from logging.config import valid_ident
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# from .google_scanned_utils import load_image_from_exr, load_seg_from_exr
import struct
from .ray_utils import *
from .nocs_utils import rebalance_mask
import glob
from objectron.schema import annotation_data_pb2 as annotation_protocol
import glob
from utils.test_poses import *

def read_poses(pose_dir, img_files):
    pose_file = os.path.join(pose_dir, 'pose.json')
    with open(pose_file, "r") as read_content:
        data = json.load(read_content)
    # fov = data['fov']
    #hard coded here 
    focal = data['focal']
    # focal = (800 / 2) / np.tan((fov / 2) / (180 / np.pi))
    # img_wh = (800,800)
    img_wh = data['img_size']
    all_c2w = []
    for img_file in img_files:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        all_c2w.append(c2w)
    return all_c2w, focal, img_wh


class PDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), white_back=False, model_type = "Vanilla"):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        self.img_wh = img_wh
        self.define_transforms()
        self.all_c2w = []
        self.white_back = white_back
        self.read_meta()
        w, h = self.img_wh
        self.image_sizes = np.array([[h, w] for i in range(len(self.all_c2w))])
        self.val_image_sizes = np.array([[h, w] for i in range(1)])
        self.model_type = model_type

    def read_meta(self):
        
        if self.split == 'test':
            split = 'val'
        else:
            split = self.split
        self.base_dir = os.path.join(self.root_dir, split)
        # self.base_dir = os.path.join(self.root_dir, 'train')
        self.img_files = os.listdir(os.path.join(self.base_dir, 'rgb'))
        self.img_files.sort()

        #for object centric
        self.near = 2.0
        self.far = 6.0

        # self.near = 0.2
        # self.far = 3.0

        #for backgrond modelling as well
        # self.near = 0.2
        # self.far = 3.0
        self.all_c2w, self.focal, self.img_size = read_poses(pose_dir = os.path.join(self.base_dir, 'pose'), img_files= self.img_files)
        w, h = self.img_wh
        print("self.focal", self.focal)
        self.focal *=(self.img_wh[0]/self.img_size[0])  # modify focal length to match size self.img_wh
        print("self.focal after", self.focal)
        if self.split == 'train' or self.split == 'test': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_rays_d = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_instance_ids = []
            self.all_radii = []

            # self.img_files = self.img_files[:5]
            for i, img_name in enumerate(self.img_files):
                c2w = self.all_c2w[i]
                directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
                img = Image.open(os.path.join(self.base_dir, 'rgb', img_name))                
                img = img.resize((w,h), Image.LANCZOS)
                #Get masks
                seg_mask = Image.open(os.path.join(self.base_dir, 'semantic_segmentation_2d', img_name))
                seg_mask = seg_mask.resize((w,h), Image.LANCZOS)
                seg_mask =  np.array(seg_mask)
                seg_mask[seg_mask!=5] =0
                seg_mask[seg_mask==5] =1
                instance_mask = seg_mask >0

                if self.white_back:
                    rgb_masked = np.ones((h,w,3), dtype=np.uint16)*255
                    instance_mask_repeat = np.repeat(instance_mask[...,None],3,axis=2)
                    rgb_masked[instance_mask_repeat] = np.array(img)[instance_mask_repeat]
                    img = Image.fromarray(np.uint8(rgb_masked))

                img = self.transform(img) # (h, w, 3)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

                instance_mask_weight = rebalance_mask(
                    instance_mask,
                    fg_weight=1.0,
                    bg_weight=0.05,
                )
                instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                    -1), self.transform(instance_mask_weight).view(-1)
                instance_ids = torch.ones_like(instance_mask).long() * 1
                self.all_rays_d+=[rays_d]
                self.all_rays += [torch.cat([rays_o, view_dirs, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_d[:, :1])],
                                1)] # (h*w, 8)
                self.all_rgbs += [img]
                self.all_instance_masks +=[instance_mask]
                self.all_instance_masks_weight +=[instance_mask_weight]
                self.all_instance_ids +=[instance_ids]
                self.all_radii +=[radii.unsqueeze(-1)]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)

            print("self.allrays", self.all_rays.shape)
            #Get near far for voxel training normalization
            rays_o = self.all_rays[:, :3]
            viewdirs = self.all_rays[:, 3:6]
            near = self.all_rays[:, 6]
            far = self.all_rays[:, 7]
            print("rays_o, rays_d", rays_o.shape, viewdirs.shape)
            pts_nf = torch.stack([rays_o + viewdirs * self.near, rays_o + viewdirs * self.far], 1
            )
            self.xyz_min = torch.amin(pts_nf, axis=(0, 1))
            self.xyz_max = torch.amax(pts_nf, axis=(0, 1))

            print("self.xyz_min, self.xyz_max", self.xyz_min, self.xyz_max)

            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_rays_d = torch.cat(self.all_rays_d, 0)
            self.all_radii = torch.cat(self.all_radii, 0)
            self.all_instance_masks = torch.cat(self.all_instance_masks, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_instance_masks_weight = torch.cat(self.all_instance_masks_weight, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_instance_ids = torch.cat(self.all_instance_ids, 0) # (len(self.meta['frames])*h*w, 3)
        # elif self.split == 'test':
            
        #     locations = get_archimedean_spiral(sphere_radius=1.5)
        #     poses_test = [look_at(loc, [0,0,0])[0] for loc in locations]
        #     self.poses_test = convert_pose_spiral(poses_test)
        #     #self.poses_test = create_spheric_poses(radius = 1.7)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 1 # only validate 8 images (to support <=8 gpus)
            # return len(self.all_c2w) # only validate 8 images (to support <=8 gpus)
        else:
            return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split == 'train' or self.split == 'test': # use data in the buffers
            # for running NeRFFactory RefNeRF ad NeRF++
            if self.model_type == "Vanilla":
                sample = {
                    "rays": self.all_rays[idx],
                    "rgbs": self.all_rgbs[idx],
                    "instance_mask": self.all_instance_masks[idx],
                    "instance_mask_weight": self.all_instance_masks_weight[idx],
                    "instance_ids": self.all_instance_ids[idx],
                }
            else:
                sample = {}
                sample["rays_o"] = self.all_rays[idx][:3]
                sample["rays_d"] = self.all_rays_d[idx]
                sample["viewdirs"] = self.all_rays[idx][3:6]
                sample["radii"] = self.all_radii[idx]
                sample["target"] = self.all_rgbs[idx]
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])
                sample["instance_mask"] = self.all_instance_masks[idx]
                sample["instance_mask_weight"] = self.all_instance_masks_weight[idx]
                sample["instance_ids"] = self.all_instance_ids[idx]
            

        # elif self.split == 'val': # create data for each image separately
        elif self.split=='val':
            idx = 65
            # idx = idx
            img_name = self.img_files[idx]
            w, h = self.img_wh
            c2w = self.all_c2w[idx]
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            # rays_o, rays_d = get_rays(directions, c2w)
            rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
            img = Image.open(os.path.join(self.base_dir, 'rgb', img_name))  
            img = img.resize((w,h), Image.LANCZOS)     
                    

            #Get masks
            seg_mask = Image.open(os.path.join(self.base_dir, 'semantic_segmentation_2d', img_name))
            seg_mask = seg_mask.resize((w,h), Image.LANCZOS)
            seg_mask =  np.array(seg_mask)
            seg_mask[seg_mask!=5] =0
            seg_mask[seg_mask==5] =1
            instance_mask = seg_mask >0
            if self.white_back:
                rgb_masked = np.ones((h,w,3), dtype=np.uint16)*255
                instance_mask_repeat = np.repeat(instance_mask[...,None],3,axis=2)
                rgb_masked[instance_mask_repeat] = np.array(img)[instance_mask_repeat]
                img = Image.fromarray(np.uint8(rgb_masked))

            img = self.transform(img) # (h, w, 3)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            instance_mask_weight = rebalance_mask(
                instance_mask,
                fg_weight=1.0,
                bg_weight=0.05,
            )
            instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                -1), self.transform(instance_mask_weight).view(-1)
            instance_ids = torch.ones_like(instance_mask).long() * 1
            rays = torch.cat([rays_o, view_dirs, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_d[:, :1])],
                                1) # (H*W, 8)

            if self.model_type == "Vanilla":
                sample = {
                    "rays": rays,
                    "rgbs": img,
                    "img_wh": self.img_wh,
                    "instance_mask": instance_mask,
                    "instance_mask_weight": instance_mask_weight,
                    "instance_ids": instance_ids
                }
            else:
                sample = {}
                sample["rays_o"] = rays[:,:3]
                sample["rays_d"] = rays_d
                sample["viewdirs"] = rays[:,3:6]
                sample["target"] = img
                sample["radii"] = radii
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])
                sample["instance_mask"] = instance_mask
                sample["instance_mask_weight"] = instance_mask_weight
                sample["instance_ids"] = instance_ids

        else:
            sample = {}
            sample["rays_o"] = self.all_rays[idx][:3]
            sample["rays_d"] = self.all_rays_d[idx]
            sample["viewdirs"] = self.all_rays[idx][3:6]
            sample["radii"] = self.all_radii[idx]
            sample["target"] = self.all_rgbs[idx]
            sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
            sample["normals"] = np.zeros_like(sample["rays_o"])

        return sample