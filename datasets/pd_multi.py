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
import random

def read_poses(pose_dir, img_files):
    pose_file = os.path.join(pose_dir, 'pose.json')
    with open(pose_file, "r") as read_content:
        data = json.load(read_content) 
    focal = data['focal']
    img_wh = data['img_size']
    all_c2w = []
    for img_file in img_files:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        all_c2w.append(c2w)
    return all_c2w, focal, img_wh


class PD_Multi(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), white_back=False, model_type = "Vanilla"):
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = white_back
        self.base_dir = root_dir
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        # w, h = self.img_wh
        # self.image_sizes = np.array([[h, w] for i in range(len(self.all_c2w))])
        # self.val_image_sizes = np.array([[h, w] for i in range(1)])
        self.model_type = model_type
        #for object centric
        self.near = 2.0
        self.far = 6.0
        self.xyz_min = np.array([-2.7014, -2.6993, -2.2807]) 
        self.xyz_max = np.array([2.6986, 2.6889, 2.2192])
        # self.samples_per_epoch = 5000
        self.samples_per_epoch = 100

    def read_train_data(self, instance_dir, image_id, latent_id):
        base_dir = os.path.join(self.base_dir, instance_dir, 'train')
        img_files = os.listdir(os.path.join(base_dir, 'rgb'))
        img_files.sort()

        all_c2w, focal, img_size = read_poses(pose_dir = os.path.join(base_dir, 'pose'), img_files= img_files)
        w, h = self.img_wh
        focal *=(self.img_wh[0]/img_size[0])  # modify focal length to match size self.img_wh
        
        
        img_name = img_files[image_id]
            
        c2w = all_c2w[image_id]
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
        img = Image.open(os.path.join(base_dir, 'rgb', img_name))                
        img = img.resize((w,h), Image.Resampling.LANCZOS)
        #Get masks
        seg_mask = Image.open(os.path.join(base_dir, 'semantic_segmentation_2d', img_name))
        seg_mask = seg_mask.resize((w,h), Image.Resampling.LANCZOS)
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
        instance_ids = (torch.ones_like(instance_mask).long())* (latent_id+1)

        return rays_o, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight, instance_ids

    def read_val_data(self, instance_dir, image_id, latent_id):
        base_dir = os.path.join(self.base_dir, instance_dir, 'val')
        img_files = os.listdir(os.path.join(base_dir, 'rgb'))
        img_files.sort()
        all_c2w, focal, img_size = read_poses(pose_dir = os.path.join(base_dir, 'pose'), img_files= img_files)
        w, h = self.img_wh
        focal *=(self.img_wh[0]/img_size[0])  # modify focal length to match size self.img_wh
        img_name = img_files[image_id]
            
        c2w = all_c2w[image_id]
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
        img = Image.open(os.path.join(base_dir, 'rgb', img_name))                
        img = img.resize((w,h), Image.Resampling.LANCZOS)
        #Get masks
        seg_mask = Image.open(os.path.join(base_dir, 'semantic_segmentation_2d', img_name))
        seg_mask = seg_mask.resize((w,h), Image.Resampling.LANCZOS)
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
        instance_ids = (torch.ones_like(instance_mask)).long()*(latent_id+1)

        return rays_o, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight, instance_ids


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return self.samples_per_epoch
            # return len(self.ids)
        elif self.split == 'val':
            return len(self.ids)

    def __getitem__(self, idx):
        random.seed()
        if self.split == 'train': # use data in the buffers
            train_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[train_idx]
            #100 is max number of images
            train_image_id = random.randint(0, 99)
            rays, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight, instance_ids =  self.read_train_data(instance_dir, train_image_id, latent_id = train_idx)
            
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
                sample["rays_o"] = rays
                sample["rays_d"] = rays_d
                sample["viewdirs"] = view_dirs
                sample["target"] = img
                sample["radii"] = radii
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])
                sample["instance_mask"] = instance_mask
                sample["instance_mask_weight"] = instance_mask_weight
                sample["instance_ids"] = instance_ids
                


        # elif self.split == 'val': # create data for each image separately
        elif self.split=='val':
            instance_dir = self.ids[idx]
            #100 is max number of images
            val_image_id = random.randint(0, 99)
            rays, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight, instance_ids =  self.read_val_data(instance_dir, val_image_id, latent_id = idx)
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
                sample["rays_o"] = rays
                sample["rays_d"] = rays_d
                sample["viewdirs"] = view_dirs
                sample["target"] = img
                sample["radii"] = radii
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])
                sample["instance_mask"] = instance_mask
                sample["instance_mask_weight"] = instance_mask_weight
                sample["instance_ids"] = instance_ids

        # else:
        #     sample = {}
        #     sample["rays_o"] = self.all_rays[idx][:3]
        #     sample["rays_d"] = self.all_rays_d[idx]
        #     sample["viewdirs"] = self.all_rays[idx][3:6]
        #     sample["radii"] = self.all_radii[idx]
        #     sample["target"] = self.all_rgbs[idx]
        #     sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
        #     sample["normals"] = np.zeros_like(sample["rays_o"])

        return sample