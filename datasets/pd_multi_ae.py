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

img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_bbox_from_mask(inst_mask):
    # bounding box
    horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
    vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    # x2 and y2 should not be part of the box. Increment by 1.
    x2 += 1
    y2 += 1
    return x1, x2, y1, y2

def read_poses(pose_dir, img_files):
    pose_file = os.path.join(pose_dir, 'pose.json')
    with open(pose_file, "r") as read_content:
        data = json.load(read_content) 
    focal = data['focal']
    img_wh = data['img_size']
    asset_pose_ = data["vehicle_pose"]
    all_c2w = []
    for img_file in img_files:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        all_c2w.append(convert_pose_PD_to_NeRF(np.linalg.inv(asset_pose_) @ c2w))

    #scale to fit inside a unit bounding box
    all_c2w = np.array(all_c2w)
    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
    # bbox_dimensions = np.array(bbox_dimensions)*pose_scale_factor
    all_c2w[:, :3, 3] *= pose_scale_factor

    return all_c2w, focal, img_wh


class PD_Multi_AE(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), white_back=False, model_type = "Vanilla"):
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = white_back
        self.base_dir = root_dir
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        self.model_type = model_type
        #for object centric
        # self.near = 2.0
        # self.far = 6.0

        self.near = 0.3
        self.far = 3.0
        self.xyz_min = np.array([-2.7014, -2.6993, -2.2807]) 
        self.xyz_max = np.array([2.6986, 2.6889, 2.2192])
        # self.samples_per_epoch = 5000
        self.samples_per_epoch = 1000

    def read_train_data(self, instance_dir, image_id, latent_id):
        base_dir = os.path.join(self.base_dir, instance_dir, 'train')
        img_files = os.listdir(os.path.join(base_dir, 'rgb'))
        img_files.sort()

        all_c2w, focal, img_size = read_poses(pose_dir = os.path.join(base_dir, 'pose'), img_files= img_files)
        w, h = self.img_wh        
        
        img_name = img_files[image_id]
            
        c2w = all_c2w[image_id]
        c2w = torch.FloatTensor(c2w)[:3, :4]
        img = Image.open(os.path.join(base_dir, 'rgb', img_name))                
        img = img.resize((w,h), Image.LANCZOS)
        #Get masks
        seg_mask = Image.open(os.path.join(base_dir, 'semantic_segmentation_2d', img_name))
        seg_mask = seg_mask.resize((w,h), Image.LANCZOS)
        seg_mask =  np.array(seg_mask)
        seg_mask[seg_mask!=5] =0
        seg_mask[seg_mask==5] =1
        instance_mask = seg_mask >0

        x1, x2, y1, y2 = get_bbox_from_mask(instance_mask)

        if self.white_back:
            rgb_masked = np.ones((h,w,3), dtype=np.uint16)*255
            instance_mask_repeat = np.repeat(instance_mask[...,None],3,axis=2)
            rgb_masked[instance_mask_repeat] = np.array(img)[instance_mask_repeat]
            # img = Image.fromarray(np.uint8(rgb_masked))

        img = rgb_masked[y1:y2, x1:x2]
        instance_mask = instance_mask[y1:y2, x1:x2]

        h_new, w_new, _ =  img.shape
        focal *=(w_new/img_size[0])  # modify focal length to match size self.img_wh
        directions = get_ray_directions(h_new, w_new, focal) # (h, w, 3)
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)


        instance_mask_weight = rebalance_mask(
            instance_mask,
            fg_weight=1.0,
            bg_weight=0.05,
        )

        return rays_o, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight

    def read_val_data(self, instance_dir, image_id, latent_id):
        base_dir = os.path.join(self.base_dir, instance_dir, 'val')
        img_files = os.listdir(os.path.join(base_dir, 'rgb'))
        img_files.sort()
        all_c2w, focal, img_size = read_poses(pose_dir = os.path.join(base_dir, 'pose'), img_files= img_files)
        
        w, h = self.img_wh
        img_name = img_files[image_id]
            
        c2w = all_c2w[image_id]
        c2w = torch.FloatTensor(c2w)[:3, :4]
        img = Image.open(os.path.join(base_dir, 'rgb', img_name))                
        img = img.resize((w,h), Image.LANCZOS)
        #Get masks
        seg_mask = Image.open(os.path.join(base_dir, 'semantic_segmentation_2d', img_name))
        seg_mask = seg_mask.resize((w,h), Image.LANCZOS)
        seg_mask =  np.array(seg_mask)
        seg_mask[seg_mask!=5] =0
        seg_mask[seg_mask==5] =1
        instance_mask = seg_mask >0
        x1, x2, y1, y2 = get_bbox_from_mask(instance_mask)

        if self.white_back:
            rgb_masked = np.ones((h,w,3), dtype=np.uint16)*255
            instance_mask_repeat = np.repeat(instance_mask[...,None],3,axis=2)
            rgb_masked[instance_mask_repeat] = np.array(img)[instance_mask_repeat]
        img = rgb_masked[y1:y2, x1:x2]
        instance_mask = instance_mask[y1:y2, x1:x2]

        h_new, w_new, _ =  img.shape
        focal *=(w_new/img_size[0])  # modify focal length to match size self.img_wh
        directions = get_ray_directions(h_new, w_new, focal) # (h, w, 3)
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)


        instance_mask_weight = rebalance_mask(
            instance_mask,
            fg_weight=1.0,
            bg_weight=0.05,
        )

        return rays_o, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return self.samples_per_epoch
            # return len(self.ids)
        elif self.split == 'val':
            return len(self.ids)
        else:
            return len(self.ids[:10])


    def __getitem__(self, idx):
        random.seed()
        if self.split == 'train': # use data in the buffers
            train_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[train_idx]
            #100 is max number of images
            train_image_id = random.randint(0, 99)
            rays, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight =  self.read_train_data(instance_dir, train_image_id, latent_id = train_idx)

            return rays, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight
                
        # elif self.split == 'val': # create data for each image separately
        elif self.split=='val':
            instance_dir = self.ids[idx]
            print("instance_dir", instance_dir)
            #100 is max number of images
            val_image_id = random.randint(0, 99)
            rays, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight =  self.read_val_data(instance_dir, val_image_id, latent_id = idx)

            return rays, view_dirs, rays_d, img, radii, instance_mask, instance_mask_weight

def collate_lambda_train(batch, model_type, ray_batch_size=1024):
    imgs = list()
    instance_masks = list()
    rays = list()
    view_dirs = list()
    rays_d = list()
    rgbs = list()
    radii = list()

    for el in batch:
        cam_rays, cam_view_dirs, cam_rays_d, img, camera_radii, instance_mask, _ = el
        img = Image.fromarray(np.uint8(img))
        img = T.ToTensor()(img)
        instance_mask = T.ToTensor()(instance_mask)
        camera_radii = torch.FloatTensor(camera_radii)
        cam_rays = torch.FloatTensor(cam_rays)
        cam_view_dirs = torch.FloatTensor(cam_view_dirs)
        cam_rays_d = torch.FloatTensor(cam_rays_d)

        _, H, W = img.shape
        pix_inds = torch.randint(0,  H * W, (ray_batch_size,))
        rgb_gt = img.permute(1,2,0).flatten(0,1)[pix_inds,...] 
        msk_gt = instance_mask.permute(1,2,0).flatten(0,1)[pix_inds,...]
        camera_radii = camera_radii.view(-1)[pix_inds]
        ray = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds]
        ray_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])[pix_inds]
        viewdir = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])[pix_inds]

        imgs.append(
            img_transform(img)
        )
        instance_masks.append(msk_gt)  
        rays.append(ray)
        view_dirs.append(viewdir)
        rays_d.append(ray_d)
        rgbs.append(rgb_gt)
        radii.append(camera_radii)
    
    imgs = torch.stack(imgs)
    # rgbs = torch.stack(rgbs, 1)  
    # instance_masks = torch.stack(instance_masks, 1)
    # rays = torch.stack(rays, 1)  
    # rays_d = torch.stack(rays_d, 1)  
    # view_dirs = torch.stack(view_dirs, 1)  
    # radii = torch.stack(radii, 1)  

    rgbs = torch.stack(rgbs, 0)  
    instance_masks = torch.stack(instance_masks, 0)
    rays = torch.stack(rays, 0)  
    rays_d = torch.stack(rays_d, 0)  
    view_dirs = torch.stack(view_dirs, 0)  
    radii = torch.stack(radii, 0)  
    
    if model_type == "Vanilla":
        sample = {
            "src_imgs": imgs,
            "rays": rays,
            "rgbs": rgbs,
            "instance_mask": instance_masks,
        }
    else:
        sample = {}
        sample["src_imgs"] = imgs
        sample["rays_o"] = rays
        sample["rays_d"] = rays_d
        sample["viewdirs"] = view_dirs
        sample["target"] = rgbs
        sample["radii"] = radii.unsqueeze(-1)
        sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
        sample["normals"] = torch.zeros_like(sample["rays_o"])
        sample["instance_mask"] = instance_masks
    return sample


def collate_lambda_val(batch, model_type):
    
    cam_rays, cam_view_dirs, cam_rays_d, img, camera_radii, instance_mask, _ = batch[0]
    h,w,_ = img.shape
    img = Image.fromarray(np.uint8(img))
    img = T.ToTensor()(img)
    instance_mask = T.ToTensor()(instance_mask)
    camera_radii = torch.FloatTensor(camera_radii)
    cam_rays = torch.FloatTensor(cam_rays)
    cam_view_dirs = torch.FloatTensor(cam_view_dirs)
    cam_rays_d = torch.FloatTensor(cam_rays_d)

    rgbs = img.permute(1,2,0).flatten(0,1)
    instance_masks = instance_mask.permute(1,2,0).flatten(0,1)
    radii = camera_radii.view(-1)
    rays = cam_rays.view(-1, cam_rays.shape[-1])
    rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
    view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])
    imgs = img_transform(img)
    
    if model_type == "Vanilla":
        sample = {
            "src_imgs": imgs,
            "rays": rays,
            "rgbs": rgbs,
            "instance_mask": instance_masks,
            "img_wh": np.array((w,h))
        }
    else:
        sample = {}
        sample["src_imgs"] = imgs
        sample["rays_o"] = rays
        sample["rays_d"] = rays_d
        sample["viewdirs"] = view_dirs
        sample["target"] = rgbs
        sample["radii"] = radii
        sample["multloss"] = torch.zeros((sample["rays_o"].shape[1], 1))
        sample["normals"] = torch.zeros_like(sample["rays_o"])
        sample["instance_mask"] = instance_masks
        sample["img_wh"] = np.array((w,h))
    return sample