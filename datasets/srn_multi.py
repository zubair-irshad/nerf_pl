import imageio
import numpy as np
import torch
# import json
# from torchvision import transforms
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *

def load_poses(pose_dir, idxs=[]):
    txtfiles = np.sort([os.path.join(pose_dir, f.name) for f in os.scandir(pose_dir)])
    posefiles = np.array(txtfiles)[idxs]
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset
    # poses = []
    # print("pose_file", posefile)
    # pose = np.loadtxt(posefile).reshape(4,4)
    # c2w = pose@srn_coords_trans

    for posefile in posefiles:
        pose = np.loadtxt(posefile).reshape(4,4)
        c2w = pose@srn_coords_trans
    #     poses.append(pose@srn_coords_trans)
    return c2w

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

class SRN_Multi():
    def __init__(self, cat='srn_cars', splits='cars_train',
                img_wh=(128, 128), data_dir = '/data/datasets/code_nerf',
                num_instances_per_obj = 1, crop_img = True, use_mask = False):
        """
        cat: srn_cars / srn_chairs
        split: cars_train(/test/val) or chairs_train(/test/val)
        First, we choose the id
        Then, we sample images (the number of instances matter)
        """
        self.define_transforms()
        self.white_back = True
        self.data_dir = os.path.join(data_dir, cat, splits)
        self.ids = np.sort([f.name for f in os.scandir(self.data_dir)])
        # self.ids = self.ids[:1102]
        self.num_instances_per_obj = num_instances_per_obj
        self.train = True if splits.split('_')[1] == 'train' else False
        self.lenids = len(self.ids)
        
        # if not self.train:
        #     # only for val, choose 10 random ids within the train set for validation
        #     id_val = np.random.choice(self.lenids, 10)
        #     self.ids = self.ids[id_val]
        #     self.lenids = len(self.ids)

        self.img_wh = img_wh
        self.crop_img = crop_img
        # bounds, common for all scenes
        self.near = 0.3
        self.far = 2.0
        self.bounds = np.array([self.near, self.far])
        self.use_mask = use_mask
        self.cat = cat

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return self.lenids

    def load_img(self, img_dir, idxs = []):
        allimgfiles = np.sort([os.path.join(img_dir, f.name) for f in os.scandir(img_dir)])
        imgfiles = np.array(allimgfiles)[idxs]
        for imgfile in imgfiles:
            img = Image.open(imgfile) 
            img = self.transform(img) # (h, w, 3)
            if self.crop_img:
                img = img[:,32:-32,32:-32]
            if self.cat == 'srn_cars':
                img = img.contiguous().view(4, -1).permute(1, 0) # (h*w, 4) RGBA 
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            else:
                img = img.contiguous().view(3, -1).permute(1, 0) # (h*w, 3) RGBA
        return img
    
    def __getitem__(self, idx):
        obj_id = self.ids[idx]
        if self.train:
            rays, img = self.return_train_data(obj_id)
            sample = { "rays": rays,
                       "rgbs": img,
                       "obj_id": idx
            }
            return sample
        else:
            rays, img = self.return_val_data(obj_id)
            sample = { "rays": rays,
                       "rgbs": img,
                       "obj_id": idx
            }
            return sample
    
    def return_train_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.random.choice(50, self.num_instances_per_obj)
        c2w = load_poses(pose_dir, instances)
        img = self.load_img(img_dir, instances)
        focal, H, W = load_intrinsic(intrinsic_path)

        w, h = self.img_wh
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1)

        if not self.use_mask:
            rays = torch.cat([rays_o, rays_d, 
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])],
                            1) # (h*w, 8)
        else:
            valid_mask = (img.sum(1)<3).flatten() # (H*W) valid color area
            ray_array = torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1) # (h*w, 8)
            rays = ray_array[valid_mask] # remove valid_mask for later epochs
            img = img[valid_mask] # remove valid_mask for later epochs
            
        return rays, img

    def return_val_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        # instances = np.arange(250)
        instances = np.random.choice(50, 1)
        c2w = load_poses(pose_dir, instances)
        img = self.load_img(img_dir, instances)
        w, h = self.img_wh
        focal, H, W = load_intrinsic(intrinsic_path)
        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1)
        return rays, img

    # def return_test_data(self, obj_id):
    #     pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
    #     img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
    #     intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
    #     # instances = np.arange(250)
    #     instances = np.random.choice(250, 1)
    #     c2w = load_poses(pose_dir, instances)
    #     img = self.load_img(img_dir, instances)
        
    #     w, h = self.img_wh
    #     focal, H, W = load_intrinsic(intrinsic_path)
    #     directions = get_ray_directions(h, w, focal) # (h, w, 3)
    #     c2w = torch.FloatTensor(c2w)[:3, :4]
    #     rays_o, rays_d = get_rays(directions, c2w)
    #     rays = torch.cat([rays_o, rays_d, 
    #                             self.near*torch.ones_like(rays_o[:, :1]),
    #                             self.far*torch.ones_like(rays_o[:, :1])],
    #                             1)
    #     return rays, img