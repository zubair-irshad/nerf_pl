import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2

# from .google_scanned_utils import load_image_from_exr, load_seg_from_exr

from .ray_utils import *
from .nocs_utils import rebalance_mask

class GoogleScannedDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1600, 1600)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):

        self.base_dir = os.path.join(self.root_dir, '00000')
        json_files = [pos_json for pos_json in os.listdir(self.base_dir) if pos_json.endswith('.json')]        
        json_files.sort()
        self.meta = json_files
        w, h = self.img_wh
        # bounds, common for all scenes
        self.near = 0.5
        self.far = 3.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_instance_ids = []

            self.instance_ids = []

            for json_file in self.meta:
                file = os.path.join(self.base_dir, json_file)
                with open(file, 'r') as f:
                    data = json.loads(f.read())
                cam2world = data["camera_data"]["cam2world"]
                intrinsics = data["camera_data"]["intrinsics"]
                focal = intrinsics["fx"]
                directions = get_ray_directions(h, w, focal) # (h, w, 3)
                c2w = np.array(cam2world).T
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, rays_d = get_rays(directions, c2w)
                img_name = json_file.split('.')[0]+ '.png'
                seg_name = json_file.split('.')[0]+ '.seg.png'
                img = Image.open(img_name)
                seg_masks = cv2.imread(seg_name, cv2.IMREAD_ANYDEPTH)                
                # img = load_image_from_exr(os.path.join(self.base_dir, img_name))
                # seg_masks = load_seg_from_exr(os.path.join(self.base_dir, seg_name))
                img = self.transform(img) # (h, w, 3)
                img = img.view(-1, 3) # (h*w, 3) RGBA
                curr_frame_instance_masks = []
                curr_frame_instance_masks_weight = []
                curr_frame_instance_ids = []
                # Load masks for each objects
                for id in range(len(data["objects"])):
                    segmentation_id = data["objects"][id]["segmentation_id"]
                    self.instance_ids.append(segmentation_id)
                    mask = seg_masks == segmentation_id
                    instance_mask_weight = rebalance_mask(
                        mask,
                        fg_weight=1.0,
                        bg_weight=0.05,
                    )
                    instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                        -1), self.transform(instance_mask_weight).view(-1)
                    instance_ids = torch.ones_like(instance_mask).long() * segmentation_id
                    curr_frame_instance_masks += [instance_mask]
                    curr_frame_instance_masks_weight += [instance_mask_weight]
                    curr_frame_instance_ids += [instance_ids]

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1)] # (h*w, 8)
                self.all_rgbs += [img]
                self.all_instance_masks += [torch.stack(curr_frame_instance_masks, -1)]
                self.all_instance_masks_weight += [
                    torch.stack(curr_frame_instance_masks_weight, -1)
                ]
                self.all_instance_ids += [torch.stack(curr_frame_instance_ids, -1)]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_instance_masks = torch.cat(self.all_instance_masks, 0)  # (len(self.meta['frames])*h*w)
            self.all_instance_masks_weight = torch.cat(self.all_instance_masks_weight, 0)  # (len(self.meta['frames])*h*w)
            self.all_instance_ids = torch.cat(self.all_instance_ids, 0).long()  # (len(self.meta['frames])*h*w)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            rand_instance_id = torch.randint(0, len(self.instance_ids), (1,))
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "instance_mask": self.all_instance_masks[idx, rand_instance_id],
                "instance_mask_weight": self.all_instance_masks_weight[
                    idx, rand_instance_id
                ],
                "instance_ids": self.all_instance_ids[idx, rand_instance_id],
            }
        else: # create data for each image separately
            json_file = self.meta[idx]
            w, h = self.img_wh
            file = os.path.join(self.base_dir, json_file)
            with open(file, 'r') as f:
                data = json.loads(f.read())
            cam2world = data["camera_data"]["cam2world"]
            intrinsics = data["camera_data"]["intrinsics"]
            focal = intrinsics["fx"]
            directions = get_ray_directions(h, w, focal) # (h, w, 3)
            c2w = np.array(cam2world).T
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            img_name = json_file.split('.')[0]+ '.exr'
            seg_name = json_file.split('.')[0]+ '.seg.exr'
            
            # img = load_image_from_exr(os.path.join(self.base_dir, img_name))
            # seg_masks = load_seg_from_exr(os.path.join(self.base_dir, seg_name))
            img = Image.open(img_name)
            seg_masks = cv2.imread(seg_name, cv2.IMREAD_ANYDEPTH)    
            img = self.transform(img) # (h, w, 3)
            img = img.view(-1, 3) # (h*w, 3) RGBA

            directions = get_ray_directions(h, w, focal) # (h, w, 3)
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            val_inst_id = 7
            for i_inst, instance_id in enumerate(self.instance_ids):
                if instance_id != val_inst_id:
                    continue
                instance_mask = seg_masks == instance_id 
                instance_mask_weight = rebalance_mask(
                    instance_mask,
                    fg_weight=1.0,
                    bg_weight=0.005,
                )
                instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                    -1), self.transform(instance_mask_weight).view(-1)
                instance_id_out = torch.ones_like(instance_mask).long() * instance_id
            sample = {
                "rays": rays,
                "rgbs": img,
                "instance_mask": instance_mask,
                "instance_mask_weight": instance_mask_weight,
                "instance_ids": instance_id_out
            }     

        return sample