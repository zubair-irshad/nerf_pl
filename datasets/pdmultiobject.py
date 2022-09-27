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


class PDMultiObject(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1600, 1600)):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        self.img_wh = img_wh
        self.define_transforms()
        self.all_c2w = []
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        self.base_dir = self.root_dir
        self.img_files = os.listdir(os.path.join(self.base_dir, 'rgb'))
        self.img_files.sort()
        #for bottle

        self.all_unique_ids = [1167, 1168, 1169, 1170]
        self.near = 0.3
        self.far = 3.0
        self.all_c2w, self.focal, self.img_size = read_poses(pose_dir = os.path.join(self.base_dir, 'pose'), img_files= self.img_files)
        w, h = self.img_wh
        print("self.focal", self.focal)
        self.focal *=(self.img_wh[0]/self.img_size[0])  # modify focal length to match size self.img_wh
        print("self.focal after", self.focal)
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_instance_ids = []

            for i, img_name in enumerate(self.img_files):
                print("img_name", img_name)
                c2w = self.all_c2w[i]
                directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, rays_d = get_rays(directions, c2w)

                img = Image.open(os.path.join(self.base_dir, 'rgb', img_name))                
                img = img.resize((w,h), Image.LANCZOS)
                img = self.transform(img) # (h, w, 3)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

                #Get masks
                inst_mask = Image.open(os.path.join(self.base_dir, 'semantic_segmentation_2d', img_name))
                inst_mask = np.array(inst_mask)
                inst_mask = cv2.resize(inst_mask, (w,h), interpolation=cv2.INTER_NEAREST)
                unique_ids = np.unique(np.array(inst_mask))[1:]
                print("unique_ids",unique_ids)

                curr_frame_instance_masks = []
                curr_frame_instance_masks_weight = []
                curr_frame_instance_ids = []

                for i_inst, instance_id in enumerate(unique_ids):
                    instance_mask = (inst_mask == instance_id)
                    print("instance_mask", instance_mask.shape)
                    instance_mask_weight = rebalance_mask(
                        instance_mask,
                        fg_weight=1.0,
                        bg_weight=0.005,
                    )
                    instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                        -1), self.transform(instance_mask_weight).view(-1)

                    sample = {
                        "instance_mask": instance_mask,
                        "instance_mask_weight": instance_mask_weight,
                        "instance_ids": torch.ones_like(instance_mask).long() * (i_inst+1)
                    }
                    curr_frame_instance_masks += [sample["instance_mask"]]
                    curr_frame_instance_masks_weight += [sample["instance_mask_weight"]]
                    curr_frame_instance_ids += [sample["instance_ids"]]
                    if i_inst ==0:
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

            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            self.all_instance_masks = torch.cat(self.all_instance_masks, 0)  # (len(self.meta['frames])*h*w)
            self.all_instance_masks_weight = torch.cat(self.all_instance_masks_weight, 0)  # (len(self.meta['frames])*h*w)
            self.all_instance_ids = torch.cat(self.all_instance_ids, 0).long()  # (len(self.meta['frames])*h*w)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 1 # only validate 8 images (to support <=8 gpus)
        return len(self.meta)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            rand_instance_id = torch.randint(0, len(self.all_unique_ids), (1,))
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "instance_mask": self.all_instance_masks[idx, rand_instance_id],
                "instance_mask_weight": self.all_instance_masks_weight[
                    idx, rand_instance_id
                ],
                "instance_ids": self.all_instance_ids[idx, rand_instance_id],
            }

        elif self.split == 'val': # create data for each image separately
            val_idx = 15
            img_name = self.img_files[val_idx]
            w, h = self.img_wh
            c2w = self.all_c2w[val_idx]
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            img = Image.open(os.path.join(self.base_dir, 'rgb', img_name))                
            img = img.resize((w,h), Image.LANCZOS)
            img = self.transform(img) # (h, w, 3)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            rays = torch.cat([rays_o, rays_d, 
                                        self.near*torch.ones_like(rays_o[:, :1]),
                                        self.far*torch.ones_like(rays_o[:, :1])],
                                        1) # (h*w, 8)  
            #Get masks
            inst_mask = Image.open(os.path.join(self.base_dir, 'semantic_segmentation_2d', img_name))
            inst_mask = np.array(inst_mask)
            inst_mask = cv2.resize(inst_mask, (w,h), interpolation=cv2.INTER_NEAREST)
            unique_ids = np.unique(np.array(inst_mask))[1:]
            
            val_id = 0
            val_inst_id = unique_ids[val_id]

            instance_mask = (inst_mask == val_inst_id).astype(np.bool)
            instance_mask_weight = rebalance_mask(
                instance_mask,
                fg_weight=1.0,
                bg_weight=0.005,
            )
            instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                -1), self.transform(instance_mask_weight).view(-1)
            instance_id_out = torch.ones_like(instance_mask).long() * (val_id+1)
                    
            sample = {
                "rays": rays,
                "rgbs": img,
                "img_wh": self.img_wh,
                "instance_mask": instance_mask,
                "instance_mask_weight": instance_mask_weight,
                "instance_ids": instance_id_out
            }
        else:
            w, h = self.img_wh
            c2w = self.all_c2w[idx]

            focal = self.all_focal[idx]
            focal *=(self.img_wh[0]/1920) # modify focal length to match size self.img_wh
            
            directions = get_ray_directions(h, w, focal) # (h, w, 3)

            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {
                "rays": rays,
                "c2w": c2w,
                "img_wh": self.img_wh
            }  
        return sample