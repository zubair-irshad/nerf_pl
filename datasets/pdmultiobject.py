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


# def read_poses(pose_dir_train, pose_dir_val, img_files):
#     pose_file_train = os.path.join(pose_dir_train, 'pose.json')
#     pose_file_val = os.path.join(pose_dir_val, 'pose.json')
#     with open(pose_file_train, "r") as read_content:
#         data = json.load(read_content)
#     with open(pose_file_val, "r") as read_content:
#         data_val = json.load(read_content)

#     focal = data['focal']
#     img_wh = data['img_size']
#     obj_location = np.array(data["obj_location"])
#     all_c2w = []

#     for img_file in img_files:
#         c2w = np.array(data['transform'][img_file.split('.')[0]])
#         c2w[:3, 3] = c2w[:3, 3] - obj_location
#         all_c2w.append(convert_pose_PD_to_NeRF(c2w))

#     for img_file in img_files:
#         c2w = np.array(data_val['transform'][img_file.split('.')[0]])
#         c2w[:3, 3] = c2w[:3, 3] - obj_location
#         all_c2w.append(convert_pose_PD_to_NeRF(c2w))

#     all_c2w = np.array(all_c2w)

#     pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
#     all_c2w[:, :3, 3] *= pose_scale_factor

#     all_c2w_train = all_c2w[:99, :, :]
#     all_c2w_test = all_c2w[99:, :, :]

#     return all_c2w_train, all_c2w_test, focal, img_wh

def read_poses(pose_dir_train, pose_dir_val, img_files_train, img_files_val, output_boxes = False):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    pose_file_val = os.path.join(pose_dir_val, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)
    with open(pose_file_val, "r") as read_content:
        data_val = json.load(read_content)
    focal = data['focal']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w = []
    all_c2w_train = []
    all_c2w_test = []
    for img_file in img_files_train:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))
    for img_file in img_files_val:
        c2w = np.array(data_val['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))
        all_c2w_test.append(convert_pose_PD_to_NeRF(c2w))
    all_c2w = np.array(all_c2w)
    all_c2w_train = np.array(all_c2w_train)
    all_c2w_test = np.array(all_c2w_test)
    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
    all_c2w_train[:, :3, 3] *= pose_scale_factor
    all_c2w_test[:, :3, 3] *= pose_scale_factor
    all_c2w_val = all_c2w_train[100:]
    all_c2w_train = all_c2w_train[:100]
    # Get bounding boxes for object MLP training only
    if output_boxes:
        all_boxes = []
        all_translations= []
        all_rotations = []
        for k,v in data['bbox_dimensions'].items():
                bbox = np.array(v)
                all_boxes.append(bbox*pose_scale_factor)
                #New scene 200 uncomment here
                all_rotations.append(data["obj_rotations"][k])
                translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor
                all_translations.append(translation)
        # Old scenes uncomment here
        # all_translations = (np.array(data['obj_translations'])- obj_location)*pose_scale_factor
        # all_rotations = data["obj_rotations"]
        RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
        return all_c2w_train, all_c2w_val, all_c2w_test, focal, img_wh, RTs
    else:
        return all_c2w_train, all_c2w_val, all_c2w_test, focal, img_wh

class PDMultiObject(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1600, 1600), 
                white_back = False, model_type= 'vanilla'):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        self.model_type = model_type
        self.img_wh = img_wh
        self.define_transforms()
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        base_dir_train = os.path.join(self.root_dir, 'train')
        img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
        img_files_train.sort()

        self.base_dir_val = base_dir_train
        
        base_dir_test = os.path.join(self.root_dir, 'val')
        img_files_test = os.listdir(os.path.join(base_dir_test, 'rgb'))
        img_files_test.sort()

        # self.all_unique_ids = [1167, 1168, 1169, 1170]
        self.near = 0.2
        self.far = 3.0
        pose_dir_train = os.path.join(self.root_dir, 'train', 'pose')
        pose_dir_val = os.path.join(self.root_dir, 'val', 'pose')

        if self.split == 'train':
            all_c2w, all_c2w_val, _, self.focal, self.img_size, _ = read_poses(pose_dir_train, pose_dir_val, img_files_train, img_files_test, output_boxes=True)
        elif self.split == 'val':
            all_c2w, all_c2w_val, _, self.focal, self.img_size, _ = read_poses(pose_dir_train, pose_dir_val, img_files_train, img_files_test, output_boxes=True)

        self.img_files_val = img_files_train[100:]        
        self.all_c2w_val = all_c2w_val
        print("all c2w", all_c2w.shape)
        w, h = self.img_wh
        print("self.focal", self.focal)
        self.focal *=(self.img_wh[0]/self.img_size[0])  # modify focal length to match size self.img_wh
        print("self.focal after", self.focal)
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_rays_d = []
            self.all_radii = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_instance_ids = []

            num = 9
            if num ==3:
                src_views_num = [7, 50, 66]
            elif num ==5:
                src_views_num = [7, 28, 50, 66, 75]
            elif num ==7:
                src_views_num = [7, 28, 39, 50, 64, 66, 75]
            elif num ==9:
                src_views_num = [7, 21, 28, 39, 45, 50, 64, 66, 75]
            elif num ==1:
                src_views_num = [7]
            NV =  100
            for train_image_id in range(0, NV):
            # for i, img_name in enumerate(self.img_files):
                if train_image_id not in src_views_num:
                    continue
                img_name = img_files_train[train_image_id]
                c2w = all_c2w[train_image_id]
                directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)

                img = Image.open(os.path.join(base_dir_train, 'rgb', img_name))                
                img = img.resize((w,h), Image.LANCZOS)
                img = self.transform(img) # (h, w, 3)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

                self.all_rays += [torch.cat([rays_o, view_dirs, 
                                            self.near*torch.ones_like(rays_o[:, :1]),
                                            self.far*torch.ones_like(rays_o[:, :1])],
                                            1)] # (h*w, 8)  
                self.all_rays_d+=[view_dirs]
                self.all_radii +=[radii.unsqueeze(-1)]
                self.all_rgbs += [img]

            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            self.all_rays_d = torch.cat(self.all_rays_d, 0)
            self.all_radii = torch.cat(self.all_radii, 0)

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
            # rand_instance_id = torch.randint(0, len(self.all_unique_ids), (1,))
            if self.model_type == "vanilla":
                sample = {
                    "rays": self.all_rays[idx],
                    "rgbs": self.all_rgbs[idx],
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

        elif self.split == 'val': # create data for each image separately
            val_idx = 15
            img_name = self.img_files_val[val_idx]
            w, h = self.img_wh
            c2w = self.all_c2w_val[val_idx]
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
            img = Image.open(os.path.join(self.base_dir_val, 'rgb', img_name))                
            img = img.resize((w,h), Image.LANCZOS)
            img = self.transform(img) # (h, w, 3)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            rays = torch.cat([rays_o, view_dirs, 
                                        self.near*torch.ones_like(rays_o[:, :1]),
                                        self.far*torch.ones_like(rays_o[:, :1])],
                                        1) # (h*w, 8)  
        if self.model_type == "vanilla":
                sample = {
                    "rays": rays,
                    "rgbs": img,
                    "img_wh": self.img_wh,
                }
        else:
                sample = {}
                sample["rays_o"] = rays[:,:3]
                sample["rays_d"] = view_dirs
                sample["viewdirs"] = rays[:,3:6]
                sample["target"] = img
                sample["radii"] = radii
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])
        return sample