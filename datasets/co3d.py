import os
import sys
from pathlib import Path
import logging
import pickle

import numpy as np
import torch
import torch.nn.functional as F

<<<<<<< HEAD
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(code_dir, "3rdparty/co3d"))

from datasets.co3d_dataset import Co3dDataset as Co3dData
from datasets.camera import PerspectiveCamera
from datasets.ray_utils import homogenise_torch
=======
# code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(os.path.join(code_dir, "3rdparty/co3d"))

from datasets.co3d_dataset import Co3dDataset as Co3dData
from datasets.camera import PerspectiveCamera
from datasets.ray_utils import *
import torchvision.transforms as T

from datasets.nocs_utils import rebalance_mask_tensor

import os
import sys

import numpy as np
import torch


class CO3D_Instance(torch.utils.data.Dataset):
    def __init__(self, data_dir, category, instance, downsample_factor = 4, split = 'train'):
        self.ds = Dataset(data_dir = data_dir, category = category, instance = instance)
        self.cameras = self.ds.get_cameras()
        self.near = 0.2
        self.far = 30.0
        self.all_rays = []
        self.all_rgbs = []
        self.all_instance_masks = []
        self.all_instance_masks_weight = []
        self.all_instance_ids = []
        
        self.split = split
        self.downsample_factor = downsample_factor
        self.white_back = False

        size = self.ds.size
        self.scale_factor = np.max(size)
        self.all_c2w = []
        for i, camera in enumerate(self.cameras):
            c2w = camera.get_pose_matrix().squeeze().cpu().numpy().copy()
            c2w[:3,3] /= self.scale_factor
            c2w = self.convert_pose(c2w)
            self.all_c2w.append(c2w)
            K = camera.get_intrinsics().squeeze().cpu().numpy()
            H, W = camera.get_image_size()[0].cpu().numpy()
            img_size = (H,W)
            H_new, W_new = int(H/self.downsample_factor), int(W/self.downsample_factor)
            focal = K[0,0]
            focal *=(H_new/H)

            directions = get_ray_directions(H_new, W_new, focal) # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            
            img = self.ds.images[i]
            resize_transform = T.Resize((H_new, W_new), antialias=True)
            img = resize_transform(img.permute(2,0,1))
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            instance_mask = self.ds.co3d_masks[i]
            instance_mask = resize_transform(instance_mask)
            instance_mask = (instance_mask>0).squeeze(0)
            instance_mask_weight = rebalance_mask_tensor(
                instance_mask,
                fg_weight=1.0,
                bg_weight=0.05,
            )
            #print("instance_mask, instance_mask_weight", instance_mask.shape, instance_mask_weight.shape)
            instance_mask = instance_mask.view(-1)
            instance_mask_weight = instance_mask_weight.view(-1)
            instance_ids = torch.ones_like(instance_mask).long() * 1

            self.all_rays += [torch.cat([rays_o, rays_d, 
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])],
                            1)] # (h*w, 8)
            self.all_rgbs += [img]
            self.all_instance_masks +=[instance_mask]
            self.all_instance_masks_weight +=[instance_mask_weight]
            self.all_instance_ids +=[instance_ids]

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs = torch.cat(self.all_rgbs, 0)
        self.all_instance_masks = torch.cat(self.all_instance_masks, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_instance_masks_weight = torch.cat(self.all_instance_masks_weight, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_instance_ids = torch.cat(self.all_instance_ids, 0) # (len(self.meta['frames])*h*w, 3)


    def convert_pose(self, C2W):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        C2W = np.matmul(C2W, flip_yz)
        return C2W

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 1 # only validate 1 image

    def ray_idx_to_img_ray(self, idx):
        for i, num in enumerate(self.num_rays):
            if idx < num:
                return i, idx
            idx -= num
        assert False

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "instance_mask": self.all_instance_masks[idx],
                "instance_mask_weight": self.all_instance_masks_weight[idx],
                "instance_ids": self.all_instance_ids[idx]
            }
        else:
            camera = self.cameras[idx]
            # c2w = camera.get_pose_matrix().squeeze().cpu().numpy().copy()
            # c2w[:3,3] /=self.scale_factor
            # c2w = self.convert_pose(c2w)
            c2w = self.all_c2w[idx]
            K = camera.get_intrinsics().squeeze().cpu().numpy()
            H, W = camera.get_image_size()[0].cpu().numpy()
            img_size = (H,W)
            H_new, W_new = int(H/self.downsample_factor), int(W/self.downsample_factor)
            focal = K[0,0]
            focal *=(H_new/H)

            directions = get_ray_directions(H_new, W_new, focal) # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            
            img = self.ds.images[idx]
            resize_transform = T.Resize((H_new, W_new), antialias=True)
            img = resize_transform(img.permute(2,0,1))
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            instance_mask = self.ds.co3d_masks[idx]
            instance_mask = resize_transform(instance_mask)
            instance_mask = (instance_mask>0).squeeze(0)
            instance_mask_weight = rebalance_mask_tensor(
                instance_mask,
                fg_weight=1.0,
                bg_weight=0.05,
            )
            #print("instance_mask, instance_mask_weight", instance_mask.shape, instance_mask_weight.shape)
            instance_mask = instance_mask.view(-1)
            instance_mask_weight = instance_mask_weight.view(-1)
            instance_ids = torch.ones_like(instance_mask).long() * 1

            rays = torch.cat([rays_o, rays_d, 
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])],
                            1) # (h*w, 8)
            sample = {
                "rays": rays,
                "rgbs": img,
                "instance_mask": instance_mask,
                "instance_mask_weight": instance_mask_weight,
                "instance_ids": instance_ids,
                "img_wh": (W_new, H_new)
            }
        return sample
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0


def to_global_path(path):
    # Converts a local path to a path relative to the source tree root
    path = Path(path)
    if not path.exists():
        path = Path(code_dir).joinpath(path)
    return path


def load_co3d(data_dir,category, instance):
    category = category
    dataset_root = to_global_path(data_dir)
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Category: {category}")
    logging.info(f"Instance: {instance}")
    frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
    sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")
    sequence_name = instance
    dataset = Co3dData(
        frame_annotations_file=frame_file,
        sequence_annotations_file=sequence_file,
        dataset_root=dataset_root,
        image_height=None,
        image_width=None,
        box_crop=False,
        load_point_clouds=True,
        pick_sequence=[sequence_name]
    )
    return dataset


def load_auto_bbox_scale(data_dir, category, instance, margin_scale, apply_scaling):
    path = os.path.join(data_dir, category, instance, "alignment.npy")
    # path = data_extra_dir.joinpath(cfg.category, cfg.instance, "alignment.npy")
    data = np.load(path, allow_pickle=True).item()
    T = data["T"]
    box = data["box_size"]

    s = np.zeros((4,4), np.float32)
    if apply_scaling:
        max_sz = np.max(box)
        scale = 2.0 / max_sz * margin_scale
        np.fill_diagonal(s, scale)
        s[3, 3] = 1.0
        total = s @ T
    else:
        np.fill_diagonal(s, 1.0)
        total = T

    return total, box, s


class Dataset:
    def __init__(self, data_dir,category, instance, split="train",
                use_auto_box=True, scaling_factor= 0.8, apply_scaling = False, 
<<<<<<< HEAD
                other=None, device='cuda'):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device(device)
=======
                other=None):
        super(Dataset, self).__init__()
        print('Load data: Begin')
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0
        # self.cfg = cfg

        if other is not None:
            co3d = other.co3d
        else:
            co3d = load_co3d(data_dir, category, instance)
        self.co3d = co3d

        point_cloud = co3d[0].sequence_point_cloud.to("cpu")
        self.point_cloud_xyz = point_cloud.points_padded()
        self.point_cloud_rgb = point_cloud.features_padded()
        self.point_cloud_quality_score = co3d[0].point_cloud_quality_score

        num_frames = len(co3d)
        self.n_images = num_frames

        self.pytorch3d_cameras = [co3d[idx].camera for idx in range(num_frames)]

        filenames = [co3d.frame_annots[i]["frame_annotation"].image.path for i in range(num_frames)]
        filenames = [os.path.split(f)[-1] for f in filenames]
        self.filenames = filenames

        auto_box = use_auto_box
        if auto_box:
            scaling_factor = scaling_factor
            scale_obj, box, scale = load_auto_bbox_scale(data_dir, category, instance, scaling_factor, apply_scaling)
            scale_obj = torch.from_numpy(scale_obj) # C2W
            self.bbox_scale_transform = scale

            max_sz = np.max(box)
            box_scaled = box / max_sz
            object_bbox_min = -box_scaled * np.array([1.0, 1.1, 1.0])
            object_bbox_max = box_scaled * np.array([1.0, 1.1, 1.1])
            self.raw_bbox_min = -box / max_sz * scaling_factor
            self.raw_bbox_max =  box / max_sz * scaling_factor

            pts = homogenise_torch(self.point_cloud_xyz.squeeze())
            pts = torch.einsum('ji,ni->nj', scale_obj, pts)[:, :3]

            bbox_min = torch.from_numpy(self.raw_bbox_min)
            bbox_max = torch.from_numpy(self.raw_bbox_max)

            mask = torch.cat([pts >= bbox_min, pts <= bbox_max], dim=1)
            mask = torch.all(mask, dim=-1)

<<<<<<< HEAD
            self.point_cloud_xyz_canonical = pts[mask].to(self.device)
=======
            # self.point_cloud_xyz_canonical = pts[mask].to(self.device)
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0
        else:
            scale_obj = None
            object_bbox_min = np.array([-1.01, -1.01, -1.01])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        self.global_alignment = scale_obj
        self.RT = scale_obj
        self.size = box

        self.images = [None] * num_frames
        self.masks = [None] * num_frames
        self.co3d_masks = [None] * num_frames
        self.co3d_depth_masks = [None] * num_frames
        self.co3d_depth_maps = [None] * num_frames
        self.cameras = []

        for idx in range(num_frames):
            frame_data = co3d[idx]
            img = frame_data.image_rgb
            self.images[idx] = img.permute(1, 2, 0).cpu()
            # self.masks[idx] = frame_data.fg_probability.cpu()
            self.masks[idx] = torch.ones_like(self.images[idx]).cpu()
            self.co3d_masks[idx] = frame_data.fg_probability
            self.co3d_depth_masks[idx] = frame_data.depth_mask
            self.co3d_depth_maps[idx] = frame_data.depth_map

            cam = frame_data.camera
            new_cam = PerspectiveCamera.from_pytorch3d(cam, img.shape[1:])
            if scale_obj is not None:
                new_cam = new_cam.left_transformed(scale_obj)
<<<<<<< HEAD
            new_cam.to(self.device)
=======
            # new_cam.to(self.device)
>>>>>>> 07e8a30f4c8670d06f3ae05f4394db30bff09ab0
            self.cameras.append(new_cam)
        
        # if cfg.trainval_split:
        #     data_extra_dir = to_global_path(cfg.data_extra_dir)
        #     path = data_extra_dir.joinpath(category, cfg.split_file)
        #     split_data = pickle.load(open(path, "rb"))
        #     instance = split_data[instance]
        #     ids = instance[split]
        #     self.n_images = len(ids)

        #     slice_ = lambda arr, idx: [arr[i] for i in idx]
        #     self.cameras = slice_(self.cameras, ids)
        #     self.images = slice_(self.images, ids)
        #     self.masks = slice_(self.masks, ids)
        #     self.filenames = slice_(self.filenames, ids)
        #     self.co3d_masks = slice_(self.co3d_masks, ids)
        #     self.co3d_depth_masks = slice_(self.co3d_depth_masks, ids)
        #     self.co3d_depth_maps = slice_(self.co3d_depth_maps, ids)

        print('Load data: End')

    def get_cameras(self):
        return self.cameras

    def get_ground_plane_z(self):
        return self.raw_bbox_min[2].item()

    # def near_far_from_sphere(self, rays):
    #     rays_o = rays.origins
    #     rays_d = rays.directions
    #     a = torch.sum(rays_d**2, dim=-1, keepdim=True).sqrt()
    #     b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    #     mid = 0.5 * (-b) / a
    #     far = mid + 1.0
    #     min_depth = float(self.cfg.min_depth)
    #     if min_depth == -1:
    #         near = mid - 1.0
    #     else:
    #         near = torch.ones_like(far) * min_depth
    #     return near, far