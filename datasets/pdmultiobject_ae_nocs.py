import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
import random
from models.nerfplusplus.helper import sample_rays_in_bbox
import cv2

# img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img_transform = T.Compose([T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def read_poses(pose_dir_train, pose_dir_val, img_files, output_boxes = False):
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

    for img_file in img_files:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    for img_file in img_files:
        c2w = np.array(data_val['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w = np.array(all_c2w)

    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
    all_c2w[:, :3, 3] *= pose_scale_factor

    all_c2w_train = all_c2w[:99, :, :]
    all_c2w_test = all_c2w[99:, :, :]

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
        return all_c2w_train, all_c2w_test, focal, img_wh, RTs
    else:
        return all_c2w_train, all_c2w_test, focal, img_wh

class PDMultiObject_AE_NOCS(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), white_back=False, model_type = "Vanilla"):
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = white_back
        self.base_dir = root_dir
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])

        # if self.split =='val':
        #     self.ids = self.ids[:10]

        #for multi scene training
        self.samples_per_epoch = 7000
        # for single scene training
        # self.samples_per_epoch = 70
        
        self.model_type = model_type
        #for object centric
        # self.near = 2.0
        # self.far = 6.0

        self.near = 0.2
        self.far = 3.0
        self.xyz_min = np.array([-2.7014, -2.6993, -2.2807]) 
        self.xyz_max = np.array([2.6986, 2.6889, 2.2192])

    def read_data(self, instance_dir, image_id):
        # base_dir = os.path.join(self.base_dir, instance_dir, self.split)
        base_dir = os.path.join(self.base_dir, instance_dir, 'train')
        img_files = os.listdir(os.path.join(base_dir, 'rgb'))
        img_files.sort()

        pose_dir_train = os.path.join(self.base_dir, instance_dir, 'train', 'pose')
        pose_dir_val = os.path.join(self.base_dir, instance_dir, 'val', 'pose')

        if self.split == 'train':
            #all_c2w, _,  focal, img_size = read_poses(pose_dir_train, pose_dir_val, img_files= img_files)
            all_c2w, _,  focal, img_size, self.RTs = read_poses(pose_dir_train, pose_dir_val, img_files= img_files, output_boxes=True)
        elif self.split == 'val':
            #all_c2w, _, focal, img_size = read_poses(pose_dir_train, pose_dir_val, img_files= img_files)
            all_c2w, _, focal, img_size, self.RTs = read_poses(pose_dir_train, pose_dir_val, img_files= img_files, output_boxes=True)
        
        w, h = self.img_wh       
        focal *=(w/img_size[0])  # modify focal length to match size self.img_wh

        c = np.array([640 / 2.0, 480/2.0])
        c*= (w/img_size[0]) 
        
        img_name = img_files[image_id]
            
        c2w = all_c2w[image_id]
        pose = torch.FloatTensor(c2w)
        c2w = torch.FloatTensor(c2w)[:3, :4]
        img = Image.open(os.path.join(base_dir, 'rgb', img_name))                
        img = img.resize((w,h), Image.LANCZOS)

        nocs_2d = Image.open(os.path.join(base_dir, 'nocs_2d', img_name))                
        nocs_2d = nocs_2d.resize((w,h), Image.LANCZOS)

        #Get masks
        seg_mask = Image.open(os.path.join(base_dir, 'semantic_segmentation_2d', img_name))
        # seg_mask = seg_mask.resize((w,h), Image.LANCZOS)
        seg_mask =  np.array(seg_mask)
        seg_mask[seg_mask!=5] =0
        seg_mask[seg_mask==5] =1
        seg_mask = cv2.resize(seg_mask, (w,h), interpolation =cv2.INTER_NEAREST)
        instance_mask = seg_mask >0

        directions = get_ray_directions(h, w, focal) # (h, w, 3)
        rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)

        return rays_o, view_dirs, rays_d, img, instance_mask, nocs_2d, radii, pose, torch.tensor(focal, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)

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
        if self.split == 'train': # use data in the buffers
            train_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[train_idx]
            
            imgs = list()
            nocs_2ds = list()
            masks = list()
            rays = list()
            view_dirs = list()
            rays_d = list()
            rgbs = list()
            radii = list()
            poses = list()
            focals = list()
            all_c = list()
            NV = 99
            src_views = 3
            ray_batch_size = 1024
        
            for train_image_id in range(0, NV):
                cam_rays, cam_view_dirs, cam_rays_d, img, instance_mask, nocs_2d, camera_radii, c2w, f, c =  self.read_data(instance_dir, train_image_id)
                img = Image.fromarray(np.uint8(img))
                instance_mask = T.ToTensor()(instance_mask)
                img = T.ToTensor()(img)

                nocs_2d = Image.fromarray(np.uint8(nocs_2d))
                nocs_2d = T.ToTensor()(nocs_2d)

                _, H, W = img.shape
                camera_radii = torch.FloatTensor(camera_radii)
                cam_rays = torch.FloatTensor(cam_rays)
                cam_view_dirs = torch.FloatTensor(cam_view_dirs)
                cam_rays_d = torch.FloatTensor(cam_rays_d)

                rgb_gt = img.permute(1,2,0).flatten(0,1)
                mask_gt = instance_mask.permute(1,2,0).flatten(0,1)
                radii_gt = camera_radii.view(-1)
                ray = cam_rays.view(-1, cam_rays.shape[-1])
                ray_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
                viewdir = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

                imgs.append(
                    img_transform(img)
                )
                nocs_2ds.append(nocs_2d)
                masks.append(mask_gt)
                rays.append(ray)
                view_dirs.append(viewdir)
                rays_d.append(ray_d)
                rgbs.append(rgb_gt)
                radii.append(radii_gt)
                poses.append(c2w)
                focals.append(f)
                all_c.append(c)
            
            imgs = torch.stack(imgs, 0)
            masks = torch.stack(masks, 0)
            poses = torch.stack(poses, 0)
            focals = torch.stack(focals, 0)
            all_c = torch.stack(all_c, 0)
            src_views_num = np.random.choice(99, src_views, replace=False)

            imgs = imgs[src_views_num, :]
            poses = poses[src_views_num, :]
            focals = focals[src_views_num]
            all_c = all_c[src_views_num, :]
  
            rgbs = torch.stack(rgbs, 0)
            nocs_2ds = torch.stack(nocs_2ds, 0)
            rays = torch.stack(rays, 0)  
            rays_d = torch.stack(rays_d, 0) 
            view_dirs = torch.stack(view_dirs, 0)  
            radii = torch.stack(radii, 0)  

            pix_inds = torch.randint(0, NV * H * W, (ray_batch_size,))
            rgbs = rgbs.reshape(-1,3)[pix_inds,...] 
            nocs_2ds = nocs_2ds.reshape(-1,3)[pix_inds,...] 
            masks = masks.reshape(-1,1)[pix_inds]
            radii = radii.reshape(-1,1)[pix_inds]
            rays = rays.reshape(-1,3)[pix_inds]
            rays_d = rays_d.reshape(-1,3)[pix_inds]
            view_dirs = view_dirs.reshape(-1,3)[pix_inds]
            
            near_obj, far_obj, _ = sample_rays_in_bbox(self.RTs, rays.numpy(), view_dirs.numpy())

            if self.model_type == "Vanilla":
                sample = {
                    "src_imgs": imgs,
                    "rays": rays,
                    "rgbs": rgbs,
                }
            else:
                sample = {}
                sample["src_imgs"] = imgs
                sample["src_poses"] = poses
                sample["src_focal"] = focals
                sample["src_c"] = all_c
                sample["near_obj"] = near_obj
                sample["far_obj"] = far_obj
                sample["instance_mask"] = masks
                sample["rays_o"] = rays
                sample["rays_d"] = rays_d
                sample["viewdirs"] = view_dirs
                sample["target"] = rgbs
                sample["nocs_2d"] = nocs_2ds
                sample["radii"] = radii
                sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = torch.zeros_like(sample["rays_o"])
                
            return sample
                
        # elif self.split == 'val': # create data for each image separately
        elif self.split=='val':
            instance_dir = self.ids[idx]
            imgs = list()
            nocs_2ds = list()
            masks = list()
            rays = list()
            view_dirs = list()
            rays_d = list()
            rgbs = list()
            radii = list()
            poses = list()
            focals = list()
            all_c = list()
            NV = 99
            src_views = 3
            for train_image_id in range(0, NV):
                cam_rays, cam_view_dirs, cam_rays_d, img, instance_mask, nocs_2d, camera_radii, c2w, f, c =  self.read_data(instance_dir, train_image_id)
                img = Image.fromarray(np.uint8(img))
                instance_mask = T.ToTensor()(instance_mask)
                img = T.ToTensor()(img)

                nocs_2d = Image.fromarray(np.uint8(nocs_2d))
                nocs_2d = T.ToTensor()(nocs_2d)

                _, H, W = img.shape
                camera_radii = torch.FloatTensor(camera_radii)
                cam_rays = torch.FloatTensor(cam_rays)
                cam_view_dirs = torch.FloatTensor(cam_view_dirs)
                cam_rays_d = torch.FloatTensor(cam_rays_d)

                rgb_gt = img.permute(1,2,0).flatten(0,1)
                mask_gt = instance_mask.permute(1,2,0).flatten(0,1)
                radii_gt = camera_radii.view(-1)
                ray = cam_rays.view(-1, cam_rays.shape[-1])
                ray_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
                viewdir = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

                imgs.append(
                    img_transform(img)
                )
                nocs_2ds.append(nocs_2d)
                masks.append(mask_gt)
                rays.append(ray)
                view_dirs.append(viewdir)
                rays_d.append(ray_d)
                rgbs.append(rgb_gt)
                radii.append(radii_gt)
                poses.append(c2w)
                focals.append(f)
                all_c.append(c)
            
            imgs = torch.stack(imgs, 0)
            poses = torch.stack(poses, 0)
            focals = torch.stack(focals, 0)
            all_c = torch.stack(all_c, 0)

            # src_views_num = np.random.choice(99, 3, replace=False)
            src_views_num = np.sort(np.random.choice(NV, src_views, replace=False))
            # src_views_num = np.random.randint(0, 15, 3)

            dest_view_num = np.random.randint(0, NV - src_views)
            for vs in range(src_views):
                dest_view_num += dest_view_num >= src_views_num[vs]

            dest_view_num = [dest_view_num]

            imgs = imgs[src_views_num, :]
            poses = poses[src_views_num, :]
            focals = focals[src_views_num]
            all_c = all_c[src_views_num, :]
  
            rgbs = torch.stack(rgbs, 0)
            nocs_2ds = torch.stack(nocs_2ds, 0)
            masks = torch.stack(masks, 0)
            rays = torch.stack(rays, 0)  
            rays_d = torch.stack(rays_d, 0) 
            view_dirs = torch.stack(view_dirs, 0)  
            radii = torch.stack(radii, 0)  

            rgbs = rgbs[dest_view_num].squeeze(0)
            nocs_2ds = nocs_2ds[dest_view_num].squeeze(0)
            masks = masks[dest_view_num].squeeze(0)
            radii = radii[dest_view_num].squeeze(0)
            rays = rays[dest_view_num].squeeze(0)
            rays_d = rays_d[dest_view_num].squeeze(0)
            view_dirs = view_dirs[dest_view_num].squeeze(0)

            near_obj, far_obj, _ = sample_rays_in_bbox(self.RTs, rays.numpy(), view_dirs.numpy())
            
            if self.model_type == "Vanilla":
                sample = {
                    "src_imgs": imgs,
                    "rays": rays,
                    "rgbs": rgbs,
                }
            else:
                sample = {}
                sample["src_imgs"] = imgs
                sample["src_poses"] = poses
                sample["src_focal"] = focals
                sample["near_obj"] = near_obj
                sample["far_obj"] = far_obj
                sample["instance_mask"] = masks
                sample["src_c"] = all_c
                sample["rays_o"] = rays
                sample["rays_d"] = rays_d
                sample["viewdirs"] = view_dirs
                sample["target"] = rgbs
                sample["nocs_2d"] = nocs_2ds
                sample["radii"] = radii
                sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = torch.zeros_like(sample["rays_o"])
                
            return sample