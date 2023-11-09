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

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    print("radius", radius)
    radius = 1.3
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)

class GoogleScannedDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1600, 1600)):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True
        # self.focal = 1931.371337890625
        # self.focal *= self.img_wh[0]/1600 # modify focal length to match size self.img_wh

    def read_meta(self):

        self.base_dir = os.path.join(self.root_dir, '00000')
        json_files = [pos_json for pos_json in os.listdir(self.base_dir) if pos_json.endswith('.json')]        
        json_files.sort()
        self.meta = json_files
        w, h = self.img_wh

        self.focal = 0.5*800/np.tan(0.5*0.785398) # original focal length
                                                                     # when W=800
        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
        # bounds, common for all scenes
        self.near = 0.03
        self.far = 4.5
        self.bounds = np.array([self.near, self.far])
        self.instance_ids = np.arange(1,21,1)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_instance_ids = []
            self.all_instance_ids = []
            self.all_valid_masks = []

            # self.instance_ids = []

            for i, json_file in enumerate(self.meta):
                if i>10:
                    continue
                file = os.path.join(self.base_dir, json_file)
                with open(file, 'r') as f:
                    data = json.loads(f.read())
                cam2world = data["camera_data"]["cam2world"]
                #intrinsics = data["camera_data"]["intrinsics"]
                # focal = intrinsics["fx"]
                directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
                c2w = np.array(cam2world).T
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, rays_d = get_rays(directions, c2w)
                img_name = json_file.split('.')[0]+ '.png'
                seg_name = json_file.split('.')[0]+ '.seg.png'
                img = Image.open(os.path.join(self.base_dir, img_name))
                #img = img.resize(self.img_wh, Image.LANCZOS)
                # seg_masks = cv2.resize(cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH),
                #                         self.img_wh, interpolation=cv2.INTER_NEAREST)

                seg_masks = cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH)

                valid_mask = seg_masks>0
                valid_mask = self.transform(valid_mask).view(
                        -1)           
                # img = load_image_from_exr(os.path.join(self.base_dir, img_name))
                # seg_masks = load_seg_from_exr(os.path.join(self.base_dir, seg_name))
                img = self.transform(img) # (h, w, 3)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
                curr_frame_instance_masks = []
                curr_frame_instance_masks_weight = []
                curr_frame_instance_ids = []
                # Load masks for each objects
                for id in range(len(data["objects"])):
                    segmentation_id = data["objects"][id]["segmentation_id"]
                    # self.instance_ids.append(segmentation_id)
                    instance_mask = seg_masks == segmentation_id
                    if instance_mask.sum() <100:
                        continue
                    instance_mask_weight = rebalance_mask(
                        instance_mask,
                        fg_weight=1.0,
                        bg_weight=0.005,
                    )
                    instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                        -1), self.transform(instance_mask_weight).view(-1)
                    instance_ids = torch.ones_like(instance_mask).long() * segmentation_id
                    curr_frame_instance_masks += [instance_mask]
                    curr_frame_instance_masks_weight += [instance_mask_weight]
                    curr_frame_instance_ids += [instance_ids]
                # self.all_current_frame_ids = len(curr_frame_instance_ids)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1)] # (h*w, 8)
                self.all_rgbs += [img]
                self.all_valid_masks+=[valid_mask]
                print("torch.stack(curr_frame_instance_masks, -1)", torch.stack(curr_frame_instance_masks, -1).shape)
                self.all_instance_masks += [torch.stack(curr_frame_instance_masks, -1)]
                self.all_instance_masks_weight += [
                    torch.stack(curr_frame_instance_masks_weight, -1)
                ]
                self.all_instance_ids += [torch.stack(curr_frame_instance_ids, -1)]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_valid_masks = torch.cat(self.all_valid_masks, 0) # (len(self.meta['frames])*h*w, 3)
            # self.all_instance_masks = torch.cat(self.all_instance_masks, 0)  # (len(self.meta['frames])*h*w)
            # self.all_instance_masks_weight = torch.cat(self.all_instance_masks_weight, 0)  # (len(self.meta['frames])*h*w)
            # self.all_instance_ids = torch.cat(self.all_instance_ids, 0).long()  # (len(self.meta['frames])*h*w)
        elif self.split =='test':
                #1.0 is radius of hemisphere
                #radius = 1.2 * 1.0
                self.poses_test = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
                #self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 1 # only validate 8 images (to support <=8 gpus)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            #print("idx", idx)
            # print("self.all_instance_masks", self.all_instance_masks.shape)
            img_size = self.img_wh[0]*self.img_wh[1]
            frame_idx = int(idx/img_size)
            frame_idx_sample = idx%img_size

            #print("frame_idx, frame_idx_sample",frame_idx, frame_idx_sample)
            #print("self.instance_ids", len(self.instance_ids), self.all_current_frame_ids)

            # print("self.all_instance_masks", len(self.all_instance_masks), len(self.all_instance_masks[0]))
            # print("self.all_current_frame_ids", self.all_current_frame_ids)
            #rand_instance_id = torch.randint(0, self.all_current_frame_ids, (1,))
            rand_id = self.all_instance_masks[frame_idx].shape[1]
            #print("rand_id", self.all_instance_masks[frame_idx].shape[1], self.all_instance_masks[frame_idx].shape)
            rand_instance_id = torch.randint(0, rand_id, (1,))
            #print("self.all_instance_masks", self.all_instance_masks_weight[frame_idx][frame_idx_sample, rand_instance_id].shape)
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "instance_mask": self.all_instance_masks[frame_idx][frame_idx_sample, rand_instance_id],
                "instance_mask_weight": self.all_instance_masks_weight[frame_idx][frame_idx_sample, rand_instance_id],
                "instance_ids": self.all_instance_ids[frame_idx][frame_idx_sample, rand_instance_id],
                "valid_mask": self.all_valid_masks[idx]
            }
            # sample = {
            #     "rays": self.all_rays[idx],
            #     "rgbs": self.all_rgbs[idx],
            #     "instance_mask": self.all_instance_masks[idx, rand_instance_id],
            #     "instance_mask_weight": self.all_instance_masks_weight[idx, rand_instance_id],
            #     "instance_ids": self.all_instance_ids[idx, rand_instance_id],
            #     "valid_mask": self.all_valid_masks[idx]
            # }
        elif self.split == 'val': # create data for each image separately
            val_idx = 100
            json_file = self.meta[val_idx]
            w, h = self.img_wh
            file = os.path.join(self.base_dir, json_file)
            with open(file, 'r') as f:
                data = json.loads(f.read())
            cam2world = data["camera_data"]["cam2world"]
            intrinsics = data["camera_data"]["intrinsics"]
            # focal = intrinsics["fx"]
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            c2w = np.array(cam2world).T
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)
            img_name = json_file.split('.')[0]+ '.png'
            seg_name = json_file.split('.')[0]+ '.seg.png'
            
            # img = load_image_from_exr(os.path.join(self.base_dir, img_name))
            # seg_masks = load_seg_from_exr(os.path.join(self.base_dir, seg_name))
            img = Image.open(os.path.join(self.base_dir, img_name))
            #img = img.resize(self.img_wh, Image.LANCZOS)
            #seg_masks = cv2.resize(cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH),
                                    # self.img_wh, interpolation=cv2.INTER_NEAREST)
            seg_masks = cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH)
            # valid_mask = (seg_masks>0).flatten()
            valid_mask = seg_masks>0
            valid_mask = self.transform(valid_mask).view(
                    -1)  
            img = self.transform(img) # (h, w, 3)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
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
                "instance_ids": instance_id_out,
                "valid_mask": valid_mask
            }  
        else:
            w, h = self.img_wh
            print("self.poses_test", self.poses_test.shape)
            c2w = torch.FloatTensor(self.poses_test[idx])[:3,:4]
            
            print("c2w", c2w.shape)

            
            directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
            print("directions", directions.shape)
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            sample = {'rays': rays,
                      'c2w': c2w}   

        return sample

# import torch
# from torch.utils.data import Dataset
# import json
# import numpy as np
# import os
# from PIL import Image
# from torchvision import transforms as T
# import cv2

# # from .google_scanned_utils import load_image_from_exr, load_seg_from_exr

# from .ray_utils import *
# from .nocs_utils import rebalance_mask

# trans_t = lambda t : torch.Tensor([
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,t],
#     [0,0,0,1]]).float()

# rot_phi = lambda phi : torch.Tensor([
#     [1,0,0,0],
#     [0,np.cos(phi),-np.sin(phi),0],
#     [0,np.sin(phi), np.cos(phi),0],
#     [0,0,0,1]]).float()

# rot_theta = lambda th : torch.Tensor([
#     [np.cos(th),0,-np.sin(th),0],
#     [0,1,0,0],
#     [np.sin(th),0, np.cos(th),0],
#     [0,0,0,1]]).float()

# def pose_spherical(theta, phi, radius):
#     print("radius", radius)
#     radius = 1.3
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#     return c2w

# def create_spheric_poses(radius, n_poses=120):
#     """
#     Create circular poses around z axis.
#     Inputs:
#         radius: the (negative) height and the radius of the circle.

#     Outputs:
#         spheric_poses: (n_poses, 3, 4) the poses in the circular path
#     """
#     def spheric_pose(theta, phi, radius):
#         trans_t = lambda t : np.array([
#             [1,0,0,0],
#             [0,1,0,-0.9*t],
#             [0,0,1,t],
#             [0,0,0,1],
#         ])

#         rot_phi = lambda phi : np.array([
#             [1,0,0,0],
#             [0,np.cos(phi),-np.sin(phi),0],
#             [0,np.sin(phi), np.cos(phi),0],
#             [0,0,0,1],
#         ])

#         rot_theta = lambda th : np.array([
#             [np.cos(th),0,-np.sin(th),0],
#             [0,1,0,0],
#             [np.sin(th),0, np.cos(th),0],
#             [0,0,0,1],
#         ])

#         c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
#         c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
#         return c2w[:3]

#     spheric_poses = []
#     for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
#         spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
#     return np.stack(spheric_poses, 0)

# class GoogleScannedDataset(Dataset):
#     def __init__(self, root_dir, split='train', img_wh=(1600, 1600)):
#         self.root_dir = root_dir
#         self.split = split
#         print("img_wh", img_wh)
#         assert img_wh[0] == img_wh[1], 'image width must equal image height!'
#         self.img_wh = img_wh
#         self.define_transforms()

#         self.read_meta()
#         self.white_back = True
#         # self.focal = 1931.371337890625
#         # self.focal *= self.img_wh[0]/1600 # modify focal length to match size self.img_wh

#     def read_meta(self):

#         self.base_dir = os.path.join(self.root_dir, '00000')
#         json_files = [pos_json for pos_json in os.listdir(self.base_dir) if pos_json.endswith('.json')]        
#         json_files.sort()
#         self.meta = json_files
#         w, h = self.img_wh

#         self.focal = 0.5*800/np.tan(0.5*0.785398) # original focal length
#                                                                      # when W=800
#         self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
#         # bounds, common for all scenes
#         self.near = 0.03
#         self.far = 4.5
#         self.bounds = np.array([self.near, self.far])
#         self.instance_ids = np.arange(1,21,1)
            
#         if self.split == 'train': # create buffer of all rays and rgb data
#             self.image_paths = []
#             self.poses = []
#             self.all_rays = []
#             self.all_rgbs = []
#             self.all_instance_masks = []
#             self.all_instance_masks_weight = []
#             self.all_instance_ids = []
#             self.all_instance_ids = []
#             self.all_valid_masks = []

#             # self.instance_ids = []

#             for i, json_file in enumerate(self.meta):
#                 if i>10:
#                     continue
#                 file = os.path.join(self.base_dir, json_file)
#                 with open(file, 'r') as f:
#                     data = json.loads(f.read())
#                 cam2world = data["camera_data"]["cam2world"]
#                 #intrinsics = data["camera_data"]["intrinsics"]
#                 # focal = intrinsics["fx"]
#                 directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
#                 c2w = np.array(cam2world).T
#                 c2w = torch.FloatTensor(c2w)[:3, :4]
#                 rays_o, rays_d = get_rays(directions, c2w)
#                 img_name = json_file.split('.')[0]+ '.png'
#                 seg_name = json_file.split('.')[0]+ '.seg.png'
#                 img = Image.open(os.path.join(self.base_dir, img_name))
#                 #img = img.resize(self.img_wh, Image.LANCZOS)
#                 # seg_masks = cv2.resize(cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH),
#                 #                         self.img_wh, interpolation=cv2.INTER_NEAREST)

#                 seg_masks = cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH)

#                 valid_mask = seg_masks>0
#                 valid_mask = self.transform(valid_mask).view(
#                         -1)           
#                 # img = load_image_from_exr(os.path.join(self.base_dir, img_name))
#                 # seg_masks = load_seg_from_exr(os.path.join(self.base_dir, seg_name))
#                 img = self.transform(img) # (h, w, 3)
#                 img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
#                 curr_frame_instance_masks = []
#                 curr_frame_instance_masks_weight = []
#                 curr_frame_instance_ids = []
#                 # Load masks for each objects
#                 for id in range(len(data["objects"])):
#                     segmentation_id = data["objects"][id]["segmentation_id"]
#                     # self.instance_ids.append(segmentation_id)
#                     instance_mask = seg_masks == segmentation_id
#                     # if instance_mask.sum() <100:
#                     #     continue
#                     instance_mask_weight = rebalance_mask(
#                         instance_mask,
#                         fg_weight=1.0,
#                         bg_weight=0.005,
#                     )
#                     instance_mask, instance_mask_weight = self.transform(instance_mask).view(
#                         -1), self.transform(instance_mask_weight).view(-1)
#                     instance_ids = torch.ones_like(instance_mask).long() * segmentation_id
#                     curr_frame_instance_masks += [instance_mask]
#                     curr_frame_instance_masks_weight += [instance_mask_weight]
#                     curr_frame_instance_ids += [instance_ids]
#                 self.all_current_frame_ids = len(curr_frame_instance_ids)

#                 self.all_rays += [torch.cat([rays_o, rays_d, 
#                                 self.near*torch.ones_like(rays_o[:, :1]),
#                                 self.far*torch.ones_like(rays_o[:, :1])],
#                                 1)] # (h*w, 8)
#                 self.all_rgbs += [img]
#                 self.all_valid_masks+=[valid_mask]
#                 self.all_instance_masks += [torch.stack(curr_frame_instance_masks, -1)]
#                 self.all_instance_masks_weight += [
#                     torch.stack(curr_frame_instance_masks_weight, -1)
#                 ]
#                 self.all_instance_ids += [torch.stack(curr_frame_instance_ids, -1)]

#             self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
#             self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
#             self.all_valid_masks = torch.cat(self.all_valid_masks, 0) # (len(self.meta['frames])*h*w, 3)
#             self.all_instance_masks = torch.cat(self.all_instance_masks, 0)  # (len(self.meta['frames])*h*w)
#             self.all_instance_masks_weight = torch.cat(self.all_instance_masks_weight, 0)  # (len(self.meta['frames])*h*w)
#             self.all_instance_ids = torch.cat(self.all_instance_ids, 0).long()  # (len(self.meta['frames])*h*w)
#         elif self.split =='test':
#                 #1.0 is radius of hemisphere
#                 #radius = 1.2 * 1.0
#                 self.poses_test = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
#                 #self.poses_test = create_spheric_poses(radius)

#     def define_transforms(self):
#         self.transform = T.ToTensor()

#     def __len__(self):
#         if self.split == 'train':
#             return len(self.all_rays)
#         if self.split == 'val':
#             return 1 # only validate 8 images (to support <=8 gpus)
#         return len(self.poses_test)

#     def __getitem__(self, idx):
#         if self.split == 'train': # use data in the buffers
#             rand_instance_id = torch.randint(0, len(self.instance_ids), (1,))
#             sample = {
#                 "rays": self.all_rays[idx],
#                 "rgbs": self.all_rgbs[idx],
#                 "instance_mask": self.all_instance_masks[idx,rand_instance_id],
#                 "instance_mask_weight": self.all_instance_masks_weight[idx,rand_instance_id],
#                 "instance_ids": self.all_instance_ids[idx,rand_instance_id],
#                 "valid_mask": self.all_valid_masks[idx]
#             }
#         elif self.split == 'val': # create data for each image separately
#             val_idx = 100
#             json_file = self.meta[val_idx]
#             w, h = self.img_wh
#             file = os.path.join(self.base_dir, json_file)
#             with open(file, 'r') as f:
#                 data = json.loads(f.read())
#             cam2world = data["camera_data"]["cam2world"]
#             intrinsics = data["camera_data"]["intrinsics"]
#             # focal = intrinsics["fx"]
#             directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
#             c2w = np.array(cam2world).T
#             c2w = torch.FloatTensor(c2w)[:3, :4]
#             rays_o, rays_d = get_rays(directions, c2w)
#             img_name = json_file.split('.')[0]+ '.png'
#             seg_name = json_file.split('.')[0]+ '.seg.png'
            
#             # img = load_image_from_exr(os.path.join(self.base_dir, img_name))
#             # seg_masks = load_seg_from_exr(os.path.join(self.base_dir, seg_name))
#             img = Image.open(os.path.join(self.base_dir, img_name))
#             #img = img.resize(self.img_wh, Image.LANCZOS)
#             #seg_masks = cv2.resize(cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH),
#                                     # self.img_wh, interpolation=cv2.INTER_NEAREST)
#             seg_masks = cv2.imread(os.path.join(self.base_dir, seg_name), cv2.IMREAD_ANYDEPTH)
#             # valid_mask = (seg_masks>0).flatten()
#             valid_mask = seg_masks>0
#             valid_mask = self.transform(valid_mask).view(
#                     -1)  
#             img = self.transform(img) # (h, w, 3)
#             img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

#             directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
#             rays_o, rays_d = get_rays(directions, c2w)
#             rays = torch.cat([rays_o, rays_d, 
#                               self.near*torch.ones_like(rays_o[:, :1]),
#                               self.far*torch.ones_like(rays_o[:, :1])],
#                               1) # (H*W, 8)
#             val_inst_id = 7
#             for i_inst, instance_id in enumerate(self.instance_ids):
#                 if instance_id != val_inst_id:
#                     continue
#                 instance_mask = seg_masks == instance_id 
#                 instance_mask_weight = rebalance_mask(
#                     instance_mask,
#                     fg_weight=1.0,
#                     bg_weight=0.005,
#                 )
#                 instance_mask, instance_mask_weight = self.transform(instance_mask).view(
#                     -1), self.transform(instance_mask_weight).view(-1)
#                 instance_id_out = torch.ones_like(instance_mask).long() * instance_id
#             sample = {
#                 "rays": rays,
#                 "rgbs": img,
#                 "instance_mask": instance_mask,
#                 "instance_mask_weight": instance_mask_weight,
#                 "instance_ids": instance_id_out,
#                 "valid_mask": valid_mask
#             }  
#         else:
#             w, h = self.img_wh
#             print("self.poses_test", self.poses_test.shape)
#             c2w = torch.FloatTensor(self.poses_test[idx])[:3,:4]
            
#             print("c2w", c2w.shape)

            
#             directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
#             print("directions", directions.shape)
#             rays_o, rays_d = get_rays(directions, c2w)
#             rays = torch.cat([rays_o, rays_d, 
#                               self.near*torch.ones_like(rays_o[:, :1]),
#                               self.far*torch.ones_like(rays_o[:, :1])],
#                               1) # (H*W, 8)
#             sample = {'rays': rays,
#                       'c2w': c2w}   

#         return sample