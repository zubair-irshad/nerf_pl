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
import pickle

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

# img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def rot_from_origin(c2w,rotation=10):
    rot = c2w[:3,:3]
    pos = c2w[:3,-1:]
    rot_mat = get_rotation_matrix(rotation)
    pos = torch.mm(rot_mat, pos)
    rot = torch.mm(rot_mat, rot)
    c2w = torch.cat((rot, pos), -1)
    return c2w

def get_rotation_matrix(rotation):
    #if iter_ is not None:
    #    rotation = self.near_c2w_rot * (self.smoothing_rate **(int(iter_/self.smoothing_step_size)))
    #else: 

    phi = (rotation*(np.pi / 180.))
    x = np.random.uniform(-phi, phi)
    y = np.random.uniform(-phi, phi)
    z = np.random.uniform(-phi, phi)
    
    rot_x = torch.Tensor([
                [1,0,0],
                [0,np.cos(x),-np.sin(x)],
                [0,np.sin(x), np.cos(x)]
                ])
    rot_y = torch.Tensor([
                [np.cos(y),0,-np.sin(y)],
                [0,1,0],
                [np.sin(y),0, np.cos(y)]
                ])
    rot_z = torch.Tensor([
                [np.cos(z),-np.sin(z),0],
                [np.sin(z),np.cos(z),0],
                [0,0,1],
                ])
    rot_mat = torch.mm(rot_x, torch.mm(rot_y, rot_z))
    return rot_mat

TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision
def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists

def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))

def get_nearest_pose_ids(tar_pose, ref_poses, num_select=4, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


# def select_key_frames(num_cameras, source_cameras, rel_translations, rel_rotations, tmax, Rmax):
#     # Create boolean masks for source cameras and valid destination cameras
#     is_source_camera = np.zeros(num_cameras, dtype=bool)
#     is_source_camera[source_cameras] = True
#     is_valid_dest_camera = ~is_source_camera
    
#     # Create boolean masks for pairs of cameras that satisfy the conditions
#     trans_mask = np.linalg.norm(rel_translations, axis=-1) < tmax
#     rot_mask = np.abs(rel_rotations) < Rmax
    
#     # Use boolean masks to select valid pairs of cameras
#     valid_pairs = np.logical_and(np.logical_and(trans_mask, rot_mask), np.outer(is_source_camera, is_valid_dest_camera))
    
#     # Find the indices of valid destination cameras
#     key_frames = np.where(np.any(valid_pairs, axis=0))[0].tolist()
    
#     # Return the list of key frames
#     return key_frames

def read_poses(pose_dir_train, img_files_train, output_boxes = False, contract = True):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data['focal']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))
    all_c2w_train[:, :3, 3] *= pose_scale_factor

    # # Trying praful's suggestion for normalizin all cameras by the first camera
        # So we don't need to keep track of c2w_0 while creating a world grid
        # During indexing into the world grid, we don't need to multiple by c2w_0 which is always idenitity
    # if contract:
    #     c2w = all_c2w_train[49]
    #     all_c2w_train_transformed = []
    #     for i in range(all_c2w_train.shape[0]):
    #         all_c2w_train_transformed.append(np.linalg.inv(c2w)@all_c2w_train[i,:,:])
    #     all_c2w_train = np.array(all_c2w_train_transformed)

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
        return all_c2w_train, all_c2w_val, focal, img_wh, RTs
    else:
        return all_c2w_train, all_c2w_val, focal, img_wh

# def read_poses(pose_dir_train, pose_dir_val, img_files_train, img_files_val, output_boxes = False):
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
#     all_c2w_train = []
#     all_c2w_test = []
#     for img_file in img_files_train:
#         c2w = np.array(data['transform'][img_file.split('.')[0]])
#         c2w[:3, 3] = c2w[:3, 3] - obj_location
#         all_c2w.append(convert_pose_PD_to_NeRF(c2w))
#         all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))
#     for img_file in img_files_val:
#         c2w = np.array(data_val['transform'][img_file.split('.')[0]])
#         c2w[:3, 3] = c2w[:3, 3] - obj_location
#         all_c2w.append(convert_pose_PD_to_NeRF(c2w))
#         all_c2w_test.append(convert_pose_PD_to_NeRF(c2w))
#     all_c2w = np.array(all_c2w)
#     all_c2w_train = np.array(all_c2w_train)
#     all_c2w_test = np.array(all_c2w_test)
#     pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
#     all_c2w_train[:, :3, 3] *= pose_scale_factor
#     all_c2w_test[:, :3, 3] *= pose_scale_factor
#     all_c2w_val = all_c2w_train[100:]
#     all_c2w_train = all_c2w_train[:100]
#     # Get bounding boxes for object MLP training only
#     if output_boxes:
#         all_boxes = []
#         all_translations= []
#         all_rotations = []
#         for k,v in data['bbox_dimensions'].items():
#                 bbox = np.array(v)
#                 all_boxes.append(bbox*pose_scale_factor)
#                 #New scene 200 uncomment here
#                 all_rotations.append(data["obj_rotations"][k])
#                 translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor
#                 all_translations.append(translation)
#         # Old scenes uncomment here
#         # all_translations = (np.array(data['obj_translations'])- obj_location)*pose_scale_factor
#         # all_rotations = data["obj_rotations"]
#         RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
#         return all_c2w_train, all_c2w_val, all_c2w_test, focal, img_wh, RTs
#     else:
#         return all_c2w_train, all_c2w_val, all_c2w_test, focal, img_wh

class PDMultiObject_AE_CV(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), 
                 white_back=False, model_type = "Vanilla",  
                 eval_inference=None, optimize = None, encoder_type = 'resnet', contract = True,
                 finetune_lpips = False):
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = white_back
        self.base_dir = root_dir
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        self.eval_inference = eval_inference
        self.optimize = optimize
        self.encoder_type = encoder_type
        self.contract = contract
        self.finetune_lpips = finetune_lpips

        # if self.contract:
        #     print("===========================\n\n")
        #     print("Dataloader: Contract samples Ref Camera, 49\n\n\n\n")
        #     print("===========================\n\n")
        # if self.split =='val':
        #     self.ids = self.ids[:10]


        # with open(os.path.join(os.path.dirname(self.base_dir), "relative_rot_trans_dict.pkl"), "rb") as f:
        #     relative_rot_trans_dict = pickle.load(f)

        # self.rel_rotations_train = relative_rot_trans_dict["rel_rotations_train"]
        # self.rel_translations_train = relative_rot_trans_dict["rel_translations_train"]
        # self.rel_translations_val = relative_rot_trans_dict["rel_translations_val"]
        # self.rel_rotations_val = relative_rot_trans_dict["rel_rotations_val"]

        #for multi scene training
        if self.encoder_type == 'resnet':
            self.img_transform = T.Compose([T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            # for custom CNN MVS nerf style
            self.img_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        if self.encoder_type == 'resnet':
            #roughly 10 epochs to see one full data i.e. 210*240*100*25
            self.samples_per_epoch = 9600
            # self.samples_per_epoch = 1875
        else:
            self.samples_per_epoch = 9600
            # self.samples_per_epoch = 1875
            #back to 10 epochs to see one full data due to increase sampling size i.e. 128 and 256 for coarse and fine
            # roughly 2 epochs to see one full data i.e. 210*240*100*25

        # self.samples_per_epoch = 1000
        # 
        w, h = self.img_wh
        if self.eval_inference is not None:
            # eval_num = int(self.eval_inference[0])
            # num =  99 - eval_num
            num = 99
            self.image_sizes = np.array([[h, w] for i in range(num)])
            # self.image_sizes = np.array([[h, w] for i in range(5)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])
            # self.image_sizes = np.array([[h, w] for i in range(100)])
        # for single scene training
        # self.samples_per_epoch = 70
        
        self.model_type = model_type
        #for object centric
        # self.near = 2.0
        # self.far = 6.0

        self.near = 0.2
        self.far = 2.5

# self.xyz_min, self.xyz_max tensor([-2.4419, -2.4701, -2.3195]) tensor([2.4696, 2.4467, 1.5973])

        self.xyz_min = np.array([-2.4419, -2.4701, -2.3195]) 
        self.xyz_max = np.array([2.4696, 2.4467, 1.5973])

        # self.xyz_min = np.array([-2.7014, -2.6993, -2.2807]) 
        # self.xyz_max = np.array([2.6986, 2.6889, 2.2192])



    def read_data(self, instance_dir, image_id, out_instance_seg = False, contract = True, smoothing_loss=False, out_src_view=False):
        base_dir_train = os.path.join(self.base_dir, instance_dir, 'train')
        img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
        img_files_train.sort()
        pose_dir_train = os.path.join(self.base_dir, instance_dir, 'train', 'pose')

        if self.split == 'train':
            all_c2w, _, focal, img_size = read_poses(pose_dir_train, img_files_train, output_boxes=False, contract = contract)
            img_files = img_files_train[:100]
            base_dir = base_dir_train
        elif self.split == 'val':
            all_c2w_train, all_c2w_val, focal, img_size = read_poses(pose_dir_train, img_files_train, output_boxes=False, contract = contract)
            all_c2w =  np.concatenate((all_c2w_train, all_c2w_val), axis=0)
            img_files = img_files_train
            base_dir = base_dir_train

        w, h = self.img_wh       
        focal *=(w/img_size[0])  # modify focal length to match size self.img_wh

        c = np.array([640 / 2.0, 480/2.0])
        c*= (w/img_size[0])

        intrinsics = torch.FloatTensor([
            [focal, 0., c[0]],
            [0., focal, c[1]],
            [0., 0., 1.],
            ])
            
 
        
        img_name = img_files[image_id]
            
        c2w = all_c2w[image_id]

        if out_src_view:
            if self.split == 'train':
                src_views_num = get_nearest_pose_ids(c2w, all_c2w)
                src_views_num = src_views_num[1:]
            else:
                src_views_num = get_nearest_pose_ids(c2w, all_c2w_train)
                src_views_num = src_views_num[1:]


        if smoothing_loss:
            c2w_near = rot_from_origin(torch.FloatTensor(c2w))
            c2w_near = c2w_near[:3, :4]

        
        pose = torch.FloatTensor(convert_pose(c2w))
        w2c = torch.linalg.inv(pose)

        #Scale factor for images + features retrieval is 2 i.e. 120, 160
        proj_mat_l = torch.eye(4)
        intrinsics = intrinsics/2
        intrinsics[-1,-1] = 1
        # intrinsics[:2] = intrinsics[:2] / 2
        proj_mat_l[:3, :4] = intrinsics @ w2c[:3, :4]
    
        c2w = pose[:3, :4]
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

        if out_instance_seg:
            inst_seg = Image.open(os.path.join(base_dir, 'instance_masks_2d', img_name))
            inst_seg = cv2.resize(np.array(inst_seg), (w,h), interpolation =cv2.INTER_NEAREST)

        rays_o, rays_d = get_rays_mvs(h, w, focal, c2w)
        # directions = get_ray_directions(h, w, focal) # (h, w, 3)
        # rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)

        if out_instance_seg:
            if out_src_view:
                return rays_o, rays_d, w2c, img, instance_mask, inst_seg, nocs_2d, proj_mat_l, pose, torch.tensor(focal, dtype=torch.float32), torch.tensor(c, dtype=torch.float32), src_views_num
            else:
                return rays_o, rays_d, w2c, img, instance_mask, inst_seg, nocs_2d, proj_mat_l, pose, torch.tensor(focal, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)
        else:
            if out_src_view:
                return rays_o, rays_d, w2c, img, instance_mask, nocs_2d, proj_mat_l, pose, torch.tensor(focal, dtype=torch.float32), torch.tensor(c, dtype=torch.float32), src_views_num
            else:
                return rays_o, rays_d, w2c, img, instance_mask, nocs_2d, proj_mat_l, pose, torch.tensor(focal, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            if self.optimize is not None:
                return 100
            else:
                return self.samples_per_epoch
            # return len(self.ids)
        elif self.split == 'val':
            if self.eval_inference is not None:
                # num = int(self.eval_inference[0])
                # return len(self.ids)*(99-num)
                return len(self.ids)*99
                # return len(self.ids)*(5)
            else:
                return len(self.ids)
        else:
            return len(self.ids[:10])


    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            train_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[train_idx]
            
            imgs = list()
            poses = list()
            focals = list()
            all_c = list()

            nocs_2ds = list()
            masks = list()
            inst_seg_masks = list()
            rays = list()
            view_dirs = list()
            rays_d = list()
            rgbs = list()
            radii = list()
            proj_mats = list()
            w2cs = list()

            NV = 100
            src_views = 3
            if self.encoder_type == 'resnet':
                ray_batch_size = 4096
            else:
                ray_batch_size = 4096

            # ray_batch_size = 512
            #optimize from 5 source views
            
            dest_view_num = np.random.randint(0, 100)
            # src_views_num = get_nearest_pose_ids(tar_pose, all_c2w_train)
            # src_views_num = np.random.choice(100, src_views, replace=False)
            # # dest_view_nums = select_key_frames(100, src_views_num, self.rel_translations_train, self.rel_rotations_train, 0.4, 90)
            # # dest_view_num = random.choice(dest_view_nums)
            # views = [i for i in range(0, 100)]
            # #dest_view_nums = random.sample(list(set(views) - set(src_views_num)), 20)
            # a = list(set(views) - set(src_views_num))
            # dest_view_num = random.sample(a,1)[0]

            cam_rays, cam_rays_d,_ , img_gt, instance_mask, nocs_2d, _, _, _, _, src_views_num =  self.read_data(instance_dir, dest_view_num, contract = self.contract, out_src_view=True)
            
            instance_mask = T.ToTensor()(instance_mask)
            nocs_2d = Image.fromarray(np.uint8(nocs_2d))
            nocs_2d = T.ToTensor()(nocs_2d)

            H, W, _ = np.array(img_gt).shape
            cam_rays = torch.FloatTensor(cam_rays)
            cam_rays_d = torch.FloatTensor(cam_rays_d)

            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = T.ToTensor()(img_gt)
            rgbs = img_gt.permute(1,2,0).flatten(0,1)

            nocs_2ds = nocs_2d.permute(1,2,0).flatten(0,1)
            masks = instance_mask.permute(1,2,0).flatten(0,1)
            rays = cam_rays.view(-1, cam_rays.shape[-1])
            rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])

            patch = self.finetune_lpips
            if patch:
                #select 64 by 64 patch for LPIPS loss
                width = self.img_wh[0]
                height = self.img_wh[1]
                x = np.random.randint(0, height - 64 + 1)
                y = np.random.randint(0, width - 64 + 1)
                rgbs = rgbs.view(height, width, 3)[x:x+64, y:y+64, :].reshape(-1,3)
                nocs_2ds = nocs_2ds.view(height, width, 3)[x:x+64, y:y+64, :].reshape(-1,3)
                masks = masks.view(height, width)[x:x+64, y:y+64].reshape(-1,1)
                rays = rays.view(height, width, 3)[x:x+64, y:y+64, :].reshape(-1,3)
                rays_d = rays_d.view(height, width, 3)[x:x+64, y:y+64, :].reshape(-1,3)
            else:
                pix_inds = torch.randint(0, H * W, (ray_batch_size,))
                rgbs = rgbs[pix_inds,...] 
                nocs_2ds = nocs_2ds[pix_inds]
                masks = masks[pix_inds]
                rays = rays[pix_inds]
                rays_d = rays_d[pix_inds]

            for train_image_id in range(0, NV):
                if train_image_id not in src_views_num:
                    continue
                _, _, w2c, img, _, _, proj_mat, c2w, f, c =  self.read_data(instance_dir, train_image_id, contract = self.contract)
                img = Image.fromarray(np.uint8(img))
                img = T.ToTensor()(img)
                imgs.append(self.img_transform(img))
                poses.append(c2w)
                proj_mats.append(proj_mat)
                w2cs.append(w2c)
                focals.append(f)
                all_c.append(c)

            proj_mats_norm = []
            for i in range(len(proj_mats)):  # reference view
                if i ==0:
                    ref_proj_inv = torch.linalg.inv(proj_mats[i])
                    proj_mats_norm += [torch.eye(4)]
                else:
                    proj_mats_norm += [proj_mats[i] @ ref_proj_inv]
            
            imgs = torch.stack(imgs, 0)
            poses = torch.stack(poses, 0)
            focals = torch.stack(focals, 0)
            all_c = torch.stack(all_c, 0)
            proj_mats = torch.stack(proj_mats_norm, 0)
            w2cs = torch.stack(w2cs, 0)
        

            #======================================================\n\n\n
            # #Load desitnation view data from 20 dest views
            #======================================================\n\n\n
            # for train_image_id in dest_view_nums:
            #     cam_rays, cam_view_dirs, cam_rays_d, img_gt, instance_mask, nocs_2d, camera_radii, _, _, _ =  self.read_data(instance_dir, train_image_id, contract = self.contract)
                
            #     instance_mask = T.ToTensor()(instance_mask)
            #     nocs_2d = Image.fromarray(np.uint8(nocs_2d))
            #     nocs_2d = T.ToTensor()(nocs_2d)

            #     _, H, W = img.shape
            #     camera_radii = torch.FloatTensor(camera_radii)
            #     cam_rays = torch.FloatTensor(cam_rays)
            #     cam_view_dirs = torch.FloatTensor(cam_view_dirs)
            #     cam_rays_d = torch.FloatTensor(cam_rays_d)

            #     img_gt = Image.fromarray(np.uint8(img_gt))
            #     img_gt = T.ToTensor()(img_gt)
            #     rgb_gt = img_gt.permute(1,2,0).flatten(0,1)

            #     nocs_2d_gt = nocs_2d.permute(1,2,0).flatten(0,1)
            #     mask_gt = instance_mask.permute(1,2,0).flatten(0,1)
            #     radii_gt = camera_radii.view(-1)
            #     ray = cam_rays.view(-1, cam_rays.shape[-1])
            #     ray_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
            #     viewdir = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

            #     nocs_2ds.append(nocs_2d_gt)
            #     masks.append(mask_gt)
            #     rays.append(ray)
            #     view_dirs.append(viewdir)
            #     rays_d.append(ray_d)
            #     rgbs.append(rgb_gt)
            #     radii.append(radii_gt)

            # rgbs = torch.stack(rgbs, 0)
            # masks = torch.stack(masks, 0)
            # nocs_2ds = torch.stack(nocs_2ds, 0)
            # rays = torch.stack(rays, 0)  
            # rays_d = torch.stack(rays_d, 0) 
            # view_dirs = torch.stack(view_dirs, 0)  
            # radii = torch.stack(radii, 0)

            # pix_inds = torch.randint(0, len(dest_view_nums) * H * W, (ray_batch_size,))
            # rgbs = rgbs.reshape(-1,3)[pix_inds,...] 
            # nocs_2ds = nocs_2ds.reshape(-1,3)[pix_inds,...] 
            # masks = masks.reshape(-1,1)[pix_inds]
            # radii = radii.reshape(-1,1)[pix_inds]
            # rays = rays.reshape(-1,3)[pix_inds]
            # rays_d = rays_d.reshape(-1,3)[pix_inds]
            # view_dirs = view_dirs.reshape(-1,3)[pix_inds] 


            #======================================================\n\n\n
            # #Load desitnation view data from one camera view
            #======================================================\n\n\n
            #Load desitnation view data

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
                sample["instance_mask"] = masks
                sample["rays_o"] = rays
                sample["rays_d"] = rays_d
                sample["proj_mats"] = proj_mats
                sample["target"] = rgbs
                sample["nocs_2d"] = nocs_2ds
                sample["w2cs"] = w2cs
                sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = torch.zeros_like(sample["rays_o"])
                
            return sample
                
        # elif self.split == 'val': # create data for each image separately
        elif self.split=='val':
            if self.eval_inference is not None:
                instance_dir = self.ids[0]
            else:
                instance_dir = self.ids[idx]
            # instance_dir = self.ids[idx]
            imgs = list()
            poses = list()
            focals = list()
            all_c = list()
            proj_mats = list()
            w2cs = list()
            NV = 199
            src_views = 3

            if self.eval_inference is not None:
                num = int(self.eval_inference[0])
                if num ==3:
                    src_views_num = [0, 38, 44]
                elif num ==5:
                    src_views_num = [0, 38, 44, 94, 48]
                elif num ==1:
                    src_views_num = [0]
                dest_view_num = idx+100

            else:
                if self.optimize is not None:
                    num = int(self.optimize[0])
                    if num ==3:
                        src_views_num = [0, 38, 44]
                    elif num ==5:
                        src_views_num = [0, 38, 44, 94, 48]
                    elif num ==1:
                        src_views_num = [0]

                    dest_view_num = np.random.randint(0, 99) + 100
                else:
                    # src_views_num = [0, 38, 44]
                    # # dest_view_num = np.random.randint(0, 99) + 100

                    # views = [i for i in range(0, 99)]
                    # a = list(set(views) - set(src_views_num))
                    # dest_view_num = random.sample(a,1)[0] + 100
                    dest_view_num = np.random.randint(0, 99) + 100
                    # dest_view_num = random.sample(list(set(views)) - list(set(src_views_num)), 1)[0] + 100
                    # while True:
                    #     dest_view_nums = select_key_frames(NV, src_views_num, self.rel_translations_val, self.rel_rotations_val, 0.5, 120)
                    #     greater_than_100 = [num for num in dest_view_nums if num > 100]
                    #     if len(greater_than_100) > 0:
                    #         dest_view_num = random.choice(greater_than_100)
                    #         break
                    # dest_view_nums = select_key_frames(100, src_views_num, self.rel_translations_val, self.rel_rotations_val, 0.4, 90)
                    # dest_view_num = random.choice(dest_view_nums)


            #Load desitnation view data
            cam_rays, cam_rays_d,_ , img_gt, instance_mask, inst_seg, nocs_2d, _, _, _, _, src_views_num = self.read_data(instance_dir, dest_view_num, out_instance_seg=True, contract = self.contract, out_src_view=True)
            # cam_rays, cam_view_dirs, cam_rays_d, img_gt, instance_mask, inst_seg, nocs_2d, camera_radii, _, _, _, src_views_num =  self.read_data(instance_dir, dest_view_num, out_instance_seg=True, contract = self.contract, out_src_view=True)
            
            instance_mask = T.ToTensor()(instance_mask)
            inst_seg = T.ToTensor()(inst_seg)
            nocs_2d = Image.fromarray(np.uint8(nocs_2d))
            nocs_2d = T.ToTensor()(nocs_2d)

            H, W, _ = np.array(img_gt).shape
            cam_rays = torch.FloatTensor(cam_rays)
            cam_rays_d = torch.FloatTensor(cam_rays_d)

            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = T.ToTensor()(img_gt)
            rgbs = img_gt.permute(1,2,0).flatten(0,1)

            nocs_2ds = nocs_2d.permute(1,2,0).flatten(0,1)
            masks = instance_mask.permute(1,2,0).flatten(0,1)
            inst_seg_masks = inst_seg.permute(1,2,0).flatten(0,1)
            rays = cam_rays.view(-1, cam_rays.shape[-1])
            rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])

            for train_image_id in range(0, NV):
                if train_image_id not in src_views_num:
                    continue
                _, _, w2c, img, _, _,_, proj_mat, c2w, f, c =  self.read_data(instance_dir, train_image_id, out_instance_seg=True, contract = self.contract)
                img = Image.fromarray(np.uint8(img))
                img = T.ToTensor()(img)
                imgs.append(self.img_transform(img))
                poses.append(c2w)
                focals.append(f)
                all_c.append(c)
                proj_mats.append(proj_mat)
                w2cs.append(w2c)

            proj_mats_norm = []
            for i in range(len(proj_mats)):  # reference view
                if i ==0:
                    ref_proj_inv = torch.linalg.inv(proj_mats[i])
                    proj_mats_norm += [torch.eye(4)]
                else:
                    proj_mats_norm += [proj_mats[i] @ ref_proj_inv]
            
            imgs = torch.stack(imgs, 0)
            poses = torch.stack(poses, 0)
            focals = torch.stack(focals, 0)
            all_c = torch.stack(all_c, 0)
            proj_mats = torch.stack(proj_mats_norm, 0)
            w2cs = torch.stack(w2cs, 0)

            # near_obj, far_obj, _ = sample_rays_in_bbox(self.RTs, rays.numpy(), view_dirs.numpy())
            
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
                sample["instance_mask"] = masks
                sample["inst_seg_mask"] = inst_seg_masks
                sample["src_c"] = all_c
                sample["rays_o"] = rays
                sample["rays_d"] = rays_d
                sample["proj_mats"] = proj_mats
                sample["target"] = rgbs
                sample["nocs_2d"] = nocs_2ds
                sample["w2cs"] = w2cs
                sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = torch.zeros_like(sample["rays_o"])
                
            return sample