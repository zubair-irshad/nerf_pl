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

def make_poses_bounds_array(frame_data, near=0.2, far=10):
    # See https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap
    # Returns an array of shape (N, 17).
    rows = []
    all_c2w = []
    all_focal = []

    adjust_matrix = np.array(
        [[0.,   1.,   0.],
        [1.,   0.,   0.],
        [0.,   0.,  -1.]])

    for frame in frame_data:
        camera = frame.camera      
        focal = camera.intrinsics[0]
        cam_to_world = np.array(camera.transform).reshape(4,4)
        all_c2w.append(cam_to_world)
        all_focal.append(focal)
    return all_c2w, all_focal

def transform_rays_to_bbox_coordinates_nocs(rays_o, rays_d, axis_align_mat):
    rays_o_bbox = rays_o
    rays_d_bbox = rays_d
    T_box_orig = axis_align_mat
    rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
    rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
    return rays_o_bbox, rays_d_bbox

def load_frame_data(geometry_filename):
    # See get_geometry_data in objectron-geometry-tutorial.ipynb
    frame_data = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        while i < len(proto_buf):
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame = ar_metadata_protocol.ARFrame()
            frame.ParseFromString(message_buf)
            frame_data.append(frame)
    return frame_data

def get_frame_annotation(annotation_filename):
    """Grab an annotated frame from the sequence."""
    result = []
    instances = []
    with open(annotation_filename, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())

        object_id = 0
        object_rotations = []
        object_translations = []
        object_scale = []
        num_keypoints_per_object = []
        object_categories = []
        annotation_types = []
        
        # Object instances in the world coordinate system, These are stored per sequence, 
        # To get the per-frame version, grab the transformed keypoints from each frame_annotation
        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            scale = np.array(obj.scale)
            points3d = np.array([[kp.x, kp.y, kp.z] for kp in obj.keypoints])
            instances.append((rotation, translation, scale, points3d))
        
        # Grab teh annotation results per frame
        for data in sequence.frame_annotations:
            # Get the camera for the current frame. We will use the camera to bring
            # the object from the world coordinate to the current camera coordinate.
            transform = np.array(data.camera.transform).reshape(4, 4)
            view = np.array(data.camera.view_matrix).reshape(4, 4)
            intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)
            projection = np.array(data.camera.projection_matrix).reshape(4, 4)
        
            keypoint_size_list = []
            object_keypoints_2d = []
            object_keypoints_3d = []
            for annotations in data.annotations:
                num_keypoints = len(annotations.keypoints)
                keypoint_size_list.append(num_keypoints)
                for keypoint_id in range(num_keypoints):
                    keypoint = annotations.keypoints[keypoint_id]
                    object_keypoints_2d.append((keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                    object_keypoints_3d.append((keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
                num_keypoints_per_object.append(num_keypoints)
                object_id += 1
            result.append((object_keypoints_2d, object_keypoints_3d, keypoint_size_list, view, projection))

    return result, instances

def read_objectron_info(base_dir, instance_name):
    annotation_data, instances = get_frame_annotation(os.path.join(base_dir, instance_name+'.pbdata'))
    instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[0]
    instance_rotation = np.reshape(instance_rotation, (3, 3))
    box_transformation = np.eye(4)
    box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    box_transformation[:3, -1] = instance_translation
    scale = instance_scale.T
    axis_align_mat = box_transformation
    bbox_bounds = np.array([-scale / 2, scale / 2])
    return axis_align_mat

class ObjectronDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1600, 1600)):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.all_c2w = []
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        instance_name = 'bottle_batch-1_37'
        self.base_dir = '/home/ubuntu/nerf_pl/data/objectron/bottle/'+instance_name
        self.axis_align_mat = torch.FloatTensor(np.linalg.inv(read_objectron_info(self.base_dir, instance_name)))
        # self.axis_align_mat = torch.FloatTensor(read_objectron_info(self.base_dir, instance_name))
                #json_files = [pos_json for pos_json in os.listdir(base_dir) if pos_json.endswith('.json')]
        sfm_arframe_filename = self.base_dir + '/'+ instance_name+'_sfm_arframe.pbdata'
        frame_data = load_frame_data(sfm_arframe_filename)

        # self.near = 0.02
        # self.far = 1.5

        #for bottle
        self.near = 0.02
        self.far = 1.0

        self.all_c2w, self.all_focal = make_poses_bounds_array(frame_data, near=self.near, far=self.far)
        self.meta = sorted(glob.glob(self.base_dir+'/masks_12/*.png'))
        self.all_c2w = self.all_c2w
        self.all_focal = self.all_focal
        self.meta = self.meta

        # if len(self.meta) >300:
        self.all_c2w = self.all_c2w[:200]
        self.all_focal = self.all_focal[:200]
        self.meta = self.meta[:200]

        #Select every 4th view for training
        # self.all_c2w = self.all_c2w[::2]
        # self.all_focal = self.all_focal[::2]
        # self.meta = self.meta[::2]
    
        w, h = self.img_wh
        self.bounds = np.array([self.near, self.far])
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_instance_masks = []
            self.all_instance_masks_weight = []
            self.all_instance_ids = []

            for i, seg_name in enumerate(self.meta):
                # if i>200:
                #     continue
                
                c2w = self.all_c2w[i]

                focal = self.all_focal[i]
                # focal *= self.img_wh[0]/1440 # modify focal length to match size self.img_wh
                focal *=(self.img_wh[0]/1920)  # modify focal length to match size self.img_wh
                directions = get_ray_directions(h, w, focal) # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, rays_d = get_rays(directions, c2w)

                img_name = os.path.basename(seg_name).split('_')[1]
                img = Image.open(os.path.join(self.base_dir, 'images_12', img_name))
                img = img.transpose(Image.ROTATE_90)
                img = self.transform(img) # (h, w, 3)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

                #Get masks
                # mask_name = 'mask_'+ os.path.split(img_name)[1]
                seg_mask = cv2.imread(os.path.join(self.base_dir, 'masks_12', os.path.basename(seg_name)), cv2.IMREAD_GRAYSCALE)
                seg_mask = np.rot90(np.array(seg_mask), axes=(0,1))

                valid_mask = seg_mask>0
                valid_mask = self.transform(valid_mask).view(
                        -1)
                instance_mask = seg_mask >0
                instance_mask_weight = rebalance_mask(
                    instance_mask,
                    fg_weight=1.0,
                    bg_weight=0.05,
                )
                instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                    -1), self.transform(instance_mask_weight).view(-1)
                instance_ids = torch.ones_like(instance_mask).long() * 1

                rays_o, rays_d = transform_rays_to_bbox_coordinates_nocs(rays_o, rays_d, self.axis_align_mat)

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
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "instance_mask": self.all_instance_masks[idx],
                "instance_mask_weight": self.all_instance_masks_weight[idx],
                "instance_ids": self.all_instance_ids[idx],
            }

            # sample = {
            #     "rays": self.all_rays[idx],
            #     "rgbs": self.all_rgbs[idx],
            # }

        elif self.split == 'val': # create data for each image separately
            val_idx = 40
            seg_name = self.meta[val_idx]
            w, h = self.img_wh
            c2w = self.all_c2w[val_idx]

            focal = self.all_focal[val_idx]
            focal *=(self.img_wh[0]/1920) # modify focal length to match size self.img_wh
            
            directions = get_ray_directions(h, w, focal) # (h, w, 3)

            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, rays_d = get_rays(directions, c2w)

            img_name = os.path.basename(seg_name).split('_')[1]
            img = Image.open(os.path.join(self.base_dir, 'images_12', img_name))
            img = img.transpose(Image.ROTATE_90) 
            img = self.transform(img) # (h, w, 3)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA


            #Get masks
            # seg_name = 'mask_'+ os.path.split(img_name)[1]
            
            seg_mask = cv2.imread(os.path.join(self.base_dir, 'masks_12', os.path.basename(seg_name)), cv2.IMREAD_GRAYSCALE)
            seg_mask = np.rot90(np.array(seg_mask), axes=(0,1))
            valid_mask = seg_mask>0
            valid_mask = self.transform(valid_mask).view(
                    -1)
            instance_mask = seg_mask >0
            # print("instance_mask", instance_mask.shape)
            instance_mask_weight = rebalance_mask(
                instance_mask,
                fg_weight=1.0,
                bg_weight=0.05,
            )
            instance_mask, instance_mask_weight = self.transform(instance_mask).view(
                -1), self.transform(instance_mask_weight).view(-1)
            instance_ids = torch.ones_like(instance_mask).long() * 1
            # print("instance_mask", instance_mask.shape)

            rays_o, rays_d = transform_rays_to_bbox_coordinates_nocs(rays_o, rays_d, self.axis_align_mat)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            sample = {
                "rays": rays,
                "rgbs": img,
                "instance_mask": instance_mask,
                "instance_mask_weight": instance_mask_weight,
                "instance_ids": instance_ids,
            }
            # sample = {
            #     "rays": rays,
            #     "rgbs": img
            # }  
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
                "c2w": c2w
            }  
        return sample