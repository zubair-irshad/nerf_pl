import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
import sys
sys.path.append('/home/ubuntu/pixel-nerf')
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# from .google_scanned_utils import load_image_from_exr, load_seg_from_exr
import struct
# from .ray_utils import *
# from .nocs_utils import rebalance_mask
import glob
from objectron.schema import annotation_data_pb2 as annotation_protocol
import glob
from random import choice
import random

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    # if image_size > 0:
    #     ops.append(T.Resize(image_size))
    ops.extend(
        [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return T.Compose(ops)

def normalize_image():
    ops = []
    ops.extend(
        [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    # ops.extend(
    #     [T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])]
    # )
    return T.Compose(ops)

def make_poses_bounds_array(frame_data, near=0.2, far=10):
    # See https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap
    # Returns an array of shape (N, 17).
    rows = []
    all_c2w = []
    all_focal = []
    all_c = []

    adjust_matrix = np.array(
        [[0.,   1.,   0.],
        [1.,   0.,   0.],
        [0.,   0.,  -1.]])
    #hard coded but change it to self.img_wh
    img_wh = (160,120)
    for frame in frame_data:
        camera = frame.camera      
        # focal = camera.intrinsics[0]
        intrinsics = np.array(camera.intrinsics).reshape(3,3)
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]

        fx *=(img_wh[0]/1920)
        fy *=(img_wh[0]/1920)
        cx *=(img_wh[0]/1920)
        cy *=(img_wh[0]/1920)

        focal = torch.tensor((fx, fy), dtype=torch.float32)
        c = torch.tensor((cx, cy), dtype=torch.float32)

        cam_to_world = np.array(camera.transform).reshape(4,4)
        all_c2w.append(cam_to_world)
        all_focal.append(focal)
        all_c.append(c)
    return all_c2w, all_focal, all_c

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

def read_objectron_info(base_dir, instance_name, idx):
    annotation_data, instances = get_frame_annotation(os.path.join(base_dir, instance_name+'.pbdata'))
    instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[0]
    instance_rotation = np.reshape(instance_rotation, (3, 3))
    box_transformation = np.eye(4)
    box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    box_transformation[:3, -1] = instance_translation
    scale = instance_scale.T
    axis_align_mat = box_transformation

    # read 3d bounding box and project it to 2D
    points_2d, points_3d, num_keypoints, view, projection = annotation_data[idx]
    vertices_3d = instance_vertices_3d * instance_scale.T
    vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
    box_vertices_3d_world = np.matmul(box_transformation, vertices_3d_homg) 
    vertices_3d_cam = np.matmul(view, box_vertices_3d_world)
    vertices_2d_proj = np.matmul(projection, vertices_3d_cam)
    # Project the points
    points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
    points2d_ndc = points2d_ndc.T
    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    x = points2d_ndc[:, 1]
    y = points2d_ndc[:, 0]
    points2d = np.copy(points2d_ndc)
    width = 1440 
    height = 1920
    points2d[:, 0] = ((1 + x) * 0.5) * width
    points2d[:, 1] = ((1 + y) * 0.5) * height   
    points2d = points2d/12
    points_2d_aray = np.array(points2d.astype(int))
    x_min=np.min(points_2d_aray[:,0])
    y_min=np.min(points_2d_aray[:,1])
    x_max=np.max(points_2d_aray[:,0])
    y_max=np.max(points_2d_aray[:,1])

    bbox_2d = (x_min, y_min, x_max, y_max)
    
    bbox_bounds = np.array([-scale / 2, scale / 2])
    return axis_align_mat, scale, bbox_2d

class ObjectronMultiDataset(Dataset):
    def __init__(self, root_dir, max_imgs = 50, stage='train', img_wh=(160, 120), num_instances_per_obj = 1, crop_img = False):
        self.base_dir = root_dir
        self.max_imgs = max_imgs
        self.split = stage
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        self.crop_img = crop_img
        self.num_instances_per_obj = num_instances_per_obj
        self.lenids = len(self.ids)
        self.normalize_img = normalize_image()
        self.img_wh = img_wh
        self.define_transforms()
        # self.near = 0.02
        # self.far = 1.0
        self.z_near = 0.08
        self.z_far = 0.8
        # self.val_instances = [30, 60, 90, 120, 145]
        self.val_instances = [10, 20, 30, 40, 45]
        self.white_back = False
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.lindisp = False

    def load_img(self, instance_dir, idxs = [], ids=None, bbox_2d = None):
        masks_dir = os.path.join(instance_dir, 'masks_12/*.png')
        allmaskfiles = sorted(glob.glob(masks_dir))
        maskfiles = np.array(allmaskfiles)[idxs]
        all_imgs =[]
        for seg_name in maskfiles:
            img_name = os.path.basename(str(seg_name)).split('_')[1]
            img = cv2.imread(os.path.join(instance_dir, 'images_12', img_name))
            focal_idx = img_name.split('.')[0]
            img = img[...,::-1]
            if self.crop_img:
                x_min, y_min, x_max, y_max = bbox_2d
                if y_min<0:
                    y_min=0
                if x_min<0:
                    x_min=0
                img = img[y_min: y_max, x_min:x_max]
            
            W,H = img.shape[0], img.shape[1]
            new_img_wh = (W,H)
            img = Image.fromarray(img)
            img = img.transpose(Image.ROTATE_90)
            img = self.image_to_tensor(img)
            # img = self.transform(img) # (h, w, 3)
            #img = img.contiguous().view(3, -1).permute(1, 0) # (h*w, 3) RGBA
        if len(idxs) == 1:
            return img, new_img_wh, int(focal_idx)
        else:
            return all_imgs, new_img_wh, int(focal_idx)


    def return_train_data(self, instance_name, idx, instances):
        instance_dir = os.path.join(self.base_dir, instance_name)
        #Only consider the first 200 instances for reconstruction
        # instances = np.random.choice(150, self.num_instances_per_obj)
        axis_align_mat, scale, bbox_2d = read_objectron_info(instance_dir, instance_name, idx)
        self.axis_align_mat = torch.FloatTensor(np.linalg.inv(axis_align_mat))
        #save relevant bonding box info with obj ids for inference
        RTs_dict = {'RT':axis_align_mat, 'scale': scale}
        sfm_arframe_filename = instance_dir + '/'+ instance_name+'_sfm_arframe.pbdata'
        frame_data = load_frame_data(sfm_arframe_filename)
        all_c2w, all_focal, all_c = make_poses_bounds_array(frame_data, near=self.z_near, far=self.z_near)
        img, new_img_wh, focal_idx = self.load_img(instance_dir, [instances], idx, bbox_2d=bbox_2d)
        w, h = self.img_wh
        if self.crop_img:
            w, h = new_img_wh
            # w -= (2*32)
            # h -=  (2*32)

        c2w = np.array(all_c2w)[focal_idx]
        focal = all_focal[focal_idx]
        c = all_c[focal_idx]
        #incase of just one instance
        c2w = np.squeeze(c2w)
        return img, c2w, focal, c

    def return_val_data(self, instance_name, idx, instances):
        instance_dir = os.path.join(self.base_dir, instance_name)
        #Only consider the first 200 instances for reconstruction
        # instances = np.random.choice(150, self.num_instances_per_obj)
        axis_align_mat, scale, bbox_2d = read_objectron_info(instance_dir, instance_name, idx)
        self.axis_align_mat = torch.FloatTensor(np.linalg.inv(axis_align_mat))
        #save relevant bonding box info with obj ids for inference
        RTs_dict = {'RT':axis_align_mat, 'scale': scale}
        sfm_arframe_filename = instance_dir + '/'+ instance_name+'_sfm_arframe.pbdata'
        frame_data = load_frame_data(sfm_arframe_filename)
        all_c2w, all_focal, all_c = make_poses_bounds_array(frame_data, near=self.near, far=self.far)
        img, new_img_wh, focal_idx = self.load_img(instance_dir, [instances], idx, bbox_2d=bbox_2d)
        w, h = self.img_wh
        if self.crop_img:
            w, h = new_img_wh
            # w -= (2*32)
            # h -=  (2*32)

        c2w = np.array(all_c2w)[focal_idx]
        focal = all_focal[focal_idx]
        c = all_c[focal_idx]
        #incase of just one instance
        c2w = np.squeeze(c2w)
        return img, c2w, focal, c

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return self.lenids

    def __getitem__(self, idx):
        obj_id = self.ids[idx]
        if self.split == "train":
            numbers = range(0,self.max_imgs)
            all_imgs = []
            all_poses = []
            # all_focal = []
            # all_c = []
            train_indices = [n for n in numbers if n not in self.val_instances]
            random.shuffle(train_indices)
            for j, index in enumerate(train_indices):
                img, c2w, focal, c = self.return_train_data(obj_id, idx, index)
                all_imgs.append(img)
                all_poses.append(torch.tensor(c2w, dtype=torch.float32))
                if j ==0:
                    all_focal = focal
                    all_c = c
                # all_focal.append(focal)
                # all_c.append(c)
            all_imgs = torch.stack(all_imgs)
            all_poses = torch.stack(all_poses)
            # all_focal = torch.stack(all_focal)
            # all_c = torch.stack(all_c)
            result = {
                "path": self.base_dir,
                "img_id": idx,
                "focal": all_focal,
                "images": all_imgs,
                "poses": all_poses,
                "c": all_c
            }
            return result
        else:
            numbers = self.val_instances
            all_imgs = []
            all_poses = []
            # all_focal = []
            # all_c = []
            val_indices = numbers
            random.shuffle(val_indices)
            for j, index in enumerate(val_indices):
                img, c2w, focal, c = self.return_train_data(obj_id, idx, index)
                all_imgs.append(img)
                all_poses.append(torch.tensor(c2w, dtype=torch.float32))
                if j ==0:
                    all_focal = focal
                    all_c = c
                # all_focal.append(focal)
                # all_c.append(c)
            all_imgs = torch.stack(all_imgs)
            all_poses = torch.stack(all_poses)
            # all_focal = torch.stack(all_focal)
            # all_c = torch.stack(all_c)
            result = {
                "path": self.base_dir,
                "img_id": idx,
                "focal": all_focal,
                "images": all_imgs,
                "poses": all_poses,
                "c": all_c
            }
            return result