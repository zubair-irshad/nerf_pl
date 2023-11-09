import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa
import copy
import numpy as np
import numba as nb  # mute some warning
import torch
from torch import nn
from collections import defaultdict
from tqdm import trange, tqdm
from typing import List, Optional, Any, Dict, Union
from omegaconf import OmegaConf

from utils.geo_utils import center_pose_from_avg
from utils.bbox_utils import BBoxRayHelper
# from utils.util import read_json
from train_compositional import NeRFSystem
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from datasets.ray_utils import get_ray_directions, get_rays, transform_rays_camera
from models.multi_rendering import render_rays_multi


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg

def center_pose_from_avg(pose_avg, pose):
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg.copy()  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_homo = np.eye(4)
    pose_homo[:3] = pose[:3]
    pose_centered = np.linalg.inv(pose_avg_homo) @ pose_homo  # (4, 4)
    # pose_centered = pose_centered[:, :3] # (N_images, 3, 4)
    return pose_centered

class EditableRenderer:
    def __init__(self, hparams):
        # load config
        self.hparams = hparams
        # self.ckpt_config = config.ckpt_config
        self.load_model(hparams.ckpt_path)
        self.poses = None

        # initialize rendering parameters
        # dataset_extra = self.ckpt_config.dataset_extra
        # self.near = config.get("near", dataset_extra.near)
        # self.far = config.get("far", dataset_extra.far)
        # self.scale_factor = dataset_extra.scale_factor
        # self.pose_avg = np.concatenate(
        #     [np.eye(3), np.array(dataset_extra["scene_center"])[:, None]], 1
        # )
        self.root_dir = hparams.root_dir
        self.object_to_remove = []
        self.active_object_ids = [0]
        # self.active_object_ids = []
        self.object_pose_transform = {}
        self.object_bbox_ray_helpers = {}
        self.bbox_enlarge = 0.0
        self.img_wh = hparams.img_wh

    def load_model(self, ckpt_path):
        self.system = NeRFSystem.load_from_checkpoint(
            ckpt_path
        ).cuda()
        self.system.eval()

    def read_meta(self):
        # Step 1: rescale focal length according to training resolution
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        self.focal = camdata[1].params[0] * self.img_wh[0]/W

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        # read successfully reconstructed images and ignore others
        self.image_paths = [os.path.join(self.root_dir, 'images', name)
                            for name in sorted([imdata[k].name for k in imdata])]
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
        
        # read bounds
        self.bounds = np.zeros((len(poses), 2)) # (N_images, 2)
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts_world = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_images, N_points)
        for i, k in enumerate(pts3d):
            pts_world[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids:
                visibilities[j-1, i] = 1
        depths = ((pts_world-poses[..., 3:4])*poses[..., 2:3]).sum(1) # (N_images, N_points)
        for i in range(len(poses)):
            visibility_i = visibilities[i]
            zs = depths[i][visibility_i==1]
            self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]
        # permute the matrices to increasing order
        poses = poses[perm]
        self.bounds = self.bounds[perm]
        
        # COLMAP poses has rotation in form "right down front", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        #self.poses, self.pose_avg = center_poses(poses)
        self.poses = poses
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        self.scale_factor = near_original*0.75 # 0.75 is the defaul`t parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= self.scale_factor
        # self.poses[..., 3] /= self.scale_factor
        self.near = self.bounds.min()
        self.far = min(8 * self.near, self.bounds.max()) # focus on central object only
        self.pose_avg = average_poses(self.poses) # (3, 4)

        return self.pose_avg, self.scale_factor, self.focal


    def get_camera_pose_by_frame_idx(self, frame_idx):
        return self.poses[frame_idx]

    def scene_inference(
        self,
        rays: torch.Tensor,
        show_progress: bool = True,
    ):
        args = {}
        # args["train_config"] = self.ckpt_config.train

        B = rays.shape[0]
        results = defaultdict(list)
        chunk = self.hparams.chunk
        for i in tqdm(range(0, B, self.hparams.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_multi(
                    models=self.system.models,
                    embeddings=self.system.embeddings,
                    code_library=self.system.code_library,
                    # rays=rays[i : i + chunk],
                    rays_list=[rays[i : i + chunk]],
                    obj_instance_ids=[0],
                    N_samples=self.hparams.N_samples,
                    use_disp=self.hparams.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.hparams.N_importance,
                    chunk=self.hparams.chunk,  # chunk size is effective in val mode
                    white_back=False,
                    **args,
                )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            # cat on rays dim
            if len(v[0].shape) <= 2:
                results[k] = torch.cat(v, 0)
            elif len(v[0].shape) == 3:
                results[k] = torch.cat(v, 1)
        return results

    def generate_rays(self, obj_id, rays_o, rays_d, c2w=None, pose_delta=None):
        near = self.near
        far = self.far
        if obj_id == 0:
            # batch_near = near / self.scale_factor * torch.ones_like(rays_o[:, :1])
            # batch_far = far / self.scale_factor * torch.ones_like(rays_o[:, :1])
            batch_near = near  * torch.ones_like(rays_o[:, :1])
            batch_far = far * torch.ones_like(rays_o[:, :1])
            # rays_o = rays_o / self.scale_factor
            rays = torch.cat([rays_o, rays_d, batch_near, batch_far], 1)  # (H*W, 8)
        else:
            bbox_mask, bbox_batch_near, bbox_batch_far = self.object_bbox_ray_helpers[
                str(obj_id)
            ].get_ray_bbox_intersections(
                rays_o,
                rays_d,
                self.scale_factor,
                # bbox_enlarge=self.bbox_enlarge / self.get_scale_factor(obj_id),
                bbox_enlarge=self.bbox_enlarge, 
                c2w = c2w,# in physical world
                pose_delta = pose_delta
            )
            # for area which hits bbox, we use bbox hit near far
            # bbox_ray_helper has scale for us, do no need to rescale
            batch_near_obj, batch_far_obj = bbox_batch_near, bbox_batch_far
            # for the invalid part, we use 0 as near far, which assume that (0, 0, 0) is empty
            batch_near_obj[~bbox_mask] = torch.zeros_like(batch_near_obj[~bbox_mask])
            batch_far_obj[~bbox_mask] = torch.zeros_like(batch_far_obj[~bbox_mask])
            
            #rays_o, rays_d = transform_rays_camera(rays_o, rays_d, c2w)
            rays = torch.cat(
                [rays_o, rays_d, batch_near_obj, batch_far_obj], 1
            )  # (H*W, 8)
        rays = rays.cuda()
        return rays

    def render_origin(
        self,
        h: int,
        w: int,
        camera_pose_Twc: np.ndarray,
        focal,
    ):
        directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        # Twc = center_pose_from_avg(self.pose_avg, camera_pose_Twc)
        # print("camera_pose_Twc", camera_pose_Twc.shape)
        Twc = center_pose_from_avg(self.pose_avg, camera_pose_Twc)
        Twc[:, 3] /= self.scale_factor
        # for scene, Two is eye
        #Twc = camera_pose_Twc
        Two = np.eye(4)
        Toc = np.linalg.inv(Two) @ Twc
        #Toc = Twc
        Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
        rays_o, rays_d = get_rays(directions, Toc)
        rays = self.generate_rays(0, rays_o, rays_d)
        results = self.scene_inference(rays)
        return results

    def render_edit(
        self,
        h: int,
        w: int,
        camera_pose_Twc: np.ndarray,
        camera_pose_Twc_origin : np.ndarray,
        focal,
        pose_delta = None,
        show_progress: bool = False,
        render_bg_only: bool = False,
        render_obj_only: bool = True,
        white_back: bool = True,
    ):
        # focal = (w / 2) / np.tan((fovx_deg / 2) / (180 / np.pi))
        directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        Twc = center_pose_from_avg(self.pose_avg, camera_pose_Twc)

        Twc_origin = center_pose_from_avg(self.pose_avg, camera_pose_Twc_origin)
        # pose_delta_T = center_pose_from_avg(self.pose_avg, pose_delta)
        # pose_delta_T = pose_delta

        #Twc, self.pose_avg = center_poses(camera_pose_Twc)
        args = {}
        results = {}
        obj_ids = []
        rays_list = []

        # only render background
        if render_bg_only:
            self.active_object_ids = [0]

        # only render objects
        if render_obj_only:
            self.active_object_ids.remove(0)
        object_pose_and_size = {}
        processed_obj_id = []
        all_obj_ids = [1,2,3,4,5]
        for obj_id in all_obj_ids:
            object_pose_and_size[str(obj_id)] = {}
            object_pose_and_size[str(obj_id)]['size'] = self.get_object_bbox_helper(str(obj_id)).size
            object_pose_and_size[str(obj_id)]['pose'] = self.get_object_bbox_helper(str(obj_id)).axis_align_mat

        for obj_id in self.active_object_ids:
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            if obj_id == 0:
                # for scene, transform is Identity
                Tow = transform = np.eye(4)
                # Toc = Twc
            else:
                object_pose = self.object_pose_transform[
                   f"{obj_id}_{obj_duplication_cnt}"
                ]
                #Tow = self.get_object_bbox_helper(
                #    obj_id
                #).get_camera_to_object_transform()
                
                #transform = np.linalg.inv(Tow) @ object_pose @ Tow
                # Toc = Toc

                # # transform in the real world scale
                c2w = copy.deepcopy(Twc)
                c2w[:, 3] /= self.scale_factor
                Tow_orig = self.get_object_bbox_helper(
                    obj_id
                ).get_world_to_object_transform(c2w)

                # get_camera_to_object_transform
                # # transform object into center, then apply user-specific object poses
                transform = np.linalg.inv(Tow_orig) @ object_pose @ Tow_orig
                # # for X_c = Tcw * X_w, when we applying transformation on X_w,
                # # it equals to Tcw * (transform * X_w). So, Tow = inv(transform) * Twc
                Tow = np.linalg.inv(transform)
                # # Tow = np.linalg.inv(Tow)  # this move obejct to center
                Tow = np.eye(4)

            processed_obj_id.append(obj_id)
            Toc = Tow @ Twc

            Toc_origin = Tow @ Twc_origin
            # resize to NeRF scale
            Toc[:, 3] /= self.scale_factor

            Toc_origin[:, 3] /= self.scale_factor
            Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
            Toc_origin = torch.from_numpy(Toc_origin).float().cuda()[:3, :4]

            # print("pose_delta_T",pose_delta_T,  pose_delta_T.shape)
            # pose_delta_T_scaled = pose_delta_T
            # pose_delta_T_scaled[..., 3] /= self.scale_factor
            # pose_delta_T_scaled = pose_delta_T_scaled[:3, :4]
            # pose_delta[3,3] = 1
        
            # all the rays_o and rays_d has been converted to NeRF scale

            # r_test = np.eye(3)
            # t_test = np.zeros((3,1))
            # c2w_test = np.concatenate((r_test,t_test), axis=1)

            # # c2w_test = torch.from_numpy(c2w_test).float()
            # c2w_test = torch.from_numpy(c2w_test).float().cuda()[:3, :4]
            # rays_o, rays_d = get_rays(directions, c2w_test) # both (h*w, 3)

            
            
            rays_o, rays_d = get_rays(directions, Toc)
            rays = self.generate_rays(obj_id, rays_o, rays_d, Toc, pose_delta.copy())
            # light anchor should also be transformed
            #Tow = torch.from_numpy(Tow).float()
            #transform = torch.from_numpy(transform).float()
            obj_ids.append(obj_id)
            rays_list.append(rays)

        # split chunk
        B = rays_list[0].shape[0]
        chunk = self.hparams.chunk
        results = defaultdict(list)
        background_skip_bbox = self.get_skipping_bbox_helper()
        for i in tqdm(range(0, B, self.hparams.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_multi(
                    models=self.system.models,
                    embeddings=self.system.embeddings,
                    code_library=self.system.code_library,
                    rays_list=[r[i : i + chunk] for r in rays_list],
                    obj_instance_ids=obj_ids,
                    N_samples=self.hparams.N_samples,
                    use_disp=self.hparams.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.hparams.N_importance,
                    chunk=self.hparams.chunk,  # chunk size is effective in val mode
                    white_back=white_back,
                    background_skip_bbox=background_skip_bbox,
                    c2w=Toc, 
                    pose_delta=pose_delta.copy(),
                    **args,
                )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v.detach().cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results, object_pose_and_size

    def remove_scene_object_by_ids(self, hparams, obj_ids, idx, poses_avg, scale_factor):
        """
        Create a clean background by removing user specified objects.
        """
        self.object_to_remove = obj_ids
        for obj_id in obj_ids:
            self.initialize_object_bbox(hparams, obj_id, idx, poses_avg, scale_factor)

    def reset_active_object_ids(self):
        self.active_object_ids = [0]

    def set_object_pose_transform(
        self,
        hparams, 
        obj_id: int,
        idx,
        pose: np.ndarray,
        obj_dup_id: int,  # for object duplication, 
        poses_avg, 
        scale_factor
    ):
        self.active_object_ids.append(obj_id)
        if obj_id not in self.active_object_ids:
            self.initialize_object_bbox(hparams, obj_id, idx, poses_avg, scale_factor)
        self.object_pose_transform[f"{obj_id}_{obj_dup_id}"] = pose

    def initialize_object_bbox(self, hparams, obj_id: int, idx, poses_avg, scale_factor):
        self.object_bbox_ray_helpers[str(obj_id)] = BBoxRayHelper(hparams, obj_id, idx, poses_avg, scale_factor
        )

    def get_object_bbox_helper(self, obj_id: int):
        return self.object_bbox_ray_helpers[str(obj_id)]

    def get_skipping_bbox_helper(self):
        skipping_bbox_helper = {}
        for obj_id in self.object_to_remove:
            skipping_bbox_helper[str(obj_id)] = self.object_bbox_ray_helpers[
                str(obj_id)
            ]
        return skipping_bbox_helper