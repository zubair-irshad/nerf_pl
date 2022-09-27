import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import numpy as np
import numba as nb  # mute some warning
import torch
from torch import nn
from collections import defaultdict
from tqdm import trange, tqdm
from typing import List, Optional, Any, Dict, Union

from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# from .google_scanned_utils import load_image_from_exr, load_seg_from_exr
import struct

from utils.bbox_utils_pd import BBoxRayHelper
from train_compositional import NeRFSystem
from datasets.ray_utils import get_ray_directions, get_rays
from datasets.co3d import CO3D_Instance as Co3d_Dataset
#for disentagled shape and appearance
# from models.multi_rendering_objectron import render_rays_multi
import json

#for only one shape code
from models.multi_rendering import render_rays_multi
from objectron.schema import annotation_data_pb2 as annotation_protocol

class EditableRenderer:
    def __init__(self, config):
        # load config
        self.config = config
        self.load_model(config.ckpt_path)

        # initialize rendering parameters
        # self.near = 0.02
        # self.far = 1.5

        #For bottle
        self.near = 0.3
        self.far = 3.0

        self.object_to_remove = []
        self.active_object_ids = [0]
        # self.active_object_ids = []
        self.object_pose_transform = {}
        self.object_bbox_ray_helpers = {}
        self.bbox_enlarge = 0.0

    def load_model(self, ckpt_path):
        self.system = NeRFSystem.load_from_checkpoint(
            ckpt_path
        ).cuda()
        self.system.eval()

    
    def load_frame_meta(self):

        data_dir = '/home/ubuntu/nerf_pl/data/PD'
        # instance = 'fullsize_sedan_01_body_red'
        instance = 'fullsize_sedan_01_body_black_circular'
        self.img_files = os.listdir(os.path.join(data_dir,instance, 'rgb'))
        self.img_files.sort()
        pose_file = os.path.join(data_dir, instance, 'pose', "pose.json")

        with open(pose_file, "r") as read_content:
            data = json.load(read_content)

        print("img_files", self.img_files)
        self.all_c2w = []
        for img_file in self.img_files:
            print("img_file", img_file)
            c2w = np.array(data['transform'][img_file.split('.')[0]])
            self.all_c2w.append(c2w)
        self.focal = data['focal']
        self.img_size = data['img_size']
        print("self.focal, self img size", self.focal, self.img_size)

    def get_camera_pose_focal_by_frame_idx(self, frame_idx):
        print("frame idx", frame_idx)
        print("img file", self.img_files[frame_idx])
        c2w = self.all_c2w[frame_idx]
        focal = self.focal
        focal *=(self.config.img_wh[0]/self.img_size[0])
        print("focal", focal)
        return c2w, focal

    def scene_inference(
        self,
        rays: torch.Tensor,
        show_progress: bool = True,
    ):
        args = {}
        # args["train_config"] = self.ckpt_config.train

        B = rays.shape[0]
        results = defaultdict(list)
        chunk = self.config.chunk
        for i in tqdm(range(0, B, self.config.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_multi(
                    models=self.system.models,
                    embeddings=self.system.embeddings,
                    code_library=self.system.code_library,
                    # rays=rays[i : i + chunk],
                    rays_list=[rays[i : i + chunk]],
                    obj_instance_ids=[0],
                    N_samples=self.config.N_samples,
                    use_disp=self.config.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.config.N_importance,
                    chunk=self.config.chunk,  # chunk size is effective in val mode
                    white_back=False,
                    bckg_latent = True,
                    disentagled = True
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

    def generate_rays(self, obj_id, rays_o, rays_d):
        near = self.near
        far = self.far
        if obj_id == 0:
            batch_near = near * torch.ones_like(rays_o[:, :1])
            batch_far = far* torch.ones_like(rays_o[:, :1])
            # rays_o = rays_o / self.scale_factor
            rays = torch.cat([rays_o, rays_d, batch_near, batch_far], 1)  # (H*W, 8)

        else:
            bbox_mask, bbox_batch_near, bbox_batch_far = self.object_bbox_ray_helpers[
                str(obj_id)
            ].get_ray_bbox_intersections(
                rays_o,
                rays_d,
                None,
                # bbox_enlarge=self.bbox_enlarge / self.get_scale_factor(obj_id),
                bbox_enlarge=self.bbox_enlarge,
                canonical_rays = True  # in physical world
            )
            # for area which hits bbox, we use bbox hit near far
            # bbox_ray_helper has scale for us, do no need to rescale
            batch_near_obj, batch_far_obj = bbox_batch_near, bbox_batch_far
            # for the invalid part, we use 0 as near far, which assume that (0, 0, 0) is empty
            batch_near_obj[~bbox_mask] = torch.zeros_like(batch_near_obj[~bbox_mask])
            batch_far_obj[~bbox_mask] = torch.zeros_like(batch_far_obj[~bbox_mask])
            rays = torch.cat(
                [rays_o, rays_d, batch_near_obj, batch_far_obj], 1
            )  # (H*W, 8)
        rays = rays.cuda()
        return rays

    def render_edit(
        self,
        h: int,
        w: int,
        camera_pose_Twc: np.ndarray,
        focal: float,
        bckg_img = None,
        show_progress: bool = True,
        render_bg_only: bool = True,
        render_obj_only: bool = False,
        white_back: bool = False,
        object_pose_Twc = None
    ):
        directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        Twc = camera_pose_Twc
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

        processed_obj_id = []
        for obj_id in self.active_object_ids:
            # count object duplication
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            if obj_id == 0:
                # for scene, transform is Identity
                Tow = transform = np.eye(4)
                print("OBJ 0")
            else:
                print("OBJ 111111111")
                object_pose = self.object_pose_transform[
                    f"{obj_id}_{obj_duplication_cnt}"
                ]
                # Tow_orig = self.get_object_bbox_helper(
                #     obj_id
                # ).get_world_to_object_transform()
                Tow_orig = np.eye(4)
                transform = np.linalg.inv(Tow_orig) @ object_pose @ Tow_orig
                # transform = object_pose
                Tow = np.linalg.inv(transform)

            processed_obj_id.append(obj_id)
            Toc = Tow @ Twc
            # if obj_id !=0:
            #     Toc = object_pose_Twc 
            Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
            # all the rays_o and rays_d has been converted to NeRF scale
            rays_o, rays_d = get_rays(directions, Toc)
            rays = self.generate_rays(obj_id, rays_o, rays_d)
            # light anchor should also be transformed
            Tow = torch.from_numpy(Tow).float()
            transform = torch.from_numpy(transform).float()
            obj_ids.append(obj_id)
            rays_list.append(rays)

        print("rays", rays.shape)

        # split chunk
        B = rays_list[0].shape[0]
        chunk = self.config.chunk
        results = defaultdict(list)
        background_skip_bbox = self.get_skipping_bbox_helper()
        background_skip_bbox = None
        for i in tqdm(range(0, B, self.config.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_multi(
                    models=self.system.models,
                    embeddings=self.system.embeddings,
                    code_library=self.system.code_library,
                    rays_list=[r[i : i + chunk] for r in rays_list],
                    obj_instance_ids=obj_ids,
                    N_samples=self.config.N_samples,
                    use_disp=self.config.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.config.N_importance,
                    chunk=self.config.chunk,  # chunk size is effective in val mode
                    white_back=white_back,
                    background_skip_bbox=background_skip_bbox,
                    # bckg_img = bckg_img[i : i + chunk],
                    bckg_latent = True,
                    disentagled = True,
                    **args,
                )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v.detach().cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def remove_scene_object_by_ids(self, obj_ids):
        """
        Create a clean background by removing user specified objects.
        """
        self.object_to_remove = obj_ids
        for obj_id in obj_ids:
            self.initialize_object_bbox(obj_id)

    def reset_active_object_ids(self):
        self.active_object_ids = [0]

    def set_object_pose_transform(
        self,
        obj_id: int,
        pose: np.ndarray,
        obj_dup_id: int = 0,  # for object duplication
    ):
        self.active_object_ids.append(obj_id)
        if obj_id not in self.active_object_ids:
            self.initialize_object_bbox(obj_id)
        self.object_pose_transform[f"{obj_id}_{obj_dup_id}"] = pose

    def initialize_object_bbox(self, obj_id: int):
        self.object_bbox_ray_helpers[str(obj_id)] = BBoxRayHelper(
            self.config, obj_id
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