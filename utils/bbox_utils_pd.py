import os
import numpy as np
import torch
import copy
from utils.geo_utils import bbox_intersection_batch
from objectron.schema import annotation_data_pb2 as annotation_protocol
torch.set_printoptions(threshold=1000)
import json

class BBoxRayHelper:
    def __init__(self, hparams, instance_id):
        super().__init__()
        self.hparams = hparams
        self.load_auto_bbox_scale(instance_id)

    def load_auto_bbox_scale(self, instance):
        # data_dir = '/home/ubuntu/nerf_pl/data/PD'
        # instance = 'fullsize_sedan_01_body_red'
        # pose_file = os.path.join(data_dir, instance, 'pose', "pose.json")

        # with open(pose_file, "r") as read_content:
        #     data = json.load(read_content)
        # box = data['bbox_dimensions']
        # scale_factor = np.max(box)
        # box = box/scale_factor
        # box_1 =  np.array([[-0.931105, 0.0988361, -2.2721235752105713],
        #                     [0.931105, 1.2862364053726196, 2.1314218044281006]])
        # box_2 = np.array([[-0.969584, 0.14402827620506287, -2.613887310028076],
        #                     [0.969584, 1.8400439023971558, 2.6055867671966553]])
        # box_3 = np.array([[-1.018, 0.0, -2.6391286849975586],
        #                     [1.018, 2.0103869438171387, 2.7681970596313477]])
        # box_4 = np.array([[-0.885007, 0.120295, -2.41961],
        #                     [0.885007, 1.38131, 2.25796]])

        # all_boxes=[box_1, box_2, box_3, box_4]
        # box = all_boxes[instance-1]
        # scale_factor = 15
        # self.bbox_bounds = box/scale_factor

        # self.bbox_bounds = np.array([-box / 2, box / 2])


        box =np.array([[-0.982011, -2.7727229595184326, 0.13078086078166962],
        [0.98367, 2.5783050060272217, 1.4612261056900024]])
        # print("")
        scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
        self.bbox_bounds = box/scale_factor

        # self.bbox_bounds = np.array([-box / 2, box / 2])

    def get_ray_bbox_intersections(
        self, rays_o, rays_d, scale_factor=None, bbox_enlarge=0, canonical_rays = False
    ):
        if not canonical_rays:
            rays_o_bbox, rays_d_bbox = self.transform_rays_to_bbox_coordinates_objectron(
                rays_o, rays_d
            )
        else:
            if type(rays_o) is torch.Tensor:
                rays_o_bbox, rays_d_bbox = (
                    rays_o.detach().cpu().numpy(),
                    rays_d.detach().cpu().numpy(),
                )
            else:
                rays_o_bbox, rays_d_bbox = rays_o, rays_d

        bbox_enlarge = 0.3
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        if bbox_enlarge > 0.0:
            bbox_bounds[0][2] -= bbox_enlarge
            bbox_bounds[1,0] += 0.1
            # bbox_bounds[0,0] -= 0.1
            # bbox_bounds[0,1] -= 0.1
        elif bbox_enlarge < 0:
            bbox_bounds[0][2] -= bbox_enlarge

        bbox_mask, batch_near, batch_far = bbox_intersection_batch(
            bbox_bounds, rays_o_bbox, rays_d_bbox
        )
        bbox_mask, batch_near, batch_far = (
            torch.Tensor(bbox_mask).bool(),
            torch.Tensor(batch_near[..., None]),
            torch.Tensor(batch_far[..., None]),
        )
        print(batch_near[batch_near > 0], batch_far[batch_far > 0])
        print("batch_near, batch_far",batch_near, batch_far)

        return bbox_mask.cuda(), batch_near.cuda(), batch_far.cuda()

    def check_xyz_in_bounds(
        self,
        xyz: torch.Tensor,
        scale_factor: float = None,
        bbox_enlarge: float = 0,
        canonical_rays = False
    ):
        """
        scale_factor: we should rescale xyz to real size
        """
        if not canonical_rays:
            xyz = self.transform_xyz_to_bbox_coordinates(xyz)
            xyz = torch.from_numpy(xyz).float().cuda()
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        bbox_enlarge = 0.3
        print("bbox bounds before enlarge", bbox_bounds)
        if bbox_enlarge > 0.0:
            bbox_bounds[0][2] -= bbox_enlarge
            bbox_bounds[1,0] += 0.1
            bbox_bounds[0,0] -= 0.1
            bbox_bounds[0,1] -= 0.1
        elif bbox_enlarge < 0:
            bbox_bounds[0][2] -= bbox_enlarge
        print("bbox bounds after enlarge", bbox_bounds)
        x_min, y_min, z_min = bbox_bounds[0]
        x_max, y_max, z_max = bbox_bounds[1]
        in_x = torch.logical_and(xyz[:, 0] >= x_min, xyz[:, 0] <= x_max)
        in_y = torch.logical_and(xyz[:, 1] >= y_min, xyz[:, 1] <= y_max)
        in_z = torch.logical_and(xyz[:, 2] >= z_min, xyz[:, 2] <= z_max)
        in_bounds = torch.logical_and(in_x, torch.logical_and(in_y, in_z))
        return in_bounds


def check_in_any_boxes(
    boxes,
    xyz: torch.Tensor,
    scale_factor: float = None,
    bbox_enlarge: float = 0.0,
    canonical_rays = False
):
    need_reshape = False
    if len(xyz.shape) == 3:
        N1, N2, _ = xyz.shape
        xyz = xyz.view(-1, 3)
        need_reshape = True
    in_bounds = torch.zeros_like(xyz[:, 0]).bool()
    for k, box in boxes.items():
        in_bounds = torch.logical_or(
            box.check_xyz_in_bounds(xyz, scale_factor, bbox_enlarge, canonical_rays), in_bounds
        )
    if need_reshape:
        in_bounds = in_bounds.view(N1, N2)
    return in_bounds