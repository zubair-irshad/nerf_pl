import os
import numpy as np
import torch
import copy
# from utils.util import read_yaml, read_json
from utils.geo_utils import bbox_intersection_batch
from datasets.nocs_utils import load_depth, process_data, get_GT_poses, rebalance_mask, Pose
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from datasets.viz_utils import get_pointclouds_abspose

def convert_points_to_homopoints(points):
  """Project 3d points (3xN) to 4d homogenous points (4xN)"""
  assert len(points.shape) == 2
  assert points.shape[0] == 3
  points_4d = np.concatenate([
      points,
      np.ones((1, points.shape[1])),
  ], axis=0)
  assert points_4d.shape[1] == points.shape[1]
  assert points_4d.shape[0] == 4
  return points_4d


def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d


class BBoxRayHelper:
    def __init__(self, hparams, instance_id, idx, pose_avg, scale_factor):
        super().__init__()
        self.hparams = hparams
        self.scale_factor = scale_factor
        self.instance_id = instance_id
        self.pose_avg3by4 = pose_avg
        self.idx = idx
        imdata = read_images_binary(os.path.join(hparams.root_dir, 'sparse/0/images.bin'))
        self.image_paths = [os.path.join(hparams.root_dir, 'images', name)
                    for name in sorted([imdata[k].name for k in imdata])]
        self.read_nocs_bbox_info()

        # self.dataset_name = full_conf["dataset_name"]
        # assert self.dataset_name in ["scannet_base", "toydesk"]

        # if self.dataset_name == "scannet_base":
        #     self.scene_id = self.conf["scene_id"]
        #     self.read_bbox_info_scannet()
        # elif self.dataset_name == "toydesk":
        #     self.read_bbox_info_desk()

    def get_axis_align_mat(self, rescaled=False):
        if rescaled:
            axis_align_mat = copy.deepcopy(self.axis_align_mat)
            axis_align_mat[:3, 3] /= self.scale_factor
            return axis_align_mat
        else:
            return self.axis_align_mat

    # def get_world_to_object_transform(self):
    #     recenter = np.eye(4)
    #     # if self.dataset_name == "scannet_base":
    #     #     recenter[:3, 3] = -self.bbox_c
    #     print("self.pose_avg", self.pose_avg.shape)
    #     trans = recenter @ self.axis_align_mat @ self.pose_avg
    #     #trans = recenter @ self.axis_align_mat
    #     return trans  # Tow

    def get_world_to_object_transform(self, c2w):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1

        axis_align_mat = self.axis_align_mat.copy()*4.4
        axis_align_mat[3,3] = 1
        obj2world = c2w @ np.linalg.inv(flip_yz) @ axis_align_mat

#        obj2world = np.linalg.inv(flip_yz) @ self.axis_align_mat.copy()

        obj2world = self.convert_pose(obj2world)

        trans = np.linalg.inv(obj2world)

        # trans = copy.deepcopy(self.axis_align_mat)
        return trans  # Tow


    def get_camera_to_object_transform(self):
        trans = self.axis_align_mat
        return trans  # Tow

    def convert_pose(self, C2W):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        C2W = np.matmul(C2W, flip_yz)
        return C2W

    def read_nocs_bbox_info(self):
        # NOCS specific data loading
        image_path = self.image_paths[self.idx]
        img_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.')[0].split('_')[0])
        depth_full_path = img_path + '_depth.png'
        depth = load_depth(depth_full_path)
        data_dir = '/home/ubuntu/Downloads/nocs_data'
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_path, depth)                
        abs_poses, model_points = get_GT_poses(data_dir, img_path, class_ids, instance_ids, model_list, bboxes, is_pcd_out=False)

        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1

        for i in range(len(instance_ids)):
            if instance_ids[i] != self.instance_id:
                continue
            # obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
            # obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
            #rotated_pc, rotated_box, size = get_pointclouds_abspose(obj2world, model_points[i])
            
            pc = model_points[i]
            pc_hp = convert_points_to_homopoints(pc.T)
            scaled_homopoints = (abs_poses[i].scale_matrix @ pc_hp)
            scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
            size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
            self.axis_align_mat = np.eye(4)
            self.axis_align_mat[:3, :3] = abs_poses[i].camera_T_object[:3,:3]
            self.axis_align_mat[:3, 3] = abs_poses[i].camera_T_object[:3,3]
            # self.axis_align_mat = self.convert_pose(self.axis_align_mat)
            # self.axis_align_mat = np.linalg.inv(self.axis_align_mat)
            self.bbox_bounds = np.array([-size / 2, size / 2])
            self.size = size
            break
        self.pose_avg = np.eye(4)
        self.pose_avg[:3] = self.pose_avg3by4

    def read_bbox_info_scannet(self):
        # read axis_align_matrix
        scene_info_file = os.path.join(
            self.conf["scans_dir"], "{}/{}.txt".format(self.scene_id, self.scene_id)
        )
        lines = open(scene_info_file).readlines()
        for line in lines:
            if "axisAlignment" in line:
                axis_align_matrix = [
                    float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
                ]
                break
        self.axis_align_mat = np.array(axis_align_matrix).reshape(4, 4)

        # read bbox bounds
        scene_bbox = np.load(
            os.path.join(self.conf["bbox_dir"], "{}_bbox.npy".format(self.scene_id))
        )
        for b in scene_bbox:
            # if b[6] != self.conf['val_instance_id']:
            if b[6] != self.instance_id:
                continue
            length = np.array([b[3], b[4], b[5]]) * 0.5
            center = np.array([b[0], b[1], b[2]])
            self.bbox_bounds = np.array([center - length, center + length])
        self.bbox_c = center  # center in ScanNet aligned coordinate
        self.pose_avg = np.eye(4)
        self.pose_avg[:3, 3] = np.array(self.conf["scene_center"])

    def read_bbox_info_desk(self):
        from scipy.spatial.transform import Rotation as R

        j = read_json(self.conf["bbox_dir"])
        labels = j["labels"]
        for l in labels:
            if int(l["id"]) != self.instance_id:
                continue
            if "position" not in l["data"]:
                continue
            pos = l["data"]["position"]
            quat = l["data"]["quaternion"]
            scale = l["data"]["scale"]
            pos, scale = np.array(pos), np.array(scale)
            r = R.from_quat(quat)
            rmat = r.as_matrix()
            # self.bbox_c = pos - scale / 2
            self.bbox_c = pos
            # bbox = o3d.geometry.OrientedBoundingBox(center=pos, R=rmat, extent=scale)
            self.axis_align_mat = np.eye(4)
            self.axis_align_mat[:3, :3] = rmat
            self.axis_align_mat[:3, 3] = pos
            self.axis_align_mat = np.linalg.inv(self.axis_align_mat)
            self.bbox_bounds = np.array([-scale / 2, scale / 2])
            break

        self.pose_avg = np.eye(4)
        self.pose_avg[:3, 3] = np.array(self.conf["scene_center"])

    # def transform_rays_to_bbox_coordinates(self, rays_o, rays_d, scale_factor):
    #     if type(rays_o) is torch.Tensor:
    #         rays_o, rays_d = (
    #             rays_o.detach().cpu().numpy(),
    #             rays_d.detach().cpu().numpy(),
    #         )
    #     #unscale
    #     rays_o = rays_o * scale_factor
    #     # de-centralize
    #     T_orig_avg = self.pose_avg.squeeze()
    #     rays_o_bbox = (T_orig_avg[:3, :3] @ rays_o.T).T + T_orig_avg[:3, 3]
    #     rays_d_bbox = (T_orig_avg[:3, :3] @ rays_d.T).T
    #     #convert to bbox coordinates

    #     rays_o_bbox = rays_o
    #     rays_d_bbox = rays_d
    #     T_box_orig = self.axis_align_mat
    #     rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
    #     rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
    #     return rays_o_bbox, rays_d_bbox

    def transform_rays_to_bbox_coordinates_nocs(self, rays_o, rays_d, scale_factor, c2w=None, pose_delta=None):
        if type(rays_o) is torch.Tensor:
            rays_o, rays_d = (
                rays_o.detach().cpu().numpy(),
                rays_d.detach().cpu().numpy(),
            )
        # bottom = torch.tensor([0,0,0,1]).unsqueeze(0).cuda()
        if c2w is not None:
            c2w = np.concatenate((c2w.cpu().numpy(), np.array([[0,0,0,1]])), axis=0)

        # if pose_delta is not None:
        #     pose_delta = np.concatenate((pose_delta, np.array([[0,0,0,1]])), axis=0)

        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        # obj2world = c2w @ np.linalg.inv(flip_yz) @ self.axis_align_mat

        # change pose using camera transformation
        # axis_align_mat = self.axis_align_mat
        # axis_align_mat[:3, 3] += axis_align_mat[:3, :3] @ pose_delta

        # obj2world = np.linalg.inv(flip_yz) @ np.linalg.inv(pose_delta) @ self.axis_align_mat 
        # #obj2world = np.linalg.inv(flip_yz) @ self.axis_align_mat
        # # obj2world = self.convert_pose(obj2world)
        # T_box_orig = np.linalg.inv(obj2world)

        # pose_delta = self.convert_pose(pose_delta)
        # obj2world = pose_delta @ self.axis_align_mat
        # obj2world = pose_delta @ np.linalg.inv(flip_yz) @ self.axis_align_mat 
        #obj2world = np.linalg.inv(flip_yz) @ pose_delta @ self.axis_align_mat

        # T_c20 =  np.linalg.inv(pose_delta) @ self.axis_align_mat 
        # print("self.axis_align_mat", self.axis_align_mat)
        # print("pose_delta", pose_delta)
        # T_c20 =  np.linalg.inv(pose_delta) @ self.axis_align_mat
        # print("T_c20", T_c20)
        # print("self.axis_align_mat", pose_delta @ self.axis_align_mat)
        # # T_c20 = self.convert_pose(T_c20)

        # print("convert_pose(pose_delta)", self.convert_pose(pose_delta))
        # T_c20 = self.convert_pose(pose_delta) @ self.axis_align_mat
        # #T_c20 = self.convert_pose(T_c20)
        
        # print("pose_delta", pose_delta)
        # print("T_c20", T_c20)
        
        # print("c2w", c2w)

        # print("self.axis_align_mat.copy()", self.axis_align_mat.copy())
        # print("pose_delta", pose_delta)
        # T_c2o = pose_delta @ self.axis_align_mat.copy()
        # print("T_c2o", T_c2o)
        # T_c2o = pose_delta @ np.linalg.inv(flip_yz) @ self.axis_align_mat.copy()
        # print("T_c2o before pose convert", T_c2o)
        
        # T_c2o = self.convert_pose(T_c2o)

        # # in opencv
        # T_c2o = pose_delta @ self.axis_align_mat.copy()
        # #in opencv

        # obj2world_1 = c2w @ np.linalg.inv(flip_yz) @ T_c2o
        # obj2world_1 = self.convert_pose(obj2world_1)

        axis_align_mat = self.axis_align_mat.copy()*4.4
        axis_align_mat[3,3] = 1
        obj2world = c2w @ np.linalg.inv(flip_yz) @ axis_align_mat
        
        #obj2world = c2w @ np.linalg.inv(flip_yz) @ self.axis_align_mat.copy() 
        #obj2world = pose_delta @ np.linalg.inv(flip_yz) @ self.axis_align_mat 
        obj2world = self.convert_pose(obj2world)

        # now move the bounding boxes


        T_box_orig = np.linalg.inv(obj2world)
        

        # obj2world = np.linalg.inv(pose_delta) @ self.axis_align_mat 
        #T_box_orig = np.linalg.inv(obj2world)
        # #unscale
        # rays_o = rays_o * scale_factor
        # # de-centralize
        # T_orig_avg = self.pose_avg.squeeze()
        # rays_o_bbox = (T_orig_avg[:3, :3] @ rays_o.T).T + T_orig_avg[:3, 3]
        # rays_d_bbox = (T_orig_avg[:3, :3] @ rays_d.T).T
        # #convert to bbox coordinates

        rays_o_bbox = rays_o
        rays_d_bbox = rays_d
        # T_box_orig = self.axis_align_mat
        rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
        rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
        return rays_o_bbox, rays_d_bbox

    def transform_xyz_to_bbox_coordinates(self, xyz, scale_factor, c2w=None, pose_delta=None):
        if type(xyz) is torch.Tensor:
            xyz = xyz.detach().cpu().numpy()
        # # unscale
        # xyz = xyz * scale_factor
        # de-centralize
        # T_orig_avg = self.pose_avg.squeeze()
        # xyz_bbox = (T_orig_avg[:3, :3] @ xyz.T).T + T_orig_avg[:3, 3]
        # convert to bbox coordinates


        if c2w is not None:
            c2w = np.concatenate((c2w.cpu().numpy(), np.array([[0,0,0,1]])), axis=0)

        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1

        axis_align_mat = self.axis_align_mat.copy()*4.4
        axis_align_mat[3,3] = 1
        obj2world = c2w @ np.linalg.inv(flip_yz) @ axis_align_mat

        #obj2world = c2w @ np.linalg.inv(flip_yz) @ self.axis_align_mat.copy()
        obj2world = self.convert_pose(obj2world)
        T_box_orig = np.linalg.inv(obj2world)
        
        # T_box_orig = self.axis_align_mat
        xyz_bbox = xyz
        xyz_bbox = (T_box_orig[:3, :3] @ xyz_bbox.T).T + T_box_orig[:3, 3]
        return xyz_bbox

    def get_ray_bbox_intersections(
        self, rays_o, rays_d, scale_factor=None, bbox_enlarge=0, c2w=None, pose_delta=None
    ):
        if scale_factor is None:
            scale_factor = self.scale_factor
        # rays_o_bbox, rays_d_bbox = self.transform_rays_to_bbox_coordinates(
        #     rays_o, rays_d, scale_factor
        # )

        rays_o_bbox, rays_d_bbox = self.transform_rays_to_bbox_coordinates_nocs(
            rays_o, rays_d, scale_factor, c2w, pose_delta
        )
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        # if bbox_enlarge > 0:
        bbox_enlarge = 0.04
        bbox_z_min_orig = bbox_bounds[0][2]
        bbox_bounds[0] -= bbox_enlarge
        bbox_bounds[1] += bbox_enlarge
        bbox_bounds[0][2] = bbox_z_min_orig

        bbox_mask, batch_near, batch_far = bbox_intersection_batch(
            bbox_bounds, rays_o_bbox, rays_d_bbox
        )
        bbox_mask, batch_near, batch_far = (
            torch.Tensor(bbox_mask).bool(),
            torch.Tensor(batch_near[..., None]),
            torch.Tensor(batch_far[..., None]),
        )
        
        # print("batch_near", batch_near)
        # print("batch_far", torch.masked_select(batch_far, bbox_mask))
        # batch_near, batch_far = batch_near / scale_factor, batch_far / scale_factor

        # # print("batch_near", torch.masked_select(batch_near, bbox_mask))
        # # print("batch_far", torch.masked_select(batch_far, bbox_mask))
        return bbox_mask.cuda(), batch_near.cuda(), batch_far.cuda()

    def check_xyz_in_bounds(
        self,
        xyz: torch.Tensor,
        scale_factor: float = None,
        bbox_enlarge: float = 0,
        c2w=None, 
        pose_delta=None
    ):
        """
        scale_factor: we should rescale xyz to real size
        """
        if scale_factor is None:
            scale_factor = self.scale_factor
        xyz = self.transform_xyz_to_bbox_coordinates(xyz, scale_factor, c2w, pose_delta)
        xyz = torch.from_numpy(xyz).float().cuda()
        bbox_bounds = copy.deepcopy(self.bbox_bounds)

        bbox_enlarge = 0.04
        bbox_z_min_orig = bbox_bounds[0][2]
        bbox_bounds[0] -= bbox_enlarge
        bbox_bounds[1] += bbox_enlarge
        bbox_bounds[0][2] = bbox_z_min_orig
        bbox_bounds[0][2] -= bbox_enlarge

        # if bbox_enlarge > 0:
        #     z_min_orig = bbox_bounds[0][2]  # keep z_min
        #     bbox_bounds[0] -= bbox_enlarge
        #     bbox_bounds[1] += bbox_enlarge
        #     bbox_bounds[0][2] = z_min_orig
        # elif bbox_enlarge < 0:
        #     # make some margin near the ground
        #     bbox_bounds[0][2] -= bbox_enlarge

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
    c2w=None, 
    pose_delta=None
):
    need_reshape = False
    if len(xyz.shape) == 3:
        N1, N2, _ = xyz.shape
        xyz = xyz.view(-1, 3)
        need_reshape = True
    in_bounds = torch.zeros_like(xyz[:, 0]).bool()
    for k, box in boxes.items():
        in_bounds = torch.logical_or(
            box.check_xyz_in_bounds(xyz, scale_factor, bbox_enlarge, c2w, pose_delta), in_bounds
        )
    if need_reshape:
        in_bounds = in_bounds.view(N1, N2)
    return in_bounds
