import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import sys
import json
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv
from .nocs_utils import Pose, umeyama
from .ray_utils import world_to_ndc
import colorsys
import os
import trimesh
import copy

def transform_xyz_to_bbox_coordinates(xyz, axis_align_mat):
    if type(xyz) is torch.Tensor:
        xyz = xyz.detach().cpu().numpy()
    xyz_bbox = xyz
    # convert to bbox coordinates
    T_box_orig = np.linalg.inv(axis_align_mat.copy())
    xyz_bbox = (T_box_orig[:3, :3] @ xyz_bbox.T).T + T_box_orig[:3, 3]
    return xyz_bbox
    
def check_xyz_in_bounds(
    xyz: torch.Tensor,
    axis_align_mat,
    bbox_bounds,
    bbox_enlarge: float = 0,
    canonical_rays = False
):
    """
    scale_factor: we should rescale xyz to real size
    """
    if not canonical_rays:
        xyz = transform_xyz_to_bbox_coordinates(xyz, axis_align_mat)
        xyz = torch.from_numpy(xyz).float()
    bbox_bounds = copy.deepcopy(bbox_bounds)
    bbox_enlarge = 0.06
    if bbox_enlarge > 0:
        bbox_z_min_orig = bbox_bounds[0][2]
        bbox_bounds[0] -= bbox_enlarge
        bbox_bounds[1] += bbox_enlarge
        bbox_bounds[0][2] = bbox_z_min_orig
        bbox_bounds[0][2] -= bbox_enlarge
    elif bbox_enlarge < 0:
        # make some margin near the ground
        bbox_bounds[0][2] -= bbox_enlarge
    x_min, y_min, z_min = bbox_bounds[0]
    x_max, y_max, z_max = bbox_bounds[1]
    in_x = torch.logical_and(xyz[:, 0] >= x_min, xyz[:, 0] <= x_max)
    in_y = torch.logical_and(xyz[:, 1] >= y_min, xyz[:, 1] <= y_max)
    in_z = torch.logical_and(xyz[:, 2] >= z_min, xyz[:, 2] <= z_max)
    in_bounds = torch.logical_and(in_x, torch.logical_and(in_y, in_z))

    print("in_bounds", in_bounds.shape)
    return in_bounds


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def draw_canonical_box(points):
  lines = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [4, 5],
      [4, 6],
      [5, 7],
      [6, 7],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
  ]

  colors = random_colors(len(lines))
  line_set = LineMesh(points, lines,colors=colors, radius=0.008)
  line_set = line_set.cylinder_segments
  return line_set

def unit_cube():
  points = np.array([
      [0, 0, 0],
      [2, 0, 0],
      [0, 2, 0],
      [2, 2, 0],
      [0, 0, 2],
      [2, 0, 2],
      [0, 2, 2],
      [2, 2, 2],
  ]) -1
  lines = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [4, 5],
      [4, 6],
      [5, 7],
      [6, 7],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
  ]

  colors = random_colors(len(lines))
  line_set = LineMesh(points, lines,colors=colors, radius=0.008)
  line_set = line_set.cylinder_segments
  return line_set

def line_set_mesh(points_array):
  open_3d_lines = [
        [0, 1],
        [7,3],
        [1, 3],
        [2, 0],
        [3, 2],
        [0, 4],
        [1, 5],
        [2, 6],
        # [4, 7],
        [7, 6],
        [6, 4],
        [4, 5],
        [5, 7],
    ]
  colors = random_colors(len(open_3d_lines))
  open_3d_lines = np.array(open_3d_lines)
  line_set = LineMesh(points_array, open_3d_lines,colors=colors, radius=0.001)
  line_set = line_set.cylinder_segments
  return line_set

class NOCS_Real():
  def __init__(self, height=480, width=640, scale_factor=1.):
    # This is to go from mmt to pyrender frame
    self.height = int(height / scale_factor)
    self.width = int(width / scale_factor)
    self.f_x = 591.0125
    self.f_y = 590.16775
    self.c_x = 322.525
    self.c_y = 244.11084
    self.stereo_baseline = 0.119559
    self.intrinsics = np.array([
            [self.f_x, 0., self.c_x, 0.0],
            [0., self.f_y, self.c_y, 0.0],
            [0., 0., 1., 0.0],
            [0., 0., 0., 1.],
        ])

x_width = 1.0
y_depth = 1.0
z_height = 1.0
_WORLD_T_POINTS = np.array([
    [0, 0, 0],  #0
    [0, 0, z_height],  #1
    [0, y_depth, 0],  #2
    [0, y_depth, z_height],  #3
    [x_width, 0, 0],  #4
    [x_width, 0, z_height],  #5
    [x_width, y_depth, 0],  #6
    [x_width, y_depth, z_height],  #7
]) - 0.5

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


def vis_ray_segmented(masks, class_ids, rays_o, rays_d, img, W, H):
    seg_mask = np.zeros([H, W])
    print("seg_mask", seg_mask.shape)
    for i in range(len(class_ids)):
        # plt.imshow(masks[:,:,i] > 0)
        # plt.show()
        seg_mask[masks[:,:,i] > 0] = np.array(class_ids)[i]
    
    ray_od = torch.stack([rays_o, rays_d], dim=1)
    print("ray_od", ray_od.shape)
    print("img[:, None, :]", img[:, None, :].shape)
    rays_rgb = np.concatenate([ray_od.numpy(), img[:, None, :]], 1)
    rays_rgb_obj = []
    rays_rgb_obj_dir = []
    select_inds=[]
    N_rays=2048

    for i in range(len(class_ids)):
        rays_on_obj = np.where(seg_mask.flatten() == class_ids[i])[0]
        print("rays_on_obj", rays_on_obj.shape)
        rays_on_obj = rays_on_obj[np.random.choice(rays_on_obj.shape[0], N_rays)]
        select_inds.append(rays_on_obj)

        obj_mask = np.zeros(len(rays_rgb), np.bool)
        obj_mask[rays_on_obj] = 1
        # rays_rgb_debug = np.array(rays_rgb)
        # rays_rgb_debug[rays_on_obj, :] += np.random.rand(3) #0.
        # img_sample = np.reshape(rays_rgb_debug[:, 2, :],[H, W, 3])
        # # plt.imshow(img_sample)
        # # plt.show()
        rays_rgb_obj.append(rays_o[rays_on_obj, :])
        rays_rgb_obj_dir.append(rays_d[rays_on_obj, :])

    select_inds = np.concatenate(select_inds, axis=0)
    obj_mask = np.zeros(len(rays_rgb), np.bool)
    obj_mask[select_inds] = 1


    print("select_inds", select_inds.shape, rays_rgb.shape)
    rays_rgb_debug = np.array(rays_rgb)

    rays_rgb_debug[select_inds, :] += np.random.rand(3) #0.
    img_sample = np.reshape(rays_rgb_debug[:, 2, :],[H, W, 3])
    plt.imshow(img_sample)
    plt.show()

    seg_vis = np.zeros([H, W]).flatten()
    seg_vis[obj_mask>0] = 1
    seg_vis = np.reshape(seg_vis, [H,W])
    plt.imshow(seg_vis)
    plt.show()
    return rays_rgb_obj, rays_rgb_obj_dir


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    return bbox_3d

def get_pointclouds_abspose(pose, pc, is_inverse = False
):
    if is_inverse:
        # pose.scale_matrix[0,0] = 1/pose.scale_matrix[0,0]
        # pose.scale_matrix[1,1] = 1/pose.scale_matrix[1,1]
        # pose.scale_matrix[2,2] = 1/pose.scale_matrix[2,2]
        print("pose scale matrix", pose.scale_matrix)
        pose.scale_matrix = np.linalg.inv(pose.scale_matrix)
        print("pose scale matrix", pose.scale_matrix)
        pose.camera_T_object = np.linalg.inv(pose.camera_T_object)

    print("pc", pc.shape)
    if is_inverse:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints = (pose.camera_T_object @ pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    else:
        # pc = pc*1000/220
        print("pose.scale_matrix", pose.scale_matrix)

        # pose.scale_matrix[0,0] = pose.scale_matrix[0,0]*1000/220
        # pose.scale_matrix[1,1] = pose.scale_matrix[1,1]*1000/220
        # pose.scale_matrix[2,2] = pose.scale_matrix[2,2]*1000/220

        pc_homopoints = convert_points_to_homopoints(pc.T)
        # pc_homopoints = pc_homopoints*1000/220
        morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T
        # morphed_pc_homopoints = morphed_pc_homopoints*1000/220

    pc_hp = convert_points_to_homopoints(pc.T)
    scaled_homopoints = (pose.scale_matrix @ pc_hp)
    scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
    size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
    box = get_3d_bbox(size)
    unit_box_homopoints = convert_points_to_homopoints(box.T)
    morphed_box_homopoints = pose.camera_T_object @ unit_box_homopoints
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    return morphed_pc_homopoints, morphed_box_points, size


def transform_rays_w2o(pose, pc, scale_matrix, is_inverse = False
):
    if is_inverse:
        scale_matrix[0,0] = 1/scale_matrix[0,0]
        scale_matrix[1,1] = 1/scale_matrix[1,1]
        scale_matrix[2,2] = 1/scale_matrix[2,2]
        pose = np.linalg.inv(pose)

    if is_inverse:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints = scale_matrix  @ (pose @ pc_homopoints)
        #morphed_pc_homopoints = (pose @ pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    else:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints =  pose @ (scale_matrix @ pc_homopoints)
        #morphed_pc_homopoints = pose @ ( pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    return morphed_pc_homopoints

def get_gt_pointclouds(pose, pc
):
    pc_homopoints = convert_points_to_homopoints(pc.T)
    unit_box_homopoints = convert_points_to_homopoints(_WORLD_T_POINTS.T)
    morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
    morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  
    morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    # box_points.append(morphed_box_points)
    return morphed_pc_homopoints, morphed_box_points


def viz_pcd_out(model_points, abs_poses):

    rotated_pc_o3d = []
    rotated_pc_box = []
    for i in range(len(model_points)):
        rotated_pc, rotated_box = get_gt_pointclouds(abs_poses[i], model_points[i])
        rotated_pc_o3d.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rotated_pc)))
        rotated_pc_box.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rotated_box)))
    o3d.visualization.draw_geometries(rotated_pc_o3d)


def transform_rays_orig(rays_o, directions, c2w, scale_matrix):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    #rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = rays_o + np.broadcast_to(c2w[:, 3][:3], rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    #print("rays_o", rays_o.shape, np.linalg.inv(scale_matrix).shape, np.linalg.inv(scale_matrix))
    # rays_o = ( rays_o @ torch.from_numpy(np.linalg.inv(scale_matrix))[:3,:3])
    # rays_d = ( rays_d @ torch.from_numpy(np.linalg.inv(scale_matrix))[:3, :3] )

    
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, rays_d


def transform_rays(rays_o, directions, o2w):
    # Rotate ray directions from camera coordinate to the world coordinate

    print("directions", directions.shape)
    rays_d = directions @ o2w[:3, :3] # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

    # Shift the camera rays to object center location in world frame and rotate by o2w
    rays_o = rays_o - np.broadcast_to(o2w[:, 3][:3], rays_d.shape)
    rays_o = rays_o @ o2w[:3, :3] # (H, W, 3)
    rays_o = rays_o.view(-1, 3)

    print("rays_d", rays_d.shape)
    print("o2w[:, 3]", o2w[:, 3].shape)

    return rays_o, rays_d

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def plot_NDC_trajectory(model_points, abs_poses, camera_poses, rays_o_all, rays_d_all, num, W, H, focal):
    
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))
    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)
    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    for i in range(len(model_points)):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        print(" model_points[i]",  model_points[i].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        #obj2world = opengl_pose @ abs_poses[i].camera_T_object
        obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        #obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        rotated_pc, rotated_box, size = get_pointclouds_abspose(obj2world, model_points[i])
        rotated_pcds.append(rotated_pc)
        rotated_boxes.append(rotated_box)
        T =  obj2world.camera_T_object
        objs2world.append(T)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.transform(T)
        mesh_frames.append(mesh_frame)

    rotated_pcds_ndc = []
    rotated_boxes_ndc = []
    for rotated_pcd, rotated_box in zip(rotated_pcds, rotated_boxes):
        ndc_pcd = world_to_ndc(rotated_pcd, W, H, focal, near=1.0)
        ndc_box = world_to_ndc(rotated_box, W, H, focal, near=1.0)
        rotated_pcds_ndc.append(ndc_pcd)
        rotated_boxes_ndc.append(ndc_box)

    fig = pv.figure()
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for pcds in rotated_pcds_ndc:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])

    for rotated_box in rotated_boxes_ndc:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])

    # Plot rays for each obj
    for rays_o, rays_d in zip(rays_o_all, rays_d_all):
        print("OBJ RAYS", rays_o.shape, rays_d.shape)
        for j in range(300):
            start = rays_o[j,:]
            end = rays_o[j,:] + rays_d[j,:]*1
            line = np.concatenate((start[None, :],end[None, :]), axis=0)
            fig.plot(line, c=(1.0, 0.5, 0.0))
    unitcube = unit_cube()
    for k in range(len(unitcube)):
        fig.add_geometry(unitcube[k]) 

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)


    # Plot all camera trajectories
    for pose in transformation_matrices:
        fig.plot_transform(A2B=pose, s=0.1)
        fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)

    fig.show()

def plot_world_trajectory(all_model_points, all_abs_poses, all_class_ids, camera_poses, poses_avg):
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))
    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)

    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    obbs = []

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = poses_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    # last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    # poses_homo = \
    #     np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    # print("pose_avg_homo", pose_avg_homo)
    # poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    print("===================================\n")
    print("===================================\n")
    R1 = []
    center = []
    for num, (model_points, abs_poses, class_ids) in enumerate(zip(all_model_points, all_abs_poses, all_class_ids)):
        # if num>10:
        #     continue
        model_points = [x for _, x in sorted(zip(class_ids, model_points))]
        abs_poses = [x for _, x in sorted(zip(class_ids, abs_poses))]

        class_ids = [y for y, _ in sorted(zip(class_ids, abs_poses))]

        print("class_ids", class_ids)

        # class_ids = [x for _, x in sorted(zip(class_ids, abs_poses))]
        for i in range(len(model_points)):
            flip_yz = np.eye(4)
            flip_yz[1, 1] = -1
            flip_yz[2, 2] = -1
            print(" model_points[i]",  model_points[i].shape)

            if num ==0:
                R1.append(abs_poses[i].camera_T_object[:3, :3])
                center.append(abs_poses[i].camera_T_object[:3, 3])
            # opengl_pose = convert_pose(transformation_matrices[num])
            #obj2world = opengl_pose @ abs_poses[i].camera_T_object
            obj2world = np.linalg.inv(transformation_matrices[num]) @ np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
            #obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object

            #poses_centered = np.linalg.inv(pose_avg_homo) @ abs_poses[i].camera_T_object # (N_images, 4, 4)
            #obj2world = abs_poses[i].camera_T_object
            # scale_inv = np.linalg.inv(abs_poses[i].scale_matrix)
            if num>0:
                R2 = abs_poses[i].camera_T_object[:3, :3]
                R = R1[i] @ R2.transpose()
                cos_theta = (np.trace(R) - 1) / 2
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                shift = np.linalg.norm(center[i] - abs_poses[i].camera_T_object[:3, 3]) * 100
                print("theta, object ", i, theta)
                print("shift ", i, shift)
            
            obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
            rotated_pc, rotated_box, size = get_pointclouds_abspose(obj2world, model_points[i])
            
            scales = size
            # print("scales", scales)
            bbox = o3d.geometry.OrientedBoundingBox(center=obj2world.camera_T_object[:3, 3], R=obj2world.camera_T_object[:3, :3], extent=scales)
            obbs.append(bbox)

            rotated_pcds.append(rotated_pc)
            rotated_boxes.append(rotated_box)
            T =  obj2world.camera_T_object
            objs2world.append(T)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            mesh_frame = mesh_frame.transform(T)
            mesh_frames.append(mesh_frame)
        print("-------------------------------\n")

    print("===================================\n")
    print("===================================\n")

    mesh_and_obbs = []
    fig = pv.figure()
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
        mesh_and_obbs.append(mesh_frame)

    print("len(obbs)", len(obbs))
    for obb in obbs:
        fig.add_geometry(obb)
        mesh_and_obbs.append(obb)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
        mesh_and_obbs.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])
            # mesh_and_obbs.append(cylinder_segments[k])

    # unitcube = unit_cube()
    # for k in range(len(unitcube)):
    #     fig.add_geometry(unitcube[k]) 
    #     mesh_and_obbs.append(unitcube[k])

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)
    # Plot all camera trajectories
    # for pose in transformation_matrices:
    #     fig.plot_transform(A2B=pose, s=0.1)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    

    # for pose in objs2world:
    #     fig.plot_transform(A2B=pose, s=1.0)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    
    # for pose in transformation_matrices_before:
    #     fig.plot_transform(A2B=pose, s=0.5)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()


def grid_generation(H, W):
    x = np.linspace(0, W-1, W)
    y = np.linspace(0, H-1, H)
    xv, yv = np.meshgrid(x, y)  # HxW
    xv = torch.from_numpy(xv.astype(np.float32)).to(dtype=torch.float32)
    yv = torch.from_numpy(yv.astype(np.float32)).to(dtype=torch.float32)
    ones = torch.ones_like(xv)
    meshgrid = torch.stack((xv, yv, ones), dim=2)  # HxWx3
    return meshgrid

def get_src_xyz_from_plane_disparity(meshgrid_src_homo,
                                     mpi_disparity_src,
                                     K_src_inv):
    """

    :param meshgrid_src_homo: 3xHxW
    :param mpi_disparity_src: BxS
    :param K_src_inv: Bx3x3
    :return:
    """
    B, S = mpi_disparity_src.size()
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33, meshgrid_src_homo_Bs3N)  # BSx3xHW
    xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3)  # BxSx3xHW
    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def plot_camera_trajectory(model_points, abs_poses, camera_poses, rays_o_all, rays_d_all, num, poses_before_center, batch_near_obj, batch_far_obj, depth,  pts_opengl, scale_factor, focal, intrinsics):
    num_id = 2
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))
    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)

    transformation_matrices_before = np.empty((len(poses_before_center), 4, 4))
    for i, camera_pose in enumerate(poses_before_center):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices_before[i] = pt.transform_from(R=R, p=p)

    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    obbs = []

    all_aamat = []
    all_bounds = []
    for i in range(len(model_points)):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        
        axis_align_mat = abs_poses[i].camera_T_object*4
        axis_align_mat[3,3] = 1
        obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ axis_align_mat
        all_aamat.append(convert_pose(obj2world))
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        rotated_pc, rotated_box, size = get_pointclouds_abspose(obj2world, model_points[i])
        scales = size
        bbox = o3d.geometry.OrientedBoundingBox(center=obj2world.camera_T_object[:3, 3], R=obj2world.camera_T_object[:3, :3], extent=scales)
        obbs.append(bbox)

        
        bbox_bounds = np.array([-size / 2, size / 2])
        all_bounds.append(bbox_bounds)
        print("axis_align_mat",axis_align_mat.shape)

        # orig_pose = Pose(camera_T_object=axis_align_mat, scale_matrix=abs_poses[i].scale_matrix)
        # _, _, orig_size = get_pointclouds_abspose(orig_pose, model_points[i])
        # original_obb = o3d.geometry.OrientedBoundingBox(center=orig_pose.camera_T_object[:3, 3], R=orig_pose.camera_T_object[:3, :3], extent=orig_size)
        
        # axis_align_mat = self.axis_align_mat.copy()*4.4
        # axis_align_mat[3,3] = 1

        

        
        

        # obbs.append(original_obb)
        rotated_pcds.append(rotated_pc)
        rotated_boxes.append(rotated_box)
        T =  obj2world.camera_T_object
        objs2world.append(T)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.transform(T)
        mesh_frames.append(mesh_frame)

    mesh_and_obbs = []

    #Plot depth
    # cam_cx = camera_real.c_x
    # cam_fx = camera_real.f_x
    # cam_cy = camera_real.c_y
    # cam_fy = camera_real.f_y

    cam_cx = 320
    cam_fx = focal
    cam_cy = 240
    cam_fy = focal

    print("cam_cx, cam_fx", cam_cx, cam_fx, cam_cy, cam_fy)

    indices = depth.flatten().shape[0]
    choose = np.random.choice(indices, 9113)
    depth_masked = depth.flatten()[choose][:, np.newaxis]
    xmap = np.array([[y for y in range(640)] for z in range(480)])
    ymap = np.array([[z for y in range(640)] for z in range(480)])
    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]

    pt2 = depth_masked/1000.0
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    # pcd_depth = o3d.geometry.PointCloud()
    # pcd_depth.points = o3d.utility.Vector3dVector(points)
    c2w_depth = transformation_matrices[num] @ np.linalg.inv(flip_yz)
    #c2w_depth = np.linalg.inv(flip_yz)
    points = points*1000/240
    pc_homopoints = convert_points_to_homopoints(points.T)
    morphed_pc_homopoints = c2w_depth @ pc_homopoints
    points = convert_homopoints_to_points(morphed_pc_homopoints).T

    # #fused depth
    # indices = fused_depth.flatten().shape[0]
    # choose = np.random.choice(indices, 9113)
    # depth_masked = depth.flatten()[choose][:, np.newaxis]
    # xmap = np.array([[y for y in range(640)] for z in range(480)])
    # ymap = np.array([[z for y in range(640)] for z in range(480)])
    # xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    # ymap_masked = ymap.flatten()[choose][:, np.newaxis]

    # pt2 = depth_masked/scale_factor
    # pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    # pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    # points_fused = np.concatenate((pt0, pt1, pt2), axis=1)
    # # pcd_depth = o3d.geometry.PointCloud()
    # # pcd_depth.points = o3d.utility.Vector3dVector(points)
    # # c2w_depth = transformation_matrices[num] @ np.linalg.inv(flip_yz)
    # c2w_depth = transformation_matrices[num]
    # #c2w_depth = np.linalg.inv(flip_yz)
    # points_fused = points_fused
    # pc_homopoints = convert_points_to_homopoints(points.T)
    # morphed_pc_homopoints = c2w_depth @ pc_homopoints
    # points_fused = convert_homopoints_to_points(morphed_pc_homopoints).T

    c2w_depth = transformation_matrices[num] @ np.linalg.inv(flip_yz)
    # c2w_depth = np.eye(4)
    print("pts_opengl", pts_opengl.shape)
    pc_homopoints = convert_points_to_homopoints(pts_opengl)
    morphed_pc_homopoints = c2w_depth @ pc_homopoints
    
    pts_opengl = convert_homopoints_to_points(morphed_pc_homopoints).T
    pts_opengl = pts_opengl/scale_factor
    



    # print("depth_masked", points)
    # print("pts_opengl", pts_opengl)
    # pts_opengl = np.transpose(pts_opengl, (1,0))
    # print("points", points.shape, pts_opengl.shape)
    # c, R, t = umeyama(points, pts_opengl) 

    # print ("Check:  a1*cR + t = a2  is", np.allclose(points.dot(c*R) + t, pts_opengl))
    # err = ((points.dot(c * R) + t - pts_opengl) ** 2).sum()
    # print ("Residual error", err)
    # print(c,R,t)
    # points_world = points.dot(c*R) + t

    pcd_depth = o3d.geometry.PointCloud()
    pcd_depth.points = o3d.utility.Vector3dVector(points)

    print("intrinsics", intrinsics)
    K_inv = np.linalg.inv(intrinsics)
    K_inv = torch.from_numpy(K_inv)
    
    disparity_end = 0.11
    disparity_start =  1.0
    S_coarse = 32
    disparity_coarse_src = torch.linspace(disparity_start, disparity_end, 
    S_coarse, dtype=torch.float32).unsqueeze(0).repeat(1, 1)  # BxS
    Height_tgt = 384
    Width_tgt = 640
    meshgrid = grid_generation(Height_tgt, Width_tgt)
    meshgrid = meshgrid.permute(2, 0, 1).contiguous()  # 3xHxW
    
    # Plot MPI syz_src_list
    xyz_src_BS3HW = get_src_xyz_from_plane_disparity(
        meshgrid,
        disparity_coarse_src,
        K_inv
    )

    print("xyz_src_BS3HW", xyz_src_BS3HW.shape, xyz_src_BS3HW)
    xyz_src = xyz_src_BS3HW[0].permute(0, 2, 3, 1).reshape(-1, 3)
    # draw.addBufferf('xyz_src', xyz_src)
    print("xyz_src", xyz_src.shape)
    pc_homopoints = convert_points_to_homopoints(xyz_src.T)
    morphed_pc_homopoints = c2w_depth @ pc_homopoints
    xyz_src = convert_homopoints_to_points(morphed_pc_homopoints).T

    xyz_src = torch.from_numpy(xyz_src)
    bbox_enlarge = 0
    canonical_rays= False
    xyz_src_BS3HW = copy.deepcopy(xyz_src)
    
    for axis_aligned_mat, bounds in zip(all_aamat,all_bounds):
        # axis_aligned_mat = all_aamat[0]
        print("axis_aligned_mat", axis_aligned_mat.shape)
        # bounds = all_bounds[0]
        in_bounds = torch.zeros_like(xyz_src[:,0]).bool()
        in_bounds = torch.logical_or(check_xyz_in_bounds(xyz_src, axis_aligned_mat, bounds,
        bbox_enlarge, canonical_rays), in_bounds
        )
        print("in_bounds", in_bounds.shape)
        print("xyz_src", xyz_src.shape)
        in_bounds = in_bounds.unsqueeze(1).repeat(1, 3)
        xyz_src_BS3HW[in_bounds] = 0

    xyz_src_BS3HW = xyz_src_BS3HW.numpy() 
    pcd_xyz_src = o3d.geometry.PointCloud()
    pcd_xyz_src.points = o3d.utility.Vector3dVector(xyz_src_BS3HW)

    
    
    # pcd_depth_fused = o3d.geometry.PointCloud()
    # pcd_depth_fused.points = o3d.utility.Vector3dVector(points_fused)

    # pcd_world_depth = o3d.geometry.PointCloud()
    # pcd_world_depth.points = o3d.utility.Vector3dVector(points_world)
    
    #pts_opengl = pts_opengl * 220/1000
    pcd_opengl = o3d.geometry.PointCloud()
    pcd_opengl.points = o3d.utility.Vector3dVector(pts_opengl)

    fig = pv.figure()
    # for camera_points in all_camera_points:
    #     fig.add_geometry(camera_points)
    fig.add_geometry(pcd_opengl)
    fig.add_geometry(pcd_depth)
    fig.add_geometry(pcd_xyz_src)
    
    # fig.add_geometry(pcd_depth_fused)
    # fig.add_geometry(pcd_world_depth)
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
        mesh_and_obbs.append(mesh_frame)
    for obb in obbs:
        fig.add_geometry(obb)
        mesh_and_obbs.append(obb)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
        mesh_and_obbs.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])
    
    for i, (rays_o, rays_d, batch_near, batch_far) in enumerate(zip(rays_o_all, rays_d_all, batch_near_obj, batch_far_obj)):
        #rays_o, rays_o, batch_near, batch_far = rays_o[10000:], rays_o[10000:], batch_near[10000:], batch_far[10000:]
        print("OBJ RAYS", rays_o.shape, rays_d.shape, batch_near.shape)
        print("batch_near", batch_near, batch_far)

        ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*0.5))

        rays_o = rays_o[ids, :]
        rays_d = rays_d[ids, :]

        batch_near = batch_near[ids]
        batch_far = batch_far[ids]

        # rays_o = np.random.choice(rays_o, rays_o.shape[0]*0.8)
        # rays_d = np.random.choice(rays_d, rays_d.shape[0]*0.8)

        # batch_near = np.random.choice(batch_near, batch_near.shape[0]*0.8)
        # batch_far = np.random.choice(batch_far, batch_far.shape[0]*0.8)

        if i ==1:
            for j in range(2500):
                # pts = [orig, orig + dmin * ray, orig + dmax * ray]
                # pts_idx = [[0, 1], [1, 2]]
                # colors = [[1, 0, 0], [0, 1, 0]]
                start = rays_o[j,:]
                end = rays_o[j,:] + rays_d[j,:]*batch_near[j, :]
                line = np.concatenate((start[None, :],end[None, :]), axis=0)
                fig.plot(line, c=(1.0, 0.5, 0.0))

                start = rays_o[j,:] + rays_d[j,:]*batch_near[j, :]
                end = rays_o[j,:] + rays_d[j,:]*4.5
                line = np.concatenate((start[None, :],end[None, :]), axis=0)
                fig.plot(line, c=(0.0, 1.0, 0.0))

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)
    # Plot all camera trajectories
    # for pose in transformation_matrices:
    #     fig.plot_transform(A2B=pose, s=0.1)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

    # for pose in objs2world:
    #     fig.plot_transform(A2B=pose, s=1.0)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    
    # for pose in transformation_matrices_before:
    #     fig.plot_transform(A2B=pose, s=0.5)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)

def plot_canonical_pcds(model_points, abs_poses, camera_poses, rays_o_all, rays_d_all, num, scale_factor):
    
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))

    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)

    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    obbs = []
    canonical_boxes = []
    all_sizes = []
    for i in range(len(model_points)):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        print(" model_points[i]",  model_points[i].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        #obj2world = opengl_pose @ abs_poses[i].camera_T_object
        #obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        #obj2world = abs_poses[i].camera_T_object
        # scale_inv = np.linalg.inv(abs_poses[i].scale_matrix)

        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        rotated_pc, rotated_box, size = get_pointclouds_abspose(obj2world, model_points[i])
        canonical_boxes.append(get_3d_bbox(size))
        all_sizes.append(np.array([-size / 2, size / 2]))
        bbox = o3d.geometry.OrientedBoundingBox(center=obj2world.camera_T_object[:3, 3], R=obj2world.camera_T_object[:3, :3], extent=size)
        obbs.append(bbox)

        rotated_pcds.append(rotated_pc)
        rotated_boxes.append(rotated_box)
        T =  obj2world.camera_T_object
        objs2world.append(T)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.transform(T)
        mesh_frames.append(mesh_frame)

    id_num = 0
    # transform back to canonical frame for one object
    canonical_pcds = []
    for i in range(len(rotated_pcds)):
        print("rotated_pcds[i]", rotated_pcds[1].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        # obj2world = opengl_pose @ abs_poses[0].camera_T_object
        #obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[0].camera_T_object
        obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = convert_pose(obj2world)
        print("IS EQUALLLL", np.equal (obj2world, abs_poses[i].camera_T_object))
        print("np.linalg.inv(flip_yz)", np.linalg.inv(flip_yz) @ flip_yz)
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        canonical_pc, canonical_box, sizes = get_pointclouds_abspose(obj2world, rotated_pcds[i], is_inverse=True)
        print("sizes", sizes)
        canonical_pcds.append(canonical_pc)

    # transform rays to caonical frames
    canonical_rays_all = []
    canonical_rays_d_all = []
    for ray_id, (rays_o, rays_d) in enumerate(zip(rays_o_all, rays_d_all)):
        if ray_id ==id_num:
            for pose_id in range(len(abs_poses)):
                #obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[0].camera_T_object
                obj2world = np.linalg.inv(flip_yz) @ abs_poses[pose_id].camera_T_object
                obj2world = convert_pose(obj2world)
                rays_o_numpy = rays_o.numpy()
                rays_d_numpy = rays_d.numpy()
                print("T_box_orig", np.linalg.inv(obj2world).shape, rays_o.shape)
                #convert rays to bbox
                T_box_orig = np.linalg.inv(obj2world)
                canonical_rays = (T_box_orig[:3, :3] @ rays_o_numpy.T).T + T_box_orig[:3, 3]
                canonical_rays_d = (T_box_orig[:3, :3] @ rays_d_numpy.T).T

                #canonical_rays, canonical_rays_d = transform_rays_orig(rays_o, rays_d, obj2world, scale_matrix)
                canonical_rays_all.append(canonical_rays)
                canonical_rays_d_all.append(canonical_rays_d)

    bbox_mask, batch_near, batch_far = get_ray_bbox_intersections(canonical_rays_all[id_num], canonical_rays_d_all[id_num], scale_factor, all_sizes[id_num])
    fig = pv.figure()
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])
    # Plot rays for each obj
    for i, (rays_o, rays_d) in enumerate(zip(rays_o_all, rays_d_all)):
        if i == id_num:
            print("OBJ RAYS", rays_o.shape, rays_d.shape)
            for j in range(1000):
                start = rays_o[j,:]
                end = rays_o[j,:] + rays_d[j,:]*1.5
                line = np.concatenate((start[None, :],end[None, :]), axis=0)
                fig.plot(line, c=(1.0, 0.5, 0.0))

    #Plot canonical pcds, unit cube and caonical rays
    for i, canonical_pc in enumerate(canonical_pcds):
        if i == id_num:
            fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(canonical_pc)))
    # unitcube = unit_cube()

    for i, canonical_box in enumerate(canonical_boxes):
        if i == id_num:
            cylinder_segments = line_set_mesh(canonical_box)
            for k in range(len(cylinder_segments)):
                fig.add_geometry(cylinder_segments[k])

    # for canonical_box in canonical_boxes:
    #     canonical_box_draw = draw_canonical_box(canonical_box)
    #     for k in range(len(canonical_box_draw)):
    #         fig.add_geometry(canonical_box_draw[k]) 

    for iter, (rays_o, rays_d) in enumerate(zip(canonical_rays_all, canonical_rays_d_all)):
        print("OBJ RAYS", rays_o.shape, rays_d.shape)
        if iter == id_num:
            for j in range(1000):
                start = rays_o[j,:]
                end = rays_o[j,:] + rays_d[j,:]*1.5
                line = np.concatenate((start[None, :],end[None, :]), axis=0)
                fig.plot(line, c=(1.0, 0.5, 0.0))

    # plot camera for all meshes:

    for obj2world in objs2world:
        obj2world_opengl = convert_pose(obj2world)
        # obj2world_opengl = np.linalg.inv(objs2world[0])
        # plot camera for first mesh
        fig.plot_camera(M=M, cam2world=obj2world_opengl, virtual_image_distance=0.1, sensor_size=sensor_size)
        fig.plot_transform(A2B=obj2world_opengl, s=0.3)

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)

    # Plot all camera trajectories
    # for pose in transformation_matrices:
    #     fig.plot_transform(A2B=pose, s=0.1)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

def get_trimesh_scales(obj_trimesh
):
    bounding_box = obj_trimesh.bounds
    current_scale = np.array([
        bounding_box[1][0] - bounding_box[0][0], bounding_box[1][1] - bounding_box[0][1],
        bounding_box[1][2] - bounding_box[0][2]
    ])
    return current_scale

def get_3D_rotated_box(pose, sizes
):
    box = get_3d_bbox(sizes)
    unit_box_homopoints = convert_points_to_homopoints(box.T)
    morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    return morphed_box_points

def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

def draw_saved_mesh_and_pose(abs_pose_outputs, model_lists, color_img):
    mesh_dir_path = '/home/zubair/Downloads/nocs_data/obj_models_transformed/real_test'
    rotated_meshes = []
    pcds = []
    camera = NOCS_Real()
    for j in range(len(abs_pose_outputs)):
        mesh_file_name = os.path.join(mesh_dir_path, model_lists[j]+'.obj')
        obj_trimesh = trimesh.load(mesh_file_name)
        sizes = get_trimesh_scales(obj_trimesh)
        obj_trimesh.apply_transform(abs_pose_outputs[j].scale_matrix)
        obj_trimesh.apply_transform(abs_pose_outputs[j].camera_T_object)
        obj_trimesh = obj_trimesh.as_open3d
        obj_trimesh.compute_vertex_normals()

        # single_mesh = obj_trimesh.as_open3d
        single_pcd = obj_trimesh.sample_points_uniformly(200000)
        points_mesh = convert_points_to_homopoints(np.array(single_pcd.points).T)
        points_2d_mesh = project(camera.intrinsics, points_mesh)
        points_2d_mesh = points_2d_mesh.T
        colors = []
        print(points_2d_mesh.shape)
        for k in range(points_2d_mesh.shape[0]):
            # im = Image.fromarray(np.uint8(color_img/255.0))
            color = color_img.getpixel((int(points_2d_mesh[k,0]), int(points_2d_mesh[k,1])))
            color = np.array(color)
            colors.append(color/255.0)
        single_pcd.colors  = o3d.utility.Vector3dVector(np.array(colors))
        single_pcd.normals = o3d.utility.Vector3dVector(np.zeros(
            (1, 3)))  # invalidate existing normals
        pcds.append(single_pcd)
        # o3d.visualization.draw_geometries([single_pcd])

        box_3D = get_3D_rotated_box(abs_pose_outputs[j], sizes)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        T = abs_pose_outputs[j].camera_T_object
        mesh_frame = mesh_frame.transform(T)
        rotated_meshes.append(obj_trimesh)
        rotated_meshes.append(mesh_frame)
        pcds.append(mesh_frame)

        cylinder_segments = line_set_mesh(box_3D)
        for k in range(len(cylinder_segments)):
            rotated_meshes.append(cylinder_segments[k])
            pcds.append(cylinder_segments[k])

    unitcube = unit_cube()
    for k in range(len(unitcube)):
        rotated_meshes.append(unitcube[k])
        pcds.append(unitcube[k])
        # fig.add_geometry(unitcube[k]) 

    #custom_draw_geometry_with_rotation(pcds, 4)
    o3d.visualization.draw_geometries(pcds)
    #custom_draw_geometry_with_rotation(pcds, 5)
    o3d.visualization.draw_geometries(rotated_meshes)
    #custom_draw_geometry_with_rotation(rotated_meshes, 5)
    return pcds

def get_object_rays_in_bbox(rays_o, rays_d, model_points, abs_poses, scale_factor, c2w=None):
    
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1

    if c2w is not None:
        c2w = np.concatenate((c2w, np.array([[0,0,0,1]])), axis=0)
    rays_o_obj_all = []
    rays_d_obj_all = []
    batch_near_obj_all = []
    batch_far_obj_all = []
    for pc, pose in zip(model_points, abs_poses):
        if c2w is not None:

            axis_align_mat = pose.camera_T_object*4.4
            axis_align_mat[3,3] = 1
            #axis_align_mat = pose.camera_T_object
            obj2world = c2w @ np.linalg.inv(flip_yz) @ axis_align_mat
            
            # obj2world = c2w @ np.linalg.inv(flip_yz) @ pose.camera_T_object
            obj2world = convert_pose(obj2world)
            # obj2world*=1000/220
            axis_aligned_mat = np.linalg.inv(obj2world)
        else:
            obj2world = np.linalg.inv(flip_yz) @ pose.camera_T_object
            # obj2world = convert_pose(obj2world)
            # obj2world*=1000/220
            axis_aligned_mat = np.linalg.inv(obj2world)

        pc_hp = convert_points_to_homopoints(pc.T)
        scaled_homopoints = (pose.scale_matrix @ pc_hp)
        scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
        size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
        bbox_bounds = np.array([-size / 2, size / 2])
        print("bbox_bounds", bbox_bounds)
        rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, scale_factor, axis_aligned_mat)
        
        #rays_o_obj, rays_d_obj = transform_rays_camera(rays_o_obj, rays_d_obj, c2w)
        rays_o_obj_all.append(rays_o_obj)
        rays_d_obj_all.append(rays_d_obj)
        batch_near_obj_all.append(batch_near_obj)
        batch_far_obj_all.append(batch_far_obj)

    return rays_o_obj_all, rays_d_obj_all, batch_near_obj_all, batch_far_obj_all


def get_object_rays_in_bbox_unscaled(rays_o, rays_d, model_points, abs_poses, scale_factor, c2w=None):
    
    if c2w is not None:
        c2w = np.concatenate((c2w, np.array([[0,0,0,1]])), axis=0)
    rays_o_obj_all = []
    rays_d_obj_all = []
    batch_near_obj_all = []
    batch_far_obj_all = []
    for pc, pose in zip(model_points, abs_poses):
        axis_align_mat = pose.camera_T_object
        axis_aligned_mat = np.linalg.inv(axis_align_mat)

        pc_hp = convert_points_to_homopoints(pc.T)
        scaled_homopoints = (pose.scale_matrix @ pc_hp)
        scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
        size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
        bbox_bounds = np.array([-size / 2, size / 2])
        rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, scale_factor, axis_aligned_mat)
        #rays_o_obj, rays_d_obj = transform_rays_camera(rays_o_obj, rays_d_obj, c2w)
        rays_o_obj_all.append(rays_o_obj)
        rays_d_obj_all.append(rays_d_obj)
        batch_near_obj_all.append(batch_near_obj)
        batch_far_obj_all.append(batch_far_obj)

    return rays_o_obj_all, rays_d_obj_all, batch_near_obj_all, batch_far_obj_all