import colorsys
from lineset import LineMesh
import cv2
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box
import numpy as np
import matplotlib.pyplot as plt
import json 
import os
import open3d as o3d


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

def unit_cube():
  points = np.array([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
  ]) - 0.5
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
  line_set = LineMesh(points, lines,colors=colors, radius=0.0008)
  line_set = line_set.cylinder_segments
  return line_set

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.5, 0.0)
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("o")] = rotate_view

    o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)

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

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W

# def read_poses_train(pose_dir_train, img_files, output_boxes = False):
#     pose_file_train = os.path.join(pose_dir_train, 'pose.json')
#     with open(pose_file_train, "r") as read_content:
#         data = json.load(read_content)

#     focal = data['focal']
#     img_wh = data['img_size']
#     obj_location = np.array(data["obj_location"])
#     all_c2w = []

#     for img_file in img_files:
#         c2w = np.array(data['transform'][img_file.split('.')[0]])
#         # c2w[:3, 3] = c2w[:3, 3] - obj_location
#         all_c2w.append(convert_pose_PD_to_NeRF(c2w))

#     all_c2w = np.array(all_c2w)
    
#     raw_boxes = data
#     # Get bounding boxes for object MLP training only
#     if output_boxes:
#         all_boxes = []
#         all_translations = []
#         all_rotations= []
#         for k,v in data['bbox_dimensions'].items():
#                 bbox = np.array(v)
#                 all_boxes.append(bbox)
#                 translation = np.array(data['obj_translations'][k])
#                 all_translations.append(translation)
#                 all_rotations.append(data["obj_rotations"][k])
#         RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
#         return all_c2w, focal, img_wh, RTs, raw_boxes
#     else:
#         return all_c2w, focal, img_wh


def read_poses(pose_dir_val, img_files, output_boxes = False):
    pose_file_val = os.path.join(pose_dir_val, 'pose.json')
    with open(pose_file_val, "r") as read_content:
        data_val = json.load(read_content)

    focal = data_val['focal']
    img_wh = data_val['img_size']
    obj_location = np.array(data_val["obj_location"])
    all_c2w = []

    for img_file in img_files:
        c2w = np.array(data_val['transform'][img_file.split('.')[0]])
        # c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w = np.array(all_c2w)
    
    raw_boxes = data_val
    # Get bounding boxes for object MLP training only
    if output_boxes:
        all_boxes = []
        all_translations = []
        all_rotations = []
        all_names = []
        for k, v in data_val['bbox_dimensions'].items():
                bbox = np.array(v)
                all_boxes.append(bbox)
                translation = np.array(data_val['obj_translations'][k])
                all_translations.append(translation)
                all_rotations.append(data_val["obj_rotations"][k])
                all_names.append(k)
        RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes, 'names': all_names}
        return all_c2w, focal, img_wh, RTs, raw_boxes
    else:
        return all_c2w, focal, img_wh

def get_RTs(obj_poses):
    all_boxes = []
    all_translations = []
    all_rotations= []
    for k,v in obj_poses['bbox_dimensions'].items():
            bbox = np.array(v)
            all_boxes.append(bbox)
            translation = np.array(obj_poses['obj_translations'][k])
            all_translations.append(translation)
            all_rotations.append(obj_poses["obj_rotations"][k])
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    return RTs

def draw_pcd_and_box(depth, obj_pose):
    RTs = get_RTs(obj_pose)
    all_pcd = []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth.flatten(1).t()))
    all_pcd.append(pcd)
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']
    for Rot,Tran,bbox in zip(all_rotations, all_translations, bbox_dimensions):
        Tran = np.array(Tran)

        bbox_diff = bbox[0,2]+((bbox[1,2]-bbox[0,2])/2)
        Tran[2] += bbox_diff
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)
        sca =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
        bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)

    custom_draw_geometry_with_key_callback(all_pcd)


def get_3d_bboxes(obj_pose):
    RTs = get_RTs(obj_pose)
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']
    for Rot,Tran,bbox in zip(all_rotations, all_translations, bbox_dimensions):
        Tran = np.array(Tran)

        bbox_diff = bbox[0,2]+((bbox[1,2]-bbox[0,2])/2)
        Tran[2] += bbox_diff
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)
        sca =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
        bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)

    custom_draw_geometry_with_key_callback(all_pcd)


def get_pointclouds(depth, intrinsics, width, height):
    xmap = np.array([[y for y in range(width)] for z in range(height)])
    ymap = np.array([[z for y in range(width)] for z in range(height)])
    cam_cx = intrinsics[0,2]
    cam_fx = intrinsics[0,0]
    cam_cy = intrinsics[1,2]
    cam_fy = intrinsics[1,1]

    depth_masked = depth.reshape(-1)[:, np.newaxis]
    xmap_masked = xmap.flatten()[:, np.newaxis]
    ymap_masked = ymap.flatten()[:, np.newaxis]
    pt2 = depth_masked
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    return points


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None