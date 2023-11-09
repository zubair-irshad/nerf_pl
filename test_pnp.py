import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import torch
import open3d as o3d
import colorsys
from lineset import LineMesh
# from scipy.spatial.transform import Rotation as R

def solvePnP(cam, image_points, object_points, return_inliers=False):
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    if image_points.shape[0] < 4:
        pose = np.eye(4)
        inliers = []
    else:
        image_points[:, [0, 1]] = image_points[:, [1, 0]]
        object_points = np.expand_dims(object_points, 1)
        image_points = np.expand_dims(image_points, 1)

        try:
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points.astype(float), cam,
                                                                                       dist_coeffs, iterationsCount=300,
                                                                                       reprojectionError=1.)[:4]
        except:
            success = False
            rotation_vector = np.zeros([3, 1])
            translation_vector = np.zeros([3, 1])
            inliers = []


        # Get a rotation matrix
        pose = np.eye(4)
        if success:
            pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            pose[:3, 3] = np.squeeze(translation_vector)

        if inliers is None:
            inliers = []

    if return_inliers:
        return pose, len(inliers)
    else:
        return pose

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

def get_RTs(obj_poses):
    all_boxes = []
    all_translations = []
    all_rotations= []
    for i, bbox in enumerate(obj_poses['bbox_dimensions']):
            all_boxes.append(bbox)
            translation = np.array(obj_poses['obj_translations'][i])
            all_translations.append(translation)
            all_rotations.append(obj_poses["obj_rotations"][i])
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    return RTs

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
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)
        sca = bbox
        bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)

    custom_draw_geometry_with_key_callback(all_pcd)

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

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W

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


def read_poses(pose_dir_train, pose_dir_val, img_files, output_boxes = False):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    pose_file_val = os.path.join(pose_dir_val, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)
    with open(pose_file_val, "r") as read_content:
        data_val = json.load(read_content)

    focal = data['focal']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w = []

    for img_file in img_files:
        print("img file", img_file)
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        # c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    for img_file in img_files:
        c2w = np.array(data_val['transform'][img_file.split('.')[0]])
        # c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w = np.array(all_c2w)

    # pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
    # all_c2w[:, :3, 3] *= pose_scale_factor

    all_c2w_train = all_c2w[:99, :, :]
    all_c2w_test = all_c2w[99:, :, :]
    
    raw_boxes = data
    # Get bounding boxes for object MLP training only
    if output_boxes:
        all_boxes = []
        all_translations = []
        all_rotations= []
        for k,v in data['bbox_dimensions'].items():
                print("k", k)
                bbox = np.array(v)
                print("bbox", bbox)
                all_boxes.append(bbox)
                translation = np.array(data['obj_translations'][k])
                all_translations.append(translation)
                all_rotations.append(data["obj_rotations"][k])
        RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
        return all_c2w_train, all_c2w_test, focal, img_wh, RTs, raw_boxes
    else:
        return all_c2w_train, all_c2w_test, focal, img_wh

base_dir = '/home/zubairirshad/pd-api-py/data/PDMultiObj_Single_Scene/SF_6thAndMission_medium2'
base_dir_train = os.path.join(base_dir, 'train')
img_files = os.listdir(os.path.join(base_dir_train, 'rgb'))
img_files.sort()
pose_dir_train = os.path.join(base_dir, 'train', 'pose')
pose_dir_val = os.path.join(base_dir, 'val', 'pose')

all_c2w, _,  focal, img_size, RTs, raw_boxes = read_poses(pose_dir_train, pose_dir_val, img_files= img_files, output_boxes=True)

folder_path = '/home/zubairirshad/pd-api-py/data/PDMultiObj_Single_Scene/SF_6thAndMission_medium2/train'
file_path = 'suv_medium_02-100.png'
depth_endpath = file_path.split('.')[0]+'.npz'
depth_path = os.path.join(folder_path,'depth', depth_endpath)

depth = np.clip(np.load(depth_path, allow_pickle=True)['arr_0'], 0,100)


img_path = os.path.join(folder_path, 'rgb', file_path)

cx = 640 / 2.0
cy = 480 / 2.0

# focal = focal/2
# cx = cx/2
# cy = cy/2
K =  np.array([
            [focal, 0., cx],
            [0., focal, cy],
            [0., 0., 1.],
        ])

# K[0,0] = K[0,0]/2
# K[1,1] = K[1,1]/2
# K[0,2] = K[0,2]/2
# K[1,2] = K[1,2]/2

import cv2
instance_id_to_name = {1071:'midsize_muscle_01_blue', 1072: 'compact_luxury_001_body_silver', 1073:'compact_sport_01_gunmetal', 1074:'suv_medium_02_red'}
seg_path = os.path.join(folder_path, 'semantic_segmentation_2d', file_path)
inst_path = os.path.join(folder_path, 'instance_segmentation_2d', file_path)
seg_mask = Image.open(seg_path)
inst_mask = Image.open(inst_path)

# inst_mask = cv2.resize(np.array(inst_mask), (320,240), interpolation =cv2.INTER_NEAREST)

segmap = np.array(inst_mask)
plt.imshow(segmap)
plt.show()
segmap[segmap<1070] = 0
instance_id_to_name = {1071:'midsize_muscle_01_blue', 1072: 'compact_luxury_001_body_silver', 1073:'compact_sport_01_gunmetal', 1074:'suv_medium_02_red'}
plt.imshow(segmap)
plt.show()


# Load NOCS image

#coord_path = '/home/zubairirshad/generalizable-object-representations/ckpts/3viewtest_novelobj/nocs_000.png'
coord_path = '/home/zubairirshad/pd-api-py/data/PDMultiObj_Single_Scene/SF_6thAndMission_medium2/train/nocs_2d/suv_medium_02-100.png'

coord_map = cv2.imread(coord_path)[:, :, :3]
#coord_map = cv2.resize(coord_map.astype(np.uint8), None, fx=2., fy=2., interpolation=cv2.INTER_NEAREST)
coord_map = coord_map[:, :, (2, 1, 0)]
# flip z axis of coord map
coord_map = np.array(coord_map, dtype=np.float32) / 255

plt.imshow(coord_map)
plt.show()

obj_poses = {}
obj_poses["bbox_dimensions"] = []
obj_poses["obj_translations"] = []
obj_poses["obj_rotations"] = []
unitcube = unit_cube()

def transform_pts(pts, poses):
    pc_homopoints = convert_points_to_homopoints(pts.T)
    morphed_pc_homopoints = poses @ pc_homopoints
    pts_world = convert_homopoints_to_points(morphed_pc_homopoints).T
    return pts_world


s = 6
all_rotated_pcds = []
ids = np.unique(segmap)
print("ids",ids)
for id in ids:
    print("id" ,id)
    if id != 0:  # ignore background
        name = instance_id_to_name[id]
        bbox = np.array(raw_boxes["bbox_dimensions"][name])
        bbox_extent = np.array([(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])])
        obj_poses["bbox_dimensions"].append(bbox_extent)

        instance_map = segmap == id
        print("instance_map", instance_map.shape)
        idxs = np.where(instance_map)
        coords = np.multiply(coord_map, np.expand_dims(instance_map, axis=-1))
        print(idxs, np.array(idxs).shape)
        # plt.imshow(coords)
        # plt.show()
        print("coords", coords.shape) 
        coord_vis = s* (coords[idxs[0], idxs[1], :] - 0.5)
        coord_vis = coord_vis.reshape(-1,3)
        #a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord_vis))
        #o3d.visualization.draw_geometries([a, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1), *unitcube])
        # plt.imshow(instance_map)
        # plt.show()
        # idxs = np.where(instance_map)
        # print(idxs, idxs.shape)

        coord_pts = s * (coords[idxs[0], idxs[1], :] - 0.5)
        coord_pts = coord_pts[:, :, None]

        img_pts = np.array([idxs[1], idxs[0]]).transpose()
        img_pts = img_pts[:, :, None].astype(float)
        distCoeffs = np.zeros((4, 1))    # no distoration
        retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, K, distCoeffs)
        R, _ = cv2.Rodrigues(rvec)
        T = np.squeeze(tvec)

        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = T

        print("R", R, "\n\n")
        print("T", T, "\n\n")

        # predicted_pose, n_inliers = solvePnP(K, img_pts, coord_pts, return_inliers=True)
        # rot = predicted_pose[:3, :3]
        # quat = R.from_matrix(rot).as_quat()
        # quat = np.concatenate([quat[3:], quat[:3]])  # reformat
        # trans = predicted_pose[:3, 3]
        # pose = np.concatenate([predicted_pose[:3], np.array([[0, 0, 0, 1]])], axis=0)
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        coordinate_frame.transform(pose)
        transformed_cords = transform_pts(coord_vis, pose)
        all_rotated_pcds.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed_cords)))
        all_rotated_pcds.append(coordinate_frame)
        obj_poses["obj_rotations"].append(R)
        obj_poses["obj_translations"].append(T)

o3d.visualization.draw_geometries(all_rotated_pcds)
depth_pts = get_pointclouds(depth, K, width = 640, height = 480)
print("depth pts",depth_pts.shape)
H = 480
W = 640

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

# extrinsics = convert_pose(np.array(all_c2w[0]))
# pc_homopoints = convert_points_to_homopoints(depth_pts.T)
# morphed_pc_homopoints = extrinsics @ pc_homopoints
# depth_pts = convert_homopoints_to_points(morphed_pc_homopoints).T
xyz_orig = torch.from_numpy(depth_pts).reshape(H,W,3).permute(2,0,1)

draw_pcd_and_box(xyz_orig, obj_poses)