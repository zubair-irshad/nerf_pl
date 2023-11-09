import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import open3d as o3d
import torch
import kornia as kn
from PIL import Image
from vis_nocs_utils import *
import os
    
def get_nocs(K, depth, segmap, obj_pose, instance_id_to_name, scale=1, extrinsics=None):
    unitcube = unit_cube()
    depth[depth >20.0] = 0.0
    depth_pts = get_pointclouds(depth, K, width = 640, height = 480)
    H = 480
    W = 640

    pc_homopoints = convert_points_to_homopoints(depth_pts.T)
    morphed_pc_homopoints = extrinsics @ pc_homopoints
    depth_pts = convert_homopoints_to_points(morphed_pc_homopoints).T
    xyz_orig = torch.from_numpy(depth_pts).reshape(H,W,3).permute(2,0,1)
    xyz = torch.zeros_like(xyz_orig)
    #draw_pcd_and_box(xyz_orig, obj_pose)
    ids = np.unique(segmap)
    for id in ids:
        if id != 0:  # ignore background
            name = instance_id_to_name[id]
            cam_rot = np.array(obj_pose['obj_rotations'][name])[:3, :3]
            cam_t = np.array(obj_pose['obj_translations'][name]) * scale

            bbox = np.array(obj_pose['bbox_dimensions'][name])
            bbox_diff = bbox[0,2]+((bbox[1,2]-bbox[0,2])/2)
            cam_t[2] += bbox_diff
            instance_map = segmap == id
            # Apply inverse translation
            xyz[:, instance_map] = xyz_orig[:, instance_map] - torch.from_numpy(cam_t).unsqueeze(-1)
            xyz[:, instance_map] = torch.from_numpy(np.linalg.inv(cam_rot)) @ xyz[:, instance_map]
            # Object specific transform
            scaling_factor = 6
            # a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.flatten(1).t()))
            # o3d.visualization.draw_geometries([a, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1), *unitcube])
            xyz[:, instance_map]/= scaling_factor
            xyz[:, instance_map] += 0.5
    nocs = xyz
    nocs[:, segmap == 0] = 0

    return nocs.clip(0, 1), xyz_orig



# folder_path = '/home/zubairirshad/pd-api-py/PDMultiObj_Single_Scene/SF_6thAndMission_medium2/train'
# file_path = 'suv_medium_02-000.png'
instance_id_to_name = {1071:'midsize_muscle_01_blue', 1072: 'compact_luxury_001_body_silver', 1073:'compact_sport_01_gunmetal', 1074:'suv_medium_02_red'}

base_dir = '/home/zubairirshad/pd-api-py/PDMultiObj_Single_Scene/SF_6thAndMission_medium0'
base_dir_train = os.path.join(base_dir, 'val')

pose_dir_train = os.path.join(base_dir, 'train', 'pose')
pose_dir_val = os.path.join(base_dir, 'val', 'pose')


img_files = os.listdir(os.path.join(base_dir_train, 'rgb'))
img_files.sort()

nocs_save_dir = os.path.join(base_dir_train, 'nocs_2d')
os.makedirs(nocs_save_dir, exist_ok=True)

instance_save_dir = os.path.join(base_dir_train, 'instance_masks_2d')
os.makedirs(instance_save_dir, exist_ok=True)

for i, img_file in enumerate(img_files):
    depth_endpath = img_file.split('.')[0]+'.npz'
    depth_path = os.path.join(base_dir_train, 'depth', depth_endpath)

    depth = np.clip(np.load(depth_path, allow_pickle=True)['arr_0'], 0,100)
    img_path = os.path.join(base_dir_train, 'rgb', img_file)
    seg_path = os.path.join(base_dir_train, 'semantic_segmentation_2d', img_file)
    inst_path = os.path.join(base_dir_train, 'instance_segmentation_2d', img_file)
    seg_mask = Image.open(seg_path)
    inst_mask = Image.open(inst_path)

    segmap = np.array(inst_mask)
    segmap[segmap<1070] = 0
    instance_save_name = os.path.join(instance_save_dir, img_file)
    inst_save_img = Image.fromarray(segmap)
    inst_save_img.save(instance_save_name)

    
    #cv2.imwrite(instance_save_name, segmap)
    # _, all_c2w,  focal, img_size, RTs, raw_boxes = read_poses_val(pose_dir_train, pose_dir_val, img_files= img_files, output_boxes=True)
    all_c2w,  focal, img_size, RTs, raw_boxes = read_poses_val(pose_dir_val, img_files= img_files, output_boxes=True)

    K =  np.array([
                [focal, 0., 640 / 2.0],
                [0., focal, 480 / 2.0],
                [0., 0., 1.],
            ])

    #convert camera pose from NeRF to opencv coordinate system
    camera_pose =  convert_pose(np.array(all_c2w[i]))
    nocs, xyz_orig = get_nocs(K, depth, segmap, raw_boxes, instance_id_to_name, scale=1, extrinsics=camera_pose)
    nocs = nocs.permute(1,2,0).numpy()
    im = Image.fromarray((nocs * 255).astype(np.uint8))
    nocs_save_name = os.path.join(nocs_save_dir, img_file)
    im.save(nocs_save_name)
    print("done with image:", img_file, "\n")




