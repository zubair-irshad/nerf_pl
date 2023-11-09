
import sys
sys.path.append('/home/zubairirshad/nerf_pl')
from save_nocs_PD.utils import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import os
    save_dir = '/home/zubairirshad/pd-api-py/Vis_Single_Scene_SFGrant'
    dirname = '/home/zubairirshad/pd-api-py/PDMultiObj_Single_Scene_SFGrant'
    folders = os.listdir(dirname)

    for folder in folders:

        print("folder", folder)
        save_folder_dir = os.path.join(save_dir, folder)
        os.makedirs(save_folder_dir, exist_ok=True)
        base_dir_train = os.path.join(dirname, folder, 'train')
        print("base_dir_train", base_dir_train)
        img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
        img_files_train.sort()

        img_dir = os.path.join(base_dir_train, 'rgb')
        nocs_dir = os.path.join(base_dir_train, 'nocs_2d')
        sem_dir = os.path.join(base_dir_train, 'semantic_segmentation_2d')
        depth_dir = os.path.join(base_dir_train, 'depth')
        instance_dir = os.path.join(base_dir_train, 'instance_masks_2d')

        pose_dir_train = os.path.join(base_dir_train, 'pose')
        all_c2w_train, all_c2w_val, focal, img_wh, RTs = read_poses(pose_dir_train, img_files_train, output_boxes = True)
        
        obj_poses = preprocess_RTS_for_vis(RTs)
        RTs = get_RTs(obj_poses)
        for i, img_name in enumerate(img_files_train):
            print("img_name", img_name)
            img_path = os.path.join(img_dir, img_name)
            sem_path = os.path.join(sem_dir, img_name)
            instance_path = os.path.join(instance_dir, img_name)
            nocs_path = os.path.join(nocs_dir, img_name)

            print("img_path", img_path)
            image = Image.open(img_path)
            seg_map = Image.open(sem_path)
            inst_map = Image.open(instance_path)
            nocs2d = Image.open(nocs_path)

            depth_endpath = img_name.split('.')[0]+'.npz'
            depth_path = os.path.join(depth_dir, depth_endpath)
            depth = np.load(depth_path, allow_pickle=True)['arr_0']
            depth_vis = depth2inv(torch.tensor(depth).squeeze().unsqueeze(0).unsqueeze(0))
            depth_vis = viz_inv_depth(depth_vis)
            print("depth_vis", depth_vis.shape)
            depth_vis = Image.fromarray((depth_vis*255).astype(np.uint8))
            seg_rgb = get_seg_rgb(seg_map)
            seg_vis = Image.fromarray((seg_rgb).astype(np.uint8))

            inst_rgb = get_inst_rgb(inst_map)
            inst_rgb = get_merged_instance_rgb(image, inst_rgb)
            inst_vis = Image.fromarray((inst_rgb).astype(np.uint8))
            fov = 80
            focal = (640 / 2) / np.tan(( fov/ 2) / (180 / np.pi))
            intrinsics = np.array([
                    [focal, 0., 640 / 2.0],
                    [0., focal, 480 / 2.0],
                    [0., 0., 1.],
                ])
            K_matrix = np.eye(4)
            K_matrix[:3,:3] = intrinsics
            bbox_save_name = os.path.join(save_folder_dir, str(i)+'bbox_'+'.png')
            vis_bounding_box_image(all_c2w_train[i], image, RTs, K_matrix, bbox_save_name)
            

            nocs_save_name = os.path.join(save_folder_dir, str(i)+'_nocs'+'.png')
            rgb_save_name = os.path.join(save_folder_dir, str(i)+'rgb_'+'.png')
            sem_save_name = os.path.join(save_folder_dir, str(i)+'sem_'+'.png')
            inst_save_name = os.path.join(save_folder_dir, str(i)+'inst_'+'.png')
            sem_save_name = os.path.join(save_folder_dir, str(i)+'sem_'+'.png')
            depth_save_name = os.path.join(save_folder_dir, str(i)+'depth_'+'.png')
            

            image.save(rgb_save_name)
            nocs2d.save(nocs_save_name)
            depth_vis.save(depth_save_name)
            seg_vis.save(sem_save_name)
            inst_vis.save(inst_save_name)


