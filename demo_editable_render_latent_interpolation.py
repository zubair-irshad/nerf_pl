import sys
import os
from opt import get_opts
import cv2
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import imageio
import numpy as np
from tqdm import tqdm
from models.editable_renderer_objectron_canonical import EditableRenderer
from scipy.spatial.transform import Rotation
from datasets.depth_utils import *
from PIL import Image
from torchvision import transforms as T

def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    # radii = 0.005
    radii = 0
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def get_pure_rotation(progress_11: float, max_angle: float = 10):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose


def get_transformation_with_duplication_offset(progress, duplication_id: int):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=30)
    offset = 0.05
    if duplication_id > 0:
        trans_pose[0, 3] -= np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] -= 0.2
    else:
        trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] += 0.55
    return trans_pose


def main(config):
    render_path = f"debug/rendered_view/render_{config.prefix}/"
    os.makedirs(render_path, exist_ok=True)
    # intialize room optimizer
    renderer = EditableRenderer(config=config)
    renderer.load_frame_meta()
    # obj_id_list = config.obj_id_list  # e.g. [4, 6]
    obj_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
    for obj_id in obj_id_list:
        renderer.initialize_object_bbox(obj_id)

    renderer.remove_scene_object_by_ids(obj_id_list)
    W, H = config.img_wh
    # total_frames = config.total_frames
    # pose_frame_idx = config.test_frame
    total_frames = 50
    pose_frame_idx = 0
    edit_type = "pure_rotation"
    # background_name = '/home/ubuntu/nerf_pl/data/objectron/backgrounds/0004_color.png'
    # bckg_img = Image.open(background_name)
    # transform = T.ToTensor()
    # bckg_img = bckg_img.transpose(Image.Transpose.ROTATE_90)
    # bckg_img = transform(bckg_img) # (h, w, 3)
    # bckg_img = bckg_img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
    obj_id_counter = 0
    for idx in tqdm(range(total_frames)):
        # an example to set object pose
        # trans_pose = get_transformation(0.2)
        processed_obj_id = []

        obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
        progress = idx / total_frames
        print("progress", progress)
        obj_id = obj_id_list[obj_id_counter]
        
        obj_id_counter+=1
        if obj_id_counter>14:
            obj_id_counter=0

        if edit_type == "duplication":
            trans_pose = get_transformation_with_duplication_offset(
                progress, obj_duplication_cnt
            )
        elif edit_type == "pure_rotation":
            trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))

        renderer.set_object_pose_transform(obj_id, trans_pose, obj_duplication_cnt)
        processed_obj_id.append(obj_id)

        # Note: uncomment this to render original scene
        # results = renderer.render_origin(
        #     h=H,
        #     w=W,
        #     camera_pose_Twc=move_camera_pose(
        #         renderer.get_camera_pose_by_frame_idx(pose_frame_idx), idx / total_frames
        #     ),
        #     fovx_deg=getattr(renderer, "fov_x_deg_dataset", 60),
        # )

        # render edited scene
        results = renderer.render_edit(
            h=H,
            w=W,
            camera_pose_Twc=move_camera_pose(
                renderer.get_camera_pose_focal_by_frame_idx(pose_frame_idx)[0],
                idx / total_frames,
            ),
            focal=renderer.get_camera_pose_focal_by_frame_idx(pose_frame_idx)[1],
            bckg_img = None
        )
        image_out_path = f"{render_path}/render_{idx:04d}.png"
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        image_np = results[f'rgb_{typ}'].view(H, W, 3).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = np.rot90(image_np, axes=(1,0))
        imageio.imwrite(image_out_path, image_np)
        
        depth_out_path = f"{render_path}/render_{idx:04d}_depth.png"
        depth_pred = results[f'depth_{typ}'].view(H, W).cpu().numpy()

        depth_vis = depth2inv(torch.tensor(depth_pred).unsqueeze(0).unsqueeze(0))
        depth_vis = viz_inv_depth(depth_vis)
        depth_vis = np.rot90(depth_vis, axes=(1,0))

        # depth_img = (depth_pred - np.min(depth_pred)) / (max(np.max(depth_pred) - np.min(depth_pred), 1e-8))
        # depth_img_ = cv2.applyColorMap((depth_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        imageio.imwrite(depth_out_path, depth_vis)

    #         if hparams.depth_format == 'pfm':
    #             save_pfm(os.path.join(render_path, f'depth_{idx:03d}.pfm'), depth_pred)
    #         else:
    #             with open(os.path.join(render_path, f'depth_{idx:03d}'), 'wb') as f:
    #                 f.write(depth_pred.tobytes())

    # if hparams.save_depth:
    #     min_depth = np.min(depth_maps)
    #     max_depth = np.max(depth_maps)
    #     depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
    #     depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
    #     imageio.mimsave(os.path.join(render_path, f'_depth.gif'), depth_imgs_, fps=30)

        renderer.reset_active_object_ids()


if __name__ == "__main__":
    hparams = get_opts()
    main(hparams)