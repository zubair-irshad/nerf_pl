import sys
import os
from opt import get_opts

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import imageio
import numpy as np
from tqdm import tqdm
from models.editable_renderer import EditableRenderer
# from utils.util import get_timestamp
from scipy.spatial.transform import Rotation
from datasets.viz_utils import plot_3d_bbox

def center_pose_from_avg(pose_avg, pose):
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_homo = np.eye(4)
    pose_homo[:3] = pose[:3]
    pose_centered = np.linalg.inv(pose_avg_homo) @ pose_homo  # (4, 4)
    # pose_centered = pose_centered[:, :3] # (N_images, 3, 4)
    return pose_centered

def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.05
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    Tc2_c1 = np.eye(4)
    Tc2_c1[:3, 3] = pose[:3, :3].copy() @ center

    pose_transformed = pose.copy()
    pose_transformed[:3, 3] += pose_transformed[:3, :3] @ center


    return pose_transformed, Tc2_c1


def get_pure_rotation(progress_11: float, max_angle: float = 20):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose


def get_transformation_with_duplication_offset(progress, duplication_id: int):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=10)
    offset = 0.05
    if duplication_id > 0:
        trans_pose[0, 3] -= np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] -= 0.2
    else:
        trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] += 0.55
    return trans_pose


def main(hparams):
    render_path = f"debug/rendered_view/{hparams.prefix}/"
    os.makedirs(render_path, exist_ok=True)
    # intialize room optimizer
    renderer = EditableRenderer(hparams=hparams)
    poses_avg, scale_factor, focal = renderer.read_meta()
    obj_id_list = [1,2,3,4,5,6]  # e.g. [4, 6]
    W, H = hparams.img_wh
    total_frames = 50
    pose_frame_idx = 350
    edit_type = 'pure_rotation'

    for obj_id in obj_id_list:
        renderer.initialize_object_bbox(hparams, obj_id, pose_frame_idx, poses_avg.copy(), scale_factor)
    #renderer.remove_scene_object_by_ids(hparams, obj_id_list, pose_frame_idx, poses_avg, scale_factor)

    for idx in tqdm(range(total_frames)):
    #     # an example to set object pose
    #     # trans_pose = get_transformation(0.2)
        processed_obj_id = []
        for obj_id in obj_id_list:
            # count object duplication, which is generally to be zero,
            # but can be increased if duplication operation happened
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            progress = idx / total_frames

            if edit_type == "duplication":
                trans_pose = get_transformation_with_duplication_offset(
                    progress, obj_duplication_cnt
                )
            elif edit_type == "pure_rotation":
                trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))

            renderer.set_object_pose_transform(hparams, obj_id, pose_frame_idx, trans_pose, obj_duplication_cnt, poses_avg, scale_factor)
            processed_obj_id.append(obj_id)

        # Note: uncomment this to render original scene
        # results = renderer.render_origin(
        #     h=H,
        #     w=W,
        #     camera_pose_Twc=move_camera_pose(
        #         renderer.get_camera_pose_by_frame_idx(pose_frame_idx), idx / total_frames
        #     ),
        #     focal = focal,
        # )
        # results = renderer.render_origin(
        #     h=H,
        #     w=W,
        #     camera_pose_Twc=renderer.poses[idx],
        #     focal = focal,
        # )

        # render edited scene
        # c2w = renderer.get_camera_pose_by_frame_idx(pose_frame_idx).copy()
        # c2w = center_pose_from_avg(poses_avg.copy(), c2w)
        # c2w[:, 3] /= scale_factor


        camera_pose_Twc, Tc2_c1 = move_camera_pose(
                renderer.get_camera_pose_by_frame_idx(pose_frame_idx).copy(),
                idx / total_frames,
            )

        Twc = center_pose_from_avg(poses_avg.copy(), camera_pose_Twc.copy())
        Twc = Twc/scale_factor


        # c2w_after = camera_pose_Twc.copy()
        # c2w_after = center_pose_from_avg(poses_avg.copy(), c2w_after)
        # Tc2_c1_scaled = np.linalg.inv(c2w_after) @ c2w
        # Tc2_c1_scaled[..., 3] /= scale_factor
        # Tc2_c1_scaled[3,3] = 1
        # Tc2_c1[..., 3] /= scale_factor
        # Tc2_c1[3,3] = 1

        results, object_pose_and_size = renderer.render_edit(
            h=H,
            w=W,
            camera_pose_Twc=camera_pose_Twc,
            camera_pose_Twc_origin = renderer.get_camera_pose_by_frame_idx(pose_frame_idx).copy(),
            focal = focal,
            pose_delta = Tc2_c1
        )
        image_out_path = f"{render_path}/render_{idx:04d}.png"
        # image_np = results["rgb_fine"].view(H, W, 3).detach().cpu().numpy()

        img_pred = np.clip(results["rgb_fine"].view(H, W, 3).detach().cpu().numpy(), 0, 1)
        image_pred = (img_pred * 255).astype(np.uint8)

        plot_3d_bbox(object_pose_and_size, image_pred, image_out_path, Tc2_c1, Twc)
        # imageio.imwrite(image_out_path, (image_np * 255).astype(np.uint8))
        # imageio.imsave(image_out_path, img_pred)

        renderer.reset_active_object_ids()


if __name__ == "__main__":

    hparams = get_opts()
    main(hparams)
