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
# from models.editable_renderer_objectron_canonical import EditableRenderer
# from models.editable_renderer_co3d import EditableRenderer
from models.editable_renderer_pd import EditableRenderer
from datasets.depth_utils import *
from PIL import Image
from torchvision import transforms as T
from utils.test_poses import *

def main(config):
    render_path = f"debug/rendered_view/render_{config.prefix}/"
    os.makedirs(render_path, exist_ok=True)
    # intialize room optimizer
    renderer = EditableRenderer(config=config)
    renderer.load_frame_meta()
    # obj_id_list = config.obj_id_list  # e.g. [4, 6]
    obj_id_list = [1]
    for obj_id in obj_id_list:
        renderer.initialize_object_bbox(obj_id)

    renderer.remove_scene_object_by_ids(obj_id_list)
    W, H = config.img_wh
    # total_frames = config.total_frames
    # pose_frame_idx = config.test_frame
    total_frames = 90
    pose_frame_idx = 15
    # edit_type = "scale"
    edit_type = "pure_rotation"

    # background_name = '/home/ubuntu/nerf_pl/grocery_bckg_6.jpg'
    # bckg_img = Image.open(background_name)
    # transform = T.ToTensor()
    # # newsize = (1440, 1920)
    # newsize = (1440, 1920)
    # bckg_img = bckg_img.resize(newsize)
    # bckg_img = bckg_img.transpose(Image.Transpose.ROTATE_90)
    # bckg_img = transform(bckg_img) # (h, w, 3)
    # print("bckg_img", bckg_img.shape)
    # bckg_img = bckg_img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
    # print("bckg_img", bckg_img.shape)

    # use spheric test poses
    #poses_test = create_spheric_poses(1.7, total_frames)
    # use train as test poses
    poses_test = renderer.all_c2w

    #use archimedan spiral
    locations = get_archimedean_spiral(sphere_radius=1.5)
    poses_test = [look_at(loc, [0,0,0])[0] for loc in locations]

    trans_pose = None
    p_rotation, p_translation =  None, None
    for idx in tqdm(range(total_frames)):
        # an example to set object pose
        # trans_pose = get_transformation(0.2)
        processed_obj_id = []
        for obj_id in obj_id_list:
            # count object duplication, which is generally to be zero,
            # but can be increased if duplication operation happened
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            progress = idx / total_frames

            print("progress", progress)

            if edit_type == "duplication":
                trans_pose = get_transformation_with_duplication_offset(
                    progress, obj_duplication_cnt
                )
            elif edit_type == "pure_rotation":
                trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))
            elif edit_type == "pure_translation":
                trans_pose = get_pure_translation(progress_11=(progress * 2 - 1), axis='x')
            elif edit_type == 'parallel_parking':
                trans_pose, p_rotation, p_translation = parallel_parking_transform(trans_pose, p_rotation, p_translation, idx, total_frames, max_angle=80)
            else:
                trans_pose = get_scale_offset(progress, obj_duplication_cnt)
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
            camera_pose_Twc = poses_test[idx],
            # camera_pose_Twc=move_camera_pose(
            #     renderer.get_camera_pose_focal_by_frame_idx(pose_frame_idx)[0],
            #     idx / total_frames,
            # ),
            focal=renderer.get_camera_pose_focal_by_frame_idx(pose_frame_idx)[1]
            # object_pose_Twc = renderer.get_camera_pose_focal_by_frame_idx(idx)[0]
            # bckg_img = bckg_img
            # bckg_img = None
        )

        # results = renderer.render_edit(
        #     h=H,
        #     w=W,
        #     camera_pose_Twc=move_camera_pose(
        #         renderer.get_camera_pose_focal_by_frame_idx(pose_frame_idx)[0],
        #         idx / total_frames,
        #     ),
        #     focal=renderer.get_camera_pose_focal_by_frame_idx(pose_frame_idx)[1]
        # )
        image_out_path = f"{render_path}/render_{idx:04d}.png"
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        image_np = results[f'rgb_{typ}'].view(H, W, 3).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        # image_np = np.rot90(image_np, axes=(1,0))
        imageio.imwrite(image_out_path, image_np)
        
        depth_out_path = f"{render_path}/render_{idx:04d}_depth.png"
        depth_pred = results[f'depth_{typ}'].view(H, W).cpu().numpy()

        depth_vis = depth2inv(torch.tensor(depth_pred).unsqueeze(0).unsqueeze(0))
        depth_vis = viz_inv_depth(depth_vis)
        # depth_vis = np.rot90(depth_vis, axes=(1,0))

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