import os
import numpy as np
import torch
import copy
from utils.geo_utils import bbox_intersection_batch
from objectron.schema import annotation_data_pb2 as annotation_protocol
torch.set_printoptions(threshold=1000)

def get_frame_annotation(annotation_filename):
    """Grab an annotated frame from the sequence."""
    result = []
    instances = []
    with open(annotation_filename, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())

        object_id = 0
        object_rotations = []
        object_translations = []
        object_scale = []
        num_keypoints_per_object = []
        object_categories = []
        annotation_types = []
        
        # Object instances in the world coordinate system, These are stored per sequence, 
        # To get the per-frame version, grab the transformed keypoints from each frame_annotation
        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            scale = np.array(obj.scale)
            points3d = np.array([[kp.x, kp.y, kp.z] for kp in obj.keypoints])
            instances.append((rotation, translation, scale, points3d))
        
        # Grab teh annotation results per frame
        for data in sequence.frame_annotations:
            # Get the camera for the current frame. We will use the camera to bring
            # the object from the world coordinate to the current camera coordinate.
            transform = np.array(data.camera.transform).reshape(4, 4)
            view = np.array(data.camera.view_matrix).reshape(4, 4)
            intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)
            projection = np.array(data.camera.projection_matrix).reshape(4, 4)
        
            keypoint_size_list = []
            object_keypoints_2d = []
            object_keypoints_3d = []
            for annotations in data.annotations:
                num_keypoints = len(annotations.keypoints)
                keypoint_size_list.append(num_keypoints)
                for keypoint_id in range(num_keypoints):
                    keypoint = annotations.keypoints[keypoint_id]
                    object_keypoints_2d.append((keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                    object_keypoints_3d.append((keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
                num_keypoints_per_object.append(num_keypoints)
                object_id += 1
            result.append((object_keypoints_2d, object_keypoints_3d, keypoint_size_list, view, projection))

    return result, instances

class BBoxRayHelper:
    def __init__(self, hparams, instance_id):
        super().__init__()
        self.hparams = hparams
        self.read_objectron_info(instance_id)


    # def read_objectron_info(self, instance_id):
    #     # base_dir = '/home/ubuntu/nerf_pl/data/objectron'
    #     base_dir = '/home/ubuntu/nerf_pl/data/objectron'
    #     self.ids = np.sort([f.name for f in os.scandir(base_dir)]) 
    #     # instance_name = self.ids[instance_id-1]
    #     instance_id = 3
    #     instance_name = self.ids[instance_id-1]
    #     print("instance_name", instance_name)
    #     print("self.ids", self.ids)
    #     base_dir = os.path.join(base_dir, instance_name)
    #     annotation_data, instances = get_frame_annotation(os.path.join(base_dir, instance_name+'.pbdata'))
    #     instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[0]
    #     instance_rotation = np.reshape(instance_rotation, (3, 3))
    #     box_transformation = np.eye(4)
    #     box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    #     box_transformation[:3, -1] = instance_translation
    #     scale = instance_scale.T
    #     self.axis_align_mat = box_transformation
    #     self.bbox_bounds = np.array([-scale / 2, scale / 2])

    def read_objectron_info(self, instance_id):
        # base_dir = '/home/ubuntu/nerf_pl/data/objectron'
        base_dir = '/home/ubuntu/nerf_pl/data/objectron/camera'
        self.ids = np.sort([f.name for f in os.scandir(base_dir)]) 
        # instance_name = self.ids[instance_id-1]
        # instance_id = 3
        instance_name = 'camera_batch-1_0'
        print("instance_name", instance_name)
        print("self.ids", self.ids)
        base_dir = os.path.join(base_dir, instance_name)
        annotation_data, instances = get_frame_annotation(os.path.join(base_dir, instance_name+'.pbdata'))
        instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[0]
        instance_rotation = np.reshape(instance_rotation, (3, 3))
        box_transformation = np.eye(4)
        box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
        box_transformation[:3, -1] = instance_translation
        scale = instance_scale.T
        self.axis_align_mat = box_transformation
        self.bbox_bounds = np.array([-scale / 2, scale / 2])

    def get_axis_align_mat(self, rescaled=False):
        if rescaled:
            axis_align_mat = copy.deepcopy(self.axis_align_mat)
            axis_align_mat[:3, 3] /= self.scale_factor
            return axis_align_mat
        else:
            return self.axis_align_mat

    def get_world_to_object_transform(self):
        trans = np.linalg.inv(self.axis_align_mat.copy()) 
        return trans  # Tow

    def transform_rays_to_bbox_coordinates_objectron(self, rays_o, rays_d):
        if type(rays_o) is torch.Tensor:
            rays_o, rays_d = (
                rays_o.detach().cpu().numpy(),
                rays_d.detach().cpu().numpy(),
            )
        rays_o_bbox = rays_o
        rays_d_bbox = rays_d
        T_box_orig = np.linalg.inv(self.axis_align_mat.copy())
        rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
        rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
        return rays_o_bbox, rays_d_bbox

    def transform_xyz_to_bbox_coordinates(self, xyz):
        if type(xyz) is torch.Tensor:
            xyz = xyz.detach().cpu().numpy()
        xyz_bbox = xyz
        # convert to bbox coordinates
        T_box_orig = np.linalg.inv(self.axis_align_mat.copy())
        xyz_bbox = (T_box_orig[:3, :3] @ xyz_bbox.T).T + T_box_orig[:3, 3]
        return xyz_bbox

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

        bbox_enlarge = 0.06
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        if bbox_enlarge > 0:
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
        bbox_enlarge = 0.06
        if bbox_enlarge > 0:
            bbox_z_min_orig = bbox_bounds[0][2]
            bbox_bounds[0] -= bbox_enlarge
            bbox_bounds[1] += bbox_enlarge
            bbox_bounds[0][2] = bbox_z_min_orig
            bbox_bounds[0][2] -= bbox_enlarge
            
            # z_min_orig = bbox_bounds[0][2]  # keep z_min
            # bbox_bounds[0] -= bbox_enlarge
            # bbox_bounds[1] += bbox_enlarge
            # bbox_bounds[0][2] = z_min_orig
        elif bbox_enlarge < 0:
            # make some margin near the ground
            bbox_bounds[0][2] -= bbox_enlarge
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