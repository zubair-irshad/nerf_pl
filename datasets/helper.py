
import numpy as np
import torch
import torch.nn.functional as F
import numba as nb

@nb.jit(nopython=True)
def bbox_intersection_batch(bounds, rays_o, rays_d):
    N_rays = rays_o.shape[0]
    all_hit = np.empty((N_rays))
    all_near = np.empty((N_rays))
    all_far = np.empty((N_rays))
    for idx, (o, d) in enumerate(zip(rays_o, rays_d)):
        hit, near, far = bbox_intersection(bounds, o, d)
        all_hit[idx] = hit
        all_near[idx] = near
        all_far[idx] = far
    # return (h*w), (h*w, 3), (h*w, 3)
    return all_hit, all_near, all_far

@nb.jit(nopython=True)
def bbox_intersection(bounds, orig, dir):
    # FIXME: currently, it is not working properly if the ray origin is inside the bounding box
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    # handle divide by zero
    dir[dir == 0] = 1.0e-14
    invdir = 1 / dir
    sign = (invdir < 0).astype(np.int64)

    tmin = (bounds[sign[0]][0] - orig[0]) * invdir[0]
    tmax = (bounds[1 - sign[0]][0] - orig[0]) * invdir[0]

    tymin = (bounds[sign[1]][1] - orig[1]) * invdir[1]
    tymax = (bounds[1 - sign[1]][1] - orig[1]) * invdir[1]

    if tmin > tymax or tymin > tmax:
        return False, 0, 0
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (bounds[sign[2]][2] - orig[2]) * invdir[2]
    tzmax = (bounds[1 - sign[2]][2] - orig[2]) * invdir[2]

    if tmin > tzmax or tzmin > tmax:
        return False, 0, 0
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax
    # additionally, when the orig is inside the box, we return False
    if tmin < 0 or tmax < 0:
        return False, 0, 0
    return True, tmin, tmax

def transform_rays_to_bbox_coordinates(rays_o, rays_d, axis_align_mat):
    rays_o_bbox = rays_o
    rays_d_bbox = rays_d
    T_box_orig = axis_align_mat
    rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
    rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
    return rays_o_bbox, rays_d_bbox

def get_rays_in_bbox(rays_o, rays_d, bbox_bounds, axis_aligned_mat):

    rays_o_bbox, rays_d_bbox = transform_rays_to_bbox_coordinates(
        rays_o, rays_d, axis_aligned_mat
    )
    bbox_mask, batch_near, batch_far = bbox_intersection_batch(
        bbox_bounds, rays_o_bbox, rays_d_bbox
    )
    bbox_mask, batch_near, batch_far = (
        torch.Tensor(bbox_mask).bool(),
        torch.Tensor(batch_near[..., None]),
        torch.Tensor(batch_far[..., None]),
    )
    return bbox_mask, batch_near, batch_far

def get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=False):
    instance_rotation = RTs['R']
    instance_translation = RTs['T']
    bbox_bounds = RTs['s']
    box_transformation = np.eye(4)
    box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    box_transformation[:3, -1] = instance_translation
    axis_aligned_mat = np.linalg.inv(box_transformation)
    bbox_mask, batch_near_obj, batch_far_obj = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, axis_aligned_mat)
    return bbox_mask, batch_near_obj, batch_far_obj

def sample_rays_in_bbox(RTs, rays_o, view_dirs):
    all_R = RTs['R']
    all_T = RTs['T']
    all_s = RTs['s']
    all_near = torch.zeros((rays_o.shape[0], 1))
    all_far = torch.zeros((rays_o.shape[0], 1))
    for Rot,Tran,sca in zip(all_R, all_T, all_s):
        RTS_single = {'R': np.array(Rot), 'T': np.array(Tran), 's': np.array(sca)}
        _, near, far = get_object_rays_in_bbox(rays_o, view_dirs, RTS_single, canonical=False)
        new_near = torch.where((all_near==0) | (near==0), torch.maximum(near, all_near), torch.minimum(near, all_near))
        all_near = new_near
        new_far = torch.where((all_far==0) | (far==0), torch.maximum(far, all_far), torch.minimum(far, all_far))
        all_far = new_far
    bbox_mask = (all_near !=0) & (all_far!=0)
    return all_near, all_far, bbox_mask