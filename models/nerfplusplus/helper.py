# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF++ (https://github.com/Kai-46/nerfplusplus)
# Copyright (c) 2020 the NeRF++ authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import numba as nb

def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]


def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    in_sphere,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)

    if in_sphere:
        if lindisp:
            t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        else:
            t_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    if in_sphere:
        coords = cast_rays(t_vals, rays_o, rays_d)
    else:
        t_vals = torch.flip(
            t_vals,
            dims=[
                -1,
            ],
        )  # 1.0 -> 0.0
        coords = depth2pts_outside(rays_o, rays_d, t_vals)

    return t_vals, coords


def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd, in_sphere, t_far=None, nocs=None, out_depth=None):

    eps = 1e-10

    if in_sphere:
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, t_far - t_vals[..., -1:]], dim=-1)
        dists *= torch.norm(dirs[..., None, :], dim=-1)
    else:
        dists = t_vals[..., :-1] - t_vals[..., 1:]
        dists = torch.cat([dists, torch.full_like(t_vals[..., :1], 1e10)], dim=-1)

    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    T = torch.cumprod(1.0 - alpha + eps, dim=-1)
    bg_lambda = T[..., -1:] if in_sphere else None
    accum_prod = torch.cat([torch.ones_like(T[..., -1:]), T[..., :-1]], dim=-1)
    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)

    if nocs is not None:
        comp_nocs = (weights[..., None] * nocs).sum(dim=-2)
        return comp_rgb, acc, weights, bg_lambda, comp_nocs
    else:
        if out_depth is not None:
            comp_depth = (weights * t_vals).sum(dim=-2)
            return comp_rgb, acc, weights, bg_lambda, comp_depth
        else:
            return comp_rgb, acc, weights, bg_lambda


def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def sample_pdf(
    bins, weights, origins, directions, t_vals, num_samples, randomized, in_sphere
):

    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values

    if in_sphere:
        coords = cast_rays(t_vals, origins, directions)
    else:
        t_vals = torch.flip(
            t_vals,
            dims=[
                -1,
            ],
        )  # 1.0 -> 0.0
        coords = depth2pts_outside(origins, directions, t_vals)

    return t_vals, coords


def intersect_sphere(rays_o, rays_d):
    """Compute the depth of the intersection point between this ray and unit sphere.
    Args:
        rays_o: [num_rays, 3]. Ray origins.
        rays_d: [num_rays, 3]. Ray directions.
    Returns:
        depth: [num_rays, 1]. Depth of the intersection point.
    """
    # note: d1 becomes negative if this mid point is behind camera

    d1 = -torch.sum(rays_d * rays_o, dim=-1, keepdim=True) / torch.sum(
        rays_d**2, dim=-1, keepdim=True
    )
    p = rays_o + d1 * rays_d
    # consider the case where the ray does not intersect the sphere
    rays_d_cos = 1.0 / torch.norm(rays_d, dim=-1, keepdim=True)
    p_norm_sq = torch.sum(p * p, dim=-1, keepdim=True)
    check_pos = 1.0 - p_norm_sq
    assert torch.all(check_pos >= 0), "1.0 - p_norm_sq should be greater than 0"
    d2 = torch.sqrt(1.0 - p_norm_sq) * rays_d_cos
    return d1 + d2

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

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def depth2pts_outside(rays_o, rays_d, depth):
    """Compute the points along the ray that are outside of the unit sphere.
    Args:
        rays_o: [num_rays, 3]. Ray origins of the points.
        rays_d: [num_rays, 3]. Ray directions of the points.
        depth: [num_rays, num_samples along ray]. Inverse of distance to sphere origin.
    Returns:
        pts: [num_rays, 4]. Points outside of the unit sphere. (x', y', z', 1/r)
    """
    # note: d1 becomes negative if this mid point is behind camera
    rays_o = rays_o[..., None, :].expand(
        list(depth.shape) + [3]
    )  #  [N_rays, num_samples, 3]
    rays_d = rays_d[..., None, :].expand(
        list(depth.shape) + [3]
    )  #  [N_rays, num_samples, 3]
    d1 = -torch.sum(rays_d * rays_o, dim=-1, keepdim=True) / torch.sum(
        rays_d**2, dim=-1, keepdim=True
    )

    p_mid = rays_o + d1 * rays_d
    p_mid_norm = torch.norm(p_mid, dim=-1, keepdim=True)
    rays_d_cos = 1.0 / torch.norm(rays_d, dim=-1, keepdim=True)

    check_pos = 1.0 - p_mid_norm * p_mid_norm
    assert torch.all(check_pos >= 0), "1.0 - p_mid_norm * p_mid_norm should be greater than 0"

    d2 = torch.sqrt(1.0 - p_mid_norm * p_mid_norm) * rays_d_cos
    p_sphere = rays_o + (d1 + d2) * rays_d

    rot_axis = torch.cross(rays_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth[..., None])  # depth is inside [0, 1]
    rot_angle = phi - theta  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = (
        p_sphere * torch.cos(rot_angle)
        + torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle)
        + rot_axis
        * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True)
        * (1.0 - torch.cos(rot_angle))
    )
    p_sphere_new = p_sphere_new / (
        torch.norm(p_sphere_new, dim=-1, keepdim=True) + 1e-10
    )
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts