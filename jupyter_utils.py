import torch
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from models.rendering import *
from models.nerf import *
import metrics
from datasets import dataset_dict
from datasets.llff import *
from torch.utils.data import DataLoader
from functools import partial
from datasets.srn_multi_ae import collate_lambda_train, collate_lambda_val
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
# torch.set_printoptions(edgeitems=20)

import plotly
import plotly.graph_objects as go

# print(obj_samples)
def contract(x, order):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

def contract_pts(pts, radius):
    mask = torch.norm(pts, dim=-1).unsqueeze(-1) > radius
    new_pts = pts.clone()/radius
    norm_pts = torch.norm(new_pts, dim=-1).unsqueeze(-1)
    contracted_points = ((1+0.2) - 0.2/(norm_pts))*(new_pts/norm_pts)*radius
    warped_points = mask*contracted_points + (~mask)*pts
    return warped_points

def get_image_coords(pixel_offset, image_height, image_width,
):
    """This gets the image coordinates of one of the cameras in this object.
    If no index is specified, it will return the maximum possible sized height / width image coordinate map,
    by looking at the maximum height and width of all the cameras in this object.
    Args:
        pixel_offset: Offset for each pixel. Defaults to center of pixel (0.5)
        index: Tuple of indices into the batch dimensions of the camera. Defaults to None, which returns the 0th
            flattened camera
    Returns:
        Grid of image coordinates.
    """
    image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
    image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
    image_coords = torch.cat([image_coords, torch.ones((*image_coords.shape[:-1], 1))], dim=-1)
    image_coords = image_coords.view(-1, 3)
    return image_coords

def get_sphere(
    radius, center = None, color: str = "black", opacity: float = 1.0, resolution: int = 32
) -> go.Mesh3d:  # type: ignore
    """Returns a sphere object for plotting with plotly.
    Args:
        radius: radius of sphere.
        center: center of sphere. Defaults to origin.
        color: color of sphere. Defaults to "black".
        opacity: opacity of sphere. Defaults to 1.0.
        resolution: resolution of sphere. Defaults to 32.
    Returns:
        sphere object.
    """
    phi = torch.linspace(0, 2 * torch.pi, resolution)
    theta = torch.linspace(-torch.pi / 2, torch.pi / 2, resolution)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.sin(theta)
    pts = torch.stack((x, y, z), dim=-1)

    pts *= radius
    if center is not None:
        pts += center

    return go.Mesh3d(
        {
            "x": pts[:, :, 0].flatten(),
            "y": pts[:, :, 1].flatten(),
            "z": pts[:, :, 2].flatten(),
            "alphahull": 0,
            "opacity": opacity,
            "color": color,
        }
    )
    
def vis_camera_rays(origins, directions, coords) -> go.Figure:  # type: ignore
    """Visualize camera rays.
    Args:
        camera: Camera to visualize.
    Returns:
        Plotly lines
    """
    lines = torch.empty((origins.shape[0] * 2, 3))
    lines[0::2] = origins
    lines[1::2] = origins + directions*3.0
    
    print("lines", lines.shape)

    colors = torch.empty((coords.shape[0] * 2, 3))
    colors[0::2] = coords
    colors[1::2] = coords

    data = []
    data.append(go.Scatter3d(
    x=lines[:, 0],
    y=lines[:, 2],
    z=lines[:, 1],
    marker=dict(
        size=4,
        color=colors)))
        
    data.append(get_sphere(radius=1.0, color="#111111", opacity=0.05))
#     data.append(get_sphere(radius=2.0, color="#111111", opacity=0.05))
    fig = go.Figure(data = data
        
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False),
            yaxis=dict(title="z", showspikes=False),
            zaxis=dict(title="y", showspikes=False),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig

def vis_camera_samples(all_samples) -> go.Figure:  # type: ignore
    """Visualize camera rays.
    Args:
        camera: Camera to visualize.
    Returns:
        Plotly lines
    """
#     samples = samples.view(-1,3)

    data = []
    
    for i in range(all_samples.shape[0]):
        samples = all_samples[i]
        samples_init = samples[:10, :]
        samples_mid = samples[10:50, :]
        samples_final = samples[50:, :]

        data.append(go.Scatter3d(
        x=samples_init[:, 0],
        y=samples_init[:, 2],
        z=samples_init[:, 1],
        mode="markers",
        marker=dict(size=2, color="blue")
        ))

        data.append(go.Scatter3d(
        x=samples_mid[:, 0],
        y=samples_mid[:, 2],
        z=samples_mid[:, 1],
        mode="markers",
        marker=dict(size=2, color="black")
        ))

        data.append(go.Scatter3d(
        x=samples_final[:, 0],
        y=samples_final[:, 2],
        z=samples_final[:, 1],
        mode="markers",
        marker=dict(size=2, color="green")
        ))
        
    data.append(get_sphere(radius=1.0, color="#111111", opacity=0.05))
    data.append(get_sphere(radius=2.0, color="#111111", opacity=0.05))
    fig = go.Figure(data = data
        
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False),
            yaxis=dict(title="z", showspikes=False),
            zaxis=dict(title="y", showspikes=False),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig


def world2camera_viewdirs(w_viewdirs, cam2world, NS):
    w_viewdirs = repeat_interleave(w_viewdirs, NS)  # (SB*NS, B, 3)
    rot = torch.copy(cam2world[:, :3, :3]).transpose(1, 2)  # (B, 3, 3)
    viewdirs = torch.matmul(rot[:, None, :3, :3], w_viewdirs.unsqueeze(-1))[..., 0]
    return viewdirs


def world2camera(w_xyz, cam2world, NS=None):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    #print(w_xyz.shape, cam2world.shape)
    if NS is not None:
        w_xyz = repeat_interleave(w_xyz, NS)  # (SB*NS, B, 3)
    rot = cam2world[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    trans = -torch.bmm(rot, cam2world[:, :3, 3:])  # (B, 3, 1)
    #print(rot.shape, w_xyz.shape)
    cam_rot = torch.matmul(rot[:, None, :3, :3], w_xyz.unsqueeze(-1))[..., 0]
    cam_xyz = cam_rot + trans[:, None, :, 0]
    # cam_xyz = cam_xyz.reshape(-1, 3)  # (SB*B, 3)
    return cam_xyz

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def projection(c_xyz, focal, c):
    """Converts [x,y,z] in camera coordinates to image coordinates 
        for the given focal length focal and image center c.
    :param c_xyz: points in camera coordinates (SB*NV, NP, 3)
    :param focal: focal length (SB, 2)
    :c: image center (SB, 2)
    :output uv: pixel coordinates (SB, NV, NP, 2)
    """
    uv = -c_xyz[..., :2] / (c_xyz[..., 2:] + 1e-9)  # (SB*NV, NC, 2); NC: number of grid cells 
    uv *= repeat_interleave(
                focal.unsqueeze(1), NV if focal.shape[0] > 1 else 1
            )
    uv += repeat_interleave(
                c.unsqueeze(1), NV if c.shape[0] > 1 else 1
            )
    return uv


def pos_enc(x, min_deg=0, max_deg=10):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)

import open3d as o3d
def get_world_grid(side_lengths, grid_size):
    """ Returns a 3D grid of points in world coordinates.
    :param side_lengths: (min, max) for each axis (3, 2)
    :param grid_size: number of points along each dimension () or (3)
    :output grid: (1, grid_size**3, 3)
    """
    if len(grid_size) == 1:
        grid_size = [grid_size[0] for _ in range(3)]
        
    w_x = torch.linspace(side_lengths[0][0], side_lengths[0][1], grid_size[0])
    w_y = torch.linspace(side_lengths[1][0], side_lengths[1][1], grid_size[1])
    w_z = torch.linspace(side_lengths[2][0], side_lengths[2][1], grid_size[2])
    # Z, Y, X = torch.meshgrid(w_x, w_y, w_z)
    X, Y, Z = torch.meshgrid(w_x, w_y, w_z)
    w_xyz = torch.stack([X, Y, Z], axis=-1) # (gs, gs, gs, 3), gs = grid_size
    print(w_xyz.shape)
    w_xyz = w_xyz.reshape(-1, 3).unsqueeze(0) # (1, grid_size**3, 3)
    return w_xyz

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

def w2i_projection(w_xyz, cam2world, intrinsics):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = torch.cat([w_xyz, torch.ones_like(w_xyz[..., :1])], dim=-1)  # [n_points, 4]
    cam_xyz = torch.inverse(cam2world).bmm(w_xyz.permute(0,2,1))
    camera_grids = cam_xyz.permute(0,2,1)[:,:,:3]
    projections = intrinsics[None, ...].repeat(cam2world.shape[0], 1, 1).bmm(cam_xyz[:,:3,:])
    projections = projections.permute(0,2,1)
    uv = projections[..., :2] / projections[..., 2:3]  # [n_views, n_points, 2]
    return camera_grids, uv

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
    print("check pos", torch.max(p_norm_sq), torch.min(p_norm_sq))
    assert torch.all(check_pos >= 0), "1.0 - p_norm_sq should be greater than 0"
    d2 = torch.sqrt(1.0 - p_norm_sq) * rays_d_cos
    return d1 + d2