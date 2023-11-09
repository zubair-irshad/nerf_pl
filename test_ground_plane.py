side_length = 5.
radius = 4.5
grid_size=[256, 16, 256]
sfactor=4
import torch

side_length = 5.
radius = 4.5
# grid_size=[256, 16, 256]
grid_size=[256, 256, 256]
sfactor=4
import torch

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
    Z, Y, X = torch.meshgrid(w_x, w_y, w_z)
    w_xyz = torch.stack([X, Y, Z], axis=-1) # (gs, gs, gs, 3), gs = grid_size
#     w_xyz = torch.stack(torch.meshgrid(w_x, w_y, w_z), axis=-1) # (gs, gs, gs, 3), gs = grid_size
    print(w_xyz.shape)
    w_xyz = w_xyz.reshape(-1, 3).unsqueeze(0) # (1, grid_size**3, 3)
    return w_xyz

# world_grid = get_world_grid([[-side_length, side_length],
#                                        [0, 5],
#                                        [-side_length, side_length],
#                                        ], [int(grid_size[0]/sfactor), grid_size[1], int(grid_size[2]/sfactor)] )  # (1, grid_size**3, 3)

world_grid = get_world_grid([[-side_length, side_length],
                                       [-side_length, 5],
                                       [-side_length, side_length],
                                       ], [int(grid_size[0]/sfactor), int(grid_size[1]/sfactor), int(grid_size[2]/sfactor)] )  # (1, grid_size**3, 3)

print(world_grid.shape)
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(world_grid.squeeze(0).numpy())

o3d.visualization.draw_geometries([pcd])