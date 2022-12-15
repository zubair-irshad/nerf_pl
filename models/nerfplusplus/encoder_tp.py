import torch
from torch import nn
import torch.nn.functional as F
from models.nerfplusplus.util import *
import numpy as np
import torch.autograd.profiler as profiler
from models.nerfplusplus.encoder import SpatialEncoder


def init_weights_kaiming(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight)
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1e-3, 1e-3)

class DepthPillarEncoder(nn.Module):
    def __init__(self, inp_ch, LS):
        super().__init__()
        self.common_branch = nn.Sequential(nn.Linear(inp_ch, LS),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(LS, LS),
                                      nn.ReLU(inplace=True),)
        self.depth_encoder = nn.Linear(LS, LS)
        self.common_branch.apply(init_weights_kaiming)
        self.depth_encoder.apply(init_weights_kaiming)

    def forward(self, x):
        out = self.common_branch(x)
        out = self.depth_encoder(out)
        return out

class GridEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            backbone="resnet34",
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="zeros",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
            grid_size=[256, 256, 256],
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflzerosection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()
        #         print("Grid size: ", grid_size, type(grid_size))
        self.grid_size = grid_size
        self.side_length = 5.
        self.radius = 4.5
        self.sfactor = 4
        self.world_grid = get_world_grid([[-self.side_length, self.side_length],
                                               [-self.side_length, self.side_length],
                                               [-self.side_length, self.side_length],
                                               ], [int(self.grid_size[0]/self.sfactor), int(self.grid_size[1]/self.sfactor), int(self.grid_size[2]/self.sfactor)] ).cuda()  # (1, grid_size**3, 3)

        self.spatial_encoder = SpatialEncoder(backbone="resnet34",
                                              pretrained=True,
                                              num_layers=4,
                                              index_interp="bilinear",
                                              index_padding="zeros",
                                              # index_padding="border",
                                              upsample_interp="bilinear",
                                              feature_scale=1.0,
                                              use_first_pool=True,
                                              norm_type="batch")
        self.latent_size = self.spatial_encoder.latent_size  # self.spatial_encoder.latent_size/8
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        LS = self.latent_size
        # To encode latent from spatial encoder with camera depth
        self.depth_fc = DepthPillarEncoder(inp_ch=self.spatial_encoder.latent_size + 3 + 3, LS=LS)

        self.pillar_aggregator = nn.Sequential(nn.Linear(LS + 1, LS),
                                                nn.ReLU(inplace=True),
                                               nn.Linear(LS, 1))


        # Creates the static grid using the per-scene floorplans.
        self.floorplan_convnet = nn.Sequential(nn.ReflectionPad2d(1),
                                                            nn.Conv2d(LS, LS, 3, 1, 0),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(LS, LS, 3, 1, 1, padding_mode="reflect"),
                                                            nn.ReLU(inplace=True),
                                                            nn.UpsamplingBilinear2d(scale_factor=2),
                                                            nn.Conv2d(LS, LS*2, 3, 1, 1, padding_mode="reflect"),
                                                            nn.ReLU(inplace=True),
                                                            nn.UpsamplingBilinear2d(scale_factor=2),
                                                            nn.Conv2d(LS*2, LS*1, 3, 1, 1, padding_mode="reflect"),
                                                            )

        self.floorplan_convnet.apply(init_weights_kaiming)
        self.pillar_aggregator.apply(init_weights_kaiming)
    
    def contract_pts(self, pts, radius):
        mask = torch.norm(pts, dim=-1).unsqueeze(-1) > radius
        new_pts = pts.clone()/radius
        norm_pts = torch.norm(new_pts, dim=-1).unsqueeze(-1)
        contracted_points = (1.1 - 0.1/(norm_pts))*(new_pts/norm_pts)*radius
        warped_points = mask*contracted_points + (~mask)*pts
        return warped_points

    def index(self, samples):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """

        uv_xz = samples[:, :, [0, 2]].reshape(-1,2).unsqueeze(0)
        uv_yz = samples[:, :, [1, 2]].reshape(-1,2).unsqueeze(0)
        uv_xy = samples[:, :, [0, 1]].reshape(-1,2).unsqueeze(0)

        # To bring the coordinates to -1, 1
        #             print("t:", t[:, 0, 0])
        scale = 1 / self.side_length
        #             print("UV shape:", uv.shape)

        uv_xz = self.contract_pts(uv_xz, 1.0)
        uv_xz = uv_xz * scale
        uv_xz = uv_xz.unsqueeze(2)  # (B, N, 1, 2)
        
        uv_yz = self.contract_pts(uv_yz, 1.0)
        uv_yz = uv_yz * scale
        uv_yz = uv_yz.unsqueeze(2)  # (B, N, 1, 2)
        
        uv_xy = self.contract_pts(uv_xy, 1.0)
        uv_xy = uv_xy * scale
        uv_xy = uv_xy.unsqueeze(2)  # (B, N, 1, 2)
        # Plot a scatter plot of the latents.
        # Compute the object latent from the warped scene grid.
        
        scene_latent_xz = F.grid_sample(self.scene_grid_xz,
                                    uv_xz,
                                    align_corners=True,
                                    mode="bilinear",  # "nearest",
                                    padding_mode="zeros", )

        scene_latent_xy = F.grid_sample(self.scene_grid_xy,
                                    uv_xy,
                                    align_corners=True,
                                    mode="bilinear",  # "nearest",
                                    padding_mode="zeros", )

        scene_latent_yz = F.grid_sample(self.scene_grid_yz,
                                    uv_yz,
                                    align_corners=True,
                                    mode="bilinear",  # "nearest",
                                    padding_mode="zeros", )

        output = (scene_latent_xz + scene_latent_yz + scene_latent_xy) / 3

        return output[..., 0]

    def forward(self, images, poses, focal, c):
        """
        For extracting ResNet's features.
        :param images (SB, NV, C, H, W)
        :param poses (SB*NV, 4, 4)
        :param focal focal length (SB) or (SB, 2)
        :param c principal point (SB) or (SB, 2)
        :return latent (SB, latent_size, H, W)
        """

        focal = focal[0].unsqueeze(-1).repeat((1, 2))
        focal[..., 1] *= -1.0
        c = c[0].unsqueeze(0)

        NV, C, H, W = images.shape
        self.spatial_encoder(images)

        # Creating a grid of x, y, z coordinates
        self.world_grids = repeat_interleave(self.world_grid.clone(),
                                                NV).cuda()  # (SB*NV, NC, 3) NC: number of grid cells

        # Transforming world points to camera view
        # self.camera_grids = world2camera(self.world_grids, poses)
        camera_grids = world2camera(self.world_grids, poses)

        # Compute mask for points outside the frustrum
        # mask = self.camera_grids[..., :, -1] < 1e-3

        # Fix as proposed in https://github.com/apchenstu/mvsnerf/issues/12#issuecomment-1171424369
        mask = camera_grids[..., :, -1].abs() < 1e-3
        print("self.camera_grids, mask", camera_grids.shape, mask.shape)
        camera_grids[mask, -1] = 1e-3

        camera_pts_dir = self.world_grids - poses[:, None, :3, -1]
        camera_pts_dir_norm = torch.norm(camera_pts_dir + 1e-9, dim=-1)
        camera_pts_dir = camera_pts_dir/camera_pts_dir_norm[:, :, None]
        camera_pts_dir = camera_pts_dir * mask[:, :, None]

        # Projecting points in camera coordinates on the image plane
        uv = projection(camera_grids, focal, c)  # [f, -f]

        latent = self.spatial_encoder.index(
            uv, None, torch.Tensor([W, H]).cuda()
        )
        _, L, _ = latent.shape  # (NV, L, grid_size**3)
        latent = latent * mask[:, None, :]
        latent = torch.cat([latent,
                            camera_grids.permute(0, -1, 1),
                            camera_pts_dir.permute(0, -1, 1)], 1)
        latent = latent.reshape(NV, L+3+3,
                                self.grid_size[0]//self.sfactor * self.grid_size[1]//self.sfactor * self.grid_size[2]//self.sfactor).permute(0, 2, 1)  # Input to the linear layer # (SB*T*NV, grid_size**3, L+1)
        latent = self.depth_fc(latent)
        latent = latent.reshape(1, NV,
                                  self.grid_size[0]//self.sfactor,
                                  self.grid_size[1]//self.sfactor,
                                  self.grid_size[2]//self.sfactor, L).permute(0, 1, 2, 4, 3, 5)

        latent = latent.mean(1)  # (SB * T, NC, L) average along the number of views
        
        latent_inp_x = torch.cat([latent, self.world_grids.reshape(1, NV, self.grid_size[0]//self.sfactor,
                                                          self.grid_size[1]//self.sfactor,
                                                          self.grid_size[2]//self.sfactor, 3)[:, 0, ..., 0:1]], -1)
        latent_inp_y = torch.cat([latent, self.world_grids.reshape(1, NV, self.grid_size[0]//self.sfactor,
                                                          self.grid_size[1]//self.sfactor,
                                                          self.grid_size[2]//self.sfactor, 3)[:, 0, ..., 1:2]], -1)
        latent_inp_z = torch.cat([latent, self.world_grids.reshape(1, NV, self.grid_size[0]//self.sfactor,
                                                          self.grid_size[1]//self.sfactor,
                                                          self.grid_size[2]//self.sfactor, 3)[:, 0, ..., 2:3]], -1)

        weights_x = torch.softmax(self.pillar_aggregator(latent_inp_x), dim=-2)  # (SB, T, X, Z, Y, 1)
        weights_y = torch.softmax(self.pillar_aggregator(latent_inp_y), dim=-2)  # (SB, T, X, Z, Y, 1)
        weights_z = torch.softmax(self.pillar_aggregator(latent_inp_z), dim=-2)  # (SB, T, X, Z, Y, 1)

        floorplans_yz = (latent * weights_x).sum(-2)  # (SB, T, X, Z, L)
        floorplans_xz = (latent * weights_y).sum(-2)  # (SB, T, X, Z, L)
        floorplans_xy = (latent * weights_z).sum(-2)  # (SB, T, X, Z, L)

        grid_yz = floorplans_yz.permute(0, -1, 1, 2) # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        self.scene_grid_yz = self.floorplan_convnet(grid_yz)  # (SB*T, L, X, Z)

        grid_xz = floorplans_xz.permute(0, -1, 1, 2) # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        self.scene_grid_xz = self.floorplan_convnet(grid_xz)  # (SB*T, L, X, Z)

        grid_xy = floorplans_xy.permute(0, -1, 1, 2) # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        self.scene_grid_xy = self.floorplan_convnet(grid_xy)  # (SB*T, L, X, Z)
        return 0