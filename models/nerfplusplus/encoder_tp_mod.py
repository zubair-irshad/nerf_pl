import torch
from torch import nn
import torch.nn.functional as F
from models.nerfplusplus.util import *
import numpy as np
from models.nerfplusplus.spatial_encoder import SpatialEncoder, ResUNet
from models.nerfplusplus.conv3d import EncoderDecoder
from torch import linalg as LA
from torchvision import transforms as T

# # def contract_samples(x, order=1):
# #     mag = LA.norm(x, order, dim=-1)[..., None]
# #     return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

def contract_samples(x, order=float('inf')):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag)), mag

def inverse_contract_samples(x, mag_origial,order=float('inf')):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (x*mag_origial)/(2-(1/mag_origial)))

def unprocess_images(normalized_images, encoder_type = 'resnet'):
    if encoder_type =='resnet':
        inverse_transform = T.Compose([T.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))])
    else:
        inverse_transform = T.Compose([T.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))])
    return inverse_transform(normalized_images)

# #mipnerf contract
# # def _contract(x):
# #     x_mag_sq = torch.sum(x**2, dim=-1, keepdim=True).clip(min=1e-32)
# #     z = torch.where(
# #         x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x
# #     )
# #     return z

def unprocess_images(normalized_images, encoder_type = 'resnet'):
    if encoder_type =='resnet':
        inverse_transform = T.Compose([T.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))])
    else:
        inverse_transform = T.Compose([T.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))])
    return inverse_transform(normalized_images)

def get_c(samples, imgs, poses, focal, c, stride = 1, encoder_type = 'resnet'):

    # if contract:
    #     #mipnerf contract with a larger grid size i.e. -2 to 2
    #     samples = _contract(samples)
    #     # samples = contract_samples(samples)

    focal = focal[0].unsqueeze(-1).repeat((1, 2))
    focal[..., 1] *= -1.0
    c = c[0].unsqueeze(0)

    # imgs_unprocess = unprocess_images(imgs, encoder_type = encoder_type)
    imgs_unprocess = unprocess_images(imgs, encoder_type = 'resnet')

    NV, C, height, width = imgs.shape

    samples = samples[:,:,:3].reshape(-1,3).unsqueeze(0).float()
    cam_xyz = world2camera(samples, poses)
    focal = focal/stride
    c = c/stride
    uv = projection(cam_xyz, focal, c)

    im_x = uv[:,:, 0]
    im_y = uv[:,:, 1]
    im_grid = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)
    im_grid = im_grid.unsqueeze(2)

    colors = torch.empty((im_grid.shape[1], NV*C), device=im_grid.device, dtype=torch.float)
    for i, idx in enumerate(range(imgs_unprocess.shape[0])):
        imgs_feat = F.grid_sample(imgs_unprocess[idx, :, :, :].unsqueeze(0), im_grid[idx, :, :].unsqueeze(0), align_corners=True, mode='bilinear', padding_mode='zeros')
        colors[...,i*C:i*C+C] = imgs_feat[0].squeeze(-1).permute(1,0)
    return colors


def index_grid(samples, scene_grid_xz, scene_grid_xy, scene_grid_yz):
    """
    Get pixel-aligned image features at 2D image coordinates
    :param uv (B, N, 2) image points (x,y)
    :param cam_z ignored (for compatibility)
    :param image_size image size, either (width, height) or single int.
    if not specified, assumes coords are in [-1, 1]
    :param z_bounds ignored (for compatibility)
    :return (B, L, N) L is latent size
    """
    # if contract:
    #     #mipnerf contract with a larger grid size i.e. -2 to 2
    #     # samples = _contract(samples)
    #     samples, _ = contract_samples(samples)

    # scale_factor = 2
    # samples = samples/scale_factor
    
    # index_x = samples[:,:,0].float().unsqueeze(-1).reshape(-1,1)
    # index_y = samples[:,:,1].float().unsqueeze(-1).reshape(-1,1)
    # index_z = samples[:,:,2].float().unsqueeze(-1).reshape(-1,1)

    # index_x = index_x*-1
    # index_y = index_y*-1

    # uv_xz = torch.cat([index_x, index_z], dim=-1).unsqueeze(0).unsqueeze(2) # (B, N, 1, 2)
    # uv_yz = torch.cat([index_y, index_z], dim=-1).unsqueeze(0).unsqueeze(2) # (B, N, 1, 2)
    # uv_xy = torch.cat([index_x, index_y], dim=-1).unsqueeze(0).unsqueeze(2) # (B, N, 1, 2)

    scale_factor = 2.4673

    uv_xz = samples[:, :, [0, 2]].reshape(-1,2).unsqueeze(0).float()
    uv_yz = samples[:, :, [1, 2]].reshape(-1,2).unsqueeze(0).float()
    uv_xy = samples[:, :, [0, 1]].reshape(-1,2).unsqueeze(0).float()

    uv_xz = (uv_xz/scale_factor).unsqueeze(2)  # (B, N, 1, 2)
    uv_yz = (uv_yz/scale_factor).unsqueeze(2)  # (B, N, 1, 2)
    uv_xy = (uv_xy/scale_factor).unsqueeze(2)  # (B, N, 1, 2)

    scene_latent_xz = F.grid_sample(scene_grid_xz,
                                uv_xz,
                                align_corners=True,
                                mode="bilinear",  # "nearest",
                                padding_mode="zeros", )

    scene_latent_xy = F.grid_sample(scene_grid_xy,
                                uv_xy,
                                align_corners=True,
                                mode="bilinear",  # "nearest",
                                padding_mode="zeros", )

    scene_latent_yz = F.grid_sample(scene_grid_yz,
                                uv_yz,
                                align_corners=True,
                                mode="bilinear",  # "nearest",
                                padding_mode="zeros", )

    output = torch.sum(torch.stack([scene_latent_xz, scene_latent_xy, scene_latent_yz]), dim=0)
    return output[..., 0]
    

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
            encoder_type = "resnet",
            contract = False,
            backbone="resnet34",
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="zeros",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
            # grid_size=[64, 64, 64],
            grid_size=[96, 96, 96],
            xyz_min = None,
            xyz_max = None
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
        # self.xyz_min = xyz_min
        # self.xyz_max = xyz_max
        self.contract = contract
        self.grid_size = grid_size
        # self.side_length = 0.5
        self.encoder_type = encoder_type
        self.conv3d = EncoderDecoder()

        # self.radius = 4.5
        # self.radius = 1.0
        # self.sfactor = 4
        # if self.contract:
        #     #current experiment earlier. This contraction seems to work great atleast for!!!
        #     # side_lengths = [3, 3, 12]
        #     side_lengths = [3, 3, 7]

        #     world_grid = get_world_grid([[-side_lengths[0], side_lengths[0]],
        #                                         [-side_lengths[1]/2, side_lengths[1]],
        #                                         [-side_lengths[2], side_lengths[2]],
        #                                         ], [int(grid_size[0]), int(grid_size[1]), int(grid_size[2])] )  # (1, grid_size**3, 3)

        # else:
        # side_lengths = [3, 3, 6]
        # world_grid = get_world_grid([[-side_lengths[0], side_lengths[0]],
        #                                     [-side_lengths[1], side_lengths[1]],
        #                                     [-side_lengths[2], side_lengths[2]],
        #                                     ], [int(grid_size[0]), int(grid_size[1]), int(grid_size[2])] )  # (1, grid_size**3, 3)


        # world_grid, self.mag_original = contract_samples(world_grid.squeeze(0))
        # norm_pose_0 = [[ 7.49440487e-01, -7.29083171e-02, 6.58045085e-01],
        #             [ 6.62071716e-01, 8.25294955e-02, -7.44882491e-01],
        #             [ 1.35916947e-17, 9.93918135e-01, 1.10121480e-01]]
        # norm_pose_0 = torch.FloatTensor(norm_pose_0).to(world_grid.device)
        # self.world_grid = (norm_pose_0 @ world_grid.T).T

        #https://github.com/zubair-irshad/generalizable-scene-representations/blob/f43f3ac73ffa7c431158e31d9f0778a1ed29b834/models/nerfplusplus/encoder_tp_mod.py
        # side_lengths = [3, 3, 6]
        # world_grid = get_world_grid([[-side_lengths[0], side_lengths[0]],
        #                                     [-side_lengths[1]/2, side_lengths[1]],
        #                                     [-side_lengths[2], 0],
        #                                     ], [int(grid_size[0]), int(grid_size[1]), int(grid_size[2])] )  # (1, grid_size**3, 3)

        # world_grid, self.mag_original = contract_samples(world_grid.squeeze(0))
        # norm_pose_0 = [[ 7.14784828e-01, -3.56641552e-01, 6.01572484e-01, 6.02849754e-01],
        #     [ 6.99344443e-01,  3.64515616e-01, -6.14854223e-01, -6.16159694e-01],
        #     [-6.48906216e-18,  8.60194844e-01,  5.09965519e-01, 5.11048288e-01],
        #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]

        # norm_pose_0 = torch.FloatTensor(norm_pose_0).to(world_grid.device)
        # wg_homo = torch.cat((world_grid, torch.ones(world_grid.shape[0],1, device=world_grid.device)), dim=-1)
        # world_grid = norm_pose_0 @ wg_homo.T
        # self.world_grid = world_grid[:3, :].T

        # self.world_grid = world_grid.unsqueeze(0)


        # norm_pose_0 = [[ 7.14784828e-01, -3.56641552e-01, 6.01572484e-01, 6.02849754e-01],
        #     [ 6.99344443e-01,  3.64515616e-01, -6.14854223e-01, -6.16159694e-01],
        #     [-6.48906216e-18,  8.60194844e-01,  5.09965519e-01, 5.11048288e-01],
        #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]

        # norm_pose_0 = torch.FloatTensor(norm_pose_0).to(world_grid.device)
        # wg_homo = torch.cat((world_grid.squeeze(0), torch.ones(world_grid.shape[1],1, device=world_grid.device)), dim=-1)
        # world_grid = norm_pose_0 @ wg_homo.T
        # world_grid = world_grid[:3, :].T
        # self.world_grid = get_world_grid([[0, self.side_length],
        #                                        [-self.side_length, self.side_length],
        #                                        [-self.side_length, self.side_length],
        #                                        ], [int(self.grid_size[0]/self.sfactor), int(self.grid_size[1]/self.sfactor), int(self.grid_size[2]/self.sfactor)] ).cuda()  # (1, grid_size**3, 3)

        side_lengths = [1, 1, 1]
        self.world_grid = get_world_grid([[-side_lengths[0], side_lengths[0]],
                                            [-side_lengths[1], side_lengths[1]],
                                            [0, side_lengths[2]],
                                            ], [int(grid_size[0]), int(grid_size[1]), int(grid_size[2])] )  # (1, grid_size**3, 3)


        if encoder_type == 'resnet':
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
        else:
            self.spatial_encoder = ResUNet()

        # self.latent_size = self.spatial_encoder.latent_size  # self.spatial_encoder.latent_size/8
        self.latent_size = 3
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        LS = self.latent_size
        # To encode latent from spatial encoder with camera depth
        self.depth_fc = DepthPillarEncoder(inp_ch=3 + 3 + 3, LS=LS)

        self.pillar_aggregator_xz = nn.Sequential(nn.Linear(LS + 1, LS),
                                                nn.ReLU(inplace=True),
                                               nn.Linear(LS, 1))

        self.pillar_aggregator_yz = nn.Sequential(nn.Linear(LS + 1, LS),
                                                nn.ReLU(inplace=True),
                                               nn.Linear(LS, 1))
        self.pillar_aggregator_xy = nn.Sequential(nn.Linear(LS + 1, LS),
                                                nn.ReLU(inplace=True),
                                               nn.Linear(LS, 1))


        # Creates the static grid using the per-scene floorplans.
        self.floorplan_convnet_xy = nn.Sequential(nn.ReflectionPad2d(1),
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

        self.floorplan_convnet_yz = nn.Sequential(nn.ReflectionPad2d(1),
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

        self.floorplan_convnet_xz = nn.Sequential(nn.ReflectionPad2d(1),
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

        self.floorplan_convnet_xy.apply(init_weights_kaiming)
        self.floorplan_convnet_yz.apply(init_weights_kaiming)
        self.floorplan_convnet_xz.apply(init_weights_kaiming)

        self.pillar_aggregator_xz.apply(init_weights_kaiming)
        self.pillar_aggregator_yz.apply(init_weights_kaiming)
        self.pillar_aggregator_xy.apply(init_weights_kaiming)
    
    def get_resnet_feats(self, cam_xyz, feats, focal, c, width, height, stride = 1):
        
        focal = focal/stride
        c = c/stride
        uv = projection(cam_xyz, focal, c)

        im_x = uv[:,:, 0]
        im_y = uv[:,:, 1]
        im_grid = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)

        mask_z = cam_xyz[:,:,2]<1e-3
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (mask_z)

        im_grid = im_grid.unsqueeze(2)
        resnet_feat = F.grid_sample(feats, im_grid, align_corners=True, mode='bilinear', padding_mode="zeros")
        resnet_feat[mask.unsqueeze(1).unsqueeze(-1).repeat(1, resnet_feat.shape[1], 1,1) == False] = 0
        return resnet_feat[:, :, :, 0]

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

        # the resnet stride to downscale the intrinsics by
        

        _, _, H, W = images.shape
        features = self.spatial_encoder(images)
        print("images", images.shape)
        print("features", features.shape)
        NV, _, H_feat, W_feat = features.shape

        features = unprocess_images(images, encoder_type = 'resnet')

        stride =  H/H_feat

        # world_grid = inverse_contract_samples(self.world_grid.clone(), self.mag_original.clone())
        world_grid = self.world_grid.unsqueeze(0)

            
        # world_grids = repeat_interleave(self.world_grid.clone(),
        #                                         NV).cuda()  # (SB*NV, NC, 3) NC: number of grid cells
        
        world_grids = repeat_interleave(self.world_grid.clone(),
                                                NV)  # (SB*NV, NC, 3) NC: number of grid cells
        camera_grids = world2camera(world_grids, poses)

        camera_pts_dir = world_grids - poses[:, None, :3, -1]
        camera_pts_dir_norm = torch.norm(camera_pts_dir + 1e-9, dim=-1)
        # print("camera_pts_dir", camera_pts_dir.shape)
        # print("camera_pts_dir_norm", camera_pts_dir_norm)
        # print("camera_pts_dir_norm", camera_pts_dir_norm.shape)
        camera_pts_dir = camera_pts_dir/camera_pts_dir_norm[:, :, None]
        # camera_pts_dir = camera_pts_dir * masks[:, :, None]

        # # Projecting points in camera coordinates on the image plane
        # uv = projection(camera_grids, focal, c)  # [f, -f]
        

        latent = self.get_resnet_feats(camera_grids, features, focal, c, W_feat, H_feat, stride = stride)

        out_latent = latent
        mask = camera_grids[:,:,2] <1e-3

        mask_expand_grid = mask.unsqueeze(-1).repeat(1, 1, camera_grids.shape[-1])
        camera_pts_dir[mask_expand_grid == False] = 0

        _, L, _ = latent.shape  # (NV, L, grid_size**3)
        # latent[mask_expand == False] = 0
        latent = torch.cat([latent,
                            camera_grids.permute(0, -1, 1),
                            camera_pts_dir.permute(0, -1, 1)], 1)
        latent = latent.reshape(NV, L+3+3,
                                self.grid_size[0] * self.grid_size[1] * self.grid_size[2]).permute(0, 2, 1)  # Input to the linear layer # (SB*T*NV, grid_size**3, L+1)
        latent = self.depth_fc(latent)
        latent = latent.reshape(1, NV,
                                  self.grid_size[0],
                                  self.grid_size[1],
                                  self.grid_size[2], L)
        
        # latent = self.conv3d()

        out_latent_2 = latent

        latent = latent.mean(1)  # (SB * T, NC, L) average along the number of views
        
        latent_inp_x = torch.cat([latent, world_grids.reshape(1, NV, self.grid_size[0],
                                                          self.grid_size[1],
                                                          self.grid_size[2], 3)[:, 0, ..., 0:1]], -1)
        latent_inp_y = torch.cat([latent, world_grids.reshape(1, NV, self.grid_size[0],
                                                          self.grid_size[1],
                                                          self.grid_size[2], 3)[:, 0, ..., 1:2]], -1)
        latent_inp_z = torch.cat([latent, world_grids.reshape(1, NV, self.grid_size[0],
                                                          self.grid_size[1],
                                                          self.grid_size[2], 3)[:, 0, ..., 2:3]], -1)


        weights_yz = torch.softmax(self.pillar_aggregator_yz(latent_inp_x), dim=1)  # (SB, T, X, Z, Y, 1)
        weights_xz = torch.softmax(self.pillar_aggregator_xz(latent_inp_y), dim=2)  # (SB, T, X, Z, Y, 1)
        weights_xy = torch.softmax(self.pillar_aggregator_xy(latent_inp_z), dim=3)  # (SB, T, X, Z, Y, 1)

        floorplans_yz = (latent * weights_yz).sum(1)  # (SB, T, X, Z, L)
        floorplans_xz = (latent * weights_xz).sum(2)  # (SB, T, X, Z, L)
        floorplans_xy = (latent * weights_xy).sum(3)  # (SB, T, X, Z, L)

        scene_grid_yz = floorplans_yz.permute(0, -1, 1, 2)
        scene_grid_xz = floorplans_xz.permute(0, -1, 1, 2)
        scene_grid_xy = floorplans_xy.permute(0, -1, 1, 2)

        # grid_yz = floorplans_yz.permute(0, -1, 1, 2) # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        # scene_grid_yz = self.floorplan_convnet_yz(grid_yz)  # (SB*T, L, X, Z)

        # grid_xz = floorplans_xz.permute(0, -1, 1, 2) # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        # scene_grid_xz = self.floorplan_convnet_xz(grid_xz)  # (SB*T, L, X, Z)

        # grid_xy = floorplans_xy.permute(0, -1, 1, 2) # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        # scene_grid_xy = self.floorplan_convnet_xy(grid_xy)  # (SB*T, L, X, Z)
        
        return scene_grid_xz, scene_grid_xy, scene_grid_yz, out_latent, out_latent_2
        # return 0