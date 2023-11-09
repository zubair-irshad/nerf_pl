import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('/home/zubairirshad/nerf_pl')
from models.nerfplusplus.util import *
import numpy as np
import torch.autograd.profiler as profiler

from models.nerfplusplus.spatial_encoder import SpatialEncoder, ResUNet
import models.nerfplusplus.helper as helper
from torch import linalg as LA
from torchvision import transforms as T
from inplace_abn import InPlaceABN

from kornia.utils import create_meshgrid
import torchvision
from models.nerfplusplus.util import get_norm_layer
norm_layer = get_norm_layer("batch")

pretrained_model = getattr(torchvision.models, "resnet34")(
    pretrained=False, norm_layer=norm_layer
)

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)
    
class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)

class ResUNet(nn.Module):
    def __init__(self, out_ch = 64):
        super().__init__()
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        filters = [64, 128, 256, 512]

        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)
        # self.layer4 = pretrained_model.layer4
        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)
        self.latent_size = out_ch
    
    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)
        return x_out
    
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory



        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid


def contract_samples(x, order=float('inf')):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag)), mag

def inverse_contract_samples(x, mag_origial,order=float('inf')):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (x*mag_origial)/(2-(1/mag_origial)))

def unprocess_images(normalized_images, encoder_type = 'resUNet'):
    if encoder_type =='resnet':
        inverse_transform = T.Compose([T.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))])
    else:
        inverse_transform = T.Compose([T.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))])
    return inverse_transform(normalized_images)

def get_c(samples, imgs, w2cs, focal, c, near=0.2, far =2.5, with_mask=True):

    NV,C, H, W = imgs.shape

    NC = NV*C

    intrinsics_ref = torch.FloatTensor([
        [focal[0], 0., c[0][0]],
        [0., focal[0], c[0][1]],
        [0., 0., 1.],
        ]).to(w2cs.device)
    #our feature size is H/2, W/2
    intrinsics_ref = intrinsics_ref/2
    intrinsics_ref[-1,-1] = 1

    C += with_mask

    B, N_samples, _ = samples.shape
    # imgs_unprocess = unprocess_images(imgs, encoder_type = encoder_type)
    imgs_unprocess = unprocess_images(imgs, encoder_type = 'resUNet')
    imgs_unprocess = F.interpolate(imgs_unprocess, size=(120,160), mode='bilinear', align_corners=False)

    world_xyz = repeat_interleave(samples.reshape(-1,3).unsqueeze(0), NV)  # (SB*NS, B, 3)
    _, uv, _ =  projection_extrinsics(world_xyz, w2cs, intrinsics_ref)

    im_x = uv[:,:, 0]
    im_y = uv[:,:, 1]
    im_grid = torch.stack([2 * im_x / (W - 1) - 1, 2 * im_y / (H - 1) - 1], dim=-1)

    data_im = F.grid_sample(imgs, im_grid.unsqueeze(2), align_corners=True, mode='bilinear', padding_mode='zeros')
    feats_c = data_im.squeeze(-1).permute(2,0,1).view(-1,NC)
    return feats_c


def index_grid(samples, volume_features, w2cs, focal, c, near=0.2, far = 2.5):
    """
    Get pixel-aligned image features at 2D image coordinates
    :param uv (B, N, 2) image points (x,y)
    :param cam_z ignored (for compatibility)
    :param image_size image size, either (width, height) or single int.
    if not specified, assumes coords are in [-1, 1]
    :param z_bounds ignored (for compatibility)
    :return (B, L, N) L is latent size
    """ 

    w2c_ref = w2cs[0].unsqueeze(0)
    _,_,_, H, W = volume_features.shape
    inv_scale = torch.tensor([W-1, H-1]).to(w2c_ref.device)

    samples = samples.reshape(-1,3).unsqueeze(0)
    intrinsics_ref = torch.FloatTensor([
        [focal[0], 0., c[0][0]],
        [0., focal[0], c[0][1]],
        [0., 0., 1.],
        ]).to(w2c_ref.device)
    intrinsics_ref = intrinsics_ref/2
    intrinsics_ref[-1,-1] = 1

    if intrinsics_ref is not None:
        point_samples_pixel = projection_extrinsics_alldim(samples, w2c_ref, intrinsics_ref)
        point_samples_pixel = point_samples_pixel.squeeze(0)
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  
        point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
    
    grid = point_samples_pixel.view(1, 1, 1, -1, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
    features = F.grid_sample(volume_features, grid, align_corners=True, mode='bilinear')[:,:,0].permute(2,3,0,1).squeeze()#, padding_mode="border"

    return features
    

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
            xyz_max = None,
            use_transformer = False,
            use_stride = False,
            NC = 9
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

        # if encoder_type == 'resnet':
        #     self.spatial_encoder = SpatialEncoder(backbone="resnet34",
        #                                         pretrained=True,
        #                                         num_layers=4,
        #                                         index_interp="bilinear",
        #                                         index_padding="zeros",
        #                                         # index_padding="border",
        #                                         upsample_interp="bilinear",
        #                                         feature_scale=1.0,
        #                                         use_first_pool=True,
        #                                         norm_type="batch")
        # else:
        #     self.spatial_encoder = ResUNet()

        self.feature = ResUNet()

        self.latent_size = 64
        # self.index_interp = index_interp
        # self.index_padding = index_padding
        # self.upsample_interp = upsample_interp
        # LS = self.latent_size
        # NC = 3*5
        self.cost_reg_2 = CostRegNet(64+NC, norm_act = InPlaceABN)


    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        _, N_img, C_img, _, _ = imgs.shape
        NC = N_img*C_img

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]

        
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        img_feat = torch.empty((B, NC + 64, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs, src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:, (i + 1) * 3:(i + 2) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i + 1] = in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        a = volume_sq_sum * count - (volume_sum * count) ** 2
        img_feat[:, -64:] = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks
    
    def forward(self, imgs, proj_mats, near = 0.2, far = 2.5, return_color=False, lindisp=False, pad =0):

        NV, C, H, W = imgs.shape

        B = 1
        imgs = imgs.reshape(NV, 3, H, W)
        features = self.feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)

        imgs = imgs.view(1, NV, 3, H, W)
        feats_l = features  # (B*V, C, h, w)
        proj_mats = proj_mats.unsqueeze(0)

        feats_l = feats_l.view(1, NV, *feats_l.shape[1:])  # (B, V, C, h, w)
        D = 128
        t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        depth_values = near * (1.-t_vals) + far * (t_vals)
        depth_values = depth_values.unsqueeze(0)
        volume_feat, in_masks = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad)

        if return_color:
            feats_l = torch.cat((volume_feat[:,:NV*3].view(B, NV, 3, *volume_feat.shape[2:]),in_masks.unsqueeze(2)),dim=2)

        volume_feat = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])

        return volume_feat, feats_l, depth_values
    

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))
    
class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x
    
if __name__ == "__main__":

    encoder = ResUNet()

    total_params = sum(p.numel() for p in encoder.parameters())
    print("total custom  pretrained params", total_params)

    img = torch.randn((3,3,240,320))
    latent = encoder(img)
    print("custom resent 34 ibrnet unet",latent.shape)

    encoder = GridEncoder()
    imgs = torch.randn((3,3, 240,320))
    proj_mats = torch.randn((1,3,3,4))

    volume_feat, feats_l, depth_values = encoder(imgs, proj_mats)

    print("volume_feat, feats_l, depth_values", volume_feat.shape, feats_l.shape, depth_values.shape)