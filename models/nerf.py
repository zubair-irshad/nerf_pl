import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
# from models.resnet_encoder import MultiHeadImgEncoder

import tinycudann as tcnn
from models.activation import trunc_exp
from models.layers import *

class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out

class NeRF_TCNN(nn.Module):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # sigma network
        self.bound = bound
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    
    def forward(self, input_xd):
        x = input_xd[:, :3]
        d = input_xd[:, 3:]
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]


        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        outputs = torch.cat([color, sigma[..., None]], -1)

        return outputs

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        # if self.bg_radius > 0:
        #     params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
        #     params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

# class NeRF_TCNN(nn.Module):
#     def __init__(self,
#                  encoding="HashGrid",
#                  encoding_dir="SphericalHarmonics",
#                  num_layers=2,
#                  hidden_dim=128,
#                  geo_feat_dim=15,
#                  num_layers_color=3,
#                  hidden_dim_color=64,
#                  bound=2,
#                  **kwargs
#                  ):
#         super().__init__(**kwargs)

#         self.bound = bound

#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim

#         per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

#         self.encoder = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": 16,
#                 "n_features_per_level": 2,
#                 "log2_hashmap_size": 19,
#                 "base_resolution": 16,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         # self.sigma_net = tcnn.Network(
#         #     n_input_dims=32,
#         #     n_output_dims=2 + self.geo_feat_dim,
#         #     network_config={
#         #         "otype": "FullyFusedMLP",
#         #         "activation": "ReLU",
#         #         "output_activation": "None",
#         #         "n_neurons": hidden_dim,
#         #         "n_hidden_layers": num_layers - 1,
#         #     },
#         # )

#         self.sigma_net = tcnn.Network(
#             n_input_dims=32,
#             n_output_dims=1 + self.geo_feat_dim,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim,
#                 "n_hidden_layers": num_layers - 1,
#             },
#         )

#         # color network
#         self.num_layers_color = num_layers_color
#         self.hidden_dim_color = hidden_dim_color

#         self.encoder_dir = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "SphericalHarmonics",
#                 "degree": 4,
#             },
#         )

#         self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

#         self.color_net = tcnn.Network(
#             n_input_dims=self.in_dim_color,
#             n_output_dims=3,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim_color,
#                 "n_hidden_layers": num_layers_color - 1,
#             },
#         )

#     def forward(self, input):
#         x = input[:, :3]
#         d = input[:, 3:]

#         # x: [N, 3], in [-bound, bound]
#         # d: [N, 3], nomalized in [-1, 1]

#         # sigma
#         x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
#         x = self.encoder(x)
#         # h = self.sigma_net(x)
#         h = self.sigma_net(x)
#         sigma = trunc_exp(h[..., 0])

#         # sigma = h[..., 0]
#         # logit = h[..., 1]
        
#         geo_feat = h[..., 1:]

#         # color
#         d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)

#         # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
#         h = torch.cat([d, geo_feat], dim=-1)
#         h = self.color_net(h)

#         # sigmoid activation for rgb
#         # color = h
#         color = torch.sigmoid(h)

#         # outputs = torch.cat([color, sigma[..., None], logit[..., None]], -1)
#         outputs = torch.cat([color, sigma[..., None]], -1)
#         return outputs

#     def get_params(self, lr):
#         params = [
#             {'params': self.encoder.parameters(), 'lr': lr},
#             {'params': self.sigma_net.parameters(), 'lr': lr},
#             {'params': self.encoder_dir.parameters(), 'lr': lr},
#             {'params': self.color_net.parameters(), 'lr': lr}, 
#         ]
#         return params

#     def density(self, x):
#         # x: [N, 3], in [-bound, bound]

#         x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
#         x = self.encoder(x)
#         h = self.sigma_net(x)

#         sigma = h[..., 0]
#         logit = h[..., 1]
#         geo_feat = h[..., 2:]

#         return {
#             'sigma': sigma,
#             'geo_feat': geo_feat,
#             'logit': logit
#         }

#     # allow masked inference
#     def color(self, x, d, mask=None, geo_feat=None, **kwargs):
#         # x: [N, 3] in [-bound, bound]
#         # mask: [N,], bool, indicates where we actually needs to compute rgb.

#         x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

#         if mask is not None:
#             rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
#             # in case of empty mask
#             if not mask.any():
#                 return rgbs
#             x = x[mask]
#             d = d[mask]
#             geo_feat = geo_feat[mask]

#         # color
#         d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)

#         h = torch.cat([d, geo_feat], dim=-1)
#         h = self.color_net(h)

#         # sigmoid activation for rgb
#         h = torch.sigmoid(h)

#         if mask is not None:
#             rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
#         else:
#             rgbs = h

#         return rgbs


class ObjectNeRF(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super(ObjectNeRF, self).__init__()
        self.hparams = hparams
        self.use_voxel_embedding = False
        # initialize neural model with config
        self.initialize_scene_branch(hparams)
        self.initialize_object_branch(hparams)

    def initialize_scene_branch(self, hparams):
        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2

        self.activation = nn.LeakyReLU(inplace=True)

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W // 2), self.activation
        )

    def initialize_object_branch(self, hparams):
        # instance encoding
        N_obj_code_length = hparams.N_obj_code_length
        
        inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.in_channels_xyz + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = hparams.inst_D
        self.inst_W = hparams.inst_W
        self.inst_skips = hparams.inst_skips

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code = inputs["obj_code"]

        print("emb_xyz")
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict

class ConditionalNeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4], latent_size=256):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(ConditionalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_xyz = in_channels_xyz
        self.in_dir = in_channels_dir
        self.in_channels_xyz = in_channels_xyz + latent_size
        self.in_channels_dir = in_channels_dir + latent_size
        self.skips = skips

        # in_channels_xyz += latent_size
        # in_channels_dir += latent_size

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+self.in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, shape_latent, texture_latent, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        shape_latent = shape_latent.repeat(x.shape[0], 1)
        texture_latent = texture_latent.repeat(x.shape[0], 1)
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_xyz, self.in_dir], dim=-1)
        else:
            input_xyz = x
        
        input_xyz = torch.cat([input_xyz, shape_latent], dim=1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir, texture_latent], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out

def PE(x, degree):
    y = torch.cat([2.**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)

class CodeNeRF(nn.Module):
    def __init__(self, shape_blocks = 3, texture_blocks = 1, W = 256, 
                 num_xyz_freq = 10, num_dir_freq = 4, latent_dim=256):
        super().__init__()
        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.num_xyz_freq = num_xyz_freq
        self.num_dir_freq = num_dir_freq
        
        self.d_xyz, self.d_viewdir = 3 + 6 * num_xyz_freq, 3 + 6 * num_dir_freq
        self.encoding_xyz = nn.Sequential(nn.Linear(self.d_xyz, W), nn.ReLU())
        for j in range(shape_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
            setattr(self, f"shape_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"shape_layer_{j+1}", layer)
        self.encoding_shape = nn.Linear(W,W)
        # self.sigma = nn.Sequential(nn.Linear(W,1), nn.Softplus())
        self.sigma = nn.Sequential(nn.Linear(W,1))
        # self.sigma = nn.Sequential(nn.Linear(W,1), nn.Softplus())
        self.encoding_viewdir = nn.Sequential(nn.Linear(W+self.d_viewdir, W), nn.ReLU())
        for j in range(texture_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"texture_layer_{j+1}", layer)
        self.rgb = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, 3),  nn.Sigmoid())
        # self.rgb = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, 3))
        
    def forward(self, x, shape_latent, texture_latent):

        # print("x", x.shape)

        xyz, viewdir = \
                torch.split(x, [self.d_xyz, self.d_viewdir], dim=-1)

        # print("xyz", xyz.shape), print("viewdir", viewdir.shape), print("shape_latent", shape_latent.shape), print("texture_latent", texture_latent.shape)

        # xyz = PE(xyz, self.num_xyz_freq)
        # viewdir = PE(viewdir, self.num_dir_freq)
        y = self.encoding_xyz(xyz)
        for j in range(self.shape_blocks):
            z = getattr(self, f"shape_latent_layer_{j+1}")(shape_latent)
            y = y + z
            y = getattr(self, f"shape_layer_{j+1}")(y)
        y = self.encoding_shape(y)
        sigmas = self.sigma(y)
        y = torch.cat([y, viewdir], -1)
        y = self.encoding_viewdir(y)
        for j in range(self.texture_blocks):
            z = getattr(self, f"texture_latent_layer_{j+1}")(texture_latent)
            y = y + z
            y = getattr(self, f"texture_layer_{j+1}")(y)
        rgbs = self.rgb(y)
        out = torch.cat([rgbs, sigmas], -1)
        return out

class ObjectBckgNeRF(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super(ObjectBckgNeRF, self).__init__()
        self.hparams = hparams
        self.use_voxel_embedding = False
        # initialize neural model with config
        self.initialize_scene_branch(hparams)
        self.initialize_object_branch(hparams)

    def initialize_scene_branch(self, hparams):
        #background latent encoding
        N_obj_code_length = hparams.N_obj_code_length

        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size

        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + 128
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2

        self.activation = nn.LeakyReLU(inplace=True)

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W // 2), self.activation
        )

    def initialize_object_branch(self, hparams):
        # instance encoding
        N_obj_code_length = hparams.N_obj_code_length
        
        inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.xyz_emb_size + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = hparams.inst_D
        self.inst_W = hparams.inst_W
        self.inst_skips = hparams.inst_skips

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        bckg_code = inputs["bckg_code"]
        input_dir = inputs.get("emb_dir", None)

        input_xyz = torch.cat([input_xyz, bckg_code], -1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code = inputs["obj_code"]
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict

class ObjectBckgNeRFConditional(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super(ObjectBckgNeRFConditional, self).__init__()
        self.hparams = hparams
        self.use_voxel_embedding = False
        # initialize neural model with config
        self.initialize_scene_branch(hparams)
        self.initialize_object_branch(hparams)

    def initialize_scene_branch(self, hparams):
        #background latent encoding
        N_obj_code_length = hparams.N_obj_code_length

        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size

        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + 128
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2

        self.activation = nn.LeakyReLU(inplace=True)

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W // 2), self.activation
        )

    def initialize_object_branch(self, hparams):
        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size
        N_obj_code_length = hparams.N_obj_code_length
        self.activation = nn.LeakyReLU(inplace=True)
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + N_obj_code_length
        in_channels_dir = 3 + 3 * self.N_freq_dir * 2
        self.in_channels_dir = in_channels_dir + N_obj_code_length
        
        inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.xyz_emb_size + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = hparams.inst_D
        self.inst_W = hparams.inst_W
        self.inst_skips = hparams.inst_skips

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code_shape = inputs["obj_code_shape"]
        obj_code_appearance = inputs["obj_code_appearance"]
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code_shape], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code_shape], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir, obj_code_appearance], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict

    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        bckg_code = inputs["bckg_code"]
        input_dir = inputs.get("emb_dir", None)

        print("input_xyz, bckg_code, input_dir", input_xyz.shape, bckg_code.shape, input_dir.shape)

        input_xyz = torch.cat([input_xyz, bckg_code], -1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict


class StyleGenerator2D(nn.Module):
    def __init__(self, out_res = 32, out_ch = 32, z_dim = 128, ch_mul=1, ch_max=256, skip_conn=False):
        super().__init__()

        self.skip_conn = skip_conn

        # dict key is the resolution, value is the number of channels
        # a trend in both StyleGAN and BigGAN is to use a constant number of channels until 32x32
        self.channels = {
            4: ch_max,
            8: ch_max,
            16: ch_max,
            32: ch_max,
            64: (ch_max // 2 ** 1) * ch_mul,
            128: (ch_max // 2 ** 2) * ch_mul,
            256: (ch_max // 2 ** 3) * ch_mul,
            512: (ch_max // 2 ** 4) * ch_mul,
            1024: (ch_max // 2 ** 5) * ch_mul,
        }

        self.latent_normalization = PixelNorm()
        self.mapping_network = []
        for i in range(3):
            self.mapping_network.append(EqualLinear(in_channel=z_dim, out_channel=z_dim, lr_mul=0.01, activate=True))
        self.mapping_network = nn.Sequential(*self.mapping_network)

        log_size_in = int(math.log(4, 2))  # 4x4
        log_size_out = int(math.log(out_res, 2))

        self.input = ConstantInput(channel=self.channels[4])

        self.conv1 = ModulatedConv2d(
            in_channel=self.channels[4],
            out_channel=self.channels[4],
            kernel_size=3,
            z_dim=z_dim,
            upsample=False,
            activate=True,
        )

        if self.skip_conn:
            self.to_rgb1 = ToRGB(in_channel=self.channels[4], out_channel=out_ch, z_dim=z_dim, upsample=False)
            self.to_rgbs = nn.ModuleList()

        self.convs = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(log_size_in + 1, log_size_out + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                ModulatedConv2d(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=3,
                    z_dim=z_dim,
                    upsample=True,
                    activate=True,
                )
            )

            self.convs.append(
                ModulatedConv2d(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    kernel_size=3,
                    z_dim=z_dim,
                    upsample=False,
                    activate=True,
                )
            )

            if self.skip_conn:
                self.to_rgbs.append(ToRGB(in_channel=out_channel, out_channel=out_ch, z_dim=z_dim, upsample=True))

            in_channel = out_channel

        # if not accumulating with skip connections we need final layer to map to out_ch channels
        if not self.skip_conn:
            self.out_rgb = ToRGB(in_channel=out_channel, out_channel=out_ch, z_dim=z_dim, upsample=False)
            self.to_rgbs = [None] * (log_size_out - log_size_in)  # dummy for easier control flow

        if self.skip_conn:
            self.n_layers = len(self.convs) + len(self.to_rgbs) + 2
        else:
            self.n_layers = len(self.convs) + 2

    def process_latents(self, z):
        # output should be list with separate latent code for each conditional layer in the model

        if isinstance(z, list):  # latents already in proper format
            pass
        elif z.ndim == 2:  # standard training, shape [B, ch]
            z = self.latent_normalization(z)
            z = self.mapping_network(z)
            z = [z] * self.n_layers
        elif z.ndim == 3:  # latent optimization, shape [B, n_latent_layers, ch]
            n_latents = z.shape[1]
            z = [self.latent_normalization(self.mapping_network(z[:, i])) for i in range(n_latents)]
        return z

    def forward(self, z):
        z = self.process_latents(z)

        out = self.input(z[0])
        B = out.shape[0]
        out = out.view(B, -1, 4, 4)

        out = self.conv1(out, z[0])

        if self.skip_conn:
            skip = self.to_rgb1(out, z[1])
            i = 2
        else:
            i = 1

        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out, z[i])
            out = conv2(out, z[i + 1])

            if self.skip_conn:
                skip = to_rgb(out, z[i + 2], skip)
                i += 3
            else:
                i += 2

        if not self.skip_conn:
            skip = self.out_rgb(out, z[i])
        return skip

class ObjectBckgNeRFGSN(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super(ObjectBckgNeRFGSN, self).__init__()
        self.hparams = hparams
        self.use_voxel_embedding = False
        # initialize neural model with config
        self.initialize_scene_branch(hparams)
        self.initialize_object_branch(hparams)
        
        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)

        self.z_decoder = StyleGenerator2D()

    def initialize_scene_branch(self, hparams):
        #background latent encoding
        N_obj_code_length = hparams.N_obj_code_length

        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size

        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + 32
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2

        self.activation = nn.LeakyReLU(inplace=True)

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W // 2), self.activation
        )

    def initialize_object_branch(self, hparams):
        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size
        N_obj_code_length = hparams.N_obj_code_length
        self.activation = nn.LeakyReLU(inplace=True)
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + N_obj_code_length
        in_channels_dir = 3 + 3 * self.N_freq_dir * 2
        self.in_channels_dir = in_channels_dir + N_obj_code_length
        
        inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.xyz_emb_size + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = hparams.inst_D
        self.inst_W = hparams.inst_W
        self.inst_skips = hparams.inst_skips

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())


    def sample_local_latents(self, z, xyz):
        local_latents = self.z_decoder(z=z)
        if local_latents.ndim == 4:
            B, local_z_dim, H, W = local_latents.shape
            # take only x and z coordinates, since our latent codes are in a 2D grid (no y dimension)
            # for the purposes of grid_sample we treat H*W as the H dimension and samples_per_ray as the W dimension
            xyz = xyz[:, :, :, [0, 2]]  # [B, H * W, samples_per_ray, 2]
        elif local_latents.ndim == 5:
            B, local_z_dim, D, H, W = local_latents.shape
            B, HW, samples_per_ray, _ = xyz.shape
            H = int(np.sqrt(HW))
            xyz = xyz.view(B, H, H, samples_per_ray, 3)

        samples_per_ray = xyz.shape[2]
        # all samples get the most detailed latent codes
        sampled_local_latents = nn.functional.grid_sample(
            input=local_latents,
            grid=xyz,
            mode='bilinear',  # bilinear mode will use trilinear interpolation if input is 5D
            align_corners=False,
            padding_mode="zeros",
        )
        # output is shape [B, local_z_dim, H * W, samples_per_ray]
        if local_latents.ndim == 4:
            # put channel dimension at end: [B, H * W, samples_per_ray, local_z_dim]
            sampled_local_latents = sampled_local_latents.permute(0, 2, 3, 1)
        elif local_latents.ndim == 5:
            sampled_local_latents = sampled_local_latents.permute(0, 2, 3, 4, 1)

        # merge everything else into batch dim: [B * H * W * samples_per_ray, local_z_dim]
        sampled_local_latents = sampled_local_latents.reshape(-1, local_z_dim)
        return sampled_local_latents, local_latents
        
    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        print("emb_xyz", emb_xyz.shape)
        input_dir = inputs.get("emb_dir", None)
        obj_code_shape = inputs["obj_code_shape"]
        obj_code_appearance = inputs["obj_code_appearance"]
        
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code_shape], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code_shape], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        print("inst_sigma", inst_sigma.shape)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir, obj_code_appearance], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict

    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        bckg_code = inputs["bckg_code"]
        input_dir = inputs.get("emb_dir", None)

        input_xyz = torch.cat([input_xyz, bckg_code], -1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict


class InstanceNeRF(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super(InstanceNeRF, self).__init__()
        self.hparams = hparams
        self.use_voxel_embedding = False
        self.initialize_object_branch(hparams)

    def initialize_object_branch(self, hparams):
        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size
        N_obj_code_length = hparams.N_obj_code_length
        self.activation = nn.LeakyReLU(inplace=True)
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + N_obj_code_length
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2
        
        inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.xyz_emb_size + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = hparams.inst_D
        self.inst_W = hparams.inst_W
        self.inst_skips = hparams.inst_skips

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code = inputs["obj_code"]
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict


class InstanceConditionalNeRF(nn.Module):
    def __init__(
        self,
        hparams
    ):
        super(InstanceConditionalNeRF, self).__init__()
        self.hparams = hparams
        self.use_voxel_embedding = False
        self.initialize_object_branch(hparams)

    def initialize_object_branch(self, hparams):
        self.D = hparams.D
        self.W = hparams.W
        self.N_freq_xyz = hparams.N_freq_xyz
        self.N_freq_dir = hparams.N_freq_dir
        self.skips = hparams.skips
        # embedding size for voxel representation
        voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.xyz_emb_size = xyz_emb_size
        N_obj_code_length = hparams.N_obj_code_length
        self.activation = nn.LeakyReLU(inplace=True)
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size + N_obj_code_length
        in_channels_dir = 3 + 3 * self.N_freq_dir * 2
        self.in_channels_dir = in_channels_dir + N_obj_code_length
        
        inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.xyz_emb_size + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = hparams.inst_D
        self.inst_W = hparams.inst_W
        self.inst_skips = hparams.inst_skips

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code_shape = inputs["obj_code_shape"]
        obj_code_appearance = inputs["obj_code_appearance"]
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code_shape], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code_shape], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir, obj_code_appearance], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict

if __name__ == '__main__':

    model = NeRF_TCNN(
        encoding="hashgrid",
    )
    print("HEREE")
    model_nerf = NeRF(in_channels_xyz=63,
                                in_channels_dir=27)

    input_nerf = torch.randn((2, 90))
    input_tnn = torch.randn((2, 6))
    print("input_nerf", input_nerf.shape)

    output = model_nerf(input_nerf)
    model = model.cuda()
    input_tnn = input_tnn.cuda()

    output_tcnn = model(input_tnn)
    print("OUTPUT",output.shape)
    print("OUTPUT TCNN",output_tcnn.shape)