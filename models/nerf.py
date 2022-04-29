import torch
from torch import nn

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
