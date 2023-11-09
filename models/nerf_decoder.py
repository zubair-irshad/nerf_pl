"""
Main model implementation
"""
import torch
import torch.autograd.profiler as profiler
import numpy as np
from .mlp_res import MlpResNet


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer("_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1))
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get("num_freqs", 6),
            d_in,
            conf.get("freq_factor", np.pi),
            conf.get("include_input", True),
        )


class ConditionalRenderer(torch.nn.Module):
    def __init__(
        self,
        global_mlp=None,
        color_mlp=None,
        density_mlp=None,
        use_xyz=True,
        normalize_z=True,
        use_code=True,
        use_code_viewdirs=False,
        use_viewdirs=False,
        symm_axis=None,
        code=None,
        positional_noise=0,
        **kwargs
    ):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.use_xyz = use_xyz

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = normalize_z
        self.use_code = use_code  # Positional encoding
        self.use_code_viewdirs = use_code_viewdirs  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = use_viewdirs

        # Aggrate all image global latents to a single one?
        d_in = 3 if self.use_xyz else 1

        self.symm_axis = symm_axis

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(code, d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        if global_mlp is not None and global_mlp.get("use", True):
            global_mlp_conf = global_mlp
            global_injections = None
            if "injections" in global_mlp_conf:
                global_injections = {int(k): dict(v) for k, v in global_mlp_conf["injections"].items()}

            self.global_mlp = MlpResNet(
                d_in=global_mlp_conf["code_size"] + self.code.d_out,
                dims=global_mlp_conf["num_layers"] * [global_mlp_conf["d_hidden"]],
                d_out=global_mlp_conf.get("d_out", 4),
                injections=global_injections,
                agg_fct=global_mlp_conf.get("agg_fct", "sum"),
                add_out_lvl=global_mlp_conf.get("add_out_lvl", None),
                add_out_dim=global_mlp_conf.get("add_out_dim", None),
            )

        if density_mlp is not None and density_mlp.get("use", True):
            density_mlp_conf = density_mlp
            density_injections = None
            if "injections" in density_mlp_conf:
                density_injections = {int(k): dict(v) for k, v in density_mlp_conf["injections"].items()}

            self.density_mlp = MlpResNet(
                d_in=density_mlp_conf["code_size"] + self.code.d_out,
                dims=density_mlp_conf["num_layers"] * [density_mlp_conf["d_hidden"]],
                d_out=density_mlp_conf.get("d_out", 1),
                injections=density_injections,
                agg_fct=density_mlp_conf.get("agg_fct", "sum"),
                add_out_lvl=density_mlp_conf.get("add_out_lvl", None),
            )

        if color_mlp is not None and color_mlp.get("use", True):
            color_mlp_conf = color_mlp
            color_injections = None
            if "injections" in color_mlp_conf:
                color_injections = {int(k): dict(v) for k, v in color_mlp_conf["injections"].items()}

            self.color_mlp = MlpResNet(
                d_in=color_mlp_conf["code_size"] + self.code.d_out,
                dims=color_mlp_conf["num_layers"] * [color_mlp_conf["d_hidden"]],
                d_out=color_mlp_conf.get("d_out", 3),
                injections=color_injections,
                agg_fct=color_mlp_conf.get("agg_fct", "sum"),
            )

        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)

        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1

        self.positional_noise = positional_noise

    def forward(self, rep_pts=None, pts=None, viewdirs=None, cond=None):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        add_output = None
        world_xyz = pts
        world_xyz_symm = torch.clone(world_xyz)
        if self.symm_axis is not None:
            # reflect sampled points along symmetry axis
            if self.symm_axis == "x":
                world_xyz_symm[..., 0] *= torch.sign(world_xyz_symm[..., 0])
            elif self.symm_axis == "y":
                world_xyz_symm[..., 1] *= torch.sign(world_xyz_symm[..., 1])
            else:
                raise NotImplementedError("Other symmetry axis not yet supported")
        query_points = None
        query_cams = None
        if len(world_xyz.shape) == 3:
            SB, B, _ = world_xyz.shape
        else:
            SB = 0
            B, _ = world_xyz.shape
        # prepare query points
        z_feature = world_xyz.reshape(-1, 3)  # (SB*B, 3)
        if self.positional_noise > 0:
            z_feature += self.positional_noise * (torch.rand(z_feature.shape) - 0.5).to(z_feature)

        z_feature_symm = world_xyz.reshape(-1, 3)  # (SB*B, 3)

        if self.use_code and not self.use_code_viewdirs:
            # Positional encoding (no viewdirs)
            z_feature = self.code(z_feature)
            z_feature_symm = self.code(z_feature_symm)
            query_points = z_feature
            query_points_symm = z_feature_symm

        viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
        if self.use_viewdirs:
            # * Encode the view directions
            assert viewdirs is not None
            # Viewdirs stay in world space
            """
            viewdirs = viewdirs.reshape(SB, B, 3, 1)
            viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
            """
            z_feature = torch.cat((z_feature, viewdirs), dim=1)  # (SB*B, 4 or 6)
            z_feature_symm = torch.cat((z_feature_symm, viewdirs), dim=1)  # (SB*B, 4 or 6)

        if self.use_code and self.use_code_viewdirs:
            # Positional encoding (with viewdirs)
            z_feature = self.code(z_feature)
            if self.model_type in ["ShapeColor", "ShapeColorEmb"]:
                query_points = self.code(z_feature[..., :-3])
                query_cams = self.code(viewdirs)
        else:
            query_cams = viewdirs
            # for each head, prepare mlp input
        if hasattr(self, "global_mlp"):
            lat = cond[0]["global"]
            lat = lat[None,].expand(len(query_points),len(lat))
            inj_data = {
                "lat,pos": torch.cat([lat, query_points], 1),
                "lat,pos_symm": torch.cat([lat, query_points_symm], 1),
                "lat": lat,
                "query_points": query_points,
                "query_points_symm": query_points_symm,
                "query_cams": query_cams,
            }
            if self.global_mlp.input is None:
                mlp_input = torch.cat([lat, query_points], 1)
            elif self.global_mlp.input == "lat,pos":
                mlp_input = torch.cat([lat, query_points], 1)
            elif self.global_mlp.input == "lat,pos_symm":
                mlp_input = torch.cat([lat, query_points_symm], 1)

            if self.global_mlp.add_out_lvl is not None:
                global_out, add_output = self.global_mlp(mlp_input, inj_data=inj_data)
            else:
                global_out = self.global_mlp(mlp_input, inj_data=inj_data)
            rgb_out = global_out[..., :3]
            sigma_out = global_out[..., 3:4]

        if hasattr(self, "density_mlp"):
            lat = cond[0]["density"]
            lat = lat[None,].expand(len(query_points),len(lat))

            inj_data = {
                "lat,pos": torch.cat([lat, query_points], 1),
                "lat,pos_symm": torch.cat([lat, query_points_symm], 1),
                "lat": lat,
                "query_points": query_points,
                "query_points_symm": query_points_symm,
                "query_cams": query_cams,
            }
            if self.density_mlp.input is None:
                mlp_input = torch.cat([lat, query_points], 1)
            elif self.density_mlp.input == "lat,pos":
                mlp_input = torch.cat([lat, query_points], 1)
            elif self.density_mlp.input == "lat,pos_symm":
                mlp_input = torch.cat([lat, query_points_symm], 1)
            if self.density_mlp.add_out_lvl is not None:
                sigma_out, density_feats = self.density_mlp(mlp_input, inj_data=inj_data)
            else:
                sigma_out = self.density_mlp(mlp_input, inj_data=inj_data)
                density_feats = None

        if hasattr(self, "color_mlp"):
            lat = cond[0]["color"]
            lat = lat[None,].expand(len(query_points),len(lat))
            inj_data = {
                "lat,pos": torch.cat([lat, query_points], 1),
                "lat,pos_symm": torch.cat([lat, query_points_symm], 1),
                "lat": lat,
                "query_points": query_points,
                "query_points_symm": query_points_symm,
                "query_cams": query_cams,
                "density_feats": density_feats,
            }
            if self.color_mlp.input is None:
                mlp_input = torch.cat([lat, query_points], 1)
            elif self.color_mlp.input == "lat,pos":
                mlp_input = torch.cat([lat, query_points], 1)
            elif self.color_mlp.input == "lat,pos_symm":
                mlp_input = torch.cat([lat, query_points_symm], 1)

            rgb_out = self.color_mlp(mlp_input, inj_data=inj_data)


        return torch.cat([sigma_out,rgb_out],1)
