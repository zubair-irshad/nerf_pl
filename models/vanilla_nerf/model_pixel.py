# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import models.vanilla_nerf.helper as helper
from models.interface import LitModel

from torch.utils.data import DataLoader
from datasets import dataset_dict
from collections import defaultdict
import torch.distributed as dist
from utils.train_helper import *
from models.vanilla_nerf.util import *
from models.vanilla_nerf.encoder import *
import wandb

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)   
random.seed(0)

class NeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 2,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        latent_size:int = 512,
        combine_layer:int = 6,
        combine_type="average"
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        pos_size += latent_size
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, x, condition_tile, latent, combine_inner_dims):

        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)

        x = torch.cat([x, latent], dim=-1)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)

            if idx == self.combine_layer:
                bottleneck = self.bottleneck_layer(x)
                #print("bottleneck", bottleneck.shape)
                x = combine_interleaved(
                    x, combine_inner_dims, self.combine_type
                )

            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            if idx == 0:
                x = combine_interleaved(
                    x, combine_inner_dims, self.combine_type
                )
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        return raw_rgb, raw_density

class PixelNeRF(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        # num_fine_samples: int = 128,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False,
        num_src_views: int = 3
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(PixelNeRF, self).__init__()

        self.encoder = SpatialEncoder()
        self.rgb_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)

    def encode(self, images):
        self.encoder(images)
        self.image_shape = torch.Tensor([images.shape[-1], images.shape[-2]]).to(images.device)
        self.latent_size = self.encoder.latent_size

    def forward(self, rays, randomized, white_bkgd, near, far):

        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                mlp = self.coarse_mlp

            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(
                    bins=t_mids,
                    weights=weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp

            B, N_samples, _ = samples.shape
            samples = samples.reshape(-1,3).unsqueeze(0)
            samples_cam = world2camera(samples, rays["src_poses"], self.num_src_views)
            focal = rays["src_focal"][0].unsqueeze(-1).repeat((1, 2))
            c = rays["src_c"][0].unsqueeze(0)
            uv = projection(samples_cam, focal, c, self.num_src_views) 
            
            latent = self.encoder.index(
                uv, None, self.image_shape
            )  # (SB * NS, latent, B) 
            latent = latent.transpose(1, 2).reshape(
                    -1, self.latent_size
                )  # (SB * NS * B, latent)

            samples_enc = helper.pos_enc(
                samples_cam,
                self.min_deg_point,
                self.max_deg_point,
            )

            viewdirs = world2camera_viewdirs(rays["viewdirs"].unsqueeze(0), rays["src_poses"], self.num_src_views)
            viewdirs_enc = helper.pos_enc(viewdirs, 0, self.deg_view)

            viewdirs_enc = torch.tile(viewdirs_enc[:, None, :], (1, N_samples, 1)).reshape(
                    -1, viewdirs_enc.shape[-1]
                )

            #viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
            
            NV, N_points, _ =  samples_enc.shape
            raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc, latent, combine_inner_dims=(self.num_src_views, N_points))

            raw_rgb = raw_rgb.reshape(B, N_samples, -1)
            raw_sigma = raw_sigma.reshape(B, N_samples, -1)

            if self.noise_std > 0 and randomized:
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std

            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)

            comp_rgb, acc, weights = helper.volumetric_rendering(
                rgb,
                sigma,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            ret.append((comp_rgb, acc))

        return ret

class LitPixelNeRF(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)

        self.hparams.update(vars(hparams))

        super(LitPixelNeRF, self).__init__()
        self.model = PixelNeRF()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]
        
        if self.hparams.dataset_name == 'pd' or self.hparams.dataset_name == 'pd_multi_obj' or self.hparams.dataset_name =='pd_multi_obj_ae':
            kwargs_train = {'root_dir': self.hparams.root_dir,
                             'img_wh': tuple(self.hparams.img_wh),
                                'white_back': self.hparams.white_back,
                                'model_type': 'pixelnerf'}
            kwargs_val = {'root_dir': self.hparams.root_dir,
                            'img_wh': tuple(self.hparams.img_wh),
                                'white_back': self.hparams.white_back,
                                'model_type': 'pixelnerf'}

        if self.hparams.run_eval:        
            kwargs_test = {'root_dir': self.hparams.root_dir,
                            'img_wh': tuple(self.hparams.img_wh),
                            'white_back': self.hparams.white_back}
            self.test_dataset = dataset(split='train', **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            self.train_dataset = dataset(split='train', **kwargs_train)
            self.val_dataset = dataset(split='val', **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back

    def training_step(self, batch, batch_idx):

        for k,v in batch.items():
            batch[k] = v.squeeze(0)

        self.model.encode(batch["src_imgs"])

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]
        target = batch["target"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
        loss = loss1 + loss0

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)

        return loss

    def render_rays(self, batch):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == 'src_imgs' or k =='src_poses' or k =='src_focal' or k=='src_c':
                   batch_chunk[k] = v 
                elif k =='radii':
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]                      
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far
            )
            #here 1 denotes fine
            ret["comp_rgb"]+=[rendered_results_chunk[1][0]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)
        psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        self.log("val/psnr", psnr_.item(), on_step=True, prog_bar=True, logger=True)
        return ret

    def validation_step(self, batch, batch_idx):
        for k,v in batch.items():
            batch[k] = v.squeeze()
            if k =='radii':
                batch[k] = v.unsqueeze(-1)
        self.model.encode(batch["src_imgs"])

        W,H = self.hparams.img_wh
        ret = self.render_rays(batch)
        # rank = dist.get_rank()
        rank =0
        if rank==0:
            grid_img = visualize_val_rgb(
                (W,H), batch, ret
            )
            self.logger.experiment.log({
                "val/GT_pred rgb": wandb.Image(grid_img)
            })

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = self.hparams.run_max_steps

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=32,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                        shuffle=False,
                        num_workers=4,
                        batch_size=self.hparams.batch_size,
                        pin_memory=True)

    # def validation_epoch_end(self, outputs):
    #     val_image_sizes = self.trainer.datamodule.val_image_sizes
    #     rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
    #     targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
    #     psnr_mean = self.psnr_each(rgbs, targets).mean()
    #     ssim_mean = self.ssim_each(rgbs, targets).mean()
    #     lpips_mean = self.lpips_each(rgbs, targets).mean()
    #     self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)

            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips