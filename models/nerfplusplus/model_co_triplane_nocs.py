# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF++ (https://github.com/Kai-46/nerfplusplus)
# Copyright (c) 2020 the NeRF++ authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import models.nerfplusplus.helper as helper
from models.utils import store_image, write_stats
from models.interface import LitModel
from torch.utils.data import DataLoader
from datasets import dataset_dict
from collections import defaultdict
import torch.distributed as dist
from utils.train_helper import *
from models.nerfplusplus.util import *
from models.nerfplusplus.encoder_tp import GridEncoder
# from models.nerfplusplus.encoder_gp import GridEncoder
import wandb
import random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)   
random.seed(0)

# @gin.configurable()
class NeRFPPMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        # netdepth: int = 8,
        netdepth: int = 4,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 64,
        # skip_layer: int = 4,
        skip_layer: int = 2,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        latent_size: int = 512,
        out_nocs: bool = False
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFPPMLP, self).__init__()

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

        if self.out_nocs:
            self.nocs_layer = nn.Linear(netwidth, num_rgb_channels)
            self.ins_layer = nn.Linear(netwidth, 1)
            init.xavier_uniform_(self.nocs_layer.weight)
            init.xavier_uniform_(self.ins_layer.weight)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, x, condition, latent, out_nocs=False):
        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        latent = latent.reshape(-1, latent.shape[-1])
        
        x = torch.cat([x, latent], dim=-1)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        if out_nocs:
            #Could consider detaching the features as mentioned in: https://github.com/vLAR-group/DM-NeRF/blob/082da8d64c6d7d77936f53d16b1a61ca35b1f9b7/networks/dm_nerf.py#L95
            raw_nocs = self.nocs_layer(x).reshape(-1, num_samples, self.num_rgb_channels)
            sem_logits = self.ins_layer(x).reshape(-1, num_samples, 1)
            
        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        if out_nocs:
            return raw_rgb, raw_density, raw_nocs, sem_logits
        else:
            return raw_rgb, raw_density


# @gin.configurable()
class NeRFPP_TP(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        # num_fine_samples: int = 128,
        num_fine_samples: int = 64,
        use_viewdirs: bool = True,
        num_src_views: int = 3,
        density_noise: float = 0.0,
        lindisp: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFPP_TP, self).__init__()

        self.encoder = GridEncoder()
        self.rgb_activation = nn.Sigmoid()
        # self.sigma_activation = nn.ReLU()
        self.sigma_activation = nn.Softplus()
        
        self.obj_coarse_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view, out_nocs = True)
        self.obj_fine_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view, out_nocs = True)
        self.fg_coarse_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view)
        self.fg_fine_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view)
        self.bg_coarse_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view, input_ch=4)
        self.bg_fine_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view, input_ch=4)

    def encode(self, images, poses, focal, c):
        self.encoder(images, poses, focal, c)

    def forward(self, rays, randomized, white_bkgd, near, far, is_train=True):
        ret = []
        near = torch.full_like(rays["rays_o"][..., -1:], 1e-4)
        far = helper.intersect_sphere(rays["rays_o"], rays["rays_d"])

        #Supress the near_obj far_obj to only keep ones inside the bounding box
        # near_obj, far_obj = rays["near_obj"], rays["far_obj"]

        #Supress the near_obj far_obj to only keep ones inside the bounding box
        near_obj, far_obj = rays["near_obj"], rays["far_obj"]

        # Suppress rays outside of instance masks of objects for object MLP
        near_obj[~rays["instance_mask"]] = torch.zeros_like(near_obj[~rays["instance_mask"]])
        far_obj[~rays["instance_mask"]] = torch.zeros_like(far_obj[~rays["instance_mask"]])

        #Do not Supress the near_obj far_obj to only keep ones inside the bounding box but rather keep all
        # near_obj = near
        # far_obj = far

        for i_level in range(self.num_levels):
            if i_level == 0:
                obj_t_vals, obj_samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near_obj,
                    far=far_obj,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    in_sphere=True,
                )
                
                # near_insphere = near
                # far_insphere = far
                
                #supress the rays for near background MLP where bounding boxes esists
                # if is_train:
                #     near_insphere[rays["instance_mask"]] = torch.zeros_like(near_insphere[rays["instance_mask"]])
                #     far_insphere[rays["instance_mask"]] = torch.zeros_like(far_insphere[rays["instance_mask"]])

                fg_t_vals, fg_samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    # near=near_insphere,
                    # far=far_insphere,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    in_sphere=True,
                )
                bg_t_vals, bg_samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    in_sphere=False,
                )
                fg_mlp = self.fg_coarse_mlp
                bg_mlp = self.bg_coarse_mlp
                obj_mlp = self.obj_coarse_mlp

            else:
                obj_t_mids = 0.5 * (obj_t_vals[..., 1:] + obj_t_vals[..., :-1])
                obj_t_vals, obj_samples = helper.sample_pdf(
                    bins=obj_t_mids,
                    weights=obj_weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=obj_t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                    in_sphere=True,
                )

                fg_t_mids = 0.5 * (fg_t_vals[..., 1:] + fg_t_vals[..., :-1])
                fg_t_vals, fg_samples = helper.sample_pdf(
                    bins=fg_t_mids,
                    weights=fg_weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=fg_t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                    in_sphere=True,
                )
                bg_t_mids = 0.5 * (bg_t_vals[..., 1:] + bg_t_vals[..., :-1])
                bg_t_vals, bg_samples = helper.sample_pdf(
                    bins=bg_t_mids,
                    weights=bg_weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=bg_t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                    in_sphere=False,
                )

                fg_mlp = self.fg_fine_mlp
                bg_mlp = self.bg_fine_mlp
                obj_mlp = self.obj_fine_mlp
        
            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)

            def predict(samples, mlp, latent, out_nocs=False):
                samples_enc = helper.pos_enc(
                    samples,
                    self.min_deg_point,
                    self.max_deg_point,
                )
                if out_nocs:
                    raw_rgb, raw_sigma, raw_nocs, sem_logits = mlp(samples_enc, viewdirs_enc, latent, out_nocs=True)
                else:
                    raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc, latent)

                if self.density_noise != 0.0 and randomized:
                    raw_sigma = (
                        raw_sigma + torch.rand_like(raw_sigma) * self.density_noise
                    )

                rgb = self.rgb_activation(raw_rgb)
                sigma = self.sigma_activation(raw_sigma)
                if out_nocs:
                    nocs = self.rgb_activation(raw_nocs)
                    return rgb, sigma, nocs, sem_logits
                else:
                    return rgb, sigma

            # Get triplane features here

            B, N_samples, _ = obj_samples.shape
            latent_obj = self.encoder.index(obj_samples)
            latent_obj = latent_obj.squeeze(0).permute(1,0).reshape(B, N_samples, -1)

            B, N_samples, _ = fg_samples.shape
            latent_fg = self.encoder.index(fg_samples)
            latent_fg = latent_fg.squeeze(0).permute(1,0).reshape(B, N_samples, -1)

            B, N_samples, _ = bg_samples.shape
            latent_bg = self.encoder.index(bg_samples)
            latent_bg = latent_bg.squeeze(0).permute(1,0).reshape(B, N_samples, -1)

            # B,N_samples, _ = fg_samples.shape
            # uv_obj = obj_samples[:, :, [0, 2]]
            # uv_obj = uv_obj.reshape(-1,2).unsqueeze(0)
            # latent_obj = self.encoder.index(uv_obj)
            # latent_obj = latent_obj.squeeze(0).permute(1,0).reshape(B, N_samples, -1)

            # B,N_samples, _ = fg_samples.shape
            # uv_fg = fg_samples[:, :, [0, 2]]
            # uv_fg = uv_fg.reshape(-1,2).unsqueeze(0)
            # latent_fg = self.encoder.index(uv_fg)
            # latent_fg = latent_fg.squeeze(0).permute(1,0).reshape(B, N_samples, -1)

            # B, N_samples, _ = bg_samples.shape
            # uv_bg = bg_samples[:, :, [0, 2]]
            # uv_bg = uv_bg.reshape(-1,2).unsqueeze(0)
            # latent_bg = self.encoder.index(uv_bg)
            # latent_bg = latent_bg.squeeze(0).permute(1,0).reshape(B, N_samples, -1)

            #Get predictions for each MLP
            obj_rgb, obj_sigma, nocs_obj, sem_logits = predict(obj_samples, obj_mlp, latent_obj, out_nocs = True)
            fg_rgb, fg_sigma = predict(fg_samples, fg_mlp, latent_fg)
            bg_rgb, bg_sigma = predict(bg_samples, bg_mlp, latent_bg)


            obj_comp_rgb, obj_acc, obj_weights, bg_lambda_obj, obj_nocs, sem_map = helper.volumetric_rendering(
                obj_rgb,
                obj_sigma,
                obj_t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
                in_sphere=True,
                t_far=far_obj,
                sem_logits = sem_logits,
                nocs = nocs_obj
            )
            
            fg_comp_rgb, fg_acc, fg_weights, bg_lambda = helper.volumetric_rendering(
                fg_rgb,
                fg_sigma,
                fg_t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
                in_sphere=True,
                t_far=far,
            )
            bg_comp_rgb, bg_acc, bg_weights, _ = helper.volumetric_rendering(
                bg_rgb,
                bg_sigma,
                bg_t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
                in_sphere=False,
            )

            comp_rgb = obj_comp_rgb + fg_comp_rgb + bg_lambda * bg_comp_rgb
            ret.append((comp_rgb, fg_comp_rgb, bg_comp_rgb, obj_comp_rgb, fg_acc, bg_acc, obj_acc, obj_nocs, sem_map))

        return ret


# @gin.configurable()
class LitNeRFPP_CO_TP_NOCS(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
        grad_max_norm: float = 0.05
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        super(LitNeRFPP_CO_TP_NOCS, self).__init__()
        self.model = NeRFPP_TP()

    def setup(self, stage: Optional[str] = None) -> None:

        dataset = dataset_dict[self.hparams.dataset_name]
        
        if self.hparams.dataset_name == 'pd' or self.hparams.dataset_name == 'pd_multi_obj' or self.hparams.dataset_name =='pd_multi_obj_ae' or self.hparams.dataset_name =='pd_multi_obj_ae_nocs':
            kwargs_train = {'root_dir': self.hparams.root_dir,
                             'img_wh': tuple(self.hparams.img_wh),
                                'white_back': self.hparams.white_back,
                                'model_type': 'nerfpp'}
            kwargs_val = {'root_dir': self.hparams.root_dir,
                            'img_wh': tuple(self.hparams.img_wh),
                                'white_back': self.hparams.white_back,
                                'model_type': 'nerfpp'}

        if self.hparams.run_eval:        
            kwargs_test = {'root_dir': self.hparams.root_dir,
                            'img_wh': tuple(self.hparams.img_wh),
                            'white_back': self.hparams.white_back}
            self.test_dataset = dataset(split='val', eval_inference=True, **kwargs_test)
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
        
        eps = 1e-6
        for k,v in batch.items():
            batch[k] = v.squeeze(0)

        self.model.encode(batch["src_imgs"], batch["src_poses"], batch["src_focal"], batch["src_c"])

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )

        obj_rgb_coarse = rendered_results[0][3]
        obj_rgb_fine = rendered_results[1][3]

        nocs_coarse = rendered_results[0][7]
        nocs_fine = rendered_results[1][7]

        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]

        target = batch["target"]
        target_nocs = batch["nocs_2d"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
        loss = loss1 + loss0

        if loss.isnan(): loss=eps
        else: loss = loss

        mask = batch["instance_mask"].view(-1, 1).repeat(1, 3)
        loss2 = helper.img2mse(obj_rgb_coarse[mask], target[mask])
        loss3 = helper.img2mse(obj_rgb_fine[mask], target[mask])
        masked_rgb_loss = (loss2 + loss3)

        if masked_rgb_loss.isnan(): masked_rgb_loss=eps
        else: masked_rgb_loss = masked_rgb_loss

        loss4 = helper.img2mse(nocs_coarse[mask], target_nocs[mask])
        loss5 = helper.img2mse(nocs_fine[mask], target_nocs[mask])
        masked_nocs_loss = (loss4 + loss5)

        if masked_nocs_loss.isnan(): masked_nocs_loss=eps
        else: masked_nocs_loss = masked_nocs_loss


        self.log("train/masked_rgb_loss", masked_rgb_loss, on_step=True)
        self.log("train/masked_nocs_loss", masked_nocs_loss, on_step=True)
        # loss += masked_rgb_loss

        loss += masked_rgb_loss
        loss += masked_nocs_loss

        #opacity loss
        opacity_loss = self.opacity_loss_CE(
                rendered_results, batch["instance_mask"].view(-1)
            )  
        if opacity_loss.isnan(): opacity_loss=eps
        else: opacity_loss = opacity_loss

        # We might not need opacity loss if we are supplying object masks
        self.log("train/sem_map_loss", opacity_loss, on_step=True)
        loss += opacity_loss

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)
        self.log("train/lr", helper.get_learning_rate(self.optimizers()))
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

            # do not suppress rays for near background mlp in validation since we don't have masks in inference time                 
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far, is_train=False
            )
            #here 1 denotes fine
            ret["comp_rgb"]+=[rendered_results_chunk[1][0]]
            ret["fg_rgb"] +=[rendered_results_chunk[1][1]]
            ret["bg_rgb"] +=[rendered_results_chunk[1][2]]
            ret["obj_rgb"] +=[rendered_results_chunk[1][3]]
            ret["obj_acc"] +=[rendered_results_chunk[1][6]]
            ret["nocs"] +=[rendered_results_chunk[1][7]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)
        psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        ssim_ = self.ssim_legacy(ret["comp_rgb"], batch["target"]).mean()
        lpips_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        self.log("val/psnr", psnr_.item(), on_step=True, prog_bar=True, logger=True)
        return ret

    # def render_rays(self, batch, batch_idx):
    #     ret = {}
    #     rendered_results = self.model(
    #         batch, False, self.white_bkgd, self.near, self.far
    #     )
    #     rgb_fine = rendered_results[1][0]
    #     target = batch["target"]
    #     ret["target"] = target
    #     ret["rgb"] = rgb_fine
    #     return ret

    def on_validation_start(self):
        self.random_batch = np.random.randint(5, size=1)[0]

    def validation_step(self, batch, batch_idx):
        for k,v in batch.items():
            batch[k] = v.squeeze()
            if k =='radii':
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k== "far_obj":
                batch[k] = batch[k].unsqueeze(-1)

        for k,v in batch.items():
            print(k,v.shape)

        self.model.encode(batch["src_imgs"], batch["src_poses"], batch["src_focal"], batch["src_c"])
        W,H = self.hparams.img_wh
        ret = self.render_rays(batch)
        rank = dist.get_rank()
        # rank =0
        if rank==0:
            if batch_idx == self.random_batch:
                grid_img = visualize_val_fb_bg_rgb_opacity_nocs(
                    (W,H), batch, ret
                )
                self.logger.experiment.log({
                    "val/GT_pred rgb": wandb.Image(grid_img)
                })

    def test_step(self, batch, batch_idx):
        for k,v in batch.items():
            print(k,v.shape)
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

        if self.grad_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_max_norm)

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
                        batch_size=1,
                        pin_memory=True)

    # def validation_epoch_end(self, outputs):
    #     val_image_sizes = self.val_dataset.val_image_sizes
    #     rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
    #     targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
    #     psnr_mean = self.psnr_each(rgbs, targets).mean()
    #     ssim_mean = self.ssim_each(rgbs, targets).mean()
    #     lpips_mean = self.lpips_each(rgbs, targets).mean()
    #     self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        # dmodule = self.trainer.datamodule
        # all_image_sizes = (
        #     dmodule.all_image_sizes
        #     if not dmodule.eval_test_only
        #     else dmodule.test_image_sizes
        # )
        all_image_sizes = self.test_dataset.image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        # psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # lpips = self.lpips(
        #     rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        # )
        psnr = self.psnr(rgbs, targets, None, None, None)
        ssim = self.ssim(rgbs, targets, None, None, None)
        lpips = self.lpips(
            rgbs, targets, None, None, None
        )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join("ckpts",self.hparams.exp_name, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image(image_dir, rgbs)

            result_path = os.path.join("ckpts",self.hparams.exp_name, "results.json")
            write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips

    def opacity_loss(self, rendered_results, instance_mask):
        opacity_lambda = 0.1
        criterion = nn.MSELoss(reduction="none")
        loss = (
            criterion(
                torch.clamp(rendered_results[0][6], 0, 1),
                instance_mask.float(),
            )
        ).mean()
        loss += (
            criterion(
                torch.clamp(rendered_results[1][6], 0, 1),
                instance_mask.float(),
            )
        ).mean()  
        #
        return loss*opacity_lambda

    def opacity_loss_CE(self, rendered_results, instance_mask):
        criterion = nn.BCEWithLogitsLoss()
        loss = (
            criterion(
                torch.clamp(rendered_results[0][8], 0, 1),
                instance_mask.float(),
            )
        ).mean()
        loss += (
            criterion(
                torch.clamp(rendered_results[1][8], 0, 1),
                instance_mask.float(),
            )
        ).mean()  
        #
        # return loss*opacity_lambda
        return loss