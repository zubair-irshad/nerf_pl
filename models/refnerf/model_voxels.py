# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Ref-NeRF (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict
from einops import rearrange
import sys
sys.path.append('/home/ubuntu/nerf_pl')

import wandb
import models.refnerf.helper as helper
import models.refnerf.ref_utils as ref_utils
from models.utils import store_image, write_stats
from models.interface import LitModel
from models.code_library import *
from datasets import dataset_dict
from utils.train_helper import *

# class Voxels(nn.Module):
#     """A voxel based radiance field model."""

#     def __init__(self, side: int, scale: float):
#         """Constructor.

#         Args:
#             side (int): The number of voxels on one side of a cube.
#             scale (float): The scale of the voxel volume, equivalent
#                            to half of one side of the volume, i.e. a
#                            scale of 1 indicates a volume of size 2x2x2.
#         """
#         nn.Module.__init__(self)
#         self.params = {
#             "side": side,
#             "scale": scale
#         }

#         voxels = torch.zeros((1, 1, side, side, side), dtype=torch.float32)
#         # voxels = torch.zeros((1, 4, side, side, side), dtype=torch.float32)
#         self.voxels = nn.Parameter(voxels)

#         # bias = torch.zeros(4, dtype=torch.float32)
#         # bias[:3] = torch.logit(torch.FloatTensor([1e-5, 1e-5, 1e-5]))
#         # bias[3] = -2
#         bias = torch.zeros(1, dtype=torch.float32)
#         bias[0] = -2
#         self.bias = nn.Parameter(bias.unsqueeze(0))
#         self.scale = scale
#         self.use_view = False

#     def forward(self, positions: torch.Tensor) -> torch.Tensor:
#         """Interpolates the positions within the voxel volume."""
#         print("positions", positions.shape, torch.max(positions), torch.min(positions))
#         positions = positions.reshape(1, -1, 1, 1, 3)
#         positions = positions / self.scale
#         output = F.grid_sample(self.voxels, positions,
#                                padding_mode="border", align_corners=False)
        
#         print("output", output.shape)
#         output = output.transpose(1, 2)
#         print("output", output.shape)
#         output = output.reshape(-1, 1)
#         # output = output.reshape(-1, 4)
#         print("output", output.shape)
#         output = output + self.bias
#         print("output", output.shape)
#         assert not output.isnan().any()
#         return output

class Generator3D(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max, train_opacity_rgb):
        super(Generator3D, self).__init__()
        self.z_size = 128       # options.z_size
        self.bias = False       # options.bias
        self.voxel_size = 64  # options.voxel_size
        padd = (1, 1, 1)
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.train_opacity_rgb = train_opacity_rgb
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.z_size, self.voxel_size * 16, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.BatchNorm3d(self.voxel_size * 16),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 16, self.voxel_size *8, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 8),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 8, self.voxel_size * 4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 4),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 4, self.voxel_size* 2, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 2),
            torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 2, self.voxel_size, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size),
            torch.nn.ReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size, int(self.voxel_size/2), kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(int(self.voxel_size/2)),
            torch.nn.ReLU())
        # self.layer7 = torch.nn.Sequential(
        #     torch.nn.ConvTranspose3d(int(self.voxel_size/2), 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
        #     torch.nn.Sigmoid())
        if self.train_opacity_rgb:
            self.layer7 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(int(self.voxel_size/2), 4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)))
        else:    
            self.layer7 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(int(self.voxel_size/2), 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)))

    def forward(self, x, z):
        out = z.view(-1, self.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        # extract density from voxels using trilinear interpolation
        x = x.reshape(1, -1, 1, 1, 3)
        self.xyz_min = self.xyz_min.to(x.device)
        self.xyz_max = self.xyz_max.to(x.device)
        ind_norm = (
            (x - self.xyz_min)
            / (self.xyz_max - self.xyz_min)
        ).flip((-1,)) * 2 - 1
        out = F.grid_sample(out, ind_norm.float(),
                                padding_mode="border", align_corners=False)
        return out

# @gin.configurable()
class GeneratorMLP(nn.Module):
    def __init__(
        self,
        xyz_min,
        xyz_max,
        train_opacity_rgb
        # tint_activation: Callable[..., Any] = nn.Sigmoid(),
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(GeneratorMLP, self).__init__()

        self.generator_3d = Generator3D(xyz_min = xyz_min, xyz_max = xyz_max, train_opacity_rgb=train_opacity_rgb)
        # self.generator_3d = Voxels(128,1)
    """
    x: torch.Tensor, [batch, num_samples, feature]
    viewdirs: torch.Tensor, [batch, viewdirs]
    """

    def forward(self, samples, shape_latent= None):    
        num_samples, feat_dim = samples.shape[1:]
        samples = samples.reshape(1, -1, 1, 1, feat_dim)
        if self.train_opacity_rgb:
            color_o = self.generator_3d(samples, shape_latent)
            color_o = color_o.transpose(1, 2)
            color_o = color_o.reshape(-1, num_samples, 4)
            color, opacity = torch.split(color_o, [3, 1], -1)
            return dict(
                rgb=color,
                density=opacity
            )
            
        else:
            density = self.generator_3d(samples, shape_latent)
            density = density.reshape(-1, num_samples, 1)
            return dict(
                density=density
            )
class GeneratorModel(nn.Module):
    def __init__(
        self,
        xyz_min,
        xyz_max,
        num_levels = 1,
        lindisp: bool = False,
        train_opacity_rgb = False,
        num_coarse_samples: int = 96,
        num_fine_samples: int = 64
    ):
        # Layers
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(GeneratorModel, self).__init__()
        print("train_opacity_rgb", train_opacity_rgb)
        self.mlp = GeneratorMLP(xyz_min = xyz_min, xyz_max = xyz_max, train_opacity_rgb = train_opacity_rgb)

    def forward(self, rays, randomized, white_bkgd, near, far, embedding_instance=None):
        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays_vanilla(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp
                )
            ray_results = self.mlp(samples, shape_latent = embedding_instance)
            if self.train_opacity_rgb:
                rgb = torch.sigmoid(ray_results["rgb"])
                desnity = F.softplus(ray_results["density"])
                comp_rgb, acc = helper.volumetric_rendering_rgb(
                    rgb,
                    desnity,
                    t_vals,
                )
                rendered_result = ray_results
                rendered_result["comp_rgb"] = comp_rgb
                rendered_result["acc"] = acc
                ret.append(rendered_result)
            else:
                # if train_only_opacity:
                desnity = F.softplus(ray_results["density"])                
                acc, weights =  helper.volumetric_rendering_opacity(
                        desnity,
                        t_vals
                    )
                rendered_result = ray_results
                rendered_result["acc"] = acc
                rendered_result["weights"] = weights
                ret.append(rendered_result)
        return ret


# @gin.configurable()
class LitVoxelGenerator(LitModel):
    def __init__(
        self,
        hparams,
        # lr_init: float = 5.0e-4,
        lr_init: float = 1e-3,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        coarse_loss_mult: float = 0.1,
        randomized: bool = True,
        orientation_loss_mult: float = 0.1,
        orientation_coarse_loss_mult: float = 0.01,
        predicted_normal_loss_mult: float = 3e-4,
        predicted_normal_coarse_loss_mult: float = 3e-5,
        compute_normal_metrics: bool = False,
        grad_max_norm: float = 0.001,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        self.automatic_optimization = False

        super(LitVoxelGenerator, self).__init__()

    def setup(self, stage: Optional[str] = None) -> None:

        dataset = dataset_dict[self.hparams.dataset_name]
        
        if self.hparams.dataset_name == 'pd' or self.hparams.dataset_name == 'pd_multi':
            kwargs_train = {'root_dir': self.hparams.root_dir,
                      'img_wh': tuple(self.hparams.img_wh),
                      'white_back': self.hparams.white_back,
                      'model_type': 'refnerf'}
            kwargs_val = {'root_dir': self.hparams.root_dir,
                      #'img_wh': tuple((int(self.hparams.img_wh[0]/8),int(self.hparams.img_wh[1]/8))),
                      'img_wh': tuple(self.hparams.img_wh),
                      'white_back': self.hparams.white_back,
                      'model_type': 'refnerf'}

        if self.hparams.run_eval:        
            kwargs_test = {'root_dir': self.hparams.root_dir,
                            'img_wh': tuple(self.hparams.img_wh),
                            'white_back': self.hparams.white_back}
            self.test_dataset = dataset(split='test', **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            self.train_dataset = dataset(split='train', **kwargs_train)
            self.val_dataset = dataset(split='val', **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back

        xyz_min = torch.from_numpy(self.train_dataset.xyz_min)
        xyz_max = torch.from_numpy(self.train_dataset.xyz_max)
        self.model = GeneratorModel(xyz_min = xyz_min, xyz_max = xyz_max, train_opacity_rgb = self.hparams.train_opacity_rgb)
        self.code_library = CodeLibraryVoxel(self.hparams)
        self.models_to_train = [self.model,
                                self.code_library]

    # Batched backprop for all 150 instances
    def training_step(self, batch, batch_idx):
        loss_all = []
        loss_all_rgb = []
        loss_all_opacity = []
        indices = torch.randperm(batch["target"].shape[1])
        for k,v in batch.items():
            batch[k] = batch[k][:, indices]
            if len(batch[k].size()) ==3:
                batch[k] = rearrange(v, 'b n c -> (b n) c')
            else:
                batch[k] = rearrange(v, 'b n -> (b n)')
        B = batch["rays_o"].shape[0]
        for i in range(0, B, self.hparams.chunk):
            extra_info = dict()
            extra_info.update(self.code_library(batch["instance_ids"][i : i + self.hparams.chunk][0]))
            batch_chunk = dict()
            for k, v in batch.items():
                batch_chunk[k] = v[i : i + self.hparams.chunk]

            rendered_results = self.model(
                batch_chunk, self.randomized, self.white_bkgd, self.near, self.far, **extra_info
            )

            opacity_coarse = rendered_results[0]["acc"]
            criterion = nn.MSELoss(reduction="none")
            instance_mask = batch_chunk["instance_mask"].view(-1)        


            opt = self.optimizers()
            opt.zero_grad()

            #manually backprop the losses
            loss_opacity = (criterion(torch.clamp(opacity_coarse, 0, 1), instance_mask.float())* batch_chunk["instance_mask_weight"]).mean()*10.0
            if self.hparams.train_opacity_rgb:
                rgb_coarse = rendered_results[0]["comp_rgb"]
                target = batch_chunk["target"]
                loss_rgb = helper.img2mse(rgb_coarse, target)
                loss = loss_rgb + loss_opacity
            else:
                loss = loss_opacity
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            opt.step()
            loss_all.append(loss.item())
            if self.hparams.train_opacity_rgb:
                loss_all_rgb.append(loss_rgb.item())
                loss_all_opacity.append(loss_opacity.item())

        #log losses
        self.log("train/loss", np.mean(loss_all), on_step=True, prog_bar=True, logger=True)
        self.log("train/lr", helper.get_learning_rate(self.optimizer))
        if self.hparams.train_opacity_rgb:
            self.log("train/loss_rgb", np.mean(loss_all_rgb), on_step=True)
            self.log("train/loss_opacity", np.mean(loss_all_opacity), on_step=True)

        return loss

    # # Batched backprop for all 150 instances
    # def training_step(self, batch, batch_idx):
    #     loss_all = []
    #     extra_info = dict()
    #     extra_info.update(self.code_library(batch))
    #     B = batch["rays_o"].shape[0]
    #     indices = torch.randperm(batch["target"].shape[1])
    #     for k,v in batch.items():
    #         batch[k] = batch[k].squeeze(0)[indices]
        
    #     for i in range(0, B, self.hparams.chunk):
    #         batch_chunk = dict()
    #         for k, v in batch.items():
    #             batch_chunk[k] = v[i : i + self.hparams.chunk]
    #         rendered_results = self.model(
    #             batch_chunk, self.randomized, self.white_bkgd, self.near, self.far, **extra_info
    #         )
    #         opacity_coarse = rendered_results[0]["acc"]
    #         criterion = nn.MSELoss(reduction="none")
    #         instance_mask = batch_chunk["instance_mask"].view(-1)

    #         opt = self.optimizers()
    #         opt.zero_grad()
    #         loss = (criterion(torch.clamp(opacity_coarse, 0, 1), instance_mask.float())* batch_chunk["instance_mask_weight"]).mean()*10.0
    #         self.manual_backward(loss)
    #         torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
    #         opt.step()
    #         loss_all.append(loss.item())

    #     self.log("train/loss", loss.mean(), on_step=True, prog_bar=True, logger=True)
    #     self.log("train/lr", helper.get_learning_rate(self.optimizer))
    #     return loss

    def training_epoch_end(self, training_step_outputs):
        sch = self.lr_schedulers()
        sch.step() 

    # def training_step(self, batch, batch_idx):
    #     extra_info = dict()
    #     extra_info.update(self.code_library(batch))
    #     rendered_results = self.model(
    #         batch, self.randomized, self.white_bkgd, self.near, self.far, **extra_info
    #     )
    #     opacity_coarse = rendered_results[0]["acc"]
    #     # opacity_fine = rendered_results[1]["acc"]
    #     criterion = nn.MSELoss(reduction="none")
    #     instance_mask = batch["instance_mask"].view(-1)
    #     loss = (criterion(torch.clamp(opacity_coarse, 0, 1), instance_mask.float())* batch["instance_mask_weight"]).mean()*10.0
    #     # loss1 = loss(torch.clamp(opacity_fine, 0, 1), instance_mask.float()).mean()
    #     # loss = loss1 + loss0 * self.coarse_loss_mult
    #     self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
    #     # self.log("train/loss1", loss1, on_step=True, prog_bar=True, logger=True)
    #     # self.log("train/loss", loss, on_step=True)
    #     #both opacity and color
    #     # else:
    #     #     rendered_results = self.model(
    #     #         batch, self.randomized, self.white_bkgd, self.near, self.far
    #     #     )
    #     #     rgb_coarse = rendered_results[0]["comp_rgb"]
    #     #     target = batch["target"]
    #     #     # loss = helper.img2mse(rgb_coarse, target)

    #     #     loss_rgb = helper.img2mse(rgb_coarse, target)
    #     #     opacity_coarse = rendered_results[0]["acc"]
    #     #     criterion = nn.MSELoss(reduction="none")
    #     #     instance_mask = batch["instance_mask"].view(-1)
    #     #     loss_opacity = (criterion(torch.clamp(opacity_coarse, 0, 1), instance_mask.float())* batch["instance_mask_weight"]).mean()*10.0
    #     #     loss = loss_rgb + loss_opacity

    #     #     psnr = helper.mse2psnr(loss_rgb)

    #     #     self.log("train/loss_rgb", loss_rgb, on_step=True)
    #     #     self.log("train/loss_opacity", loss_opacity, on_step=True, prog_bar=True, logger=True)
    #     #     self.log("train/psnr", psnr, on_step=True, prog_bar=True, logger=True)

    #     #     self.log("train/loss", loss, on_step=True)
    #     return loss

    def render_rays(self, batch, extra=None):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                batch_chunk[k] = v[i : i + self.hparams.chunk]
                                    
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far, **extra
            )
            # rendered_results_chunk = self.model(
            #     batch_chunk, False, self.white_bkgd, self.near, self.far
            # )
            #here 1 denotes fine
            for k, v in rendered_results_chunk[0].items():
                ret[k] += [v]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)
            # opacity = rendered_results[1]["acc"]
            # target_mask = batch["instance_mask"]
            # ret["opacity"] = opacity
            # ret["target_mask"] = target_mask
        
        # psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        # self.log("val/psnr", psnr_.item(), on_epoch=True, sync_dist=True)
        self.log("val/psnr", 1.0, on_epoch=True, sync_dist=True)
        return ret

    def on_validation_start(self):
        self.random_batch = np.random.randint(5, size=1)[0]
        #self.random_batch = 0

    def validation_step(self, batch, batch_idx):
        for k,v in batch.items():
            batch[k] = v.squeeze()
            if k =='radii':
                batch[k] = v.unsqueeze(-1)
        W,H = self.hparams.img_wh
        extra_info = dict()
        extra_info.update(self.code_library(batch["instance_ids"][0]))
        
        ret = self.render_rays(batch, extra_info)
        # ret = self.render_rays(batch, batch_idx)
        print("random_batch", self.random_batch)
        if batch_idx == self.random_batch:
            if self.hparams.train_opacity_rgb:
                grid_img = visualize_val_rgb_opacity(
                    (W,H), batch, ret
                )
                self.logger.experiment.log({
                    "val/GT_pred rgb": wandb.Image(grid_img)
                })
            else:
                grid_img = visualize_val_opacity(
                    (W,H), batch, ret
                )
                self.logger.experiment.log({
                    "val/GT_pred opacity": wandb.Image(grid_img)
                })

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        parameters = helper.get_parameters(self.models_to_train)
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr_init, betas=(0.9, 0.999))
        scheduler = LambdaLR(self.optimizer, lambda epoch: (1-epoch/(self.hparams.num_epochs))**self.hparams.poly_exp)
        return [self.optimizer], [scheduler]

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu,
    #     using_native_amp,
    #     using_lbfgs,
    # ):
    #     step = self.trainer.global_step
    #     max_steps = self.hparams.run_max_steps

    #     if self.lr_delay_steps > 0:
    #         delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
    #             0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
    #         )
    #     else:
    #         delay_rate = 1.0

    #     t = np.clip(step / max_steps, 0, 1)
    #     scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
    #     new_lr = delay_rate * scaled_lr

    #     for pg in optimizer.param_groups:
    #         pg["lr"] = new_lr
        
    #     if self.grad_max_norm > 0:
    #         parameters = helper.get_parameters(self.models_to_train)
    #         nn.utils.clip_grad_norm_(parameters, self.grad_max_norm)

    #     optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        # return DataLoader(self.train_dataset,
        #                   shuffle=False,
        #                   num_workers=4,
        #                   batch_size=self.hparams.batch_size,
        #                   pin_memory=True)
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=16,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                        shuffle=False,
                        num_workers=4,
                        batch_size=self.hparams.batch_size,
                        pin_memory=True)

    def test_epoch_end(self, outputs):
        all_image_sizes = self.test_dataset.image_sizes
        # dmodule = self.trainer.datamodule
        # all_image_sizes = (
        #     dmodule.all_image_sizes
        #     if not dmodule.eval_test_only
        #     else dmodule.test_image_sizes
        # )
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
            self.write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips

if __name__ == '__main__':
    # voxel_ckpt_path = '/home/ubuntu/nerf_pl/ckpts/voxel_generator_singlelatent/last.ckpt'
    
    # voxel_gen = GeneratorMLP(xyz_min = [-2,-2,-2], xyz_max = [2,2,2])
    # helper.load_ckpt(voxel_gen, voxel_ckpt_path, model_name='nerf_coarse')
    # model, code_library = helper.load_model(voxel_ckpt_path)

    x =  torch.randn((1024,129,3))
    a = torch.randn((1,128))
    xyz_min = torch.from_numpy(np.array([-2,-2,-2]))
    xyz_max = torch.from_numpy(np.array([2,2,2]))

    generator = Generator3D(xyz_min = xyz_min, xyz_max=xyz_max, train_opacity_rgb=True)

    out = generator(x,a)

    print(out.shape)
    