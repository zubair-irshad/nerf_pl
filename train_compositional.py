import os

from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nerf import *
# from models.rendering_compositional import *
from models.rendering_compositional_combined import *

# from models.rendering_compositional_symmetric import *
from models.code_library import *
from utils.train_helper import visualize_val_image
# optimizer, scheduler, visualization
from utils import *

# losses
from losses import get_loss, get_sym_loss

# metrics
from metrics import *
from dotmap import DotMap
# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import wandb
from pytorch_lightning.loggers import WandbLogger

# wandb_logger = WandbLogger()

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print("hparams", hparams)
        #only do this for loading checkpoint/inference
        if type(hparams) is dict:
            hparams = DotMap(hparams)
        # self.loss = loss_dict['color'](coef=1)
        self.loss = get_loss(hparams)
        # self.loss = get_sym_loss(hparams)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        # self.nerf_coarse = ObjectNeRF(hparams)
        # self.nerf_coarse = ObjectBckgNeRFConditional(hparams)
        self.nerf_coarse = ObjectBckgNeRFGSN(hparams)
        self.models = {'coarse': self.nerf_coarse}
        # load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            # self.nerf_fine = ObjectNeRF(hparams)
            self.nerf_fine = ObjectBckgNeRFGSN(hparams)
            self.models['fine'] = self.nerf_fine
            # load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

        # self.code_library = CodeLibrary(hparams)
        self.code_library = CodeLibraryBckgObjShapeApp(hparams)

        self.models_to_train = [
            self.models,
            self.code_library,
            self.embedding_xyz,
        ]

    def forward(self, rays, extra=dict()):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            extra_chunk = dict()
            for k, v in extra.items():
                if k == 'embedding_backgrounds':
                    extra_chunk[k] = v
                elif isinstance(v, torch.Tensor):
                    extra_chunk[k] = v[i : i + self.hparams.chunk]
                else:
                    extra_chunk[k] = v

            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            **extra_chunk)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        if self.hparams.dataset_name == 'llff' or self.hparams.dataset_name == 'llff_nocs':
            kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        elif self.hparams.dataset_name == 'objectron' or self.hparams.dataset_name=='pd' or self.hparams.dataset_name=='pdmultiobject':
            kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        elif self.hparams.dataset_name == 'co3d':
            kwargs = {'data_dir': self.hparams.root_dir}
            kwargs['category'] = 'car'
            kwargs['instance'] = '106_12662_23043'
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=6,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        for k,v in batch.items():
            print(k,v.shape)
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)

        extra_info = dict()
        extra_info["is_eval"] = False
        # extra_info["instance_mask"] = batch["instance_mask"]
        extra_info["rays_in_bbox"] = False
        extra_info["frustum_bound_th"] = -1
        extra_info.update(self.code_library(batch))

        results = self(rays, extra_info)
        loss_sum, loss_dict = self.loss(results, batch)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss_sum)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss_sum

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        for k,v in batch.items():
            if torch. is_tensor(v):
                print(k,v.shape)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)

        extra_info = dict()
        extra_info["is_eval"] = True
        extra_info["rays_in_bbox"] = False
        extra_info["frustum_bound_th"] = -1
        extra_info.update(self.code_library(batch))

        for k,v in extra_info.items():
            if torch. is_tensor(v):
                print(k,v.shape)
        results = self(rays, extra_info)
        loss_sum, loss_dict = self.loss(results, batch)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v)
        log = {"val_loss": loss_sum}
        log.update(loss_dict)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = batch['img_wh']

            grid_img = visualize_val_image(
                (W,H), batch, results, typ=typ
            )
            self.logger.experiment.log({
                "val/GT_pred images": wandb.Image(grid_img)
            })
            
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'ckpts/{hparams.exp_name}',
        filename="{epoch:d}",
        monitor="val/psnr",
        mode="max",
        # save_top_k=5,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, pbar]
    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=wandb_logger,
                      enable_model_summary=False,
                      gpus=hparams.num_gpus,
                      accelerator="gpu" if hparams.num_gpus > 1 else "auto",
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                    #   val_check_interval=0.75,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)
    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)

