import os

from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
# from losses import loss_dict, get_background_loss
from losses import loss_dict

# metrics
from metrics import *

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

        self.loss = loss_dict['color'](coef=1)
        # self.loss = get_background_loss(hparams)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        # self.embeddings = None

        self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        # self.nerf_coarse = NeRF_TCNN(
        #                         encoding="hashgrid",
        #                     )
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            # self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
            #                       in_channels_dir=6*hparams.N_emb_dir+3)
            
            self.nerf_fine =  NeRF_TCNN(
                                encoding="hashgrid",
                            )
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        print("B", B)
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            print("=========================\n\n\n")
            print("i", i)
            print("i+self.hparams.chunk")
            print("rayssssssss", rays.shape)
            print("self.hparams.chunk", self.hparams.chunk)
            print("rays[i:i+self.hparams.chunk]", rays[i:i+self.hparams.chunk].shape)
            print("=========================\n\n\n")
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
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        if self.hparams.dataset_name == 'llff' or self.hparams.dataset_name == 'llff_nocs' or self.hparams.dataset_name =='nocs_bckg' or self.hparams.dataset_name=='llff_nsff':
            kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        elif self.hparams.dataset_name == 'co3d':
            kwargs = {'data_dir': self.hparams.root_dir}
            kwargs['category'] = 'laptop'
            kwargs['instance'] = '112_13277_23636'
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        # self.optimizer = get_optimizer_tcnn(self.hparams, self.models)
        # scheduler = get_scheduler_tcnn(self.hparams, self.optimizer)
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        for k,v in batch.items():
            print(k,v.shape)
        results = self(rays)
        loss = self.loss(results, batch)
        # loss_sum, loss_dict = self.loss(results, batch)
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)

        # for k, v in loss_dict.items():
        #     self.log(f"train/{k}", v)

        self.log('train/psnr', psnr_, prog_bar=True)
        return loss
        # return loss_sum

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        for k,v in batch.items():
            if k =='img_wh':
                continue
            print(k,v.shape)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)

        batch['rgbs'] = batch['rgbs'].squeeze()
        # batch['fused_depth'] = batch['fused_depth'].squeeze()
        # loss_sum, loss_dict = self.loss(results, batch) 
        # log = {"val_loss": loss_sum}
        # log.update(loss_dict)
        log = {'val_loss': self.loss(results, batch)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            H, W = batch['img_wh']
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            print("img gt deth before",img_gt.shape, depth.shape)

            if self.hparams.dataset_name == 'objectron':
                 img_gt = torch.rot90(img_gt.permute(1,2,0), dims=(1, 0)).permute(2,0,1)
                 depth = torch.rot90(depth.permute(1,2,0), dims=(1, 0)).permute(2,0,1)
                 img = torch.rot90(img.permute(1,2,0), dims=(1, 0)).permute(2,0,1)

            images = {"gt":img_gt, "predicted": img, "depth": depth}
            self.logger.experiment.log({
                "Val images": [wandb.Image(img, caption=caption)
                for caption, img in images.items()]
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
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    # logger = TensorBoardLogger(save_dir="logs",
    #                            name=hparams.exp_name,
    #                            default_hp_metric=False)
    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=wandb_logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)

