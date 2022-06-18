from multiprocessing import Condition
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
from losses import loss_dict

# metrics
from metrics import *

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # set to DETAIL for runtime logging.

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import wandb
from pytorch_lightning.loggers import WandbLogger
import math

# wandb_logger = WandbLogger()


def save_latent_vectors(shape_latent, texture_latent, save_dir, epoch):
    save_dict = { 'shape_code_params': shape_latent.state_dict(),
                  'texture_code_params': texture_latent.state_dict()}
    torch.save(save_dict, os.path.join(save_dir, str(epoch) + '.pth'))
    
def make_codes(embdim, d):
    shape_codes = nn.Embedding(d, embdim)
    texture_codes = nn.Embedding(d, embdim)
    shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
    texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
    return shape_codes, texture_codes

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        # self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
        #                         in_channels_dir=6*hparams.N_emb_dir+3)

        # self.nerf_coarse = CodeNeRF()

        self.nerf_coarse = ConditionalNeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                            in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            # self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
            #                       in_channels_dir=6*hparams.N_emb_dir+3)
            # self.nerf_fine = CodeNeRF()
            self.nerf_fine = ConditionalNeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                            in_channels_dir=6*hparams.N_emb_dir+3)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')
        self.shape_codes, self.texture_codes = make_codes(hparams.latent_dim, hparams.emb_dim)
        # self.automatic_optimization = False
        self.log_frequency = 10

    def forward(self, rays, shape_codes, texture_codes):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays_conditional(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            shape_codes,
                            texture_codes,
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
        # kwargs = {'root_dir': self.hparams.root_dir,
        #           'img_wh': tuple(self.hparams.img_wh)}
        # if self.hparams.dataset_name == 'llff' or self.hparams.dataset_name == 'llff_nocs':
        #     kwargs['spheric_poses'] = self.hparams.spheric_poses
        #     kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(splits=self.hparams.splits, cat = self.hparams.cat,img_wh = self.hparams.img_wh)
        self.val_dataset = dataset(splits=self.hparams.splits, cat = self.hparams.cat, img_wh = self.hparams.img_wh)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        self.optimizer_latent = get_optimizer_latent(hparams, self.shape_codes, self.texture_codes)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        scheduler_latent = get_scheduler(self.hparams, self.optimizer_latent)
        return [self.optimizer, self.optimizer_latent], [scheduler, scheduler_latent]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                        #   batch_size=self.hparams.batch_size,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb, optimizer_idx):
        rays, rgbs, obj_idx = batch['rays'], batch['rgbs'], batch['obj_id']
        rays = rays.squeeze(0)
        rgbs = rgbs.squeeze(0)

        results = self(rays, self.shape_codes(obj_idx), self.texture_codes(obj_idx))
        loss_img = self.loss(results, rgbs)

        reg_loss = torch.norm(self.shape_codes(obj_idx), dim=-1) + torch.norm(self.texture_codes(obj_idx), dim=-1)
        loss_reg = 1e-4 * torch.mean(reg_loss)
        loss = loss_img + loss_reg
            
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss_img', loss_img)
        self.log('train/loss_latent', loss_reg)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss
    
    # def training_step(self, batch, batch_nb):
    #     rays, rgbs, obj_idx = batch['rays'], batch['rgbs'], batch['obj_id']
    #     rays = rays.squeeze(0)
    #     rgbs = rgbs.squeeze(0)
    #     loss_img_all = []
    #     loss_latent_all = []
    #     generated_img = []
    #     gt_img = []
    #     psnr_all = []

    #     for i in range(0, rays.shape[0], self.hparams.batch_size):
    #         shape_codes = self.shape_codes(obj_idx)
    #         texture_codes = self.texture_codes(obj_idx)
    #         results = self(rays[i:i+self.hparams.batch_size], shape_codes, texture_codes)
    #         loss_img = self.loss(results, rgbs[i:i+self.hparams.batch_size])

    #         typ = 'fine' if 'rgb_fine' in results else 'coarse'

    #         if batch_nb == 0:
    #             W, H = self.hparams.img_wh
    #             generated_img.append(results[f'rgb_{typ}'])
    #             gt_img.append(rgbs[i:i+self.hparams.batch_size])
            
    #         if i == 0:
    #             reg_loss = torch.norm(shape_codes, dim=-1) + torch.norm(texture_codes, dim=-1)
    #             loss_reg = 1e-4 * torch.mean(reg_loss)
    #             loss = loss_img + loss_reg
    #         else:
    #             loss = loss_img

    #         for optimizer in  self.optimizers():
    #             optimizer.zero_grad()
    #         self.manual_backward(loss)
    #         for optimizer in  self.optimizers():
    #             optimizer.step()
                
    #         with torch.no_grad():
    #             typ = 'fine' if 'rgb_fine' in results else 'coarse'
    #             psnr_ = psnr(results[f'rgb_{typ}'], rgbs[i:i+self.hparams.batch_size])
    #             psnr_all.append(psnr_.item())
    #         loss_img_all.append(loss_img.item())
    #         loss_latent_all.append(loss_reg.item())

    #     if batch_nb ==0:
    #         gt_img = torch.cat(gt_img).detach()
    #         gt_img = gt_img.reshape(H,W,3).permute(2, 0, 1).cpu()
    #         self.logger.experiment.log({"Train image/gt": [wandb.Image(gt_img)]})
    #         generated_img = torch.cat(generated_img)
    #         generated_img = generated_img.reshape(H,W,3).permute(2, 0, 1).cpu()
    #         self.logger.experiment.log({"Train image/pred": [wandb.Image(generated_img)]})

    #     # for scheduler in self.lr_schedulers():
    #     #     scheduler.step()            
    #     # psnr = -10*np.log(np.mean(loss_img_all)) / np.log(10)
    #     #psnr = -10*torch.log10(torch.from_numpy(np.asarray(np.mean(loss_img_all))))
    #     self.log('lr', get_learning_rate(self.optimizer))
    #     self.log('train/loss_img', np.mean(loss_img_all))
    #     self.log('train/loss_latent', np.mean(loss_latent_all))
    #     self.log('train/psnr', np.mean(psnr_all), prog_bar=True)

    #     return loss

    def on_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()  
        if self.current_epoch % self.log_frequency:
            save_latent_vectors(self.shape_codes, self.texture_codes, f'ckpts/{self.hparams.exp_name}', self.current_epoch)


    def validation_step(self, batch, batch_nb):
        rays, rgbs, obj_idx = batch['rays'], batch['rgbs'], batch['obj_id']
        print("rays, rgbs", rays.shape, rgbs.shape)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays, self.shape_codes(obj_idx), self.texture_codes(obj_idx))
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
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

    trainer = Trainer(enable_model_summary=True,
                      max_epochs=hparams.num_epochs,
                      limit_val_batches=1000,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=wandb_logger,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      strategy=DDPPlugin(find_unused_parameters=True) if hparams.num_gpus>1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)

