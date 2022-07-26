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
from losses import loss_dict, loss_dict_latent_reg

# metrics
from metrics import *
# torch.set_printoptions(profile="full")

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

class NeRFEncoderRegression(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict_latent_reg['latent'](coef=1)
        self.resnet_encoder = MultiHeadImgEncoder()

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        self.train_dataset = dataset(splits=self.hparams.splits, cat = self.hparams.cat,
                                     img_wh = self.hparams.img_wh, crop_img = self.hparams.crop_img, 
                                     encoder_reg=True, latent_code_path = self.hparams.latent_code_path)
        self.val_dataset = dataset(splits=self.hparams.splits, cat = self.hparams.cat,
                                     img_wh = self.hparams.img_wh, crop_img = self.hparams.crop_img, 
                                     encoder_reg=True, latent_code_path = self.hparams.latent_code_path)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.resnet_encoder)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                        #   batch_size=self.hparams.batch_size,
                          batch_size=64,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        shape_code, texture_code = batch['shape_code'], batch['texture_code']     
        latents = self.resnet_encoder(batch['enc_img'])
        shape_code_predicted = latents["density"].squeeze(0)
        texture_codes_predicted = latents["color"].squeeze(0)

        shape_latent_loss   = self.loss(shape_code_predicted, shape_code)
        texture_latent_loss = self.loss(texture_codes_predicted, texture_code)
        loss = shape_latent_loss + texture_latent_loss

        self.log('train/shape_latent_loss', shape_latent_loss)
        self.log('train/texture_latent_loss', shape_latent_loss)

        self.log('lr', get_learning_rate(self.optimizer))
        return loss

    def validation_step(self, batch, batch_nb):

        shape_code, texture_code = batch['shape_code'], batch['texture_code']
        
        latents = self.resnet_encoder(batch['enc_img'])
        shape_code_predicted = latents["density"].squeeze(0)
        texture_codes_predicted = latents["color"].squeeze(0)

        shape_latent_loss   = self.loss(shape_code_predicted, shape_code)
        texture_latent_loss = self.loss(texture_codes_predicted, texture_code)
        loss = shape_latent_loss + texture_latent_loss

        # self.log('train/shape_latent_loss', shape_latent_loss)
        # self.log('train/texture_latent_loss', shape_latent_loss)

        log = {}
        log = {'val_loss_shape': shape_latent_loss}
        log ['val_loss_texture'] = texture_latent_loss

        return log

    def validation_epoch_end(self, outputs):
        mean_loss_shape = torch.stack([x['val_loss_shape'] for x in outputs]).mean()
        mean_loss_texture = torch.stack([x['val_loss_texture'] for x in outputs]).mean()
        self.log('val/loss_shape', mean_loss_shape)
        self.log('val/loss_texture', mean_loss_texture)


def main(hparams):
    if hparams.ckpt_path:
        system = NeRFEncoderRegression.load_from_checkpoint(hparams.ckpt_path, hparams=hparams)
    else:
        system = NeRFEncoderRegression(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/loss_shape',
                              mode='min',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]
    wandb_logger = WandbLogger()

    trainer = Trainer(enable_model_summary=True,
                      max_epochs=hparams.num_epochs,
                      limit_val_batches=100,
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

