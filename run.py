
from opt import get_opts
import wandb
import torch
# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
wandb.login(key = '996ee27de02ee214ded37d491317d5a0567f6dc8')
wandb_logger = WandbLogger()
import numpy as np
import random
#baselines models
from models.nerfplusplus.model import LitNeRFPP
from models.vanilla_nerf.model_pixel import LitPixelNeRF
#
from models.vanilla_nerf.model import LitNeRF
# from models.mipnerf360.model import LitMipNeRF360
from models.refnerf.model import LitRefNeRF

#Symmetric Voxel based pretraining + RefNeRF autoDecoder
from models.refnerf.model_voxels import LitVoxelGenerator
from models.refnerf.model_conditional import LitRefNeRFConditional
from models.refnerf.model_conditional_ae import LitRefNeRFConditionalAE
from models.nerfplusplus.model_ae import LitNeRFPP_AE
from models.nerfplusplus.model_groundplan import LitNeRFPP_GP
from models.nerfplusplus.model_triplane import LitNeRFPP_TP
from models.nerfplusplus.model_co_groundplan import LitNeRFPP_CO_GP
from models.nerfplusplus.model_co_groundplan_nocs import LitNeRFPP_CO_GP_NOCS
from models.nerfplusplus.model_co_triplane import LitNeRFPP_CO_TP
from models.nerfplusplus.model_co_triplane_nocs import LitNeRFPP_CO_TP_NOCS

# For debugging
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)   
random.seed(0)

def main(hparams):
    if hparams.exp_type == 'pixelnerf':
        system = LitPixelNeRF(hparams=hparams)
    elif hparams.exp_type == 'vanilla':
        system = LitNeRF(hparams=hparams) # Needs to modify this to train for 3 test images
    elif hparams.exp_type == 'groundplanar':
        system = LitNeRFPP_CO_GP(hparams=hparams)
    elif hparams.exp_type == 'groundplanar_nocs':
        system = LitNeRFPP_CO_GP_NOCS(hparams=hparams)
    elif hparams.exp_type == 'triplanar':
        system = LitNeRFPP_CO_TP(hparams=hparams)

    # ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
    #                           filename='{epoch:d}',
    #                           monitor='val/psnr',
    #                           mode='max',
    #                           save_top_k=5)

    ckpt_cb = ModelCheckpoint(
        dirpath=f'ckpts/{hparams.exp_name}',
        monitor='val/psnr',
        filename='{epoch:d}',
        save_top_k=5,
        mode="max",
        save_last=True,
        every_n_epochs=10,
        # every_n_epochs=50,
    )

    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]
    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=hparams.num_epochs,
                    callbacks=callbacks,
                    resume_from_checkpoint=hparams.ckpt_path,
                    logger=wandb_logger,
                    enable_model_summary=False,
                    accelerator='auto',
                    devices=hparams.num_gpus,
                    num_sanity_val_steps=1,
                    detect_anomaly=True,
                    benchmark=True,
                    check_val_every_n_epoch=1,
                    limit_val_batches=1, # for single scene scenario
                    profiler="simple" if hparams.num_gpus==1 else None,
                    strategy=DDPPlugin(find_unused_parameters=True) if hparams.num_gpus>1 else None)

    if hparams.run_eval:
        ckpt_path = (
            f"ckpts/{hparams.exp_name}/last.ckpt"
        )
        trainer.test(system, ckpt_path=ckpt_path)
        # self.val_dataset = dataset(split='val', **kwargs_test)
    else:
        trainer.fit(system)



if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)