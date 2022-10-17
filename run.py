
from opt import get_opts
import wandb
# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
wandb.login(key = '996ee27de02ee214ded37d491317d5a0567f6dc8')
wandb_logger = WandbLogger()

#baselines models
from models.nerfplusplus.model import LitNeRFPP
# from models.mipnerf360.model import LitMipNeRF360
from models.refnerf.model import LitRefNeRF

#Symmetric Voxel based pretraining + RefNeRF autoDecoder
# from models.refnerf.model_voxels import LitVoxelGenerator
from models.refnerf.model_conditional import LitRefNeRFConditional


def main(hparams):
    system = LitRefNeRF(hparams=hparams)

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
        # every_n_epochs=10,
        every_n_epochs=50,
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
                    benchmark=True,
                    limit_val_batches=5,
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