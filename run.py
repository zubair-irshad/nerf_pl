
from opt import get_opts

# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger()
from models.nerfplusplus.model import LitNeRFPP
# from models.refnerf.model import LitRefNeRF
from models.refnerf.model_disentagled import LitRefNeRF
# from models.mipnerf360.model import LitMipNeRF360

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
                    profiler="simple" if hparams.num_gpus==1 else None,
                    strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)

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