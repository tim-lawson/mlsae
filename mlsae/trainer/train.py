import math
from typing import cast

import wandb
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from mlsae.model import MLSAETransformer
from mlsae.model.data import get_train_dataloader
from mlsae.trainer.config import RunConfig, initialize


def train(config: RunConfig) -> None:
    initialize(config.seed)

    train_dataloader = get_train_dataloader(
        config.data.path,
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.data.num_workers or 1,
    )

    val_dataloader = get_train_dataloader(
        config.data.path,
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.data.num_workers or 1,
    )

    model: MLSAETransformer = MLSAETransformer(
        config.model_name,
        config.layers,
        config.autoencoder.expansion_factor,
        config.autoencoder.k,
        config.autoencoder.auxk,
        config.autoencoder.auxk_coef,
        config.autoencoder.dead_tokens_threshold,
        config.autoencoder.dead_threshold,
        config.autoencoder.lr,
        config.autoencoder.standardize,
        config.autoencoder.skip_special_tokens,
        config.data.max_length,
        config.data.batch_size,
        config.trainer.accumulate_grad_batches,
        config.autoencoder.tuned_lens,
    )  # type: ignore

    wandb.login()  # type: ignore

    trainer = Trainer(
        precision=cast(_PRECISION_INPUT, config.trainer.precision),
        logger=WandbLogger(
            name=config.run,
            save_dir="wandb_logs",
            project=config.project,
            log_model=True,
        ),
        max_steps=config.trainer.max_steps
        or math.ceil(config.data.max_steps / config.trainer.accumulate_grad_batches),
        limit_val_batches=config.trainer.limit_val_batches,
        val_check_interval=config.trainer.val_check_interval,
        log_every_n_steps=config.trainer.log_every_n_steps,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        deterministic=True,
        default_root_dir=config.trainer.default_root_dir,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=config.trainer.checkpoint_path,
    )

    wandb.finish()  # type: ignore
