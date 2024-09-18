import os
from typing import cast

import pandas as pd
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer

from mlsae.model import MLSAETransformer
from mlsae.model.data import get_test_dataloader
from mlsae.trainer.config import RunConfig, initialize
from mlsae.utils import get_repo_id


def test(config: RunConfig) -> None:
    initialize(config.seed)

    repo_id = get_repo_id(
        config.model_name,
        config.autoencoder.expansion_factor,
        config.autoencoder.k,
        transformer=True,
        tuned_lens=config.autoencoder.tuned_lens,
    )

    model = MLSAETransformer.from_pretrained(repo_id)
    model.requires_grad_(False)

    dataloader = get_test_dataloader(
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.data.num_workers or 1,
    )

    trainer = Trainer(
        precision=cast(_PRECISION_INPUT, config.trainer.precision),
        limit_test_batches=config.data.max_steps,
        deterministic=True,
    )

    output = trainer.test(model, dataloaders=dataloader)

    filename = f"test_{repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output).to_csv(os.path.join("out", filename), index=False)
