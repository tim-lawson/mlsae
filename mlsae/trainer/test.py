import os
from typing import cast

import pandas as pd
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer

from mlsae.model import DataModule, MLSAETransformer
from mlsae.trainer.config import RunConfig, initialize
from mlsae.utils import get_repo_id


def test(config: RunConfig) -> None:
    initialize(config.seed)

    data = DataModule(config.model_name, config=config.data)

    repo_id = get_repo_id(
        config.model_name,
        config.autoencoder.expansion_factor,
        config.autoencoder.k,
        transformer=True,
    )

    model = MLSAETransformer.from_pretrained(repo_id)
    model.requires_grad_(False)

    trainer = Trainer(
        precision=cast(_PRECISION_INPUT, config.trainer.precision),
        limit_test_batches=config.data.max_steps,
        deterministic=True,
    )

    output = trainer.test(model, datamodule=data)

    filename = f"test_{repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output).to_csv(os.path.join("out", filename), index=False)
