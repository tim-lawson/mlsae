import os
from typing import cast

import pandas as pd
import torch
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer

from mlsae.model import DataModule, MLSAETransformer
from mlsae.trainer.config import RunConfig, initialize


@torch.no_grad()
def test(config: RunConfig, repo_id: str):
    initialize(config.seed)

    model = MLSAETransformer.from_pretrained(repo_id)
    data = DataModule(config.model_name, config=config.data)

    trainer = Trainer(
        precision=cast(_PRECISION_INPUT, config.trainer.precision),
        limit_test_batches=config.data.max_steps,
        deterministic=True,
    )
    output = trainer.test(model, datamodule=data)

    filename = f"test_{repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output).to_csv(os.path.join("out", filename), index=False)