import os
from typing import cast

import pandas as pd
from datasets import IterableDataset, load_dataset
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from transformers import AutoTokenizer

from mlsae.model import MLSAETransformer
from mlsae.model.data import concat_and_tokenize, get_dataloader
from mlsae.trainer.config import RunConfig, initialize
from mlsae.utils import get_repo_id


def test(config: RunConfig) -> None:
    initialize(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    max_length = min(tokenizer.model_max_length, config.data.max_length)

    dataset = load_dataset(
        "json",
        data_files="./data/test.jsonl.zst",
        split="train",
        streaming=True,
    )
    assert isinstance(dataset, IterableDataset)

    dataloader = get_dataloader(
        concat_and_tokenize(dataset, tokenizer, max_length),
        config.data.batch_size,
        num_workers=1,
    )

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

    output = trainer.test(model, dataloaders=dataloader)

    filename = f"test_{repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output).to_csv(os.path.join("out", filename), index=False)
