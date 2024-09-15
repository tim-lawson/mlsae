import math
from dataclasses import dataclass

import torch
from datasets import IterableDataset, load_dataset
from datasets.formatting.formatting import LazyBatch
from jaxtyping import Int
from simple_parsing import Serializable
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class DataConfig(Serializable):
    """The data configuration."""

    path: str = "monology/pile-uncopyrighted"
    """The path to a HuggingFace text dataset."""

    max_length: int = 2048
    """The maximum length of a tokenized input sequence."""

    batch_size: int = 1
    """The number of sequences in a batch."""

    max_tokens: float = 1_000_000_000
    """The maximum number of tokens to train on."""

    num_workers: int | None = None
    """The number of workers to use for data loading."""

    @property
    def max_steps(self) -> int:
        """The maximum number of batches to train on."""

        return math.ceil(self.max_tokens / (self.batch_size * self.max_length))


def concat_and_tokenize(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> IterableDataset:
    return dataset.map(
        _concat_and_tokenize,
        batched=True,
        # Large batch size minimizes the number of tokens dropped
        batch_size=1024,
        # TODO: Column names are not always available
        remove_columns=dataset.column_names or ["text", "meta"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    ).with_format("torch")


# Based on https://github.com/EleutherAI/sae/blob/19d95a401e9d17dbf7d6fb0fa7a91081f1b0d01f/sae/data.py
def _concat_and_tokenize(
    batch: LazyBatch, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> dict:
    output = tokenizer(
        # Concatenate the batch of text with the EOS token
        tokenizer.eos_token.join([""] + batch["text"]),  # type: ignore
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_overflowing_tokens=True,
    )

    overflowing_tokens = output.pop("overflowing_tokens", None)
    _ = output.pop("overflow_to_sample_mapping", None)

    # Split the overflowing tokens into sequences of the maximum length
    if overflowing_tokens is not None:
        output["input_ids"] += [
            overflowing_tokens[i * max_length : (i + 1) * max_length]
            for i in range(math.ceil(len(overflowing_tokens) / max_length))
        ]  # type: ignore

    # Drop the last batch, which is probably incomplete
    return {k: v[:-1] for k, v in output.items()}


def get_dataloader(
    dataset: IterableDataset,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_workers: int = 1,
) -> DataLoader[Int[Tensor, "batch pos"]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Constrain the maximum length of a tokenized input sequence
    max_length = min(tokenizer.model_max_length, max_length)

    return DataLoader(
        concat_and_tokenize(dataset, tokenizer, max_length),  # type: ignore
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_train_dataloader(
    path: str, model_name: str, max_length: int, batch_size: int, num_workers: int = 1
) -> DataLoader[torch.Tensor]:
    return get_dataloader(
        load_dataset(path, split="train", streaming=True),  # type: ignore
        model_name,
        max_length,
        batch_size,
        num_workers,
    )


def get_test_dataloader(
    model_name: str, max_length: int, batch_size: int, num_workers: int = 1
) -> DataLoader[torch.Tensor]:
    return get_dataloader(
        load_dataset(
            "json",
            data_files="./data/test.jsonl.zst",
            split="train",
            streaming=True,
        ),  # type: ignore
        model_name,
        max_length,
        batch_size,
        num_workers,
    )
