import math
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Literal

from datasets import Dataset, IterableDataset, load_dataset
from datasets.formatting.formatting import LazyBatch
from jaxtyping import Int
from lightning.pytorch import LightningDataModule
from simple_parsing import Serializable
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class DataConfig(Serializable):
    """The data configuration."""

    path: str = "monology/pile-uncopyrighted"
    """The path to a HuggingFace text dataset."""

    # TODO: Support different splits
    name: str | None = None
    """The subset of the dataset to train on."""

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


class DataModule(LightningDataModule):
    def __init__(self, model_name: str, config: DataConfig) -> None:
        """
        Args:
            model_name (str): The name of a pretrained GPTNeoXForCausalLM model.

            config (DataConfig): The data configuration.
        """

        super().__init__()

        self.model_name = model_name
        self.config = config
        self.num_workers = config.num_workers or cpu_count() // 2

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Constrain the maximum length of a tokenized input sequence
        self.max_length = min(self.tokenizer.model_max_length, self.config.max_length)

    def _dataset(
        self, split: Literal["train", "validation", "test"]
    ) -> IterableDataset | Dataset:
        dataset: IterableDataset | Dataset = load_dataset(
            self.config.path,
            name=self.config.name,
            split=split,
            # TODO: This is specific to monology/pile-uncopyrighted
            streaming=split == "train",
        )  # type: ignore

        return dataset.map(
            concat_and_tokenize,
            batched=True,
            # Large batch size minimizes the number of tokens dropped
            batch_size=1024,
            # TODO: Column names are not always available
            remove_columns=dataset.column_names or ["text", "meta"],
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": self.max_length},
        ).with_format("torch")

    def _dataloader(
        self, dataset: IterableDataset | Dataset, num_workers: int | None = None
    ) -> DataLoader[Int[Tensor, "batch pos"]]:
        return DataLoader(
            dataset,  # type: ignore
            batch_size=self.config.batch_size,
            num_workers=num_workers or self.num_workers,
        )

    def train_dataloader(
        self, num_workers: int | None = None
    ) -> DataLoader[Int[Tensor, "batch pos"]]:
        return self._dataloader(self._dataset("train"), num_workers)

    def val_dataloader(
        self, num_workers: int | None = None
    ) -> DataLoader[Int[Tensor, "batch pos"]]:
        return self._dataloader(self._dataset("train"), num_workers)

    def test_dataloader(
        self, num_workers: int | None = None
    ) -> DataLoader[Int[Tensor, "batch pos"]]:
        return self._dataloader(self._dataset("test"), num_workers)


# Based on https://github.com/EleutherAI/sae/blob/19d95a401e9d17dbf7d6fb0fa7a91081f1b0d01f/sae/data.py
def concat_and_tokenize(
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
