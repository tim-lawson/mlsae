import os
from collections.abc import Generator
from dataclasses import dataclass, field
from itertools import product

import torch
from lightning.pytorch import seed_everything
from simple_parsing import Serializable

from mlsae.model import DataConfig, MLSAEConfig
from mlsae.utils import get_repo_id


@dataclass
class TrainerConfig(Serializable):
    """The trainer configuration."""

    checkpoint_path: str | None = None
    """The path to a model checkpoint to resume training."""

    precision: str = "16-mixed"
    """The precision of the training parameters."""

    accumulate_grad_batches: int = 64
    """The number of batches over which to accumulate gradients."""

    max_steps: int | None = None
    """The maximum number of training batches. If None, uses the maximum tokens."""

    log_every_n_steps: int | None = 8
    """The number of training steps between logging metrics."""

    val_check_interval: int | float | None = 64 * 64
    """The number of training batches between validation steps."""

    limit_val_batches: int | float | None = 64 * 8  # 1M tokens with batch size 2048
    """The number of batches to validate on."""

    default_root_dir: str | None = None
    """The default root directory for model checkpoints."""


@dataclass
class RunConfig(Serializable):
    autoencoder: MLSAEConfig = field(default_factory=MLSAEConfig)
    """The autoencoder configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    """The data configuration."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """The trainer configuration."""

    seed: int = 42
    """The seed for global random state."""

    model_name: str = "EleutherAI/pythia-70m-deduped"
    """The name of a pretrained HuggingFace GPTNeoXForCausalLM model."""

    layers: list[int] | None = None
    """The layers to train on. If None, trains on all layers."""

    project: str | None = None
    """The Weights & Biases project name."""

    run: str | None = None
    """The Weights & Biases run name."""


@dataclass
class SweepConfig(Serializable):
    model_name: list[str] = field(
        default_factory=lambda: ["EleutherAI/pythia-70m-deduped"]
    )
    """The names of pretrained HuggingFace GPTNeoXForCausalLM models."""

    expansion_factor: list[int] = field(default_factory=list)
    """The ratios of the number of latents to the number of inputs."""

    k: list[int] = field(default_factory=list)
    """The numbers of largest latents to keep."""

    tuned_lens: bool = False
    """Whether to apply a pretrained tuned lens before the encoder."""

    seed: int = 42
    """The seed for global random state."""

    def __iter__(self) -> Generator[tuple[str, int, int], None, None]:
        yield from product(self.model_name, self.expansion_factor, self.k)

    def repo_ids(
        self, transformer: bool = True, tuned_lens: bool = False
    ) -> Generator[str, None, None]:
        for param in self:
            yield get_repo_id(
                *param, transformer=transformer, tuned_lens=self.tuned_lens
            )


def initialize(seed: int) -> None:
    # Deterministic for reproducibility
    seed_everything(seed=seed, workers=True)

    # Fork processes via multiprocessing in Python
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Avoid PyTorch DataLoader "too many open files"
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Improve matmul performance
    torch.set_float32_matmul_precision("high")
