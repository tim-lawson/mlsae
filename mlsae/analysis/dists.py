import os
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from safetensors.torch import load_file, save_file
from simple_parsing import Serializable, field, parse
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAETransformer, TopK, get_train_dataloader
from mlsae.trainer import initialize
from mlsae.utils import get_device


@dataclass
class Config(Serializable):
    repo_id: str
    """
    The name of a pretrained autoencoder and transformer from HuggingFace, or the path
    to a directory that contains them.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """The data configuration. Remember to set max_tokens to a reasonable value!"""

    seed: int = 42
    """The seed for global random state."""

    log_every_n_steps: int | None = None
    """The number of steps between logging statistics."""

    push_to_hub: bool = True
    """Whether to push the dataset to HuggingFace."""


class Metric:
    def __init__(
        self, n_layers: int, n_latents: int, device: torch.device | str = "cpu"
    ) -> None:
        self.n_layers = n_layers
        self.n_latents = n_latents
        self.counts = torch.zeros((n_layers, n_latents), device=device)
        self.totals = torch.zeros((n_layers, n_latents), device=device)
        self.layers = torch.arange(n_layers, device=device).unsqueeze(1)
        self.device = device

    def update(self, x: TopK) -> None:
        for layer in range(self.n_layers):
            indices = x.indices[layer].squeeze().view(-1)
            values = x.values[layer].squeeze().view(-1)
            ones = torch.ones_like(indices, dtype=torch.float)
            self.counts[layer].put_(indices, ones, accumulate=True)
            self.totals[layer].put_(indices, values, accumulate=True)

    def compute(self) -> dict[str, torch.Tensor]:
        return dict(counts=self.counts, totals=self.totals)


def get_stats(layer_std: torch.Tensor) -> dict[str, float]:
    values = layer_std.cpu().numpy()
    std = np.nanstd(values).item()
    return {
        "mean": np.nanmean(values).item(),
        "var": np.nanvar(values).item(),
        "std": std,
        "sem": std / np.sqrt(len(values)),
    }


@torch.no_grad()
def get_tensors(
    config: Config, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    model = MLSAETransformer.from_pretrained(config.repo_id).to(device)

    dataloader = get_train_dataloader(
        config.data.path,
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
    )

    tokens_per_step = config.data.batch_size * config.data.max_length

    metric = Metric(model.n_layers, model.n_latents, device)
    rows: list[dict[str, str | int | float]] = []

    for i, batch in enumerate(tqdm(dataloader, total=config.data.max_steps)):
        inputs = model.transformer.forward(batch["input_ids"].to(device))
        topk = model.autoencoder.encode(inputs).topk
        metric.update(topk)

        if config.log_every_n_steps is not None and i % config.log_every_n_steps == 0:
            dists = Dists.from_tensors(metric.compute(), metric.device)
            rows.append(
                {
                    "model_name": model.model_name,
                    "n_layers": model.n_layers,
                    "n_latents": model.n_latents,
                    "expansion_factor": model.expansion_factor,
                    "k": model.k,
                    "step": i,
                    "tokens": (i + 1) * tokens_per_step,
                    **get_stats(dists.layer_std),
                }
            )

        if i > config.data.max_steps:
            break

    if len(rows) > 0:
        repo_id = config.repo_id.split("/")[-1]
        pd.DataFrame(rows).to_csv(
            os.path.join("out", f"dists_layer_std_step_{repo_id}.csv"), index=False
        )

    return metric.compute()


class Dists:
    def __init__(
        self,
        tensors: dict[str, torch.Tensor] | None = None,
        filename: str | os.PathLike[str] | None = None,
        device: torch.device | str | int = "cpu",
    ):
        device = str(device) if isinstance(device, torch.device) else device
        if tensors is not None:
            self.tensors = {k: v.to(device) for k, v in tensors.items()}
        elif filename is not None:
            self.tensors = load_file(filename, device)
        else:
            raise ValueError("either tensors or filename must be provided")
        self.counts = self.tensors["counts"]  # n_layers n_latents
        self.totals = self.tensors["totals"]  # n_layers n_latents
        self.n_layers, self.n_latents = self.counts.shape
        self.layers = torch.arange(self.n_layers, device=device).unsqueeze(1)

    @cached_property
    def count(self) -> torch.Tensor:
        return self.counts.sum(0)  # n_latents

    @cached_property
    def total(self) -> torch.Tensor:
        return self.totals.sum(0)  # n_latents

    @cached_property
    def mean(self) -> torch.Tensor:
        return self.total / (self.count + 1e-8)  # n_latents

    @cached_property
    def means(self) -> torch.Tensor:
        return self.totals / (self.counts + 1e-8)  # n_layers n_latents

    @cached_property
    def probs(self) -> torch.Tensor:
        return self.totals / self.totals.sum(0)  # n_layers n_latents

    @cached_property
    def layer_mean(self) -> torch.Tensor:
        return (self.probs * self.layers).sum(0)  # n_latents

    @cached_property
    def layer_var(self) -> torch.Tensor:
        return (self.probs * self.layers**2).sum(0) - self.layer_mean**2  # n_latents

    @cached_property
    def layer_std(self) -> torch.Tensor:
        return self.layer_var.sqrt()  # n_latents

    def __iter__(self) -> Generator[dict[str, list[float] | float], None, None]:
        for latent in range(self.n_latents):
            yield {
                "latent": latent,
                "count": self.count[latent].item(),
                "total": self.total[latent].item(),
                "mean": self.mean[latent].item(),
                "layer_mean": self.layer_mean[latent].item(),
                "layer_var": self.layer_var[latent].item(),
                "layer_std": self.layer_std[latent].item(),
                "counts": self.counts[:, latent].tolist(),
                "totals": self.totals[:, latent].tolist(),
                "means": self.means[:, latent].tolist(),
                "probs": self.probs[:, latent].tolist(),
            }

    @staticmethod
    def load(repo_id: str, device: torch.device | str | int) -> "Dists":
        repo_id = Dists.repo_id(repo_id)
        try:
            filename = Dists.filename(repo_id)
            return Dists.from_file(filename, str(device))
        except Exception:
            return Dists.from_hub(repo_id, str(device))

    @staticmethod
    def from_tensors(
        tensors: dict[str, torch.Tensor], device: torch.device | str | int = "cpu"
    ) -> "Dists":
        return Dists(tensors=tensors, device=device)

    @staticmethod
    def from_file(
        filename: str | os.PathLike[str], device: torch.device | str | int = "cpu"
    ) -> "Dists":
        return Dists(filename=filename, device=device)

    @staticmethod
    def from_dataset(
        dataset: Dataset, device: torch.device | str | int = "cpu"
    ) -> "Dists":
        n_layers, n_latents = len(dataset["counts"][0]), len(dataset["counts"])
        tensors = {
            "counts": torch.zeros((n_layers, n_latents), device=device),
            "totals": torch.zeros((n_layers, n_latents), device=device),
        }
        for i, item in enumerate(dataset):
            assert isinstance(item, dict)
            tensors["counts"][:, i] = torch.tensor(item["counts"], device=device)
            tensors["totals"][:, i] = torch.tensor(item["totals"], device=device)
        return Dists(tensors=tensors, device=device)

    @staticmethod
    def from_hub(repo_id: str, device: torch.device | str | int = "cpu") -> "Dists":
        dataset = load_dataset(Dists.repo_id(repo_id))
        assert isinstance(dataset, DatasetDict)
        return Dists.from_dataset(dataset["train"], device)

    @staticmethod
    def repo_id(repo_id: str) -> str:
        if repo_id.endswith("-dists"):
            logger.warning(f"repo_id {repo_id} already ends with '-dists'")
            return repo_id
        if repo_id.endswith("-tfm"):
            return repo_id.replace("-tfm", "-dists")
        return repo_id + "-dists"

    @staticmethod
    def filename(repo_id: str) -> str:
        os.makedirs("out", exist_ok=True)
        return os.path.join(
            "out", f"{Dists.repo_id(repo_id).replace('/', '-')}.safetensors"
        )


def save_dists(config: Config, device: torch.device | str = "cpu") -> None:
    tensors = get_tensors(config, device)
    repo_id = Dists.repo_id(config.repo_id)
    filename = Dists.filename(repo_id)

    save_file(tensors, filename)
    _test = Dists.from_tensors(tensors, device)
    _test = Dists.from_file(filename, device)

    if config.push_to_hub:
        dataset = Dataset.from_generator(Dists(tensors).__iter__)
        assert isinstance(dataset, Dataset)
        dataset.push_to_hub(repo_id, commit_description=config.dumps_json())
        _test = Dists.from_dataset(dataset, device)
        _test = Dists.from_hub(repo_id, device)


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)
    save_dists(config, device)
