import os
from dataclasses import dataclass, field

import pandas as pd
import torch
from datasets import Dataset
from safetensors.torch import save_file
from simple_parsing import Serializable, parse
from tqdm import tqdm

from mlsae.analysis.dists import Dists, Metric, get_stats
from mlsae.model import get_test_dataloader
from mlsae.model.data import DataConfig
from mlsae.trainer import initialize
from mlsae.utils import forward_single_layer, get_device, get_repo_id, load_single_layer


@dataclass
class Config(Serializable):
    model_name: str
    layer: int
    expansion_factor: int = 64
    k: int = 32
    tuned_lens: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    """The data configuration. Remember to set max_tokens to a reasonable value!"""

    seed: int = 42
    """The seed for global random state."""

    log_every_n_steps: int | None = 8
    """The number of steps between logging statistics."""

    push_to_hub: bool = True
    """Whether to push the dataset to HuggingFace."""


@torch.no_grad()
def get_tensors(config: Config, device: torch.device) -> dict[str, torch.Tensor]:
    model = load_single_layer(
        config.model_name,
        config.layer,
        device,
        expansion_factor=config.expansion_factor,
        k=config.k,
        tuned_lens=config.tuned_lens,
    )

    dataloader = get_test_dataloader(
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
    )

    tokens_per_step = config.data.batch_size * config.data.max_length

    metric = Metric(model.n_layers, model.n_latents, device)
    rows: list[dict[str, str | int | float]] = []

    for i, batch in enumerate(tqdm(dataloader, total=config.data.max_steps)):
        inputs, recons, topk = forward_single_layer(
            model, batch["input_ids"].to(device)
        )
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
        repo_id = get_repo_id(
            config.model_name,
            config.expansion_factor,
            config.k,
            config.tuned_lens,
            True,
            [config.layer],
        ).split("/")[-1]
        pd.DataFrame(rows).to_csv(
            os.path.join("out", f"dists_layer_std_step_{repo_id}.csv"), index=False
        )

    return metric.compute()


def main(config: Config, device: torch.device) -> None:
    initialize(config.seed)

    tensors = get_tensors(config, device)
    repo_id = get_repo_id(
        config.model_name,
        config.expansion_factor,
        config.k,
        config.tuned_lens,
        True,
        [config.layer],
    )
    repo_id = Dists.repo_id(repo_id)
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
    main(parse(Config), get_device())
