import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from simple_parsing import parse

from figures.test import parse_mlsae_repo_id
from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    filename: str = "num_layers.csv"
    """The filename to save the results to."""

    threshold: int = 10_000
    """The minimum non-zero activations to be considered 'active' at a layer."""


def main(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for repo_id in config.repo_ids(transformer=True, tuned_lens=config.tuned_lens):
        dists = Dists.load(repo_id, device)
        values = torch.where(dists.counts >= config.threshold, 1, 0).sum(0).float()

        repo_id = repo_id.split("/")[-1]
        model_name, expansion_factor, k, tuned_lens = parse_mlsae_repo_id(repo_id)

        rows.append(
            {
                "model_name": model_name,
                "n_layers": dists.n_layers,
                "n_latents": dists.n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "tuned_lens": tuned_lens,
                "mean": values.mean().item(),
                "var": values.var().item(),
                "std": values.std().item(),
                "sem": values.std().item() / values.size(0) ** 0.5,
                "rel": values.mean().item() / dists.n_layers,
            }
        )

        values = values.cpu().numpy()
        hist, bins = np.histogram(
            values, bins=dists.n_layers, range=(0, dists.n_layers)
        )
        hist = np.append(hist, 0)
        pd.DataFrame({"bins": bins, "hist": hist}).to_csv(
            os.path.join(out, f"num_layers_{repo_id}_{config.threshold}.csv"),
            index=False,
        )

    pd.DataFrame(rows).to_csv(os.path.join(out, config.filename), index=False)


if __name__ == "__main__":
    main(parse(Config), get_device())
