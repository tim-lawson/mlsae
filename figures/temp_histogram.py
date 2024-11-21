import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    mode: str = "counts"
    """Whether to plot counts or totals."""


def get_data(dists: Dists, mode: str) -> torch.Tensor:
    if mode == "counts":
        return dists.counts
    if mode == "totals":
        return dists.totals
    raise ValueError(f"Invalid mode: {mode}")


def get_csv(repo_id: str, mode: str) -> str:
    return f"temp_{mode}_{repo_id.split('/')[-1]}.csv"


def get_pdf(repo_id: str, mode: str) -> str:
    return f"temp_{mode}_{repo_id.split('/')[-1]}.pdf"


def main(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    for repo_id in config.repo_ids():
        dists = Dists.load(repo_id, device)

        data = get_data(dists, config.mode) / 1e7  # n_tokens
        max = data.max().item()

        hists, bins = [], np.array([])
        for layer in range(dists.n_layers):
            hist, bins = np.histogram(
                data[layer, :].cpu().numpy(),
                bins=10,
                range=(0, max),
            )
            hist = hist / hist.sum()  # percentages
            hist = np.append(hist, 0).tolist()  # bins has one more element
            hists.append(hist)

        df = pd.DataFrame(hists, columns=bins.tolist())
        df.to_csv(os.path.join(out, get_csv(repo_id, config.mode)), index=False)


if __name__ == "__main__":
    main(parse(Config), get_device())
