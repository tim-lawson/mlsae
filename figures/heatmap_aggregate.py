import os
from dataclasses import dataclass

import torch
from matplotlib.colors import PowerNorm
from simple_parsing import parse

from figures.heatmap import save_heatmap
from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    mode: str = "probs"
    """Whether to plot counts, totals, or probabilities."""

    gamma: float = 0.5
    """Gamma value for PowerNorm. Only applies to counts and totals."""


def get_heatmap_data(dists: Dists, mode: str) -> torch.Tensor:
    if mode == "counts":
        return dists.counts
    if mode == "totals":
        return dists.totals
    if mode == "probs":
        return dists.probs
    raise ValueError(f"Invalid mode: {mode}")


def get_heatmap_filename(repo_id: str, mode: str) -> str:
    return f"heatmap_aggregate_{mode}_{repo_id.split('/')[-1]}.pdf"


def main(
    repo_id: str,
    config: Config,
    device: torch.device,
    out: str | os.PathLike[str] = ".out",
):
    os.makedirs(out, exist_ok=True)
    norm = None if config.mode == "probs" else PowerNorm(config.gamma)
    dists = Dists.load(repo_id, device)
    _, indices = dists.layer_mean.sort(descending=True)
    save_heatmap(
        get_heatmap_data(dists, config.mode)[:, indices].cpu(),
        os.path.join(out, get_heatmap_filename(repo_id, config.mode)),
        norm=norm,
    )


def sweep(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    for repo_id in config.repo_ids(transformer=True, tuned_lens=config.tuned_lens):
        main(repo_id, config, device, out)


if __name__ == "__main__":
    device = get_device()
    sweep(parse(Config), device)
