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
    mode: str,
    device: torch.device,
    out: str | os.PathLike[str] = ".out",
):
    norm = None if mode == "probs" else PowerNorm(0.5)
    dists = Dists.load(repo_id, device)
    _, indices = dists.layer_mean.sort(descending=True)
    save_heatmap(
        get_heatmap_data(dists, mode)[:, indices].cpu(),
        os.path.join(out, get_heatmap_filename(repo_id, mode)),
        norm=norm,
    )


def sweep(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    for repo_id in config.repo_ids():
        main(repo_id, config.mode, device, out)


def sweep_layers() -> None:
    for repo_id in [
        "tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-0-dists",
        "tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-1-dists",
        "tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-2-dists",
        "tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-3-dists",
        "tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-4-dists",
        "tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-5-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-0-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-1-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-2-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-3-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-4-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-5-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-6-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-7-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-8-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-9-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-10-dists",
        "tim-lawson/sae-pythia-160m-deduped-x64-k32-tfm-layers-11-dists",
    ]:
        main(repo_id, "probs", device)
        main(repo_id, "counts", device)
        main(repo_id, "totals", device)


if __name__ == "__main__":
    device = get_device()
    sweep(parse(Config), device)
