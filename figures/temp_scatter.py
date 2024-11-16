import os

import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


def main(
    config: SweepConfig, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    for repo_id in config.repo_ids():
        dists = Dists.load(repo_id, device)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=600)
        colors = cm.get_cmap("rainbow")
        colors.resampled(dists.n_layers)
        for layer in range(dists.n_layers):
            ax.scatter(
                dists.counts[layer].cpu().numpy(),
                dists.totals[layer].cpu().numpy(),
                color=colors(layer),
                alpha=0.5,
            )
        ax.set_xlabel("Counts")
        ax.set_ylabel("Totals")
        ax.set_title(repo_id)
        fig.savefig(
            os.path.join(out, f"scatter_{repo_id.split('/')[-1]}.png"),
            format="png",
        )
        plt.close(fig)


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
