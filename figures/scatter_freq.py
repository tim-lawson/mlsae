import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


def main(
    config: SweepConfig, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    figsize, dpi = (6, 6), 300

    for repo_id in config.repo_ids(tuned_lens=config.tuned_lens):
        model_name = repo_id.split("/")[-1]
        dists = Dists.load(repo_id, device)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.set_xlim(0, 1e7)
        cmap = plt.colormaps["viridis"]
        colors = cmap(np.linspace(0, 1), dists.n_layers)

        for layer, color in zip(range(dists.n_layers), colors, strict=False):
            ax.scatter(
                dists.counts[layer],
                dists.totals[layer],
                s=2,
                alpha=0.5,
                color=color,
            )
        ax.legend([f"Layer {i}" for i in range(dists.n_layers)], loc="upper left")

        fig.savefig(os.path.join(out, f"scatter_freq_{model_name}.png"), format="png")
        plt.close(fig)


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
