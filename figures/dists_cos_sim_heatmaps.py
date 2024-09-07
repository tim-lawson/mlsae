import os

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.model import MLSAE
from mlsae.trainer.config import SweepConfig
from mlsae.utils import get_device, normalize


@torch.no_grad()
def get_dists_cos_sim(
    repo_id: str, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    mlsae = MLSAE.from_pretrained(repo_id).to(device)
    W_dec = mlsae.decoder.weight.detach()
    W_dec = normalize(W_dec)

    # Sort latents in descending order of mean layer
    dists = Dists.load(repo_id, device)
    _, indices = dists.layer_mean.sort(descending=True)
    W_dec = W_dec[:, indices]

    # Pairwise differences between mean layers
    layer_mean = dists.layer_mean.view(-1, 1) - dists.layer_mean.view(1, -1)

    # Pairwise cosine similarities between decoder weight vectors
    cos_sim = torch.mm(W_dec.T, W_dec)

    # Remove duplicates and self-similarities
    triu_indices = torch.triu_indices(*cos_sim.shape, offset=1)
    x = layer_mean[*triu_indices].cpu()
    y = cos_sim[*triu_indices].cpu()

    return x, y


def save_heatmap(
    x: torch.Tensor,
    y: torch.Tensor,
    filename: str,
    figsize: tuple[float, float] = (2, 2),
    dpi: int = 300,
    cmap: str | Colormap | None = "magma_r",
) -> None:
    plt.rcParams.update({"axes.linewidth": 0})
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.hist2d(x, y, bins=[64, 512], range=[(0, 5), (-0.25, 0.25)], cmap=cmap)
    ax.set_axis_off()
    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    device = get_device()
    for repo_id in parse(SweepConfig).repo_ids(transformer=False):
        filename = f"dists_cos_sim_heatmap_{repo_id.split('/')[-1]}.pdf"
        x, y = get_dists_cos_sim(repo_id, device)
        save_heatmap(x, y, os.path.join("out", filename))
