import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap


def save_heatmap(
    data: torch.Tensor,
    filename: str,
    figsize: tuple[float, float] = (5.5, 1.25),
    dpi: int = 1200,
    cmap: str | Colormap | None = "magma_r",
) -> None:
    # Exclude latents with only NaN values
    data = data[:, ~torch.all(data.isnan(), dim=0)]

    n_layers, n_latents = data.shape
    extent = (0, n_latents, 0, n_layers)

    plt.rcParams.update({"axes.linewidth": 0})
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.imshow(data, cmap=cmap, aspect="auto", extent=extent, interpolation="nearest")
    ax.set_axis_off()

    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
