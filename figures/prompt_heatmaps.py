import os
from dataclasses import dataclass

import torch
from simple_parsing import parse

from mlsae.analysis.heatmaps import save_heatmap
from mlsae.model import MLSAETransformer
from mlsae.model.decoder import scatter_topk
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    prompt: str = "When Mary and John went to the store, John gave a drink to"
    """The prompt to generate heatmaps for."""

    dead_threshold: float = 1e-3
    """The threshold activation to exclude latents."""


@torch.no_grad()
def get_data(config: Config, repo_id: str, device: torch.device | str) -> torch.Tensor:
    model = MLSAETransformer.from_pretrained(repo_id).to(device)

    tokens = model.transformer.tokenizer.encode(config.prompt)
    tokens = torch.tensor(tokens).unsqueeze(0)
    inputs = model.transformer.forward(tokens.to(device))
    topk = model.autoencoder.forward(inputs).topk
    latents = scatter_topk(topk, model.n_latents).squeeze()

    # Exclude latents with maximum activation below the threshold
    latents = latents[:, latents.max(dim=0).values.gt(config.dead_threshold)]

    # Sort latents in ascending order of mean layer
    layers = torch.arange(0, model.n_layers, device=device).unsqueeze(-1)
    _, indices = ((latents / latents.sum(0)) * layers).sum(0).sort(descending=True)

    return latents[:, indices]


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    for repo_id in config.repo_ids():
        filename = f"prompt_heatmap_{repo_id.split('/')[-1]}.pdf"
        data = get_data(config, repo_id, device)
        save_heatmap(data.cpu(), os.path.join("out", filename))
