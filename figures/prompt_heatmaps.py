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

    tuned_lens: bool = False
    """Whether to apply a pretrained tuned lens before the encoder."""


@torch.no_grad()
def get_data(config: Config, repo_id: str, device: torch.device | str) -> torch.Tensor:
    model = MLSAETransformer.from_pretrained(repo_id).to(device)

    tokens = model.transformer.tokenizer.encode(config.prompt)
    tokens = torch.tensor(tokens).unsqueeze(0)
    inputs = model.transformer.forward(tokens.to(device))
    topk = model.autoencoder.forward(inputs).topk

    # Sum over the tokens in the prompt
    latents = scatter_topk(topk, model.n_latents).squeeze().sum(dim=1)

    # Exclude latents with maximum activation below the threshold
    latents = latents[:, latents.max(dim=0).values.gt(config.dead_threshold)]

    # Convert activations to probabilities
    probs = latents / latents.sum(dim=0, keepdim=True)

    # Sort latents in ascending order of mean layer
    layers = torch.arange(0, model.n_layers, device=device).unsqueeze(-1)
    _, indices = (probs * layers).sum(0).sort(descending=True)

    return probs[:, indices]


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    for repo_id in config.repo_ids(tuned_lens=config.tuned_lens):
        filename = f"prompt_heatmap_{repo_id.split('/')[-1]}.pdf"
        data = get_data(config, repo_id, device)
        save_heatmap(data.cpu(), os.path.join("out", filename), figsize=(5.5, 1.25))
