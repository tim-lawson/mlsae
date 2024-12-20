import os
from dataclasses import dataclass

import torch
from matplotlib.colors import PowerNorm
from simple_parsing import parse

from figures.heatmap import save_heatmap
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

    mode: str = "probs"
    """Whether to plot counts, totals, or probabilities."""

    gamma: float = 0.5
    """Gamma value for PowerNorm. Only applies to counts and totals."""


@torch.no_grad()
def get_heatmap_data(
    config: Config, repo_id: str, device: torch.device | str
) -> torch.Tensor:
    model = MLSAETransformer.from_pretrained(repo_id).to(device)
    model.transformer.tokenizer.pad_token = model.transformer.tokenizer.eos_token
    assert model.transformer.tokenizer.pad_token_id is not None

    tokens = torch.tensor(
        model.transformer.tokenizer.encode(
            config.prompt,
            padding="max_length",
            max_length=model.max_length,
        )
    )

    inputs = model.transformer.forward(tokens.unsqueeze(0).to(device))
    inputs = inputs[:, :, tokens.ne(model.transformer.tokenizer.pad_token_id), :]

    topk = model.autoencoder.forward(inputs).topk

    latents = scatter_topk(topk, model.n_latents).squeeze()

    probs = latents.sum(dim=1) / latents.sum(dim=1).sum(dim=0, keepdim=True)

    if config.mode == "counts":
        data = latents.where(latents.gt(config.dead_threshold), 0).float().sum(dim=1)
    elif config.mode == "totals":
        data = latents.sum(dim=1)
    elif config.mode == "probs":
        latents = latents.sum(dim=1)
        data = latents / latents.sum(dim=0, keepdim=True)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    # Exclude latents that never activate
    mask = torch.any(data.gt(0), dim=0)
    data = data[:, mask]

    layers = torch.arange(0, model.n_layers, device=device).unsqueeze(-1)

    _, indices = (probs[:, mask] * layers).sum(0).sort(descending=True)

    return data[:, indices]


def get_heatmap_filename(repo_id: str, mode: str) -> str:
    return f"heatmap_prompt_{mode}_{repo_id.split('/')[-1]}.pdf"


def sweep(
    config: Config, device: torch.device | str, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    norm = None if config.mode == "probs" else PowerNorm(config.gamma)
    for repo_id in config.repo_ids(transformer=True, tuned_lens=config.tuned_lens):
        data = get_heatmap_data(config, repo_id, device)
        save_heatmap(
            data.cpu(),
            os.path.join(out, get_heatmap_filename(repo_id, config.mode)),
            norm=norm,
        )


if __name__ == "__main__":
    sweep(parse(Config), get_device())
