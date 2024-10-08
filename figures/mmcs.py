import os
from dataclasses import dataclass

import pandas as pd
import torch
from simple_parsing import parse
from tqdm import tqdm

from mlsae.model import MLSAE
from mlsae.trainer import SweepConfig, initialize
from mlsae.utils import get_device, get_repo_id, normalize


@dataclass
class Config(SweepConfig):
    filename: str = "mmcs.csv"
    """The name of the file to save the results to."""


@torch.no_grad()
def get_max_cos_sim(
    model_name: str,
    expansion_factor: int,
    k: int,
    tuned_lens: bool,
    max_latents: int = 16384,
    chunk_size: int = 1024,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    repo_id = get_repo_id(model_name, expansion_factor, k, False, tuned_lens)
    mlsae = MLSAE.from_pretrained(repo_id).to(device)
    W_dec = normalize(mlsae.decoder.weight.detach())

    _, n_latents = W_dec.shape
    if n_latents < max_latents:
        # Compute the full cosine similarity matrix
        cos_sim = torch.triu(torch.mm(W_dec.T, W_dec), diagonal=1)
        max_cos_sim = cos_sim.max(dim=0).values
    else:
        # Compute the maximum cosine similarities in chunks
        max_cos_sim = torch.zeros(n_latents, device=device)
        for i in tqdm(range(0, n_latents, chunk_size), total=n_latents // chunk_size):
            chunk_W_dec = W_dec[:, i : i + chunk_size]
            chunk_cos_sim = torch.mm(W_dec.T, chunk_W_dec)
            mask = torch.ones_like(chunk_cos_sim, dtype=torch.bool, device=device)
            mask[: i + chunk_size, :] = torch.triu(
                mask[: i + chunk_size, :], diagonal=1
            )
            chunk_cos_sim = chunk_cos_sim.masked_fill(~mask, float("-inf"))
            chunk_max_cos_sim = torch.max(chunk_cos_sim, dim=0).values
            max_cos_sim[i : i + chunk_size] = torch.max(
                max_cos_sim[i : i + chunk_size], chunk_max_cos_sim
            )
    return max_cos_sim.cpu(), mlsae.n_latents


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)

    rows: list[dict[str, str | int | float]] = []
    for model_name, expansion_factor, k in config:
        max_cos_sim, n_latents = get_max_cos_sim(
            model_name, expansion_factor, k, config.tuned_lens, device=device
        )
        rows.append(
            {
                "model_name": model_name,
                "n_latents": n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "tuned_lens": config.tuned_lens,
                "mean": max_cos_sim.mean().item(),
                "var": max_cos_sim.var().item(),
                "std": max_cos_sim.std().item(),
                "sem": max_cos_sim.std().item() / max_cos_sim.size(0) ** 0.5,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join("out", config.filename), index=False)
