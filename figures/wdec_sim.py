import os

import numpy as np
import pandas as pd
import torch
from simple_parsing import parse

from mlsae.model import MLSAETransformer
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device, normalize


def get_filename(repo_id: str, mode: str) -> str:
    return f"wdec_sim_{mode}_{repo_id.split('/')[-1]}.csv"


@torch.no_grad()
def get_wdec_sims(
    repo_id: str,
    max_latents: int = 16384,
    chunk_size: int = 1024,
    device: torch.device | str = "cpu",
):
    mlsae = MLSAETransformer.from_pretrained(repo_id).to(device).autoencoder
    W_dec = normalize(mlsae.decoder.weight.detach())

    _, n_latents = W_dec.shape
    if n_latents < max_latents:
        # Compute the full cosine similarity matrix
        cos_sim = torch.triu(torch.mm(W_dec.T, W_dec), diagonal=1)
    else:
        # Compute the maximum cosine similarities in chunks
        cos_sim = torch.zeros(n_latents * (n_latents + 1) // 2, device=device)
        for i in range(0, n_latents, chunk_size):
            chunk_W_dec = W_dec[:, i : i + chunk_size]
            chunk_cos_sim = torch.mm(W_dec.T, chunk_W_dec)
            mask = torch.ones_like(chunk_cos_sim, dtype=torch.bool, device=device)
            mask[: i + chunk_size, :] = torch.triu(
                mask[: i + chunk_size, :], diagonal=1
            )
            chunk_cos_sim = chunk_cos_sim.masked_fill(~mask, float("-inf"))
            cos_sim[i : i + chunk_size] = torch.max(chunk_cos_sim, dim=0).values

    return cos_sim[*torch.triu_indices(*cos_sim.shape, offset=1)].cpu()


def main(
    config: SweepConfig, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    for repo_id in config.repo_ids(transformer=True):
        values = get_wdec_sims(repo_id, device=device)
        hist, bins = np.histogram(values, bins=100, range=(-1, 1))
        hist = np.append(hist, 0)  # bins has one more element
        pd.DataFrame({"layer": bins, "density": hist}).to_csv(
            os.path.join(out, f"wdec_sim_{repo_id.split('/')[-1]}.csv"), index=False
        )


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
