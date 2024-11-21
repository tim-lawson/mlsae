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


def get_pairwise_sims(x: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    _, n_elements = x.shape
    cos_sim = torch.empty((n_elements * (n_elements - 1)) // 2, device=x.device)
    idx = 0
    for i in range(0, n_elements, chunk_size):
        chunk_i_end = min(i + chunk_size, n_elements)
        chunk_i = x[:, i:chunk_i_end]
        for j in range(i, n_elements, chunk_size):
            if j < i:
                continue
            chunk_j_end = min(j + chunk_size, n_elements)
            chunk_j = x[:, j:chunk_j_end]
            chunk_cos_sim = torch.mm(chunk_i.T, chunk_j)
            if i == j:
                triu_indices = torch.triu_indices(
                    chunk_i_end - i, chunk_j_end - j, offset=1
                )
                chunk_cos_sim = chunk_cos_sim[triu_indices[0], triu_indices[1]]
            else:
                chunk_cos_sim = chunk_cos_sim.view(-1)
            next_idx = idx + chunk_cos_sim.shape[0]
            cos_sim[idx:next_idx] = chunk_cos_sim
            idx = next_idx
    return cos_sim


@torch.no_grad()
def get_wdec_sims(
    repo_id: str,
    chunk_size: int = 1024,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    mlsae = MLSAETransformer.from_pretrained(repo_id).to(device).autoencoder
    W_dec = normalize(mlsae.decoder.weight.detach())

    _, n_latents = W_dec.shape
    cos_sim = torch.empty((n_latents * (n_latents - 1)) // 2, device=device)

    idx = 0
    for i in range(0, n_latents, chunk_size):
        chunk_i_end = min(i + chunk_size, n_latents)
        chunk_i = W_dec[:, i:chunk_i_end]
        for j in range(i, n_latents, chunk_size):
            if j < i:
                continue
            chunk_j_end = min(j + chunk_size, n_latents)
            chunk_j = W_dec[:, j:chunk_j_end]
            chunk_cos_sim = torch.mm(chunk_i.T, chunk_j)
            if i == j:
                triu_indices = torch.triu_indices(
                    chunk_i_end - i, chunk_j_end - j, offset=1
                )
                chunk_cos_sim = chunk_cos_sim[triu_indices[0], triu_indices[1]]
            else:
                chunk_cos_sim = chunk_cos_sim.view(-1)
            next_idx = idx + chunk_cos_sim.shape[0]
            cos_sim[idx:next_idx] = chunk_cos_sim
            idx = next_idx
    return cos_sim


def main(
    config: SweepConfig, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    for repo_id in config.repo_ids(transformer=True):
        mlsae = MLSAETransformer.from_pretrained(repo_id).to(device).autoencoder

        Wdec_real = normalize(mlsae.decoder.weight.detach())
        Wdec_fake = normalize(torch.normal(0, 1, Wdec_real.shape, device=device))

        values_real = get_pairwise_sims(Wdec_real).cpu().numpy()
        values_fake = get_pairwise_sims(Wdec_fake).cpu().numpy()

        hist_real, bins = np.histogram(values_real, bins=200, range=(-1, 1))
        hist_real = np.append(hist_real, 0)
        hist_fake, _ = np.histogram(values_fake, bins=200, range=(-1, 1))
        hist_fake = np.append(hist_fake, 0)

        pd.DataFrame({"bin": bins, "real": hist_real, "fake": hist_fake}).to_csv(
            os.path.join(out, f"wdec_sim_{repo_id.split('/')[-1]}.csv"), index=False
        )


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
