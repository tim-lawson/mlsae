import math
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


def get_positive(
    shape: torch.Size, n_repeats: int, std: float, device: torch.device
) -> torch.Tensor:
    positive = torch.normal(
        0,
        1,
        (shape[0], math.ceil(shape[1] / n_repeats)),
        device=device,
    ).repeat(1, n_repeats)[:, : shape[1]]
    positive += torch.normal(0, std, positive.shape, device=device)
    return normalize(positive)


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


def get_hist(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    hist, bins = np.histogram(
        get_pairwise_sims(normalize(x)).cpu().numpy(), bins=200, range=(-1, 1)
    )
    hist = np.append(hist, 0)
    return bins, hist


def main(
    config: SweepConfig, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    for repo_id in config.repo_ids(transformer=True, tuned_lens=config.tuned_lens):
        model = MLSAETransformer.from_pretrained(repo_id).to(device)
        autoencoder = model.autoencoder
        shape = autoencoder.decoder.weight.shape

        bins, actual = get_hist(autoencoder.decoder.weight.detach())

        # Negative control: n_latents IID Gaussian vectors
        _, negative = get_hist(torch.normal(0, 1, shape, device=device))

        # Positive control: n_latents // n_layers IID Gaussian vectors, repeated
        # n_layers times with a small amount of noise
        _, positive1 = get_hist(get_positive(shape, model.n_layers, 0.1, device))
        _, positive2 = get_hist(get_positive(shape, model.n_layers, 0.2, device))
        _, positive3 = get_hist(get_positive(shape, model.n_layers, 0.3, device))
        _, positive4 = get_hist(get_positive(shape, model.n_layers, 0.4, device))
        _, positive5 = get_hist(get_positive(shape, model.n_layers, 0.5, device))
        _, positive6 = get_hist(get_positive(shape, model.n_layers, 0.6, device))
        _, positive7 = get_hist(get_positive(shape, model.n_layers, 0.7, device))
        _, positive8 = get_hist(get_positive(shape, model.n_layers, 0.8, device))
        _, positive9 = get_hist(get_positive(shape, model.n_layers, 0.9, device))
        _, positive10 = get_hist(get_positive(shape, model.n_layers, 1.0, device))
        _, positive11 = get_hist(get_positive(shape, model.n_layers, 1.1, device))
        _, positive12 = get_hist(get_positive(shape, model.n_layers, 1.2, device))
        _, positive13 = get_hist(get_positive(shape, model.n_layers, 1.3, device))
        _, positive14 = get_hist(get_positive(shape, model.n_layers, 1.4, device))
        _, positive15 = get_hist(get_positive(shape, model.n_layers, 1.5, device))
        _, positive16 = get_hist(get_positive(shape, model.n_layers, 1.6, device))
        _, positive17 = get_hist(get_positive(shape, model.n_layers, 1.7, device))
        _, positive18 = get_hist(get_positive(shape, model.n_layers, 1.8, device))
        _, positive19 = get_hist(get_positive(shape, model.n_layers, 1.9, device))
        _, positive20 = get_hist(get_positive(shape, model.n_layers, 2.0, device))

        pd.DataFrame(
            {
                "bins": bins,
                "actual": actual,
                "negative": negative,
                "positive1": positive1,
                "positive2": positive2,
                "positive3": positive3,
                "positive4": positive4,
                "positive5": positive5,
                "positive6": positive6,
                "positive7": positive7,
                "positive8": positive8,
                "positive9": positive9,
                "positive10": positive10,
                "positive11": positive11,
                "positive12": positive12,
                "positive13": positive13,
                "positive14": positive14,
                "positive15": positive15,
                "positive16": positive16,
                "positive17": positive17,
                "positive18": positive18,
                "positive19": positive19,
                "positive20": positive20,
            }
        ).to_csv(
            os.path.join(out, f"wdec_sim_{repo_id.split('/')[-1]}.csv"), index=False
        )


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
