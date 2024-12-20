import math
import os
from dataclasses import dataclass

import numpy
import pandas as pd
import torch
from simple_parsing import parse

from figures.test import parse_mlsae_repo_id
from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    filename: str = "entropy.csv"
    """The filename to save the results to."""


def main(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for repo_id in config.repo_ids(transformer=True, tuned_lens=config.tuned_lens):
        dists = Dists.load(repo_id, device)
        values = dists.entropies
        values = values[~torch.isnan(values)]

        repo_id = repo_id.split("/")[-1]
        model_name, expansion_factor, k, tuned_lens = parse_mlsae_repo_id(repo_id)

        rows.append(
            {
                "model_name": model_name,
                "n_layers": dists.n_layers,
                "n_latents": dists.n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "tuned_lens": tuned_lens,
                "mean": values.mean().item(),
                "var": values.var().item(),
                "std": values.std().item(),
                "sem": values.std().item() / values.size(0) ** 0.5,
                "rel": values.mean().item() / math.log(dists.n_layers),
            }
        )

        hist, bins = numpy.histogram(
            values.cpu().numpy(),
            bins=dists.n_latents // expansion_factor,
            range=(0, math.log(dists.n_layers)),
        )
        hist = numpy.append(hist, 0)
        pd.DataFrame({"bins": bins, "hist": hist}).to_csv(
            os.path.join(out, f"entropy_{repo_id.split("/")[-1]}.csv"), index=False
        )

    pd.DataFrame(rows).to_csv(os.path.join(out, config.filename), index=False)


if __name__ == "__main__":
    main(parse(Config), get_device())
