import os
from dataclasses import dataclass

import pandas as pd
from simple_parsing import parse

from mlsae.analysis.dists import Dists, get_stats
from mlsae.trainer import SweepConfig, initialize
from mlsae.utils import get_device, get_repo_id


@dataclass
class Config(SweepConfig):
    filename: str = "dists_layer_std.csv"
    """The name of the file to save the results to."""


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)

    rows: list[dict[str, str | int | float]] = []
    for model_name, expansion_factor, k in config:
        repo_id = get_repo_id(model_name, expansion_factor, k, True, config.tuned_lens)
        dists = Dists.load(repo_id, device)
        stats = get_stats(dists.layer_std)
        rows.append(
            {
                "model_name": model_name,
                "n_layers": dists.n_layers,
                "n_latents": dists.n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "tuned_lens": config.tuned_lens,
                **stats,
                **{f"{k}_rel": v / dists.n_layers for k, v in stats.items()},
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join("out", config.filename), index=False)
