import os

import numpy as np
import pandas as pd
from simple_parsing import parse

from mlsae.analysis.dists import get_dists
from mlsae.trainer import SweepConfig, initialize
from mlsae.utils import get_device, get_repo_id

if __name__ == "__main__":
    device = get_device()
    config = parse(SweepConfig)
    initialize(config.seed)

    rows: list[dict[str, str | int | float]] = []
    for model_name, expansion_factor, k in config:
        repo_id = get_repo_id(model_name, expansion_factor, k, True)
        dist = get_dists(repo_id, device)
        layer_std = dist.layer_std.cpu().numpy()
        rows.append(
            {
                "model_name": model_name,
                "n_layers": dist.n_layers,
                "n_latents": dist.n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "mean": np.nanmean(layer_std).item(),
                "var": np.nanvar(layer_std).item(),
                "std": np.nanstd(layer_std).item(),
                "sem": np.nanstd(layer_std).item() / np.sqrt(len(layer_std)),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join("out", "dists_layer_std.csv"), index=False)
