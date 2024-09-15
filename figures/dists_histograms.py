import os
from dataclasses import dataclass

import numpy
import pandas as pd
import torch
from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig, initialize
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    noninteger: bool = False
    """Whether to plot the non-integer component of the center of mass."""

    tuned_lens: bool = False
    """Whether to apply a pretrained tuned lens before the encoder."""


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)

    for repo_id in config.repo_ids(tuned_lens=config.tuned_lens):
        dists = Dists.load(repo_id, device)
        values = dists.layer_mean[~torch.isnan(dists.layer_mean)].cpu().numpy()

        repo_id = repo_id.split("/")[-1]
        bins = 16 * dists.n_layers

        if config.noninteger:
            values = numpy.abs(values - numpy.round(values))
            range = (0, 0.5)
            filename = f"dists_histogram_noninteger_{repo_id}.csv"
        else:
            range = (0, dists.n_layers - 1)
            filename = f"dists_histogram_{repo_id}.csv"

        hist, bins = numpy.histogram(values, bins=bins, range=range, density=True)
        hist = numpy.append(hist, 0)  # bins has one more element
        pd.DataFrame({"layer": bins, "density": hist}).to_csv(
            os.path.join("out", filename), index=False
        )
