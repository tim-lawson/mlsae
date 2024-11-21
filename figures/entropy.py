import math
import os

import numpy
import pandas as pd
import torch
from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


def main(
    config: SweepConfig, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    for repo_id in config.repo_ids():
        filename = f"entropy_{repo_id.split("/")[-1]}.csv"

        dists = Dists.load(repo_id, device)

        a = dists.entropies
        a = a[~torch.isnan(a)]
        a = a.cpu().numpy()

        hist, bins = numpy.histogram(
            a, bins=32, range=(0, math.log(dists.n_layers)), density=True
        )
        hist = numpy.append(hist, 0)  # bins has one more element

        pd.DataFrame({"bins": bins, "hist": hist}).to_csv(
            os.path.join(out, filename), index=False
        )


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
