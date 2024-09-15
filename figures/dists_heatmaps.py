import os
from dataclasses import dataclass

from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.analysis.heatmaps import save_heatmap
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    tuned_lens: bool = False
    """Whether to apply a pretrained tuned lens before the encoder."""


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    for repo_id in config.repo_ids(tuned_lens=config.tuned_lens):
        dists = Dists.load(repo_id, device)
        filename = f"dists_heatmap_{repo_id.split('/')[-1]}.pdf"
        _, indices = dists.layer_mean.sort(descending=True)
        save_heatmap(
            dists.probs[:, indices].cpu(),
            os.path.join("out", filename),
            figsize=(5.5, 1.75),
        )
