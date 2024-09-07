import os

from simple_parsing import parse

from mlsae.analysis.dists import Dists
from mlsae.analysis.heatmaps import save_heatmap
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device

if __name__ == "__main__":
    device = get_device()
    for repo_id in parse(SweepConfig).repo_ids():
        dists = Dists.load(repo_id, device)
        filename = f"dists_heatmap_{repo_id.split('/')[-1]}.pdf"
        _, indices = dists.layer_mean.sort(descending=True)
        save_heatmap(dists.probs[:, indices].cpu(), os.path.join("out", filename))
