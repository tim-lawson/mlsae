import os
from dataclasses import dataclass

from figures import heatmap_aggregate, heatmap_prompt, mmcs
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
pythia_410m = "EleutherAI/pythia-410m-deduped"

expansion_factor = 64
k = 32


@dataclass
class Sweep(SweepConfig):
    id: str | None = None
    """The identifier to use for filenames."""


@dataclass
class Config:
    out: str
    """The directory to save the results to."""

    heatmap_aggregate: bool = True
    heatmap_prompt: bool = True
    mmcs: bool = True


def main(sweeps: list[Sweep]) -> None:
    device = get_device()
    config = Config(out=".out")
    os.makedirs(config.out, exist_ok=True)

    for sweep in sweeps:
        _ = sweep.__dict__.pop("id")
        sweep_dict = sweep.__dict__

        for mode in ["probs"]:
            if config.heatmap_aggregate:
                heatmap_aggregate_config = heatmap_aggregate.Config(
                    **sweep_dict, mode=mode
                )
                heatmap_aggregate.main(heatmap_aggregate_config, device, config.out)

            if config.heatmap_prompt:
                heatmap_prompt_config = heatmap_prompt.Config(**sweep_dict, mode=mode)
                heatmap_prompt.main(heatmap_prompt_config, device, config.out)

        if config.mmcs:
            mmcs_config = mmcs.Config(**sweep_dict, filename=f"mmcs_{sweep.id}.csv")
            mmcs.main(mmcs_config, device, config.out)


sweeps: list[Sweep] = [
    Sweep(
        id="pythia-70m-deduped_layers",
        model_name=[pythia_70m],
        expansion_factor=[64],
        k=[32],
        tuned_lens=False,
        layers=list([[i] for i in range(6)]),
    ),
    Sweep(
        id="pythia-160m-deduped_layers",
        model_name=[pythia_160m],
        expansion_factor=[64],
        k=[32],
        tuned_lens=False,
        layers=list([[i] for i in range(11)]),
    ),
]

if __name__ == "__main__":
    main(sweeps)
