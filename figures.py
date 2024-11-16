import os
from dataclasses import dataclass

from figures import (
    embed_sim,
    heatmap_aggregate,
    heatmap_prompt,
    layer_hist,
    layer_sim,
    layer_std,
    mmcs,
)
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
pythia_410m = "EleutherAI/pythia-410m-deduped"
pythia_1b = "EleutherAI/pythia-1b-deduped"

expansion_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ks = [16, 32, 64, 128, 256, 512]


@dataclass
class Config:
    out: str
    """The directory to save the results to."""

    heatmap_aggregate: bool = True
    heatmap_prompt: bool = True
    mmcs: bool = True

    embed_sim: bool = False
    layer_hist: bool = False
    layer_sim: bool = False
    layer_std: bool = False


def main(sweeps: list[tuple[str, SweepConfig]]) -> None:
    device = get_device()
    config = Config(out=".out")
    os.makedirs(config.out, exist_ok=True)

    for id, sweep_config in sweeps:
        sweep = sweep_config.__dict__

        for mode in ["probs"]:
            if config.heatmap_aggregate:
                heatmap_aggregate_config = heatmap_aggregate.Config(**sweep, mode=mode)
                heatmap_aggregate.main(heatmap_aggregate_config, device, config.out)

            if config.heatmap_prompt:
                heatmap_prompt_config = heatmap_prompt.Config(**sweep, mode=mode)
                heatmap_prompt.main(heatmap_prompt_config, device, config.out)

        if config.mmcs:
            mmcs_config = mmcs.Config(**sweep, filename=f"mmcs_{id}.csv")
            mmcs.main(mmcs_config, device, config.out)

        if config.embed_sim:
            embed_sim_config = embed_sim.Config(**sweep, filename=f"embed_sim_{id}.csv")
            embed_sim.main(embed_sim_config, device, config.out)

        if config.layer_hist:
            layer_hist_config = layer_hist.Config(**sweep)
            layer_hist.main(layer_hist_config, device, config.out)

        if config.layer_sim:
            layer_sim.main(sweep_config, device, config.out)

        if config.layer_std:
            layer_std_config = layer_std.Config(**sweep, filename=f"layer_std_{id}.csv")
            layer_std.main(layer_std_config, device, config.out)


sweeps: list[tuple[str, SweepConfig]] = [
    # Varying model for R = 64 and k = 32
    (
        "model_name",
        SweepConfig(
            model_name=[pythia_70m, pythia_160m, pythia_410m, pythia_1b],
            expansion_factor=[64],
            k=[32],
            tuned_lens=False,
        ),
    ),
    # Varying model with tuned lens for R = 64 and k = 32
    (
        "lens_model_name",
        SweepConfig(
            model_name=[pythia_70m, pythia_160m, pythia_410m],
            expansion_factor=[64],
            k=[32],
            tuned_lens=True,
        ),
    ),
    # Varying R for Pythia-70m and k = 32
    (
        "pythia-70m-deduped_expansion_factor",
        SweepConfig(
            model_name=[pythia_70m],
            expansion_factor=expansion_factors,
            k=[32],
            tuned_lens=False,
        ),
    ),
    # Varying k for Pythia-70m and R = 64
    (
        "pythia-70m-deduped_k",
        SweepConfig(
            model_name=[pythia_70m],
            expansion_factor=[64],
            k=ks,
            tuned_lens=False,
        ),
    ),
    # Varying R for Pythia-160m and k = 32
    (
        "pythia-160m-deduped_expansion_factor",
        SweepConfig(
            model_name=[pythia_160m],
            expansion_factor=expansion_factors,
            k=[32],
            tuned_lens=False,
        ),
    ),
    # Varying k for Pythia-160m and R = 64
    (
        "pythia-160m-deduped_k",
        SweepConfig(
            model_name=[pythia_160m],
            expansion_factor=[64],
            k=ks,
            tuned_lens=False,
        ),
    ),
    # Varying R for Pythia-70m with tuned lens and k = 32
    (
        "pythia-70m-deduped_lens_expansion_factor",
        SweepConfig(
            model_name=[pythia_70m],
            expansion_factor=expansion_factors,
            k=[32],
            tuned_lens=True,
        ),
    ),
    # Varying k for Pythia-70m with tuned lens and R = 64
    (
        "pythia-70m-deduped_lens_k",
        SweepConfig(
            model_name=[pythia_70m],
            expansion_factor=[64],
            k=ks,
            tuned_lens=True,
        ),
    ),
]

sweeps = sweeps[:1]
if __name__ == "__main__":
    main(sweeps)
