import os
from dataclasses import dataclass

from figures import (
    embed_sim,
    entropy,
    heatmap_aggregate,
    heatmap_prompt,
    layer_hist,
    layer_sim,
    layer_std,
    mmcs,
    temp_scatter,
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
class FigureSweep(SweepConfig):
    id: str | None = None
    """The identifier to use for filenames."""

    enabled: bool = True
    """Whether to enable this sweep."""


@dataclass
class FigureConfig:
    out: str
    """The directory to save the results to."""

    heatmap_aggregate: bool = True
    heatmap_prompt: bool = True
    mmcs: bool = True

    embed_sim: bool = False
    layer_std: bool = False
    layer_hist: bool = False
    layer_sim: bool = False
    heatmap_freq: bool = False
    entropy: bool = False


def main(sweeps: list[FigureSweep]) -> None:
    device = get_device()
    config = FigureConfig(out=".out")
    os.makedirs(config.out, exist_ok=True)

    for sweep in sweeps:
        if not sweep.enabled:
            continue
        _ = sweep.__dict__.pop("id")
        _ = sweep.__dict__.pop("enabled")
        sweep_dict = sweep.__dict__

        for mode in ["probs", "counts", "totals"]:
            if config.heatmap_aggregate:
                heatmap_aggregate_config = heatmap_aggregate.Config(
                    **sweep_dict, mode=mode
                )
                heatmap_aggregate.sweep(heatmap_aggregate_config, device, config.out)

            if config.heatmap_prompt:
                heatmap_prompt_config = heatmap_prompt.Config(**sweep_dict, mode=mode)
                heatmap_prompt.main(heatmap_prompt_config, device, config.out)

        if config.mmcs:
            mmcs_config = mmcs.Config(**sweep_dict, filename=f"mmcs_{sweep.id}.csv")
            mmcs.main(mmcs_config, device, config.out)

        if config.embed_sim:
            embed_sim_config = embed_sim.Config(
                **sweep_dict, filename=f"embed_sim_{sweep.id}.csv"
            )
            embed_sim.main(embed_sim_config, device, config.out)

        if config.layer_std:
            layer_std_config = layer_std.Config(
                **sweep_dict, filename=f"layer_std_{sweep.id}.csv"
            )
            layer_std.main(layer_std_config, device, config.out)

        if config.layer_hist:
            layer_hist_config = layer_hist.Config(**sweep_dict)
            layer_hist.main(layer_hist_config, device, config.out)

        if config.layer_sim:
            layer_sim.main(sweep, device, config.out)

        if config.heatmap_freq:
            temp_scatter.main(sweep, device, config.out)

        if config.entropy:
            entropy.main(sweep, device, config.out)


sweeps: list[FigureSweep] = [
    # Varying model for R = 64 and k = 32
    FigureSweep(
        id="model_name",
        model_name=[pythia_70m, pythia_160m, pythia_410m, pythia_1b],
        expansion_factor=[64],
        k=[32],
        tuned_lens=False,
    ),
    # Varying model with tuned lens for R = 64 and k = 32
    FigureSweep(
        id="lens_model_name",
        model_name=[pythia_70m, pythia_160m, pythia_410m],
        expansion_factor=[64],
        k=[32],
        tuned_lens=True,
    ),
    # Varying R for Pythia-70m and k = 32
    FigureSweep(
        id="pythia-70m-deduped_expansion_factor",
        model_name=[pythia_70m],
        expansion_factor=expansion_factors,
        k=[32],
        tuned_lens=False,
    ),
    # Varying k for Pythia-70m and R = 64
    FigureSweep(
        id="pythia-70m-deduped_k",
        model_name=[pythia_70m],
        expansion_factor=[64],
        k=ks,
        tuned_lens=False,
    ),
    # Varying R for Pythia-160m and k = 32
    FigureSweep(
        id="pythia-160m-deduped_expansion_factor",
        model_name=[pythia_160m],
        expansion_factor=expansion_factors,
        k=[32],
        tuned_lens=False,
    ),
    # Varying k for Pythia-160m and R = 64
    FigureSweep(
        id="pythia-160m-deduped_k",
        model_name=[pythia_160m],
        expansion_factor=[64],
        k=ks,
        tuned_lens=False,
    ),
    # Varying R for Pythia-70m with tuned lens and k = 32
    FigureSweep(
        id="pythia-70m-deduped_lens_expansion_factor",
        model_name=[pythia_70m],
        expansion_factor=expansion_factors,
        k=[32],
        tuned_lens=True,
    ),
    # Varying k for Pythia-70m with tuned lens and R = 64
    FigureSweep(
        id="pythia-70m-deduped_lens_k",
        model_name=[pythia_70m],
        expansion_factor=[64],
        k=ks,
        tuned_lens=True,
    ),
]

if __name__ == "__main__":
    main(sweeps)
