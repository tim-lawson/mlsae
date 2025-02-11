import os
from dataclasses import dataclass

from simple_parsing import Serializable, parse

from figures import (
    embed_sim,
    entropy,
    heatmap_aggregate,
    heatmap_prompt,
    layer_hist,
    layer_sim,
    layer_std,
    mmcs,
    num_layers,
    scatter_freq,
    wdec_sim,
)
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
pythia_410m = "EleutherAI/pythia-410m-deduped"
pythia_1b = "EleutherAI/pythia-1b-deduped"
pythia_1_4b = "EleutherAI/pythia-1.4b-deduped"
gpt2_small = "openai-community/gpt2"
llama_3b = "meta-llama/Llama-3.2-3B"
gemma_2b = "google/gemma-2-2b"

expansion_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ks = [16, 32, 64, 128, 256, 512]


@dataclass
class FigureSweep(SweepConfig):
    id: str | None = None
    """The identifier to use for filenames."""

    enabled: bool = True
    """Whether to enable this sweep."""


@dataclass
class FigureConfig(Serializable):
    out: str = ".out"
    """The directory to save the results to."""

    # in the paper
    heatmap_aggregate: bool = False
    heatmap_prompt: bool = False
    mmcs: bool = False
    wdec_sim: bool = False
    num_layers: bool = False
    entropy: bool = False

    # not in the paper
    embed_sim: bool = False
    layer_std: bool = False
    layer_hist: bool = False
    layer_sim: bool = False
    scatter_freq: bool = False


def main(config: FigureConfig, sweeps: list[FigureSweep]) -> None:
    device = get_device()
    os.makedirs(config.out, exist_ok=True)

    for sweep in sweeps:
        id = sweep.__dict__.pop("id")
        print(id)
        enabled = sweep.__dict__.pop("enabled")
        if not enabled:
            continue
        sweep_dict = sweep.__dict__

        for mode in ["probs", "counts", "totals"]:
            gamma = 0.25

            if config.heatmap_aggregate:
                print(f"> heatmap_aggregate ({mode})")
                heatmap_aggregate_config = heatmap_aggregate.Config(
                    **sweep_dict, mode=mode, gamma=gamma
                )
                heatmap_aggregate.sweep(
                    heatmap_aggregate_config,
                    device,
                    os.path.join(config.out, f"heatmap_aggregate_{mode}"),
                )

            if config.heatmap_prompt:
                print(f"> heatmap_prompt ({mode})")
                heatmap_prompt_config = heatmap_prompt.Config(
                    **sweep_dict, mode=mode, gamma=gamma
                )
                heatmap_prompt.sweep(
                    heatmap_prompt_config,
                    device,
                    os.path.join(config.out, f"heatmap_prompt_{mode}"),
                )

        if config.mmcs:
            print("> mmcs")
            mmcs_config = mmcs.Config(**sweep_dict, filename=f"mmcs_{id}.csv")
            mmcs.main(mmcs_config, device, os.path.join(config.out, "mmcs"))

        if config.wdec_sim:
            print("> wdec_sim")
            wdec_sim.main(sweep, device, os.path.join(config.out, "wdec_sim"))

        if config.num_layers:
            print("> num_layers")
            for threshold in [1, 10, 100, 1000, 10000, 100000, 1000000]:
                num_layers_config = num_layers.Config(
                    **sweep_dict,
                    filename=f"num_layers_{id}_{threshold}.csv",
                    threshold=threshold,
                )
                num_layers.main(
                    num_layers_config, device, os.path.join(config.out, "num_layers")
                )

        if config.entropy:
            print("> entropy")
            entropy_config = entropy.Config(**sweep_dict, filename=f"entropy_{id}.csv")
            entropy.main(entropy_config, device, os.path.join(config.out, "entropy"))

        if config.embed_sim:
            print("> embed_sim")
            embed_sim_config = embed_sim.Config(
                **sweep_dict, filename=f"embed_sim_{id}.csv"
            )
            embed_sim.main(
                embed_sim_config, device, os.path.join(config.out, "embed_sim")
            )

        if config.layer_std:
            print("> layer_std")
            layer_std_config = layer_std.Config(
                **sweep_dict, filename=f"layer_std_{id}.csv"
            )
            layer_std.main(
                layer_std_config, device, os.path.join(config.out, "layer_std")
            )

        if config.layer_hist:
            print("> layer_hist")
            layer_hist_config = layer_hist.Config(**sweep_dict)
            layer_hist.main(
                layer_hist_config, device, os.path.join(config.out, "layer_hist")
            )

        if config.layer_sim:
            print("> layer_sim")
            layer_sim.main(sweep, device, os.path.join(config.out, "layer_sim"))

        if config.scatter_freq:
            print("> scatter_freq")
            scatter_freq.main(sweep, device, os.path.join(config.out, "scatter_freq"))


sweeps: list[FigureSweep] = [
    # Non-Pythia models for R = 64 and k = 32
    FigureSweep(
        id="other",
        enabled=True,
        model_name=[gpt2_small, llama_3b, gemma_2b],
        expansion_factor=[64],
        k=[32],
        tuned_lens=False,
    ),
    # Varying model for R = 64 and k = 32
    FigureSweep(
        id="model_name",
        enabled=False,
        model_name=[pythia_70m, pythia_160m, pythia_410m, pythia_1b, pythia_1_4b],
        expansion_factor=[64],
        k=[32],
        tuned_lens=False,
    ),
    # Varying model with tuned lens for R = 64 and k = 32
    FigureSweep(
        id="lens_model_name",
        enabled=False,
        model_name=[pythia_70m, pythia_160m, pythia_410m],
        expansion_factor=[64],
        k=[32],
        tuned_lens=True,
    ),
    # Varying R for Pythia-70m and k = 32
    FigureSweep(
        id="pythia-70m-deduped_expansion_factor",
        enabled=False,
        model_name=[pythia_70m],
        expansion_factor=expansion_factors,
        k=[32],
        tuned_lens=False,
    ),
    # Varying k for Pythia-70m and R = 64
    FigureSweep(
        id="pythia-70m-deduped_k",
        enabled=False,
        model_name=[pythia_70m],
        expansion_factor=[64],
        k=ks,
        tuned_lens=False,
    ),
    # Varying R for Pythia-160m and k = 32
    FigureSweep(
        id="pythia-160m-deduped_expansion_factor",
        enabled=False,
        model_name=[pythia_160m],
        expansion_factor=expansion_factors,
        k=[32],
        tuned_lens=False,
    ),
    # Varying k for Pythia-160m and R = 64
    FigureSweep(
        id="pythia-160m-deduped_k",
        enabled=False,
        model_name=[pythia_160m],
        expansion_factor=[64],
        k=ks,
        tuned_lens=False,
    ),
    # Varying R for Pythia-70m with tuned lens and k = 32
    FigureSweep(
        id="pythia-70m-deduped_lens_expansion_factor",
        enabled=False,
        model_name=[pythia_70m],
        expansion_factor=expansion_factors,
        k=[32],
        tuned_lens=True,
    ),
    # Varying k for Pythia-70m with tuned lens and R = 64
    FigureSweep(
        id="pythia-70m-deduped_lens_k",
        enabled=False,
        model_name=[pythia_70m],
        expansion_factor=[64],
        k=ks,
        tuned_lens=True,
    ),
]

if __name__ == "__main__":
    main(parse(FigureConfig), sweeps)
