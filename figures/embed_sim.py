import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from simple_parsing import field, parse
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from mlsae.model import MLSAETransformer
from mlsae.trainer.config import SweepConfig
from mlsae.utils import get_device, get_repo_id, normalize


@dataclass
class Config(SweepConfig):
    filename: str = "embed_sim.csv"
    """The name of the file to save the results to."""

    latents: list[int] = field(default_factory=lambda: [])
    """The latent indices to find the most similar embeddings to."""

    n_embeds: int = 8
    """The number of most similar embeddings to save."""

    seed: int = 42
    """The seed for global random state."""


@torch.no_grad()
def get_similar_embeds(
    config: Config, repo_id: str, model_name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mlsae = MLSAETransformer.from_pretrained(repo_id).to(device).autoencoder
    W_dec = normalize(mlsae.decoder.weight)
    if len(config.latents) > 0:
        W_dec = W_dec[:, config.latents]
    latents = (
        config.latents if len(config.latents) > 0 else list(range(mlsae.n_latents))
    )

    def save_csv(topk: torch.return_types.topk, path: Path | str):
        rows = [
            {
                "latent": latent,
                "token": tokenizer.decode(topk.indices[embed_index, latent_index]),
                "sim": topk.values[embed_index, latent_index].detach().item(),
            }
            for latent_index, latent in enumerate(latents)
            for embed_index in range(config.n_embeds)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)

    model: GPTNeoXForCausalLM = GPTNeoXForCausalLM.from_pretrained(model_name)  # type: ignore
    embed_in = normalize(model.get_input_embeddings().weight.to(device), dim=1)
    embed_out = normalize(model.get_output_embeddings().weight.to(device), dim=1)

    topk_in = torch.topk(embed_in @ W_dec, k=config.n_embeds, dim=0)
    topk_out = torch.topk(embed_out @ W_dec, k=config.n_embeds, dim=0)

    repo_id = repo_id.split("/")[-1]
    save_csv(topk_in, os.path.join("out", f"embed_in_cos_sim_{repo_id}.csv"))
    save_csv(topk_out, os.path.join("out", f"embed_out_cos_sim_{repo_id}.csv"))

    return topk_in.values[0, :], topk_out.values[0, :]


def main(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for model_name, expansion_factor, k in config:
        repo_id = get_repo_id(
            model_name=model_name,
            expansion_factor=expansion_factor,
            k=k,
            tuned_lens=config.tuned_lens,
            transformer=False,
        )
        topk_in, topk_out = get_similar_embeds(config, repo_id, model_name, device)
        n_latents = topk_in.shape[0]
        rows.append(
            {
                "model_name": model_name,
                "n_latents": n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "tuned_lens": config.tuned_lens,
                "in_mean": topk_in.mean().item(),
                "in_var": topk_in.var().item(),
                "in_std": topk_in.std().item(),
                "in_sem": topk_in.std().item() / np.sqrt(n_latents),
                "out_mean": topk_out.mean().item(),
                "out_var": topk_out.var().item(),
                "out_std": topk_out.std().item(),
                "out_sem": topk_out.std().item() / np.sqrt(n_latents),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out, config.filename), index=False)


if __name__ == "__main__":
    main(parse(Config), get_device())
