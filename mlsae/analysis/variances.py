from dataclasses import dataclass

import einops
import pandas as pd
import torch
from simple_parsing import field, parse
from tqdm import tqdm

from mlsae.model.data import DataConfig, get_test_dataloader
from mlsae.model.decoder import scatter_topk
from mlsae.model.lightning import MLSAETransformer
from mlsae.trainer.config import SweepConfig, initialize
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    data: DataConfig = field(default_factory=DataConfig)
    """The data configuration. Remember to set max_tokens to a reasonable value!"""

    seed: int = 42
    """The seed for global random state."""

    filename: str = "variances.csv"
    """The name of the file to save the results to."""


class Metric:
    def __init__(
        self,
        n_layers: int,
        n_tokens: int,
        n_latents: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_layers = n_layers
        self.n_tokens = n_tokens
        self.n_latents = n_latents
        self.layers = torch.arange(self.n_layers, device=device)

        self.exp_var_l_f = []
        self.exp_var_l_tf = []
        self.var_l = []
        self.rel_var_f = []
        self.rel_var_t = []

    def var(self, x: torch.Tensor):
        layers = self.layers.view((self.n_layers, *([1] * (len(x.shape) - 1))))

        ell = (layers * x).sum(dim=0)
        ell_sq = ((layers**2) * x).sum(dim=0)
        return ell_sq - ell**2

    def update(self, latents: torch.Tensor):
        assert latents.shape == (self.n_layers, self.n_tokens, self.n_latents)

        probs = latents / latents.sum(dim=0)
        probs = probs.nan_to_num_(0.0)

        e_var_l_f = self.var(probs.mean(1)).mean()
        e_var_l_tf = self.var(probs).mean()
        var_l = self.var(probs.mean((1, 2)))

        self.exp_var_l_f.append(e_var_l_f)
        self.exp_var_l_tf.append(e_var_l_tf)
        self.var_l.append(var_l)
        self.rel_var_f.append(e_var_l_f / var_l)
        self.rel_var_t.append(e_var_l_tf / e_var_l_f)

    def compute(self) -> dict[str, float]:
        return dict(
            exp_var_l_f=torch.stack(self.exp_var_l_f).mean().item(),
            exp_var_l_tf=torch.stack(self.exp_var_l_tf).mean().item(),
            var_l=torch.stack(self.var_l).mean().item(),
            rel_var_f=torch.stack(self.rel_var_f).mean().item(),
            rel_var_t=torch.stack(self.rel_var_t).mean().item(),
        )


@torch.no_grad()
def get_variances(
    repo_id: str,
    max_length: int,
    batch_size: int,
    max_steps: float,
    device: torch.device | str = "cpu",
) -> dict:
    model = MLSAETransformer.from_pretrained(repo_id).to(device)

    dataloader = get_test_dataloader(model.model_name, max_length, batch_size)

    tokens_per_step = batch_size * max_length

    metric = Metric(model.n_layers, tokens_per_step, model.n_latents, device)

    for i, batch in enumerate(tqdm(dataloader, total=max_steps)):
        inputs = model.transformer.forward(batch["input_ids"].to(device))
        topk = model.autoencoder.encode(inputs).topk

        latents = scatter_topk(topk, model.n_latents)
        latents = einops.rearrange(latents, "l b t f -> l (b t) f")

        metric.update(latents)

        if i > max_steps:
            break

    return {
        "model_name": model.model_name,
        "n_layers": model.n_layers,
        "n_latents": model.n_latents,
        "expansion_factor": model.expansion_factor,
        "k": model.k,
        "step": i,
        "tokens": (i + 1) * tokens_per_step,
        **metric.compute(),
    }


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)

    rows: list[dict] = []
    for repo_id in config.repo_ids(transformer=True):
        row = get_variances(
            repo_id,
            config.data.max_length,
            config.data.batch_size,
            config.data.max_steps,
            device=device,
        )
        pd.DataFrame({k: [v] for k, v in row.items()}).to_csv(
            f"out/variances_{repo_id.split("/")[-1]}.csv", index=False
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"out/{config.filename}", index=False)
