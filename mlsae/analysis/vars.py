from dataclasses import dataclass

import einops
import torch
from simple_parsing import Serializable, field, parse
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAETransformer, TopK
from mlsae.model.data import get_test_dataloader
from mlsae.trainer import initialize
from mlsae.utils import get_device


@dataclass
class Config(Serializable):
    repo_id: str
    """
    The name of a pretrained autoencoder and transformer from HuggingFace, or the path
    to a directory that contains them.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """The data configuration. Remember to set max_tokens to a reasonable value!"""

    seed: int = 42
    """The seed for global random state."""

    log_every_n_steps: int | None = 1
    """The number of steps between logging statistics."""

    push_to_hub: bool = True
    """Whether to push the dataset to HuggingFace."""


class Metric:
    def __init__(
        self, n_layers: int, n_latents: int, device: torch.device | str = "cpu"
    ) -> None:
        self.n_layers = n_layers
        self.n_latents = n_latents

        self.layers = torch.arange(n_layers, device=device).unsqueeze(1).unsqueeze(2)
        self.device = device

    # layer batch pos k
    def update(self, x: TopK) -> None:
        indices = einops.rearrange(x.indices, "l b p k -> l (b p) k")
        values = einops.rearrange(x.values, "l b p k -> l (b p) k")

        probs = torch.zeros(
            (self.n_layers, self.n_latents, indices.shape[1]), device=self.device
        )
        for layer in range(self.n_layers):
            probs[layer, ...].put_(indices[layer], values[layer], accumulate=True)
        probs /= probs.sum(dim=0, keepdim=True)

        exp_var_layer_latent = self.var(probs.mean(dim=1)).mean()
        exp_var_layer_token_latent = self.var(probs).mean()

        print(exp_var_layer_latent, exp_var_layer_token_latent)

    def var(self, probs: torch.Tensor) -> torch.Tensor:
        sum = (self.layers * probs).sum(dim=0)
        sum_squared = (self.layers**2 * probs).sum(dim=0)
        return sum_squared - sum**2

    def compute(self):
        pass


@torch.no_grad()
def get_variances(config: Config, device: torch.device | str = "cpu"):
    model = MLSAETransformer.from_pretrained(config.repo_id).to(device)

    dataloader = get_test_dataloader(
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
    )

    tokens_per_step = config.data.batch_size * config.data.max_length

    metric = Metric(model.n_layers, model.n_latents, device)

    for i, batch in enumerate(tqdm(dataloader, total=config.data.max_steps)):
        inputs = model.transformer.forward(batch["input_ids"].to(device))
        topk = model.autoencoder.encode(inputs).topk
        metric.update(topk)

        if config.log_every_n_steps is not None and i % config.log_every_n_steps == 0:
            print(metric.compute())

        if i > config.data.max_steps:
            break

    return metric.compute()


def save_variances(config: Config, device: torch.device | str = "cpu") -> None:
    variances = get_variances(config, device)


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)
    save_variances(config, device)
