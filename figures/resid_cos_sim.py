import math
import os

import einops
import pandas as pd
import torch
from simple_parsing import parse
from tqdm import tqdm

from mlsae.model import Transformer
from mlsae.model.data import get_train_dataloader
from mlsae.trainer import RunConfig, initialize
from mlsae.utils import get_device, normalize


class VarianceMetric:
    def __init__(
        self, size: tuple[int, ...] = (1,), device: torch.device | str = "cpu"
    ) -> None:
        self.count = 0
        self.mean = torch.zeros(size, device=device)
        self.squared = torch.zeros(size, device=device)

    def update(self, x: torch.Tensor) -> None:
        self.count += x.shape[0]
        delta = x - self.mean
        self.mean += torch.sum(delta, dim=0) / self.count
        self.squared += torch.sum(delta * delta, dim=0)

    def compute(self) -> dict[str, torch.Tensor]:
        var = self.squared / (self.count - 1)
        std = var.sqrt()
        sem = std / math.sqrt(self.count)
        return dict(mean=self.mean, var=var, std=std, sem=sem)


@torch.no_grad()
def get_resid_cos_sim(
    config: RunConfig, device: torch.device | str = "cpu"
) -> list[dict[str, float]]:
    transformer = Transformer(
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.autoencoder.skip_special_tokens,
        layers=config.layers,
        device=device,
    )
    transformer.model.to(device)  # type: ignore

    dataloader = get_train_dataloader(
        config.data.path,
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
    )

    resid_mean = [
        VarianceMetric(size=(transformer.config.hidden_size,), device=device)
        for _ in range(transformer.n_layers)
    ]
    resid_cos_sim = [
        VarianceMetric(size=(1,), device=device)
        for _ in range(transformer.n_layers - 1)
    ]

    # First, compute the mean residual stream activation vectors over the dataset
    # https://www.lesswrong.com/s/6njwz6XdSYwNhtsCJ/p/eLNo7b56kQQerCzp2
    for i, batch in tqdm(enumerate(dataloader), total=config.data.max_steps):
        x = transformer.forward(batch["input_ids"].to(device))
        x = einops.rearrange(x, "l b p i -> l (b p) i")
        for layer in range(transformer.n_layers - 1):
            resid_mean[layer].update(x[layer, ...])
        if i > config.data.max_steps:
            break

    resid_mean = [metric.compute() for metric in resid_mean]
    resid_mean = torch.stack([metric["mean"] for metric in resid_mean])  # l i
    assert resid_mean.shape == (transformer.n_layers, transformer.config.hidden_size)

    # Then, compute the mean cosine similarities between centered residual stream
    # activation vectors at adjacent layers
    for i, batch in tqdm(enumerate(dataloader), total=config.data.max_steps):
        x = transformer.forward(batch["input_ids"].to(device))
        x = einops.rearrange(x, "l b p i -> l (b p) i")
        x = x - resid_mean.unsqueeze(1)
        x = normalize(x, -1)
        for layer in range(transformer.n_layers - 1):
            cos_sim = einops.einsum(x[layer], x[layer + 1], "bp i, bp i -> bp")
            resid_cos_sim[layer].update(cos_sim.flatten())
        if i > config.data.max_steps:
            break

    data = [metric.compute() for metric in resid_cos_sim]
    return [{k: v.item() for k, v in layer.items()} for layer in data]


def main(config: RunConfig, device: torch.device | str = "cpu") -> None:
    data = get_resid_cos_sim(config, device)
    filename = f"resid_cos_sim_{config.model_name.split('/')[-1]}.csv"
    df = pd.DataFrame(data)
    df.index.name = "start_at_layer"
    df.to_csv(os.path.join("out", filename))


if __name__ == "__main__":
    config = parse(RunConfig)
    device = get_device()
    initialize(config.seed)
    main(config, device)
