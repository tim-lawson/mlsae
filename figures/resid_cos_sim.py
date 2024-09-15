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
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.squared = 0.0

    def update(self, x: torch.Tensor) -> None:
        self.count += x.shape[0]
        delta = x - self.mean
        self.mean += torch.sum(delta).item() / self.count
        self.squared += torch.sum(delta * delta).item()

    def compute(self) -> dict[str, float]:
        var = self.squared / (self.count - 1)
        std = math.sqrt(var)
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

    metrics = [VarianceMetric() for _ in range(transformer.n_layers - 1)]

    for i, batch in tqdm(enumerate(dataloader), total=config.data.max_steps):
        x = transformer.forward(batch["input_ids"].to(device))
        x = normalize(einops.rearrange(x, "l b p i -> l (b p) i"), -1)
        for layer in range(transformer.n_layers - 1):
            cos_sim = einops.einsum(x[layer], x[layer + 1], "bp i, bp i -> bp")
            metrics[layer].update(cos_sim.flatten())

        if i > config.data.max_steps:
            break
    return [metric.compute() for metric in metrics]


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
