import math
import os

import einops
import pandas as pd
import torch
from simple_parsing import parse
from tqdm import tqdm
from tuned_lens import TunedLens

from mlsae.model import Transformer
from mlsae.model.data import get_test_dataloader
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
def save_resid_cos_sim(config: RunConfig, device: torch.device | str = "cpu") -> None:
    transformer = Transformer(
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.autoencoder.skip_special_tokens,
        layers=config.layers,
        device=device,
    )
    transformer.model.to(device)  # type: ignore

    lens = (
        TunedLens.from_model_and_pretrained(
            transformer.model,
            transformer.model_name,
            map_location=device,
        )
        if config.autoencoder.tuned_lens
        else None
    )
    lens_name = "lens_" if lens is not None else ""

    def forward_lens(inputs: torch.Tensor) -> torch.Tensor:
        if lens is None:
            return inputs
        lens.to(inputs.device)
        for layer in range(transformer.n_layers):
            inputs[layer, ...] = lens.transform_hidden(inputs[layer, ...], layer)
        return inputs

    dataloader = get_test_dataloader(
        config.model_name,
        config.data.max_length,
        config.data.batch_size,
    )

    model_name = config.model_name.split("/")[-1]

    means = [
        VarianceMetric(size=(transformer.config.hidden_size,), device=device)
        for _ in range(transformer.n_layers)
    ]
    l2_norms = [
        VarianceMetric(size=(1,), device=device) for _ in range(transformer.n_layers)
    ]
    cos_sims = [
        VarianceMetric(size=(1,), device=device)
        for _ in range(transformer.n_layers - 1)
    ]

    # First, compute the mean residual stream activation vectors over the dataset
    # https://www.lesswrong.com/s/6njwz6XdSYwNhtsCJ/p/eLNo7b56kQQerCzp2
    for i, batch in tqdm(enumerate(dataloader), total=config.data.max_steps):
        x = forward_lens(transformer.forward(batch["input_ids"].to(device)))
        x = einops.rearrange(x, "l b p i -> l (b p) i")
        for layer in range(transformer.n_layers):
            means[layer].update(x[layer, ...])
            l2_norms[layer].update(x[layer, ...].norm(dim=-1))
        if i > config.data.max_steps:
            break

    l2_norms = [metric.compute() for metric in l2_norms]
    df = pd.DataFrame([{k: v.item() for k, v in layer.items()} for layer in l2_norms])
    df.index.name = "layer"
    df.to_csv(os.path.join("out", f"resid_l2_norm_{lens_name}{model_name}.csv"))

    means = [metric.compute() for metric in means]
    means = torch.stack([metric["mean"] for metric in means])  # l i
    assert means.shape == (transformer.n_layers, transformer.config.hidden_size)

    # Then, compute the mean cosine similarities between centered residual stream
    # activation vectors at adjacent layers
    for i, batch in tqdm(enumerate(dataloader), total=config.data.max_steps):
        x = forward_lens(transformer.forward(batch["input_ids"].to(device)))
        x = einops.rearrange(x, "l b p i -> l (b p) i")
        x = x - means.unsqueeze(1)
        x = normalize(x, -1)
        for layer in range(transformer.n_layers - 1):
            cos_sim = einops.einsum(x[layer], x[layer + 1], "bp i, bp i -> bp")
            cos_sims[layer].update(cos_sim.flatten())
        if i > config.data.max_steps:
            break

    data = [metric.compute() for metric in cos_sims]
    data = [{k: v.item() for k, v in layer.items()} for layer in data]

    df = pd.DataFrame(data)
    df.index.name = "start_at_layer"
    df.to_csv(os.path.join("out", f"resid_cos_sim_{lens_name}{model_name}.csv"))


if __name__ == "__main__":
    config = parse(RunConfig)
    device = get_device()
    initialize(config.seed)
    save_resid_cos_sim(config, device)
