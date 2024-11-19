import torch
from simple_parsing import parse
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAEConfig
from mlsae.trainer import RunConfig, SweepConfig, test
from mlsae.utils import get_device


def main(config: SweepConfig, device: torch.device | None = None) -> None:
    for model_name, expansion_factor, k in tqdm(config):
        test(
            RunConfig(
                autoencoder=MLSAEConfig(
                    expansion_factor=expansion_factor, k=k, tuned_lens=config.tuned_lens
                ),
                data=DataConfig(max_tokens=1_000_000, num_workers=1),
                model_name=model_name,
            ),
            device,
        )


if __name__ == "__main__":
    main(parse(SweepConfig), get_device())
