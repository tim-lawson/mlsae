from dataclasses import dataclass

from simple_parsing import parse
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAEConfig
from mlsae.trainer import RunConfig, SweepConfig, test
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    tuned_lens: bool = False
    """Whether to apply a pretrained tuned lens before the encoder."""


if __name__ == "__main__":
    device = get_device()
    config = parse(SweepConfig)
    for model_name, expansion_factor, k in tqdm(config):
        test(
            RunConfig(
                model_name=model_name,
                autoencoder=MLSAEConfig(
                    expansion_factor=expansion_factor, k=k, tuned_lens=config.tuned_lens
                ),
                data=DataConfig(max_tokens=1_000_000, num_workers=1),
            )
        )
