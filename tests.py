from simple_parsing import parse
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAEConfig
from mlsae.trainer import RunConfig, SweepConfig, test
from mlsae.utils import get_device

if __name__ == "__main__":
    device = get_device()
    for model_name, expansion_factor, k in tqdm(parse(SweepConfig)):
        test(
            RunConfig(
                model_name=model_name,
                autoencoder=MLSAEConfig(expansion_factor=expansion_factor, k=k),
                data=DataConfig(max_tokens=1_000_000, num_workers=1),
            )
        )
