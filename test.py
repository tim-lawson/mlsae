from simple_parsing import parse

from mlsae.model.lightning import MLSAEConfig
from mlsae.trainer import RunConfig, SweepConfig, test
from mlsae.utils import get_repo_id

if __name__ == "__main__":
    config = RunConfig()
    for model_name, expansion_factor, k in parse(SweepConfig):
        repo_id = get_repo_id(model_name, expansion_factor, k, transformer=True)
        config.model_name = model_name
        config.autoencoder = MLSAEConfig(expansion_factor=expansion_factor, k=k)
        test(config, repo_id)
