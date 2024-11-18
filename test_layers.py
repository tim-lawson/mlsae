from mlsae.model import DataConfig, MLSAEConfig
from mlsae.trainer import RunConfig, test

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
# pythia_410m = "EleutherAI/pythia-410m-deduped"
# pythia_1b = "EleutherAI/pythia-1b-deduped"

layers = {
    pythia_70m: range(6),
    pythia_160m: range(12),
    # pythia_410m: range(24),
    # pythia_1b: range(16),
}


def main() -> None:
    for model_name in [pythia_70m, pythia_160m]:
        for layer in layers[model_name]:
            test(
                RunConfig(
                    autoencoder=MLSAEConfig(
                        expansion_factor=64, k=32, tuned_lens=False
                    ),
                    data=DataConfig(max_tokens=1_000_000, num_workers=1),
                    model_name=model_name,
                    layers=[layer],
                )
            )


if __name__ == "__main__":
    main()
