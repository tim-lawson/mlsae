from mlsae.analysis import dists, variances
from mlsae.model.data import DataConfig
from mlsae.utils import get_repo_id


def main() -> None:
    repo_ids = [
        get_repo_id(
            model_name="EleutherAI/pythia-70m-deduped",
            expansion_factor=64,
            k=32,
            tuned_lens=False,
            transformer=True,
            layers=[i],
        )
        for i in range(6)
    ]
    repo_ids += [
        get_repo_id(
            model_name="EleutherAI/pythia-160m-deduped",
            expansion_factor=64,
            k=32,
            tuned_lens=False,
            transformer=True,
            layers=[i],
        )
        for i in range(12)
    ]

    data = DataConfig(max_tokens=1e7)
    for repo_id in repo_ids:
        dists.main(dists.Config(repo_id, data=data))
        variances.main(repo_id, data)


if __name__ == "__main__":
    main()
