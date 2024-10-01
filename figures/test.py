import os

import pandas as pd
from natsort import natsorted


def parse_repo_id(repo_id: str) -> tuple[str, int, int, bool]:
    split = repo_id.split("-")
    model_name = split[1] + "-" + split[2] + "-" + split[3]
    expansion_factor = int(split[4].lstrip("x"))
    k = int(split[5].lstrip("k"))
    tuned_lens = "-lens" in repo_id
    return model_name, expansion_factor, k, tuned_lens


if __name__ == "__main__":
    dfs: list[pd.DataFrame] = []
    columns: list[str] = []

    for root, _, files in os.walk("out"):
        for name in files:
            if name.startswith("test_mlsae"):
                df = pd.read_csv(os.path.join(root, name))
                model_name, expansion_factor, k, tuned_lens = parse_repo_id(name)

                columns = list(set(columns + list(df.columns)))
                df["model_name"] = model_name
                df["expansion_factor"] = expansion_factor
                df["k"] = k
                df["tuned_lens"] = tuned_lens

                if model_name == "pythia-70m-deduped":
                    df["n_latents"] = df["expansion_factor"] * 512
                elif model_name == "pythia-160m-deduped":
                    df["n_latents"] = df["expansion_factor"] * 768
                elif model_name == "pythia-410m-deduped":
                    df["n_latents"] = df["expansion_factor"] * 1024
                elif model_name == "pythia-1b-deduped":
                    df["n_latents"] = df["expansion_factor"] * 2048

                dfs.append(df)

    df = pd.concat(dfs)
    columns = [
        "model_name",
        "expansion_factor",
        "k",
        "tuned_lens",
        "n_latents",
    ] + natsorted(columns)
    df = df[columns]
    df = df.sort_values(["n_latents", "expansion_factor", "k", "tuned_lens"])
    df.to_csv("out/test.csv", index=False)

    is_70m = df["model_name"] == "pythia-70m-deduped"
    is_160m = df["model_name"] == "pythia-160m-deduped"
    is_x64 = df["expansion_factor"] == 64
    is_k32 = df["k"] == 32
    is_tuned_lens = df["tuned_lens"]

    df[is_70m & is_x64 & ~is_tuned_lens].to_csv(
        "out/test_pythia-70m-deduped_k.csv", index=False
    )
    df[is_70m & is_k32 & ~is_tuned_lens].to_csv(
        "out/test_pythia-70m-deduped_expansion_factor.csv", index=False
    )
    df[is_70m & is_x64 & is_tuned_lens].to_csv(
        "out/test_pythia-70m-deduped_lens_k.csv", index=False
    )
    df[is_70m & is_k32 & is_tuned_lens].to_csv(
        "out/test_pythia-70m-deduped_lens_expansion_factor.csv", index=False
    )
    df[is_160m & is_k32 & ~is_tuned_lens].to_csv(
        "out/test_pythia-160m-deduped_expansion_factor.csv", index=False
    )
    df[is_160m & is_x64 & ~is_tuned_lens].to_csv(
        "out/test_pythia-160m-deduped_k.csv", index=False
    )

    df = df[
        [
            "model_name",
            "train/fvu/avg",
            "train/mse/avg",
            "train/l1/avg",
            "val/loss/delta/avg",
            "val/logit/kldiv/avg",
        ]
    ]
    df["model_name"] = (
        df["model_name"].str.replace("pythia", "Pythia").str.replace("-deduped", "")
    )
    df = df.rename(
        columns={
            "model_name": "Model",
            "train/fvu/avg": "FVU",
            "train/mse/avg": "MSE",
            "train/l1/avg": "L1 Norm",
            "val/loss/delta/avg": "Delta CE Loss",
            "val/logit/kldiv/avg": "KL Divergence",
        }
    )
    df[is_x64 & is_k32 & ~is_tuned_lens].transpose().to_csv(
        "out/test_model_name.csv", header=False, index=True
    )
    df[is_x64 & is_k32 & is_tuned_lens].transpose().to_csv(
        "out/test_lens_model_name.csv", header=False, index=True
    )
