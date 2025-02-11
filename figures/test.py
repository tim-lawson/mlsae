# TODO: tidy this up!

import os
import re

import pandas as pd
from natsort import natsorted


def parse_mlsae_repo_id(repo_id: str) -> tuple[str, int, int, bool]:
    split = repo_id.split("-")
    if split[1] == "pythia" or split[1] == "gemma" or split[1] == "Llama":
        model_name = split[1] + "-" + split[2] + "-" + split[3]
        expansion_factor = int(split[4].lstrip("x"))
        k = int(split[5].lstrip("k"))
    elif split[1] == "gpt2":
        model_name = split[1]
        expansion_factor = int(split[2].lstrip("x"))
        k = int(split[3].lstrip("k"))
    else:
        raise ValueError(f"unknown model: {split[1]}")
    tuned_lens = "-lens" in repo_id
    return model_name, expansion_factor, k, tuned_lens


def parse_sae_repo_id(repo_id: str) -> tuple[str, int, int, bool, int]:
    split = repo_id.split("-")
    model_name = split[1] + "-" + split[2] + "-" + split[3]
    expansion_factor = int(split[4].lstrip("x"))
    k = int(split[5].lstrip("k"))
    tuned_lens = "-lens" in repo_id
    layer = int(split[9].rstrip(".csv")) if tuned_lens else int(split[8].rstrip(".csv"))
    return model_name, expansion_factor, k, tuned_lens, layer


def matrix_plot(
    df: pd.DataFrame,
    filename: str,
    pattern: str | re.Pattern[str] = r"train/fvu/layer_\d+",
) -> None:
    cols = [col for col in df.columns if re.match(pattern, col)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    rows = []
    for train_layer in df["layer"].unique():
        if train_layer is None:
            train_layer = -1
            row = df[df["layer"].isnull()].iloc[0]
        else:
            row = df[df["layer"] == train_layer].iloc[0]
        for col in cols:
            rows.append(
                {"x": int(col.split("_")[-1]), "y": train_layer, "value": row[col]}
            )
    rows = [row for row in rows if not pd.isna(row["value"])]
    pd.DataFrame(rows).sort_values(["y", "x"]).to_csv(filename, index=False)


if __name__ == "__main__":
    dfs: list[pd.DataFrame] = []
    columns: list[str] = []

    for root, _, files in os.walk("out"):
        for name in files:
            if name.startswith("test_mlsae"):
                df = pd.read_csv(os.path.join(root, name))
                model_name, expansion_factor, k, tuned_lens = parse_mlsae_repo_id(name)

                columns = list(set(columns + list(df.columns)))
                df["model_name"] = model_name
                df["expansion_factor"] = expansion_factor
                df["k"] = k
                df["tuned_lens"] = tuned_lens
                df["layer"] = None

                if model_name == "pythia-70m-deduped":
                    df["n_latents"] = df["expansion_factor"] * 512
                elif model_name == "pythia-160m-deduped":
                    df["n_latents"] = df["expansion_factor"] * 768
                elif model_name == "pythia-410m-deduped":
                    df["n_latents"] = df["expansion_factor"] * 1024
                elif model_name == "pythia-1b-deduped":
                    df["n_latents"] = df["expansion_factor"] * 2048

                dfs.append(df)

            if name.startswith("test_sae"):
                df = pd.read_csv(os.path.join(root, name))
                model_name, expansion_factor, k, tuned_lens, layer = parse_sae_repo_id(
                    name
                )

                columns = list(set(columns + list(df.columns)))
                df["model_name"] = model_name
                df["expansion_factor"] = expansion_factor
                df["k"] = k
                df["tuned_lens"] = tuned_lens
                df["layer"] = layer

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
        "layer",
        "n_latents",
    ] + natsorted(columns)
    df = df[columns]
    df = df.sort_values(["n_latents", "expansion_factor", "k", "tuned_lens", "layer"])
    df.to_csv("out/test.csv", index=False)

    is_70m = df["model_name"] == "pythia-70m-deduped"
    is_160m = df["model_name"] == "pythia-160m-deduped"
    is_410m = df["model_name"] == "pythia-410m-deduped"
    is_x64 = df["expansion_factor"] == 64
    is_k32 = df["k"] == 32
    is_tuned_lens = df["tuned_lens"]
    is_layer = df["layer"].notnull()

    df[is_70m & is_x64 & ~is_tuned_lens & ~is_layer].to_csv(
        "out/test_pythia-70m-deduped_k.csv", index=False
    )
    df[is_70m & is_k32 & ~is_tuned_lens & ~is_layer].to_csv(
        "out/test_pythia-70m-deduped_expansion_factor.csv", index=False
    )
    df[is_70m & is_x64 & is_tuned_lens & ~is_layer].to_csv(
        "out/test_pythia-70m-deduped_lens_k.csv", index=False
    )
    df[is_70m & is_k32 & is_tuned_lens & ~is_layer].to_csv(
        "out/test_pythia-70m-deduped_lens_expansion_factor.csv", index=False
    )
    df[is_160m & is_k32 & ~is_tuned_lens & ~is_layer].to_csv(
        "out/test_pythia-160m-deduped_expansion_factor.csv", index=False
    )
    df[is_160m & is_x64 & ~is_tuned_lens & ~is_layer].to_csv(
        "out/test_pythia-160m-deduped_k.csv", index=False
    )

    df[is_70m & is_layer & ~is_tuned_lens].to_csv(
        "out/test_pythia-70m-deduped_layer.csv", index=False
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-70m-deduped_layer_fvu.csv",
        pattern=r"train/fvu/layer_\d+",
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-70m-deduped_layer_mse.csv",
        pattern=r"train/mse/layer_\d+",
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-70m-deduped_layer_loss_delta.csv",
        pattern=r"val/loss/delta/layer_\d+",
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-70m-deduped_layer_kl_div.csv",
        pattern=r"val/logit/kldiv/layer_\d+",
    )

    df[is_70m & is_layer & is_tuned_lens].to_csv(
        "out/test_pythia-70m-deduped_layer.csv", index=False
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & is_tuned_lens],
        "out/test_pythia-70m-deduped_lens_layer_fvu.csv",
        pattern=r"train/fvu/layer_\d+",
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & is_tuned_lens],
        "out/test_pythia-70m-deduped_lens_layer_mse.csv",
        pattern=r"train/mse/layer_\d+",
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & is_tuned_lens],
        "out/test_pythia-70m-deduped_lens_layer_loss_delta.csv",
        pattern=r"val/loss/delta/layer_\d+",
    )
    matrix_plot(
        df[is_70m & is_x64 & is_k32 & is_tuned_lens],
        "out/test_pythia-70m-deduped_lens_layer_kl_div.csv",
        pattern=r"val/logit/kldiv/layer_\d+",
    )

    df[is_160m & is_layer & ~is_tuned_lens].to_csv(
        "out/test_pythia-160m-deduped_layer.csv", index=False
    )
    matrix_plot(
        df[is_160m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-160m-deduped_layer_fvu.csv",
        pattern=r"train/fvu/layer_\d+",
    )
    matrix_plot(
        df[is_160m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-160m-deduped_layer_mse.csv",
        pattern=r"train/mse/layer_\d+",
    )
    matrix_plot(
        df[is_160m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-160m-deduped_layer_loss_delta.csv",
        pattern=r"val/loss/delta/layer_\d+",
    )
    matrix_plot(
        df[is_160m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-160m-deduped_layer_kl_div.csv",
        pattern=r"val/logit/kldiv/layer_\d+",
    )

    df[is_410m & is_layer & ~is_tuned_lens].to_csv(
        "out/test_pythia-410m-deduped_layer.csv", index=False
    )
    matrix_plot(
        df[is_410m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-410m-deduped_layer_fvu.csv",
        pattern=r"train/fvu/layer_\d+",
    )
    matrix_plot(
        df[is_410m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-410m-deduped_layer_mse.csv",
        pattern=r"train/mse/layer_\d+",
    )
    matrix_plot(
        df[is_410m & is_x64 & is_k32 & ~is_tuned_lens],
        "out/test_pythia-410m-deduped_layer_loss_delta.csv",
        pattern=r"val/loss/delta/layer_\d+",
    )

    df = df[
        [
            "model_name",
            "train/fvu/avg",
            "train/mse/avg",
            "train/l1/avg",
            "val/loss/delta/avg",
            # "val/logit/kldiv/avg",
        ]
    ]
    df["model_name"] = (
        df["model_name"]
        .str.replace("pythia", "Pythia")
        .str.replace("-deduped", "")
        .str.replace("gpt2", "GPT-2 small")
        .str.replace("google/", "")
        .str.replace("gemma-2-2b", "Gemma 2 2B")
        .str.replace("meta-llama/", "")
        .str.replace("Llama-3.2-3B", "Llama 3.2 3B")
    )
    df["model_name"] = pd.Categorical(
        df["model_name"],
        categories=[
            "Pythia-70m",
            "Pythia-160m",
            "Pythia-410m",
            "Pythia-1b",
            "Pythia-1.4b",
            "GPT-2 small",
            "Gemma 2 2B",
            "Llama 3.2 3B",
        ],
    )
    df = df.rename(
        columns={
            "model_name": "Model",
            "train/fvu/avg": "FVU",
            "train/mse/avg": "MSE",
            "train/l1/avg": "L1 Norm",
            "val/loss/delta/avg": "Delta CE Loss",
            # "val/logit/kldiv/avg": "KL Divergence",
        }
    )
    # is_14b = df["Model"] == "Pythia-1.4b"
    df[is_x64 & is_k32 & ~is_tuned_lens & ~is_layer].to_csv(
        "out/test_model_name.csv",
        header=True,
        index=False,
    )
    df[is_x64 & is_k32 & is_tuned_lens & ~is_layer].to_csv(
        "out/test_lens_model_name.csv",
        header=True,
        index=False,
    )
