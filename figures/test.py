import os

import pandas as pd


def parse_repo_id(repo_id: str) -> tuple[str, int, int]:
    split = repo_id.split("-")
    model_name = split[1] + "-" + split[2] + "-" + split[3]
    expansion_factor = int(split[4].lstrip("x"))
    k = int(split[5].lstrip("k"))
    return model_name, expansion_factor, k


if __name__ == "__main__":
    dfs: list[pd.DataFrame] = []
    for root, _, files in os.walk("out"):
        for name in files:
            if name.startswith("test_mlsae"):
                df = pd.read_csv(os.path.join(root, name))
                model_name, expansion_factor, k = parse_repo_id(name)

                columns = list(df.columns)
                df["model_name"] = model_name
                df["expansion_factor"] = expansion_factor
                df["k"] = k

                dfs.append(df)

    df = pd.concat(dfs)
    columns = ["model_name", "expansion_factor", "k"] + columns
    df = df[columns]
    df = df.sort_values(["model_name", "expansion_factor", "k"])
    df.to_csv("out/test.csv", index=False)

    is_pythia_70m = df["model_name"] == "pythia-70m-deduped"
    is_expansion_factor_64 = df["expansion_factor"] == 64
    is_k_32 = df["k"] == 32

    df[is_pythia_70m & is_expansion_factor_64].to_csv("out/test_k.csv", index=False)
    df[is_pythia_70m & is_k_32].to_csv("out/test_expansion_factor.csv", index=False)
