import os
from dataclasses import dataclass

import diptest
import numpy as np
import pandas as pd
import scipy.stats
import torch
from simple_parsing import parse

from figures.test import parse_mlsae_repo_id
from mlsae.analysis.dists import Dists
from mlsae.trainer import SweepConfig
from mlsae.utils import get_device


@dataclass
class Config(SweepConfig):
    filename: str = "bimodality.csv"
    """The filename to save the results to."""


def main(
    config: Config, device: torch.device, out: str | os.PathLike[str] = ".out"
) -> None:
    os.makedirs(out, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for repo_id in config.repo_ids(transformer=True, tuned_lens=config.tuned_lens):
        dists = Dists.load(repo_id, device)

        counts = dists.counts.cpu().numpy()
        totals = dists.totals.cpu().numpy()
        layer_means = dists.layer_mean.cpu().numpy()
        layer_vars = dists.layer_var.cpu().numpy()
        entropies = dists.entropies.cpu().numpy()

        repo_id = repo_id.split("/")[-1]
        model_name, expansion_factor, k, tuned_lens = parse_mlsae_repo_id(repo_id)

        skewness = scipy.stats.skew(totals, axis=0, nan_policy="omit")
        kurtosis = scipy.stats.kurtosis(totals, axis=0, fisher=False, nan_policy="omit")
        bimodality = (skewness**2 + 1) / kurtosis

        dips: list[float] = []
        for i in range(dists.n_latents):
            dips.append(diptest.dipstat(totals[:, i]))  # type: ignore

        # Save the bimodality stats for each latent
        df = pd.DataFrame(
            {
                "count": counts.sum(axis=0),
                "total": totals.sum(axis=0),
                "layer_mean": layer_means,
                "layer_var": layer_vars,
                "entropy": entropies,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "bimodality": bimodality,
                "dip": dips,
            }
        )
        df.to_csv(os.path.join(out, f"bimodality_{repo_id.split("/")[-1]}.csv"))

        skewness = skewness[~np.isnan(skewness)]
        kurtosis = kurtosis[~np.isnan(kurtosis)]
        bimodality = bimodality[~np.isnan(bimodality)]

        # Save the bimodality stats over all latents
        rows.append(
            {
                "model_name": model_name,
                "n_layers": dists.n_layers,
                "n_latents": dists.n_latents,
                "expansion_factor": expansion_factor,
                "k": k,
                "tuned_lens": tuned_lens,
                "skewness": np.mean(skewness),
                "skewness_var": np.var(skewness),
                "skewness_std": np.std(skewness),
                "skewness_sem": scipy.stats.sem(skewness),
                "kurtosis": np.mean(kurtosis),
                "kurtosis_var": np.var(kurtosis),
                "kurtosis_std": np.std(kurtosis),
                "kurtosis_sem": scipy.stats.sem(kurtosis),
                "bimodality": np.mean(bimodality),
                "bimodality_var": np.var(bimodality),
                "bimodality_std": np.std(bimodality),
                "bimodality_sem": scipy.stats.sem(bimodality),
                "dip": np.mean(dips),  # type: ignore
                "dip_var": np.var(dips),  # type: ignore
                "dip_std": np.std(dips),  # type: ignore
                "dip_sem": scipy.stats.sem(dips),  # type: ignore
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out, config.filename), index=False)


def regress(xcol: str, ycol: str, out: str | os.PathLike[str] = ".out") -> None:
    rows = []
    for filename in os.listdir(out):
        if "bimodality_" in filename:
            df = pd.read_csv(os.path.join(out, filename))
            try:
                name = filename.replace("bimodality_", "").replace(".csv", "")
                x = df[xcol].to_numpy()
                y = df[ycol].to_numpy()
                mask = np.isnan(x) | np.isnan(y)
                slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(
                    x[~mask], y[~mask]
                )
                rows.append(
                    {
                        "name": name,
                        "slope": slope,
                        "intercept": intercept,
                        "rvalue": rvalue,
                        "pvalue": pvalue,
                        "stderr": stderr,
                    }
                )
            except KeyError:
                pass
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"bimodality_regress_{xcol}_{ycol}.csv"), index=False)


if __name__ == "__main__":
    # main(parse(Config), get_device())
    out = ".out/bimodality"
    regress(xcol="count", ycol="bimodality", out=out)
    regress(xcol="total", ycol="bimodality", out=out)
    regress(xcol="count", ycol="dip", out=out)
    regress(xcol="total", ycol="dip", out=out)
    regress(xcol="count", ycol="entropy", out=out)
    regress(xcol="total", ycol="entropy", out=out)
