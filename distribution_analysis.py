"""
distribution_analysis.py
--------------------------
Tests whether artist-level streams/popularity follow a power-law (Pareto)
distribution using the Clauset-Shalizi-Newman (2009) method, and compares
the power-law fit against log-normal and exponential alternatives.

Key functions:
  - fit_power_law(data)          -> fitted powerlaw.Fit object
  - compare_distributions(fit)   -> dict of log-likelihood ratio tests
  - rank_size_plot(data, ...)    -> saves a log-log rank-size plot
  - run_by_year(df, value_col)   -> fits the distribution separately per year
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import powerlaw

import config

OUT_DIR = "outputs"


def fit_power_law(data, discrete=True):
    """
    Fit a power law to `data` using the Clauset-Shalizi-Newman method.
    `powerlaw.Fit` automatically estimates the optimal xmin (the point
    above which the tail is power-law-like) via KS-statistic minimization,
    rather than assuming the whole distribution is Pareto.
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
    return fit


def compare_distributions(fit):
    """
    Likelihood-ratio comparisons of power law vs. alternative distributions.
    R > 0 and p < 0.05 means the first distribution in the pair is favored.
    """
    comparisons = {}
    for alt in ["lognormal", "exponential", "truncated_power_law", "stretched_exponential"]:
        R, p = fit.distribution_compare("power_law", alt)
        comparisons[f"power_law_vs_{alt}"] = {"loglikelihood_ratio_R": R, "p_value": p}
    return comparisons


def summarize_fit(fit, label=""):
    comps = compare_distributions(fit)
    summary = {
        "label": label,
        "alpha": fit.power_law.alpha,
        "xmin": fit.power_law.xmin,
        "ks_distance": fit.power_law.D,
        "n_obs_above_xmin": int((fit.data >= fit.power_law.xmin).sum()),
        "n_obs_total": len(fit.data),
    }
    summary.update(comps)
    return summary


def rank_size_plot(data, title, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.sort(np.asarray(data, dtype=float))[::-1]
    ranks = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, data, marker=".", linestyle="none", markersize=4, alpha=0.6)
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Streams / Popularity proxy (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_by_year(df, value_col="streams_est", year_col="year"):
    """
    Fits the power law separately for each year in the panel and returns
    a tidy DataFrame of alpha, xmin, KS distance, and distribution
    comparisons over time -- this is the key input for showing whether
    the tail has gotten "fatter" (lower alpha) over time.
    """
    records = []
    for year, sub in df.groupby(year_col):
        fit = fit_power_law(sub[value_col].values)
        summary = summarize_fit(fit, label=str(year))
        summary[year_col] = year
        records.append(summary)
        rank_size_plot(
            sub[value_col].values,
            title=f"Rank-size distribution, {year}",
            filename=f"rank_size_{year}.png",
        )

    result = pd.DataFrame(records).sort_values(year_col)
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(os.path.join(OUT_DIR, "power_law_fits_by_year.csv"), index=False)
    return result


def plot_alpha_over_time(fit_results, year_col="year"):
    """
    A falling alpha over time = the tail is getting fatter = more
    winner-take-all concentration. This is one of the two headline
    charts for the paper (the other being the Gini/Top-1% trend).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fit_results[year_col], fit_results["alpha"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Power-law exponent (alpha)")
    ax.set_title("Tail exponent over time (lower = more concentration)")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "alpha_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    df = pd.read_csv(config.PANEL_FILE)
    value_col = "streams_est" if "streams_est" in df.columns else "popularity"
    results = run_by_year(df, value_col=value_col)
    plot_alpha_over_time(results)
    print(results[["year", "alpha", "xmin", "ks_distance"]])
