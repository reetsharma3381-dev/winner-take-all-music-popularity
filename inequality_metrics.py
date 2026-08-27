"""
inequality_metrics.py
-----------------------
Computes standard inequality metrics (Gini coefficient, Theil index,
top-X% share) on artist-level streams/popularity, per year, so you can
chart whether concentration has risen -- the second headline result
for the paper (alongside the falling power-law alpha).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT_DIR = "outputs"


def gini(values):
    """Standard Gini coefficient. 0 = perfect equality, 1 = max inequality."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    cum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])


def theil(values):
    """Theil's T index. 0 = perfect equality; decomposable into within/between groups."""
    x = np.asarray(values, dtype=float)
    x = x[x > 0]
    n = len(x)
    if n == 0:
        return np.nan
    mean_x = x.mean()
    return np.mean((x / mean_x) * np.log(x / mean_x))


def top_share(values, share):
    """Share of total value held by the top `share` fraction of units (e.g. 0.01 = top 1%)."""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    n = len(x)
    k = max(1, int(np.ceil(n * share)))
    total = x.sum()
    if total == 0:
        return np.nan
    return x[:k].sum() / total


def theil_decompose_by_group(df, value_col, group_col):
    """
    Decomposes Theil's T into within-group and between-group components.
    Useful for testing e.g. "is rising inequality driven by genre
    concentration (between) or superstars within each genre (within)?"
    """
    x = df[value_col].astype(float)
    total = x.sum()
    if total == 0:
        return {"within": np.nan, "between": np.nan, "total": np.nan}

    overall_mean = x.mean()
    n_total = len(x)

    within = 0.0
    between = 0.0
    for _, sub in df.groupby(group_col):
        n_g = len(sub)
        mean_g = sub[value_col].mean()
        share_g = sub[value_col].sum() / total
        within += share_g * theil(sub[value_col].values)
        if mean_g > 0:
            between += share_g * np.log(mean_g / overall_mean)

    return {"within": within, "between": between, "total": within + between}


def compute_inequality_by_year(df, value_col="streams_est", year_col="year", group_col="genre_query"):
    records = []
    for year, sub in df.groupby(year_col):
        vals = sub[value_col].values
        rec = {
            "year": year,
            "gini": gini(vals),
            "theil": theil(vals),
        }
        for s in config.TOP_SHARES:
            rec[f"top_{int(s*100)}pct_share"] = top_share(vals, s)

        decomp = theil_decompose_by_group(sub, value_col, group_col)
        rec["theil_within_genre"] = decomp["within"]
        rec["theil_between_genre"] = decomp["between"]

        records.append(rec)

    result = pd.DataFrame(records).sort_values("year")
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(os.path.join(OUT_DIR, "inequality_by_year.csv"), index=False)
    return result


def plot_inequality_trends(inequality_df):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(inequality_df["year"], inequality_df["gini"], marker="o", label="Gini")
    axes[0].plot(inequality_df["year"], inequality_df["theil"], marker="s", label="Theil")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Inequality index")
    axes[0].set_title("Gini and Theil over time")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.3)

    for s in config.TOP_SHARES:
        col = f"top_{int(s*100)}pct_share"
        axes[1].plot(inequality_df["year"], inequality_df[col], marker="o", label=f"Top {int(s*100)}%")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Share of total streams")
    axes[1].set_title("Top-share concentration over time")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "inequality_trends.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    df = pd.read_csv(config.PANEL_FILE)
    value_col = "streams_est" if "streams_est" in df.columns else "popularity"
    result = compute_inequality_by_year(df, value_col=value_col)
    plot_inequality_trends(result)
    print(result)
