"""
naive_vs_csn_comparison.py
-----------------------------
Demonstrates why this project uses the Clauset-Shalizi-Newman (CSN) method
instead of the traditional "naive" log-log OLS slope for estimating a
power-law exponent.

THE NAIVE METHOD
-----------------
Sort the data descending, plot log(rank) vs log(value), fit a straight
line with OLS, and treat the slope as the power-law exponent. This is
intuitive and was the standard approach for decades, but it's known to be
statistically biased (Clauset, Shalizi & Newman, 2009, SIAM Review):
  - It assumes the ENTIRE distribution is power-law shaped, when usually
    only the extreme tail actually is.
  - OLS on ranked/logged data violates OLS's assumptions (the errors
    aren't independent -- rank order guarantees correlation between
    adjacent points).
  - The estimate is highly sensitive to which points you include.

THE CSN METHOD
---------------
Uses maximum likelihood estimation instead of OLS, and automatically
finds the optimal `xmin` -- the threshold above which the tail actually
behaves like a power law -- by minimizing the Kolmogorov-Smirnov distance
between the fitted model and the data. This is what the `powerlaw`
package implements, and what `distribution_analysis.py` uses throughout
this project.

Run this after you have real data (data/artist_panel.csv) to see the two
estimates side by side for your own results, e.g. for a methods-section
comparison paragraph.
"""

import numpy as np
import pandas as pd
import powerlaw

import config


def naive_loglog_slope(values):
    """
    The traditional (biased) approach: OLS slope of log(rank) vs log(value)
    on the full sorted dataset.
    """
    data_sorted = np.sort(np.asarray(values, dtype=float))[::-1]
    data_sorted = data_sorted[data_sorted > 0]
    ranks = np.arange(1, len(data_sorted) + 1)
    slope, intercept = np.polyfit(np.log(ranks), np.log(data_sorted), 1)
    # By convention the power-law exponent alpha = 1 - slope of the
    # rank-size relationship (rank ~ value^(-1/(alpha-1))); many papers
    # instead just report the raw rank-size slope directly and are explicit
    # about which convention they use. We report the raw slope here since
    # that's what's most commonly plotted.
    return slope


def csn_fit(values):
    """The rigorous approach: maximum-likelihood fit with automatic xmin."""
    data = np.asarray(values, dtype=float)
    data = data[data > 0]
    fit = powerlaw.Fit(data, discrete=True, verbose=False)
    return fit


def compare_for_year(df, year, value_col="streams_est", year_col="year"):
    sub = df[df[year_col] == year][value_col].values

    naive_slope = naive_loglog_slope(sub)
    fit = csn_fit(sub)
    R, p = fit.distribution_compare("power_law", "lognormal")

    return {
        "year": year,
        "n_obs": len(sub),
        "naive_loglog_slope": naive_slope,
        "csn_alpha": fit.power_law.alpha,
        "csn_xmin": fit.power_law.xmin,
        "csn_n_above_xmin": int((fit.data >= fit.power_law.xmin).sum()),
        "csn_ks_distance": fit.power_law.D,
        "powerlaw_vs_lognormal_R": R,
        "powerlaw_vs_lognormal_p": p,
    }


def run_comparison(value_col=None, year_col="year"):
    df = pd.read_csv(config.PANEL_FILE)
    if value_col is None:
        value_col = "streams_est" if "streams_est" in df.columns else "popularity"

    records = [compare_for_year(df, year, value_col, year_col)
               for year in sorted(df[year_col].unique())]
    result = pd.DataFrame(records)

    pd.set_option("display.width", 140)
    print(result.to_string(index=False))
    result.to_csv("outputs/naive_vs_csn_comparison.csv", index=False)
    print("\nSaved -> outputs/naive_vs_csn_comparison.csv")

    print("\nHow to read the powerlaw_vs_lognormal columns:")
    print("  R > 0  -> power law fits better than log-normal")
    print("  R < 0  -> log-normal fits better than power law")
    print("  p < 0.05 -> the difference is statistically meaningful;")
    print("             large p means the two fits are indistinguishable")
    print("             for this data, and that should be reported honestly.")

    return result


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    run_comparison()
