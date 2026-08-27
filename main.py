"""
main.py
--------
Runs the full pipeline end-to-end:
  1. Load the artist panel (data/artist_panel.csv)
  2. Fit power-law distributions by year + compare vs. lognormal/exponential
  3. Compute Gini / Theil / top-share inequality by year
  4. Run the structural-break panel regressions
  5. Write all tables/figures to outputs/ and print a plain-text summary

Usage:
    python generate_synthetic_data.py   # if you don't have real data yet
    python main.py
"""

import os
import pandas as pd

import config
import distribution_analysis as da
import inequality_metrics as im
import panel_regression as pr


def main():
    if not os.path.exists(config.PANEL_FILE):
        raise FileNotFoundError(
            f"No panel file found at {config.PANEL_FILE}. "
            f"Run `python generate_synthetic_data.py` for a demo dataset, "
            f"or `python collect_data.py` (run repeatedly over time) for real data."
        )

    df = pd.read_csv(config.PANEL_FILE)
    value_col = "streams_est" if "streams_est" in df.columns else "popularity"
    print(f"Loaded panel: {len(df)} rows, {df['artist_id'].nunique()} unique artists, "
          f"years {df['year'].min()}-{df['year'].max()}")
    print(f"Using '{value_col}' as the streaming-volume proxy.\n")

    # --- 1. Power-law / distributional analysis ---
    print(">>> Fitting power-law distributions by year...")
    fit_results = da.run_by_year(df, value_col=value_col)
    da.plot_alpha_over_time(fit_results)
    print(fit_results[["year", "alpha", "xmin", "ks_distance",
                        "power_law_vs_lognormal"]].to_string(index=False))

    # --- 2. Inequality metrics ---
    print("\n>>> Computing inequality metrics by year...")
    ineq_df = im.compute_inequality_by_year(df, value_col=value_col)
    im.plot_inequality_trends(ineq_df)
    print(ineq_df.to_string(index=False))

    # --- 3. Regressions ---
    print("\n>>> Running structural-break regressions...")
    m1 = pr.run_inequality_trend_regression(ineq_df, y_col="gini")
    pr.summarize_model(m1, name="Aggregate Gini trend regression")

    m2 = pr.run_artist_level_panel_regression(df, value_col=value_col)
    pr.summarize_model(m2, name=f"Artist-level panel regression ({m2._outcome_label})")

    # --- 4. Plain-text summary ---
    alpha_start = fit_results.iloc[0]["alpha"]
    alpha_end = fit_results.iloc[-1]["alpha"]
    gini_start = ineq_df.iloc[0]["gini"]
    gini_end = ineq_df.iloc[-1]["gini"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Power-law exponent (alpha): {alpha_start:.2f} ({fit_results.iloc[0]['year']}) "
          f"-> {alpha_end:.2f} ({fit_results.iloc[-1]['year']})")
    print("  (a falling alpha = fatter tail = more winner-take-all concentration)")
    print(f"Gini coefficient: {gini_start:.3f} -> {gini_end:.3f}")
    print(f"Top 1% share: {ineq_df.iloc[0]['top_1pct_share']:.3f} -> "
          f"{ineq_df.iloc[-1]['top_1pct_share']:.3f}")
    print(f"\nAll figures and tables saved to ./outputs/")


if __name__ == "__main__":
    main()
