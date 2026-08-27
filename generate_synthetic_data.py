"""
generate_synthetic_data.py
---------------------------
Generates a realistic synthetic artist panel so you can test/demo the
entire analysis pipeline immediately, without waiting on live Spotify
API collection or sourcing third-party historical data.

The synthetic process is deliberately built to mimic superstar dynamics:
  - streams are drawn from a mixture of log-normal (the bulk) and a
    Pareto tail (the superstars), which is exactly the pattern the
    empirical literature (e.g. Rosen 1981, Chung & Cox 1994) finds in
    real cultural markets.
  - inequality is set to *increase* over the sample years, and a
    structural break is injected around the TikTok-era date so the
    panel regression has a genuine effect to recover.

Replace this with real data (from collect_data.py and/or a third-party
historical source) when you're ready — just match the column schema.
"""

import numpy as np
import pandas as pd
import os

import config

RNG = np.random.default_rng(42)

N_ARTISTS = 1200
YEARS = list(range(2014, 2025))  # 2014-2024 inclusive


def simulate_year_streams(n, year, base_year=2014, breakpoint_year=2019):
    """
    Mixture of log-normal (bulk of artists) + Pareto tail (superstars).
    Inequality (Pareto share and shape) increases after `breakpoint_year`
    to simulate a TikTok/algorithmic-discovery-driven winner-take-all shift.
    """
    years_since_break = max(0, year - breakpoint_year)

    # Pareto tail gets fatter (lower alpha = more inequality) after the break
    base_alpha = 2.6
    alpha = max(1.15, base_alpha - 0.09 * years_since_break)

    # Share of artists drawn from the superstar tail grows slightly over time
    tail_share = min(0.08, 0.03 + 0.004 * years_since_break)

    is_tail = RNG.random(n) < tail_share

    lognormal_part = RNG.lognormal(mean=11.5, sigma=1.1, size=n)  # bulk streams
    pareto_part = (RNG.pareto(alpha, size=n) + 1) * 5_000_000      # superstar streams

    streams = np.where(is_tail, pareto_part, lognormal_part)

    # Slight overall growth in total listening year over year
    growth = 1.0 + 0.04 * (year - base_year)
    streams = streams * growth

    return streams.astype(np.int64)


def build_panel():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    artist_ids = [f"artist_{i:04d}" for i in range(N_ARTISTS)]
    genres = RNG.choice(config.GENRES, size=N_ARTISTS)
    # Give each artist a persistent "quality" draw so rank isn't pure noise year to year
    artist_quality = RNG.normal(0, 1, size=N_ARTISTS)

    rows = []
    for year in YEARS:
        base_streams = simulate_year_streams(N_ARTISTS, year)
        # Blend in persistent quality so high performers tend to stay high performers
        quality_adj = np.exp(artist_quality * 0.3)
        streams = (base_streams * quality_adj).astype(np.int64)

        # popularity score (0-100) as a monotonic, noisy transform of rank
        pct_rank = pd.Series(streams).rank(pct=True).to_numpy()
        popularity = np.clip((pct_rank * 100) + RNG.normal(0, 3, N_ARTISTS), 0, 100).astype(int)

        followers = (streams * RNG.uniform(0.02, 0.06, N_ARTISTS)).astype(np.int64)

        for i in range(N_ARTISTS):
            rows.append(
                {
                    "artist_id": artist_ids[i],
                    "artist_name": f"Artist {i:04d}",
                    "genre_query": genres[i],
                    "genres_tagged": genres[i],
                    "year": year,
                    "snapshot_date": f"{year}-06-15",
                    "streams_est": streams[i],   # SYNTHETIC ONLY - real API won't give this
                    "popularity": popularity[i],
                    "followers": followers[i],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(config.PANEL_FILE, index=False)
    print(f"Synthetic panel written to {config.PANEL_FILE}")
    print(f"  {N_ARTISTS} artists x {len(YEARS)} years = {len(df)} rows")
    return df


if __name__ == "__main__":
    build_panel()
