# Spotify Superstar Economics

Tests whether artist popularity/streams follow a power-law (Pareto) distribution
consistent with "winner-take-all" dynamics, and whether concentration has
increased over time — with a panel regression testing candidate drivers
(Discover Weekly-era algorithmic curation, TikTok-era virality).

Analysis has been done for a span of a decade (2014-2024)

## The one thing to read before you do anything else

**The Spotify Web API does not expose historical raw stream counts.** It gives
you a `popularity` score (0–100) and follower counts, both as *current
snapshots only* — you can't ask "what was this artist's popularity in 2018."

You have three honest paths forward, and the code supports all three:

1. **Pull real historical data right now via `collect_kworb_wayback_data.py`.**
   Kworb.net's "Daily Chart - Global" page only ever shows *today's* top-200
   chart — but the Internet Archive's Wayback Machine has been snapshotting
   that exact page for years. This script fetches archived snapshots from
   one date per year (2016–2025 by default), parses the top-200 chart as it
   looked on that real historical date, and builds a genuine multi-year
   artist panel. **This is the recommended option** — it gives you real
   data today instead of waiting.
2. **Build your own panel over time.** Run `collect_data.py` on a schedule
   (e.g. a weekly cron job) and it will append live snapshots to
   `data/artist_panel.csv`. Fully legitimate, but you'll only have as much
   history as time has passed since you started.
3. **Merge in other third-party historical data.** Services like
   Chartmetric or Last.fm give you actual historical volume too. Reformat
   that data to match the schema below and skip the other collection
   scripts entirely.

To let you build and test the whole pipeline *today* regardless, `generate_synthetic_data.py`
creates a realistic fake panel (2014–2024) with an injected winner-take-all
trend, so you can see all the analysis working before you have real data.

## Important scope note on the Wayback/Kworb data (option 1)

This method captures **top-200-chart concentration**, not whole-market
concentration — it measures how unequal streaming is *among songs that
make the global daily chart* on a given day, and says nothing about the
much larger number of artists who never chart. This is a standard scope
restriction in the superstar-economics literature (Billboard Hot 100
concentration studies use exactly this kind of top-chart data), so it's
legitimate and defensible — just say so explicitly in your write-up rather
than implying it covers "all Spotify artists." Also note: featured artists
on a collab track are each credited with the track's full streams (a
simplifying convention, not a stream-split estimate) — mention this if you
want to be precise about it.

## Data schema (`data/artist_panel.csv`)

| column         | meaning                                              |
|----------------|-------------------------------------------------------|
| `artist_id`    | stable unique identifier                              |
| `artist_name`  | display name                                           |
| `genre_query`  | genre bucket used to sample/group the artist           |
| `year`         | calendar year of this observation                      |
| `streams_est`  | streaming-volume proxy (only in synthetic/3rd-party data) |
| `popularity`   | Spotify's 0–100 popularity score (if using live API)    |
| `followers`    | artist follower count                                   |

Only one of `streams_est` / `popularity` needs to be present — the pipeline
auto-detects which to use.

## Setup

```bash
pip install -r requirements.txt

# only needed if pulling live from Spotify:
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"
```

## Running it

```bash
# Option A: instant demo with synthetic data
python generate_synthetic_data.py
python main.py

# Option B: real historical data via Wayback Machine + Kworb (recommended)
python collect_kworb_wayback_data.py
python main.py

# Option C: live data collection (repeat over weeks/months to build a panel)
python collect_data.py

# Option D: drop your own historical CSV at data/artist_panel.csv, matching
# the schema above, then just run:
python main.py
```

`collect_kworb_wayback_data.py` needs `requests` and `beautifulsoup4`
(already in `requirements.txt`). It prints which archived snapshot date it
actually used for each year (the Wayback Machine doesn't always have a
snapshot on the exact date requested, so it searches +/-10 days and picks
the closest one) — check that printout to confirm the dates make sense
before you write them into your paper.

All figures and tables land in `outputs/`:
- `rank_size_<year>.png` — log-log rank-size plots per year
- `alpha_over_time.png` — power-law exponent trend (falling = more concentration)
- `power_law_fits_by_year.csv` — alpha, xmin, KS distance, distribution comparisons per year
- `inequality_trends.png` — Gini/Theil and top-1%/5%/10% share over time
- `inequality_by_year.csv` — same, in table form

## Methodology notes

- **Power-law fitting** uses the Clauset-Shalizi-Newman (2009) method via the
  `powerlaw` package, which estimates the optimal `xmin` (the point above
  which the tail behaves like a power law) rather than assuming the entire
  distribution is Pareto. A naive log-log OLS fit is known to be badly
  biased — don't use it as your primary estimate.
- **Distribution comparisons**: for each year, the power law is compared
  against log-normal, exponential, truncated power law, and stretched
  exponential alternatives via a log-likelihood ratio test. A common finding
  in cultural markets (Chung & Cox 1994; Salganik et al. 2006) is that
  log-normal fits the bulk of the distribution better, with a Pareto-like
  tail only among genuine superstars — don't be surprised if your results
  show this.
- **Inequality metrics**: Gini and Theil are both reported since Theil
  decomposes cleanly into within-genre and between-genre components, letting
  you test whether rising inequality is about superstars *within* every
  genre or about certain genres pulling away from others.
- **Structural break regressions**: the aggregate year-level regression
  (`panel_regression.run_inequality_trend_regression`) has very low power
  with ~10 annual observations — treat it as descriptive. The artist-level
  panel regression with clustered standard errors
  (`run_artist_level_panel_regression`) is the more defensible inferential
  test, since n = artists × years.
- **Known limitation**: cross-sectional survivorship — new/small artists
  cycle in and out of "Top artist by genre" searches. If your real data
  collection uses a fixed search-based sampling method (as `collect_data.py`
  does), note this explicitly as a limitation in your write-up, and consider
  restricting analysis to a fixed cohort of artists tracked consistently
  across all years.

## File overview

| file                          | purpose                                            |
|--------------------------------|-----------------------------------------------------|
| `config.py`                    | genres, structural break dates, settings            |
| `collect_data.py`               | live Spotify API pull -> panel CSV                 |
| `collect_kworb_wayback_data.py` | real historical panel via Wayback Machine snapshots of Kworb charts |
| `generate_synthetic_data.py`    | realistic fake panel for testing/demo               |
| `distribution_analysis.py`      | power-law fitting, rank-size plots, CSN method      |
| `inequality_metrics.py`         | Gini, Theil (+decomposition), top-share metrics     |
| `panel_regression.py`           | structural-break OLS regressions                    |
| `main.py`                       | runs the full pipeline end-to-end                   |
