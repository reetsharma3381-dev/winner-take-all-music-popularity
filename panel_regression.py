"""
panel_regression.py
----------------------
Regresses the year-level inequality metrics on time trend + structural-break
era dummies (Discover Weekly launch, TikTok era) to test whether algorithmic
curation / virality platforms are statistically associated with rising
concentration. Also includes an artist-level panel regression relating
individual popularity to genre and time effects, using clustered SEs.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config


def add_era_dummies(df, year_col="year"):
    df = df.copy()
    dw_year = int(config.STRUCTURAL_BREAKS["discover_weekly_launch"][:4])
    tt_year = int(config.STRUCTURAL_BREAKS["tiktok_music_era"][:4])
    df["post_discover_weekly"] = (df[year_col] >= dw_year).astype(int)
    df["post_tiktok"] = (df[year_col] >= tt_year).astype(int)
    df["time_trend"] = df[year_col] - df[year_col].min()
    return df


def run_inequality_trend_regression(inequality_df, y_col="gini"):
    """
    Time-series regression of a year-level inequality metric on a linear
    trend plus era dummies. With only ~10 annual observations this is a
    low-power test (n=years), so treat it as descriptive/suggestive rather
    than a definitive causal estimate -- the artist-level regression below
    with clustered SEs is the more defensible inferential test.
    """
    data = add_era_dummies(inequality_df)
    formula = f"{y_col} ~ time_trend + post_discover_weekly + post_tiktok"
    model = smf.ols(formula, data=data).fit(cov_type="HC1")
    return model


def run_artist_level_panel_regression(panel_df, value_col="popularity",
                                       year_col="year", genre_col="genre_query",
                                       cluster_col="artist_id"):
    """
    Artist x year panel regression:
        value ~ time_trend + post_discover_weekly + post_tiktok + C(genre)
    with standard errors clustered by artist to account for repeated
    observations of the same artist over time (serial correlation).

    This tests whether, controlling for genre composition, an individual
    artist's expected popularity/streams shifted after each platform-era
    break -- a proxy for whether the *typical* artist's position changed,
    complementary to the aggregate inequality-index regression above.
    """
    data = add_era_dummies(panel_df, year_col=year_col)

    # log-transform the outcome if it's a raw stream/follower count (skewed, >0)
    if data[value_col].min() > 0 and data[value_col].max() > 1000:
        data["_y"] = np.log(data[value_col])
        y_label = f"log({value_col})"
    else:
        data["_y"] = data[value_col]
        y_label = value_col

    formula = f"_y ~ time_trend + post_discover_weekly + post_tiktok + C({genre_col})"
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )
    model._outcome_label = y_label  # stash for reporting
    return model


def summarize_model(model, name=""):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(model.summary())
    return model.summary()


if __name__ == "__main__":
    import inequality_metrics as im

    df = pd.read_csv(config.PANEL_FILE)
    value_col = "streams_est" if "streams_est" in df.columns else "popularity"

    ineq_df = im.compute_inequality_by_year(df, value_col=value_col)
    m1 = run_inequality_trend_regression(ineq_df, y_col="gini")
    summarize_model(m1, name="Aggregate Gini trend regression")

    m2 = run_artist_level_panel_regression(df, value_col=value_col)
    summarize_model(m2, name=f"Artist-level panel regression ({m2._outcome_label})")
