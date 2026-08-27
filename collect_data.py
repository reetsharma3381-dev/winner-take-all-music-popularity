"""
collect_data.py
----------------
Pulls artist-level data from the Spotify Web API and appends a timestamped
snapshot to a panel CSV file.

IMPORTANT CAVEAT (read this before you design your paper's data section):
The Spotify Web API does NOT expose raw historical stream counts. The best
available proxies are:
  - `popularity` (0-100 integer, Spotify's own algorithmic popularity score)
  - `followers.total` (artist follower count)
These are cross-sectional-in-time snapshots — Spotify does not let you query
"popularity in 2018" retroactively. To build a genuine time-series panel you
have two options:
  1. Run this script repeatedly over weeks/months to accumulate your own
     panel (slow, but 100% legitimate and defensible methodologically).
  2. Supplement with third-party historical data (e.g., Kworb.net chart
     archives, Chartmetric, or Last.fm scrobble counts) merged in by artist
     name/ID. This is the more common approach in published work on this
     topic, and I'd recommend it for a course-length project.

This script handles option (1) and produces data in the exact shape needed
by the rest of the pipeline. If you already have historical data from
another source, just format it to match the columns in `data/artist_panel.csv`
(see the README) and skip this script.
"""

import os
import time
import datetime
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import config


def get_client():
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET environment variables. "
            "Create an app at https://developer.spotify.com/dashboard and set them."
        )
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=auth)


def fetch_artists_for_genre(sp, genre, limit=50):
    """Search for top artists tagged with a genre, paging through results."""
    rows = []
    fetched = 0
    offset = 0
    page_size = 50
    while fetched < limit:
        batch = min(page_size, limit - fetched)
        try:
            results = sp.search(
                q=f'genre:"{genre}"', type="artist", limit=batch, offset=offset
            )
        except Exception as e:
            print(f"  [warn] search failed for genre={genre} offset={offset}: {e}")
            break

        items = results.get("artists", {}).get("items", [])
        if not items:
            break

        for a in items:
            rows.append(
                {
                    "artist_id": a["id"],
                    "artist_name": a["name"],
                    "genre_query": genre,
                    "genres_tagged": ";".join(a.get("genres", [])),
                    "popularity": a.get("popularity"),
                    "followers": a.get("followers", {}).get("total"),
                }
            )
        fetched += len(items)
        offset += len(items)
        time.sleep(0.1)  # be polite to the API
    return rows


def collect_snapshot():
    sp = get_client()
    os.makedirs(config.DATA_DIR, exist_ok=True)

    snapshot_date = datetime.date.today().isoformat()
    all_rows = []
    for genre in config.GENRES:
        print(f"Fetching genre: {genre}")
        rows = fetch_artists_for_genre(sp, genre, limit=config.ARTISTS_PER_GENRE)
        for r in rows:
            r["snapshot_date"] = snapshot_date
        all_rows.extend(rows)
        print(f"  -> {len(rows)} artists")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data collected. Check your credentials and network access.")
        return

    # Deduplicate within this snapshot (an artist can match multiple genre queries)
    df = df.sort_values("popularity", ascending=False).drop_duplicates(
        subset=["artist_id", "snapshot_date"], keep="first"
    )

    if os.path.exists(config.PANEL_FILE):
        existing = pd.read_csv(config.PANEL_FILE)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["artist_id", "snapshot_date"], keep="last"
        )
    else:
        combined = df

    combined.to_csv(config.PANEL_FILE, index=False)
    print(f"\nSaved {len(df)} artist rows for {snapshot_date}.")
    print(f"Panel file now has {len(combined)} total rows -> {config.PANEL_FILE}")


if __name__ == "__main__":
    collect_snapshot()
