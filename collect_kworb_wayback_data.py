"""
collect_kworb_wayback_data.py
--------------------------------
Builds a REAL multi-year artist-level stream panel without waiting months
for live data collection, by pulling historical snapshots of Kworb's
Spotify "Daily Chart - Global" page from the Internet Archive's Wayback
Machine.

WHY THIS WORKS
--------------
kworb.net/spotify/country/global_daily.html only ever shows *today's*
top-200 chart -- there's no per-date URL. But the Wayback Machine has been
crawling and snapshotting that exact page for years. Fetching an archived
snapshot from, say, January 2019 gives you the chart exactly as it looked
that day: rank, artist, track, and that day's stream count for the top 200
tracks worldwide. Pull one snapshot per year and you get a genuine
cross-sectional distribution of streams-across-artists for each year --
precisely the multi-year panel the power-law/inequality analysis needs.

WHAT THIS MEASURES (read before writing your methods section)
---------------------------------------------------------------
This is TOP-200-CHART concentration, not whole-market concentration. It
captures how unequal streaming is *among songs that make the global daily
chart* on a given day -- it says nothing about the much larger number of
artists who never chart at all. This is actually a completely standard
scope restriction in the "superstar economics" literature (e.g. Billboard
Hot 100 concentration studies use exactly this kind of top-chart data), so
it's a legitimate and defensible measure -- just be explicit in your
write-up that your inequality metrics describe concentration *within the
hit-making tier*, not across all artists on Spotify.

Multiple songs by the same artist on the same day's chart are summed
together into one artist-day observation. Featured/collaborating artists
are recorded too (each gets credited with the song's full streams -- a
common simplifying convention; note it as a limitation if you split credit
differently in your write-up).

USAGE
-----
    python collect_kworb_wayback_data.py

Edit SNAPSHOT_DATES below to control which dates get pulled. One date per
year is enough for the year-over-year inequality trend; add more dates
per year (e.g. one per quarter) for finer-grained analysis.
"""

import json
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

import config

TARGET_PAGE = "kworb.net/spotify/country/global_daily.html"

# One representative date per year. Pick a date unlikely to be a chart
# outlier (avoid Dec 25-31, which is dominated by holiday songs, unless
# you're specifically studying seasonality).
SNAPSHOT_DATES = [
    "2016-06-15",
    "2017-06-15",
    "2018-06-15",
    "2019-06-15",
    "2020-06-15",
    "2021-06-15",
    "2022-06-15",
    "2023-06-15",
    "2024-06-15",
    "2025-06-15",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research script; econometrics coursework; "
                  "contact: replace-with-your-email@example.com)"
}

CDX_API = "http://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "http://web.archive.org/web"


def find_snapshot(target_url, date_str, search_window_days=10):
    """
    Uses the Wayback CDX API to find the snapshot of `target_url` closest
    to `date_str` (YYYY-MM-DD), searching up to `search_window_days` before
    and after if no exact-day snapshot exists.
    """
    date_compact = date_str.replace("-", "")
    from_date = (pd.Timestamp(date_str) - pd.Timedelta(days=search_window_days)).strftime("%Y%m%d")
    to_date = (pd.Timestamp(date_str) + pd.Timedelta(days=search_window_days)).strftime("%Y%m%d")

    params = {
        "url": target_url,
        "from": from_date,
        "to": to_date,
        "output": "json",
        "filter": "statuscode:200",
        "limit": 50,
    }
    resp = requests.get(CDX_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    rows = resp.json()

    if len(rows) <= 1:  # first row is just the header
        return None

    header, *data_rows = rows
    idx_timestamp = header.index("timestamp")

    # Pick the snapshot timestamp closest to the target date
    target_ts = pd.Timestamp(date_str)
    best_row = min(
        data_rows,
        key=lambda r: abs(pd.Timestamp(r[idx_timestamp][:8]) - target_ts),
    )
    timestamp = best_row[idx_timestamp]
    snapshot_url = f"{WAYBACK_BASE}/{timestamp}/https://{target_url}"
    actual_date = pd.Timestamp(timestamp[:8]).strftime("%Y-%m-%d")
    return snapshot_url, actual_date


ARTIST_HREF_RE = re.compile(r"/spotify/artist/([A-Za-z0-9]+)\.html")


def parse_chart_page(html):
    """
    Parses a Kworb 'Daily Chart - Global' page (archived or live) into a
    list of dicts: one row per (artist, chart entry), with that day's
    stream count. Handles multiple credited artists per track (primary +
    featured "(w/ ...)").
    """
    soup = BeautifulSoup(html, "html.parser")

    # Kworb's chart table is the largest table on the page.
    tables = soup.find_all("table")
    if not tables:
        return []
    chart_table = max(tables, key=lambda t: len(t.find_all("tr")))

    rows = chart_table.find_all("tr")
    records = []

    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 7:
            continue  # skip header row / malformed rows

        # Identify which cell holds the artist/title links (has artist hrefs)
        title_cell = None
        for c in cells:
            if c.find("a", href=ARTIST_HREF_RE):
                title_cell = c
                break
        if title_cell is None:
            continue

        artist_links = title_cell.find_all("a", href=ARTIST_HREF_RE)
        artists = []
        for a in artist_links:
            m = ARTIST_HREF_RE.search(a["href"])
            if m:
                artists.append((m.group(1), a.get_text(strip=True)))

        if not artists:
            continue

        # Streams column: Kworb's daily chart has Streams as the first
        # numeric column after Pk/(x?) -- identify by picking the numeric
        # cell that follows the title cell.
        title_idx = cells.index(title_cell)
        streams_val = None
        for c in cells[title_idx + 1:]:
            text = c.get_text(strip=True).replace(",", "")
            if text.lstrip("-").isdigit() and int(text) > 1000:
                streams_val = int(text)
                break

        if streams_val is None:
            continue

        for artist_id, artist_name in artists:
            records.append(
                {
                    "artist_id": artist_id,
                    "artist_name": artist_name,
                    "streams_est": streams_val,
                }
            )

    return records


def build_panel_from_wayback():
    all_rows = []

    for date_str in SNAPSHOT_DATES:
        year = int(date_str[:4])
        print(f"Fetching snapshot near {date_str}...")
        result = find_snapshot(TARGET_PAGE, date_str)
        if result is None:
            print(f"  [warn] no snapshot found near {date_str}, skipping.")
            continue

        snapshot_url, actual_date = result
        print(f"  -> using snapshot from {actual_date} ({snapshot_url})")

        try:
            resp = requests.get(snapshot_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [warn] failed to fetch snapshot: {e}")
            continue

        records = parse_chart_page(resp.text)
        if not records:
            print(f"  [warn] no chart rows parsed for {actual_date}")
            continue

        # Sum multiple chart entries for the same artist on the same day
        df_day = pd.DataFrame(records)
        df_day = (
            df_day.groupby(["artist_id", "artist_name"], as_index=False)
            ["streams_est"].sum()
        )
        df_day["year"] = year
        df_day["snapshot_date"] = actual_date
        df_day["genre_query"] = "unknown"  # Kworb doesn't provide genre

        all_rows.append(df_day)
        print(f"  -> {len(df_day)} unique artists captured")

        time.sleep(1.5)  # be polite to archive.org

    if not all_rows:
        print("\nNo data collected. Check your network connection and the "
              "SNAPSHOT_DATES list.")
        return None

    panel = pd.concat(all_rows, ignore_index=True)
    panel = panel.sort_values(["year", "streams_est"], ascending=[True, False])
    panel.to_csv(config.PANEL_FILE, index=False)
    print(f"\nSaved {len(panel)} rows across {panel['year'].nunique()} years "
          f"-> {config.PANEL_FILE}")
    return panel


if __name__ == "__main__":
    build_panel_from_wayback()
