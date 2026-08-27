"""
Configuration for the Spotify Superstar Economics project.
"""

# --- Spotify API credentials ---
# Set these as environment variables before running collect_data.py:
#   export SPOTIPY_CLIENT_ID="your_client_id"
#   export SPOTIPY_CLIENT_SECRET="your_client_secret"
# Get credentials at https://developer.spotify.com/dashboard

# --- Genres to sample artists from (broad coverage reduces genre-specific bias) ---
GENRES = [
    "pop", "hip-hop", "rock", "country", "latin", "edm",
    "r-n-b", "indie", "classical", "jazz", "k-pop", "reggaeton",
]

# --- How many artists to pull per genre per collection run ---
ARTISTS_PER_GENRE = 50

# --- Data storage ---
DATA_DIR = "data"
PANEL_FILE = f"{DATA_DIR}/artist_panel.csv"

# --- Known platform/algorithm shocks for structural break testing ---
# (dates are approximate, used to build era dummies in the regression)
STRUCTURAL_BREAKS = {
    "discover_weekly_launch": "2015-07-01",   # Spotify's algorithmic playlist push
    "tiktok_music_era": "2019-01-01",         # TikTok becomes a major discovery/virality channel
}

# --- Inequality metric settings ---
TOP_SHARES = [0.01, 0.05, 0.10]  # top 1%, 5%, 10% stream/popularity share
