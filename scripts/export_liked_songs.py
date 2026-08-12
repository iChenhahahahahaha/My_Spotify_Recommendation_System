"""
Export My Spotify Liked Songs (Saved Tracks) metadata to CSV

Fields included:
  track_id, song_name, artist_name, artist_id, album_name,
  release_date, release_date_precision, duration_ms, popularity,
  explicit, added_at, artist_genres, spotify_url

Usage:
1. pip install spotipy
2. Replace CLIENT_ID / CLIENT_SECRET below with your own values
3. python export_liked_songs.py
   This will generate data/liked_songs.csv
"""

import csv
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ====== Replace with your own values ======
CLIENT_ID = "xxxx"
CLIENT_SECRET = "xxxx"
REDIRECT_URI = "http://127.0.0.1:8888/callback"
OUTPUT_FILE = "data/liked_songs.csv"
# ===========================================

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-library-read"
))

print("Fetching liked songs...")

all_items = []
limit = 50
offset = 0

while True:
    resp = sp.current_user_saved_tracks(limit=limit, offset=offset)
    items = resp["items"]
    if not items:
        break
    all_items.extend(items)
    print(f"  Fetched {len(all_items)} tracks so far...")
    offset += limit
    if resp["next"] is None:
        break
    time.sleep(0.1)  # avoid hitting rate limits

print(f"Total tracks fetched: {len(all_items)}")
print()

# ---- Batch fetch artist genres (not included in track object, needs separate call) ----
# Spotify /artists endpoint supports up to 50 artist ids per request; cache to avoid duplicates
artist_ids = set()
for item in all_items:
    for artist in item["track"]["artists"]:
        artist_ids.add(artist["id"])

artist_ids = list(artist_ids)
artist_genre_map = {}

print(f"Fetching genre info for {len(artist_ids)} artists...")
for i in range(0, len(artist_ids), 50):
    batch = artist_ids[i:i + 50]
    resp = sp.artists(batch)
    for artist in resp["artists"]:
        artist_genre_map[artist["id"]] = artist.get("genres", [])
    time.sleep(0.1)

print("Artist info fetched")
print()

# ---- Build rows and write to CSV ----
rows = []
for item in all_items:
    track = item["track"]
    if track is None:
        continue  # rare case: track has been removed/unavailable, track will be null

    album = track["album"]
    main_artist = track["artists"][0]
    all_artist_names = ", ".join(a["name"] for a in track["artists"])
    genres = artist_genre_map.get(main_artist["id"], [])

    rows.append({
        "track_id": track["id"],
        "song_name": track["name"],
        "artist_name": all_artist_names,
        "artist_id": main_artist["id"],
        "album_name": album["name"],
        "release_date": album["release_date"],
        "release_date_precision": album["release_date_precision"],
        "duration_ms": track["duration_ms"],
        "popularity": track["popularity"],
        "explicit": track["explicit"],
        "added_at": item["added_at"],
        "artist_genres": "; ".join(genres),
        "spotify_url": track["external_urls"]["spotify"],
    })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Done! Exported {len(rows)} tracks to {OUTPUT_FILE}")
