"""
Merge my liked songs with the public catalogue (data/tracks.csv) to
attach audio features, keeping only tracks that appear in both. Reads
data/liked_songs.csv and data/tracks.csv, writes data/merged_cleaned.csv.
"""

import pandas as pd

liked_songs = pd.read_csv('data/liked_songs.csv')
liked_songs['release_year'] = liked_songs['release_date'].str[:4].astype(int)
liked_songs_up_to_2024 = liked_songs[liked_songs['release_year'] <= 2024]
print(f"{len(liked_songs_up_to_2024)} out of {len(liked_songs)} tracks released in or before 2024")

public_catalogue = pd.read_csv('data/tracks.csv')
merged = liked_songs_up_to_2024.merge(public_catalogue, on='track_id')
print(f"{len(merged)} out of {len(liked_songs_up_to_2024)} tracks merged")

# The public catalogue has its own version of a few columns we already have
# from Spotify (explicit, popularity, duration_ms, album_name, added_at),
# plus some chart-specific columns we don't need at all. Drop those, and
# keep the "_x" (liked_songs) version of the ones that collided.
cleaned = merged.drop(columns=[
    'release_date', 'release_date_precision', 'explicit_x', 'streams', 'track_artists',
    'explicit_y', 'chart', 'album_release_date', 'added_at_y', 'popularity_y', 'name',
    'track_album_album', 'duration_ms_y', 'available_markets', 'track_track_number',
    'rank', 'album_name_y', 'region', 'trend',
]).rename(columns={
    'album_name_x': 'album_name',
    'duration_ms_x': 'duration_ms',
    'popularity_x': 'popularity',
    'added_at_x': 'added_at',
})

cleaned['added_at'] = cleaned['added_at'].str[:4].astype(int)
cleaned.to_csv('data/merged_cleaned.csv', index=False, encoding="utf-8-sig")
print(f"Saved {len(cleaned)} tracks to data/merged_cleaned.csv")
