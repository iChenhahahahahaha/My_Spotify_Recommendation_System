"""
Offline refresh job.

This script does all of the slow / heavy work, one time:

1. Loads your saved songs (data/merged_cleaned.csv).
2. Groups them into taste clusters (K-Means), same as the notebook.
3. Builds the item-to-item recommender (cosine / knn / hybrid).
4. Loads the public song catalogue (data/tracks.csv, can be ~900k rows)
   and scores every song in it against your taste, one taste cluster at
   a time, so every cluster ends up with its own recommendations.
5. Saves everything the API needs into the "artifacts/" folder.

Run it from the project root (the folder that contains "app/" and
"scripts/"), like this:

    python -m scripts.refresh

The API (app/main.py) never runs any of this itself — it only reads the
files this script writes to "artifacts/". That's why the API stays fast
even though step 4 here can take a while on a big catalogue.

Note: the notebook also computed a "category_score" (key/mode/time
signature match) in the personalization section, but that score was
never actually used in the final personalized_score formula. It's left
out here to keep this script simple.
"""

import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.recommender import SpotifyRecommender, parse_genres

load_dotenv()

# ---------------------------------------------------------------------
# Settings — change these in your .env file, not here.
# ---------------------------------------------------------------------

PERSONAL_DATA_PATH = os.getenv("PERSONAL_DATA_PATH", "data/merged_cleaned.csv")
PUBLIC_DATA_PATH = os.getenv("PUBLIC_DATA_PATH", "data/tracks.csv")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

N_CLUSTERS = 5
TOP_GENRE_COUNT = 15
AUDIO_WEIGHT = 0.80
GENRE_WEIGHT = 0.20
RECOMMENDATIONS_PER_CLUSTER = 40  # so every cluster is guaranteed some recommendations
MAX_PER_ARTIST = 2

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "tempo",
]


def load_personal_data():
    """Load and clean your saved songs, same as notebook cells 4-5."""
    print("Step 1/5: loading your saved songs...")

    data = pd.read_csv(PERSONAL_DATA_PATH)

    numeric_columns = AUDIO_FEATURES + [
        "key", "mode", "time_signature",
        "duration_ms", "popularity", "release_year", "added_at",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["added_at"] = data["added_at"].astype(int)
    data["release_year"] = data["release_year"].astype(int)

    data = data.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)

    print(f"  loaded {len(data)} personal tracks")
    return data


def add_taste_clusters(data):
    """Group your songs into N_CLUSTERS taste clusters, same as notebook cell 36-38."""
    print("Step 2/5: grouping your songs into taste clusters...")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[AUDIO_FEATURES])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(scaled_features)

    # Label each cluster with its two strongest audio features, just like
    # the notebook does, so recommendations can show a friendly label.
    cluster_top_2_map = {}
    for cluster_id in range(N_CLUSTERS):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = np.argsort(center)[::-1][:2]
        top_features = [AUDIO_FEATURES[i].replace("_", " ").title() for i in top_indices]
        cluster_top_2_map[cluster_id] = " + ".join(top_features)

    data["cluster_top_2_features"] = data["cluster"].map(cluster_top_2_map)

    print(f"  created {N_CLUSTERS} clusters")
    return data


def build_item_recommender(data):
    """Build the item-to-item recommender (cosine/knn/hybrid), same as notebook cell 51-53."""
    print("Step 3/5: building the item-to-item recommender...")
    return SpotifyRecommender(data)


def select_diverse_tracks(scored_tracks, n_recommendations, max_per_artist):
    """Pick the top scoring tracks, but cap how many come from one artist."""
    selected_rows = []
    artist_counts = {}

    sorted_tracks = scored_tracks.sort_values("personalized_score", ascending=False)

    for _, row in sorted_tracks.iterrows():
        artist = row["artist_name"]
        count = artist_counts.get(artist, 0)

        if count >= max_per_artist:
            continue

        selected_rows.append(row)
        artist_counts[artist] = count + 1

        if len(selected_rows) >= n_recommendations:
            break

    return pd.DataFrame(selected_rows)


def build_personalized_recommendations(personal_tracks):
    """
    Score the whole public catalogue against your taste, and keep the
    best, most diverse recommendations FOR EACH CLUSTER. Same scoring
    logic as notebook cells 81-113, but the final selection is done one
    cluster at a time so every taste cluster ends up with some
    recommendations, instead of the biggest/most-popular cluster
    crowding out the others (that used to happen when picking a single
    global top-200 list).
    """
    print("Step 4/5: loading the public song catalogue...")

    public_columns = [
        "track_id", "name", "track_artists", "album_name",
        "album_release_date", "popularity", "genres", "explicit", "duration_ms",
    ] + AUDIO_FEATURES + ["key", "mode", "time_signature"]

    public_tracks = pd.read_csv(PUBLIC_DATA_PATH, usecols=public_columns, low_memory=False)

    public_tracks = public_tracks.rename(columns={
        "name": "song_name",
        "track_artists": "artist_name",
        "album_release_date": "release_date",
    })
    public_tracks = public_tracks.drop_duplicates(subset="track_id").copy()

    numeric_columns = AUDIO_FEATURES + ["key", "mode", "time_signature", "popularity", "duration_ms"]
    for column in numeric_columns:
        public_tracks[column] = pd.to_numeric(public_tracks[column], errors="coerce")
    public_tracks = public_tracks.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)
    public_tracks["artist_name"] = public_tracks["artist_name"].fillna("Unknown Artist")
    public_tracks["track_id"] = public_tracks["track_id"].astype(str)

    print(f"  loaded {len(public_tracks)} public tracks")

    print("Step 5/5: scoring every public track against your taste (this is the slow step)...")

    # --- audio score: how close is each public song to one of your taste clusters? ---
    scaler = MinMaxScaler()
    catalogue_audio_matrix = scaler.fit_transform(public_tracks[AUDIO_FEATURES]).astype("float32")
    personal_audio_matrix = scaler.transform(personal_tracks[AUDIO_FEATURES]).astype("float32")

    cluster_ids = sorted(personal_tracks["cluster"].unique())
    user_profile_list = [
        personal_audio_matrix[personal_tracks["cluster"] == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ]
    user_profile_matrix = np.vstack(user_profile_list).astype("float32")

    distances = pairwise_distances(
        catalogue_audio_matrix, user_profile_matrix, metric="euclidean", n_jobs=-1
    )
    best_profile_positions = np.argmin(distances, axis=1)
    minimum_distances = np.min(distances, axis=1)
    max_possible_distance = np.sqrt(len(AUDIO_FEATURES))

    public_tracks["audio_profile_score"] = np.clip(
        1 - minimum_distances / max_possible_distance, 0, 1
    )
    public_tracks["matched_taste_cluster"] = [cluster_ids[p] for p in best_profile_positions]

    # --- genre score: how much do each song's genres overlap with your favorite genres? ---
    personal_genres = []
    for genre_value in personal_tracks["genres"]:
        personal_genres.extend(parse_genres(genre_value))
    personal_genre_counts = Counter(personal_genres)

    favorite_genre_items = personal_genre_counts.most_common(TOP_GENRE_COUNT)
    favorite_genre_names = [genre for genre, _ in favorite_genre_items]
    user_genre_vector = np.array([np.sqrt(count) for _, count in favorite_genre_items])
    user_genre_vector_norm = np.linalg.norm(user_genre_vector)

    def get_personal_genre_score(genre_value):
        track_genres = set(parse_genres(genre_value))
        track_genre_vector = np.array(
            [1.0 if genre in track_genres else 0.0 for genre in favorite_genre_names]
        )
        track_vector_norm = np.linalg.norm(track_genre_vector)

        if track_vector_norm == 0 or user_genre_vector_norm == 0:
            return 0.0

        return float(
            np.dot(user_genre_vector, track_genre_vector)
            / (user_genre_vector_norm * track_vector_norm)
        )

    public_tracks["genre_preference_score"] = public_tracks["genres"].apply(get_personal_genre_score)

    # --- final score ---
    public_tracks["personalized_score"] = (
        AUDIO_WEIGHT * public_tracks["audio_profile_score"]
        + GENRE_WEIGHT * public_tracks["genre_preference_score"]
    )

    # --- remove songs you already saved ---
    saved_track_ids = set(personal_tracks["track_id"].astype(str))
    candidate_tracks = public_tracks[~public_tracks["track_id"].isin(saved_track_ids)].copy()

    # --- pick the best, most diverse songs, ONE CLUSTER AT A TIME ---
    # This is what guarantees that filtering /recommend/personalized by any
    # cluster_id returns something, instead of some clusters being empty.
    per_cluster_results = []
    for cluster_id in cluster_ids:
        cluster_candidates = candidate_tracks[candidate_tracks["matched_taste_cluster"] == cluster_id]

        cluster_picks = select_diverse_tracks(
            cluster_candidates,
            n_recommendations=RECOMMENDATIONS_PER_CLUSTER,
            max_per_artist=MAX_PER_ARTIST,
        )
        per_cluster_results.append(cluster_picks)

        print(f"  cluster {cluster_id}: {len(cluster_picks)} recommendations")

    personalized_recommendations = pd.concat(per_cluster_results, ignore_index=True)
    personalized_recommendations = personalized_recommendations.sort_values(
        "personalized_score", ascending=False
    ).reset_index(drop=True)

    personalized_recommendations["spotify_url"] = (
        "https://open.spotify.com/track/" + personalized_recommendations["track_id"].astype(str)
    )

    print(f"  kept {len(personalized_recommendations)} personalized recommendations total")
    return personalized_recommendations


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    personal_data = load_personal_data()
    personal_data = add_taste_clusters(personal_data)

    item_recommender = build_item_recommender(personal_data)
    recommender_path = os.path.join(ARTIFACTS_DIR, "recommender.pkl")
    joblib.dump(item_recommender, recommender_path)
    print(f"  saved {recommender_path}")

    personalized_recommendations = build_personalized_recommendations(personal_data)
    recommendations_path = os.path.join(ARTIFACTS_DIR, "personalized_recommendations.csv")
    personalized_recommendations.to_csv(recommendations_path, index=False)
    print(f"  saved {recommendations_path}")

    print("\nDone! Restart (or rebuild/redeploy) the API so it picks up the new files.")


if __name__ == "__main__":
    main()
