"""
The recommendation system itself.

This is almost a direct copy of the SpotifyRecommender class from the
analysis notebook (see notebooks/). It lives in its own module so both
the offline refresh script (scripts/refresh.py) and the online API
(app/main.py) can import it without copy-pasting code.

Nothing in this file talks to the internet or reads huge files. It only
works with whatever DataFrame you give it.
"""

import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def parse_genres(value):
    """
    Turn the "genres" column into a normal Python list.

    In the CSV, genres are usually stored as a string that looks like a
    list, e.g. "['pop', 'k-pop']". This function turns that string back
    into an actual list. If it can't, it falls back to splitting on commas.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
    except (ValueError, SyntaxError):
        pass

    return [genre.strip() for genre in str(value).split(",") if genre.strip()]


class SpotifyRecommender:
    """
    A simple item-to-item recommendation system for a personal Spotify
    library. Give it a DataFrame of songs (with audio features, genres,
    key/mode/time_signature, and a spotify_url column) and it can suggest
    songs that are similar to a given song, using three methods:

    - "cosine": cosine similarity on audio features
    - "knn":    nearest neighbours (Euclidean distance) on audio features
    - "hybrid": audio features + genres + key/mode/time_signature together
    """

    def __init__(self, tracks_data):
        print("Starting the recommendation system...")

        self.data = tracks_data.copy().reset_index(drop=True)

        self.audio_features = [
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

        self._prepare_audio_features()
        self._prepare_cosine_model()
        self._prepare_knn_model()
        self._prepare_hybrid_model()

        print("Recommendation system is ready")

    def _prepare_audio_features(self):
        """Scale the continuous audio features to a 0-1 range."""
        self.scaler = MinMaxScaler()
        self.audio_matrix = self.scaler.fit_transform(self.data[self.audio_features])

    def _prepare_cosine_model(self):
        """Calculate cosine similarity between every pair of songs."""
        self.cosine_matrix = cosine_similarity(self.audio_matrix)

    def _prepare_knn_model(self):
        """Build a simple KNN model using Euclidean distance."""
        self.knn_model = NearestNeighbors(metric="euclidean", algorithm="brute")
        self.knn_model.fit(self.audio_matrix)

    def _prepare_hybrid_model(self):
        """Combine audio features, genres and music categories."""
        genre_lists = self.data["genres"].apply(parse_genres)

        self.genre_encoder = MultiLabelBinarizer()
        genre_matrix = self.genre_encoder.fit_transform(genre_lists)

        category_data = self.data[["key", "mode", "time_signature"]].astype(str)
        category_matrix = pd.get_dummies(category_data, dtype=float).values

        # Audio features get the largest weight; genres and categories
        # just add some extra context on top.
        self.hybrid_matrix = np.hstack(
            [self.audio_matrix, genre_matrix * 0.5, category_matrix * 0.3]
        )
        self.hybrid_similarity_matrix = cosine_similarity(self.hybrid_matrix)

    def find_song(self, song_name, artist_name=None):
        """Find the row position of a song in the dataset."""
        matches = self.data[
            self.data["song_name"].str.contains(
                song_name, case=False, na=False, regex=False
            )
        ]

        if artist_name is not None:
            matches = matches[
                matches["artist_name"].str.contains(
                    artist_name, case=False, na=False, regex=False
                )
            ]

        if len(matches) == 0:
            raise ValueError(f"Song not found: {song_name}")

        # If there are several matches, just use the first one.
        return matches.index[0]

    def recommend(self, song_name, artist_name=None, n_recommendations=10, method="cosine"):
        """
        Recommend songs that are similar to the given song.

        method can be "cosine", "knn" or "hybrid".
        """
        song_index = self.find_song(song_name, artist_name)

        if method == "cosine":
            scores = self.cosine_matrix[song_index]
            ranked_indices = np.argsort(scores)[::-1]

        elif method == "knn":
            neighbor_count = min(n_recommendations + 1, len(self.data))
            distances, indices = self.knn_model.kneighbors(
                self.audio_matrix[song_index].reshape(1, -1),
                n_neighbors=neighbor_count,
            )
            ranked_indices = indices[0]
            scores = np.zeros(len(self.data))
            scores[indices[0]] = 1 / (1 + distances[0])

        elif method == "hybrid":
            scores = self.hybrid_similarity_matrix[song_index]
            ranked_indices = np.argsort(scores)[::-1]

        else:
            raise ValueError("method must be 'cosine', 'knn', or 'hybrid'")

        # Don't recommend the song to itself.
        ranked_indices = [index for index in ranked_indices if index != song_index]
        ranked_indices = ranked_indices[:n_recommendations]

        result_columns = [
            "track_id",
            "song_name",
            "genres",
            "artist_name",
            "album_name",
            "popularity",
            "release_year",
            "added_at",
            "cluster",
            "cluster_top_2_features",
            "spotify_url",
        ]
        result_columns = [column for column in result_columns if column in self.data.columns]

        results = self.data.iloc[ranked_indices][result_columns].copy()
        results["similarity_score"] = scores[ranked_indices]

        return results.reset_index(drop=True)
