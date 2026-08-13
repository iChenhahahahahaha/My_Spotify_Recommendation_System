"""
The online API.

This app does NOT read data/merged_cleaned.csv or data/tracks.csv, and it
does NOT train anything. It only loads the files that scripts/refresh.py
already produced (in artifacts/) once, when the app starts. Every
request after that just looks things up in memory, so it stays fast no
matter how big the public catalogue was.

Run it from the project root with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

load_dotenv()

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

# These start empty and get filled in by load_artifacts() below, when the
# app starts up. Nothing here talks to Spotify or does any heavy work.
recommender = None
personalized_recommendations = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the API starts, before it accepts any requests."""
    global recommender, personalized_recommendations

    recommender_path = os.path.join(ARTIFACTS_DIR, "recommender.pkl")
    recommendations_path = os.path.join(ARTIFACTS_DIR, "personalized_recommendations.csv")

    if not os.path.exists(recommender_path) or not os.path.exists(recommendations_path):
        raise RuntimeError(
            f"Could not find artifacts in '{ARTIFACTS_DIR}'. "
            "Run `python -m scripts.refresh` first to create them."
        )

    print("Loading recommender.pkl ...")
    recommender = joblib.load(recommender_path)

    print("Loading personalized_recommendations.csv ...")
    personalized_recommendations = pd.read_csv(recommendations_path)

    print("API is ready.")

    yield  # the app runs while paused here

    # nothing to clean up on shutdown


app = FastAPI(title="Personal Spotify Recommender API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Quick check that the API is up and the artifacts loaded correctly."""
    return {
        "status": "ok",
        "recommender_loaded": recommender is not None,
        "personalized_songs_available": (
            0 if personalized_recommendations is None else len(personalized_recommendations)
        ),
    }


@app.get("/recommend")
def recommend(
    song: str = Query(..., description="Song name, e.g. 平行世界"),
    artist: Optional[str] = Query(None, description="Artist name (optional, narrows the search)"),
    method: str = Query("cosine", description="cosine, knn, or hybrid"),
    n: int = Query(10, ge=1, le=50, description="How many recommendations to return"),
):
    """
    Item-to-item recommendations: songs from YOUR library that are similar
    to the song you asked about.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender is not loaded yet.")

    try:
        results = recommender.recommend(
            song_name=song, artist_name=artist, n_recommendations=n, method=method
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

    # Some rows in the public dataset are missing a value here or there
    # (e.g. no popularity score). Pandas stores that as NaN, which is not
    # valid JSON, so swap any NaN for None before returning.
    results = results.where(pd.notnull(results), None)
    return results.to_dict(orient="records")


@app.get("/recommend/personalized")
def recommend_personalized(
    cluster_id: Optional[int] = Query(
        None, description="Only show songs matched to this taste cluster"
    ),
    n: int = Query(10, ge=1, le=100, description="How many recommendations to return"),
):
    """
    New songs (not already in your library) from the public catalogue,
    picked because they match your overall taste. These scores were all
    computed offline by scripts/refresh.py — this endpoint just filters
    and sorts a table that was already sitting on disk.
    """
    if personalized_recommendations is None:
        raise HTTPException(status_code=503, detail="Recommendations are not loaded yet.")

    results = personalized_recommendations

    if cluster_id is not None:
        results = results[results["matched_taste_cluster"] == cluster_id]

    results = results.sort_values("personalized_score", ascending=False).head(n)

    # Same NaN issue as /recommend: the public dataset has some missing
    # values that dropna() during refresh didn't catch, so clean those up
    # here too before returning JSON.
    results = results.where(pd.notnull(results), None)
    return results.to_dict(orient="records")
