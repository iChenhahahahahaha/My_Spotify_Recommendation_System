# This image contains the API code (app/) AND the artifacts/ folder
# that scripts/refresh.py already produced. Cloud platforms like Render
# can't mount a folder from your laptop, so the artifacts have to be
# baked into the image instead.
#
# This means: every time you rerun refresh.py locally, you need to
# rebuild and redeploy this image so the container gets the new
# artifacts/. For a small personal project that refreshes occasionally,
# that's an acceptable trade-off for staying simple.
#
# Before building, make sure artifacts/ exists locally:
#   python -m scripts.refresh

FROM python:3.11-slim

WORKDIR /code

# Install dependencies first, so Docker can reuse this layer as long as
# requirements.txt doesn't change (makes rebuilds faster).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what the API actually needs to run: the app/ package and
# the artifacts/ folder. notebooks/, tests/, scripts/ and data/ are
# excluded by .dockerignore — they're not needed to serve requests.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Cloud hosts like Render tell your app which port to listen on through
# the PORT environment variable (it's not always 8000). This falls back
# to 8000 for local "docker run", where nobody sets PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
