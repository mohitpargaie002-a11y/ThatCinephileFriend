# That Cinephile Friend

A semantic recommendation service for films, series, and anime. It embeds catalog descriptions locally with `all-MiniLM-L6-v2` and stores vectors in Qdrant Cloud.

## Run locally

Copy `.env.example` to `.env`, add credentials, install `requirements.txt`, then populate the catalog with:

```sh
python ingest/fetch_data.py
python ingest/build_index.py
```

Start the app with `uvicorn app.main:app --reload` and visit `http://localhost:8000`. `/discover` searches a mood or genre, while `/similar?title=...` uses an indexed title vector. `media_type`, `niche=true`, and `limit` are supported filters.

## Deploy

Create a Docker Hugging Face Space, push this repository, and add `TMDB_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` as Space secrets. The container exposes port 7860. Run ingestion separately before deployment; the running Space only reads Qdrant.
