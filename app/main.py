from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer
import httpx

from .schemas import MediaType, ResultsResponse
from .search import build_filter, result_from_point, search_points
from .text import embedding_text

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SentenceTransformer(MODEL_NAME)
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    app.state.client = QdrantClient(url=url, api_key=key, timeout=20) if url else None
    app.state.collection = os.getenv("QDRANT_COLLECTION", "titles")
    yield


app = FastAPI(title="That Cinephile Friend", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.mount("/frontend", StaticFiles(directory=FRONTEND), name="frontend")


def dependencies():
    if app.state.client is None:
        raise HTTPException(503, "Qdrant is not configured. Set QDRANT_URL and QDRANT_API_KEY.")
    return app.state.client, app.state.model, app.state.collection


async def tmdb_fallback_vector(title: str, model: SentenceTransformer) -> list[float]:
    """Embed an unindexed TMDB movie without adding it to Qdrant."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise HTTPException(404, f"No indexed title named '{title}', and TMDB fallback is unavailable.")
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            search = await http.get("https://api.themoviedb.org/3/search/movie", params={"api_key": api_key, "query": title})
            search.raise_for_status()
            candidates = search.json().get("results", [])
            if not candidates:
                raise HTTPException(404, f"No indexed or TMDB movie named '{title}'.")
            selected = next((m for m in candidates if m.get("title", "").casefold() == title.casefold()), candidates[0])
            details = await http.get(f"https://api.themoviedb.org/3/movie/{selected['id']}", params={"api_key": api_key})
            details.raise_for_status()
        movie = details.json()
        text = embedding_text(movie.get("title", title), movie.get("overview", ""), [g["name"] for g in movie.get("genres", [])])
        return model.encode(text, normalize_embeddings=True).tolist()
    except HTTPException:
        raise
    except httpx.HTTPError:
        raise HTTPException(502, "Failed to retrieve title details from TMDB.")


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(FRONTEND / "index.html")


@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
async def favicon():
    return FileResponse(FRONTEND / "favicon.svg", media_type="image/svg+xml")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    configured = app.state.client is not None
    return {"status": "ok" if configured else "degraded", "qdrant_configured": configured}


@app.get("/similar", response_model=ResultsResponse)
async def similar(
    title: str = Query(min_length=1, max_length=200),
    media_type: MediaType | None = None,
    limit: int = Query(default=10, ge=1, le=50),
):
    client, model, collection = dependencies()
    title_filter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=title))])
    matches, _ = client.scroll(collection_name=collection, scroll_filter=title_filter, limit=1, with_vectors=True)
    source_id = None
    if matches:
        vector = matches[0].vector
        source_id = str(matches[0].id)
        if isinstance(vector, dict):
            vector = next(iter(vector.values()))
    else:
        vector = await tmdb_fallback_vector(title, model)
    points = search_points(client, collection, vector, build_filter(media_type), limit + 1)
    results = [result_from_point(p) for p in points if str(p.id) != source_id][:limit]
    return ResultsResponse(results=results, total=len(results))


@app.get("/discover", response_model=ResultsResponse)
async def discover(
    query: str | None = Query(default=None, max_length=500),
    media_type: MediaType | None = None,
    niche: bool = False,
    limit: int = Query(default=10, ge=1, le=50),
):
    client, model, collection = dependencies()
    query_filter = build_filter(media_type, niche)
    if query and query.strip():
        vector = model.encode(query.strip(), normalize_embeddings=True).tolist()
        points = search_points(client, collection, vector, query_filter, limit)
    else:
        points, _ = client.scroll(collection_name=collection, scroll_filter=query_filter, limit=limit, with_payload=True)
    results = [result_from_point(p) for p in points]
    return ResultsResponse(results=results, total=len(results))
