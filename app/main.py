from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .embedder import OnnxEmbedder
from .schemas import MediaType, ResultsResponse
from .search import build_filter, result_from_point, search_points
from .text import embedding_text

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cinephile")

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# In-memory LRU cache for query embeddings
_EMBEDDING_CACHE: dict[str, list[float]] = {}
MAX_CACHE_SIZE = 2500


async def get_embedding(text: str, embedder: OnnxEmbedder) -> list[float]:
    """Encode text to normalized vector asynchronously on a worker thread with LRU caching."""
    clean_text = text.strip()
    if clean_text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[clean_text]

    vector = await asyncio.to_thread(embedder.encode, clean_text)
    if len(_EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
        _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
    _EMBEDDING_CACHE[clean_text] = vector
    return vector


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ultra-lightweight ONNX embedder for %s...", MODEL_NAME)
    app.state.embedder = OnnxEmbedder(MODEL_NAME)

    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if url:
        logger.info("Connecting to Qdrant cluster: %s", url)
        app.state.client = QdrantClient(url=url, api_key=key, timeout=20)
    else:
        logger.warning("QDRANT_URL is not set; running in degraded mode.")
        app.state.client = None
    app.state.collection = os.getenv("QDRANT_COLLECTION", "titles")
    logger.info("That Cinephile Friend is online and ready for requests.")
    yield
    logger.info("Shutting down application.")


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
async def log_and_secure_requests(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Unhandled exception on %s %s (%.1fms): %s\n%s",
            request.method,
            request.url.path,
            duration_ms,
            exc,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred while processing your request."},
            headers={
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "SAMEORIGIN",
                "Referrer-Policy": "strict-origin-when-cross-origin",
            },
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    if request.url.path not in ("/health", "/favicon.ico") or response.status_code >= 400:
        logger.info(
            "%s %s%s [%d] in %.1fms",
            request.method,
            request.url.path,
            f"?{request.url.query}" if request.url.query else "",
            response.status_code,
            duration_ms,
        )

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP %d on %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        },
    )


app.mount("/frontend", StaticFiles(directory=FRONTEND), name="frontend")


def dependencies():
    if app.state.client is None:
        raise HTTPException(503, "Qdrant is not configured. Set QDRANT_URL and QDRANT_API_KEY.")
    return app.state.client, app.state.embedder, app.state.collection


async def tmdb_fallback_vector(title: str, embedder: OnnxEmbedder) -> list[float]:
    """Search TMDB for movies/TV/anime or fallback to semantic text encoding."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.info("No TMDB_API_KEY set; embedding query text directly: '%s'", title)
        return await get_embedding(title, embedder)

    try:
        async with httpx.AsyncClient(timeout=6) as http:
            search = await http.get(
                "https://api.themoviedb.org/3/search/multi",
                params={"api_key": api_key, "query": title, "include_adult": "false"},
            )
            if search.status_code == 200:
                results = [
                    item for item in search.json().get("results", [])
                    if item.get("media_type") in ("movie", "tv") and item.get("overview")
                ]
                if results:
                    selected = next(
                        (
                            m for m in results
                            if (m.get("title") or m.get("name", "")).casefold() == title.casefold()
                        ),
                        results[0],
                    )
                    item_title = selected.get("title") or selected.get("name") or title
                    overview = selected.get("overview", "")
                    genre_names = [str(g) for g in selected.get("genre_ids", [])]
                    text = embedding_text(item_title, overview, genre_names)
                    logger.info("TMDB match found for '%s': '%s' -> embedding overview", title, item_title)
                    return await get_embedding(text, embedder)
    except Exception as exc:
        logger.warning("TMDB lookup exception for '%s': %s. Falling back to direct query embedding.", title, exc)

    # Fallback to direct semantic concept embedding
    logger.info("No exact TMDB entry for '%s'; embedding as semantic concept.", title)
    return await get_embedding(title, embedder)


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
    client, embedder, collection = dependencies()
    clean_title = title.strip()

    title_filter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=clean_title))])
    matches, _ = client.scroll(collection_name=collection, scroll_filter=title_filter, limit=1, with_vectors=True)
    source_id = None
    if matches:
        vector = matches[0].vector
        source_id = str(matches[0].id)
        if isinstance(vector, dict):
            vector = next(iter(vector.values()))
    else:
        vector = await tmdb_fallback_vector(clean_title, embedder)

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
    client, embedder, collection = dependencies()
    query_filter = build_filter(media_type, niche)
    if query and query.strip():
        vector = await get_embedding(query.strip(), embedder)
        points = search_points(client, collection, vector, query_filter, limit)
    else:
        points, _ = client.scroll(collection_name=collection, scroll_filter=query_filter, limit=limit, with_payload=True)
    results = [result_from_point(p) for p in points]
    return ResultsResponse(results=results, total=len(results))
