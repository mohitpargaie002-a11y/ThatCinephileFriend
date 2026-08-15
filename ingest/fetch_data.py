"""Fetch TMDB movies/TV and anime into a normalized JSONL file."""
from __future__ import annotations
import asyncio, json, os
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()
OUT = Path(__file__).resolve().parent.parent / "data" / "raw_items.jsonl"
TMDB = "https://api.themoviedb.org/3"


def enough_words(text: str | None) -> bool:
    return bool(text and len(text.split()) >= 20)


async def tmdb_records(client, kind: str, target: int, genres: dict[int, str]):
    path, media_type = ("movie/popular", "movie") if kind == "movie" else ("tv/popular", "series")
    yielded = 0
    for page in range(1, min(500, (target + 19) // 20) + 1):
        response = await client.get(f"{TMDB}/{path}", params={"api_key": os.environ["TMDB_API_KEY"], "page": page})
        response.raise_for_status()
        for item in response.json().get("results", []):
            overview = item.get("overview", "")
            if enough_words(overview):
                title = item.get("title") if media_type == "movie" else item.get("name")
                yield {
                    "id": f"{media_type}:{item['id']}",
                    "title": title,
                    "media_type": media_type,
                    "overview": overview,
                    "genres": [genres[g] for g in item.get("genre_ids", []) if g in genres],
                    "vote_average": item.get("vote_average", 0),
                    "vote_count": item.get("vote_count", 0),
                    "poster_url": f"https://image.tmdb.org/t/p/w500{item['poster_path']}" if item.get("poster_path") else None,
                }
                yielded += 1
                if yielded >= target:
                    return
        await asyncio.sleep(0.2)


async def tmdb_anime_records(client, target: int, genre_maps: dict[str, dict[int, str]]):
    """Fetch top anime series & movies from TMDB with Japanese original language and Animation genre."""
    yielded = 0
    endpoints = [("discover/tv", "tv", target * 3 // 4), ("discover/movie", "movie", target // 4)]
    for path, kind, sub_target in endpoints:
        sub_yielded = 0
        genre_map = genre_maps.get(kind, {})
        for page in range(1, min(500, (sub_target + 19) // 20) + 1):
            response = await client.get(
                f"{TMDB}/{path}",
                params={
                    "api_key": os.environ["TMDB_API_KEY"],
                    "with_genres": "16",
                    "with_original_language": "ja",
                    "sort_by": "popularity.desc",
                    "page": page,
                },
            )
            response.raise_for_status()
            for item in response.json().get("results", []):
                overview = item.get("overview", "")
                if enough_words(overview):
                    title = item.get("name") if kind == "tv" else item.get("title")
                    yield {
                        "id": f"anime:{item['id']}",
                        "title": title,
                        "media_type": "anime",
                        "overview": overview,
                        "genres": [genre_map[g] for g in item.get("genre_ids", []) if g in genre_map],
                        "vote_average": item.get("vote_average", 0),
                        "vote_count": item.get("vote_count", 0),
                        "poster_url": f"https://image.tmdb.org/t/p/w500{item['poster_path']}" if item.get("poster_path") else None,
                    }
                    yielded += 1
                    sub_yielded += 1
                    if sub_yielded >= sub_target or yielded >= target:
                        break
            if sub_yielded >= sub_target or yielded >= target:
                break
            await asyncio.sleep(0.2)


async def jikan_records(client, target: int):
    yielded = 0
    for page in range(1, min(100, (target + 24) // 25) + 1):
        response = await jikan_get(client, "https://api.jikan.moe/v4/top/anime", {"page": page, "limit": 25})
        if response is None:
            break
        for item in response.json().get("data", []):
            overview = item.get("synopsis", "")
            if enough_words(overview):
                yield {
                    "id": f"anime:{item['mal_id']}",
                    "title": item["title"],
                    "media_type": "anime",
                    "overview": overview,
                    "genres": [g["name"] for g in item.get("genres", [])],
                    "vote_average": item.get("score") or 0,
                    "vote_count": item.get("scored_by") or 0,
                    "poster_url": item.get("images", {}).get("jpg", {}).get("large_image_url"),
                }
                yielded += 1
                if yielded >= target:
                    return
        await asyncio.sleep(1)


async def jikan_get(client, url: str, params: dict, attempts: int = 2):
    """Attempt Jikan API, returning None if service is down/failing."""
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                return response
            if response.status_code not in (429, 500, 502, 503, 504):
                return None
            retry_after = float(response.headers.get("Retry-After", 0) or 0)
        except httpx.RequestError:
            retry_after = 0
        delay = max(retry_after, min(5, 1.5 * (2 ** attempt)))
        await asyncio.sleep(delay)
    return None


async def main():
    if not os.getenv("TMDB_API_KEY"):
        raise RuntimeError("TMDB_API_KEY is required")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as output:
        async with httpx.AsyncClient(timeout=30) as client:
            genre_maps = {}
            for kind in ("movie", "tv"):
                response = await client.get(f"{TMDB}/genre/{kind}/list", params={"api_key": os.environ["TMDB_API_KEY"]})
                response.raise_for_status()
                genre_maps[kind] = {g["id"]: g["name"] for g in response.json()["genres"]}

            for kind, count in (("movie", 10_000), ("tv", 5_000)):
                print(f"Fetching {count} {kind} titles...", flush=True)
                total = 0
                async for record in tmdb_records(client, kind, count, genre_maps[kind]):
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    if total % 1000 == 0:
                        print(f"  Fetched {total}/{count} {kind} items", flush=True)

            print("Fetching 2,000 anime titles...", flush=True)
            anime_count = 0
            # Try Jikan first, but if it fails/times out, seamlessly fallback to TMDB anime
            async for record in jikan_records(client, 2_000):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                anime_count += 1
                if anime_count % 500 == 0:
                    print(f"  Fetched {anime_count}/2000 anime items (Jikan)", flush=True)

            if anime_count < 2_000:
                needed = 2_000 - anime_count
                print(f"Fetching remaining {needed} anime titles via TMDB...", flush=True)
                async for record in tmdb_anime_records(client, needed, genre_maps):
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    anime_count += 1
                    if anime_count % 500 == 0 or anime_count == 2_000:
                        print(f"  Fetched {anime_count}/2000 anime items", flush=True)

            print("Finished fetching all catalog items.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
