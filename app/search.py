"""Qdrant query helpers."""
from __future__ import annotations
from typing import Any
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
from .schemas import TitleResult


def result_from_point(point: Any) -> TitleResult:
    payload = point.payload or {}
    return TitleResult(id=str(payload.get("id", point.id)), title=payload["title"], media_type=payload["media_type"], overview=payload.get("overview", ""), genres=payload.get("genres", []), vote_average=float(payload.get("vote_average") or 0), vote_count=int(payload.get("vote_count") or 0), poster_url=payload.get("poster_url"), score=getattr(point, "score", None))


def build_filter(media_type: str | None = None, niche: bool = False) -> Filter | None:
    conditions = []
    if media_type:
        conditions.append(FieldCondition(key="media_type", match=MatchValue(value=media_type)))
    if niche:
        conditions.extend([FieldCondition(key="vote_average", range=Range(gte=7.5)), FieldCondition(key="vote_count", range=Range(lte=1500))])
    return Filter(must=conditions) if conditions else None


def search_points(client: Any, collection: str, vector: list[float], query_filter: Filter | None, limit: int):
    try:
        return client.query_points(collection_name=collection, query=vector, query_filter=query_filter, limit=limit, with_payload=True).points
    except AttributeError:
        return client.search(collection_name=collection, query_vector=vector, query_filter=query_filter, limit=limit, with_payload=True)
