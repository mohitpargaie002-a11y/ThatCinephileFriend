"""Fast text cleanup and token-optimized formatting for embeddings."""
from __future__ import annotations

import html
import re

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    """Decode entities, remove markup, and collapse unusual whitespace."""
    plain = html.unescape(value or "")
    return _SPACE.sub(" ", _TAGS.sub(" ", plain)).strip()


def truncate_synopsis(overview: str, max_words: int = 40) -> str:
    """Keep the most salient initial sentences/words to maximize embedding speed while keeping full semantic intent."""
    words = overview.split()
    if len(words) <= max_words:
        return overview
    return " ".join(words[:max_words])


def embedding_text(title: str, overview: str, genres: list[str]) -> str:
    concise_overview = truncate_synopsis(normalize_text(overview), max_words=40)
    genre_str = ", ".join(normalize_text(g) for g in genres if g)
    return f"{normalize_text(title)}. {concise_overview} Genres: {genre_str}".strip()
