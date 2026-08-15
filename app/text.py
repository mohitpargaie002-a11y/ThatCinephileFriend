"""Small, dependency-free cleanup for text sent to the embedding model."""
from __future__ import annotations

import html
import re

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    """Decode entities, remove markup, and collapse unusual whitespace."""
    plain = html.unescape(value or "")
    return _SPACE.sub(" ", _TAGS.sub(" ", plain)).strip()


def embedding_text(title: str, overview: str, genres: list[str]) -> str:
    return f"{normalize_text(title)}. {normalize_text(overview)} Genres: {', '.join(normalize_text(g) for g in genres)}"
