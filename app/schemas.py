from typing import Literal

from pydantic import BaseModel, Field

MediaType = Literal["movie", "series", "anime"]


class TitleResult(BaseModel):
    id: str
    title: str
    media_type: MediaType
    overview: str
    genres: list[str] = Field(default_factory=list)
    vote_average: float = 0
    vote_count: int = 0
    poster_url: str | None = None
    score: float | None = None


class ResultsResponse(BaseModel):
    results: list[TitleResult]
    total: int
