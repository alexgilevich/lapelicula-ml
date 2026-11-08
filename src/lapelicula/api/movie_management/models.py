# Generated from the OpenAPI spec at http://127.0.0.1:5003/docs/v1.json
# (Internal Movie Management API v1). Regenerate with:
#   curl -s http://127.0.0.1:5003/docs/v1.json -o movie_api.json
#   uvx --from datamodel-code-generator datamodel-codegen --input movie_api.json --input-file-type openapi \
#     --output-model-type pydantic_v2.BaseModel --output src/movie_api/models.py

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, confloat, conint, constr


class Movie(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    id: conint(ge=1, le=9223372036854776000)
    tmdbId: conint(ge=1, le=9223372036854776000)
    title: constr(min_length=1)
    description: constr(min_length=1)
    year: conint(ge=1800, le=2100)
    releaseDate: date
    ratingCount: conint(ge=0, le=9223372036854776000)
    ratingAverage: confloat(ge=0.0, le=10.0)
    adult: bool
    tagline: str | None = None
    imdbId: int | None = None
    posterUri: str | None = None
    budget: float | None = None
    revenue: float | None = None
    action: bool | None = None
    adventure: bool | None = None
    animation: bool | None = None
    comedy: bool | None = None
    crime: bool | None = None
    documentary: bool | None = None
    drama: bool | None = None
    fantasy: bool | None = None
    filmNoir: bool | None = None
    horror: bool | None = None
    kids: bool | None = None
    musical: bool | None = None
    mystery: bool | None = None
    romance: bool | None = None
    sciFi: bool | None = None
    thriller: bool | None = None
    war: bool | None = None
    western: bool | None = None
    anger: float | None = None
    anticipation: float | None = None
    disgust: float | None = None
    fear: float | None = None
    joy: float | None = None
    love: float | None = None
    optimism: float | None = None
    pessimism: float | None = None
    sadness: float | None = None
    surprise: float | None = None
    trust: float | None = None
    originCountries: list[str] | None = None
    productionCountries: list[str] | None = None
    embedding: list[float] = Field(..., max_length=384, min_length=384)
