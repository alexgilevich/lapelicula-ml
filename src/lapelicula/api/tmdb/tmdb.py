import datetime
import json
from dataclasses import asdict, dataclass

import requests

from lapelicula.api.tmdb.iso_3166 import country_name
from lapelicula.config import SecretsManager


@dataclass
class TMDBMovie:
    id: int
    imdb_id: int
    title: str
    poster_uri: str  # poster_path
    budget: int
    description: str  # overview
    release_date: datetime.date
    origin_countries: list[dict]  # [{"code": "US", "name": "United States of America"}]
    vote_average: float
    revenue: int
    production_countries: list[dict]  # [{"code": "US", "name": "United States of America"}]
    tagline: str
    adult: bool

    def to_dict(self) -> dict:
        return asdict(self)


class TMDBClient:
    def __init__(self, secrets_manager: SecretsManager):
        self.base_url = "https://api.themoviedb.org"
        self.api_key = secrets_manager.get("TMDB_API_KEY")

    def get_movie_by_id(self, id: int) -> TMDBMovie:
        url = f"{self.base_url}/3/movie/{id}?language=en-US"

        headers = {"accept": "application/json", "Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        movie_info = json.loads(response.text)
        return TMDBMovie(
            id=movie_info["id"],
            title=movie_info["title"],
            poster_uri="https://image.tmdb.org/t/p/w500" + movie_info["poster_path"],
            budget=movie_info["budget"],
            description=movie_info["overview"],
            release_date=datetime.date.fromisoformat(movie_info["release_date"]),
            # TMDB provides only codes for origin countries; resolve names via ISO 3166
            origin_countries=[{"code": code, "name": country_name(code)} for code in movie_info["origin_country"]],
            vote_average=movie_info["vote_average"],
            revenue=movie_info["revenue"],
            production_countries=[
                {"code": country["iso_3166_1"], "name": country["name"]} for country in movie_info["production_countries"]
            ],
            tagline=movie_info["tagline"],
            adult=movie_info["adult"],
            imdb_id=movie_info["imdb_id"],
        )
