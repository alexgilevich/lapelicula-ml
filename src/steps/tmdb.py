import datetime
import requests
import os
import json
from dataclasses import dataclass, asdict
from config import SecretsManager


@dataclass
class TMDBMovie:
    id: int
    title: str
    poster_uri: str #poster_path
    budget: float
    description: str #overview
    release_date: datetime.date
    origin_countries: list[str]

    def to_dict(self) -> dict:
        return asdict(self)



class TMDBClient:
    def __init__(self, secrets_manager: SecretsManager):
        self.base_url = "https://api.themoviedb.org"
        self.api_key = secrets_manager.get("TMDB_API_KEY")

    def get_movie_by_id(self, id: int) -> TMDBMovie:
        url = f"{self.base_url}/3/movie/{id}?language=en-US"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        movie_info = json.loads(response.text)
        return TMDBMovie(
            id = movie_info["id"],
            title = movie_info["title"],
            poster_uri = 'https://image.tmdb.org/t/p/w500' + movie_info["poster_path"],
            budget = movie_info["budget"],
            description = movie_info["overview"],
            release_date = datetime.date.fromisoformat(movie_info["release_date"]),
            origin_countries = movie_info["origin_country"]
        )
