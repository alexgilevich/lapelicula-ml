"""Client for the Internal Movie Management API (PUT /api/internal/movies)."""

from __future__ import annotations

import time
from typing import Optional, Protocol, Sequence

import requests

from lapelicula.config import Config, SecretsManager
from lapelicula.logs.logging_factory import get_logger

from .models import Movie

logger = get_logger(__name__)

UPSERT_MOVIES_PATH = "/api/internal/movies"
_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)
_MAX_RETRIES = 3


class TokenProvider(Protocol):
    def get_token(self) -> str: ...


class MovieApiClient:
    """
    Thin client over the generated API contract.

    base_url: scheme+host(+port) of the API, e.g. http://127.0.0.1:5003
    token_provider: optional bearer token provider (Cognito). When None,
        requests are sent unauthenticated (local development).
    """

    def __init__(self, base_url: str, token_provider: Optional[TokenProvider] = None, timeout: int = 300):
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._timeout = timeout

    def upsert_movies(self, movies: Sequence[Movie]) -> None:
        """Bulk idempotent upsert. The API validates the entire array and rejects the request if any movie is invalid."""
        payload = [movie.model_dump(mode="json", exclude_none=True) for movie in movies]
        self._request("PUT", UPSERT_MOVIES_PATH, json=payload)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {self._token_provider.get_token()}"

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = requests.request(method, url, headers=headers, timeout=self._timeout, **kwargs)
                if response.status_code == 200:
                    return response
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = 2**attempt
                    logger.warning(
                        "API %s %s returned %s, retrying in %ss (attempt %s/%s)",
                        method,
                        path,
                        response.status_code,
                        delay,
                        attempt,
                        _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"API {method} {url} failed with status {response.status_code}: {response.text[:2000]}")
            except requests.ConnectionError as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = 2**attempt
                    logger.warning(
                        "API %s %s connection error, retrying in %ss (attempt %s/%s): %s",
                        method,
                        path,
                        delay,
                        attempt,
                        _MAX_RETRIES,
                        e,
                    )
                    time.sleep(delay)
                    continue
        raise RuntimeError(f"API {method} {url} failed after {_MAX_RETRIES} attempts") from last_error


def create_movie_api_client(config: Config, secrets_manager: SecretsManager) -> MovieApiClient:
    """
    Build the client from job configuration and secrets.

    Configuration:
    - MOVIE_API_URL: base URL of the Movie Management API (required).
    - MOVIE_API_SCOPE: optional OAuth2 scope for the Cognito token request.

    Secrets (all three must be present to enable authentication; when absent,
    requests are sent unauthenticated - local development):
    - COGNITO_TOKEN_URL: Cognito OAuth2 token endpoint
    - COGNITO_CLIENT_ID
    - COGNITO_CLIENT_SECRET
    """
    from .auth import CognitoTokenProvider

    base_url = config.string("MOVIE_API_URL")
    if not base_url:
        raise ValueError("MOVIE_API_URL is not configured. Set it in the job parameters/environment.")

    def try_get_secret(key: str) -> Optional[str]:
        try:
            return secrets_manager.get(key)
        except Exception:
            return None

    token_url = config.string("COGNITO_TOKEN_URL")
    client_id = try_get_secret("COGNITO_CLIENT_ID")
    client_secret = try_get_secret("COGNITO_CLIENT_SECRET")

    token_provider: Optional[TokenProvider] = None
    if token_url and client_id and client_secret:
        token_provider = CognitoTokenProvider(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            scope=config.string("MOVIE_API_SCOPE"),
        )
        logger.info("Movie API client will authenticate against %s", token_url)
    else:
        logger.info("Cognito credentials not configured; calling Movie API unauthenticated (local mode)")

    return MovieApiClient(base_url=base_url, token_provider=token_provider)
