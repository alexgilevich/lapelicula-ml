"""
Python client for the Internal Movie Management API.

- models.py: pydantic models generated from the OpenAPI spec (http://.../docs/v1.json)
- auth.py: Cognito OAuth2 client-credentials token provider
- client.py: HTTP client + factory wiring configuration/secrets
"""

from .auth import CognitoTokenProvider
from .client import MovieApiClient, TokenProvider, create_movie_api_client
from .models import Movie

__all__ = ["CognitoTokenProvider", "Movie", "MovieApiClient", "TokenProvider", "create_movie_api_client"]
