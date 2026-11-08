"""Cognito OAuth2 (client credentials) token provider for the Movie Management API."""

from __future__ import annotations

import threading
import time
from typing import Optional

import requests

from lapelicula.logs.logging_factory import get_logger

logger = get_logger(__name__)

# Refresh this many seconds before the token actually expires.
_EXPIRY_BUFFER_SECONDS = 60


class CognitoTokenProvider:
    """
    Obtains and caches an OAuth2 access token from Amazon Cognito using the
    client_credentials grant.

    token_url is the Cognito hosted UI OAuth2 token endpoint, e.g.
    https://<domain>.auth.<region>.amazoncognito.com/oauth2/token
    """

    def __init__(self, token_url: str, client_id: str, client_secret: str, scope: Optional[str] = None, timeout: int = 30):
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._timeout = timeout
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        with self._lock:
            if self._token is None or time.time() >= self._expires_at:
                token, expires_in = self._fetch_token()
                self._token = token
                self._expires_at = time.time() + max(expires_in - _EXPIRY_BUFFER_SECONDS, 0)
            return self._token

    def _fetch_token(self) -> tuple[str, int]:
        data = {"grant_type": "client_credentials"}
        if self._scope:
            data["scope"] = self._scope

        logger.info("Requesting Cognito access token from %s", self._token_url)
        response = requests.post(
            self._token_url,
            data=data,
            auth=(self._client_id, self._client_secret),
            timeout=self._timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Cognito token request failed with status {response.status_code}: {response.text}")

        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"Cognito token response did not contain an access_token: {payload}")

        expires_in = int(payload.get("expires_in", 3600))
        logger.info("Obtained Cognito access token valid for %s seconds", expires_in)
        return token, expires_in
