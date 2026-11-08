from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .config import Config


class SecretsManager(ABC):
    """
    Abstract secrets manager used by pipeline steps.
    """

    @abstractmethod
    def get(self, key: str) -> str:
        """Return a string configuration value."""
        raise NotImplementedError


class DBUtilsSecretsManager(SecretsManager):
    def __init__(self, config: Config, dbutils: Any):
        self._config = config
        self._dbutils = dbutils

    def get(self, key: str) -> str:
        return self._dbutils.secrets.get(scope=self._config.string("SCOPE", "lapelicula"), key=key)


class EnvSecretsManager(SecretsManager):
    def get(self, key: str) -> str:
        return os.environ.get(key)
