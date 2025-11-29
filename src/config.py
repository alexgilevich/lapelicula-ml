from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
from logging_factory import get_logger


logger = get_logger(__name__)

class Config(ABC):
    """
    Abstract configuration provider used by pipeline steps.

    Implementations should provide methods to fetch string, bool, and int values.
    """

    @abstractmethod
    def string(self, name: str, default: str = "") -> str:
        """Return a string configuration value."""
        raise NotImplementedError

    def bool(self, name: str, default: bool = False) -> bool:
        """Return a boolean configuration value, with common true-like values support."""
        val = (self.string(name, "true" if default else "false") or "").strip().lower()
        return val in ("1", "true", "yes", "y")

    def int(self, name: str, default: int = 0) -> int:
        """Return an integer configuration value."""
        raw = self.string(name, str(default))
        try:
            return int(raw)
        except Exception:
            return default


class ArgumentsConfig(Config):
    """
    CLI arguments config implementation that reads values from job parameters using argparse
    """

    def __init__(self) -> None:
        import sys

        self._config = {}
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                try:
                    key, value = arg[2:].split('=', 1)
                    self._config[key] = value
                except ValueError:
                    logger.warning(f"Skipping malformed argument: {arg}")

        logger.info("The job was initialized with the following configuration: %s", self._config)


    def string(self, name: str, default: str = "") -> str:
        return self._config.get(name, default)


class SecretsManager(ABC):
    """
    Abstract secrets manager used by pipeline steps.
    """

    @abstractmethod
    def get(self, key: str) -> str:
        """Return a string configuration value."""
        raise NotImplementedError

class DBUtilsSecretsManager(SecretsManager):

    def __init__(self, scope: str, dbutils: Any):
        self._scope = scope
        self._dbutils = dbutils

    def get(self, key: str) -> str:
        return self._dbutils.secrets.get(scope = self._scope, key = key)
