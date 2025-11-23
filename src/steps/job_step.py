from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from spark_utils import get_spark
from config import Config, ArgumentsConfig


class JobStep:
    """
    Base class for all job steps.

    New contract:
    - load(): Read input tables and initialize protected instance fields on self.
    - process(): Transform data using the fields previously initialized in load(); write results
                 back into protected instance fields on self.
    - save(): Persist the final results using the protected fields; no parameters.

    Subclasses should define meaningful protected attributes (e.g., _raw_movies_df, _ratings_df, ...)
    and set them in load() / process().
    """

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None):
        # Keep same defaults to ease usage from scripts
        self.spark: SparkSession = spark or get_spark(self.__class__.__name__)
        self.config: Config = config or ArgumentsConfig()

    # The base class provides the methods to be overridden by subclasses.
    # They intentionally do not accept or return dataframes anymore.
    def load(self) -> None:  # pragma: no cover - abstract contract
        raise NotImplementedError

    def process(self) -> None:  # pragma: no cover - abstract contract
        raise NotImplementedError

    def save(self) -> None:  # pragma: no cover - abstract contract
        raise NotImplementedError
