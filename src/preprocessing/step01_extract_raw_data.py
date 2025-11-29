import os
from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from spark_utils import get_spark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from config import Config, ArgumentsConfig
from job_step import JobStep


class ExtractRawDataJobStep(JobStep):
    """
    Step 1: Extract raw data
    - Read MovieLens CSV files with Spark
    - Persist as Unity Catalog Delta tables: raw_movies, raw_ratings, raw_links

    Config:
    - UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - RAW_DATA_PATH: Base path to CSV folder (default: ./data)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None):
        super().__init__(spark or get_spark("step01_extract_raw_data"), config or ArgumentsConfig())
        self._raw_movies_df: DataFrame | None = None
        self._raw_ratings_df: DataFrame | None = None
        self._raw_links_df: DataFrame | None = None

    # ---------------------- step contract ----------------------
    def load(self) -> None:
        """
        Initialize raw movies, raw ratings, raw links as DataFrames
        """
        data_path = self.config.string("RAW_DATA_PATH", "./data")

        # Schemas
        movies_schema = T.StructType([
            T.StructField("movieId", T.IntegerType(), False),
            T.StructField("title", T.StringType(), True),
            T.StructField("genres", T.StringType(), True),
        ])
        ratings_schema = T.StructType([
            T.StructField("userId", T.IntegerType(), False),
            T.StructField("movieId", T.IntegerType(), False),
            T.StructField("rating", T.DoubleType(), False),
            T.StructField("timestamp", T.LongType(), False),
        ])
        links_schema = T.StructType([
            T.StructField("movieId", T.IntegerType(), False),
            T.StructField("imdbId", T.IntegerType(), True),
            T.StructField("tmdbId", T.IntegerType(), True),
        ])

        self._raw_movies_df = (
            self.spark.read.option("header", True).schema(movies_schema).csv(os.path.join(data_path, "movies.csv"))
        )
        self._raw_ratings_df = (
            self.spark.read.option("header", True).schema(ratings_schema).csv(os.path.join(data_path, "ratings.csv"))
        )
        self._raw_links_df = (
            self.spark.read.option("header", True).schema(links_schema).csv(os.path.join(data_path, "links.csv"))
        )

    def process(self) -> None:
        # Minimal cleanup: ensure types are correct (Spark schema enforces types). Also trim strings.
        assert self._raw_movies_df is not None
        self._raw_movies_df = self._raw_movies_df.withColumn("title", F.trim(F.col("title"))).withColumn("genres", F.trim(F.col("genres")))

    def save(self) -> None:
        assert self._raw_movies_df is not None and self._raw_ratings_df is not None and self._raw_links_df is not None
        overwrite = self.config.bool("OVERWRITE", True)
        mode = "overwrite" if overwrite else "errorifexists"
        self._raw_movies_df.write.format("delta").mode(mode).saveAsTable("raw_movies")
        self._raw_ratings_df.write.format("delta").mode(mode).saveAsTable("raw_ratings")
        self._raw_links_df.write.format("delta").mode(mode).saveAsTable("raw_links")


if __name__ == "__main__":
    from logging_factory import get_logger
    logger = get_logger(__name__)
    config = ArgumentsConfig()
    spark = get_spark("step01_extract_raw_data", config)
    step = ExtractRawDataJobStep(spark, config)
    step.load()
    step.process()
    step.save()