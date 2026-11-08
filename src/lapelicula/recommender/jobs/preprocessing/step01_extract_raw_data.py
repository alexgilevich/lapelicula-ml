import os

from dependency_injector.wiring import Provide, inject
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from lapelicula.config import Config
from lapelicula.di.containers import ContainerFactory
from lapelicula.recommender.jobs.job_step import JobStep
from lapelicula.spark.dataframe import DataFrameWriter


class ExtractRawDataJobStep(JobStep):
    """
    Step 1: Extract bronze data
    - Read MovieLens CSV files with Spark
    - Persist as Unity Catalog Delta tables: bronze_movies, bronze_ratings, bronze_links

    Config:
    -UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - RAW_DATA_PATH: Base path to CSV folder (default: ./data)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._bronze_movies_df: DataFrame | None = None
        self._bronze_ratings_df: DataFrame | None = None
        self._bronze_links_df: DataFrame | None = None

    # ---------------------- step contract ----------------------
    def load(self) -> None:
        """
        Initialize bronze movies, bronze ratings, bronze links as DataFrames
        """
        data_path = self.config.string("RAW_DATA_PATH", "./data")

        # Schemas
        movies_schema = T.StructType(
            [
                T.StructField("movieId", T.IntegerType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("genres", T.StringType(), True),
            ]
        )
        ratings_schema = T.StructType(
            [
                T.StructField("userId", T.IntegerType(), False),
                T.StructField("movieId", T.IntegerType(), False),
                T.StructField("rating", T.DoubleType(), False),
                T.StructField("timestamp", T.LongType(), False),
            ]
        )
        links_schema = T.StructType(
            [
                T.StructField("movieId", T.IntegerType(), False),
                T.StructField("imdbId", T.IntegerType(), True),
                T.StructField("tmdbId", T.IntegerType(), True),
            ]
        )

        self._bronze_movies_df = (
            self.spark.read.option("header", True)
            .schema(movies_schema)
            .csv(os.path.join(os.path.expanduser(data_path), "movies.csv"))
            .withColumnRenamed("movieId", "movie_id")
        )
        self._bronze_ratings_df = (
            self.spark.read.option("header", True)
            .schema(ratings_schema)
            .csv(os.path.join(os.path.expanduser(data_path), "ratings.csv"))
            .withColumnRenamed("movieId", "movie_id")
            .withColumnRenamed("userId", "user_id")
        )
        self._bronze_links_df = (
            self.spark.read.option("header", True)
            .schema(links_schema)
            .csv(os.path.join(os.path.expanduser(data_path), "links.csv"))
            .withColumnRenamed("movieId", "movie_id")
            .withColumnRenamed("imdbId", "imdb_id")
            .withColumnRenamed("tmdbId", "tmdb_id")
        )

    def process(self) -> None:
        # Minimal cleanup: ensure types are correct (Spark schema enforces types). Also trim strings.
        assert self._bronze_movies_df is not None
        self._bronze_movies_df = self._bronze_movies_df.withColumn("title", F.trim(F.col("title"))).withColumn(
            "genres", F.trim(F.col("genres"))
        )

    def save(self) -> None:
        assert (
            self._bronze_movies_df is not None and self._bronze_ratings_df is not None and self._bronze_links_df is not None
        )
        self.dataframe_writer.write(self._bronze_movies_df, "bronze_movies")
        self.dataframe_writer.write(self._bronze_ratings_df, "bronze_ratings")
        self.dataframe_writer.write(self._bronze_links_df, "bronze_links")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"],
):
    step = ExtractRawDataJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()


def main(parameters: list[str] | None = None) -> None:
    """Entry point for the Databricks Python wheel task and console scripts."""
    import sys

    if parameters:
        sys.argv[1:] = list(parameters)
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()


if __name__ == "__main__":
    main()
