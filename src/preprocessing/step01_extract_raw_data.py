import _includes
import os
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from pyspark.sql import types as T
from config import Config
from job_step import JobStep


class ExtractRawDataJobStep(JobStep):
    """
    Step 1: Extract raw data
    - Read MovieLens CSV files with Spark
    - Persist as Unity Catalog Delta tables: raw_movies, raw_ratings, raw_links

    Config:
    -UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - RAW_DATA_PATH: Base path to CSV folder (default: ./data)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
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
            self.spark.read.option("header", True).schema(movies_schema).csv(os.path.join(os.path.expanduser(data_path), "movies.csv"))
        )
        self._raw_ratings_df = (
            self.spark.read.option("header", True).schema(ratings_schema).csv(os.path.join(os.path.expanduser(data_path), "ratings.csv"))
        )
        self._raw_links_df = (
            self.spark.read.option("header", True).schema(links_schema).csv(os.path.join(os.path.expanduser(data_path), "links.csv"))
        )

    def process(self) -> None:
        # Minimal cleanup: ensure types are correct (Spark schema enforces types). Also trim strings.
        assert self._raw_movies_df is not None
        self._raw_movies_df = self._raw_movies_df.withColumn("title", F.trim(F.col("title"))).withColumn("genres", F.trim(F.col("genres")))

    def save(self) -> None:
        assert self._raw_movies_df is not None and self._raw_ratings_df is not None and self._raw_links_df is not None
        self.dataframe_writer.write(self._raw_movies_df, "raw_movies")
        self.dataframe_writer.write(self._raw_ratings_df, "raw_ratings")
        self.dataframe_writer.write(self._raw_links_df, "raw_links")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = ExtractRawDataJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()