import _includes
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from pyspark.sql import types as T
from config import Config
from job_step import JobStep
from logging_factory import get_logger

logger = get_logger(__name__)

class GenerateTrainingDataJobStep(JobStep):
    """
    Step 1: Generate training data
    - Join oversampled ratings with users_with_features and movies_preprocessed
    - Build per-row movie one-hot genre features (optionally include combined genres)
    - Produce three aligned tables keyed by row_id:
      training_movies(row_id, <movie features + onehot>),
      training_users(row_id, <user features>),
      training_labels(row_id, rating)

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - ENABLE_COMBINED_GENRES: 'false' | 'true'
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._ratings_oversampled_df: DataFrame | None = None
        self._users_with_features_df: DataFrame | None = None
        self._movies_preprocessed_df: DataFrame | None = None
        self._training_base_df: DataFrame | None = None

    def load(self) -> None:
        self._ratings_oversampled_df = self.spark.table("ratings_oversampled")
        self._users_with_features_df = self.spark.table("users_with_features")
        self._movies_preprocessed_df = self.spark.table("movies_preprocessed")

    def process(self) -> None:
        assert self._ratings_oversampled_df is not None and self._users_with_features_df is not None and self._movies_preprocessed_df is not None
        # Prepare movie slim columns (drop training-unwanted columns)
        drop_cols = set([c for c in self._movies_preprocessed_df.columns if c.startswith("genre_partition")] + ["rating_count", "rating_avg", "weight", "year"])
        movie_cols_keep = [c for c in self._movies_preprocessed_df.columns if c not in drop_cols]
        movies_slim_df = self._movies_preprocessed_df.select(*movie_cols_keep)
        
        # Build base by joining ratings -> users -> movies
        # We need to keep the same number of rows in all training data dataframes as they are later used in dot product in Neural Network.
        # Therefore, we join and then split again.
        base = self._ratings_oversampled_df.alias("r") \
            .join(self._users_with_features_df.alias("u"), on="userId", how="inner") \
            .join(movies_slim_df.alias("m"), on="movieId", how="inner") \
            .orderBy(F.col('userId'), F.col('movieId'))

        # Create stable row_id for alignment
        base = base.withColumn("row_id", F.monotonically_increasing_id()).select(
            ["row_id", "rating"] +
            [F.col(f"u.userId").alias("user_id"), F.col(f"m.movieId").alias("movie_id"), F.col(f"m.title").alias("title")] +
            [F.col(f"u.{c}").alias(f"u_{c}") for c in self._users_with_features_df.columns if c != "userId"] +
            [F.col(f"m.{c}").alias(f"m_{c}") for c in movie_cols_keep if c != "genres" and c != "title" and c != "movieId"]
        )
        
        self._training_base_df = base

    def save(self) -> None:
        assert self._training_base_df is not None
        self.dataframe_writer.write(self._training_base_df, "training_base")

@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = GenerateTrainingDataJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()