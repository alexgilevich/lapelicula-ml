import os
from typing import Tuple, List
import _includes
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from spark_utils import get_spark
from config import Config, ArgumentsConfig
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

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None):
        super().__init__(spark or get_spark("step07_generate_training_data"), config or ArgumentsConfig())
        self._ratings_oversampled_df: DataFrame | None = None
        self._users_with_features_df: DataFrame | None = None
        self._movies_preprocessed_df: DataFrame | None = None
        self._training_movies_df: DataFrame | None = None
        self._training_users_df: DataFrame | None = None
        self._training_labels_df: DataFrame | None = None

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
            .join(movies_slim_df.alias("m"), on="movieId", how="inner")

        # Create stable row_id for alignment
        base = base.withColumn("row_id", F.monotonically_increasing_id())

        # Movie one-hot from genres (array<string>)
        exploded = base.select("row_id", F.explode(F.col("genres")).alias("genre"))
        onehot_counts = exploded.groupBy("row_id", "genre").count()
        onehot = onehot_counts.groupBy("row_id").pivot("genre").agg(F.max("count")).fillna(0)

        # Build training tables
        # Users columns are exactly users_with_features_df columns
        user_cols = self._users_with_features_df.columns
        training_users = base.select(["row_id"] + user_cols)

        # Movie features are movies_slim_df columns, but drop 'genres' and add onehot
        movie_cols = [c for c in movie_cols_keep if c != "genres"]
        training_movies_partial = base.select(["row_id"] + movie_cols)
        training_movies = training_movies_partial.join(onehot, on="row_id", how="left").fillna(0)

        # Labels
        training_labels = base.select("row_id", F.col("rating").alias("rating"))

        self._training_movies_df = training_movies
        self._training_users_df = training_users
        self._training_labels_df = training_labels

    def save(self) -> None:
        assert self._training_movies_df is not None and self._training_users_df is not None and self._training_labels_df is not None
        overwrite = self.config.bool("OVERWRITE", True)
        mode = "overwrite" if overwrite else "errorifexists"
        self._training_movies_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("training_movies")
        self._training_users_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("training_users")
        self._training_labels_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("training_labels")


if __name__ == "__main__":
    config = ArgumentsConfig()
    spark = get_spark("step01_generate_training_data", config)
    step = GenerateTrainingDataJobStep(spark, config)
    step.load()
    step.process()
    step.save()
