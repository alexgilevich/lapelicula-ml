import _includes
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from pyspark.sql import types as T, Column
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
        self._gold_movies_df: DataFrame | None = None
        self._training_base_df: DataFrame | None = None

    def load(self) -> None:
        self._ratings_oversampled_df = self.spark.table("silver_ratings_oversampled")
        self._users_with_features_df = self.spark.table("silver_users_with_features")
        self._gold_movies_df = self.spark.table("gold_movie")

    def process(self) -> None:
        assert self._ratings_oversampled_df is not None and self._users_with_features_df is not None and self._gold_movies_df is not None
        # Prepare movie slim columns (drop training-unwanted columns)
        movie_cols_keep = ["movie_id", 
                           "title", 
                           "year", 
                           "adult",
                           "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Kids", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", 
                           "description_embedding", 
                           "anger", "contempt", "disgust", "fear", "frustration", "gratitude", "joy", "love", "neutral", "sadness", "surprise"]
        
        movies_slim_df = self._gold_movies_df.select(*movie_cols_keep)
        movie_col_names_keep = [(c.name() if isinstance(c, tuple) else str(c)) for c in movie_cols_keep]

        # Build base by joining ratings -> users -> movies
        # We need to keep the same number of rows in all training data dataframes as they are later used in dot product in Neural Network.
        # Therefore, we join and then split again.
        base = self._ratings_oversampled_df.alias("r") \
            .join(self._users_with_features_df.alias("u"), on="user_id", how="inner") \
            .join(movies_slim_df.alias("m"), on="movie_id", how="inner") \
            .orderBy(F.col('user_id'), F.col('movie_id'))
        
        # Create stable row_id for alignment
        base = base.withColumn("row_id", F.monotonically_increasing_id()).select(
            ["row_id", "rating"] +
            [F.col(f"u.user_id").alias("user_id"), F.col(f"m.movie_id"), F.col(f"m.title").alias("title")] +
            [F.col(f"u.{c}").alias(f"u_{c}") for c in self._users_with_features_df.columns if c != "user_id"] +
            [F.col(f"m.{c}").alias(f"m_{c}") for c in movie_col_names_keep if c != "genres" and c != "title" and c != "movie_id"] 
        )
        
        self._training_base_df = base

    def save(self) -> None:
        assert self._training_base_df is not None
        self.dataframe_writer.write(self._training_base_df, "silver_training_base")

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