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

class SplitTrainingDataJobStep(JobStep):
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
        self._training_base_df: DataFrame | None = None
        self._training_movies_df: DataFrame | None = None
        self._training_users_df: DataFrame | None = None
        self._training_labels_df: DataFrame | None = None

    def load(self) -> None:
        self._training_base_df = self.spark.table("training_base").orderBy("row_id")

    def process(self) -> None:
        logger.info('Number of base rows = %d', self._training_base_df.count())
        
        # Build training tables
        # Users columns are exactly users_with_features_df columns
        user_cols = ["row_id", "user_id"] + [F.col(c).alias(c[2:]) for c in self._training_base_df.columns if c.startswith("u_")]
        training_users = self._training_base_df.select(user_cols).orderBy("row_id")

        logger.info('Schema of the training_users table = %s', training_users.schema)
        
        # Movie features are movies_slim_df columns, but drop 'genres' and add onehot
        movie_cols = ["row_id", "movie_id", "title"] + [F.col(c).alias(c[2:]) for c in self._training_base_df.columns if c.startswith("m_")]
        training_movies = self._training_base_df.select(movie_cols)

        logger.info('Schema of the training_movies table = %s', training_movies.schema)
        
        # Labels
        training_labels = self._training_base_df.select("row_id", F.col("rating").alias("rating")).orderBy("row_id")
        logger.info('Schema of the training_labels table = %s', training_labels.schema)

        self._training_movies_df = training_movies
        self._training_users_df = training_users
        self._training_labels_df = training_labels

    def save(self) -> None:
        assert self._training_movies_df is not None and self._training_users_df is not None and self._training_labels_df is not None
        self.dataframe_writer.write(self._training_movies_df, "training_movies")
        self.dataframe_writer.write(self._training_users_df, "training_users")
        self.dataframe_writer.write(self._training_labels_df, "training_labels")

@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = SplitTrainingDataJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()