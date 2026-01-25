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

class GenerateSyntheticUsersJobStep(JobStep):
    """
    Step 4: Generate synthetic users
    - For each genre partition column (genre_partition0, genre_partition1, ...),
      keep users that have ratings in more than one partition and remap their userId to
      100_000_000 * (i + 1) + userId * 1000 + partition_value
    - Concatenate such frames across partitions

    Input: ratings_with_movies
    Output: ratings_with_synth_users

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - GENRES_N_CLUSTERS: comma-separated integers, default '45,100' (drives number of partition columns)
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._ratings_df: DataFrame | None = None
        self._movies_df: DataFrame | None = None
        self._ratings_df: DataFrame | None = None

    def _parse_clusters(self):
        raw = self.config.string("GENRES_N_CLUSTERS", "45,100")
        try:
            return [int(x.strip()) for x in raw.split(',') if x.strip()]
        except Exception:
            return [45, 100]

    def load(self) -> None:
        self._ratings_df = self.spark.table("raw_ratings")
        self._movies_df = self.spark.table("movies_shortlist")

    def process(self) -> None:
        assert self._ratings_df is not None
        assert self._movies_df is not None
        ratings_with_movies_df = (
            self._ratings_df.alias("r").join(self._movies_df.alias("m"),
                                             on=F.col("r.movieId") == F.col("m.movieId"), how="inner")
            .drop(self._movies_df["movieId"])  # avoid duplicate movieId columns
        )
        
        genres_n_clusters = self._parse_clusters()
        final_df: DataFrame | None = None
        
        logger.info('Generating synthetic users... Current ratings count = %d', ratings_with_movies_df.count())
        for i in range(len(genres_n_clusters)):
            partition_col = f"genre_partition{i}"
            if partition_col not in ratings_with_movies_df.columns:
                # if missing, skip
                logger.warning("Expected column %s but the column was not found, skipping")
                continue
            # there still might be cases when partitions were not assigned because of too few samples in the centroids
            sdf = ratings_with_movies_df.where(F.col(partition_col) > -1)
            # users with >1 distinct partition
            user_part_counts = sdf.groupBy("userId").agg(F.countDistinct(F.col(partition_col)).alias("part_cnt"))
            # we need to break apart only users with multiple partitions because users with 1 partition already have "focused" interest around one specific type of movies
            users_multi = user_part_counts.where(F.col("part_cnt") > 1).select("userId")
            sdf = sdf.join(users_multi, on="userId", how="inner")
            # new synthetic user id generation
            sdf = sdf.withColumn(
                "userId",
                (F.lit(100_000_000) * (F.lit(i + 1)) + F.col("userId") * F.lit(1000) + F.col(partition_col)).cast(T.LongType())
            )
            if final_df is None:
                final_df = sdf
            else:
                final_df = final_df.unionByName(sdf)
        self._ratings_df = (final_df if final_df is not None else ratings_with_movies_df.limit(0)).select(*self._ratings_df.columns).cache()
        logger.info('Generated synthetic users... Current ratings count = %d', self._ratings_df.count())
        
    def save(self) -> None:
        assert self._ratings_df is not None
        self.dataframe_writer.write(self._ratings_df, "ratings_synthetic")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = GenerateSyntheticUsersJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()