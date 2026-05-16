import _includes
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from pyspark import Row
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from pyspark.sql import types as T
from config import Config
from job_step import JobStep
from logging_factory import get_logger
import numpy as np

logger = get_logger(__name__)

class OversampleRatingsJobStep(JobStep):
    """
    Step 6: Oversample ratings
    - First, balance genres to reach at least 75th percentile count across genres
    - Optionally, balance rating bins [0,2.5], (2.5,4.5], (4.5,5]

    Input: ratings_with_synth_users
    Output: ratings_oversampled

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - ENABLE_OVERSAMPLING_BY_RATING: 'false' | 'true'
    - RANDOM_SEED: integer (default 42)
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._ratings_df: DataFrame | None = None
        self._ratings_oversampled_df: DataFrame | None = None
        self._movies_df: DataFrame | None = None

    def load(self) -> None:
        self._ratings_df = self.spark.table("ratings_synthetic")
        self._movies_df = self.spark.table("movies_shortlist")

    def _oversample_by_genre(self, ratings_df: DataFrame, seed: int) -> DataFrame:
        # explode genres
        exploded = ratings_df.withColumn("genre_exploded", F.explode(F.col("genres")))
        counts = exploded.groupBy("genre_exploded").agg(F.count(F.lit(1)).alias("cnt")).orderBy("genre_exploded")
        counts_lst: list[Row] = counts.collect()
        # 75th percentile
        cnt_list = [r.cnt for r in counts_lst]
        if not cnt_list:
            return ratings_df
        
        desired_min = int(np.percentile(np.array(cnt_list), 75))
        logger.info('Desired min rating per genre (75-th percentile) = %d', desired_min)
        # Build union of additional samples
        resampled = ratings_df
        for row in counts_lst:
            genre = row["genre_exploded"]
            current = row["cnt"]
            if current >= desired_min or current == 0:
                continue
            logger.info('Oversampling low-represented genre %s with %d ratings to %d ratings', genre, current, desired_min)
            target = desired_min - current
            # Filter rows for this genre
            subset = ratings_df.where(F.array_contains(F.col("genres"), F.lit(genre)))
            # Fraction for approximate sampling with replacement
            fraction = target / float(current)
            sampled = subset.sample(withReplacement=True, fraction=fraction, seed=seed)
            resampled = resampled.unionByName(sampled)

        logger.info('Completed oversampling ratings b y genre... Current ratings count = %d', resampled.count())
        
        return resampled

    def _oversample_by_rating_bins(self, ratings_df: DataFrame, seed: int) -> DataFrame:
        # Define bins
        def label_for(r: float) -> int:
            if r <= 2.5:
                return 1
            elif r <= 4.5:
                return 2
            else:
                return 3
        label_udf = F.udf(label_for, T.IntegerType())
        binned = ratings_df.withColumn("rating_bin", label_udf(F.col("rating")))
        counts = binned.groupBy("rating_bin").agg(F.count(F.lit(1)).alias("cnt")).collect()
        by_bin = {row["rating_bin"]: row["cnt"] for row in counts}
        if not by_bin:
            return ratings_df
        target_per_bin = {
            1: by_bin.get(1, 0),
            2: by_bin.get(2, 0),
            3: max(by_bin.get(3, 0), int(by_bin.get(2, 0) * 0.8), by_bin.get(1, 0))
        }
        resampled = binned
        for label in [1, 2, 3]:
            current = by_bin.get(label, 0)
            target = target_per_bin[label]
            if current >= target or current == 0:
                continue
            logger.info('Oversampling bin %d with %d ratings to %d ratings', label, by_bin[label], target)
            fraction = (target - current) / float(current)
            subset = binned.where(F.col("rating_bin") == label)
            sampled = subset.sample(withReplacement=True, fraction=fraction, seed=seed)
            resampled = resampled.unionByName(sampled)
        return resampled.drop("rating_bin")

    def process(self) -> None:
        assert self._ratings_df is not None
        assert self._movies_df is not None

        seed = self.config.int("RANDOM_SEED", 42)
        enable_by_rating = self.config.bool("ENABLE_OVERSAMPLING_BY_RATING", False)

        ratings_with_movies_df = (
            self._ratings_df.alias("r").join(self._movies_df.alias("m"),
                                             on=F.col("r.movieId") == F.col("m.movieId"), how="inner")
            .drop(self._movies_df["movieId"])  # avoid duplicate movieId columns
        )

        resampled_ratings_df = self._oversample_by_genre(ratings_with_movies_df, seed)
        if enable_by_rating:
            resampled_ratings_df = self._oversample_by_rating_bins(resampled_ratings_df, seed)
        self._ratings_oversampled_df = resampled_ratings_df.select(*self._ratings_df.columns)

    def save(self) -> None:
        assert self._ratings_oversampled_df is not None
        assert self._movies_df is not None
        self.dataframe_writer.write(self._ratings_oversampled_df, "ratings_oversampled")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = OversampleRatingsJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()
