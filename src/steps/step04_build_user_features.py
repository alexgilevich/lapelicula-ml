import os
from typing import Tuple

from pyspark.sql import SparkSession, DataFrame
from spark_utils import get_spark
from pyspark.sql import functions as F
from pyspark.sql import types as T

from features import COMBINED_GENRE_FEATURES
from config import Config, ArgumentsConfig
from job_step import JobStep


class BuildUserFeaturesJobStep(JobStep):
    """
    Step 5: Enrich users with features derived from ratings by genre.
    Input: ratings_with_synth_users (produced by step04)
    Output: users_with_features

    Logic mirrors data_pipeline._get_users_with_features:
    - Optionally extend genres with combined genre features
    - Extract year from UNIX timestamp
    - Compute per-user, per-genre average rating (optionally weighted by 'weight' column if present and ENABLE_WEIGHTED_AVERAGE)
    - Pivot to wide user feature vector
    - Optionally add high_rating_count, low_rating_count, few_genres_lover

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - ENABLE_WEIGHTED_AVERAGE: 'false' | 'true'
    - ENABLE_COMBINED_GENRES: 'false' | 'true'
    - ENABLE_EXTRA_USER_FEATURES: 'false' | 'true'
    """

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None):
        super().__init__(spark or get_spark("step05_build_user_features"), config or ArgumentsConfig())
        self._ratings_df: DataFrame | None = None
        self._users_with_features_df: DataFrame | None = None
        self._movies_df: DataFrame | None = None

    def load(self) -> None:
        self._ratings_df = self.spark.table("ratings_synthetic")
        self._movies_df = self.spark.table("movies_shortlist")

    def process(self) -> None:
        assert self._ratings_df is not None
        assert self._movies_df is not None
        enable_weighted = self.config.bool("ENABLE_WEIGHTED_AVERAGE", False)
        enable_combined = self.config.bool("ENABLE_COMBINED_GENRES", False)
        enable_extra_user = self.config.bool("ENABLE_EXTRA_USER_FEATURES", False)

        sdf = (
            self._ratings_df.alias("r").join(self._movies_df.alias("m"),
                                             on=F.col("r.movieId") == F.col("m.movieId"), how="inner")
            .drop(self._movies_df["movieId"])  # avoid duplicate movieId columns
        )

        # Optionally extend genres with combined features
        if enable_combined and len(COMBINED_GENRE_FEATURES) > 0:
            # UDF to add combined genres
            def add_combined(genres: list[str]) -> list[str]:
                if genres is None:
                    return []
                s = set(genres)
                for comb in COMBINED_GENRE_FEATURES:
                    if comb.issubset(s):
                        s.add('_'.join(sorted(comb)))
                return sorted(list(s))
            add_combined_udf = F.udf(add_combined, T.ArrayType(T.StringType()))
            sdf = sdf.withColumn("genres", add_combined_udf(F.col("genres")))

        # Extract year from timestamp
        sdf = sdf.withColumn("year", F.year(F.from_unixtime(F.col("timestamp"))))

        # Explode genres and compute per-user per-genre rating
        exploded = sdf.select("userId", "genres", "rating", *( ["weight"] if "weight" in sdf.columns else [] )) \
                     .withColumn("genre", F.explode("genres"))

        if enable_weighted and "weight" in exploded.columns:
            avg_rating_by_user_genre_df = exploded.groupBy("userId", "genre").agg((F.sum(F.col("rating") * F.col("weight")) / F.sum(F.col("weight"))).alias("rating"))
        else:
            avg_rating_by_user_genre_df = exploded.groupBy("userId", "genre").agg(F.avg("rating").alias("rating"))

        # Pivot to wide
        users_features = avg_rating_by_user_genre_df.groupBy("userId").pivot("genre").agg(F.first("rating")).fillna(0.0)

        if enable_extra_user:
            high_counts = avg_rating_by_user_genre_df.where(F.col("rating") >= 4.5).groupBy("userId").count().withColumnRenamed("count", "high_rating_count")
            low_counts = avg_rating_by_user_genre_df.where(F.col("rating") <= 2).groupBy("userId").count().withColumnRenamed("count", "low_rating_count")
            users_features = users_features.join(high_counts, on="userId", how="left").join(low_counts, on="userId", how="left")
            users_features = users_features.fillna({"high_rating_count": 0, "low_rating_count": 0})

        # Ensure userId as first column (optional)
        cols = ["userId"] + [c for c in users_features.columns if c != "userId"]
        self._users_with_features_df = users_features.select(*cols)

    def save(self) -> None:
        assert self._users_with_features_df is not None
        overwrite = self.config.bool("OVERWRITE", True)
        mode = "overwrite" if overwrite else "errorifexists"
        self._users_with_features_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("users_with_features")


if __name__ == "__main__":
    config = ArgumentsConfig()
    spark = get_spark("step05_build_user_features", config)
    step = BuildUserFeaturesJobStep(spark, config)
    step.load()
    step.process()
    step.save()
