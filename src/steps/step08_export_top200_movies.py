import os
from typing import Tuple

from pyspark.sql import SparkSession, DataFrame
from spark_utils import get_spark
from pyspark.sql import functions as F

from config import Config, ArgumentsConfig
from job_step import JobStep


class ExportTop200MoviesJobStep(JobStep):
    """
    Step 10: Export top 200 movies
    - Sort movies by rating_count desc (fallback to compute from ratings if missing)
    - Persist top-200 to UC table top200_movies and optionally to CSV path

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - EXPORT_PATH: optional external path to write CSV (e.g., dbfs:/FileStore/top200_movies.csv)
    """

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None):
        super().__init__(spark or get_spark("step10_export_top200_movies"), config or ArgumentsConfig())
        self._movies_df: DataFrame | None = None
        self._ratings_df: DataFrame | None = None
        self._top200_df: DataFrame | None = None

    def _table_exists(self, full_name: str) -> bool:
        try:
            return self.spark.catalog.tableExists(full_name)
        except Exception:
            return False

    def load(self) -> None:
        # Prefer movies_enriched; if not exists, use movies_preprocessed
        movies_tbl = "movies_enriched" if self._table_exists("movies_enriched") else "movies_preprocessed"
        self._movies_df = self.spark.table(movies_tbl)
        self._ratings_df = self.spark.table("raw_ratings")

    def process(self) -> None:
        assert self._movies_df is not None
        assert self._ratings_df is not None

        # Ensure rating_count column exists
        movies_df = self._movies_df
        if "rating_count" not in movies_df.columns and self._ratings_df is not None:
            counts = self._ratings_df.groupBy("movieId").agg(F.count(F.lit(1)).alias("rating_count"))
            movies_df = movies_df.join(counts, on="movieId", how="inner")
        self._top200_df = movies_df.orderBy(F.col("rating_count").desc_nulls_last()).limit(200)


    def save(self) -> None:
        overwrite = self.config.bool("OVERWRITE", True)
        mode = "overwrite" if overwrite else "errorifexists"
        self._top200_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("movies_top200")



if __name__ == "__main__":
    config = ArgumentsConfig()
    spark = get_spark("step10_export_top200_movies", config)
    step = ExportTop200MoviesJobStep(spark, config)
    step.load()
    step.process()
    step.save()
