import os
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from spark_utils import get_spark
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import DBUtilsSecretsManager
from tmdb import TMDBClient
from config import Config, ArgumentsConfig
from job_step import JobStep


class EnrichMoviesJobStep(JobStep):
    """
    Step 8: Enrich movies (one-hot genres + external attributes)
    - One-hot encode movie genres
    - Join with links (imdbId, tmdbId)
    - Fetch and attach TMDB attributes (cached in UC table)

    Produces: movies_enriched

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - TMDB_FETCH_THREADS: default 10
    """

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None, client: TMDBClient = None):
        super().__init__(spark or get_spark("step08_enrich_movies"), config or ArgumentsConfig())
        self._movies_preprocessed_df: DataFrame | None = None
        self._links_df: DataFrame | None = None
        self._movies_enriched_df: DataFrame | None = None
        self._movies_tmdb_info_df: DataFrame | None = None
        self._client = client

    def _table_exists(self, full_name: str) -> bool:
        try:
            return self.spark.catalog.tableExists(full_name)
        except Exception:
            return False

    def load(self) -> None:
        self._movies_preprocessed_df = self.spark.table("movies_preprocessed")
        self._links_df = self.spark.table("raw_links")
        tmdb_tbl = "movies_tmdb_info"
        if self._table_exists(tmdb_tbl):
            self._movies_tmdb_info_df = self.spark.table(tmdb_tbl)

    def _one_hot_genres(self, movies_df: DataFrame) -> DataFrame:
        # explode genres and pivot back into one-hot
        exploded = movies_df.select("movieId", F.explode(F.col("genres")).alias("genre"))
        onehot_counts = exploded.groupBy("movieId", "genre").count()
        onehot = onehot_counts.groupBy("movieId").pivot("genre").agg(F.when(F.max("count") > 0, F.lit(1)).otherwise(F.lit(0)).cast(T.IntegerType())).fillna(0)
        return movies_df.join(onehot, on="movieId", how="left").fillna(0)

    def _fetch_missing_tmdb(self, missing_tmdb_ids: List[int]) -> DataFrame:
        if not missing_tmdb_ids:
            return self.spark.createDataFrame([], schema=T.StructType([
                T.StructField("id", T.IntegerType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("poster_uri", T.StringType(), True),
                T.StructField("budget", T.DoubleType(), True),
                T.StructField("description", T.StringType(), True),
                T.StructField("release_date", T.DateType(), True),
                T.StructField("origin_countries", T.ArrayType(T.StringType()), True),
            ]))


        def get_one(tmdb_id: int) -> Dict[str, Any]:
            if tmdb_id is None:
                return {}
            try:
                res = self._client.get_movie_by_id(int(tmdb_id)).to_dict()
                # Normalize types
                if isinstance(res.get("release_date"), datetime.date):
                    res["release_date"] = res["release_date"].isoformat()
                return res
            except Exception:
                return {}

        with ThreadPoolExecutor() as pool:
            results = list(pool.map(get_one, missing_tmdb_ids))

        # filter empties
        results = [r for r in results if r and r.get("id") is not None]
        schema = T.StructType([
            T.StructField("id", T.IntegerType(), False),
            T.StructField("title", T.StringType(), True),
            T.StructField("poster_uri", T.StringType(), True),
            T.StructField("budget", T.DoubleType(), True),
            T.StructField("description", T.StringType(), True),
            T.StructField("release_date", T.StringType(), True),
            T.StructField("origin_countries", T.ArrayType(T.StringType()), True),
        ])
        df = self.spark.createDataFrame(results, schema=schema)
        # Cast release_date to date
        df = df.withColumn("release_date", F.to_date("release_date"))
        return df

    def process(self) -> None:
        assert self._movies_preprocessed_df is not None
        assert self._links_df is not None

        movies_onehot_df = self._one_hot_genres(self._movies_preprocessed_df)
        movies_with_links = movies_onehot_df.join(self._links_df, on="movieId", how="left")

        # Prepare/cached TMDB attributes table
        tmdb_attr_df = self._movies_tmdb_info_df if self._movies_tmdb_info_df else self.spark.createDataFrame([], schema=T.StructType([
            T.StructField("id", T.IntegerType(), False),
            T.StructField("title", T.StringType(), True),
            T.StructField("poster_uri", T.StringType(), True),
            T.StructField("budget", T.DoubleType(), True),
            T.StructField("description", T.StringType(), True),
            T.StructField("release_date", T.DateType(), True),
            T.StructField("origin_countries", T.ArrayType(T.StringType()), True),
        ]))

        needed_ids_df = movies_with_links.select("tmdbId").where(F.col("tmdbId").isNotNull()).distinct()
        have_ids_df = tmdb_attr_df.select(F.col("id").alias("tmdbId")).distinct()
        missing_ids = needed_ids_df.join(have_ids_df, on="tmdbId", how="left_anti").select("tmdbId")
        missing_ids_list = [int(r.tmdbId) for r in missing_ids.collect()]

        missing_tmdb_info_df = self._fetch_missing_tmdb(missing_ids_list)
        self._movies_tmdb_info_df = self._movies_tmdb_info_df.unionByName(missing_tmdb_info_df)

        # Join attributes (drop original title to mimic original behavior)
        movies_no_title = movies_with_links.drop("title") if "title" in movies_with_links.columns else movies_with_links
        enriched = movies_no_title.join(self._movies_tmdb_info_df, movies_with_links["tmdbId"] == self._movies_tmdb_info_df["id"], how="inner")
        self._movies_enriched_df = enriched

    def save(self) -> None:
        assert self._movies_enriched_df is not None
        overwrite = self.config.bool("OVERWRITE", True)
        mode = "overwrite" if overwrite else "errorifexists"
        self._movies_tmdb_info_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("movies_tmdb_info")
        self._movies_enriched_df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable("movies_enriched")


if __name__ == "__main__":
    dbutils: Any
    config = ArgumentsConfig()
    client = TMDBClient(DBUtilsSecretsManager("lapelicula", dbutils))
    spark = get_spark("step08_enrich_movies", config)
    step = EnrichMoviesJobStep(spark, config, client)
    step.load()
    step.process()
    step.save()
