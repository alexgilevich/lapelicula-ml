import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from dependency_injector.wiring import Provide, inject
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from lapelicula.api.tmdb import TMDBClient
from lapelicula.config import Config
from lapelicula.di.containers import ContainerFactory
from lapelicula.logs.logging_factory import get_logger
from lapelicula.recommender.jobs.job_step import JobStep
from lapelicula.spark.dataframe import DataFrameWriter

logger = get_logger(__name__)


class EnrichMoviesJobStep(JobStep):
    """
    Step 8: Enrich movies (one-hot genres + external attributes)
    - One-hot encode movie genres
    - Join with links (imdbId, tmdb_id)
    - Fetch and attach TMDB attributes (cached in UC table)

    Produces: movies_enriched

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - TMDB_FETCH_THREADS: default 10
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, client: TMDBClient = None):
        super().__init__(spark, config, dataframe_writer)
        self._movies_preprocessed_df: DataFrame | None = None
        self._links_df: DataFrame | None = None
        self._movies_enriched_df: DataFrame | None = None
        self._movies_tmdb_info_df: DataFrame | None = None
        self._client = client
        self.enriched_movie_schema = T.StructType(
            [
                T.StructField("id", T.IntegerType(), True),
                T.StructField("title", T.StringType(), True),
                T.StructField("poster_uri", T.StringType(), True),
                T.StructField("budget", T.LongType(), True),
                T.StructField("description", T.StringType(), True),
                T.StructField("release_date", T.StringType(), True),
                T.StructField("origin_country_codes", T.ArrayType(T.StringType()), True),
                T.StructField("origin_country_names", T.ArrayType(T.StringType()), True),
                T.StructField("production_country_codes", T.ArrayType(T.StringType()), True),
                T.StructField("production_country_names", T.ArrayType(T.StringType()), True),
                T.StructField("vote_average", T.FloatType(), True),
                T.StructField("revenue", T.LongType(), True),
                T.StructField("tagline", T.StringType(), True),
                T.StructField("adult", T.BooleanType(), True),
            ]
        )

    def _table_exists(self, full_name: str) -> bool:
        try:
            return self.spark.catalog.tableExists(full_name)
        except Exception:
            return False

    def _upgrade_legacy_tmdb_info(self, cached: DataFrame) -> DataFrame:
        """
        The TMDB cache predates storing country codes and names side by side:
        origin_countries held ISO codes and production_countries held full names.
        Derive the missing half of each pair from the ISO 3166 maps.
        """
        from lapelicula.api.tmdb.iso_3166 import ISO_3166_1_ALPHA_2_TO_NAME, ISO_3166_1_NAME_TO_CODE

        code_to_name = F.create_map(*[F.lit(v) for pair in ISO_3166_1_ALPHA_2_TO_NAME.items() for v in pair])
        name_to_code = F.create_map(*[F.lit(v) for pair in ISO_3166_1_NAME_TO_CODE.items() for v in pair])

        return (
            cached.withColumn("origin_country_codes", F.col("origin_countries"))
            .withColumn("origin_country_names", F.transform("origin_countries", lambda c: code_to_name[c]))
            .withColumn("production_country_names", F.col("production_countries"))
            .withColumn(
                "production_country_codes",
                F.filter(F.transform("production_countries", lambda c: name_to_code[c]), lambda c: c.isNotNull()),
            )
            .drop("origin_countries", "production_countries")
        )

    def load(self) -> None:
        self._movies_preprocessed_df = self.spark.table("silver_movies_preprocessed")
        self._links_df = self.spark.table("bronze_links")
        tmdb_tbl = "silver_movies_tmdb_info"
        if self._table_exists(tmdb_tbl):
            cached = self.spark.table(tmdb_tbl)
            if "origin_country_codes" not in cached.columns:
                cached = self._upgrade_legacy_tmdb_info(cached)
            self._movies_tmdb_info_df = cached.select(*[field.name for field in self.enriched_movie_schema]).to(
                self.enriched_movie_schema
            )

    ### Helper to fetch missing TMDB info in parallel
    ### This is done with a thread pool and not with Spark for now
    def _fetch_missing_tmdb(self, missing_tmdb_ids: List[int]) -> DataFrame:
        if not missing_tmdb_ids:
            return self.spark.createDataFrame([], schema=self.enriched_movie_schema)

        total = len(missing_tmdb_ids)

        def get_one(tmdb_id: int) -> Dict[str, Any]:
            if tmdb_id is None:
                return {}
            try:
                res = self._client.get_movie_by_id(int(tmdb_id)).to_dict()
                # Normalize types
                if isinstance(res.get("release_date"), datetime.date):
                    res["release_date"] = res["release_date"].isoformat()
                # Flatten [{"code", "name"}, ...] into parallel code/name arrays
                origins = res.pop("origin_countries", None) or []
                productions = res.pop("production_countries", None) or []
                res["origin_country_codes"] = [c["code"] for c in origins]
                res["origin_country_names"] = [c.get("name") for c in origins]
                res["production_country_codes"] = [c["code"] for c in productions]
                res["production_country_names"] = [c.get("name") for c in productions]
                return res
            except Exception as e:
                logger.warning(f"Failed to fetch TMDB info for movie {tmdb_id}: {e}")
                return {}
            finally:
                nonlocal total
                total -= 1
                if total % 100 == 0:
                    logger.info("Left %d movies to load", total)

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(get_one, missing_tmdb_ids))

        # filter empties
        results = [r for r in results if r and r.get("id") is not None]

        df = self.spark.createDataFrame(results, schema=self.enriched_movie_schema)
        # Cast release_date to date
        df = df.withColumn("release_date", F.to_date("release_date"))
        return df

    def process(self) -> None:
        assert self._movies_preprocessed_df is not None
        assert self._links_df is not None

        movies_with_links = self._movies_preprocessed_df.join(self._links_df, on="movie_id", how="left")

        # Prepare/cached TMDB attributes table
        tmdb_attr_df = (
            self._movies_tmdb_info_df
            if self._movies_tmdb_info_df
            else self.spark.createDataFrame([], schema=self.enriched_movie_schema)
        )

        needed_ids_df = movies_with_links.select("tmdb_id").where(F.col("tmdb_id").isNotNull()).distinct()
        have_ids_df = tmdb_attr_df.select(F.col("id").alias("tmdb_id")).distinct()
        missing_ids = needed_ids_df.join(have_ids_df, on="tmdb_id", how="left_anti").select("tmdb_id")
        missing_ids_list = [int(r.tmdb_id) for r in missing_ids.collect()]

        missing_tmdb_info_df = self._fetch_missing_tmdb(missing_ids_list)
        self._movies_tmdb_info_df = tmdb_attr_df.unionByName(missing_tmdb_info_df)

        # Join attributes
        movies_no_title = movies_with_links.drop("title") if "title" in movies_with_links.columns else movies_with_links
        enriched = movies_no_title.join(
            self._movies_tmdb_info_df, movies_with_links["tmdb_id"] == self._movies_tmdb_info_df["id"], how="inner"
        ).withColumn(
            "combined_description",
            F.concat_ws(
                "\n",
                F.concat(F.lit("Title: "), F.col("title")),
                F.concat(F.lit("Tag line: "), F.coalesce(F.col("tagline"), F.lit(""))),
                F.concat(F.lit("Genres: "), F.array_join(F.col("genres"), ", ")),
                F.concat(F.lit("\nDescription: "), F.col("description")),
            ),
        )

        self._movies_enriched_df = enriched

    def save(self) -> None:
        assert self._movies_enriched_df is not None
        assert self._movies_tmdb_info_df is not None
        # self.dataframe_writer.write(self._movies_tmdb_info_df, "silver_movies_tmdb_info")
        self.dataframe_writer.write(self._movies_enriched_df, "silver_movies_enriched")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"],
    tmdb_client: TMDBClient = Provide["tmdb_client"],
):
    step = EnrichMoviesJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer, client=tmdb_client)
    step.load()
    step.process()
    step.save()


def main(parameters: list[str] | None = None) -> None:
    """Entry point for the Databricks Python wheel task and console scripts."""
    import sys

    if parameters:
        sys.argv[1:] = list(parameters)
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()


if __name__ == "__main__":
    main()
