from dependency_injector.wiring import Provide, inject
from pydantic import ValidationError
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from lapelicula.api.movie_management.client import MovieApiClient
from lapelicula.api.movie_management.models import Movie
from lapelicula.config import Config
from lapelicula.di.containers import ContainerFactory
from lapelicula.logs.logging_factory import get_logger
from lapelicula.recommender.jobs.job_step import JobStep
from lapelicula.spark.dataframe import DataFrameWriter

logger = get_logger(__name__)


# API-side hard limits used to pre-filter rows client side. The API validates
# every movie in the array and rejects the entire request if any one is invalid.
MIN_YEAR, MAX_YEAR = 1800, 2100
EMBEDDING_SIZE = 384

DEFAULT_BATCH_SIZE = 500


GENRE_COLUMNS = {
    "Action": "action",
    "Adventure": "adventure",
    "Animation": "animation",
    "Comedy": "comedy",
    "Crime": "crime",
    "Documentary": "documentary",
    "Drama": "drama",
    "Fantasy": "fantasy",
    "Film-Noir": "filmNoir",
    "Horror": "horror",
    "Kids": "kids",
    "Musical": "musical",
    "Mystery": "mystery",
    "Romance": "romance",
    "Sci-Fi": "sciFi",
    "Thriller": "thriller",
    "War": "war",
    "Western": "western",
}

SENTIMENT_COLUMNS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]


class SyncDataToApiJobStep(JobStep):
    """
    Step 10: Sync all movies (with embeddings and sentiments) from gold_movie
    to the Internal Movie Management API via bulk idempotent upsert.

    Config:
    - MOVIE_API_URL: base URL of the API (required)
    - MOVIE_API_BATCH_SIZE: number of movies per PUT request (default 500)
    - MOVIE_API_SCOPE: optional OAuth2 scope
    Authentication (Cognito client credentials) is enabled automatically when
    COGNITO_TOKEN_URL, COGNITO_CLIENT_ID and COGNITO_CLIENT_SECRET secrets are
    present; otherwise requests are sent unauthenticated (local development).
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, api_client: MovieApiClient):
        super().__init__(spark, config, dataframe_writer)
        self._movies_df: DataFrame | None = None
        self._api_client = api_client

    def load(self) -> None:
        self._movies_df = self.spark.table("gold_movie")

    def process(self) -> None:
        assert self._movies_df is not None

        select = [
            F.col("movie_id").alias("id"),
            F.col("tmdb_id").alias("tmdbId"),
            F.col("title"),
            F.col("description"),
            F.col("year"),
            F.col("release_date").cast(T.StringType()).alias("releaseDate"),
            F.coalesce("rating_count", F.lit(0)).alias("ratingCount"),
            F.coalesce("rating_avg", F.lit(0.0)).alias("ratingAverage"),
            (F.col("adult") == 1).alias("adult"),
            F.col("tagline"),
            F.col("imdb_id").alias("imdbId"),
            F.col("poster_uri").alias("posterUri"),
            F.col("budget"),
            F.col("revenue"),
            F.col("origin_country_codes").alias("originCountries"),
            F.col("production_country_codes").alias("productionCountries"),
            F.col("description_embedding").alias("embedding"),
        ]
        for source, target in GENRE_COLUMNS.items():
            select.append((F.col(f"`{source}`") >= 1).alias(target))
        for name in SENTIMENT_COLUMNS:
            select.append(F.col(name))

        self._movies_df = (
            self._movies_df.select(*select)
            .dropDuplicates(["id"])
            .filter(
                (F.col("year") >= MIN_YEAR)
                & (F.col("year") <= MAX_YEAR)
                & (F.length("title") > 0)
                & (F.length("description") > 0)
                & (F.size("embedding") == EMBEDDING_SIZE)
            )
        )

    def save(self) -> None:
        assert self._movies_df is not None

        batch_size = self.config.int("MOVIE_API_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        sent = 0
        rejected = 0
        batch: list[Movie] = []

        for row in self._movies_df.toLocalIterator():
            try:
                batch.append(Movie(**row.asDict()))
            except ValidationError as e:
                rejected += 1
                logger.warning("Skipping movie %s: does not satisfy API schema: %s", row["id"], e)
                continue
            if len(batch) >= batch_size:
                self._api_client.upsert_movies(batch)
                sent += len(batch)
                logger.info("Upserted %s movies", sent)
                batch = []

        if batch:
            self._api_client.upsert_movies(batch)
            sent += len(batch)
            logger.info("Upserted %s movies", sent)

        logger.info("Sync complete: %s movies upserted, %s rejected by client-side validation", sent, rejected)


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"],
    api_client: MovieApiClient = Provide["movie_api_client"],
):
    step = SyncDataToApiJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer, api_client=api_client)
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
