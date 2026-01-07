from typing import Iterable

from boto3.resources.base import ServiceResource
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame, Row
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F, types as T
from config import Config, SecretsManager
from job_step import JobStep
import boto3
from boto3.dynamodb.table import BatchWriter
import boto3.dynamodb.types as ddbtypes
from logging_factory import get_logger
from decimal import Decimal

logger = get_logger(__name__)

class SyncDataToDynamoDbJobStep(JobStep):
    """
    Step 10: Export top 200 movies
    - Sort movies by rating_count desc (fallback to compute from ratings if missing)
    - Persist top-200 to UC table top200_movies and optionally to CSV path

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    - EXPORT_PATH: optional external path to write CSV (e.g., dbfs:/FileStore/top200_movies.csv)
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, secrets_manager: SecretsManager):
        super().__init__(spark, config, dataframe_writer)
        self._movies_df: DataFrame | None = None
        self._top200_df: DataFrame | None = None
        self._secrets_manager = secrets_manager

    def _table_exists(self, full_name: str) -> bool:
        try:
            return self.spark.catalog.tableExists(full_name)
        except Exception:
            return False

    def load(self) -> None:
        self._movies_df = self.spark.table("movies_enriched")

    def process(self) -> None:
        self._movies_df = self._movies_df.select(
            F.col("movieId").alias("movie_id"),
            F.col("genres").alias("genres"),
            F.col("year").alias("year"),
            F.col("rating_count").alias("rating_count"),
            F.col("rating_avg").alias("rating_avg"),
            F.col("tmdbId").alias("tmdb_id"),
            F.col("imdbId").alias("imdb_id"),
            F.col("title").alias("title"),
            F.col("description").alias("description"),
            F.col("poster_uri").alias("poster_uri"),
            F.col("budget").alias("budget"),
            F.col("release_date").cast(T.StringType()).alias("release_date"),
            F.col("origin_countries").alias("origin_countries"),
        ).dropDuplicates(["movie_id"])





    def save(self) -> None:
        assert self._movies_df is not None

        # although boto3 does not require passing the credentials explicitly if they are set using environment variables,
        # it is still required to do so in the Databricks cluster where credentials should retrieved first from the secrets

        secrets_manager = self._secrets_manager
        access_key = secrets_manager.get("AWS_ACCESS_KEY_ID")
        secret_key = secrets_manager.get("AWS_SECRET_ACCESS_KEY")
        region = secrets_manager.get("AWS_DEFAULT_REGION")

        def get_dynamodb_resource() -> ServiceResource:
            return boto3.resource('dynamodb',
                                  aws_access_key_id=access_key,
                                  aws_secret_access_key=secret_key,
                                  region_name=region)


        def create_table(table_name: str):
            dynamodb = get_dynamodb_resource()

            existing_tables = dynamodb.meta.client.list_tables()['TableNames']
            if table_name not in existing_tables:
                logger.info("Creating table %s", table_name)
                table = dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[{'AttributeName': 'movie_id', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[
                        {'AttributeName': 'movie_id', 'AttributeType': ddbtypes.NUMBER}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )

                logger.info("Waiting for table %s to be created", table_name)
                table.meta.client.get_waiter('table_exists').wait(TableName=table_name)

        create_table("movies")

        def process_partition(partition: Iterable[Row]):
            dynamodb_resource = get_dynamodb_resource()
            with dynamodb_resource.Table("movies").batch_writer() as writer:
                for row in partition:
                    writer.put_item(Item={
                        "movie_id": row["movie_id"],
                        "genres": row["genres"],
                        "year": row["year"],
                        "rating_count": row["rating_count"],
                        "rating_avg": Decimal(str(row["rating_avg"])),
                        "tmdb_id": row["tmdb_id"],
                        "imdb_id": row["imdb_id"],
                        "title": row["title"],
                        "description": row["description"],
                        "poster_uri": row["poster_uri"],
                        "budget": Decimal(str(row["budget"])),
                        "release_date": row["release_date"],
                        "origin_countries": row["origin_countries"],
                    })

        self._movies_df.foreachPartition(process_partition)


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"],
    secrets_manager: SecretsManager = Provide["secrets_manager"]
):
    step = SyncDataToDynamoDbJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer, secrets_manager=secrets_manager)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()