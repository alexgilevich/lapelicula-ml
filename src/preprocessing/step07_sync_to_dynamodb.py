import _includes
import boto3
from typing import Iterable
from boto3.resources.base import ServiceResource
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame, Row
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F, types as T
from config import Config, SecretsManager
from job_step import JobStep
from logging_factory import get_logger
from decimal import Decimal

logger = get_logger(__name__)


TARGET_TABLE = "movies"


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
            F.col("production_countries").alias("production_countries"),
            F.col("vote_average").alias("vote_average"),
            F.col("revenue").alias("revenue"),
            F.col("tagline").alias("tagline"),
            F.col("adult").alias("adult"),
        ).dropDuplicates(["movie_id"])
    

    def save(self) -> None:
        assert self._movies_df is not None

        self.create_table()
        
        
        self._movies_df.foreachPartition(self.create_partition_processor())

    def create_table(self) -> None:
        access_key = self._secrets_manager.get("AWS_ACCESS_KEY_ID")
        secret_key = self._secrets_manager.get("AWS_SECRET_ACCESS_KEY")
        region = self._secrets_manager.get("AWS_DEFAULT_REGION")

        dynamodb = boto3.resource('dynamodb',
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key,
                                    region_name=region)

        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if TARGET_TABLE not in existing_tables:
            logger.info("Creating table %s", TARGET_TABLE)
            table = dynamodb.create_table(
                TableName=TARGET_TABLE,
                KeySchema=[{'AttributeName': 'movie_id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[
                    {'AttributeName': 'movie_id', 'AttributeType': boto3.dynamodb.types.NUMBER}
                ],
                BillingMode='PAY_PER_REQUEST'
            )

            logger.info("Waiting for table %s to be created", TARGET_TABLE)
            table.meta.client.get_waiter('table_exists').wait(TableName=TARGET_TABLE)

    def create_partition_processor(self) -> callable:
        # create DynamoDB resource inside the worker to avoid serialization issues
        access_key = self._secrets_manager.get("AWS_ACCESS_KEY_ID")
        secret_key = self._secrets_manager.get("AWS_SECRET_ACCESS_KEY")
        region = self._secrets_manager.get("AWS_DEFAULT_REGION")
        table_name = TARGET_TABLE

        def _process(partition: Iterable[Row]):
            dynamodb_resource = boto3.resource('dynamodb',
                                                aws_access_key_id=access_key,
                                                aws_secret_access_key=secret_key,
                                                region_name=region)


            with dynamodb_resource.Table(table_name).batch_writer() as writer:
                for row in partition:
                    writer.put_item(Item={
                        "movie_id": row["movie_id"],
                        "genres": row["genres"],
                        "year": row["year"],
                        "rating_count": row["rating_count"],
                        "rating_avg": Decimal(str(row["rating_avg"])) if row["rating_avg"] is not None else Decimal(),
                        "vote_average": Decimal(row["vote_average"])if row["vote_average"] is not None else Decimal(),
                        "tmdb_id": row["tmdb_id"],
                        "imdb_id": row["imdb_id"],
                        "title": row["title"],
                        "description": row["description"],
                        "tagline": row["tagline"],
                        "poster_uri": row["poster_uri"],
                        "budget": Decimal(str(row["budget"])) if row["budget"] is not None else None,
                        "revenue": Decimal(str(row["revenue"])) if row["revenue"] is not None else None,
                        "release_date": row["release_date"],
                        "origin_countries": row["origin_countries"],
                        "production_countries": row["production_countries"],
                        "adult": row["adult"],
                    })
            return _process

        return _process


    


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