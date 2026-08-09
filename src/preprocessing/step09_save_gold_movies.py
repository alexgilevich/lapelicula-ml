import _includes
from dependency_injector.wiring import inject, Provide
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.sql.functions import pandas_udf, sentences
import pandas as pd
from config import Config
from job_step import JobStep
from logging_factory import get_logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


logger = get_logger(__name__)



class SaveMoviesJobStep(JobStep):
    """
    Step 8: Detect sentiment in movie descriptions using a pre-trained model
    - Use a sentiment analysis model from Hugging Face

    Produces: gold_movies

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._movies_df: DataFrame | None = None
        self._movie_sentiment_df: DataFrame | None = None
        self._movie_embeddings_df: DataFrame | None = None
        self._gold_movies_df: DataFrame | None = None

    def load(self) -> None:
        self._movies_df = self.spark.table("silver_movies_enriched")
        self._movie_sentiment_df = self.spark.table("silver_movie_sentiment")
        self._movie_embeddings_df = self.spark.table("silver_movie_embedding")
        


    def process(self) -> None:
        assert self._movies_df is not None
        assert self._movie_sentiment_df is not None
        assert self._movie_embeddings_df is not None

        movies_df = self._movies_df.selectExpr("* EXCEPT(adult)", "CAST(adult AS INT) AS adult")
        self._gold_movies_df = (movies_df.join(self._movie_sentiment_df, on="movie_id", how="inner")
                                .join(self._movie_embeddings_df, on="movie_id", how="inner")
                                .selectExpr("movie_id as movie_id", "* EXCEPT(movie_id)"))
        

    def save(self) -> None:
        assert self._gold_movies_df is not None
        self.dataframe_writer.write(self._gold_movies_df, "gold_movie")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = SaveMoviesJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()