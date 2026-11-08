import pandas as pd
import torch
from dependency_injector.wiring import Provide, inject
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import pandas_udf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lapelicula.config import Config
from lapelicula.di.containers import ContainerFactory
from lapelicula.logs.logging_factory import get_logger
from lapelicula.recommender.jobs.job_step import JobStep
from lapelicula.spark.dataframe import DataFrameWriter

logger = get_logger(__name__)


class DetectMovieSentimentJobStep(JobStep):
    """
    Step 8: Detect sentiment in movie descriptions using a pre-trained model
    - Use a sentiment analysis model from Hugging Face

    Produces: movie_sentiment

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._movies_df: DataFrame | None = None

    def load(self) -> None:
        self._movies_df = self.spark.table("silver_movies_enriched")
        self._movies_embeddings_df: DataFrame | None = None
        self.spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "250")

    def process(self) -> None:
        assert self._movies_df is not None

        model_name = "ayoubkirouane/BERT-Emotions-Classifier"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        LABELS = [
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

        @pandas_udf(
            T.StructType(
                [T.StructField(label, T.FloatType()) for label in LABELS]
                + [T.StructField("top_sentiments", T.ArrayType(T.StringType()))]
            )
        )
        def predict_sentiments(raw_texts: pd.Series) -> pd.DataFrame:
            logger.info(f"Detecting sentiments for {len(raw_texts)} texts")
            with torch.no_grad():
                inputs = tokenizer(raw_texts.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=384)
                probs_for_raw_texts = torch.sigmoid(model(**inputs).logits).cpu().numpy()
                results = []
                for row in probs_for_raw_texts:
                    # we are going to need sentiments for all labels for model training and evaluation
                    sentiments = [float(row[i]) for i in range(len(LABELS))]
                    # we are going to need top sentiments for prefiltering
                    top_sentiments = [(LABELS[i], float(row[i])) for i in range(len(LABELS)) if row[i] >= 0.25]
                    top_sentiments.sort(key=lambda x: -x[1])
                    top_sentiments_labels = [top_sentiments[i][0] for i in range(0, len(top_sentiments))] or ["neutral"]
                    results.append(sentiments + [top_sentiments_labels])

            logger.info(f"Detected sentiments for {len(results)} texts")
            return pd.DataFrame(data=results, columns=LABELS + ["top_sentiments"])

        self._movies_embeddings_df = self._movies_df.withColumn(
            "_sentiment", predict_sentiments(F.col("combined_description"))
        ).select("movie_id", F.col("_sentiment.*"))

    def save(self) -> None:
        assert self._movies_embeddings_df is not None
        self.dataframe_writer.write(self._movies_embeddings_df, "silver_movie_sentiment")


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"],
):
    step = DetectMovieSentimentJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
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
