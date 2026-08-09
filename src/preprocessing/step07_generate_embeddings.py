import _includes
from dependency_injector.wiring import inject, Provide
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.sql.functions import pandas_udf, sentences
import pandas as pd
from config import Config
from job_step import JobStep
import model
from tmdb import TMDBClient
from logging_factory import get_logger
from transformers import AutoTokenizer, AutoModel
from torch import no_grad, clamp, sum
from torch.nn.functional import normalize as torch_normalize

logger = get_logger(__name__)



class GenerateEmbeddingsJobStep(JobStep):
    """
    Step 7: Generate embeddings for movie descriptions using Sentence Transformers
    - Use the 'all-MiniLM-L6-v2' model from Hugging Face

    Produces: movies_with_embeddings

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, client: TMDBClient = None):
        super().__init__(spark, config, dataframe_writer)
        self._movies_df: DataFrame | None = None

    def load(self) -> None:
        self._movies_df = self.spark.table("movies_enriched")
        self._movies_embeddings_df: DataFrame | None = None
        self.spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "250")


    def process(self) -> None:
        assert self._movies_df is not None

        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return sum(token_embeddings * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)

        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


        @pandas_udf(T.ArrayType(T.FloatType()))
        def generate_embeddings(raw_texts: pd.Series) -> pd.Series:
            logger.info(f"Generating embeddings for {len(raw_texts)} texts")
            # Tokenize sentences
            encoded_input = tokenizer(raw_texts.tolist(), padding=True, truncation=True, return_tensors='pt')
            # Compute token embeddings
            with no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = torch_normalize(sentence_embeddings, p=2, dim=1)

            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # embeddings = model.encode(raw_texts.tolist(), convert_to_numpy=True)
            
            emdeddings = sentence_embeddings.numpy().tolist()
            logger.info(f"Generated embeddings for {len(emdeddings)} texts")
            return pd.Series(emdeddings)
        
        self._movies_embeddings_df = self._movies_df.withColumn("description_embedding", generate_embeddings(F.col("description"))).select("movie_id", "description_embedding")
        self._movies_embeddings_df.show(5)

        

    def save(self) -> None:
        assert self._movies_embeddings_df is not None
        self.dataframe_writer.write(self._movies_embeddings_df, "silver_movie_embedding")

    
@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = GenerateEmbeddingsJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()