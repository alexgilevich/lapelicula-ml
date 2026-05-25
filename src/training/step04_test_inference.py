# The script loads a trained model using MLFlow and tests inference by making predictions for different user preferences. It retrieves the preprocessed movies data from a Unity Catalog table, converts it to a format suitable for the model, and then uses the model to generate movie recommendations based on the provided user preferences. The recommendations are then joined with the original movies data to display the recommended movies along with their details.
# The script is not included in the main training pipeline and is meant to be run separately for testing purposes. It demonstrates how to load a trained model, prepare input data, and perform inference to get recommendations based on user preferences.

import os

import pandas as pd
from pyspark.sql.types import * 

import _includes
import mlflow
from features import UserPreferences
from mlflow_utils import MLFlowModelManager
from model import Model
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from config import Config, SecretsManager
from job_step import JobStep
from logging_factory import get_logger
from mlflow.models import infer_signature
import boto3
import shutil
import numpy as np

DEFAULT_MODEL_LOCAL_SAVE_PATH = "../../model_artifacts/mlflow_models/"

logger = get_logger(__name__)

DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-movie-recommender-model-18-feb"
DEFAULT_NUM_MODEL_LAYER_OUTPUTS = 256
DEFAULT_NUM_EPOCHS = 2

class TestInferenceWithTrainedModelJobStep(JobStep):
    """
    Train and publish a model on Unity Catalog
    
    Config:
    - UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, model_manager: MLFlowModelManager, secrets_manager: SecretsManager):
        super().__init__(spark, config, dataframe_writer)
        self._model_manager: MLFlowModelManager = model_manager
        self._secrets_manager: SecretsManager = secrets_manager

    # ---------------------- step contract ----------------------
    def load(self) -> None:
        """
        Initialize dataframes for training
        """
        self._movies_preprocessed_df = self.spark.table("movies_preprocessed").withColumnRenamed("movieId", "movie_id")

    def process(self) -> None:
        model_name = self.config.string("MLFLOW_MODEL_NAME", DEFAULT_MLFLOW_MODEL_NAME)
        local_model_save_path = self.config.int("MODEL_LOCAL_SAVE_PATH", DEFAULT_MODEL_LOCAL_SAVE_PATH)
        full_local_model_save_path = os.path.join(local_model_save_path, model_name)
        
        inference_params = { "limit": 3000 }
        all_movies_data = self._movies_preprocessed_df.toPandas().drop(columns=['row_id', 'title', 'genres', 'year', 'rating_count', 'rating_avg', 'genre_partition0', 'genre_partition1'], errors="ignore").to_numpy()
        
        model = mlflow.pyfunc.load_model(full_local_model_save_path)
        examples = [
            UserPreferences(kids = 5, animation = 5, adventure = 0, comedy = 0, mystery = 0, crime = 0, horror = 0, sci_fi = 0),
            UserPreferences(action = 5, adventure = 3.5, mystery = 4, horror = 1, sci_fi = 4, western = 3, drama = 3, animation = 0.5),
            UserPreferences(kids = 5, animation = 5, adventure = 4.5, comedy = 4.5, mystery = 2, crime = 1, horror = 0.5, sci_fi = 4),
            UserPreferences( comedy = 4.5, romance = 5, mystery = 2, crime = 0.5, horror = 0.5, sci_fi = 1.5),
            UserPreferences(kids = 5, animation = 5, adventure = 4.5),
            UserPreferences(action = 5,  sci_fi = 4.5 ),
            UserPreferences(comedy = 4.5,  romance = 4.5 ),
            UserPreferences(war=5),
            UserPreferences(western=5),
        ]
        
        for i, example in enumerate(examples):
            print(f"Predicting for user ###{i + 1}: ", example.to_dict()) 
            input = pd.DataFrame([{
                "user_preferences": example.to_dict(),
                "movies": all_movies_data
            }])
            predictions = model.predict(input, inference_params)
            #logger.info("loaded: %s", predictions)
            recommendations = [(int(movie_id), float(rating)) for movie_id, rating in predictions[0]]
            recommendations_df = self.spark.createDataFrame(recommendations, schema=StructType([
                StructField(name="movie_id", dataType=IntegerType(), nullable=True),
                StructField(name="rating", dataType=StringType(), nullable=True)
            ]))
            recommendations_df.join(self._movies_preprocessed_df, 'movie_id', how='inner').withColumn('rn', F.expr('row_number() over(order by rating desc)')).show(1000,  truncate = False)
            print("\n\n\n\n")
        #logger.info("Model max output difference is: %f", np.max(np.abs(np.array(orig)[:, 1:] - np.array(loaded)[:, 1:])))

    

        
    def save(self) -> None:
        pass
        


@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"],
    model_manager: MLFlowModelManager = Provide["model_manager"],
    secrets_manager: SecretsManager = Provide["secrets_manager"]
):
    step = TestInferenceWithTrainedModelJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer, model_manager=model_manager, secrets_manager=secrets_manager)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()