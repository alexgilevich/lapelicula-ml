# The script loads a trained model using MLFlow and tests inference by making predictions for different user preferences. It retrieves the preprocessed movies data from a Unity Catalog table, converts it to a format suitable for the model, and then uses the model to generate movie recommendations based on the provided user preferences. The recommendations are then joined with the original movies data to display the recommended movies along with their details.
# The script is not included in the main training pipeline and is meant to be run separately for testing purposes. It demonstrates how to load a trained model, prepare input data, and perform inference to get recommendations based on user preferences.

import os

import _includes
import mlflow
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession
from containers import ContainerFactory
from dataframe import DataFrameWriter
from config import Config, SecretsManager
from job_step import JobStep
from logging_factory import get_logger
from mlflow_utils import MLFlowModelManager
from inference_test import prepare_inference_input, run_inference_test, build_recommendations_df, DEFAULT_INFERENCE_LIMIT

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
        self._gold_movies_df = self.spark.table("gold_movie")

    def process(self) -> None:
        model_name = self.config.string("MLFLOW_MODEL_NAME", DEFAULT_MLFLOW_MODEL_NAME)
        local_model_save_path = self.config.int("MODEL_LOCAL_SAVE_PATH", DEFAULT_MODEL_LOCAL_SAVE_PATH)
        full_local_model_save_path = os.path.join(local_model_save_path, model_name)

        movies_data, embeddings = prepare_inference_input(self._gold_movies_df, self.config)
        inference_params = { "limit": DEFAULT_INFERENCE_LIMIT }
        model = mlflow.pyfunc.load_model(full_local_model_save_path)

        for result in run_inference_test(model, movies_data, embeddings, inference_params):
            print(f"Predicting for user ###{result['user_index']}: ", result['preferences'])
            build_recommendations_df(self.spark, self._gold_movies_df, result['recommendations']).show(150, truncate=False)
            print("\n\n\n\n")

    

        
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