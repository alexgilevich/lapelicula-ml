import os

import pandas as pd

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

DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-recommender-model"
DEFAULT_NUM_MODEL_LAYER_OUTPUTS = 256
DEFAULT_NUM_EPOCHS = 2

class TrainModelJobStep(JobStep):
    """
    Train and publish a model on Unity Catalog
    
    Config:
    - UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, model_manager: MLFlowModelManager, secrets_manager: SecretsManager):
        super().__init__(spark, config, dataframe_writer)
        self._training_movies_df: DataFrame | None = None
        self._training_users_df: DataFrame | None = None
        self._training_labels_df: DataFrame | None = None
        self._model_manager: MLFlowModelManager = model_manager
        self._secrets_manager: SecretsManager = secrets_manager

    # ---------------------- step contract ----------------------
    def load(self) -> None:
        """
        Initialize dataframes for training
        """
        self._training_movies_df = self.spark.table("training_movies")
        self._training_users_df = self.spark.table("training_users")
        self._training_labels_df = self.spark.table("training_labels")
        self._movies_preprocessed_df = self.spark.table("test_catalog.default.movies_preprocessed")

    def process(self) -> None:
        assert self._training_movies_df is not None
        assert self._training_users_df is not None
        assert self._training_labels_df is not None

        model_name = self.config.string("MLFLOW_MODEL_NAME", DEFAULT_MLFLOW_MODEL_NAME)
        num_epochs = self.config.int("NUM_EPOCHS", DEFAULT_NUM_EPOCHS)
        num_model_layer_outputs = self.config.int("NUM_MODEL_LAYER_OUTPUTS", DEFAULT_NUM_MODEL_LAYER_OUTPUTS)
        local_model_save_path = self.config.int("MODEL_LOCAL_SAVE_PATH", DEFAULT_MODEL_LOCAL_SAVE_PATH)


        self._model_manager.start_experiment()
        
        movie_train_data_pdf = self._training_movies_df.orderBy(F.col('row_id')).toPandas()
        user_train_data_pdf = self._training_users_df.orderBy(F.col('row_id')).toPandas()
        y_pdf = self._training_labels_df.orderBy(F.col('row_id')).toPandas()
        model = Model(num_outputs = num_model_layer_outputs, num_epochs=num_epochs)

        with mlflow.start_run() as run:
            logger.info("User training initial df data shape: %s, columns: %s", movie_train_data_pdf.shape, movie_train_data_pdf.columns)
            logger.info("Movie training initial df data shape: %s, columns: %s", user_train_data_pdf.shape, user_train_data_pdf.columns)
            logger.info("Label training initial df data shape: %s, columns: %s", y_pdf.shape, y_pdf.columns)
            
            
            movie_train_data = movie_train_data_pdf.drop(columns=['row_id', 'title'], errors="ignore").to_numpy()
            user_train_data  = user_train_data_pdf.drop(columns=['row_id'], errors="ignore").to_numpy()
            y_labels         = y_pdf.drop(columns=['row_id']).to_numpy()
            
            metrics = model.train(user_train_data, movie_train_data, y_labels)

            # log params
            mlflow.log_params({ 
                "num_outputs": model.num_outputs,
                "num_epochs": model.num_epochs,
            })
            
            mlflow.log_metrics(metrics)


            all_movies_data = self._movies_preprocessed_df.toPandas().drop(columns=['row_id', 'title', 'genres', 'year', 'rating_count', 'rating_avg', 'genre_partition0', 'genre_partition1'], errors="ignore").to_numpy()
            # log model
            model_input_signature = pd.DataFrame([{
                "user_preferences": UserPreferences(action=0, animation=0, comedy=0, crime=5, documentary=5, drama=0, family=0, fantasy=0, film_noir=0, history=2, horror=3, music=0, mystery=4.5, romance=0, sci_fi=0, thriller=0, war=3, western=0).to_dict(),
                "movies": all_movies_data[:3000]
            }])
            inference_params = { "limit": 1000 }
            signature = infer_signature(
                model_input = model_input_signature, 
                model_output = model.predict(model_input_signature, inference_params),
                params = inference_params
            )

            #temp
            orig = model.predict(model_input_signature, inference_params)
            logger.info("orig: %s", orig)
            
            
            logger.info("Model signature is: %s", signature)
            model_log_info = mlflow.pyfunc.log_model(model_name, python_model=model, code_paths=["../model.py", "../features.py"])
            logger.info("Successfully logged the trained model: %s", model_log_info)
            
            
            
            
            model_save_bucket = self.config.string("MODEL_SAVE_S3_BUCKET")
            model_save_prefix = self.config.string("MODEL_SAVE_S3_PREFIX")
            if model_save_bucket:
                full_local_model_save_path = os.path.join(local_model_save_path, model_name)
                try:
                    if os.path.exists(full_local_model_save_path):
                        shutil.rmtree(full_local_model_save_path)
                    model_save_info = mlflow.pyfunc.save_model(full_local_model_save_path, python_model=model, signature=signature,
                                                               code_paths=["../model.py", "../features.py"])
                    logger.info("Successfully saved the trained model to the path `%s` with the following details: %s", full_local_model_save_path, model_save_info)
                except Exception as e:
                    logger.warning("Failed to save the trained model to the path `%s`", full_local_model_save_path, exc_info=e)

                #temp
                loaded = mlflow.pyfunc.load_model(full_local_model_save_path).predict(model_input_signature, inference_params)
                logger.info("Model max output difference is: %f", np.max(np.abs(np.array(orig)[:, 1:] - np.array(loaded)[:, 1:])))

                try:
                    access_key = self._secrets_manager.get("AWS_ACCESS_KEY_ID")
                    secret_key = self._secrets_manager.get("AWS_SECRET_ACCESS_KEY")
                    region = self._secrets_manager.get("AWS_DEFAULT_REGION")
                    client = boto3.client(
                        's3',
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        region_name=region
                    )

                    exclude = {'__pycache__'}
                    for path, dirs, files in os.walk(local_model_save_path, topdown=True):
                        dirs[:] = [d for d in dirs if d not in exclude]
                        for file in files:
                            file_s3_key = os.path.join(model_save_prefix, os.path.normpath(path[len(local_model_save_path):] + '/' + file))
                            file_local_path = os.path.join(path, file)
                            logger.info("Uploading file `%s` to S3 bucket `%s` with key `%s`", file_local_path, model_save_bucket, file_s3_key)
                            client.upload_file(file_local_path, model_save_bucket, file_s3_key)
                            logger.info("Uploaded file `%s` to S3 bucket `%s` with key `%s`", file_local_path, model_save_bucket, file_s3_key)
                except Exception as e:
                    logger.warning("Failed to copy the trained model to the S3 path `s3://%s/%s/`", model_save_bucket, model_save_prefix,
                                   exc_info=e)



        model_details = self._model_manager.register_model(model_log_info)
        self._model_manager.wait_until_registered(model_details.name, model_details.version)
        try:
            self._model_manager.promote_to_production(model_details.name, model_details.version)
        except Exception as e:
            logger.warning("Promotion to production failed (probably because you are using a locally deployed Unity Catalog OSS): %s", e)

        
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
    step = TrainModelJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer, model_manager=model_manager, secrets_manager=secrets_manager)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()