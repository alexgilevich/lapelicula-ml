import os

import pandas as pd

import _includes
import mlflow
from features import UserPreferences
from mlflow_utils import MLFlowModelManager
from model import Model
from inference_test import prepare_inference_input, run_inference_test, build_recommendations_df, DEFAULT_INFERENCE_LIMIT
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

UC_TRAINING_MOVIES_TABLE = "gold_training_movie"
UC_TRAINING_USERS_TABLE = "gold_training_user"
UC_TRAINING_LABELS_TABLE = "gold_training_label"
UC_GOLD_MOVIES_TABLE = "gold_movie"

logger = get_logger(__name__)

DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-recommender-model"
DEFAULT_NUM_MODEL_LAYER_OUTPUTS = 32
DEFAULT_NUM_EPOCHS = 10

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
        self._training_movies_version: int | None = None
        self._training_users_df: DataFrame | None = None
        self._training_users_version: int | None = None
        self._training_labels_df: DataFrame | None = None
        self._training_labels_version: int | None = None
        self._model_manager: MLFlowModelManager = model_manager
        self._secrets_manager: SecretsManager = secrets_manager

    # ---------------------- Delta version helpers ----------------------

    def _get_delta_version(self, table_name: str) -> int | None:
        """Return the latest Delta table version via DESCRIBE HISTORY."""
        row = self.spark.sql(f"DESCRIBE HISTORY {table_name}").orderBy(F.col("version").desc()).first()
        if row is None or row["version"] is None:
            return None
        return int(row["version"])

    # ---------------------- step contract ----------------------
    def load(self) -> None:
        """
        Initialize dataframes for training, capturing Delta table versions.
        """
        self._training_movies_df = self.spark.table(UC_TRAINING_MOVIES_TABLE)
        self._training_movies_version = self._get_delta_version(UC_TRAINING_MOVIES_TABLE)
        self._training_users_df = self.spark.table(UC_TRAINING_USERS_TABLE)
        self._training_users_version = self._get_delta_version(UC_TRAINING_USERS_TABLE)
        self._training_labels_df = self.spark.table(UC_TRAINING_LABELS_TABLE)
        self._training_labels_version = self._get_delta_version(UC_TRAINING_LABELS_TABLE)
        self._gold_movies_df = self.spark.table(UC_GOLD_MOVIES_TABLE)

    def process(self) -> None:
        assert self._training_movies_df is not None
        assert self._training_users_df is not None
        assert self._training_labels_df is not None

        model_name = self.config.string("MLFLOW_MODEL_NAME", DEFAULT_MLFLOW_MODEL_NAME)
        num_epochs = self.config.int("NUM_EPOCHS", DEFAULT_NUM_EPOCHS)
        num_model_layer_outputs = self.config.int("NUM_MODEL_LAYER_OUTPUTS", DEFAULT_NUM_MODEL_LAYER_OUTPUTS)
        local_model_save_path = self.config.int("MODEL_LOCAL_SAVE_PATH", DEFAULT_MODEL_LOCAL_SAVE_PATH)


        self._model_manager.start_experiment()

        MOVIE_COLUMNS = ['movie_id', 
                        "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust", 
                        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Kids', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 
                        'description_embedding',
                        'adult']
        
        movie_with_embs_train_data_pdf = self._training_movies_df.select('row_id', *MOVIE_COLUMNS).orderBy(F.col('row_id')).toPandas()
        movie_train_data_pdf = movie_with_embs_train_data_pdf.drop(columns=['row_id', 'title', 'year', 'description_embedding'], errors="ignore")
        user_train_data_pdf = self._training_users_df.orderBy(F.col('row_id')).toPandas().drop(columns=['row_id'], errors="ignore")
        y_pdf = self._training_labels_df.orderBy(F.col('row_id')).toPandas().drop(columns=['row_id'])
        all_movies_pdf = self._gold_movies_df.select(*MOVIE_COLUMNS).drop('description_embedding').toPandas()

        model = Model(num_outputs = num_model_layer_outputs, num_epochs=num_epochs, verbose_level=2)

        with mlflow.start_run() as run:

            # Log training datasets with Delta table versions
            if self._training_movies_version is not None:
                movies_dataset = mlflow.data.from_spark(
                    self._training_movies_df,
                    table_name=UC_TRAINING_MOVIES_TABLE,
                    version=str(self._training_movies_version),
                )
                mlflow.log_input(movies_dataset, context=f"uc_{UC_TRAINING_MOVIES_TABLE}", tags={f"{UC_TRAINING_MOVIES_TABLE}.delta_version": str(self._training_movies_version)})
            if self._training_users_version is not None:
                users_dataset = mlflow.data.from_spark(
                    self._training_users_df,
                    table_name=UC_TRAINING_USERS_TABLE,
                    version=str(self._training_users_version),
                )
                mlflow.log_input(users_dataset, context=f"uc_{UC_TRAINING_USERS_TABLE}", tags={f"{UC_TRAINING_USERS_TABLE}.delta_version": str(self._training_users_version)})
            if self._training_labels_version is not None:
                labels_dataset = mlflow.data.from_spark(
                    self._training_labels_df,
                    table_name=UC_TRAINING_LABELS_TABLE,
                    version=str(self._training_labels_version),
                )
                mlflow.log_input(labels_dataset, context=f"uc_{UC_TRAINING_LABELS_TABLE}", tags={f"{UC_TRAINING_LABELS_TABLE}.delta_version": str(self._training_labels_version)})

            logger.info("Movie training initial dataframe data shape: %s, columns: %s", movie_with_embs_train_data_pdf.shape, movie_with_embs_train_data_pdf.columns)
            logger.info("Movie training cut down dataframe data shape: %s, columns: %s", movie_train_data_pdf.shape, movie_train_data_pdf.columns)
            logger.info("User training initial dataframe data shape: %s, columns: %s", user_train_data_pdf.shape, user_train_data_pdf.columns)
            logger.info("Label training initial dataframe data shape: %s, columns: %s", y_pdf.shape, y_pdf.columns)
            
            movie_train_data = movie_train_data_pdf.to_numpy(dtype=np.float32)
            user_train_data  = user_train_data_pdf.to_numpy(dtype=np.float32)
            y_labels         = y_pdf.to_numpy(dtype=np.float32)
            all_movies_data = all_movies_pdf.to_numpy(dtype=np.float32)
            embeddings = np.stack(movie_with_embs_train_data_pdf['description_embedding'].apply(lambda x: np.array(x)).values)

            # Diagnostic: train with zeroed embeddings so the model learns to rely
            # on genres/sentiments/adult only. All learned parameters adapt to a
            # world without content signal — this is the faithful "pre-embeddings"
            # model (unlike zeroing at inference, which is out-of-distribution).
            # NOTE: inference on this artifact must also use DISABLE_EMBEDDINGS=true.
            if self.config.bool("DISABLE_EMBEDDINGS", False):
                logger.info("DISABLE_EMBEDDINGS=true — training with zeroed description embeddings")
                embeddings = np.zeros_like(embeddings)

            train_result = model.train(user_train_data, movie_train_data, embeddings, y_labels, all_movies_data)
            metrics = train_result["metrics"]

            # log params
            mlflow.log_params(model.get_params())

            # log test-set metrics (single values per run)
            mlflow.log_metrics({f"{name}_test": value for name, value in metrics.items()})

            # log per-epoch training metrics under one name per metric (step = epoch),
            # so MLflow keeps the full sequence and shows the trend
            for epoch, epoch_logs in enumerate(train_result["epoch_metrics"]):
                for name, value in epoch_logs.items():
                    mlflow.log_metric(f"{name}_training", value, step=epoch)

            # log architecture summary (Keras summary table)
            mlflow.log_text(train_result["architecture_summary"], artifact_file="model_architecture.txt")


            
            # log model
            model_input_signature = pd.DataFrame([{
                "user_preferences": UserPreferences(action=0, animation=0, comedy=0, crime=5, documentary=5, drama=0, family=0, fantasy=0, film_noir=0, history=2, horror=3, music=0, mystery=4.5, romance=0, sci_fi=0, thriller=0, war=3, western=0).to_dict(),
                "movies": all_movies_data[:1000],
                "embeddings": embeddings[:1000]
            }])
            inference_params = { "limit": 1000 }
            signature = infer_signature(
                model_input = model_input_signature, 
                model_output = model.predict(model_input_signature, inference_params),
                params = inference_params
            )
            
            
            logger.info("Model signature is: %s", signature)
            included_code_paths = [os.path.abspath("../model.py"), os.path.abspath("../features.py"), os.path.abspath("../logging_factory.py")]

            model_log_info = mlflow.pyfunc.log_model(model_name, python_model=model, code_paths=included_code_paths)
            logger.info("Successfully logged the trained model: %s", model_log_info)

            # Test the trained model on the full catalog and log the output to MLFlow as
            # artifacts to facilitate tracking training experiments.
            test_movies_data, test_embeddings = prepare_inference_input(self._gold_movies_df, self.config)
            test_inference_params = { "limit": DEFAULT_INFERENCE_LIMIT }
            inference_results = run_inference_test(model, test_movies_data, test_embeddings, test_inference_params)

            mlflow.log_dict(
                [{"user_index": result["user_index"], "preferences": result["preferences"]} for result in inference_results],
                artifact_file="test_inference/user_preferences.json"
            )
            for result in inference_results:
                recommendations_pdf = build_recommendations_df(self.spark, self._gold_movies_df, result["recommendations"]).toPandas()
                mlflow.log_table(recommendations_pdf, artifact_file=f"test_inference/user_{result['user_index']}.json")

            model_save_bucket = self.config.string("MODEL_SAVE_S3_BUCKET")
            model_save_prefix = self.config.string("MODEL_SAVE_S3_PREFIX")
            if model_save_bucket:
                full_local_model_save_path = os.path.join(local_model_save_path, model_name)
                try:
                    if os.path.exists(full_local_model_save_path):
                        shutil.rmtree(full_local_model_save_path)
                    model_save_info = mlflow.pyfunc.save_model(full_local_model_save_path, python_model=model, signature=signature,
                                                               code_paths=included_code_paths)
                    logger.info("Successfully saved the trained model to the path `%s` with the following details: %s", full_local_model_save_path, model_save_info)
                except Exception as e:
                    logger.warning("Failed to save the trained model to the path `%s`", full_local_model_save_path, exc_info=e)

                # test loading the model back and doing a prediction to ensure the saved model is not corrupted before uploading to S3
                _ = mlflow.pyfunc.load_model(full_local_model_save_path).predict(model_input_signature, inference_params)
                

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
                    for path, dirs, files in os.walk(full_local_model_save_path, topdown=True):
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