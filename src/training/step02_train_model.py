import os
from typing import Tuple
from mlflow.pyfunc.stdin_server import params
import _includes
import mlflow
from mlflow import MlflowClient
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from features import UserPreferences
from spark_utils import get_spark
from config import Config, ArgumentsConfig
from job_step import JobStep
from mlflow_utils import MLFlowModelManager
from model import Model, InferenceRequest
from mlflow.models import infer_signature

MLFLOW_MODEL_NAME = "lapelicula-recommender-model"

class TrainModelJobStep(JobStep):
    """
    Train and publish a model on Unity Catalog
    
    Config:
    - UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession | None = None, config: Config | None = None):
        super().__init__(spark, config)
        self._training_movies_df: DataFrame | None = None
        self._training_users_df: DataFrame | None = None
        self._training_labels_df: DataFrame | None = None

    # ---------------------- step contract ----------------------
    def load(self) -> None:
        """
        Initialize dataframes for training
        """
        self._training_movies_df = self.spark.table("training_movies")
        self._training_users_df = self.spark.table("training_users")
        self._training_labels_df = self.spark.table("training_labels")

    def process(self) -> None:
        assert self._training_movies_df is not None
        assert self._training_users_df is not None
        assert self._training_labels_df is not None

        catalog_name = self.config.string("UC_DEFAULT_CATALOG_NAME")
        schema_name = self.config.string("UC_DEFAULT_SCHEMA_NAME")
        mlflow_model_manager = MLFlowModelManager(catalog_name, schema_name, MLFLOW_MODEL_NAME, "default")
        mlflow_model_manager.start_experiment()
        
        movie_train_data_pdf = self._training_movies_df.orderBy(F.col('row_id')).toPandas()
        user_train_data_pdf = self._training_users_df.orderBy(F.col('row_id')).toPandas()
        y_pdf = self._training_labels_df.orderBy(F.col('row_id')).toPandas()
        model = Model(num_outputs = 256, num_epochs=30)
        
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
            
            # log model
            requests = [{
                "user_preferences": UserPreferences(action=0, animation=0, comedy=0, crime=0, documentary=0, drama=0, family=0, fantasy=0, film_noir=0, history=0, horror=0, music=0, mystery=0, romance=0, sci_fi=0, thriller=0, war=0, western=0).to_dict(),
                "movies": movie_train_data[:10]
            }]
            inference_params = { "limit": 50 }
            signature = infer_signature(
                model_input = requests, 
                model_output = model.predict(requests, inference_params),
                params = inference_params
            )
            model_info = mlflow.pyfunc.log_model(MLFLOW_MODEL_NAME, python_model=model, signature=signature, code_paths=["../model.py", "../features.py"])

        model_details = mlflow_model_manager.register_model(model_info)
        mlflow_model_manager.wait_until_registered(model_details.name, model_details.version)
        mlflow_model_manager.promote_to_production(model_details.name, model_details.version)
        
    def save(self) -> None:
        pass
        


if __name__ == "__main__":
    from logging_factory import get_logger
    import training
    logger = get_logger(__name__)
    config = ArgumentsConfig()
    spark = get_spark("step02_train_model", config)
    step = TrainModelJobStep(spark, config)
    step.load()
    step.process()
    step.save()