import datetime
import os
import pathlib
import time
from typing import Any
from datetime import datetime, timezone
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
import mlflow
from pyspark.sql import functions as F, DataFrame
from logging_factory import get_logger
from config import Config

logger = get_logger(__name__)

MLFLOW_MODEL_PRODUCTION_ALIAS = "Production"
DEFAULT_SCHEMA_NAME = "default"
DEFAULT_CATALOG_NAME = "lapelicula"
DEFAULT_MLFLOW_TRAINING_ARTIFACTS_LOCATION = "/Workspace/"
DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-recommender-model"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "default"
DEFAULT_MLFLOW_REGISTRY_URI = "databricks-uc"
DEFAULT_MLFLOW_TRACKING_URI = "databricks"


class MLFlowModelManager:
    MODEL_URI = "models:/{catalog_name}.{schema_name}.{model_name}@{label}".format

    def __init__(self, config: Config):
        self._default_training_artifacts_location: str = config.string("MLFLOW_TRAINING_ARTIFACTS_LOCATION", DEFAULT_MLFLOW_TRAINING_ARTIFACTS_LOCATION)
        self._catalog_name: str = config.string("UC_DEFAULT_CATALOG_NAME", DEFAULT_CATALOG_NAME)
        self._schema_name: str = config.string("UC_DEFAULT_SCHEMA_NAME", DEFAULT_SCHEMA_NAME)
        self._model_name: str = config.string("UC_MODEL_NAME", DEFAULT_MLFLOW_MODEL_NAME)
        self._tracking_uri: str = config.string("MLFLOW_TRACKING_SERVER_URI", None)
        self._registry_uri: str = config.string("MLFLOW_REGISTRY_SERVER_URI", None)
        self._client: MlflowClient = MlflowClient(self._tracking_uri, self._registry_uri)
        self._experiment_name: str = config.string("MLFLOW_MODEL_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT_NAME)


    def get_experiments_path(self) -> str:
        result = os.path.join(self._default_training_artifacts_location, f"experiments/{self._experiment_name}")
        pathlib.Path(result).parent.mkdir(parents=True, exist_ok=True)
        return result

    def start_experiment(self) -> None:
        exp_name = self.get_experiments_path()
    
        logger.info(f"Starting an MLFlow experiment with name {exp_name}")

        if self._tracking_uri is not None:
            logger.info(f"Using MLFlow tracking server at {self._tracking_uri}")
            mlflow.set_tracking_uri(self._tracking_uri)

        if self._registry_uri is not None:
            logger.info(f"Using MLFlow registry server at {self._registry_uri}")
            mlflow.set_registry_uri(self._registry_uri)
        
        existing_experiment = mlflow.get_experiment_by_name(exp_name)
        if existing_experiment:
            mlflow_experiment = mlflow.set_experiment(exp_name)
            logger.info(f"Using existing experiment: id={mlflow_experiment.experiment_id}, name={mlflow_experiment.name}, location={mlflow_experiment.artifact_location}, creation_time={mlflow_experiment.creation_time}")
        else:
            _ = mlflow.create_experiment(name=exp_name)
            mlflow_experiment = mlflow.set_experiment(exp_name)
            logger.info(f"Created new experiment: id={mlflow_experiment.experiment_id}, name={mlflow_experiment.name}, location={mlflow_experiment.artifact_location}, creation_time={mlflow_experiment.creation_time}")

    
    def load_model_as_pyfunc(self, model_name: str, label: str = MLFLOW_MODEL_PRODUCTION_ALIAS):
        model_uri = self.MODEL_URI(catalog_name=self._catalog_name, schema_name=self._schema_name, model_name=model_name, label=label)
        logger.info("Loading model with URI=%s", model_uri)
        return mlflow.pyfunc.load_model(model_uri=model_uri)
    
    
    def load_model_as_spark_udf(self, spark_session, catalog_name: str, model_name: str, label: str = MLFLOW_MODEL_PRODUCTION_ALIAS):
        model_uri = self.MODEL_URI(catalog_name=catalog_name, model_name=model_name, label=label)
        logger.info("Loading model with URI=%s", model_uri)
        return mlflow.pyfunc.spark_udf(spark_session, model_uri=model_uri)
    
    
    @staticmethod
    def apply_model_as_spark_udf(df: DataFrame, model_as_spark_udf, feature_names: list[str], predicted_column_name: str) -> DataFrame:
        return df.withColumn(
            predicted_column_name, model_as_spark_udf(F.struct(*map(F.col, feature_names)))[0]
        )
    
    
    def promote_to_production(self, model_name: str, version: str) -> None:
        self._client.set_registered_model_alias(model_name, MLFLOW_MODEL_PRODUCTION_ALIAS, version)
    
    def register_model(self, model_info) -> Any:
        mlflow_registry_model_name = f"{self._catalog_name}.{self._schema_name}.{self._model_name}"
        result = mlflow.register_model(model_info.model_uri, mlflow_registry_model_name)
        print(result)
        return result


    def wait_until_registered(self, model_name: str, model_version: str, max_seconds: int = 60):
        start = time.time()
        while time.time() - start < max_seconds:
            model_version_details = self._client.get_model_version(
                name=model_name,
                version=model_version,
            )
            status = ModelVersionStatus.from_string(model_version_details.status)

            if status == ModelVersionStatus.READY:
                logger.info("Model '%s' version '%s' has been registered successfully in MLFlow", model_name, model_version)
                return
            time.sleep(1)

        logger.warning("Model '%s' version '%s' is not ready", model_name, model_version)
