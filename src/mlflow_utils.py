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

class MLFlowModelManager:
    MODEL_URI = "models:/{catalog_name}.{schema_name}.{model_name}@{label}".format
    
    
    def __init__(self, config: Config):
        self._client: MlflowClient = MlflowClient()
        self._default_model_location: str = config.string("ML_MODEL_LOCATION_PREFIX", "/Workspace/")
        self._catalog_name: str = config.string("UC_DEFAULT_CATALOG_NAME", "lapelicula")
        self._schema_name: str = config.string("UC_DEFAULT_SCHEMA_NAME", "default")
        self._model_name: str = config.string("UC_MODEL_NAME", "lapelicula-recommender-model")
        self._experiment_name: str = config.string("ML_MODEL_EXPERIMENT_NAME", "default")


    def get_experiments_path(self) -> str:
        result = os.path.join(self._default_model_location, f"{self._catalog_name}/mlflow/experiments/{self._model_name}/{self._experiment_name}")
        pathlib.Path(result).parent.mkdir(parents=True, exist_ok=True)
        return result

    def get_experiments_name(self) -> str:
        return f"{self._model_name}_{self._experiment_name}"

    def start_experiment(self) -> None:
        exp_name = self.get_experiments_path()
    
        logger.info(f"Creating experiment with name {exp_name}")
    
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        
        existing_experiment = mlflow.get_experiment_by_name(exp_name)
        if existing_experiment:
            mlflow_experiment = mlflow.set_experiment(exp_name)
            logger.info(f"Using existing experiment: {mlflow_experiment.experiment_id}")
        else:
            _ = mlflow.create_experiment(name=exp_name)
            mlflow_experiment = mlflow.set_experiment(exp_name)
            logger.info(f"Created new experiment: {mlflow_experiment.experiment_id}")


    
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
