from dependency_injector import containers, providers
from os import environ

__all__ = ["ContainerFactory"]

from config import ArgumentsConfig, EnvConfig, DBUtilsSecretsManager, EnvSecretsManager
from dataframe import DataFrameWriter
from mlflow_utils import MLFlowModelManager
from spark_utils import get_spark
from catalog_utils import CatalogUtils
from tmdb import TMDBClient


class ContainerFactory:
    @staticmethod
    def create_container() -> containers.DeclarativeContainer:
        match environ.get("CONTAINER_TYPE", ""):

            case "local-development":
                return _LocalDevelopmentContainer()
            case _:
                return _DatabricksContainer()



class _DatabricksContainer(containers.DeclarativeContainer):
    def get_dbutils():
        try:
            import IPython
            return IPython.get_ipython().user_ns["dbutils"]
        except ImportError:
            raise ImportError("IPython is not available. Make sure you're not in a non-IPython environment.")
        
    spark_application_name = providers.Object("lapelicula-databricks-job")
    dbutils = providers.Factory(get_dbutils)
    config = providers.Factory(ArgumentsConfig)
    secrets_manager = providers.Factory(DBUtilsSecretsManager, config=config, dbutils=dbutils)
    spark_session = providers.Factory(get_spark, spark_application_name, config)
    model_manager = providers.Factory(MLFlowModelManager, config=config)
    dataframe_writer = providers.Factory(DataFrameWriter, spark_session=spark_session, config=config)
    tmdb_client = providers.Factory(TMDBClient, secrets_manager=secrets_manager)

class _LocalDevelopmentContainer(containers.DeclarativeContainer):
    spark_application_name = providers.Object("local-development")
    dbutils = providers.Object(None)
    config = providers.Factory(EnvConfig)
    secrets_manager = providers.Factory(EnvSecretsManager)
    spark_session = providers.Factory(get_spark, spark_application_name, config)
    model_manager = providers.Factory(MLFlowModelManager, config=config)
    dataframe_writer = providers.Factory(DataFrameWriter, spark_session=spark_session, config=config)
    tmdb_client = providers.Factory(TMDBClient, secrets_manager=secrets_manager)