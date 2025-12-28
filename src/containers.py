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
    dbutils = providers.Object(None)
    config = providers.Factory(ArgumentsConfig)
    secrets_manager = providers.Factory(DBUtilsSecretsManager, config=config, dbutils=dbutils)

class _LocalDevelopmentContainer(containers.DeclarativeContainer):
    spark_application_name = providers.Object("local-development")
    dbutils = providers.Object(None)
    config = providers.Factory(EnvConfig)
    secrets_manager = providers.Factory(EnvSecretsManager)
    spark_session = providers.Factory(get_spark, spark_application_name, config)
    default_table_location = providers.Object("/tmp/unity-catalog/lapelicula/")
    default_model_location = providers.Object("/tmp/unity-catalog/")
    model_manager = providers.Factory(MLFlowModelManager, config=config)
    dataframe_writer = providers.Factory(DataFrameWriter, spark_session=spark_session, config=config, default_location=default_table_location)
    tmdb_client = providers.Factory(TMDBClient, secrets_manager=secrets_manager)