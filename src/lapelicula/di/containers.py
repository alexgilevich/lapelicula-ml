from os import environ

from dependency_injector import containers, providers

__all__ = ["ContainerFactory"]

from lapelicula.api.movie_management.client import create_movie_api_client
from lapelicula.api.tmdb.tmdb import TMDBClient
from lapelicula.config import ArgumentsConfig, DBUtilsSecretsManager, EnvConfig, EnvSecretsManager
from lapelicula.recommender.mlflow_utils import MLFlowModelManager
from lapelicula.spark.dataframe import DataFrameWriter
from lapelicula.spark.spark_utils import get_spark


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
    movie_api_client = providers.Factory(create_movie_api_client, config=config, secrets_manager=secrets_manager)


class _LocalDevelopmentContainer(containers.DeclarativeContainer):
    spark_application_name = providers.Object("local-development")
    dbutils = providers.Object(None)
    config = providers.Factory(EnvConfig)
    secrets_manager = providers.Factory(EnvSecretsManager)
    spark_session = providers.Factory(get_spark, spark_application_name, config)
    model_manager = providers.Factory(MLFlowModelManager, config=config)
    dataframe_writer = providers.Factory(DataFrameWriter, spark_session=spark_session, config=config)
    tmdb_client = providers.Factory(TMDBClient, secrets_manager=secrets_manager)
    movie_api_client = providers.Factory(create_movie_api_client, config=config, secrets_manager=secrets_manager)
