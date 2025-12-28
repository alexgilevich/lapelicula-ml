from __future__ import annotations
import os
import sys
from config import Config
from pyspark.sql import SparkSession
from logging_factory import get_logger

logger = get_logger(__name__)

def is_databricks_runtime() -> bool:
    """
    Best-effort detection of running inside a Databricks cluster/notebook.
    We prefer an already-created SparkSession if present.
    """
    try:
        active = SparkSession.getActiveSession()
        if active is not None:
            return True
    except Exception as e:
        logger.error("Unexpected error occurred", exc_info=e)
    return os.getenv("DATABRICKS_RUNTIME_VERSION") is not None



def get_spark(app_name: str, config: Config) -> SparkSession:
    """
    Obtain a SparkSession according to the environment:
    - If running on Databricks and a session already exists, return it.
    - Otherwise, try to create a Databricks Connect session using environment variables.
      Expected env vars (Databricks SDK / Connect):
        - DATABRICKS_HOST (https://<workspace-url>)
        - DATABRICKS_TOKEN
        - DATABRICKS_CLUSTER_ID (for All-purpose/job clusters) OR DATABRICKS_SQL_WAREHOUSE_ID
        - UC_DEFAULT_CATALOG_NAME
        - UC_DEFAULT_SCHEMA_NAME
        - Optional: DATABRICKS_CONFIG_PROFILE (if using ~/.databrickscfg)
    - If Databricks Connect is unavailable or misconfigured, fall back to a local Spark session
      to keep local development unblocked, but warn the user.
    """

    # noinspection PyShadowingNames
    def use_default_catalog(spark_session: SparkSession, catalog_name: str, schema_name: str) -> None:
        spark_session.catalog.setCurrentCatalog(catalog_name)
        spark_session.catalog.setCurrentDatabase(schema_name)

    # 1) If an active session already exists (Databricks jobs/notebooks), prefer it
    try:
        active = SparkSession.getActiveSession()
        if active is not None:
            catalog_name = config.string("UC_DEFAULT_CATALOG_NAME")
            schema_name = config.string("UC_DEFAULT_SCHEMA_NAME")
            use_default_catalog(active, catalog_name, schema_name)
            return active
    except Exception as e:
        logger.error("Unexpected error occurred", exc_info=e)

    # # 2) Try Databricks Connect
    # try:
    #     # Databricks Connect v14+ exposes DatabricksSession in databricks.connect
    #     from databricks.connect import DatabricksSession  # type: ignore
    #
    #     # Ensure required env vars are provided; Databricks Connect also reads ~/.databrickscfg
    #     # host = os.getenv("DATABRICKS_HOST")
    #     # token = os.getenv("DATABRICKS_TOKEN")
    #     # cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")
    #     # warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID") or os.getenv("DATABRICKS_WAREHOUSE_ID")
    #     catalog_name = os.getenv("UC_DEFAULT_CATALOG_NAME")
    #     schema_name = os.getenv("UC_DEFAULT_SCHEMA_NAME")
    #
    #
    #     builder = DatabricksSession.builder.appName(app_name)
    #
    #     # If host/token provided, they will be picked up by the underlying Databricks SDK via env vars.
    #     # If neither cluster nor warehouse is provided, Connect may still leverage default profile.
    #     # We don't pass sdk config explicitly to avoid tight coupling with SDK types; env vars are standard.
    #     active = builder.getOrCreate()
    #     use_default_catalog(active, catalog_name, schema_name)
    #     return active
    # except Exception:
    #     # Either library missing or configuration incomplete
    #     warnings.warn(
    #         "Databricks Connect is not available or not configured (check databricks-connect package and env vars). "
    #         "Falling back to a local Spark session for development.")

    catalog_name = config.string("UC_DEFAULT_CATALOG_NAME")
    schema_name = config.string("UC_DEFAULT_SCHEMA_NAME")

    # 3) Fallback to a local standalone Spark cluster (spark://127.0.0.1:7077) with a local Unity Catalog server (http://localhost:8080")

    # We still need to set PYSPARK_PYTHON env var because the Worker daemon (which is not part of this process)
    # needs to know which python to use when spawning executors, and it often looks at the env var inherited
    # or passed during context init. For local standalone, sys.executable ensures the venv is used.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    return (
        SparkSession.builder
        .appName(app_name)
        .master("spark://127.0.0.1:7077")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.cores", "2")
        .config("spark.executor.memory", "4g")
        .config("spark.python.worker.memory", "4g")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "io.unitycatalog.spark.UCSingleCatalog") # spark_catalog is the system catalog created automatically by UC
        .config(f"spark.sql.catalog.{catalog_name}", "io.unitycatalog.spark.UCSingleCatalog") # lapelicula is our own catalog used to read/write tables/models
        .config(f"spark.sql.catalog.{catalog_name}.uri", "http://localhost:8080")
        .config(f"spark.sql.catalog.{catalog_name}.token", "")
        .config("spark.sql.defaultCatalog", catalog_name)
        .config("spark.sql.defaultSchema", schema_name)
        .getOrCreate()
    )
