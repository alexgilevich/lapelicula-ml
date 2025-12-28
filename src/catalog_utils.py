import os.path

from pyspark.sql import SparkSession


class CatalogUtils:
    def __init__(self, spark_session: SparkSession, default_location: str = None):
        self.spark_session = spark_session
        self.default_location = default_location

    def create_tables_if_not_exist(self, *tables: str):
        for table in tables:
            if self.default_location:
                self.spark_session.sql(f"CREATE OR REPLACE TABLE {table} USING DELTA LOCATION '{os.path.join(self.default_location, table)}'")
            else:
                self.spark_session.sql(f"CREATE OR REPLACE TABLE {table} USING DELTA")