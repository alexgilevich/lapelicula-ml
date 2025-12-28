import os.path
from pyspark.sql import SparkSession, DataFrame
from config import Config


class DataFrameWriter:
    def __init__(self, spark_session: SparkSession, config: Config, default_location: str = None):
        self.spark_session = spark_session
        self.default_location = default_location
        self.config = config

    def write(self, dataframe: DataFrame, table_name: str):
        # this condition is needed because the open-source version of Unity Catalog does not yet support managed tables (refer to https://github.com/unitycatalog/unitycatalog/issues/1143)
        if self.default_location:
            # we need to first write the data to the external table location
            dataframe.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(os.path.join(self.default_location, table_name))
            # and then create a table in Unity Catalog
            self.spark_session.sql(f"CREATE OR REPLACE TABLE {table_name} USING DELTA LOCATION '{os.path.join(self.default_location, table_name)}'")
        else:
            overwrite = self.config.bool("OVERWRITE", True)
            mode = "overwrite" if overwrite else "errorifexists"
            dataframe.write.format("delta").mode(mode).option("mergeSchema", "true").saveAsTable("raw_movies")
