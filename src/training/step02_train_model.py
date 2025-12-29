import _includes
import mlflow
from features import UserPreferences
from mlflow_utils import MLFlowModelManager
from model import Model
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from containers import ContainerFactory
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from config import Config
from job_step import JobStep
from logging_factory import get_logger
from mlflow.models import infer_signature

logger = get_logger(__name__)

DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-recommender-model"
DEFAULT_NUM_MODEL_LAYER_OUTPUTS = 256
DEFAULT_NUM_EPOCHS = 2

class TrainModelJobStep(JobStep):
    """
    Train and publish a model on Unity Catalog
    
    Config:
    - UC_CATALOG: Unity Catalog catalog name (default: hive_metastore)
    - UC_SCHEMA: Unity Catalog schema/database (default: default)
    - OVERWRITE: If 'true', overwrite existing tables
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter, model_manager: MLFlowModelManager):
        super().__init__(spark, config, dataframe_writer)
        self._training_movies_df: DataFrame | None = None
        self._training_users_df: DataFrame | None = None
        self._training_labels_df: DataFrame | None = None
        self._model_manager: MLFlowModelManager = model_manager

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

        model_name = self.config.string("MLFLOW_MODEL_NAME", DEFAULT_MLFLOW_MODEL_NAME)
        num_epochs = self.config.int("NUM_EPOCHS", DEFAULT_NUM_EPOCHS)
        num_model_layer_outputs = self.config.int("NUM_MODEL_LAYER_OUTPUTS", DEFAULT_NUM_MODEL_LAYER_OUTPUTS)

        self._model_manager.start_experiment()
        
        movie_train_data_pdf = self._training_movies_df.orderBy(F.col('row_id')).toPandas()
        user_train_data_pdf = self._training_users_df.orderBy(F.col('row_id')).toPandas()
        y_pdf = self._training_labels_df.orderBy(F.col('row_id')).toPandas()
        model = Model(num_outputs = num_model_layer_outputs, num_epochs=num_epochs)
        
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
            model_info = mlflow.pyfunc.log_model(model_name, python_model=model, signature=signature, code_paths=["../model.py", "../features.py"])

        model_details = self._model_manager.register_model(model_info)
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
    model_manager: MLFlowModelManager = Provide["model_manager"]
):
    step = TrainModelJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer, model_manager=model_manager)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()