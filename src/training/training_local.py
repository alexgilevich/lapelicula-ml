import os

import pandas as pd
from pyspark.sql.types import * 

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
from config import Config, SecretsManager
from job_step import JobStep
from logging_factory import get_logger
from mlflow.models import infer_signature
import boto3
import shutil
import numpy as np
import tensorflow as tf

DEFAULT_MODEL_LOCAL_SAVE_PATH = "../../model_artifacts/mlflow_models/"

logger = get_logger(__name__)

DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-movie-recommender-model-18-feb"
DEFAULT_NUM_MODEL_LAYER_OUTPUTS = 256
DEFAULT_NUM_EPOCHS = 30


movie_train_data_schema = {"row_id":"int64","movie_id":"int32","title":"object","Action":"int64","Adventure":"int64","Animation":"int64","Comedy":"int64","Crime":"int64","Documentary":"int64","Drama":"int64","Fantasy":"int64","Film-Noir":"int64","Horror":"int64","Kids":"int64","Musical":"int64","Mystery":"int64","Romance":"int64","Sci-Fi":"int64","Thriller":"int64","War":"int64","Western":"int64"}
user_train_data_schema = {"row_id":"int64","user_id":"int64","Action":"float64","Adventure":"float64","Animation":"float64","Comedy":"float64","Crime":"float64","Documentary":"float64","Drama":"float64","Fantasy":"float64","Film-Noir":"float64","Horror":"float64","Kids":"float64","Musical":"float64","Mystery":"float64","Romance":"float64","Sci-Fi":"float64","Thriller":"float64","War":"float64","Western":"float64"}
y_schema = {"row_id":"int64","rating":"float64"}

movie_train_data_pdf = pd.read_csv("train_data_movies.csv", dtype=movie_train_data_schema)
user_train_data_pdf = pd.read_csv("train_data_users.csv", dtype=user_train_data_schema)
y_pdf = pd.read_csv("train_data_y.csv", dtype=y_schema)
model_save_path = "model_binary_v4.keras"

model = Model(num_outputs = DEFAULT_NUM_MODEL_LAYER_OUTPUTS, num_epochs=DEFAULT_NUM_EPOCHS, model_save_path=model_save_path)


logger.info("User training initial df data shape: %s, columns: %s", movie_train_data_pdf.shape, movie_train_data_pdf.columns)
logger.info("Movie training initial df data shape: %s, columns: %s", user_train_data_pdf.shape, user_train_data_pdf.columns)
logger.info("Label training initial df data shape: %s, columns: %s", y_pdf.shape, y_pdf.columns)


movie_train_data = movie_train_data_pdf.drop(columns=['row_id', 'title'], errors="ignore").to_numpy()
user_train_data  = user_train_data_pdf.drop(columns=['row_id'], errors="ignore").to_numpy()
y_labels         = y_pdf.drop(columns=['row_id']).to_numpy()

metrics = model.train(user_train_data, movie_train_data, y_labels)

model.save()

model.load()
