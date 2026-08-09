import sys
import os

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/')
sys.path.append(script_dir)

import pandas as pd
from pyspark.sql.types import * 
from model import Model
from pyspark.sql import functions as F
from logging_factory import get_logger
import numpy as np
import tensorflow as tf
from spark_utils import get_spark
from config import EnvConfig

spark = get_spark('default', EnvConfig())
spark.sql('DROP TABLE movies_enriched').show()