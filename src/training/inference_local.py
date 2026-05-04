import os

import pandas as pd
from pyspark.sql.types import * 
from features import UserPreferences
from model import Model
from pyspark.sql import functions as F
from job_step import JobStep
from logging_factory import get_logger
import numpy as np
from tabulate import tabulate

DEFAULT_MODEL_LOCAL_SAVE_PATH = "../../model_artifacts/mlflow_models/"

logger = get_logger(__name__)

DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-movie-recommender-model-18-feb"
DEFAULT_NUM_MODEL_LAYER_OUTPUTS = 256
DEFAULT_NUM_EPOCHS = 2

inference_params = { "limit": 3000 }

schema = {"movie_id": "int32", "title": "object", "genres": "object", "year": "int32", "rating_count": "float64", "rating_avg": "float64", "genre_partition0": "int64", "genre_partition1": "int64", "Action": "int64", "Adventure": "int64", "Animation": "int64", "Comedy": "int64", "Crime": "int64", "Documentary": "int64", "Drama": "int64", "Fantasy": "int64", "Film-Noir": "int64", "Horror": "int64", "Kids": "int64", "Musical": "int64", "Mystery": "int64", "Romance": "int64", "Sci-Fi": "int64", "Thriller": "int64", "War": "int64", "Western": "int64"}
all_movies_pdf = pd.read_csv("all_movies.csv", dtype=schema)
all_movies_data = all_movies_pdf.drop(columns=['row_id', 'title', 'genres', 'year', 'rating_count', 'rating_avg', 'genre_partition0', 'genre_partition1'], errors="ignore").to_numpy()

examples = [
    UserPreferences(action = 5, adventure = 3.5, mystery = 4, horror = 1, sci_fi = 4, western = 3, drama = 3, animation = 0.5),
    UserPreferences(kids = 5, animation = 5, adventure = 4.5, comedy = 4.5, mystery = 2, crime = 1, horror = 0.5, sci_fi = 4),
    UserPreferences( comedy = 4.5, romance = 5, mystery = 2, crime = 0.5, horror = 0.5, sci_fi = 1.5),
    UserPreferences(kids = 5, animation = 5, adventure = 4.5),
    UserPreferences(action = 5,  sci_fi = 4.5 ),
    UserPreferences(comedy = 4.5,  romance = 4.5 ),
    UserPreferences(war=5),
    UserPreferences(western=5),
]


model_save_path = "model_binary_v10.keras"

model = Model(num_outputs = DEFAULT_NUM_MODEL_LAYER_OUTPUTS, num_epochs=DEFAULT_NUM_EPOCHS, model_save_path=model_save_path)

model.load()

all_movies_pdf = all_movies_pdf.set_index('movie_id')[['title', 'genres', 'year', 'rating_count', 'rating_avg']]

for i, example in enumerate(examples):
    logger.info(f"Predicting for user ###{i + 1}: ", example.to_dict()) 
    input = pd.DataFrame([{
        "user_preferences": example.to_dict(),
        "movies": all_movies_data
    }])
    predictions = model.predict(input, inference_params)
    # #logger.info("loaded: %s", predictions)
    recommendations = [(int(movie_id), float(rating)) for movie_id, rating in predictions[0]]
    recommendations_pdf = pd.DataFrame(recommendations, columns=['movie_id', 'rating'])

    recommendations_pdf = recommendations_pdf.join(all_movies_pdf, 'movie_id', how='inner')
    print(tabulate(recommendations_pdf.iloc[:200], headers = 'keys', tablefmt = 'psql', maxcolwidths=50))
    
    print("\n\n\n\n")
#logger.info("Model max output difference is: %f", np.max(np.abs(np.array(orig)[:, 1:] - np.array(loaded)[:, 1:])))

    

        