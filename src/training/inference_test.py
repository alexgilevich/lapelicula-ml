"""
Shared helper to test a trained movie-recommender model by running inference for a set of
example user preferences over the full catalog of gold movies.

The inference-test logic that used to live in `step04_test_inference.py::process` now lives
here so it can be reused:
- `step04_test_inference.py` loads a saved model and prints the recommendations to the console.
- `step03_train_model.py` tests the in-memory trained model and logs the output to MLFlow
  as artifacts (to facilitate tracking training experiments).
"""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from features import UserPreferences
from config import Config
from logging_factory import get_logger

logger = get_logger(__name__)

# Columns used to build the model's movie input. `movie_id` is the first column so that the
# predicted (movie_id, rating) pairs can be produced from the scored movie matrix.
MOVIE_COLUMNS = [
    'movie_id',
    "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust",
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Kids', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
    'description_embedding',
    'adult'
]

# Number of top recommendations to produce per user.
DEFAULT_INFERENCE_LIMIT = 300

# Schema used to turn raw (movie_id, rating) pairs back into a Spark DataFrame.
RECOMMENDATIONS_SCHEMA = StructType([
    StructField(name="movie_id", dataType=IntegerType(), nullable=True),
    StructField(name="rating", dataType=StringType(), nullable=True)
])

# The fixed set of example users the model is tested against.
DEFAULT_EXAMPLE_USER_PREFERENCES = [
    UserPreferences(kids=5, animation=5, adventure=0, comedy=0, mystery=0, crime=0, horror=0, sci_fi=0),
    UserPreferences(action=5, adventure=3.5, mystery=4, horror=1, sci_fi=4, western=3, drama=3, animation=0.5),
    UserPreferences(kids=5, animation=5, adventure=4.5, comedy=4.5, mystery=2, crime=1, horror=0.5, sci_fi=4),
    UserPreferences(comedy=4.5, romance=5, mystery=2, crime=0.5, horror=0.5, sci_fi=1.5),
    UserPreferences(kids=5, animation=5, adventure=4.5),
    UserPreferences(action=5, sci_fi=4.5),
    UserPreferences(comedy=4.5, romance=4.5),
    UserPreferences(war=5),
    UserPreferences(western=5),
    UserPreferences(kids=5),
    UserPreferences(horror=5),
    UserPreferences(thriller=5),
    UserPreferences(drama=5),
    UserPreferences(film_noir=5),
]


def prepare_inference_input(gold_movies_df: DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch all gold movies and their description embeddings, aligned row-by-row, in a form
    suitable for the model's `predict` method.

    :param gold_movies_df: The `gold_movie` Spark DataFrame.
    :param config: Configuration provider (used for the `DISABLE_EMBEDDINGS` diagnostic flag).
    :return: (movies_data, embeddings) where `movies_data` is a float32 array of the movie
        features (movie_id first, no embedding column) and `embeddings` is an array of the
        per-movie description embeddings with the same number of rows.
    """
    all_movies_with_embs_pdf = gold_movies_df.select(*MOVIE_COLUMNS).toPandas()
    all_movies_without_embs_pdf = all_movies_with_embs_pdf.drop(columns=['description_embedding'])
    movies_data = all_movies_without_embs_pdf.to_numpy(dtype=np.float32)
    embeddings = np.stack(all_movies_with_embs_pdf['description_embedding'].apply(lambda x: np.array(x)).values)

    # Diagnostic: with DISABLE_EMBEDDINGS=true the embedding path receives zeros, so it produces
    # a constant (bias-only) vector per movie genre -- no content signal. Recommendations then
    # depend only on genres/sentiments/adult, i.e. the same behavior as before embeddings existed.
    if config.bool("DISABLE_EMBEDDINGS", False):
        logger.info("DISABLE_EMBEDDINGS=true -- zeroing out all description embeddings for this diagnostic run")
        embeddings = np.zeros_like(embeddings)

    return movies_data, embeddings


def run_inference_test(model, movies_data: np.ndarray, embeddings: np.ndarray,
                       inference_params: dict, examples: list[UserPreferences] | None = None) -> list[dict]:
    """
    Run the model for each example user preference and collect the recommendations.

    :param model: A model exposing `predict(input_df, params)` (either the in-memory `Model`
        or a `mlflow.pyfunc.load_model` wrapper).
    :param movies_data: Movie features array (see `prepare_inference_input`).
    :param embeddings: Movie description embeddings array (see `prepare_inference_input`).
    :param inference_params: Parameters passed to the model, e.g. `{"limit": 3000}`.
    :param examples: Example user preferences to test; defaults to `DEFAULT_EXAMPLE_USER_PREFERENCES`.
    :return: A list of result dicts, one per example user:
        `{"user_index": int, "preferences": dict, "recommendations": list[(movie_id, rating)]}`,
        where `recommendations` is sorted by rating (highest first).
    """
    if examples is None:
        examples = DEFAULT_EXAMPLE_USER_PREFERENCES

    results = []
    for i, example in enumerate(examples):
        sorted_dict = {k: v for k, v in sorted(example.to_dict().items(), key=lambda item: -item[1])}
        input_df = pd.DataFrame([{
            "user_preferences": sorted_dict,
            "movies": movies_data,
            "embeddings": embeddings
        }])
        predictions = model.predict(input_df, inference_params)
        recommendations = [(int(movie_id), float(rating)) for movie_id, rating in predictions[0]]

        results.append({
            "user_index": i + 1,
            "preferences": sorted_dict,
            "recommendations": recommendations,
        })

    return results


def build_recommendations_df(spark: SparkSession, gold_movies_df: DataFrame,
                             recommendations: list[tuple[int, float]]) -> DataFrame:
    """
    Turn a list of (movie_id, rating) pairs into a ranked, enriched Spark DataFrame by joining
    with the gold movies. Shared so that the console output (step04) and the MLFlow artifacts
    (step03) are built from the exact same logic.
    """
    recommendations_df = spark.createDataFrame(recommendations, schema=RECOMMENDATIONS_SCHEMA)
    return recommendations_df \
        .join(gold_movies_df, 'movie_id', how='inner') \
        .withColumn('rn', F.expr('row_number() over(order by rating desc)')) \
        .select('movie_id', 'rating', 'title', 'genres', 'year', 'rating_count', 'top_sentiments', 'vote_average')
