import _includes
from dependency_injector.wiring import inject, Provide
from pyspark.sql import SparkSession, DataFrame
from containers import ContainerFactory
from typing import List
from dataframe import DataFrameWriter
from pyspark.sql import functions as F
from pyspark.sql import types as T
from config import Config
from job_step import JobStep
from logging_factory import get_logger
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances
import kmedoids
from features import ORIGINAL_GENRE_FEATURES

logger = get_logger(__name__)

class PreprocessMoviesJobStep(JobStep):
    """
    Step 2: Preprocess movies
    - Clean and normalize genres
    - Derive year and optional flags
    - Compute rating_count, rating_avg from ratings
    - Compute genre partitions via weighted Jaccard + k-medoids in Pandas, then attach back

    Produces tables: movies_preprocessed, movies_shortlist

    Config:
    - UC_CATALOG, UC_SCHEMA
    - OVERWRITE: 'true' to overwrite
    - ENABLE_WEIGHTED_AVERAGE: 'false' | 'true'
    - ENABLE_EXTRA_MOVIE_FEATURES: 'false' | 'true'
    - GENRES_N_CLUSTERS: comma-separated integers, default '45,100'
    """

    def __init__(self, spark: SparkSession, config: Config, dataframe_writer: DataFrameWriter):
        super().__init__(spark, config, dataframe_writer)
        self._raw_movies_df: DataFrame | None = None
        self._raw_ratings_df: DataFrame | None = None
        self._movies_preprocessed_df: DataFrame | None = None
        self._movies_shortlist_df: DataFrame | None = None

    def _parse_clusters(self) -> List[int]:
        raw = self.config.string("GENRES_N_CLUSTERS", "45,100")
        try:
            return [int(x.strip()) for x in raw.split(',') if x.strip()]
        except Exception:
            return [45, 100]

    def load(self) -> None:
        self._raw_movies_df = self.spark.table("raw_movies")
        self._raw_ratings_df = self.spark.table("raw_ratings")

    def process(self) -> None:
        assert self._raw_movies_df is not None and self._raw_ratings_df is not None
        enable_extra_movie_features = self.config.bool("ENABLE_EXTRA_MOVIE_FEATURES", False)
        enable_weighted_average = self.config.bool("ENABLE_WEIGHTED_AVERAGE", False)
        genres_n_clusters = self._parse_clusters()

        # Clean genres into arrays, map 'Children'->'Kids', remove '(no genres listed)', 'IMAX', sort
        split_genres = F.split(F.col("genres"), "\\|")
        cleaned_genres = F.expr("filter(transform(genres_col, g -> case when g = 'Children' then 'Kids' else g end), g -> g not in ('(no genres listed)', 'IMAX'))")
        movies_df = (
            self._raw_movies_df
            .withColumn("genres_col", split_genres)
            .withColumn("genres", cleaned_genres)
            .drop("genres_col")
        )
        movies_df = movies_df.withColumn("genres", F.array_sort(F.col("genres")))
        movies_df = movies_df.where(F.size(F.col("genres")) > 0)

        # Extract year from title '(YYYY)'
        movies_df = movies_df.withColumn("year_str", F.regexp_extract(F.col("title"), "\\((\\d{4})\\)", 1))
        movies_df = movies_df.withColumn("year", F.when(F.col("year_str").isNull() | (F.trim(F.col("year_str")) == ""), F.lit(0)).otherwise(F.lit(1)).cast("int")).drop("year_str")

        if enable_extra_movie_features:
            movies_df = (
                movies_df
                .withColumn("specific_target_audience", (F.size("genres").between(1, 2)).cast("int"))
                .withColumn("broad_target_audience", (F.size("genres").between(3, 4)).cast("int"))
                .withColumn("broader_target_audience", (F.size("genres") > 4).cast("int"))
            )

        # rating_count, rating_avg from ratings
        # although these features are not used during training, it is convenient to have them for data interpretation purposes
        agg_df = (
            self._raw_ratings_df.groupBy("movieId")
            .agg(F.count(F.lit(1)).alias("rating_count"), F.avg("rating").alias("rating_avg"))
        )
        movies_df = movies_df.join(agg_df, on="movieId", how="left")

        # genre partitions via Pandas k-medoids using weighted Jaccard
        # Collect minimal columns to driver
        pdf = movies_df.select("movieId", "genres").toPandas()
        logger.info('Converting to Pandas dataframe to calculate medoids, shape = %s, features = %s', pdf.shape, pdf.columns.tolist())

        if not pdf.empty:
            # Binarize using ORIGINAL_GENRE_FEATURES; ensure mapping for 'Kids'
            mlb = MultiLabelBinarizer(classes=ORIGINAL_GENRE_FEATURES)
            one_hot = mlb.fit_transform(pdf["genres"])
            x = np.array(one_hot, dtype=bool)
            # weights aligned to classes
            # 0 'Action', 1 'Adventure', 2 'Animation', 3 'Comedy',
            # 4 'Crime', 5 'Documentary', 6 'Drama', 7 'Fantasy', 8 'Film-Noir', 9 'Horror',
            # 10 'Kids', 11 'Musical', 12 'Mystery', 13 'Romance', 14 'Sci-Fi', 15 'Thriller', 16 'War',
            # 17 'Western'
            weights = np.array([1, 1, 2, 1, 4, 3, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 4, 3])
            assert x.shape[1] == weights.shape[0]

            d = pairwise_distances(x, metric="jaccard", w=weights)
            # For each n_clusters create labels
            for i, k in enumerate(genres_n_clusters):
                if len(pdf) >= k and k > 1:
                    km = kmedoids.KMedoids(k, method='fasterpam', random_state=42)
                    labels = km.fit_predict(d)
                    logger.info('Detected medoids: %s',
                                [pdf.iloc[idx]['genres'] for idx in km.medoid_indices_])
                else:
                    labels = np.zeros(len(pdf), dtype=int)
                pdf[f"genre_partition{i}"] = labels.astype(np.int32)

        logger.info('Done calculating medoids based on Pandas dataframe, shape = %s, features = %s', pdf.shape, pdf.columns.tolist())

        # Convert labels back to Spark and join
        labels_cols = [c for c in pdf.columns if c.startswith("genre_partition")]
        if labels_cols:
            labels_pdf = pdf[["movieId"] + labels_cols]
            labels_sdf = self.spark.createDataFrame(labels_pdf)
            movies_df = movies_df.join(labels_sdf, on="movieId", how="left")
        else:
            # If no labels, set -1
            for i, _ in enumerate(genres_n_clusters):
                movies_df = movies_df.withColumn(f"genre_partition{i}", F.lit(-1))


        if enable_weighted_average:
            def get_weight(avg_rating: float) -> float:
                if avg_rating is None:
                    return 1.0
                if avg_rating < 2.5:
                    return 0.5
                elif avg_rating < 3:
                    return 0.75
                else:
                    return 1.0
            get_weight_udf = F.udf(get_weight, T.DoubleType())
            movies_df = movies_df.withColumn("weight", get_weight_udf(F.col("rating_avg")))

        # shortlist
        movies_shortlist_df = movies_df.where(F.col("rating_count") >= F.lit(10))
        self._movies_preprocessed_df = movies_df
        self._movies_shortlist_df = movies_shortlist_df

    def save(self) -> None:
        assert self._movies_preprocessed_df is not None and self._movies_shortlist_df is not None
        self.dataframe_writer.write(self._movies_preprocessed_df, "movies_preprocessed")
        self.dataframe_writer.write(self._movies_shortlist_df, "movies_shortlist")



@inject
def run(
    spark_session: SparkSession = Provide["spark_session"],
    config: Config = Provide["config"],
    dataframe_writer: DataFrameWriter = Provide["dataframe_writer"]
):
    step = PreprocessMoviesJobStep(spark=spark_session, config=config, dataframe_writer=dataframe_writer)
    step.load()
    step.process()
    step.save()

if __name__ == "__main__":
    container = ContainerFactory.create_container()
    container.wire(modules=[__name__])
    run()
