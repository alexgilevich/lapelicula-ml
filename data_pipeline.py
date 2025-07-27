import datetime
import itertools
import logging
import os
import ast
import random
from concurrent.futures import ThreadPoolExecutor
from os import path
import pandas as pd
import sklearn.utils
from pandas.core.array_algos import quantile
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jaccard
import kmedoids
from features import COMBINED_GENRE_FEATURES
from tmdb import TMDBClient, TMDBMovie

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MovieLensPipeline:
    """
    Handles the complete processing pipeline for the MovieLens 100K dataset.

    The class is designed to load, preprocess, and transform data from the MovieLens 100K
    dataset into formats suitable for training recommendation models and conducting
    analysis. This involves steps such as loading CSV data, encoding genres, generating
    user and movie feature vectors, and preparing training data.

    :ivar csv_data_path: Path to the folder where MovieLens 100K dataset CSV files are
        stored.
    :type csv_data_path: str
    """

    def __init__(self, csv_data_path, enable_weighted_average = False, enable_combined_genres = False, enable_extra_user_features = False, enable_extra_movie_features = False, enable_oversampling_by_rating = False):
        self.csv_data_path = csv_data_path
        self.enable_weighted_average = enable_weighted_average
        self.enable_combined_genres = enable_combined_genres
        self.enable_extra_movie_features = enable_extra_movie_features
        self.enable_extra_user_features = enable_extra_user_features
        self.enable_oversampling_by_rating = enable_oversampling_by_rating
        self.genres_n_clusters = [45, 100]


    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # dict[int, dict[str, str | int]]]:
        """
        Executes the entire data pipeline for MovieLens 100K dataset.

        :return: 
            A tuple containing multiple data structures:

            - movie_train_data_df (pd.DataFrame): Training data with movies.
            - user_train_data_df (pd.DataFrame): Training data with users.
            - y_df (pd.DataFrame): Y labels.
            - users_with_features_df (pd.DataFrame): User-related features (preferences).
            - all_movies_df (pd.DataFrame): All movies with links and other related metadata.
            - movies_dict (dict[int, dict[str, str | int]]): Dictionary representation of movies dataframe above plus links.
        """
        all_movies_df, all_ratings_df, all_links_df = self._load_csvs()

        random.seed(42)

        # split genres into list and filter out irrelevant movies
        all_movies_df, shortened_movie_list_df = self._preprocess_movies(all_movies_df, all_ratings_df)

        # join movies and ratings dataframes
        all_ratings_df = self._get_ratings_with_movies_info(shortened_movie_list_df, all_ratings_df)

        all_ratings_df = self._generate_synthetic_users(all_ratings_df)

        users_with_features_df = self._get_users_with_features(all_ratings_df)

        # oversample ratings
        all_ratings_df = self._oversample_ratings(all_ratings_df)


        # noinspection PyShadowingNames
        movie_train_data_df, user_train_data_df, y_df = self._get_training_data(all_movies_df, all_ratings_df, users_with_features_df)
        # we are doing it here to avoid column conflicts with user_with_features_df
        all_movies_df = self._get_all_movies_with_onehot_genres(all_movies_df)

        # we need to generate movies dict later on to pass back to C#
        all_movies_df = all_movies_df.join(all_links_df, how="left", lsuffix='_movies', rsuffix='_links')
        all_movies_df = self.load_additional_movie_attributes(all_movies_df)
        return movie_train_data_df, user_train_data_df, y_df, users_with_features_df, all_movies_df, all_ratings_df

    def _get_training_data(self, all_movies_df: pd.DataFrame, all_ratings_df: pd.DataFrame, users_with_features_df: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Retrieves and preprocesses training data necessary for neural network training 

        :param all_movies_df: DataFrame containing information about movies. Expected fields 
            include movie-related attributes (e.g., movieId, genres, etc.).
        :param all_ratings_df: DataFrame containing user ratings for movies as provided by MovieLens.
        :param users_with_features_df: DataFrame containing user-related features that are 
            used as input data for the users' neural network.
        :return: A tuple of DataFrames containing: (1) training data for the movies' network 
            after including encoded genres, (2) training data for the users' network, and (3) 
            the Y labels used as the output labels.
        """

        all_movies_df = all_movies_df.drop(columns=['genre_partition' + str(i) for i in range(len(self.genres_n_clusters))] + ['rating_count', 'rating_avg', 'weight', 'year'], errors='ignore')

        # We need to keep the same number of rows in all training data dataframes as they are later used in dot product in Neural Network. 
        # Therefore, we join and then split again.
        all_training_data_df = self._get_all_ratings_with_movies_and_users(all_ratings_df, users_with_features_df)

        # Training data for users' network
        final_user_train_data_df = all_training_data_df[users_with_features_df.columns.tolist()]

        # Training data for movies' network
        final_movie_train_data_df = all_training_data_df[all_movies_df.columns.tolist()]
        one_hot_df = self._encode_movie_genres(all_training_data_df['genres'])
        final_movie_train_data_df = final_movie_train_data_df.drop(columns=['genres'])
        final_movie_train_data_df = pd.concat([final_movie_train_data_df, one_hot_df], axis=1)

        # Y
        y_df = all_training_data_df[['rating']]


        return final_movie_train_data_df, final_user_train_data_df, y_df

    def _get_ratings_with_movies_info(self, all_movies_df: pd.DataFrame, all_ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get ratings information combined with movies information.
        
        :param all_movies_df: DataFrame containing movie information.
        :param all_ratings_df: DataFrame containing ratings information.
        :return: A DataFrame resulting from the inner join of the movies and ratings DataFrames, 
                 merging on the 'movieId' key.
        """
        # Get ratings with movies info 
        return all_ratings_df.join(all_movies_df, on="movieId", how="inner", lsuffix='_ratings',
                                   rsuffix='_movies').drop(columns=['movieId_ratings', 'movieId_movies'], errors='ignore')

    def _get_all_movies_with_onehot_genres(self, all_movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes DataFrame of movies to include one-hot encoded genre information. Calls _encode_movie_genres() under the hood.

        :param all_movies_df: A pandas DataFrame containing movie data with at least a 'genres' column, where genres are 
                              expected to be represented in a raw, unprocessed format (e.g., as strings or lists).
        :return: A pandas DataFrame which combines the original movie data with one-hot encoded genre information, with the
                 original 'genres' column removed.
        """
        return pd.concat([all_movies_df, self._encode_movie_genres(all_movies_df['genres'])], axis=1)

    def _encode_movie_genres(self, genres_series: pd.Series) -> pd.DataFrame:
        """
        Encodes a series of movie genres into a one-hot encoded DataFrame. Each genre is represented 
        as a binary feature in the resulting DataFrame.

        :param genres_series: A pandas Series where each row is a list of genres associated 
            with a movie.
        :return: A pandas DataFrame where rows correspond to the input movies, and columns are one-hot 
            encoded representation of genres. Each cell has a binary value indicating whether a movie 
            has a particular genre or not.
        """
        mlb = MultiLabelBinarizer()
        one_hot_matrix = mlb.fit_transform(genres_series)
        one_hot_df = pd.DataFrame(one_hot_matrix, columns=mlb.classes_, index=genres_series.index)

        if self.enable_combined_genres:
            def build_combined_genre_feature(genres_in_feature: set[str]) -> (str, pd.Series):
                return "_".join(sorted(genres_in_feature)), genres_series.apply(lambda genres: genres_in_feature.issubset(set(genres))).astype(int)

            for combined_genre in COMBINED_GENRE_FEATURES:
                feature_name, series = build_combined_genre_feature(combined_genre)
                one_hot_df[feature_name] = series

        return one_hot_df

    def _get_all_ratings_with_movies_and_users(self, all_ratings_df, users_with_features_df):
        return all_ratings_df.join(users_with_features_df, on="userId", how="inner", lsuffix='_ratings',
                                   rsuffix='_users')

    def _get_users_with_features(self, all_ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and returns a DataFrame containing user feature vectors based on ratings by genres.

        :param all_ratings_df: DataFrame containing user ratings and genres information (as an array of strings). 
                               Expected to have columns 'userId', 'genres' (array of strings), and 'rating'.
        :return: DataFrame where the rows represent users and columns represent genres,
                 with each value depicting the mean rating a user has given to items 
                 of the corresponding genre.
        """


        all_ratings_df = all_ratings_df.copy()

        if self.enable_combined_genres:
            def generate_combined_genres(genres: list[str]) -> list[str] :
                genres = set(genres)
                for combined_genre in COMBINED_GENRE_FEATURES:
                    if combined_genre.issubset(genres):
                        genres.add('_'.join(sorted(combined_genre)))
                return genres
            all_ratings_df['genres'] = all_ratings_df['genres'].apply(generate_combined_genres)


        import datetime
        all_ratings_df['year'] = all_ratings_df['timestamp'].apply(lambda ts: datetime.datetime.fromtimestamp(ts).year)

        def weighted_average_rating(group: pd.DataFrame) -> pd.Series:
            d = group['rating']
            w = group['weight']
            weighting_rating: pd.Series = (d * w).sum() / w.sum()
            return weighting_rating

        average_rating_per_genre_per_user: pd.DataFrame = \
            (all_ratings_df
             .explode('genres').reset_index(drop=True)
             .groupby(['userId', 'genres']))

        if self.enable_weighted_average:
            average_rating_per_genre_per_user = average_rating_per_genre_per_user.apply(weighted_average_rating).rename('rating').reset_index()
        else:
            average_rating_per_genre_per_user = average_rating_per_genre_per_user['rating'].agg('mean').rename('rating').reset_index()


        users_with_features_df: pd.DataFrame = \
            (average_rating_per_genre_per_user
             .pivot(index='userId', columns='genres', values='rating')
             .reset_index()
             .set_index('userId', drop=False)
             .fillna(0.0))

        if self.enable_extra_user_features:
            high_rating_count = average_rating_per_genre_per_user[
                average_rating_per_genre_per_user['rating'] >= 4.5].groupby('userId').size()
            low_rating_count = average_rating_per_genre_per_user[
                average_rating_per_genre_per_user['rating'] <= 2].groupby('userId').size()
            users_with_features_df['high_rating_count'] = users_with_features_df['userId'].map(
                high_rating_count).fillna(0).astype(int)
            users_with_features_df['low_rating_count'] = users_with_features_df['userId'].map(low_rating_count).fillna(
                0).astype(int)
            users_with_features_df['few_genres_lover'] = users_with_features_df['userId'].map(high_rating_count).fillna(
                0).astype(int)


        logger.info('Preprocessed users, shape = %s, features = %s', users_with_features_df.shape, users_with_features_df.columns.tolist())
        for user_id, row in itertools.islice(users_with_features_df.iterrows(), 10):
            logger.info('User #%d features are: %s', user_id, row.to_dict())

        return users_with_features_df

    def _preprocess_movies(self, all_movies_df: pd.DataFrame, all_ratings_df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the movies data by splitting the 'genres' column string into a list of genres.

        :param all_movies_df: A pandas DataFrame containing movie data, where the 'genres' column is a string
            of genre names separated by the "|" character.
        :return: A pandas DataFrame with the 'genres' column converted to a list of genres.
        """

        movie_to_ratings_df = all_ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])


        all_movies_df['genres'] = all_movies_df['genres'].str.split('|')
        all_movies_df['genres'] = all_movies_df['genres'].apply(lambda genres: [genre if genre != 'Children' else 'Kids' for genre in genres if genre not in ['(no genres listed)', 'IMAX']])
        all_movies_df: pd.DataFrame = all_movies_df[all_movies_df['genres'].apply(lambda x: len(x) > 0)].copy()
        all_movies_df['genres'] = all_movies_df['genres'].apply(lambda genres: list(sorted(genres)))
        all_movies_df['year'] = pd.to_numeric(all_movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)).fillna(0).astype(int)
        if self.enable_extra_movie_features:
            all_movies_df['specific_target_audience'] = all_movies_df['genres'].apply(lambda x: 1 <= len(x) <= 2).astype(int)
            all_movies_df['broad_target_audience'] = all_movies_df['genres'].apply(lambda x: 3 <= len(x) <= 4).astype(int)
            all_movies_df['broader_target_audience'] = all_movies_df['genres'].apply(lambda x: len(x) > 4).astype(int)

        mlb = MultiLabelBinarizer()
        one_hot_genre_matrix = mlb.fit_transform(all_movies_df['genres'])
        genres_encoded = one_hot_genre_matrix.tolist()

        x_stacked = np.vstack(genres_encoded, dtype=np.dtypes.BoolDType(), casting='unsafe')

        # genre weights
        # 0 'Action', 1 'Adventure', 2 'Animation', 3 'Comedy',
        # 4 'Crime', 5 'Documentary', 6 'Drama', 7 'Fantasy', 8 'Film-Noir', 9 'Horror',
        # 10 'Kids', 11 'Musical', 12 'Mystery', 13 'Romance', 14 'Sci-Fi', 15 'Thriller', 16 'War',
        # 17 'Western'
        weights = np.array([1, 1, 2, 1, 4, 3, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 4, 3])
        assert x_stacked.shape[1] == weights.shape[0]


        jaccard_dist = pairwise_distances(x_stacked, w=weights, metric="jaccard")

        for i, n_clusters in enumerate(self.genres_n_clusters):
            km = kmedoids.KMedoids(n_clusters, method='fasterpam', random_state=42)
            c = km.fit_predict(jaccard_dist)
            logger.info('Detected medoids: %s', [all_movies_df.iloc[idx]['genres'] for idx in km.medoid_indices_])
            col_name = 'genre_partition' + str(i)
            all_movies_df[col_name] = c.astype(np.dtypes.Int64DType())
            all_movies_df[col_name] = all_movies_df[col_name].fillna(0)


        logger.info('Preprocessed movies, shape = %s, features = %s', all_movies_df.shape, all_movies_df.columns.tolist())

        # although these feature are not used during training, it is convenient to have them for data interpretation purposes
        all_movies_df['rating_count'] = movie_to_ratings_df['count']
        all_movies_df['rating_avg'] = movie_to_ratings_df['mean']


        if self.enable_weighted_average:
            def get_weight(avg_rating: int) -> float:
                if avg_rating < 2.5:
                    return 0.5
                elif avg_rating < 3:
                    return 0.75
                else:
                    return 1.0

            all_movies_df['weight'] = 1.0
            all_movies_df['weight'] = all_movies_df['rating_avg'].apply(get_weight)

        shortened_movies_list_df = all_movies_df[all_movies_df['rating_count'] >= 10]

        return all_movies_df, shortened_movies_list_df




    def _load_csvs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads data from CSV files corresponding to movies, ratings, and links
        available in the MovieLens 100K dataset. The method reads the respective CSV
        files by specifying column data types explicitly to ensure data consistency
        during the loading process.

        :raises FileNotFoundError: If any of the CSV files cannot be found.
        :raises pd.errors.EmptyDataError: If any of the CSV files are empty.
        :raises pd.errors.ParserError: If there's an error in parsing the CSV files.
        :raises ValueError: If the specified CSV columns contain data incompatible
           with the given dtypes.

        :return: A tuple containing three pandas DataFrames - one for movies, one for
           ratings, and one for links - loaded from their respective CSV files.
        """


        # Load all available raw data first (MovieLens 100K)

        dtypes = {
            'movieId': pd.Int64Dtype(),
            'title': pd.StringDtype(),
            'genres': pd.StringDtype()
        }
        all_movies_df = pd.read_csv(f'{self.csv_data_path}/movies.csv', dtype=dtypes, index_col='movieId')
        logger.info('Loaded movies.csv, shape = %s', all_movies_df.shape)

        dtypes = {
            'userId': pd.Int64Dtype(),
            'movieId': pd.Int64Dtype(),
            'rating': pd.Float64Dtype(),
            'timestamp': pd.Int64Dtype()
        }
        all_ratings_df = pd.read_csv(f'{self.csv_data_path}/ratings.csv', dtype=dtypes)
        logger.info('Loaded ratings.csv, shape = %s', all_ratings_df.shape)

        dtypes = {
            'movieId': pd.Int64Dtype(),
            'imdbId': pd.Int64Dtype(),
            'tmdbId': pd.Int64Dtype(),
        }
        all_links_df = pd.read_csv(f'{self.csv_data_path}/links.csv', dtype=dtypes, index_col='movieId')#.set_index('movieId', drop=True)
        logger.info('Loaded links.csv, shape = %s', all_links_df.shape)

        return all_movies_df, all_ratings_df, all_links_df

    def _oversample_ratings(self, all_ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Oversamples the ratings DataFrame to ensure that there are enough ratings for meaningful predictions. The basic version oversamples genres with fewer ratings.
        Also, if `self.enable_oversampling_by_rating` = True, the algorithm tries to balance out samples for different buckets of ratings with the emphasis on the worst and the best ratings.
        Approximate target – 600k ratings (approximately 60 ratings per 1 movie, although that is not guaranteed by the algorithm).
        The algorithm is far from ideal. Ideally, we need to look more into various genres and balance based on them as well.
        """
        logger.info('Oversampling ratings... Current shape = %s', all_ratings_df.shape)
        all_ratings_exploded = all_ratings_df.copy()
        all_ratings_exploded['genre_exploded'] = all_ratings_exploded['genres']
        all_ratings_exploded = all_ratings_exploded.explode('genre_exploded')
        grouped_by_genre = all_ratings_exploded.groupby(['genre_exploded'])['rating'].count()

        desired_min_per_genre = int(grouped_by_genre.quantile(0.75))
        genres_with_few_ratings = grouped_by_genre[grouped_by_genre < desired_min_per_genre].index.tolist()

        resampled_ratings_df = all_ratings_df
        for genre in genres_with_few_ratings:
            logger.info('Oversampling low-represented genre %s, last known row number = %d ratings', genre, grouped_by_genre[genre])
            ratings_to_sample_df = resampled_ratings_df[resampled_ratings_df['genres'].apply(lambda genres: genre in genres) == True]
            target_sample_amount = desired_min_per_genre - grouped_by_genre[genre]
            additional_sampled_ratings_df = sklearn.utils.resample(ratings_to_sample_df, replace=True, n_samples=target_sample_amount, random_state=42)
            logger.info('Adding new %d ratings for genre %s', len(additional_sampled_ratings_df), genre)
            resampled_ratings_df = pd.concat([resampled_ratings_df, additional_sampled_ratings_df])

        logger.info('Completed oversampling ratings by genre... Current ratings shape = %s', resampled_ratings_df.shape)

        # The following code oversampling by rating
        if self.enable_oversampling_by_rating:

            logger.info('Still oversampling ratings... Current ratings shape = %s', resampled_ratings_df.shape)

            bins = [ 0, 2.5, 4.5, 5 ]
            labels = [ 1, 2, 3 ]
            resampled_ratings_df['rating_bin'] = pd.cut(
                resampled_ratings_df['rating'],
                bins=bins,
                labels=labels,
                right=True
            )

            grouped_by_rating = resampled_ratings_df.groupby(['rating_bin'])['rating'].count()

            target_per_bin = { 1: grouped_by_rating[1], 2: grouped_by_rating[2], 3: max(grouped_by_rating[3], int(grouped_by_rating[2] * 0.8), grouped_by_rating[1])}
            for idx, label in enumerate(labels):
                logger.info('Oversampling bin %.1f-%.1f, current size = %d ratings', bins[idx], bins[idx + 1], grouped_by_rating[label])
                ratings_to_sample_df = resampled_ratings_df[resampled_ratings_df['rating_bin'] == label]
                target_sample_amount = target_per_bin[label] - grouped_by_rating[label]
                if target_sample_amount <= 0:
                    logger.info('Nothing to oversample for bin %.1f-%.1f', bins[idx], bins[idx + 1])
                    continue
                additional_sampled_ratings_df = sklearn.utils.resample(ratings_to_sample_df, replace=True, n_samples=target_sample_amount, random_state=42)
                logger.info('Adding new %d ratings for bin %.1f-%.1f', len(additional_sampled_ratings_df), bins[idx], bins[idx + 1])
                resampled_ratings_df = pd.concat([resampled_ratings_df, additional_sampled_ratings_df])

            logger.info('Completed oversampling ratings... New ratings shape = %s', resampled_ratings_df.shape)

            resampled_ratings_df.drop(columns=['rating_bin'], inplace=True)
            resampled_ratings_df.reset_index(drop=True, inplace=True)

        return resampled_ratings_df

    def _generate_synthetic_users(self, all_ratings_df: pd.DataFrame) -> pd.DataFrame:
        # the method should group similar genres together
        final_resampled_df = None
        for i in range(len(self.genres_n_clusters)):
            logger.info('Generating synthetic users... Current ratings shape = %s', all_ratings_df.shape if final_resampled_df is None else final_resampled_df.shape)

            partition_column_name = 'genre_partition' + str(i)
            resampled_ratings_with_partitions_df = all_ratings_df[all_ratings_df[partition_column_name] > -1]

            user_to_partition_count_df: pd.DataFrame = resampled_ratings_with_partitions_df.groupby(by=['userId'])[partition_column_name].nunique()

            users_with_partitions: list[int] = user_to_partition_count_df[user_to_partition_count_df > 1].keys().tolist()
            resampled_ratings_with_partitions_df: pd.DataFrame = resampled_ratings_with_partitions_df.loc[
                resampled_ratings_with_partitions_df['userId'].isin(users_with_partitions)]

            resampled_ratings_with_partitions_df['userId'] = 100_000_000 * (i + 1) + resampled_ratings_with_partitions_df[
                'userId'] * 1000 + resampled_ratings_with_partitions_df[partition_column_name]

            if final_resampled_df is None:
                final_resampled_df = resampled_ratings_with_partitions_df
            else:
                final_resampled_df = pd.concat([final_resampled_df, resampled_ratings_with_partitions_df], ignore_index=True)

            logger.info('%d synthetic users were added... Current ratings shape = %s', user_to_partition_count_df.sum(), final_resampled_df.shape)

        return final_resampled_df

    def load_additional_movie_attributes(self, all_movies_df: pd.DataFrame) -> pd.DataFrame:
        additional_movie_attributes_file_path = path.join(self.csv_data_path, 'additional_movie_attributes.csv')

        dtypes = {
            'id': pd.Int64Dtype(),
            'title': pd.StringDtype(),
            'poster_uri': pd.StringDtype(),
            'budget': pd.Float64Dtype(),
            'description': pd.StringDtype(),
            'release_date': pd.StringDtype(),
            'origin_countries': pd.StringDtype()
        }

        if not os.path.exists(additional_movie_attributes_file_path):
            all_movies_df: pd.DataFrame = all_movies_df
            total = all_movies_df.shape[0]
            processed = 0
            client = TMDBClient()


            def get_tmdb_info(movie_id: int) -> dict[str, str | float | int | datetime.date]:
                nonlocal processed
                if movie_id is None:
                    logger.warning('Error %d while requesting movie #%d', err.response.status_code, movie_id)
                    return {}
                    
                import requests
                try:
                    res = client.get_movie_by_id(movie_id).to_dict()
                    processed += 1
                    if processed % 100 == 0:
                        logger.info('Processed %d out of %d movies', processed, total)
                    return res
                except requests.exceptions.HTTPError as err:
                    logger.warning('Error %d while requesting movie #%s', err.response.status_code, movie_id)
                    return {}
                except BaseException as err:
                    logger.error('General error while requesting movie #%s: %s', movie_id, str(err))
                    return {}

            keys = all_movies_df['tmdbId'].tolist()

            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(get_tmdb_info, keys))

            tmdb_attributes_df = pd.DataFrame(results)
            tmdb_attributes_df = tmdb_attributes_df[~tmdb_attributes_df['id'].isna()]
            tmdb_attributes_df.to_csv(additional_movie_attributes_file_path, index=False)

        tmdb_attributes_df = pd.read_csv(additional_movie_attributes_file_path, dtype=dtypes, index_col='id')
        tmdb_attributes_df["origin_countries"] = tmdb_attributes_df["origin_countries"].apply(ast.literal_eval)
        
        all_movies_df = (all_movies_df
                         .reset_index(drop=False)
                         .drop(columns=['title'])
                         .merge(tmdb_attributes_df, how='inner', left_on='tmdbId', right_on='id', right_index=True)
                         .set_index('movieId'))

        return all_movies_df




if __name__ == "__main__":
    from tabulate import tabulate
    pipeline = MovieLensPipeline('./data')
    movie_train_data_df, user_train_data_df, y_df, users_with_features_df, all_movies_with_links_df, all_ratings_df = pipeline.run()

    print(tabulate(movie_train_data_df.iloc[:100], headers="keys", tablefmt="pretty"))
    print(tabulate(user_train_data_df.iloc[:250], headers="keys", tablefmt="pretty"))
    print(tabulate(y_df.iloc[:200], headers="keys", tablefmt="pretty"))
    print(tabulate(users_with_features_df.iloc[:100], headers="keys", tablefmt="pretty"))
    print(tabulate(all_movies_with_links_df.iloc[:100], headers="keys", tablefmt="pretty"))

    print('=== Export ===')

    all_movies_with_links_df['genres'] = all_movies_with_links_df['genres'].apply(lambda genres: ','.join(genres))
    all_movies_with_links_df['origin_countries'] = all_movies_with_links_df['origin_countries'].apply(lambda origin_countries: ','.join(origin_countries))
    all_movies_with_links_df.sort_values(by='rating_count', ascending=False).iloc[0:200].to_csv('./data/top200_movies.csv')
    all_movies_with_links_df.to_csv('./data/all_movies.csv')
    users_with_features_df.to_csv('./data/user_with_features.csv')
    all_ratings_df.to_csv('./data/ratings_with_movies.csv')
