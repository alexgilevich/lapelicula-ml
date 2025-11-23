from model import Model, UserPreferences
from data_pipeline import MovieLensPipeline


pipeline = MovieLensPipeline(movielens_data_path)
movie_train_data_df, user_train_data_df, y_df, users_with_features_df, all_movies_with_links_df, _ = pipeline.run()

movies_dict = {
    movieId: movie_row.to_dict() for movieId, movie_row in all_movies_with_links_df.iterrows()
}

all_movies_with_links_df.drop(
    columns=['title', 'year', 'genres', 'genre_partition0', 'genre_partition1', 'rating_count', 'rating_avg', 'weight',
             'imdbId', 'tmdbId', 'origin_countries', 'release_date', 'description', 'budget', 'poster_uri'],
    errors='ignore',
    inplace=True)