from model import Model, UserPreferences
from data_pipeline import MovieLensPipeline

# This module is just a facade for data preprocessing, model training and inference
# The module is used to generate C# bindings automatically using CSnake source code generator

model: Model = None
movie_train_data_df = user_train_data_df = y_df = users_with_features_df = all_movies_with_links_df = movies_dict = None
def preprocess(movielens_data_path: str) -> dict[int, dict[str, str | int]]:
    """
    Preprocesses movie and user training data, creating and returning a structured 
    dictionary of movies with corresponding information.

    This method runs the data pipeline, and saves the resulting data *in-memory*.

    :return: A structured dictionary containing processed movie data with titles, genres and links. The map is consumed later by C# code.
    """
    global movie_train_data_df, user_train_data_df, y_df, users_with_features_df, all_movies_with_links_df, movies_dict
    pipeline = MovieLensPipeline(movielens_data_path)
    movie_train_data_df, user_train_data_df, y_df, users_with_features_df, all_movies_with_links_df, _ = pipeline.run()

    movies_dict = {
        movieId: movie_row.to_dict() for movieId, movie_row in all_movies_with_links_df.iterrows()
    }

    all_movies_with_links_df.drop(
        columns=['title', 'year', 'genres', 'genre_partition0', 'genre_partition1', 'rating_count', 'rating_avg', 'weight', 'imdbId', 'tmdbId', 'origin_countries', 'release_date', 'description', 'budget', 'poster_uri'],
        errors='ignore',
        inplace=True)

    return movies_dict

def train(model_save_path: str) -> None:
    """
    Train Neural Network model on preprocessed data.

    :raises ValueError: Raised if `movie_train_data_df` is `None`, indicating 
        that data preprocessing has not been completed.
    """
    if movie_train_data_df is None: 
        raise ValueError("Data is not preprocessed yet")
    
    global model
    model = Model(movie_train_data_df, user_train_data_df, y_df, model_save_path, num_outputs=128)
    model.train()

def recommend(movie_ids: list[int] = None, preferences: dict[str, float] = None) -> list[tuple[str, float]]:
    """
    Recommend items based on the given user preferences.

    This function takes a dictionary of user preferences as input, converts 
    it into an appropriate format, and generates recommendations using a 
    pre-trained model. The recommendations are returned as a list of tuples,
    where each tuple consists of an item and its predicted rating.

    :param movie_ids: Limited set of movie IDs to recommend instead of full-blown dataset.
    :param preferences: A dictionary representing user preferences with item 
        keys and their respective preference scores/ratings as values.
    :return: A list of tuples, where each tuple contains an item (str) and 
        its predicted score (float).
    """
    if not is_trained() or all_movies_with_links_df is None:
        raise ValueError("Model is not trained yet")
    
    if __debug__:
        print("Predicting for preferences: ", preferences)

    preferences = UserPreferences(**preferences)
    if __debug__:
        print("All preferences: ", preferences.to_dict())

    if movie_ids:
        movie_ids = all_movies_with_links_df.index.intersection(movie_ids)
        predictions = model.predict(preferences, all_movies_with_links_df.loc[movie_ids])
    else:
        predictions = model.predict(preferences, all_movies_with_links_df)
    
    return [
        (movie_id, score, movies_dict[movie_id]['title'], movies_dict[movie_id]['genres'], movies_dict[movie_id]['tmdbId'])
        for movie_id, score in predictions
    ]

def is_trained() -> bool:
    """
    Determines whether the model has been trained.

    :return: A boolean indicating whether the model has been trained.
    """
    return model and model.is_trained()



if __name__ == '__main__':
    preprocess('./data/')
    train('./artifacts/model.keras')
    
    def recommend_internal(movie_ids: list[int] = None, **preferences: float) -> list[tuple[str, float]]:
        return recommend(movie_ids, preferences)
    
    print('\n\nfirst user (should be more action and adventure movies):')
    predictions = recommend_internal(action = 5, adventure = 3.5, mystery = 4, horror = 1, sci_fi = 4, western = 3, drama = 3, animation = 0.5 )
    for pred in predictions:
        print(pred)

    print('\n\nsecond user (should be more kids-oriented movies and cartoons):')
    predictions = recommend_internal(kids = 5, animation = 5, adventure = 4.5, comedy = 4.5, mystery = 2, crime = 1, horror = 0.5, sci_fi = 4)
    for pred in predictions:
        print(pred)

    print('\n\nthird user (should be more romance-oriented movies):')
    predictions = recommend_internal( comedy = 4.5, romance = 5, mystery = 2, crime = 0.5, horror = 0.5, sci_fi = 1.5)
    for pred in predictions:
        print(pred)

    print('\n\nfourth user (should be only kids-oriented movies and cartoons):')
    predictions = recommend_internal(kids = 5, animation = 5, adventure = 4.5)
    for pred in predictions:
        print(pred)


    print('\n\nfifth user (should be more action and sci-fi movies):')
    predictions = recommend_internal(action = 5,  sci_fi = 4.5 )
    for pred in predictions:
        print(pred)

    print('\n\nsixth user (should be more comedy and romance movies):')
    predictions = recommend_internal(comedy = 4.5,  romance = 4.5 )
    for pred in predictions:
        print(pred)
    
    #existing users
    pipeline = MovieLensPipeline("./data")
    all_movies_df, all_ratings_df, _ = pipeline._load_csvs()
    all_movies_df, _ = pipeline._preprocess_movies(all_movies_df, all_ratings_df)
    all_ratings_df = pipeline._get_ratings_with_movies_info(all_movies_df, all_ratings_df)
    existing_users_with_features_df = pipeline._get_users_with_features(all_ratings_df)
    all_ratings_df.set_index('userId', inplace=True)

    
    # check the recommendations for the users from the training data set – sanity check that the model is working
    for user_id, user_row in existing_users_with_features_df.iloc[0:3].iterrows():
        print(f"\n\nPredicting for existing user {user_id}:")
        user_dict = user_row.to_dict()
        #del user_dict['high_rating_count']
        #del user_dict['low_rating_count']
        user_dict = { feature.lower().replace('-', '_') : value for feature, value in user_dict.items() if feature != 'userId' }
        rated_movies_df = all_ratings_df.loc[user_id]
        rated_movies = rated_movies_df['movieId'].unique()
        
        
        predictions = recommend_internal(rated_movies, **user_dict)
        for pred in predictions:
            pred = (rated_movies_df[rated_movies_df['movieId'] == pred[0]]['rating'].item(), ) + pred
            print(pred)
        
    print("Done")