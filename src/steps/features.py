import math

ORIGINAL_GENRE_FEATURES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Kids', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

COMBINED_GENRE_FEATURES = [
#     { 'Action', 'Sci-Fi' },
#     { 'Action', 'Crime' },
#     { 'Action', 'Drama' },
#     { 'Action', 'Horror' },
#     { 'Action', 'Comedy' },
#     { 'Romance', 'Western' },
#     { 'Comedy', 'Romance' },
#     { 'Drama', 'Romance'},
#     { 'Drama', 'Horror'},
#     { 'Adventure', 'Sci-Fi' },
#     { 'Adventure', 'Kids' },
#     { 'Adventure', 'Fantasy' },
#     { 'Adventure', 'Animation' },
#     { 'Kids', 'Fantasy' },
#     { 'Animation', 'Kids'  },
#     { 'Adventure', 'Kids'},
#     { 'Drama', 'War' },
#     { 'Crime', 'Drama' },
#     { 'Horror', 'Thriller' },
#     { 'Comedy', 'Horror' },
#     { 'Horror', 'Sci-Fi' },
#     { 'Horror', 'Fantasy' },
#     { 'Horror', 'Mystery' },
#     { 'Animation', 'Comedy' },
]

class UserPreferences:
    GENRES = [genre_name.lower().replace('-', '_') for genre_name in ORIGINAL_GENRE_FEATURES]

    def __init__(self, action = 0, adventure = 0, animation = 0,
                 comedy = 0, crime = 0, documentary = 0, drama = 0, fantasy = 0, film_noir = 0, horror = 0, kids = 0, musical = 0, mystery = 0,
                 romance = 0, sci_fi = 0, thriller = 0, war = 0, western = 0, **kwargs):
        for genre in self.GENRES:
            setattr(self, genre, float(locals().get(genre)))
            
        for simple_genres in COMBINED_GENRE_FEATURES:
            combined_name = self.get_combined_genre_name(simple_genres)
            if combined_name not in locals(): 
                continue
            setattr(self, self.get_combined_genre_name(combined_name), float(locals().get(combined_name)))
        
        unexpected_genres = set(self.__dict__.keys()) - set(locals().keys())
        if len(unexpected_genres) > 0:
            raise ValueError(f"Unexpected genres: {unexpected_genres}")

    def get_normalized_genre_name(genre_name: str) -> str:
        return genre_name.lower().replace('-', '_')
    
    def get_genre_value(self, genre_name: str) -> float:
        genre_name = genre_name.lower().replace('-', '_')
        return -math.inf if genre_name not in self.__dict__ or self.__dict__[genre_name] == 0 else self.__dict__[genre_name]

    def get_combined_genre_name(self, simple_genres: set[str]):
        combined_genre_name = '_'.join(
            sorted(UserPreferences.get_normalized_genre_name(genre_name) for genre_name in simple_genres))
        return combined_genre_name

    def to_list(self):
        def build_combined_genres_dict():
            combined_genres_dict = {}
            for COMBINED_GENRE in COMBINED_GENRE_FEATURES:
                combined_genre_name = self.get_combined_genre_name(COMBINED_GENRE)
                combined_genre_value = self.get_genre_value(combined_genre_name)
                if math.isinf(combined_genre_value):
                    combined_genre_value = round(
                        sum([
                            self.get_genre_value(genre_name)
                            for genre_name in COMBINED_GENRE
                        ]) / len(COMBINED_GENRE), 2
                    )
                
                combined_genres_dict[combined_genre_name] = combined_genre_value if not math.isinf(combined_genre_value) else 0

            return combined_genres_dict

        

        all_genres_dict = self.__dict__ | build_combined_genres_dict()

            
        genres = [
            float(all_genres_dict[genre])
            for genre in sorted(all_genres_dict.keys())
        ]

        high_rating_count, low_rating_count = 0, 0
        for g in genres:
            if g >= 4.5:
                high_rating_count += 1
            elif g < 2:
                low_rating_count += 1
        return genres# + [high_rating_count, low_rating_count]

    def to_dict(self):
        return self.__dict__.copy()