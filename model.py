import pandas as pd
import tensorflow as tf
from keras import Layer
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
pd.set_option("display.precision", 1)
import logging
import numpy as np
import os
from features import UserPreferences

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@tf.keras.utils.register_keras_serializable()
class L2Norm(Layer):
    def __init__(self, axis=1, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis, "epsilon": self.epsilon})
        return cfg
    
class Model:
    def __init__(self, movie_train_data_df, user_train_data_df, y_df, model_save_path, num_outputs = 32):
        self.y_df = y_df
        self.user_train_data_df: pd.DataFrame = user_train_data_df
        self.movie_train_data_df: pd.DataFrame = movie_train_data_df
        self.target_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        self._tf_model = None
        self.model_save_path = model_save_path
        self.num_outputs = num_outputs
        
        
    def train(self) -> tf.keras.Model:
        
        logger.info("User training data shape: %s, columns: %s", self.user_train_data_df.shape, self.user_train_data_df.columns)
        logger.info("Movie training data shape: %s, columns: %s", self.movie_train_data_df.shape, self.movie_train_data_df.columns)
        logger.info("Label training data shape: %s, columns: %s", self.y_df.shape, self.y_df.columns)

        # drop non-relevant columns
        movie_train_data_df = self.movie_train_data_df.drop(columns=['movieId', 'title'], errors="ignore").to_numpy()
        user_train_data_df = self.user_train_data_df.drop(columns=['userId'], errors="ignore").to_numpy()
        
        self.user_scaler.fit(user_train_data_df)
        user_train_data_df = self.user_scaler.transform(user_train_data_df)
        
        y_df = self.y_df.to_numpy().reshape(-1, 1)
        self.target_scaler.fit(y_df)
        y_df = self.target_scaler.transform(y_df)


        tf.random.set_seed(42)
        
        movie_train_df, movie_test_df = train_test_split(movie_train_data_df, train_size=0.9, shuffle=True, random_state=42)
        user_train_df, user_test_df   = train_test_split(user_train_data_df,  train_size=0.9, shuffle=True, random_state=42)
        y_train_df, y_test_df         = train_test_split(y_df,                train_size=0.9, shuffle=True, random_state=42)

        if os.path.exists(self.model_save_path):
            logger.warning("Found model in path `%s`. No training will be performed. Loading existing model instead...", self.model_save_path)
            self._tf_model = tf.keras.models.load_model(self.model_save_path, safe_mode=False)
            metrics = self._tf_model.evaluate([user_test_df, movie_test_df], y_test_df, return_dict=True)
            logger.info("Evaluation phase. Here are your metrics: %s", metrics)
            return
        else:
            logger.info("No model found in path `%s`. Training will be performed. Saving model to this path after training completes...",
                           self.model_save_path)
                
        # Neural Network architecture

        tf.random.set_seed(42)
        user_network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        
        movie_network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        
        # create the user input and point to the base network
        user_input_layer = tf.keras.layers.Input(shape=(user_train_data_df.shape[1],), name='user_input_layer')
        user_network_output = user_network(user_input_layer)
        user_network_output = L2Norm(axis=1)(user_network_output)
        
        
        # create the item input and point to the base network
        movie_input_layer = tf.keras.layers.Input(shape=(movie_train_data_df.shape[1],), name='movie_input_layer')
        movie_network_output = movie_network(movie_input_layer)
        movie_network_output = L2Norm(axis=1)(movie_network_output)
        
        # compute the dot product of the two vectors vu and vm
        output = tf.keras.layers.Dot(axes=1)([user_network_output, movie_network_output])
        
        # specify the inputs and output of the model
        model = tf.keras.Model([user_input_layer, movie_input_layer], output)
        
        model.summary()
        
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.1), 
            loss = tf.keras.losses.MeanSquaredError())
        
        model.fit([user_train_df, movie_train_df], y_train_df, epochs=30)
        
        metrics = model.evaluate([user_test_df, movie_test_df], y_test_df, return_dict=True)
        logger.info("Evaluation phase. Here are your metrics: %s", metrics)
        
        # save the model
        model.save(self.model_save_path, save_format='keras')
        self._tf_model = model

    def predict(self, preferences: UserPreferences, all_movies_df: pd.DataFrame) -> list[tuple]:
        user_vector = np.array(preferences.to_list())
        
        # generate and replicate the user vector to match the number movies in the data set.
        user_matrix = np.tile(user_vector, (len(all_movies_df), 1))
        
        # scale our user and item vectors
        scaled_user_vec = self.user_scaler.transform(user_matrix)
        # TODO: call reset_index() earlier
        movies_matrix = all_movies_df.reset_index().to_numpy(dtype=int)
    
        # make a prediction
        y_p = self._tf_model.predict([scaled_user_vec, movies_matrix[:, 1:]])
    
        # unscale y prediction 
        y_pu = self.target_scaler.inverse_transform (y_p)
    
        # sort the results, highest rating first
        sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu   = y_pu[sorted_index]
        sorted_movies_matrix = movies_matrix[sorted_index][:50]  #using unscaled vectors for display
        
        
        return [
            (row[0], np.round(sorted_ypu[i, 0], 1))
            for i, row in enumerate(sorted_movies_matrix)
        ]
        
    def is_trained(self) -> bool:
        return self._tf_model is not None



