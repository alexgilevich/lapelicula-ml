import numpy
import pandas as pd
import tensorflow as tf
from keras import Layer
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import mlflow
pd.set_option("display.precision", 1)
import logging
import numpy as np
import os
from features import UserPreferences
import pydantic

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

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, num_outputs = 32, num_epochs = 30):
        self.target_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        self._tf_model = None
        self.num_outputs = num_outputs
        self.num_epochs = num_epochs
        
    
    def train(self, user_train_data: numpy.ndarray, movie_train_data: numpy.ndarray, y_labels: numpy.ndarray) -> dict:
        """
        Trains the model using the provided training data
        :param user_train_data: User training data
        :param movie_train_data: Movie training data
        :param y_labels: Y-labels (ground truth)
        :return: Train metrics
        """
        
        assert user_train_data.shape[0] == movie_train_data.shape[0]
        assert len(user_train_data) > 0 and len(movie_train_data) > 0 and len(y_labels) > 0
        
        logger.info("User training data numpy shape: %s, first row: %s", user_train_data.shape, user_train_data[0])
        logger.info("Movie training data numpy shape: %s, first row: %s", movie_train_data.shape, movie_train_data[0])
        logger.info("Label training data numpy shape: %s, first row: %s", y_labels.shape, y_labels[0])
        
        user_train_data = user_train_data[:, 1:]
        movie_train_data = movie_train_data[:, 1:]
        
        self.user_scaler.fit(user_train_data)
        user_train_data = self.user_scaler.transform(user_train_data)
        
        y_labels = y_labels.reshape(-1, 1)
        self.target_scaler.fit(y_labels)
        y_labels = self.target_scaler.transform(y_labels)

        tf.random.set_seed(42)
        
        movie_train_split, movie_test_split = train_test_split(movie_train_data, train_size=0.9, shuffle=True, random_state=42)
        user_train_split, user_test_split   = train_test_split(user_train_data, train_size=0.9, shuffle=True, random_state=42)
        y_train_split, y_test_split         = train_test_split(y_labels, train_size=0.9, shuffle=True, random_state=42)


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
        user_input_layer = tf.keras.layers.Input(shape=(user_train_data.shape[1],), name='user_input_layer')
        user_network_output = user_network(user_input_layer)
        user_network_output = L2Norm(axis=1)(user_network_output)
        
        
        # create the item input and point to the base network
        movie_input_layer = tf.keras.layers.Input(shape=(movie_train_data.shape[1],), name='movie_input_layer')
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
        
        model.fit([user_train_split, movie_train_split], y_train_split, epochs=self.num_epochs, verbose=2)
        
        metrics = model.evaluate([user_test_split, movie_test_split], y_test_split, return_dict=True, verbose=2)
        logger.info("Evaluation phase. Here are your metrics: %s", metrics)

        # save the model
        self._tf_model = model
        
        return metrics

    # noinspection PyMethodOverriding
    def predict(self, model_input: list[dict[str, dict[str, float] | numpy.ndarray]], params: dict[str, int|float]) -> list[list[tuple]]:
        if not self.is_trained():
            raise ValueError("Model is not trained. Please train the model first.")
        
        assert isinstance(model_input, list)
        
        return [self._predict_single(request['user_preferences'], request['movies'], params) for request in model_input]
        
        
    def _predict_single(self, user_preferences: dict[str, float], movies_matrix: numpy.ndarray, params: dict[str, int|float]) -> list[tuple]:
        user_preferences = user_preferences or {}
        params = params or {}
        
        assert isinstance(user_preferences, dict)
        assert isinstance(params, dict)
        assert isinstance(movies_matrix, numpy.ndarray)
        
        # parameters
        limit = params.get("limit", 50)
        
        user_preferences_model = UserPreferences(**user_preferences)
        user_vector = np.array(user_preferences_model.to_list())

        # generate and replicate the user vector to match the number movies in the data set.
        user_matrix = np.tile(user_vector, (movies_matrix.shape[0], 1))

        # scale our user and item vectors
        scaled_user_vec = self.user_scaler.transform(user_matrix)

        # make a prediction
        y_p = self._tf_model.predict([scaled_user_vec, movies_matrix[:, 1:]], verbose=0)

        # unscale y prediction 
        y_p_unscaled = self.target_scaler.inverse_transform(y_p)

        # sort the results, highest rating first
        sorted_index = np.argsort(-y_p_unscaled, axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu = y_p_unscaled[sorted_index]
        sorted_movies_matrix = movies_matrix[sorted_index][:limit]  #using unscaled vectors for display

        return [
            (row[0], np.round(sorted_ypu[i, 0], 1))
            for i, row in enumerate(sorted_movies_matrix)
        ]
        
    def is_trained(self) -> bool:
        return self._tf_model is not None



