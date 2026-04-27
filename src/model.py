import numpy
import pandas as pd
import tensorflow as tf
from keras import Layer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import mlflow
pd.set_option("display.precision", 1)
import logging
import numpy as np
import os
from features import UserPreferences
from logging_factory import get_logger
import pydantic

logger = get_logger(__name__)


@tf.keras.utils.register_keras_serializable()
class L2Norm(Layer):
    def __init__(self, axis=1, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs, name=None):
        return tf.linalg.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon, name=name)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis, "epsilon": self.epsilon})
        return cfg




class CustomTrainingStepModel(tf.keras.Model):
    """
    This class implements a custom training step to perform global negative sampling for the ranking loss component of the model's loss function. The global negative sampling is performed by sampling negative movies from the entire set of movies, rather than just from the batch, which allows for a more diverse set of negative samples and can lead to better model performance.
    @param base_model: The base Keras model that takes user and movie inputs and produces a predicted rating.
    @param all_movies: A numpy array containing all movies information (IDs, features), used for global negative sampling
    """
    def __init__(self, base_model: tf.keras.Model, all_movies: np.ndarray):
        super().__init__()
        self.base_model = base_model
        self.all_movies = all_movies
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.rank_tracker = tf.keras.metrics.Mean(name="rank_loss")


    def train_step(self, data):
        # extract what we pass into the fit function
        (user_matrix, movie_matrix, movie_ids), y_ratings = data

        batch_size = tf.shape(movie_matrix)[0]
        num_movies = tf.shape(self.all_movies)[0]

        # global negative sampling
        neg_indices = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=num_movies,
            dtype=tf.int32
        )

        # avoid sampling the same movie for the same user in the same batch (collision), resample if collision occurs (up to 2 attempts)
        for _ in range(2):
            neg_movies = tf.gather(self.all_movies, neg_indices)
            collision_mask = tf.equal(neg_movies[:, 0], movie_ids)

            resample_indices = tf.random.uniform(
                shape=(batch_size,),
                minval=0,
                maxval=num_movies,
                dtype=tf.int32
            )

            neg_indices = tf.where(collision_mask, resample_indices, neg_indices)

        # final negative movies after resampling
        neg_movies = tf.gather(self.all_movies, neg_indices)[:, 1:] # skip movie ids, keep only features for training

        with tf.GradientTape() as tape:
            # positive scores
            pos_scores = self.base_model([user_matrix, movie_matrix])['y_scaled']
            # negative scores
            neg_scores = self.base_model([user_matrix, neg_movies])['y_scaled']


            # MSE loss (real ratings)
            mse_loss = tf.reduce_mean(tf.math.square(y_ratings['y_scaled'] - pos_scores), axis=-1)

            # ranking loss
            rank_loss = -tf.reduce_mean(
                tf.math.log(tf.sigmoid(pos_scores - neg_scores) + 1e-8),
                axis=-1
            )

            # combined loss
            loss = mse_loss + 0.2 * rank_loss

        grads = tape.gradient(loss, self.base_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.base_model.trainable_variables)
        )

        # update built-in metrics
        # use batch size as weight to get average loss per sample
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(tf.keras.metrics.mse(y_ratings['y_scaled'], pos_scores)) 
        self.rank_tracker.update_state(rank_loss)
        

        # 👇 update metrics manually
        for metric in self.metrics:
            metric.update_state(y_ratings['y_scaled'], pos_scores)

        return {
            "loss": self.loss_tracker.result(),
            "mse_loss": tf.reduce_mean(self.mse_tracker.result()),
            "rank_loss": self.rank_tracker.result(),
            **{m.name: m.result() for m in self.metrics}
        }

    def call(self, inputs):
        return self.base_model(inputs)



class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, num_outputs = 32, num_epochs = 30, model_save_path = "model_binary.keras"):
        self._tf_model = None
        self.num_outputs = num_outputs
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        
    
    def train(self, user_train_data: numpy.ndarray, movie_train_data: numpy.ndarray, y_labels: numpy.ndarray, all_movies: numpy.ndarray) -> dict:
        """
        Trains the model using the provided training data
        :param user_train_data: User training data
        :param movie_train_data: Movie training data
        :param y_labels: Y-labels (ground truth)
        :param all_movies: Array containing all movies information (used for negative sampling)
        :return: Train metrics
        """
        
        assert user_train_data.shape[0] == movie_train_data.shape[0]
        assert len(user_train_data) > 0 and len(movie_train_data) > 0 and len(y_labels) > 0
        
        logger.info("User training data numpy shape: %s, first row: %s", user_train_data.shape, user_train_data[0])
        logger.info("Movie training data numpy shape: %s, first row: %s", movie_train_data.shape, movie_train_data[0])
        logger.info("Label training data numpy shape: %s, first row: %s", y_labels.shape, y_labels[0])
        
        
        user_scaler_layer = tf.keras.layers.Normalization(axis=-1)
        user_scaler_layer.adapt(user_train_data[:, 1:])  # fit scaler


        y_labels = y_labels.reshape(-1, 1)
        y_scaler = tf.keras.layers.Normalization(axis=-1, name="y_scaler_layer")
        y_scaler.adapt(y_labels)

        # 🔑 Construct the denormalization layer using the adapted normalization one
        y_unscaler = tf.keras.layers.Normalization(axis=-1, mean=y_scaler.mean, variance=y_scaler.variance, invert=True, name="y_unscaler_layer")
        
        tf.random.set_seed(42)
        (
            movie_train_split,
            movie_test_split,
            user_train_split,
            user_test_split,
            y_train_split,
            y_test_split
        ) = train_test_split(
            movie_train_data,
            user_train_data,
            y_labels,
            train_size=0.9,
            shuffle=True,
            random_state=42
        )

        # skip movie ids, save movie ids separately for training (negative sampling)
        movie_train_split, movie_ids_train_split, movie_test_split = movie_train_split[:, 1:], movie_train_split[:, 0], movie_test_split[:, 1:] 
        # skip user ids for training, save user ids separately for evaluation (NDCG calculation) 
        user_train_split, user_test_split, user_ids_test_split = user_train_split[:, 1:], user_test_split[:, 1:], user_test_split[:, 0] 
        
        # skip user ids
        movie_train_data = movie_train_data[:, 1:] # skip movie ids

        # Neural Network architecture
        user_network = tf.keras.models.Sequential([
            user_scaler_layer,
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        
        movie_network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        
        # create the user input and point to the base network
        user_input_layer = tf.keras.layers.Input(shape=(user_train_split.shape[1],), name='user_input_layer')
        user_network_output = user_network(user_input_layer)
        user_network_output = L2Norm(axis=1)(user_network_output, name='user_l2_normalization_layer')
        
        
        # create the item input and point to the base network
        movie_input_layer = tf.keras.layers.Input(shape=(movie_train_split.shape[1],), name='movie_input_layer')
        movie_network_output = movie_network(movie_input_layer)
        movie_network_output = L2Norm(axis=1)(movie_network_output, name='movie_l2_normalization_layer')
        
        # compute the dot product of the two vectors: the output user vector and the output movie vector
        dot_product_output_layer = tf.keras.layers.Dot(axes=1, name="dot_product_layer")([user_network_output, movie_network_output])
        # Inverse output scaling layer
        unscaled_output = y_unscaler(dot_product_output_layer)
        
        # specify the inputs and output of the model
        model = tf.keras.Model(
            inputs=[user_input_layer, movie_input_layer], 
            outputs={
                "y_scaled": dot_product_output_layer,
                "y_unscaled": unscaled_output
            })

        rec_model = CustomTrainingStepModel(model, all_movies)
        
        model.summary()
        rec_model.summary()
        
        compile_args = {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.1),
            "loss": {
                "y_scaled": tf.keras.losses.MeanSquaredError(), # dummy loss function because Model.fit() expects a loss defined in compile() even though we compute the loss inside the custom train_step()
            },
            "metrics": [
                tf.keras.metrics.MeanSquaredError(name="tf_mse"),
                tf.keras.metrics.MeanAbsoluteError(name="tf_mae")
            ]
        }
        rec_model.compile(**compile_args)
        model.compile(**compile_args) # compile the base model as well, just for evaluation purposes (we won't use its train_step)

        rec_model.fit(
            [user_train_split, movie_train_split, movie_ids_train_split],
            {"y_scaled": y_scaler(y_train_split).numpy()}, 
            epochs=self.num_epochs, 
            verbose=1
        )

        metrics = self._evaluate_trained_model(model, movie_test_split, user_test_split, user_ids_test_split, y_test_split) 

        logger.info("Evaluation phase. Here are your metrics: %s", metrics)

        # save the model
        self._tf_model = model
        
        return metrics

    def _evaluate_trained_model(self, model, movie_test_split, user_test_split, user_ids_test_split, y_test_split):
        """
        Evaluate the trained model using MSE, MAE and NDCG metrics. The NDCG is computed offline by generating predictions for the test set and comparing them to the true ratings, grouped by user.
        :param model: The trained Keras model to evaluate
        :param movie_test_split: Movie features for the test set
        :param user_test_split: User features for the test set
        :param user_ids_test_split: User IDs for the test set (used for NDCG calculation)
        :param y_test_split: True ratings for the test set (unscaled!)
        :return: A dictionary containing the computed metrics (MSE, MAE, NDCG) for the test set
        """

        metrics = model.evaluate(
            [user_test_split, movie_test_split],
            y_test_split,
            return_dict=True,
            verbose=2
        )
        preds = model.predict([user_test_split, movie_test_split])["y_scaled"]
        ndcg = self._compute_ndcg_offline(
            y_test_split,
            preds.flatten(),
            user_ids_test_split  # you MUST keep these!
        )
        metrics["ndcg"] = ndcg
        return metrics


    @staticmethod
    def _compute_ndcg_offline(y_true, y_pred, user_ids, k=10):
        """Compute NDCG@k offline by generating predictions for the test set and comparing them to the true ratings, grouped by user. This function assumes that the input y_true, y_pred and user_ids are aligned (i.e. the i-th element of each corresponds to the same user-movie pair) and that user_ids contains the user identifiers for each prediction, which are necessary for grouping the predictions by user."""

        # group by user
        unique_users = np.unique(user_ids)
        ndcgs = []

        for user_id in unique_users:
            mask = user_ids == user_id
            y_t = y_true[mask]
            y_p = y_pred[mask]

            # if there are less than k movies for this user still caculate the metric but adjust k to the number of ratings for this user
            k = min(k, len(y_t))

            # sort by prediction
            order = np.argsort(-y_p)
            y_true_sorted_by_prediction = y_t[order][:k]

            gains = (2 ** y_true_sorted_by_prediction - 1)
            discounts = np.log2(np.arange(2, k + 2))
            dcg = np.sum(gains / discounts)

            # ideal
            y_ideal_sorted = np.sort(y_t)[::-1][:k]
            ideal_dcg = np.sum((2 ** y_ideal_sorted - 1) / discounts)

            ndcgs.append(dcg / (ideal_dcg + 1e-8))

        return np.mean(ndcgs)
    
    def save(self):
        if not self._tf_model:
            raise RuntimeError()
        self._tf_model.save(self.model_save_path, save_format='keras')
        
    def load(self):
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(self.model_save_path)
        self._tf_model = tf.keras.models.load_model(self.model_save_path, safe_mode=False)


    
    # noinspection PyMethodOverriding
    def predict(self, model_input, params):
        if not self.is_trained():
            raise ValueError("Model is not trained. Please train the model first.")

        params: dict[str, int|float] = params or {}
        
        logger.debug("Args type = %s ,params type = %s, params = %s", type(model_input), type(params), params)
        assert isinstance(model_input, pd.DataFrame), "Input is not a Pandas DataFrame"
        #assert isinstance(model_input, dict), "Params is not a dictionary"
        
        return [self._predict_single(request['user_preferences'], request['movies'], params) for _, request in model_input.iterrows()]
        
        
    def _predict_single(self, user_preferences: dict[str, float], movies_matrix: numpy.ndarray, params: dict[str, int|float]) -> list[list]:
        user_preferences = user_preferences or {}
        params = params or {}
        
        assert isinstance(user_preferences, dict)
        assert isinstance(params, dict)
        assert isinstance(movies_matrix, numpy.ndarray)
        
        # parameters
        limit = params.get("limit", 50)

        logger.info("Predicting top %d movies for the following preferences: %s", limit, user_preferences)
        
        user_preferences_model = UserPreferences(**user_preferences)
        user_vector = np.array(user_preferences_model.to_list())

        # generate and replicate the user's vector to match the movies' vector shape
        user_matrix = np.tile(user_vector, (movies_matrix.shape[0], 1))

        # make a prediction
        y_p = self._tf_model.predict([user_matrix, movies_matrix[:, 1:]], verbose=0)

        # take the unscaled y prediction 
        y_p_unscaled = y_p["y_unscaled"]

        # sort the results, highest rating first
        sorted_index = np.argsort(-y_p_unscaled, axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu = y_p_unscaled[sorted_index]
        sorted_movies_matrix = movies_matrix[sorted_index][:limit]  #using unscaled vectors for display

        return [
            [row[0], np.round(sorted_ypu[i, 0], 1)]
            for i, row in enumerate(sorted_movies_matrix)
        ]
        
    def is_trained(self) -> bool:
        return self._tf_model is not None

