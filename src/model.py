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

    def call(self, inputs, name=None):
        return tf.linalg.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon, name=name)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis, "epsilon": self.epsilon})
        return cfg



class CustomTrainingStepModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def train_step(self, data):
        (user_matrix, movie_matrix), y_ratings = data

        batch_size = tf.shape(movie_matrix)[0]

        # sample negatives by shuffling movies in batch
        neg_indices = tf.random.shuffle(tf.range(batch_size))
        neg_movies = tf.gather(movie_matrix, neg_indices)

        with tf.GradientTape() as tape:
            # positive scores
            pos_scores = self.base_model([user_matrix, movie_matrix])['y_scaled']
            # negative scores
            neg_scores = self.base_model([user_matrix, neg_movies])['y_scaled']
            # predicted ratings
            pred_ratings = pos_scores

            # MSE loss (real ratings)
            mse_loss = tf.reduce_mean((y_ratings['y_scaled'] - pred_ratings) ** 2)

            # ranking loss
            rank_loss = -tf.reduce_mean(
                tf.math.log(tf.sigmoid(pos_scores - neg_scores) + 1e-8)
            )

            # combined loss
            loss = mse_loss + 0.2 * rank_loss

        grads = tape.gradient(loss, self.base_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.base_model.trainable_variables)
        )

        # 👇 update metrics manually
        for metric in self.metrics:
            metric.update_state(y_ratings['y_scaled'], pred_ratings)

        return {
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
        
        
        user_scaler_layer = tf.keras.layers.Normalization(axis=-1)
        user_scaler_layer.adapt(user_train_data[:, 1:])  # fit scaler


        y_labels = y_labels.reshape(-1, 1)
        y_scaler = tf.keras.layers.Normalization(axis=-1, name="y_scaler_layer")
        y_scaler.adapt(y_labels)

        # 🔑 Construct the denormalization layer using the adapted normalization one
        y_unscaler = tf.keras.layers.Normalization(axis=-1, mean=y_scaler.mean, variance=y_scaler.variance, invert=True, name="y_unscaler_layer")
        
        tf.random.set_seed(42)
        
        movie_train_split, movie_test_split = train_test_split(movie_train_data, train_size=0.9, shuffle=True, random_state=42)
        user_train_split, user_test_split   = train_test_split(user_train_data, train_size=0.9, shuffle=True, random_state=42)
        y_train_split, y_test_split         = train_test_split(y_labels, train_size=0.9, shuffle=True, random_state=42)

        movie_train_split, movie_test_split = movie_train_split[:, 1:], movie_test_split[:, 1:] # skip movie ids
        user_train_split, user_test_split, user_ids_test_split = user_train_split[:, 1:], user_test_split[:, 1:], user_test_split[:, 0] # skip user ids for training, save user ids separately for evaluation (NDCG calculation) 
        
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

        rec_model = CustomTrainingStepModel(model)
        
        model.summary()
        rec_model.summary()
        
        compile_args = {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.1),
            "loss": {
                "y_scaled": tf.keras.losses.MeanSquaredError(), # dummy loss function because Model.fit() expects a loss defined in compile() even though we compute the loss inside the custom train_step()
            },
            "metrics": [
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.MeanAbsoluteError(name="mae")
            ]
        }
        rec_model.compile(**compile_args)
        model.compile(**compile_args)

        rec_model.fit(
            [user_train_split, movie_train_split],
            {"y_scaled": y_scaler(y_train_split).numpy()}, 
            epochs=self.num_epochs, 
            verbose=1
        )

        metrics = self._evaluate_trained_model(model, movie_test_split, user_test_split, user_ids_test_split,
                                               y_scaler(y_test_split).numpy())

        logger.info("Evaluation phase. Here are your metrics: %s", metrics)

        # save the model
        self._tf_model = model
        
        return metrics

    def _evaluate_trained_model(self, model, movie_test_split, user_test_split, user_ids_test_split,
                                y_test_split_scaled):
        metrics = model.evaluate(
            [user_test_split, movie_test_split],
            y_test_split_scaled,
            return_dict=True,
            verbose=2
        )
        preds = model.predict([user_test_split, movie_test_split])["y_scaled"]
        ndcg = self._compute_ndcg_offline(
            y_test_split_scaled.flatten(),
            preds.flatten(),
            user_ids_test_split  # you MUST keep these!
        )
        metrics["ndcg"] = ndcg
        return metrics


    @staticmethod
    def _compute_ndcg_offline(y_true, y_pred, user_ids, k=10):
        # group by user
        unique_users = np.unique(user_ids)
        ndcgs = []

        for user_id in unique_users:
            mask = user_ids == user_id
            y_t = y_true[mask]
            y_p = y_pred[mask]

            if len(y_t) < k:
                continue

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

