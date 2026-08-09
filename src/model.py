import numpy
import pandas as pd
import tensorflow as tf
from keras import Layer
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np
import os
from features import UserPreferences
from logging_factory import get_logger

pd.set_option("display.precision", 1)
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



@tf.keras.utils.register_keras_serializable()
class TrainingKerasModel(tf.keras.Model):
    """
    This class implements a custom training step to perform global negative sampling for the ranking loss component of the model's loss function. The global negative sampling is performed by sampling negative movies from the entire set of movies, rather than just from the batch, which allows for a more diverse set of negative samples and can lead to better model performance.
    @param base_model: The base Keras model that takes user and movie inputs and produces a predicted rating.
    @param all_movies: A numpy array containing all movies information (IDs, features), used for global negative sampling
    """
    def __init__(self, base_model: tf.keras.Model, all_movies: np.ndarray, output_names: list[str], **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.all_movies = all_movies
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.rank_tracker = tf.keras.metrics.Mean(name="rank_loss")
        self.output_names = output_names


    def train_step(self, data):
        # extract what we pass into the fit function
        (user_matrix, movie_matrix, embedding_matrix, movie_ids), y_true = data

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
            pos_scores = self.base_model([user_matrix, movie_matrix, embedding_matrix])['y_scaled']
            # negative scores
            neg_scores = self.base_model([user_matrix, neg_movies, embedding_matrix])['y_scaled']

            # MSE loss (real ratings)
            mse_loss = tf.reduce_mean(tf.math.square(y_true['y_scaled'] - pos_scores), axis=-1)

            # ranking loss
            rank_loss = -tf.reduce_mean(
                tf.math.log(tf.sigmoid(pos_scores - neg_scores) + 1e-8), 
                axis=-1
            )

            # combined loss
            total_loss = mse_loss + 0.2 * rank_loss
            
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.base_model.trainable_variables)
        )

        return self.compute_metrics((user_matrix, movie_matrix, movie_ids), y_true['y_scaled'], pos_scores, mse_loss, rank_loss, total_loss)
    
    def test_step(self, data):
        # extract what we pass into the fit function
        (user_matrix, movie_matrix, embedding_matrix), y_true = data
        
        # Compute predictions
        y_pred = self([user_matrix, movie_matrix, embedding_matrix], training=False)
        
        mse = tf.reduce_mean(tf.math.square(y_true['y_scaled'] - y_pred['y_scaled']))
        mae = tf.reduce_mean(tf.abs(y_true['y_scaled'] - y_pred['y_scaled']))


        # Return a dict with standard metrics values (e.g. {"mse": mse, "mae": mae})
        return {
            "mse": mse, 
            "mae": mae
        }

    def call(self, inputs):
        return self.base_model(inputs)
    
    def compute_metrics(self, x, y, y_pred, mse_loss, rank_loss, total_loss):
        metric_results = super().compute_metrics(x, y, y_pred)
        
        # update built-in metrics
        self.total_loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.rank_tracker.update_state(rank_loss)
        
        # compute and return a dictionary of metrics
        return {
            **{k: v for k, v in metric_results.items() if k != "loss"},  # remove the default loss metric since we compute our own combined loss
            self.total_loss_tracker.name: self.total_loss_tracker.result(),
            self.mse_tracker.name: self.mse_tracker.result(),
            self.rank_tracker.name: self.rank_tracker.result()
        }
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        result = [self.total_loss_tracker, self.mse_tracker, self.rank_tracker]
        result.extend(super().metrics)
        return result
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"output_names": self.output_names})
        cfg.update({"base_model": self.base_model.get_config()})
        return cfg
    
    @classmethod
    def from_config(cls, config):
        return cls(tf.keras.Model.from_config(config["base_model"]), None, config["output_names"])

class Model(mlflow.pyfunc.PythonModel):
    """
    A custom model class for training and evaluating a neural network for movie recommendation.
    The model is trained using a custom training loop implemented in the TrainingKerasModel class, which performs global negative sampling for the ranking loss component of the loss function. 
    The model is evaluated using MSE, MAE and NDCG metrics, where the NDCG is computed offline by generating predictions for the test set and comparing them to the true ratings, grouped by user.
    @param num_outputs: The number of output dimensions for the user and movie embeddings in the model (default: 32)
    @param num_epochs: The number of epochs to train the model for (default: 30)
    @param batch_size: The batch size to use during training (default: 2048)
    @param model_save_path: The local file path to save the trained model (default: "model_binary.keras")
    @param verbose_level: The verbosity level for training output (default: 1)
    """
    def __init__(self, num_outputs = 32, num_epochs = 30, batch_size = 2048, model_save_path = "model_binary.keras", verbose_level = 1):
        self._tf_model = None
        self.num_outputs = num_outputs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.verbose_level = verbose_level
    
    def get_params(self):
        return {
            "num_outputs": self.num_outputs,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "model_save_path": self.model_save_path
        }
    
    def train(self, user_train_data: numpy.ndarray, movie_train_data: numpy.ndarray, embedding_train_data: numpy.ndarray, y_labels: numpy.ndarray, all_movies: numpy.ndarray) -> dict:
        """
        Trains the model using the provided training data
        :param user_train_data: User training data
        :param movie_train_data: Movie training data
        :param embedding_train_data: Movie description embeddings
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
            embedding_train_split,
            embedding_test_split,
            user_train_split,
            user_test_split,
            y_train_split,
            y_test_split
        ) = train_test_split(
            movie_train_data,
            embedding_train_data,
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
        



        
        # create the user input and point to the base network
        user_input = tf.keras.layers.Input(shape=(user_train_split.shape[1],), name = 'user_input_layer')
        user_seq = tf.keras.models.Sequential([
            user_scaler_layer,
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        user_output = L2Norm(axis=1)(user_seq(user_input), name = 'user_l2_normalization_layer')
        
        
        # create the item input and point to the base network
        movie_cat_features_input = tf.keras.layers.Input(shape=(movie_train_split.shape[1],), name = 'movie_cat_features_input_layer')
        movie_embedding_input = tf.keras.layers.Input(shape=(embedding_train_split.shape[1],), name = 'movie_embedding_input_layer')


        movie_embedding_preprocessing = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = 128, activation = 'relu')
        ])

        movie_embedding_preprocessing_output = movie_embedding_preprocessing(movie_embedding_input)
        
        movie_features_concatenation = tf.keras.layers.Concatenate(axis = -1, name = "movie_embedding_concatenate_layer")([movie_cat_features_input, movie_embedding_preprocessing_output])
        movie_seq = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 256, activation = 'relu'),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        movie_network_output = movie_seq(movie_features_concatenation)
        movie_network_output = L2Norm(axis=1)(movie_network_output, name = 'movie_l2_normalization_layer')
        
        # compute the dot product of the two vectors: the output user vector and the output movie vector
        dot_product_output = tf.keras.layers.Dot(axes=1, name="dot_product_layer")([user_output, movie_network_output])

        # Inverse output scaling layer
        unscaled_output = y_unscaler(dot_product_output)
        
        # specify the inputs and output of the model
        model = tf.keras.Model(
            inputs=[user_input, movie_cat_features_input, movie_embedding_input], 
            outputs={
                "y_scaled": dot_product_output,
                "y_unscaled": unscaled_output
            }
        )

        rec_model = TrainingKerasModel(model, all_movies, ['y_scaled', 'y_unscaled'])
        
        rec_model.summary()
        
        compile_args = {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.1),
            "loss": {
                "y_scaled": tf.keras.losses.MeanSquaredError(), # dummy loss function because Model.fit() expects a loss defined in compile() even though we compute the loss inside the custom train_step()
            },
            "metrics": {
                "y_scaled": [tf.keras.metrics.MeanSquaredError(name="tf_mse"), tf.keras.metrics.MeanAbsoluteError(name="tf_mae")],
                "y_unscaled": []
            },
            "run_eagerly": True #uncomment for eager execution (slower but allows for step-by-step debugging and printing of intermediate values in the train_step)
        }
        rec_model.compile(**compile_args)

        rec_model.fit(
            [user_train_split, movie_train_split, embedding_train_split, movie_ids_train_split],
            {"y_scaled": y_scaler(y_train_split).numpy()}, 
            epochs=self.num_epochs,
            verbose=self.verbose_level,
            batch_size = self.batch_size
        )

        metrics = self._evaluate_trained_model(
            rec_model, 
            movie_test_split, 
            embedding_test_split,
            user_test_split, 
            user_ids_test_split, 
            {
                "y_scaled": y_scaler(y_test_split).numpy(),
                "y_unscaled": y_test_split
            }
        ) 

        logger.info("Evaluation phase. Here are your metrics: %s", metrics)

        # save the model
        self._tf_model = rec_model
        
        return metrics

    def _evaluate_trained_model(self, model, movie_test_split, embedding_test_split, user_test_split, user_ids_test_split, y_test_split):
        """
        Evaluate the trained model using MSE, MAE and NDCG metrics. The NDCG is computed offline by generating predictions for the test set and comparing them to the true ratings, grouped by user.
        :param model: The trained Keras model to evaluate
        :param movie_test_split: Movie features for the test set
        :param embedding_test_split: Movie embeddings for the test set
        :param user_test_split: User features for the test set
        :param user_ids_test_split: User IDs for the test set (used for NDCG calculation)
        :param y_test_split: True ratings for the test set (unscaled!)
        :return: A dictionary containing the computed metrics (MSE, MAE, NDCG) for the test set
        """
        
        metrics = model.evaluate(
            [user_test_split, movie_test_split, embedding_test_split],
            y_test_split,
            return_dict=True,
            verbose=2
        )
        metrics = {k: v.numpy() for k, v in metrics.items()}
        preds = model.predict([user_test_split, movie_test_split, embedding_test_split])
        ndcg = self._compute_ndcg_offline(
            y_test_split,
            preds,
            user_ids_test_split
        )
        metrics["ndcg"] = ndcg
        return metrics


    @staticmethod
    def _compute_ndcg_offline(y_true, y_pred, user_ids, k=10):
        """Compute NDCG@k offline by generating predictions for the test set and comparing them to the true ratings, grouped by user. This function assumes that the input y_true, y_pred and user_ids are aligned (i.e. the i-th element of each corresponds to the same user-movie pair) and that user_ids contains the user identifiers for each prediction, which are necessary for grouping the predictions by user."""
        
        y_true_scaled = y_true['y_scaled'].flatten()
        y_pred_scaled = y_pred['y_scaled'].flatten()

        # group by user
        unique_users = np.unique(user_ids)
        ndcgs = []

        for user_id in unique_users:
            mask = user_ids == user_id
            y_t = y_true_scaled[mask]
            y_p = y_pred_scaled[mask]

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
        
        
    def _predict_single(self, user_preferences: dict[str, float], movies_matrix: numpy.ndarray, params: dict[str, int|float]) -> list[tuple[int, float]]:
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
            (row[0], np.round(sorted_ypu[i, 0], 1))
            for i, row in enumerate(sorted_movies_matrix)
        ]
        
    def is_trained(self) -> bool:
        return self._tf_model is not None

