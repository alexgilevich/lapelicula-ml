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



class EpochMetricsCallback(tf.keras.callbacks.Callback):
    """Collects the per-epoch training metrics (fit()'s logs dict) so they can
    be reported to MLflow after training finishes."""

    def __init__(self):
        super().__init__()
        self.epoch_metrics: list[dict[str, float]] = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_metrics.append({k: float(v) for k, v in (logs or {}).items()})


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
        self.mae_tracker = tf.keras.metrics.Mean(name="mae_loss")
        self.rank_tracker = tf.keras.metrics.Mean(name="rank_loss")
        self.ndcg_tracker = tf.keras.metrics.Mean(name="ndcg")

        self.output_names = output_names
        self.at_k = 100
        # which phase's metrics get_metrics_result() should report;
        # train_step/test_step keep it up to date
        self._in_test = False


    def train_step(self, data):
        self._in_test = False
        # extract what we pass into the fit function
        (user_matrix, movie_matrix, embedding_matrix, movie_ids, user_ids), y_true = data

        batch_size = tf.shape(movie_matrix)[0]
        num_movies = tf.shape(self.all_movies)[0]

        # Build per-user mask of invalid negative candidates:
        # Exclude movies whose genres match any genre the user likes (avg rating >= 3).
        num_genres = 18
        user_genres = user_matrix[:, -num_genres:]  # batch x 18, values in [0, 5]
        liked_genre_mask = tf.cast(user_genres >= 3.0, dtype=tf.float32)  # batch x 18

        all_movie_genres = self.all_movies[:, 12:12 + num_genres]  # num_movies x 18
        # invalid_negatives[i, j] = True if movie j shares any liked genre with user i
        invalid_mask = tf.matmul(liked_genre_mask, tf.transpose(all_movie_genres)) > 0.0  # batch x num_movies

        num_negs_per_pos = 5
        total_negs = batch_size * num_negs_per_pos

        # Repeat each positive row num_negs_per_pos times so we can sample and compare element-wise
        movie_ids_expanded = tf.repeat(movie_ids, num_negs_per_pos)  # (B*N,)
        invalid_mask_expanded = tf.repeat(invalid_mask, num_negs_per_pos, axis=0)  # (B*N, num_movies)

        # global negative sampling with rejection of invalid candidates
        neg_indices = tf.random.uniform(
            shape=(total_negs,),
            minval=0,
            maxval=num_movies,
            dtype=tf.int32
        )

        # Rejection sampling: keep resampling invalid indices (up to 10 attempts)
        for _ in range(10):
            neg_movies = tf.gather(self.all_movies, neg_indices)
            collision_mask = tf.equal(neg_movies[:, 0], movie_ids_expanded)  # (B*N,) element-wise
            row_idx = tf.range(total_negs)
            genre_invalid_mask = tf.gather_nd(invalid_mask_expanded, tf.stack([row_idx, neg_indices], axis=1))  # (B*N,)

            invalid = tf.logical_or(collision_mask, genre_invalid_mask)
            resample_indices = tf.random.uniform(
                shape=(total_negs,),
                minval=0,
                maxval=num_movies,
                dtype=tf.int32
            )
            neg_indices = tf.where(invalid, resample_indices, neg_indices)

        # final negative movies after resampling: (B*N, features)
        neg_movies = tf.gather(self.all_movies, neg_indices)[:, 1:]

        with tf.GradientTape() as tape:
            # positive scores: (B,) — keep both scales; the loss uses y_scaled,
            # NDCG is computed on the raw-scale output of the inverse layer
            base_out = self.base_model([user_matrix, movie_matrix, embedding_matrix])
            pos_scores = base_out['y_scaled']
            pos_scores_unscaled = base_out['y_unscaled']

            # repeat user and embedding rows to match the expanded batch for negatives: (B*N, features)
            user_expanded = tf.repeat(user_matrix, num_negs_per_pos, axis=0)
            embedding_expanded = tf.repeat(embedding_matrix, num_negs_per_pos, axis=0)

            # negative scores: (B*N,)
            neg_scores = self.base_model([user_expanded, neg_movies, embedding_expanded])['y_scaled']

            # MSE loss
            mse_loss = tf.reduce_mean(tf.math.square(y_true['y_scaled'] - pos_scores), axis=-1)

            # MAE loss
            mae_loss = tf.reduce_mean(tf.abs(y_true['y_scaled'] - pos_scores), axis=-1)

            # ranking loss: reshape neg scores to (B, N) for per-positive comparison
            neg_scores_reshaped = tf.reshape(neg_scores, [batch_size, num_negs_per_pos])
            rank_loss = -tf.reduce_mean(
                tf.math.log(tf.sigmoid(pos_scores[:, None] - neg_scores_reshaped) + 1e-8),
                axis=[-1, -2]
            )

            # combined loss
            total_loss = mse_loss + 1.0 * rank_loss

        grads = tape.gradient(total_loss, self.base_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.base_model.trainable_variables)
        )

        return self.compute_metrics(
            (user_matrix, movie_matrix, embedding_matrix, movie_ids, user_ids),
            y_true['y_scaled'],
            pos_scores,
            mae_loss,
            mse_loss,
            rank_loss,
            total_loss,
            y_true['y_unscaled'],
            pos_scores_unscaled,
        )
    
    def test_step(self, data):
        self._in_test = True
        # extract what we pass into the fit function
        (user_matrix, movie_matrix, embedding_matrix, user_ids), y_true = data
        
        # Compute predictions
        y_pred = self([user_matrix, movie_matrix, embedding_matrix], training=False)
        
        mse = tf.reduce_mean(tf.math.square(y_true['y_scaled'] - y_pred['y_scaled']))
        mae = tf.reduce_mean(tf.abs(y_true['y_scaled'] - y_pred['y_scaled']))

        # NDCG on RAW (unscaled) ratings: the 2**x - 1 gain assumes non-negative
        # relevance grades where 0 means "not relevant", which StandardScaler
        # output violates (below-mean ratings become negative gains). The model's
        # y_unscaled output is a monotonic inverse transform of y_scaled, so the
        # ranking is unchanged — only the gain magnitudes are now well defined.
        preds = self([user_matrix, movie_matrix, embedding_matrix])
        ndcg_per_user = self._compute_ndcg_per_user(
            y_true['y_unscaled'],
            preds['y_unscaled'], 
            user_ids
        )
        ndcg = tf.reduce_mean(ndcg_per_user)

        # update the evaluation trackers so get_metrics_result() at the end of
        # evaluate() can collect them (Keras resets all self.metrics before
        # evaluate, so they must be (re)built here)
        self.mse_tracker.update_state(mse)
        self.mae_tracker.update_state(mae)
        # Feed the per-user vector (not the batch mean) into the Mean tracker:
        # it then accumulates total = sum of per-user NDCGs and count = number
        # of users, so its result is the exact per-user average over ALL
        # batches (each user counted once, no matter which batch their rows
        # landed in). Averaging the per-batch means instead would weight
        # batches equally, not users.
        self.ndcg_tracker.update_state(ndcg_per_user)

        # Return a dict keyed by the tracker names so that its keys match
        # get_metrics_result() — Keras's _get_metrics_result_or_logs() then
        # reports the tracker values (averaged over all batches) instead of
        # only the last batch's raw values
        return {
            "mse_loss": mse,
            "mae_loss": mae,
            "ndcg": ndcg
        }

    def summary(self, *args, **kvargs) -> None:
        self.base_model.summary(*args, **kvargs)

    def call(self, inputs):
        return self.base_model(inputs)
    
    def compute_metrics(self, x, y_true, y_pred, mae_loss, mse_loss, rank_loss, total_loss, y_true_unscaled, y_pred_unscaled):
        metric_results = super().compute_metrics(x, y_true, y_pred)

        _, _, _, _, user_ids = x

        # update built-in metrics
        self.total_loss_tracker.update_state(total_loss)
        self.mae_tracker.update_state(mae_loss)
        self.mse_tracker.update_state(mse_loss)
        self.rank_tracker.update_state(rank_loss)
        # NDCG on RAW (unscaled) ratings — see the note in test_step()
        self.ndcg_tracker.update_state(self._compute_ndcg_per_user(y_true_unscaled, y_pred_unscaled, user_ids))
        # compute and return a dictionary of metrics
        return {
            **{k: v for k, v in metric_results.items() if k != "loss"},  # remove the default loss metric since we compute our own combined loss
            self.total_loss_tracker.name: self.total_loss_tracker.result(),
            self.mse_tracker.name: self.mse_tracker.result(),
            self.mae_tracker.name: self.mae_tracker.result(),
            self.rank_tracker.name: self.rank_tracker.result(),
            self.ndcg_tracker.name: self.ndcg_tracker.result()
        }

    
    def _compute_ndcg(self, y_true, y_pred, user_ids):
        """Mean NDCG@k over the users present in the input (scalar, for logging)."""
        return tf.reduce_mean(self._compute_ndcg_per_user(y_true, y_pred, user_ids))

    def _compute_ndcg_per_user(self, y_true, y_pred, user_ids):
        """Compute NDCG@k by comparing predictions to the true ratings, grouped by user.

        The input y_true, y_pred and user_ids must be aligned (the i-th element of each corresponds
        to the same user-movie pair); user_ids holds the user identifier for each row and is used
        to group rows by user. For each user the items are ranked by predicted score and the top
        `self.at_k` items (fewer if the user has less) contribute to the DCG.

        Returns one NDCG value per distinct user in the input (a 1-D tensor), so that callers
        can aggregate with exact per-user weighting across batches (see test_step).

        Implemented entirely with TF-native ops (no numpy, no .numpy(), no Python loops over
        data-dependent values), so it works on numpy arrays, eager tensors and symbolic tensors
        (graph mode, e.g. inside test_step during model.evaluate()).
        """
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        user_ids = tf.reshape(user_ids, [-1])

        # DCG ranking by predicted scores vs. ideal DCG ranking by true scores
        dcg = self._grouped_dcg(rank_scores=y_pred_flat, gain_scores=y_true_flat, user_ids=user_ids)
        ideal_dcg = self._grouped_dcg(rank_scores=y_true_flat, gain_scores=y_true_flat, user_ids=user_ids)

        return dcg / (ideal_dcg + 1e-8)

    def _grouped_dcg(self, rank_scores, gain_scores, user_ids):
        """Per-user DCG@at_k: within each user group rank items by `rank_scores` (descending) and
        sum the discounted gains (2 ** x - 1) of the top at_k items' `gain_scores`.

        Vectorized and batch-size agnostic (graph-safe):
        1. `tf.unique` maps user ids to dense cluster ids (0..U-1), usable directly as segment ids.
        2. Two stable argsorts (by rank score desc, then by cluster id) yield a global order in
           which each user's items are contiguous and sorted by rank score desc.
        3. The rank of an item within its user is its position minus the group's start position
           (group starts come from the cumulative group sizes).
        4. Items with rank >= at_k are zeroed out (users with fewer items simply contribute
           fewer terms), then `tf.math.segment_sum` aggregates the discounted gains per user.
        """
        n = tf.shape(rank_scores)[0]
        _, cluster_ids = tf.unique(user_ids)

        # global order: sorted by cluster id, ties broken by rank score descending (stable)
        order_by_score = tf.argsort(rank_scores, direction="DESCENDING", stable=True)
        order = tf.gather(order_by_score, tf.argsort(tf.gather(cluster_ids, order_by_score), stable=True))

        gains_sorted = tf.gather(gain_scores, order)
        cluster_sorted = tf.gather(cluster_ids, order)

        # 0-based rank of each item within its user in the sorted order
        positions = tf.range(n, dtype=tf.int32)
        group_sizes = tf.math.segment_sum(tf.ones_like(positions), cluster_sorted)
        group_starts = tf.concat([tf.constant([0], dtype=tf.int32), tf.cumsum(group_sizes)[:-1]], axis=0)
        rank_in_group = positions - tf.gather(group_starts, cluster_sorted)

        # only the top at_k items of each user contribute to the DCG
        ranks = tf.cast(rank_in_group, dtype=gains_sorted.dtype)
        gains = tf.pow(2.0, gains_sorted) - 1.0
        discounts = tf.math.log(ranks + 2.0) / tf.cast(tf.math.log(2.0), ranks.dtype)
        dcg_terms = tf.where(rank_in_group < self.at_k, gains / discounts, tf.zeros_like(gains_sorted))

        return tf.math.segment_sum(dcg_terms, cluster_sorted)

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
    The model is evaluated using MSE, MAE and NDCG metrics, where the NDCG is computed by generating predictions for the test set and comparing them to the true ratings, grouped by user.
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
        user_train_split, user_ids_train_split, user_test_split, user_ids_test_split = user_train_split[:, 1:], user_train_split[:, 0], user_test_split[:, 1:], user_test_split[:, 0]

        # skip user ids
        movie_train_data = movie_train_data[:, 1:] # skip movie ids

        # Scale movie features (analogous to user scaling) so they're on the same scale as embeddings before concatenation
        movie_scaler_layer = tf.keras.layers.Normalization(axis=-1)
        movie_scaler_layer.adapt(movie_train_split)

        # Neural Network architecture
        



        
        # create the user input and point to the base network
        user_input = tf.keras.layers.Input(shape=(user_train_split.shape[1],), name = 'user_input_layer')
        user_seq = tf.keras.models.Sequential([
            user_scaler_layer,
            tf.keras.layers.Dense(units = 128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(units = self.num_outputs, activation = 'linear')
        ])
        user_output = L2Norm(axis=1)(user_seq(user_input), name = 'user_l2_normalization_layer')
        
        
        # create the item input and point to the base network
        movie_cat_features_input = tf.keras.layers.Input(shape=(movie_train_split.shape[1],), name = 'movie_cat_features_input_layer')
        movie_embedding_input = tf.keras.layers.Input(shape=(embedding_train_split.shape[1],), name = 'movie_embedding_input_layer')

        
        movie_embedding_preprocessing = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(units = 32, activation = 'relu'),
            tf.keras.layers.Normalization(axis=-1)
        ])

        movie_embedding_preprocessing_output = movie_embedding_preprocessing(movie_embedding_input)

        # Genre-aware gate: learns per-movie scalar to suppress/amplify embeddings based on genre
        # Low bias init: sigmoid(-1)≈0.27 → gate starts at 0.1+0.9*0.27≈0.34,
        # so embeddings start at ~34% influence and the gate only learns to
        # amplify (up to 1.0) where the loss says it is important
        embedding_gate = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            bias_initializer=tf.constant_initializer(-1.0),
            name='embedding_gate'
        )(movie_cat_features_input[:, :])
        embedding_gate = 0.1 + embedding_gate * 0.9  # scale to [0.1, 1.0]
        gated_embedding = embedding_gate * movie_embedding_preprocessing_output

        movie_scaled_features = movie_scaler_layer(movie_cat_features_input)
        
        movie_features_concatenation = tf.keras.layers.Concatenate(axis = -1, name = "movie_embedding_concatenate_layer")(
            [
                movie_scaled_features,
                gated_embedding
            ]
        )
        movie_seq = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(units = 64, activation = 'relu'),
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
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
            "loss": {
                "y_scaled": tf.keras.losses.MeanSquaredError(), # dummy loss function because Model.fit() expects a loss defined in compile() even though we compute the loss inside the custom train_step()
            },
            "metrics": {
                "y_scaled": [tf.keras.metrics.MeanSquaredError(name="tf_mse"), tf.keras.metrics.MeanAbsoluteError(name="tf_mae")],
                "y_unscaled": []
            },
            #"run_eagerly": True #uncomment for eager execution (slower but allows for step-by-step debugging and printing of intermediate values in the train_step)
        }
        rec_model.compile(**compile_args)

        epoch_metrics_callback = EpochMetricsCallback()

        rec_model.fit(
            [user_train_split, movie_train_split, embedding_train_split, movie_ids_train_split, user_ids_train_split],
            {
                "y_scaled": y_scaler(y_train_split).numpy(),
                "y_unscaled": y_train_split,
            },
            epochs=self.num_epochs,
            verbose=self.verbose_level,
            batch_size = self.batch_size,
            callbacks=[epoch_metrics_callback]
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

        def extract_architecture_summary(model: tf.keras.Model) -> str:
            """Capture the standard Keras model.summary() table as a string."""
            summary = []
            model.summary(print_fn=lambda x: summary.append(x))
            return "\n".join(summary)

        return {
            "metrics": metrics,
            "architecture_summary": extract_architecture_summary(model),
            "epoch_metrics": epoch_metrics_callback.epoch_metrics,
        }

    def _evaluate_trained_model(self, model: TrainingKerasModel, movie_test_split, embedding_test_split, user_test_split, user_ids_test_split, y_test_split):
        """
        Evaluate the trained model using MSE, MAE and NDCG metrics. The NDCG is computed by generating predictions for the test set and comparing them to the true ratings, grouped by user.
        :param model: The trained Keras model to evaluate
        :param movie_test_split: Movie features for the test set
        :param embedding_test_split: Movie embeddings for the test set
        :param user_test_split: User features for the test set
        :param user_ids_test_split: User IDs for the test set (used for NDCG calculation)
        :param y_test_split: True ratings for the test set (unscaled!)
        :return: A dictionary containing the computed metrics (MSE, MAE, NDCG) for the test set
        """

        # NDCG is a per-user metric: every rating of a user must be evaluated
        # together. The test split is shuffled, so small batches fragment each
        # user's ratings across batches (the NDCG of a 1-item fragment is 1.0)
        # and inflate the metric toward 1.0. A batch size larger than the whole
        # test set puts all rows in a single batch, so every user is scored on
        # their full rating list. NOTE: if the test split can ever exceed this
        # many rows, users split across batches would fragment again — raise
        # the limit if that happens.
        metrics = model.evaluate(
            [user_test_split, movie_test_split, embedding_test_split, user_ids_test_split],
            y_test_split,
            batch_size=100_000,
            return_dict=True,
            verbose=2
        )
        
        metrics = {k: v.numpy() for k, v in metrics.items()}
        preds = model.predict([user_test_split, movie_test_split, embedding_test_split])
        ndcg_np = self._compute_ndcg_offline(
            y_test_split,
            preds,
            user_ids_test_split,
            k=model.at_k
        )
        metrics["ndcg_numpy"] = ndcg_np

        return metrics


    @staticmethod
    def _compute_ndcg_offline(y_true, y_pred, user_ids, k=10):
        """Compute NDCG@k offline by generating predictions for the test set and comparing them to the true ratings, grouped by user. This function assumes that the input y_true, y_pred and user_ids are aligned (i.e. the i-th element of each corresponds to the same user-movie pair) and that user_ids contains the user identifiers for each prediction, which are necessary for grouping the predictions by user."""
        
        # NDCG on RAW (unscaled) ratings — see the note in test_step(): the
        # exponential gain is only well defined on non-negative relevance grades
        y_true_raw = y_true['y_unscaled'].flatten()
        y_pred_raw = y_pred['y_unscaled'].flatten()

        # group by user
        unique_users = np.unique(user_ids)
        ndcgs = []

        for user_id in unique_users:
            mask = user_ids == user_id
            y_t = y_true_raw[mask]
            y_p = y_pred_raw[mask]

            # if there are less than k movies for this user still caculate the metric but adjust k to the number of ratings for this user
            # NOTE: use a per-user variable — assigning back to `k` used to leak
            # the reduced cutoff into every subsequent user's NDCG
            k_eff = min(k, len(y_t))

            # sort by prediction
            order = np.argsort(-y_p, kind="stable")
            y_true_sorted_by_prediction = y_t[order][:k_eff]

            gains = (2 ** y_true_sorted_by_prediction - 1)
            discounts = np.log2(np.arange(2, k_eff + 2))
            dcg = np.sum(gains / discounts)

            # ideal
            y_ideal_sorted = np.sort(y_t)[::-1][:k_eff]
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
        
        return [self._predict_single(request['user_preferences'], request['movies'], request['embeddings'], params) for _, request in model_input.iterrows()]
        
        
    def _predict_single(self, user_preferences: dict[str, float], movies_matrix: numpy.ndarray, embedding_matrix: numpy.ndarray, params: dict[str, int|float]) -> list[tuple[int, float]]:
        user_preferences = user_preferences or {}
        params = params or {}
        
        assert isinstance(user_preferences, dict)
        assert isinstance(params, dict)
        assert isinstance(movies_matrix, numpy.ndarray)
        assert isinstance(embedding_matrix, numpy.ndarray)
        
        # parameters
        limit = params.get("limit", 50)

        logger.info("Predicting top %d movies for the following preferences: %s", limit, user_preferences)
        
        user_preferences_model = UserPreferences(**user_preferences)
        user_vector = np.array(user_preferences_model.to_list())

        # generate and replicate the user's vector to match the movies' vector shape
        user_matrix = np.tile(user_vector, (movies_matrix.shape[0], 1))

        # make a prediction
        y_p = self._tf_model.predict([user_matrix, movies_matrix[:, 1:], embedding_matrix], verbose=0)

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


    