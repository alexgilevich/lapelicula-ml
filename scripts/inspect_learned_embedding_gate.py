"""
Inspect the LEARNED embedding gate weights from a trained model artifact.

The gate is a Dense(1, sigmoid) layer over all 30 raw movie features
(11 sentiments + 18 genres + adult), rescaled to [0.1, 1.0]:

    gate = 0.1 + 0.9 * sigmoid(features @ W + b)

It is initialized low (bias = -1.0 → sigmoid ≈ 0.27 → gate start ≈ 0.34)
so embeddings start at ~34% influence and the gate only learns to amplify
toward 1.0 where the loss says embedding signal is useful.

Run after step03 with the artifact in model_artifacts/mlflow_models/<name>
(update DEFAULT_MLFLOW_MODEL_NAME if you saved it under a different name).
"""
import os

import numpy as np
import mlflow
import tensorflow as tf

DEFAULT_MODEL_LOCAL_SAVE_PATH = "../model_artifacts/mlflow_models/"
DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-movie-recommender-model-new"

SENTIMENTS = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Kids', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
FEATURES = SENTIMENTS + GENRES + ["adult"]


def movie_vector(*genres: str, adult: bool = False, **sentiments: float) -> np.ndarray:
    """Build a 30-dim raw movie feature vector: [11 sentiments, 18 genre one-hots, adult].

    Sentiments default to 0.0; genre bits are 1.0 for the given genres.
    """
    vec = np.zeros(30, dtype=np.float32)
    for name, value in sentiments.items():
        vec[SENTIMENTS.index(name)] = value
    for g in genres:
        vec[len(SENTIMENTS) + GENRES.index(g)] = 1.0
    if adult:
        vec[-1] = 1.0
    return vec


def main():
    model_path = os.path.join(DEFAULT_MODEL_LOCAL_SAVE_PATH, DEFAULT_MLFLOW_MODEL_NAME)
    print(f"Loading model from: {model_path}")
    model = mlflow.pyfunc.load_model(model_path)

    base = model._model_impl.python_model._tf_model.base_model

    # --- 1. Gate weights ---
    gate_layer = base.get_layer("embedding_gate")
    kernel, bias = gate_layer.get_weights()  # kernel: (30, 1), bias: (1,)

    print("\n=== EMBEDDING GATE WEIGHTS (30 features) ===")
    print(f"Kernel shape: {kernel.shape}, Bias shape: {bias.shape}")
    print(f"Bias value: {bias[0]:.4f} (init was -1.0 → sigmoid ≈ 0.27 → gate start ≈ 0.34)")

    # Per-feature weight, sorted by value (most negative = most suppressing)
    weights_flat = kernel.flatten()
    sorted_idx = np.argsort(weights_flat)
    print("\nPer-feature weights (most negative = most suppressing):")
    for i in sorted_idx:
        w = weights_flat[i]
        bar = "+" * int(abs(w) * 20) if w > 0 else "-" * int(abs(w) * 20)
        print(f"  {FEATURES[i]:15s}  {w:+.4f}  {bar}")

    # --- 2. Compute gate values for test profiles ---
    print("\n=== GATE VALUES FOR TEST MOVIE PROFILES ===")
    # Synthetic raw feature vectors: genre one-hots, sentiments 0, adult as noted
    test_profiles = {
        "Kids only":                    movie_vector("Kids"),
        "Kids + Animation":             movie_vector("Kids", "Animation"),
        "Animation only":               movie_vector("Animation"),
        "Horror only":                  movie_vector("Horror"),
        "Horror + Thriller (adult)":    movie_vector("Horror", "Thriller", adult=True),
        "Crime + Drama (adult)":        movie_vector("Crime", "Drama", adult=True),
        "Crime + Drama + Western":      movie_vector("Crime", "Drama", "Western"),
        "Sci-Fi + Action":              movie_vector("Sci-Fi", "Action"),
        "Mystery + Drama":              movie_vector("Mystery", "Drama"),
        "Comedy + Romance":             movie_vector("Comedy", "Romance"),
        "War only (adult)":             movie_vector("War", adult=True),
        "Western only":                 movie_vector("Western"),
        "Drama only":                   movie_vector("Drama"),
        "Drama only (adult)":           movie_vector("Drama", adult=True),
    }

    k = kernel.flatten()  # (30,)
    b = bias[0]  # scalar

    print(f"\n{'Profile':30s}  {'Logit':>8s}  {'Sigmoid':>8s}  {'Gate [0.1-1]':>12s}")
    print("-" * 62)
    for name, vec in test_profiles.items():
        logit = float(vec @ k + b)
        sig = float(1.0 / (1.0 + np.exp(-logit)))
        gate = 0.1 + 0.9 * sig
        print(f"{name:30s}  {logit:+8.4f}  {sig:8.4f}  {gate:12.4f}")

    # --- 3. Check embedding preprocessing output scale ---
    print("\n=== EMBEDDING SCALE CHECK ===")
    norm_layers = [l for l in base.layers if isinstance(l, tf.keras.layers.Normalization)]
    for nl in norm_layers:
        print(f"Normalization layer '{nl.name}': mean shape={nl.mean.shape}, "
              f"mean[:5]={nl.mean.numpy().flatten()[:5] if nl.mean is not None else 'NOT ADAPTED'}")

    # Check the embedding preprocessing Dense layer weights scale
    for layer in base.layers:
        if "dense" in layer.name.lower() and layer.get_weights():
            w = layer.get_weights()[0]
            print(f"Layer '{layer.name}': weight shape={w.shape}, "
                  f"mean={np.mean(w):.4f}, std={np.std(w):.4f}, max_abs={np.max(np.abs(w)):.4f}")


if __name__ == "__main__":
    main()
