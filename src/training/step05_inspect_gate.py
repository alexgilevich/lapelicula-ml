"""
Inspect the embedding gate weights from the trained model artifact.
Run after training to see what the gate learned per genre.
"""
import os
import numpy as np
import mlflow
import tensorflow as tf

DEFAULT_MODEL_LOCAL_SAVE_PATH = "../../model_artifacts/mlflow_models/"
DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-movie-recommender-model-new"

GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Kids', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def genre_vector(*genres: str) -> np.ndarray:
    """Build a one-hot genre vector (18,) for the given genre names."""
    vec = np.zeros(18, dtype=np.float32)
    for g in genres:
        idx = GENRES.index(g)
        vec[idx] = 1.0
    return vec


def main():
    model_path = os.path.join(DEFAULT_MODEL_LOCAL_SAVE_PATH, DEFAULT_MLFLOW_MODEL_NAME)
    print(f"Loading model from: {model_path}")
    model = mlflow.pyfunc.load_model(model_path)

    base = model._model_impl.python_model._tf_model.base_model

    # --- 1. Gate weights ---
    gate_layer = base.get_layer("embedding_gate")
    kernel, bias = gate_layer.get_weights()  # kernel: (18, 1), bias: (1,)

    print("\n=== EMBEDDING GATE WEIGHTS ===")
    print(f"Kernel shape: {kernel.shape}, Bias shape: {bias.shape}")
    print(f"Bias value: {bias[0]:.4f}")

    # Per-genre weight, sorted by value (most suppressing first)
    weights_flat = kernel.flatten()
    sorted_idx = np.argsort(weights_flat)
    print("\nPer-genre weights (most negative = most suppressing):")
    for i in sorted_idx:
        if i < 11 or i > 28: continue
        
        bar = "+" * int(abs(weights_flat[i]) * 20) if weights_flat[i] > 0 else "-" * int(abs(weights_flat[i]) * 20)
        print(f"  {GENRES[i - 11]:15s}  {weights_flat[i]:+.4f}  {bar}")

    # --- 2. Compute gate values for test profiles ---
    print("\n=== GATE VALUES FOR TEST GENRE PROFILES ===")
    test_profiles = {
        "Kids only":              genre_vector("Kids"),
        "Kids + Animation":       genre_vector("Kids", "Animation"),
        "Animation only":         genre_vector("Animation"),
        "Horror only":            genre_vector("Horror"),
        "Horror + Thriller":      genre_vector("Horror", "Thriller"),
        "Crime + Drama":          genre_vector("Crime", "Drama"),
        "Crime + Drama + Western": genre_vector("Crime", "Drama", "Western"),
        "Sci-Fi + Action":        genre_vector("Sci-Fi", "Action"),
        "Mystery + Drama":        genre_vector("Mystery", "Drama"),
        "Comedy + Romance":       genre_vector("Comedy", "Romance"),
        "War only":               genre_vector("War"),
        "Western only":           genre_vector("Western"),
        "Drama only":             genre_vector("Drama"),
    }

    k = kernel[11 : 11 + 18, :] # (18, 1)
    b = bias[0]  # scalar

    print(f"\n{'Profile':30s}  {'Logit':>8s}  {'Sigmoid':>8s}  {'Gate [0.1-2.0]':>15s}")
    print("-" * 65)
    for name, vec in test_profiles.items():
        logit = float(vec @ k.flatten() + b)
        sig = float(1.0 / (1.0 + np.exp(-logit)))
        gate = 0.1 + sig * 1.9
        print(f"{name:30s}  {logit:+8.4f}  {sig:8.4f}  {gate:15.4f}")

    # --- 3. Check embedding preprocessing output scale ---
    print("\n=== EMBEDDING SCALE CHECK ===")
    # Find the Normalization layer inside the embedding preprocessing
    # The embedding preprocessing is a Sequential sub-model
    # Look for Normalization layers in the base model
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
