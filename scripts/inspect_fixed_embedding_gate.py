"""
Show the fixed per-genre embedding gate used by the model.
The gate is no longer learned — it is a fixed table (GENRE_EMBEDDING_SCALES in
src/model.py), so this script prints the scale each genre profile gets and, if a
trained artifact exists, verifies the saved model uses the same table.
"""
import os
import sys

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/')
sys.path.append(script_dir)

import numpy as np
from model import GENRE_ORDER, GENRE_EMBEDDING_SCALES, FixedGenreEmbeddingScale

DEFAULT_MODEL_LOCAL_SAVE_PATH = "../model_artifacts/mlflow_models/"
DEFAULT_MLFLOW_MODEL_NAME = "lapelicula-movie-recommender-model-new"

TEST_PROFILES = [
    ("Kids only",                ["Kids"]),
    ("Kids + Animation",         ["Kids", "Animation"]),
    ("Animation only",           ["Animation"]),
    ("Horror only",              ["Horror"]),
    ("Horror + Thriller",        ["Horror", "Thriller"]),
    ("Crime + Drama",            ["Crime", "Drama"]),
    ("Crime + Drama + Western",  ["Crime", "Drama", "Western"]),
    ("Sci-Fi + Action",          ["Sci-Fi", "Action"]),
    ("Mystery + Drama",          ["Mystery", "Drama"]),
    ("Comedy + Romance",         ["Comedy", "Romance"]),
    ("War only",                 ["War"]),
    ("Western only",             ["Western"]),
    ("Drama only",               ["Drama"]),
]


def gate_for(genres: list[str]) -> float:
    """Most restrictive present genre wins; no genres -> 1.0 (no suppression)."""
    if not genres:
        return 1.0
    return min(GENRE_EMBEDDING_SCALES[GENRE_ORDER.index(g)] for g in genres)


def main():
    print("=== FIXED EMBEDDING GATE SCALES (per genre) ===")
    for genre, scale in zip(GENRE_ORDER, GENRE_EMBEDDING_SCALES):
        bar = "+" * int(scale * 20)
        print(f"  {genre:15s}  {scale:4.2f}  {bar}")

    print("\n=== GATE VALUES FOR TEST GENRE PROFILES ===")
    print(f"{'Profile':30s}  {'Gate [0-1]':>12s}")
    print("-" * 44)
    for name, genres in TEST_PROFILES:
        print(f"{name:30s}  {gate_for(genres):12.4f}")

    # --- Verify the trained artifact uses the same fixed table ---
    model_path = os.path.join(DEFAULT_MODEL_LOCAL_SAVE_PATH, DEFAULT_MLFLOW_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"\nModel artifact not found at {model_path} — skipping artifact check.")
        return

    import mlflow
    print(f"\n=== ARTIFACT CHECK ===")
    print(f"Loading model from: {model_path}")
    model = mlflow.pyfunc.load_model(model_path)
    base = model._model_impl.python_model._tf_model.base_model

    gate_layers = [l for l in base.layers if isinstance(l, FixedGenreEmbeddingScale)]
    if not gate_layers:
        print("WARNING: no FixedGenreEmbeddingScale layer found — this artifact was "
              "trained with an OLDER model version (learned gate). Retrain to pick up the fixed gate.")
        return

    gate_layer = gate_layers[0]
    saved_scales = gate_layer.scales
    print(f"Gate layer '{gate_layer.name}' found. Saved scales match current table: "
          f"{saved_scales == GENRE_EMBEDDING_SCALES}")

    # Sanity check: run the saved gate on a Kids one-hot and compare with gate_for()
    kids_vec = np.zeros((1, 18), dtype=np.float32)
    kids_vec[0, GENRE_ORDER.index("Kids")] = 1.0
    print(f"Saved gate output for Kids one-hot: {float(gate_layer(kids_vec)[0, 0]):.4f} "
          f"(expected {gate_for(['Kids']):.4f})")


if __name__ == "__main__":
    main()
