"""Ridge residual features copied exactly from train_residual_ridge.py."""
from pathlib import Path
import os
import joblib
import numpy as np


def model_path() -> Path:
    return Path(os.getenv(
        "BATT_RIDGE_MODEL_PATH",
        "backend/ML/Ringe_residual_learning/residual_ridge_model.joblib"
    )).expanduser().resolve()


def make_features(cycles: np.ndarray, temperatures_C: np.ndarray) -> np.ndarray:
    n = np.asarray(cycles, dtype=float) / 1000.0
    t = (np.asarray(temperatures_C, dtype=float) - 25.0) / 15.0
    if np.any(n < 0):
        raise ValueError("Cycle count must be non-negative")
    sn = np.sqrt(n)
    return np.column_stack((sn, n, t * sn, t * n, t**2 * sn, t**2 * n))


def predict_residual(cycles: np.ndarray, temperature_C: float) -> np.ndarray:
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Trained Ridge artifact not found: {path}")
    model = joblib.load(path)
    features = make_features(
        np.asarray(cycles, dtype=float),
        np.full(len(cycles), float(temperature_C), dtype=float)
    )
    if getattr(model, "n_features_in_", 6) != 6:
        raise ValueError("Ridge artifact does not match six training features")
    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Ridge produced non-finite corrections")
    return prediction
