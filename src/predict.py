from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def save_model(model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_charges(model, X: pd.DataFrame) -> np.ndarray:
    missing = [col for col in FEATURE_COLS if col not in X.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return np.asarray(model.predict(X[FEATURE_COLS]), dtype=float)


def predict_single(model, record: dict) -> float:
    return float(predict_charges(model, pd.DataFrame([record]))[0])
