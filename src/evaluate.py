from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .train import BASELINE, baseline_prediction


def regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2 Score": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return regression_metrics(y, model.predict(X))


def baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    return regression_metrics(y_test, baseline_prediction(y_train, len(y_test)))


def evaluate_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    y_train: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    if y_train is not None:
        rows.append({"Model": BASELINE, **baseline_metrics(y_train, y)})
    rows += [
        {"Model": name, **evaluate_model(model, X, y)}
        for name, model in models.items()
    ]
    return pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2 Score"])


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MAE"] = out["MAE"].map("${:,.2f}".format)
    out["RMSE"] = out["RMSE"].map("${:,.2f}".format)
    out["R2 Score"] = out["R2 Score"].map("{:.4f}".format)
    return out.set_index("Model")
