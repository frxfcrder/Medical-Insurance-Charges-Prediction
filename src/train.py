from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

TEST_SIZE = 0.2
RANDOM_STATE = 42
BASELINE = "Baseline"
LINEAR = "Linear Regression"
RANDOM_FOREST = "Random Forest"
RANDOM_FOREST_LOG = "Random Forest (Log-Transform)"
XGBOOST = "XGBoost"


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def baseline_prediction(y_train: pd.Series | np.ndarray, n_samples: int) -> np.ndarray:
    return np.full(n_samples, float(np.mean(y_train)))


def _pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def _forest(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )


def build_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    pipelines = {
        LINEAR: _pipeline(LinearRegression()),
        RANDOM_FOREST: _pipeline(_forest(random_state)),
        RANDOM_FOREST_LOG: _pipeline(
            TransformedTargetRegressor(
                regressor=_forest(random_state), func=np.log, inverse_func=np.exp
            )
        ),
    }
    if XGBRegressor is not None:
        pipelines[XGBOOST] = _pipeline(
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
            )
        )
    return pipelines


def train_all(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> dict[str, Pipeline]:
    return {
        name: pipeline.fit(X_train, y_train)
        for name, pipeline in build_pipelines(random_state).items()
    }
