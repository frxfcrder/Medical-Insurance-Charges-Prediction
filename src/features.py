from __future__ import annotations

import pandas as pd

TARGET = "charges"
CATEGORICAL_COLS = ["sex", "smoker", "region"]
NUMERICAL_COLS = ["age", "bmi", "children"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLS].copy()


def select_target(df: pd.DataFrame, target: str = TARGET) -> pd.Series:
    if target not in df.columns:
        raise KeyError(f"Missing target column: {target!r}")
    return df[target].copy()


def get_feature_names(preprocessor) -> list[str]:
    names = preprocessor.get_feature_names_out()
    return [
        str(name).replace("cat__", "").replace("remainder__", "") for name in names
    ]


def feature_importance(pipeline, top_n: int | None = None) -> pd.Series:
    model = pipeline[-1]
    if hasattr(model, "regressor_"):
        model = model.regressor_
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} does not expose feature_importances_"
        )
    importances = pd.Series(
        model.feature_importances_,
        index=get_feature_names(pipeline[:-1]),
        name="importance",
    ).sort_values(ascending=False)
    return importances.head(top_n) if top_n else importances
