from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_COLS, FEATURE_COLS, TARGET


def encode_smoker(df: pd.DataFrame, column: str = "smoker") -> pd.DataFrame:
    out = df.copy()
    mapped = (
        out[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    out[column] = mapped.fillna(pd.to_numeric(out[column], errors="coerce"))
    return out


def missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = encode_smoker(df)
    ordered = [col for col in FEATURE_COLS + [TARGET] if col in out.columns]
    extra = [col for col in out.columns if col not in ordered]
    return out[ordered + extra]


def build_preprocessor(categorical_cols: list[str] | None = None) -> ColumnTransformer:
    categorical_cols = CATEGORICAL_COLS if categorical_cols is None else categorical_cols
    return ColumnTransformer(
        [("cat", OneHotEncoder(drop="first"), categorical_cols)],
        remainder="passthrough",
    )
