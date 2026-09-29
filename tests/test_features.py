import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS,
    TARGET,
    feature_importance,
    get_feature_names,
    select_features,
    select_target,
)
from src.preprocessing import build_preprocessor
from src.train import RANDOM_FOREST, build_pipelines


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, 46, 33, 60],
            "sex": ["female", "male", "male", "female"],
            "bmi": [17.0, 22.5, 27.5, 34.0],
            "children": [0, 1, 3, 2],
            "smoker": ["yes", "no", "yes", "no"],
            "region": ["southwest", "southeast", "northwest", "northeast"],
            "charges": [16884.92, 4720.0, 21984.47, 8000.0],
        }
    )


def test_feature_schema():
    assert FEATURE_COLS == ["sex", "smoker", "region", "age", "bmi", "children"]
    assert TARGET == "charges"


def test_select_features_excludes_target(sample_df):
    X = select_features(sample_df)
    assert list(X.columns) == FEATURE_COLS
    assert TARGET not in X.columns
    assert len(X) == len(sample_df)


def test_select_features_missing_column_raises(sample_df):
    with pytest.raises(KeyError, match="age"):
        select_features(sample_df.drop(columns=["age"]))


def test_select_target(sample_df):
    y = select_target(sample_df)
    assert y.name == TARGET
    with pytest.raises(KeyError):
        select_target(sample_df.drop(columns=[TARGET]))


def test_get_feature_names_matches_transform_shape(sample_df):
    pre = build_preprocessor().fit(sample_df)
    names = get_feature_names(pre)
    assert len(names) == pre.transform(sample_df).shape[1]
    assert "age" in names and "sex_male" in names


def test_feature_importance_sorted(sample_df):
    pipeline = build_pipelines()[RANDOM_FOREST]
    pipeline.fit(select_features(sample_df), select_target(sample_df))

    importances = feature_importance(pipeline)
    assert isinstance(importances, pd.Series)
    assert len(importances) == len(get_feature_names(pipeline.named_steps["preprocessor"]))
    assert list(importances) == sorted(importances, reverse=True)
    assert np.isclose(importances.sum(), 1.0)
    assert len(feature_importance(pipeline, top_n=2)) == 2


def test_feature_importance_unsupported_model_raises(sample_df):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("model", LinearRegression())]
    )
    pipeline.fit(select_features(sample_df), select_target(sample_df))
    with pytest.raises(AttributeError, match="feature_importances_"):
        feature_importance(pipeline)
