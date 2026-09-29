import numpy as np
import pandas as pd
import pytest

from src.features import get_feature_names
from src.preprocessing import (
    build_preprocessor,
    clean_dataframe,
    encode_smoker,
    missing_values,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, 46, 33],
            "sex": ["female", "male", "male"],
            "bmi": [27.9, 33.1, 33.0],
            "children": [0, 1, 3],
            "smoker": ["yes", "no", "yes"],
            "region": ["southwest", "southeast", "southeast"],
            "charges": [16884.92, 4720.0, 21984.47],
        }
    )


def test_encode_smoker_yes_no():
    df = pd.DataFrame({"smoker": ["yes", "no", "YES", "No"]})
    encoded = encode_smoker(df)
    assert encoded["smoker"].tolist() == [1, 0, 1, 0]
    assert pd.api.types.is_numeric_dtype(encoded["smoker"])


def test_encode_smoker_keeps_numeric_and_does_not_mutate_input():
    df = pd.DataFrame({"smoker": [1, 0, 1]})
    encoded = encode_smoker(df)
    assert encoded["smoker"].tolist() == [1, 0, 1]
    assert df["smoker"].tolist() == [1, 0, 1]


def test_encode_smoker_missing_column_raises():
    with pytest.raises(KeyError):
        encode_smoker(pd.DataFrame({"age": [30]}))


def test_missing_values(sample_df):
    sample_df.loc[0, "bmi"] = np.nan
    counts = missing_values(sample_df)
    assert counts["bmi"] == 1
    assert counts.drop("bmi").sum() == 0


def test_build_preprocessor_output(sample_df):
    pre = build_preprocessor()
    transformed = pre.fit_transform(sample_df)
    names = get_feature_names(pre)

    assert transformed.shape == (len(sample_df), len(names))
    assert "sex_male" in names
    assert "region_southwest" in names
    assert {"age", "bmi", "children"} <= set(names)
    assert "smoker" not in names


def test_build_preprocessor_drop_first_no_dummy_trap():
    df = pd.DataFrame(
        {
            "sex": ["male", "female", "male"],
            "smoker": ["yes", "no", "yes"],
            "region": ["north", "south", "north"],
            "age": [20, 30, 40],
            "bmi": [22.0, 30.0, 26.0],
            "children": [0, 1, 2],
        }
    )
    names = get_feature_names(build_preprocessor().fit(df))
    assert len([n for n in names if n.startswith("sex_")]) == 1
    assert len([n for n in names if n.startswith("region_")]) == 1


def test_clean_dataframe_orders_columns(sample_df):
    shuffled = sample_df[
        ["charges", "region", "smoker", "children", "bmi", "sex", "age"]
    ]
    cleaned = clean_dataframe(shuffled)
    assert list(cleaned.columns) == [
        "sex",
        "smoker",
        "region",
        "age",
        "bmi",
        "children",
        "charges",
    ]
    assert cleaned["smoker"].tolist() == [1, 0, 1]
    assert len(cleaned) == 3
