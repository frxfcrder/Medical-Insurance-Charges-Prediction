import numpy as np
import pandas as pd
import pytest

from src.data_loader import split_features_target
from src.evaluate import (
    baseline_metrics,
    evaluate_models,
    format_metrics_table,
    regression_metrics,
)
from src.predict import load_model, predict_charges, predict_single, save_model
from src.preprocessing import clean_dataframe
from src.train import LINEAR, RANDOM_FOREST, RANDOM_FOREST_LOG, split_data, train_all

EXPECTED_MODELS = {LINEAR, RANDOM_FOREST, RANDOM_FOREST_LOG}


@pytest.fixture
def insurance_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    age = rng.integers(18, 64, n)
    bmi = rng.uniform(18, 45, n)
    children = rng.integers(0, 4, n)
    smoker = rng.choice(["yes", "no"], n, p=[0.3, 0.7])
    charges = (
        250 * age
        + 30 * bmi
        + 4000 * (smoker == "yes").astype(int)
        + 500 * children
        + rng.normal(0, 1000, n)
        + 1000
    )
    return pd.DataFrame(
        {
            "age": age,
            "sex": rng.choice(["male", "female"], n),
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": rng.choice(
                ["northwest", "northeast", "southeast", "southwest"], n
            ),
            "charges": charges,
        }
    )


@pytest.fixture
def split(insurance_df):
    df = clean_dataframe(insurance_df)
    X, y = split_features_target(df)
    return split_data(X, y)


def test_split_shapes(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 320 and len(X_test) == 80
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)
    assert "charges" not in X_train.columns


def test_train_all_fits_every_model(split):
    X_train, _, y_train, _ = split
    models = train_all(X_train, y_train)
    assert EXPECTED_MODELS <= set(models)
    for pipeline in models.values():
        assert hasattr(pipeline, "predict")


def test_metrics_capture_signal(split):
    X_train, X_test, y_train, y_test = split
    models = train_all(X_train, y_train)
    table = evaluate_models(models, X_test, y_test, y_train=y_train)

    assert list(table.columns) == ["Model", "MAE", "RMSE", "R2 Score"]
    assert table.iloc[0]["Model"] == "Baseline"
    assert table["Model"].tolist()[1:] == list(models)

    linear = table.loc[table["Model"] == LINEAR, "R2 Score"].item()
    best = table.drop(0)["R2 Score"].max()
    assert linear > 0.7
    assert best > 0.8
    assert table.loc[0, "R2 Score"] < 0.1


def test_metrics_values(split):
    X_train, X_test, y_train, y_test = split
    models = train_all(X_train, y_train)
    metrics = regression_metrics(y_test, models[LINEAR].predict(X_test))

    assert set(metrics) == {"MAE", "RMSE", "R2 Score"}
    assert metrics["RMSE"] >= metrics["MAE"] > 0
    assert 0 < metrics["R2 Score"] <= 1


def test_baseline_metrics(split):
    _, _, y_train, y_test = split
    metrics = baseline_metrics(y_train, y_test)
    assert np.isclose(metrics["MAE"], np.mean(np.abs(y_test - y_train.mean())))
    assert metrics["R2 Score"] <= 0


def test_predictions_are_positive_and_well_shaped(split):
    X_train, X_test, y_train, _ = split
    models = train_all(X_train, y_train)

    for name, pipeline in models.items():
        preds = predict_charges(pipeline, X_test)
        assert preds.shape == (len(X_test),), name
        assert (preds > 0).all(), name


def test_predict_single_matches_dataframe_prediction(split):
    X_train, X_test, y_train, _ = split
    pipeline = train_all(X_train, y_train)[RANDOM_FOREST]

    record = X_test.iloc[0].to_dict()
    assert predict_single(pipeline, record) == pytest.approx(
        predict_charges(pipeline, X_test.iloc[:1])[0]
    )


def test_predict_charges_requires_all_features(split):
    X_train, X_test, y_train, _ = split
    pipeline = train_all(X_train, y_train)[LINEAR]
    with pytest.raises(KeyError, match="bmi"):
        predict_charges(pipeline, X_test.drop(columns=["bmi"]))


def test_save_load_roundtrip(split, tmp_path):
    X_train, X_test, y_train, _ = split
    pipeline = train_all(X_train, y_train)[RANDOM_FOREST_LOG]

    path = save_model(pipeline, tmp_path / "model.joblib")
    restored = load_model(path)
    np.testing.assert_allclose(
        predict_charges(restored, X_test), predict_charges(pipeline, X_test)
    )


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_format_metrics_table(split):
    X_train, X_test, y_train, y_test = split
    models = train_all(X_train, y_train)
    table = format_metrics_table(evaluate_models(models, X_test, y_test))

    assert table.index.name == "Model"
    assert table["MAE"].str.startswith("$").all()
    assert table["R2 Score"].str.match(r"^-?\d\.\d{4}$").all()
