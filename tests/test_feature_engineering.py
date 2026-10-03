from pathlib import Path

import numpy as np
import pandas as pd

from src.data_readiness import TabularPreprocessor, regression_metrics
from src.feature_engineering import engineer_house_features

REAL_DATA_PATH = Path("data/raw/kc_house_data.csv")


def test_feature_engineering_removes_identifier_and_adds_safe_features():
    raw = pd.read_csv(REAL_DATA_PATH, nrows=5).drop(columns=["price"])

    features = engineer_house_features(raw)

    assert "id" not in features.columns
    assert "date" not in features.columns
    assert "yr_renovated" not in features.columns
    assert features["zipcode"].dtype.name == "string"
    assert "house_age_at_sale" in features.columns
    assert "years_since_renovation" in features.columns
    assert "sqft_per_bedroom" in features.columns
    assert np.isfinite(features.select_dtypes(include="number")).all().all()


def test_regression_metrics_are_perfect_for_perfect_predictions():
    actual = pd.Series([100_000.0, 200_000.0, 300_000.0])

    metrics = regression_metrics(actual, actual.to_numpy())

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1
    assert metrics["mape_percent"] == 0
    assert metrics["within_20_percent"] == 1


def test_preprocessor_uses_train_categories_and_accepts_unseen_zipcode():
    raw = pd.read_csv(REAL_DATA_PATH, nrows=5).drop(columns=["price"])
    validation = raw.iloc[[0]].copy()
    validation["zipcode"] = 99999

    preprocessor = TabularPreprocessor().fit(raw)
    transformed = preprocessor.transform(validation)

    assert "99999" not in preprocessor.zipcode_categories
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
