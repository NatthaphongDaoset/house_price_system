"""Reusable house-price model utilities."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(data: pd.DataFrame):
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)
