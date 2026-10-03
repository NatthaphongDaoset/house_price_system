"""Train the house-price model when the team's real dataset is available."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.model import build_pipeline, save_model


DATA_PATH = Path("data/raw/house_prices.csv")
MODEL_PATH = Path("artifacts/serving_model/house_price_model.joblib")
TARGET_COLUMN = "price"


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    data = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = build_pipeline(x_train)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    save_model(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
