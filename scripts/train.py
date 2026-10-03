"""Train the house-price model when the team's real dataset is available."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.model import build_pipeline, save_model

DATA_PATH = Path("data/raw/house_prices.csv")
MODEL_PATH = Path("artifacts/serving_model/house_price_model.joblib")
TARGET = "SalePrice"

if not DATA_PATH.exists():
    print("Dataset is not available yet. Training is skipped.")
    raise SystemExit(0)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = build_pipeline(X_train)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

save_model(model, str(MODEL_PATH))

print("Training complete")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")
print(f"Model saved to: {MODEL_PATH}")
