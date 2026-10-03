"""FastAPI service for house-price prediction."""
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import load_model

MODEL_PATH = Path("artifacts/serving_model/house_price_model.joblib")
app = FastAPI(title="House Price Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained yet")

    model = load_model(str(MODEL_PATH))
    input_data = pd.DataFrame([request.features])
    prediction = float(model.predict(input_data)[0])
    return {"predicted_price": prediction}
