# House Price System

Machine Learning Engineering / MLOps project for house-price prediction.

## Current stage

The project is prepared so the team can develop the MLOps system before the
real dataset arrives. The real CSV will be placed at:

`data/raw/house_prices.csv`

The expected target column is `SalePrice`. If the team's dataset uses a
different target name, update `config/config.yaml` and the training scripts.

## Workflow

```text
Real Dataset
    ↓
data/raw/house_prices.csv
    ↓
Data Validation
    ↓
Preprocessing
    ↓
Random Forest Regression
    ↓
MAE / RMSE / R²
    ↓
artifacts/serving_model/house_price_model.joblib
    ↓
FastAPI /predict
    ↓
Monitoring + CI/CD
```

## Run after receiving the dataset

```bash
pip install -r requirements.txt
python scripts/01_data_validation.py
python scripts/train.py
uvicorn serving.app:app --reload
```

API health check:

`GET /health`

Prediction:

`POST /predict`

Example body:

```json
{
  "features": {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2
  }
}
```

The feature names in the request must match the columns used by the real
training dataset.

## CI

GitHub Actions checks code quality and tests. Dataset-dependent steps are
skipped safely until the real dataset is supplied.
