"""Basic validation for the real house-price CSV."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/house_prices.csv")
TARGET = "SalePrice"

if not DATA_PATH.exists():
    print("No real dataset yet. Data validation is skipped.")
    raise SystemExit(0)

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("Dataset is empty")
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found")
if df[TARGET].isna().all():
    raise ValueError(f"Target column '{TARGET}' contains no usable values")

print("Data validation passed")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Target: {TARGET}")
