"""Prepare the raw-data location.

The real dataset will be supplied by the team. This script intentionally does
not download an external dataset. It only checks whether the expected file
exists, so CI can be used before the real data is added.
"""
from pathlib import Path
import sys

DATA_PATH = Path("data/raw/house_prices.csv")

if DATA_PATH.exists():
    print(f"Dataset found: {DATA_PATH}")
else:
    print(f"Dataset is not available yet: {DATA_PATH}")
    print("Waiting for the team's real house-price dataset.")
