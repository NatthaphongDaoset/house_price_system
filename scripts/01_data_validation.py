"""Basic validation for the real house-price CSV."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/house_prices.csv")


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    data = pd.read_csv(DATA_PATH)
    print("Rows:", len(data))
    print("Columns:", len(data.columns))
    print("Missing values:")
    print(data.isnull().sum())


if __name__ == "__main__":
    main()
