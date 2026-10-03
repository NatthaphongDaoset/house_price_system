"""Prepare the data directory for the team's real dataset."""

from pathlib import Path


DATA_PATH = Path("data/raw/house_prices.csv")


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
    else:
        print(
            "No dataset downloaded automatically. "
            "Place the team's house_prices.csv in data/raw/."
        )


if __name__ == "__main__":
    main()
