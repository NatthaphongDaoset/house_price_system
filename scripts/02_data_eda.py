"""Generate reproducible EDA evidence without changing the raw dataset."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_eda import write_eda_evidence
from src.data_ingestion import DEFAULT_RAW_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"EDA_ALERT: dataset not found: {args.input}", file=sys.stderr)
        return 1
    # EDA reads the raw CSV but never overwrites it.
    data = pd.read_csv(args.input)
    summary = write_eda_evidence(data, args.output_dir)
    print("EDA_EVIDENCE_CREATED")
    print(json.dumps(summary["approved_cleaning"], indent=2))
    print(f"Evidence directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
