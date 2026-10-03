"""Validate and split the raw house-sale dataset by sale date."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_PATH,
    DEFAULT_REJECTED_DIR,
    DataIngestionError,
    ingest_and_split,
)
from src.data_validation import DataValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rejected-dir", type=Path, default=DEFAULT_REJECTED_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("artifacts/data_validation_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        # One function runs validation, splitting, saving, and manifest creation.
        manifest = ingest_and_split(
            raw_path=args.input,
            output_dir=args.output_dir,
            rejected_dir=args.rejected_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            validation_report_path=args.validation_report,
        )
    except (DataValidationError, DataIngestionError) as exc:
        print(f"DATA_INGESTION_ALERT: {exc}", file=sys.stderr)
        return 1

    summary = {}
    for split_name, split_details in manifest["splits"].items():
        summary[split_name] = split_details["rows"]
    print("DATA_INGESTION_PASSED")
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {args.output_dir / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
