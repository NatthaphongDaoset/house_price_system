"""Validate the raw King County housing data before downstream jobs run."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import DataValidationError, validate_csv

DEFAULT_DATA_PATH = Path("data/raw/kc_house_data.csv")
DEFAULT_REPORT_PATH = Path("artifacts/data_validation_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV to validate (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"JSON evidence report (default: {DEFAULT_REPORT_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        # A critical error is converted to exit code 1 for Airflow and CI.
        _, report = validate_csv(args.input, report_path=args.report)
    except DataValidationError as exc:
        print("DATA_VALIDATION_ALERT: critical checks failed", file=sys.stderr)
        print(json.dumps(exc.report.errors[:10], indent=2, default=str), file=sys.stderr)
        print(f"Evidence report: {args.report}", file=sys.stderr)
        return 1

    # Exit code 0 means downstream tasks are allowed to continue.
    print(
        f"DATA_VALIDATION_PASSED: {report.row_count} rows, "
        f"{report.column_count} columns"
    )
    if report.warnings:
        print("Warnings for manual review:")
        print(json.dumps(report.warnings, indent=2, default=str))
    print(f"Evidence report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
