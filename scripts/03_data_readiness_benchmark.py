"""Benchmark the prepared data before handing it to the model team."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_readiness import run_data_readiness_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/data_readiness"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_data_readiness_benchmark(PROJECT_ROOT, args.output_dir)
    summary = {
        "selected_model": report["selected_model"],
        "validation_results": report["validation_results"],
        "test_result": report["test_result"]["selected_model_metrics"],
        "readiness_gate": report["readiness_gate"],
    }
    print("DATA_READINESS_BENCHMARK_COMPLETED")
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0 if report["readiness_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
