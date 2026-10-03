"""Validate the raw King County housing dataset before it is used.

There are two levels of checks:

* Critical errors stop the pipeline, for example a negative price, a missing
  column, or inconsistent living-area columns.
* Warnings are recorded for review but do not stop the pipeline, for example a
  repeated property ID or an unusual bedroom count.

Use :func:`validate_csv` when reading a file and :func:`validate_house_data`
when a pandas DataFrame is already available.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

# The expected order is part of the raw-data contract.
EXPECTED_COLUMNS = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


# Pandera performs column types, required fields, and value-range checks.
HOUSE_DATA_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(int, checks=pa.Check.gt(0), nullable=False),
        "date": pa.Column(
            str,
            checks=pa.Check.str_matches(r"^\d{8}T\d{6}$"),
            nullable=False,
        ),
        "price": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "bedrooms": pa.Column(
            int, checks=[pa.Check.ge(0), pa.Check.le(100)], nullable=False
        ),
        "bathrooms": pa.Column(
            float, checks=[pa.Check.ge(0), pa.Check.le(20)], nullable=False
        ),
        "sqft_living": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "sqft_lot": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "floors": pa.Column(
            float, checks=[pa.Check.gt(0), pa.Check.le(10)], nullable=False
        ),
        "waterfront": pa.Column(
            int, checks=pa.Check.isin([0, 1]), nullable=False
        ),
        "view": pa.Column(
            int, checks=pa.Check.isin([0, 1, 2, 3, 4]), nullable=False
        ),
        "condition": pa.Column(
            int, checks=pa.Check.isin([1, 2, 3, 4, 5]), nullable=False
        ),
        "grade": pa.Column(
            int, checks=[pa.Check.ge(1), pa.Check.le(13)], nullable=False
        ),
        "sqft_above": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "sqft_basement": pa.Column(
            float, checks=pa.Check.ge(0), nullable=False
        ),
        "yr_built": pa.Column(
            int, checks=[pa.Check.ge(1800), pa.Check.le(2100)], nullable=False
        ),
        "yr_renovated": pa.Column(
            int,
            checks=pa.Check(
                lambda series: (series == 0)
                | ((series >= 1800) & (series <= 2100)),
                name="zero_or_valid_year",
            ),
            nullable=False,
        ),
        "zipcode": pa.Column(
            int, checks=[pa.Check.ge(10_000), pa.Check.le(99_999)], nullable=False
        ),
        "lat": pa.Column(
            float, checks=[pa.Check.ge(-90), pa.Check.le(90)], nullable=False
        ),
        "long": pa.Column(
            float, checks=[pa.Check.ge(-180), pa.Check.le(180)], nullable=False
        ),
        "sqft_living15": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "sqft_lot15": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
    },
    checks=[
        pa.Check(
            lambda frame: ~frame.duplicated(subset=["id", "date"]),
            name="unique_property_sale",
        ),
        pa.Check(
            lambda frame: (
                frame["sqft_above"] + frame["sqft_basement"]
            ).eq(frame["sqft_living"]),
            name="living_area_components_match",
        ),
        pa.Check(
            lambda frame: (frame["yr_renovated"] == 0)
            | (frame["yr_renovated"] >= frame["yr_built"]),
            name="renovation_not_before_construction",
        ),
    ],
    strict=True,
    coerce=True,
    ordered=True,
)


@dataclass
class ValidationReport:
    """Serializable evidence produced by every validation run."""

    status: str
    dataset_path: str
    checked_at_utc: str
    row_count: int
    column_count: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)


class DataValidationError(ValueError):
    """Raised when raw data violates the critical data contract."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(
            f"Data validation failed with {len(report.errors)} critical error(s)."
        )


def _warning_summary(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return suspicious-but-allowed observations for manual review."""

    warnings: list[dict[str, Any]] = []

    # Each tuple contains a readable warning name and the matching rows.
    warning_checks = [
        ("repeated_property_id", data["id"].duplicated(keep=False)),
        ("bedrooms_above_10", data["bedrooms"] > 10),
        ("zero_bedrooms", data["bedrooms"] == 0),
        ("zero_bathrooms", data["bathrooms"] == 0),
    ]

    for check_name, matching_rows in warning_checks:
        count = int(matching_rows.sum())
        if count:
            warnings.append(
                {
                    "check": check_name,
                    "count": count,
                    "sample_rows": data.index[matching_rows].tolist()[:5],
                }
            )
    return warnings


def _schema_errors(exc: SchemaErrors) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for record in exc.failure_cases.to_dict(orient="records"):
        normalized = {
            key: (None if pd.isna(value) else value)
            for key, value in record.items()
        }
        if (
            normalized.get("check") == "column_in_dataframe"
            and normalized.get("column") is None
        ):
            normalized["column"] = normalized.get("failure_case")
        errors.append(normalized)
    return errors


def _write_report(report: ValidationReport, report_path: Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def validate_house_data(
    data: pd.DataFrame,
    *,
    dataset_path: str = "<dataframe>",
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a DataFrame and stop the caller when critical checks fail."""

    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        # lazy=True collects all schema problems instead of stopping at the first.
        validated = HOUSE_DATA_SCHEMA.validate(data, lazy=True)

        # The regular-expression check verifies the shape of a date. Parsing it
        # separately also catches impossible calendar dates such as month 13.
        parsed_dates = pd.to_datetime(
            validated["date"], format="%Y%m%dT%H%M%S", errors="coerce"
        )
        invalid_date_rows = validated.index[parsed_dates.isna()].tolist()
        if invalid_date_rows:
            raise ValueError(
                "date contains invalid calendar values at rows "
                f"{invalid_date_rows[:10]}"
            )
    except SchemaErrors as exc:
        report = ValidationReport(
            status="failed",
            dataset_path=dataset_path,
            checked_at_utc=checked_at,
            row_count=len(data),
            column_count=len(data.columns),
            errors=_schema_errors(exc),
        )
        _write_report(report, report_path)
        raise DataValidationError(report) from exc
    except (TypeError, ValueError) as exc:
        report = ValidationReport(
            status="failed",
            dataset_path=dataset_path,
            checked_at_utc=checked_at,
            row_count=len(data),
            column_count=len(data.columns),
            errors=[{"check": "date_is_valid", "failure_case": str(exc)}],
        )
        _write_report(report, report_path)
        raise DataValidationError(report) from exc

    # Only warnings remain, so this dataset is safe for the next pipeline step.
    report = ValidationReport(
        status="passed",
        dataset_path=dataset_path,
        checked_at_utc=checked_at,
        row_count=len(validated),
        column_count=len(validated.columns),
        warnings=_warning_summary(validated),
    )
    _write_report(report, report_path)
    return validated, report


def validate_csv(
    data_path: Path,
    *,
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load a CSV and validate it against the raw-data contract."""

    if not data_path.is_file():
        report = ValidationReport(
            status="failed",
            dataset_path=str(data_path),
            checked_at_utc=datetime.now(timezone.utc).isoformat(),
            row_count=0,
            column_count=0,
            errors=[{"check": "file_exists", "failure_case": str(data_path)}],
        )
        _write_report(report, report_path)
        raise DataValidationError(report)

    try:
        data = pd.read_csv(data_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        report = ValidationReport(
            status="failed",
            dataset_path=str(data_path),
            checked_at_utc=datetime.now(timezone.utc).isoformat(),
            row_count=0,
            column_count=0,
            errors=[{"check": "csv_is_readable", "failure_case": str(exc)}],
        )
        _write_report(report, report_path)
        raise DataValidationError(report) from exc

    return validate_house_data(
        data,
        dataset_path=str(data_path),
        report_path=report_path,
    )
