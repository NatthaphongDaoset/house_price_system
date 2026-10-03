"""Load, clean, and split house-sale data in a reproducible way.

The main function is :func:`ingest_and_split`. It performs these steps:

1. Validate the raw CSV.
2. Reject rows that are outside the project scope or clearly inconsistent.
3. Group transactions by property ID.
4. Split properties by their latest sale date.
5. Save train, validation, test, rejected rows, and a JSON manifest.

No missing value is filled here. Imputation belongs in the model pipeline and
must be fitted with the training set only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_validation import ValidationReport, validate_csv

DEFAULT_RAW_PATH = Path("data/raw/kc_house_data.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REJECTED_DIR = Path("data/rejected")
DATE_COLUMN = "date"
TARGET_COLUMN = "price"


class DataIngestionError(ValueError):
    """Raised when validated data cannot be prepared or split safely."""


def sha256_file(path: Path) -> str:
    """Return a stable content hash for data-version evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_reason(
    reasons: pd.Series,
    matching_rows: pd.Series,
    reason: str,
) -> None:
    """Add one reason to every matching row without hiding overlapping rules."""

    for row_index in reasons.index[matching_rows]:
        current_reason = reasons.at[row_index]
        if current_reason:
            reasons.at[row_index] = f"{current_reason};{reason}"
        else:
            reasons.at[row_index] = reason


def clean_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply approved row-level cleaning rules.

    Rejected rows are returned instead of silently deleted. This keeps an audit
    trail while preventing out-of-scope rows from reaching model training.
    """

    sale_dates = pd.to_datetime(
        data[DATE_COLUMN], format="%Y%m%dT%H%M%S", errors="raise"
    )
    sale_years = sale_dates.dt.year

    # The 33-bedroom, 1,620 sqft record is rejected by this density rule. The
    # 11-bedroom, 3,000 sqft record is retained because its density is plausible.
    sqft_per_bedroom = data["sqft_living"].div(data["bedrooms"].replace(0, pd.NA))

    rule_masks = {
        "zero_bedrooms": data["bedrooms"].eq(0),
        "zero_bathrooms": data["bathrooms"].eq(0),
        "bedrooms_above_20_and_below_100_sqft_each": data["bedrooms"].gt(20)
        & sqft_per_bedroom.lt(100),
        "built_after_sale": data["yr_built"].gt(sale_years),
        "renovated_after_sale": data["yr_renovated"].ne(0)
        & data["yr_renovated"].gt(sale_years),
    }

    rejection_reasons = pd.Series("", index=data.index, dtype="string")
    for rule_name, matching_rows in rule_masks.items():
        _append_reason(rejection_reasons, matching_rows, rule_name)

    rejected_mask = rejection_reasons.ne("")
    rejected = data.loc[rejected_mask].copy()
    rejected["rejection_reasons"] = rejection_reasons.loc[rejected_mask]
    cleaned = data.loc[~rejected_mask].copy().reset_index(drop=True)

    rule_counts = {}
    for rule_name, matching_rows in rule_masks.items():
        rule_counts[rule_name] = int(matching_rows.sum())

    cleaning_report = {
        "input_rows": len(data),
        "rule_counts_before_overlap_removal": rule_counts,
        "rejected_rows": len(rejected),
        "output_rows": len(cleaned),
        "imputation_applied": False,
        "policy": (
            "Rejected rows are saved for audit. Large price and area values are "
            "retained. Model-side imputation must be fitted on train only."
        ),
    }
    return cleaned, rejected.reset_index(drop=True), cleaning_report


def _find_cutoff_date(rows_by_date: pd.Series, target_rows: float) -> pd.Timestamp:
    """Return the first date whose cumulative row count reaches the target."""

    cumulative_rows = rows_by_date.cumsum()
    matching_dates = cumulative_rows.loc[cumulative_rows >= target_rows]
    if matching_dates.empty:
        raise DataIngestionError("Could not calculate a temporal split cutoff.")
    return pd.Timestamp(matching_dates.index[0])


def split_properties_by_latest_sale(
    data: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Keep all sales of one property in the same temporal partition.

    A property's latest sale date decides its partition. Earlier transactions
    follow that property into the same partition. This keeps legitimate resales
    while preventing the same property ID from appearing in two datasets.
    """

    test_fraction = 1.0 - train_fraction - validation_fraction
    if not (0 < train_fraction < 1):
        raise DataIngestionError("train_fraction must be between 0 and 1.")
    if not (0 < validation_fraction < 1):
        raise DataIngestionError("validation_fraction must be between 0 and 1.")
    if test_fraction <= 0:
        raise DataIngestionError("train and validation fractions must sum to less than 1.")

    working = data.copy()
    working["_sale_date"] = pd.to_datetime(
        working[DATE_COLUMN], format="%Y%m%dT%H%M%S", errors="raise"
    )
    working["_property_latest_sale"] = working.groupby("id")["_sale_date"].transform(
        "max"
    )

    # Count transactions by the latest-sale date of their property. Cutoffs are
    # based on row counts so the final partitions stay close to 70/15/15.
    rows_by_latest_sale = working.groupby("_property_latest_sale").size().sort_index()
    if len(rows_by_latest_sale) < 3:
        raise DataIngestionError(
            "Temporal splitting requires at least three property sale dates."
        )

    total_rows = len(working)
    train_end = _find_cutoff_date(rows_by_latest_sale, total_rows * train_fraction)
    validation_end = _find_cutoff_date(
        rows_by_latest_sale,
        total_rows * (train_fraction + validation_fraction),
    )

    train = working.loc[working["_property_latest_sale"] <= train_end]
    validation = working.loc[
        (working["_property_latest_sale"] > train_end)
        & (working["_property_latest_sale"] <= validation_end)
    ]
    test = working.loc[working["_property_latest_sale"] > validation_end]

    if train.empty or validation.empty or test.empty:
        raise DataIngestionError("Temporal split produced an empty dataset partition.")

    split_cutoffs = {
        "property_latest_sale_train_end": train_end.date().isoformat(),
        "property_latest_sale_validation_end": validation_end.date().isoformat(),
    }

    # Sort every file for deterministic output, then remove helper columns.
    output_columns = data.columns.tolist()
    train = train.sort_values(["_sale_date", "id"], kind="mergesort")
    validation = validation.sort_values(["_sale_date", "id"], kind="mergesort")
    test = test.sort_values(["_sale_date", "id"], kind="mergesort")
    train = train.loc[:, output_columns].reset_index(drop=True)
    validation = validation.loc[:, output_columns].reset_index(drop=True)
    test = test.loc[:, output_columns].reset_index(drop=True)

    return train, validation, test, split_cutoffs


def _date_range(data: pd.DataFrame) -> dict[str, str]:
    dates = pd.to_datetime(data[DATE_COLUMN], format="%Y%m%dT%H%M%S")
    return {
        "min": dates.min().date().isoformat(),
        "max": dates.max().date().isoformat(),
    }


def _cross_split_id_counts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, int]:
    train_ids = set(train["id"])
    validation_ids = set(validation["id"])
    test_ids = set(test["id"])
    return {
        "train_validation": len(train_ids & validation_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(validation_ids & test_ids),
    }


def _repeated_sale_summary(data: pd.DataFrame) -> dict[str, int]:
    sales_per_property = data.groupby("id").size()
    repeated_groups = sales_per_property.loc[sales_per_property > 1]
    return {
        "properties_with_multiple_sales": len(repeated_groups),
        "rows_from_properties_with_multiple_sales": int(repeated_groups.sum()),
        "maximum_sales_for_one_property": int(sales_per_property.max()),
    }


def _build_manifest(
    *,
    raw_path: Path,
    validation_report: ValidationReport,
    cleaning_report: dict[str, Any],
    cleaned_data: pd.DataFrame,
    rejected_path: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    split_cutoffs: dict[str, str],
    output_paths: dict[str, Path],
    requested_fractions: dict[str, float],
) -> dict[str, Any]:
    total_split_rows = len(train) + len(validation) + len(test)
    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    split_details: dict[str, dict[str, Any]] = {}
    for split_name, split_data in split_frames.items():
        split_path = output_paths[split_name]
        split_details[split_name] = {
            "path": str(split_path),
            "rows": len(split_data),
            "actual_fraction": round(len(split_data) / total_split_rows, 6),
            "transaction_date_range": _date_range(split_data),
            "unique_property_ids": int(split_data["id"].nunique()),
            "sha256": sha256_file(split_path),
        }

    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
            "rows": validation_report.row_count,
        },
        "feature_contract": {
            "target": TARGET_COLUMN,
            "identifier_columns": ["id"],
            "excluded_model_features": ["id"],
            "categorical_features": ["zipcode"],
            "datetime_columns": [DATE_COLUMN],
        },
        "cleaning": cleaning_report,
        "rejected_records": {
            "path": str(rejected_path),
            "rows": cleaning_report["rejected_rows"],
            "sha256": sha256_file(rejected_path),
        },
        "split_strategy": {
            "name": "group_aware_temporal_by_property_latest_sale",
            "description": (
                "All transactions for one property ID stay together. The latest "
                "sale date of that property decides its partition."
            ),
            "requested_fractions": requested_fractions,
            "cutoffs": split_cutoffs,
        },
        "splits": split_details,
        "cross_split_property_ids": _cross_split_id_counts(
            train, validation, test
        ),
        "repeated_sales": _repeated_sale_summary(cleaned_data),
        "validation": {
            "status": validation_report.status,
            "warnings": validation_report.warnings,
        },
        "limitations": [
            "Sales cover May 2014 to May 2015 in King County only.",
            "The data may not represent current house prices or other regions.",
            "Earlier sales of a repeated property follow its latest sale into the same split.",
        ],
    }


def ingest_and_split(
    raw_path: Path = DEFAULT_RAW_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    rejected_dir: Path = DEFAULT_REJECTED_DIR,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    validation_report_path: Path | None = None,
) -> dict[str, Any]:
    """Run validation, cleaning, splitting, saving, and manifest creation."""

    # Step 1: Validate the untouched raw data.
    validated, validation_report = validate_csv(
        raw_path, report_path=validation_report_path
    )

    # Step 2: Apply approved cleaning rules and preserve rejected rows.
    cleaned, rejected, cleaning_report = clean_data(validated)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = rejected_dir / "rejected_rows.csv"
    rejected.to_csv(rejected_path, index=False, lineterminator="\n")

    # Step 3: Split by property group and latest sale date.
    train, validation, test, split_cutoffs = split_properties_by_latest_sale(
        cleaned,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    # Step 4: Save each partition.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
    }
    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    for split_name, split_data in split_frames.items():
        split_data.to_csv(
            output_paths[split_name], index=False, lineterminator="\n"
        )

    # Step 5: Record decisions, versions, counts, and checksums.
    requested_fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    manifest = _build_manifest(
        raw_path=raw_path,
        validation_report=validation_report,
        cleaning_report=cleaning_report,
        cleaned_data=cleaned,
        rejected_path=rejected_path,
        train=train,
        validation=validation,
        test=test,
        split_cutoffs=split_cutoffs,
        output_paths=output_paths,
        requested_fractions=requested_fractions,
    )
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
