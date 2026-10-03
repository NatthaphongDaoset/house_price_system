from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import ingest_and_split
from src.data_validation import DataValidationError, validate_csv, validate_house_data

REAL_DATA_PATH = Path("data/raw/kc_house_data.csv")


def test_project_data_directories_exist():
    assert Path("data/raw").exists()
    assert Path("data/processed").exists()
    assert Path("data/drift").exists()


def test_real_dataset_passes_schema(tmp_path):
    report_path = tmp_path / "valid-report.json"

    validated, report = validate_csv(REAL_DATA_PATH, report_path=report_path)

    assert report.status == "passed"
    assert report.row_count == 21_613
    assert len(validated.columns) == 21
    assert report_path.exists()


def test_negative_price_stops_pipeline(tmp_path):
    data = pd.read_csv(REAL_DATA_PATH, nrows=3)
    data.loc[0, "price"] = -1
    report_path = tmp_path / "negative-price-report.json"

    with pytest.raises(DataValidationError) as caught:
        validate_house_data(data, report_path=report_path)

    assert caught.value.report.status == "failed"
    assert report_path.exists()
    assert any(
        error.get("column") == "price" for error in caught.value.report.errors
    )


def test_missing_required_column_stops_pipeline():
    data = pd.read_csv(REAL_DATA_PATH, nrows=3).drop(columns=["zipcode"])

    with pytest.raises(DataValidationError) as caught:
        validate_house_data(data)

    assert any(
        error.get("column") == "zipcode" for error in caught.value.report.errors
    )


def test_inconsistent_living_area_stops_pipeline():
    data = pd.read_csv(REAL_DATA_PATH, nrows=3)
    data.loc[0, "sqft_living"] += 1

    with pytest.raises(DataValidationError) as caught:
        validate_house_data(data)

    assert any(
        error.get("check") == "living_area_components_match"
        for error in caught.value.report.errors
    )


def test_invalid_calendar_date_stops_pipeline():
    data = pd.read_csv(REAL_DATA_PATH, nrows=3)
    data.loc[0, "date"] = "20141340T000000"

    with pytest.raises(DataValidationError) as caught:
        validate_house_data(data)

    assert caught.value.report.errors[0]["check"] == "date_is_valid"


def test_temporal_ingestion_creates_reproducible_splits(tmp_path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = ingest_and_split(
        REAL_DATA_PATH,
        first_output,
        rejected_dir=tmp_path / "first-rejected",
    )
    second = ingest_and_split(
        REAL_DATA_PATH,
        second_output,
        rejected_dir=tmp_path / "second-rejected",
    )

    assert sum(split["rows"] for split in first["splits"].values()) == 21_578
    assert first["cleaning"]["rejected_rows"] == 35
    assert (
        first["split_strategy"]["cutoffs"]
        == second["split_strategy"]["cutoffs"]
    )
    assert {
        name: details["sha256"] for name, details in first["splits"].items()
    } == {
        name: details["sha256"] for name, details in second["splits"].items()
    }
    assert (first_output / "split_manifest.json").exists()


def test_property_ids_do_not_cross_splits(tmp_path):
    output_dir = tmp_path / "processed"
    manifest = ingest_and_split(
        REAL_DATA_PATH,
        output_dir,
        rejected_dir=tmp_path / "rejected",
    )

    train = pd.read_csv(output_dir / "train.csv")
    validation = pd.read_csv(output_dir / "validation.csv")
    test = pd.read_csv(output_dir / "test.csv")

    train_ids = set(train["id"])
    validation_ids = set(validation["id"])
    test_ids = set(test["id"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert manifest["cross_split_property_ids"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert manifest["repeated_sales"]["properties_with_multiple_sales"] == 176


def test_cleaning_is_auditable(tmp_path):
    manifest = ingest_and_split(
        REAL_DATA_PATH,
        tmp_path / "processed",
        rejected_dir=tmp_path / "rejected",
    )

    rejected = pd.read_csv(tmp_path / "rejected" / "rejected_rows.csv")
    cleaning = manifest["cleaning"]
    assert len(rejected) == 35
    assert cleaning["rule_counts_before_overlap_removal"] == {
        "zero_bedrooms": 13,
        "zero_bathrooms": 10,
        "bedrooms_above_20_and_below_100_sqft_each": 1,
        "built_after_sale": 12,
        "renovated_after_sale": 6,
    }
    assert cleaning["imputation_applied"] is False
    assert "rejection_reasons" in rejected.columns


def test_manifest_defines_feature_roles(tmp_path):
    manifest = ingest_and_split(
        REAL_DATA_PATH,
        tmp_path / "processed",
        rejected_dir=tmp_path / "rejected",
    )

    contract = manifest["feature_contract"]
    assert contract["excluded_model_features"] == ["id"]
    assert contract["categorical_features"] == ["zipcode"]


def test_validation_warnings_do_not_stop_ingestion(tmp_path):
    manifest = ingest_and_split(
        REAL_DATA_PATH,
        tmp_path / "processed",
        rejected_dir=tmp_path / "rejected",
    )

    warning_names = {
        warning["check"] for warning in manifest["validation"]["warnings"]
    }
    assert "repeated_property_id" in warning_names
    assert "bedrooms_above_10" in warning_names
    assert "zero_bedrooms" in warning_names
    assert "zero_bathrooms" in warning_names
