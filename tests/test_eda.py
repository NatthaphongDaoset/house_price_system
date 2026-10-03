from pathlib import Path

import pandas as pd

from src.data_eda import build_eda_summary, write_eda_evidence

REAL_DATA_PATH = Path("data/raw/kc_house_data.csv")


def test_eda_summary_records_cleaning_evidence():
    data = pd.read_csv(REAL_DATA_PATH)

    summary = build_eda_summary(data)

    cleaning = summary["approved_cleaning"]
    assert summary["rows"] == 21_613
    assert cleaning["rule_counts_before_overlap_removal"] == {
        "zero_bedrooms": 13,
        "zero_bathrooms": 10,
        "bedrooms_above_20_and_below_100_sqft_each": 1,
        "built_after_sale": 12,
        "renovated_after_sale": 6,
    }
    assert cleaning["unique_rows_rejected"] == 35
    assert cleaning["rows_after_cleaning"] == 21_578
    assert cleaning["status"] == "approved_and_applied_by_ingestion"


def test_eda_writes_reviewable_evidence(tmp_path):
    data = pd.read_csv(REAL_DATA_PATH)

    write_eda_evidence(data, tmp_path)

    expected_files = [
        "data_eda_summary.json",
        "anomaly_rows.csv",
        "data_eda.md",
        "figures/price_distribution.png",
        "figures/bedrooms_vs_living_area.png",
        "figures/monthly_sales.png",
    ]
    for relative_path in expected_files:
        assert (tmp_path / relative_path).is_file()
