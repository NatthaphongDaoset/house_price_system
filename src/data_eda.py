"""Create EDA evidence without changing the raw dataset.

The functions in this file only inspect data and write reports or figures.
Approved cleaning rules are shown as evidence; this EDA module never drops or
fills a value. The ingestion module applies the approved rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVIDENCE_COLUMNS = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "yr_built",
    "yr_renovated",
    "zipcode",
]


def approved_cleaning_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Return approved cleaning rules without mutating the raw data."""

    # Division by zero becomes NaN, so zero-bedroom rows are handled safely.
    sqft_per_bedroom = data["sqft_living"].div(data["bedrooms"].replace(0, np.nan))
    sale_dates = pd.to_datetime(
        data["date"], format="%Y%m%dT%H%M%S", errors="raise"
    )
    sale_years = sale_dates.dt.year
    return {
        "zero_bedrooms": data["bedrooms"].eq(0),
        "zero_bathrooms": data["bathrooms"].eq(0),
        "bedrooms_above_20_and_below_100_sqft_each": data["bedrooms"].gt(20)
        & sqft_per_bedroom.lt(100),
        "built_after_sale": data["yr_built"].gt(sale_years),
        "renovated_after_sale": data["yr_renovated"].ne(0)
        & data["yr_renovated"].gt(sale_years),
    }


def approved_rejection_mask(data: pd.DataFrame) -> pd.Series:
    """Combine all approved rejection rules into one True/False Series."""

    masks = approved_cleaning_masks(data)
    result = pd.Series(False, index=data.index)
    for mask in masks.values():
        result |= mask
    return result


def anomaly_evidence(data: pd.DataFrame) -> pd.DataFrame:
    """Return rows relevant to the proposed cleaning decision."""

    evidence_mask = approved_rejection_mask(data) | data["bedrooms"].gt(10)
    result = data.loc[evidence_mask, EVIDENCE_COLUMNS].copy()
    result["sqft_per_bedroom"] = result["sqft_living"].div(
        result["bedrooms"].replace(0, np.nan)
    )
    result["approved_rejection"] = approved_rejection_mask(data).loc[result.index]
    return result.sort_values(["bedrooms", "bathrooms", "sqft_living"])


def build_eda_summary(data: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(data["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    masks = approved_cleaning_masks(data)
    drop_mask = approved_rejection_mask(data)
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_id_date": int(data.duplicated(["id", "date"]).sum()),
        "date_range": {
            "min": dates.min().date().isoformat(),
            "max": dates.max().date().isoformat(),
            "invalid": int(dates.isna().sum()),
        },
        "target_price": {
            "min": float(data["price"].min()),
            "median": float(data["price"].median()),
            "mean": float(data["price"].mean()),
            "max": float(data["price"].max()),
        },
        "approved_cleaning": {
            "rule_counts_before_overlap_removal": {
                name: int(mask.sum()) for name, mask in masks.items()
            },
            "unique_rows_rejected": int(drop_mask.sum()),
            "rows_after_cleaning": int((~drop_mask).sum()),
            "status": "approved_and_applied_by_ingestion",
        },
        "other_observations": {
            "bedrooms_above_10": int(data["bedrooms"].gt(10).sum()),
            "bedrooms_equal_11": int(data["bedrooms"].eq(11).sum()),
            "repeated_property_id_rows": int(data["id"].duplicated(keep=False).sum()),
            "nonpositive_price": int(data["price"].le(0).sum()),
            "nonpositive_living_area": int(data["sqft_living"].le(0).sum()),
            "living_area_component_mismatch": int(
                (data["sqft_above"] + data["sqft_basement"])
                .ne(data["sqft_living"])
                .sum()
            ),
        },
    }


def _write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    cleaning = summary["approved_cleaning"]
    counts = cleaning["rule_counts_before_overlap_removal"]
    target = summary["target_price"]
    report = f"""# King County House Data — EDA Evidence

## Dataset overview

- Rows: {summary['rows']:,}
- Columns: {summary['columns']}
- Sale dates: {summary['date_range']['min']} to {summary['date_range']['max']}
- Missing cells in the raw CSV: {summary['missing_cells']}
- Exact duplicate rows: {summary['duplicate_rows']}
- Duplicate `(id, date)` sale keys: {summary['duplicate_id_date']}
- Price range: {target['min']:,.0f} to {target['max']:,.0f}
- Median price: {target['median']:,.0f}

## Approved cleaning evidence

These rules are approved for ingestion. EDA records evidence but does not alter
the raw CSV itself:

| Candidate rule | Matching rows before overlap removal |
|---|---:|
| `bedrooms == 0` | {counts['zero_bedrooms']} |
| `bathrooms == 0` | {counts['zero_bathrooms']} |
| `bedrooms > 20` and `< 100 sqft/bedroom` | {counts['bedrooms_above_20_and_below_100_sqft_each']} |
| `yr_built > sale year` | {counts['built_after_sale']} |
| nonzero `yr_renovated > sale year` | {counts['renovated_after_sale']} |

The union contains **{cleaning['unique_rows_rejected']} rows**, leaving
**{cleaning['rows_after_cleaning']} rows** for splitting. No value is
imputed during ingestion. Any model-side imputer must be fitted on train only.

The 11-bedroom observation and large price/area values are retained. Repeated
sales are kept together in one split based on each property's latest sale date.
See `anomaly_rows.csv` for row-level evidence.

## Figures

- `figures/price_distribution.png`
- `figures/bedrooms_vs_living_area.png`
- `figures/monthly_sales.png`
"""
    output_path.write_text(report, encoding="utf-8")


def _write_figures(data: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data["price"], bins=50, color="#3274a1")
    axes[0].set_title("Sale price")
    axes[0].set_xlabel("Price")
    axes[0].set_ylabel("Count")
    axes[1].hist(np.log1p(data["price"]), bins=50, color="#e1812c")
    axes[1].set_title("log(1 + sale price)")
    axes[1].set_xlabel("Log price")
    figure.tight_layout()
    figure.savefig(figures_dir / "price_distribution.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        data["bedrooms"],
        data["sqft_living"],
        alpha=0.25,
        s=12,
        color="#3274a1",
    )
    axis.set_title("Bedrooms vs living area")
    axis.set_xlabel("Bedrooms")
    axis.set_ylabel("Living area (sqft)")
    figure.tight_layout()
    figure.savefig(figures_dir / "bedrooms_vs_living_area.png", dpi=140)
    plt.close(figure)

    dates = pd.to_datetime(data["date"], format="%Y%m%dT%H%M%S")
    monthly_sales = dates.dt.to_period("M").value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(monthly_sales.index.astype(str), monthly_sales.values, color="#55a868")
    axis.set_title("Sales per month")
    axis.set_xlabel("Month")
    axis.set_ylabel("Rows")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(figures_dir / "monthly_sales.png", dpi=140)
    plt.close(figure)


def write_eda_evidence(data: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Write machine-readable, row-level, narrative, and visual evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON is useful for code and the Markdown file is useful for reviewers.
    summary = build_eda_summary(data)
    (output_dir / "data_eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    anomaly_evidence(data).to_csv(output_dir / "anomaly_rows.csv", index=False)
    _write_markdown(summary, output_dir / "data_eda.md")
    _write_figures(data, output_dir / "figures")
    return summary
