# House Price System

Machine Learning Engineering / MLOps project for house-price prediction.

## Data validation

Ingest the tracked raw dataset, validate and clean it, then create group-aware
temporal train/validation/test partitions (approximately 70/15/15):

```powershell
python scripts/00_data_ingestion.py
```

The command writes `data/processed/train.csv`, `validation.csv`, `test.csv`,
and `split_manifest.json`. Rejected rows are preserved in
`data/rejected/rejected_rows.csv`. The manifest records cleaning decisions,
checksums, feature roles, row counts, and validation warnings.

All sales for one property ID stay in the same partition. The property's latest
sale date decides the partition, preventing entity leakage while retaining real
repeat sales. Actual fractions can differ slightly from 70/15/15.

Run the critical raw-data checks before training:

```powershell
python scripts/01_data_validation.py
```

The command validates the exact schema, value ranges, calendar dates, unique
property-sale keys, living-area consistency, and renovation years. A successful
run writes `artifacts/data_validation_report.json` and exits with code 0.

Demonstrate that corrupted data stops the pipeline and emits an alert:

```powershell
python scripts/01_data_validation.py `
  --input tests/fixtures/kc_house_data_invalid.csv `
  --report artifacts/invalid_data_validation_report.json
$LASTEXITCODE
```

The corrupted fixture must print `DATA_VALIDATION_ALERT` and exit with code 1.
Repeated property IDs and suspicious bedroom/bathroom counts are reported as
warnings because the dataset can contain legitimate repeat sales and incomplete
listings; duplicate `(id, date)` sales remain a critical failure.

Generate separate, reproducible EDA evidence for the approved cleaning rules:

```powershell
python scripts/02_data_eda.py
```

The command does not modify the raw data. It creates a Markdown report, JSON
summary, anomaly-row CSV, and three figures under `reports/eda/`.

See `docs/data_limitations.md` for scope, drift, and interpretation limits.

## Data readiness benchmark

After Data Ingestion has created the three split files, run:

```powershell
python scripts/03_data_readiness_benchmark.py
```

The benchmark checks split hashes before training, learns preprocessing from
train only, selects a ridge regularization value using validation MAE, and
evaluates test once after selection. It writes metrics, feature importance,
predictions, and diagnostic figures to `reports/data_readiness/`.

This is a data-quality baseline for handoff. The model team can replace the
ridge model while preserving the split, feature contract, and leakage controls.
See `docs/model_handoff.md` for the exact contract and current findings.

