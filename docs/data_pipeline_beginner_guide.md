# Beginner guide: Data Ingestion, Validation, and EDA

The data flow is intentionally kept in three separate commands:

```text
kc_house_data.csv
        |
        v
Data Validation  -- critical error --> stop with exit code 1
        |
        v
Data Ingestion   --> train.csv + validation.csv + test.csv + manifest

EDA reads the raw CSV separately and creates evidence. It never changes data.
```

## 1. Data Validation

Run:

```powershell
python scripts/01_data_validation.py
```

Code entrypoint: `scripts/01_data_validation.py`  
Reusable functions: `src/data_validation.py`

Pandera checks required columns, data types, allowed ranges, and relationships
between columns. Critical failures create a JSON report and exit with code 1.
Warnings are included in the report but return exit code 0.

## 2. Data Ingestion

Run:

```powershell
python scripts/00_data_ingestion.py
```

Code entrypoint: `scripts/00_data_ingestion.py`  
Reusable functions: `src/data_ingestion.py`

The command validates first, sorts by sale date, and creates chronological
70/15/15 partitions. A sale date is never shared by two partitions. The JSON
manifest records row counts, cutoff dates, file hashes, and warnings.

## 3. EDA evidence

Run:

```powershell
python scripts/02_data_eda.py
```

Code entrypoint: `scripts/02_data_eda.py`  
Reusable functions: `src/data_eda.py`

The command writes a Markdown report, JSON summary, anomaly-row CSV, and PNG
figures to `reports/eda/`. EDA reports the approved rules but never changes the
raw CSV. Cleaning happens in Data Ingestion and rejected rows are preserved in
`data/rejected/rejected_rows.csv`.

## 4. Feature engineering and data readiness

Run Data Ingestion first, then run:

```powershell
python scripts/03_data_readiness_benchmark.py
```

Code entrypoint: `scripts/03_data_readiness_benchmark.py`  
Reusable functions: `src/feature_engineering.py`, `src/data_readiness.py`

Fixed feature formulas add sale month, house age, renovation age, basement
status, area ratios, interactions, and log-area features. The training pipeline
removes `id`, treats `zipcode` as a category, and learns missing-value medians,
scaling values, and ZIP categories from train only.

Three ridge regularization values are compared on validation against a median
baseline. The winner is fitted again with train + validation, then test is read
for the final evaluation. Do not use the test result to tune these choices.

## 5. Tests

Run only this work area:

```powershell
python -m pytest tests/test_data.py tests/test_eda.py tests/test_feature_engineering.py -v
```

Tests use temporary output folders, so they do not overwrite processed project
data. A green run currently contains 15 tests.
