# Data readiness benchmark

## Purpose

This benchmark checks whether the validated feature set contains useful signal.
It is evidence for handoff, not the model team's production model selection.

## Validation comparison

| Model | MAE | RMSE | R² | Within 20% |
|---|---:|---:|---:|---:|
| dummy_median | 220,095 | 355,561 | -0.045 | 28.9% |
| ridge_log_alpha_0_1 | 70,742 | 123,151 | 0.875 | 78.0% |
| ridge_log_alpha_10 | 71,796 | 124,719 | 0.871 | 77.8% |
| ridge_log_alpha_100 | 82,928 | 143,533 | 0.830 | 73.9% |

Selected from validation MAE: **ridge_log_alpha_0_1**

## Final test (evaluated after selection)

- MAE: 75,591
- RMSE: 130,059
- R²: 0.880
- Within 20%: 78.6%
- MAE improvement over median baseline: 66.4%
- Readiness gate passed: **True**

## Leakage controls

- Split hashes were checked against `split_manifest.json`.
- Property IDs do not cross train, validation, and test.
- `id` is removed before modeling and `zipcode` is encoded as categorical.
- Imputation, scaling, and ZIP categories are learned from train only.
- The chosen alpha is refitted on train + validation, then test is evaluated once.
