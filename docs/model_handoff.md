# Model-team handoff: data and feature contract

## Approved inputs

- Target: `price`, interpreted as the historical sale price.
- Entity key: `id`, retained for traceability and excluded from model features.
- Time field: `date`, converted to explicit calendar features.
- Location: `zipcode` is categorical. It must not be treated as a continuous
  number.
- Split files: `data/processed/train.csv`, `validation.csv`, and `test.csv`.
- Split proof: `data/processed/split_manifest.json` contains row counts,
  cutoffs, hashes, and zero-overlap checks.

All transactions for one property remain in one partition. This prevents the
same house from appearing in both fitting and evaluation data while retaining
real repeat-sale records.

## Feature engineering

`src/feature_engineering.py` is the shared deterministic transformation. It
adds sale year/month and cyclical month values, house and renovation age,
basement and renovation flags, area-per-room ratios, a living-area/grade
interaction, squared house age, and log-area values.

The following operations learn from data and therefore belong inside the fitted
training/serving pipeline:

- numeric median imputation;
- numeric scaling;
- the list of known ZIP-code categories;
- model coefficients or tree parameters.

Fit these operations on train only during experiments. After choosing the
configuration with validation, fit the chosen configuration on train +
validation and evaluate test once.

## Current benchmark evidence

The NumPy ridge baseline selected `alpha=0.1` using validation MAE.

| Partition | MAE | RMSE | R² | Within 20% |
|---|---:|---:|---:|---:|
| Validation | 70,742 | 123,151 | 0.875 | 78.0% |
| Test | 75,591 | 130,059 | 0.880 | 78.6% |

MAE improved by 67.9% over the validation median baseline and 66.4% over the
test median baseline. This passes the data-readiness rule of at least 20%
improvement on both partitions.

Permutation importance shows the strongest current signals are `zipcode`,
`sqft_living`, `grade`, latitude, and longitude. Test MAE is highest in the top
price quartile (144,181), so the model team should report price-band metrics and
consider nonlinear models or a specialized high-price strategy.

## Limits

The data covers King County sales from May 2014 through May 2015. These results
show that the fields contain predictive signal for that historical setting;
they do not establish present-day accuracy or transfer to Thailand. Production
use requires newer local transactions, drift monitoring, and a retraining rule.

Full evidence is stored in `reports/data_readiness/`.
