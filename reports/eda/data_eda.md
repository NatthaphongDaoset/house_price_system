# King County House Data — EDA Evidence

## Dataset overview

- Rows: 21,613
- Columns: 21
- Sale dates: 2014-05-02 to 2015-05-27
- Missing cells in the raw CSV: 0
- Exact duplicate rows: 0
- Duplicate `(id, date)` sale keys: 0
- Price range: 75,000 to 7,700,000
- Median price: 450,000

## Approved cleaning evidence

These rules are approved for ingestion. EDA records evidence but does not alter
the raw CSV itself:

| Candidate rule | Matching rows before overlap removal |
|---|---:|
| `bedrooms == 0` | 13 |
| `bathrooms == 0` | 10 |
| `bedrooms > 20` and `< 100 sqft/bedroom` | 1 |
| `yr_built > sale year` | 12 |
| nonzero `yr_renovated > sale year` | 6 |

The union contains **35 rows**, leaving
**21578 rows** for splitting. No value is
imputed during ingestion. Any model-side imputer must be fitted on train only.

The 11-bedroom observation and large price/area values are retained. Repeated
sales are kept together in one split based on each property's latest sale date.
See `anomaly_rows.csv` for row-level evidence.

## Figures

- `figures/price_distribution.png`
- `figures/bedrooms_vs_living_area.png`
- `figures/monthly_sales.png`
