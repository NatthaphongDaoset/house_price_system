# Data limitations and decisions

## Scope

The dataset contains King County house sales from 2 May 2014 to 27 May 2015.
It can support a course demonstration of asset valuation in that historical
market. It does not directly represent current prices, Thailand, or markets
outside King County. Data drift and concept drift should be expected before the
model is used with a different period or location.

## Approved row cleaning

The raw CSV remains unchanged. Data Ingestion excludes rows with:

- zero bedrooms;
- zero bathrooms;
- more than 20 bedrooms and less than 100 living sqft per bedroom;
- a construction year after the sale year; or
- a nonzero renovation year after the sale year.

Rejected rows are saved with one or more explicit reasons in
`data/rejected/rejected_rows.csv`. Large prices, lots, and living areas remain
because luxury houses and large land parcels are plausible in this domain.
No value is imputed during ingestion.

## Repeated property sales

Repeated property IDs are valid transactions rather than duplicate rows. In the
raw data, 176 properties have multiple sales, the shortest gap is 61 days, and
no `(id, date)` key is duplicated. Recorded physical features are unchanged
between repeated sales even when prices change, so unrecorded renovation or
market effects may exist.

To prevent entity leakage, all transactions for one property stay in one split.
The property's latest sale date determines whether it belongs to train,
validation, or test. Earlier transactions therefore follow the latest sale into
that partition, so transaction-date ranges can overlap even though property IDs
never cross partitions.

## Feature roles

- `price` is the regression target.
- `id` is retained for lineage and split grouping but excluded from model input.
- `zipcode` is categorical even though the CSV stores it as an integer.
- `date` requires explicit feature engineering before model input.
- `yr_renovated = 0` means no recorded renovation and is not calendar year zero.
- `sqft_basement = 0` means no basement and is not a missing value.

Any imputer, encoder, scaler, or learned transformation must be fitted on the
training split only, then reused unchanged for validation, test, and serving.
