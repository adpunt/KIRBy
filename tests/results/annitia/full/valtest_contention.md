# val_test ensemble contention map

Aggregated across 25 variant CSVs.

## Interpretation

- **High contention (IQR rank > 0.30)**: variants strongly disagree on patient ranking.
  These patients dominate the LB gap. Different recipes will move them in different directions.
- **Stable high-rank**: ensemble consensus — likely true high-risk patients. Submission strategy
  should preserve these rankings.
- **Stable low-rank**: ensemble consensus low-risk. Same.

## Per-endpoint files

Detailed per-patient tables: `valtest_contention_hepatic.csv` and `valtest_contention_death.csv`
