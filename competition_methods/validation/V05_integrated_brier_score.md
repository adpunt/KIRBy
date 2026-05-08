# V05 — Integrated Brier Score (IPCW-weighted)

**Type:** Validation metric (calibration + discrimination)
**Domain:** Survival
**Cite:** Graf et al., "Assessment and comparison of prognostic classification schemes for survival data", Stat Med 1999

## What it is
Brier score at time t for survival = mean squared error between predicted survival probability S(t|x) and the true survival indicator (with IPCW for censoring). Integrated Brier Score (IBS) averages this over a grid of t values. Lower = better.

## Why it helps
C-index measures *discrimination* (rank correctness) only. Two models with identical C can have very different *calibration* — one predicts probabilities at the right scale, the other doesn't. IBS jointly captures both via the Murphy decomposition:
```
IBS = uncertainty - resolution + reliability
```
A variant that wins on C but loses on IBS is *distorting calibration to chase rank* — exactly the failure mode of OOF-tuned fusion variants.

## When to use
- Whenever C-index is the primary metric, run IBS as second opinion
- Comparing variants whose C-index diff is < 0.005 (often calibration distinguishes them)
- Detecting "rank-only" hacks in fusion recipes

## When not to use
- Ranking-only competitions (LB literally ignores calibration)
- Models that don't output S(t|x) — need probabilistic output, not just risk score

## Implementation
```python
from sksurv.metrics import integrated_brier_score
# survs: n_samples × n_times array of S(t_i | x_j)
# y_train, y_test: structured arrays
ibs = integrated_brier_score(y_train, y_test, survs, times)
```

`times` should span from t_min (above 0) to t_max (just below max follow-up).

## Trade-offs
- Requires probabilistic survival output, not just risk score
- Tightly coupled to choice of `times` grid
- Less interpretable than C-index in absolute terms (typical IBS 0.05–0.20)

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [V04 Uno IPCW C](V04_uno_ipcw_concordance.md), [V11 Brier residual diversity](V11_brier_residual_diversity.md)
[← back to index](../README.md)
