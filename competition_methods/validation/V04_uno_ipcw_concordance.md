# V04 — Uno's IPCW concordance index

**Type:** Validation metric
**Domain:** Survival with moderate-to-high right-censoring
**Cite:** Uno et al., "On the C-statistics for Evaluating Overall Adequacy of Risk Prediction Procedures with Censored Survival Data", Stat Med 2011

## What it is
Concordance estimator weighted by inverse probability of censoring weights (IPCW). For each comparable pair (i, j) where t_i < t_j and i had an event:
```
weight_ij = 1 / G(t_i)^2
```
where G is the Kaplan-Meier estimate of the censoring distribution. Concordance is the IPCW-weighted fraction of pairs ranked correctly.

## Why it helps over Harrell's C
Harrell's C is biased upward as censoring increases — it implicitly conditions on "non-censored" pairs only. Under heavy censoring (>30%), this introduces a censoring-dependent bias that varies *across model fits*. Uno's IPCW C has a limiting value that does not depend on the censoring distribution, so it's both **less biased** and **more comparable across studies**.

For our typical regime (90%+ censoring), Harrell's C can over-report by 0.02–0.05.

## When to use
- Right-censored survival
- Censoring fraction > 30%
- Comparing models across cohorts with different censoring patterns

## When not to use
- Censoring < 10% (Harrell's C is fine, simpler)
- Time-truncated data without proper G-estimation
- When G(t) is unstable (very small risk set near tau)

## Implementation
```python
from sksurv.metrics import concordance_index_ipcw
# y_train: structured array used to fit G
# y_test:  structured array on which we evaluate
# risk:    predicted risk score (higher = worse)
c, *_ = concordance_index_ipcw(y_train, y_test, risk, tau=tau_horizon)
```

`tau_horizon` should be set just below the maximum follow-up to keep G(tau) > 0.

## Trade-offs
- Requires fitting an additional KM model for G(t) — trivial cost
- Sensitive to choice of tau; report sensitivity over [t_max * 0.8, t_max * 0.95]
- Does not subsume Harrell's C — report both for transparency

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | TBD | TBD | Currently using Harrell's C; 96% censoring on hep so Uno expected to re-rank top candidates |

---
**See also:** [V05 IBS](V05_integrated_brier_score.md), [V06 t-AUC](V06_cumulative_dynamic_auc.md)
[← back to index](../README.md)
