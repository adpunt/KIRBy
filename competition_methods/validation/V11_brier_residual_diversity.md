# V11 — Per-patient Brier residual diversity

**Type:** Validation diagnostic (ensemble-component selection)
**Domain:** Survival / classification where ensembles are built from many candidates

## What it is
For each candidate variant, compute the per-patient Brier (or absolute error) at multiple horizons. Cluster patients by *which variants miss them*. Select fusion components that miss *different* patients, not different predictions on the same patients.

## Why it helps
"Decorrelation by prediction" measures cosine of full prediction vectors — but two variants can have correlated predictions globally yet differ on the hard cases that decide LB. The actually-useful diversity is *residual diversity*: which patients does this variant get wrong?

Two variants with high residual disagreement are genuinely complementary; two with low disagreement are redundant even if their predictions look different on average.

## When to use
- Selecting fusion components from a library of 50+ candidates
- Diagnosing ensemble-failure cohorts (which patients does the *whole pool* miss?)
- Designing specialist routers (V07-style cohort assignment)

## When not to use
- Pure ranking metrics (C-index without horizon-specific Brier) — residual undefined
- Tiny libraries (<10 candidates) where simpler diversity metrics suffice

## Implementation
```python
from sksurv.metrics import brier_score
# brier per patient per time horizon
horizons = [5, 10, 15]   # years
residual = {}
for name, S_pred in candidates.items():
    _, br = brier_score(y_train, y_test, S_pred, horizons)
    residual[name] = br  # shape: (n_patients, n_horizons)

# Disagreement matrix: how often do A and B miss DIFFERENT patients?
import numpy as np
def miss_disagreement(rA, rB, threshold):
    miss_A = (rA > threshold).any(axis=1)
    miss_B = (rB > threshold).any(axis=1)
    return (miss_A != miss_B).mean()  # 0 = identical, 1 = orthogonal misses
```

Selection rule: greedy add candidates that maximize sum of miss-disagreement with already-selected pool.

## Trade-offs
- Requires probabilistic survival output, not just rank
- Threshold-dependent; report sensitivity over a range
- More expensive than rank-correlation diversity

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [F06 diversity-penalized selection](../fusion/F06_diversity_penalized_selection.md)
[← back to index](../README.md)
