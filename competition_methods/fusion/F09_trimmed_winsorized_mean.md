# F09 — Trimmed / winsorized mean of predictions

**Type:** Fusion / ensembling (robust pooling)
**Domain:** Any aggregation across a noisy candidate library

## What it is
- **Trimmed mean:** drop the top q% and bottom q% of predictions per patient before averaging.
- **Winsorized mean:** clip the top q% and bottom q% to the q-th and (1-q)-th percentile, then average.

Typical q = 10–20%.

## Why it helps
- Single rogue base (a variant that's accidentally outputting garbage on a subset of patients) can shift an arithmetic mean substantially
- Trimming removes the contribution; winsorizing dampens it
- Cheap insurance against silent base failures (a model that crashed and returned all-NaN, then was filled with a constant)

## When to use
- Library size > 10 (trimming 20% leaves enough)
- Suspect 1–2 unreliable bases but don't know which
- Want a robust safety net behind a primary fusion method (geomean / Caruana)

## When not to use
- Tiny libraries (< 5) — trimming removes too much signal
- Bases known to be uniformly reliable — trimming wastes signal

## Implementation
```python
import numpy as np
def trimmed_mean_pool(stack, q=0.1):
    # stack: shape (n_candidates, n_patients)
    sorted_ = np.sort(stack, axis=0)
    n = sorted_.shape[0]
    lo = int(np.floor(q * n))
    hi = n - lo
    return sorted_[lo:hi].mean(axis=0)

def winsorized_mean_pool(stack, q=0.1):
    lo_q = np.quantile(stack, q, axis=0)
    hi_q = np.quantile(stack, 1 - q, axis=0)
    clipped = np.clip(stack, lo_q, hi_q)
    return clipped.mean(axis=0)
```

## Hyperparameter notes
- `q = 0.1` is a typical default
- Higher q = more robustness but less signal — tune via V08 hold-out
- Trimming preserves the mean's interpretability; winsorizing preserves all data

## Trade-offs
- Loses information from extreme bases that may actually be informative
- Doesn't *learn* which bases to trim — pure data-driven per-patient
- Excellent backup; rarely the absolute best; almost never catastrophic

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | hold-out C (5-seed) | not submitted | Top-20 by train C, q=0.1 trim. **Highest hold C of all 8 methods tested** (combined LB 0.9258 ±0.022) but ΔC vs anchor not significant (p=1.0 hep, p=0.722 death). Empirical lift over anchor +0.0029 well within SE. Recommendation: best of the parameter-free methods on this regime, but doesn't beat anchor when bases are correlated. |

---
[← back to index](../README.md)
