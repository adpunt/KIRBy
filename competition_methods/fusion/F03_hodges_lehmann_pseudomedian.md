# F03 — Hodges-Lehmann pseudomedian

**Type:** Fusion / ensembling (parameter-free, robust location)
**Domain:** Any setting requiring per-patient aggregation across candidates
**Cite:** Hodges & Lehmann, "Estimates of Location Based on Rank Tests", Annals of Math Stat 1963

## What it is
Per-patient prediction aggregator: median of all pairwise means.
```
HL(x_1, ..., x_n) = median{ (x_i + x_j) / 2 : i ≤ j }
```
Asymptotically equivalent to a Wilcoxon-rank-test-based location estimator. Strictly more efficient than the median for symmetric distributions, while retaining ~95% of the robustness.

## Why it helps
- More efficient than the median (closer to mean efficiency) when the underlying distribution is roughly symmetric
- More robust than the mean against single-variant outliers
- A natural "in-between" between median (overly conservative) and mean (insufficiently robust)
- Zero parameters → no OOF fitting → no selection bias

## When to use
- Robust per-patient pooling when 1–2 candidates are suspected to be unreliable
- Approximately-symmetric prediction distributions (true for ranks; often approximately true for risk scores)
- Pools of 5–20 candidates (pairwise enumeration cost manageable)

## When not to use
- Highly asymmetric prediction distributions (geometric mean / log-pool better)
- Very large pools (n > 100) — pairwise enumeration becomes expensive

## Implementation
```python
import numpy as np
def hodges_lehmann(stack):
    # stack: shape (n_candidates, n_patients)
    out = np.empty(stack.shape[1])
    for p in range(stack.shape[1]):
        col = stack[:, p]
        pairs = (col[:, None] + col[None, :]).flatten() / 2
        out[p] = np.median(pairs)
    return out
```

For large libraries use the standard $O(n \log n)$ algorithm via the Wilcoxon walking-median; for small (<20) the naive $O(n^2)$ is fine.

## Trade-offs
- Per-patient cost = $O(N^2)$ where N = candidate count; negligible for small N
- Less interpretable than mean / median in absolute terms
- Empirical lifts over median typically small but consistent

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | hold-out C (5-seed) | not submitted | Top-15 by train C → HL pseudomedian. Tied with anchor: hep ΔC +0.012 (p=0.88), death ΔC +0.0056 (p=0.228) on permutation null R=500. Multi-seed hold-out 0.9253 ± 0.024 — within SE of anchor. Conclusion: when bases are highly correlated (single-domain pool of 206 candidates), HL behaves like rank-mean. |

---
[← back to index](../README.md)
