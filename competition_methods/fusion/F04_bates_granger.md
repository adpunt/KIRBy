# F04 — Bates-Granger variance-weighted combination

**Type:** Fusion / ensembling (one weight per base, fit on hold-out)
**Domain:** Forecast / prediction combination with reliable variance estimates
**Cite:** Bates & Granger, "The Combination of Forecasts", Operational Research Quarterly 1969

## What it is
For each base i, estimate the variance σ²_i of its prediction error on a hold-out set. Combination weights:
```
w_i ∝ 1 / σ²_i
ŷ = Σ w_i p_i  (with weights normalized to sum to 1)
```
Theoretically optimal under independent base errors with known variances. Robust extensions (Diebold-Pauly 1987, Stock-Watson 2004) handle correlated errors.

## Why it helps
- Variance is a *more stable* statistic across bootstraps than the C-index itself
- One parameter per base (the variance) — minimal selection-bias surface
- Has a closed-form theoretical optimum under independence
- Naturally penalizes high-variance / unreliable bases without explicit selection

## When to use
- A reliable hold-out set (V08) is available
- Bases have meaningfully different error variances
- Probability or rank-scale predictions where MSE is interpretable

## When not to use
- Highly correlated base errors (use shrinkage extension or covariance-weighted form)
- Hold-out is too small to estimate σ²_i reliably (n_hold < 50)

## Implementation
```python
def bates_granger(oof_dict, y_holdout, holdout_idx):
    bases = list(oof_dict.keys())
    var = {b: np.var(oof_dict[b][holdout_idx] - y_holdout) for b in bases}
    inv = np.array([1 / var[b] for b in bases])
    w = inv / inv.sum()
    return {b: w_i for b, w_i in zip(bases, w)}, w

# At submission time:
def apply(weights_dict, oof_dict_full):
    return sum(w * oof_dict_full[b] for b, w in weights_dict.items())
```

For correlated errors, replace 1/σ²_i with the row sum of Σ⁻¹ where Σ is the residual covariance matrix on hold-out.

## Trade-offs
- Single hold-out → single weight set; can be high-variance for small hold-outs
- Doesn't account for systematic bias in any base — pair with bias correction (V02 / V03) per base before combining
- Often dominated by F01 / F05 in practice but provides a clean theoretical baseline

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
[← back to index](../README.md)
