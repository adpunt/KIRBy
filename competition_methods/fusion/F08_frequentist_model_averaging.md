# F08 — Frequentist Model Averaging (Hjort & Claeskens)

**Type:** Fusion / ensembling (information-criterion-weighted)
**Domain:** Parametric model classes with comparable AIC/BIC
**Cite:** Hjort & Claeskens, "Frequentist Model Average Estimators", JASA 2003

## What it is
Like BMA (F07) but weights derived from a frequentist criterion (typically smoothed AIC or focused IC). For AIC:
```
w_k ∝ exp(-0.5 * AIC_k)
```
"Focused information criterion" (FIC) variant tunes the weights to a specific *focus parameter* (e.g. survival at a particular time horizon).

## Why it helps
- Doesn't require Bayesian priors — purely frequentist
- AIC weights have explicit risk-minimization properties (predictively optimal under regularity conditions)
- Focused variant lets you weight bases by their performance at the prediction target you actually care about (single horizon, single subgroup)
- Like BMA, avoids OOF-metric-driven greedy selection

## When to use
- Comparing models from the same parametric family (Cox / AFT variants)
- Have a specific target functional (e.g. 5-year hep risk) — use FIC variant
- Want a non-Bayesian alternative to F07

## When not to use
- Nonparametric / deep bases with ill-defined AIC
- Strong dependence between candidates (AIC weights collapse onto one model)

## Implementation
```python
def fma_weights(aic_dict, smoothing=1.0):
    names = list(aic_dict)
    aic = np.array([aic_dict[n] for n in names])
    log_w = -0.5 * (aic - aic.min()) / smoothing
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return dict(zip(names, w))
```

`smoothing > 1.0` flattens weights toward uniform (reduces overfit to AIC noise); `smoothing < 1.0` sharpens.

## Trade-offs
- Like F07: AIC requires effective parameter count
- Weights can collapse onto a single best-AIC model under exponential weighting; smoothing fixes this
- Less common in ML practice than BMA — fewer canonical implementations

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [F07 BMA](F07_bayesian_model_averaging.md)
[← back to index](../README.md)
