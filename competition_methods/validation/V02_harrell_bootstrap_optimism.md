# V02 — Harrell bootstrap optimism correction

**Type:** Validation protocol (bias correction)
**Domain:** Small-sample survival prognostic models
**Cite:** Harrell, *Regression Modeling Strategies* (2nd ed.), ch. 5; Steyerberg 2001

## What it is
1. Compute "apparent" performance: fit on full data, evaluate on full data → C_app.
2. For b = 1..B (typically 200): draw a bootstrap sample (with replacement, size n) → fit model → evaluate on bootstrap (C_boot) and on original full data (C_orig).
3. Optimism = mean_b(C_boot − C_orig).
4. Corrected = C_app − Optimism.

The corrected statistic estimates how much the apparent (training) performance overstates the true generalization performance.

## Why it helps
Apparent training C is upward biased — the model fit those exact patients. The bootstrap measures how much "fitting harder" inflates apparent performance vs how much the same model generalizes. Subtracting that gap is a principled internal-validation estimate.

## When to use
- Small samples (events per variable < 20)
- Survival prognostic models
- When you cannot afford a held-out test set

## When not to use
- When heavy regularization is applied (use V03 .632+ or compare both)
- Very heavy censoring (>90%) — pair with V04 Uno's C as the inner metric
- When R\*K-fold is already cheap (V01) and you have a held-out set

## Implementation
```python
def harrell_optimism(fit_predict_fn, X, y_struct, B=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_struct)
    C_app = c_index(fit_predict_fn(X, y_struct, X), y_struct)
    optimisms = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        risk_b = fit_predict_fn(X[idx], y_struct[idx], X[idx])
        risk_o = fit_predict_fn(X[idx], y_struct[idx], X)
        optimisms.append(c_index(risk_b, y_struct[idx]) - c_index(risk_o, y_struct))
    return C_app - np.mean(optimisms)
```

## Trade-offs
- B=200 means 200 model fits per candidate — expensive but parallelizable
- Returns a *point estimate*, not a CI; pair with bootstrap SE if needed
- Documented to slightly *over-correct* under heavy regularization (Iba 2021)

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [V03 .632+](V03_632plus_bootstrap.md), [V04 Uno IPCW C](V04_uno_ipcw_concordance.md)
[← back to index](../README.md)
