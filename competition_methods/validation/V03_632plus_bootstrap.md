# V03 — .632+ bootstrap estimator

**Type:** Validation protocol (bias correction)
**Domain:** Small-sample classification + survival
**Cite:** Efron & Tibshirani, "Improvements on Cross-Validation: The .632+ Bootstrap Method", JASA 1997

## What it is
Convex combination of training error and out-of-bag error:
```
err_.632+ = (1 - w) * err_train + w * err_oob
w = 0.632 / (1 - 0.368 * R')
R' = (err_oob - err_train) / (γ - err_train)   // relative overfit
γ = no-information error rate (random pairing)
```
The plain `.632` estimator uses w = 0.632 fixed; `.632+` adapts w to how much the model is actually overfitting (R').

## Why it helps
- Training error alone is hopelessly optimistic.
- Pure OOB / leave-one-out is downward biased and high-variance for small samples.
- The convex combination reduces both biases simultaneously. .632+ fixes the original .632's failure when overfitting is severe.

## When to use
- Small-sample classification or survival
- Models with light-to-moderate regularization
- When K-fold variance is too high (rare events)

## When not to use
- Heavy regularization (Iba 2021 finds .632+ underperforms here vs Harrell's optimism, V02)
- Large data (single hold-out works fine)

## Implementation
- Survival: `c_index_oob` from sksurv random-survival-forest is a free OOB; for general models implement via `np.random.choice(n, n, replace=True)` and track which indices are *not* sampled per bootstrap.
- General-purpose: `mlxtend.evaluate.bootstrap_point632_score`

```python
from mlxtend.evaluate import bootstrap_point632_score
err = bootstrap_point632_score(estimator, X, y, n_splits=200, method='.632+')
```

## Trade-offs
- B=200 fits — same cost as Harrell's optimism (V02)
- Adapts to overfitting magnitude — better default for varied model libraries
- Survival adaptation requires custom implementation (mlxtend is classification-only)

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [V02 Harrell optimism](V02_harrell_bootstrap_optimism.md)
[← back to index](../README.md)
