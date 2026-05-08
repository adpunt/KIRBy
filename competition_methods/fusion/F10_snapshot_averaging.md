# F10 — Snapshot averaging within base models

**Type:** Variance reduction (pre-fusion)
**Domain:** Any base model with a stochastic training process (random init, random fold split, random feature subsampling)

## What it is
Train each base model R times with different random seeds (or different bootstrap samples). Average the predictions before passing to the fusion stage. This becomes the "single base" in the fusion library.

## Why it helps
- Most base-model variance comes from random initialization, fold-split assignment, or stochastic feature subsampling — *not* from epistemic uncertainty about the data
- Averaging across seeds is a pure variance reduction: bias unchanged, variance / R
- Reduces the noise the *fusion* layer has to absorb — fusion can then focus on real inter-base differences instead of intra-base seed jitter

## When to use
- Bases are stochastic (random forests, random survival forests, neural nets, gradient boosting with `subsample < 1`)
- Computationally affordable (R = 5 typically, R = 10 for nets)
- Before any fusion / stacking step

## When not to use
- Deterministic bases (closed-form Cox, ridge regression)
- When R is too small to actually reduce variance (R < 3 has high estimator variance itself)

## Implementation
```python
def snapshot_average(model_factory, X_train, y_train, X_predict, R=5, base_seed=0):
    preds = []
    for r in range(R):
        m = model_factory(random_state=base_seed + r)
        m.fit(X_train, y_train)
        preds.append(m.predict_risk(X_predict))
    return np.mean(preds, axis=0)
```

For survival K-fold CV:
```python
oof_seeded = []
for seed in range(R):
    oof = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y_event):
        m = model_factory(random_state=seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_risk(X[va])
    oof_seeded.append(oof)
oof_avg = np.nanmean(oof_seeded, axis=0)
```

## Hyperparameter notes
- R = 5 is a sane default for forests / boosting; R = 10 for nets where variance is larger
- Average ranks rather than raw scores if base output scales drift between seeds
- Combine with V01 repeated K-fold: outer loop over seeds is natural

## Trade-offs
- Linear cost in R
- Doesn't help with bias — only variance
- Often the cheapest free lunch in an ensemble pipeline

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | Combined LB | TBD | Multi-seed RSF (task #88) gave death +0.0005 free, hep no gain |

---
**See also:** [V01 repeated K-fold](../validation/V01_repeated_stratified_kfold.md)
[← back to index](../README.md)
