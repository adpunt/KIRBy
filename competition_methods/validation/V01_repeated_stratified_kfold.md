# V01 — Repeated Stratified K-Fold CV

**Type:** Validation protocol
**Domain:** Any classification or survival with imbalanced events
**Cite:** Standard practice; Royston & Altman 2013 for survival prognostic models

## What it is
Run StratifiedKFold(K) `R` times with different seeds, stratifying on the event flag (or class label). Aggregate per-patient OOF predictions across repeats; report mean and SE of the validation metric across repeats. Typical settings: 5 folds × 5 repeats, or 10×5 for very small datasets.

## Why it helps
Single-split K-fold has fold-assignment variance that dominates real model differences when events are rare. With n=1106 / 47 events / 5 folds, you have ~9 events per fold; one unlucky split moves apparent C by ±0.02. Repeating reduces this variance proportionally to √R.

## When to use
- n < few thousand
- Events rare (<10% prevalence)
- Comparing models whose claimed lifts are <0.02

## When not to use
- Truly large data (n > 50k) where single-seed fold variance is already small
- Computationally prohibitive base models (deep nets that take hours to fit)

## Implementation
```python
from sklearn.model_selection import StratifiedKFold
oofs = []
for seed in range(R):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan)
    for tr, va in skf.split(X, y_event):
        m.fit(X[tr], y[tr]); oof[va] = m.predict_risk(X[va])
    oofs.append(oof)
oof_mean = np.mean(oofs, axis=0)  # rank-stable per patient
```

## Trade-offs
- Linear cost in R
- Per-patient OOF *averaged across repeats* is more stable than any single repeat — this becomes the OOF used downstream
- Per-repeat metric SE is the headline number — anything within 1× SE of anchor is noise

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | Harrell C | TBD | Currently using single seed=42; switch pending |

---
[← back to index](../README.md)
