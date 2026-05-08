# F05 — Stacking with regularized meta-learner + nested CV

**Type:** Fusion / ensembling (learned meta-weights)
**Domain:** Many bases with calibrated outputs
**Cite:** Wolpert, "Stacked Generalization", Neural Networks 1992; Cawley & Talbot 2010 for nested-CV protocol

## What it is
1. Split data into outer K_o folds.
2. For each outer fold:
   - Within the outer-train, run inner K_i-fold CV to produce base OOF predictions.
   - Fit a meta-learner (regularized logistic / NNLS / isotonic) on (base_OOF, y_inner).
   - Refit each base on full outer-train, predict on outer-test.
   - Apply meta-learner to outer-test base predictions, score outer-test.
3. Average outer-fold scores; final meta-learner = retrain on full-data OOF.

## Why it helps
- Meta-learner can express *any* combination, not just the convex weights of F01–F04
- Regularization (L2 on logistic, sum-to-one + non-negative on NNLS) controls weight overfitting
- Nested CV cleanly separates selection from evaluation — exactly the leak that destroyed cherry_pick
- Can express interactions between bases that linear pooling cannot

## When to use
- Bases output well-calibrated probabilities (or probability-equivalent risks)
- Library size 5–50 (above 50, use F01 first to prune, then F05 on survivors)
- Hold-out set too small to use F04 / F08 alone

## When not to use
- Bases output rank-only scores — meta-learner has nothing to model on the probability scale
- < 200 events — meta-learner overfits; stick to F02 / F03 / F04

## Implementation
```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
# See V09 implementation sketch — meta-learner inside the inner loop

# For survival, convert risk scores to AFT-residuals or KM-target before stacking.
# Penalize meta-learner heavily; we want a near-uniform prior on bases.
meta = LogisticRegressionCV(Cs=[1e-2, 1e-1, 1.0], penalty='l2', cv=5)
meta.fit(base_oof_inner, y_event_inner)
test_pred = meta.predict_proba(base_oof_outer)[:, 1]
```

For monotonic blending, isotonic regression on a single index (e.g. variance-weighted mean of bases, then isotonic-mapped to event indicator) often matches L2-logistic with fewer pitfalls.

## Hyperparameter notes
- L2 strength: cross-validate over 3 orders of magnitude
- Restrict meta-learner to *positive* weights (NNLS or non-negative L2): more interpretable, often more robust
- Meta-learner family: try logistic, NNLS, isotonic, and pick whichever has smallest *outer-fold* metric variance, not best mean

## Trade-offs
- Compute = K_o × (K_i × base_fits + meta_fit + base_refits)
- For K_o=5, K_i=5: 30× base-fit cost
- Sensitive to meta-learner family and regularization strength

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [V09 nested CV](../validation/V09_nested_cross_validation.md)
[← back to index](../README.md)
