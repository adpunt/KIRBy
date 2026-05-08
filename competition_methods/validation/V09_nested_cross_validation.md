# V09 — Nested cross-validation

**Type:** Validation protocol (proper stacking / hyperparameter tuning)
**Domain:** Any setting where model selection or weight fitting touches OOF predictions
**Cite:** Cawley & Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", JMLR 2010

## What it is
Two nested CV loops:
- **Outer fold (K_o splits):** evaluation. Each outer fold's test split is used *only* for measuring final ensemble performance.
- **Inner fold (K_i splits):** selection. Within the outer fold's training portion, run an inner CV to (a) generate OOF base predictions, (b) fit meta-learner / fusion weights, (c) tune hyperparameters.

Critically, the meta-learner is *retrained per outer fold* using only that fold's inner OOF — so the outer test split is never seen by anything except the final evaluation.

## Why it helps
Standard "fit base models with K-fold CV, then fit meta-learner on the resulting OOF, then evaluate on the same OOF" is the textbook leak that produces all selection-bias-on-OOF disasters. The meta-learner has effectively seen the test labels (through the OOF predictions of base models that were fit excluding those rows). Nested CV breaks the loop.

## When to use
- Fitting a stacking meta-learner (logistic regression, isotonic, NNLS)
- Hyperparameter tuning where the tuning metric is the same as the evaluation metric
- Any "fit weights on OOF" fusion variant

## When not to use
- Pure averaging (no fitted weights → no leak)
- True hold-out test set already exists (V08 simpler)
- Computationally infeasible (K_o × K_i × base-fit cost can be 25× a single fit)

## Implementation
```python
from sklearn.model_selection import StratifiedKFold
outer = StratifiedKFold(K_o, shuffle=True, random_state=0)
outer_perf = []
for tr_o, te_o in outer.split(X, y_event):
    # Inner CV fits base models + builds meta-OOF
    inner = StratifiedKFold(K_i, shuffle=True, random_state=1)
    base_oof = np.zeros((len(tr_o), n_bases))
    for tr_i, va_i in inner.split(X[tr_o], y_event[tr_o]):
        for b, model in enumerate(bases):
            model.fit(X[tr_o][tr_i], y[tr_o][tr_i])
            base_oof[va_i, b] = model.predict_risk(X[tr_o][va_i])
    meta = MetaLearner().fit(base_oof, y[tr_o])
    # Predict on outer test using bases retrained on the full outer-train
    base_te = np.column_stack([
        b.fit(X[tr_o], y[tr_o]).predict_risk(X[te_o]) for b in bases])
    outer_perf.append(c_index(meta.predict(base_te), y[te_o]))
print(f'nested-CV C: {np.mean(outer_perf):.4f} ± {np.std(outer_perf):.4f}')
```

## Trade-offs
- Cost = K_o × (K_i × base_fit + meta_fit + base_refit)
- For K_o=5, K_i=5, that's 30 base fits per base model — multiply by library size
- Variance reduction often justifies cost; results are publication-quality

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [V08 hold-out gate](V08_holdout_fusion_test_gate.md), [F05 stacking + nested CV](../fusion/F05_stacking_nested_cv.md)
[← back to index](../README.md)
