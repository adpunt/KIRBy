# F06 — Diversity-penalized ensemble selection

**Type:** Fusion / ensembling (regularized greedy)
**Domain:** Large model libraries with many near-duplicate models
**Cite:** Caruana, Munson & Niculescu-Mizil, "Getting the Most Out of Ensemble Selection", ICDM 2006

## What it is
Forward greedy selection (à la F01) but each step's score is:
```
score(m) = metric(running ∪ m) − λ * mean_pairwise_corr(m, running)
```
where corr is on the prediction vectors (or residuals — see V11). λ controls the strength of the diversity penalty.

## Why it helps
- F01 alone tends to collapse onto one base family when that family dominates the metric: select the best XGBoost variant, then the second-best XGBoost variant, etc.
- Diversity penalty *forces* breadth: once a family is represented, the next pick has to be from a different region of model space to win
- Empirically (ICDM 2006) gives 0.005–0.015 lift over plain greedy, and substantially more under strong base correlations

## When to use
- Library has many similar candidates (e.g. RSF with different hparams + Cox with different penalties + lightgbm with different seeds)
- F01 produced an ensemble dominated by one family
- Want robustness to single-family failure (one base goes bad → diversity-penalized fallback degrades less)

## When not to use
- Already-diverse library where diversity is the bottleneck reason for a high penalty
- Tiny library (< 10 candidates)

## Implementation
```python
def caruana_div_penalized(oof_dict, y, lam=0.1, max_size=30, init_k=5):
    names = list(oof_dict.keys())
    init = sorted(names, key=lambda n: metric(oof_dict[n], y), reverse=True)[:init_k]
    ensemble = list(init)
    running = np.mean([oof_dict[n] for n in ensemble], axis=0)
    while len(ensemble) < max_size:
        best, best_score = None, -np.inf
        for n in names:
            cand_pred = (running * len(ensemble) + oof_dict[n]) / (len(ensemble) + 1)
            m = metric(cand_pred, y)
            corr = np.mean([
                np.corrcoef(oof_dict[n], oof_dict[e])[0, 1] for e in ensemble])
            s = m - lam * corr
            if s > best_score: best_score, best = s, n
        ensemble.append(best)
        running = (running * (len(ensemble) - 1) + oof_dict[best]) / len(ensemble)
    return ensemble
```

## Hyperparameter notes
- `lam = 0.1`: standard starting point; tune via hold-out (V08)
- Use *residual* correlation (V11) instead of prediction correlation when probability calibration is the priority
- Combine with bagging (F01 outer loop) for variance reduction

## Trade-offs
- Hyperparameter λ requires V08 hold-out to tune
- Marginal lift over F01 in already-diverse libraries
- Strong fix when diversity is the actual bottleneck

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [F01 plain bagged selection](F01_caruana_bagged_selection.md), [V11 residual diversity](../validation/V11_brier_residual_diversity.md)
[← back to index](../README.md)
