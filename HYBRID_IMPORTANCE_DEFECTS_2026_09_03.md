# Two defects in `compute_feature_importance` — needs a decision, then a separate fix session

**Date:** 2026-09-03 · **Branch:** `similarity-metrics-study` · **Owner:** Adelaide
**Found while building** `tests/experiments/framework/f30_importance_methods.py`.
**Nothing here has been fixed.** The rebuild roadmap locks `hybrid.py` and
`find_best_hybrid.py`, so both of these need Adelaide's decision first.

Both defects are in `src/kirby/hybrid.py`. Both affect **past results**, not just f30.


> **Line numbers verified 2026-09-03 against the working tree.** `src/kirby/hybrid.py`
> has uncommitted changes from a parallel session, so these shifted once already and may
> shift again. Every reference below also names the function or the exact code, which is
> the durable anchor — trust the name over the number.

---

## Defect 1 — the 2,000-feature pre-screen ranks on floating-point rounding error

### Where
Three branches, identical code:

| method | line |
|---|---|
| `treeshap` | `hybrid.py:3414` |
| `xgboost_gain` | `hybrid.py:3709` |
| `lightgbm_gain` | `hybrid.py:3774` |

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clipped)
...
if X_scaled.shape[1] > max_features_lgb:          # default 2000
    variances = np.var(X_scaled, axis=0)          # <-- on the SCALED matrix
    selected_idx = np.argsort(variances)[-max_features_lgb:]
```

Features outside the kept 2,000 are written as **exactly 0.0**
(`hybrid.py:3464-3465`, `:3722-3727`, `:3780-3785`).

### What is wrong
`StandardScaler` sets every non-constant column's variance to 1. The screen then sorts a
vector of ones.

MEASURED 2026-09-03 on the local cache:

| representation | width | variance range after scaling | features written 0.0 |
|---|---|---|---|
| caco2/grover | 4800 | 0.999998 → 1.000000 (spread 4.8e-06) | 2,800 |
| caco2/ecfp4 | 2048 | 0 → 1.00011 (194 genuinely constant columns) | 48 |
| koenig/esm3_cdr | 3072 | 0.999885 → 1.000110 (spread 2.3e-04) | 1,072 |

Apart from genuinely constant columns, the ordering is float64 rounding error in the sixth
decimal place. On grover, 2,800 of 4,800 features are made unselectable on that basis.

### MEASURED 2026-09-04: on caco2 it costs nothing

f30 runs `lightgbm_gain` twice — once as production runs it, once with the screen
lifted through the existing `max_features_lgb` kwarg — so the effect is a number
rather than an inference. caco2, 3 group-aware folds, budget 100, five of its ten
representations wider than the 2000 cap:

| cell | capped (production) | uncapped | difference |
|---|---|---|---|
| power_2, downstream lightgbm | +0.0069 | +0.0022 | −0.0047 |
| power_2, downstream rf | +0.0100 | +0.0085 | −0.0015 |
| rank_exp, downstream lightgbm | +0.0062 | −0.0023 | −0.0085 |
| rank_exp, downstream rf | +0.0077 | +0.0057 | −0.0020 |

Lifting the screen is **worse in four cells of four**, and costs 56% more compute
(65.5s vs 42.0s of scoring).

**This corrects the framing above.** The screen is still arbitrary — it ranks a
vector of ones and discards 2,800 of grover's 4,800 features on rounding noise, and
that is worth fixing for correctness. But there is no evidence it costs predictive
performance, and on this dataset removing it hurts. The most likely reading is that
the discarded features are mostly noise on caco2, so halving the candidate pool acts
as regularisation.

Treat this as a correctness defect to tidy, NOT as an urgent one. Differences are
small and this is three folds on one dataset; the other three datasets will say more.

### USER DECISION 2026-09-04: KEEP THE SCREEN

Adelaide's call after seeing the four-cell comparison above. The screen stays as it
is. It costs nothing in accuracy on caco2, saves 36% of the scoring compute, and
removing it was worse in every cell. Do not re-raise this as a defect to fix.

The `lightgbm_gain_nocap` arm stays in f30 as the measurement that supports the
decision, not as a proposal to change production.

### Reach
Any representation wider than 2,000, scored by any of those three methods. In the four
f30 datasets: **5 of caco2's 10, 3 of buchwald_hartwig's 10, 7 of
flab_expression_koenig's 10, 0 of tcga_brca_pam50's 10.**

`deeplift` has no such screen, so it is the one incumbent that scores every feature.

### Options
1. **Compute the variance on `X_clipped`, before scaling.** One line per branch. Restores
   the screen's stated intent — keep the most variable features.
2. **Drop the screen and cap by a real cost limit** (e.g. a time budget or a sampled-row
   fit), since the screen exists only to keep the fit fast.
3. **Raise `max_features_lgb` / `_shap` / `_xgb`** at the call site. No code change — these
   are already `**kwargs` (`hybrid.py:3758`, `:3381`, `:3677`). This is what f30 does for
   its `lightgbm_gain_nocap` arm.
4. Leave it.

### If it is changed
Representations are not in the cache key, so a change to how features are scored needs a
scoped delete: for each affected dataset, `result_db/<ds>.db`, that dataset's search and
baseline fold caches, and its result JSONs. Not a blanket wipe.

---

## Defect 2 — `lightgbm_gain` cannot score most of a wide representation

### The mechanism
LightGBM's `gain` importance is the total gain of the splits that used a feature. A feature
that no tree ever split on has gain **exactly 0**. So the number of features that can
receive a non-zero score is bounded by the number of splits in the model:

```
max scorable features  ≤  n_estimators × (num_leaves − 1)
```

`compute_feature_importance` uses `n_estimators=50`, `max_depth=6`
(`hybrid.py:3755-3756`), leaving `num_leaves` at LightGBM's default 31. That is a ceiling
of **50 × 30 = 1,500** features, and the real count is far lower because boosting reuses
the features that work.

### Measured
On caco2/ecfp4 (2,048 features), through `compute_feature_importance`:

| method | features with a non-zero score |
|---|---|
| `lightgbm_gain` | **201 / 2048** |
| `xgboost_gain` | 441 / 2048 |
| `treeshap` | 865 / 2048 |

The other 1,847 features tie at exactly 0.0, and `np.argsort` then orders those ties by
column index.

### Why it matters here and does not matter there
- **Selection at budget 100 is fine.** Each representation's quota is roughly 10–50
  features, comfortably inside the ~200 that got scored. The hybrid is not picking
  arbitrary features.
- **Any agreement statistic between two methods is not fine.** A Spearman correlation over
  all 2,048 features is dominated by how two methods' tied-zero blocks happen to line up,
  not by how they rank the features they actually scored. f30 therefore reports every rank
  statistic twice: over all features, and over the union of features both methods scored
  non-zero.
- **At larger budgets it would start to matter for selection too** — a per-representation
  quota above ~200 would begin drawing from the tied zeros, i.e. by column index.

### Growing the model does NOT fix it — MEASURED 2026-09-03

caco2/ecfp4 (2158 x 2048). Target built as a linear combination of **40 planted
features** plus noise, so "how many of the 40 does the method's top 40 recover" is a
direct quality measure rather than a coverage one.

| n_estimators | max_depth | num_leaves | features given a score | planted features recovered |
|---|---|---|---|---|
| 50 (production) | 6 | 31 | 132 / 2048 | **12 / 40** |
| 200 | 6 | 31 | 268 / 2048 | **12 / 40** |
| 500 | 6 | 31 | 384 / 2048 | **12 / 40** |
| 200 | unlimited | 127 | 527 / 2048 | **9 / 40** |

Ten times the trees quadruples the number of features that get a non-zero score and
recovers **exactly the same 12** real ones. Widening the trees scores more features
still and recovers **fewer**.

So the sparsity is not a tuning problem. Extra capacity spreads gain over more
features without finding more of the true ones, and deeper trees actively hurt.

Caveat, stated: one representation, one seed, and a *linear* target over sparse binary
features — a shape trees are poor at. The remaining configurations in the sweep vary
depth and leaves further and were still running when this was written.

### Options
1. ~~**Raise `n_estimators` / `num_leaves`.**~~ **Ruled out by the table above.** It
   fixes the tie-at-zero symptom and does not improve which features get selected.
2. **Keep it as is and treat the tied zeros as "unranked"** rather than "worst", i.e. fix
   the analysis rather than the scorer.
3. **Cut `lightgbm_gain`.** Not recommended on this evidence: the sparsity is a property of
   gain importance at these hyperparameters, not a defect in the method, and option 1 is
   cheap to test.

---

## What f30 does about both, without changing any code

- Runs `lightgbm_gain` twice — once as production runs it, once with the pre-screen lifted
  through the existing kwarg — so the size of defect 1 is measured rather than assumed.
- Reports every rank-agreement statistic over all features **and** over the union of
  features both methods scored non-zero, so defect 2 cannot silently inflate agreement.
- Adds `rf_mdi`, which has no branch today: a call with `method='random_forest'` currently
  raises at `hybrid.py:4008`, despite the docstring at `hybrid.py:3362` referring to it.
