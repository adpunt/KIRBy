# Framework Experiment Roadmap

**Goal:** Empirically test every component of the hybrid search machinery across domains,
identify what's useless, refine the speed tiers and algorithms, and produce validated
final search settings.

**Status:** Wave 1 partial results in (2026-03-18)

---

## The Problem

The hybrid search (`find_best_hybrid`) has a large combinatorial configuration space:

| Component | Options | Count |
|-----------|---------|-------|
| Allocation methods | greedy, greedy_feature, performance_weighted (6 weighting schemes), dominant_greedy, two_stage, stability_selection, fixed | 7 base / 17 named |
| Importance methods | treeshap, xgboost_gain, lightgbm_gain, deeplift, permutation, kernelshap, integrated_gradients, gradientshap | 8 |
| Budgets | 50, 100, 200, 500 | 4 |
| Step sizes (greedy-based) | 1, 2, 5, 10, 25, 50 | 6 |
| Patience (greedy-based) | 2, 3, 5, None | 4 |
| Models (downstream) | ridge, rf, xgb, lgbm, knn, mlp, svm, gp, catboost, dnn, cnn | 11 |
| Augmentation strategies | all, none, greedy_ablation | 3 |
| Internal evaluator | RF (hardcoded in `_evaluate_allocation` and `_create_hybrid_with_allocation`) | 1 |
| Adaptive pruning threshold | 0.3–0.7 | continuous |
| Rep screening | None (all reps always enter search) | 0 |

The current speed tiers (`SPEED_TIERS` in `find_best_hybrid.py`) were designed theoretically,
not empirically. The `fast` tier (5 effective configs) might miss the best config. The `long`
tier (424 effective configs) might waste 90% of compute on methods that never win.

**We need to test each component, identify dead weight, and rebuild the tiers from data.**

### Key coupling to understand

Not all components are independent:

- **Importance-insensitive allocations** (greedy, dominant_greedy) ignore importance_method
  entirely — they use RF `feature_importances_` internally. So importance × allocation is
  NOT a full cross-product.
- **Step size and patience** only matter for greedy-based allocations (greedy, dominant_greedy,
  greedy_feature). Performance_weighted/fixed/two_stage/stability_selection ignore them.
- **The internal evaluator** (RF in `_evaluate_allocation`) is used inside greedy allocation
  loops to score trial allocations. This is separate from the downstream model used in
  `find_best_hybrid`. Changing the internal evaluator could change which allocations greedy picks.
- **Augmentations** are evaluated AFTER the base hybrid is built. They are additive — testing
  them is somewhat independent of the core allocation/importance/budget grid.

---

## Experiment Inventory

### All framework experiments (tests/experiments/framework/)

| Script | What it tests | Tasks | Finding | Implemented? |
|--------|-------------|-------|---------|-------------|
| **f00** | Rep screening strategies | **Done** 3+7 retest | D_augmentation most consistent. C_relative fixed, threshold 0.1–0.3 optimal depending on domain | `_screen_reps()` exists in find_best_hybrid.py. Not yet wired as default |
| **f01** | 17 allocation strategies | **3/7** (luad, brca, fcrnrt). 4 submitted | stability_selection best on large data. greedy/greedy_feature worst. Performance_weighted safest default | **Yes** — greedy/greedy_feature removed from SPEED_TIERS. performance_weighted is fast-tier default |
| **f02** | Adaptive pruning simulation | **Blocked** on f01+f03 | — | — |
| **f03** | Greedy step_size × patience | **Resubmitted** (all 7 timed out). Now: 4 models, budgets 50/150/300, 1 fold | — | — |
| **f04** | Two-phase vs full grid | **Resubmitted** (all 7 timed out, 1 FAILED). Now: 4 models, budgets 50/150/300, 1 fold | — | — |
| **f05** | Importance method redundancy | **Done** 7/7 | 3 families: treeshap, deeplift, lightgbm_gain. Drop 5 others | **Yes** — SPEED_TIERS uses only these 3 |
| **f06** | Augmentation strategies | **6/7** (missing logd) | greedy_ablation wins molecular, core+all wins antibody. All beat single reps | **No** — augmentation_strategy still defaults to greedy_ablation everywhere |
| **f07** | Internal evaluator (RF vs XGB vs Ridge) | **Done** 5/7, cancelled rest | Ridge wins antibody, fastest overall | **No** — `internal_evaluator` param not added. Still hardcoded RF |
| **f08** | Iterative augmentation hyperparams | **6/7** (missing logd) | auto_threshold=50, base_step_size=2–10 matter most. candidate_ranking/ab_eval_model don't matter | **No** — iterative augmentation not in hybrid.py yet |
| **f09** | Cross-domain synthesis | **Not written** | — | — |
| **f10** | Small dataset fixes (n<200) | **Done** 4/4 | No fix recovers hybrid on n~130. Budget cap, ridge eval, CV3 all fail | **No** — no small-dataset guard |
| **f11** | Allocation × evaluator interaction | **Done** 5/5 | Marginal gains with power_2+ridge_cv3 on some tasks. Root issue unfixable by tuning | **No** — same as f10 |

---

## Results (2026-03-18)

### F00: Rep Screening (3/7 tasks complete)

**Datasets:** herg (molecular, RMSE↓, n=1412), buchwald_hartwig (reaction, RMSE↓, n=3955), flab_pk_jain_fcrnrt (antibody, Spearman↑, n=132)

#### Summary table

| Strategy | herg RMSE↓ | buchwald RMSE↓ | fcrnrt ρ↑ | Notes |
|----------|-----------|----------------|-----------|-------|
| Best single rep | 0.618 (ecfp4) | 14.26 (reactant_ecfp4) | 0.483 (igt5) | |
| A_hard_gate_p50 | **0.613** | 14.53 | — | Sensitive to percentile |
| A_hard_gate_p75 | 0.611 | 14.75 | — | |
| B_implicit | 0.618 | 14.67 | 0.259 | Slow (~2500s), no clear advantage |
| C_relative_* | **ALL NaN** | **ALL NaN** | 0.240–0.289 | **BUG** on lower-is-better metrics |
| D_augmentation_core3 | 0.622 | **14.06** | **0.436** | Consistent, fast |
| D_augmentation_core4 | 0.610 | **14.01** | 0.384 | Best on reaction |
| D_augmentation_core5 | **0.608** | 15.17 | 0.346 | Best on molecular |

#### C_relative sign bug

**Location:** `f00_rep_screening.py:278-281`

```python
comparable_scores = {k: _score_to_comparable(v, lower_is_better) ...}
best_comparable = max(comparable_scores.values())  # = -0.618 for herg
cutoff_comparable = fraction * best_comparable      # = 0.1 × (-0.618) = -0.0618
# BUG: cutoff (-0.0618) is ABOVE all scores (-0.618 to -0.688)
# → all reps screened out
```

When `lower_is_better=True`, comparable scores are negative. Multiplying by a positive fraction makes the cutoff *less negative* than all scores → everything filtered.

**Fix:** Use gap-based threshold:
```python
best_comparable = max(comparable_scores.values())
cutoff_comparable = best_comparable - fraction * abs(best_comparable)
```

**Status:** Fix needed, then re-run herg + buchwald_hartwig. fcrnrt results are valid (positive scores).

#### Key findings

1. **D_augmentation is the most consistent strategy** — beats best single rep on 2/3 tasks.
2. **A_hard_gate is sensitive to percentile** — p50 sometimes wins but p25 is too aggressive.
3. **B_implicit is slow** (~2500s) without clear advantage over D.
4. **Small datasets (n=132) resist hybridization** — no strategy beats best single rep on fcrnrt.
5. **Remaining tasks needed:** logd, flab_thermo_tresanco, tcga_luad_vs_lusc, tcga_brca_pam50 (plus re-runs after C_relative fix).

---

### F05: Importance Method Correlation (7/7 tasks complete)

**All 8 methods ran on all 7 datasets.** Key patterns are consistent across domains.

#### Redundancy clusters

| Cluster | Methods | Typical ρ | Fastest in cluster |
|---------|---------|-----------|-------------------|
| Gradient-based | deeplift ≈ integrated_gradients ≈ gradientshap | 0.89–1.00 | **deeplift** (2s) |
| Tree-permutation | treeshap ≈ permutation | 0.90–0.96 | **treeshap** (6–75s) |
| Tree-gain | xgboost_gain ≈ lightgbm_gain | 0.47–0.87 | **lightgbm_gain** (1–8s) |
| Outlier | kernelshap | avg ρ = 0.27–0.54 | N/A (2800–4500s, slowest by far) |

#### Per-dataset highlights

- **buchwald_hartwig** (reaction): Extreme redundancy — deeplift/IG/gradientshap ρ=0.998–1.000. treeshap/deeplift ρ=0.988. xgboost/lightgbm ρ=0.872.
- **herg** (molecular): deeplift/IG ρ=0.951. treeshap/permutation ρ=0.916. KernelSHAP isolated (avg ρ=0.39).
- **logd** (molecular): Same pattern. treeshap/permutation ρ=0.962. deeplift/IG ρ=0.943.
- **flab_pk_jain_fcrnrt** (antibody, n=132): Lower correlations overall (max ρ=0.81). All methods more independent on small data.
- **flab_thermo_tresanco** (antibody, n=672): deeplift/IG ρ=0.987. treeshap/permutation ρ=0.899.
- **tcga_brca_pam50** (TCGA): Data pending detailed analysis.
- **tcga_luad_vs_lusc** (TCGA): Data pending detailed analysis.

#### Decision (pending confirmation from remaining experiments)

**Use 3 methods: treeshap + deeplift + lightgbm_gain**

Rationale:
- Covers 3 independent families (tree-permutation, gradient, tree-gain)
- Combined wall time ~10–80s vs ~3500s for all 8
- KernelSHAP dropped: slowest by 10–100x, lowest correlation with others (unique signal or noise — not worth the cost)
- permutation dropped: redundant with treeshap (ρ>0.90)
- integrated_gradients + gradientshap dropped: redundant with deeplift (ρ>0.95), 15–300x slower

**TODO:** Confirm with f01/f03/f04 results that downstream performance doesn't depend on the dropped methods.

---

### F06: Augmentation Strategy Test (4/7 tasks complete)

**Datasets:** buchwald_hartwig, flab_pk_jain_fcrnrt, tcga_luad_vs_lusc, tcga_brca_pam50

Splits reps into core (top 5 by single-rep baseline) and candidates (rest). Tests 3 augmentation strategies.

#### Summary table

| Dataset | Best single | core+greedy_ablation | core+iterative | core+all |
|---------|------------|---------------------|----------------|----------|
| buchwald_hartwig (RMSE↓) | 15.99 | **14.20** (rf, b=200) | 14.54 (rf, b=200) | 14.61 (rf, b=200) |
| fcrnrt (ρ↑) | 0.427 | 0.326 (rf) | 0.309 (rf) | **0.377** (ridge) |
| tcga_luad_vs_lusc (acc↑) | 0.942 | 0.963 (xgb) | **0.968** (xgb) | 0.963 (xgb) |
| tcga_brca_pam50 (acc↑) | 0.817 | 0.884 (ridge) | **0.884** (ridge) | 0.876 (xgb) |

#### Key findings

1. **All strategies beat single reps** on 3/4 tasks (exception: fcrnrt n=132).
2. **greedy_ablation and iterative are comparable** — neither consistently dominates.
3. **augmentation_budget has minimal impact** (10 vs 50 give similar results).
4. **Model choice matters** — ridge best for TCGA/antibody, rf/xgb best for reaction.

#### Relationship to F08

F06 answers *which strategy* (greedy_ablation vs iterative vs all). F08 answers *how to tune iterative's hyperparameters*. F06's iterative results use default hyperparams → lower bound. F06 is NOT redundant.

---

### F07: Internal Evaluator (4/7 tasks — CANCELLED remaining)

**Datasets:** flab_pk_jain_fcrnrt, flab_thermo_tresanco, tcga_brca_pam50, tcga_luad_vs_lusc

Tests RF vs XGBoost vs Ridge as the internal evaluator in greedy allocation loops. Also tests greedy vs dominant_greedy allocation and budget 100 vs 200.

#### Summary table (best downstream score per dataset)

| Dataset | Best internal evaluator | Allocation | Budget | Best downstream | Score |
|---------|----------------------|------------|--------|----------------|-------|
| flab_pk_jain_fcrnrt (ρ↑) | **ridge** | greedy | 100 | ridge | 0.435 |
| flab_thermo_tresanco (ρ↑) | **ridge** | greedy | 100 | lightgbm | 0.796 |
| tcga_brca_pam50 (acc↑) | rf | dominant_greedy | 200 | xgboost | 0.896 |
| tcga_luad_vs_lusc (acc↑) | xgboost | greedy | 100 | xgboost | 0.963 |

#### Decision

**Internal evaluator becomes a configurable parameter with `ridge` as default.**

Rationale:
- Ridge wins on antibody (both tasks).
- No single evaluator dominates across all domains.
- Ridge is fastest of the three.
- Expose as `internal_evaluator` kwarg in `find_best_hybrid()` / `create_hybrid()`.

**Remaining F07 runs cancelled** — enough data to make the decision.

---

### F08: Iterative Augmentation Hyperparams (6/6 tasks complete)

**Datasets:** buchwald_hartwig, flab_pk_jain_fcrnrt, flab_thermo_tresanco, herg, tcga_brca_pam50, tcga_luad_vs_lusc

Tests 7 hyperparameter axes for iterative augmentation: auto_threshold, candidate_ranking, worst_feature_method, patience, aug_step_size, base_step_size, ab_eval_model. 1152 configs per dataset.

#### What matters (ranked by % of score variance explained)

| Axis | Matters? | Best value | Notes |
|------|----------|-----------|-------|
| **base_step_size** | **YES — dominant** | 2 or 5 | step=1 is terrible (starves reps). Controls initial hybrid quality. |
| **auto_threshold** | **YES** | 50 | Prunes weak reps, frees candidates. 10 often leaves zero candidates. |
| **ab_eval_model** | Moderate | rf_proxy | Slightly better than downstream in 4/6. |
| aug_step_size | Weak | 1 | Fine-grained slightly better. |
| patience | No (<1%) | any | Search converges fast. |
| worst_feature_method | No (<1%) | any | permutation ≈ allocation_scores. |
| candidate_ranking | No (<1%) | any | rf_importance ≈ residual_correlation. |

#### Augmentation impact per dataset

| Dataset | Best base | Best augmented | Improvement | Note |
|---------|----------|---------------|-------------|------|
| buchwald_hartwig (RMSE↓) | 17.29 | 15.74 | **-8.9%** | 3 fills + 5 replaces |
| flab_pk_jain_fcrnrt (ρ↑) | 0.402 | 0.786 | **+79.4%** | Massive gain |
| flab_thermo_tresanco (ρ↑) | 0.754 | 0.823 | **+5.3%** | Modest |
| herg (RMSE↓) | 0.593 | 0.593 | 0% | No candidates (all reps pass threshold) |
| tcga_brca_pam50 (acc↑) | 1.000 | 1.000 | 0% | Ceiling effect |
| tcga_luad_vs_lusc (acc↑) | 0.980 | 0.996 | **+1.2%** | Near ceiling |

#### Key finding

Augmentation essentially never hurts (0% of configs worse than their own base). The dominant factors are base_step_size and auto_threshold — getting the initial hybrid right matters far more than tuning the augmentation search.

#### Decision (2026-03-25)

- `base_step_size` default changed to 5 (was 10)
- `auto_threshold` irrelevant — user hand-picks augmentation reps in DATASET_REPS
- `patience=20`, `candidate_ranking='rf_importance'`, `worst_feature_method='permutation'` as safe defaults
- Implemented in `hybrid.py` and `find_best_hybrid.py`

---

### F11: Allocation x Evaluator Interaction (5/5 tasks complete)

**Datasets:** flab_aggregation_jain_hicrt (n=137), flab_pk_jain_fcrnrt (n=132), flab_pk_jain_heprt (n=130), flab_polyreact_jain_cic (n=137), flab_thermo_tresanco (n=672, control)

Tests 5 allocation strategies x 4 evaluator conditions x 2 budgets + 2 importance-weighted allocations = 44 conditions. All small antibody datasets + one control.

#### Summary table

| Dataset | n | Best single rep | Best hybrid | Delta |
|---------|---|----------------|-------------|-------|
| aggregation_hicrt | 137 | prot_t5/ridge 0.504 | softmax_T2/rf_single b=100: 0.420 | -16.7% |
| pk_fcrnrt | 132 | ablang2/ridge 0.595 | power_2/rf_single b=100: 0.460 | -22.6% |
| pk_heprt | 130 | igt5/ridge 0.783 | imp_treeshap b=100: 0.777 | -0.7% |
| polyreact_cic | 137 | prot_t5/ridge 0.585 | power_2/ridge_cv3 b=100: 0.516 | -11.8% |
| thermo_tresanco | 672 | blosum/lgbm 0.780 | min_floor/rf_single b=100: 0.767 | -1.8% |

#### Key findings

1. **No hybrid condition beats best single rep on any dataset.** 44 conditions tested, zero consistently win.
2. **Evaluator condition is irrelevant.** Switching rf_single → ridge_cv3 changes nothing (max delta 0.017).
3. **No allocation strategy dominates.** power_2 has best mean rank (3.2) but never enough to close the gap.
4. **Gap scales with 1/n.** -1.8% at n=672, -0.7% to -22.6% at n=130.
5. **Root cause is structural:** ~22 training samples per fold-split cannot reliably distinguish signal from noise in greedy search.

#### Decision

**Root issue is unfixable by tuning.** No small-dataset guard added — the problem is fundamental sample efficiency, not a hyperparameter to optimize.

---

## Implementation Next Steps

### Immediate (before more experiments finish)

1. **Fix C_relative sign bug** in `f00_rep_screening.py:281`
   - Change `cutoff_comparable = fraction * best_comparable` to gap-based threshold
   - Re-run f00 on herg + buchwald_hartwig (the two lower-is-better tasks that failed)
   - Also run the 4 missing tasks (logd, tresanco, luad_vs_lusc, brca_pam50)

2. **Add `internal_evaluator` parameter** to `find_best_hybrid()` and `create_hybrid()`
   - Default: `'ridge'`
   - Accepted values: `'rf'`, `'xgboost'`, `'ridge'`
   - Wire into `_evaluate_allocation()` and `_create_hybrid_with_allocation()`

### After f01/f03/f04 results arrive

3. **Confirm importance method reduction** — verify that f01/f03 downstream scores don't depend on dropped methods (permutation, kernelshap, IG, gradientshap)
4. **Update SPEED_TIERS** — reduce importance_methods from 8 to 3 (treeshap, deeplift, lightgbm_gain)

### After f08 results arrive

5. **Compare F06 vs F08** — do F08's tuned iterative hyperparams change the strategy ranking from F06?
6. **Set iterative augmentation defaults** based on F08's best hyperparameters

### After all Wave 1 + Wave 2 results

7. **Run f09 cross-domain synthesis** — consumes all JSON outputs, produces final recommendations
8. **Rebuild SPEED_TIERS from data** — allocation methods, importance methods, budgets, step sizes

### Code changes (after f09 synthesis)

9. **Implement rep screening** — if D_augmentation holds across all tasks, add `screen_reps='augmentation'` as default pathway in `find_best_hybrid()`
10. **Remove dead importance methods** from default configs (keep available for explicit use)
11. **Update default budget/step_size** based on f03 plateau analysis

---

### Why f01–f04 need rewriting (not updating)

The existing scripts were written against an older, regression-only, single-domain codebase.
The problems are structural, not cosmetic:

**`_shared.py` (the foundation):**
- `get_model_factories()` has 6 models. `DEFAULT_MODELS` in `find_best_hybrid.py` has 11.
  GP, CatBoost, DNN, CNN are never tested in framework experiments.
- Every evaluation helper (`evaluate_cv`, `evaluate_single_rep_cv`, etc.) hardcodes
  `r2_score`. Broken for classification (TCGA), Spearman (antibody), CI (DTA).
- `INTERMEDIARY` and `FULL` configs hardcode model lists and allocation methods that don't
  match current `SPEED_TIERS`.

**f01 (allocation strategies):**
- Line 130: hardcodes `r2_score(y_val, model.predict(X_val_s))`. Broken for classification.
- Line 71: hardcodes `importance_method = 'lightgbm_gain'`.
- Doesn't log allocation dicts — can't see which reps got features.
- Missing 4 allocation families: dominant_greedy, two_stage, stability_selection, greedy_feature.

**f02 (adaptive pruning):**
- Pruning threshold comparison (line 98) uses `best < threshold * global_best_score` which
  is directionally wrong for lower-is-better metrics (RMSE, MAE).
- Only simulates pruning of models. Current `_run_search` prunes models, importance methods,
  AND allocation methods. f02 doesn't test the latter two.

**f03 (greedy step size):**
- Line 97: hardcodes `r2_score`. Line 80: hardcodes `importance_method='lightgbm_gain'`.
- Only tests greedy. Should also test dominant_greedy and greedy_feature (the other
  step-size-sensitive methods).
- Has the best logging of the 4 (allocation dicts, timing, stop reasons) — this design
  should be preserved in the rewrite.

**f04 (two-phase search):**
- Line 98: hardcodes `r2_score`. Line 85: hardcodes `importance_method='lightgbm_gain'`.
- Full grid (manual) and two-phase (`find_best_hybrid`) use different model sets — results
  not directly comparable.

**Cross-cutting:**
- No experiment supports classification tasks (TCGA, trastuzumab, humanness).
- No experiment uses domain-appropriate metrics.
- 5 of 11 models (GP, SVM, CatBoost, DNN, CNN) never tested.
- No structured logging format — each script has ad-hoc output.

---

## Implementation Plan

### Step 0: Rewrite _shared.py

**This unblocks everything. Do this first.**

Changes needed:

1. **Replace `get_model_factories()`** — use `find_best_hybrid.DEFAULT_MODELS` and
   `find_best_hybrid.get_model()` instead of maintaining a separate catalog. All 11 models
   available, with the same size guards and fallbacks.

2. **Add metric dispatch** — new function `get_domain_metric(data)` that returns
   `(metric_name, metric_fn, lower_is_better)` based on `data['domain']` and
   `data['task_type']`. All evaluation helpers use this instead of hardcoded `r2_score`.

3. **Rewrite evaluation helpers** — `evaluate_cv()`, `evaluate_cv_with_train()`,
   `evaluate_single_rep_cv()` all accept a `metric_fn` parameter. Remove hardcoded
   `r2_score` calls.

4. **Update INTERMEDIARY/FULL configs** — sync model lists with `DEFAULT_MODELS`. Sync
   allocation/importance lists with current `SPEED_TIERS`.

5. **Add structured logging** — `save_experiment_results()` writes JSON with common schema:
   ```python
   {
       "experiment": "f01",
       "dataset": "logd",
       "domain": "molecular",
       "timestamp": "...",
       "primary_metric": "rmse",
       "lower_is_better": true,
       "config": { ... },
       "results": [ ... ],
       "summary": { ... },
       "meta": { "n_samples": ..., "n_reps": ..., "rep_dims": ..., ... }
   }
   ```

6. **Add cross-domain runner** — helper that runs an experiment function across a list of
   (dataset, rep_preset) pairs, saving results per task.

### Step 1: Rewrite f01 (allocation strategies)

Preserve the experiment design (strategy grid × budgets × models → heatmap). Rebuild:

- Use `get_domain_metric()` for scoring, not hardcoded `r2_score`.
- Use `get_model()` from `find_best_hybrid` for model instantiation.
- Accept `importance_method` as CLI arg (default: treeshap).
- Add all 17 NAMED_ALLOCATIONS (currently missing dominant_greedy, two_stage,
  stability_selection, greedy_feature).
- **Log allocation dict per config** — which reps got how many features. This is the
  key missing output.
- Log wall time per config.
- Use structured JSON output.

### Step 2: Rewrite f03 (greedy step size)

Preserve the experiment design (step × patience × budget × model grid with diagnostics).
Rebuild:

- Domain-aware metrics.
- Models from `DEFAULT_MODELS`.
- **Expand beyond greedy-only** — also test dominant_greedy and greedy_feature (both are
  step-size-sensitive).
- Configurable importance method.
- Keep the good parts: allocation dict logging, timing, stop reason analysis, utilization
  tracking.

### Step 3: Rewrite f02 (adaptive pruning)

Preserve the simulation approach. Rebuild:

- **Fix directional bug** — use `_score_to_comparable()` from `find_best_hybrid` instead
  of raw `best < threshold * global_best_score`.
- **Prune all three dimensions** — models, importance methods, AND allocation methods.
  Current f02 only prunes models.
- Accept pre-computed results JSON from f01/f03 (not just its own `find_best_hybrid` run).
- Domain-aware metrics.
- Use structured JSON output.

### Step 4: Rewrite f04 (two-phase search)

Preserve the full-grid-vs-two-phase comparison. Rebuild:

- Domain-aware metrics.
- Models from `DEFAULT_MODELS` for both full grid and two-phase (currently mismatched).
- Configurable allocation list and importance method (currently hardcoded).
- Use structured JSON output.

### Step 5: Write f05 (importance correlation)

New script. For each task:
1. Concatenate all reps into one feature matrix.
2. Compute feature importance with all 8 methods.
3. Compute pairwise Spearman rank correlation.
4. Measure wall time per method.
5. Output: 8×8 correlation matrix + timing table.

### Step 6: Write f06 (augmentation test)

New script. For each task:
1. Split reps into "core" (top N by baseline) and "candidate" (rest).
2. Test 4 conditions: all-as-base, core+candidates-as-greedy_ablation,
   core+candidates-as-all, core-only.
3. Sweep augmentation_budget: [10, 20, 50].
4. Compare scores, which augmentations were kept.

### Step 7: Write f07 (internal evaluator)

New script. Monkey-patch `hybrid._evaluate_allocation` with configurable model
(RF vs XGBoost vs Ridge). For each task:
1. Run greedy allocation with each internal evaluator.
2. Compare: resulting allocation dicts, final hybrid scores with real downstream model.

### Step 8: Write f09 (cross-domain analysis)

New script. Consumes JSON outputs from f00–f08 across all tasks. Produces:
- Allocation method ranking per domain (which wins most often?)
- Importance method ranking (correlation analysis from f05 + downstream validation from f01)
- Budget/step sensitivity summary (plateau points from f03)
- Model ranking comparison (baseline vs hybrid, from f01+f03)
- Rep screening recommendation (from f00)
- Pruning threshold recommendation (from f02)
- **Proposed new SPEED_TIERS** based on all data.

---

## Execution Phases

### Phase 0: Foundation (Step 0)
**Implement:** Rewrite `_shared.py`.
**Validate:** Run one existing experiment (e.g. f03 on logd) to confirm new infrastructure works.
**Deliverable:** Working `_shared.py` with domain-aware metrics, all 11 models, structured logging.

### Phase 1: Rep Screening (f00)
**Implement:** f00 is written. Test it.
**Run:** f00 on logd (molecular), flab_thermo_tresanco (antibody), tcga_luad_vs_lusc (TCGA), buchwald_hartwig (reaction).
**Deliverable:** Decision on where rep filtering goes. If pre-search wins, implement `screen_reps` in `find_best_hybrid`.

### Phase 2: Core Search Components (Steps 1–4)
**Implement:** Rewrite f01, f03, f02, f04 (in that order — f01 first since f02 depends on its output).
**Run:** All 4 on the same task set as Phase 1. Add DTA (davis) when baselines land.
**Deliverable:** Per-task results for allocation comparison, budget/step sensitivity, pruning simulation, two-phase validation.

### Phase 3: New Experiments (Steps 5–8)
**Implement:** Write f05, f06, f07, f08.
**Run:** On same task set. f05 runs locally (fast). f06, f07, f08 on HTC.
**Deliverable:** Importance correlation matrix, augmentation comparison, internal evaluator comparison, iterative augmentation.

### Phase 4: Synthesis (Step 9)
**Implement:** Write f09.
**Run:** Locally (analysis script, consumes JSONs from Phases 1–3).
**Deliverable:** Cross-domain analysis, proposed new `SPEED_TIERS`, list of algorithm changes needed.

### Phase 5: Algorithm Refinement
**Implement:** Code changes to `hybrid.py` and `find_best_hybrid.py` based on Phase 4 findings.
Examples:
- New `SPEED_TIERS` dict.
- `screen_reps` parameter (if Phase 1 recommends it).
- Internal evaluator swap (if Phase 3 f07 recommends it).
- Default budget/step_size changes.
- Removal of dead allocation/importance methods.
- Any greedy algorithm logic changes.

### Phase 6: A/B Validation
**Implement:** Rewritten f04 + f09.
**Run:** On all tasks with **full rep sets** (not curated subsets). Include forward-compat
reps (MINT, SaProt, MolE, IgT5, Boltz-2). Compare old long tier vs new fast/medium/long.
**Acceptance criteria:**
- New `fast` ≥95% of old `long`'s score
- New `fast` ≤10% of old `long`'s wall time
- Rep screening regret <1% on all tasks

### Phase 7: Production
Run `find_best_hybrid` on all viable BOM tasks with validated tiers.

---

## Dependency Graph

```
Phase 0: _shared.py rewrite
    ↓
Phase 1: f00 rep screening  →  screen_reps decision
    ↓
Phase 2: f01 allocation  →  f03 budget/step  →  f02 pruning  →  f04 two-phase
    ↓                                                    (post-hoc on f01+f03 data)
Phase 3: f05 importance  +  f06 augmentation  +  f07 internal evaluator  +  f08 iterative augmentation
    ↓         (can run in parallel with Phase 2)
Phase 4: f09 cross-domain analysis  ←  consumes all outputs
    ↓
Phase 5: algorithm refinement (code changes)
    ↓
Phase 6: A/B validation (full rep sets)
    ↓
Phase 7: production
```

Phase 0 is a prerequisite for everything. Phase 1 should run immediately after.
Phases 2 and 3 can partially overlap. Phase 4 is synthesis. Phase 5 is code changes.
Phase 6 is the gate. Phase 7 is the payoff.

---

## Estimated Compute

| Phase | What | Estimate |
|-------|------|----------|
| 0 | _shared.py rewrite | 1 session |
| 1 | f00 × 4 tasks | 2–3 hours HTC |
| 2 | f01+f03+f02+f04 rewrites + runs × 4 tasks | 2–3 sessions + 6–10 hours HTC |
| 3 | f05+f06+f07 writes + runs × 4 tasks | 1–2 sessions + 2–4 hours HTC |
| 4 | f08 analysis | 1 session (local) |
| 5 | Code changes | 1 session |
| 6 | A/B validation (full rep sets) | 4–8 hours HTC |
| 7 | Production runs | Depends on BOM task count |
| **Total pre-production** | | **~5–8 sessions + ~15–25 hours HTC** |

---

## Task Selection

Run all experiments on this common task set (expand when baselines land):

| Domain | Task | Samples | Groups | Metric | Reps |
|--------|------|---------|--------|--------|------|
| Molecular | `logd` | 5,028 | scaffold | RMSE | `all` (36 reps, Phase 5 done) |
| Antibody | `flab_thermo_tresanco` | 672 | identity | Spearman | `all` (20–23 reps tested) |
| TCGA | `tcga_luad_vs_lusc` | 1,017 | TSS | AUROC | `all` (36/57 reps tested) |
| Reaction | `buchwald_hartwig` | 3,955 | additive | RMSE | `all` (8 reps) |
| DTA | `davis` | 30,000 | cold-target | CI | `full` (when baselines land) |

Use `all`/`full` preset (not curated subsets) — rep screening (f00) handles filtering.
