# Framework Experiment Roadmap

**Goal:** Empirically test every component of the hybrid search machinery across domains,
identify what's useless, refine the speed tiers and algorithms, and produce validated
final search settings.

**Status:** Planning (2026-03-15)

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

| Script | What it tests | Status |
|--------|-------------|--------|
| **f00** `f00_rep_screening.py` | Where should rep filtering go? Pre-search gate vs implicit vs relative threshold vs augmentation pathway | **Written** (needs testing) |
| **f01** `f01_allocation_strategies.py` | 21 allocation strategies × budgets × models | **Done** |
| **f02** `f02_adaptive_pruning.py` | Post-hoc pruning simulation at thresholds 0.3–0.7 | **Done** |
| **f03** `f03_greedy_step_size.py` | step_size × patience × budget × model for greedy only | **Done** |
| **f04** `f04_two_phase_search.py` | Two-phase (coarse + refinement) vs full grid | **Done** |
| **f05** `f05_importance_correlation.py` | Are the 8 importance methods redundant? Rank correlation + wall time | **Written** |
| **f06** `f06_augmentation_test.py` | greedy_ablation vs all vs none. augmentation_budget sweep. | **Written** |
| **f07** `f07_internal_evaluator.py` | RF vs XGBoost vs Ridge as the internal evaluator inside greedy loops | **Written** |
| **f08** `f08_iterative_augmentation.py` | Iterative augmentation strategy testing | **Written** |
| **f09** `f09_cross_domain_analysis.py` | Consumes outputs from f00–f08 across all tasks. Identifies cross-domain patterns. | **Not written** |

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
