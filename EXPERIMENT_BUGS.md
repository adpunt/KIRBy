# Experiment Bug Tracker

**Created:** 2026-03-26
**Context:** Deep audit of complexity/similarity/noise experiments.

---

## FIXED (this session)

### Tier 1
- [x] `_shared.py load_data()` — added two-pass MUTATION_REPS support for antibody datasets
- [x] `complexity_helpers.py` DATASET_REPS — added `mutation_features` back to all DMS datasets (abagym_*, ddg_combined)
- [x] N07 recovery % — verified NOT a bug (formula handles both metric directions correctly)

---

## Tier 2 — Wrong metrics on classification datasets — FIXED (2026-03-26)

- [x] **C02** — PCA baselines now use `get_domain_metric(data)` instead of `r2_score`
- [x] **C08** — uses `get_domain_metric(data)`, skips entirely on classification (conformal is regression-only)
- [x] **C09** — 9d bias-variance passes `metric_fn` to `evaluate_cv_with_train()`. 9b kept as r2_score (PCA reconstruction diagnostic, not task metric)
- [x] **C10** — uses `get_domain_metric(data)` instead of `r2_score`
- [x] **C11** — uses `get_domain_metric(data)` instead of `r2_score`
- [x] **N01** — uses `get_model_factories(task_type=...)` and `metric_fn` from `get_domain_metric()`. Removed hardcoded r2_score/RMSE from cliff-stratified results.
- [x] **N07** — uses `get_model_factories(task_type=...)` instead of `RandomForestRegressor`
- [x] Stale `r2_score` imports removed from C16, C17, C18, C19, N07

### Similarity
- [ ] **S01-S09** — hardcode `r2_score` or similarity-specific metrics. Lower priority since similarity experiments are by nature distance-based, but S04/S05/S06 compute R² on predictions which is wrong for classification.

---

## Tier 3 — Augmentation reps loaded but not used — FIXED (2026-03-26)

### Complexity
- [x] **C11** — passes augmentations to `create_hybrid()` with capped strategy; applies `apply_augmentation_selection` to val
- [x] **C16** — `_evaluate_cell()` now accepts and passes augmentations (capped); val augmentation applied
- [x] **C17** — hybrid strategy passes augmentations (capped) to `create_hybrid()`
- [x] **C18** — natural hybrid allocation passes augmentations (capped) to `create_hybrid()`
- [x] **C19** — hybrid input space passes augmentations (capped) to `create_hybrid()`
- [x] **cached_create_hybrid** in `complexity_helpers.py` — now accepts `augmentations` and `augmentation_strategy` kwargs; cache key includes augmentation data

### Similarity
- [x] **similarity_helpers.py** shared search — passes `augmentations=data.get('augmentations')` to `get_hybrid_features_and_weights()`
- [x] **S02-S07** — receive augmented search via shared_search
- [x] **S08, S10, S12** — standalone searches now pass augmentations
- S11, S13 use `create_hybrid` directly (augmentations not applicable at that level)

### Noise
- [x] **N01-N08** — all `create_hybrid()` calls now pass `augmentations=data.get('augmentations'), augmentation_strategy='uncapped'`
- [x] N01, N03 sub-functions updated to accept and forward augmentation params

---

## Blocked on F12/F13 — Hardcoded `allocation_method='greedy'`

F01 showed `greedy` is the WORST allocation method. These experiments all hardcode it. Once F12/F13 finalize the allocation winner, update all of them.

### Complexity (direct `create_hybrid` calls)
- C08, C09 (9b/9c), C10, C11, C16, C17, C18, C19

### Similarity
- S11, S13

### Noise
- N01, N02, N03, N04, N05, N06, N07, N08

**C20, C21** correctly use `performance_weighted`. Experiments that go through `find_best_hybrid()` (C05-C07, C13, C15) use library defaults which are already correct.

---

## Hybrid usage audit — what C01-C07 actually test

### C01 (rep characterization)
- **No hybrid.** Standalone profiling of individual reps (intrinsic dim, MI, effective rank).
- No augmentation, no find_best_hybrid.

### C02 (complexity grid)
- **Uses `find_best_hybrid()`** — library defaults for allocation/importance.
- Tests: best_hybrid vs best_single_rep vs PCA_baselines across budgets.
- Augmentation: ✅ passed to `find_best_hybrid()` when available.
- Conditions tested: multiple budgets, multiple models.
- **Missing:** no frozen-vs-FT condition, no capped-vs-uncapped augmentation comparison (only runs default strategy).

### C04 (importance as metric)
- **No hybrid.** Tests whether feature importance scores predict downstream utility.
- Standalone analysis.

### C05 (learning curves)
- **Uses `find_best_hybrid()`** — library defaults.
- Tests: subsampled training sizes → hybrid performance curve.
- Augmentation: ✅ passed through.
- **Issue:** Fixed budget=100 (TODO in code).

### C06 (within-family gradient)
- **Uses `find_best_hybrid()`** per model family variant.
- Tests: how model capacity interacts with hybrid dimensionality.
- Augmentation: ✅ passed through.

### C07 (data complexity partitions)
- **Uses `find_best_hybrid()`** on dataset partitions.
- Tests: does hybrid help more on "hard" vs "easy" subsets?
- Augmentation: ✅ passed through.

### What C02 currently tests:
- ✅ Multiple models × multiple budgets × allocations × importance methods (full grid)
- ✅ No-augmentation vs uncapped augmentation (two search passes)
- ✅ PCA baselines at each budget for comparison
- ✅ Best single rep vs hybrid delta

### Design gaps — FIXED (2026-03-26)
1. [x] **Capped vs uncapped augmentation** — C02 now runs 4 conditions: no-aug, uncapped, capped, top-tier-only
2. [x] **Frozen vs fine-tuned** — C02 has TODO + wiring stub for a fifth search pass (`search_ft`). Commented out until FT .npy files land.
3. [x] **Budget sweep** — C05 now passes `config['budgets']` ([25,50,100,200,500]) to `find_best_hybrid`, capped to subsample size. C02 budgets extended to include 1000, 1500, 2000, total_dim.
4. **Fast vs full search** — blocked on F12/F13 SPEED_TIERS finalization. C02 currently uses explicit params rather than speed tiers.
5. [x] **Top-reps-only vs all-reps** — C02 now runs `top_tier` from `DATASET_REPS` as a separate search condition.

---

## Other issues (low priority)

- **N04, N06** return raw dicts without `make_structured_output()` — breaks output contract
- **S04-S06, S13** swallow CV fold exceptions without logging
- **C13** may have stale `_LEGACY_ANTIBODY` rep names (needs verification)
- **C09** swallows exceptions in information/effective rank computation; produces NaN curves silently
