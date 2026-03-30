# Experiment Hybrid Architecture Roadmap

**Created:** 2026-03-30
**Status:** Phase 1-7 DONE (code in place, not yet tested on server)

---

## Problem

Every experiment currently calls `create_hybrid` directly with hardcoded params. They skip `find_best_hybrid` entirely — no search, no optimization, no caching. This means experiments don't showcase the `find_best_hybrid` methodology, which is core to the paper.

## Target Architecture

### Step 1: Precompute step in each helper's `run_all()`

Before experiments run, call `find_best_hybrid` for each (budget, aug_condition) combination any experiment in that category will need. All calls use `cache=True` — first run computes, subsequent runs hit disk cache.

```python
# In complexity_helpers.run_all(), before experiment loop:
from kirby.find_best_hybrid import find_best_hybrid

# Budget is an INPUT — one call per budget value
for budget in config['budgets']:
    # Uncapped augmentation
    find_best_hybrid(reps, labels, groups, budgets=[budget],
                     augmentations=augs, augmentation_strategy='uncapped',
                     speed=config['search_speed'], cache=True, verbose=False)
    # No augmentation
    find_best_hybrid(reps, labels, groups, budgets=[budget],
                     augmentations=None,
                     speed=config['search_speed'], cache=True, verbose=False)
    # Capped augmentation (for experiments that need it)
    find_best_hybrid(reps, labels, groups, budgets=[budget],
                     augmentations=augs, augmentation_strategy='capped',
                     speed=config['search_speed'], cache=True, verbose=False)
```

Each `find_best_hybrid` call searches over (allocations × importance_methods × internal_evaluators) at that budget. Results cached to `~/.cache/kirby/hybrid_search/`.

### Step 2: Experiments consume cached search results

Instead of calling `create_hybrid` directly, experiments call `find_best_hybrid` with their specific (budget, aug_condition). Since the precompute step already ran these exact calls, they hit cache instantly.

```python
# BEFORE (wrong):
X_train, fi = create_hybrid(train_reps, y_train, allocation_method='entropy_1.0',
                             importance_method='lightgbm_gain', budget=100)

# AFTER (correct):
search = find_best_hybrid(reps, labels, groups, budgets=[100],
                          augmentations=augs, augmentation_strategy='uncapped',
                          speed='fast', cache=True, return_fitted=True, verbose=False)
# Use search.fitted_result for the best hybrid at budget=100
# Or use search.best_by_model for per-model best
# Or use search.all_results for the full grid at this budget
```

### Step 3: Per-fold reconstruction from search results

For experiments that need per-fold train/val splits (most of them), use the winning Blueprint from the search to reconstruct per fold:

```python
bp = search.best_overall['blueprint']
for train_idx, val_idx in fold_splits:
    train_reps = {k: v[train_idx] for k, v in reps.items()}
    val_reps = {k: v[val_idx] for k, v in reps.items()}
    X_train, fi = create_hybrid(train_reps, labels[train_idx],
                                allocation_method=bp.allocation_method,
                                importance_method=bp.importance_method,
                                internal_evaluator=bp.internal_evaluator,
                                budget=bp.budget)
    X_val = apply_feature_selection(val_reps, fi)
```

This create_hybrid call uses the WINNING config from find_best_hybrid, not hardcoded params. And create_hybrid_cached (fold-aware disk cache) deduplicates identical fold calls across experiments.

---

## Per-Experiment Audit

### Experiments that DON'T use hybrids (no changes needed)

| Experiment | What it does |
|---|---|
| C01 | Rep characterization (standalone rep analysis) |
| C04 | Importance as metric (importance scores only) |
| S01 | Similarity baseline (pairwise rep distances) |
| S07 | Noise bridge (label noise vs similarity) |
| S09 | Cross-dataset summary (meta-analysis) |

### Experiments that already call find_best_hybrid (check they use cache)

| Experiment | Current call | Change needed |
|---|---|---|
| C02 complexity grid | find_best_hybrid with budget sweep | Ensure cache=True, use precomputed |
| C03 cross-dataset | find_best_hybrid per dataset | Ensure cache=True |
| C05 learning curves | find_best_hybrid at sub-budgets | Ensure cache=True |
| C06 within-family | find_best_hybrid with config budgets | Ensure cache=True |
| C07 data complexity | find_best_hybrid | Ensure cache=True |
| C08 calibration | find_best_hybrid + create_hybrid | Convert create_hybrid to use blueprint |
| C09 formal analysis | find_best_hybrid + create_hybrid | Convert create_hybrid to use blueprint |
| C13 forward compat | find_best_hybrid + create_hybrid | Convert create_hybrid to use blueprint |
| C14 cross-domain | find_best_hybrid | Ensure cache=True |
| All 6 OPIG | find_best_hybrid | Ensure cache=True |

### Experiments that call create_hybrid directly (MUST convert)

#### Uses "the best" hybrid — convert to find_best_hybrid best_overall

| Experiment | Current params | Aug condition | Budgets |
|---|---|---|---|
| C11 ensemble vs hybrid | entropy_1.0 / lightgbm_gain | not passed | budget sweep |
| C17 info-theoretic | entropy_1.0 / lightgbm_gain | uncapped | budget sweep [25,50,100,200] |
| C18 diversity vs depth | entropy_1.0 / lightgbm_gain | uncapped | variable |
| C19 importance-weighted | entropy_1.0 / lightgbm_gain | uncapped | variable |
| N01 activity cliff | from blueprint | uncapped | from blueprint |
| N02 sample weighting | entropy_1.0 / lightgbm_gain | uncapped | 100 |
| N03 SAS landscape | entropy_1.0 / lightgbm_gain | uncapped | 50 |
| N04 GP-LOO | entropy_1.0 / lightgbm_gain | uncapped | from config |
| N05 noise injection | entropy_1.0 / lightgbm_gain | uncapped | from kwargs |
| N06 BindingDB | entropy_1.0 / lightgbm_gain | not passed | variable |
| N07 remediation | entropy_1.0 / lightgbm_gain | not passed | budget sweep |
| N08 estimation | entropy_1.0 / lightgbm_gain | uncapped | from config |
| N09 LNL survey | entropy_1.0 / lightgbm_gain | not passed | variable |
| N10 hybrid neighborhood | entropy_1.0 / lightgbm_gain | uncapped | from config |
| N11 est×remediation | entropy_1.0 / lightgbm_gain | uncapped | from config |
| N12 importance distance | entropy_1.0 / lightgbm_gain | uncapped | 100 |
| N13 noise distributions | entropy_1.0 / lightgbm_gain | uncapped | 100 |
| S13 clustering | entropy_1.0 / lightgbm_gain | not passed | variable |

#### Uses specific budget/dim — convert to find_best_hybrid with specific budget

| Experiment | Current params | Aug condition | Special needs |
|---|---|---|---|
| C10 noise robustness | entropy_1.0 / treeshap | capped | Budget sweep + noise-injected labels (sigma>0 can't use precomputed) |
| C15 ML theory depth | entropy_1.0 / lightgbm_gain | not passed | Multiple budget sweeps, complexity analysis |
| C16 feature selection | entropy_1.0 / lightgbm_gain | capped | Budget sweep as regularization |
| S08 optimal dimension | from blueprint | uncapped | Budget sweep, alloc sweep |
| S11 Bayesian opt | entropy_1.0 / lightgbm_gain | not passed | Iterative acquisition (can't precompute — training set changes) |

#### Uses custom rep subsets — likely can't use standard precompute

| Experiment | Why custom |
|---|---|
| C20 aug value decomposition | Per-pair rep subsets, budget=base_dim |
| C21 single-rep aug | Single rep + augmentation, budget=base_dim |

#### Similarity shared search (already close to correct)

| Experiment | Current pattern | Change needed |
|---|---|---|
| S02 hybrid similarity | Produces shared_search via get_hybrid_features_and_weights | Convert to find_best_hybrid with cache |
| S03 feature concentration | Uses shared_search | Use precomputed search result |
| S04 activity cliffs | Uses shared_search + per-fold create_hybrid | Per-fold: use blueprint from search |
| S05 kNN prediction | Uses shared_search + per-fold create_hybrid | Per-fold: use blueprint from search |
| S06 applicability domain | Uses shared_search + per-fold create_hybrid | Per-fold: use blueprint from search |
| S10 importance downstream | Per-fold create_hybrid from blueprint | Use blueprint from search |
| S12 cross-domain cliffs | Per-fold create_hybrid from blueprint | Use blueprint from search |

#### Federated (5 experiments)

| Experiment | Current pattern | Change needed |
|---|---|---|
| fed01 noise heterogeneity | create_hybrid direct | Convert to find_best_hybrid |
| fed02 rep diversity silos | create_hybrid direct | Convert to find_best_hybrid |
| fed03 synthetic private | create_hybrid direct | Convert to find_best_hybrid |
| fed04 cluster-aware | create_hybrid direct | Convert to find_best_hybrid |
| fed05 antibody federated | create_hybrid direct | Convert to find_best_hybrid |

---

## Precompute Grid

Per dataset, the precompute step calls find_best_hybrid for:

- **Aug conditions:** None, uncapped, capped (3)
- **Budgets:** from config (screening: 8 values, full: 11 values)
- **Speed:** fast or full (determines alloc/imp/evaluator grid)

Total precompute calls per dataset:
- Screening: 3 aug × 8 budgets = 24 find_best_hybrid calls (each cached)
- Full: 3 aug × 11 budgets = 33 find_best_hybrid calls (each cached)

Each find_best_hybrid call at fast tier searches:
- 5 effective alloc×imp combos × 3 evaluators × 3 folds = 45 configs

So screening precompute = 24 × 45 = 1,080 hybrid evaluations per dataset. With cache, this only happens once.

---

## Implementation Order

### Phase 1: Infrastructure ✓ DONE
1. ✓ Fixed `_hybrid_cache_key` to include augmentations + augmentation_strategy
2. ✓ Added `precompute_hybrid_searches()` + `get_precomputed_search()` to `_shared.py`
3. ✓ Added `get_fitted_hybrid()` convenience wrapper to `_shared.py`
4. ✓ Wired precompute into all 4 helper `run_all()` functions (complexity, similarity, noise, federated)

### Phase 2: Convert "best hybrid" experiments ✓ DONE
5. ✓ N01-N13: All 13 noise experiments converted (Pattern A/B/C as appropriate)
6. ✓ C11, C17-C19: All 4 complexity experiments converted
7. ✓ S13: Similarity experiment converted
   - Also fixed augmentation slicing bug in N02, N07, N09-N13 (was passing unsliced augs to per-fold create_hybrid)

### Phase 3: Convert budget-sweep experiments ✓ DONE
8. ✓ C10: Blueprint + create_hybrid (noisy labels prevent build_fold_hybrid; treeshap→search winner)
9. ✓ C15: Mixed — get_fitted_hybrid (15a/b full-dataset) + build_fold_hybrid (15d per-fold)
10. ✓ C16: build_fold_hybrid (identical to C11 pattern, capped aug)
11. ✓ S08: build_fold_hybrid for per-fold (was already close, used blueprint)
12. ✓ S11: Blueprint + create_hybrid (iterative BO, training set grows — can't precompute)

### Phase 4: Convert shared search pattern ✓ DONE
10. ✓ Removed shared_search from similarity_helpers.py
11. ✓ S02-S07: get_fitted_hybrid + flatten_weights (self-contained, cached)
12. ✓ S10, S12: get_precomputed_search (only need blueprint)
13. ✓ Made _flatten_weights public (→ flatten_weights)

### Phase 5: Convert find_best_hybrid users (ensure cache=True) ✓ DONE
11. ✓ Audit C02, C03, C05, C06, C07 — all already correct (cache=True, augmentations)
12. ✓ Audit all 6 OPIG experiments — all already correct (cache=True)
13. ✓ C14: Added missing cache=True to find_best_hybrid call
14. ✓ C08: Added internal_evaluator + augmentations to create_hybrid, extracted from blueprint
15. ✓ C09: Added internal_evaluator + augmentations to 2 create_hybrid calls (9b, 9c) + evaluate_cv_with_train caller (9d)
16. ✓ C13: Added augmentations to find_best_hybrid, converted create_hybrid to build_fold_hybrid
17. ✓ _shared.py evaluate_cv_with_train: Added internal_evaluator, augmentations, augmentation_strategy params

### Phase 6: Federated ✓ DONE
13. ✓ FED01: get_precomputed_search at start, 2 create_hybrid calls converted to use blueprint params
14. ✓ FED02: get_precomputed_search at start, 2 create_hybrid calls converted
15. ✓ FED03: 3 parts — Parts A/B use get_precomputed_search, Part C uses per-silo find_best_hybrid(cache=True). 5 create_hybrid calls converted.
16. ✓ FED04: get_precomputed_search at start, _compute_cluster_importance + global baseline converted. Smooth factor sweep also updated.
17. ✓ FED05: Per-silo find_best_hybrid(cache=True), _silo_baseline + 3 condition create_hybrid calls converted. All 14 create_hybrid calls across 5 files verified.

### Phase 7: Special cases ✓ DONE
14. ✓ S11 (Bayesian opt) — already converted in Phase 3 (blueprint + create_hybrid, iterative BO)
15. ✓ C20 deleted — merged into C21. Raw concat condition removed.
16. ✓ C21 rewritten: find_best_hybrid per base rep (budget=base_dim, augmentation_strategy='uncapped', cache=True). Search discovers importance_method × internal_evaluator. Per-fold construction via build_fold_hybrid.

---

## Verification

For each converted experiment:
1. Run on herg with fast tier
2. Verify it produces results (no crashes)
3. Verify the hybrid config in results matches find_best_hybrid's best (not hardcoded entropy_1.0)
4. Verify cache files exist in `~/.cache/kirby/hybrid_search/`
5. Run again — verify it hits cache (faster second run)
