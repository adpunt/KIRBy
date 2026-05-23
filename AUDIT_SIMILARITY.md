# Similarity Experiments Audit — 2026-05-23

Scope: `tests/experiments/similarity/s01..s13` + `similarity_helpers.py`.
Grounded against `src/kirby/find_best_hybrid.py` (signature line 3756) and
`tests/experiments/_shared.py` (pool helpers, distance tier pool, splitters).

## Files audited

| File | Lines |
| --- | --- |
| s01_similarity_baseline.py | 159 |
| s02_hybrid_similarity.py | 270 |
| s03_feature_concentration.py | 187 |
| s04_activity_cliffs.py | 388 |
| s05_knn_prediction.py | 318 |
| s06_applicability_domain.py | 393 |
| s07_noise_bridge.py | 215 |
| s08_optimal_dimension.py | 398 |
| s09_cross_dataset_summary.py | 238 |
| s10_importance_downstream.py | 852 |
| s11_bayesian_optimization.py | 1248 |
| s12_cross_domain_cliffs.py | 296 |
| s13_clustering_retrieval.py | 309 |
| similarity_helpers.py | 785 |

## P0 (broken / silently wrong results)

- **s06_applicability_domain.py:104** — Dict comprehension `aug_tr = {k: v[train_idx] for k, v in augmentations.items()}` rebinds the loop variable `k` from the outer `for k in k_values:` (line 87). After this comprehension `k` becomes the last augmentation key (a string), so every subsequent `NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))` (lines 165, 171, 181, 188, 194, 200) either raises `TypeError` (str vs int) when `augmentations` is non-empty, or — if the only key happens to compare OK — silently uses the wrong k for the entire AD analysis. Even line 105's `aug_va` repeats the same comprehension and re-shadows. Fix: rename loop var to `aug_name` / `aug_v` (and the symmetric line 105). Same pattern needs an audit on the `for k in k_values:` block: `for k_kv` outer or rename comprehension keys (consistent with s05 line 133–135 which uses `k2`).

- **s10_importance_downstream.py:700** — `get_best_search_for_downstream(..., task_type=task_type, ...)` references `task_type` before it's bound; the local binding does not happen until line 723 (`task_type = data['task_type']`). The surrounding bare `except Exception as e:` (line 713) swallows the `NameError`, sets `top3 = None`, and `hybrid_blueprint` stays `None`. Result: every hybrid branch in Task 1 (kNN), Task 2 (GP), Task 3 (SVR), and Task 4 (q-RASAR) is silently skipped — the experiment runs only the per-rep baselines but reports as if it produced hybrid results. Fix: move `task_type = data['task_type']` (and `augmentations = data.get('augmentations')`) above the search call, or pass `data['task_type']` directly.

## P1 (methodology drift / incorrect-but-recoverable)

- **s12_cross_domain_cliffs.py:263** — `if top3:` is unconditional, but `top3` is only assigned inside the `try` block at line 169 *and only if* `search is not None`. If the try succeeds but `search is None` (no candidate in the pool), `top3` is never defined → `NameError` at line 263 inside the same `run()`, which aborts the experiment after CV ran. Fix: initialise `top3 = None` immediately before the try (mirror s10/s13).

- **s11_bayesian_optimization.py:549–580 + 1087–1119** — "hybrid_50d" / "hybrid_50d_refit" conditions are misleading: when `bp` is non-None from `get_best_search_for_downstream(... budget=None)`, `build_hybrid_features` and `_run_bo_with_rebuild` use `bgt = bp.budget`, *ignoring* the `--hybrid-budget` (default 50) argument. The condition name advertises a fixed 50-D hybrid but the actual dim is whatever the pool's best blueprint had (often other budgets). Fix: either (a) honor `hybrid_budget` and override `bp.budget` for these conditions, or (b) rename the conditions to reflect that the blueprint's native budget is used.

- **s11_bayesian_optimization.py:567 / 1113 / 1146** — `create_hybrid(...)` is called directly without `step_size` or `augmentation_strategy`, and without using `find_best_hybrid` caching. That's defensible for per-seed BO (the search is per-iteration, not amenable to disk cache), but the call drops the blueprint's `step_size` (silently defaulted to 5) and ignores augmentations entirely — i.e. `data['augmentations']` is never threaded into S11 at all, even when present. If S11 is run on datasets with augmentations, the hybrid BO condition is built with a different feature space than the search blueprint was scored against. Fix: pass `step_size=getattr(bp, 'step_size', 5)` and thread augmentations through `build_hybrid_features` / `_run_bo_with_rebuild`.

- **s07_noise_bridge.py:122–140** — The pool-helper call wraps everything in a bare `except Exception`. On failure (e.g. `RuntimeError: no pool candidate produced a result` from `get_fitted_pool_hybrid` line 2574), `X_hybrid = None`, the hybrid FP-rate block (lines 143–160) is silently skipped, and `results['summary']` (line 191) still reports per-rep baselines as if the experiment ran. There is no error or skip flag in the saved JSON. Fix: surface the failure as `results['hybrid'] = {'error': str(e)}` so downstream s09 aggregation knows to drop the dataset, and log a clearer message than `f"Hybrid search failed: {e}"`.

- **s01_similarity_baseline.py:98 and s08_optimal_dimension.py:127** — `compute_distance_matrix(X, feature_weights=importance[rep_name], standardize=True)` passes raw importance to the helper, and that helper does `X_proc * np.sqrt(feature_weights)` (noise_mitigation_methods.py:88) with no `np.maximum(., 0)` guard. `compute_feature_importance(method='lightgbm_gain')` yields non-negative gains today, but any future swap to a signed importance (permutation, SHAP signed) will silently inject NaNs into every downstream distance/SAC/violation value. The rest of similarity (s02 line 192, s04 line 222 / 288, s05 line 166, s06 line 125 / 156–158, s07 line 153) consistently uses `np.sqrt(np.maximum(.., 0))` — s01 and s08 are the outliers. Fix: clamp before passing, or call `np.maximum(., 0)` inline.

- **s09_cross_dataset_summary.py:131–145** — AD-corr aggregation `break`s after the first `k_key` in `distance_error_correlation`. If a dataset has multiple `k` values (as the s06 default supports), only the smallest-k correlation is recorded; the aggregate becomes a hidden mix of "k=5 for dataset A" and (after a config change) "k=10 for dataset B". Fix: prefer a canonical k (e.g. `corrs = exp6['distance_error_correlation'].get('k5')`) and record it explicitly.

- **s09_cross_dataset_summary.py:153–157** — Noise FP reduction divides by `best_rep_fp` from `summary.best_rep_fp_rate`; if best_rep_fp_rate is 0 (no false positives), the test skips silently. Fine for noise data but the summary then drops the dataset without indicating why. Fix: log skipped-dataset reasons in the aggregate.

- **s11_bayesian_optimization.py:855–860** — When `groups is None` or `n_unique_groups < 20`, S11 falls back to a *random* shuffle (`rng.permutation(n_samples)`) for both seed and pool sets. Per `feedback_no_random_splits.md`, the codebase forbids random splits. For BO this is a conceptual gray area (BO simulates library screening), but the user directive is explicit. Fix: either gate S11 to datasets with sufficient groups, or document this as an intentional BO-simulation deviation in the docstring (currently the docstring does not warn).

- **s11_bayesian_optimization.py:776–798** — Conditions are iterated even when their feature builders depend on resources that may be absent (`tanimoto_ecfp4` falls through to "max-dim rep" when ecfp4 is missing; `multi_obj_*` requires `data['smiles']`; `hybrid_gauche_gp` requires `gauche` install). Errors land in `_run_bo_loop` returning `success=False` for every iteration → `surrogate_failures` is incremented for the entire budget and a "best curve" is reported that is actually pure random acquisition. Fix: skip those conditions up front when their precondition fails, instead of reporting a degenerate curve.

## P2 (smells, future-proofing)

- **s06_applicability_domain.py:131 + 144–145, s11:1122–1124** — `StandardScaler` is fit on training-fold features for AD distance computation, then used to transform both train and val — that's correct. But `_concat = np.hstack([reps[k] for k in sorted(reps.keys())])` (s06 line 60) is computed on the *full* dataset before CV. The concat-baseline distance matrix in s06 effectively uses test-set features when fit-transformed inside the fold (line 162–164 re-scales per fold so this is OK in practice). No bug — flagged because the same pattern in s13:174 is also full-dataset and *not* re-scaled per fold for the clustering/retrieval analysis. s13 is full-dataset unsupervised anyway, so it's intentional, but adds tautological advantage for `hybrid` (acknowledged in s13:191–194 comment).

- **s08_optimal_dimension.py:109** — Top-3 reporting `for n, s in top3` shadows the outer scalar `n = len(labels)` (line 79). The shadowed `n` is only used at line 80 (already executed before the budget loop) and never referenced afterward inside the loop, so it's functionally harmless. Flag as smell — rename to `_n3` or `rep_n`.

- **s11_bayesian_optimization.py:1086–1199** — `_run_bo_with_rebuild` uses `last_optimize = -10` while `refit_interval=10` is hard-coded for the GP refit inside the rebuild path (line 1169), but the *outer* `run_bo_loop` uses `refit_interval=10` defaulted (line 992). The two refit logics are not aligned; if `refit_interval` is ever changed in `run_bo_loop` callsites, `_run_bo_with_rebuild` won't notice. Fix: pass `refit_interval` through.

- **s10_importance_downstream.py:99–100** — `is_self_query = X_query is X_train or (X_query.shape == X_train.shape and np.allclose(X_query, X_train))` will silently mis-classify when the val set happens to be identical-shape and bit-equal to the train fold (impossible with proper splits, but `np.allclose` is O(n·d) and runs every query call). Fix: rely on the `is` check only, or pass an explicit `self_query=True` flag.

- **s04_activity_cliffs.py:181–182, s12_cross_domain_cliffs.py:199–205** — `val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])` uses Python `in` against a `set` per element. Fine for small folds; expect O(n) cliff-membership checks per fold but the set lookup is amortized O(1). No bug. Flag only because s12 builds *lists* (line 199–200) then indexes `val_idx[val_cliff]` — works because val_idx is a numpy array and lists-of-ints are valid indexers.

- **similarity_helpers.py:441–449** — `antibody_sequence_cliff_sweep` per-threshold loop wraps the inner call in `try/except Exception`, recording `{'error': str(e)}` on failure. Silent on the console (no print of the exception class) — caller has to read the JSON to find out which threshold blew up. Lower priority; documentation gap.

- **similarity_helpers.py:629** — `'budgets': SCREENING_BUDGETS` is hard-coded; experiments may want a longer budget grid. Not buggy, but the default is opaque. Flag for documentation.

- **similarity_helpers.py:720–737** — `precompute_hybrid_searches` is only called for `tier='general'` and `tier='top'`. The distance tier searches (`tier='general_distance'`, `tier='top_distance'`) used by s04/s05/s06/s07/s10/s13 are *not* precomputed by `run_all`; every experiment that needs them triggers an on-demand `find_best_hybrid` via the disk-cache path. First-run latency on a fresh dataset will be high. Not a bug, but worth flagging if SLURM precompute scripts need updating.

- **s05_knn_prediction.py:116–117, s06_applicability_domain.py:84–85** — A new `make_cv_splitter` is constructed inside per-budget / per-k loops. The splitter is deterministic (random_state=seed), so the folds are identical, but recreating it is wasteful. Smell only.

- **cv_split (_shared.py:486)** — `mode` parameter is accepted but ignored (the branch is purely on `n_splits`). S04/S10/S12 pass `mode='intermediary'` or `mode='full'` derived from `n_folds`; harmless dead arg. Not similarity-specific; flag for cleanup.

- **s13_clustering_retrieval.py:174** — `X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])` rebinds `k` for the rest of the function (line 263, 268). Since `k_retrieval` is used for the retrieval kNN and `k` is only used as iteration var inside subsequent dict comprehensions (also rebinds), there's no functional bug. Rename to avoid confusion.

## Verified-clean

- **s01_similarity_baseline.py** — apart from the P1 sqrt-of-negative weight flag; passes augmentations correctly via `run_antibody_cliff_analysis(data, ...)`; permutation test seeded (`RandomState(42)`).
- **s02_hybrid_similarity.py** — pool helper used correctly, `augmentation_strategy` reaches `find_best_hybrid` via `get_fitted_pool_hybrid` → `create_hybrid`; uses `get_fitted_pool_hybrid` rather than hardcoded `(tier, aug_cond)`; classification path branches via `is_classification`.
- **s03_feature_concentration.py** — uses `get_fitted_pool_hybrid` correctly; MI / Gini math is straightforward; `np.random.RandomState(42)` seeded.
- **s04_activity_cliffs.py** — distance-pool helper, `is_clf` branching for cliff detection, weighted/unweighted both evaluated per-fold; classification metric routing via `get_domain_metric`. No P0/P1 issues beyond inherited splitter behavior.
- **s05_knn_prediction.py** — distance pool, per-budget blueprint lookup, splitter shared across conditions for fair comparison; `is_clf` branches for `KNeighborsClassifier`; weighted/unweighted both implemented.
- **s07_noise_bridge.py** — separate branch for antibody (edit distance) vs molecular (ECFP4 Tanimoto); standardize/weight order documented; FP-rate computation correct. (Failure-swallowing in P1 list.)
- **s08_optimal_dimension.py** — Pareto frontier math handles `lower_is_better` correctly; per-model best-budget search uses pool helper per budget; classification path supplied via `is_clf`.
- **s09_cross_dataset_summary.py** — `_metric_lower_is_better` inferred from per-rep metric strings; Wilcoxon test gated to ≥3 datasets; per-dataset summary tolerates missing exp files.
- **s12_cross_domain_cliffs.py** — domain-agnostic Euclidean cliffs; auto-threshold fallback; per-model evaluation grid; pool helper used (apart from P1 NameError edge case).
- **s13_clustering_retrieval.py** — explicit acknowledgement of supervised-hybrid tautology in docstring; PCA dim capped at `len(labels)-1`; uses pool helper with `distance=True`; classification branch uses class purity.
- **similarity_helpers.py** — DATASET_REPS sourced from `complexity_helpers` (single source of truth); `precompute_hybrid_searches` called with explicit `augmentation_strategy` per the cache-collision fix; `pruning_mode` plumbed; NaN guards on rep features (line 660–665).

## Cross-cutting observations (not bugs)

- All similarity experiments now use the pool helper (`get_fitted_pool_hybrid` / `get_best_search_for_downstream`) rather than hardcoded `(tier, aug_cond)` picks — consistent with the May 2026 pool-refactor (`commit 4afd225`).
- `augmentation_strategy` is *not* defaulted to `'uncapped'` anywhere; it is always either set explicitly (in `precompute_hybrid_searches` / `get_precomputed_search` to match `aug_condition`) or picked from `meta['aug_condition']` after pool selection (in `get_fitted_pool_hybrid` line 2580). Cache collision risk per `project_cache_collision_fix.md` is not present in similarity.
- No references to the removed `best_overall` evaluator anywhere in `tests/experiments/similarity/`.
- `internal_evaluators` defaults flow from `INTERNAL_EVALUATORS` / `DISTANCE_EVALUATORS` constants. The distance pool sets `DISTANCE_EVALUATORS = ['knn', 'ridge', 'rf', 'mlp', 'lightgbm', 'svm']` (_shared.py:261) — note this includes `'ridge'`, which `feedback_no_ridge.md` forbids as a "sensitivity check". Ridge here is an internal evaluator inside the hybrid search (not a downstream model), but worth flagging because the user has been explicit about Ridge being a special case.
- No Mutation-reps `load_data` two-pass concern: similarity uses `complexity_helpers.DATASET_REPS` directly and `load_data` is called inside `run_all`. SKEMPI/mutation datasets are not in the SINGLE_DATASET_EXPERIMENTS rotation.
- No direct `create_hybrid` use that bypasses the cache *for the main feature-selection path* — the only `create_hybrid` calls are in s11 BO (per-seed, intentional) and inside `get_fitted_pool_hybrid` (one-shot full-data build using the cached blueprint), which is the intended architecture per `project_experiment_hybrid_architecture.md`.
