# Federated Experiments Audit — 2026-05-23

Read-only audit of `/Users/apunt/repos/KIRBy/tests/experiments/federated/`. No files in that directory were modified.

## Files audited
| File | Lines |
|---|---|
| `__init__.py` | 0 |
| `caco2.py` | 41 |
| `fed01_noise_heterogeneity.py` | 693 |
| `fed02_rep_diversity_silos.py` | 481 |
| `fed03_synthetic_private.py` | 1107 |
| `fed04_cluster_aware_importance.py` | 391 |
| `fed05_antibody_federated.py` | 810 |
| `federated_helpers.py` | 1929 |
| `herg.py` | 41 |
| `lhasa_herg_loader.py` | 207 |
| `logd.py` | 41 |

Cross-references read: `FEDERATED_EXPERIMENT_ROADMAP.md`, `src/kirby/find_best_hybrid.py:3756`, `src/kirby/hybrid.py:3188`, `tests/experiments/_shared.py:486` (cv_split), `:2083` (precompute_hybrid_searches), `:2387` (get_best_search_for_downstream).

## P0 (broken / silently wrong results)

**P0-1 — `fluid_within_kirby_across` calls `create_hybrid` without importing it.** `tests/experiments/federated/federated_helpers.py:1436` references `create_hybrid` but the local imports at `:1378-1381` only bring in `KMeans, PCA, StandardScaler, apply_feature_selection`. There is also no module-level `create_hybrid` import. Every call raises `NameError` and is silently captured by `fed02:391` → fed02 Condition C6 (the roadmap's "strongest publication angle") has never produced a real number. Fix: add `from kirby.hybrid import create_hybrid` to the local imports.

**P0-2 — augmentations dropped from every `create_hybrid` / `federated_feature_selection` call.** All five experiments compute `augs = data.get('augmentations')` and pass them ONLY to `get_best_search_for_downstream` (blueprint discovery). Downstream feature-construction calls never receive `augmentations=augs`. Sites: `fed01:237-241, 320-325, 422-427, 609-612`; `fed02:291-296, 333-338, 367-371`; `fed03:412-416, 635-639, 684-688, 879-883, 949-953`; `fed04:75-81, 204-209`; `fed05:106-110, 302-306, 481-485, 587-591`. Federated hybrids therefore operate on a strictly smaller feature space than the rest of KIRBy, and the blueprint chosen for the `augs`-present search is applied to a `augs`-absent build. Fix: thread `augmentations=augs` (and `augmentation_strategy=aug_cond`) into `create_hybrid` and add the same parameters to `federated_feature_selection` / `fluid_within_kirby_across`.

**P0-3 — Ridge used as silo internal evaluator (user-forbidden).** `federated_helpers.py:746,785-786` (`federated_feature_selection`) uses `RidgeClassifierCV`/`RidgeCV` to compute the silo score that drives cross-silo budget allocation. Same at `federated_helpers.py:1418,1448` (`fluid_within_kirby_across`) and `fed03:675,693` (Part B C3). Per `feedback_no_ridge.md`, Ridge is explicitly forbidden as a sensitivity check. Every "noisier silo gets less budget" interpretability claim in Fed01 Phase 4 is a Ridge artefact. Fix: replace with the `internal_evaluator` already passed in (matched-to-downstream).

**P0-4 — Single 80/20 split everywhere; ISSUE S1 not actually closed.** Roadmap lines 502-505 claim a fix via `cv_split(groups, 1, ...)`. But `_shared.py:492-499` returns ONE `GroupShuffleSplit` fold when `n_splits=1`. Per `feedback_no_random_splits.md`: "Single 80/20 insufficient — need 3-5 fold CV." Sites: `fed01:151`; `fed02:149-150`; `fed03:233-234, 580-581, 867, 942-944, 993-994`; `fed04:167-168`; `fed05:_split_silo` at `:86-90` (used everywhere). Every reported metric is one-fold, no CI. Fix: use `n_splits >= 3` and average; update `_split_silo`.

**P0-5 — `federated_feature_selection` inner validation split is not group-aware.** `federated_helpers.py:782-789` uses `ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+s)` on the silo's data. Same at `:1446` (`fluid_within_kirby_across`) and `fed03:691-694`. Outer split is group-aware, inner silo score is random → scaffold neighbours can sit on both sides of the silo's internal val split, inflating `silo_scores` and skewing budget allocation. Fix: use `GroupShuffleSplit` with within-silo groups.

**P0-6 — `is_classification = len(np.unique(labels)) <= 10` heuristic.** `federated_helpers.py:749` infers task type from label cardinality. Several federated regression datasets (`flab_thermo_tresanco` Tm scores, small DMS subsamples) can fall under 10 unique values → misrouted to `RidgeClassifierCV`. Fix: require explicit `task_type=` arg propagated from caller.

**P0-7 — `compute_clf_metrics` silently nans on multiclass.** `federated_helpers.py:1004-1008` calls `roc_auc_score(y_true, y_prob)` with a 1D positive-class probability; multiclass raises `ValueError` which is silently caught and returns `nan`. fed02 forwards `predict_proba[:, 1]` for any classification target (`fed02:184, 304, 346`). Only binary tasks are correct today; multiclass would silently lose AUROC. Fix: branch on `n_classes` and use `roc_auc_score(..., multi_class='ovr', average='macro')` with full proba matrix.

## P1 (methodology drift / incorrect-but-recoverable)

**P1-1 — Stale server path `/data/stat-cadd/`.** `federated_helpers.py:49` and `lhasa_herg_loader.py:20` still point at `stat-cadd`; per memory the server moved to `stat-ecr` in 2026-05. Both lookups will quietly fall back to the local path unless env vars are set on the server.

**P1-2 — `builtins.display` permanently overwritten.** `federated_helpers.py:44-67`. On non-IPython environments `_original_display is None`, so the no-op lambda is left bound to `builtins.display` after import. Fix: `delattr(builtins, 'display')` in the `is None` branch.

**P1-3 — `fed03` Part C (`_run_antibody_federated`) hardcodes regression.** `fed03:887, 956, 1010-1012` always use `RandomForestRegressor` + `compute_reg_metrics`; no check on `sd['task_type']`. Any classification silo (e.g. some `abagym_HER2` loader paths) gets wrong model and metric.

**P1-4 — `fed05` always uses `RandomForestRegressor` regardless of task_type.** `fed05:113, 240, 488, 552, 594, 637`. `_silo_baseline` at `:106-118` too. Same risk as P1-3.

**P1-5 — `fed03` Part B C3 `silo_indices_all` is dead code.** `fed03:668-670` builds `silo_indices_all` but never uses it; the loop calls `_ch` per-silo independently and hstacks. Misleading: implies federation when none is happening.

**P1-6 — fed01 C0 "centralized upper bound" trained on noisy labels.** `fed01:237-247`. The label set is `train_labels_noisy`, same as the federated conditions; it is NOT a clean-label upper bound. Phase 4's noise-detection narrative compares against this baseline assuming clean labels. Fix: rename, or add a `c0b_centralized_kirby_clean` variant.

**P1-7 — `class_weight='balanced'` on FLuID teachers vs FLuID's RUS.** `fed01:226-228, 593-595`. The audit log calls this a documented deviation. For noise-heterogeneity specifically, balanced reweighting interacts asymmetrically with skewed-by-noise silos and biases the vote. Worth re-checking.

**P1-8 — Cluster reconstruction may diverge from FLuID's smoothed views.** `fed01:200-207, 570-578` rebuilds per-cluster DataFrames via `train_df[train_df['CLUSTER'] == s]`. With `smooth_factor > 0`, FLuID's `cluster_data_space` reassigns rows AFTER computing the returned views — the rebuilt views may not equal the originals returned by `cluster_data`. Fix: pass the returned `cluster_data` views directly.

**P1-9 — `federate_labels_and_train_student` hardcodes `random_state=0` for the student.** `federated_helpers.py:698-701` ignores the caller's `seed`. Mixed seeding inside one function.

**P1-10 — `predict_fluid_quality_ensemble` uses `KFold(shuffle=False)`.** `federated_helpers.py:544, 1322`. On a silo whose rows are sorted (FLuID's K-Means + smooth ordering), `shuffle=False` puts contiguous chemical neighbours in the same fold → MCC underestimates generalization → quality weights distorted.

**P1-11 — `fed05._split_silo` always seeds 42; no replicate sweep.** `fed05:86-90`. Combined with P0-4, single-fold single-seed. Roadmap row 10 (line 616) tracks `replicate_count=3` as TODO; still open.

**P1-12 — `fed01:417-431` claims "Reuse C5's hybrid" but actually rebuilds.** The code path only uses C5's success flag; it calls `create_hybrid` fresh on cleaned data. Cosmetic — obscures intent.

**P1-13 — `fed05:670-673` fragile weight-prediction zip.** `w_arr` and `preds` rely on insertion-order dict iteration. Works today; any future parallel teacher prediction silently misaligns. Fix: capture `silo_order = list(preds)` once.

**P1-14 — Bare `except Exception` swallows everything in every inner condition.** All fed01–fed05 conditions wrap in `try: ... except Exception as e: results[..] = {'error': str(e)}`. Hides P0-1 NameError, ImportError on FLuID paths, shape mismatches. Recommend `traceback.format_exc()` or narrowing exception types.

## P2 (smells, future-proofing)

**P2-1 — `RandomForest(n_estimators=150, max_depth=15, class_weight='balanced')` hardcoded everywhere.** Documented in ISSUE S2 (FLuID parity), but `DOWNSTREAM_MODELS` imported at `federated_helpers.py:36` is then unused — federated experiments never honour the 7-model downstream catalogue.

**P2-2 — `K=5` clusters hardcoded; no sweep.** `fed01:49, fed04:39`. Roadmap (line 484) wants `K=[3,5,7,10]`.

**P2-3 — `precompute_hybrid_searches` runs in `run_all` but most fed conditions bypass the cache.** `federated_helpers.py:1855-1894` precomputes top + general searches; downstream code calls `create_hybrid` directly per silo, only the blueprint lookup actually consumes the precompute. Most expensive step in `run_all` for marginal value.

**P2-4 — `internal_evaluator='match'` passed to `create_hybrid`.** `fed01:241, fed02:296, fed04:80`, etc. `create_hybrid`'s default is `'svm'` (`hybrid.py:3208`); `'match'` is `find_best_hybrid`'s semantic ("match downstream model"). Inside `create_hybrid` it likely flows through with no special handling — verify it doesn't silently fall through to a default evaluator. Fix: map `'match'` → `'rf'` before calling `create_hybrid`.

**P2-5 — `compute_clf_metrics` lacks accuracy and `f1_weighted` (the registered metric for several antibody clf datasets).** `federated_helpers.py:994-1009`. Only MCC/BAC/F1/AUROC. If any fed experiment routes an antibody clf dataset through here, primary-metric mismatch.

**P2-6 — fed03 Part A C7 cross_val_predict inconsistencies.** `fed03:454-458`: fresh RF with `n_estimators=100` (vs 150 elsewhere), `cv=3`, not group-aware.

**P2-7 — `cluster_antibody_silos` silently concatenates all reps as a last-resort fallback.** `federated_helpers.py:1599-1610`. For high-dim antibody reps this is a memory bomb and changes clustering geometry without warning. Should raise.

**P2-8 — Test-cluster assignment in fed04 uses a KMeans with overwritten centroids.** `federated_helpers.py:309-321`. Overwriting `cluster_centers_` leaves `labels_`, `inertia_`, `n_iter_` stale and inconsistent. fed04:193 only uses `.predict()` so currently OK but fragile.

**P2-9 — Dead branch `dataset_name == 'herg'` in run_all special-case.** `federated_helpers.py:1769, 1797`. The per-dataset wrappers (`herg.py`) pass `DATASET = 'lhasa_herg'` — the `'herg'` string never reaches `run_all`.

**P2-10 — No per-replicate seed sweep.** All `random_state=0`/`42` are fixed. Variance unknown.

**P2-11 — `_run_antibody_federated` cross-task transfer allows mismatched task_type pairs.** `fed03:976-1030` iterates all (source, target) pairs blindly; `importance_profile_transfer` (helpers `:1636`) does not check task type.

**P2-12 — `compare_importance_profiles` returns nan with < 3 reps, fed04 treats nan as "stable".** `federated_helpers.py:1548-1549` returns `(nan, {})` when fewer than 3 reps; `fed04:225-229` prints `mean_spearman` then `if mean_spearman < 0.7:` — false for nan → always prints "STABLE across clusters". False-positive stability claim.

**P2-13 — `DOWNSTREAM_MODELS` imported but only used to populate an unused config field.** `federated_helpers.py:36, :130`. Dead-ish.

**P2-14 — `_FLUID_AVAILABLE = False` path issues a warning at import time** but every experiment then crashes deep in execution with a less helpful trace. Recommend a hard-fail at `run_all` entry (which already exists at `:1755-1758`) AND skip imports of fluid-dependent helpers when `_FLUID_AVAILABLE = False`.

## Verified-clean
- `__init__.py`, `caco2.py`, `herg.py`, `logd.py` — thin wrappers, no logic.
- `lhasa_herg_loader.py` aside from P1-1 — dedup, FP computation, and binarisation logic correct; UNKNOWN-key handling documented.
- `extract_importance_profile`, `compute_reg_metrics` — correct.
- `inject_noise_per_silo` (`helpers:330-385`) — silo-local range and MAD computation match the audit-log claims; both label_flip and Gaussian paths correct.
- `predict_fluid_similarity_ensemble` / `predict_fluid_quality_ensemble` — math matches FLuID's `annotate_transfer_data`/`federate_labels`; `pred_probs = fed_probs[:, 1]` confirmed.
- `dedup_across_silos` (`lhasa_herg_loader.py:160`) — first-wins with UNKNOWN warning correct.
- MUTATION_REPS two-pass concern: N/A. fed05 loads antibody silos via `load_data`, which handles MUTATION_REPS in `_shared.py:1647-1670`.
- `best_overall` — confirmed removed; no references in any federated file.
- `find_best_hybrid` calls in fed03 Part C (`:818-826`) and fed05 (`:145-153`) pass `augmentation_strategy='none'` EXPLICITLY — correct per the cache-collision fix rule.

## Priority summary
- **P0: 7** — top-impact fixes are P0-1 (one-line import; unblocks fed02 C6) and P0-2 (thread `augmentations=` everywhere).
- **P1: 14** — most damaging are P1-3/P1-4 (regression hardcoded) and P1-1 (stale `stat-cadd` path).
- **P2: 14** — most consequential are P2-4 (`'match'` evaluator semantics in `create_hybrid`) and P2-12 (false stability claim).
