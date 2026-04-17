# Precompute + Experiment Bug Audit — 2026-04-17

**Findings only. Nothing fixed.** This audit re-checks the state of the precompute/experiment stack after the 2026-04-15 fix round (commits `93d8da7`, `94c02d7`, `a8eb383`, `ecc97fa`, `f01a68c`). Verified against current code; prior audit `PRECOMPUTE_AUDIT_2026_04_15.md` is **not authoritative** for this report — some of its bugs are now fixed, and new ones have surfaced.

Scope: `src/kirby/find_best_hybrid.py`, `src/kirby/hybrid.py`, `tests/precompute_hybrids.py`, `tests/check_precompute_status.py`, `tests/experiments/_shared.py`, and every experiment file under `tests/experiments/{complexity,similarity,noise,federated}/` plus all `opig_*.py` and `imrie_transfer_learning.py`. Framework (F00–F15) deliberately excluded.

Severity legend:
- **P0** — wrong numbers possible OR runtime crash.
- **P1** — guaranteed cache miss / wasted compute (correct numbers).
- **P2** — hygiene / drift risk / dead code.
- **P3** — by design, flagged for visibility.

---

## TL;DR — the five things that will hurt you first

1. **Eleven experiment files have outright runtime crashes** from undefined variables (`task_type` or `FULL_BUDGETS`). Most are in files that currently rely on a try/except wrapper; under normal invocation they raise `NameError` before a single fold runs. Files listed in §2.1.
2. **`_make_search_meta` still omits ~13 behaviour-affecting parameters.** `candidate_ranking`, `greedy_tolerance`, `screen_threshold`, `two_phase`, `refinement_top_n`, `refinement_step_sizes`, `test_rep_subsets`, `dimension_tiers`, `weight_method`, `step_size`, `cv_splitter`, `rep_factories` (identity), `secondary_metrics`. Precompute uses defaults for all; any experiment that passes a non-default value silently collides on the same cache file. Details in §1.1.
3. **Synthetic `dataset_name` suffixes bypass the entire precompute.** c05, c13, c23, `opig_graphinity_ddg`, `opig_cdr_flexibility`, `opig_activity_cliff_qsar`, `opig_cgr_comparison`, `opig_kingsmith_component_splits`, `opig_ginkgo_developability` all build `dataset_name` from `{real_name}_{suffix}`. `DATASET_REPS` contains the real name only; precompute never iterates the suffix form. Every call is a full recompute. §4.
4. **`opig_ginkgo_developability.py:266` passes a custom `cv_splitter` whose identity is not in cache metadata.** Two runs with different fold assignments collide on the same cache file and silently serve the wrong folds. §1.2.
5. **`s04_activity_cliffs.py` silently treats every dataset as regression** — it never passes `is_clf` to `identify_cliff_compounds`, so on classification datasets cliffs are detected by `|Δy|` threshold instead of class mismatch. Wrong numbers, no warning. §3.4.

---

## 1. Infrastructure bugs (`find_best_hybrid.py`, `_shared.py`, `precompute_hybrids.py`)

### 1.1 Cache metadata omissions — P0

`_make_search_meta` at `src/kirby/find_best_hybrid.py:191-240` now captures 20+ fields including the ones the prior audit flagged (`internal_evaluators`, `allocation_methods`, `importance_methods`, `adaptive_pruning`, `augmentation_threshold`, `augmentation_budget`). Good.

Still missing, despite all being parameters of `find_best_hybrid` (signature at 2743-2781):

| Param | Default | Used at | Why it matters |
|---|---|---|---|
| `candidate_ranking` | `'rf_importance'` | 1524, 1607, 2209, 2425, 2515 | Changes which features enter the greedy search. |
| `greedy_tolerance` | `0.0` | 1532, 1613, 2213 | Changes early-stopping behaviour of greedy allocation. |
| `screen_threshold` | `None` | 2772, passes through to `_screen_reps` | Pre-search rep filter — can drop reps entirely. |
| `two_phase` | `False` | 2766 | Enables a second-phase refinement search. |
| `refinement_top_n` | `5` | 2767 | Count of configs refined in phase 2. |
| `refinement_step_sizes` | `None` | 2768, 2443 | Alternate step sizes explored in phase 2. |
| `test_rep_subsets` | `False` | 2757, 1990 | Evaluates leave-one-rep-out subsets. |
| `dimension_tiers` | `[50,100,200,500]` | 2760, 2549 | Determines `best_by_dimension` aggregation. |
| `weight_method` | `None` | 2759 | Feature importance method for the fitted result. |
| `step_size` | `5` | 1519, 1600, 1700, 1718 | Greedy search step — changes per-iteration feature adds. |
| `cv_splitter` | `None` | 2771, 1211, 1549, 2057 | Arbitrary sklearn splitter — no identity possible. |
| `rep_factories` | `None` | 2769, 1242-1328 | Per-fold callables producing FT embeddings. Identity cannot be compared. |
| `secondary_metrics` | `None` | 2770 | Extra metrics computed alongside primary. |

Every one of these is an input to the search and is not covered by `_cache_meta`. Example failure mode: SLURM job A writes the cache with `screen_threshold=None`; experiment B later calls with `screen_threshold=0.2`, which drops half the reps — but the filename+meta match, so B loads A's result and sees unfiltered numbers.

The specific ones most likely to bite today:
- `cv_splitter`: OPIG ginkgo uses a custom splitter (see §1.2).
- `rep_factories`: used by any FT experiment. If a factory is swapped (different LoRA weights) the cache won't notice.
- `candidate_ranking`, `greedy_tolerance`, `screen_threshold`: precompute always uses defaults, so this is latent unless an experiment customises them.
- `step_size`: precompute always uses default 5, same story.

### 1.2 `cv_splitter` identity not in cache key — P0 for OPIG ginkgo

`src/kirby/find_best_hybrid.py:2771` accepts `cv_splitter=None`. When a caller passes a custom splitter, neither the object nor any hashable summary of it is recorded in cache meta. `opig_ginkgo_developability.py:266-267` passes `_PrecomputedFoldSplitter(fold_assignments, n_folds)`. If the `fold_assignments` tensor changes (new Ginkgo fold CSV, different clustering run), the next search will load the old cache content as if the folds were the same. CV numbers become incomparable across runs.

### 1.3 `rep_factories` identity unhashable — P0 for FT experiments

`rep_factories` is a dict of callables. `_make_search_meta` skips it entirely. Any FT experiment (`imrie_transfer_learning.py`, `molecular_finetune_comparison.py`, `antibody_finetune_comparison.py`, `dta_finetune_comparison.py`, any c13 forward-compat path that hits a factory) risks cross-contaminating cache with embeddings from different factory versions. Mitigation requires hashing factory source or version string — neither present today.

### 1.4 Checkpoint files loaded without validation — P0

`_search_checkpoint_path` exists at `find_best_hybrid.py:97-100` and is read around lines 3082-3110. Grep confirms checkpoints are written during long searches and read on resume. There is no `_load_cache_with_meta` equivalent for checkpoints. If a parallel job crashed mid-search and left a `.ckpt.pkl` partial, a subsequent run with different parameters (different budgets, different allocation_methods, different seed) could silently resume from incompatible fold-zero results.

### 1.5 `_folds_cache_path` defined but unused — P2

`_folds_cache_path` at lines 110-114 and `_make_folds_meta` at 270-283 are defined but never called from anywhere in the codebase. Dead code today; if ever enabled without adding CV splitter type to meta, GroupKFold vs StratifiedGroupKFold would collide on the same folds cache.

### 1.6 Three parallel metric dispatchers — P2 drift risk

- `tests/precompute_hybrids.py::_get_metric(domain, task_type, labels)` line 156
- `tests/experiments/_shared.py::_resolve_search_metric(labels, task_type, metric, domain)` line 1827
- `tests/experiments/_shared.py::get_domain_metric(data)` line 397

All three produce the same `metric_name` for the same inputs today, but they are three independent implementations. Any future change to one (new domain, new metric) must be mirrored to the other two by hand. Single source of truth not enforced.

### 1.7 No per-dataset authoritative metric table — P2 drift risk (your user explicitly called this out)

There is no canonical `{dataset_name: (metric, lower_is_better, task_type)}` table in the repo. Metric and direction are derived on the fly from `data['task_type']` + domain dispatch. `DATASET_REPS` (in `complexity_helpers.py:70`) has `general`, `augmentation`, `top_tier`, `ft_reps` — but **no metric, no task_type, no direction**. If a loader's `task_type` flips, the metric flips silently with no cross-check.

This is the structural source of the "classification/regression got fucked up" concern. The fix is to put the authoritative triple next to DATASET_REPS or in a new registry.

### 1.8 `_detect_task_type` heuristic still present — P2

`find_best_hybrid.py:938-943` still has the auto-detect heuristic. `find_best_hybrid` itself rejects `task_type` outside `{'classification','regression'}` at lines 2917-2921, so the heuristic is only reachable via `tune_models` at 1049. Dead for the main path, but a second entry point (`tune_models`) could still silently misclassify a regression dataset with few unique labels.

### 1.9 Inconsistent entry-point contracts — P2

- `precompute_hybrid_searches(dataset_name, tier='general', task_type='auto', metric='auto')` — in-session precompute. `task_type='auto'` default contradicts `_resolve_search_metric`'s strict guard.
- `get_precomputed_search(dataset_name, tier='general', task_type=None)` — on-demand. `task_type=None` default also contradicts.
- `get_fitted_hybrid(dataset_name, tier='general', task_type=None)` — same.

All three will raise when `_resolve_search_metric` runs, but the defaults in the signatures say otherwise. This is exactly how callers end up passing unvalidated `task_type` through — the signature invites it.

### 1.10 Silent `try: ... except: pass` around cache calls — P2

Many experiment files wrap `get_precomputed_search` / `get_fitted_hybrid` / `find_best_hybrid` in bare exception handlers. Cache misses and NameErrors alike are swallowed. A couple of them even print nothing in the fallback path. No hit/miss counters are produced in any SLURM log by default (`verbose=False` is the norm).

### 1.11 Atomic write is POSIX-only — P3

`_atomic_write_pickle` relies on `os.replace(tmp, final)`. Fine on the ARC/Linux cluster you run on; breaks under NFS with non-atomic semantics and under Windows. No NFS locking guard. If two SLURM jobs ever write the same condition cache in parallel, the "later wins" is not guaranteed to be atomic under heavy filesystem contention.

### 1.12 `KIRBY_CACHE_DIR` read independently in two helpers — P3

`_cache_base()` in `find_best_hybrid.py:79` reads the env var each call. `_shared.py::_hybrid_fold_cache_dir` reads it separately. If two parts of the process see different values (env changes mid-run, subprocess doesn't inherit), the caches fragment silently. Low probability, but not zero on SLURM where modules can mutate env.

### 1.13 `check_precompute_status.py` — not audited in depth

Did not trace through. Note: the prior audit flagged this as filename-presence-only instead of content validation. Recent fix commits don't mention `check_precompute_status` specifically — worth a separate audit pass before trusting its output.

### 1.14 SLURM scripts not audited in this pass

`tests/slurm_scripts/submit_precompute*.sh` is out of scope here (user files, JSON-only rsync policy). Flagged: any CLI drift between the scripts and the precompute_hybrids.py argparse would create silent mis-submissions.

---

## 2. Complexity experiments (`tests/experiments/complexity/`)

### 2.1 Runtime-crash bugs (NameError) — P0

Confirmed by `grep -n "task_type = \|task_type=task_type\|data\['task_type'\]\|^def " …`. Each of these will raise `NameError: name 'task_type' is not defined` (or `FULL_BUDGETS`) before any fold runs.

| File | Line | Bug |
|---|---|---|
| `c08_calibration.py` | 98 | `budgets=FULL_BUDGETS` — imports only `SCREENING_BUDGETS` at line 23. |
| `c10_noise_robustness.py` | 125 | `task_type=task_type` — `run()` (line 89) never assigns `task_type`. Only `lower_is_better` is extracted at 106. |
| `c11_ensemble_vs_hybrid.py` | 92 | `task_type=task_type` — file only ever uses `data['task_type']` at line 60, no variable assignment. |
| `c13_forward_compatibility.py` | 188 | `budgets=FULL_BUDGETS` — imports only `SCREENING_BUDGETS` at line 57. |
| `c15_ml_theory_depth.py` | 113, 310, 344, 627 | `task_type=task_type` in **helper functions** `_run_distance_concentration` (line 58), `_run_kernel_analysis`, `_run_double_descent`, `_run_approximation_theory` — none take `task_type` as a parameter. The variable is only defined inside `run()` at line 752, which is scoped out. Helpers crash on first call. |
| `c17_information_theoretic_decomposition.py` | 233 | `task_type=task_type` — undefined. |
| `c19_importance_weighted_training.py` | 127 | `task_type=task_type` — undefined. File defines `data['task_type']` at 102 but never assigns to a variable. |

Seven files, 11 crash sites. If any of these experiments has ever produced results, it did so through a code path I haven't found — or it has never actually run after the a8eb383 commit tightened `task_type` to required.

### 2.2 Custom `models=` parameter breaks cache alignment — P1

`c02_complexity_grid.py:102` passes `models=model_instances` (actual sklearn instances) to `find_best_hybrid`. Precompute calls with `models=None` (which `_resolve_models` turns into the speed-tier default dict). Even though `_make_search_meta` normalises `models` to a sorted list of names, custom instance dicts are normalised via `sorted(models.keys())` — if the keys differ at all from the default set (e.g., contains only `'rf'`), the cache key changes.

Check `c02:75-91` vs `c02:98-114`: the first block is OK (default), the second block is the expensive recompute.

### 2.3 Single-budget loops prevent cache reuse — P1

`c09_formal_analysis.py:299, 432, 563` pass `budgets=[single_budget]`. Precompute writes with `budgets=FULL_BUDGETS`; `budgets` is in cache meta (tuple). Guaranteed miss.

### 2.4 Dataset-name suffixing — P1

`c05_learning_curves.py` builds `dataset_name=f"{dataset_name}_n{n_sub}_s{seed}"` on line 114. Unique cache key per subsample × seed. Intentional per subsample, but `DATASET_REPS` has no entry for any `_n*_s*` variant — precompute cannot seed these.

`c13_forward_compatibility.py:467, 502` builds `dataset_name=f"{dataset_name}_{without,with}_{rep_name}"`. Same story.

`c23_evaluator_model_matching.py` suffixes dataset_name with model/evaluator — intentional, and memory says c23 is deprioritized anyway.

### 2.5 Other complexity-specific concerns — mostly OK

- `c01`, `c04`: no hybrid calls. OK.
- `c06_within_family_gradient.py`: passes `models={var_name: var_factory()}` per variant at line 74. Custom models dict → cache miss by design.
- `c07_data_complexity_partitions.py`: partition-local reps, legitimately unique cache keys.
- `c12_modality_ablation.py`: imports `find_best_hybrid_tcga` (separate function). Audit whether that variant has its own validated cache — not covered here.
- `c14, c16, c18, c22, c24`: `task_type = data['task_type']` present in `run()`. Cache calls look aligned.
- `c21_single_rep_augmentation.py`: calls `find_best_hybrid(cache=True, ...)` — verify it passes `dataset_name` and `tier` (didn't trace).

---

## 3. Similarity experiments (`tests/experiments/similarity/`)

### 3.1 Runtime-crash bugs — P0

| File | Line | Bug |
|---|---|---|
| `s02_hybrid_similarity.py` | 154 | `task_type=task_type` — only `is_classification = data['task_type'] == 'classification'` at line 48, no scalar variable. |
| `s03_feature_concentration.py` | 60 | `task_type=task_type` — undefined. |
| `s11_bayesian_optimization.py` | 713 | `task_type=task_type` — undefined. |

### 3.2 Distance tier alignment — NOW CORRECT — P3 flagged only

The s04–s08, s10, s13 files that the prior audit flagged for `tier='general'` + `DISTANCE_EVALUATORS` mismatch have all been updated to `tier='general_distance'` (commit `93d8da7`). Verified:
```
s04:122, s05:77, s06:75, s07:132, s08:105, s10:701, s13:204 → tier='general_distance'
s02:154, s03:60, s11:713, s12:163 → tier='general'   (standard evaluators, correct)
```

### 3.3 `similarity_helpers.py:shared_search` passes tiers cleanly — OK

Lines 695 (`tier='general'`) and 715 (`tier='top'`). Clean, matches precompute.

### 3.4 `s04_activity_cliffs.py` always treats dataset as regression — P0

`s04:77-81` calls `identify_cliff_compounds(labels, …)` without passing `is_clf`. The helper defaults to regression-mode cliff detection (|Δy| > MAD threshold). On classification datasets, this is wrong — cliffs should be class-mismatch events. No guard, no warning. Wrong numbers for any clf dataset that goes through s04.

### 3.5 Silent fold-exception suppression — P2

`s04:155`, `s05:145`, `s05:172`, `s06:311`, `s13:93` wrap per-fold logic in `except Exception: continue` or `except Exception: ... = float('nan')`. Fold failures are invisible in the output, and the same masking could hide NameErrors from (3.1).

### 3.6 `s11_bayesian_optimization.py` direct `create_hybrid` calls — P3

Lines 567, 1109, 1142 build per-iteration hybrids directly. No forbidden `allocation_method='greedy'`, but the allocation method is read from config — confirm config doesn't allow greedy to sneak in.

---

## 4. Noise experiments (`tests/experiments/noise/`)

### 4.1 Runtime-crash bug — P0

| File | Line | Bug |
|---|---|---|
| `n06_bindingdb_ground_truth.py` | 97 | `task_type=task_type` — undefined. |

### 4.2 `n03_sas_landscape.py` — cache bypassed in 3 of 4 sub-analyses — P1

`n03:554` uses `get_hybrid_features_and_weights(...)` (from similarity helpers) for sub-analyses 1–3. This entry point does NOT know about `dataset_name`/`tier` and builds hybrids on the fly. Sub-analysis 4 (line 414) correctly uses `get_fitted_hybrid`. The two paths produce different hybrid construction, and the first three ignore precomputed caches. Not wrong numbers, just wasted compute + methodological inconsistency inside one experiment.

### 4.3 Everything else (n01, n02, n04, n05, n07–n13) — OK

All correctly pass `dataset_name`, `tier='general'` (or `'general_distance'` where appropriate), `speed`, `aug_condition`, and `task_type=data['task_type']`. Classification-guard logic appears sound:
- n02 branches on `is_clf` for reliability scoring.
- n04 early-returns when `is_clf`.
- n05, n09–n13 handle both regression (Gaussian injection) and classification (label flipping).
- n07's recovery percentage (line 298) is metric-agnostic by construction.
- n08's cliff discrimination (line 186) uses `rho < 0.1` — convention is "low estimator score on cliff compounds", which matches current estimator implementations.

### 4.4 Dead `budget=` argument — P2

Memory flag confirmed: n02:244, n07, n09, n10:419, n11, n12, n13 still pass `budget=` to `get_precomputed_search`. The parameter is ignored silently (deprecated in the docstring). No wrong numbers — this is a signal that authors didn't read the current API, so other assumptions about per-budget selection via `best_by_model_budget` may also be shaky in those files. Worth a second reading of each one, not covered by this audit.

### 4.5 N04, N06 return raw dict — P3

Per contract (memory). Not a bug but breaks aggregation uniformity.

---

## 5. OPIG experiments (`tests/experiments/opig_*.py` + `imrie_transfer_learning.py`)

### 5.1 Runtime bugs — P0

| File | Line | Bug |
|---|---|---|
| `imrie_transfer_learning.py` | 189-194 | `find_best_hybrid(..., cache=True, ..., dataset_name=..., tier=..., metric='rmse')` — **`task_type` is not passed**. `find_best_hybrid` raises on line 2917-2921 with `task_type={task_type!r}`. |
| `opig_blueprint_transfer.py` | 74 | `task_type = target_data['task_type']` — OK. But note: source and target may have different task_types. The code uses target's task_type for the source search at line 105-110 as well, which is wrong if source and target differ. |

### 5.2 Dataset-name suffixing — every custom-loader OPIG experiment — P1

| Experiment | dataset_name pattern | Problem |
|---|---|---|
| `opig_graphinity_ddg.py:328` | `graphinity_foldx_n###` | Size-suffixed names not in DATASET_REPS. (Note: `graphinity_foldx`, `_rosetta`, `_augmented` ARE in DATASET_REPS — the `_n###` suffix is the issue.) |
| `opig_humanness.py` | `humanness_species` | Not in DATASET_REPS (only `humanness_therapeutic` is). |
| `opig_cdr_flexibility.py:224` | `cdr_flexibility_{cdr_types}` | Not in DATASET_REPS. |
| `opig_activity_cliff_qsar.py:311` | `chembl_qsar_{target}` | Not in DATASET_REPS. |
| `opig_cgr_comparison.py:298` | `{ds}_{pool_name}` | Suffix not in DATASET_REPS. |
| `opig_kingsmith_component_splits.py:320` | `{ds}_loo_{component}` | Suffix not in DATASET_REPS. |
| `opig_ginkgo_developability.py:279` | `ginkgo_{prop}` | Only 6 of 9 props are in DATASET_REPS (missing `ginkgo_tm2`, `ginkgo_aggregation`, `ginkgo_purity`). |

Every one of these experiments does a full cold recompute on every run. For the user's "stuck in survey stage" concern, this is likely the single biggest contributor. These experiments can't benefit from SLURM precompute unless either (a) DATASET_REPS gets their actual dataset-name variants, or (b) the suffixing is removed and the cache is keyed on a `condition_suffix` slot instead.

### 5.3 `opig_ginkgo_developability.py` custom splitter — P0

See §1.2. The `cv_splitter=_PrecomputedFoldSplitter(fold_assignments, n_folds)` at line 266-267 is not in cache meta. If the Ginkgo hierarchical fold CSV ever changes, the cache silently serves stale fold assignments.

### 5.4 Cross-domain blueprint transfer direction handling — P1 / P0 depending on input

`opig_blueprint_transfer.py:79-81` only `print`s a warning if source and target domains differ. If a future user actually transfers rmse→spearman, the delta logic at lines 357-361 uses target's `lower_is_better`; it is sign-normalised correctly for target but the source scores it's comparing against live in a different metric space. "Which rep is better on source" does not translate to "which rep is better on target" — the **blueprint** transfers, not a comparison. The current code is structurally fine for the "transfer blueprint + re-score on target" workflow, but a reader could easily misread it as comparing cross-domain scores. Flag rather than bug.

### 5.5 OPIG loaders set `task_type` — OK, mostly

| Loader | Sets `task_type` | Sets `domain` | Notes |
|---|---|---|---|
| `trastuzumab.py:219` | `'classification'` | no | Relies on dispatch default. |
| `graphinity.py:312` | `'regression'` | no | |
| `humanness.py:237` | `'classification'` | no | |
| `ginkgo_dev.py:125` | `'regression'` | no | Provides `fold_assignments`. |
| `itsflexible.py:178` | `'classification'` | no | |
| `chembl_qsar.py:207` | `'regression'` | no | Uses MAE (in BUILTIN_METRICS). |
| `reaction_yields.py:177,310,363,486` | `'regression'` | no | |

None set `domain` explicitly. All rely on the dispatch default in `_shared.py` / `precompute_hybrids.py` / `complexity_helpers.py`. That dispatch is identical across the three, but another P2 drift risk (§1.6).

### 5.6 `opig_trastuzumab_enrichment.py` — OK

`task_type='classification'`, metric `auc`, passes `dataset_name='tz_her2_binary'` + `tier='general'`. Aligned.

### 5.7 `imrie_transfer_learning.py` direct `find_best_hybrid` pattern — P2

All other cache-using experiments go through `get_precomputed_search` / `get_fitted_hybrid` in `_shared.py`. imrie calls `find_best_hybrid(cache=True, ...)` directly — see 5.1 for the missing-task_type bug. Inconsistent entry point means the shared helpers' metric/task resolution never applies.

---

## 6. Forward compatibility (c13) and federated (fed01–fed05)

### 6.1 c13 crash bug (P0) — see §2.1

c13:188 NameError on `FULL_BUDGETS`. Pattern identical to c08.

### 6.2 c13 behaviour otherwise — OK

Per-fold synthetic dataset_name usage (§4 in complexity) is intentional: c13 measures the effect of adding/removing individual reps, so each rep variant gets its own cache. Metric direction handling (delta sign, win-rate) is correct (`lower_is_better` respected). Legacy-rep detection (line 130) is dynamic and robust.

### 6.3 Federated fed01–fed05 — cache alignment looks OK

All five files pass `dataset_name`, `tier='general'`, `task_type=data['task_type']` to `get_precomputed_search`. Per-silo `create_hybrid` calls are intentional (privacy), with no `allocation_method='greedy'` hardcoded.

### 6.4 fed03 Part C datasets ARE in `DATASET_REPS` — OK

`ddg_combined`, `abagym_HER2`, `flab_thermo_tresanco`, `graphinity_foldx` all appear in `DATASET_REPS`. Precompute can seed these. (Note that the `abagym_*` loader expects task_type regression — confirm with loader audit; not done here.)

### 6.5 fed03 `herg_asymmetric` — P1

Synthetic dataset name `herg_asymmetric` is not in `DATASET_REPS`. First run builds and caches; every dataset_name variant becomes its own cache entry. Same pattern as the OPIG suffix problem.

### 6.6 fed05 SELF_LOADING — OK

`federated_helpers.py:147-149` registers correctly. fed05 uses `find_best_hybrid` per silo directly and does not depend on precompute.

### 6.7 federated/caco2.py, herg.py, logd.py, lhasa_herg_loader.py — not audited deeply

Memory flags `herg` (classification, TDC) vs `herg_ki` (regression, ChEMBL) disambiguation. `lhasa_herg_loader.py` returns binary labels; `federated_helpers.py:106` wraps with `'task_type': 'classification'`. Fed experiments that load the Lhasa version are consistent. No fed experiment appears to load `herg` and treat it as regression.

---

## 7. Cross-cutting patterns and user's specific concerns

### 7.1 "classification/regression got fucked up" — TRUE and PARTIAL

**What is fixed:**
- `task_type` is now required by `find_best_hybrid` (raises 2917-2921).
- `precompute_hybrids.py:214-219` fails loudly if a loader returns `task_type` that's not `classification`/`regression`.
- `_require_task_type` in `_shared.py:1248-1264` guards `load_data`.
- Loaders audited (see §5.5) all set `task_type`.
- `get_domain_metric` now hard-fails on unknown task_type.

**What is still broken:**
- **No per-dataset authoritative metric/direction table** — all derived on-the-fly. `DATASET_REPS` has no `metric`/`task_type` field. Adding those would give the "single source of truth" you asked for.
- **Three parallel metric dispatchers** (§1.6) means a change to one silently drifts the others.
- **s04 hardcodes regression cliff detection** (§3.4) — wrong numbers on any clf dataset that reaches s04.
- **`_detect_task_type` heuristic** still reachable from `tune_models` (§1.8).

### 7.2 "caching has been fucked" — PARTIAL

**Progress since 2026-04-15:**
- `aug_condition='none'` remap removed.
- `internal_evaluators`, `allocation_methods`, `importance_methods`, `adaptive_pruning`, `augmentation_threshold`, `augmentation_budget` in cache meta.
- `general_distance` / `top_distance` tiers written by precompute, consumed correctly by s04–s08, s10, s13.
- `dataset_name` + `tier` threaded through the main 42 direct callers (commit `ecc97fa`).
- Atomic writes via `tempfile.mkstemp` + `os.replace`.
- Content validation on load (meta mismatch = silent recompute, not wrong answer).

**Still structurally brittle:**
- 13 search parameters not in cache meta (§1.1). Precompute doesn't customize any of them, so today's runs align, but any experiment that does customize silently collides.
- `cv_splitter` and `rep_factories` cannot be hashed — P0 for any run that uses them.
- Checkpoint files validated weakly (§1.4).
- Synthetic-dataset_name OPIG experiments can never hit precompute (§5.2).

### 7.3 "ONLY FULL tier is precomputed" — STILL TRUE

Nothing in the audit found precompute writing `speed='fast'`. All experiment callers use `speed='full'`. If anyone ever flips an experiment to `'fast'`, every call cache-misses because `speed` is in the filename. Not a problem today; latent trap.

### 7.4 "additional hybrids get computed for certain experiments" — EXPECTED

Federated (fed01–fed05): per-silo `create_hybrid` by design.
F10–F15: deliberate cache bypass to test algorithm choices.
c07 partitions: legitimately unique cache keys.
c12 TCGA uses a separate variant of `find_best_hybrid` — not audited here; flag for follow-up.
All OPIG suffixed-dataset-name experiments (§5.2) also do additional compute — this is the likely source of "stuck in survey stage".

### 7.5 "federated experiments require specific datasets" — true and OK

fed01/02/04 depend on `herg` (Lhasa classification).
fed03 Part A/B on `herg`, Part C on `ddg_combined`/`abagym_HER2`/`graphinity_foldx`/`flab_thermo_tresanco`.
fed05 on `ddg_combined`/`abagym_HER2`/`flab_thermo_tresanco`.

Every one of those names is in `DATASET_REPS`. Precompute can seed them — only question is whether the current SLURM submission actually includes them. Not audited.

### 7.6 "OPIG experiments are a hodgepodge mess" — true

Every OPIG experiment except `opig_blueprint_transfer`, `opig_trastuzumab_enrichment`, `opig_humanness_therapeutic` uses a synthetic dataset_name (§5.2). None of those are seed-able by the current precompute path. The OPIG suite can't benefit from precompute until the suffixing is addressed.

### 7.7 "some things are still missing and stuck in the survey stage" — MOST LIKELY CAUSES

1. OPIG suffix problem (§5.2) — 7+ experiments cannot use precompute.
2. c05/c13/c23 suffix problem — by design but no precompute.
3. fed03 `herg_asymmetric` — synthetic, no precompute.
4. Non-`DATASET_REPS` datasets (`humanness_species`, 3 ginkgo props) — no precompute coverage.
5. Any experiment on the 11 NameError-crashing files (§2.1, §3.1, §4.1, §5.1) — obviously produces nothing.
6. FT experiments with `rep_factories` — cache identity unreliable.

---

## 8. Prioritized fix queue (you asked for findings only; these are the ordering recommendations)

**Must fix before any correctness-dependent run:**
1. The 11 NameError crash bugs (§2.1, §3.1, §4.1, §5.1). These are either blocking the experiment entirely or being masked by try/except.
2. `s04_activity_cliffs.py:77-81` — `is_clf` not passed. P0 silent wrong numbers on clf datasets.
3. `imrie_transfer_learning.py:189-194` — missing `task_type=`.
4. OPIG ginkgo `cv_splitter` not in cache meta (§1.2).

**Should fix to stop wasting compute:**
5. Add all 13 missing params to `_make_search_meta` (§1.1). Most important: `cv_splitter`, `rep_factories`, `candidate_ranking`, `greedy_tolerance`, `screen_threshold`, `step_size`.
6. c02 `models=model_instances` — align with precompute `models=None` (§2.2).
7. c09 single-budget loops (§2.3) — use FULL_BUDGETS + per-budget selection via `best_by_model_budget`.
8. OPIG synthetic `dataset_name` pattern (§5.2). Two options: add the variants to `DATASET_REPS`, or key the cache on a separate `condition_suffix` field.

**Should fix for durability:**
9. Collapse the three parallel metric dispatchers to one (§1.6).
10. Add a per-dataset authoritative metric/task/direction registry (§1.7 / §7.1).
11. Checkpoint load validation (§1.4).
12. Hit/miss logging / counters for cache (§1.10).
13. `check_precompute_status.py` content validation audit.
14. `_detect_task_type` removal or narrow scoping (§1.8).
15. Deprecated `budget=` removal from noise experiments (§4.4).

**Out of scope of this audit (worth separate passes):**
- `find_best_hybrid_tcga` vs main `find_best_hybrid` — c12, TCGA survey scripts.
- Framework F00–F15 (user said do not touch).
- Rep caching (`~/.cache/kirby/reps/`, `rep_cache.py`) — user handles separately.
- `tests/slurm_scripts/submit_precompute*.sh` — JSON-only rsync policy.
- Result aggregation / BOM.

---

## 9. What this audit did NOT prove

- Whether SLURM job output currently contains the condition cache files you expect (cannot check server).
- Whether results JSONs already on disk were produced before or after each fix commit — `CACHE_FIX_PLAN.md` section 2 lists the suspect set, reproduce that analysis before trusting anything.
- That the 11 NameError bugs are actually being hit — some may be behind bare except clauses producing empty output. Either way the code must be wrong.
- Content validation of baselines/folds caches at scale — only structural review here.
- Whether `precompute_hybrid_searches` (in-session) and `precompute_hybrids.py` (SLURM) align byte-for-byte on identical datasets. The entry points differ (`speed='full'` hardcoded in one, `--speed full` default in the other; `models=None` in both). Worth a one-shot comparison for one dataset.

These are the right next questions, not blockers.
