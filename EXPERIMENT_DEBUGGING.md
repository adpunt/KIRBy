# EXPERIMENT_DEBUGGING.md

Audit of all federated + OPIG experiments. One Opus agent per experiment, audited 2026-05-25.

**Coverage:** 15 experiments — 5 federated (fed01–fed05) + 10 OPIG (activity_cliff_qsar, blueprint_transfer, cdr_flexibility, cgr_comparison, ginkgo_developability, graphinity_ddg, humanness, kingsmith_component_splits, trastuzumab_enrichment, imrie_transfer_learning).

**Severity legend:** P0 catastrophic (results invalid or run crashes) · P1 wrong results (silent) · P2 inefficient / suspicious · P3 cosmetic / docstring.

---

## Cross-cutting bugs (affect multiple experiments)

### X1. `augmentation_strategy=` kwarg silently overrides pool iteration — P0
**Where:** `tests/experiments/_shared.py:2480` — `call_kwargs.update(kwargs)` in `get_precomputed_search` overrides the per-pool `aug_strat = aug_condition`.

**How it leaks:** `get_best_search_for_downstream(..., augmentation_strategy=X)` forwards via `**kwargs` → `extra` → `get_precomputed_search` → `call_kwargs.update`.

**Effect:** every pool iteration's cache key for `augmentation_strategy` is forced to the experiment's hard-coded value. 'capped' and 'uncapped' pool entries load under the wrong cache file (collision per `project_cache_collision_fix.md`); pool selection becomes meaningless.

**Affected callers:**
- `opig_activity_cliff_qsar.py:305-318`
- `opig_graphinity_ddg.py:357`
- `opig_kingsmith_component_splits.py:343`
- `opig_cgr_comparison.py:295-310`
- `imrie_transfer_learning.py:191` (benign: augs=None)
- `opig_blueprint_transfer.py` (benign: augs=None)

**Fix:** strip `augmentation_strategy` from forwarded kwargs in `get_best_search_for_downstream`/`get_precomputed_search`; remove the redundant kwarg from each caller.

---

### X2. `kirby_hybrid` rebuild omits augmentation features — P1
**Pattern:**
```python
fi = {r: {'selected_indices': idx} for r, idx in fitted.selected_indices.items()}
reps['kirby_hybrid'] = apply_feature_selection(all_reps, fi)
```

**Effect:** `fitted.selected_indices` only has base reps; augmentation selections live in `fitted.augmentation_info`. The reported `kirby_hybrid` is a strict subset of what find_best_hybrid scored.

**Affected:** `opig_activity_cliff_qsar.py:324-329`, `opig_graphinity_ddg.py:372-377`, `opig_kingsmith_component_splits.py:358-366`.

**Fix:** mirror `_shared.py:1996-2000` — call `apply_augmentation_selection(augmentations, fitted.augmentation_info)` and `np.hstack`.

---

### X3. Hybrid feature-selection leakage — P0/P1
**Pattern:** `return_fitted=True` returns `FittedResult` built on FULL labels via `_compute_fitted_result` (`find_best_hybrid.py:3661`). Experiment then re-evaluates under 5-fold CV with same labels → every val fold sees its own labels during selection.

**Affected:** `opig_cdr_flexibility.py:234-239`, `opig_kingsmith_component_splits.py:339-393`, `opig_trastuzumab_enrichment.py:281-313`, `imrie_transfer_learning.py:188-201`.

**Fix:** build hybrid per-fold (find_best_hybrid on `train_idx`, `apply_feature_selection` on `val_idx`), OR report search's own CV score directly.

---

### X4. `'match'` evaluator silently degrades to equal split in federated_helpers — P0
**Where:** `federated_helpers.py:820-822`, `:1497-1502`, `:1514-1517`.

**Mechanism:** Blueprint default `internal_evaluator='match'` (find_best_hybrid.py:718). `get_model_factories(['match'], …)` doesn't know 'match' → returns `{}` → silo_scores all 0.0 → `total_score <= 0` → equal-budget fallback (`budget // k`).

**Effect:** the KIRBy-vs-FLuID quality-weighted allocation claim is structurally invalid. Phase 4 correlations are zero-by-construction.

**Affected:** fed01, fed02, fed05.

**Fix:** resolve `'match'` to downstream model family before `get_model_factories`, mirroring `hybrid.py:3331-3355`.

---

### X5. `'match'` → 'ridge' silent degradation in `create_hybrid` (fed02) — P0
**Where:** `fed02_rep_diversity_silos.py:309-316, 356-363` + `federated_helpers.py:1497-1502`.

**Mechanism:** `create_hybrid(..., internal_evaluator='match')` without `model=`. In `hybrid.py:3331-3355`, model=None → 'match' resolves to 'ridge'. Violates `feedback_no_ridge.md`.

**Fix:** thread `model=` through every silo-level `create_hybrid` call.

---

### X6. Single 80/20 split across federated experiments — P1
`cv_split(groups, 1, …)` yields one GroupShuffleSplit. Per `feedback_no_random_splits.md`: forbidden. Worse: `federated_helpers.py:107-110` silently falls back to ShuffleSplit (random) if scaffold assignment raises.

**Affected:** fed01:156-157, fed02:149-156, fed03 (5 sites), fed04:172-173, fed05 (various), opig_blueprint_transfer:489.

**Fix:** wrap in 5-fold CV; hard-fail when groups unavailable.

---

### X7. `adaptive_pruning=0.25` silently ignored — P2/P3
**Mechanism:** callers pass `adaptive_pruning=0.25` AND `**_pkw` (with `adaptive_pruning_schedule`). Per `find_best_hybrid.py:1734`, schedule wins → 0.25 is dead code (actual threshold=0.15).

**Affected:** `opig_cgr_comparison.py:290-310`, `opig_ginkgo_developability.py:289`, `opig_blueprint_transfer.py:126,140` (causes precompute miss), `imrie_transfer_learning` indirectly.

---

# Per-experiment audits

## 1. fed01_noise_heterogeneity.py — 1 P0, 4 P1, 3 P2, 4 P3

### P0
- **federated_helpers.py:820-823 (+ fed01:354) — `'match'` → equal-budget (X4).** C5 silently non-quality-weighted. Phase 4 r_budget ≈ 0 by construction.

### P1
- **fed01:156-157 — single 80/20 split.** All conditions/sigmas on one test fold.
- **federated_helpers.py:107-110 — silent random-split fallback if scaffold assignment fails.**
- **fed01:247-280 (C0) — docstring "upper bound" but trains on NOISY labels.**
- **fed01:564 — Phase 4 noise pattern degenerate.** `noise_levels_det = {K: 0.2}` only → Spearman across [0,0,0,0,0.2] is statistically empty.

### P2
- Lhasa path skips precompute (federated_helpers:1869-1875).
- Phase 2 retrains teacher RFs from scratch per sigma_frac.
- O(n²) distance matrix on full train set.

### P3
- fed01:39 dead import; helpers:1000-1001 dead variable; helpers:1041 single-class f1 crash risk.

### Verified clean
- All `create_hybrid` calls pass `augmentation_strategy` explicitly.
- Metric directions correct. No SVR cap issue.

---

## 2. fed02_rep_diversity_silos.py — 3 P0, 5 P1, 4 P2, 4 P3

### P0
- **fed02:149-156 — single 80/20 split.**
- **fed02:309-316, 356-363 — `'match'` → 'ridge' (X5).** All C4/C5 driven by Ridge.
- **federated_helpers:1497-1502 — same bug in C6 `fluid_within_kirby_across`.**

### P1
- **fed02:124 — uses `bp.augmentation_strategy` instead of `meta['aug_condition']`** (canonical at _shared.py:2747).
- **fed02:107-118 + 313, 360 — blueprint selected on full reps, applied to silo subset.**
- **fed02:286-287, 307-308, 354-355 — `train_augs` computed 3× under different names.**
- **federated_helpers:1237 — inverse-RMSE weight blows up on overfit tiny silo** (1.0/max(rmse, 1e-6)).
- **federated_helpers:1221, 1373-1375 — quality ensemble uses random KFold** (within-scaffold leakage).

### P2
- helpers:466-467, 648-649 — `rng.choice(replace=False)` can request > population.
- fed02:217, 231 — `teacher_models[0]` shape inconsistency.
- fed02:159-160 — slices every rep in `reps` but only 5 silo reps used.

### P3
- Docstring describes regression weighting only; clf path is MCC-weighted.
- Datasets not enforced.

---

## 3. fed03_synthetic_private.py — 2 P0, 7 P1, 5 P2, 4 P3

### P0
- **fed03:927 — `apply_augmentation_selection` NameError** (Part C). Swallowed by bare except.
- **fed03:329→335 — `teacher_models`/`cluster_data_train` defined inside C2 try block, read by C3.** Partial-C2-failure silently corrupts C3.

### P1
- **fed03:850-863 — Part C will NEVER hit precompute cache** (adaptive_pruning=0.25 + rep_set mismatch).
- **fed03:1079 — Part C C3 always uses `RandomForestRegressor`** regardless of task_type.
- **fed03:559-565, 645-651 — Part B uses 'capped' blueprint on silo subset it wasn't tuned for.**
- **fed03:235-236, 586-587, 899, 1002-1004, 1057-1058 — all cv_split n_splits=1.**
- **fed03:460-464 — Part A C7 uses unshuffled `cross_val_predict(cv=3)`** on silo-concatenated rows → fold 0 entirely one silo.
- **fed03:865-868 — Part C silently falls back to hardcoded defaults** when `silo_blueprints` empty.
- **fed03:1139 — Dispatcher uses 'herg_ki' but run_all uses 'herg'/'lhasa_herg'** → falls through to Part B silently.

### P2
- fed03:108 dead `aug_cond` in Part A.
- fed03:106-108 vs 914 — inconsistent "trust pool" pattern.
- fed03:289 — ValueError if all silos error.
- fed03:1099-1110 — clf silos print nan.
- helpers:629-712 — `student_size=10000` hardcoded cap.

### P3
- fed03:90, 546 unused metric_fn/lower_is_better; fed03:84-127 data['smiles'] assumption; lazy import; default blueprint heuristic.

---

## 4. fed04_cluster_aware_importance.py — 1 P0, 3 P1, 5 P2, 5 P3

### P0
- **fed04:149 — augmentation row misalignment when valid_mask drops molecules.** Latent for Lhasa hERG; corrupts non-Lhasa datasets.

### P1
- **fed04:326-332 — Phase 5 smooth-factor sweep drops augmentations** (Phase 2 keeps them). Not comparable.
- **fed04:152-169 — pool blueprint selected on regression labels, applied to binarized labels.**
- **fed04:172-173 — single 80/20 split.**

### P2
- fed04:164 — bare except swallows pool failures silently.
- fed04:212-219 — global hybrid recomputed despite pool already searched.
- fed04:292-300 — per-sample predict loop.
- fed04:78-86, 212-219 — `create_hybrid` without groups → non-group val split.
- pruning_mode not threaded.

### P3
- Dead unpacking; `random_state=0` instead of 42; stale comments; doc note on cluster_augs alignment; defensive `.copy()`.

---

## 5. fed05_antibody_federated.py — 2 P0, 4 P1, 4 P2, 5 P3

### P0
- **fed05:713 (C7), :791 (C8) — cross-silo `reg.predict(X_te)` dim/PCA-basis incompatibility.** Each silo PCA'd to different dim/basis. Crashes or returns garbage.
- **fed05:706-714, 707-792 — label-scale incompatibility.** kcal/mol vs norm vs °C — weighted ensemble dominated by largest-magnitude silo. Headline FLuID-vs-KIRBy claim hinges on broken C7/C8.

### P1
- **fed05:335-345 (C3) — `c_budget` computed but never passed to `create_hybrid`** (default 100 used despite "conservative" comment).
- **federated_helpers:820-822, 1514-1517 — `'match'` → equal-budget (X4).** C4 cannot detect proportional-to-quality.
- **fed05:271 — C2 hardcoded `RandomForestRegressor` ignoring task_type.**

### P2
- C1 uses per-silo blueprints; C3/C4/C5/C6 use first-iter silo's hyperparams.
- fed05:496-510 — `offset` reassigned not accumulated (brittle ≥3 silos).
- fed05:167-186 — full find_best_hybrid per silo just to extract 3 strings.
- fed05:718 — hardcoded `cv=3`.

### P3
- Metric mismatch (spearman search, RMSE report).
- C5 augmentations skipped silently.
- `random_state=0` throughout.

---

## 6. opig_activity_cliff_qsar.py — 1 P0, 1 P1, 1 P2, 4 P3

### P0
- **305-318 — augmentation_strategy kwarg leak (X1).**

### P1
- **324-329 — kirby_hybrid excludes augmentation features (X2).**

### P2
- 328 — kirby_hybrid in-sample bias (selection + eval on same arrays).

### P3
- best_mcc default 0; R² fallback 0.0; top3_base RF-only; docstring vs paper "3 reps × 3 models".

### Verified clean
- MAE direction correct everywhere.
- SVR cap not triggered (n≤6333).
- `condition_suffix=target` partitions cache.

---

## 7. opig_blueprint_transfer.py — 4 P1, 2 P2, 4 P3

### P1
- **489 — default mode='intermediary' → n_splits=1.**
- **126, 140 — adaptive_pruning=0.25 mismatches precompute + intersection rep set = cache miss.**
- **264-265 etc. — `m.predict()` for classification with AUC metric.** No `predict_proba` branch.
- **79-84 — no task_type compatibility check** between source/target.

### P2
- Source search uses target's metric.
- Silent skipping of gp/catboost/dnn/cnn.

### P3
- Dead `--augment` flag; "pretrained hybrid" overstates; implicit fold-match assumption; arbitrary 0.001 threshold.

### Verified clean
- `apply_feature_selection` indexing correct; per-fold refit; `augmentation_strategy='none'` explicit.

---

## 8. opig_cdr_flexibility.py — 0 P0, 2 P1, 3 P2, 5 P3

### P1
- **113 — effective random splitting** (dedup'd CDR3 → 1 sample per group).
- **234-239 — kirby_hybrid feature-selection leakage (X3).**

### P2
- Paper baseline comparison is apples-to-oranges; hybrid=RF only; broad except swallows.

### P3
- Paratope 8D (Tier 1) used as base; condition_suffix order-sensitive; --reps whitespace not stripped; argparse default; TCR docstring vs antibody reps.

### Verified clean
- AUC direction; `augmentation_strategy='none'` explicit.

---

## 9. opig_cgr_comparison.py — 0 P0, 2 P1, 5 P2, 7 P3

### P1
- **350 — `top3=None` crashes pool** (silently drops dataset via outer except).
- **295-310 — augmentation_strategy kwarg leak (X1).**

### P2
- adaptive_pruning ignored (X7); Spearman NaN→0; uint8 fast-path dead; in-sample bias; CGR rep silent zero-fill.

### P3
- R² fallback 0; minor inconsistencies; cgr_smiles_lm grouping; top3 RF-only; two RF baselines under different keys; broad except.

### Verified clean
- RMSE direction; registry confirms regression; SVR cap not triggered.

---

## 10. opig_ginkgo_developability.py — 2 P1, 5 P2, 7 P3

### P1
- **287, 279 — double-prefix `"ginkgo_ginkgo_hic"`** → cache always misses precompute.
- **270-274 — cv_splitter override bypasses precompute cache key.**

### P2
- adaptive_pruning ignored; top-10% recall retrains model already trained; random KFold fallback; double-count augs; baseline recompute.

### P3
- Dead PAPER_METHOD_BASELINES; dead imports; "all 9" claim but only 6 in DATASET_REPS; "structural" docstring vs actual augs; max-on-NaN; redundant per-rep recompute; docstring missing 'ci'.

### Verified clean
- Spearman direction; top-K recall sign flip; SVM cap fine; argsort correct.

---

## 11. opig_graphinity_ddg.py — 0 P0, 5 P1, 4 P2, 9 P3

### P1
- **357 — augmentation_strategy kwarg leak (X1).**
- **372-377 — kirby_hybrid drops augmentations (X2).** AAC lost.
- **138-153 — `evaluate_rep_ddg` runs SECOND full CV pass** for pearson_global. 2× wall time.
- **140 — second pass skips `_filter_high_dim_per_fold`.** pearson and pearson_global on different feature spaces.
- **484-497 — gap-to-EGNN compares per-fold-mean to paper's concat Pearson.** Apples-to-oranges.

### P2
- max(NaN); best_r mixes hybrid into individual max; transfer skips filter; one --max-samples for both presets; groups=None → per-row KFold.

### P3
- Docstring "ALL" vs max_samples=5000; transfer ignores --n-folds; redundant scale; transfer flat-float; SVR gate by full-n; loader default 50000 vs experiment 5000; redundant merge; silent build failure.

---

## 12. opig_humanness.py — 0 P0, 1 P1, 2 P2, 4 P3

### P1
- **86 — `esmc_llr`, `antiberty_llr` are all-zero vectors** (loader sets empty mutations).

### P2
- Hybrid built for RF only, evaluated under 6 models.
- Identity grouping on unique OAS sequences ≈ random.

### P3
- Dead `rep_name=''`; sort tie-break; metric_val edge case; smoke-test docstring.

### Verified clean
- AUC direction; `augmentation_strategy='none'` explicit; task_type='classification'; VH-only; species map correct.

---

## 13. opig_kingsmith_component_splits.py — 2 P0, 3 P1, 3 P2, 2 P3

### P0
- **343 — augmentation_strategy kwarg leak (X1).** Cache files clobbered.
- **339-393 — leakage: blueprint selected on same CV groups used to evaluate it.** Defeats King-Smith extrapolation entirely.

### P1
- **358-366 — augmentations dropped (X2).**
- **30-34, 50 — docstring claims LOO but only base (3) and ligand (4) actually do LOO.** Others run 5-fold GroupKFold under "loo_" cache key.
- **339-352 — n_folds not pinned, cache says 5-fold but small-group splits run fewer.**

### P2
- Redundant pool runs; R²=0 fallback; best-of-6-models cherry-picks.

### P3
- Dead uint8 branch; dead case-insensitive warning.

### Verified clean
- RMSE registered; SVR cap fine (n≤4599); per-fold StandardScaler.

---

## 14. opig_trastuzumab_enrichment.py — 2 P0, 4 P1, 3 P2, 3 P3

### P0
- **281-313 — hybrid feature-selection leakage (X3).**
- **325-336 — LLR aggregation `X.mean(axis=1)` of [sum_llr, mean_llr, min_llr].** sum_llr dominates → ranking by mutation count, not binder probability.

### P1
- Hybrid built for RF only;
- 185-189 — silent fold drop on fit failure;
- 144-148 — `_is_binary_rep` fragile on all-zero LLR;
- 106-141 — argsort ties index-ordered → boundary leak.

### P2
- Pool overhead with augs=None; SVC probability=True 5× internal CV; double iteration.

### P3
- Mixed zero-shot/supervised docstring; PAPER_BASELINES units; E@1% at n=10/fold high variance.

### Verified clean
- Stratified 5K subsample; `augmentation_strategy='none'` explicit; AUC direction; seed=42; SVM cap fine.

---

## 15. imrie_transfer_learning.py — 1 P0, 4 P1, 2 P2, 3 P3

### P0
- **188-201 — kirby_hybrid feature-selection leakage (X3).**

### P1
- **135 + dnn_pretrained.py:169-171 — double/inconsistent feature scaling.** DNN-scratch sees raw binaries; DNN-TL sees z-scored binaries.
- **dnn_pretrained.py:169 — fine-tune fits new scaler instead of using pretrain's `feat_mean`/`feat_scale`.**
- **dnn_pretrained.py:179-182 + dnn.py:60-63 — internal ES holdout uses `rng.permutation`** (not group-aware).
- **97, 106, 110 — single `random_state=42` shared across all folds.**

### P2
- augmentation_strategy='none' fragile pattern; pool overhead with augs=None.

### P3
- 5 vs 4 conditions docstring drift; --datasets example mismatch; no end-of-run rep×condition summary.

### Verified clean
- freeze_layers=-1 correct; optimizer on trainable only; InChIKey leakage filter applied; collect_downstream_inchikeys covers all default datasets; metric direction internally consistent; no SVR cap issue.

---

# Triage / fix order

**Fix first (cross-cutting, max leverage):**
1. X1 — `augmentation_strategy` kwarg leak. Central fix in `_shared.py` + remove from 5+ callers.
2. X4 — `'match'` evaluator → equal-budget. Central fix in `federated_helpers.py`.
3. X5 — `'match'` → 'ridge' in fed02. Thread `model=` through.
4. X3 — hybrid feature-selection leakage. Per-fold rebuild in 4 OPIG experiments.
5. X2 — kirby_hybrid dropping augmentation features. 3 OPIG experiments.

**Fix next (experiment-specific P0/P1):**
6. fed05 C7/C8 cross-silo predict() + label-scale.
7. fed03 missing import + cross-condition variable leak.
8. opig_ginkgo_developability double-prefix dataset_name.
9. opig_humanness LLR all-zero vectors.
10. opig_trastuzumab_enrichment LLR aggregation.
11. opig_kingsmith_component_splits LOO mismatch.
12. imrie_transfer_learning DNN scaler inconsistency.

**Policy:**
13. X6 — single 80/20 split → 5-fold CV across federated.

**Cleanup:**
14. X7 — adaptive_pruning=0.25 ignored.
15. fed01 Phase 4 noise pattern + Lhasa precompute skip.

---

# Noise Experiments — Bug Audit (2026-05-25)

One Opus agent per experiment file. Findings quoted from agent reports — verify each `file:line` before fixing.

## Suite-wide patterns (read first)

Recurring bug classes that span multiple noise experiments:

1. **Matched-evaluator rule violated.** Blueprint built for one downstream model (often `'rf'`) is reused across all other downstream models. The "`best_overall` REMOVED, matched evaluators enforced" rule is broken in n01, n02, n07, n13.
2. **`bp.augmentation_strategy` trusted instead of `meta['aug_condition']`.** Documented 2026-04-21 cache-collision pattern. Affects n07, n10, n13.
3. **AUROC computed on hard labels rather than `predict_proba`.** Confirmed in n02; same pattern likely in n01/n07/n09 binary-clf paths.
4. **Single `'rf'` downstream where suite convention is `DOWNSTREAM_MODELS_NOISE`** (6 models). Affects n02, n04, n07.
5. **Bare `except Exception: pass` swallowing real failures.** Affects n04, n07, n11, n13. Combined with sentinel defaults this produces fabricated headline numbers.
6. **Budget passed but ignored, or budget mismatch with `FULL_BUDGETS`.** n06 + n09 ignore `budget` kwarg; n10 requests `budget=200` which is not in `FULL_BUDGETS=[10,25,50,100,250,...]`, pool silently falls back, result mislabelled `hybrid_200`.
7. **Hardcoded `seed=42` with no multi-seed averaging.** Point estimates with no variance across noise realisations. Affects n04, n05, n07, n08, n11, n13.
8. **PCA/hybrid fit on full data including val rows, or on clean labels then evaluated under noise injection** — transductive/oracle leakage. n03, n09.
9. **No-op or near no-op transformations.** `_weight_features` in n12 is mathematically cancelled by the downstream `StandardScaler`. n09's weighted hybrid normalises `w` to sum=1, making layer-1 inputs O(1e-3).
10. **Dead config knobs.** `--budget`, `--mode`, `n_folds`, `augmentation_strategy` parameters accepted but never wired through. n02, n06, n08, n09.

---

## n01 — Activity Cliff Framework (`n01_activity_cliff_framework.py`)

### P0 — KNN-built hybrid evaluated with RF on the primary metric
**Location:** `n01_activity_cliff_framework.py:596-601` and `:370-372`
**Bug:** Pool-best hybrid is selected with `downstream_model='knn'`, but every cliff-stratified prediction (sub-analysis 4) uses an RF model.
**Evidence:**
```python
# L598 build hybrid
            downstream_model='knn', budget=None,
# L370 downstream eval
    model_factory = get_model_factories(
        ['rf'], task_type=data['task_type'],
        n_samples=len(labels))['rf']
```
**Why wrong:** The hybrid blueprint was optimised for kNN. Scoring with RF reports performance for a hybrid never optimised for RF. The headline `cliff_noncliff_gap` is reported against a mismatched configuration. Violates the matched-evaluator rule.
**Fix:** Set `downstream_model='rf'`, or build separate hybrids per analysis with matched evaluators.

### P1 — `augmentation_strategy='uncapped'` hardcoded, `aug_condition` from pool ignored
**Location:** `:671-678`
**Bug:** `_run_cliff_stratified` is called with `augmentation_strategy='uncapped'` regardless of which aug-condition the pool selected. Inside the function the parameter is also never referenced. Classic 2026-04-21 collision pattern.
**Fix:** Pull `aug_condition` from `meta` and pass it down; gate `train_augs`/`val_augs` on `aug_condition != 'none'`.

### P1 — Sub-analysis 4 ignores the model set; only RF tested
**Location:** `:369-372`
**Bug:** Only RF is run. Sibling experiments iterate `DOWNSTREAM_MODELS_NOISE`.
**Fix:** Loop over `DOWNSTREAM_MODELS_NOISE`, matched-evaluator hybrid per model.

### P1 — `_run_consensus` / `_run_sali_smoothing` / `_run_cliff_mitigation` each recompute the same distance matrix
**Location:** lines 210, 263, 307, 642 (per-rep); 271, 320, 632 (hybrid)
**Bug:** `compute_distance_matrix(X_rep, standardize=True)` called 4× per rep with identical args.
**Fix:** Compute once per rep, dict by rep_name.

### P1 — Cliff identification leaks val-fold labels into ground truth
**Location:** `:381-416` vs `:442-471`
**Bug:** `cliff_indices` computed on full `labels` before the CV loop. Cliff membership of a val compound depends on its delta to other val compounds.
**Fix:** Document Bajorath convention explicitly, or recompute cliff_mask per fold from train-only similarities.

### P2 — Antibody branch reads keys from `cliff_info` that may not exist
**Location:** `:384-391`, `:419-421`

### P2 — `_run_cliff_stratified` writes `fold_domain_all` but never reads it (dead)
**Location:** `:459-460`

### P2 — `sim_threshold=0.55` (Tanimoto-tuned) applied to Euclidean-derived similarities for non-molecular domains
**Location:** `:87-96, 573, 411-417, 734`

---

## n02 — Sample Weighting (`n02_sample_weighting.py`)

### P1 — `top3_base_reps` reflects only the LAST model's top-3
**Location:** `:425, 448`
**Bug:** `top3` overwritten on each iteration of the model loop. Final saved `results['top3_base_reps']` is ridge's top-3 (last key in `WEIGHTED_MODELS`), not a meaningful aggregate.

### P1 — AUROC fed hard predictions, not probabilities
**Location:** `:184, 212, 299, 326, 381`
**Bug:** For binary-clf with `metric='auc'`, calls `model.predict(X_val)` (returns 0/1) and feeds those to AUROC. `_safe_roc_auc` expects probability-like scores.
**Fix:** Use `predict_proba(X_val)[:,1]` when `metric_name == 'auc'`. **Suite-wide:** same pattern likely in n01/n07/n09.

### P2 — Docstring lists 4 conditions, `hybrid_cliff_aware` is a hidden 5th
**Location:** `:1-27, 458-461`

### P2 — Feature importance computed on unscaled X, reliability uses scaled X
**Location:** `:175-199, 289-313`

### P2 — Dead `augmentation_strategy = 'uncapped'` next to a fragile caching API
**Location:** `:142`

### P2 — Blueprint selected for `'rf'`, reused for xgboost/lightgbm/ridge
**Location:** `:243-259`
**Bug:** Matched-evaluator violation.

---

## n03 — SAS Landscape (`n03_sas_landscape.py`)

### P1 — Pool uses `downstream_model='gp'`, but GP has `_max_samples=1000` → hybrid_50d silently dropped on larger datasets
**Location:** `:412-423`
**Bug:** For datasets >1k samples, GP is excluded from `best_by_model_budget` → `get_fitted_pool_hybrid` raises → broad `except Exception` swallows it as a warning → hybrid_50d missing from sub-analysis 4. The central GP-cliff claim cannot be tested on the datasets that motivate the experiment.
**Fix:** Use a guaranteed-present model (`'rf'`), or fall back when GP absent. Don't swallow `Exception`.

### P2 — Sub-2 builds RF-optimised hybrid; sub-4 builds GP-optimised hybrid; conclusion conflates them
**Location:** `:553-566` vs `:412-420`

### P2 — PCA / hybrid fit on FULL X before train/val split (transductive leakage)
**Location:** `:391-409, 449`

### P2 — `get_hybrid_features_and_weights` doesn't pass `augmentation_strategy` when augs present
**Location:** `:553-566`

### P2 — Random projection scaling: `1/sqrt(50)` even when `k = min(50, d) < 50`
**Location:** `:392-394, 427-428`

### P2 — Hstack of mixed binary + continuous features then StandardScaler
**Location:** `:392, 396`

### P2 — `successful = {k: v ... if v['gp_success']}` will KeyError on future entries lacking that key
**Location:** `:497-506`

---

## n04 — GP-LOO Detection (`n04_gp_loo_detection.py`)

### P0 — Headline hybrid condition silently fails: GP not in precompute model set
**Location:** `:155-200, 368-374`
**Bug:** Precompute uses `DOWNSTREAM_MODELS_NOISE` which excludes `'gp'`. `get_fitted_pool_hybrid(..., downstream_model='gp', ...)` therefore returns no candidate → `RuntimeError` → caught by `try/except` with just a print. The hybrid space is never created.
**Fix:** Either include `'gp'` in noise precompute, or change `downstream_model` to something in `DOWNSTREAM_MODELS_NOISE`. The hybrid is just a feature space here — its downstream choice doesn't have to be GP.

### P0 — Missing-hybrid summary defaults to `cliff_fp_rate=1.0` sentinel
**Location:** `:345-362`
**Bug:** `h_cfp = level_res.get('hybrid', {}).get('cliff_fp_rate', 1.0)`. `1.0` is a valid value, not a sentinel. The win-rate is fabricated whenever hybrid didn't run.
**Fix:** Only emit win-rate when hybrid actually ran; otherwise `hybrid_unavailable: True`.

### P0 — Both example datasets in the docstring (`logd`, `flab_thermo_tresanco`) are exactly the cases where GP is unavailable
**Location:** `:25-26`

### P1 — Docstring claims "PCA-50 (dimension-matched)" but code uses `min(budget, total_dim, len(labels)-1)` with default `budget=100` → PCA-100
**Location:** `:19` vs `:205-208`

### P1 — Single hardcoded `seed=42` for noise; no multi-seed averaging
**Location:** `:139-152, 235-237`

### P1 — `cross_val_predict(rf, ..., cv=5)` is random KFold, not group-aware
**Location:** `:39, 322`
**Bug:** Violates "NO random splits anywhere; all splits must be scaffold-group-aware."
**Fix:** `GroupKFold(n_splits=5)` with `groups=groups[sub_idx]`.

### P2 — Cliffs defined on concat-Euclidean, then concat is one of the conditions under test
**Location:** `:65-78`

### P2 — Noise injection unbounded Gaussian at fixed `0.3 * y_range`
**Location:** `:152`

### P2 — GP-LOO refits kernel hyperparameters per noise fraction without making that explicit
**Location:** `:248-254`

### P2 — `top3` computed twice
**Location:** `:196, 223`

### P2 — `get_domain_metric` unpacked but never used
**Location:** `:172`

---

## n05 — Noise Injection Benchmark (`n05_noise_injection_benchmark.py`)

### P1 — Per-rep results misleading for y-only methods
**Location:** `:206-230`
**Bug:** `zscore` only uses `y`. Running it once per rep produces 6 identical F1s reported as if per-rep, and "best space" picks an arbitrary winner from those ties.
**Fix:** Skip rep-independent methods in the per-rep loop.

### P1 — "Best method" summary collapses across sigmas
**Location:** `:252-262`
**Bug:** Records `best_method` and `best_f1` per space but not `best_sigma`. High F1 at σ=0.05 (easy) beats moderate F1 at σ=0.3 (hard).

### P2 — Docstring says "all 15 binary detectors" but `ALL_METHODS` has 16
**Location:** `:2-4, 37-44`

### P2 — Double standardisation
**Location:** `:92, 95, 110, 122-123, 183`

### P2 — Dead imports / dead `get_domain_metric` unpacking
**Location:** `:20-21, 17, 86`

### P2 — `extract_top3_base_reps` called twice
**Location:** `:112, 129`

### P2 — Regression ground truth `np.abs(noise) > MAD(y)` is sigma-dependent
**Location:** `:64-70`
**Bug:** At low σ, n_noisy → 0 (flagging nothing scores perfectly). At high σ, n_noisy → N. Classification path sets `noise_mask[flip_idx] = True`; regression should do the same.

### P2 — `per_rep_D` computed on raw X, `per_rep_Xs` computed on capped X
**Location:** `:121-123`

### P3 — Same seed across sigmas → flip_idx for high σ is a superset of low σ
**Location:** `:47, 140`

---

## n06 — BindingDB Ground Truth (`n06_bindingdb_ground_truth.py`)

### P0 — `groups` are protein IDs, NOT replicates
**Location:** `:54-71`
**Bug:** `data['groups']` from `assign_cold_target_groups(metadata['protein_ids'])` groups by protein for cold-target CV. The experiment treats `np.std(labels[mask])` as inter-replicate noise — it's actually the spread of Kd across different drugs binding the same protein. The entire ground truth premise is wrong.
**Fix:** Custom BindingDB loader that retains per-pair replicates and exposes them as `data['replicate_groups']`, or drop this experiment. **Decision pending — user pushing back on these options.**

### P0 — `n_with_replicates < 10` "fallback" silently runs without ground truth
**Location:** `:73-78`
**Bug:** Print promises a synthetic-noise fallback; code provides none.

### P1 — Spearman correlation includes samples with `replicate_std == 0`
**Location:** `:62-68, 143-145`
**Bug:** `has_replicate = replicate_stds > 0` is computed but never applied.

### P1 — `GP_MAX=1000` subsample applied to ALL estimators, not just GP
**Location:** `:42-43, 107-127`

### P2 — `budget = kwargs.get('budget', 100)` is dead
**Location:** `:80, 91-98`

### P2 — `top3` extracted twice
**Location:** `:100, 113`

### P2 — GP subsample uses hardcoded `RandomState(42)` regardless of experiment seed
**Location:** `:108`

### P2 — Hybrid built for `'rf'` then scored against arbitrary noise estimators
**Location:** `:91-98, 100`

---

## n07 — Remediation (`n07_remediation.py`)

### P1 — Only `'rf'` downstream model
**Location:** `:163-165`

### P1 — `robust_loss` is HuberRegressor while every other strategy uses RF
**Location:** `:122-128`
**Bug:** Recovery `(strategy - noisy_baseline)/(clean - noisy_baseline)` where both baselines are RF. Huber row mixes model-choice with remediation strategy. **Decision pending.**

### P1 — `_evaluate_fold` recomputes noise scores per `(rep, strategy, fold)` when they only depend on `(rep, fold)`
**Location:** `:254-276`
**Bug:** `ActivityCliffAwareEstimator.fit` runs 4× per (rep, fold). ~5× slowdown.

### P1 — `best_strategy` is `max(recovery, ...)` across keys with incomparable denominators
**Location:** `:311-313`
**Bug:** Per-rep recovery percentages have different reference gaps. A 200% per-rep recovery (tiny noisy denominator) silently beats a stable 60% hybrid recovery.

### P2 — `bp.augmentation_strategy` trusted instead of `meta['aug_condition']`
**Location:** `:181-189, 194`

### P2 — Single hardcoded noise `seed=42`
**Location:** `:33-51`

### P2 — Bare `except Exception` on estimator failure → `noise_scores = zeros` → label_smooth/knn_smooth flag ALL samples
**Location:** `:69-74`

### P2 — `label_smooth` and `knn_smooth` collapse to the same operation under classification
**Location:** `:97-111`

### P2 — Hybrid passed through `StandardScaler` before a `metric='cosine'` cliff estimator
**Location:** `:181-189`

### P3 — Docstring says "6 remediation strategies"; code has 5

---

## n08 — Estimation Survey (`n08_estimation_survey.py`)

### P0 — "Best cliff discriminator" uses signed rho, not `abs(rho)`
**Location:** `:186, 229-236`
**Bug:** `cliff_discriminates = rho_cliff < 0.1` and `if rho < best_cliff_rho` — a strongly negatively-correlated estimator (rates cliffs as cleanest) is reported as "best discriminator". Stated goal is `|rho| ≈ 0`.
**Fix:** Use `abs(rho)`; initialise `best_cliff_abs_rho = float('inf')`.

### P1 — Docstring promises "downstream impact via sample weighting"; never implemented
**Location:** `:5-7, 21-22, 78`
**Bug:** `RandomForestRegressor`, `r2_score`, `metric_fn`, `lower_is_better` imported/unpacked then never referenced.

### P1 — `n_folds`, `mode` computed then never used; `--mode intermediary` has no effect
**Location:** `:74-75`

### P2 — Cross-estimator correlation aggregation uses substring matching on dict keys
**Location:** `:217-219`

### P2 — `compute_distance_matrix(X_concat, standardize=True)` re-standardises a separate copy of an already-z-scored matrix
**Location:** `:83-86, 112`

---

## n09 — LNL Survey (`n09_lnl_survey.py`)

### P0 — Classification path broken end-to-end
**Location:** `:98-100, 67-73`
**Bug:** Filters `heteroscedastic_gp` as regression-only but misses `evidential_kernel` (also regression-only, NIG loss). More importantly, `_run_baseline` and every LNL method use `_standard_dnn_train_predict`, which is hard-coded for regression (MSE, z-normalise on y, no softmax head). For `is_clf=True`, the experiment z-normalises 0/1 labels, fits MSE regression, calls F1w on continuous predictions.
**Fix:** Either gate to regression-only, or wire a classification head through every method. **Decision pending — user suggesting sample-removal conversion path.**

### P1 — `budget=100` recorded in results is fictitious
**Location:** `:94, 156`

### P1 — R² retention gives nonsense (sign flips, unbounded ratios)
**Location:** `:381-411`
**Bug:** `_retention = noisy/clean * 100`. With `clean=-0.1, noisy=-0.5`, retention = 500%. With `clean=0.5, noisy=-0.5`, retention = -100%.

### P1 — Blueprint chosen on CLEAN labels; docstring claims "end-to-end noise resilience"
**Location:** `:106-122, 196-207`
**Bug:** Pool blueprint fetched from precomputed searches that ran on clean labels. Real noisy-label practitioners never see clean labels.

### P2 — "Top-3 base reps (by RF)" print, but blueprint may be DNN-optimised
**Location:** `:124-126, 157`

### P2 — Dead `augmentation_strategy = 'uncapped'`
**Location:** `:86`

### P2 — Weighted hybrid normalises `w` to sum=1 → layer-1 inputs scaled to O(1e-3)
**Location:** `:246-249`

### P2 — `n_splits=1` (intermediary mode) is a single 80/20 GroupShuffleSplit
**Location:** `:88, 175-177`

---

## n10 — Hybrid Neighborhood LNL (`n10_hybrid_neighborhood_lnl.py`)

### P1 — `budget=200` not in `FULL_BUDGETS`; pool silently falls back, blueprint labelled `hybrid_200` uses different budget
**Location:** `:413`
**Bug:** `hybrid_budgets = [25, 100, 200]` but `FULL_BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000]`. `get_best_for_downstream` falls back to model-best across budgets when 200 is absent. Q4 "budget effect" silently uses a non-200 budget masquerading as 200.
**Fix:** Change to `[25, 100, 250]`. Assert `bp.budget == budget` after retrieval.

### P2 — Docstring says "6 neighbor space conditions"; code adds per-rep spaces silently → 10–20+
**Location:** `:14-16, 102-204`

### P2 — Hardcoded `metric='euclidean'`, `k=10` for all LNL estimators
**Location:** `:55-57`

### P2 — `y_noisy` passed as `y_train` into `_build_neighbor_spaces` without rename
**Location:** `:102-204`

### P2 — Q3 "hybrid rescues" compares COUNTS of positive recoveries instead of averages
**Location:** `:354-368`

### P2 — `mad` recomputed per (fold, sigma) instead of per fold
**Location:** `:518-519`

### P2 — Sigma loop is outside the fold loop
**Location:** `:472-492`

---

## n11 — Estimation × Remediation Grid (`n11_estimation_remediation_grid.py`)

### P1 — Smoothing silently broken for classification datasets
**Location:** `:213-226`
**Bug:** `smooth_by_scores` outputs continuous floats; `RandomForestClassifier.fit(X, y_smoothed)` raises `ValueError: Unknown label type: continuous`, swallowed by bare `except Exception: pass`. Half the grid silently produces no results for clf datasets.
**Fix:** Round/clip for clf, or explicitly skip smoothing when `is_clf` and emit `smoothing_skipped_clf=True`.

### P2 — `n_samples = len(labels)` (full dataset) passed to `_run_estimator_grid` but GP cap check uses fold-train size; the guarded branch is a `pass` (dead)
**Location:** `:246, 369, 186`

### P2 — `n_total` config count inflated by duplicate base rep space
**Location:** `:126-149`
**Bug:** Base rep added twice (`{base_rep_name}_raw` then via per-rep loop). Gives it two shots at being "best space".

### P2 — `denoised_score` in `top_10` doesn't carry direction
**Location:** `:407-409`

### P2 — `best_recovery = -inf` with `best_config = None` if all recoveries are NaN; print silently skipped
**Location:** `:419-436`

### P2 — Hardcoded `'hybrid_100'` key; actual hybrid uses `bp.budget`
**Location:** `:142`

### P2 — Bare `except Exception: pass` × 4 hides real failures
**Location:** `:143-144, 193-194, 209-210, 225-226`

### P3 — `leave_one_out` estimator is O(n) refits per config
**Location:** `:50-55`

### P3 — `_get_correction_target` returns first attribute found in fixed order
**Location:** `:89-99`

---

## n12 — Importance-Weighted Distance (`n12_importance_weighted_distance.py`)

### P0 — `_weight_features` is a mathematical no-op
**Location:** `:75-79`, with `:115-116, 147-148, 161-162`
**Bug:** Per-feature multiplication by a positive scalar `w_j` is exactly cancelled by the subsequent per-column `StandardScaler`. After `X * w_j` → `StandardScaler.fit_transform` → `(w_j·X_j − w_j·μ_j) / (w_j·σ_j) = (X_j − μ_j) / σ_j`. Euclidean distances used by noise estimators are identical between `base_unweighted` and `base_imp_weighted`. The entire central ablation question is unanswerable.
**Fix:** Standardise FIRST, then multiply by `sqrt(importance)`; do NOT scale again.

### P1 — `create_hybrid` called with ALL reps, ignoring `bp.reps`
**Location:** `:121-129`
**Bug:** Blueprint's allocation/importance/evaluator/budget/aug_strategy reused, but the rep pool is not subset to `bp.reps`. If extra reps are present, allocation may rank them in and the hybrid differs from the one the pool scored.
**Fix:** `train_reps_for_bp = {r: train_reps[r] for r in bp.reps if r in train_reps}`.

### P1 — Per-fold hybrid and `imp_hybrid_via_blueprint` recomputed once per `imp_method`
**Location:** `:303-306, 121-129, 143-145`
**Bug:** Loop body double-counts the unweighted hybrid scores across imp_methods.

### P2 — `bp.augmentation_strategy` may default to `'uncapped'` even when no augs
**Location:** `:127-128`

### P2 — `K_VALUES`, `SIGMAS` not aligned with N09–N11

### P2 — Classification path uses regression-style neighbor residuals on integer labels
**Location:** `:289` + `noise_estimation.py:125-303`

---

## n13 — Noise Distributions (`n13_noise_distributions.py`)

### P0 — Undefined `top3` reference (suite-wide pattern)

### P0 — Missing `val_augs` in Track A hybrid → hybrid Track A silently skipped on every augmentation dataset
**Bug:** Augmentations passed to train but not val; `build_fold_hybrid` rejects the asymmetric call and the exception is swallowed.
**Fix:** Pass `augmentations_val=val_augs`.

### P1 — Cross-space "best" mix in summary
**Bug:** Best-config selection pools spaces with different denominators.

### P1 — Track B trains on base rep only
**Bug:** Recovery formula compares to a hybrid clean baseline but Track B retrain uses base only.

### P1 — Blueprint `aug_strategy` collision (suite pattern)

### P2 — Classification "partial" semantics for label-flipping
### P2 — RF-only retention baseline (matched-evaluator)
### P2 — Cross-strength seed reuse
### P2 — Dead `gap` variable
### P2 — Per-strength inflation in `hybrid_advantage_persists`

### P2 — SVM `_max_samples=10000` cap silently triggers on large datasets (QM9 etc.)
**Location:** `:181-184, 320-325, 394-407`
**Bug:** Only SVM's training is truncated; cross-model comparison becomes apples-to-oranges.

---

## Noise — Headline triage

1. **n09 P0** (classification end-to-end broken — gate vs. sample-removal conversion: decision pending)
2. **n12 P0** (no-op feature weighting — invalidates the entire experiment)
3. **n06 P0** (wrong ground truth — decision pending, BindingDB used widely elsewhere but n06 is the only noise consumer)
4. **n04 P0×2** (hybrid silently absent + sentinel = 1.0 → fabricated headline)
5. **n01 P0** (KNN/RF mismatch on the primary metric)
6. **n08 P0** (signed-rho cliff selection)
7. **n13 P0×2** (undefined `top3`, missing `val_augs`)
8. **n11 P1** (clf smoothing silently dead)
9. **n10 P1** (`hybrid_200` mislabelled)
10. Suite-wide AUROC-on-labels fix (n02 + audit n01/n07/n09 binary-clf paths)
11. Suite-wide matched-evaluator enforcement (n01, n02, n07, n13)
12. Suite-wide `bp.augmentation_strategy` → `meta['aug_condition']` swap (n07, n10, n13)

---

# Complexity experiments audit (2026-05-25)

25 parallel Sonnet agents — one per file (c01–c19, c21–c24, plus `complexity_helpers.py`).
Cross-cutting bugs in `complexity_helpers.py` propagate to every consumer; fix those first.

## complexity_helpers — Shared Configuration (`complexity_helpers.py`) — 2 P0, 3 P1, 3 P2

### P0 — Augmentation loading: mixed-presence branch silently drops already-loaded reps
**Location:** `:1083–1092`
**Bug:** When `aug_rep_names` contains a mix of reps already in `data['reps']` and reps that are not, the `if missing:` branch builds `aug_reps` only from the newly-loaded ones. Already-loaded aug candidates (e.g. `onehot_components`, `diff_maccs` for `bh_doyle_plates`) are silently dropped from `data['augmentations']`. The pool has fewer aug reps than DATASET_REPS declares.
**Fix:** After building `aug_reps` from `aug_data`, merge in already-loaded names: `for rn in aug_rep_names: if rn in reps and rn not in aug_reps: aug_reps[rn] = reps[rn]`.

### P0 — `codon_usage` double-included as base AND augmentation on `gb_demo_human_or_worm`
**Location:** `:1094` (the `else` branch) + DATASET_REPS for `gb_demo_human_or_worm`
**Bug:** `codon_usage` lives in both `general` and `augmentation`. After load_data it is in `reps`. The `else` branch builds `data['augmentations'] = {'codon_usage': ...}` but never removes it from `data['reps']`, so find_best_hybrid receives it in both base_reps and augmentations — polluted cache key, corrupted search.
**Fix:** Remove `codon_usage` from `gb_demo_human_or_worm['general']`, and have the `else` branch also pop aug names out of `reps`.

### P1 — `AUG_GROUPS['dta']` uses wrong rep-name prefixes (c21 silently empty on DTA)
**Location:** `:881–882`
**Bug:** Group lists `prot_esm2`, `prot_prost`, `prot_t5`, `prot_aac` but the actual rep names are `protein_esm2`, `protein_saprot`, etc. (`protein_` prefix). `interaction` lists `mordred` (actual: `drug_mordred`) and `drug_sns` (not in any DTA general). `get_aug_groups` filters to available reps → empty. c21 runs only A_baseline/B_all on DTA.
**Fix:** Rewrite with actual rep names from DATASET_REPS (`protein_esm2`, `protein_saprot`, `protein_esmc`, `protein_esm2_midlayers`, etc.).

### P1 — `AUG_GROUPS['tcga']` is entirely stale — c21 gets zero domain groups on TCGA
**Location:** `:884–891`
**Bug:** Every TCGA rep name listed is wrong (`rnaseq_pca50`, `rnaseq_hallmark`, `mutation_binary`, `methylation_pca50`, `cnv_arm`, etc.). Actual TCGA reps: `gene_geneformer_v2`, `gene_bulkrnabert`, `mutation_matrix`, `mut_pca50`, `methyl_pca50`, `cnv_pca50`, `wsi_titan`. 5 of 6 groups empty after filtering.
**Fix:** Replace with actual TCGA DATASET_REPS rep names.

### P1 — `c12` missing from `_experiment_full_name` → TCGA c12 results saved to wrong path
**Location:** `:1248–1272`
**Bug:** `_import_experiment` maps `c12` → `c12_modality_ablation`. `run_all` auto-inserts `c12` for tcga_* datasets. But `_experiment_full_name` has no entry, so save path is `output_dir/c12/...` not `output_dir/c12_modality_ablation/...`. Any consumer reading via `load_experiment_results(..., 'c12_modality_ablation', ds)` finds nothing.
**Fix:** Add `'c12': 'c12_modality_ablation'` to the mapping.

### P2 — `bh_doyle_plates`: `onehot_components` and `diff_maccs` in both `general` and `augmentation`
**Location:** `:603–612`
**Fix:** Remove the two reps from `augmentation` (they are already base reps).

### P2 — `c07` hardcodes `augmentation_strategy='capped'` on general-tier subpopulations
**Location:** `c07_data_complexity_partitions.py :178`
**Fix:** Use `'uncapped' if sub_augs else 'none'`.

### P2 — Stale module docstring
**Location:** `:8`
**Bug:** "Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search" — actual is 5-fold, 6 models from DOWNSTREAM_MODELS, FULL_BUDGETS.
**Fix:** Update header.

---

## c01 — Rep Characterization (`c01_rep_characterization.py`) — 1 P0, 2 P1, 2 P2

### P0 — `args.reps` passed as `rep_preset` instead of `rep_names`
**Location:** `:124`
**Bug:** `load_data(args.dataset, args.reps, ...)` passes `args.reps` as the second positional (`rep_preset`). When the user passes `--reps esm2,prot_t5`, it lands as `rep_preset='esm2,prot_t5'` which doesn't match any preset; auto-filter then reads DATASET_REPS['general'] anyway, ignoring `--reps` entirely. Every other CLI uses `parse_rep_args(args)` first. Standalone runs silently load the wrong rep set.
**Fix:** `rep_preset, rep_names = parse_rep_args(args)` then `load_data(args.dataset, rep_preset, rep_names=rep_names, ...)`.

### P1 — `lower_is_better` fetched but never used — summary direction-agnostic, missing from JSON
**Location:** `:46`, `:99–109`, `:111`
**Fix:** Sort summary table by direction; include `lower_is_better` + `metric_name` in the returned JSON.

### P1 — Double PCA reduction corrupts MI estimates on small datasets
**Location:** `:63–66`
**Bug:** c01 PCA-reduces X then passes to `estimate_mutual_information_knn`, which re-standardises and applies its own PCA. For small antibody datasets (n<250) the outer cap differs from the inner cap and the inner StandardScaler sees already-whitened PCA scores — KSG neighbour distances distorted.
**Fix:** Remove the outer PCA; pass raw X.

### P2 — Falsy check `if dim_info.get('mle'):` silently hides MLE=0.0
**Location:** `:57`, `:102`
**Fix:** Use `is not None`.

### P2 — `lower_is_better` / `metric_name` absent from JSON output
**Location:** `:111`
**Fix:** Add `_meta` block.

---

## c02 — Complexity Grid (`c02_complexity_grid.py`) — 2 P0, 3 P1, 4 P2

### P0 — PCA baseline uses `m.predict()` for AUC/F1w classification metrics
**Location:** `:385`
**Bug:** `metric_fn` for classification is `_safe_roc_auc` / F1w — requires probabilities. `m.predict()` returns class labels. Every PCA baseline row on a classification dataset is silently wrong.
**Fix:** Use `_predict_for_metric(m, X_val_p, metric_fn)` helper from find_best_hybrid.

### P0 — `dta_protein_lora` factory type silently falls through `_build_rep_factories`, empty FT search
**Location:** `:189–223` (c02) and `_shared.py:1479–1480`
**Bug:** `_build_rep_factories` handles `molecular_lora`/`molecular_contrastive`/`molecular_gnn`/`antibody_lora` only; `dta_protein_lora` hits the `else` branch with just a warning. For `kiba`/`bindingdb_kd` the entire FT analysis is silently dropped.
**Fix:** Add `dta_protein_lora` branch in `_build_rep_factories`.

### P1 — PCA baseline budget list diverges from hybrid search budget list when called via `run_all`
**Location:** `:44, :57, :365`
**Fix:** Use `full_budgets` (the 8-budget list) for the PCA loop, not `config['budgets']` (the 4-budget screening list).

### P1 — `delta` sign uninformative for lower-is-better metrics
**Location:** `:348, :412`
**Fix:** Flip the sign for lower_is_better OR suffix print with `↓ better` / `↑ better`.

### P1 — Per-model-top-rep search passes pre-instantiated model dict; never hits precompute cache
**Location:** `:62, :327`
**Fix:** Document explicitly that the per-model pass bypasses search cache; only fold-level cache hits.

### P2 — `model_instances` for gp/knn not gated by `_max_samples` if skip set ever changes
**Location:** `:62, :306–327`
**Fix:** Pass `models=[_model_name]` (list) not dict, so `_resolve_models` re-applies `_max_samples`.

### P2 — `ft_id_parts` uses bare subscript `DATASET_REPS[dataset_name]`
**Location:** `:196`
**Fix:** Use `.get(...)` with safe defaults.

### P2 — Heatmap col-set (`full_budgets`) and PCA baseline key-set (`budgets`) diverge
**Location:** `:422–431`
**Fix:** Resolved by the P1 budget-list fix above.

---

## c03 — Cross-Dataset Transfer (`c03_cross_dataset_transfer.py`) — 1 P0, 3 P1, 4 P2

### P0 — Does not implement "blueprint transfer" at all
**Location:** `:2-12`, `:76-177`
**Bug:** File name + docstring claim cross-dataset blueprint transfer. Actual code runs `get_best_search_for_downstream` independently per dataset and compares optimal configs side-by-side. Never applies source's blueprint to target. Stated conclusion ("optimal configs transfer") is not supported by the code.
**Fix:** For each (source, target) pair: load source blueprint, filter to reps that exist in target, call `build_fold_hybrid` with source blueprint on target data, compute transfer gap vs target's native optimal.

### P1 — `create_hybrid` called with all train reps, ignoring `bp.reps`
**Location:** `:126-134`
**Fix:** Filter `train_reps`/`val_reps` to `{k: v for k, v in train_reps.items() if k in bp.reps}` before calling create_hybrid.

### P1 — `bp.augmentation_strategy` used instead of `meta['aug_condition']` (documented stale-cache bug)
**Location:** `:107`, `:133`
**Fix:** `aug_strat = meta['aug_condition']`; pass that.

### P1 — Novel models (SVR, ElasticNet) silently fail on classification datasets, exception swallowed
**Location:** `:77-154`
**Fix:** Guard with `if data['task_type'] == 'regression':` or substitute classifier equivalents.

### P2 — `create_hybrid` called `n_novel × n_folds` times; feature_info recomputed identically per novel model
**Location:** `:113-149`
**Fix:** Hoist fold loop above novel-model loop; reuse `fi`/`X_train`.

### P2 — Complexity metrics computed on full-rep concatenation; cross-dataset comparison apples-to-oranges
**Location:** `:65`
**Fix:** Use a fixed common rep set (intersection or top_tier).

### P2 — Variance decomposition can produce negative interaction percentage
**Location:** `:294-310`
**Fix:** Clamp `ss_interaction = max(0, ss_interaction)` or rename to `pct_residual`.

### P2 — `DISTANCE_BUDGETS`/`DISTANCE_EVALUATORS` imported but never used
**Location:** `:31`
**Fix:** Remove imports.

---

## c04 — Importance as Metric (`c04_importance_as_metric.py`) — 0 P0, 1 P1, 4 P2

### P1 — Importance scores not cross-calibrated; concatenated weights on incomparable scales
**Location:** `:62–66`, `:152–159`, `:203–206`
**Bug:** `compute_feature_importance` trains one model per rep independently — importance magnitudes are within-rep scaled. Concatenating per-rep vectors treats them as cross-calibrated weights. The kNN and similarity retrieval are scientifically invalid.
**Fix:** Compute importance once on the full concatenated `X_train_concat`, not per-rep.

### P2 — kNN R² (per-fold mean) vs similarity R² (pooled) — different estimators
**Location:** `:110–130`
**Fix:** Per-fold mean for both.

### P2 — Cross-test calls `compute_feature_importance` twice per fold for diagonal entries
**Location:** `:151–158`
**Fix:** Precompute importance dict per fold; reuse across cells.

### P2 — `_evaluate_low_dim` uses pooled R²; mismatched with `_evaluate_knn_and_sim` aggregation
**Location:** `:237–241`
**Fix:** Per-fold accumulation.

### P2 — Cross-test budget hardcoded to `kwargs.get('budget', 25)`, ignores `DISTANCE_BUDGETS`
**Location:** `:267`, `:309–311`
**Fix:** Iterate `DISTANCE_BUDGETS`.

### P3 — `except Exception: pass` silently swallows cross-test failures
**Location:** `:314`
**Fix:** Log failure with key set to None.

### P3 — `has_delta=True, has_augment=True` registered but never used in `run()`
**Location:** `:366`, `:378–379`
**Fix:** Remove from parser and load_data call.

---

## c05 — Learning Curves (`c05_learning_curves.py`) — 2 P0, 4 P1, 3 P2

### P0 — Subsampling not scaffold-aware: scaffold groups shattered across folds
**Location:** `:122–126`
**Bug:** `np.random.choice(n_total, n_sub, replace=False)` picks individual molecules. For scaffold-grouped molecular datasets, this places scaffold-mates into both train and val of the same fold within the subsample.
**Fix:** Subsample at the group level (draw `k` scaffold groups, take all members).

### P0 — `_max_samples` model cap inconsistency across subsample sizes
**Location:** `:39–41, 141–156, 171–188`
**Bug:** `get_model_factories(n_samples=len(data['labels']))` computed once at full N; SVM excluded forever even at small subsamples where it would normally be included. Also: hybrid search re-resolves SVM at small n via `_resolve_models`, but `evaluate_single_rep_cv` uses the pre-filtered factories → asymmetric model set between hybrid and single-rep at small n.
**Fix:** Move `get_model_factories` inside the `n_sub` loop with `n_samples=n_sub`.

### P1 — `augmentation_strategy='capped'` hardcoded for subsample search but `'uncapped'` for full-data blueprint
**Location:** `:148, 91–98`
**Fix:** Force the full-data pool search to `'capped'` blueprints, or document the confound and surface `bp_fd.augmentation_strategy` per point.

### P1 — `np.random.seed(seed)` global, but `find_best_hybrid` uses seed=42 internally
**Location:** `:121–122`
**Fix:** Pass `seed=seed` to `find_best_hybrid` and `cv_split`.

### P1 — Full-data blueprint's `bp.budget` may exceed `n_sub`; silently skips at small-n
**Location:** `:93, 198–201`
**Fix:** Use `build_fold_hybrid(..., budget_override=min(bp_fd_budget, n_sub))` instead of skipping.

### P1 — `evaluate_single_rep_cv` `seed` not forwarded; all repeats use same 80/20 split for baseline
**Location:** `:176–185`
**Fix:** Add and forward `seed=seed`.

### P2 — `fit_learning_curve` hardcoded for [0,1] R² bounds; wrong for AUROC/MCC/Spearman/RMSE
**Location:** `:267, _shared.py:1208–1209`
**Fix:** Accept `lower_is_better` + `metric_name`; rename output key `r2_max` → `score_max`.

### P2 — `last_search` for `top3` extraction is from last seed's run, not the full-data search
**Location:** `:231–234, 258`
**Fix:** Explicitly extract from full-data (n=1.0) search.

### P2 — `sub_budgets = [min(sweep_budgets)]` dead-code path is subtly wrong
**Location:** `:138–139`
**Fix:** Use `[sorted(sweep_budgets)[0]]`.

---

## c06 — Within-Family Gradient (`c06_within_family_gradient.py`) — 0 P0, 3 P1, 3 P2

### P1 — `INTERNAL_EVALUATORS` imported but never passed
**Location:** `:30`, `:142–160`
**Fix:** Add `internal_evaluators=INTERNAL_EVALUATORS` to find_best_hybrid call.

### P1 — Budget mismatch: c06 pre-filters `eff_budgets` but find_best_hybrid injects `n_samples` as at-n boundary
**Location:** `:67–68`, `:171`
**Fix:** Iterate `per_budget.keys()` instead of `eff_budgets`.

### P1 — Importance cache fetched with `aug_condition='none'` even when dataset uses 'uncapped'
**Location:** `:79`, `:113`
**Fix:** Pass `aug_condition=aug_strategy`.

### P2 — Output keys `'val_r2'`/`'r2'` hardcoded regardless of actual domain metric
**Location:** `:180`, `:164`, `:196`, `:205`, `:234`
**Fix:** Use `metric_name` as key.

### P2 — kNN family has no `_max_samples`; runs at full size on QM9/nt_*
**Location:** `:39`, `:85–99`
**Fix:** Add `_max_samples=2000` to kNN DEFAULT_MODELS entry, or dataset-size guard.

### P2 — `dataset_name` mangled to `"{ds}__c06_{fam}_{var}"` breaks `_applied_skips` lookup
**Location:** `:154`
**Fix:** Pass real `dataset_name` + `condition_suffix=f"c06_{fam}_{var}"`.

---

## c07 — Data Complexity Partitions (`c07_data_complexity_partitions.py`) — 0 P0, 3 P1, 6 P2

### P1 — `augmentation_strategy='capped'` hardcoded; fulldata blueprints use 'uncapped'
**Location:** `:178`
**Fix:** `'uncapped' if sub_augs else 'none'`.

### P1 — `target_variance` partition uses full-dataset labels (label leakage into partition definition)
**Location:** `:111–124`
**Fix:** Document the limitation; optionally restrict to training-fold labels.

### P1 — Failed-SMILES molecules silently land in 'easy' partition in `molecular_complexity`
**Location:** `:102–105`
**Fix:** Track `valid_mask`; drop invalid rows from both partitions.

### P2 — NaN score not guarded in best-model selection loop
**Location:** `:194–208`
**Fix:** `if np.isnan(m_best.get('score_mean')): continue`.

### P2 — Singleton-scaffold samples default to var=0; conflated with "easy"
**Location:** `:122–123`
**Fix:** Exclude singleton-scaffold samples or assign NaN.

### P2 — Unused `KFold` import inside inner loop
**Location:** `:227`
**Fix:** Delete.

### P2 — SMILES parsed twice per molecule
**Location:** `:102–103`
**Fix:** Cache `mol = Chem.MolFromSmiles(s)`.

### P2 — Second-pass `cv_split` missing explicit `seed`
**Location:** `:242`
**Fix:** Pass `seed` explicitly.

---

## c08 — Calibration (`c08_calibration.py`) — 2 P0, 3 P1, 3 P2

### P0 — `exp2_results` path drops `internal_evaluator` and `augmentation_strategy`; hardcodes 'ridge' + 'uncapped'
**Location:** `:78–91`
**Fix:** Default to `'match'`; or re-fetch via pool helper.

### P0 — `val_score` from `model` (full-train) but calibration metrics from `model_conf` (proper-train subset) — irreconcilable train-set mismatch
**Location:** `:219–224` and `:236–302`
**Fix:** Compute `val_score` from `model_conf` (or move `val_score` AFTER model_conf is fit) so both come from the same model.

### P1 — `sharpness_ratio` direction backwards (comment says "1/n_classes = trivial, 1 = perfect" — inverted)
**Location:** `:282–283`
**Fix:** Correct comment OR invert formula to `1 - avg_set_size/n_classes`.

### P1 — `optimal_internal_eval = 'ridge'` default; project canonical default is `'match'`
**Location:** `:73`
**Fix:** Default to `'match'`.

### P1 — `underfit` condition hardcodes budget=25 regardless of `total_dim`
**Location:** `:168`
**Fix:** `min(25, max(5, total_dim // 4))`.

### P2 — NaN averaging in fold_metrics silently drops folds with no `predict_proba`
**Location:** `:370–374`
**Fix:** Track `n_valid_cal_folds` separately.

### P2 — Classification `val_score` uses hard labels for AUC metric
**Location:** `:224`
**Fix:** Branch on `metric_name == 'auc'` and use `predict_proba`.

### P2 — Regression summary print crashes on KeyError `a['val_r2']` if no valid folds
**Location:** `:409`
**Fix:** `.get('val_r2', float('nan'))`.

---

## c09 — Formal Analysis (`c09_formal_analysis.py`) — 1 P0, 3 P1, 3 P2

### P0 — Raw `create_hybrid` in 9b/9c; no fold-level caching, and the cached wrapper exists
**Location:** `:29`, `:338`, `:468`
**Fix:** Use `create_hybrid_cached` from `experiments._shared`.

### P1 — Dead c06 fast-path in `_run_bias_variance_analysis`: key mismatch
**Location:** `:534–557`
**Bug:** Checks `'train_r2' in metrics` but c06 only stores `'val_r2'` — condition is always False; c06 speedup is silently dead.
**Fix:** Either add `train_r2` to c06 heatmap, or remove the c06 fast-path.

### P1 — `bias_proxy = 1 - train_score` is R²-specific; broken for AUROC/MCC/RMSE
**Location:** `:546`, `:606`
**Fix:** Domain-aware formula using `lower_is_better`.

### P1 — ANOVA output key `'grand_mean_r2'` wrong for non-R² datasets
**Location:** `:253`, `:165`, `:181`, `:186`, `:191–199`
**Fix:** Rename to `grand_mean_score`.

### P2 — Inflection point off-by-one (reports end-of-plateau budget instead of start)
**Location:** `:388–397`
**Fix:** `inflection_budget = budgets[inflection_idx]` not `[inflection_idx + 1]`.

### P2 — Dead variable `X_concat`
**Location:** `:287`
**Fix:** Delete.

### P2 — Dead variable `model_names`
**Location:** `:564`
**Fix:** Delete.

---

## c10 — Noise Robustness (`c10_noise_robustness.py`) — 2 P0, 3 P1, 3 P2

### P0 — Blueprint budget mismatch: `bp.budget` overwrites the outer loop `budget`
**Location:** `:252–260`, `:356`
**Bug:** Outer loop iterates a budget grid; the blueprint pulled for that loop step carries its own `bp.budget` (from where the precompute thought it was best for the requested model). The fold loop then calls `create_hybrid(budget=bp.budget)` instead of the loop variable — the budget sweep is broken; the heatmap rows are mislabelled.
**Fix:** Pass `budget=budget` (loop variable). Use `bp.allocation_method/importance_method/internal_evaluator` from blueprint but pin budget to the sweep axis.

### P0 — `NoiseInjectorRegression` state carries across folds and sigmas; noise NOT i.i.d. per fold
**Location:** `:170–181`, `:234–244`, `:340–349`
**Fix:** Re-instantiate inside the fold loop with a deterministic per-(sigma, fold) seed.

### P1 — Blueprint may come from cross-model fallback (different model than the evaluator)
**Location:** `:136`
**Fix:** Assert `meta['best_config']['model'] == 'rf'`; warn if fallback fired.

### P1 — Lower-is-better metric variable named `baseline_r2` and printed as `R²=` — confusing for RMSE
**Location:** `:204`, `:222/307`
**Fix:** Rename to `baseline_score`; harden zero-check with `abs(baseline_r2) > 1e-9`.

### P2 — `augmentation_strategy=bp.augmentation_strategy` (known stale-cache bug)
**Location:** `:260`
**Fix:** Store `aug_conditions[budget] = meta['aug_condition']` and use that.

### P2 — `create_hybrid` called directly inside fold loop (expensive, not cached)
**Location:** `:253`, `:354`
**Fix:** Document the cost; reduce defaults if needed.

### P2 — Nogueira SE formula breaks for M=2 folds; NaN written to JSON
**Location:** `:83–84`, `:393–395`
**Fix:** Guard before computing CI.

---

## c11 — Ensemble vs Hybrid (`c11_ensemble_vs_hybrid.py`) — 0 P0, 4 P1, 4 P2

### P1 — `n_comp` stale when folds produce different training sizes or when all folds skip a budget
**Location:** `:245`
**Fix:** Compute `n_comp` per fold; record the last-successful-fold value explicitly.

### P1 — Docstring promises ensemble; code runs `pca_matched` — comparison no longer answers the stated question
**Location:** `:10–16`, `:199`
**Fix:** Either reinstate prediction-averaging ensemble (no leakage) or rename to `c11_pca_vs_hybrid`.

### P1 — `'augmented': False` hardcoded; "with vs without augmented features" condition silently absent
**Location:** `:180`, `:249`, `:308`
**Fix:** Add explicit outer loop over `['none', 'uncapped']`, OR remove `has_augment` and docstring claim.

### P1 — `last_search` reflects only last budget's pool winner; `top3_base_reps` uses wrong search
**Location:** `:93`, `:106`, `:112`, `:378`
**Fix:** Capture `'none'` tier search explicitly, or derive top3 from `single_rep_results`.

### P2 — Print labels say `"R2="` unconditionally; wrong for AUROC/Spearman/CI/MCC datasets
**Location:** `:175`, `:302`, `:331–332`
**Fix:** Use `metric_name`.

### P2 — `n_models_trained=1` for all conditions; dead column post-ensemble removal
**Location:** `:181`, `:247`, `:306`
**Fix:** Drop the field.

### P2 — PCA-matched excludes augmentations but hybrid includes them; asymmetric information
**Location:** `:209–210`
**Fix:** Either include augmentations in `X_train_cat` or document the confound.

### P2 — `mode` mapping from `n_splits` is fragile
**Location:** `:63–64`
**Fix:** Document.

---

## c12 — Modality Ablation (`c12_modality_ablation.py`) — 0 P0, 2 P1, 4 P2

### P1 — `gene_geneformer_v2` silently dropped from ablation (not in MODALITY_GROUPS)
**Location:** `:71–130`
**Fix:** Add `gene_geneformer_v2` to `expression`; add `wsi_virchow2/conch/hoptimus1` to `wsi`. Add warning logging uncovered reps.

### P1 — Pairwise and all-combined concatenation has no PCA/cap; curse-of-dimensionality biases against "all"
**Location:** `:299–337`, `:502–543`, `:552–592`
**Fix:** Apply per-fold PCA via `cap_dimensions` in `eval_concat_reps`.

### P2 — `get_clf_model_factories` missing SVM and `_max_samples` guards
**Location:** `:200–235`
**Fix:** Route through `get_model_factories(task_type='classification', n_samples=...)`.

### P2 — `XGBClassifier(use_label_encoder=False)` deprecated in xgboost ≥ 1.6
**Location:** `:219`
**Fix:** Remove the kwarg.

### P2 — `FACTORY_EXPRESSION_REPS` is dead code with a misleading comment
**Location:** `:133–154`, `:419–424`, `:750–751`
**Fix:** Delete; fix comment.

### P2 — Docstring modality-group table disagrees with code for expression
**Location:** `:9–42`
**Fix:** Sync docstring to actual MODALITY_GROUPS contents.

---

## c13 — Forward Compatibility (`c13_forward_compatibility.py`) — 2 P0, 3 P1, 3 P2

### P0 — Conditions C and D use fake dataset names; bypass precompute cache; live find_best_hybrid sees all labels → leakage
**Location:** `:481`, `:517–518`
**Fix:** Pass real `dataset_name`; cache differentiation comes naturally from rep_shapes hash. Or pass `data=data` so in-memory `precomputed_searches` is hit.

### P0 — Condition E is literally identical to Condition D (`e_scores = d_scores`)
**Location:** `:588–598`
**Fix:** Either remove E and rename delta to "D-B (augmentation gain)", or make E genuinely different (e.g. apply hybrid feature selection to single-rep r).

### P1 — `_LEGACY_MOLECULAR` is stale — auto-detect wrongly flags grover/mhg_gnn/etc. as "new"
**Location:** `:68–126`
**Fix:** Audit legacy sets against current DATASET_REPS; add grover, mhg_gnn, pdv, molformer, chemberta, smited, atom_pair, chembert, graph_kernel, graphmvp, selformer, rdkfp, top_torsion to molecular legacy.

### P1 — `_eval_hybrid_per_fold` picks blueprint by best across 5 models even though `model_factory` is fixed
**Location:** `:190–215`
**Fix:** Constrain to single `best_b_model`; or pass `downstream_model_name` arg.

### P1 — `_eval_single_rep_per_fold` skips `cap_dimensions`; high-dim reps tanked on small antibody datasets
**Location:** `:146–158`
**Fix:** Add `cap_dimensions(X_train, X_val)` after scaler.

### P2 — `_make_row_shuffled` actually permutes columns (per-column), not rows
**Location:** `:329–334`
**Fix:** Rename to `_col_permuted` or implement true row-shuffle.

### P2 — `_eval_hybrid_per_fold` ignores `model_factory` for blueprint selection
**Location:** `:190–215`
**Fix:** Same as P1 above.

### P2 — Win-rate counting double-applies direction correction → inverted win rate for lower_is_better
**Location:** `:724–729`
**Fix:** `if dc > 0: win += 1` (already sign-normalized).

---

## c14 — Cross-Domain Summary (`c14_cross_domain_summary.py`) — 1 P0, 2 P1, 3 P2

### P0 — Apples-to-oranges: single-rep baseline uses 11 models; hybrid pool searches only 5 wrong models (missing xgboost, ridge)
**Location:** `:185` vs `:106–107`
**Fix:** Use `DOWNSTREAM_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'mlp', 'lightgbm']` for both candidate loop and single-rep baseline factories. Remove `knn`.

### P1 — Fold count/split mismatch in intermediary mode: single-rep is 1-fold 80/20, hybrid is 5-fold KFold
**Location:** `:74–75`, `:123–124`, `:219`
**Fix:** Read single-rep baselines from `search.baselines[rep][model]` (5-fold) instead of rerunning independent CV.

### P1 — `mean_delta`/`std_delta` aggregated across heterogeneous metric scales is meaningless
**Location:** `:322–323`, `:376–377`, docstring `:19`
**Fix:** Drop the cross-metric scalar deltas; keep win_rate + Wilcoxon + per-domain breakdowns.

### P2 — Single-rep baseline missing `cap_dimensions`; SVM/GP fail silently swallowed
**Location:** `:125–136`
**Fix:** Add cap_dimensions; replace bare `except`.

### P2 — `pct_improvement` formula explodes for near-zero `best_single_score`
**Location:** `:224`, `:227`
**Fix:** Floor denominator at `max(abs(score), 0.05)` or drop pct from cross-domain aggregate.

### P2 — `output_dir` parameter of `run()` is documented but never used
**Location:** `:412`, `:499`
**Fix:** Move save into `run()` or drop the parameter.

---

## c15 — ML Theory Depth (`c15_ml_theory_depth.py`) — 0 P0, 1 P1, 3 P2

### P1 — Instability detection always reports UNSTABLE (15c)
**Location:** `:601–621`
**Bug:** Budgets are `[0.8n, 0.9n, 1.0n, 1.1n, 1.2n]` — all fall inside `near_n_mask` bounds 0.7–1.3, so `far_scores` is always empty, `far_var=1e-10` sentinel; ratio always ~1e10; "UNSTABLE" always printed. The entire scientific claim of 15c is dead.
**Fix:** Extend sweep with `[0.1, 0.3, 0.5]` so `far_scores` is non-empty.

### P2 — Duplicate `get_fitted_pool_hybrid` for budget=50 in 15b
**Location:** `:291–306`, `:321–331`
**Fix:** Save `X_hybrid` from budget=50 iteration; reuse.

### P2 — Theoretical `sqrt(d/n)` uses `min(budget, X.shape[1])` but actual features capped at `min(..., n_samples-1)`
**Location:** `:875`
**Fix:** Use the same cap.

### P2 — Rademacher hybrid built using REAL labels (`y[train_idx]`) → memorization test biased
**Location:** `:975–976`
**Fix:** Use val-split features (built without those labels) for the random-label fit.

---

## c16 — Feature Selection as Regularization (`c16_feature_selection_as_regularization.py`) — 1 P0, 3 P1, 3 P2

### P0 — `augmentations` referenced before assignment → NameError on every run
**Location:** `:209` (use), `:224` (assignment)
**Bug:** `_c16_strategy = 'uncapped' if augmentations else 'none'` at :209 references a name not in scope; the assignment happens at :224. Unconditional crash.
**Fix:** Move the assignment before :209.

### P1 — Blueprint reuse ignores per-budget intent: `build_fold_hybrid` uses `blueprint.budget`, not loop `budget`
**Location:** `:241–244`, `:291–298`
**Fix:** Pass `budget_override=budget` to `build_fold_hybrid`.

### P1 — `LogReg` missing from DEFAULT_MODELS; `_family_to_canonical` maps to non-existent `'logreg'`
**Location:** `:205–206`
**Fix:** Map `'LogReg': 'ridge'` (same family for gating).

### P1 — Pool budget filter `b <= n_samples` is too aggressive; small datasets see only tiny budgets
**Location:** `:193–196`
**Fix:** Filter only by `b <= total_dim`.

### P2 — `capped_full` budget search uses `speed='fast'` while everything else uses `'full'`
**Location:** `:263`
**Fix:** Use `'full'`.

### P2 — `'best_tuned_full_dim'` is mislabelled (it's `capped_full`, not true full-dim)
**Location:** `:319`, `:370–388`
**Fix:** Rename key to `'best_tuned_capped_full'`.

---

## c17 — Information-Theoretic Decomposition (`c17_information_theoretic_decomposition.py`) — 0 P0, 4 P1, 3 P2

### P1 — `downstream_r2` is not direction-corrected; RMSE/MAE values are wrong-signed in the field
**Location:** `:170–178`, `:204`, `:287–291`
**Fix:** Pass `lower_is_better`; rename key to `downstream_score`; apply sign.

### P1 — Hybrid raw output vs comparator pre-scaled output → asymmetric standardisation
**Location:** `:153–154`, `:165–166`, `:209`, `:265–285`, `:287`
**Fix:** Don't pre-scale `X_concat`. Let `_evaluate_strategy` own the single scaling.

### P1 — mRMR and RFE run feature selection on full data (label leakage into downstream R²)
**Location:** `:229–235`, `:272–280`
**Fix:** Either remove `downstream_r2` from the cross-strategy comparison, or implement proper nested CV.

### P1 — `_rfe_select` hardcodes `RandomForestRegressor`; ignores task_type; field misnamed
**Location:** `:133–140`, `:193–197`
**Fix:** Rename `downstream_r2` → `downstream_score`; document RF-fixed protocol.

### P2 — Budget filter `b <= n_samples` allows budget == n_samples (degenerate CV)
**Location:** `:217`
**Fix:** `b <= n_samples // max(n_folds, 2)`.

### P2 — `estimate_mutual_information_knn` called on pre-scaled data → double standardisation
**Location:** `:153–154`
**Fix:** Don't pre-standardise; estimator is self-standardising.

---

## c18 — Diversity vs Depth (`c18_diversity_vs_depth.py`) — 1 P0, 1 P1, 3 P2

### P0 — Augmentation asymmetry between arms invalidates comparison
**Location:** `:146–218`
**Bug:** `depth_1/2/3` and `uniform_all` use base reps only; `natural` arm gets augmentations passed to `build_fold_hybrid`. Any win by natural is uninterpretable (it has a larger feature pool).
**Fix:** Either drop augs from all arms or include augs in all arms.

### P1 — `uniform_all` uses fewer features than budget when budget not divisible by n_reps
**Location:** `:67–85`
**Fix:** Distribute remaining slots across reps in importance order until budget exhausted.

### P2 — Natural arm records `budget` (outer) as `n_features` but may use `blueprint.budget` (different)
**Location:** `:235`
**Fix:** Record `bp.budget` for the natural condition.

### P2 — `_score_prediction` silently falls back to class labels if `predict_proba` raises
**Location:** `:99–106`
**Fix:** Narrow except; log warning.

### P2 — Rep importance for depth ranking is noisy at small fold sizes (<200)
**Location:** `:41–45`, `:172`
**Fix:** For n<200 use internal 3-fold averaged gain, or permutation importance.

---

## c19 — Importance-Weighted Training (`c19_importance_weighted_training.py`) — 1 P0, 2 P1, 2 P2

### P0 — `combined` condition double-weights features before computing reliability scores
**Location:** `:193–198`
**Bug:** `X_tr_s = X_tr_s * imp_weights` then passes already-weighted X to `_compute_reliability_scores` which multiplies again — kNN runs in `X * imp_weights²` space. Reliability scores are not what the docstring claims.
**Fix:** Compute reliability on unweighted standardised X first, then apply feature weighting.

### P1 — `_compute_importance_weights` uses `sum(w)=1` normalisation → StandardScaler cancels the weighting
**Location:** `:45–47`, `:184–186`, `:194–195`
**Bug:** Same no-op bug as n12. With sum-to-1 normalisation and hundreds of features, each weight is ~0.005; multiplying standardised columns by ~0.005 just uniformly crushes them. Feature-weighted ≈ unweighted geometrically.
**Fix:** Use mean-1 normalisation: `w = w / w.sum() * len(w)`. Apply same fix to `_compute_reliability_scores`.

### P1 — Bare `except Exception` silently falls back to unweighted fit; corrupts delta if weighting fails for non-kwarg reasons
**Location:** `:211–221`
**Fix:** Catch only `TypeError`; let other errors propagate.

### P2 — `importance_pruned` median threshold = random 50% drop under current normalisation
**Location:** `:200–205`
**Note:** Resolved transitively if the normalisation fix above is applied.

### P2 — `mode` variable dead (already correctly forwarded — non-bug, retracted by agent).

---

## c21 — Single-Rep Augmentation (`c21_single_rep_augmentation.py`) — 2 P0, 3 P1, 2 P2

### P0 — Missing `task_type` + `dataset_name` causes every augmentation search to silently fail
**Location:** `:149–165`
**Bug:** `find_best_single_rep_augmented` now raises ValueError if `task_type=None` or if `cache=True` without `dataset_name`/`tier`. The bare `except Exception` swallows both; `bp = None`; every aug condition fills with NaN. Experiment runs to completion with hollow output.
**Fix:** Pass `task_type=task_type`, `metric=metric_name`, `dataset_name=dataset_name`, `tier=f'c21_{base_name}'`, `condition_suffix=cond_name`. Remove silent except.

### P0 — AUG_GROUPS contain phantom rep names for TCGA and DTA (also flagged in helpers audit)
**Location:** `complexity_helpers.py :876–899`
**Fix:** Per the complexity_helpers audit above.

### P1 — Single rf blueprint applied to all downstream models (knn, xgboost get rf-optimal blueprint)
**Location:** `:160–162`, `:167–210`
**Fix:** Inside model loop: `bp = best_by_model.get(model_name) or best_by_model.get('rf')`.

### P1 — `relative_improvement` stored without direction correction for lower-is-better
**Location:** `:249`
**Fix:** Apply `sign`.

### P1 — `n_folds` not passed to `find_best_single_rep_augmented`; cache misses under intermediary mode
**Location:** `:149–159`
**Fix:** Pass `n_folds=n_folds`.

### P2 — `bh_doyle_plates` `codon_usage` collision (covered by helpers audit).

### P2 — Missing `condition_suffix` makes cache files not per-condition human-readable
**Location:** `:149–159`
**Fix:** Pass `condition_suffix=cond_name`.

---

## c22 — Explainability (`c22_explainability.py`) — 2 P0, 4 P1, 3 P2

### P0 — SHAP re-trains a fresh model instance; discards the scored model
**Location:** `:141`
**Fix:** Save the fitted `best_model_obj` from the scoring loop; pass that to SHAP.

### P0 — `aggregate_importance_by_rep` and `explain_hybrid` silently ignore augmentation features → SHAP indices misaligned
**Location:** `:199–204`, `:211–212`
**Fix:** Pass `augmentation_info` explicitly to both functions; warn if augmentations present but not aggregated.

### P1 — `feature_info` from last fold used; SHAP averaged across folds → feature indices may differ per fold
**Location:** `:188`, `:199`
**Fix:** Restrict to n_folds=1 OR accumulate SHAP per named feature using feature_info to map (rep, local_idx).

### P1 — `shap_vals` shape handling is fragile across SHAP versions/binary-vs-multiclass
**Location:** `:154–158`
**Fix:** Explicit normalisation to `(n_samples, n_features)`.

### P1 — Only `['rf', 'xgboost']` evaluated; ignores DOWNSTREAM_MODELS
**Location:** `:65–67`
**Fix:** Use DOWNSTREAM_MODELS or document the deliberate restriction.

### P2 — `min_len` truncation can silently misalign features
**Location:** `:194–195`
**Fix:** Assert all folds have same feature count; warn+skip if diverge.

### P2 — `search_score` field name is ambiguous (it's the internal-evaluator CV score, not downstream val)
**Location:** `:83`
**Fix:** Rename `search_score_internal_cv`.

### P2 — `task_type` not forwarded to `get_best_search_for_downstream`
**Location:** `:71–78`
**Fix:** Pass `task_type=task_type`.

---

## c23 — Evaluator-Model Matching (`c23_evaluator_model_matching.py`) — 0 P0, 0 P1, 5 P2

### P2 — Comment claims catboost/dnn/cnn use ridge for matched; code uses themselves (correctly)
**Location:** `:47–48`
**Fix:** Delete/rewrite the parenthetical comment.

### P2 — `metric_fn` dead variable
**Location:** `:131`
**Fix:** `_` it.

### P2 — Condition D for `('dnn', 'mlp')` is exact cache hit of ABC B_slight_off for dnn
**Location:** `:64–68`, `:185–220`
**Fix:** Remove `('dnn', 'mlp')` from NN_PAIRS.

### P2 — Condition D and E deltas not printed in summary
**Location:** `:279–306`
**Fix:** Add D/E summary blocks.

### P2 — Inconsistent delta storage: D at result root; ABC inside per-condition scores
**Location:** `:214–220` vs `:172`
**Fix:** Store D's delta inside scores too.

---

## c24 — Blueprint Transfer (`c24_blueprint_transfer.py`) — 2 P0, 3 P1, 3 P2

### P0 — Classification: `model.predict()` always called; AUC requires `predict_proba`
**Location:** `:226`, `:288`
**Fix:** Use a `_predict_safe` helper that dispatches to `predict_proba` when metric is AUC.

### P0 — Own-fold score vs transfer-matrix diagonal computed independently; can diverge
**Location:** `:191–230`, `:241–291`, `:299`
**Fix:** Eliminate `own_fold_scores` and rely on `transfer_matrix[(i,i)]`; or vice versa with explicit reuse.

### P1 — Cache misalignment: uses `f"{dataset_name}_fold{i}"` + `augmentation_strategy='none'`/'uncapped'` inconsistent with global precompute; per-fold cache always recomputes
**Location:** `:167`
**Fix:** Use precomputed search via `get_precomputed_search`; or align `augmentation_strategy` consistently.

### P1 — Cross-fold transfer uses source fold's best model; should use target fold's
**Location:** `:173–176`, `:277–282`
**Bug:** Model type confounds the feature-selection comparison.
**Fix:** Use `fold_best_models[tgt_fold]` for all entries in `tgt_fold` column.

### P1 — `nogueira_stability` `total_features` uses full-dataset sum; doesn't match per-fold available features when reps get dropped
**Location:** `:133`, `:333`
**Fix:** Track per-fold available count.

### P2 — No comparison against a target-native baseline blueprint
**Location:** Structural omission
**Fix:** Add naive-blueprint null baseline.

### P2 — `augmentations` passed as `{}` even when `augmentation_strategy='none'`
**Location:** `:156–169`
**Fix:** Use `None` consistently when no augs.

### P2 — `fold_feature_sets` not populated for folds whose own-fold eval failed; stability subset diverges from transfer matrix
**Location:** `:212–215`, `:332`
**Fix:** Assert key-set equality; print warning on partial failure.

---

## Complexity suite — Headline triage

1. **complexity_helpers P0×2** (aug-loading drops reps on bh_doyle_plates; codon_usage double-included on gb_demo) — propagates to every aug-using c0X experiment.
2. **c16 P0** — NameError on every run (`augmentations` used before assignment).
3. **c21 P0** — every augmentation search silently fails (missing task_type/dataset_name); experiment output is hollow.
4. **c13 P0×2** — Conditions C/D bypass precompute → data leakage; Condition E is literally Condition D.
5. **c10 P0×2** — budget sweep broken (`bp.budget` overrides loop); noise RNG state carries across folds/sigmas → not i.i.d.
6. **c14 P0** — single-rep baseline uses 11 models, hybrid pool only 5 (missing xgboost+ridge).
7. **c08 P0×2** — exp2_results path drops evaluator/aug; val_score from one model, calibration from another.
8. **c24 P0×2** — classification `predict` vs `predict_proba`; own-fold/cross-fold diagonal inconsistent.
9. **c22 P0×2** — SHAP re-trains a new model; augmentation features silently ignored by SHAP aggregation.
10. **c05 P0×2** — scaffold groups shattered by random subsampling; `_max_samples` model-set asymmetric across n.
11. **c02 P0×2** — PCA baseline uses hard labels for AUC; DTA `dta_protein_lora` FT factory silently empty.
12. **c01 P0** — `--reps` CLI silently no-op when comma-separated.
13. **c03 P0** — does not actually transfer blueprints (just compares native optima).
14. **c09 P0** — raw `create_hybrid` no caching in 9b/9c (≥2026-03-30 architecture violation).
15. **c18 P0** — augmentation asymmetry between arms.
16. **c19 P0** — `combined` double-weights → kNN in `X * w²` space.
17. **c15 P1** — instability detection always prints UNSTABLE (the whole experimental claim is dead).
18. Many P1s: metric direction in stored keys (R²-named fields actually CI/AUROC/RMSE), `bp.augmentation_strategy` vs `meta['aug_condition']`, blueprint-vs-loop budget mismatches, missing INTERNAL_EVALUATORS plumbing, sum(w)=1 weighting no-op (c19 + n12 pattern).
