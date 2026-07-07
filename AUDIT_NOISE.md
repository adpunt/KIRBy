# Noise Experiments Audit — 2026-05-23

## Files audited
- `tests/experiments/noise/n01_activity_cliff_framework.py` (754 lines)
- `tests/experiments/noise/n02_sample_weighting.py` (489 lines)
- `tests/experiments/noise/n03_sas_landscape.py` (659 lines)
- `tests/experiments/noise/n04_gp_loo_detection.py` (374 lines)
- `tests/experiments/noise/n05_noise_injection_benchmark.py` (279 lines)
- `tests/experiments/noise/n06_bindingdb_ground_truth.py` (196 lines)
- `tests/experiments/noise/n07_remediation.py` (331 lines)
- `tests/experiments/noise/n08_estimation_survey.py` (255 lines)
- `tests/experiments/noise/n09_lnl_survey.py` (531 lines)
- `tests/experiments/noise/n10_hybrid_neighborhood_lnl.py` (734 lines)
- `tests/experiments/noise/n11_estimation_remediation_grid.py` (561 lines)
- `tests/experiments/noise/n12_importance_weighted_distance.py` (478 lines)
- `tests/experiments/noise/n13_noise_distributions.py` (576 lines)
- `tests/experiments/noise/noise_helpers.py` (275 lines)

Supporting modules cross-checked:
- `src/kirby/find_best_hybrid.py` (signature, cache contract)
- `src/kirby/utils/lnl_methods.py` (registry surface)
- `src/kirby/utils/noise_estimation.py` (ESTIMATOR_REGISTRY, classes)
- `src/kirby/utils/noise_mitigation_methods.py` (DISTANCE_AWARE_METHODS)
- `tests/experiments/_shared.py` (helpers, INTERNAL_EVALUATORS, DOWNSTREAM_MODELS_NOISE)

(Note: this file previously held the 2026-05-08 ANNITIA noise-mitigation
candidate notes. Overwritten with the noise-experiments code audit.)

---

## 2026-07-07 VIABILITY REVIEW — cuts and rescopes

A design-level (not code-level) viability pass. The P0/P1/P2 sections below
are code bugs; this section records experiments judged unsound *by construction*
and the honest rescoping applied to the survivors. Metric context: the old NSI/
NDS linear slope has been dropped everywhere in favour of auc_norm + Weibull
(`src/kirby/noise_robustness.py`).

### CUT — deleted (irredeemable by design; recoverable via git history)
The file was `git rm`'d and its ID removed from the `audit_full_wave.py` /
`audit_done_dataset_gaps.py` registries. Stale result dirs under
`tests/results/noise/<id>/` were left in place (data not destroyed) but are now
orphaned and should not be reported.

- **n03 sas_landscape** — circular zone-shift (zones defined from the same
  embedding they're scored on) and the one predictive test (GP bridge) is
  leakage-biased AND an n=4 correlation.
- **n04 gp_loo_detection** ("killer experiment") — rigged: the ground truth is
  the injected Gaussian itself, and the hybrid is fit on the clean labels it is
  then asked to separate, so its win is by construction.
- **n05 noise_injection_benchmark** — F1 scored against an artifact mask
  (|noise| > MAD of the injection itself); homoscedastic Gaussian ≠ real assay
  noise.
- **n08 estimation_survey** (noise-estimation cliff-discrimination) — circular:
  cliffs and noise scores are both derived from the same embedding distances +
  labels.
- **n10 hybrid_neighborhood_lnl** — synthetic + self-fulfilling: the injected
  Gaussian *is* a neighbour-deviation, so a distance detector "wins" trivially.

### RESCOPED — kept, framing/bugs fixed (survivors are synthetic-noise only)
- **n01** — kept ONLY sub-analysis 4 (cliff-stratified prediction error;
  rep-independent cliff reference + held-out CV). Sub-analyses 1–3 were circular
  (cliffs from each rep's own embedding → supervised hybrid wins by
  construction) and were removed. Molecular only.
- **n07** — sound within the synthetic-noise regime; fixed the per-rep `is_clf`
  bug (P0 below) and scoped the write-up to synthetic Gaussian label noise, no
  fiction-scored detector.
- **n09** — honest *relative* question (does the hybrid change LNL-method
  ranking); fixed the weighted-hybrid scale collapse (P1); synthetic noise on
  train only.
- **n11** — the held-out multi-seed recovery metric is the claim; the
  hybrid-vs-base comparison is oracle-biased (blueprint chosen on clean labels)
  and is relabelled as best-case, not a fair baseline. Molecular, synthetic.
- **n12** — legit narrow ablation; scoped molecular-only and flagged that
  effects may sit within run-to-run variance; oracle-leak comment corrected.
- **n13** — the soundest framing (synthetic-noise sensitivity study); dropped
  multiplicative noise, fixed partial-noise clf semantics and the MLP
  unweighted-fallback control.

n02 and n06 were not flagged in this pass and are unchanged.

---

## P0 (broken / silently wrong results)

### N07 per-rep evaluation forgets `is_clf` for classification datasets
`tests/experiments/noise/n07_remediation.py:267-269`
The hybrid branch correctly passes `is_clf=is_clf` (line 244), but the per-rep
branch calls `_evaluate_fold(...)` without it. Inside `_evaluate_fold`,
`_inject_noise(y_train, noise_fraction)` then runs the regression path —
adding Gaussian-scaled-by-`y.max()-y.min()` noise to integer class labels and
training a regressor where the per-rep config calls for a classifier.
- Symptom: on every clf dataset (trastuzumab, humanness, ginkgo clf,
  bh_doyle if clf, etc.) every per-rep N07 row except the hybrid branch is
  junk. Recovery numbers vs hybrid baseline will look spuriously favourable
  to hybrid.
- Fix: pass `is_clf=is_clf` at the per-rep call site, mirroring the hybrid
  branch on line 244.

### N03 hybrid for sub-analyses 1–3 silently fails (cache contract violation)
`tests/experiments/noise/n03_sas_landscape.py:553-562`
The call `get_hybrid_features_and_weights(reps, labels, groups, search_speed,
'lightgbm_gain', augmentations=augmentations)` lands in
`tests/experiments/_shared.py:1930-1973`, which invokes `find_best_hybrid`
with `cache=True` but without `dataset_name`, `tier`, `task_type`, or
`augmentation_strategy`. Per `src/kirby/find_best_hybrid.py:4018-4036` that
combination raises `ValueError` immediately (`cache=True requires
dataset_name=...`). The `except Exception` on n03:561 swallows the error and
the experiment proceeds with `X_hybrid=None`, so the SALI smoothing, hybrid
consensus Jaccard, and cliff mitigation sub-analyses produce only the "rep
only" branches — the hybrid columns/rows in those tables are silently empty
for every dataset. (Sub-analysis 4 uses `get_fitted_pool_hybrid` correctly.)
- Fix: replace the `get_hybrid_features_and_weights` call with
  `get_fitted_pool_hybrid(reps, labels, groups, downstream_model='rf',
  budget=None, augmentations=augmentations, speed=search_speed, data=data,
  dataset_name=dataset_name, task_type=task_type)` and drop the
  `get_hybrid_features_and_weights` import.

### N02 hybrid reliability scores ignore `is_clf`
`tests/experiments/noise/n02_sample_weighting.py:311-313`
`compute_reliability_scores(X_train_s, y_train, imp_scores, k=k_neighbors)`
omits `is_clf=is_clf` while the equivalent per-rep call on line 197-199 does
pass it. For any classification dataset the hybrid path takes the regression
branch (line 102-109) — dividing class labels by `y.max()-y.min()` (=
`n_classes-1`) and treating "neighbor mean" agreement as a continuous
deviation. Reliability weights for the hybrid+sample-weight condition are
nonsense, so condition-4 vs condition-3 deltas on any clf dataset are
meaningless. The same omission applies implicitly to condition 5
(`compute_cliff_aware_weights`, line 377) which calls
`ActivityCliffAwareEstimator` — an estimator with no clf branch — on
classification labels.
- Fix: pass `is_clf=is_clf` on line 312. For condition 5 either guard the
  call with `if not is_clf` or document that cliff-aware weighting is
  regression-only and skip on clf datasets.

---

## P1 (methodology drift / incorrect-but-recoverable)

### N09 weighted-hybrid scale collapse
`tests/experiments/noise/n09_lnl_survey.py:246-249`
After computing `w = sqrt(max(fold_weights, 0))` and `w = w / (w.sum() +
1e-10)`, the code does `X_whybrid_train_s = X_hybrid_train_s * w` (and the
same for val) with no follow-up rescaling. For a ~100-feature hybrid each
column ends up multiplied by ~0.01. The DNN baseline (Adam, lr=0.001, no
input re-normalisation) then trains on near-zero inputs while the LNL
methods inherit the same scale. Conditions E and F (weighted hybrid + LNL)
are systematically handicapped by an arbitrary scale shrink rather than by
the weighting hypothesis the experiment claims to test. n02 / n10 do not
have this issue because they apply `StandardScaler` after multiplication.
- Fix: wrap with `StandardScaler().fit_transform(X_hybrid_train_s * w)`
  (fit on train, transform on val) like n10:177-178 does.

### Blueprint selected on clean labels; per-fold rebuild uses noisy labels
- `n07_remediation.py:181-196,232-234`
- `n09_lnl_survey.py:106-122,205-207`
- `n10_hybrid_neighborhood_lnl.py:418-435,489-491`
- `n11_estimation_remediation_grid.py:278-289,339-341`
- `n12_importance_weighted_distance.py:237-248,303-306`
- `n13_noise_distributions.py:211-226,330-332`

Blueprint is selected from the precomputed pool, which was fit on CLEAN
labels (`precompute_hybrid_searches` in `noise_helpers.py:206-218`). Per-fold
`build_fold_hybrid(...)` / `create_hybrid(...)` then reuses that blueprint's
allocation, importance method and internal evaluator under noisy labels.
The intent comments (e.g. n09:197-199, n11:331) call this "no oracle leak" /
"realistic" — but the *choice* of allocation method and importance method
sees clean labels, biasing every hybrid condition toward higher retention
than a fair baseline. Either re-derive the blueprint per fold under noise
(expensive) or correct the inline comments to "blueprint from clean-label
pool, per-fold construction under noise — best-case hybrid".

### N10 ground-truth noise mask uses MAD threshold
`tests/experiments/noise/n10_hybrid_neighborhood_lnl.py:515-519`
For regression the noise mask is `|noise_vec| > MAD(y_train_clean)`. At
σ=0.3 × y_range that threshold often eliminates a large minority of the
injected samples (Gaussian draws inside ±MAD are labelled "not noisy"),
shrinking the positive class for the detection F1 calculation in
`_run_detection` (line 267-303). The reported `n_noisy` and the
precision/recall denominators reflect a thresholded subset, not the actual
injected set.
- Fix: either always use `|noise_vec| > 0` (anything injected counts) or
  document the MAD threshold in the writeup.

### N04 noise injection applied to FULL labels (single realisation)
`tests/experiments/noise/n04_gp_loo_detection.py:237-241`
`_inject_noise(labels, frac)` runs once per fraction with a fixed seed, then
GP-LOO uses the full noisy vector. Because the experiment has no train/val
split (LOO detector benchmark) this is not a leakage bug, but the noise
pattern is identical across every space within a fraction — any "hybrid
wins on cliff_fp_rate" claim is conditional on this single realisation.
Either add a 3-5 seed loop and report mean ± std, or flag the single-shot
caveat in the writeup.

### N13 partial-noise semantics collapse for classification
`tests/experiments/noise/n13_noise_distributions.py:77-78`
For `noise_type='partial'` under classification, `flip_frac = strength *
partial_fraction` reduces "partial" noise to "Gaussian noise at a smaller
rate". The regression path (line 113-119) corrupts a *subset* of samples at
full strength — a structurally different concept. Either skip 'partial' for
classification or implement subset-flipping (flip strength fraction of
samples within a partial_fraction-sized subset).

### N13 silent unweighted fallback for MLP in Track B
`tests/experiments/noise/n13_noise_distributions.py:400-404`
`try: m.fit(X, y, sample_weight=weights) except TypeError: m.fit(X, y)`. For
`mlp` (sklearn MLPRegressor/Classifier do not accept `sample_weight`), the
Track-B "remediated" score becomes a fresh fit on noisy labels — equal to
the noisy baseline by construction, inflating "no recovery" counts. Either
drop MLP from Track B model loop when remediation is via sample weights, or
log that MLP rows are unweighted controls, not remediated.

---

## P2 (smells, future-proofing)

### N02 inconsistent top-3 source
`tests/experiments/noise/n02_sample_weighting.py:252,447`
`top3` from the search (RF-ranked, line 252) is overwritten by the per-model
loop variable (line 424, a 4-tuple of `(rep, uw, w, d)`). The final
`results['top3_base_reps']` thus contains the *last* model's per-rep
ranking, not the search-derived rep ranking that the field name implies.
- Fix: rename the search-derived variable to `top3_search` and assign
  `results['top3_base_reps'] = [(n, s) for n, s in top3_search]`.

### N08 string-format crash on missing cliff_correlation
`tests/experiments/noise/n08_estimation_survey.py:198`
`f", cliff_rho={entry.get('cliff_correlation', '?'):.3f}"`. If
`'cliff_correlation' not in entry` and the `'?'` sentinel is hit, `:.3f`
raises `TypeError` mid-loop. Won't trigger as long as `n_cliffs >= 5`
always.
- Fix: split into two branches based on key presence.

### N05 single-realisation noise across all reps and sigmas
`tests/experiments/noise/n05_noise_injection_benchmark.py:140`
`_inject_noise(labels, sigma, is_clf=is_clf)` uses the default `seed=42`
once per sigma — every method then sees the same noise pattern. Detection
F1 is computed on a single noise realisation per σ, so per-rep F1
differences include single-draw variance. A 3-seed loop with mean would
make rankings credible.

### N09 retention >100% for lower-is-better metrics
`tests/experiments/noise/n09_lnl_survey.py:381-394`
Formula `clean_val / noisy_val * 100` for lower-is-better is correct in
sign but can yield >100% when LNL actually beats the clean baseline (noise
+ denoising leaving lower-than-clean RMSE). Output is plausible but readers
will be confused. Add a clamp or rename `retention_pct` to
`score_ratio_pct` with a docstring note.

### N10 importance recomputation uses noisy labels
`tests/experiments/noise/n10_hybrid_neighborhood_lnl.py:155-178`
`compute_feature_importance` is called with `y_train = y_noisy` (line
156-158) — the lightgbm gain ranking is fitted on noisy labels. Consistent
with the "realistic noisy training" framing but means the
`imp_weighted_*` spaces are doubly noisy (noisy labels for both importance
and downstream). Worth noting in the write-up.

### N10 imp_weighted spaces partially undone by StandardScaler
`tests/experiments/noise/n10_hybrid_neighborhood_lnl.py:162-178`
After multiplying features by `sqrt(importance)` and re-running
`StandardScaler`, low-importance columns (near-zero variance) are whitened
back to unit variance — partially undoing the importance weighting. Either
skip StandardScaler for `imp_weighted_*` spaces or document that the
"importance weighting" only affects feature orientation, not column
weighting.

### N13 partial-noise mask not surfaced
`tests/experiments/noise/n13_noise_distributions.py:113-119`
`corrupt_idx` is generated and consumed locally — never returned. Detection
/ recovery tracks have no way to compute "did the estimator find the
corrupted samples" — the most natural metric for partial noise.
- Fix: return `corrupt_idx` alongside `y_noisy` (return tuple) or add a
  separate hook the caller can use.

### Shared noise seed across noise experiments
All N0x `_inject_noise` paths use `seed=42 + fold_idx`. Across the 5-fold
loop that gives 5 distinct realisations per σ — fine for averaging. But two
different experiments (e.g. N09 vs N11) on the same dataset/σ use
*identical* noise vectors. Cross-experiment "agreement" then partly
reflects shared randomness, not method robustness. Either pass a unique
`base_seed` per experiment ID or randomise per (experiment, fold).

### N13 `partial_fraction` × `strength` axis mismatch
`tests/experiments/noise/n13_noise_distributions.py:296-302`
`strength_variants = [(0.3, f) for f in PARTIAL_FRACTIONS]` reports partial
at one strength value only (0.3). Other noise types sweep
`STRENGTHS = [0.1, 0.3, 0.6]`. The 'partial' track is therefore a 1D scan
over `partial_fraction` when the others are 1D scans over strength —
different axes plotted as if comparable. Either also sweep
strength × partial_fraction (3×3) or label the partial axis differently in
the summary.

### N09/N13 leading-underscore import surface
Both files import `_standard_dnn_train_predict` from `lnl_methods` — a
leading underscore means the module author considers it private. A refactor
to `lnl_methods.py` could silently break these imports. Either expose a
public alias or accept the coupling and add a comment.

### N04 / N06 return raw dict, not `make_structured_output`
`n04_gp_loo_detection.py:365` and `n06_bindingdb_ground_truth.py:183`
return plain dicts. Other noise experiments use
`make_structured_output(...)`. Downstream aggregation expecting the
standard envelope (`experiment_id`, `primary_metric`, `lower_is_better`,
etc.) will miss n04/n06 rows. Already noted in
`PRECOMPUTE_EXPERIMENT_BUGS_2026_04_17.md §4.5` — still unfixed.

### N08 estimators have no classification branch
`tests/experiments/noise/n08_estimation_survey.py:130-167`
Most estimators in `ESTIMATOR_REGISTRY` (GP, conformal, BayesianRF,
NeighborResidual, LocalVarianceRatio, etc.) are regression-design. n08
runs them on whatever labels are in `data['labels']` with no clf guard
(line 167 calls `estimator.fit(X_sub, y_sub)` unconditionally). For
classification datasets the estimator may silently fit a regressor on
integer-valued labels and produce nonsense `noise_scores_`. Either restrict
the experiment to regression datasets or audit each estimator class for
clf compatibility before opening n08 to clf tasks.

---

## Verified-clean

- **`noise_helpers.py`** — precompute pool calls pass `tier`,
  `dataset_name`, `augmentation_strategy` implicitly through
  `precompute_hybrid_searches`; scaffold-grouped CV via `cv_split`; uses
  `DOWNSTREAM_MODELS_NOISE` and `INTERNAL_EVALUATORS` correctly; no
  references to retired `best_overall`.
- **`n04_gp_loo_detection.py`** — early-returns for classification (line
  164-166), uses pool helper with `task_type=task_type`, noise seeded,
  noise applied after representation construction (no train/val split by
  design). Only smells: P1 single-shot noise realisation and P2 raw-dict
  return.
- **`n05_noise_injection_benchmark.py`** — handles clf via label flipping
  in `_inject_noise`, pool helper used correctly, all 15 detectors
  evaluated. Only smell: P2 single-noise-realisation per σ.
- **`n06_bindingdb_ground_truth.py`** — `task_type` bug from prior audit
  is fixed (line 82, 98). Real ground truth (replicate disagreement), no
  synthetic noise injection, no leakage concerns.
- **`n12_importance_weighted_distance.py`** — proper scaffold splits via
  `cv_split`, noise injected per-fold with `seed=42+fold_idx`, pool helper
  with `task_type` and `dataset_name`, classification branches in
  `_inject_noise`. Main concern is the shared P1 oracle-leak comment.
- **MUTATION_REPS / two-pass** — none of the noise experiments load
  SKEMPI-family data directly; they go through `noise_helpers.run_all` →
  `load_data` which handles two-pass mutation loading upstream. No N0x
  file re-implements its own load path.
- **Random splits** — every noise experiment routes through `cv_split` /
  `make_cv_splitter`, which dispatches to `GroupShuffleSplit` /
  `GroupKFold` / `StratifiedGroupKFold` whenever groups exist. No raw
  `train_test_split` or `KFold(shuffle=True)` is used on grouped data.
- **`best_overall` removed** — no `best_overall` references in the noise
  tree.
- **`augmentation_strategy` explicit on caching path** — every
  `get_best_search_for_downstream` / `get_fitted_pool_hybrid` call in
  n01/n02/n04/n05/n06/n07/n08/n09/n10/n11/n12/n13 routes to
  `get_precomputed_search` which sets
  `augmentation_strategy=aug_condition` explicitly on the
  `find_best_hybrid` call (see `_shared.py:2300`). The cache-collision bug
  from `project_cache_collision_fix.md` is closed in this subdirectory
  **except for the n03 path through `get_hybrid_features_and_weights`**
  (P0 above).
- **No mean baselines, no Ridge sensitivity, no PCA-reduced variants** —
  PCA appears only as an explicit *condition* in n03 (sub-analysis 4,
  comparing hybrid vs PCA in BO bridge) and n04 (`pca_matched` space as
  one of many competing distance spaces); neither uses PCA as a
  "sensitivity check" variant of a hybrid. Ridge appears only in
  `DOWNSTREAM_MODELS` (one member of the standard 6) and `WEIGHTED_MODELS`
  (n02), never as a bolted-on sensitivity check.
- **DOWNSTREAM_MODELS_NOISE vs DOWNSTREAM_MODELS** — `noise_helpers.py:31`
  and `n13:31` import `DOWNSTREAM_MODELS_NOISE`. Per `_shared.py:413` it
  is currently identical to `DOWNSTREAM_MODELS` (ngboost removed
  2026-05-22 — see comment lines 400-412). Behaviour matches memory.
- **Noise applied AFTER train/val split** — every experiment that injects
  noise (n07/n09/n10/n11/n12/n13) calls `_inject_noise(y_train_clean,
  ...)` inside the per-fold loop. Test set is never noised. n04 and n05
  inject into the full label vector because neither has a train/val split
  (LOO detection benchmarks).
- **`find_best_hybrid` direct calls** — none of the noise experiments call
  `find_best_hybrid` directly; all go through cached pool helpers
  (`get_best_search_for_downstream`, `get_fitted_pool_hybrid`,
  `precompute_hybrid_searches`). The only exception was n03 via
  `get_hybrid_features_and_weights`, flagged P0 above.
- **No `cv_splitter` overrides bypassing groups** — searched for
  `cv_splitter=`, `KFold(`, `train_test_split` in the noise tree; only
  internal-CV `cross_val_predict(rf, ..., cv=5)` in n04:322 (used for the
  prediction-error baseline detector, not the hybrid evaluation loop).
