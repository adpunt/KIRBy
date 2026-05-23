# Complexity Experiments Audit — 2026-05-23

## Files audited

24 files in `/Users/apunt/repos/KIRBy/tests/experiments/complexity/`:
- c01 (128), c02 (554), c03 (355), c04 (382), c05 (315), c06 (186), c07 (314), c08 (416), c09 (695), c10 (480), c11 (401), c12 (779), c13 (812), c14 (495), c15 (1006), c16 (352), c17 (316), c18 (251), c19 (262), c21 (322), c22 (283), c23 (331), c24 (395), complexity_helpers (1221). c20 is absent (numbering gap).

## P0 (broken / silently wrong results)

- **`c24_blueprint_transfer.py:158`** — `budgets=FULL_BUDGETS` but the import at line 44 only brings in `SCREENING_BUDGETS`, not `FULL_BUDGETS`. Every fold raises NameError → caught by the bare `except Exception as e` at line 176 → all folds skipped → returns `{'error': 'insufficient_folds'}`. Experiment never produces output. **Fix:** add `FULL_BUDGETS` to the import list at line 44.

- **`c11_ensemble_vs_hybrid.py:376`** — `top3 = extract_top3_base_reps(search) if search is not None else []`. `search` is the loop variable from line 94's tuple unpack. If `budgets` is empty or the try block raises before any iteration succeeds, `search` is undefined → NameError. Lines 109-112 already correctly use the `last_search` variable. **Fix:** replace line 376 with `extract_top3_base_reps(last_search)`.

- **`c10_noise_robustness.py:28-31, 157, 183, 271`** — Regression-only, no `task_type` guard. Hardcodes `NoiseInjectorRegression` and `RandomForestRegressor` throughout. `NoiseInjectorRegression.inject(y_train, sigma)` mangles integer class labels into floats; classification `metric_fn` then runs on regression-style RF predictions → silently wrong on every classification dataset. **Fix:** gate the whole experiment on `task_type == 'regression'` (mirror c04 line 259), or branch to classifier variants.

- **`c15_ml_theory_depth.py:592-713`** — `_eval_at_budget`, `_eval_at_budget_pca`, `_eval_at_budget_random` are regression-only. All three hardcode `RandomForestRegressor` (lines 603, 660, 693) and `Ridge` (lines 600, 657, 690) regardless of `task_type`. Called from `_run_double_descent` (lines 515, 523, 529, 535, 541) and `_run_approximation_theory` (lines 769, 787, 797) with classification `metric_fn` → AUROC/accuracy/F1 fed continuous regressor outputs. `_fit_random_label_capacity` at line 821 already shows the correct dispatch pattern.

- **`c17_information_theoretic_decomposition.py:137, 168`** — `_rfe_select` and `_evaluate_strategy` hardcode `RandomForestRegressor`. Captures `task_type` at line 198 but never uses it. Downstream R² printed in summary calls classification `metric_fn` on regression outputs. mRMR/PCA/random branches likewise. **Fix:** dispatch RF on `task_type` or gate to regression like c04.

- **`c06_within_family_gradient.py`** — uses shared `get_model_family_variants()` (`_shared.py:997-1062`) which hardcodes `RandomForestRegressor` / `Ridge` / `KNeighborsRegressor` / `MLPRegressor` / `XGBRegressor`. On classification datasets the regressor variants are trained on integer labels and scored with classification `metric_fn`. **Fix:** add `task_type='regression'` argument to `get_model_family_variants` and switch to Classifier variants.

## P1 (incorrect-but-recoverable, methodology drift)

- **`c13_forward_compatibility.py:529-551`** — `modality_novelty` never computes. `feature_info = d_info.get('feature_info', {})` but `_eval_hybrid_per_fold` (line 264-275) never stores `'feature_info'` in `result_info` — only `'blueprint'`, `'allocation'`, `'baselines'`, `'n_features'`, `'best_result'`. `modality_novelty` is always NaN; cosine-distance diversity analysis is silently dead.

- **`c24_blueprint_transfer.py:276-277`** — Dead loop:
  ```
  for m_name, m_best in fold_blueprints.items():
      pass  # We need to know the model — store it
  ```
  `best_model_src` stays None. Cross-fold evaluation always uses RF (lines 280-282) regardless of the source fold's actually-best downstream model. The "blueprint transfer" therefore does not honor the source fold's model. **Fix:** store `best_model` alongside the blueprint when collecting it (line 184-185) and use it instead of the RF fallback.

- **`c08_calibration.py:244`** — Silent calibration-point dropout: `if int(y_cal[i]) < cal_probs.shape[1]` filters out cal points whose label index exceeds seen-class count. Fires in StratifiedGroupKFold splits where a class is missing from training data → conformal coverage computed from biased subset. **Fix:** index via `np.searchsorted(model_conf.classes_, y_cal)` and warn (don't silently drop) on truly-unseen classes.

- **`c19_importance_weighted_training.py:50-71`** — `_compute_reliability_scores` uses `y_range = max - min` and neighbor-mean averaging — nominal-on-ordinal arithmetic on multi-class integer labels. `sample_weighted` and `combined` conditions silently report meaningless numbers on multi-class. Binary borderline OK.

- **`c14_cross_domain_summary.py:251-253`** — `'hybrid_config'` reads wrong dict keys: `best_entry.get('allocation_method', '')` returns `''` because the blueprint fields live inside `best_entry['blueprint']`, not as top-level keys of `best_entry`. Same for `'model'`. The hybrid_config recorded per task is mostly empty strings.

- **`tests/experiments/_shared.py:515` (consumed by all complexity exps)** — `cv_split` silently falls back to `KFold(shuffle=True, random_state=seed)` when `groups is None` or `len(unique(groups)) < n_splits`. Per project rules random splits are forbidden; this fallback should raise/warn loudly. Can fire on partitioned subpops (c07) or small datasets.

- **`c10_noise_robustness.py:21` / `c11_ensemble_vs_hybrid.py:39`** — Unused imports left over from the pool-helper migration (`get_precomputed_search`, `get_best_for_downstream`, `_select_top_features_per_fold`, `HIGH_DIM_FILTER_THRESHOLD`). Harmless today but expands breakage surface.

## P2 (smells, future-proofing)

- **`c18_diversity_vs_depth.py:124`** — `budgets = kwargs.get('budgets', [50, 100])`. `__main__` never passes `budgets` in kwargs → budget sweep permanently locked to `[50, 100]` regardless of mode config.

- **`c23_evaluator_model_matching.py:129`** — `aug_strategy = data.get('augmentation_strategy', 'uncapped')`. `load_data` never sets that key, so this is always `'uncapped'`. Dead lookup or missing wire-through.

- **`c19_importance_weighted_training.py:247`** — String key `'best_overall'` in summary dict. Not the deprecated *evaluator* but shares the name — risk of confusion. Consider renaming to e.g. `'most_winning_condition'`.

- **`c07_data_complexity_partitions.py:274`** — `'baselines': search.baselines if best_config else {}` — `search` is only defined inside the preceding try block. Currently safe because `best_config` truthy implies search succeeded, but a later refactor could trip NameError. Capture `search = None` before the try as defensive.

- **`c13_forward_compatibility.py:67-126`** — Hand-maintained `LEGACY_REPS` lists will drift as new reps land. No test flags when a rep present in DATASET_REPS is also in (or missing from) LEGACY_REPS. Consider a `since=` tag in DATASET_REPS rather than a static legacy list.

- **`c12_modality_ablation.py:161-252`** — Reimplements `score_predictions`, `metric_name`, `higher_is_better`, `get_clf_model_factories`, `get_predict_fn` instead of using shared `get_domain_metric` + `get_model_factories`. Future metric/model fixes in `_shared` won't propagate.

- **`c02_complexity_grid.py:299`** — `skipped_already_tested = set(full_budgets) | {len(labels)}`. Including `len(labels)` as a skip-key is peculiar; the `allow_budget_exceeds_n=True` path explicitly wants budgets above n. Worth a comment explaining intent.

- **`c03_cross_dataset_transfer.py:90`** — Pool helper requests `downstream_model='svm'` to seed SVR/ElasticNet blueprints. Reasonable proxy but undocumented design choice.

## Verified-clean

- `c01_rep_characterization.py` — domain-aware metric, no hybrid calls
- `c04_importance_as_metric.py` — explicitly gated on regression (line 258-263)
- `c05_learning_curves.py` — second-pass pool-best blueprint wired (lines 79-105); random sample-size subsampling is intentional
- `c09_formal_analysis.py` — 9b/9c/9d converted to pool helper; ANOVA path uses cached c02 configs
- `c12_modality_ablation.py` — own metric stack but classification-aware throughout
- `c16_feature_selection_as_regularization.py` — classification dispatch via `_get_regularization_families(task_type)`; pool helper per budget
- `c21_single_rep_augmentation.py` — task_type wired through factories
- `c22_explainability.py` — pool helper for blueprint, task_type wired through `get_model_factories`
- `complexity_helpers.py` — registry/config, no execution paths
