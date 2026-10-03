# Noise search space, n01–n42

Read from the code on 2026-09-25 (branch `similarity-metrics-study`, uncommitted
working tree). Section 1.7 and every count derived from it (the ×236 entries of
section 2, n35, and sections 3.2–3.3) were updated on 2026-10-03 for the owner's
change to the similarity feature spaces; nothing else was re-read then. Every value below is a constant or a function call in the file
named. Where a count depends on the data or on a loop not traced statically, it
says **not derived** and why. Nothing here was run.

Plan section 9: the prelim results are read against this list, and the cuts are
decided by the owner. This file contains no decision rules.

---

## 1. Axes shared by several experiments

### 1.1 Noise condition

One row = one list of conditions an experiment can sweep.

| list | values | count | defined in |
|---|---|---|---|
| regression registry, in `SWEEP_ORDER` | grouped_shifted, outlier_only_p10, outlier_p10, gaussian, student_t_nu3, censoring, outlier_only_p05, outlier_p05, grouped_wider, laplace, outlier_only_p01, outlier_p01, student_t_nu5, student_t_nu10 | 14 | `kirby/noise_spec.py:132` (`REGRESSION_STRATEGIES`), order `noise_helpers.py:1009` |
| classification registry | uniform, class_imbalance, binary_asymmetric, instance_noise, class_dependent, confusion_directed | 6 | `noise_spec.py:134` |
| `conditions_for_task(data)` | the regression 14 or the classification 6, by task type | 14 or 6 | `noise_helpers.py:1016` |
| `DETECTION_IS_DEFINED` | outlier_p01/p05/p10, outlier_only_p01/p05/p10, grouped_wider, censoring, shift_only_p05, shift_only_p10, sparse_dprime_p02/p05/p10, sparse_dprime_up_p02/p05/p10 | 16 | `noise_helpers.py:580` |
| relative conditions (noiseInject) | dense_oofmse, sparse_dprime_p02/p05/p10, sparse_dprime_up_p02/p05/p10, hetero_seen, hetero_hidden | 9 | `noiseInject/relative.py:55` |
| shift-only | shift_only_p05, shift_only_p10 | 2 | `noise_helpers.py:546` |

The two grouped conditions are skipped on a dataset with no SMILES
(`sweep_conditions`, `groups_needed`).

### 1.2 Noise level

One row = one level grid. A pass is one (level, draw).

| grid | values | count | passes per condition (draws 0,1,2; one draw at level 0 on regression) |
|---|---|---|---|
| regression default `_DEFAULT_NOISE_LEVELS` | 0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5 (fraction of label SD) | 7 | 1 + 6×3 = 19 |
| `CENSORING_LEVELS` | 0, 0.10, 0.20, 0.25, 0.30, 0.40 (fraction clipped) | 6 | 1 + 5×3 = 16 |
| `CENSORING_LEVELS_DETECT` (n09, n11, n12) | 0, 0.10, 0.25, 0.40 | 4 | 1 + 3×3 = 10 |
| `CLASSIFICATION_NOISE_LEVELS` | 0, 0.05, 0.10, 0.20, 0.30, 0.40 (flip rate) | 6 | 6×3 = 18 (no single-draw rule on classification) |
| relative dense (`RELATIVE_DENSE_LEVELS` + 0) | 0, 0.1, 0.3, 1.0 (share of OOF MSE) | 4 | 1 + 3×3 = 10 |
| relative sparse (`RELATIVE_DPRIME_LEVELS` + 0) | 0, 1.0, 2.0, 4.0 (residual SDs) | 4 | 1 + 3×3 = 10 |
| shift-only deltas (`SHIFT_ONLY_DELTAS_SD`) | 0.5, 1.0, 2.0 (label SDs) | 3 | not derived (n31/n32 own loops) |
| `reference_noise_level` (single level) | 0.5 | 1 | 1 × draws |

Sources: `dataset_policy.py:429-471`, `noise_helpers.py:546-576`. `_NOISE_LEVELS`
is empty, so every dataset gets the default grid.

### 1.3 Noise draw

| value | experiments |
|---|---|
| (0, 1, 2) | n07 `NOISE_SEEDS`, n09, n11, n12 (`noise_seeds` default), n13, n20, n21, n23, n24, n27, n31, n32, n33, n34, n38, n39, n40, n41, n42 |
| (0,) | n25, n30, n35, n37 |
| 3 replicates | n16 `N_REPLICATES`, n17 `N_DRAWS`, n19 `N_DRAWS` |

### 1.4 Fold

5 everywhere the config is used (`config['n_splits']`, default
`find_best_hybrid.DEFAULT_N_FOLDS = 5`). n06 and n22 carry their own
`N_FOLDS = 5`. Folds come from `_shared.cv_split`.

### 1.5 Downstream model

`DOWNSTREAM_MODELS_NOISE` = rf, xgboost, svm, dnn, lightgbm (5), `_shared.py:565,825`.
The D-versus-D' rows add rf and svm to whatever the experiment fits
(`n07.SWEEP_ROW_MODELS`, `d_vs_dprime.DEFAULT_MODELS`). n42 adds the pass's own
model to rf and svm only when it is one of rf, svm, lightgbm, xgboost
(`n23.CACHEABLE_MODELS`), so a dnn unit has 2 row models and an xgboost or
lightgbm unit has 3.

### 1.6 Model input (the matrix the model is fitted on)

| value | what it is | experiments |
|---|---|---|
| base_1, base_2 | the model's rank-1 and rank-2 single representation in its own stored search (`_shared.noise_base_reps_by_model`, `NOISE_BASE_REP_COUNT = 2`) | n17–n41 `BASE_SPACES`, n42 |
| hybrid | the prediction-built recipe from the precompute | most experiments, n42 |
| hybrid_similarity | the similarity-built recipe | n07, n11, n12, n13, n17, n18, n21, n23, n25 and others via `SPACE_FOR_FAMILY` (per-file list not derived) |

Count per model: 3 in n42; 4 in n25 (`SPACES`); per-file elsewhere, not derived.
Whether hybrid_similarity exists is data-dependent: it does not on
tcga_brca_pam50, tz_her2_multiclass, tz_her2_binary
(`dataset_policy.SIMILARITY_OBJECTIVE_SKIP_DATASETS`).

### 1.7 Similarity setting (where neighbours are found)

`similarity_space.py`. One setting = space × weighting × distance.

| component | values | count |
|---|---|---|
| feature space | base_1, base_2, base_1_aug_uncapped, base_2_aug_uncapped, hybrid (predictive general rank 1), hybrid__general2 (predictive general rank 2), hybrid_similarity (similarity general rank 1) | 7 |
| weighting (feature spaces) | none, lightgbm_gain, treeshap, deeplift | 4 |
| distance (feature spaces) | manhattan, euclidean | 2 |
| fixed settings | tanimoto_ecfp4\|none\|tanimoto, forest_proximity\|none\|forest | 2 |
| today's behaviour | model_input\|none\|process (`n07.sweep_settings` puts it first) | 1 |

Settings: 7 × 4 × 2 + 2 = 58, plus model_input = **59** per model per fold when
every space exists. Fewer when a space is missing (no SMILES drops Tanimoto; no
augmentation representation drops the two aug spaces; no predictive rank-2
recipe drops hybrid__general2; no similarity recipe drops hybrid_similarity):
**not derived** per dataset.

Changed 2026-10-03 (owner decision): the two capped augmented spaces
(base_1_aug_capped, base_2_aug_capped) were dropped and hybrid__general2 was
added. Before that: 8 feature spaces, 66 settings, 67 with model_input, 268
(setting, k) pairs.

k: `SWEPT_KS` = 5, 9, 10, 20 → **4**. A k above the lists' length (`k_eff`) is
left out (`n07.usable_ks`), so small training folds or group exclusion can give
fewer: not derived.

(setting, k) pairs per model per fold, all spaces present: 59 × 4 = **236**.

`KIRBY_SIMILARITY_SETTINGS` narrows the settings: unset or `all` = 59;
`model_input` = 1.

Experiments whose code reads the setting list (grep of `sweep_settings()`,
`settings_from_env(`, `swept_similarity(`, `sim_settings`): n01, n02, n07, n09,
n11, n12, n13, n14, n17, n18, n20, n21, n22, n23, n24, n25, n32, n35, n36, n37,
n40, n41, n42. Which methods inside each are swept, and whether k is swept for
all of them, was taken from the code only for n07, n23, n32, n35 and n42 (read
this session); for the others it is **not derived here**.

Where a k is not read: n35's close-pair rule (a percentile of all pairs) sweeps
the setting only.

### 1.8 Controls

`KIRBY_NOISE_CONTROLS=1` writes a random twin and a shuffled twin of each D'
(`controls.py`) where a twin is defined: up to ×3 rows per D'. Undefined
(no twin) for a D' that adds rows (n33 c_mixup) and for a classification label
move. Off by default.

---

## 2. Method axes, per experiment

One row = one experiment. "Arms" are the data-changing or model-changing
variants the experiment compares; counts are from the constants named.
"Multiplier per pass" is the product of the method axes only; it multiplies the
passes (1.2), folds (1.4), models (1.5) and model inputs (1.6).

| exp | conditions | method axes (values) | multiplier per pass | similarity axis |
|---|---|---|---|---|
| c10 | `REGRESSION_STRATEGIES` (14) | stability selection per (level, fold); no D' | not derived | none (no neighbours) |
| n01 | none injected | cliff definitions; `SIMILAR_PERCENTILE = 2.0` | not derived | cliff = one of the k nearest (per n01 wiring report, not re-read) |
| n02 | none injected | `WEIGHTED_MODELS` 5; `DEFAULT_IMPORTANCE_METHODS` 2 (lightgbm_gain, deeplift); `DEFAULT_STRENGTH = 3.0`; reliability_weight, cliff_aware_weight | not derived | 2 weights × 236 |
| n06 | none (ground-truth providers) | estimators; `GP_MAX_ROWS = 1000` | not derived | none |
| n07 | `conditions_for_task` (14 / 6) | `CLEANUPS` 11 (remove, downweight, label_smooth, knn_smooth, oof_smooth, gp_smooth, quantile_clip, huber_weight, js_shrink, prox_smooth, invvar_weight; 4 regression-only); `DETECTOR` cross_val; `DETECTORS_GRADED` 2; noise_fraction 0.1; spaces × `N_TOP_MODELS` 2 × `N_TOP_REPS_PER_MODEL` 2 for the three model-fitting cleanups | not derived | label_smooth, knn_smooth × 236; prox_smooth × 4 (forest only) |
| n09 | `conditions_for_task` | `LNL_REGISTRY` 12 methods; `NOISY_SEARCH_BUDGETS` 50, 100, 250 | not derived | distance_selection × 236 (per wiring report) |
| n11 | `conditions_for_task` | `BASE_ESTIMATORS` 10; `SMOOTH_ALPHAS` 4 × `SMOOTH_METHODS` 4; `WEIGHT_STRENGTHS` 4 × `WEIGHT_METHODS` 3; `EXTRA_TARGETS` 2; `NEIGHBOR_METRICS` | not derived | the 3 neighbour estimators that take lists × 236 (hybrid estimator not swept) |
| n12 | `conditions_for_task` | `K_VALUES` 5, 10, 20; `IMPORTANCE_METHODS` 2; hybrid suffixes 3 | not derived | neighbor_residual, local_variance_ratio × 236 |
| n13 | all (`noise_types`) | `LNL_METHODS` 12; `ESTIMATORS` 2 | not derived | Track B estimator, distance_selection × 236 |
| n14 | none injected | cliff graph; `ECFP4_SIM_THRESHOLD = 0.55`; `AC_POTENCY_THRESHOLD = 2.0` | not derived | spaces model_input, rf base_1/2, tanimoto, forest × k (aug and hybrid spaces recorded unavailable) |
| n15 | its own (`noisy_search_specs`) | clean vs noisy recipe; one search per (level, draw) | not derived | none |
| n16 | `conditions_for_task` | `IMPORTANCE_METHODS` 3; `N_REPLICATES` 3; `TOP_K = 100` | not derived | none |
| n17 | none (real replicate disagreement) | `CORRECTIONS` 7 + 3 reference arms; `SPREAD_SCORES` 2; `FLAG_FRACTION = 0.10`; `N_DRAWS` 3 | not derived | detector + label_smooth, knn_smooth × 236 |
| n18 | none (labels as they come) | `ALL_CLEANUPS` 7; `FLAG_RATES` 0.01, 0.05, 0.20; `STRENGTHS` 3 each for 4 cleanups; `DETECTORS` 2; `SHUFFLE_SEEDS` 3 | not derived | detector + label_smooth, knn_smooth × 236 |
| n19 | none (real replicates) | `CONDITIONS` 10; `LABEL_BUILDS` 2; `MODEL_ARMS` 6; `N_DRAWS` 3 | not derived | none |
| n20 | default `outlier_p10`; `REGIMES` 2 | `CLEANUPS` 9; `FLAG_RATES` 0.05, 0.10, 0.20; `ARMS` 4 (search × fit on as-is/cleaned); one search per cleaned label set | not derived | only search_asis__fit_cleaned × 236 |
| n21 | `DETECTION_IS_DEFINED` (16) | `DETECTORS` 2; `K_VALUES` 3; `IMPORTANCE_METHODS` 2; `DISTANCE_METRICS_SWEPT` 2; `IMPORTANCE_EXPONENTS` 2 | not derived | shared sweep beside its own distances |
| n22 | `curator_flag` (herg_ic50 only) | 4 neighbour detectors | not derived | 4 detectors × 236 |
| n23 | `SWEPT_CONDITIONS` = 16; `REGIMES` 2 | `SOURCES` 5; `DROP_SHARES` 3; `MOVE_STRENGTHS` 3 × `MOVE_METHODS` 2 → 9 arms + none | 5 × 9 = 45 per pass per space, before the sweep | activity_cliff_aware and neighbour-mean targets × 236 |
| n24 | one, default `outlier_p10` | cleanups 6 (remove; label_smooth, knn_smooth, oof_smooth; downweight, huber_weight); `FLAG_RATES` 3; `DETECTORS` 2; `RANDOM_DELETION_SEEDS` 5 | not derived | neighbor_residual × 236 |
| n25 | one, default `outlier_only_p05`, only `DETECTION_IS_DEFINED` | 10 cleanups; 3 setting slots (`FLAG_RATES` 0.01, 0.05, 0.20 or `STRENGTHS`); `DETECTORS` 3; `ARMS` 3 | 10 × 3 × 3 × 3 = 270 per pass per space (quantile_clip has 1 slot: not exact) | neighbour detectors and smoothers × 236 |
| n26 | none (repeats) | `COUNT_BINS` 5; `INJECTED_VARIANCES` 3; `N_SPLIT_HALF_REPEATS` 20 | not derived | none |
| n27 | `gaussian` | `SHARES` 6 × `REPEATS` 20; `ALPHAS` 3 | not derived | none |
| n29 | none (documents, herg_ic50) | `ARMS` 7 | not derived | none |
| n30 | `censoring` only | `WINSOR_C` 3; lift arms 3 + `SHUFFLE_SEEDS` 3; `COMPARATORS` 2 + AFT | not derived | none |
| n31 | own 5 (`VARIANCE_CONDITIONS` 2, shift-only 2, gaussian) | `PRECISIONS` 5 × `FLAG_SEEDS` 2 × `OPERATIONS` 4; detector flags × 4 operations | not derived | none |
| n32 | `DETECTION_IS_DEFINED` (16) | `ITLM_ALPHAS` 3 (+ `RANDOM_SEEDS` 3 each); two_learner (+3); shrink 2; `DETECTORS` 6 | not derived | knn_shapley × 236 (graded only) |
| n33 | `conditions_for_task` (regression) | `LABEL_ARMS` 5; swaps from `HUBER_C` 4, `EXPECTILE_TAUS` 2, Tukey | not derived | none |
| n34 | `GROUPED_CONDITIONS` 2 | `ARMS` 5 | not derived | none |
| n35 | `gaussian` (check) | `ARMS` 3 → downweight_pairs per setting | 1 + 58 settings | 58 settings, no k |
| n36 | none (repeats) | `RULES` 5 × `BUDGET_SHARES` 3; `RANDOM_DRAWS` 3; `SEEDS` 2 | not derived | leverage rules × 236 |
| n37 | `gaussian` (check) | `SIMEX_LAMBDAS` 3 × `SIMEX_DRAWS` 3; Gamma intercept | not derived | Gamma intercept × 236 |
| n38 | `RELATIVE_CONDITIONS` 9 | `LABEL_BASES` 2; arms: winsorise (c 1.5), robust_loss, itlm (alpha 0.9), invvar_ngboost, + `RANDOM_SEEDS` 3 twins | not derived | none |
| n39 | own simulation | `RHOS` 3 × `TAU_FRACS` 2 × `COUNT_DISTRIBUTIONS` 2; `ARMS` 8 | not derived | none |
| n40 | `DEFAULT_CONDITIONS` 2 (gaussian, dense_oofmse) | `MODEL_ARMS` 7; `KNN_K_GRID` 8; `GP_NUGGET_MULT` 7; `TUNE_N_ITER` 20 | not derived | Gamma and 4 knn/GP arms × 236 |
| n41 | `DETECTION_IS_DEFINED` (16) | `DETECTORS` 5; calibrations (`CALIBRATION_SHARES` 2 + oracle + trusted); `BH_QS` 3; `ACTIONS` 2; `RANDOM_SEEDS` 3 | not derived | LOF × 236 (graded; Tanimoto and forest unavailable) |
| n42 | `conditions_for_task` (14 / 6) | section 3 | section 3 | section 3 |

Fixed constants that act as an axis of one value: `n07.REPAIR_OOF_SPLITS = 5`,
`n23.ESTIMATOR_FOLDS = 5`, `n23.NEIGHBOUR_K = 10`, `n32.ITLM_ROUNDS = 4`,
`n32.EXCHANGE_ROUNDS = 2`, `n35.CLOSE_PERCENTILE = 1.0`,
`n35.MAX_PAIR_ROWS = 12000`, every `OOF_SPLITS = 5`.

---

## 3. n42 counted

n42 (`n42_similarity_noise_sweep.py`) crosses the methods it imports with every
similarity setting and model input. The counts below assume a regression
dataset where every similarity space exists, every k is usable, the n25 cliff
reference can be built, and there are at most 12,000 training molecules.

### 3.1 Units

One unit = (fold, level, draw, model, model input).

| axis | values | count |
|---|---|---|
| condition | `conditions_for_task` | 14 |
| (level, draw) passes | 19 per condition; 16 under censoring | 13 × 19 + 16 = 263 per dataset |
| fold | `n_splits` | 5 |
| model | `DOWNSTREAM_MODELS_NOISE` | 5 |
| model input | base_1, base_2, hybrid | 3 |

Units per dataset: 263 × 5 × 5 × 3 = **19,725**.

### 3.2 D' per unit

One row = one family. "(s,k)" = the 236 (setting, k) pairs of 1.7; 232 of
them are not forest_proximity and 4 are.

| family | D' per unit | derivation |
|---|---|---|
| n07 score-free | 2 × 3 arms = 6 | quantile_clip, invvar_weight, one slot each; arms plain, cliff_exempt, random_exempt |
| n07 cross_val | 19 × 3 = 57 | remove 4 flag rates; downweight, oof_smooth, gp_smooth, js_shrink, huber_weight 3 strengths each |
| n07 cross_val, neighbour cleanups | (232 × 7 + 4 × 11) × 3 = 5,004 | label_smooth 3 + knn_smooth 4 per pair; + prox_smooth 4 under forest |
| n07 neighbour detectors | (232 × 52 + 4 × 60) × 3 = 36,912 | 2 detectors × (remove 4, downweight 3, label_smooth 3, knn_smooth 4, oof 3, gp 3, js 3, huber 3 = 26; + prox_smooth 4 under forest) |
| n02 | 236 × 2 = 472 | reliability_weight, cliff_aware_weight |
| n23 model sources | 4 × 9 = 36 to 4 × (3 + 6 × 236) = 5,676 | 4 sources × 9 arms; a source's 6 move arms are swept only when its target is the neighbours' mean, which depends on whether the estimator sets `predictions_`: **not derived** |
| n23 activity_cliff_aware | 236 × 9 = 2,124 | |
| n30 | 3 (6 under censoring at level > 0) | winsorise × 3 c; + 3 lift columns |
| n31 | 3 | remove, downweight, rewrite |
| n32 | 6 | itlm × 3 alphas, two_learner, 2 shrink columns |
| n33 | 2 | invvar_ngboost, c_mixup |
| n35 | 59 | model-input distance + 58 settings |
| n41 | up to 12 | 2 calibration shares × 3 q × 2 actions; an arm with no flag is not written |

Total per unit: 44,696 to 50,336 D' (the range is the n23 line above).
Before 2026-10-03: 50,720 to 57,128.

Row models per D': rf, svm and dnn units are fitted with rf and svm (2
models); xgboost and lightgbm units add their own model (3). Mean over the 5
models: (2 + 3 + 2 + 2 + 3) / 5 = 2.4 fits per D'. The D and clean fits are cached per
unit and model, so they add 2 × 2.4 fits per unit, not per D'.

### 3.3 Totals, per dataset, regression, controls off

| quantity | count |
|---|---|
| units | 19,725 |
| D' | 19,725 × 44,696 = 8.82 × 10⁸ to 19,725 × 50,336 = 9.93 × 10⁸ |
| downstream fits (× 2.4) | 2.12 × 10⁹ to 2.38 × 10⁹ |
| rows written (one per fit) | same as fits |
| with `KIRBY_NOISE_CONTROLS=1` | up to × 3 (not derived exactly: twins are skipped where undefined) |

Not counted above, each also per unit: the detector fits (cached per labels,
lists and k), the n23 uncertainty estimators (5 per unit), the out-of-fold
LightGBM prediction (1), itlm (up to 4 rounds × 5 folds × 3 alphas), the
two-learner exchange, the NGBoost sigma, the neighbour lists (236 per model per
pass, cached), and the hybrid and augmented builds (cached per labels).

Classification: families n30, n31, n32, n33, n35, n41 are skipped; the n07 menu
loses its 4 regression-only cleanups; n23's move arms are skipped; 6 conditions
× 18 passes. Counts not derived.
