# Handoff: restructure the noise experiments

Written 2026-10-30 by Claude from Adelaide's decisions in the chat of that day. Every
decision below marked "Adelaide" is hers. Everything else is a reading of the code and of
`~/Documents/ch7_noise_methods_audit.html` made that day, and is to be re-checked against the
files before it is acted on.

Read `CLAUDE.md` first. The rules that bite here: no pruning of any axis, arm, dataset or
setting for compute; local runs are debug, not results; no sbatch; one heavy local job at a
time; large files on `/Volumes/seagate`; never mention re-run or cache cost as a reason; every
stored result of a method whose code changes is on the old footing and the commit says so.

## 1. Adelaide's decisions, 2026-10-30

- The 37 numbered noise experiments are too fragmented. They are to be regrouped into a small
  number of experiments built around three lists: noise-aware training methods, detectors,
  cleanups.
- Every method that has a test and an action inside it is split: the published method runs as
  is on the noise-aware list, its test joins the detector list, its action joins the cleanup
  list. Where the detector-by-cleanup sweep already contains the published method as one cell,
  the method is not run twice; the cell is named as the published method.
- Jo-SRC (Yao 2021) is added as a published noise-aware method. The code today called
  `contrast_to_divide` is renamed: it is Jo-SRC's test used as a weight, not Zheltonozhskii 2021.
- Two combinations of Jo-SRC and DivideMix are built and tested (section 4.1).
- The hybrid representation is not the point of these experiments. Runs use the base
  representations, or a pre-computed hybrid, on a wider set of datasets. Fewer folds, more
  datasets.
- Breaking published methods into their detector and cleanup parts and recombining them is a
  research contribution of the chapter, and is to be written up as one.

## 2. Target layout

Five experiments and three standalone ones. Working names only; Adelaide names them.

| Working name | What it runs | Replaces |
|---|---|---|
| NA, noise-aware training | every method in list A, base representations, one run | n09 track A, n13 track A, n33, n40 |
| DC, detector by cleanup | every detector in list B crossed with every cleanup in list C it can pair with; published pairs are named cells | n07, n11, n23, n24, n32, n41, n42, n02, n12, n21, n25 |
| DG, detector grading | each detector against injected noise, real database disagreement, curator flags, and its ceiling | n06, n22, n31, n13 track B |
| ST, structure-specific fixes | fixes that need something only one noise kind or dataset has: censoring, grouped shifts, sources, repeat counts | n19, n26, n29, n30, n30b, n34, n35, n39 |
| MS, measurements | numbers about the noise and the data that train no cleanup | n01, n14, n16, n27, n37 |
| n15 | configuration searched on clean against damaged labels | stays |
| n20 | cleaning before against after the configuration search | stays |
| n36 | which molecules to measure again | stays |

The hybrid question of n09, does the ranking of methods change on a hybrid, becomes one value
of the feature-space axis in NA and DC, not an experiment.

## 3. Axes and controls

The two method experiments carry these. Each row says which experiment it is lifted from and
where the code is, so it is copied and not rewritten.

### 3.1 Axes

| Axis | Values | Lifted from | Code |
|---|---|---|---|
| Noise shape and level | the six decided regression kinds (gaussian, outlier_p05, grouped_shifted, censoring, shift_only_p10, sparse_dprime_up_p10), the classification flip kinds, at the decided levels; 3 draws per non-zero level, 1 under censoring; level 0 once | n13 | `noise_helpers.decided_conditions`, `decided_levels`, `inject_condition`, `draws_for`, `conditions_for_task`, `sweep_conditions` |
| Noise unit | level in label SD, or in the model's out-of-fold residual SD | n38 | `noise_helpers.inject_relative_train_labels`, `oof_residual_sd`, `level_unit` |
| Label regime | injected on clean labels; labels as delivered, nothing injected; one lab's measurement as the training label, graded against the other labs' consensus | n07 / n18 / n17 | n17 `build_entities`; n19 reuses it |
| Feature space | base_1, base_2; the two general hybrids; the importance-scaled copy | n09, n02, n12 | `noise_helpers.fold_configurations`, `fold_model_inputs`, n09 `WEIGHTED_SPACE` |
| Neighbour settings | distance family (plain, importance-weighted, forest proximity, hybrid's features), counting rule (rank, inverse square), k | n21, n12, n42 | n07 `sweep_settings`, `screen_settings`; `KNN_SEARCH_SPACE_RECORD.md` |
| Flag share and strength | flagged share 0.05, 0.10, 0.20; move strength 0.25, 0.5, 1.0; move rule proportional or confidence; n11's alpha | n07, n18, n23, n11 | n07 `FLAG_SHARES`, `flag_shares_for`; n23 `DROP_SHARES`, `MOVE_STRENGTHS`, `MOVE_METHODS` |
| Rounds | a detector-cleanup pair repeated: n32's trimmed refit at keep-shares 0.8, 0.9, 0.95 for 3 to 5 rounds; n09's multistage at 10 percent for 3 rounds | n32, n09 | n32 `itlm`, `ITLM_ALPHAS` |
| Modifier | cliff molecules put back after cleaning | n25 | n25 `cliff_mask_for_dataset`, `put_back` |

Folds: `noise_helpers.noise_outer_folds`, any count including 1 (`config['n_splits']`,
`KIRBY_NOISE_N_FOLDS`, `KIRBY_NOISE_ONLY_FOLDS`). Adelaide: fewer folds, more datasets. The
count is hers (section 8).

### 3.2 Controls

Every one of these exists today and must be an arm of DC or NA, or it is lost.

| Control | What it checks | Lifted from | Code |
|---|---|---|---|
| Matched random deletion | delete the same number of molecules at random, 5 seeds | n24 | `random_deletion_scores`, `RANDOM_DELETION_SEEDS` |
| Random put-back | put back as many non-cliff molecules as cliff molecules | n25 | `random_exempt_mask` |
| Random flags | the same number of flags at random | n41 | `random_flags` |
| Shuffled scores | the detector's scores shuffled across molecules | n42, n18 | n42 control twins; n32 `control_twins`, `random_keep` |
| Set-precision flags | flag sets whose share of true hits is set to 0, 0.25, 0.5, 0.75, 1.0 | n31 | `synthetic_flags`, `PRECISIONS` |
| Noise kind the fix should not touch | gaussian noise under a grouped-offset fix | n34 | n34 `ARMS` under `gaussian` |
| Clean QM9 | labels from a calculation; a cleanup that helps here found something other than error | n18 | n18 docstring "QM9 IS THE CONTROL" |
| Not-at-floor | move the same number of unclipped labels by the censoring lift | n30b | `not_at_floor_control` arm |
| Removal cost curve | train on a random 25, 50, 70, 80, 90 percent of molecules, every label correct, 20 draws | n27 | n27 `removal_cost` |

## 4. The three lists, with placement

Names are the audit's (`~/Documents/ch7_noise_methods_audit.html`, blocks m1 to m79). The
m-number is given so every block can be ticked off (section 7).

### 4.1 List A: noise-aware training

Each row: the published method as is; its detector part and cleanup part if it has them; and
the DC cell that reproduces it, if one does. "Runs in NA" means the method is trained as
published in the NA experiment. "DC cell" means the sweep already produces it and it is named
there instead.

| Method (audit block) | Code today | Detector part → list B | Cleanup part → list C | Runs as |
|---|---|---|---|---|
| Early learning (m59) | `lnl_methods.EarlyLearning` | none | move every label toward the network's running-average prediction | NA. Its cleanup part also goes to C as "running-average target" |
| Self-paced learning, was uncertainty curriculum (m60) | `lnl_methods.SelfPacedLearning` | loss rank from the network being trained | delete, with a kept share growing 25, 50, 75, 100 percent | NA, and DC cell: small-loss rank × delete with growing share (rounds axis) |
| DivideMix (m61) | `lnl_methods.DivideMixReg` | two-part mixture on the scaled per-molecule loss | replace flagged labels with a model's prediction | NA (two networks, refit every epoch). DC cell: mixture detector × replace-by-prediction, once |
| Meta-Weight-Net (m62) | `lnl_methods.MetaWeightNet` | weight learned from the loss on a small clean set | down-weight | NA. DC cell: meta-learned weight × down-weight |
| MentorNet PD, predefined curriculum (m63) | `lnl_methods.MentorNet`. Decision H closed 2026-10-30 (Adelaide: fix it): the mentor is now fitted before weighted training and frozen, as published, after a 10-epoch burn-in | the self-paced rule on the loss, cut-off rising each epoch (`lnl_parts.self_paced_weight`) = `net_small_loss` under the rounds axis | down-weight = `score_weight` | NA as published. DC cell: `lnl_parts.repeated_weighting`, the rule repeated every epoch with no mentor |
| MentorNet DD, data-driven (Jiang 2018) | `lnl_methods.MentorNetDD`, key `mentornet_dd`, built 2026-10-30 | a weight from a small network fitted on a trusted set with half its labels corrupted = detector `net_trusted_weight` | down-weight = `score_weight` | NA as published, and DC cell `net_trusted_weight` × `score_weight`. Trusted set defaults to the selection carve, not clean (section 8) |
| Jo-SRC weighting, was contrast-to-divide (m64) | `lnl_methods.ContrastToDivide`, rename pending | high loss plus prediction that holds still under feature jitter | down-weight at strength 3 | DC cell: Jo-SRC test × down-weight. It is not a published method and does not run in NA |
| Jo-SRC as published (Yao 2021) | not built | the jitter test above | relabel flagged molecules with a slowly updated second network | NA. Regression version of its clean test is Adelaide's decision (section 8) |
| Jo-SRC + DivideMix, combination 1 | not built | DivideMix's mixture splits; Jo-SRC's test decides which noisy molecules get guessed labels and which are dropped | | NA |
| Jo-SRC + DivideMix, combination 2 | not built | Jo-SRC's test splits; DivideMix's two networks and guessed labels train | | NA |
| Heteroscedastic GP (m65) | `lnl_methods.HeteroscedasticGP` | its noise GP = registry detector `hetero_gp` (audit m7) | per-molecule noise term = inverse-variance weight | DC cell: heteroscedastic GP spread × inverse-variance weight, GP model. Also NA as published, since the two GPs are fitted in turn |
| Evidential network (m66) | `lnl_methods.EvidentialKernel` | its own per-molecule spread | down-weight | NA. DC cell: evidential spread × down-weight |
| Co-teaching (m69) | n32 `two_learner_exchange` | small loss from the other learner | delete | DC cell: small-loss × delete, two learners. Already split in n32 |
| Small-loss selection (m41) | `lnl_methods.SmallLossSelection` | loss rank | delete | DC cell only |
| Confident learning (m42) | `lnl_methods.ConfidentLearning` (cleanlab) | cleanlab score (m11) | delete | DC cell only |
| Distance-based selection (m43) | `lnl_methods.DistanceBasedSelection` | neighbour-label gap in the fitted space | delete | DC cell only |
| Multistage cleaning (m44) | `lnl_methods.MultistageCleaning` | loss rank | delete 10 percent, 3 rounds | DC cell only, rounds axis |
| Huber loss DNN (m67) | n33 `HUBER_C` | none | none | NA |
| Each model's own robust loss (m68) | n33 `build_swaps`, `tukey_objective`, `expectile_objective` | none | none | NA |
| Yeo-Johnson target transform (m74) | n33 `LABEL_ARMS` | none | none | NA |
| Averaging fits on jittered labels (m75) | n33 `target_noise_bagging` | none | none | NA |
| Label-distance mixup (m76) | n33 `c_mixup` | none | none | NA |
| NGBoost Student-t (m77) | n33 `make_ngboost` | none | none | NA |
| Retuning (m55) | n40 `tune`, `tuned_model` | none | none | NA |
| Seed averaging (m56) | n40 `seeded_model` | none | none | NA |
| Stacking (m57) | n40 | none | none | NA |
| 1,000-tree random forest (m58) | n40 `MODEL_ARMS` | none | none | NA |
| Nearest-neighbour mean as the predictor (m78) | n40 `knn_scheme_predict` | none | none | NA |
| GP noise term from a noise estimate | n40 `GP_PAIRS`, `gamma_sigma2`, `gp_cp_choice`, `gp_cv_choice` | none | none | NA. Uses n37's Gamma estimate |
| Censored likelihood (part of m30 to m34) | n30 `fit_aft_predict`; m73 interval model | none | none | ST, censoring only |
| One row per measurement (m70) | n29 `per_record` | none | none | ST, repeat datasets |
| Log-count feature, plain and NGBoost (m71, m72) | n19 `MODEL_ARMS` | none | none | ST, repeat datasets |
| Ranking within each paper (m79) | n29; never ran | none | none | ST; record that it never ran |

### 4.2 List B: detectors

One score per training molecule. Registry: `src/kirby/utils/noise_estimation.py`
`ESTIMATOR_REGISTRY` (22 names). Detectors that score once before training are the kind DC
runs today. Detectors marked "in-loop" rescore every epoch inside a network's training; DC
needs a second detector kind for them (one hook: score after a warm-up, or every k epochs).

| Detector (audit block) | Code | Kind | Used to clean today? |
|---|---|---|---|
| Forest cross-validated residual (m1) | registry `cross_val`; n07 `DETECTOR` | once | yes, n07 |
| LightGBM cross-validated residual (m2) | n32 `oof_residual`; `kirby.utils.noisy_labels.oof_predict` | once | yes, n31, n32 |
| Neighbour-label gap (m3) | registry `neighbor_residual` | once | yes |
| Gap over neighbour spread (m4) | registry `local_variance_ratio` | once | yes |
| Cliff-aware gap (m5) | registry `activity_cliff_aware` | once | yes |
| GP spread and residual (m6) | registry `gaussian_process` | once | yes |
| Heteroscedastic GP spread (m7) | registry `hetero_gp` | once | yes, n23 |
| Variance-head network spread (m8) | registry `heteroscedastic_nn` | once | yes, n23 |
| NGBoost spread (m9) | registry `ngboost` | once | yes, n23 |
| Residual divided by a spread (m12) | n32 `_rescaled`: `ngboost_rescaled`, `quantile_forest_rescaled`, `conformal` | once | yes, n32 |
| Nearest-neighbour Shapley value (m10) | n32 `knn_shapley_values` | once | never |
| Cleanlab score (m11) | `noise_helpers.cleanlab_suspicion`; n41 | once | never on its own; n09 confident learning deletes by it |
| Conformal p-values with false-flag control (m13) | n41 `conformal_pvalues`, `benjamini_hochberg`; calibration `random_calibration`, `oracle_calibration`, `trusted_calibration` | once; gives flags | yes, n41 |
| Isolation forest on features plus residual | n41 `isolation_forest_suspicion` | once | yes, n41 |
| Local outlier factor on features plus residual | n41 `lof_suspicion` | once | yes, n41 |
| Residual with low disagreement across refits | n41 `residual_low_disagreement` | once | yes, n41 |
| Close-pair label disagreement (m25) | n35 `intercept_estimate` and the pair test | once | yes, n35 |
| Small-loss rank | `lnl_methods.SmallLossSelection`; n32 `lowest_share` | once (after warm-up) or in-loop | yes, via delete |
| Distance-based selection score | `lnl_methods.DistanceBasedSelection` | once | yes, via delete |
| DivideMix's two-part mixture on the loss | inside `lnl_methods.DivideMixReg._divide`; split out | in-loop | no |
| Jo-SRC's jitter test | inside `lnl_methods.ContrastToDivide`; split out | in-loop | yes, as a weight |
| Meta-learned weight | inside `lnl_methods.MetaWeightNet`; split out | in-loop | yes, inside the method |
| MentorNet PD weight | inside `lnl_methods.MentorNet`; = `net_small_loss` with a rising cut-off, `lnl_parts.repeated_weighting`; no new detector | in-loop | yes, inside the method |
| Trusted-set-trained weight (MentorNet DD) | registry `net_trusted_weight`; `lnl_parts.trusted_weight_scores` | in-loop | yes, inside the method |
| Evidential spread | inside `lnl_methods.EvidentialKernel`; split out | in-loop | yes, inside the method |
| Never tested on their own (m16) | registry `deep_ensemble`, `bayesian_rf`, `quantile_regression`, `quantile_forest`, `conformal`, `leave_one_out`, `ensemble`, `ensemble_disagreement`, `distance_weighted`, `distance_weighted_residual`, `cross_val_residual`, `hybrid`, `hybrid_noise` | once | no |

Also in `src/kirby/utils/noise_mitigation_methods.py`, used by no experiment: `KNNConsistencyFilter`,
`LOFFilter`, `DistanceWeightedPredictionFilter`, `NeighborhoodConsensusFilter`,
`ActivityCliffDetector`, `MahalanobisOutlierDetector`, `CVDisagreementFilter`,
`BootstrapEnsembleFilter`, `CoTeachingFilter`, `DivideMixFilter`, `ZScoreFilter`,
`PredictionErrorFilter`. Check each against the registry; a duplicate is deleted, a new one is
added to list B. Do not leave the file as a third place detectors live.

Not detectors, but settings of the neighbour detectors (neighbour-settings axis):
importance-scaled space (m14; n02, n12) and distance family (m15; n21).

### 4.3 List C: cleanups

Each reads a detector's scores or flags and changes the training set. Code: n07
`cleaned_training_set(strategy, ...)` dispatches on the strategy name.

| Cleanup (audit block) | Code | Needs | Pairs with |
|---|---|---|---|
| Delete flagged (m17) | n07 `remove` | flags | every detector |
| Score down-weight (m19) | n07 `downweight` | scores; a model that takes weights | every detector; not GP |
| Huber-shaped down-weight (m50) | n07 `huber_weight` | scores | as above |
| Exponential down-weight (m24) | n02, n11 | scores | as above |
| Top-share down-weight (m26) | n31 `apply_operation` | flags | as above |
| Inverse-variance weight (m38 part) | n07 `invvar_weight`; n33 `invvar_weights` | a spread detector | spread detectors only |
| Neighbour weights (m23) | n12, n18, n21 | neighbour detector, a space | neighbour detectors |
| Neighbour-agreement weights (m27) | n02 `compute_reliability_scores` | as above | neighbour detectors |
| Close-pair weights (m25) | n35 | the close-pair detector | that detector |
| Move toward prediction (m21) | n07 `label_smooth` | an out-of-fold prediction; regression | every detector |
| Score-scaled shrink (m22) | n07 `js_shrink` | as above | every detector |
| Replace flagged label by prediction (m48) | n31 `apply_operation('rewrite')` | as above | every detector |
| Shrink flagged label toward prediction (m49) | n32 `shrink_flagged`, `estimated_shrink_weight` | as above | every detector |
| Move toward GP prediction (m47) | n07 `gp_smooth` | a GP fit; regression | every detector |
| Move toward out-of-fold prediction (m20 part) | n07 `oof_smooth` | regression | every detector |
| Neighbour-mean smoothing (m20) | n07 `label_smooth` toward the ten-neighbour mean | a space; regression | every detector |
| Neighbour-mean replacement of flagged labels (m45) | n07 `knn_smooth` | a space; not run on classification | every detector |
| Forest-neighbour smoothing (m46) | n07 `prox_smooth` | forest proximity; regression | every detector |
| Clip the residual, winsorise (m51) | n30 `winsorise`; n41 `winsorise_flagged` | an out-of-fold prediction | every detector |
| Huber blend of label and prediction (m40) | n33 `huber_teacher_blend` | a prediction; no flags | runs on every molecule, no detector |
| Running-average target (from early learning) | split out of `lnl_methods.EarlyLearning` | a network | no detector |
| Replace flagged label with the network's guess (from DivideMix) | split out of `lnl_methods.DivideMixReg` | flags; a network | every detector |
| Relabel flagged with a slowly updated second network (from Jo-SRC) | not built | flags; a network | every detector |
| Delete with growing kept share (from self-paced) | split out of `lnl_methods.SelfPacedLearning` | scores | every detector; rounds axis |
| Five-detector delete or move (m18) | n23 | five spread detectors | those five |
| Cliff put-back (m28) | n25 `put_back` | a cliff list | a modifier on any cleanup, not a cleanup |
| Quantile clip | n07 `quantile_clip`; cut by f36 decision 2026-10-05 | | stays cut; record it |

### 4.4 Cleanups that take no detector (ST experiment)

| Fix (audit block) | Code | Needs |
|---|---|---|
| Constant lift, truncated-normal mean, censored EM, own lift, delete clipped (m30 to m34) | n30 `lift_columns`, `censored_em_column`, `rows_at_limit`; n30b on real censoring | clipped labels |
| Scaffold-family offset, mean and random intercept (m35) | n34 `ARMS` | the injector's groups |
| Source offset, source weights (m36, m37) | n39 `ARMS`; n29 document offsets | a source per measurement |
| Count and spread weights, shrink by spread (m38, m39) | n19 arms | repeat measurements |
| Per-paper quantile map (m53) | n29 `quantile_map_labels` | a paper per measurement |
| Random removal (m52) | n27 | a control, see 3.2 |
| Cleaning before the search (m29) | n20 | stays as n20 |
| Re-measurement priority rules (m54) | n36 | stays as n36 |

## 5. Where each of the 37 experiments goes

"Copy" names the pieces to lift. "Do not lose" names what disappears if it is not carried.

| Experiment | Goes to | Copy | Do not lose |
|---|---|---|---|
| n01 | MS | cliff reference per domain, `cliff_reference_independent` | the circularity flag |
| n02 | DC | reliability weights, importance-weighted space | the base-against-hybrid question → feature-space axis |
| n06 | DG | BindingDB and Lhasa/ChEMBL ground truth, rank-correlation transfer | the transfer headline, the hybrid-minus-single table |
| n07 | DC | `CLEANUPS`, `cleaned_training_set`, `flag_shares_for`, `_compute_noise_scores`, `sweep_settings`, `screen_settings` | flag-share axis; the two baselines no_noise and noisy_baseline |
| n09 | NA, DC | `LNL_REGISTRY`; the five feature spaces | the ranking-change question → feature-space axis; dnn is the model for every method |
| n11 | DC | the alpha grid, the recovery metric | strength axis; recovery share |
| n12 | DC | two neighbour counting rules | neighbour-settings axis |
| n13 | NA, DC, DG | the six kinds and the flip kinds | noise axis; track B's detection grading → DG |
| n14 | MS | cliff list, prevalence, geometry, stage-0 diagnostics | feasibility flags |
| n15 | stays | | |
| n16 | MS | rank agreement, floors, turnover | the two level-0 floors |
| n17 | DC (regime), ST | `build_entities`, `MIN_MEASUREMENTS` | the third label regime; entities never on both sides |
| n18 | DC (regime) | `FLAG_RATES`, the folds-won headline | the second label regime; QM9 control |
| n19 | ST | six training-label arms, single against averaged | the single-against-averaged test |
| n20 | stays | | |
| n21 | DC | distance families | neighbour-settings axis |
| n22 | DG | the 301 curator flags, two texts graded apart | "outside typical range" is partly the label itself |
| n23 | DC | `SOURCES`, drop and move, two move rules | the confidence rule in label units |
| n24 | DC | `random_deletion_scores`, 5 seeds | matched random deletion; per-seed reporting, no mean |
| n25 | DC | `cliff_mask_for_dataset`, `put_back`, `random_exempt_mask` | the random put-back; the paired design |
| n26 | ST, MS | error by measurement count | measurement |
| n27 | MS, control | `removal_cost`, injected propagation, relabelled | removal cost curve |
| n29 | ST | document offsets, per-record arms, known-bias check | herg_ic50 only |
| n30 | ST | `lift_columns`, `censored_em_column`, `winsorise`, `fit_aft_predict` | the censored likelihood |
| n30b | ST | negated-label trick, `not_at_floor_control` | real censoring regime |
| n31 | DG, control | `synthetic_flags`, `PRECISIONS`, predicted-against-measured AUROC | set-precision control; break-even precision |
| n32 | DC | `DETECTORS`, `itlm`, `two_learner_exchange`, `shrink_flagged`, `knn_shapley_values`, `control_twins` | rounds axis; the two-learner cell |
| n33 | NA | `LABEL_ARMS`, `HUBER_C`, `build_swaps`, `residual_kurtosis` | kurtosis beside every pass; the robust scale `1.4826 × median` |
| n34 | ST | `ARMS`, random intercept | gaussian as the negative control |
| n35 | ST, DC | `intercept_estimate`, pair test, pair weights | the noise estimate without a model |
| n36 | stays | | |
| n37 | MS | `gamma_test` per representation, the check under gaussian | negative intercepts written, not averaged |
| n38 | DC, NA (unit) | `inject_relative_train_labels`, `N38_CONDITIONS` | the noise-unit axis; `hetero_seen`, `hetero_hidden` |
| n39 | ST | injected source structure, mixed model, the 1/(…) weight | rho and tau axes |
| n40 | NA | `MODEL_ARMS`, `GP_PAIRS`, `gamma_sigma2` | `dense_oofmse` levels; paired against `fixed` |
| n41 | DC, DG | `conformal_pvalues`, `benjamini_hochberg`, three calibrations, `random_flags` | calibration-share axis; random-flag control |
| n42 | DC | `FAMILIES`, D-against-D' row, control twins | neighbour-settings axis; `CLF_NOT_RUN_CLEANUPS` |
| c10 (complexity) | MS | score drop under damage, hybrid against single | no treatment arm; keep as a measurement |

Shared code that stays where it is: `noise_helpers.py`, `noise_cache.py`, `similarity_space.py`,
`noise_ceiling_analyse.py`, `_ground_truth_providers`.

## 6. Build order

1. Rename `contrast_to_divide` to Jo-SRC weighting. Grep `thesis.tex`, `THESIS_OUTLINE.md`,
   `ch7_numbers.py`, `tables_tex.py`, `build.py`, the audit html, `NOISE_CODE_GIT_HISTORY.md`.
   Thesis row at `thesis.tex` ~3737 cites Yao 2021 and says what was left out.
2. Split the in-loop methods. For each of DivideMix, self-paced, Meta-Weight-Net, MentorNet
   (PD: no new part; DD: the trusted-set-trained weight), evidential, early learning, Jo-SRC: a detector function that takes a
   network and returns one score per molecule, and a cleanup function that takes flags or
   scores. Register the detectors in `ESTIMATOR_REGISTRY` under a kind that says "in-loop".
   Register the cleanups beside n07's `cleaned_training_set` names.
3. Build Jo-SRC as published (after Adelaide's decision on the regression clean test) and the
   two combinations. Add all three to `LNL_REGISTRY`.
4. Fold `noise_mitigation_methods.py` into the registry or delete its duplicates.
5. Build DC from n07's loop with the axes of 3.1 and the controls of 3.2. A base-only
   switch: n09 and n07 have none today; `fold_configurations` is where the five spaces come
   from, and the switch keeps base_1 and base_2 only.
6. Build NA from n09's loop with the same switch and axes. Every list A method runs; the
   DC-cell rows are not run here, they are named from DC's result.
7. DG from n06, n22, n31 and n13 track B. ST from n30, n34, n39, n19, n29, n35, n30b. MS from
   n01, n14, n16, n27, n37, c10.
8. Walk section 7 and tick every block.
9. Leave the old 37 files in place until Adelaide says the new ones replace them. The stored
   results under `/Volumes/seagate/kirby_server_results_*/noise/` are from the old files and
   stay on the old footing.

## 7. Nothing left behind: audit blocks m1 to m79

Every block must appear in exactly one row of section 4 or 5. Checked 2026-10-30:

m1–m13 → 4.2 detectors. m14, m15 → neighbour-settings axis. m16 → 4.2 never-tested row.
m17–m28 → 4.3 cleanups (m18 five-detector, m28 cliff put-back). m29 → n20. m30–m34 → 4.4
censoring. m35 → 4.4 grouped. m36, m37 → 4.4 sources. m38, m39 → 4.4 repeats (inverse-variance
weight also in 4.3). m40 → 4.3 Huber blend. m41–m44 → 4.1 DC-cell-only rows. m45–m51 → 4.3.
m52 → control 3.2. m53 → 4.4. m54 → n36. m55–m58 → 4.1 n40 rows. m59–m66 → 4.1 n09 rows.
m67, m68 → 4.1 n33 rows. m69 → 4.1 co-teaching. m70–m72 → 4.1 ST rows. m73 → censored
likelihood row. m74–m77 → 4.1 n33 rows. m78 → 4.1 n40 row. m79 → 4.1 never-ran row.

Audit section s6, the two lists of methods that raised the top score, is a reading of the old
results and is not moved. It stays as the record of what the old files found.

## 8. Decisions that are Adelaide's

- Fold count for NA and DC. One fold gives no spread; two gives one.
  DECIDED 2026-10-30 (Adelaide): all 5 folds of the existing scaffold-grouped split. The
  "fewer folds" line above was made in the interest of time and is withdrawn; the caches are
  keyed on each fold's training rows, so the 5-fold split keeps every entry.
- Which datasets. The 2026-10-29 screen in memory lists nine non-chemistry datasets that
  passed her conditions. No patient data (TCGA, ANNITIA, NHANES) anywhere.
  DECIDED 2026-11-01 (Adelaide), small molecules: QM9 HOMO-LUMO gap (qm9), QM9 dipole
  moment (qm9_mu), QM9 polarizability (qm9_alpha), hERG Ki, hERG IC50 (herg_ic50), Caco-2,
  LogD, rat microsomal clearance (adme_rlm), human microsomal clearance (adme_hlm),
  MDR1-MDCK efflux (adme_mdr1), hERG blocking (herg), and chembl_qsar (D2, factor Xa,
  Mpro). What each needs before it runs is tracked in NEW_DATASETS.md, "Small molecules
  for Papers 2 and 3"; adme_hlm and the three chembl_qsar targets joined the run list
  2026-11-01.
  Every dataset runs every experiment (complexity, OPIG, similarity, noise); each paper
  chooses which to show. Plan, before the new noise experiments are launched:
  - put adme_hlm back in the run list (cut 2026-10-09 for time only);
  - wire chembl_qsar: one dataset name per target, a loader branch with scaffold groups,
    metric rmse, out of the excluded list;
  - give herg_ic50 a registry entry and list the new noise experiments in its row.
  Antibodies DECIDED 2026-11-01 (Adelaide): flab_expression_koenig, flab_binding_koenig,
  abagym_EGFR, ddg_combined, abagym_Ang2, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic,
  ginkgo_titer, cdr_flexibility_h3, cdr_flexibility_l3, abcov_kd, tz_her2_binary,
  tz_her2_multiclass.
  humanness_therapeutic runs every experiment except the noise ones. abcov_kd gets its own
  short part in Paper 3 (labels pooled from different papers). Tracked in NEW_DATASETS.md,
  "Antibodies for Papers 2 and 3": cdr_flexibility is now two datasets, one per loop,
  which is how its authors handle it, and both are wired; tz_her2's split is open
  (its authors split randomly, so there is nothing to copy) and tz_her2_binary and
  tz_her2_multiclass stay in CUT_DATASETS until it is decided.
  flab_binding_adams and flab_expression_adams are KEPT too (Adelaide, 2026-11-01). Their
  2026-10-08 cut was for time, and the server logs of run_2026_10_06 put the time on one
  machine: folds 0 and 3 ran on the GPU machine htc-g011 and were stopped and restarted
  about every four hours, while folds 1 and 2 on ordinary compute nodes reached 802 of 840
  and 1,003 of 1,960 configurations in 22.1 hours. LEDGER HANDOFF: the Adams searches go to
  ordinary compute nodes, never to htc-g011. Numbers and the full reading are in
  NEW_DATASETS.md, "Antibodies for Papers 2 and 3".
- Base representations only, or base plus a pre-computed hybrid as one more feature space.
  DECIDED 2026-10-30 (Adelaide). NA keeps n09's feature sets, searched on each condition's
  damaged labels (a hybrid searched on clean labels and used under noise would be leakage).
  DC runs three chains per model per fold instead of n07's four:
    1. model on the base representation, detector in the base: the reference;
    2. model on the base, detector in the hybrid: against 1, does the hybrid make a
       better detector;
    3. model on the hybrid, detector in the hybrid: the best achievable; against 1, does
       the hybrid improve prediction under noise, cleaning included.
  The fourth (model on the hybrid, detector in the base) is dropped. The main table is
  chains 1 and 2, one row per cleanup; chain 3 is one extra column. A cleanup hands back
  labels, weights or a keep mask, so a cleaned set made in the hybrid space trains the
  base-representation model as it is. "Base" means the model's two best single
  representations from the per-fold search, as today.
- Jo-SRC's clean test for regression. The paper's test is the Jensen-Shannon divergence between
  a prediction and a class label. A regression version has to be chosen: the squared error on
  the z-scored label is the plain option.
- The trusted set for MentorNet DD and Meta-Weight-Net. Today Meta-Weight-Net uses the
  held-back selection rows of the training half, which carry the training labels' noise. Either
  DD uses the same rows with that caveat written beside it, or a real trusted set is named, for
  example molecules measured three or more times with a small spread on the repeat datasets.
- Names for the five experiments.
  DECIDED 2026-10-31 (Adelaide): the five new experiments are n43_noise_aware_training (NA),
  n44_detect_and_clean (DC), n45_detector_grading (DG), n46_structure_specific_fixes (ST),
  n47_noise_measurements (MS). The K-bucket naming of the same day was a misreading and is
  withdrawn. New numbers start at n43 so that n01 to n42 keep meaning the old files the
  thesis and outline cite. n15, n20 and n36 keep
  their numbers.
- Whether the old 37 files are deleted after the new ones run, or kept.
  DECIDED 2026-10-31 (Adelaide, second answer, replacing the first): the five new experiments REPLACE
  the old n-files; one copy of the old files is saved. To do once n43 to n47 all exist: move
  the 37 n-files to an archive folder beside the noise experiments (git history keeps them
  too), take them out of SINGLE_DATASET_EXPERIMENTS and the import maps; n15, n20 and n36
  stay live. Until then the n-files stay as they are.

## 9. Re-check against the files and build state, 2026-10-30 (evening)

Written by Claude after reading the code named above in that session. Each line says
what was found and what was done; "hers" marks a decision left to Adelaide.

### 9.1 Done

- Step 1, the rename: commit `3e106cd`. `ContrastToDivide` / `contrast_to_divide` is
  `JoSRCWeighting` / `josrc_weighting`. No code changed. The thesis LNL-sweep row cites
  `yao2021josrc` (added to `refs.bib`, CVPR 2021, pages 5192 to 5201) and lists what Jo-SRC
  does that the arm does not. Chapter 3's study rows keep the study's name with a TODO.
  `ch7_numbers.py` reads the stored n09 files under the old key. OPEN_THREADS A15.
- Step 2, the split: commit `f238dea`. `src/kirby/utils/lnl_parts.py` holds the arithmetic.
  Five detectors in `ESTIMATOR_REGISTRY`, listed in `IN_LOOP_DETECTORS`, each scored once after
  a 10-epoch warm-up of the suite's DNN (the warm-up small-loss selection and DivideMix both
  use): `net_small_loss`, `net_loss_mixture`, `net_josrc_test`, `net_meta_weight`,
  `net_evidential_spread`. Four cleanups as branches of n07 `cleaned_training_set`, named in
  `SPLIT_CLEANUPS` and NOT in `CLEANUPS` (n07's own sweep is unchanged): `score_weight`,
  `spread_invvar_weight`, `network_rewrite`, `ema_rewrite`. Smoke
  `tests/experiments/noise/_smoke_lnl_parts.py`, 24 checks pass. `_smoke_n09_n13.py` has two
  failures (`KeyError: 'base_reps_by_model'`) that predate this session: the smoke was last
  changed 2026-10-03 and n09's result keys moved on 2026-10-05.

### 9.2 Where the handoff above disagreed with the code

- Section 4.1, early learning. The row says its cleanup part is "move every label toward the
  network's running-average prediction". The code (`lnl_methods.EarlyLearning`) keeps no
  running average: each epoch, every molecule whose loss is above the median has its weight
  multiplied by 0.95, and the weights are divided by their mean. So its test is the loss rank
  (the same quantity as `net_small_loss`) and its action is a down-weight. Nothing new was
  split out of it; no "running-average target" cleanup exists in the repository.
- Section 4.1, self-paced learning. Its test is the loss rank of the network being trained,
  which is `net_small_loss`; its action, deleting with a kept share that grows 25, 50, 75,
  100 percent, is a training schedule, not a one-shot change to the training set, so it is
  not a `cleaned_training_set` branch. It is the rounds axis: n32's `itlm` already repeats a
  detector-cleanup pair over rounds at a kept share, and a growing share is one more schedule
  for that loop. Still to build when DC is built.
- Section 4.2, "Also in noise_mitigation_methods.py, used by no experiment". Wrong: fed01
  condition 6 and `federated_helpers` run `get_mitigation_method` (three detectors vote), and
  `representations/molecular.py` cross-checks its cliff flags with `ActivityCliffDetector`.
  Nothing was deleted. The comparison against the registry, each filter read in full:

  | Filter | What it computes | Registry equivalent | Verdict |
  |---|---|---|---|
  | KNNConsistencyFilter | regression: the spread (SD) of the k neighbours' labels, flagged above its 90th percentile; two classes: share of neighbours in the other class above one half | regression: none (the registry has the gap, `neighbor_residual`, and gap over spread, `local_variance_ratio`, not the spread alone); classification: `neighbor_residual` | regression half is a candidate for list B: "neighbour-label spread" |
  | LOFFilter | local outlier factor on the features only; no label read | none; n41 `lof_suspicion` is LOF on features plus residual | not a label detector; candidate as a feature-space outlier control |
  | DistanceWeightedPredictionFilter | gap to the inverse-distance weighted neighbour mean, flagged above 2 label SD | `distance_weighted` | duplicate (threshold shape only) |
  | NeighborhoodConsensusFilter | gap above 2 SD and neighbour spread below 1 SD | `activity_cliff_aware` (gap over 1 plus the spread ratio, continuous) | duplicate in substance (hard thresholds on the same two quantities) |
  | ActivityCliffDetector | count of close pairs with a large label gap per molecule | none; n01 does pair cliffs | a cliff list, not a noise detector (n25 puts cliffs back) |
  | MahalanobisOutlierDetector | Mahalanobis distance on the features only | none | not a label detector |
  | CVDisagreementFilter | out-of-fold forest residual, 50 trees, 5 folds | `cross_val` | duplicate |
  | BootstrapEnsembleFilter | out-of-bag residual or vote variance over 10 bootstrap forests | `ensemble` (SD across models plus residual) | duplicate in mechanism |
  | CoTeachingFilter | two forests exchanging small-loss sets for 20 rounds; rejection count | n32 `two_learner_exchange` | duplicate of n32's split |
  | DivideMixFilter | two-component mixture on the forest's out-of-bag absolute loss | `net_loss_mixture` (network, min-max scaled loss) | same test, different model and scaling |
  | ZScoreFilter | label z-score above 3, no features | none | candidate for list B: "label outlier"; cannot fire on two classes (fed01 note) |
  | PredictionErrorFilter | out-of-bag forest residual, top share | `cross_val` | duplicate |

  DECIDED 2026-10-30 (Adelaide): all three join list B. Registry names
  `neighbor_label_spread`, `label_zscore`, `feature_outlier` (noise_estimation.py). The
  originals stay in noise_mitigation_methods.py for fed01 and molecular.py.

### 9.3 Settings the split introduced, each with its reason

- `lnl_parts.WARMUP_EPOCHS = 10`: the warm-up before small-loss selection and DivideMix first
  read their test. Jo-SRC weighting as a method has no warm-up (it scores at epoch 0); the
  detector uses the shared 10.
- `lnl_parts.EMA_OMEGA = 0.99`: Jo-SRC's mean-teacher update (Yao 2021, implementation details).
- `lnl_parts.EMA_EPOCHS = 30`: the student's epochs for the one-shot `ema_rewrite`: the warm-up
  plus the plain DNN's patience of 20. No sweep behind it; a candidate for the strength axis.
- `network_rewrite` trains the guessing network on the UNFLAGGED molecules. DivideMix's networks
  train on every molecule (the noisy ones under the previous epoch's guesses); once, with no
  previous epoch, a network trained on the flagged labels would partly reproduce them.
- `net_meta_weight`'s meta-network is not checkpointed by the method, so the weight is the
  meta-network's state at the method's last epoch, read on the kept network's loss.
- `EvidentialKernel` now checkpoints its three spread heads with the backbone (before, the
  recorded epistemic spread mixed the kept backbone with the last epoch's heads). The
  prediction is unchanged.

### 9.4 Not built, and why

- Step 3, Jo-SRC as published and the two Jo-SRC + DivideMix combinations: CUT, Adelaide's
  decision 2026-10-30 after a one-fold check of both regression clean tests (decisions file,
  heading I): both lost to the plain DNN in every cell with noise on herg_ki and logd, because
  relabelling with the teacher's prediction pulls a regression fit toward the mean. The
  combinations are not built. `josrc_weighting`, `net_josrc_test` and `ema_rewrite` stay.
- MentorNet: decision H closed 2026-10-30 (fix). PD rebuilt with the mentor fitted first and frozen; DD built (`mentornet_dd`); detector `net_trusted_weight`; `repeated_weighting` is the rounds loop for an in-loop detector with a weight rule. The trusted set for DD and Meta-Weight-Net stays the author's choice (section 8). PD needs no new part (`net_small_loss` ×
  `score_weight` with a rising cut-off). DD, the trusted-set-trained weight detector and the
  published method, waits on the trusted-set decision (section 8).
- Steps 5 to 7 (DC, NA, DG, ST, MS): not started. They need the fold count, the datasets and
  the feature-space list (section 8) before the loop can be written around n07's, and they
  are hers to shape.

### 9.5 Step 5, DC, built 2026-10-31 (first commit)

`tests/experiments/noise/n44_detect_and_clean.py` (was dc_detector_by_cleanup.py, then
k02_detect_and_clean.py), run id `n44` (registered in
noise_helpers' module and name maps and HONOURS_NOISE_CONDITION; NOT in
SINGLE_DATASET_EXPERIMENTS until Adelaide says the new files replace the old). Built on
n42's unit, neighbour screen, row sink and checkpoints. What it runs:

- the three chains of section 8 (model inputs base_1, base_2, hybrid; detector space own
  or hybrid; `method_setting` carries `dspace=`);
- detectors: every registry name (26, aliases dropped, the net_* tests and the three moved
  filters included), n32's six residual detectors, n41's three, conformal flags at every
  (calibration share, q), the random control, and oracle flags at every (precision, flag
  share) where the corrupted subset is defined: 54 detector columns on a regression set;
- cleanups: 25 (n07's eleven, the four split ones, n31's three, n32's two shrinks,
  winsorise and winsorise_flagged, n11's exponential weight, n23's confidence move,
  rounds_delete); 13 on a classification set; each at its own settings (flag shares
  0.05/0.10/0.20, n25's strengths, n30's winsor c, 3 rounds);
- pairing rules in `CLEANUPS` and `_pairs_with`; neighbour detectors and cleanups under the
  kept neighbour settings (n42._Similarity); detection AUROC per score detector written
  into each unit's checkpoint; twins under KIRBY_NOISE_CONTROLS=1; cliff put-back on every D'.

Smoke `_smoke_n44.py` (one unit, 128 random molecules, rf, no neighbour axis): 4,693 rows,
0 errors, every cleanup scored, 91 minutes on this laptop. One row is one fit of the
downstream model on one cleaned set; the cache reports 3,316 distinct fits, 307 detector
computations (the rounds axis re-scores a detector twice per flag share) and 180
out-of-fold LightGBM fits for that unit.

Size, for Adelaide: one unit per (fold, level, draw, model, model input). A full-grid
regression dataset has 5 x 49 x 2 x 3 = 1,470 units, each about 3,300 distinct fits of the
downstream model with the cheapest model, before the neighbour axis adds its kept settings.
The crash check on herg_ki (local debug search settings, fold 0, gaussian) spent three hours
in the per-fold hybrid search without reaching a DC unit and was stopped; the search, not DC,
is that cost. Nothing is cut; the size decision is hers.

Not yet in DC (each a later commit): n38's noise-unit axis, n17's one-lab regime, n30b's
not-at-floor control, n27's removal cost curve, n30's lifts under censoring, n35's pair weights.

Fixed on the way: `find_best_hybrid._run_search` read an undefined name `batch_mode` (an
uncommitted edit from another session, the batch-mode switch of 2026-10-29), so every per-fold
search failed with NameError; it now reads `kirby.hybrid.BATCH_MODE`. That file carries the
other session's uncommitted edits and is left uncommitted.

### 9.6 Decisions of 2026-10-31 on DC's size (Adelaide)

- The oracle-precision flags (flag sets built from the injected noise at a set accuracy,
  n31's PRECISIONS) answer one question, how accurate a detector must be for deletion to
  help, and that result is reported FIRST in the noise paper, before any detector
  (PAPER_3_OUTLINE.md, "Can the suspect labels be found", first bullet). In DC they read no
  features, so they run in one detector space, not two; the fits per fold, condition, model
  and feature set stay. Still to move into their own first pass.
- The rounds axis (rounds_delete) is CUT from DC. Reason, from the stored n32 rows
  (kirby_results_2026-10-10b): the trimmed refit raised the top score in 10 to 18 of 37
  dataset-condition-level cells against 14 to 22 for deleting the same number at random, and
  under outliers and shifts (n42's cells) in 4 of 6 and 2 of 6 against 6 of 6 for deleting
  once. Multistage cleaning stays whole in NA. The branch remains in apply_cleanup and runs
  only when named.

### 9.7 Steps 6 and 7 built, 2026-11-02 (uncommitted)

Written by Claude. n43, n45, n46 and n47 exist, each with a smoke on random data
(`_smoke_n43.py` to `_smoke_n47.py`) that passes. None has run on a real dataset: each
needs that dataset's per-fold searches. All five (n43 to n47) are in noise_helpers' module
and name maps and in HONOURS_NOISE_CONDITION, and NOT in SINGLE_DATASET_EXPERIMENTS, so no
ledger runs them yet. n46 is in plan.py's remainder tuple and in HERG_IC50_EXPERIMENTS.

| File | Lines | Smoke |
|---|---|---|
| n43_noise_aware_training.py | 1,072 | 160 random molecules, every listed method scored, 0 errors, 314 s |
| n45_detector_grading.py | 1,610 | all 37 detectors graded or given a reason, about 4 minutes |
| n46_structure_specific_fixes.py | 1,744 | 83 rows, every fix scored or recorded "not applicable", 69 s |
| n47_noise_measurements.py | 1,678 | 123 rows, every measurement returned a value, 48 s |

Also on 2026-11-02: n44 gained n33's Huber-teacher blend as a cleanup (it was in list C and
in no new file); n44's rows now carry experiment `n44_detect_and_clean` (they carried n42's
name, because n44 builds its units with n42's class). n44's oracle flags run in their own
first pass, in the model's own space only (section 9.6).

The new files import code from the old ones (n43: n09, n33, n40; n45: n06, n22, n31; n46:
n17, n19, n26, n29, n30, n30b, n34, n35, n39; n47: n01, n14, n16, n25, n27, n37; all: n07,
n32 or n42). Moving the old files to an archive folder breaks those imports.

DECIDED 2026-11-02 (Adelaide): the old files stay where they are. Done the same day:
SINGLE_DATASET_EXPERIMENTS is now n43, n44, n45, n46, n47, n15, n20, n36 (c10 and the
other n-files are out of it, still in the import maps); plan.HERG_IC50_EXPERIMENTS is n36,
n45, n46. A new ledger plans the new experiments from now on.

## 10. QM9 as the clean base (Adelaide, 2026-11-01). To do, not yet started

Her words, kept: "In some sense, QM9 will function as any other dataset. In others, it
will be used as a comparison point."

- QM9's three targets (qm9, qm9_mu, qm9_alpha) have computed labels. They are the only
  clean data in the suite.
- Every measured dataset (hERG, Caco-2, LogD, the ADME sets, chembl_qsar) carries real
  noise as delivered, even with no noise added.
- A cleanup that raises the score on a measured dataset with no noise added is a SUCCESS.
  It found noise that was already there. Never count it as a failure.
- The ideal method also does nothing, or does slightly worse, on QM9 with no noise added.
  A method that raises QM9's score with no noise added is improving training, not removing
  noise.

What to do:
- Go through every noise experiment (n43 to n47, and the old ones until they are
  replaced). For each, mark where QM9 is one dataset among the others and where it is the
  clean comparison point. Keep the two uses apart where they need to be.
- Expect few or no changes to the experiment code. The change is in the analysis, which
  has no script yet. When that script is written, it must report each method's QM9
  no-noise result beside its no-noise result on the measured datasets, and read them as
  above.
- Paper 3's outline carries this (PAPER_3_OUTLINE.md, Methods, the controls bullet, and
  "Does acting on them help").

### 10.1 What the code does with QM9, read 2026-11-01

Found by grepping every file in `tests/experiments/noise/` for "qm9" and reading each hit.
n43, n45, n46 and n47 do not exist yet.

- Level 0 (no noise added) is the first level of every injected-noise grid:
  `noise_helpers.DECIDED_LEVELS`, `DECIDED_CENSORING_LEVELS` and
  `SCREEN_CLASSIFICATION_LEVELS` all start at 0.0. So every experiment that sweeps levels
  already writes each method's no-noise result on QM9 and on every measured dataset. The
  comparison needs no new runs.
- Rows written through `n07.dprime_rows` (n42, n44) carry a `dataset` field, so the
  analysis can split QM9 from the measured datasets by name.
- New constant `noise_helpers.CLEAN_BASE_DATASETS = ('qm9', 'qm9_mu', 'qm9_alpha')`. The
  analysis script reads it.

One row per experiment that names QM9. Every other noise experiment never names it, so
QM9 is one dataset among the others there.

| Experiment | What it does with QM9 | Use |
|---|---|---|
| n18 | writes `is_computed_label_control` into the result file | comparison point. Was true for `qm9` only; now true for all three targets (reads `CLEAN_BASE_DATASETS`) |
| n30b | `--inject-censoring`: clips QM9's training labels, where the true labels are known | comparison point, but only when run with that flag. The flag works on any regression dataset; nothing ties it to QM9 |
| n23 | docstring cites a QM9 study in `~/repos/qsar_qm_models`; its level-0 control runs on every dataset | one dataset among the others |
| n31 | docstring names qm9 as a dataset with high R^2 (0.912) | one dataset among the others |
| n09, n16 | usage examples only | one dataset among the others |
| n44 | nothing | one dataset among the others; the level-0 rows are the comparison |

Not changed: n30b's QM9 run clips the gap only (`--dataset qm9`); it is not run on qm9_mu
or qm9_alpha by any command in the repository.

## 11. Metric (Adelaide, 2026-11-01)

- RMSE stays the default for regression; AUC for hERG blocking. chembl_qsar moves from MAE
  to RMSE (NEW_DATASETS.md).
- Every fit also saves MAE, R^2, Spearman and Pearson (classification: AUC, accuracy, MCC,
  weighted F1), from the same held-out predictions, so no extra fit. Built in
  `d_vs_dprime.py` (ALL_METRICS; row fields all_D, all_Dprime, all_clean). Every
  experiment that writes its rows through `n07.dprime_rows` or `score_d_dprime` gets them:
  n42 and n44 today. n43, n45, n46 and n47 must write rows the same way.
- The old experiments that score with their own `_fit_score` (n20, n23, n24, n25, n29 to
  n32, n35, n38, n41, among others) still save one metric. They are being replaced, so they
  were not changed.

