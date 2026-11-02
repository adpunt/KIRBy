# Novelty brief: what of the 79 noise methods can be claimed

Written 2026-10-26, at the author's request. **This file is a judgement, not a measurement.**
Every method in it was graded by reading its entry in the audit at
`~/Documents/ch7_noise_methods_audit.html`, where the measured part lives: whether the original
authors' own code exists, whether this repository's code is it, and every difference between
the two with a link to the source. The grade here answers one question only: is this something
to claim, or is it too small a change to claim. It says nothing about whether the method works.

## The grades

| Grade | Meaning | Methods |
|---|---|---|
| T1 | Claimable as a contribution in its own right | 2 |
| T2 | Claimable as a stated design decision, not as a contribution | 22 |
| T3 | Not claimable | 49 |
| T4 | Cannot be claimed while the code does not do what its name says | 6 |

Two of the 79 can carry an originality claim of their own. Twenty-two are worth a named
sentence or paragraph as a stated design decision. Forty-nine cannot be claimed: either the
method is published and this code is a settings change or a move to another task, or nothing
published code was found and the change is a single expression or a single constant. Six cannot
be claimed at all while the name promises something the code does not do.

**The distinction that matters most.** "No published code does this" is not the same as "this is
a contribution". Ten of the forty-nine in the bottom grade have no published counterpart at all:
a fixed one-over-one-plus-score weight, a constant weight of a quarter on a top-share flag set,
deleting the molecules at a detection limit, one extra input feature holding a measurement count,
random removal as a control. What the audit established for these ten is that no published code
implementing them was found. Why none was found is not established: the search covered GitHub
repositories named in the papers and the published packages, and GitHub-wide keyword code search
returned HTTP 403 for every agent, so a one-line transformation sitting inside somebody else's
script would not have been seen. Each of the ten is one expression or one constant, which is the
reason this file grades them unclaimable, and that is a judgement about the size of the change,
not a measurement of the literature.

## T1 (Claimable as a contribution in its own right, 2)

Both of these are mechanisms, not parameters, and no published code implements either.
Both also appear on the list of eleven methods that need a code fix; a method can be
original and still have a defect in how it is measured. See `HANDOFF_NOISE_CODE_FIXES.md`.

### Cliff put-back

Audit anchor `#m28`. Code: `tests/experiments/noise/n25_cliff_exemption.py:838-944. random_exempt_mask is 838-865, put_back is 868-912, n_spared is 915-944. The first agent's ranges were off by one at two ends: put_back ends at 912, n_spared at 944.`

Exempting named molecules from a cleaning step. No published label-noise filter can spare a named row: imbalanced-learn exempts whole classes, cleanlab protects per-class counts, scikit-clean and NoiseFiltersR take no exemption argument at all. It is a mechanism, it is chemistry-motivated, and it comes with a matched control.

Needs a fix first: Compare the exempt-molecule count against the un-renormalised weight column, so that under the down-weighting cleanups it stops reading every exempt molecule as touched. The put-back itself needs no change.

### Cleaning before the hybrid search

Audit anchor `#m29`. Code: `tests/experiments/noise/n20_clean_before_search.py:13-19 (the four methods), :83-99 (huber_weight and invvar_weight refused), :494 (SEARCH_SCORES_ON = 'asis'), :1402-1477 (_cleaned_side, which builds the search on the cleaned rows and labels). The first agent gave 85-110 for the refusals and 1402-1478 for _cleaned_side; the right ranges are 83-99 and 1402-1477.`

Cleaning the labels before the representation search rather than after. The one published pipeline that can order these two steps runs the opposite order, and no package can even run the comparison. It is a general machine-learning point, not a chemistry one, which makes it the most portable thing in the set.

## T2 (Claimable as a stated design decision, not as a contribution, 22)

These are not contributions. Each one is a design decision worth a sentence in the text,
usually because a measurement or a recorded reason stands behind it. The failure mode to
avoid is writing any of them up as new.

### Gap over neighbor spread

Audit anchor `#m4`. Code: `src/kirby/utils/noise_estimation.py:425-497 (LocalVarianceRatioEstimator), with the weighted mean at :340-342 and the weighted standard deviation at :344-349. Read in full; the line ranges are right.`

Published source: https://github.com/metno/titanlib

The buddy-check statistic with one pass instead of iteration. The floor on the denominator has a measured reason and is worth a sentence.

### Cliff-aware gap

Audit anchor `#m5`. Code: `src/kirby/utils/noise_estimation.py:557-628 (class ActivityCliffAwareEstimator). The regression branch is :615-628. It calls the neighbour-weighting helpers at :319-348 and the neighbour search at :204-262. The line range the first agent gave is right.`

One extra term in the denominator of the published gap-over-spread statistic. A new formula, but one expression different from a method you already run as its own method.

Needs a fix first: Correct the docstring to the formula the code computes, because the code is what produced every number. The thesis has to carry the code's formula, the gap times one over one plus the neighbour-spread ratio.

### GP spread and residual

Audit anchor `#m6`. Code: `src/kirby/utils/noise_estimation.py:914-1080, class GaussianProcessEstimator, plus the out-of-fold helper at :141-197 (the first agent gave :141-190; the helper runs to 197)`

Published source: https://github.com/scikit-learn/scikit-learn

scikit-learn's process throughout. The length-scale start at the median distance has a measured reason and is worth a sentence.

### Heteroscedastic GP spread

Audit anchor `#m7`. Code: `src/kirby/utils/noise_estimation.py:1838-2021, class HeteroscedasticGPEstimator`

Ported from your own earlier repository, not from a paper. The learning-rate and passes sweep has a measured reason: rank correlation against the true noise width rose from +0.180 to +0.657.

### Conformal p-values with Benjamini-Hochberg flags

Audit anchor `#m13`. Code: `tests/experiments/noise/n41_conformal_fdr_detection.py:248-385 (conformal_pvalues, benjamini_hochberg, rescaled_residual, the three calibration rules, min_bh_flags, bh_record), with the call site at 1089-1096`

Published source: https://github.com/msesia/conditional-conformal-pvalues

Bates's p-value arithmetic on a different quantity. The three calibration rules are yours and are worth naming.

### Importance-scaled neighbor space

Audit anchor `#m14`. Code: `tests/experiments/noise/n12_importance_weighted_distance.py:291-352 (_weight_features), with the exponent constant at :287 and the importance method list at :179. Also tests/experiments/noise/n21_neighbour_distance.py:348-401 (weight_features), the same code with the exponent as an argument; the first agent's end line of 400 was one short. Both take an already z-scored feature matrix, raise each feature's importance to a power (0.5 in n12), rescale those weights so they average one across features, multiply the matrix by them, and drop every feature whose weight came out zero.`

Feature-weighted neighbours is published (Wettschereck 1997) with no code released. Computing the importance on the damaged labels is the part that is yours.

### Other detectors (not on the Chapter 7 list)

Audit anchor `#m16`. Code: `tests/experiments/noise/n41_conformal_fdr_detection.py:388-544 (joined_matrix, isolation_forest_suspicion, lof_metric_kwargs, lof_suspicion, residual_low_disagreement, detector_scores, _lof_sweep) and tests/experiments/noise/noise_helpers.py:782-816 (gamma_test)`

Published source: https://github.com/haythorn/sr

Giving a feature-space outlier detector the out-of-fold residual as one extra scaled feature turns it into a label-noise detector. Nothing published does that, and it is worth a named paragraph.

### Delete flagged

Audit anchor `#m17`. Code: `tests/experiments/noise/n07_remediation.py:957-964, the 'remove' branch inside cleaned_training_set (806-1050). The cut-off percentile is built at line 899 from the flag share; the flag shares are FLAG_SHARES = (0.05, 0.10, 0.20) at line 271 and CLF_FLAG_SHARES = (0.10,) at line 235, picked by flag_shares_for at 291-298. The detector that produces the scores is DETECTOR = 'cross_val' at line 550.`

Published source: https://github.com/cleanlab/cleanlab

Deleting flagged molecules is Brodley and Friedl, done in QSAR before. What is yours is the rule that no method may read the injected noise rate, which is a methodological point worth stating once for the whole chapter.

### Five-detector delete or move

Audit anchor `#m18`. Code: `tests/experiments/noise/n23_uncertainty_cleaning.py:1019-1044 (drop_keep_mask) and 1047-1071 (cleaned_training_set), with SOURCES at 708-709, DROP_SHARES at 764, MOVE_STRENGTHS at 767 and MOVE_METHODS at 771; tests/experiments/noise/n42_similarity_noise_sweep.py:707-722 (_n23_dprime) calls those same two functions; the label move itself is smooth_by_scores at src/kirby/utils/noise_estimation.py:2084-2134`

The standard cleanups driven by a predicted per-molecule spread rather than a neighbour comparison. Nothing published does it; the construction is a recombination, not a new mechanism.

### Neighbor-mean smoothing

Audit anchor `#m20`. Code: `tests/experiments/noise/n07_remediation.py:1103-1136, with SMOOTH_STRENGTH at n07_remediation.py:214, the neighbour constants at n07_remediation.py:227-237, the flag share at n07_remediation.py:899, and the neighbour helpers at n07_remediation.py:920-952. The neighbour search itself is make_nearest_neighbors at tests/experiments/_shared.py:3417-3436.`

Published source: https://github.com/statsmodels/statsmodels

Kernel smoothing cut to a flat neighbour mean, with a partial step scaled by the suspicion score. The quarter-step was measured on three datasets, which is what makes it statable.

### Neighbor weights

Audit anchor `#m23`. Code: `tests/experiments/noise/n12_importance_weighted_distance.py:418-462 (the score, _neighbour_suspicion), 627-629 and 768-770 (the weight), with the two constants at 158-159. tests/experiments/noise/n21_neighbour_distance.py:496-556 (the score, suspicion_from_neighbours) and 1363-1365 (the weight), with the two constants at 309-310. The weight itself is kirby.utils.noise_estimation.weights_from_scores, read at src/kirby/utils/noise_estimation.py:2178-2206. The ranges I was handed, n12:291-352 and n21:597-666, hold feature weighting and the own-distance screen. They are the wrong pointers for this method. I read them as well.`

Published source: https://github.com/metno/titanlib

Down-weighting instead of removing, at a strength that won 5 of 5 folds on two datasets in an n18 sweep. The measured strength is the statable part.

### Constant lift

Audit anchor `#m30`. Code: `tests/experiments/noise/n30_censored_likelihood.py:293-300 (limit_from_labels), :303-311 (rows_at_limit), :314-322 (residual_sd), :325-328 (mills_ratio), :331-368 (lift_columns, the constant_lift column at :350-351); call site at :840-845 and :903-912. Also tests/experiments/noise/n07_remediation.py:1345-1382.`

One shared lift for every censored molecule, where every published imputation gives each row its own. Kept as a port for reproducibility, with the correct quantity running beside it; that pairing is the statable part, not the lift.

### Censored EM

Audit anchor `#m32`. Code: `tests/experiments/noise/n30_censored_likelihood.py:371-394 (censored_em_column), :397-407 (_own_model_oof), :273 (CENSORED_EM_ROUNDS = 3), call site at :915-927. The first agent gave :371-407, which runs through a blank gap into the helper; the function itself ends at :394.`

Published source: https://github.com/harrelfe/rms

Buckley-James with a normal error and the configuration's own model, three rounds. Running it through any model is what gives the support-vector and neural-network datasets a censoring result at all, and that is worth a sentence.

### Scaffold-family offset (family mean and random intercept)

Audit anchor `#m35`. Code: `tests/experiments/noise/n34_group_random_effects.py:216-301 (_group_index 216-221, group_center_offsets 223-230, moments_tau2 232-243, random_intercept_offsets 245-301); constants at 181-188; the call site at 729-747; the held-out prediction at 368-371`

Published source: https://github.com/manifoldai/merf

The published shrinkage with the two variances by moments instead of EM, written by hand so neither merf nor gpboost is needed. Applying it to scaffold families rather than to study groups is the statable part.

### Count and spread weights

Audit anchor `#m38`. Code: `tests/experiments/noise/n19_measurement_count.py:496-521 (the variance), 524-564 (the curve-fitted variance), 567-597 (the three weights) and 692-699 (the inverse-variance weight), with the constants at 358 and n17_replicate_removal.py:215.`

Published source: https://github.com/wviechtb/metafor

Inverse-variance weighting is textbook. Estimating a molecule's measurement variance from a curve of variance against measurement count is not, and that part is worth naming.

### Huber blend

Audit anchor `#m40`. Code: `tests/experiments/noise/n33_robust_training.py:526-528 (the blend, inside label_arm_predict at 506), 603-607 (the teacher), 314-317 (the scale) and 241-242 (the two constants), with the inner fold count at 245. The range I was handed, n33:328-425, holds the loss swaps, which are a different method; I read it as well.`

Published source: https://github.com/microsoft/LightGBM

LightGBM's own Huber loss, with the downstream model fitted on half the recorded label plus half the robust teacher's out-of-fold prediction. Nothing published blends the two, and it is worth a named paragraph.

### Move toward GP prediction (gp_smooth; not on the Chapter 7 list)

Audit anchor `#m47`. Code: `tests/experiments/noise/n07_remediation.py:1266-1343, calling GaussianProcessEstimator.estimate at src/kirby/utils/noise_estimation.py:958-1079, with that class's own defaults at noise_estimation.py:925-927 and GP_SMOOTH_MAX_ROWS at n07_remediation.py:388`

Published source: https://github.com/scikit-learn/scikit-learn

scikit-learn's process, with the length scale started at the median distance because the package default returned one constant per fold on 1,280 features. The measured reason is the statable part.

### Retuning

Audit anchor `#m55`. Code: `tests/experiments/noise/n40_model_side_regularisation.py:244-250 (the tuner's settings), 258-295 (TUNE_GRIDS), 359-366 (tuned_model), 369-406 (tune), 1047-1084 (where the retuned and tuned_clean_reused methods are fitted)`

Published source: https://github.com/scikit-learn/scikit-learn

sklearn's random search with five things changed around it. Putting the model's own defaults in as candidate zero came from a measured finding and is worth a sentence.

### Meta-Weight-Net

Audit anchor `#m62`. Code: `src/kirby/utils/lnl_methods.py:1437-1625 (class MetaWeightNet), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/xjtushujun/meta-weight-net

The bilevel step runs. Measuring the meta loss on a carve of the training molecules, because this data has no clean labels to hold out, is a real adaptation and worth naming.

### Each model's own robust loss

Audit anchor `#m68`. Code: `tests/experiments/noise/n33_robust_training.py:295-311 (which swap replaces which model), 314-318 (the scale s = 1.4826 times the median absolute residual), 328-350 (the Tukey objective), 352-365 (the expectile objective), 367-387 (the median-centring wrapper), 389-426 (make_swap). The rf_l1 forest comes from src/kirby/find_best_hybrid.py:2960-2969. Reused at tests/experiments/noise/n38_noise_relative_to_model_error.py:269-276 and 876-895, with the threshold fixed at 1.5 units of s.`

Published source: https://github.com/microsoft/LightGBM

Mostly the libraries' own losses at a threshold in units of the robust residual scale. Tukey's biweight and the expectile loss are written out in your file, and the floor on the second derivative has a stated reason a boosted tree makes necessary.

### Two-learner selection (co-teaching)

Audit anchor `#m69`. Code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:1244-1270 (two_learner_exchange), with 1171-1194 (subset_loss) and 1197-1205 (lowest_share), 251-252 (2 exchange rounds, LightGBM and a support vector machine), 240 (the swept shares) and 1924-1940 (the call site). Also src/kirby/utils/noise_mitigation_methods.py:1030-1085 (CoTeachingFilter), which n32 says at its lines 22-24 that this directory does not use.`

Published source: https://github.com/bhanML/Co-teaching

Co-teaching with two different model families rather than two copies of one network, and neither as the predictor. That combination is not published and is worth naming.

### XGBoost accelerated-failure-time interval model

Audit anchor `#m73`. Code: `tests/experiments/noise/n30_censored_likelihood.py:428-456 (fit_aft_predict), settings at :250-255, the error scale at :903, call site at :1026-1029. Also tests/experiments/noise/n19_measurement_count.py:451-468 (interval_bounds), :471-493 (fit_interval_aft_predict), call site at :1434-1447.`

Published source: https://github.com/dmlc/xgboost

xgboost's own survival objective. The exponential label map, which makes the normal apply to the label rather than a shifted copy, was worked out and checked here and is worth a sentence.

## T4 (Cannot be claimed while the code does not do what its name says, 6)

Do not report a result under these names until the code is fixed or the method is renamed.
The fix for each is in `HANDOFF_NOISE_CODE_FIXES.md`.

### Confident learning

Audit anchor `#m42`. Code: `src/kirby/utils/lnl_methods.py:659-745, class ConfidentLearning, read in full. Three networks are trained on all the training molecules, seeds differing by one, epochs capped at 100. Classification: the per-molecule error is one minus the mean predicted probability of the given class, and the spread is the standard deviation across the three networks, averaged over classes. Regression: the error is the absolute gap between the label and the mean of the three predictions, and the spread is the standard deviation of the three. A molecule is flagged when its error is above the 85th percentile of all training molecules' errors AND its spread is below the 25th percentile. Flagged molecules are dropped and a network is refitted on the rest, with the whole drop undone if fewer than 10 molecules would remain.`

Published source: https://github.com/cleanlab/cleanlab

Filed under confident learning without the per-class thresholds or the count table. Not claimable under that name, and the package it names is already installed.

Needs a fix first: Call cleanlab's find_label_issues, which this repository already installs for another detector, instead of the two hand-set percentiles. That makes the method the published method rather than a percentile cut under its name.

### Uncertainty curriculum

Audit anchor `#m60`. Code: `src/kirby/utils/lnl_methods.py:1075-1232, class UncertaintyCurriculum, read in full. One fifth of the training molecules are carved off to choose the kept checkpoint; the fractions below are of the four fifths that remain. The first 20 epochs train on all of them. The remaining 180 epochs are split into four stages of 45 epochs each. In each stage every molecule's loss is recomputed at the start of each epoch, and only the lowest-loss share trains that epoch: a quarter of the molecules in the first stage, then half, then three quarters, then all of them, never fewer than 10 molecules. Training stops when the carved molecules' loss has not improved for 20 epochs. The score that orders the molecules is each molecule's own loss; no spread across models or draws is computed anywhere in the class, despite the name.`

Published source: https://doi.org/10.1145/1553374.1553380

Named for a spread it never computes. Not claimable under that name; it is self-paced learning.

Needs a fix first: Compute a spread across the seeds or dropout draws the class already runs and order the curriculum by that, or rename the method self-paced, since it currently orders by training loss. Separately, four stages of 45 epochs against an early-stopping patience of 20 epochs means a run can finish before the later stages are reached.

### DivideMix

Audit anchor `#m61`. Code: `src/kirby/utils/lnl_methods.py:1239-1434 (class DivideMixReg), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/LiJunnan1992/DivideMix

DivideMix without the half that defines it. Not claimable under that name.

Needs a fix first: Train on the high-loss molecules with guessed targets, as the published code does, or drop the name and call the method loss-mixture co-selection. Line 1372 drops them from the epoch instead, and that is the step the published method is built on.

### MentorNet

Audit anchor `#m63`. Code: `src/kirby/utils/lnl_methods.py:1628-1833 (class MentorNet), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/google/mentornet

The learned curriculum is gone. The collapse that forced the change is recorded and is itself worth reporting as a negative result.

Needs a fix first: Supply a target weight per molecule and fit the mentor against it before student training, or rename the method to self-paced weighting. Reverting the objective is not advised: the commit that changed it says the old one collapsed.

### Contrast-to-divide (renamed Jo-SRC weighting, 2026-10-30)

Renamed by the author 2026-10-30: the code is unchanged and is reported as Jo-SRC's (Yao 2021) in-distribution-noisy test used as a down-weight, key `josrc_weighting`. The lines below are the record under the old name.

Audit anchor `#m64`. Code: `src/kirby/utils/lnl_methods.py:1836-2006 (class ContrastToDivide), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/ContrastToDivide/C2D

Neither the contrast nor the second network. Not claimable under that name.

Needs a fix first: Load pretrained encoder weights and run the second network, or rename the method, because the contrast the method is named for is the part that is missing. The substitute selection rule shares no computation with the published one.

### Heteroscedastic GP (LNL method)

Audit anchor `#m65`. Code: `src/kirby/utils/lnl_methods.py:2013-2147, class HeteroscedasticGP, plus the shared defaults at :427-441 and the downstream fit at :514-542 (the first agent gave :2012-2145)`

Published source: https://github.com/LiuHaiTao01/SVSHGP

The two halves are not coupled, so the method the name points at is not what runs.

Needs a fix first: Feed the predicted noise back into the Gaussian process so the two halves are coupled, as in the paper. n19 already holds gp_per_molecule_noise, a process fitted with a known noise per molecule, so the piece exists in the repository.

## T3 (Not claimable, 49)

### Forest residual

Audit anchor `#m1`. Code: `src/kirby/utils/noise_estimation.py:780-843 (LeaveOneOutEstimator) and :728-777 (EnsembleDisagreementEstimator). Both go through the fold engine at :141-197 (_oof_apply). Read in full; the first agent's line ranges are right.`

Published source: https://github.com/cran/randomForest

Breiman's out-of-bag residual with a different tree count; the divide by the label spread is a rescaling.

### LightGBM residual

Audit anchor `#m2`. Code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:1084-1089 (detector_scores, the oof_residual branch). The prediction it subtracts is made at tests/experiments/noise/noise_cache.py:554-574, which calls src/kirby/utils/noisy_labels/core.py:151-186 (oof_predict) and :98-103 (make_model), with the parameters at :32-42. Also src/kirby/utils/noise_estimation.py:635-725 (CrossValResidualEstimator), the same idea with a random forest instead of LightGBM. Read in full; the line ranges are right.`

Published source: https://github.com/cran/rgnoisefilt

Cross-validated filtering returned as a score instead of a vote. The published method is the same method.

### Neighbor-label gap

Audit anchor `#m3`. Code: `src/kirby/utils/noise_estimation.py:361-422 (NeighborResidualEstimator), with the neighbour search at :204-262, the counting scheme at :319-337 and the weighted mean at :340-342. Read in full; the line ranges are right. The default distance is manhattan (src/kirby/similarity/distance_config.py:68, :142-144).`

Published source: https://github.com/scikit-learn-contrib/imbalanced-learn

Edited nearest neighbours as a score. Published many times, including in QSAR.

### Variance-head network

Audit anchor `#m8`. Code: `src/kirby/utils/noise_estimation.py:1083-1201, class HeteroscedasticNNEstimator`

Published source: https://github.com/martius-lab/beta-nll

beta-NLL rewritten rather than called. An exponential head in place of a softplus head is an implementation choice.

### NGBoost spread

Audit anchor `#m9`. Code: `src/kirby/utils/noise_estimation.py:1704-1834, class NGBoostEstimator`

Published source: https://github.com/stanfordmlgroup/ngboost

The package's own spread, read at a shorter schedule.

### Nearest-neighbor Shapley value

Audit anchor `#m10`. Code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:995-1076 (knn_shapley_values); the neighbour count KNN_SHAPLEY_K is set at line 255`

Published source: https://github.com/AI-secure/Shapley-Study

Jia's exact recursion with a squared label gap in place of the class-match indicator. The recursion is theirs and the swap is what regression requires.

### Cleanlab score

Audit anchor `#m11`. Code: `tests/experiments/noise/noise_helpers.py:762-779 (cleanlab_suspicion)`

Published source: https://github.com/cleanlab/cleanlab

cleanlab's own scoring, called. The sign flip is not a contribution.

### LightGBM residual divided by a spread (NGBoost, quantile forest, conformal interval)

Audit anchor `#m12`. Code: `src/kirby/utils/noise_estimation.py:1521-1599 (QuantileForestEstimator), 1601-1699 (QuantileRegressionEstimator), 1704-1835 (NGBoostEstimator); tests/experiments/noise/n41_conformal_fdr_detection.py:275-289 (rescaled_residual)`

Published source: https://github.com/stanfordmlgroup/ngboost

Three published packages at different settings; the score built on top is a ratio of two of their outputs.

### Distance families for neighbors (plain, importance-weighted, forest proximity, hybrid's features)

Audit anchor `#m15`. Code: `src/kirby/utils/noise_estimation.py:2234-2256 (forest_leaf_ids) and :2258-2277 (forest_proximity_distance), with the constants at :2228-2231. The settings are chosen in tests/experiments/noise/n07_remediation.py:1906-1981 (screen_settings). The space names are listed in tests/experiments/noise/similarity_space.py:125-147 and enumerated at :326-365; the first agent's pointer of :270-364 covered the SimilaritySetting class rather than the list. The Tanimoto space is built at :634-652 and used at :780-802; the forest space is rebuilt at :655-679. Read in full.`

Published source: https://github.com/cran/randomForest

Breiman and Cutler's proximity, with the out-of-bag restriction dropped. Dropping a restriction is not a contribution.

Needs a fix first: Count a pair of molecules only in the trees that left both out of the bootstrap, and divide by that count, which is what Breiman and Cutler's code does. Separately, an unparseable SMILES is given methane's fingerprint, which makes it a near neighbour of everything.

### Score down-weight

Audit anchor `#m19`. Code: `tests/experiments/noise/n07_remediation.py:966-971, which calls weights_from_scores at src/kirby/utils/noise_estimation.py:2178-2206`

A fixed 1/(1+score) weight. Nothing published uses it because it is too simple to publish.

### Move toward prediction

Audit anchor `#m21`. Code: `tests/experiments/noise/n07_remediation.py:1247-1264, calling oof_predict at src/kirby/utils/noisy_labels/core.py:151-186 and smooth_by_scores at src/kirby/utils/noise_estimation.py:2084-2135`

Published source: https://doi.org/10.48550/arXiv.1412.6596

Blending a label toward a prediction is Reed 2014. The group-aware inner folds are your pipeline rule, not a method.

### Score-scaled shrink

Audit anchor `#m22`. Code: `src/kirby/utils/noise_estimation.py:2084-2135`

Four ways to mix a label with a target. No published method behind it and nothing in it to claim.

### Exponential down-weight

Audit anchor `#m24`. Code: `src/kirby/utils/noise_estimation.py:2178-2206.`

Three generic weight shapes with a mean normalisation. Too small to claim.

### Close-pair weights

Audit anchor `#m25`. Code: `tests/experiments/noise/n35_near_duplicate_disagreement.py:262-293 (close pairs), 296-312 (the distance bins), 315-332 (the noise estimate), 368-372 (the flag) and 375-379 (the weight), with the four constants at 199-204 and the flagging call at 870-872.`

Published source: https://doi.org/10.1007/BF01413858

The delta test of Pi and Peterson under swept distances. The published step it is built on is the one currently bypassed.

Needs a fix first: Pass the zero-distance intercept the file already computes to the flag, instead of the closest bin's value. It is a one-line change at n35:870, and the intercept is the published test's defining step.

### Top-share down-weight

Audit anchor `#m26`. Code: `tests/experiments/noise/n31_regime_and_breakeven.py:384-389 (detector_flags) and 411-424 (apply_operation, the 'downweight' branch), with DOWNWEIGHT_TO = 0.25 at line 256 and FLAG_SHARES = (0.05, 0.10, 0.20) at line 261`

A constant weight of a quarter on a top-share flag set. Too small to claim, and the quarter has no recorded reason.

### Neighbor-agreement weights

Audit anchor `#m27`. Code: `tests/experiments/noise/n02_sample_weighting.py:222-252 (the cliff-aware weight), 255-311 (the feature weighting) and 313-362 (the agreement weight), with the constants at 137 and 172 and the cliff-aware call at 569-571. The estimator the first one drives is src/kirby/utils/noise_estimation.py:557-628, and its weight step is get_weights at 100-120, with the weight itself on line 119.`

Two generic weight forms, neither published and neither substantial. The two currently disagree with each other.

Needs a fix first: Pass the same lean and the same scaling to both weights in this file, so the two methods differ in one thing rather than two. As it stands the cliff-aware call leans 2 to 1 and the agreement weight leans 4 to 1.

### Truncated-normal mean

Audit anchor `#m31`. Code: `tests/experiments/noise/n30_censored_likelihood.py:325-328 (mills_ratio) and :353-354 (the trunc column inside lift_columns). The first agent gave :325-358 and :355-357; the trunc column is two lines earlier than that.`

Published source: https://github.com/scipy/scipy

The truncated-normal mean, verified to agree with scipy to fifteen decimal places. It is the textbook quantity.

### Own lift

Audit anchor `#m33`. Code: `tests/experiments/noise/n30_censored_likelihood.py:347-348 (the per_molecule_lift column inside lift_columns) and :358-363 (its shuffled twins), with mills_ratio at :325-328 and residual_sd at :314-322. The first agent gave :352-354 and :359-363; per_molecule_lift is five lines earlier than that.`

Neither the Tobit mean nor the Buckley-James imputation: it exceeds the truncated mean by the gap between the limit and the prediction. Kept so an older findings file reproduces. Not claimable, and the file already says so.

### Delete clipped

Audit anchor `#m34`. Code: `tests/experiments/noise/n30_censored_likelihood.py:913 (the method itself, reached only when some training molecule sits at the limit and the residual spread is finite), reading rows_at_limit at 303-311 and limit_from_labels at 293-301, with MIN_SHARE_AT_LIMIT = 0.01 at line 290`

Deleting molecules at a detection limit. The nondetect literature discusses it and advises against it.

### Source offset

Audit anchor `#m36`. Code: `tests/experiments/noise/n29_document_bias.py:340-368 (ridge_offsets), 414-440 (moment_variances), 442-491 (fit_document_offsets), 334-337 (eligible_documents), 408-411 (offsets_per_record); constants at 220-249. tests/experiments/noise/n39_correlated_sources.py:315-372 (offsets_als 315-339, offsets_mixedlm 341-355, fit_offsets 357-372); constants at 217-220`

Published source: https://github.com/fmpr/CrowdLayer

Additive per-source bias, published as annotator bias and as CrowdLayer. Fitting it outside the model is an implementation choice.

### Source weights

Audit anchor `#m37`. Code: `tests/experiments/noise/n39_correlated_sources.py:315-339 (the fallback), 341-355 (the statsmodels call), 357-371 (which of the two runs), 396-398 (the weight) and 401-416 (the two variance pieces), with the constants at 217-223 and the call sites at 958 and 969-983.`

Published source: https://github.com/statsmodels/statsmodels

statsmodels's mixed model called as published. The weight around it is a one-line inverse.

### Shrink by spread

Audit anchor `#m39`. Code: `tests/experiments/noise/n19_measurement_count.py:600-621 (the label shrink) and 664-689 (the variance shrink), with the prior constant at 292 and the call sites at 1298-1300 and 1330.`

Published source: https://github.com/bioc/limma

limma-style moderation with the prior weight fixed by hand instead of fitted. Fixing a fitted quantity is a simplification, not a contribution.

### Small-loss selection

Audit anchor `#m41`. Code: `src/kirby/utils/lnl_methods.py:568-656, class SmallLossSelection, read in full, plus the retrain helper _default_predict at lines 514-542 and the class defaults at 427-442. One network trains on every training molecule for 10 warmup epochs. The per-molecule loss is then computed once, in eval mode, over all training rows at once: squared error on z-scored labels for regression, cross-entropy for classification. The molecules with the smallest loss are kept, 70 out of every 100 of them (selection_rate=0.7), never fewer than 10 molecules. A fresh network is then fitted on the survivors through _default_predict, at the class default of 200 epochs.`

Published source: https://github.com/bhanML/Co-teaching

One network picking its own low-loss molecules once. That is small-loss filtering, which predates co-teaching.

### Distance-based selection

Audit anchor `#m43`. Code: `src/kirby/utils/lnl_methods.py:748-830, class DistanceBasedSelection, read in full, plus tests/experiments/noise/n09_lnl_survey.py:253-306, class _SchemeDistanceSelection. Features are scaled, then each molecule's 10 nearest neighbours are found, itself excluded, under the process-wide distance (distance_config.sklearn_kwargs, manhattan by default). Regression: the suspicion score is the gap between the label and the mean of the neighbours' labels, divided by the spread of the neighbours' labels. Classification: the score is the share of the 10 neighbours carrying a different class. Molecules above the 85th percentile of that score are dropped, so 15 of every 100 training molecules go, and a network is refitted on the rest. The n09 subclass weights each neighbour by rank or by inverse square distance when the sweep hands a weighting scheme over, and leaves everything after the score unchanged.`

Edited nearest neighbours again, at a fixed removal share. No paper named in the code and nothing new in it.

### Multistage cleaning

Audit anchor `#m44`. Code: `src/kirby/utils/lnl_methods.py:837-923, class MultistageCleaning, read in full. Three rounds. Each round fits a network on the molecules still surviving, epochs capped at 100, then scores those same molecules: squared error for regression, minus the log probability of the given class for classification. The 10 in every 100 with the largest score are deleted, at least one molecule per round, and the rounds stop early if fewer than 10 molecules remain. After the third round a network is fitted on the survivors, 73 of every 100 training molecules.`

Published source: https://doi.org/10.48550/arXiv.2512.04319

Three rounds of top-share removal. MANTRA released no code, but what runs here is simpler than MANTRA, not different from it.

### Neighbour-mean replacement of flagged labels (knn_smooth; not on the Chapter 7 list)

Audit anchor `#m45`. Code: `tests/experiments/noise/n07_remediation.py:1139-1168, with the flag share at n07_remediation.py:221, 235 and 291-298, and the percentile at n07_remediation.py:899`

Published source: https://github.com/scikit-learn-contrib/imbalanced-learn

Wilson editing with the label overwritten instead of the row removed. The 5-in-100 share has a measured reason; the method is published.

### Forest-neighbour smoothing (prox_smooth; not on the Chapter 7 list)

Audit anchor `#m46`. Code: `tests/experiments/noise/n07_remediation.py:1170-1236, calling forest_leaf_ids at src/kirby/utils/noise_estimation.py:2234-2254 and forest_proximity_distance at noise_estimation.py:2257-2277, with FOREST_TREES and FOREST_MIN_SAMPLES_LEAF at noise_estimation.py:2228-2229 and PROX_SMOOTH_K and PROX_SMOOTH_SHARE at n07_remediation.py:227-228`

Published source: https://github.com/cran/randomForest

Forest proximity as a smoothing kernel, suggested by an AI-written review. The proximity is published and here it is missing its out-of-bag restriction.

### Replace flagged labels by the prediction (n31 rewrite and bayes_shrink; not on the Chapter 7 list by these names)

Audit anchor `#m48`. Code: `tests/experiments/noise/n31_regime_and_breakeven.py:411-433, with 'rewrite' itself at n31_regime_and_breakeven.py:422-425 and the flag built by detector_flags at n31_regime_and_breakeven.py:384-389`

Published source: https://doi.org/10.48550/arXiv.1412.6596

Replacing a flagged label by the prediction. Too small to claim, and it came from an AI-written report.

### Shrink flagged labels toward the prediction (n32 and n42 shrink_flagged; not on the Chapter 7 list by this name)

Audit anchor `#m49`. Code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:1273-1302, with the flag built by lowest_share at n32_iterative_and_dual_selection.py:1198-1205`

Published source: https://doi.org/10.1080/01621459.1975.10479864

A James-Stein shape on mean squares. The reason mean squares replaced variances is recorded and is worth a line in the text, but the rule itself is too small to claim.

### Huber-shaped down-weight (huber_weight; not the Chapter 7 "Huber blend")

Audit anchor `#m50`. Code: `tests/experiments/noise/n07_remediation.py:973-998`

Published source: https://github.com/statsmodels/statsmodels

Huber's own weight rule and tuning constant, applied to a detector score instead of a model residual.

### Clip the residual (winsorise; not on the Chapter 7 list)

Audit anchor `#m51`. Code: `src/kirby/utils/noise_estimation.py:2138-2175 (clip_to_range) and tests/experiments/noise/n30_censored_likelihood.py:410-425 (winsorise), with the sweep of c at n30_censored_likelihood.py:232 and the clip_to_range caller at tests/experiments/noise/n07_remediation.py:1383-1400`

Published source: https://github.com/scipy/scipy

Clipping the residual rather than the label. A variant of winsorising, with the widths taken from an AI-written report.

### Random removal of training molecules (n27; a control, not on the Chapter 7 list)

Audit anchor `#m52`. Code: `tests/experiments/noise/n27_removal_cost_and_propagation.py:186-204 (removal_seed and kept_rows), with SHARES at 151 and REPEATS = 20 at 154; tests/experiments/noise/n24_delete_or_keep.py:426-445 (random_deletion_scores) and 447-462 (replicate_seed), with RANDOM_DELETION_SEEDS = (0, 1, 2, 3, 4) at 262 and the method list at 408-425`

Random removal as a control. It is a control, not a method, and it should be presented as one.

### Per-paper quantile map (n29; not on the Chapter 7 list)

Audit anchor `#m53`. Code: `tests/experiments/noise/n29_document_bias.py:530-551 (quantile_map_labels); threshold at 236-237; the mapped labels averaged per molecule at 598-600`

Published source: https://github.com/bmbolstad/preprocessCore

Quantile normalisation (Bolstad 2003) with a pooled reference. Changing what the reference is built from is a settings choice.

### Re-measurement priority rules (n36; not on the Chapter 7 list)

Audit anchor `#m54`. Code: `tests/experiments/noise/n36_remeasure_priority.py:327-352 (rank_candidates) and 354-358 (labels_after_budget); the two neighbour quantities at 224-263 (neighbour_stats) and 266-277 (neighbour_stats_from_lists); the rule list at 188; N_NEIGHBOURS = 5 at 200`

Published source: https://doi.org/10.1145/1401890.1401965

Five re-measurement rules read off Sheng 2008, whose code was not found. The leverage score is invented in the file and has no measured backing.

### Seed averaging

Audit anchor `#m56`. Code: `tests/experiments/noise/n40_model_side_regularisation.py:297-303 (SEED_AVG_N and SEED_SUBSAMPLE), 414-418 (seeded_model), 1084-1101 (the seed_one and seed_avg methods). Ten fits of the same model on the same training molecules, differing only in random_state, and the ten held-out predictions averaged. seed_one keeps the first fit, so the averaging is not confounded with the subsampling. What makes two seeds differ is set per model: lightgbm gets row subsampling 0.8 with bagging every iteration and feature subsampling 0.8, xgboost gets row subsampling 0.8 and nothing else, and dnn gets no subsampling at all, so only its weight initialisation and its batch order move.`

Averaging ten fits over seeds. Standard practice with no single originating implementation.

### Stacking

Audit anchor `#m57`. Code: `tests/experiments/noise/n40_model_side_regularisation.py:307 (OOF_SPLITS), 574-580 (_oof_model), 1107-1113 (the out-of-fold predictions), 1117-1130 (the stack). Out-of-fold predictions of the dataset's decided models go into non-negative least squares (scipy.optimize.nnls) against the noisy training labels; the weights are applied to the same models' held-out predictions from the fixed method.`

Published source: https://github.com/scikit-learn/scikit-learn

Stacking with a non-negative combiner instead of a ridge one. A combiner swap inside a published method.

### 1,000-tree RF

Audit anchor `#m58`. Code: `tests/experiments/noise/n40_model_side_regularisation.py:304 (RF_BIG_TREES) and 1100-1106 (the rf_1000_trees method), with the forest it starts from at src/kirby/find_best_hybrid.py:2757-2763. The method is that registry forest with the tree count set to 1,000 and everything else left alone: sklearn's RandomForestRegressor with max_features 'sqrt', random_state 42 and n_jobs -1.`

Published source: https://github.com/cran/randomForest

A forest with a thousand trees instead of a hundred. A setting.

### Early learning

Audit anchor `#m59`. Code: `src/kirby/utils/lnl_methods.py:926-1072, class EarlyLearning, read in full, plus the carve helper at lines 469-501. One fifth of the training molecules are carved off to choose the kept checkpoint. Every molecule starts with a weight of one. Each epoch trains on the weighted loss, weighted squared error for regression or weighted cross-entropy for classification. After each epoch every training molecule's loss is recomputed, and the weight of each molecule above the median loss is multiplied by 0.95 (gamma). Weights are then clipped to between 0.01 and 1.0 and divided by their own mean. Training stops when the carved molecules' loss has not improved for 20 epochs, the class default, out of at most 200 epochs.`

Published source: https://github.com/shengliu66/ELR

The cited paper released no code. What runs is a 0.95 decay on above-median losses, which is simpler than ELR, not different in kind. Defensible as a stated simplification, not as a contribution.

### Evidential network

Audit anchor `#m66`. Code: `src/kirby/utils/lnl_methods.py:2149-2303, class EvidentialKernel, plus the base-class defaults at :427-441 and the carve at :469-503`

Published source: https://github.com/aamini/evidential-deep-learning

Amini's loss and penalty term for term. A constant penalty weight in place of their ramp is a settings choice.

### Huber loss

Audit anchor `#m67`. Code: `tests/experiments/noise/n33_robust_training.py:214 (HUBER_C), :294-311 (build_swaps), :314-317 (robust_scale), :367-386 (_Centred), :389-425 (make_swap), :573-601 (where s is set per fold and matrix, at :593), :907-922. Also tests/experiments/noise/n30_censored_likelihood.py:246, src/kirby/find_best_hybrid.py:2560-2578, and src/kirby/models/dnn.py:120-125 and :169-176.`

Published source: https://github.com/microsoft/LightGBM

The libraries' own Huber losses. Setting the threshold from the robust residual scale is a sensible default, not a contribution.

### One row per measurement

Audit anchor `#m70`. Code: `tests/experiments/noise/n29_document_bias.py:592-616 (arm_training_sets); the method is 'per_record': (X_rec_tr, y_rec_tr) at line 612; described at 28-32`

Training on measurement rows. Under squared loss it differs from the molecule-mean method only by weighting a molecule by its measurement count, as the docstring says.

### Log-count feature

Audit anchor `#m71`. Code: `tests/experiments/noise/n19_measurement_count.py:702-716 (with_count_feature), 1340-1342 (where the two matrices are built) and 1356-1358 (where the count_feature condition picks them). The natural logarithm of the molecule's measurement count becomes one extra input feature, standardised on the training molecules, cast to the matrix's own dtype.`

One extra input feature. Too small to claim.

### NGBoost with log-count feature

Audit anchor `#m72`. Code: `tests/experiments/noise/n19_measurement_count.py:1422-1429 (the ngboost_plain and ngboost_count_feature methods), fed by with_count_feature at 702-716. The model itself comes from src/kirby/find_best_hybrid.py:2936-2942.`

Published source: https://github.com/stanfordmlgroup/ngboost

The package at a different learning rate, with one extra feature.

### Yeo-Johnson target transform (not on the Chapter 7 list)

Audit anchor `#m74`. Code: `tests/experiments/noise/n33_robust_training.py:508-512, the yeo_johnson branch of label_arm_predict. The transformer is fitted on the fold's noisy training labels, the model is fitted on the transformed labels, and the held-out predictions go back through inverse_transform.`

Published source: https://github.com/scikit-learn/scikit-learn

sklearn's transform at its own defaults.

### Averaging fits on jittered labels (target_noise_bagging; not on the Chapter 7 list)

Audit anchor `#m75`. Code: `tests/experiments/noise/n33_robust_training.py:519-525 (the target_noise_bagging branch), with the two settings at 236-237 and the residual spread it reads at 593-594. Five fits, each on the training labels plus fresh Gaussian noise whose spread is half the standard deviation of the out-of-fold residual, with the five held-out predictions averaged. No row is resampled.`

Published source: https://doi.org/10.1023/A:1007682208299

Averaging fits on jittered labels. Smearing is old and nothing here departs from it.

### Label-distance mixup (c_mixup; not on the Chapter 7 list)

Audit anchor `#m76`. Code: `tests/experiments/noise/n33_robust_training.py:464-495 (c_mixup) and 516-518 (the method), with the two settings at 231-232.`

Published source: https://github.com/huaxiuyao/C-Mixup

Yao's C-Mixup built once instead of per batch, with the original molecules kept. A simplification of a published method.

### NGBoost Student-t (not on the Chapter 7 list)

Audit anchor `#m77`. Code: `tests/experiments/noise/n33_robust_training.py:428-440 (make_ngboost), the two constants at :291-292, and the call site at :973-979 (the first agent gave :428-441 and :974-982)`

Published source: https://github.com/stanfordmlgroup/ngboost

The package's own Student-t likelihood at a different learning rate.

### Nearest-neighbour mean as the predictor (not on the Chapter 7 list)

Audit anchor `#m78`. Code: `tests/experiments/noise/n40_model_side_regularisation.py:469-472 (knn_scheme_predict) and 1187-1218 (the kNN methods, one per neighbour setting, neighbour count and counting scheme), calling tests/experiments/noise/similarity_space.py:448-458 and src/kirby/similarity/neighbour_weighting.py:158-186 (the zero-distance guard), 189-197 (one over distance to a power) and 259-272 (weight_array). The two counting schemes swept are 'rank' and 'inverse_square' (similarity_space.py:170) and the neighbour counts are 5, 10 and 20.`

Published source: https://github.com/scikit-learn/scikit-learn

A neighbour mean with two counting schemes, read from the swept representation spaces. The spaces are the contribution, not this method.

### Ranking within each paper (not on the Chapter 7 list)

Audit anchor `#m79`. Code: `tests/experiments/noise/n29_document_bias.py:554-583 (rank_within_documents); RANKER_PARAMS at 241-244`

Published source: https://github.com/dmlc/xgboost

xgboost's own ranker with a different objective string.

## What this means for splitting the papers

On this reading the noise work carries two originality claims, not a shelf of them. Cliff
put-back is chemistry-motivated and will read as a chemistry contribution. Cleaning before the
representation search is a general machine-learning point about where feature selection sits
relative to label cleaning, and it is the more portable of the two.

Everything else is a methods chapter: a wide, honest survey of published noise methods applied
to molecular labels, with each departure from the published code stated. That is a real
contribution of a different kind, and the twenty-two graded T2 are what make it one, but none of
them is a claim on its own.

The six graded T4 are a liability until they are fixed or renamed, because each reports a number
under a published name that the code does not earn.

## How to use this with the other files

- `~/Documents/ch7_noise_methods_audit.html` - the measured comparison, per method, with links.
- `thesis_provenance/NOISE_CODE_GIT_HISTORY.md` - when each difference was written and whether
  the commit gave a reason.
- `thesis_provenance/METHOD_PROVENANCE.md` - the earlier comparison against the papers' text.
- `HANDOFF_NOISE_CODE_FIXES.md` - the eleven fixes, cheapest first.
