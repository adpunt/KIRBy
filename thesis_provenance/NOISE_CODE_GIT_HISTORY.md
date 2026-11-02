# Noise methods: when each difference from the published code was written

Written 2026-10-26. One section per method, in the order of the audit at
`~/Documents/ch7_noise_methods_audit.html`. The differences themselves are in that file;
this file records only what git says about when each one entered the repository and whether
the commit gave a reason.

The judgement line on each method is an agent's reading, not a measured result.
Of the 273 commits traced across all 79 methods, 155 carry no reason in the message.


| Method | Judgement | Shortcuts | Stronger | Commits traced |
|---|---|---|---|---|
| Forest residual | Moved to another task | 4 | 4 | 4 |
| LightGBM residual | A deliberate change | 4 | 4 | 4 |
| Neighbor-label gap | A deliberate change | 2 | 5 | 5 |
| Gap over neighbor spread | A deliberate change | 4 | 4 | 4 |
| Cliff-aware gap | Fixed 2026-10-26 (docstring) | 4 | 5 | 6 |
| GP spread and residual | A deliberate change | 5 | 7 | 4 |
| Heteroscedastic GP spread | A deliberate change | 4 | 6 | 2 |
| Variance-head network | A deliberate change | 4 | 4 | 2 |
| NGBoost spread | Settings only | 2 | 5 | 4 |
| Nearest-neighbor Shapley value | Moved to another task | 2 | 4 | 4 |
| Cleanlab score | Settings only | 2 | 3 | 2 |
| LightGBM residual divided by a spread (NGBoost, quantile forest, conformal interval) | Settings only | 3 | 4 | 6 |
| Conformal p-values with Benjamini-Hochberg flags | Moved to another task | 3 | 4 | 4 |
| Importance-scaled neighbor space | A deliberate change | 3 | 5 | 4 |
| Distance families for neighbors (plain, importance-weighted, forest proximity, hybrid's features) | Fixed 2026-10-26 (out-of-bag proximity) and 2026-10-29 (unparseable SMILES skipped) | 4 | 5 | 5 |
| Other detectors (not on the Chapter 7 list) | Moved to another task | 3 | 5 | 4 |
| Delete flagged | A deliberate change | 2 | 4 | 6 |
| Five-detector delete or move | A deliberate change | 3 | 5 | 5 |
| Score down-weight | A deliberate change | 3 | 3 | 4 |
| Neighbor-mean smoothing | A deliberate change | 3 | 4 | 4 |
| Move toward prediction | A deliberate change | 2 | 4 | 3 |
| Score-scaled shrink | A deliberate change | 4 | 2 | 2 |
| Neighbor weights | A deliberate change | 3 | 5 | 4 |
| Exponential down-weight | A deliberate change | 3 | 3 | 2 |
| Close-pair weights | Fixed 2026-10-26 | 3 | 5 | 3 |
| Top-share down-weight | A deliberate change | 3 | 4 | 2 |
| Neighbor-agreement weights | Fixed 2026-10-29 (same scaling and lean for both weights) | 4 | 4 | 6 |
| Cliff put-back | Fixed 2026-10-26 | 2 | 6 | 6 |
| Cleaning before the hybrid search | A deliberate change | 2 | 5 | 6 |
| Constant lift | A deliberate change | 4 | 5 | 3 |
| Truncated-normal mean | Settings only | 0 | 4 | 2 |
| Censored EM | A deliberate change | 4 | 5 | 3 |
| Own lift | A deliberate change | 4 | 4 | 4 |
| Delete clipped | A deliberate change | 2 | 4 | 3 |
| Scaffold-family offset (family mean and random intercept) | A deliberate change | 4 | 7 | 6 |
| Source offset | A deliberate change | 4 | 7 | 5 |
| Source weights | A deliberate change | 3 | 5 | 3 |
| Count and spread weights | A deliberate change | 3 | 5 | 3 |
| Shrink by spread | A deliberate change | 3 | 4 | 3 |
| Huber blend | A deliberate change | 3 | 5 | 3 |
| Small-loss selection | A deliberate change | 4 | 2 | 3 |
| Confident learning | Fixed 2026-10-29 (calls cleanlab) | 4 | 2 | 3 |
| Distance-based selection | Moved to another task | 3 | 4 | 5 |
| Multistage cleaning | A deliberate change | 3 | 2 | 3 |
| Neighbour-mean replacement of flagged labels (knn_smooth; not on the Chapter 7 list) | A deliberate change | 3 | 4 | 4 |
| Forest-neighbour smoothing (prox_smooth; not on the Chapter 7 list) | A deliberate change | 3 | 5 | 5 |
| Move toward GP prediction (gp_smooth; not on the Chapter 7 list) | A deliberate change | 3 | 5 | 5 |
| Replace flagged labels by the prediction (n31 rewrite and bayes_shrink; not on the Chapter 7 list by these names) | A deliberate change | 2 | 4 | 2 |
| Shrink flagged labels toward the prediction (n32 and n42 shrink_flagged; not on the Chapter 7 list by this name) | A deliberate change | 2 | 5 | 1 |
| Huber-shaped down-weight (huber_weight; not the Chapter 7 "Huber blend") | Moved to another task | 3 | 4 | 3 |
| Clip the residual (winsorise; not on the Chapter 7 list) | A deliberate change | 2 | 5 | 3 |
| Random removal of training molecules (n27; a control, not on the Chapter 7 list) | A deliberate change | 2 | 4 | 5 |
| Per-paper quantile map (n29; not on the Chapter 7 list) | A deliberate change | 3 | 5 | 2 |
| Re-measurement priority rules (n36; not on the Chapter 7 list) | Moved to another task | 4 | 6 | 5 |
| Retuning | A deliberate change | 1 | 6 | 3 |
| Seed averaging | A deliberate change | 2 | 2 | 1 |
| Stacking | A deliberate change | 1 | 4 | 1 |
| 1,000-tree RF | Settings only | 1 | 1 | 2 |
| Early learning | A deliberate change | 3 | 5 | 4 |
| Uncertainty curriculum (now Self-paced learning) | Renamed and fixed 2026-10-29 | 4 | 4 | 4 |
| DivideMix | Fixed 2026-10-30 (author's decision F): trains on the noisy set with guessed labels | 5 | 4 | 4 |
| Meta-Weight-Net | A deliberate change | 5 | 4 | 4 |
| MentorNet | Fixed 2026-10-30 (author's decision H): mentor fitted before training and frozen; DD variant added | 5 | 5 | 5 |
| Contrast-to-divide (now Jo-SRC weighting) | Renamed 2026-10-30 (author): the code is Jo-SRC's noisy-sample test as a weight; unchanged | 6 | 5 | 3 |
| Heteroscedastic GP (LNL method) | Fixed 2026-10-30 (author's decision E): runs Kersting 2007 | 5 | 5 | 4 |
| Evidential network | A deliberate change | 4 | 5 | 4 |
| Huber loss | A deliberate change | 5 | 6 | 5 |
| Each model's own robust loss | A deliberate change | 2 | 6 | 3 |
| Two-learner selection (co-teaching) | Moved to another task | 3 | 4 | 2 |
| One row per measurement | Moved to another task | 2 | 3 | 2 |
| Log-count feature | A deliberate change | 0 | 4 | 2 |
| NGBoost with log-count feature | Settings only | 1 | 1 | 2 |
| XGBoost accelerated-failure-time interval model | Moved to another task | 4 | 5 | 4 |
| Yeo-Johnson target transform (not on the Chapter 7 list) | A deliberate change | 1 | 2 | 1 |
| Averaging fits on jittered labels (target_noise_bagging; not on the Chapter 7 list) | A deliberate change | 1 | 2 | 1 |
| Label-distance mixup (c_mixup; not on the Chapter 7 list) | Moved to another task | 2 | 5 | 2 |
| NGBoost Student-t (not on the Chapter 7 list) | Settings only | 3 | 4 | 1 |
| Nearest-neighbour mean as the predictor (not on the Chapter 7 list) | A deliberate change | 1 | 4 | 2 |
| Ranking within each paper (not on the Chapter 7 list) | Settings only | 2 | 5 | 3 |

## The eleven that need fixing

One row per method. "Git records" says whether a commit message gives the reason for
the change that broke the method. The fixes themselves are written up as a handoff in
`HANDOFF_NOISE_CODE_FIXES.md` in the repository root.

| Method | Git records | What to fix |
|---|---|---|
| Cliff-aware gap | Git records no reason for this | Correct the docstring to the formula the code computes, because the code is what produced every number. The thesis has to carry the code's formula, the gap times one over one plus the neighbour-spread ratio. |
| Distance families for neighbors (plain, importance-weighted, forest proximity, hybrid's features) | Git records no reason for this | Count a pair of molecules only in the trees that left both out of the bootstrap, and divide by that count, which is what Breiman and Cutler's code does. Separately, an unparseable SMILES is given methane's fingerprint, which makes it a near neighbour of everything. |
| Close-pair weights | Git records no reason for this | Pass the zero-distance intercept the file already computes to the flag, instead of the closest bin's value. It is a one-line change at n35:870, and the intercept is the published test's defining step. |
| Neighbor-agreement weights | Git records part of this as deliberate | Pass the same lean and the same scaling to both weights in this file, so the two arms differ in one thing rather than two. As it stands the cliff-aware call leans 2 to 1 and the agreement weight leans 4 to 1. |
| Cliff put-back | Git records this change as deliberate | Compare the exempt-molecule count against the un-renormalised weight column, so that under the down-weighting cleanups it stops reading every exempt molecule as touched. The put-back itself needs no change. |
| Confident learning | Git records no reason for this | Call cleanlab's find_label_issues, which this repository already installs for another detector, instead of the two hand-set percentiles. That makes the arm the published method rather than a percentile cut under its name. |
| Uncertainty curriculum | Git records no reason for this | Compute a spread across the seeds or dropout draws the class already runs and order the curriculum by that, or rename the arm self-paced, since it currently orders by training loss. Separately, four stages of 45 epochs against an early-stopping patience of 20 epochs means a run can finish before the later stages are reached. |
| DivideMix | Git records no reason for this | Train on the high-loss molecules with guessed targets, as the published code does, or drop the name and call the arm loss-mixture co-selection. Line 1372 drops them from the epoch instead, and that is the step the published method is built on. |
| MentorNet | Git records this change as deliberate | Supply a target weight per molecule and fit the mentor against it before student training, or rename the arm to self-paced weighting. Reverting the objective is not advised: the commit that changed it says the old one collapsed. |
| Contrast-to-divide | Git records no reason for this | Load pretrained encoder weights and run the second network, or rename the arm, because the contrast the method is named for is the part that is missing. The substitute selection rule shares no computation with the published one. |
| Heteroscedastic GP (LNL method) | Git records no reason for this | Feed the predicted noise back into the Gaussian process so the two halves are coupled, as in the paper. n19 already holds gp_per_molecule_noise, a process fitted with a known noise per molecule, so the piece exists in the repository. |

## Forest residual

Her code: `src/kirby/utils/noise_estimation.py:780-843 (LeaveOneOutEstimator) and :728-777 (EnsembleDisagreementEstimator). Both go through the fold engine at :141-197 (_oof_apply). Read in full; the first agent's line ranges are right.`

Published source: https://github.com/cran/randomForest

Her out-of-bag average is scikit-learn's, not Breiman and Cutler's, so the forest grows 200 trees where randomForest grows 500, and a leaf may hold one molecule where randomForest stops at five. The out-of-bag residual is then divided by the standard deviation of the labels to make a suspicion score, which neither published file computes. No reason is recorded for the tree count or the leaf floor; the commit that wrote them is titled only "Label estimation".

**Judgement: Moved to another task.** The out-of-bag arithmetic is scikit-learn's unchanged; what is new is reusing the residual as a per-molecule suspicion score.

**Shortcuts taken.**
- The default path reads scikit-learn's oob_prediction_ without checking how many trees left each molecule out (src/kirby/utils/noise_estimation.py:826-834).
- A molecule that no tree left out gets a prediction of zero from scikit-learn, and her code then scores its whole distance from zero as suspicion (src/kirby/utils/noise_estimation.py:836).
- The leaf floor stays at scikit-learn's one molecule, where randomForest stops a regression node at five molecules, with no reason recorded (src/kirby/utils/noise_estimation.py:826-832).
- EnsembleDisagreementEstimator adds the spread of ten forests to the absolute residual with equal weight and no recorded reason (src/kirby/utils/noise_estimation.py:769).

**Stronger than the published code.**
- When scaffold groups are passed every fold is group-disjoint, so no scaffold sits on both sides of a fold (src/kirby/utils/noise_estimation.py:141-197).
- Passing exclude_groups replaces out-of-bag with group-disjoint folds, so a molecule's prediction never sees its own repeat measurements, and the docstring gives that reason (src/kirby/utils/noise_estimation.py:800-824).
- A molecule no fold covered is filled with the mean of the covered molecules, not zero, so it cannot look like the cleanest molecule (src/kirby/utils/noise_estimation.py:186-196).
- The number of folds actually used is recorded on the estimator (src/kirby/utils/noise_estimation.py:808, :814).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The out-of-bag forest was set to 200 trees and the absolute residual divided by the standard deviation of the labels. | `8b20490` | 2026-01-09 | no reason in the message |
| The group-disjoint fold path was added, replacing out-of-bag whenever exclude_groups is passed; up to 20 folds, each fitting its own 200-tree forest. | `9b288a0` | 2026-09-04 | no reason in the message for this path |
| Every model-based detector was routed through the shared out-of-fold helper, which is where the group-disjoint folds come from. | `7a98aa1` | 2026-08-19 | "so every row is scored by a model that never trained on it" |
| The sum of ten forests' prediction spread and the absolute residual was written as one score. | `8b20490` | 2026-01-09 | no reason in the message; this line has existed in one commit only, written once and never revised |

## LightGBM residual

Her code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:1084-1089 (detector_scores, the oof_residual branch). The prediction it subtracts is made at tests/experiments/noise/noise_cache.py:554-574, which calls src/kirby/utils/noisy_labels/core.py:151-186 (oof_predict) and :98-103 (make_model), with the parameters at :32-42. Also src/kirby/utils/noise_estimation.py:635-725 (CrossValResidualEstimator), the same idea with a random forest instead of LightGBM. Read in full; the line ranges are right.`

Published source: https://github.com/cran/rgnoisefilt

The published filter has ten tree models predict every molecule and removes one on six disagreeing votes of ten. Her detector predicts each molecule once, from the fold that held it out, and returns the absolute residual as a continuous score. The reason is in the code: the docstring asks for "one suspicion score per training molecule", and n32 grades those scores as a ranking against the molecules whose labels moved, so no vote or threshold is needed.

**Judgement: A deliberate change.** The committee vote and the removal threshold are replaced by one continuous residual that is graded as a ranking, which the surrounding code shows is deliberate.

**Shortcuts taken.**
- Each molecule is predicted once, by the single fold-model that held it out, so there is no agreement count across models (tests/experiments/noise/n32_iterative_and_dual_selection.py:1089).
- The LightGBM settings are fixed by hand at 600 boosting rounds, learning rate 0.05, 63 leaves and 90 percent of molecules and features, with no reason recorded (src/kirby/utils/noisy_labels/core.py:32-42).
- The n32 branch divides the residual by nothing, while the random forest version of the same idea divides by the standard deviation of the labels, so the two residual detectors are on different scales (n32:1089 against src/kirby/utils/noise_estimation.py:720-721).
- Neither the features nor the labels are scaled before the residual is taken, where the published filter min-max scales both to the unit interval (n32:1086-1089).

**Stronger than the published code.**
- The out-of-fold prediction refuses to run without groups rather than splitting molecules at random, and says so in the error (src/kirby/utils/noisy_labels/core.py:113-128).
- It raises if any molecule was left unpredicted, so partial coverage cannot pass as a prediction (src/kirby/utils/noisy_labels/core.py:184-188).
- Folds are GroupKFold, so a scaffold group is never on both sides, where the published filter splits molecules into ten random folds (src/kirby/utils/noisy_labels/core.py:134-145).
- The out-of-fold draw is cached on hashes of the features, labels, groups and the thread count, and reused across many noise experiments rather than refitted (tests/experiments/noise/noise_cache.py:554-574).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The LightGBM settings were written: 600 rounds, learning rate 0.05, 63 leaves, 90 percent of molecules and features per round. | `87ac6be` | 2026-07-10 | no reason in the message; this line has existed in one commit only, written once and never revised |
| Random fold splitting from the ported housing study was replaced by group-aware folds. | `87ac6be` | 2026-07-10 | "random splits are banned and leak scaffold analogues" |
| The oof_residual detector was added to n32 alongside three other new detectors. | `1e7b813` | 2026-09-24 | "from the second research report" — an AI-written report, not a paper |
| The surrogate's thread count was changed from every core to the repository's switch; run time only, not scores. | `9cb3f43` | 2026-09-19 | "ran on every core on the cluster: 467x slower, measured" |

## Neighbor-label gap

Her code: `src/kirby/utils/noise_estimation.py:361-422 (NeighborResidualEstimator), with the neighbour search at :204-262, the counting scheme at :319-337 and the weighted mean at :340-342. Read in full; the line ranges are right. The default distance is manhattan (src/kirby/similarity/distance_config.py:68, :142-144).`

Published source: https://github.com/scikit-learn-contrib/imbalanced-learn

The published method removes a molecule when its nearest neighbours disagree with it; hers removes nothing and returns one score per molecule. The neighbour count is ten rather than three, the search is manhattan rather than Euclidean, and the regression gap is divided by the standard deviation of the labels rather than compared to a fixed threshold on min-max scaled labels. The docstring records a reason for the classification branch only; no reason is recorded for the neighbour count or the distance.

**Judgement: A deliberate change.** The comparison of a label to its neighbours' labels is intact; removal and the fixed threshold are replaced by a continuous score that the suite grades as a ranking.

**Shortcuts taken.**
- The estimator applies no feature scaling and searches whatever matrix the caller hands it, where the published regression form min-max scales features first; nothing here checks (src/kirby/utils/noise_estimation.py:386-389).
- The default of ten neighbours has no reason recorded in the file or in any commit message (src/kirby/utils/noise_estimation.py:375).

**Stronger than the published code.**
- A molecule never takes a neighbour sharing its exclude_groups label, so repeat measurements of one drug-target pair are not its own nearest neighbours, and the docstring gives that reason (src/kirby/utils/noise_estimation.py:204-262).
- The code raises when the largest group leaves fewer than one neighbour, rather than silently scoring on fewer neighbours (src/kirby/utils/noise_estimation.py:250-255).
- The classification branch reports the share of disagreeing neighbours, because distance from a fractional average of class codes measures nothing (src/kirby/utils/noise_estimation.py:361-372).
- minkowski with p of one half reaches sklearn as a metric name plus p, with a comment saying the single string would silently become Euclidean (src/kirby/utils/noise_estimation.py:227-230).
- Neighbours can be weighted by rank or by one over squared distance, which the published code does not do (src/kirby/utils/noise_estimation.py:319-337).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The estimator was written with ten neighbours, the absolute gap to the neighbour mean, and division by the standard deviation of the labels. | `8b20490` | 2026-01-09 | no reason in the message; these lines have existed in one commit only, written once and never revised |
| The classification branch replaced a residual on class codes with the share of disagreeing neighbours. | `1421b3d` | 2026-05-26 | "class-disagreement scoring instead of nonsensical value-residuals on integer class labels" |
| Molecules sharing an exclude_groups label were dropped from every neighbour list, replacing the plain skip-yourself rule. | `9b288a0` | 2026-09-04 | no reason in the message; the reason is in the docstring, repeat measurements of one pair |
| The counting scheme, rank or one over squared distance, reached the detectors. | `84e6ce2` | 2026-10-06 | "The four neighbour detectors read the counting scheme; detector grading runs under both schemes" |
| The distance became whatever resolve_distance_metric returns, which is manhattan when nobody passes a name. | `7546bae` | 2026-09-10 | no reason in the message |

## Gap over neighbor spread

Her code: `src/kirby/utils/noise_estimation.py:425-497 (LocalVarianceRatioEstimator), with the weighted mean at :340-342 and the weighted standard deviation at :344-349. Read in full; the line ranges are right.`

Published source: https://github.com/metno/titanlib

The published check divides the gap by an inflated spread, square root of the neighbour variance plus that variance over the neighbour count, and repeats the whole pass until no station is newly flagged. Hers divides by the plain standard deviation of the neighbour labels and runs one pass, returning a score instead of a flag. The floor on the denominator has a recorded reason: without it a molecule in a near-constant neighbourhood scores around ten billion, which dominates the later min-max normalisation and collapses every other molecule's score toward zero.

**Judgement: A deliberate change.** The gap over local spread is kept and the floor has a stated, measured reason; the inflation term, the radius and the repeat pass are dropped.

**Shortcuts taken.**
- One pass only: a suspect molecule stays in its neighbours' spread, where the published check drops flagged stations from later neighbour lists and repeats until nothing new is flagged (src/kirby/utils/noise_estimation.py:483-497).
- The denominator is the plain standard deviation of the neighbour labels, not the published inflated spread, and no comment or commit message says why the inflation term was dropped (src/kirby/utils/noise_estimation.py:485).
- The ten nearest molecules are taken however far away they are, where the published check requires a minimum count inside a radius and refuses the station otherwise (src/kirby/utils/noise_estimation.py:481-484).
- The spread is computed from the same neighbour labels the molecule is judged against, so one wrecked neighbour widens the denominator that would have exposed it (src/kirby/utils/noise_estimation.py:485-494).

**Stronger than the published code.**
- Both denominators are floored, with the comment giving the reason: an unfloored ratio dominates the min-max normalisation applied downstream (src/kirby/utils/noise_estimation.py:473-480, :486-492).
- The regression floor is a hundredth of the label standard deviation, so it carries the label's own unit, where the published check takes an absolute number from the caller (src/kirby/utils/noise_estimation.py:492).
- The classification branch divides disagreement by one minus the neighbour purity, floored at one over the neighbour count; the published check has no class path (src/kirby/utils/noise_estimation.py:473-480).
- Neighbours can be weighted by rank or by one over squared distance in both the mean and the spread (src/kirby/utils/noise_estimation.py:340-349).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The estimator was written: gap to the neighbour mean over the neighbour standard deviation, at ten neighbours. | `8b20490` | 2026-01-09 | no reason in the message |
| The classification denominator was floored at one over the neighbour count. | `1421b3d` | 2026-05-26 | "floored at 1/k to avoid 1e10 score explosions when neighbourhoods are perfectly pure" |
| The regression denominator was floored at a hundredth of the label standard deviation. | `7a98aa1` | 2026-08-19 | no reason in the message; the reason is in the code comment |
| The counting scheme reached this estimator's neighbour mean and neighbour spread. | `84e6ce2` | 2026-10-06 | "The four neighbour detectors read the counting scheme" |

## Cliff-aware gap

Her code: `src/kirby/utils/noise_estimation.py:557-628 (class ActivityCliffAwareEstimator). The regression branch is :615-628. It calls the neighbour-weighting helpers at :319-348 and the neighbour search at :204-262. The line range the first agent gave is right.`

Her cliff-aware score divides a molecule's gap from its neighbour mean by two spreads added together: the spread of all training labels, plus the spread of the neighbour labels. RegENN, the nearest published code, compares that gap against one spread times a fixed multiplier and deletes the molecule; hers returns a score for every molecule and deletes nothing. No reason is recorded: the commit that wrote the formula is titled only "Label estimation", and no published code implements this score (the nearest is RegENN in alvarag/DROPForRegression).

**Judgement: Fixed 2026-10-26.** The docstring said the score is the gap times one minus the neighbour-spread ratio, and the code computes the gap times one over one plus that ratio, so a formula written from the docstring was not the one that produced the numbers.

**What was done (2026-10-26).** The docstring now states the code's formula, with the date and the reason. The code is unchanged. The thesis detectors table and `THESIS_OUTLINE.md` already carried the code's formula, 1 over 1 plus the ratio. No number moves.

**Git records no reason for this.** Commit 8b20490 on 2026-01-09 wrote the score and the contradicting docstring in the same commit, with no reason in the message, and neither line was ever revised.

**Shortcuts taken.**
- The docstring's formula and the computed formula disagree (noise_estimation.py:565 against :620-623).
- The neighbour spread divides by the number of neighbours; RegENN divides by one less than that number (noise_estimation.py:346, AlgorithmReg.java:159).
- The two cut-offs that set the cliff flag, one training-label spread and one half, are fixed in the code and never fitted from the data (noise_estimation.py:627).
- Every molecule is scored against the whole training set in one pass; RegENN deletes as it goes, so a molecule already judged noisy stops being a neighbour (noise_estimation.py:592-599).

**Stronger than the published code.**
- The neighbour search can skip rows sharing a molecule's group, so repeat measurements of the same drug-target pair are not its own neighbours (noise_estimation.py:204-262).
- The spread of all training labels is floored by a tiny constant, so a dataset whose labels are all one value does not divide by zero (noise_estimation.py:615).
- The neighbour count is cut to one less than the number of molecules, so a small fold cannot ask for more neighbours than exist (noise_estimation.py:592).
- Neighbours can be weighted by rank or by inverse square distance; Weka's RegENN leaves its distance-weighting lines commented out (noise_estimation.py:319-348).
- Class labels get their own branch, scoring neighbour disagreement times neighbour purity, instead of a numeric residual on class codes (noise_estimation.py:601-612).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The score: gap over the spread of all training labels, damped by one over one plus the neighbour spread over that same spread. | `8b20490` | 2026-01-09 | no reason in the message |
| The docstring claiming the score is the gap times one minus the neighbour-spread ratio. Written once and never revised. | `8b20490` | 2026-01-09 | no reason in the message |
| The default of ten nearest neighbours, where RegENN defaults to one. | `8b20490` | 2026-01-09 | no reason in the message |
| The class-label branch, which scores neighbour disagreement times neighbour purity instead of a value residual. | `1421b3d` | 2026-05-26 | "class-disagreement scoring instead of nonsensical value-residuals on integer class labels" |
| The neighbour search can exclude rows sharing a molecule's group, so repeats do not score the molecule against themselves. | `9b288a0` | 2026-09-04 | "scored five of ten estimators from rows that include a molecule's own repeat measurements" |
| The neighbour mean and the neighbour spread became weighted, by rank or by inverse square distance. | `84e6ce2` | 2026-10-06 | "The four neighbour detectors read the counting scheme" |

## GP spread and residual

Her code: `src/kirby/utils/noise_estimation.py:914-1080, class GaussianProcessEstimator, plus the out-of-fold helper at :141-197 (the first agent gave :141-190; the helper runs to 197)`

Published source: https://github.com/scikit-learn/scikit-learn

She calls scikit-learn's Gaussian process and scores each molecule by its posterior spread plus its absolute out-of-fold residual. The length-scale start, its bounds and a fitted white-noise term replace the package defaults. Her comment dated 2026-09-21 gives the reason: a start of 1.0 made the fit return one constant value per fold on 1,280 features.

**Judgement: A deliberate change.** The package's Gaussian process runs unchanged, and the measured length-scale start and the added residual term are stated substitutions rather than a missing mechanism.

**Shortcuts taken.**
- The class docstring says a separate Gaussian process models the log variance (noise_estimation.py:922); the code fits one shared noise term for the whole fold.
- The white-noise level is capped at 1.0 by hand (noise_estimation.py:1046) where the package allows 1e5, and the class comment records a fit sitting on that cap.
- The training fold is cut to 1,000 molecules drawn once at seed 42 for cost (noise_estimation.py:986-992), with no measurement of what the cut costs in detection.
- length_scale_init_ is written inside each fold, so only the last fold's starting value survives on the object (noise_estimation.py:1047).
- The posterior spread and the absolute residual are added with equal weight, fixed by hand and never swept (noise_estimation.py:1069-1070).

**Stronger than the published code.**
- Every molecule is scored by a fit that never saw it, through the out-of-fold helper at noise_estimation.py:1064 and :141-197, where scikit-learn fits and predicts on the same rows.
- Folds keep whole scaffold families on one side when groups are passed (noise_estimation.py:174) and never shuffle otherwise (:177).
- A molecule no fold covered is set to NaN and then to the mean of the rest, never 0.0, so it cannot rank as the cleanest molecule (noise_estimation.py:191, :196).
- Features and labels are standardised on the training fold alone and the held-out molecules go through the same transform (noise_estimation.py:969-971).
- The length scale starts at the median distance between up to 256 training molecules (noise_estimation.py:1027-1037), and the class comment carries the measurement that justifies it.
- The kernel carries a fitted observation-noise term, which the package's own default kernel does not have (noise_estimation.py:1045).
- The fitted white-noise level is converted back into the label's own units and averaged over folds (noise_estimation.py:1057, :1078).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The class itself: the 1,000-molecule cap, 3 optimiser restarts, the white-noise start of 0.1 with its ceiling of 1.0, and the score as posterior spread plus residual. These lines have existed in one commit only; they were written once and never revised. | `8b20490` | 2026-01-09 | no reason in the message |
| The fit was wrapped in 5 out-of-fold folds, scaffold-group-disjoint when groups are passed. | `7a98aa1` | 2026-08-19 | "kill in-sample leakage ... every row is scored by a model that never trained on it" |
| The length scale now starts at the median distance between training molecules, with bounds a hundredfold either side of that start. | `2f1ceb6` | 2026-09-21 | no reason in the message; the measurement is in the code comment only |
| The optional per-feature weights, applied as sqrt of the weight after standardisation. | `2f1ceb6` | 2026-09-21 | no reason in the message; the code comment carries the 2026-09-19 measurement |

## Heteroscedastic GP spread

Her code: `src/kirby/utils/noise_estimation.py:1838-2021, class HeteroscedasticGPEstimator`

No published implementation exists for this construction, so the source link is empty; the class says it was ported from her own earlier repository, qsar_qm_models. The noise network's learning rate rises from 0.001 to 0.01 and its passes from 100 to 400. The class comment gives the reason and the sweep: rank correlation against the true noise width of +0.180 at the old settings and +0.657 at the new.

**Judgement: A deliberate change.** Both changed numbers were swept on her own fixture before the defaults moved, and the class records the table that chose them.

**Shortcuts taken.**
- The port source blanks the epistemic column when the fit returns one constant for every molecule; hers has no such check (noise_estimation.py:2005-2012).
- The port source offers a Tanimoto kernel as its default and a Matern as an option; hers builds an RBF only (noise_estimation.py:1937-1938), with no recorded reason.
- The training fold is cut to 1,000 molecules drawn once at the seed (noise_estimation.py:1956-1959), on a cost argument with no measurement of the cut.
- The two learning rates and the 400 passes were chosen on one synthetic fixture of 300 molecules and 8 features, which the class itself says is not a real dataset (noise_estimation.py:1884-1886).

**Stronger than the published code.**
- The noise network's target is the residual of the fitted process, taken by switching gpytorch to eval mode (noise_estimation.py:1990-1994); the prior residual left the network flat.
- Every molecule's noise is predicted by a fit that never saw it (noise_estimation.py:2015), where the port source fits once on the rows it is handed.
- The predicted noise is returned as a standard deviation in the label's own units (noise_estimation.py:2008-2010), so a smoothing rule can read the raw value.
- The posterior variance is kept beside the noise score (noise_estimation.py:2020), so a caller can check the two are not one signal reported twice.
- A test named in the class fails if the eval-mode switch is removed, and that file exists at tests/test_hetero_gp_learns_the_noise.py.
- The class states that nothing here has been swept on a real dataset, only on an eight-feature synthetic fixture (noise_estimation.py:1884-1886).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class, including the noise learning rate of 0.01, the 400 passes, the median length-scale start and the eval-mode residual target. The lines exist in one commit only; they were written once and never revised. | `c959ac5` | 2026-09-18 | "Their learning rate and epoch count leave the network flat here"; sweep table quoted in the message |
| The comment pointing the Gaussian process class at this file's median-distance heuristic; no code in this class changed. | `2f1ceb6` | 2026-09-21 | no reason in the message |

## Variance-head network

Her code: `src/kirby/utils/noise_estimation.py:1083-1201, class HeteroscedasticNNEstimator`

Published source: https://github.com/martius-lab/beta-nll

She rewrote the beta-NLL loss in her own file rather than calling the authors' package. The variance comes from a linear head read as a log variance and clamped between -10 and 10, where theirs is a softplus head floored at 1e-8 and capped at 100. No reason is recorded for the head, the dropout of 0.1, the weight decay of 1e-4 or the beta of 0.5; the whole class arrived in one commit whose message is two words.

**Judgement: A deliberate change.** The beta weighting that defines the method is present and detached from the gradient, so only the output head and the added regularisation are substitutions.

**Shortcuts taken.**
- The loss drops the constant log(2*pi) the authors keep inside the weighted bracket (noise_estimation.py:1147), so the printed loss is not their negative log-likelihood.
- Dropout at rate 0.1 and weight decay of 1e-4 are set by hand and never swept (noise_estimation.py:1130, :1162); the published network and optimiser carry neither.
- The run length is a fixed 100 passes over the training fold with no stopping rule and no checkpoint selection (noise_estimation.py:1100, :1168), where the published script runs 20,000 gradient updates.
- Beta is fixed at 0.5 (noise_estimation.py:1100); the paper's own repository sweeps it, and no measurement here chose 0.5.

**Stronger than the published code.**
- Every molecule's predicted spread comes from a fit that never saw it (noise_estimation.py:1188 and :141-197), where their script fits one model per split.
- Folds are scaffold-group-disjoint when groups are passed and unshuffled otherwise (noise_estimation.py:174, :177).
- Features and labels are standardised on the training fold alone, and the predictions are returned in the label's own units so folds are comparable (noise_estimation.py:1156-1158, :1183-1187).
- The log variance is clamped to the range -10 to 10 (noise_estimation.py:1141), so the variance can neither underflow to zero nor explode.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class: the exponentiated clamped variance head, dropout 0.1, weight decay 1e-4, beta 0.5, batches of 64 molecules and 100 passes. These lines exist in one commit only; they were written once and never revised. | `8b20490` | 2026-01-09 | no reason in the message |
| The fit was wrapped in 5 out-of-fold folds, scaffold-group-disjoint when groups are passed. | `7a98aa1` | 2026-08-19 | "kill in-sample leakage ... every row is scored by a model that never trained on it" |

## NGBoost spread

Her code: `src/kirby/utils/noise_estimation.py:1704-1834, class NGBoostEstimator`

Published source: https://github.com/stanfordmlgroup/ngboost

Her fold calls the package's NGBRegressor and reads the package's own predicted spread. The schedule is 50 boosting stages at rate 0.1 rather than 500 at 0.01, changed on 2026-09-21 because at 500 stages the experiment that asks for it was abandoned after 18 minutes with no reading. Numpy's global seed is pinned and restored around the fit, because NGBoost draws from that stream and not from the random_state it is handed.

**Judgement: Settings only.** The package's method runs unchanged; the stage count, the learning rate, the progress printing and the seed are the only differences.

**Shortcuts taken.**
- The shared parameter spec's early-stopping and best-iteration keys are not read (noise_estimation.py:1755-1771); the schedule was shortened instead of carving a validation split inside each inner fold.
- The 0.0187 average precision the shorter schedule costs was measured on one fold of one antibody dataset (noise_estimation.py:1733-1737), and no other fold or dataset was checked.

**Stronger than the published code.**
- Numpy's global seed is pinned and then restored around the fit (noise_estimation.py:1812-1817), so the scores repeat exactly and no later random draw in the process moves.
- Every molecule's spread comes from a fit that never saw it (noise_estimation.py:1821 and :141-197), where the package fits once on the rows handed to it.
- The docstring records what the shorter schedule costs in detection on one fold: 0.1313 against 0.1500 average precision, at 21.9 against 219.6 CPU-seconds.
- The docstring states that the condition under which both numbers were taken corrupts molecules at random, so neither reading is evidence about whether this estimator can detect.
- The cost per boosting stage is recorded in CPU-seconds at two representation widths, so a loaded machine cannot move the reading (noise_estimation.py:1719-1724).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The class as first written, at the package's 500 stages and rate 0.01 with the Normal likelihood and minibatch fraction 1.0. | `8b20490` | 2026-01-09 | no reason in the message |
| The schedule became 50 boosting stages at rate 0.1, keeping the same product of stages and rate. | `55b9684` | 2026-09-21 | "ngboost had never given n23 a reading. The run ... was abandoned after 18 minutes" |
| Numpy's global seed is pinned to 42 and the previous state restored around the fit. | `3aaba6e` | 2026-09-23 | "NGBOOST DRAWS FROM NUMPY'S GLOBAL RNG, not from the random_state it is handed" |
| The fit was wrapped in 5 out-of-fold folds, scaffold-group-disjoint when groups are passed. | `7a98aa1` | 2026-08-19 | "kill in-sample leakage ... every row is scored by a model that never trained on it" |

## Nearest-neighbor Shapley value

Her code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:995-1076 (knn_shapley_values); the neighbour count KNN_SHAPLEY_K is set at line 255`

Published source: https://github.com/AI-secure/Shapley-Study

Her code keeps Jia et al.'s exact recursion and swaps the published class-match indicator for minus the squared label gap, because these labels are continuous. Every training molecule becomes the test point in turn, with itself struck out, since a training fold carries no separate labelled test set. Both changes are written out in the docstring at n32 lines 1012 to 1038.

**Judgement: Moved to another task.** The recursion is the published one; only the per-neighbour worth moved from classification to regression.

**Shortcuts taken.**
- Each molecule's value is scored against its own label (n32:1058-1066), where the published code scores against a held-out test set's own labels.
- The neighbour count is five, set by hand at n32:255, against the published ten, and no comment or commit message says why five.

**Stronger than the published code.**
- _smoke_n32 checks the recursion against a brute-force Shapley value over every subset of seven molecules (tests/experiments/noise/_smoke_n32.py:99-136), a check the published code has none of.
- Molecules at exactly equal distance are ordered by a stable sort (n32:1064), so two tied molecules cannot swap and move each other's values.
- One neighbour ordering per test molecule serves every neighbour count asked for (n32:1058-1078), where the published code recomputes a ranking per neighbour count.
- The docstring states that the worth is a sum over neighbours and not the squared error of the neighbours' mean (n32:1012-1018), which is what the recursion requires.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole function: the regression worth minus the squared label gap, five neighbours, the stable sort. | `1e7b813` | 2026-09-24 | no reason in the message, beyond "from the second research report" |
| The caller may hand in distances instead of squared Euclidean distance in the features, through dist_rows. | `aa98979` | 2026-09-25 | "similarity axis wired into every noise experiment" |
| The call site stopped striking out the test molecule's scaffold group, passing no groups. | `5566961` | 2026-10-05 | no reason in the message; the code comment at n32:1105-1107 cites the neighbour-list decision of 2026-10-05 |
| Each of the neighbours keeps an equal share, recorded as outside the neighbour-counting axis. | `f2d2c98` | 2026-10-06 | "the exact Shapley recursion needs equal shares", in the code the commit wrote |

## Cleanlab score

Her code: `tests/experiments/noise/noise_helpers.py:762-779 (cleanlab_suspicion)`

Published source: https://github.com/cleanlab/cleanlab

The scoring is cleanlab's own code: she calls find_label_issues and returns one minus the label quality, so a higher score means more suspicious. She hands it KIRBy's LightGBM regressor in place of the package's default straight-line fit, and passes a seed the package leaves unset. Her docstring records that cleanlab's internal folds are random rather than scaffold-grouped.

**Judgement: Settings only.** The package computes the score line for line; the model, the seed and the sign of the output are all that differ.

**Shortcuts taken.**
- cleanlab's folds ignore scaffold groups, so a molecule's scaffold-mates can train the model that scores it (noise_helpers.py:776-778), unlike every other detector in the suite.
- The package searches for how many worst-residual molecules to hold out and returns its own flag column; her code reads the quality score only and drops the flag (noise_helpers.py:779).

**Stronger than the published code.**
- A seed is passed at noise_helpers.py:777, where the package default leaves its fold shuffle on numpy's global state, so her score repeats run to run.
- The model handed over is a LightGBM regressor (noise_helpers.py:776), which fits matrices of hundreds of features better than the package's default straight-line fit.
- The docstring says plainly that cleanlab's own folds are random and that the held-out fold never reaches this detector (noise_helpers.py:768-770).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole wrapper: the LightGBM model, the seed, five folds, and one minus the label quality. | `bd29125` | 2026-09-25 | no reason in the message; it says "cleanlab added to the noise extras" and "from the third research report" |
| These lines have only ever existed in that one commit: written once, never revised. | `bd29125` | 2026-09-25 | no reason in the message |

## LightGBM residual divided by a spread (NGBoost, quantile forest, conformal interval)

Her code: `src/kirby/utils/noise_estimation.py:1521-1599 (QuantileForestEstimator), 1601-1699 (QuantileRegressionEstimator), 1704-1835 (NGBoostEstimator); tests/experiments/noise/n41_conformal_fdr_detection.py:275-289 (rescaled_residual)`

Published source: https://github.com/stanfordmlgroup/ngboost

All three estimators call the published packages and differ in settings: two hundred trees against a hundred, depth five against three, and fifty boosting stages at a tenth against five hundred at a hundredth. The shorter NGBoost schedule has a recorded reason: at five hundred stages the experiment that asks for it never returned a number, and the docstring at lines 1736 to 1756 gives both detection readings and both costs. No reason is recorded for the tree count or the depth. The noise score itself is hers in each case, because none of these packages produces one.

**Judgement: Settings only.** Every change is a count, a depth, a rate or a seed; the packages do the fitting unaltered.

**Shortcuts taken.**
- NGBoost fits every stage with no early stopping, although the shared parameter spec carries a stopping rule; the docstring at lines 1758 to 1772 says honouring it would need a validation split with no scaffold ids.
- The quantile regression score is an interval width divided by the spread of that width across molecules (noise_estimation.py:1690-1695), a quantity her own docstring calls a property of the neighbourhood rather than of this molecule's label.
- The QuantileRegressionEstimator docstring (noise_estimation.py:1601-1609) still describes a plain interval width and says nothing about the out-of-fold loop the code runs.

**Stronger than the published code.**
- Each estimator is wrapped in a five-fold out-of-fold loop that keeps scaffold groups together (noise_estimation.py:1573, 1690, 1821), where the packages fit and score the same molecules.
- NGBoost's fit is run with numpy's global seed pinned and the previous state restored afterwards (noise_estimation.py:1812-1817), because the package's own random_state never reaches its base tree.
- The quantile forest score is how far a label falls outside its own plausible range in label units, and the docstring says why an interval width is not that quantity (noise_estimation.py:1536-1540).
- Divisors are floored before any division (noise_estimation.py:1585, 1695) and n41's spread is floored at a tenth of the median gap (n41:285), so a zero spread cannot divide.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The quantile forest estimator arrived with two hundred trees and the out-of-fold loop. | `2d80c00` | 2026-09-17 | no reason in the message |
| The quantile regression estimator arrived at depth five with a hundred boosting stages. | `8b20490` | 2026-01-09 | no reason in the message |
| The out-of-fold loop with scaffold groups was added around these estimators. | `7a98aa1` | 2026-08-19 | "kill in-sample leakage ... scaffold-group plumbing" |
| NGBoost went from five hundred stages at a hundredth to fifty stages at a tenth. | `55b9684` | 2026-09-21 | "ngboost had never given n23 a reading. The run ... was abandoned after 18 minutes" |
| NGBoost's fit got numpy's global seed pinned and restored around it. | `3aaba6e` | 2026-09-23 | "NGBOOST DRAWS FROM NUMPY'S GLOBAL RNG, not from the random_state it is handed" |
| n41's rescaled residual, with the divisor clipped at a tenth of the median gap. | `bd29125` | 2026-09-25 | no reason in the message |

## Conformal p-values with Benjamini-Hochberg flags

Her code: `tests/experiments/noise/n41_conformal_fdr_detection.py:248-385 (conformal_pvalues, benjamini_hochberg, rescaled_residual, the three calibration rules, min_bh_flags, bh_record), with the call site at 1089-1096`

Published source: https://github.com/msesia/conditional-conformal-pvalues

The p-value arithmetic is the published formula with the comparison flipped, because her score counts high as suspicious and the published one counts low as abnormal. What moved is the task: the published code scores held-out inliers with a one-class model, and hers scores training molecules with a residual divided by a fitted spread. The three calibration rules are hers; the code comment at n41 lines 216 to 222 records the owner's reason for sweeping two calibration shares.

**Judgement: Moved to another task.** The p-value formula and the Benjamini-Hochberg step are unchanged; the score and the data they run on moved to label noise in a training fold.

**Shortcuts taken.**
- Under two of the three rules the calibration molecules are ordinary training molecules that may themselves be corrupted (n41:294-305), where the published calibration data is clean by construction.
- Benjamini-Hochberg is hand-written as the plain procedure (n41:259-272); the published experiment also runs the Storey-corrected version, which hers does not.
- Only the marginal p-value is computed (n41:248-256); the four conditionally calibrated p-values the published predict returns are absent.

**Stronger than the published code.**
- min_bh_flags computes the fewest molecules Benjamini-Hochberg can flag at all at a given level (n41:355-366), a bound the published code does not record.
- bh_record counts how many moved molecules landed in the calibration set rather than among the tested ones (n41:369-385); the published experiment counts rejections, false discoveries and power only.
- trusted_calibration returns no mask and a written reason when a dataset has no repeat measurements (n41:322-323), instead of approximating one.
- The calibration set that reads the injection's own mask is labelled an oracle in its docstring (n41:304-308), so the arm cannot be read as a usable rule.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The p-value function, the hand-written Benjamini-Hochberg step, the trusted calibration rule and the flag-count floor. | `bd29125` | 2026-09-25 | no reason in the message, beyond "from the third research report" |
| The random calibration set became two shares of the training fold rather than one. | `9031b91` | 2026-09-25 | no reason in the message; the code comment at n41:216-222 records the owner's decision and the flag-count arithmetic |
| The oracle calibration set was restricted to molecules the injection did not move. | `633f553` | 2026-09-25 | no reason in the message |
| The three rejection levels and the minimum calibration size have only ever existed in the first commit: written once, never revised. | `bd29125` | 2026-09-25 | no reason in the message |

## Importance-scaled neighbor space

Her code: `tests/experiments/noise/n12_importance_weighted_distance.py:291-352 (_weight_features), with the exponent constant at :287 and the importance method list at :179. Also tests/experiments/noise/n21_neighbour_distance.py:348-401 (weight_features), the same code with the exponent as an argument; the first agent's end line of 400 was one short. Both take an already z-scored feature matrix, raise each feature's importance to a power (0.5 in n12), rescale those weights so they average one across features, multiply the matrix by them, and drop every feature whose weight came out zero.`

No published implementation exists to differ from: the 1997 feature-weighting review surveys many weighting methods and releases no code, and the two importance scorers her code consumes are called as published, LightGBM's gain and shap's TreeExplainer. Her construction raises each feature's importance to the power of one half, rescales the weights to average one across features, multiplies the already z-scored features, then drops features whose weight came out zero. Every step has a reason written in the file.

**Judgement: A deliberate change.** There is no published implementation for the weighting itself, so nothing can be absent; the construction is hers and each step carries a recorded reason.

**Shortcuts taken.**
- The exponent is fixed by hand at one half (n12:287), while n21 measured detection average precision 0.2098 at one half against 0.2388 at one under manhattan, on 672 antibodies over two folds.
- The power is taken before the rescaling here and after it in src/kirby/similarity/preprocessing.py's fold_weights, so the two give different weights at any exponent but one; the comment at n12:308-315 says so and keeps this order.
- n21's weight_features duplicates n12's arithmetic rather than importing it, so the two can drift (tests/experiments/noise/n21_neighbour_distance.py:348-401).

**Stronger than the published code.**
- The caller must pass an already z-scored matrix, because weighting first and standardizing after exactly cancels, which had made the weighted and unweighted conditions identical (tests/experiments/noise/n12_importance_weighted_distance.py:291-300).
- Features the scorer gave no importance are dropped, which commit ca249de measured as bit-identical suspicion scores across every metric, exponent, estimator and neighbour count tried (n12:338-352).
- How many features each condition's distance was computed over is recorded, so two importance scorers are not compared over spaces of very different width with nothing saying so (commit ca249de).
- The weights take the matrix's own precision, so a float32 representation stays float32 instead of being promoted (n12:327-330).
- At an exponent of zero no feature is dropped and the space is the unweighted control, stated in the comment (n12:347-349).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The bare square root of the importance became a named exponent, still set to one half, so no number n12 had produced changed. | `183e2a7` | 2026-09-21 | "the power that makes a feature's weight equal its importance is 1.0" under manhattan |
| Features whose weight came out zero were dropped from the space before the neighbour search. | `ca249de` | 2026-09-21 | "n12 computed 97 percent of every distance over columns of exact zeros" |
| The weights were rescaled to average one across features, so z-scored features are stretched rather than crushed. | `df7c7fe` | 2026-09-12 | no reason in the message; this line has existed in one commit only, written once and never revised |
| The weighting was moved to after standardization, which is when the weighted conditions stopped being identical to the unweighted ones. | `ba1a577` | 2026-05-26 | no reason in the message |

## Distance families for neighbors (plain, importance-weighted, forest proximity, hybrid's features)

Her code: `src/kirby/utils/noise_estimation.py:2234-2256 (forest_leaf_ids) and :2258-2277 (forest_proximity_distance), with the constants at :2228-2231. The settings are chosen in tests/experiments/noise/n07_remediation.py:1906-1981 (screen_settings). The space names are listed in tests/experiments/noise/similarity_space.py:125-147 and enumerated at :326-365; the first agent's pointer of :270-364 covered the SimilaritySetting class rather than the list. The Tanimoto space is built at :634-652 and used at :780-802; the forest space is rebuilt at :655-679. Read in full.`

Published source: https://github.com/cran/randomForest

Breiman and Cutler's code counts a pair of molecules only in the trees that left both out of the bootstrap, and divides by that count. Hers counts every pair in every tree, on the molecules the forest was fitted to, and divides by the total number of trees. The forest's tree count and leaf floor have reasons recorded in the file; dropping the out-of-bag restriction has none, in the code or in any commit message.

**Judgement: Fixed 2026-10-26 (the proximity) and 2026-10-29 (unparseable SMILES skipped).** The out-of-bag restriction is what keeps a forest proximity from reporting the trees' memory of the molecules they were grown on, and it was absent.

**What was done (2026-10-26).** `forest_leaf_ids` and `similarity_space._forest_leaves` return the out-of-bag mask beside the leaf ids (sklearn's `estimators_samples_`), and `forest_proximity_distance` and `_forest_distance` count a pair only in the trees that left both molecules out, divided by that count; a pair no tree left out together gets distance 1. A validation molecule was in no bootstrap, so a train-by-validation pair counts in the trees that left the training molecule out. Every caller was updated (n07 prox_smooth, n14, n21 and its inner forest, n32, the n21 loop script, the n21 smoke, `noise_cache.cached_forest_leaf_ids`). The thesis sentence on the proximity says so. On 2026-10-29 (author's decision B): an unparseable SMILES is skipped in the Tanimoto space, with +inf to and from every molecule in `similarity_space` and n32 and similarity 0 in n01; the count is recorded. OPEN_THREADS A9.

**Git records no reason for this.** Commit 2d80c00 on 2026-09-17 wrote the proximity without the out-of-bag restriction and gave no reason. The tree count and the leaf floor in the same code do have recorded reasons, so this omission is specifically unexplained.

**Shortcuts taken.**
- The proximity is computed on the very molecules the forest was fitted to (src/kirby/utils/noise_estimation.py:2246, :2270-2277).
- The divisor is the total number of trees, not the number of trees in which both molecules were left out of the bootstrap (src/kirby/utils/noise_estimation.py:2276).
- A SMILES RDKit cannot parse is given methane's fingerprint, so every unparseable molecule becomes a nearest neighbour of the others (tests/experiments/noise/similarity_space.py:646-649).
- similarity_space.py rebuilds the forest field for field instead of reusing forest_leaf_ids, and its docstring points at line numbers that have since moved (tests/experiments/noise/similarity_space.py:656-682).

**Stronger than the published code.**
- The forest is fitted on the fold's training molecules and their damaged labels only; the docstring says fitting on clean labels would hand the distance the answer the experiment is asking for (src/kirby/utils/noise_estimation.py:2234-2246).
- The leaf floor is five molecules for both regression and classification, with the reason recorded: a fully grown forest isolates every molecule in its own leaf (src/kirby/utils/noise_estimation.py:2224-2229).
- The Tanimoto family calls RDKit's Morgan fingerprint and scipy's jaccard rather than reimplementing either (tests/experiments/noise/similarity_space.py:635-653, :793-795).
- The forest's thread count honours the repository's switch, where it ran on every core before (src/kirby/utils/noise_estimation.py:2248-2253).
- The screen that picks the space, the neighbour count and the counting scheme scores a held-back fifth of the training molecules, whole scaffold groups together (tests/experiments/noise/n07_remediation.py:1928-1941).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The forest proximity was written in n21: every pair counted in every tree, divided by the tree count, at 100 trees. | `2d80c00` | 2026-09-17 | no reason in the message |
| The leaf floor and the two forest functions moved from n21 into the shared library, unchanged. | `4217d3b` | 2026-09-23 | "The compass review's first proposal was to use an RF proximity as the kernel" — an AI-written report |
| The forest's thread count was changed from every core to the repository's switch. | `7b791e9` | 2026-09-20 | "n21's forest ran on every core" |
| The Tanimoto space was written, with methane's fingerprint for an unparseable SMILES and scipy's jaccard as the distance. | `a07a612` | 2026-09-25 | no reason in the message for the methane fallback; the code comment says it copies n01 |
| The leaf floor in n21 was touched again when the delegated decisions were applied. | `f2d2c98` | 2026-10-06 | "One agent per experiment file applied that file's decisions" |

## Other detectors (not on the Chapter 7 list)

Her code: `tests/experiments/noise/n41_conformal_fdr_detection.py:388-544 (joined_matrix, isolation_forest_suspicion, lof_metric_kwargs, lof_suspicion, residual_low_disagreement, detector_scores, _lof_sweep) and tests/experiments/noise/noise_helpers.py:782-816 (gamma_test)`

Published source: https://github.com/haythorn/sr

Isolation forest and local outlier factor are sklearn's own code; what changed is the matrix they are given, which carries one extra feature, the out-of-fold residual scaled so it weighs as much as the whole feature block. That turns two feature-space outlier detectors into label-noise detectors, and the docstring at n41 lines 388 to 392 records the construction. The gamma test is reimplemented in numpy; it fits the same two averages as the published R code, and divides the intercept by a variance with a different denominator. No reason is recorded for the tree count or for that denominator.

**Judgement: Moved to another task.** The detectors themselves are untouched; the residual feature appended to their input is what moves them to a label question.

**Shortcuts taken.**
- The gamma_test docstring says each point is the mean squared distance to the k nearest neighbours (noise_helpers.py:787-789); the code takes the k-th neighbour alone (noise_helpers.py:807-808), which is the published arithmetic, so the docstring describes neither.
- gamma_test divides the intercept by numpy's variance, whose denominator is the number of molecules (noise_helpers.py:811-814); the published R code divides by one fewer, so the two ratios differ.
- gamma_test squares whatever distance matrix the caller hands it (noise_helpers.py:804-807), and its one caller at n40 line 456 passes a similarity setting's distance, which is not always Euclidean.

**Stronger than the published code.**
- The residual is z-scored and multiplied by the square root of the number of features (n41:388-399), so one residual feature weighs as much as the whole feature block instead of being one of thousands.
- gamma_test returns nothing when fewer than two neighbour positions exist (noise_helpers.py:800-801); the published R code has no such guard.
- lof_metric_kwargs raises on a distance name it does not map (n41:428), where handing the name straight to sklearn had made every Minkowski setting below one raise instead.
- residual_low_disagreement floors the spread across the five fitted seeds at a tenth of its median (n41:464), so a molecule the seeds agree on cannot divide by zero.
- Local outlier factor under Tanimoto and forest proximity is recorded as unavailable rather than approximated (n41:519-525), because those distances give no coordinates to join the residual to.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The joined matrix, the isolation forest at two hundred trees with its sign flipped, the local outlier factor call, and the seed-disagreement detector. | `bd29125` | 2026-09-25 | no reason in the message, beyond "from the third research report" |
| The gamma test reimplemented in numpy, fitted with polyfit and divided by numpy's variance. | `bd29125` | 2026-09-25 | no reason in the message |
| Local outlier factor got a distance mapping for manhattan, cosine and Minkowski below one. | `5566961` | 2026-10-05 | no reason in the message; the code comment at n41:418-426 says every Minkowski setting below one had raised |
| The joined matrix stopped casting the features to double precision and keeps their own dtype. | `5566961` | 2026-10-05 | no reason in the message; the docstring cites a handoff item of 2026-10-05 |

## Delete flagged

Her code: `tests/experiments/noise/n07_remediation.py:957-964, the 'remove' branch inside cleaned_training_set (806-1050). The cut-off percentile is built at line 899 from the flag share; the flag shares are FLAG_SHARES = (0.05, 0.10, 0.20) at line 271 and CLF_FLAG_SHARES = (0.10,) at line 235, picked by flag_shares_for at 291-298. The detector that produces the scores is DETECTOR = 'cross_val' at line 550.`

Published source: https://github.com/cleanlab/cleanlab

Her code removes a share of the training molecules that the sweep sets, 5, 10 or 20 percent on a continuous label. cleanlab's find_label_issues counts the mislabelled rows per pair of classes from predicted class probabilities and removes exactly that count. The reason for the swept share is written at n07:263-270: until 2026-09-28 the share was `noise_fraction`, the same number the injector uses, so the owner decided no method may read it.

**Judgement: A deliberate change.** Deleting the most suspect molecules survives; only how many go changed, and that change was an owner's decision of 2026-09-28 with a stated reason in the code.

**Shortcuts taken.**
- thesis_provenance/METHOD_PROVENANCE.md line 59 names Brodley and Friedl 1999, whose filter removes every molecule the ensemble misclassifies; no arm computes that count.
- The cut at n07:958-959 keeps molecules strictly below an interpolated percentile, so molecules tied on the cut all go and the count is not the share the name promises.

**Stronger than the published code.**
- The detector that produces the scores is fitted on inner folds that keep a molecule's scaffold family on one side, at n07:1638-1642.
- The arm refuses and records the reason when fewer than ten training molecules would remain, at n07:960-963, where cleanlab keeps at least one row per class.
- The cut runs on any per-molecule score, so the arm also runs on a continuous label, which cleanlab's function cannot take.
- The flag share is swept over three values rather than read from the injector, at n07:271.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The flag share became a swept (0.05, 0.10, 0.20) instead of `noise_fraction`, the injector's own corruption share. | `01e309a` | 2026-09-28 | "no method may read what only exists because the noise was injected" |
| The percentile cut and `keep = noise_scores < threshold` first appeared with the file. | `f101232` | 2026-03-25 | no reason in the message |
| The floor that declines the arm below ten remaining training molecules was written once and never revised. | `03f5abc` | 2026-09-17 | "no score for this fold rather than a fit on nothing" |
| The refusal became a recorded reason string instead of a silent None. | `60d635a` | 2026-10-08 | "n18/n20 record the specific cleanup refusal" |
| The detector feeding the scores became the named `cross_val` instead of one hardcoded class. | `e0e0301` | 2026-09-18 | "THE DETECTOR WAS ONE HARDCODED CLASS IN FIVE PLACES" |
| Classification sets got their own single flag share of 10 percent. | `f2d2c98` | 2026-10-06 | "Adelaide delegated the open decisions of chats 2-5" |

## Five-detector delete or move

Her code: `tests/experiments/noise/n23_uncertainty_cleaning.py:1019-1044 (drop_keep_mask) and 1047-1071 (cleaned_training_set), with SOURCES at 708-709, DROP_SHARES at 764, MOVE_STRENGTHS at 767 and MOVE_METHODS at 771; tests/experiments/noise/n42_similarity_noise_sweep.py:707-722 (_n23_dprime) calls those same two functions; the label move itself is smooth_by_scores at src/kirby/utils/noise_estimation.py:2084-2134`

No published code deletes or relabels training molecules by a predicted per-molecule spread, so there is nothing to differ from: this arm is her own construction. The design reason is in the commit that built it, 711bbeb of 2026-09-17: every other detector in the suite compares a label to its neighbours' labels, and a predicted spread is a different quantity. The share removed and the move strength are swept because, as n23:764 says, nobody knows the right value.

**Judgement: A deliberate change.** There is no published method behind it, so the flag judges her construction alone: it is deliberate, documented, and reportable as written.

**Shortcuts taken.**
- The move target is a neighbour label built elsewhere and the labels are moved in one pass, at n23:1069-1071; nothing iterates.
- smooth_by_scores scales the scores between the lowest and highest in the fold, at noise_estimation.py:2103, so one extreme molecule sets how far every other label moves.
- smooth_by_scores carries four methods and the arms call two, at n23:771, so 'threshold' and 'soft_threshold' are untested code in the path.

**Stronger than the published code.**
- Every drop arm records how many molecules it removed and how many of those the injector had moved, at n23:1929-1939.
- The docstring at n23:1019-1040 states the removed count is not exactly the share in the arm's name, with the measured counts.
- Each uncertainty source is scored on inner folds that keep a scaffold family together, at n23:775-776.
- The arm returns None when fewer than ten training molecules would remain, at n23:1065-1067.
- gaussian_process was taken out of the feature-reading list after measuring that half its score is the molecule's own residual, at n23:717-737.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The three drop shares, the three move strengths, the two move rules and the ten-molecule floor were all written once in the file that introduced the method. | `711bbeb` | 2026-09-17 | "nobody knows the right share here, so it is swept rather than looked up" |
| The percentile cut moved into drop_keep_mask and the removed counts began to be recorded. | `97b40da` | 2026-09-21 | no reason in the message |
| smooth_by_scores, the function that moves the labels, was written once and never revised. | `8b20490` | 2026-01-09 | no reason in the message |
| gaussian_process was removed from the list of sources that read noise from the features alone. | `c959ac5` | 2026-09-18 | no reason in the message |
| The count of injector-moved molecules among the removed ones was added to every drop row. | `f2d2c98` | 2026-10-06 | "Adelaide delegated the open decisions of chats 2-5" |

## Score down-weight

Her code: `tests/experiments/noise/n07_remediation.py:966-971, which calls weights_from_scores at src/kirby/utils/noise_estimation.py:2178-2206`

No published code uses a fixed 1/(1+score) sample weight for label noise; every published weighting of this kind learns the weight with a network or a meta objective. Hers scales the detector score between the lowest and highest in the fold, sets the weight to 1/(1+strength times that scaled score), and divides the weights by their mean. The reason for the min-max scaling is in commit e0e0301 of 2026-09-18: it makes the spread of the weights come from the spread of the evidence rather than from the label's range.

**Judgement: A deliberate change.** There is no published counterpart, so the flag judges her construction: a fixed weight shape, documented, with the strength swept elsewhere in the suite.

**Shortcuts taken.**
- The scaling at noise_estimation.py:2193 is set by the single highest and single lowest score in the fold, so one extreme molecule compresses every other weight.
- n07:966-971 fixes the strength at 1.0, while n18:297 sweeps the same weight over 1.0, 3.0 and 9.0 and n18:290-296 says a method winning at its weakest setting has not had its strength measured.
- n02:137 sets the default strength for the same weight to 3.0, so two files drive one weight at different strengths.

**Stronger than the published code.**
- The range in the denominator is floored by adding a small constant at noise_estimation.py:2193, so a fold where every score is equal does not divide by zero.
- The weights are divided by their mean at noise_estimation.py:2206, so the total weight does not change with the detector's scale.
- The detector score is produced out of fold with scaffold groups kept together, at n07:1638-1642.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| weights_from_scores, the whole weight, was written once and never revised. | `8b20490` | 2026-01-09 | no reason in the message |
| n07's call fixed the strength at 1.0 when the file was created. | `f101232` | 2026-03-25 | no reason in the message |
| The call became strength-settable, and n18 began sweeping 1.0, 3.0 and 9.0 while n07 kept passing 1.0. | `dabb0bd` | 2026-09-18 | "sweep the knob each cleanup reads, on a grid the measured error share justifies" |
| n02's reliability score was routed through this weight at a default strength of 3.0. | `e0e0301` | 2026-09-18 | "the spread of the weights comes from the spread of the evidence" |

## Neighbor-mean smoothing

Her code: `tests/experiments/noise/n07_remediation.py:1103-1136, with SMOOTH_STRENGTH at n07_remediation.py:214, the neighbour constants at n07_remediation.py:227-237, the flag share at n07_remediation.py:899, and the neighbour helpers at n07_remediation.py:920-952. The neighbour search itself is make_nearest_neighbors at tests/experiments/_shared.py:3417-3436.`

Published source: https://github.com/statsmodels/statsmodels

The published kernel regression weights every training molecule by its distance; her code takes a flat mean over the 9 nearest molecules and ignores the rest. She then keeps the original label and moves it a quarter of the way toward that mean, scaled by the molecule's suspicion score. The file records the reason for the quarter: it was measured on logd, flab_expression_koenig and herg with SVM over two folds, and lost at most 0.004 against a half.

**Judgement: A deliberate change.** Her code cites no paper for this cleanup, and each departure from kernel regression is a measured setting recorded beside the constant.

**Shortcuts taken.**
- The comment at n07_remediation.py:237 says the regression neighbours are Euclidean, while the code passes no metric and Manhattan runs.
- The neighbour count is fixed at 9 molecules by hand, where the published code fits its bandwidth from the data by cross-validation (n07_remediation.py:237).
- The classification path replaces the flagged labels outright and never moves one part way, so the two paths share a name but not an arithmetic (n07_remediation.py:1108-1121).

**Stronger than the published code.**
- The neighbour mean excludes the molecule itself, so no label is smoothed toward its own value (n07_remediation.py:1127-1130).
- Neighbours are found under Manhattan distance, chosen over Euclidean on 14 representation blocks of two datasets (src/kirby/similarity/distance_config.py:1-40 and :68).
- A similarity setting this cleanup does not read is refused with an error, so one set of numbers cannot be filed under many setting labels (n07_remediation.py:902-918).
- The smoothing strength and both neighbour counts carry the datasets, model and fold count they were measured on (n07_remediation.py:205-237).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| label_smooth was first written, taking a neighbour mean over the nearest molecules in the feature space. | `f101232` | 2026-03-25 | no reason in the message |
| The neighbour search switched from scikit-learn's Euclidean default to the suite's process-wide Manhattan setting. | `7546bae` | 2026-09-10 | no reason in the message; the module's own note gives the f18 measurement |
| The smoothing strength of a quarter and the regression neighbour count of 9 molecules became named constants. | `5566961` | 2026-10-05 | no reason in the message; the file records the f36 measurement beside each constant |
| The classification path became the majority class of the 5 nearest molecules by Manhattan distance, at 10 of every 100 molecules flagged. | `5566961` | 2026-10-05 | no reason in the message; the file records herg gaining 0.001 auc |

## Move toward prediction

Her code: `tests/experiments/noise/n07_remediation.py:1247-1264, calling oof_predict at src/kirby/utils/noisy_labels/core.py:151-186 and smooth_by_scores at src/kirby/utils/noise_estimation.py:2084-2135`

Published source: https://doi.org/10.48550/arXiv.1412.6596

No code by the authors of the published method exists, so nothing was compared line by line. Her code moves each label a quarter of the way toward a LightGBM prediction made on 5 folds that keep each scaffold group whole. The reason is in the code: a random inner split would put a molecule's scaffold family on both sides, so the value a label moves toward would be partly that label itself.

**Judgement: A deliberate change.** There is no published code to depart from, and the out-of-fold group-aware prediction is a stated, defensible choice rather than a missing mechanism.

**Shortcuts taken.**
- The step size of a quarter is fixed by hand and is the same on every dataset (n07_remediation.py:214).
- The LightGBM model behind the prediction uses one fixed parameter set that no run searches (core.py:98-103).

**Stronger than the published code.**
- Every prediction comes from folds that never place a scaffold group on both sides (src/kirby/utils/noisy_labels/core.py:143-152).
- A missing group argument raises rather than falling back to a random split (core.py:120-125).
- The function raises if any molecule was left without an out-of-fold prediction, so partial coverage cannot pass silently (core.py:181-185).
- The cleanup refuses a classification set outright rather than approximating a class label with a part-way move (n07_remediation.py:1242-1245).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| oof_smooth was added, with the inner fold count of 5 set in the same commit. | `03f5abc` | 2026-09-17 | no reason in the message for the cleanup itself |
| Every noise estimator was routed through a shared out-of-fold helper with scaffold-group folds. | `7a98aa1` | 2026-08-19 | "so every row is scored by a model that never trained on it" |
| The step size became the shared SMOOTH_STRENGTH constant of a quarter. | `5566961` | 2026-10-05 | no reason in the message; the file records the f36 measurement |

## Score-scaled shrink

Her code: `src/kirby/utils/noise_estimation.py:2084-2135`

This function is her own construction, with no published method behind it, so there is nothing to compare line by line. It rescales the suspicion scores to run from zero to one, then applies one of four rules that mix each label with a target value. No reason is recorded for the steepness and midpoint written into the soft-threshold rule, and the function has never been revised since it was written.

**Judgement: A deliberate change.** No published code exists for a score-scaled mix of a label and a target, so nothing of a published method is missing.

**Shortcuts taken.**
- In the threshold rule one number sets both how many molecules move and how far each moves (noise_estimation.py:2111-2115).
- The steepness and midpoint inside the soft-threshold rule are written in and cannot be passed by a caller (noise_estimation.py:2120).
- The confidence rule reads the raw suspicion scores while the other three read the rescaled ones, so one alpha does not mean the same thing across the four (noise_estimation.py:2127-2129).
- Two of the four rules have no caller outside one grid sweep, so they are carried untested by the rest of the suite (n11_estimation_remediation_grid.py:704).

**Stronger than the published code.**
- The rescaling adds a small floor to the divisor, so a column of identical suspicion scores does not divide by zero (src/kirby/utils/noise_estimation.py:2103).
- An unknown rule name raises rather than falling through to a default (noise_estimation.py:2130-2131).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole function, all four rules and the rescaling, was written in one commit and has never been revised. | `8b20490` | 2026-01-09 | no reason in the message |
| n07's three calls were fixed to the proportional rule at a quarter. | `5566961` | 2026-10-05 | no reason in the message; the file records the f36 measurement |

## Neighbor weights

Her code: `tests/experiments/noise/n12_importance_weighted_distance.py:418-462 (the score, _neighbour_suspicion), 627-629 and 768-770 (the weight), with the two constants at 158-159. tests/experiments/noise/n21_neighbour_distance.py:496-556 (the score, suspicion_from_neighbours) and 1363-1365 (the weight), with the two constants at 309-310. The weight itself is kirby.utils.noise_estimation.weights_from_scores, read at src/kirby/utils/noise_estimation.py:2178-2206. The ranges I was handed, n12:291-352 and n21:597-666, hold feature weighting and the own-distance screen. They are the wrong pointers for this method. I read them as well.`

Published source: https://github.com/metno/titanlib

titanlib flags an observation and removes it; her code never removes a molecule, it refits every molecule at a smaller weight. The strength of that weight, 3.0, has a recorded reason: the comment at n12:150-159 says n18 swept 2-to-1, 4-to-1 and 10-to-1 over the same molecules and folds, and 3.0 won 5 of 5 folds on both datasets. No reason is recorded for scoring each molecule once where titanlib repeats the check and drops the flagged observations between passes.

**Judgement: A deliberate change.** The deviation score is titanlib's; removal was replaced by a down-weight whose strength was measured on her own molecules and folds, not guessed.

**Shortcuts taken.**
- Each molecule is scored once (n12:620-629, n21:1358-1365); titanlib iterates up to num_iterations and drops flagged observations between passes.
- weights_from_scores min-max scales the score first (noise_estimation.py:2196), so one extreme molecule compresses every other molecule's weight in that fold.
- The counting scheme and the neighbour count are swept, but the weight rule and its strength are one fixed pair for every dataset (n12:158-159, n21:309-310).

**Stronger than the published code.**
- Every molecule stays in the fit at a smaller weight (n12:627-629, n21:1363-1365), so the number of training molecules is the same in the weighted and unweighted arms.
- The neighbours' spread is floored at the larger of 1% of sd(y) and 0.001 (n12:460-461, n21:554-555), so a molecule whose neighbours share one label cannot get an infinite score.
- The strength of 3.0 carries its measurement in the comment above it (n12:150-159, n21:304-310): 5 of 5 folds, +0.0069 rmse on herg_ki, +0.0447 spearman on flab_thermo_tresanco.
- n21 keeps an equal-weight path that _smoke_n21.py asserts reproduces the two estimator classes bit for bit (n21:509-512).
- The setting screen scores on a held-back fifth of the training molecules with whole scaffold groups kept together (n21:558-571).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The repair strength went from 0.1 under the exponential rule to 3.0 under the inverse rule, in both n12 and n21. | `e52f220` | 2026-09-19 | "a repair step within a tenth of not weighting at all" |
| The weighted neighbour mean and spread, and the counting schemes rank and inverse_square, replaced the equal-weight forms in both files. | `5566961` | 2026-10-05 | no reason in the message; the docstring at n12:425-427 says the record cut equal weighting |
| The spread floor, the larger of 1% of sd(y) and 0.001, was written into n21's scorer. | `5566961` | 2026-10-05 | no reason in the message |
| weights_from_scores, the function both files call for the weight, was written. It has never been revised; two later commits only moved it down the file. | `8b20490` | 2026-01-09 | no reason in the message |

## Exponential down-weight

Her code: `src/kirby/utils/noise_estimation.py:2178-2206.`

No published code was found for turning a per-molecule noise score into a training weight. Her weights_from_scores min-max scales the score, applies one of three rules, then divides by the mean so the weights average one (noise_estimation.py:2178-2206). No reason is recorded: the docstring names no paper, and the commit that wrote it says only "Label estimation".

**Judgement: A deliberate change.** Three hand-written rules with no published counterpart to lose; the one knob, strength, was later set from a measured sweep.

**Shortcuts taken.**
- The min-max step makes each molecule's weight depend on the single most suspect molecule in that fold (line 2196).
- The denominator carries a bare 1e-10 rather than a named floor (line 2196); the two callers in n02 test for a flat score vector themselves (n02:356-359).
- The docstring gives no units and no source (lines 2181-2190), so which of the three rules an experiment should use is written down nowhere.

**Stronger than the published code.**
- Every rule's output is divided by its own mean (line 2206), so a weighted fit and an unweighted fit see the same total weight.
- The soft rule clips the weights to between 0.1 and 1 (line 2202), so no molecule can be weighted to near zero.
- An unknown rule name raises rather than falling back silently (lines 2203-2204).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| All three rules, the min-max step, the 1e-10 and the mean normalisation were written in one commit. They have never been revised. | `8b20490` | 2026-01-09 | no reason in the message |
| The strength the callers pass went from 0.1 under the exponential rule to 3.0 under the inverse rule, which is what the function's output actually turns on. | `e52f220` | 2026-09-19 | "a repair step within a tenth of not weighting at all" |

## Close-pair weights

Her code: `tests/experiments/noise/n35_near_duplicate_disagreement.py:262-293 (close pairs), 296-312 (the distance bins), 315-332 (the noise estimate), 368-372 (the flag) and 375-379 (the weight), with the four constants at 199-204 and the flagging call at 870-872.`

Published source: https://doi.org/10.1007/BF01413858

The published test reads the noise variance by carrying the mean squared label difference out to zero distance. Her code computes that intercept (n35:326-332) but hands the flag the closest bin's value instead (n35:870). The flag rule and the 0.25 down-weight have a recorded source: the file header at n35:8-13 says an AI-written report of 2026-09-24 proposed them as its items D5 and A7.

**Judgement: Fixed 2026-10-26.** The zero-distance extrapolation, the published test's defining step, was computed and written to the result file but never reached a weight.

**What was done (2026-10-26).** The flag reads `intercept_estimate`. Two fields were added to each distance space's record: `flag_estimate` names the estimate the flag read, and `flag_estimate_usable` is False when the intercept is negative or undefined, in which case no pair is flagged (the file's existing rule for a negative estimate). The file header and the result file's `estimate_note` say so.

**Git records no reason for this.** Commit 1e7b813 on 2026-09-24 gives the reason as "from the second research report", which is an AI-written report, not a paper.

**Shortcuts taken.**
- The flag at n35:870 is passed closest_bin_estimate, so the intercept the file computes never changes a weight.
- The down-weight of 0.25 and the threshold of 3 times the square root of 2 sigma are fixed at n35:203-204, with no sweep and no sensitivity check.
- The close-pair cut is fixed at the 1st percentile of all pairwise distances (n35:199) and the bin count at 5 (n35:201); neither is swept.

**Stronger than the published code.**
- The intercept is written to the result file beside the closest-bin value (n35:331-332), so the two can be compared without a re-run.
- The result file states that both estimates hold some real activity difference and are upper bounds on the noise variance (n35:652-656).
- Close pairs are split into same-molecule and different-molecule pairs (n35:361-364), a matched control.
- A fold over 12,000 training molecules is refused with the reason recorded, before any of that fold's searches run (n35:745-759).
- The untouched arm is averaged over only the passes this arm was refitted on, with the per-pass scores kept beside it (n35, added 2026-10-08).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole method was written in one commit: the 1st percentile, the 5 bins, the closest-bin estimate, the intercept, the 3*sqrt(2) flag, the 0.25 weight and the 12,000-molecule limit. | `1e7b813` | 2026-09-24 | "from the second research report" |
| The forest-proximity row cap this file also carries was made optional, so proximity close pairs now run at any size. | `ec5b3f3` | 2026-09-25 | "owner decision" |
| The arm's score is now compared against the untouched arm on the same passes only, not against a mean over every pass. | `60d635a` | 2026-10-08 | "Fault fixes from the 2026-10-08 analysis" |

## Top-share down-weight

Her code: `tests/experiments/noise/n31_regime_and_breakeven.py:384-389 (detector_flags) and 411-424 (apply_operation, the 'downweight' branch), with DOWNWEIGHT_TO = 0.25 at line 256 and FLAG_SHARES = (0.05, 0.10, 0.20) at line 261`

No published code puts a constant weight on a top-share flag set, so there is nothing to differ from: this arm is her own. It flags the 5, 10 or 20 percent of training molecules with the largest absolute out-of-fold residual, then gives each flagged molecule a weight of a quarter and every other molecule one. The file header at n31:1-60 names no paper; it names a research report of 2026-09-24 as the source of the file's design, and the weight itself has no recorded reason.

**Judgement: A deliberate change.** There is no published counterpart, so the flag judges her construction: a deliberate arm in a four-operation comparison, with one constant nobody fitted.

**Shortcuts taken.**
- DOWNWEIGHT_TO at n31:256 fixes the weight of a flagged molecule at a quarter; nothing in the file fits or sweeps it, and the comment gives no source.
- The weight is one number for every flagged molecule, at n31:417-419, so how suspect a molecule is does not change its weight.
- The flag set and the weights are computed once per fold; nothing refits after the weighted fit.

**Stronger than the published code.**
- The flag is built from an out-of-fold residual, at n31:384-389, not from an in-sample fit.
- The flagged share is swept over three values rather than read from the injector, at n31:258-261.
- The sibling operation bayes_shrink estimates its weight from the model's own out-of-fold errors, at n31:393-408, where this one fixes a constant.
- detector_flags uses a stable sort at n31:386, so molecules tied on the residual are flagged in a fixed order rather than at random.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The weight of a quarter on a flagged molecule and detector_flags were both written once, in the commit that created the file. | `1e7b813` | 2026-09-24 | "from the second research report" |
| The flagged share became a swept (0.05, 0.10, 0.20) instead of the number the injection moved. | `1c13c88` | 2026-09-28 | "no longer the number the injection moved or the condition's nominal share" |

## Neighbor-agreement weights

Her code: `tests/experiments/noise/n02_sample_weighting.py:222-252 (the cliff-aware weight), 255-311 (the feature weighting) and 313-362 (the agreement weight), with the constants at 137 and 172 and the cliff-aware call at 569-571. The estimator the first one drives is src/kirby/utils/noise_estimation.py:557-628, and its weight step is get_weights at 100-120, with the weight itself on line 119.`

No published code was found for either weight in this file, and the two do not agree with each other. The agreement weight min-max scales the deviation and leans at strength 3.0 (n02:137, 359-362); the cliff-aware weight feeds the raw score into one over one plus the score at strength 1.0, because the call at n02:569-571 passes no strength. The 3.0 has a recorded reason (n02:130-137, from an n18 sweep); the mismatch between the two has none.

**Judgement: Fixed 2026-10-29 (author's decision A, option 1).** The two arms this file compares differed in two ways at once, the arithmetic and the lean, so the difference between them could not be read as cliff-awareness.

**What was done (2026-10-29).** Both weights now go through `weights_from_scores` (min-max scaling, strength 3.0, divided by the mean): `compute_cliff_aware_weights` reads the estimator's scores and hands them to that function at DEFAULT_STRENGTH, and the n02 cleanup arm and n42's copy of it do the same. The two arms differ in the score only. Every stored n02 and n42 result for the cliff-aware weight is on the old footing. OPEN_THREADS A10.

**Git records part of this as deliberate.** The strength of 3.0 has a recorded reason in e0e0301 on 2026-09-18: "a deviation is on the scale of the label's SPREAD, not its range". The cliff-aware call that leaves the strength at its default of 1.0 came from f101232 on 2026-03-25 with no reason, so the mismatch is accidental.

**Shortcuts taken.**
- The cliff-aware call at n02:569-571 passes no strength, so it takes the function default of 1.0, a 2-to-1 lean against the agreement weight's 4-to-1.
- The cliff-aware weight at noise_estimation.py:119 feeds the raw score into one over one plus the score, while the agreement weight min-max scales first.
- The classification branch clips the weights to between 0.1 and 1 (n02:341); the regression branch has no clip at all.
- The ActivityCliffAwareEstimator docstring says noise_score = deviation * (1 - local_variance_normalized) at noise_estimation.py:565, but the code computes deviation times one over one plus that quantity, at lines 621-623.

**Stronger than the published code.**
- The agreement weight returns all ones when every molecule's deviation is within 1e-12 of the others (n02:356-359), instead of min-max scaling a flat vector.
- The classification branch is divided by its mean (n02:348) with the reason in the comment: SVC multiplies C by each row's weight, so weights averaging below one were also a smaller-C fit.
- Both functions refuse to run without a counting scheme (n02:243-246, 330-334) rather than falling back to equal weighting silently.
- DEFAULT_STRENGTH of 3.0 carries the sweep that set it in the comment above it (n02:130-137).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The agreement weight stopped dividing the deviation by the label's range and went through weights_from_scores at strength 3.0. | `e0e0301` | 2026-09-18 | "a deviation is on the scale of the label's SPREAD, not its range" |
| The guard that returns all ones when every deviation is the same was added in the same commit. | `e0e0301` | 2026-09-18 | "min-max normalising a flat vector is undefined" |
| The classification branch began dividing its weights by their mean, which changes every weighted classification fit in this file. | `5566961` | 2026-10-05 | no reason in the message; the code comment at n02:342-347 names SVC's C scaling |
| compute_cliff_aware_weights, the call that leaves strength at 1.0, was added. | `f101232` | 2026-03-25 | no reason in the message |
| ActivityCliffAwareEstimator and its get_weights, with the strength default of 1.0 and the raw-score weight, were written. Neither has been revised since. | `8b20490` | 2026-01-09 | no reason in the message |
| reliability_from_neighbours was split out of compute_reliability_scores so the similarity sweep could hand it neighbours. | `aa98979` | 2026-09-25 | no reason in the message |

## Cliff put-back

Her code: `tests/experiments/noise/n25_cliff_exemption.py:838-944. random_exempt_mask is 838-865, put_back is 868-912, n_spared is 915-944. The first agent's ranges were off by one at two ends: put_back ends at 912, n_spared at 944.`

No published label-noise filter puts named molecules back after cleaning: hers restores the molecules on activity cliffs, undoing all three shapes a cleanup takes, a dropped row, a lowered weight and a rewritten label. The reason is in the commit that wrote the file: a molecule on a cliff carries a correct label that looks wrong to a detector comparing it with its neighbours. The nearest published code is imbalanced-learn's NeighbourhoodCleaningRule, which exempts whole classes, never named molecules.

**Judgement: Fixed 2026-10-26.** The reported count of exempt molecules the cleanup would have touched compared the cleaned weight column against a renormalised one, so under the weighting cleanups every exempt molecule read as touched; the commit that wrote it recorded this as open.

**What was done (2026-10-26).** `n_spared` reads a molecule as touched by a weighting cleanup when its cleaned weight sits below the column's largest weight, which is the level both weighting cleanups give the molecules they leave alone (1 before the column is divided by its mean). Comparing against an un-renormalised column, as first proposed, would not have done it, because the cleaned column is itself divided by its mean. The put-back is unchanged. Under `huber_weight` the count now reads only the molecules beyond the scale; under `downweight`, whose weight falls from the very first molecule, nearly every molecule still reads as touched, which is what that weight does.

**Git records this change as deliberate.** Commit 4c85148 on 2026-09-23 records why cliff molecules are spared, and the same message records this exact count defect as open: "n_spared compares against a rescaled weight rather than the neutral one".

**Shortcuts taken.**
- n_spared compares the cleaned weight column against the renormalised one, so under downweight every exempt molecule counts as touched (n25_cliff_exemption.py:902-905, :931-935).
- The control is matched on how many molecules are put back, not on how many the cleanup would have touched (n25_cliff_exemption.py:838-865).

**Stronger than the published code.**
- A matched control puts back as many non-cliff molecules as the cliff arm puts back, drawn with a fixed seed (n25_cliff_exemption.py:838-865).
- The control draws only from molecules the cliff mask did not name, so the two exempt sets never overlap (n25_cliff_exemption.py:858-860).
- Passes where too few non-cliff molecules exist are left out of the cliff-against-random count, and the number left out is recorded (n25_cliff_exemption.py:838-865).
- Restoring an exempt molecule's weight renormalises the whole column to mean one, which leaves the ratios among the other molecules as the cleanup set them (n25_cliff_exemption.py:902-905).
- A restored label keeps its own type, so a class label is not cast to a float before a classifier is fitted (n25_cliff_exemption.py:889-896).
- The count of molecules spared is defined the same way for a cleanup that deletes rows, one that lowers weights and one that rewrites labels (n25_cliff_exemption.py:915-944).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| put_back written: a dropped row goes back into the keep mask, a lowered weight is set to one, a rewritten label is restored. | `4c85148` | 2026-09-23 | "a molecule on a cliff carries a CORRECT label that looks wrong to a detector" |
| The weight column is renormalised to mean one after an exempt molecule's weight is set to one. Written once, never revised. | `4c85148` | 2026-09-23 | the same message records it as open: "n_spared compares against a rescaled weight rather than the neutral one" |
| The matched random control arm, drawn with a fixed seed from the non-cliff molecules. | `4c85148` | 2026-09-23 | no reason in the message; the docstring gives one, that the two exempt sets never overlap |
| n_spared written, counting exempt molecules whose presence, weight or label differs between the two arms. | `4c85148` | 2026-09-23 | no reason in the message |
| Passes where the control drew fewer molecules than the cliff arm are left out of the comparison and counted. | `5566961` | 2026-10-05 | no reason in the message; the docstring names it as delegated and decided 2026-10-06 |
| A restored class label stopped being cast to a float; the non-numeric case falls back to an equality test. | `5566961` | 2026-10-05 | no reason in the message; the code comment dates the decision to 2026-10-05 |

## Cleaning before the hybrid search

Her code: `tests/experiments/noise/n20_clean_before_search.py:13-19 (the four arms), :83-99 (huber_weight and invvar_weight refused), :494 (SEARCH_SCORES_ON = 'asis'), :1402-1477 (_cleaned_side, which builds the search on the cleaned rows and labels). The first agent gave 85-110 for the refusals and 1402-1478 for _cleaned_side; the right ranges are 83-99 and 1402-1477.`

scikit-clean's own runnable pipeline chooses features first and cleans labels second; her code runs the opposite order, and crosses which labels the search sees with which labels the model is fitted on, giving four arms. The reason is in the commit: every cleanup in her suite is applied after the recipe is fixed, so the cleaning never reaches the feature selection, and nothing had measured what that costs. No published package can run this comparison, so there is no source to link.

**Judgement: A deliberate change.** The reversed order is her own stated comparison with all four arms named, and the two cleanups whose output a feature search cannot read are refused in the output rather than approximated.

**Shortcuts taken.**
- One detector is fixed in the file, cross_val, and no axis sweeps it (n20_clean_before_search.py:483, :1790).
- The two hybrid representations are standardised by one scaler and the two single representations by another, which the comment names and leaves unfixed (n20_clean_before_search.py:1461-1463).

**Stronger than the published code.**
- The two cleanups that return a per-molecule weight only are refused, and both their searched-on-cleaned arms are written out as undefined (n20_clean_before_search.py:83-99).
- A no-cleaning control hands the two searches identical label columns, so a difference between arms cannot come from the code path (n20_clean_before_search.py:130, :297-300, :2428).
- Each cleaned search is keyed by its row set and its label column, so two searches never read each other's stored rows (n20_clean_before_search.py:1414, :1425).
- Candidates inside the cleaned search are scored on the labels the practitioner holds, not on the cleaned column the search trained on (n20_clean_before_search.py:494, :1423-1431).
- For a cleanup that drops rows, every training row is transformed through the hybrid built on the kept rows, so the fit-as-is arm still trains on every molecule (n20_clean_before_search.py:1405-1411).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The four arms: features chosen from the labels as they are or from the cleaned ones, crossed with the labels the model is fitted on. | `03f5abc` | 2026-09-17 | "the cleaning never reaches the feature selection" |
| downweight's two searched-on-cleaned arms written out as undefined rather than faked. | `03f5abc` | 2026-09-17 | "sample_weight appears zero times in hybrid.py and zero times in find_best_hybrid.py" |
| The axis that sets what candidates in the cleaned search are scored on, with the labels as they are as the default. | `5bd407c` | 2026-09-20 | "the candidate that best predicts corrupted labels won the selection in the arm" |
| The detector's name is passed explicitly, so the result file and the call cannot name different detectors. | `5bd407c` | 2026-09-20 | "named one detector in its result file and called another" |
| huber_weight and invvar_weight refused outright, so neither gets a cleaned search. | `c519221` | 2026-09-25 | no reason in the message |
| Scoring moved back from the cleaned column to the labels as they are, after one day on the cleaned column. | `f2d2c98` | 2026-10-06 | no reason in the message; the code comment records it as decided 2026-10-06 |

## Constant lift

Her code: `tests/experiments/noise/n30_censored_likelihood.py:293-300 (limit_from_labels), :303-311 (rows_at_limit), :314-322 (residual_sd), :325-328 (mills_ratio), :331-368 (lift_columns, the constant_lift column at :350-351); call site at :840-845 and :903-912. Also tests/experiments/noise/n07_remediation.py:1345-1382.`

Her code gives every training label at the limit one shared new value: the limit plus the average lift over the molecules sitting on it (n30:350-351). The published imputation in Frank Harrell's rms gives each censored row its own number, and no published code assigns one shared value. The reason recorded is a port: the docstring says the arm comes from censoring_probe.py, a script outside the repository, kept so the numbers in its FINDINGS.md reproduce (n30:334-336, :88-97).

**Judgement: A deliberate change.** The arm is her own averaged control, named constant_lift rather than any published method, and the per-molecule version and the correct truncated mean both run beside it.

**Shortcuts taken.**
- Every molecule at the limit gets the same value, so the per-molecule part of the Mills ratio is averaged away (n30:350-351).
- The lift being averaged is limit plus spread times Mills, which exceeds the mean of a normal truncated at the limit by (limit minus prediction) (n30:347-348).
- The script it was ported from, censoring_probe.py, is not in the repository, so the port cannot be checked against its source (n30:334-336).
- The probe function it ports is named impute_constant_mean_lift beside impute_tobit, and the arithmetic is not the Tobit conditional mean (n30:334-336).

**Stronger than the published code.**
- The spread is read off the molecules not at the limit, so how far the clip moved a label never enters it (n30:314-322).
- The limit is read off the recorded training labels, never from the injector, so the arm uses only what a practitioner holding clipped labels can see (n30:293-300).
- A limit is declared only when at least 2 molecules sit on the top label and they are at least one in a hundred of the training molecules, otherwise the lift arms do not run (n30:299-300).
- Two counts are written for every pass: molecules at the limit, and molecules the clip moved (n30:852-853).
- The prediction is out of fold with scaffold groups kept together, where the published imputation predicts on the same molecules it was fitted on.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The constant_lift column: every label at the limit set to the limit plus the mean lift over those molecules. | `2703afa` | 2026-09-24 | no reason in the message; it lists n30 as 'clipped labels lifted, dropped, winsorised' and was written once, never revised |
| The spread taken over the molecules not at the limit, which sets the size of the lift. | `2703afa` | 2026-09-24 | no reason in the message; the reason is in the code docstring, not the commit |
| The limit changed from the injector's value to the top recorded training label, which changes which molecules are lifted. | `1c13c88` | 2026-09-28 | "No noise method reads what only the injector knows" |

## Truncated-normal mean

Her code: `tests/experiments/noise/n30_censored_likelihood.py:325-328 (mills_ratio) and :353-354 (the trunc column inside lift_columns). The first agent gave :325-358 and :355-357; the trunc column is two lines earlier than that.`

Published source: https://github.com/scipy/scipy

Her code sets each label at the limit to prediction plus spread times the inverse Mills ratio, floored at the limit (n30:353-354). Run against scipy's truncated-normal mean on five prediction, spread and limit combinations, the two agreed to fifteen decimal places, so nothing in the formula differs. The reason it exists is written in the file: censoring_probe.py used a different lift, and this arm is the correct quantity put beside it (n30:88-97).

**Judgement: Settings only.** The arithmetic matches the standard truncated-normal mean exactly; only a floor at the limit is added, and it never bound in the five cases tested.

**Stronger than the published code.**
- The floor at the limit stops an imputed label landing below the limit when a prediction sits above it (n30:354).
- The spread comes from the molecules not at the limit, so the clip's own displacement does not inflate it (n30:314-322).
- The denominator of the Mills ratio is clipped away from zero, so a molecule far above the limit cannot divide by zero (n30:328).
- The file states in its own docstring that this arm and per_molecule_lift compute different quantities, and says which one the earlier numbers came from (n30:88-97).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The truncated_normal_mean column, prediction plus spread times Mills, floored at the limit. | `2703afa` | 2026-09-24 | no reason in the message; written once and never revised |
| The Mills ratio helper, with the survival term clipped away from zero. | `2703afa` | 2026-09-24 | no reason in the message |

## Censored EM

Her code: `tests/experiments/noise/n30_censored_likelihood.py:371-394 (censored_em_column), :397-407 (_own_model_oof), :273 (CENSORED_EM_ROUNDS = 3), call site at :915-927. The first agent gave :371-407, which runs through a blank gap into the helper; the function itself ends at :394.`

Published source: https://github.com/harrelfe/rms

Her code repeats three rounds of fitting the configuration's own model out of fold and replacing every label at the limit by the mean of a normal truncated there (n30:371-394). Buckley-James in rms instead estimates the residual distribution by Kaplan-Meier with no assumed shape, refits least squares in sample, and iterates until successive sums of squared errors settle. The reason is recorded in the code and the commit: the three model swaps only exist for three of the models, so datasets whose two decided models are support vector and neural network had no lower-bound result at all (n30:263-273).

**Judgement: A deliberate change.** The distribution-free Kaplan-Meier step is replaced by a normal, which makes it the EM-Tobit her own comment names, and the iteration that defines the method is still there.

**Shortcuts taken.**
- It runs exactly three rounds with no stopping test, where the published loop checks whether successive sums of squared errors settle and watches for cycles (n30:273, :381).
- The published code marks the molecule with the largest residual as uncensored before each Kaplan-Meier step; hers has no such rule, so every molecule at the limit is imputed in every round.
- The docstring names Buckley-James, whose imputation assumes no shape for the residuals, while the code draws from a normal (n30:129, :388-390).
- Which molecules count as censored is read off the labels rather than carried per molecule, so a molecule that genuinely sat on the top value is imputed too (n30:293-300).

**Stronger than the published code.**
- Each round predicts out of fold over a split that keeps scaffold groups together, where the published loop predicts on the molecules it just fitted (n30:397-407).
- It returns nothing and records how many rounds ran when the spread cannot be read from fewer than two molecules off the limit, rather than imputing from a broken spread (n30:379-382).
- The spread used in every round is written out per round, so the iteration can be inspected afterwards (n30:393-394).
- It runs for whichever two models a dataset decided, not only for least squares, so every dataset gets a lower-bound arm (n30:263-270).
- Each imputed label is floored at the limit, where the published code applies no floor (n30:390).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole censored_em arm: three rounds, the configuration's own out-of-fold model, imputation by the truncated mean. | `fce9a69` | 2026-10-06 | "n30 had no 'keep the clipped labels as lower bounds' result" for datasets whose models are support vector and neural network |
| The round count fixed at three, which sets how far the labels move. | `fce9a69` | 2026-10-06 | no reason in the message for the number three; written once and never revised |
| The out-of-fold helper that splits by scaffold group inside each round. | `fce9a69` | 2026-10-06 | no reason in the message |

## Own lift

Her code: `tests/experiments/noise/n30_censored_likelihood.py:347-348 (the per_molecule_lift column inside lift_columns) and :358-363 (its shuffled twins), with mills_ratio at :325-328 and residual_sd at :314-322. The first agent gave :352-354 and :359-363; per_molecule_lift is five lines earlier than that.`

Her code sets each label at the limit to the limit plus spread times the inverse Mills ratio (n30:347-348). The mean of a normal truncated at the limit is prediction plus spread times Mills, so hers exceeds it by (limit minus prediction), which is negative when a prediction sits above the limit, and no published code computes this quantity. The reason is written in the file: censoring_probe.py used this form, every number in its FINDINGS.md came from it, and the arm keeps it so those numbers reproduce, with the correct quantity running beside it as truncated_normal_mean (n30:88-97).

**Judgement: A deliberate change.** The departure from the standard quantity is deliberate, stated in the file with both formulas written out, and the correct version runs as its own arm.

**Shortcuts taken.**
- The lift is computed off the limit rather than off the molecule's own prediction, so it is neither the Tobit conditional mean nor the Buckley-James imputation (n30:347-348).
- It is a single pass, where the published imputation repeats until the sum of squared errors settles.
- The residual distribution is assumed normal with one spread, where the published code estimates it by Kaplan-Meier with no assumed shape (n30:314-322).
- censoring_probe.py, which the docstring names as the source of the formula, is not in the repository (n30:334-336).

**Stronger than the published code.**
- The file states both formulas, says which one is the truncated mean and which one the earlier measurement used, and runs both (n30:88-97).
- Three shuffled twins deal the same lifts to different molecules at the limit, which separates 'the labels were raised' from 'each molecule got its own amount' (n30:358-363).
- Each shuffle's permutation is keyed on the fold, the level and the draw, so the same pass reproduces the same permutation (n30:361-362).
- The spread is read off the molecules not at the limit (n30:314-322), and each imputed label is floored at the limit (n30:347).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The per_molecule_lift column, limit plus spread times Mills, floored at the limit. | `2703afa` | 2026-09-24 | no reason in the message; written once and never revised |
| The three shuffled twins of the lift, at three seeds. | `2703afa` | 2026-09-24 | no reason in the message |
| The shuffle's draw keyed on fold, level and draw, which changes which molecule gets which lift. | `aa98979` | 2026-09-25 | no reason in the message for this line; the message covers the whole similarity wiring pass |
| The limit moved from the injector's value to the top recorded training label, changing which molecules are lifted. | `1c13c88` | 2026-09-28 | "No noise method reads what only the injector knows" |

## Delete clipped

Her code: `tests/experiments/noise/n30_censored_likelihood.py:913 (the arm itself, reached only when some training molecule sits at the limit and the residual spread is finite), reading rows_at_limit at 303-311 and limit_from_labels at 293-301, with MIN_SHARE_AT_LIMIT = 0.01 at line 290`

No package by an author of the method drops observations sitting at a detection limit; the nondetect literature discusses the practice and advises against it. Hers takes the largest recorded training label as the limit when at least two training molecules and at least one percent of them sit exactly on it, then masks those molecules out of the fit. The reason the limit comes from the labels is recorded at n30:285-290: it is an owner's decision of 2026-09-28 that the limit is read off the recorded training labels, never off the injector.

**Judgement: A deliberate change.** There is no published implementation to differ from; this is complete-case deletion run as a named comparison arm beside lifting, winsorising and the censored likelihood.

**Shortcuts taken.**
- limit_from_labels at n30:296-301 takes the largest training label as the only candidate limit, so a lower limit with values recorded above it is never found.
- The arm runs only when some training molecule sits at the limit and the residual spread is finite, at n30:905-913; on folds where it does not run, no row is written for it.

**Stronger than the published code.**
- The limit is read off the recorded training labels, at n30:293-301, so the arm never uses what only the injector knows.
- A limit exists only when at least two molecules and at least one percent of training molecules sit on the top value, at n30:301.
- The number of molecules at the limit is recorded on every row, at n30:852.
- The residual spread is computed over the molecules not at the limit, at n30:313-319, so how far the clip moved a label is kept out of it.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The drop_clipped arm itself was written once, in the commit that created the file. | `2703afa` | 2026-09-24 | "clipped labels lifted, dropped, winsorised, or fitted with XGBoost's censored likelihood" |
| The limit moved from the injector to the recorded training labels, with the one-percent and two-molecule test added. | `1c13c88` | 2026-09-28 | "the clip limit is read off the training labels ... not from the injector" |
| residual_sd, which computes the spread over molecules not at the limit, was written once and never revised. | `2703afa` | 2026-09-24 | no reason in the message |

## Scaffold-family offset (family mean and random intercept)

Her code: `tests/experiments/noise/n34_group_random_effects.py:216-301 (_group_index 216-221, group_center_offsets 223-230, moments_tau2 232-243, random_intercept_offsets 245-301); constants at 181-188; the call site at 729-747; the held-out prediction at 368-371`

Published source: https://github.com/manifoldai/merf

Her random intercept keeps the published shrinkage: a group's offset is its residual sum over (molecule count + sigma^2/tau^2). She estimates the two variances by the method of moments, not by the published EM update. The docstring at n34:80-96 gives the reason: written by hand so it needs neither merf nor gpboost, neither of which is installed.

**Judgement: A deliberate change.** The shrinkage that defines the method is present; only the variance estimator and the model inside the loop are swapped, both stated in the docstring.

**Shortcuts taken.**
- The loop runs at most 5 rounds, where the published MERF runs up to 20 (EM_MAX_ROUNDS, n34:182).
- Convergence is judged on the largest offset move alone; no log-likelihood is computed, so a stopped-early fit shows only in em_converged (n34:293-301).
- The within-group and between-group variances are estimated on the same residuals the offsets are fitted from, with nothing held out for them (n34:232-243, 285-290).
- The two group_center arms subtract a whole group mean residual, which is a group mean and not a published estimator (n34:223-230).

**Stronger than the published code.**
- Every residual is predicted out of fold over 5 scaffold-aware folds, where the published MERF fits and predicts the same rows (OOF_SPLITS at n34:181, call site 729-747).
- She kept the posterior-variance term the published EM has, and recorded a measured check for it in the comment at n34:273-280.
- The between-group variance is floored at one millionth of the training label variance, so no offset divides by zero (TAU2_FLOOR_FRAC, n34:187, 254).
- The recovery correlation is reported only over groups of 3 or more training molecules, with the count of groups left out beside it (n34:258-286).
- A recovery share is refused when the clean-to-noisy gap is under 2 percent of the clean score (GAP_MIN_FRAC, n34:192, 355-361).
- The count of held-out molecules sharing a scaffold group with training molecules is written every pass (n34:694).
- The share a group of one molecule keeps of its mean residual is written per pass, so the shrinkage can be read off (n34:302-310).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole random-intercept arm: the EM loop, the method-of-moments between-group variance, the floor and the 5-round cap. Written once and never revised. | `1e7b813` | 2026-09-24 | no reason in the message; it says only "from the second research report" |
| The posterior-variance term added to the squared residual in the sigma^2 update. Written once and never revised. | `1e7b813` | 2026-09-24 | no reason in the message; the code comment at n34:273-280 carries a measured toy check |
| A held-out molecule is predicted with no group offset added back, where the published predict adds a known group's offset. Written once. | `1e7b813` | 2026-09-24 | no reason in the message |
| The offset-recovery correlation moved to groups of 3 or more training molecules; the all-groups number was kept under a new name. | `f7f0b3f` | 2026-09-25 | "the all-groups correlation is kept as corr_all_groups_contaminated" |
| Features and hybrid recipes now come from searches on each fold's training rows, two decided models, four configurations each. | `5566961` | 2026-10-05 | "searches on each outer fold's training rows ... whole-dataset searches removed" |
| gaussian replaced grouped_wider as the no-offset control condition. | `f2d2c98` | 2026-10-06 | "Adelaide delegated the open decisions of chats 2-5" |

## Source offset

Her code: `tests/experiments/noise/n29_document_bias.py:340-368 (ridge_offsets), 414-440 (moment_variances), 442-491 (fit_document_offsets), 334-337 (eligible_documents), 408-411 (offsets_per_record); constants at 220-249. tests/experiments/noise/n39_correlated_sources.py:315-372 (offsets_als 315-339, offsets_mixedlm 341-355, fit_offsets 357-372); constants at 217-220`

Published source: https://github.com/fmpr/CrowdLayer

The published CrowdLayer learns one bias per source inside the network, by gradient descent. Hers fits the offsets outside the model, alternating a LightGBM fit and a ridge fit 3 times. A document's offset is its residual sum over its measurement count plus a penalty worth 1 measurement, and no reason for that penalty is recorded.

**Judgement: A deliberate change.** One offset per source, taken off the training labels, is still the method; the fit moves outside the model and gains a ridge penalty, both stated in the docstring.

**Shortcuts taken.**
- The alternating fit stops after 3 passes with no convergence test; only the largest offset move per pass is written (N_ITER, n29:220-226, 485-489).
- The ridge penalty of 1 measurement is fixed by hand in the main path, where the published bias is a free parameter (RIDGE_ALPHA, n29:226).
- The sign-flip bootstrap refits the ridge step alone on the final residuals, so no model variability enters it (n29:493-500).
- n39's alternating least squares has no published original, and its residual degrees of freedom are written out by hand (n39:315-339).

**Stronger than the published code.**
- A document gets an offset only when it shares a training molecule with another document, because otherwise its offset cannot be told from the molecules' values (n29:334-337).
- Offsets are centred to sum to zero inside each connected group of documents, since a shift common to a group is not identifiable either (n29:360-367).
- The out-of-fold predictions keep scaffold groups together, so no molecule predicts its own other measurements (n29:466-471).
- Per document, the share of 200 molecule resamples in which its offset changes sign is written (sign_flip_shares, n29:493-520).
- The random-effect path re-estimates the penalty as the within-document variance over the between-document variance on every pass, instead of fixing it (moment_variances, n29:414-440).
- In n39, when MixedLM raises or the measurement count passes 20000, her own alternating least squares runs and the reason is written onto the pass (n39:357-372).
- n39 centres the fitted source effects to mean zero, which statsmodels does not do (n39:351).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole offset fit: 3 alternating passes, the ridge penalty of 1 measurement, the group-of-2 rule and the centring. Written once and never revised. | `2703afa` | 2026-09-24 | no reason for the constants; the message says "fitted by alternating a model fit and a ridge fit" |
| The random-effect path, where the penalty is re-estimated each pass by method of moments. Written once. | `1e7b813` | 2026-09-24 | no reason in the message |
| n39's source offsets: statsmodels MixedLM with REML, the alternating least squares fallback, and the 20000-measurement switch. Written once. | `bd29125` | 2026-09-25 | no reason in the message; it says only "from the third research report" |
| Features and hybrid recipes moved to per-fold searches on the fold's own training rows. | `5566961` | 2026-10-05 | "searches on each outer fold's training rows ... whole-dataset searches removed" |
| herg_ic50's per-fold searches refiled under one name shared with n17, n19, n26 and n39. | `2cfa5e5` | 2026-10-07 | "shared by n17, n19, n26, n29 (and n39) under one name (Adelaide 2026-10-07)" |

## Source weights

Her code: `tests/experiments/noise/n39_correlated_sources.py:315-339 (the fallback), 341-355 (the statsmodels call), 357-371 (which of the two runs), 396-398 (the weight) and 401-416 (the two variance pieces), with the constants at 217-223 and the call sites at 958 and 969-983.`

Published source: https://github.com/statsmodels/statsmodels

statsmodels MixedLM is called as published, with reml=True and a per-molecule random intercept (n39:341-355). What is hers is what happens around it: the source offsets are centred at zero, and the weight, one over the model's own error plus the molecule's label variance, has no published code. The file header at n39:6-15 records where it came from, an AI-written report of 2026-09-25, its items A8 and D4.

**Judgement: A deliberate change.** The published fit is called unchanged; the centring, the weight and the alternating-least-squares fallback are stated additions around it.

**Shortcuts taken.**
- Above 20,000 measurements statsmodels is never called and her own alternating least squares runs instead (n39:357-360); the two have not been compared on the same data.
- The fallback records how many passes it took but not whether the tolerance of 1e-10 was met (n39:338-339), where the statsmodels path records converged.
- The pass limit of 200, the tolerance of 1e-10 and the 20,000-measurement cut-off are fixed at n39:218-220, with no comment giving a reason for any of the three.

**Stronger than the published code.**
- Two oracle arms read the simulation's true error correlation and spread, and their names say so (n39:216-218, 978-983), so the gap to the best possible weight is visible.
- The result file states that those two arms use the true values (n39:701).
- The model-error estimate is floored at 1% of the mean squared out-of-fold error (n39:221-223, 401-406), so no weight is infinite.
- When statsmodels is not used, the reason is written into the result (n39:367-371, fell_back_from_mixedlm_because).
- The two measured arms were rewritten so they read nothing the injector knows; their variance comes from the within-molecule spread and the model's out-of-fold error (n39:959-972).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole file was written at once: the MixedLM call, the centring, the alternating-least-squares fallback, the weight and the oracle arms. | `bd29125` | 2026-09-25 | "from the third research report" |
| The two weight arms that read the injected error spread were replaced by arms whose variance comes from the within-molecule spread and the out-of-fold error. | `1c13c88` | 2026-09-28 | "No noise method reads what only the injector knows" |
| The pass limit of 200, the tolerance of 1e-10, the 20,000-measurement cut-off and the 1% model-error floor were written once and never revised. | `bd29125` | 2026-09-25 | no reason in the message for any of the four numbers |

## Count and spread weights

Her code: `tests/experiments/noise/n19_measurement_count.py:496-521 (the variance), 524-564 (the curve-fitted variance), 567-597 (the three weights) and 692-699 (the inverse-variance weight), with the constants at 358 and n17_replicate_removal.py:215.`

Published source: https://github.com/wviechtb/metafor

statsmodels and metafor weight by one over the variance of the estimate and nothing else; hers offers three weights, the measurement count, one over the spread, and the count divided by the spread (n19:567-597). The floor and the fill have recorded reasons in the comments: a molecule whose measurements all agreed has zero variance and an infinite weight (n19:354-357), and dropping a molecule measured once would change how many molecules train (n19:499-507). No reason is recorded for the floor being 1% of the median rather than another share.

**Judgement: A deliberate change.** The inverse-variance weight is kept and two further weights are added beside it, each named and reportable as its own arm.

**Shortcuts taken.**
- The curve path takes the plain mean variance at each count and gives a count never seen the nearest count that was (n19:557-563); nothing is fitted or smoothed between counts.
- The three weights add no model-level error before inverting, where metafor adds a between-study variance; that term enters only through a separate function (n19:692-699).
- The floor share of 1% and the pooled prior are single fixed values for every dataset (n19:292, 358), neither swept.

**Stronger than the published code.**
- Every molecule's measurement variance is floored at 1% of the median variance (n19:358, 519-520), with the reason written above the constant.
- A molecule measured once is given the median variance of the molecules measured more than once, and how many were filled is written to the result file (n19:499-521).
- The curve path writes the fitted variance at each measurement count to the result file, so a reader can see whether it falls with the count or is flat (n19:538-543).
- The code raises rather than approximating when no training molecule was measured twice (n19:513-516) and when the weights cannot be normalised (n19:594-597).
- The 3-measurement threshold the curve path trusts carries its reason at n17:213-215: hold one measurement out of two and the consensus of the rest is a single measurement.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The three weights, the label variance with its floor and median fill, and the curve-fitted variance were all written when the file was created. | `2d80c00` | 2026-09-17 | no reason in the message |
| inverse_variance_weights, one over sigma_u squared plus the label variance, was added, along with the pooled-spread path. | `2703afa` | 2026-09-24 | "An outside research report on why label cleaning fails at herg_ki's accuracy" |
| MIN_MEASUREMENTS of 3, the threshold the curve path trusts, was written at n17:215 and has not been revised. | `1d9487b` | 2026-09-13 | no reason in the message; the comment above the constant gives one |

## Shrink by spread

Her code: `tests/experiments/noise/n19_measurement_count.py:600-621 (the label shrink) and 664-689 (the variance shrink), with the prior constant at 292 and the call sites at 1298-1300 and 1330.`

Published source: https://github.com/bioc/limma

limma fits both of its prior quantities from the spread of the variances; hers fixes the prior weight at 3 measurements' worth (n19:292) and uses the pooled variance as the prior value. The comment above the constant says what 3 does at two and at ten measurements and adds "One value, not swept", but gives no reason for 3 itself. The second half, moving the label itself toward an out-of-fold prediction, has no limma counterpart; the docstring at n19:605-609 names it as the James-Stein form.

**Judgement: A deliberate change.** The posterior-mean arithmetic is limma's and the prior value is still estimated from the data; only the prior's weight is fixed by hand.

**Shortcuts taken.**
- The prior weight is fixed at 3 measurements' worth (n19:292) where limma fits it from the variances themselves.
- There is no robust option; limma can winsorise the tails at 5% and 10% before fitting its prior.
- shrink_toward's spread is one number, the variance of label minus out-of-fold prediction over all training molecules (n19:614), reused for every molecule.

**Stronger than the published code.**
- The pooled variance is built from the training molecules only, and the docstring says to pass those only (n19:673-676).
- A molecule measured once gets exactly the pooled variance (n19:679-682), which is what limma returns when a variance has no degrees of freedom.
- shrink_toward leaves the labels untouched when the out-of-fold prediction reproduces them exactly, instead of moving every label the whole way (n19:615-619).
- The shrinkage fraction and the spread it shrinks toward are recorded per fold and per feature space (n19:1301-1305).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| shrink_toward, which moves each training label toward the out-of-fold prediction, was written when the file was created and has not been revised. | `2d80c00` | 2026-09-17 | no reason in the message |
| POOLED_SPREAD_PRIOR of 3 and pooled_spread_variance were added, which is the prior that replaces limma's fitted one. | `2703afa` | 2026-09-24 | "the spread weight pulled toward the pooled spread", from an outside research report |
| Neither the prior of 3 nor the pooled-variance arithmetic has changed since that commit; both were written once. | `2703afa` | 2026-09-24 | no reason in the message for the value 3 |

## Huber blend

Her code: `tests/experiments/noise/n33_robust_training.py:526-528 (the blend, inside label_arm_predict at 506), 603-607 (the teacher), 314-317 (the scale) and 241-242 (the two constants), with the inner fold count at 245. The range I was handed, n33:328-425, holds the loss swaps, which are a different arm; I read it as well.`

Published source: https://github.com/microsoft/LightGBM

The Huber loss itself is LightGBM's, run through the objective string unchanged (n19 cache call at n33:603-607). Two things around it are hers: the threshold is 1.4826 times the median absolute out-of-fold residual, recomputed per fold and feature space, where LightGBM's default is 0.9 in label units; and the downstream model is fitted on half the recorded label plus half the teacher's prediction (n33:526-527), so it never sees the recorded labels. The file header at n33:8-14 names an AI-written report of 2026-09-24, its item G, as the source; no reason is recorded for the half-and-half weight.

**Judgement: A deliberate change.** LightGBM's Huber objective is called unchanged; the data-set threshold and the blended label are stated additions, each its own named arm.

**Shortcuts taken.**
- The teacher's weight is fixed at 0.5 (n33:241) and never swept, so the downstream model is always fitted half on a prediction.
- The Huber threshold multiplier is fixed at 1.0 times the robust scale (n33:242), where scikit-learn fits the residual scale jointly with the coefficients.
- The teacher is computed once per fold and feature space and reused for every downstream model and every arm (n33:603-607).

**Stronger than the published code.**
- The Huber threshold is recomputed from the median absolute out-of-fold residual for every fold and every feature space (n33:313-317, 603-607), not fixed in label units.
- The teacher's prediction is out of fold over 5 inner folds (n33:244-245, 603-607), so a molecule's blended label does not contain its own fitted value.
- The residual kurtosis is written beside every pass (n33:320-325, 595-598), so the gain from a robust loss can be read against how heavy-tailed the residuals are.
- The arm is skipped when the robust scale is not finite or not positive (n33:600-602) rather than run with a bad threshold.
- The arm is twinned with a control through controls.py, named in the result file as a label move.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The teacher weight of 0.5, the Huber threshold multiplier of 1.0, the robust scale and the blended-label arm were all written in one commit. | `1e7b813` | 2026-09-24 | "from the second research report" |
| The teacher's out-of-fold prediction moved to the cached path, keyed on the label hash, and the arm was twinned with a control. | `aa98979` | 2026-09-25 | no reason in the message; the code comment says the same labels give the same prediction |
| Neither constant has been revised since it was written; the later commits to this file changed folds, models and feature spaces, not the blend. | `1e7b813` | 2026-09-24 | no reason in the message for either number |

## Small-loss selection

Her code: `src/kirby/utils/lnl_methods.py:568-656, class SmallLossSelection, read in full, plus the retrain helper _default_predict at lines 514-542 and the class defaults at 427-442. One network trains on every training molecule for 10 warmup epochs. The per-molecule loss is then computed once, in eval mode, over all training rows at once: squared error on z-scored labels for regression, cross-entropy for classification. The molecules with the smallest loss are kept, 70 out of every 100 of them (selection_rate=0.7), never fewer than 10 molecules. A fresh network is then fitted on the survivors through _default_predict, at the class default of 200 epochs.`

Published source: https://github.com/bhanML/Co-teaching

Co-teaching's own code trains two networks that pick each other's low-loss molecules, re-picking inside every mini-batch. Hers trains one network for 10 warmup epochs, picks the lowest-loss 70 of every 100 training molecules once, then refits. No reason is recorded; her docstring states that one-time rule and the registry name is small_loss, not co_teaching.

**Judgement: A deliberate change.** The class, its docstring and its registry key all say one-time small-loss selection, so no score is reported under Co-teaching's name.

**Shortcuts taken.**
- The low-loss molecules are picked once, after the 10 warmup epochs, and never re-picked, at lines 626-638; the published code re-picks inside every mini-batch of every epoch.
- One network picks its own training molecules, at lines 586-635; the second network that does the picking in the published method is absent.
- The share dropped is a flat 30 of every 100 training molecules, at line 575; the published code ramps the drop from none to the noise rate over the first 10 epochs.
- The warmup network is discarded and a new network is fitted on the survivors at line 641, so nothing carries over from the 10 warmup epochs.

**Stronger than the published code.**
- Records the per-molecule suspicion score, how many molecules were kept and removed, and the mean loss on each side, at lines 644-655; Co-teaching's code records none of these.
- Keeps at least 10 training molecules however small the fold, at line 637.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class arrived in one commit: the 70-in-100 keep rate, the 10 warmup epochs, the one-time pick and the refit. Written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message |
| The classification path was added: cross-entropy loss and a network with one output per class beside the regression path. The same day's commit 1421b3d describes the swap. | `ba1a577` | 2026-05-26 | no reason in the message |
| The per-molecule suspicion score was added to the returned metadata, which n09 grades against the molecules whose label really moved. The code comment credits an owner decision of 2026-09-12. | `1d9487b` | 2026-09-13 | no reason in the message |

## Confident learning

Her code: `src/kirby/utils/lnl_methods.py:659-745, class ConfidentLearning, read in full. Three networks are trained on all the training molecules, seeds differing by one, epochs capped at 100. Classification: the per-molecule error is one minus the mean predicted probability of the given class, and the spread is the standard deviation across the three networks, averaged over classes. Regression: the error is the absolute gap between the label and the mean of the three predictions, and the spread is the standard deviation of the three. A molecule is flagged when its error is above the 85th percentile of all training molecules' errors AND its spread is below the 25th percentile. Flagged molecules are dropped and a network is refitted on the rest, with the whole drop undone if fewer than 10 molecules would remain.`

Published source: https://github.com/cleanlab/cleanlab

cleanlab, the authors' own package, sets one probability threshold per class and predicts each molecule from a fold that excluded it. Hers trains 3 networks on all training molecules and flags those with error above the 85th percentile and spread below the 25th. No reason is recorded; the class never imports cleanlab, which the repository does install for a different detector.

**Judgement: Fixed 2026-10-29 (author's decision C, option 1).** Confident learning's defining mechanism, the per-class threshold and the count table of given label against confident class, was absent, so a score filed under confident_learning was not that method.

**What was done (2026-10-29).** `ConfidentLearning` now calls cleanlab 2.9.0. Classification: the network's class probabilities for every training molecule, predicted out of fold (5 folds, scaffold-grouped when the experiment passes groups), into `cleanlab.filter.find_label_issues`, which sets one threshold per class and reads the flags off the count table; the suspicion score is 1 minus cleanlab's label quality. Regression: `cleanlab.regression.learn.CleanLearning.find_label_issues`, the same call `noise_helpers.cleanlab_suspicion` makes for the detectors, with this module's network as the refitted model by default (`issue_model='dnn'`, about 80 network fits per call) or LightGBM (`issue_model='lightgbm'`, the detector's call; a LightGBM fit followed by a torch fit in one process crashes on a laptop, so it was not checked locally); the number flagged comes from cleanlab's own search, no percentile. The network is then retrained on the kept molecules. Which issue model the sweep runs is the author's sub-choice. Every stored n09 and n13 result for `confident_learning` is on the old footing. OPEN_THREADS A11.

**Git records no reason for this.** Written once in 9a0d827 with no reason in the message. The class never imports cleanlab, although setup.py installs it.

**Shortcuts taken.**
- The 3 networks predict the same molecules they trained on, at lines 686-713; cleanlab refits on 5 folds and predicts each molecule from the fold that excluded it.
- The share removed is fixed by two percentiles, at most 15 of every 100 training molecules, at lines 666 and 720-723; cleanlab's regression path searches five fractions from 0.01 to 0.2 for the best held-out R squared.
- The 3 networks differ only by a seed shifted by one and train on the same rows, at lines 692 and 711, so their spread is narrower than cleanlab's 5 refits with 2-fold cross-validation.
- The 3 scoring networks stop at 100 epochs while the refit at line 730 runs the class default of 200 epochs, so the model that scores and the model that predicts are trained for different lengths.

**Stronger than the published code.**
- Undoes the whole removal when fewer than 10 training molecules would survive, at lines 726-727.
- Records how many molecules were flagged and both threshold values, at lines 733-744.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class arrived in one commit: 3 networks, the 85th-percentile error cut and the 25th-percentile spread cut. Written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message |
| The classification path was added: per-molecule error as one minus the mean predicted probability of the given class, and the spread averaged over classes. | `ba1a577` | 2026-05-26 | no reason in the message |
| The per-molecule suspicion score was added to the returned metadata. The code comment at lines 735-738 credits an owner decision of 2026-09-12. | `1d9487b` | 2026-09-13 | no reason in the message |

## Distance-based selection

Her code: `src/kirby/utils/lnl_methods.py:748-830, class DistanceBasedSelection, read in full, plus tests/experiments/noise/n09_lnl_survey.py:253-306, class _SchemeDistanceSelection. Features are scaled, then each molecule's 10 nearest neighbours are found, itself excluded, under the process-wide distance (distance_config.sklearn_kwargs, manhattan by default). Regression: the suspicion score is the gap between the label and the mean of the neighbours' labels, divided by the spread of the neighbours' labels. Classification: the score is the share of the 10 neighbours carrying a different class. Molecules above the 85th percentile of that score are dropped, so 15 of every 100 training molecules go, and a network is refitted on the rest. The n09 subclass weights each neighbour by rank or by inverse square distance when the sweep hands a weighting scheme over, and leaves everything after the score unchanged.`

This class names no paper and no authors, so there is no published code to compare, and the source link is empty. It scores each molecule by the gap from its 10 nearest neighbours' labels, in units of those neighbours' spread. The worst 15 of every 100 training molecules are dropped; no reason is recorded for that share.

**Judgement: Moved to another task.** Nothing published was found to compare against; the construction is hers, closest to Wilson's 1972 nearest-neighbour editing rule moved from classification to regression.

**Shortcuts taken.**
- The suspicion score is computed once from the noisy training labels themselves and never recomputed, at lines 800-806.
- The feature scaler is fitted on every training molecule, including the ones about to be dropped, at lines 775-776.
- The 10 neighbours and the 85th-percentile cut are fixed by hand at line 754; nothing fits either from the data.

**Stronger than the published code.**
- Floors the neighbours' label spread at 1e-10 before dividing, at line 804, so neighbours carrying identical labels cannot divide by zero.
- Undoes the whole removal when fewer than 10 training molecules would survive, at line 812.
- Since 2026-09-25 the neighbour search reads the process-wide distance instead of silently using Euclidean, with the old behaviour written into the comment at lines 769-775.
- The n09 subclass weights each neighbour by rank or by inverse square distance when the sweep asks for it, at tests/experiments/noise/n09_lnl_survey.py lines 269-306.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class arrived in one commit: 10 neighbours, the label-gap-over-spread score and the 85th-percentile cut. Written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message |
| The classification path was added: the score becomes the share of the 10 neighbours carrying a different class. | `ba1a577` | 2026-05-26 | no reason in the message |
| The neighbour search changed from sklearn's Euclidean default to the process-wide distance, manhattan by default, so every score under the default moved. | `aa98979` | 2026-09-25 | "Every neighbour lookup in the noise experiments takes a similarity setting" |
| The feature scaler changed from sklearn's StandardScaler to the repository's FeatureScaler, which changes the scaled features the neighbour search runs on. | `5566961` | 2026-10-05 | no reason in the message |
| The n09 subclass that weights each neighbour by rank or inverse square distance was added; its own docstring, not the message, gives the reason. | `5566961` | 2026-10-05 | no reason in the message |

## Multistage cleaning

Her code: `src/kirby/utils/lnl_methods.py:837-923, class MultistageCleaning, read in full. Three rounds. Each round fits a network on the molecules still surviving, epochs capped at 100, then scores those same molecules: squared error for regression, minus the log probability of the given class for classification. The 10 in every 100 with the largest score are deleted, at least one molecule per round, and the rounds stop early if fewer than 10 molecules remain. After the third round a network is fitted on the survivors, 73 of every 100 training molecules.`

Published source: https://doi.org/10.48550/arXiv.2512.04319

MANTRA released no code, so nothing could be read against her rounds, and the link given is the paper's DOI. Each of 3 rounds fits a network on the survivors, scores those same molecules by loss, and deletes the worst 10 of every 100. That leaves 73 of every 100 training molecules; no reason is recorded for the rate or the count of rounds.

**Judgement: A deliberate change.** No MANTRA code exists, so no mechanism can be shown absent; hers is a plain fixed three-round deletion that the docstring states outright.

**Shortcuts taken.**
- Each round scores the molecules its own network has just trained on, at lines 873-897; nothing is held out.
- The stop check runs after the deletion, at lines 909-911, so the survivors can fall below 10 molecules.
- The rate of 10 in every 100 and the count of 3 rounds are fixed by hand at line 845; nothing fits them from the data.

**Stronger than the published code.**
- Removes at least one molecule per round, at line 899, so a round on a small fold is never a no-op.
- Records, per round, how many molecules were there before, how many were deleted and their mean loss, at lines 903-907.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class arrived in one commit: 3 rounds, 10 of every 100 deleted per round, in-sample loss as the score. Written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message |
| The classification path was added: each round scores a molecule by minus the log probability of its given class. | `ba1a577` | 2026-05-26 | no reason in the message |
| The method moved from the training-loop group to the data-level group, changing which group it is reported in and no number. | `e47e692` | 2026-10-09 | "scores, deletes, refits the final model on the survivors" |

## Neighbour-mean replacement of flagged labels (knn_smooth; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n07_remediation.py:1139-1168, with the flag share at n07_remediation.py:221, 235 and 291-298, and the percentile at n07_remediation.py:899`

Published source: https://github.com/scikit-learn-contrib/imbalanced-learn

The published editing rule deletes a molecule whose nearest neighbours disagree with its label; her code keeps the molecule and overwrites the label with its 9 neighbours' mean. The share relabelled is fixed at 5 of every 100 training molecules, and the file records why: 10 and 20 of every 100 lost 0.04 to 0.11 rmse on every logd pair. The published rule removes however many molecules disagree, so her count is set by the share and not by the data.

**Judgement: A deliberate change.** Relabelling rather than deleting, at a measured share, is a stated change and the neighbour-disagreement idea is still what drives it.

**Shortcuts taken.**
- The number of molecules relabelled is fixed by a share, where the published rule acts on however many disagree with their neighbours (n07_remediation.py:221).
- The comment at n07_remediation.py:237 says these neighbours are Euclidean, while the regression path runs Manhattan.
- The neighbour count is set by hand at 9 molecules on regression and 5 on classification, against a published default of 3 (n07_remediation.py:234-237).

**Stronger than the published code.**
- The share relabelled no longer reads how many molecules the injector moved, which a real user could never know (n07_remediation.py:271-291).
- Neighbour lists drop the molecule itself on both the regression and the classification path (n07_remediation.py:920-926 and :1148-1152).
- Only the flagged labels change; every other label keeps its exact value (n07_remediation.py:1163-1167).
- Neighbours are found under Manhattan distance, chosen over Euclidean on 14 representation blocks of two datasets (src/kirby/similarity/distance_config.py:1-40).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| knn_smooth was first written, replacing flagged labels by their neighbours' mean. | `f101232` | 2026-03-25 | no reason in the message |
| The flag share stopped being the fraction the injector corrupted and became a swept set of shares. | `01e309a` | 2026-09-28 | no reason in the message; the file records the owner's decision of that date |
| The regression flag share was cut to the single value of 5 of every 100 training molecules. | `5566961` | 2026-10-05 | no reason in the message; the file records the rmse loss at the two larger shares |
| The neighbour search switched from Euclidean to the suite's Manhattan setting. | `7546bae` | 2026-09-10 | no reason in the message |

## Forest-neighbour smoothing (prox_smooth; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n07_remediation.py:1170-1236, calling forest_leaf_ids at src/kirby/utils/noise_estimation.py:2234-2254 and forest_proximity_distance at noise_estimation.py:2257-2277, with FOREST_TREES and FOREST_MIN_SAMPLES_LEAF at noise_estimation.py:2228-2229 and PROX_SMOOTH_K and PROX_SMOOTH_SHARE at n07_remediation.py:227-228`

Published source: https://github.com/cran/randomForest

The authors' own proximity counts a pair of molecules only in trees where neither was in the bootstrap sample; hers counts all 100 trees for every pair. Their filling step weights every observed molecule by its proximity and repeats for 5 rounds; hers takes a flat mean over the 5 nearest and moves every label a quarter of the way, once. The commit that built it says the idea came from an AI-written review, not from a paper.

**Judgement: A deliberate change.** The forest-proximity neighbour list is intact and each setting carries a measurement, but the out-of-bag restriction and the repeated rounds of the published code are gone.

**Shortcuts taken.**
- Every tree counts toward every pair's proximity, where the published code counts only the trees in which neither molecule was in the bootstrap sample (noise_estimation.py:2257-2277).
- One pass moves the labels, where the published filling repeats for 5 rounds and refits the forest on the values it just wrote.
- The 5 nearest molecules count equally, where the published code weights each observed molecule by its proximity (n07_remediation.py:1230-1231).

**Stronger than the published code.**
- The forest is fitted on the corrupted labels on purpose, so the neighbour list is never handed the clean answer (noise_estimation.py:2243-2246).
- The proximity diagonal is set to infinity, so a molecule is never its own forest-neighbour (n07_remediation.py:1217).
- The tree count and leaf size were each measured on 36 paired cells of fold, corruption draw, detector and neighbour count, and the numbers are recorded (noise_estimation.py:2220-2230).
- The file states the memory cost as a square matrix and the training size at which the cleanup should be refused (n07_remediation.py:1182-1187).
- The forest runs single-threaded so it obeys the same thread policy as the rest of the run (noise_estimation.py:2247-2254).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| prox_smooth was built, taking the neighbour list from the forest's proximity instead of a feature-space distance. | `4217d3b` | 2026-09-23 | "The compass review's first proposal was to use an RF proximity as the kernel" |
| The forest went back to 100 trees after a half-finished sweep had moved it to 200. | `ea8a23e` | 2026-09-23 | "100 stands and the doubled forest fit is not paid" |
| The leaf size was settled at 5 molecules, against scikit-learn's default of 1. | `ea8a23e` | 2026-09-23 | "50 molecules a leaf loses 0.0214 average precision in 26 of the 36 cells" |
| The regression path stopped replacing a flagged share and now moves every label a quarter of the way toward its 5 forest-neighbours' mean. | `5566961` | 2026-10-05 | no reason in the message; the file records the f36 measurement |
| The 2,000-molecule limit on prox_smooth was removed. | `ec5b3f3` | 2026-09-25 | "owner decision" |

## Move toward GP prediction (gp_smooth; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n07_remediation.py:1266-1343, calling GaussianProcessEstimator.estimate at src/kirby/utils/noise_estimation.py:958-1079, with that class's own defaults at noise_estimation.py:925-927 and GP_SMOOTH_MAX_ROWS at n07_remediation.py:388`

Published source: https://github.com/scikit-learn/scikit-learn

The Gaussian process is scikit-learn's, called directly. She starts its length scale at the median distance between the training molecules instead of the package default of one, and narrows the bounds to a factor of 100 either side. The file records why: at the package default the kernel matrix underflows to the identity on a 1280-feature representation, and the process returned one constant per inner fold instead of a prediction.

**Judgement: A deliberate change.** The fitted process is the package's own, and the one change that matters is a data-driven starting point with its before-and-after measured on a named fold.

**Shortcuts taken.**
- Inside the estimator a training fold over 1000 molecules is still cut to a fixed-seed random subset, which n07's refusal avoids but other callers reach (noise_estimation.py:986-993).
- The white-noise level starts at a tenth with an upper bound of one, both set by hand, where the package leaves the bound at 1e5 (noise_estimation.py:1044-1045).
- The number of restarts of the hyperparameter optimiser is fixed at 3 in the class signature and no caller changes it (noise_estimation.py:925).

**Stronger than the published code.**
- The length scale starts at the data's own median distance, with the fitted values and the rank correlation against the clean labels recorded before and after (noise_estimation.py:994-1038).
- A fold with more than 1000 training molecules is refused with a printed reason rather than fitted on a subsample (n07_remediation.py:1325-1335).
- The features and labels are standardised on the inner training fold only, then the prediction is converted back to label units (noise_estimation.py:966-971 and :1058-1062).
- The posterior mean is read out of fold and the folds keep each scaffold group whole (n07_remediation.py:1336-1338, noise_estimation.py:1065).
- The file records that every gp_smooth number from before the fix is a number for a different quantity (n07_remediation.py:1306-1314).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The RBF length scale moved from one to the median distance between training molecules, with bounds a factor of 100 either side. | `2f1ceb6` | 2026-09-21 | no reason in the message, which is about a different experiment's recovery gap |
| The row cap moved into the branch that does the fitting, so n07 refuses above 1000 molecules instead of fitting an uncapped process. | `d90f3c9` | 2026-09-20 | "a target fitted on a subsample is a different quantity from one fitted on the fold" |
| The 1000-molecule fixed-seed subset inside the estimator was added. | `7a98aa1` | 2026-08-19 | no reason in the message |
| The white-noise starting level of a tenth and its upper bound of one were written, and have never been revised. | `8b20490` | 2026-01-09 | no reason in the message |
| gp_smooth was added to the cleanups n07 runs. | `03f5abc` | 2026-09-17 | no reason in the message for the cleanup itself |

## Replace flagged labels by the prediction (n31 rewrite and bayes_shrink; not on the Chapter 7 list by these names)

Her code: `tests/experiments/noise/n31_regime_and_breakeven.py:411-433, with 'rewrite' itself at n31_regime_and_breakeven.py:422-425 and the flag built by detector_flags at n31_regime_and_breakeven.py:384-389`

Published source: https://doi.org/10.48550/arXiv.1412.6596

No code by the authors of the published method exists, so nothing was compared line by line. Her code takes the training molecules with the largest absolute out-of-fold residual and sets their labels to the out-of-fold prediction at those molecules. The commit that built the file says it came from an AI-written research report, not from a paper, and no reason is recorded for the down-weight value beside it.

**Judgement: A deliberate change.** There is no published code to depart from, and the operation is complete in itself rather than a published mechanism with a piece missing.

**Shortcuts taken.**
- One set of out-of-fold residuals both picks the flagged molecules and supplies their new labels, so the selection signal and the replacement value are the same quantity (n31:875, :948 and :422-425).
- The flag count is a fixed share of the training molecules, so labels are rewritten even on a run where nothing was corrupted (n31:933-936).

**Stronger than the published code.**
- The share flagged is swept over three values and no longer reads how many molecules the injection moved (n31_regime_and_breakeven.py:256-260).
- The prediction is out of fold and cached per dataset, so the same molecules are never scored by a model that trained on them (n31:865-868).
- Only the flagged labels change; every other label keeps its exact value (n31:422-425).
- The flagged molecules are chosen with a stable sort, so ties break the same way on every run (n31:384-389).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The rewrite operation and the down-weight value were written in one commit, together with the whole file. | `1e7b813` | 2026-09-24 | "from the second research report" — an AI-written report, not a paper |
| The flagged share stopped being the number the injection moved and became a swept set of shares. | `1c13c88` | 2026-09-28 | "no longer the number the injection moved or the condition's nominal share" |

## Shrink flagged labels toward the prediction (n32 and n42 shrink_flagged; not on the Chapter 7 list by this name)

Her code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:1273-1302, with the flag built by lowest_share at n32_iterative_and_dual_selection.py:1198-1205`

Published source: https://doi.org/10.1080/01621459.1975.10479864

There is no published code that shrinks a label toward a regression prediction, so nothing was compared line by line. Her weight is the extra mean square on the flagged molecules divided by the total mean square, applied once per flag set and once per molecule. The comment records why mean squares replaced variances: under a one-direction shift the flagged residuals cluster, so a variance-based weight came out zero and nothing moved.

**Judgement: A deliberate change.** No published implementation exists to depart from, and the one arithmetic choice that changes a number is recorded with the observation that drove it.

**Shortcuts taken.**
- One set of out-of-fold residuals picks the flagged molecules and sets the shrink weight, so the flag and the weight are not independent (n32:1276-1289).
- The same weight rule is written a second time in another file, so a change in one place does not reach the other (n31_regime_and_breakeven.py:392-408).

**Stronger than the published code.**
- The comment records the measurement that chose mean squares over variances, on the smoke test's data (n32_iterative_and_dual_selection.py:1282-1286).
- The function returns nothing and names the reason when either side holds fewer than 2 molecules (n32:1278-1281).
- The extra mean square is floored at zero, so the weight cannot go negative (n32:1288).
- The per-molecule weight is set to zero for every molecule when the unflagged mean square is zero, rather than dividing by it (n32:1293).
- The molecule count, both mean squares and both weights are written into the output (n32:1298-1303).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| shrink_flagged, both weight columns and the mean-squares note were written in one commit and have never been revised. | `1e7b813` | 2026-09-24 | "from the second research report" — an AI-written report, not a paper |

## Huber-shaped down-weight (huber_weight; not the Chapter 7 "Huber blend")

Her code: `tests/experiments/noise/n07_remediation.py:973-998`

Published source: https://github.com/statsmodels/statsmodels

Her weight rule and tuning constant are the published ones: statsmodels returns one inside the cut and the constant times scale over the score outside it, and 1.345 is the default in both. What is reweighted differs. statsmodels divides the residual of the linear model it is fitting and recomputes weights and scale each pass, up to fifty; hers divides a per-molecule detector score once and hands the weight to whatever model the arm fits next. The comment at n07:977-986 gives the reason for having this arm beside the smooth down-weight: this weight is flat for every molecule inside the scale, so an ordinary-looking label is not touched.

**Judgement: Moved to another task.** The weight function and its constant are unchanged; what moved is the thing reweighted, from a linear M-estimator's own residual to a sample weight for any downstream model.

**Shortcuts taken.**
- The comment at n07:974 calls the weight redescending; the code at n07:993 floors it above zero and never returns it to zero, so the comment describes different arithmetic.
- The weights and the scale are computed once at n07:987-993, where the published fit recomputes both each pass until the deviance moves by less than a tiny tolerance.
- The scale at n07:988 is the median of the absolute detector scores, not of the downstream model's residuals, so the cut is set on a different quantity from the one the weights act on.

**Stronger than the published code.**
- A fold where every score is identical gets a flat weight rather than a division by zero, at n07:985-987, with the reason written beside it.
- The denominator is floored by a small constant at n07:993, so a molecule with a zero score does not divide by zero.
- The weights are divided by their mean at n07:996, so the total weight does not change with the detector's scale.
- The detector score is produced out of fold with scaffold groups kept together, at n07:1638-1642, where statsmodels fits in sample.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole arm appeared at once: the weight, the tuning constant 1.345, the scale constant 1.4826 and the division by the mean. | `c2836f4` | 2026-09-23 | "from the label-noise review ... the family that review recommends over deletion on small datasets" |
| The arm began running under all fourteen noise conditions instead of one. | `c46b71e` | 2026-09-23 | "n07 ran one noise condition for the life of the project" |
| The arm is recorded as not run, rather than raising, for a model that takes no per-molecule weight. | `069fa32` | 2026-10-07 | "gp takes no per-molecule weight: weighting cleanups recorded as not run, never raised" |

## Clip the residual (winsorise; not on the Chapter 7 list)

Her code: `src/kirby/utils/noise_estimation.py:2138-2175 (clip_to_range) and tests/experiments/noise/n30_censored_likelihood.py:410-425 (winsorise), with the sweep of c at n30_censored_likelihood.py:232 and the clip_to_range caller at tests/experiments/noise/n07_remediation.py:1383-1400`

Published source: https://github.com/scipy/scipy

The published code clips the labels themselves to the value sitting at a rank, so a fixed count of labels always moves. Her winsorise clips the residual against an out-of-fold prediction and adds it back, so how many labels move depends on the spread. The clip width is swept over three values the constant's own comment attributes to an AI-written report, not to the data.

**Judgement: A deliberate change.** Clipping a residual against a held-out prediction rather than clipping the raw labels is a stated change that suits a regression target.

**Shortcuts taken.**
- The scale subtracts no median before taking the absolute residual, so it differs from the published robust scale whenever the residuals sit off zero (n30:417).
- The three clip widths come from a range quoted out of an AI-written report rather than from the data (n30:230-232).

**Stronger than the published code.**
- A label that did not move keeps its exact value rather than the prediction plus its own residual (n30_censored_likelihood.py:421-423).
- The function returns how many labels moved, which the published code does not report (n30:424-425).
- The out-of-fold prediction splits by scaffold group (n30:395-405).
- A scale that is not positive or not finite returns the labels untouched instead of clipping to nothing (n30:418-419).
- clip_to_range sorts each molecule's two edges, so passing them the wrong way round is not an error, and it checks all three arrays are the same length (noise_estimation.py:2164-2172).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| winsorise and the three clip widths were written in one commit; the arithmetic has never been revised. | `2703afa` | 2026-09-24 | "from the research report"; the widths' own comment says "The report's B5 range" |
| The function moved down the file when a censored-regression arm was added, with no change to the arithmetic. | `fce9a69` | 2026-10-06 | no reason in the message |
| clip_to_range was written, leaving every label inside its own range unchanged. | `2d80c00` | 2026-09-17 | no reason in the message; it was written once and never revised |

## Random removal of training molecules (n27; a control, not on the Chapter 7 list)

Her code: `tests/experiments/noise/n27_removal_cost_and_propagation.py:186-204 (removal_seed and kept_rows), with SHARES at 151 and REPEATS = 20 at 154; tests/experiments/noise/n24_delete_or_keep.py:426-445 (random_deletion_scores) and 447-462 (replicate_seed), with RANDOM_DELETION_SEEDS = (0, 1, 2, 3, 4) at 262 and the arm list at 408-425`

No published code removes training molecules at random as a control, so there is nothing to differ from: this is her own control, written twice. n27 keeps a random share of the training molecules at six shares with twenty draws at each share below all of them; n24 runs five independent replicates of the same procedure. The reason the seed is built with zlib.crc32 is stated at n27:186-196 and n24:447-460: Python randomises string hashing per process, so hash() would keep different molecules on every run of the same command.

**Judgement: A deliberate change.** There is no published counterpart, so the flag judges her control: it is deliberate, seeded reproducibly, and documented in both files.

**Shortcuts taken.**
- The control exists twice with different draw counts, twenty in n27:154 and five in n24:262, so the two files' controls are not read against each other.
- The share-1.0 reference in n27:151 is the only unremoved arm, and the shares below it are drawn independently rather than nested, so the kept set at a quarter is not a subset of the kept set at a half.

**Stronger than the published code.**
- The feature space is left out of the seed in both files, at n27:193-196 and n24:453-457, so the same molecules are removed in a base representation and in a hybrid and the space is the only difference between two rows.
- In n24 the control is a two-valued score handed to the same remove branch as every detector-driven deletion, at n24:426-445, so no second removal rule exists.
- Each random seed in n24 is a replicate of the whole procedure, drawing its own molecules per fold, at n24:447-452, and no mean across seeds is written anywhere.
- n27 floors the kept count at two molecules at n27:204, so no share leaves a fit with nothing.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The random keep, its seed and the twenty draws per share were written once in the commit that created n27. | `2703afa` | 2026-09-24 | "what training on 70-100% of the molecules costs" |
| The kept shares were widened from four values to six, adding a quarter and a half of the training molecules. | `1e7b813` | 2026-09-24 | no reason in the message |
| The n24 control, its five seeds and the two-valued score handed to the shared remove branch were written once. | `4c85148` | 2026-09-23 | "an arm that cleared all three random deletions AND was worse than leaving the labels alone" |
| The random-deletion seeds went from three to five, each reported on its own. | `4c85148` | 2026-09-23 | "three seeds is a spread read off two points" |
| The n27 seed moved to zlib.crc32 so the same command keeps the same molecules. | `aa98979` | 2026-09-25 | no reason in the message |

## Per-paper quantile map (n29; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n29_document_bias.py:530-551 (quantile_map_labels); threshold at 236-237; the mapped labels averaged per molecule at 598-600`

Published source: https://github.com/bmbolstad/preprocessCore

The published code builds the reference distribution as the mean of the sorted values across columns, so every column counts equally. Hers reads the reference from the quantiles of all training measurements pooled, so a document holding many measurements weighs more in it. Her code names no paper, and no reason for the pooled reference is recorded.

**Judgement: A deliberate change.** The mechanism, mapping each value's rank onto a reference distribution, is present; the reference is built differently so documents of unequal size can be mapped.

**Shortcuts taken.**
- The threshold of 20 measurements is fixed by hand, where the published code maps every column it is handed (QUANTILE_MAP_MIN_MEASUREMENTS, n29:237).
- A pooled reference lets one large document pull the reference toward itself, which the published per-column mean prevents (n29:549).
- A document that tested a stronger series is flattened toward the pooled distribution; the docstring says so and nothing in the file tests it (n29:76-79).

**Stronger than the published code.**
- The reference comes from that fold's training measurements only, so no held-out label enters the map (n29:961).
- Only a document with at least 20 training measurements is mapped; smaller documents keep their labels, because their quantiles are a handful of points (n29:236-237, 545-547).
- How many documents and how many measurements were mapped is written per fold (n29:961-965).
- The map reads labels only, so it is the same in every feature space, and the docstring says so (n29:80-83).
- Mapping documents of any size is her addition: the published code needs every column to hold the same number of rows.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| quantile_map_labels, the threshold of 20 training measurements, and the arm that averages the mapped labels per molecule. Written once and never revised. | `1e7b813` | 2026-09-24 | no reason in the message; it says only "new arms in n19/n27/n29, from the second research report" |
| The features the mapped labels are fitted in moved to per-fold searches on the fold's training rows. | `5566961` | 2026-10-05 | "searches on each outer fold's training rows ... whole-dataset searches removed" |

## Re-measurement priority rules (n36; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n36_remeasure_priority.py:327-352 (rank_candidates) and 354-358 (labels_after_budget); the two neighbour quantities at 224-263 (neighbour_stats) and 266-277 (neighbour_stats_from_lists); the rule list at 188; N_NEIGHBOURS = 5 at 200`

Published source: https://doi.org/10.1145/1401890.1401965

No code by the authors of the 2008 paper her docstring cites was found, so the five rules are her own reading of it. Her docstring at n36:53-58 says the report names no formula for leverage; she uses 1 plus the count of training molecules that hold a candidate among their nearest neighbours. The variance factor the report multiplies in was dropped: it is the same for every candidate and cannot change the order (n36:44-52).

**Judgement: Moved to another task.** The published work relabels crowd items over rounds; here a molecule has at most one spare measurement, so a rule ranks once and the budget is spent once.

**Shortcuts taken.**
- Each rule ranks the candidates once and the top share takes the budget; nothing is re-ranked as labels change (n36:858-880).
- Leverage has no published formula behind it, and the count of neighbourhoods a molecule feeds is her own choice (n36:53-58).
- A chosen molecule's label becomes the mean of 2 measurements, so a third measurement is never asked for (labels_after_budget, n36:354-358).
- neighbour_stats stays in the file although run() no longer calls it, and it still carries the old meaning of leverage (n36:224-263).

**Stronger than the published code.**
- Every rule is paired against 3 seeded random allocations, which the docstring names as the thing to beat (RANDOM_DRAWS, n36:192, 858-861).
- Ties are broken by a seeded draw, so no rule inherits the order the molecules sit in (n36:337-338, 350).
- The dropped variance factor was checked first: 2500 of 2500 toy orders were identical (commit f7f0b3f).
- Two references bracket every gain: every candidate on its one start measurement, and every training molecule on the mean of all its measurements (n36:79-84).
- Since 2026-10-06 the per-fold searches run on each seed's start labels, so a held-back measurement never picks the representations (n36:105-110).
- Neighbours use the process-wide distance after a bare sklearn Euclidean default was found slipping in, and the comment says so (n36:196-208).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The five rules, rank_candidates and labels_after_budget. Written once and never revised. | `1e7b813` | 2026-09-24 | no reason in the message; it says only "from the second research report" |
| The variance factor was dropped and the rules renamed to leverage and leverage_x_disagreement. | `f7f0b3f` | 2026-09-25 | "the variance factor was the same for every molecule, so the rule ranked by leverage alone" |
| The neighbour search moved from sklearn's Euclidean default to the process-wide distance, and every lookup took a similarity setting. | `aa98979` | 2026-09-25 | "Every neighbour lookup in the noise experiments takes a similarity setting" |
| Per-fold searches, two decided models and four configurations each, plus the neighbour settings read through lists. | `5566961` | 2026-10-05 | "searches on each outer fold's training rows ... whole-dataset searches removed" |
| Where two kept neighbour combinations differ only in the counting scheme, the next distinct setting and k down the ranking is fitted instead. | `f2d2c98` | 2026-10-06 | "Adelaide delegated the open decisions of chats 2-5" |

## Retuning

Her code: `tests/experiments/noise/n40_model_side_regularisation.py:244-250 (the tuner's settings), 258-295 (TUNE_GRIDS), 359-366 (tuned_model), 369-406 (tune), 1047-1084 (where the retuned and tuned_clean_reused arms are fitted)`

Published source: https://github.com/scikit-learn/scikit-learn

Her tuner is sklearn's draw of settings with five things changed around it: the model's own default settings go in as candidate zero, the inner folds are scaffold-grouped and cut from this fold's training molecules only, the winner is picked on the dataset's own metric, the draw seed is fixed, and the support vector machine's epsilon is converted from label standard deviations per fit. The docstring at n40:377-381 gives the reason for candidate zero: a dirty-run finding of 2026-10-05, "n40: retuning never tries the default". The comment at n40:259-263 gives the reason for the epsilon conversion: one grid then means the same thing on pKi and on logD.

**Judgement: A deliberate change.** The drawn search is sklearn's; what she changed around it is grouped inner folds, the dataset's metric and an extra default candidate, each written down as a choice.

**Shortcuts taken.**
- The draw seed is fixed, so every fold and every injection scores the same twenty drawn settings instead of fresh draws (n40:248-250, 386-387).

**Stronger than the published code.**
- The inner folds hold scaffold groups together and are cut from this fold's training molecules alone, so the held-out fold never reaches the tuner (n40:369-392).
- The winner is picked on the dataset's own metric with a direction flag, not on the regressor's R squared (n40:396-402).
- The model's default settings are candidate zero, so the retuned arm can lose to the fixed arm only through the inner folds' choice (n40:386-387).
- The tuner raises rather than scoring when a training fold holds fewer than two scaffold groups (n40:389-391).
- The support vector machine's epsilon is multiplied by the standard deviation of the fold's training labels at every fit (n40:357-364).
- A second arm tunes on the clean labels and fits the chosen setting to the noisy ones, so retuning for the noise is separated from tuning at all (n40:1056-1070).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The tuner's settings written: twenty drawn settings, three scaffold-grouped inner folds, a fixed draw seed, and the grids widened toward stronger regularisation | `bd29125` | 2026-09-25 | no reason in the message |
| The model's default settings added as candidate zero, and chose_default recorded per fold | `5566961` | 2026-10-05 | no reason in the message; the code's docstring names the dirty-run finding "n40: retuning never tries the default" |
| The support vector machine's epsilon_sd conversion, written with the tuner and never revised since | `bd29125` | 2026-09-25 | no reason in the message |

## Seed averaging

Her code: `tests/experiments/noise/n40_model_side_regularisation.py:297-303 (SEED_AVG_N and SEED_SUBSAMPLE), 414-418 (seeded_model), 1084-1101 (the seed_one and seed_avg arms). Ten fits of the same model on the same training molecules, differing only in random_state, and the ten held-out predictions averaged. seed_one keeps the first fit, so the averaging is not confounded with the subsampling. What makes two seeds differ is set per model: lightgbm gets row subsampling 0.8 with bagging every iteration and feature subsampling 0.8, xgboost gets row subsampling 0.8 and nothing else, and dnn gets no subsampling at all, so only its weight initialisation and its batch order move.`

Ten fits of one model on the same training molecules, differing only in the random seed, with the ten held-out predictions averaged. No published code for this exists, so nothing was left out of one; the comment at n40:297 records what makes two seeds differ, which is row and feature subsampling set per model. No reason is recorded for the number of fits being ten.

**Judgement: A deliberate change.** No published implementation exists to depart from, and the arm is standard practice built with a matched single-fit control beside it.

**Shortcuts taken.**
- The dnn gets no subsampling, so its ten fits differ only in weight initialisation and batch order while the two boosters also resample rows and features (n40:297-303).
- The fit count is fixed at ten and the subsampling shares at four fifths, with no reason recorded for either (n40:297-303).

**Stronger than the published code.**
- A single-fit arm keeps the first of the ten fits, so the averaging is not confounded with the subsampling that makes the seeds differ (n40:1084-1095).
- The mean spread of the ten predictions per molecule is written into the result beside the score (n40:1096-1099).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| Ten seeds, the per-model subsampling shares, and the two arms written | `bd29125` | 2026-09-25 | no reason in the message |

## Stacking

Her code: `tests/experiments/noise/n40_model_side_regularisation.py:307 (OOF_SPLITS), 574-580 (_oof_model), 1107-1113 (the out-of-fold predictions), 1117-1130 (the stack). Out-of-fold predictions of the dataset's decided models go into non-negative least squares (scipy.optimize.nnls) against the noisy training labels; the weights are applied to the same models' held-out predictions from the fixed arm.`

Published source: https://github.com/scikit-learn/scikit-learn

The combiner is scipy's non-negative least squares with no intercept, where sklearn's stacking uses a ridge fit with an intercept and allows negative weights, and the out-of-fold predictions that feed it come from scaffold-grouped folds. Each model is stacked in its own feature matrix for the configuration being stacked. The header at n40:66-70 states what the arm computes; no comment or commit message gives a reason for the non-negativity.

**Judgement: A deliberate change.** Both halves of stacking are intact, out-of-fold inputs and weights applied to models refitted on all the training molecules; only the combiner is simpler.

**Shortcuts taken.**
- The combiner fits no intercept, so a bias shared by both models' out-of-fold predictions cannot be corrected (n40:1120).

**Stronger than the published code.**
- The out-of-fold predictions are built on scaffold-grouped folds, cut to the number of scaffold groups when there are fewer (n40:574-580).
- Each model is stacked in its own feature matrix for the configuration, so neither model is forced into the other's features (n40:1107-1113).
- The stack is dropped for that pass when the weights sum to zero, instead of a prediction being returned anyway (n40:1122-1125).
- The weight given to each model is written into the result (n40:1126).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The stack written: non-negative least squares on out-of-fold predictions, five inner folds, the dropped-stack guard | `bd29125` | 2026-09-25 | no reason in the message |

## 1,000-tree RF

Her code: `tests/experiments/noise/n40_model_side_regularisation.py:304 (RF_BIG_TREES) and 1100-1106 (the rf_1000_trees arm), with the forest it starts from at src/kirby/find_best_hybrid.py:2757-2763. The arm is that registry forest with the tree count set to 1,000 and everything else left alone: sklearn's RandomForestRegressor with max_features 'sqrt', random_state 42 and n_jobs -1.`

Published source: https://github.com/cran/randomForest

The arm is the registry's forest with the tree count raised from one hundred to one thousand and nothing else touched. The registry forest tries the square root of the number of features at each split and lets a terminal node hold one molecule, where Breiman and Cutler's forest tries a third of the features and holds at least five molecules. The header at n40:52 says only "RF_BIG_TREES trees in place of the default 100"; no reason is recorded for the number one thousand.

**Judgement: Settings only.** Only numbers differ: the tree count, the features tried per split and the smallest terminal node.

**Shortcuts taken.**
- Two of the three differences from Breiman and Cutler's forest, the square root of the features and the single-molecule leaf, come from the registry entry and are carried into this arm unexamined (find_best_hybrid.py:2757-2761).

**Stronger than the published code.**
- The arm changes the tree count alone from the fixed-default forest it is paired against, so the pair differs in one thing (n40:1100-1106).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| One thousand trees and the arm that fits them | `bd29125` | 2026-09-25 | no reason in the message |
| The registry forest set to try the square root of the number of features at each split | `d7f6adf` | 2026-03-04 | no reason in the message |

## Early learning

Her code: `src/kirby/utils/lnl_methods.py:926-1072, class EarlyLearning, read in full, plus the carve helper at lines 469-501. One fifth of the training molecules are carved off to choose the kept checkpoint. Every molecule starts with a weight of one. Each epoch trains on the weighted loss, weighted squared error for regression or weighted cross-entropy for classification. After each epoch every training molecule's loss is recomputed, and the weight of each molecule above the median loss is multiplied by 0.95 (gamma). Weights are then clipped to between 0.01 and 1.0 and divided by their own mean. Training stops when the carved molecules' loss has not improved for 20 epochs, the class default, out of at most 200 epochs.`

Published source: https://github.com/shengliu66/ELR

The paper her docstring cites released no code, so the link is ELR's repository, a different paper by different authors. ELR adds a loss term pulling each prediction toward that molecule's own running average; hers adds no term. Instead it multiplies the weight of every molecule above the median loss by 0.95 each epoch; no reason is recorded.

**Judgement: A deliberate change.** A per-molecule weight decay replaces ELR's extra loss term, simpler and stated in the docstring, and the class claims early learning rather than ELR.

**Shortcuts taken.**
- The per-molecule loss is recomputed on the same molecules the network has just trained on, at lines 1013-1025.
- The median split decays half the training molecules every epoch, at lines 1026-1027, however clean the labels are.
- The decay of 0.95, the clip bounds and the median split are fixed by hand, at lines 934 and 1028.

**Stronger than the published code.**
- Carves a fifth of the training molecules out to choose which checkpoint to keep, group-disjoint, at line 948 and lines 469-501, so the held-out fold is only ever predicted.
- Floors every weight at 0.01, at line 1028, so no molecule is ever fully dropped.
- Divides the weights by their own mean, at line 1029, so the loss scale does not drift as weights decay.
- Caps regression predictions to the training-label range, at lines 1061-1062.
- Records the mean and spread of the weights and the share of molecules below weight 0.5, at lines 1067-1070.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class arrived in one commit: the decay of 0.95, the median split, the clip to between 0.01 and 1, the divide by the mean. Written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message |
| The classification path was added: weighted cross-entropy beside the weighted squared error. | `ba1a577` | 2026-05-26 | no reason in the message |
| A fifth of the training molecules is now carved off to choose the kept checkpoint; before this the method read the held-out fold's clean labels, so every score moved. | `edd07b0` | 2026-09-04 | "methods stop on training rows, not on the labels they are scored against" |
| Regression predictions are now capped to the training-label range, so predictions outside that range changed. | `60d635a` | 2026-10-08 | "n09 survey networks clip predictions to the training-label range" |

## Uncertainty curriculum

Her code: `src/kirby/utils/lnl_methods.py:1075-1232, class UncertaintyCurriculum, read in full. One fifth of the training molecules are carved off to choose the kept checkpoint; the fractions below are of the four fifths that remain. The first 20 epochs train on all of them. The remaining 180 epochs are split into four stages of 45 epochs each. In each stage every molecule's loss is recomputed at the start of each epoch, and only the lowest-loss share trains that epoch: a quarter of the molecules in the first stage, then half, then three quarters, then all of them, never fewer than 10 molecules. Training stops when the carved molecules' loss has not improved for 20 epochs. The score that orders the molecules is each molecule's own loss; no spread across models or draws is computed anywhere in the class, despite the name.`

Published source: https://doi.org/10.1145/1553374.1553380

Neither cited source released code, so the link is the DOI of Bengio et al. 2009. No spread across models, seeds or draws is computed anywhere in the class, although it is named uncertainty curriculum. It sorts molecules by their own training loss and trains on the lowest quarter, then half, then three quarters, then all.

**Judgement: Renamed Self-paced learning and fixed 2026-10-29 (author's decision D).** The quantity the old name promised, a spread across models or draws, was computed nowhere, so a score reported as an uncertainty curriculum was ordered by training loss instead.

**What was done (2026-10-29).** Renamed `SelfPacedLearning`, registry key `self_paced_learning`, citing Kumar, Packer and Koller 2010 with Bengio 2009 for the easy-first schedule; the thesis LNL-sweep row, the two results-table labels and the analysis lists (`ch7_numbers.py`, `ch7_numbers_expected.json`, `group_table/build.py`, `tables_tex.py`) carry the new name. Early stopping: the patience counter restarts at each stage, and running out of patience inside a stage that is not the last opens the next stage instead of ending the run; before, on one fold of buchwald_hartwig the run ended at epoch 40 in stage 1 of 4. `stages_reached` is written into the metadata. Two variants the author asked for are added under local test: `self_paced_uncertainty_weighted` (selection by loss; each molecule's loss divided by the spread of 10 dropout passes, mean 1 over the active set) and `self_paced_uncertainty_ranked` (selection by loss divided by that spread, so a confident mistake comes in last). Whether they stay in the sweep is the author's call after the one-fold check (OPEN_THREADS A12). Every stored n09 and n13 result for `uncertainty_curriculum` is on the old footing.

**Closed 2026-10-30 (author).** The two uncertainty variants were tested on one fold each of buchwald_hartwig, logd and herg_ki (numbers in OPEN_THREADS A12): neither separated from the plain rule by more than 0.01 rmse, and every self-paced arm lost to the plain DNN at the highest noise level. Only regular self-paced learning stays, registry `self_paced_learning`; the variants, the `mode` and `checkpoint_from` arguments and the dropout-spread code are removed. The kept checkpoint is the last stage, so the prediction is the self-paced network; before, it could come from the warm-up, and on buchwald_hartwig fold 0 it always did (epoch 9), so the stored `uncertainty_curriculum` scores are a 9-epoch plain DNN.

**Git records no reason for this.** Written once in 9a0d827 with no reason in the message.

**Shortcuts taken.**
- No spread across models, seeds or dropout draws is computed anywhere in lines 1075-1232, although the class is called UncertaintyCurriculum.
- Each stage sorts molecules by a loss the network computed on the very rows it trained on, at lines 1152-1166.
- The four stages run 45 epochs each while early stopping has a patience of 20 epochs, at lines 1134-1135 and 1209-1211, so a run can end before the later stages are ever reached.
- The four shares and the 20 warmup epochs are fixed by hand at line 1082.

**Stronger than the published code.**
- Carves a fifth of the training molecules out to choose which checkpoint to keep, at line 1097, so the held-out fold is only ever predicted.
- Keeps at least 10 molecules in every stage, at line 1163.
- Caps regression predictions to the training-label range, at lines 1223-1224.
- Records the share of molecules included at the end, at lines 1229-1230.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class arrived in one commit: 20 warmup epochs, the four shares of a quarter, half, three quarters and all, and the loss-based ordering. Written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message |
| The classification path was added: cross-entropy per molecule as the ordering score. | `ba1a577` | 2026-05-26 | no reason in the message |
| A fifth of the training molecules is now carved off to choose the kept checkpoint; before this the method read the held-out fold's clean labels, so every score moved. | `edd07b0` | 2026-09-04 | "methods stop on training rows, not on the labels they are scored against" |
| Regression predictions are now capped to the training-label range, so predictions outside that range changed. | `60d635a` | 2026-10-08 | "n09 survey networks clip predictions to the training-label range" |

## DivideMix

Her code: `src/kirby/utils/lnl_methods.py:1239-1434 (class DivideMixReg), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/LiJunnan1992/DivideMix

Her code keeps the DivideMix skeleton: fit a two-component mixture to each of two networks' per-sample losses, then train each network on the molecules the other called clean. The high-loss molecules are then dropped from that epoch, where the published code trains on them with guessed labels, blends clean labels with the network's own prediction, and mixes sample pairs. Her class docstring at line 1242 calls it "Regression-adapted DivideMix: dual DNNs + GMM on squared errors"; no reason is recorded for leaving out the semi-supervised half.

**Judgement: Fixed 2026-10-30 (author's decision F).** Before: the published method's defining step is training on the high-loss molecules with guessed labels, and line 1372 dropped them instead.

**What was done (2026-10-30).** The noisy set stays in the epoch with each molecule's label replaced by the average of the two networks' predictions; each network trains on all molecules with the other network's split deciding the label. Two faults found in the one-fold comparison were fixed as well: the mixture is fitted to the min-max scaled loss as published, not the log loss (which called 73% of clean herg_ki noisy), and no checkpoint is kept until 10 divide-and-train epochs after the warm-up (the kept checkpoint had been epoch 11 in four of six cells). Co-refinement and pair mixing stay out. Numbers in OPEN_THREADS A14. Every stored `dividemix` result (n09, n13) is on the old footing.

**What to fix.** Train on the high-loss molecules with guessed targets, as the published code does, or drop the name and call the arm loss-mixture co-selection. Line 1372 drops them from the epoch instead, and that is the step the published method is built on.

**Git records no reason for this.** The whole class landed in one commit, 9a0d827 on 2026-04-04, with no reason in the message, and those lines were never revised.

**Shortcuts taken.**
- The molecules in the high-loss component are dropped from the epoch's index list at lines 1372-1373, instead of being trained on with guessed labels.
- No line in the class blends a training label with the network's own prediction, so the published co-refinement of clean labels is absent.
- The mixture is fit on the natural log of the loss at lines 1349 and 1361, where the published code scales the loss to the range zero to one first.
- The mixture's covariance floor is left at sklearn's default at lines 1351 and 1363; the published code sets a floor five hundred times larger.
- Only the current epoch's losses are used, line 1348, where the published code averages the last five epochs' losses at the highest noise rate.

**Stronger than the published code.**
- The checkpoint is chosen on a fifth of the training molecules carved out with scaffold groups kept together, lines 1259-1261 and base lines 469-501, where the published code holds nothing out and keeps no checkpoint.
- Each mixture fit is wrapped so that a failure keeps every training molecule rather than stopping the run, lines 1350-1358 and 1362-1370.
- The percentage of training molecules each of the two networks called clean is recorded, lines 1431-1432; the published code records no such count.
- Regression predictions are capped to the range of the training labels, line 1424, so one extreme held-out molecule cannot send a fold's error to thousands.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class, including dropping the high-loss molecules, the log-loss mixture input, the default mixture settings and the ten warm-up epochs | `9a0d827` | 2026-04-04 | no reason in the message; the class arrived with the file and these lines were written once and never revised |
| The two cross-training calls were refactored into one shared step function so classification could share it; the molecules dropped stayed the same | `ba1a577` | 2026-05-26 | no reason in the message for this class |
| The checkpoint is now chosen on a fifth of the training molecules instead of the held-out fold's clean labels | `edd07b0` | 2026-09-04 | "a method that read those rows well looked good at the thing the survey measures" |
| Regression predictions capped to the training-label range | `60d635a` | 2026-10-08 | "an unbounded network output on an extreme held-out feature put one fold's RMSE in the thousands" |

## Meta-Weight-Net

Her code: `src/kirby/utils/lnl_methods.py:1437-1625 (class MetaWeightNet), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/xjtushujun/meta-weight-net

The bilevel step of Meta-Weight-Net is present: the detached per-sample loss feeds a small weight network, one virtual parameter step is taken, and the loss under those virtual parameters is pushed back into the weight network. Three things are substituted: the meta loss is measured on a fifth of the training molecules rather than a clean held-out set, the step runs every fifth epoch rather than every batch, and the weights are reused for a whole epoch. Her data has no clean labels to hold out, and commit edd07b0 records why the carve replaced the held-out fold.

**Judgement: A deliberate change.** The learned loss-to-weight mapping and its second-order path both run; what changed is how often it runs and which molecules the meta loss is measured on.

**Shortcuts taken.**
- The class docstring at line 1442 says "first-order approximation (no second-order gradients)", and line 1554 passes create_graph=True, so the code keeps the second-order path the docstring denies.
- The weights are computed once per epoch over all training molecules at lines 1567-1570 and reused for every batch; the published code recomputes them for each batch.
- The meta step runs on every fifth epoch, line 1532, where the published code takes one for every batch.
- The step size inside the virtual update is fixed at one thousandth, lines 1555-1556; the published code uses the main rate, which decays twice during training.
- The virtual step is computed on one pass over all training molecules, line 1541, where the published code uses the current batch.

**Stronger than the published code.**
- The meta loss and the checkpoint both use a fifth of the training molecules carved with scaffold groups kept together, lines 1462-1463 and 1558-1559, so the fold the method is scored on is only ever predicted.
- Dropout is switched off for the forward pass the meta-gradient is taken through, line 1540, so that gradient is the same given the seed; the published code leaves training mode on.
- The mean and spread of the weights the weight network produced are recorded, lines 1622-1623; the published code records neither.
- Regression predictions are capped to the range of the training labels, line 1615.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The weight network with thirty-two hidden units, the five-epoch interval, and the docstring line claiming no second-order gradients | `9a0d827` | 2026-04-04 | no reason in the message; written once with the file and never revised |
| The weights stopped being detached before the inner step; the virtual update with create_graph and the dropout-off forward pass were added | `7a98aa1` | 2026-08-19 | "the validation loss had no gradient path back — it never learned" |
| The meta loss moved from the held-out fold's clean labels to the carved fifth of the training molecules | `edd07b0` | 2026-09-04 | "The held-out fold is only ever predicted" |
| Regression predictions capped to the training-label range | `60d635a` | 2026-10-08 | "an unbounded network output on an extreme held-out feature put one fold's RMSE in the thousands" |

## MentorNet

Her code: `src/kirby/utils/lnl_methods.py:1628-1833 (class MentorNet), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/google/mentornet

The published mentor is fit in a separate run against a supplied target weight per sample, then frozen while the student trains, and it reads the loss, the loss minus a running average, the label and the training phase through a recurrent layer and two embeddings. Hers is a fresh one-hidden-layer network fit online, one step per epoch, to reproduce a fixed self-paced formula of the current loss with an annealed threshold. Commit 7a98aa1 records why: the earlier objective's best answer was to set every weight to zero.

**Fixed 2026-10-30 (author's decision H).** Before: the mentor was fitted one step per epoch during
student training to copy a self-paced rule, never frozen, no burn-in. Now: 10 burn-in epochs at
weight 1; the mentor fitted on the burn-in's loss features against the self-paced rule at five
epoch fractions (`lnl_parts.self_paced_weight`), then frozen; weighted training under it. This
is the predefined-curriculum variant (MentorNet PD). The data-driven variant (MentorNet DD) is
added as `MentorNetDD`, key `mentornet_dd`: mentor fitted on a trusted set with half its labels
corrupted, weight 1 untouched / 0 corrupted; the trusted set defaults to the selection carve
(not clean; the author's open choice). Still different from the published code: no moving
average subtracted from the loss, no recurrent layer, the label is not an input, the neighbour
distance is. Every stored n09 and n13 result for `mentornet` is on the old footing.

**What to fix (before 2026-10-30).** The weight was a hand-written function of the current loss, so the learned, transferred curriculum that the published method is about was gone.

**What to fix.** Supply a target weight per molecule and fit the mentor against it before student training, or rename the arm to self-paced weighting. Reverting the objective is not advised: the commit that changed it says the old one collapsed. Owner's decision, H in `NOISE_CODE_FIX_DECISIONS_2026-10-26.md`. Done 2026-10-26: the duplicated input (next list, first item) is fixed; the history is shifted after the inputs are stacked, so the seven inputs are this epoch's loss, the five before it, and the neighbour distance.

**Git records this change as deliberate.** Commit 7a98aa1 on 2026-08-19 changed the mentor objective because the "trivial optimum was all-zero weights -> collapse". The change fixed a real collapse; what it cost is the learned curriculum the name refers to.

**Shortcuts taken.**
- Fixed 2026-10-26: the current loss entered the mentor twice, line 1748 wrote it into the last history column and line 1751 stacked it again, so two of the seven features held the same values.
- The history columns hold raw losses and nothing is subtracted, lines 1746-1748, where the published mentor reads the loss minus a running average of the batch's seventieth-percentile loss.
- The mentor is fit one step per epoch inside student training, lines 1792-1798, where the published mentor is fit in a separate run of thirty thousand steps and then frozen.
- There is no burn-in: the untrained mentor's weights are applied from the first epoch, lines 1756-1760.
- The mentor is fit to a fixed formula of the current loss, lines 1786-1789, not to a supplied target weight per molecule.

**Stronger than the published code.**
- The mentor reads the mean distance to the ten nearest molecules in feature space, lines 1684-1704, a feature the published mentor does not have.
- That distance uses the process-wide distance setting rather than whichever default the library would pick, lines 1694-1700.
- The distance is z-scored with the spread floored by a tiny constant, line 1704, so a dataset where every molecule sits at one distance does not divide by zero.
- The number of neighbours is capped at one less than the number of training molecules, line 1688, so a small fold cannot ask for more neighbours than exist.
- The mean mentor weight is recorded, line 1831; the published code records no such number.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The seven-feature mentor with one thirty-two-unit layer, the duplicated current-loss feature, and the five-epoch loss history | `9a0d827` | 2026-04-04 | no reason in the message; written once with the file and never revised |
| The mentor's objective changed from minimising weight times loss to reproducing a self-paced target with an annealed threshold | `7a98aa1` | 2026-08-19 | "trivial optimum was all-zero weights -> collapse" |
| The neighbour distance changed from the library's Euclidean default to the process-wide distance setting | `aa98979` | 2026-09-25 | "Every neighbour lookup in the noise experiments takes a similarity setting" |
| The checkpoint moved from the held-out fold's clean labels to a carved fifth of the training molecules | `edd07b0` | 2026-09-04 | "The held-out fold is only ever predicted" |
| Regression predictions capped to the training-label range | `60d635a` | 2026-10-08 | "an unbounded network output on an extreme held-out feature put one fold's RMSE in the thousands" |

## Contrast-to-divide

**Renamed 2026-10-30 (author's decision; `HANDOFF_NOISE_RESTRUCTURE.md` section 1).** Class
`JoSRCWeighting`, registry key `josrc_weighting`. No code changed. The selection rule below,
high loss together with a prediction that holds still across perturbed copies of the input, is
the test Jo-SRC (Yao et al. 2021, CVPR) uses to call a sample in-distribution noisy; the arm
applies it as a down-weight. The thesis row cites yao2021josrc and lists what Jo-SRC does that
this arm does not (relabelling by the mean-teacher copy, the two-view consistency loss, the
threshold schedule). Everything below is the record against Zheltonozhskii 2021 as it stood.

Her code: `src/kirby/utils/lnl_methods.py:1836-2006 (class ContrastToDivide), plus the base defaults at lines 427-441 and the 20 percent carve at lines 469-501`

Published source: https://github.com/ContrastToDivide/C2D

The published method pretrains the encoder without labels, then fits a two-component mixture to the scaled per-sample loss, splits at a clean probability of one half, and trains two networks on each other's splits with guessed targets for the dropped molecules. Hers trains one network from random starting weights, with no pretraining and no mixture: it ranks the loss, ranks the variance of the prediction across five copies of the features with small Gaussian noise added, multiplies the two ranks and down-weights by the product. Her docstring at lines 1840-1842 states the substitute rule, but no reason is recorded for dropping the pretraining or the second network.

**Judgement: Needs fixing.** The self-supervised pretraining the method is named for is absent and the selection score shares no computation with the published one.

**What to fix.** Load pretrained encoder weights and run the second network, or rename the arm, because the contrast the method is named for is the part that is missing. The substitute selection rule shares no computation with the published one.

**Git records no reason for this.** Written once in 9a0d827 with no reason in the message. The substitute rule is stated in the docstring at lines 1840-1842 but never justified.

**Shortcuts taken.**
- The network starts from random weights and the class has no path to load pretrained ones, lines 1884-1891, so the self-supervised half of the published method is absent.
- One network is trained, so no molecule is ever selected by a second network, where the published code has each of two networks use the other's split.
- The selection scores are recomputed only on every fifth epoch and held fixed in between, line 1917.
- There is no warm-up, so the scores at epoch 0 come from an untrained network, line 1917.
- The multiplier of three inside the down-weight at line 1950 is fixed by hand; nothing fits it from the data.
- The size of the noise added to the features, one hundredth, is fixed by hand at line 1847.

**Stronger than the published code.**
- The weights are divided by their own mean at line 1951, so down-weighting does not quietly shrink the whole gradient.
- The noise added to the features is drawn with a seed that advances with the epoch, line 1927, so the scores repeat exactly on a re-run.
- The mean and spread of the weights are recorded, lines 2003-2004; the published code records neither.
- The checkpoint is chosen on a fifth of the training molecules carved with scaffold groups kept together, lines 1859-1861, where the published code keeps no checkpoint.
- Regression predictions are capped to the range of the training labels, line 1996.

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole selection rule: the rank product of loss and prediction variance, the multiplier of three, the five perturbed copies and the five-epoch recompute | `9a0d827` | 2026-04-04 | no reason in the message; written once with the file and never revised |
| The checkpoint moved from the held-out fold's clean labels to a carved fifth of the training molecules | `edd07b0` | 2026-09-04 | "a method that read those rows well looked good at the thing the survey measures" |
| Regression predictions capped to the training-label range | `60d635a` | 2026-10-08 | "an unbounded network output on an extreme held-out feature put one fold's RMSE in the thousands" |

## Heteroscedastic GP (LNL method)

Her code: `src/kirby/utils/lnl_methods.py:2013-2147, class HeteroscedasticGP, plus the shared defaults at :427-441 and the downstream fit at :514-542 (the first agent gave :2012-2145)`

Published source: https://github.com/LiuHaiTao01/SVSHGP

The published method fits a second Gaussian process over the log noise jointly with the mean process, under one likelihood. Hers fits one scikit-learn Gaussian process, trains a network on its residuals for 50 passes, then fits a DNN with weights of 1 over 1 plus the predicted noise. No reason is recorded for the substitution; the method arrived in a commit whose subject is a precompute job tracker.

**Judgement: Fixed 2026-10-30 (author's decision E).** Before: the jointly fitted log-noise Gaussian process was absent, the predicted noise never re-entered the mean fit, and the predictions came from a DNN.

**What was done (2026-10-30).** `HeteroscedasticGP` runs Kersting, Plagemann, Pfaff and Burgard 2007 (most-likely heteroscedastic GP regression): a value GP and a noise GP fitted in turn, three passes, at most 1000 training molecules; per-molecule noise = squared residual plus posterior variance of the value GP, the noise GP fitted to its log, its exponentiated prediction set as that molecule's noise term in the value GP; the value GP predicts. The old code was kept under a second name for a one-fold comparison on herg_ki and logd (numbers in OPEN_THREADS A13): the published method scores like a plain GP, the old code like a plain DNN, and the author chose the published method; the old code is removed. Every stored `heteroscedastic_gp` result (n09, n13, n19, n40) is on the old footing.

**What to fix.** Feed the predicted noise back into the Gaussian process so the two halves are coupled, as in the paper. n19 already holds gp_per_molecule_noise, a process fitted with a known noise per molecule, so the piece exists in the repository.

**Git records no reason for this.** Written once in 9a0d827, under a commit subject about a precompute tracker. A later commit, 7fada00 on 2026-09-11, says in its own message "This is NOT my work and I have not reviewed it".

**Shortcuts taken.**
- The network's target is the residual on rows the Gaussian process was fitted on (lnl_methods.py:2080, :2098), so the residual is in sample for up to 1,000 molecules.
- The predicted noise is never fed back into the Gaussian process, which keeps one constant white-noise term and is never refitted (lnl_methods.py:2069-2080).
- The noise network runs a fixed 50 passes with no stopping rule (lnl_methods.py:2113), and this method alone does not carve selection rows the way the rest of the file does (lnl_methods.py:469, :2182).
- The sample weights are 1 over (1 plus the predicted noise) (lnl_methods.py:2131); a heteroscedastic likelihood weighs by 1 over the variance, and no measurement chose this form.
- The Gaussian process sees at most 1,000 training molecules (lnl_methods.py:2024, :2057-2061), on a cost argument with no measurement of the cut.

**Stronger than the published code.**
- It refuses classification rather than approximating it, and names the Dirichlet form as the classification analogue (lnl_methods.py:2032-2039).
- The residual is divided by the label's standard deviation (lnl_methods.py:2093-2098), so how much the method downweights no longer depends on the label's spread.
- The length scale starts at the square root of the number of features (lnl_methods.py:2068); a start of 1 left the fit predicting one value on wide representations.
- A Gaussian process fit that raises falls back to a plain DNN and records gp_fallback in the metadata rather than failing silently (lnl_methods.py:2083-2086).
- The fitted process's R-squared and the predicted noise's mean and spread are recorded per fold (lnl_methods.py:2138-2143).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole method: the two-step fit, the 50 passes, the cap of 1,000 molecules and the weighting of 1 over 1 plus the predicted noise. These lines exist in one commit only for the structure; it was written once and never revised. | `9a0d827` | 2026-04-04 | no reason in the message; its subject is a precompute job tracker |
| The residual is divided by the label's standard deviation before the network is trained on it. | `7fada00` | 2026-09-11 | no reason in the message; it says "This is NOT my work and I have not reviewed it" |
| The Gaussian process length scale starts at the square root of the number of features instead of 1. | `60d635a` | 2026-10-08 | "The internal gp never left length scale 1 on abagym_Ang2 antiberty" |
| Classification raises NotImplementedError instead of running a swapped loss. | `ba1a577` | 2026-05-26 | no reason in the message; the reason is in the docstring the commit adds |

## Evidential network

Her code: `src/kirby/utils/lnl_methods.py:2149-2303, class EvidentialKernel, plus the base-class defaults at :427-441 and the carve at :469-503`

Published source: https://github.com/aamini/evidential-deep-learning

The negative log-likelihood and the evidence penalty match the authors' own code term for term. The evidence outputs are exponentiated instead of passed through softplus, and the penalty weight is a constant 0.01 where their trainer raises it after every step toward a target. No reason is recorded for either; the class arrived in a commit whose subject is a precompute job tracker.

**Judgement: A deliberate change.** Both defining expressions run unchanged, and their own file carries the exponential output form commented out, so the differences are substitutions rather than a missing mechanism.

**Shortcuts taken.**
- The penalty weight is fixed at 0.01 and never updated (lnl_methods.py:2159, :2237); their trainer fits it from the running penalty after every step.
- The saved checkpoint holds the backbone and the mean head only (lnl_methods.py:2273-2276, :2284-2285), so the three evidence heads used at the end are the last pass's, not the best pass's.
- The uncertainty recorded in the metadata drops the square root the authors' own example plots (lnl_methods.py:2294).
- Nothing is subtracted from the penalty before it is weighted (lnl_methods.py:2237); their loss weighs the penalty minus 0.01.

**Stronger than the published code.**
- The checkpoint is chosen on a fifth of the training rows, carved off before training and kept group-disjoint (lnl_methods.py:2182-2183 and :469-501); their trainer saves checkpoints against a running error on the test set.
- Predictions are capped to the training-label range recorded before the carve (lnl_methods.py:2291 and :486-491).
- It refuses classification and names Sensoy's Dirichlet form as the classification analogue (lnl_methods.py:2167-2175).
- The two evidence scale outputs are floored at 1e-6 and alpha is offset by 1 (lnl_methods.py:2223-2225), so the loss cannot divide by zero.
- The gradient norm is clipped at 1.0 and the learning rate halves after 5 passes without improvement (lnl_methods.py:2261, :2211-2212).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole class: exponentiated evidence heads, the fixed penalty weight of 0.01, three hidden layers of 256, 128 and 64 units, and the gradient clipping. Written once in this commit. | `9a0d827` | 2026-04-04 | no reason in the message; its subject is a precompute job tracker |
| The checkpoint is now chosen on rows carved out of the training half instead of on the held-out fold. | `edd07b0` | 2026-09-04 | "a method that read those rows well looked good at the thing the survey measures" |
| Predictions are capped to the training-label range. | `60d635a` | 2026-10-08 | "n09 survey networks clip predictions to the training-label range" |
| Classification raises NotImplementedError instead of running a swapped loss. | `ba1a577` | 2026-05-26 | no reason in the message; the reason is in the docstring the commit adds |

## Huber loss

Her code: `tests/experiments/noise/n33_robust_training.py:214 (HUBER_C), :294-311 (build_swaps), :314-317 (robust_scale), :367-386 (_Centred), :389-425 (make_swap), :573-601 (where s is set per fold and matrix, at :593), :907-922. Also tests/experiments/noise/n30_censored_likelihood.py:246, src/kirby/find_best_hybrid.py:2560-2578, and src/kirby/models/dnn.py:120-125 and :169-176.`

Published source: https://github.com/microsoft/LightGBM

Her code calls the published Huber objectives by name in LightGBM, xgboost and PyTorch, so the losses themselves are the library ones; Huber wrote no code. What is hers is the threshold: instead of each package's fixed default in label units, she passes a multiple of the robust residual scale, which is 1.4826 times the median absolute out-of-fold residual, swept over four multiples (n33:214, :314-317, :593). The reason recorded is a comment citing an AI-written report rather than a paper: "The report's A6 asks for a grid of multiples of MAD(residual)" (n33:212-214).

**Judgement: A deliberate change.** A fixed default threshold in label units is replaced by a data-driven multiple of the residual scale, swept rather than picked, which is defensible across datasets with different label ranges.

**Shortcuts taken.**
- n30's lightgbm_huber passes no threshold, so it runs at LightGBM's default of nine tenths of a label unit, while n33's lightgbm_huber_c arms run at a multiple of the residual scale: two thresholds under names that differ only by a suffix (n30:246, find_best_hybrid.py:2576-2578, n33:398-400).
- xgboost's reg:pseudohubererror is the smooth pseudo-Huber, whose gradient is never piecewise, while LightGBM's huber is Huber's piecewise loss, and both arms carry huber in the name (n33:410-412).
- The threshold is fixed before the fit from out-of-fold residuals, where scikit-learn's Huber estimator fits the scale jointly with the coefficients so it adapts during the fit.
- The four multiples trace to an AI-written report, not to a paper, which the comment above them says outright (n33:212-214).
- The neural network's threshold is divided by the label standard deviation because the network standardises its labels, so the same multiple means a different thing there than in the two boosted-tree families (dnn.py:120-125, :173-175).

**Stronger than the published code.**
- The threshold is in the data's own residual units, so one setting means the same thing on datasets whose labels span different ranges (n33:314-317).
- Four multiples are swept rather than one value chosen, so the threshold is a reported axis and not a hidden constant (n33:214).
- The residual scale is read from out-of-fold predictions split by scaffold group, not from in-sample residuals (n33:586-593).
- The out-of-fold prediction is cached on the label hash and shared with the other noise experiments that need the same one (n33:585-588).
- The wrapper that fits a regressor to label minus the median and adds it back fixes LightGBM starting every molecule at zero under a custom objective, where on activity labels every residual would sit past the threshold and every gradient would be zero (n33:367-386).
- The scale is floored just above zero, so a dataset with zero median absolute residual cannot produce a zero threshold (n33:396).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The four threshold multiples and the robust residual scale they multiply. | `1e7b813` | 2026-09-24 | "from the second research report"; written once and never revised |
| The Huber loss added to the neural network, with the threshold divided by the label standard deviation. | `1e7b813` | 2026-09-24 | "DNNRegressor loss='huber' (default mse unchanged)" |
| The xgboost pseudo-Huber arms with the fitted intercept switched off and the median-centring wrapper added instead. | `1e7b813` | 2026-09-24 | no reason in the message for switching off the fitted intercept |
| n30's lightgbm_huber comparator, which passes no threshold and so runs at the package default. | `ad8e920` | 2026-09-23 | "every cleaning result the noise suite has produced was measured against squared error alone" |
| The centring wrapper that fits to label minus the median and adds it back. | `1e7b813` | 2026-09-24 | no reason in the message; the reason is in the class docstring |

## Each model's own robust loss

Her code: `tests/experiments/noise/n33_robust_training.py:295-311 (which swap replaces which model), 314-318 (the scale s = 1.4826 times the median absolute residual), 328-350 (the Tukey objective), 352-365 (the expectile objective), 367-387 (the median-centring wrapper), 389-426 (make_swap). The rf_l1 forest comes from src/kirby/find_best_hybrid.py:2960-2969. Reused at tests/experiments/noise/n38_noise_relative_to_model_error.py:269-276 and 876-895, with the threshold fixed at 1.5 units of s.`

Published source: https://github.com/microsoft/LightGBM

The losses themselves are LightGBM's, XGBoost's, PyTorch's and sklearn's own; what is hers is the one setting each takes, set to a multiple of s, the robust residual scale of that fold and feature matrix, where the libraries fix it at one or at nine tenths in raw label units. The comment at n33:211-213 gives the reason: the research report asked for "a grid of multiples of MAD(residual)". Tukey's biweight and an expectile loss are written out in her file, and the comment at n33:215-224 states why the second derivative is floored: it goes negative, and a boosted tree divides by it.

**Judgement: A deliberate change.** Each loss is the library's own and the substitution is the threshold's unit, from raw labels to multiples of the fold's robust residual scale, stated in the file.

**Shortcuts taken.**
- The scale s is fixed once per fold and matrix, where iteratively reweighted least squares re-estimates the scale at every iteration (n33:585-595).
- Her Tukey gradient is the published one multiplied by s while her second derivative is unchanged, so the Newton step is s times larger (n33:328-350).

**Stronger than the published code.**
- The scale s comes from an out-of-fold LightGBM prediction, so the threshold is not set by the fit's own residuals (n33:585-595).
- Tukey's second derivative is floored, which stops a boosted tree dividing by a negative or zero denominator, and the comment states this as a choice (n33:215-224, 328-350).
- Tukey and the expectile loss are fitted around the median of the training labels, because LightGBM starts a custom objective at zero where every pKi residual sits past the threshold (n33:367-387).
- The XGBoost swaps set the starting value to zero and subtract the median by hand, so the pseudo-Huber and absolute-error swaps start the same way (n33:412-418).
- Each swap replaces exactly one of the dataset's decided models and is paired against that model in the same feature matrix (n33:295-311).
- The residual kurtosis and the scale s are written into the result for every fold and matrix (n33:322-326, 593-598).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The swaps written: Tukey with its floored second derivative, the thresholds as multiples of s, the median-centring wrapper, the XGBoost start at zero | `1e7b813` | 2026-09-24 | no reason in the message; the file's comments carry the reasons |
| The expectile loss added at two levels | `bd29125` | 2026-09-25 | no reason in the message; the code names the third research report's item B3 |
| The rf_l1 forest, which splits on absolute error, added to the registry | `ad8e920` | 2026-09-23 | "every cleaning result the noise suite has produced was measured against squared error alone" |

## Two-learner selection (co-teaching)

Her code: `tests/experiments/noise/n32_iterative_and_dual_selection.py:1244-1270 (two_learner_exchange), with 1171-1194 (subset_loss) and 1197-1205 (lowest_share), 251-252 (2 exchange rounds, LightGBM and a support vector machine), 240 (the swept shares) and 1924-1940 (the call site). Also src/kirby/utils/noise_mitigation_methods.py:1030-1085 (CoTeachingFilter), which n32 says at its lines 22-24 that this directory does not use.`

Published source: https://github.com/bhanML/Co-teaching

Each learner still picks the molecules the other trains on, but hers computes the losses on all the training molecules, exchanges twice and refits both learners from scratch, where the published code re-selects inside every mini-batch across two hundred epochs. The two learners are LightGBM and a support vector machine rather than two copies of one network, and neither is the predictor: the dataset's own models are fitted on the molecules both learners keep. The comment at n32:236-240 gives the reason for the dropped share being swept rather than taken from the injection: an owner's decision of 2026-09-28 that no method reads how many molecules were moved.

**Judgement: Moved to another task.** The exchange is intact; the method was moved from two image networks trained by epochs to two model families fitted on features, and from classes to numbers.

**Shortcuts taken.**
- Two exchange rounds, where the published code re-selects at every mini-batch of two hundred epochs (n32:251, 1244-1262).
- Both learners are refit from scratch in each round, so nothing either has learned carries between rounds (n32:1244-1262).
- The dropped share is held at the same value in every round, where the published schedule raises it from zero over the first ten epochs (n32:1247-1256).

**Stronger than the published code.**
- A kept molecule's loss is computed out of fold inside the kept set with scaffold groups held together, where the published code scores a row under the network training on it (n32:1171-1194).
- The dropped share is swept at five, ten and twenty molecules in every hundred and never read from the injector (n32:236-240, 1923-1930).
- How many molecules each learner flags, how many both flag and how many either flags are recorded (n32:1265-1273).
- A matched random removal of the same number of molecules runs beside each swept share (n32:1940-1944).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The exchange written: two rounds, LightGBM and a support vector machine, the intersection of what both keep | `1e7b813` | 2026-09-24 | no reason in the message |
| The dropped share changed from the number the injection moved to three swept shares | `1c13c88` | 2026-09-28 | "no longer the number the injection moved or the condition's nominal share" |

## One row per measurement

Her code: `tests/experiments/noise/n29_document_bias.py:592-616 (arm_training_sets); the arm is 'per_record': (X_rec_tr, y_rec_tr) at line 612; described at 28-32`

This arm trains on the loader's measurement rows, with a molecule's features repeated on each of its measurements. No published implementation was found, so there is nothing it departs from. The docstring at n29:28-32 gives the reason: under squared loss it differs from the molecule-mean arm only by weighting a molecule by its measurement count.

**Judgement: Moved to another task.** It is a weighting baseline applied to molecules measured more than once, with no published code behind it, so nothing is absent or replaced.

**Shortcuts taken.**
- A molecule measured many times can dominate the fit, and no cap or per-molecule weight is applied (n29:612).
- The docstring's equality holds under squared loss; of herg_ic50's two decided models, svm is SVR, whose loss is not squared (n29:28-32, 795-797).

**Stronger than the published code.**
- It is paired against the molecule-mean arm on the same held-out molecules, folds and features, so the weighting is the only thing that differs (n29:592-616).
- Held-out molecules are graded on the mean of their raw measurements in every arm, so the two arms are comparable (n29:119-124).
- The docstring states that a test label still carries its documents' offsets and is not a clean label (n29:119-124).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The per_record arm, one row per measurement with the molecule's features repeated. Written once and never revised. | `2703afa` | 2026-09-24 | no reason in the message; it names the experiment, not the arm |
| The features both arms are fitted in moved to per-fold searches on the fold's training rows. | `5566961` | 2026-10-05 | "searches on each outer fold's training rows ... whole-dataset searches removed" |

## Log-count feature

Her code: `tests/experiments/noise/n19_measurement_count.py:702-716 (with_count_feature), 1340-1342 (where the two matrices are built) and 1356-1358 (where the count_feature condition picks them). The natural logarithm of the molecule's measurement count becomes one extra input feature, standardised on the training molecules, cast to the matrix's own dtype.`

The natural logarithm of a molecule's measurement count is standardised on the training molecules and added as one extra input feature. No published implementation of this was found, so there is nothing to differ from: it is built on the measurement counts n19 already computes. The result file's note at n19:1013-1018 records why it holds up here: every graded held-out molecule was measured at least the minimum number of times, so the count exists at prediction time.

**Judgement: A deliberate change.** No published code exists for adding a measurement count as an input feature, so nothing was replaced or left out.

**Stronger than the published code.**
- The count is standardised on the training molecules only, so no held-out molecule sets the scale (n19:702-716).
- The standard deviation is floored at one, so a count shared by every training molecule gives a feature of zeros rather than a division by zero (n19:712-713).
- The new feature takes the matrix's own dtype, so a float32 matrix stays float32 (n19:710-716).
- The result file records that every graded held-out molecule was measured at least the minimum number of times, so the count is available at prediction time (n19:1013-1018).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The count feature written and the two conditions that use it | `2703afa` | 2026-09-24 | "an outside research report on why label cleaning fails at herg_ki's accuracy made 15 suggestions" |
| The new feature cast to the matrix's own dtype | `5566961` | 2026-10-05 | no reason in the message; the code names handoff item 2.3 number eighteen |

## NGBoost with log-count feature

Her code: `tests/experiments/noise/n19_measurement_count.py:1422-1429 (the ngboost_plain and ngboost_count_feature arms), fed by with_count_feature at 702-716. The model itself comes from src/kirby/find_best_hybrid.py:2936-2942.`

Published source: https://github.com/stanfordmlgroup/ngboost

Her code calls the package's own NGBRegressor, so the boosting, the Normal likelihood and the five hundred boosting stages are the package's. Three registry settings differ from the package defaults: the learning rate is five hundredths against one hundredth, the seed is fixed, and progress printing is off. The extra input feature is hers, and ngboost sees it as one more ordinary feature; no reason is recorded for the learning rate.

**Judgement: Settings only.** Only numbers differ: the learning rate, the seed and whether progress prints.

**Shortcuts taken.**
- The learning rate and the seed come from the registry entry written for the whole pipeline and were not revisited for this arm (find_best_hybrid.py:2936-2942).

**Stronger than the published code.**
- A plain ngboost arm runs on the same folds in the same matrices, so the count feature is the only difference between the pair (n19:1422-1429).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The registry's ngboost settings: five hundred stages, learning rate five hundredths, fixed seed, printing off | `1474bd1` | 2026-04-02 | no reason in the message; the line has existed in one commit only, written once and never revised |
| The two ngboost arms added, one plain and one with the count feature | `2703afa` | 2026-09-24 | "an outside research report on why label cleaning fails at herg_ki's accuracy made 15 suggestions" |

## XGBoost accelerated-failure-time interval model

Her code: `tests/experiments/noise/n30_censored_likelihood.py:428-456 (fit_aft_predict), settings at :250-255, the error scale at :903, call site at :1026-1029. Also tests/experiments/noise/n19_measurement_count.py:451-468 (interval_bounds), :471-493 (fit_interval_aft_predict), call site at :1434-1447.`

Published source: https://github.com/dmlc/xgboost

Her code calls xgboost's own survival:aft objective, so the likelihood, the gradients and the censoring branches are the published ones (n30:451). What is hers is the label map and the error scale: each label enters as the exponential of label minus a centre, and the scale is the spread of label minus out-of-fold prediction rather than the package default of one (n30:440-443, :453). The reason is written in the docstring: the published loss takes the logarithm of the bound, so the exponential map makes the normal apply to the label itself rather than to a shifted copy, and the inverse was measured on toy data in session (n30:104-112).

**Judgement: Moved to another task.** The published survival-time objective is moved onto activity labels by an exact exponential-and-logarithm map, and nothing in the loss itself is changed.

**Shortcuts taken.**
- In n30 the scale is the spread over the molecules not at the limit, while in n19 it is the spread over every training molecule, so the same objective takes its scale from two different sets of molecules (n30:903, n19:1441).
- In n30 the centre subtracted from the labels is the median of the training labels, while in n19 it is the median of the interval midpoints, so two centres sit under one map (n30:440, n19:482).
- n19's xgboost_aft_point arm sets the lower and upper bound equal for every molecule, so every molecule takes the uncensored branch and nothing in that arm is censored (n19:1442-1443).
- The boosting rounds, the share of features per tree and the seed are fixed by hand to match another function in the repository rather than fitted (n30:250-255).

**Stronger than the published code.**
- The error scale is passed from the data instead of left at the package default of one, so the likelihood uses the spread this dataset actually has (n30:453, :903).
- The scale is floored just above zero, so a dataset with no spread cannot pass a zero scale into the likelihood (n30:453).
- The feature matrix is handed to xgboost in its own dtype rather than widened, which the comment ties to a recorded handoff item (n30:445-447).
- The claim that the package returns the exponential of the margin, so the logarithm inverts it, was measured on toy data in session rather than taken from documentation (n30:109-111).
- A molecule at the limit gets the interval from the limit upwards and every other molecule a zero-width interval at its own label, which is exactly what the published censored and uncensored branches expect (n30:441-443).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole AFT arm: the exponential label map, the normal error distribution, and the scale passed from the residual spread. | `2703afa` | 2026-09-24 | "the censored likelihood the research report of 2026-09-24 proposed as its B4" is in the code, not the message; no reason in the message |
| The boosting rounds set to one hundred and the share of features per tree set to half, against package defaults of ten rounds and all features. | `2703afa` | 2026-09-24 | no reason in the message; the code comment says they are matched to find_best_hybrid's own regressor, and the lines were written once |
| n19's interval version of the same fit, with its own centre and its own scale. | `1e7b813` | 2026-09-24 | no reason in the message; it lists the arms added from a second research report |
| n19's point arm skipped per fold when every training molecule's interval has zero width. | `f7f0b3f` | 2026-09-25 | "skipped per fold only when every training molecule's interval has zero width" |

## Yeo-Johnson target transform (not on the Chapter 7 list)

Her code: `tests/experiments/noise/n33_robust_training.py:508-512, the yeo_johnson branch of label_arm_predict. The transformer is fitted on the fold's noisy training labels, the model is fitted on the transformed labels, and the held-out predictions go back through inverse_transform.`

Published source: https://github.com/scikit-learn/scikit-learn

She calls sklearn's PowerTransformer and passes its own defaults, so the transform and the power it picks by maximum likelihood are sklearn's. The one thing added is sending the model's held-out predictions back through the inverse transform. The header at n33:52-57 gives the reason for choosing this transform: the research report asked for a log or a Box-Cox transform, both undefined on negative labels, and logd's labels are negative.

**Judgement: A deliberate change.** Nothing of the library's transform is changed; the substitution is which transform, and the file states why the two the report asked for cannot be used.

**Shortcuts taken.**
- The held-out predictions go through the inverse transform, which is written for the transformed label itself and not for a prediction of it (n33:508-512).

**Stronger than the published code.**
- The transformer is fitted on the fold's noisy training labels only, so no held-out label reaches it (n33:508-512).
- The header records why Yeo-Johnson and not a log or a Box-Cox transform: both are undefined on the negative labels of logd (n33:52-57).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The arm written: fit the transform on the training labels, fit the model on them, invert the predictions | `1e7b813` | 2026-09-24 | no reason in the message; the file's header gives the reason |

## Averaging fits on jittered labels (target_noise_bagging; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n33_robust_training.py:519-525 (the target_noise_bagging branch), with the two settings at 236-237 and the residual spread it reads at 593-594. Five fits, each on the training labels plus fresh Gaussian noise whose spread is half the standard deviation of the out-of-fold residual, with the five held-out predictions averaged. No row is resampled.`

Published source: https://doi.org/10.1023/A:1007682208299

Five fits, each on the training labels plus fresh Gaussian noise whose spread is half the standard deviation of the out-of-fold residual, with the five held-out predictions averaged and no row resampled. No code by Breiman for output smearing was found, so her arm could not be compared line by line against a program; the link below is the paper. The header at n33:69-71 states what the arm computes, and no reason is recorded for the fit count or for the half.

**Judgement: A deliberate change.** No published code exists, so nothing was replaced; the arm follows the paper's description with its own noise size and fit count.

**Shortcuts taken.**
- The fit count and the share of the residual spread are fixed by hand, with no reason recorded for either (n33:236-237).

**Stronger than the published code.**
- The noise spread is read from that fold's out-of-fold residual rather than fixed in raw label units (n33:519-525, 593-594).
- No row is resampled, so the arm differs from the fixed-default arm in the label noise alone (n33:519-525).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The arm written: five fits, noise at half the out-of-fold residual spread, predictions averaged | `1e7b813` | 2026-09-24 | no reason in the message |

## Label-distance mixup (c_mixup; not on the Chapter 7 list)

Her code: `tests/experiments/noise/n33_robust_training.py:464-495 (c_mixup) and 516-518 (the arm), with the two settings at 231-232.`

Published source: https://github.com/huaxiuyao/C-Mixup

Hers builds one mixed training set once and fits once, keeping every original molecule and adding one mixed molecule for each, where the published code trains on the mixed pairs alone with fresh partners drawn at every mini-batch over twenty to two hundred epochs. A molecule cannot draw itself as its partner, and the mixing fraction is drawn per molecule rather than per mini-batch. The comment at n33:229-231 gives the reason for the kernel width: Yao and co-authors tune it per dataset, and hers is a fixed default, stated as one.

**Judgement: Moved to another task.** The partner draw and the Beta mixing are intact; the method moved from a network trained by epochs to whichever model the dataset decided, fitted once.

**Shortcuts taken.**
- One mixed set is built once and fitted once, where the published code redraws partners and the mixing fraction at every mini-batch of every epoch (n33:464-495, 516-518).
- The kernel width is fixed at one fifth of a label standard deviation where the published code tunes it per dataset (n33:229-231).

**Stronger than the published code.**
- A molecule cannot draw itself as its partner, where under the published kernel the self-draw is the likeliest of all and mixes a molecule with itself (n33:479).
- A row whose kernel weights all underflow draws uniformly among the other molecules instead of dividing by zero (n33:483-487).
- The kernel is computed in chunks of a thousand rows, so the full square of molecule pairs is never held at once (n33:473-490).
- Every original molecule is kept alongside its mixed one, so no real molecule is lost to the mixing (n33:493).
- The kernel width is written down as a fixed default rather than presented as tuned (n33:229-231).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The arm written: the partner draw, the per-molecule mixing fraction, the doubled training set, the kernel width and the Beta shape | `1e7b813` | 2026-09-24 | no reason in the message |
| The mixed features cast to float32 so the matrix's dtype is kept | `5566961` | 2026-10-05 | no reason in the message; the code names handoff item 2.3 number eighteen |

## NGBoost Student-t (not on the Chapter 7 list)

Her code: `tests/experiments/noise/n33_robust_training.py:428-440 (make_ngboost), the two constants at :291-292, and the call site at :973-979 (the first agent gave :428-441 and :974-982)`

Published source: https://github.com/stanfordmlgroup/ngboost

Her function builds the package's NGBRegressor with the package's own Student-t likelihood, at 500 boosting stages and learning rate 0.05 instead of 0.01. Her constant NGB_T_DF, set to 3, is never passed to the constructor, so the 3 degrees of freedom in the run come from the package class. The free-degrees-of-freedom class was tried first and raised "Input y contains NaN" on 100 toy molecules, which her docstring records with the date.

**Judgement: Settings only.** The package's likelihood and fit run unchanged, and only the learning rate, the progress printing and the seed differ from the package defaults.

**Shortcuts taken.**
- NGB_T_DF sets nothing: it is never passed to the constructor, yet the note text reports the degrees of freedom as though it did (n33_robust_training.py:292, :439-440, :740).
- The fit is not wrapped in the numpy global-seed pin her other NGBoost call site uses (n33_robust_training.py:436-440 against noise_estimation.py:1812-1817), so these scores do not repeat exactly.
- The spread this likelihood predicts is never read here; the call site takes the point prediction only (n33_robust_training.py:977), and the weighting function fits a separate Normal (n33_robust_training.py:443-452).

**Stronger than the published code.**
- The failure of the free-degrees-of-freedom class is recorded with its date and the error text rather than dropped silently (n33_robust_training.py:431-435).
- The Student-t arm is fitted against a Normal arm at the same settings, in the same feature matrices and on the same folds (n33_robust_training.py:973-979), so the likelihood is the only thing that differs.
- The note written into the result file states the host model, the 500 stages and the rate of 0.05, so a reader of the results sees the settings (n33_robust_training.py:739-746).
- The stage count is tied to the registry's own value through one named constant rather than typed at the call (n33_robust_training.py:291, :439).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The whole function, the constant of 500 boosting stages, the unused NGB_T_DF of 3, the learning rate of 0.05 and the Student-t import. These lines exist in one commit only; they were written once and never revised. | `1e7b813` | 2026-09-24 | no reason in the message; its subject says the file came "from the second research report" |

## Nearest-neighbour mean as the predictor (not on the Chapter 7 list)

Her code: `tests/experiments/noise/n40_model_side_regularisation.py:469-472 (knn_scheme_predict) and 1187-1218 (the kNN arms, one per neighbour setting, neighbour count and counting scheme), calling tests/experiments/noise/similarity_space.py:448-458 and src/kirby/similarity/neighbour_weighting.py:158-186 (the zero-distance guard), 189-197 (one over distance to a power) and 259-272 (weight_array). The two counting schemes swept are 'rank' and 'inverse_square' (similarity_space.py:170) and the neighbour counts are 5, 10 and 20.`

Published source: https://github.com/scikit-learn/scikit-learn

The prediction is the neighbours' labels counted by one of two swept schemes, one over the neighbour's rank or one over its squared distance, where sklearn counts every neighbour the same and returns a plain mean. The neighbours come from the swept representation spaces, each with its own distance, not from the matrix the model was fitted on. The comment at n40:463-466 records that until 2026-10-05 these arms were sklearn's regressor with equal counting and Euclidean distance, and the comment at n40:313-318 names the author's decision of 2026-10-06 that set the neighbour counts.

**Judgement: A deliberate change.** The prediction is still read off the nearest neighbours' labels; the counting scheme and the neighbour space are swept axes the author decided.

**Shortcuts taken.**
- Only the two weighted schemes are swept, so no arm counts every neighbour equally and the two schemes have nothing plain to be read against (n40:738, 1207-1218).

**Stronger than the published code.**
- Each arm's name carries its neighbour count, its counting scheme and its neighbour setting, so no two arms are reported under one name (n40:1207-1218).
- A row whose weights are not finite falls back to counting that row's neighbours equally, where sklearn would return a NaN for the molecule (neighbour_weighting.py:158-186).
- A neighbour at zero distance takes the whole weight in its row, the same as sklearn (neighbour_weighting.py:158-186).
- One neighbour list per setting is built at the largest count and cut, so both schemes and all three counts read the same list (n40:1195-1210).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| The arms changed from sklearn's regressor with equal counting to the two swept counting schemes on the swept neighbour spaces | `5566961` | 2026-10-05 | no reason in the message; the code comment records what the arms were until that date |
| The neighbour counts set to five, ten and twenty, and the arms that found neighbours in the model's own features removed | `f2d2c98` | 2026-10-06 | "Adelaide delegated the open decisions of chats 2-5" |

## Ranking within each paper (not on the Chapter 7 list)

Her code: `tests/experiments/noise/n29_document_bias.py:554-583 (rank_within_documents); RANKER_PARAMS at 241-244`

Published source: https://github.com/dmlc/xgboost

She calls xgboost's own XGBRanker, so the ranking itself is the package's. What differs is settings: objective rank:pairwise against the package default rank:ndcg, half the features per tree, a fixed seed and silent output. The comment at n29:239-240 gives the reason: the settings of the downstream xgboost regressor, with the ranking objective swapped in.

**Judgement: Settings only.** The ranking code is the installed package's; only constructor arguments differ, plus her own straight line mapping scores to label units.

**Shortcuts taken.**
- The arm never runs on herg_ic50, the only dataset n29 uses, because its two decided models are svm and dnn (n29:795-800).
- The score-to-label map is a straight line; nothing checks that the ranker's score is linear in the label beyond one correlation (n29:569-583).

**Stronger than the published code.**
- The ranker's scores carry no scale, so she maps them to label units with a straight line fitted on out-of-fold scores (n29:569-581).
- Those out-of-fold scores come from scaffold-group folds, so a molecule never scores itself (n29:573-577).
- The slope, the intercept and the correlation of the out-of-fold score with the label are all written (n29:581-583).
- Rows are sorted by document before fitting, which xgboost requires of query groups (n29:565-567).
- The arm is refused rather than approximated where xgboost is not a decided model, with the reason written into the result (n29:795-800, 852-855).

**Git history.**

| What changed | Commit | Date | Reason in the message |
|---|---|---|---|
| RANKER_PARAMS and rank_within_documents, including objective rank:pairwise and half the features per tree. Written once and never revised. | `1e7b813` | 2026-09-24 | no reason in the message; it says only "new arms in n19/n27/n29, from the second research report" |
| The written reason for the arm not running, under within_document_rank_why_not. | `5566961` | 2026-10-05 | no reason in the message for this item |
| The arm was settled as staying out on herg_ic50, with its reason written into the result. | `f2d2c98` | 2026-10-06 | "Adelaide delegated the open decisions of chats 2-5" |
