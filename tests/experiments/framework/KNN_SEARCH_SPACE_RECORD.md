# The kNN search space for the noise and similarity experiments: working record

Started 2026-10-05. This is the live record of deciding which neighbour settings the noise
experiments (n01-n42) and the similarity experiments (s01-s13) fix and which they sweep.

**Read first:** `CLAUDE.md` at the repo root, then `F38_DECISIONS.md` (signed off, not to be
re-argued), then `F38_ANALYSIS_HANDOFF.md`.

**How to read this file.** Every number was recounted from a result file, not copied from a
write-up; the source is named beside it. A DECISION line is blank until Adelaide fills it. Claude
never fills one. Anything headed "Claude's reading" is a proposal, not a result.

---

## 1. What is being decided

Seven things have to have a value before any neighbour can be found.

| # | the choice | values in play |
|---|---|---|
| 1 | which features the distance is measured on | a single representation, a hybrid, raw fingerprint bits, the model's own input, a random forest's leaves |
| 2 | whether features are weighted by importance first | no weighting, lightgbm_gain, treeshap, deeplift |
| 3 | what power the importance is raised to | 0.5, 1, or the power the distance implies |
| 4 | which distance formula | manhattan, euclidean, cosine, minkowski below p=1, tanimoto, dice, soergel, correlation |
| 5 | how many neighbours | 5, 9, 10, 20, and above |
| 6 | how much each neighbour counts | equally, one over distance, one over distance squared, gaussian, one over position |
| 7 | whether a molecule's own scaffold group is excluded | excluded, or only the molecule itself excluded |

## 2. What is at stake

**Cost.** Some methods refit a model once per neighbour setting. label_smooth and knn_smooth in
n07 do, and so do the sample-weighting methods in n02, n11, n12, n13, n21, n23, n24 and n25. For
those, each value kept on each axis multiplies the fits. Today one method option costs 59
settings x 4 neighbour counts = 236 fits per model, fold, draw and noise level. n25 is the
largest: 232 settings x 186 cleanup options per model, fold, draw, level and feature matrix.

Other methods only score molecules and never refit: n01, n14, n22, n32, n37, n40, n41. Keeping
more values there costs scoring time, not fits.

**Correctness.** A value fixed wrongly makes the experiment's answer wrong, and nothing in the
output says so. The censoring result in section 6 is that failure in its clearest form.

**The experiments do not agree with each other today.** Read in the code 2026-10-05:

- 23 noise experiments read one shared sweep: n01, n02, n07, n09, n11, n12, n13, n14, n17, n18,
  n20, n21, n22, n23, n24, n25, n32, n35, n36, n37, n40, n41, n42.
- Noise experiments that find no neighbours: n06 (its own, k=10 manhattan), n15, n16, n19, n26,
  n27, n29, n30, n31, n33, n34, n38, n39, and c10.
- The similarity experiments each choose for themselves. s04, s05, s08's kNN arm and s10's
  prediction task pass no distance, so sklearn uses euclidean. s01, s02, s06, s08's distance
  matrices, s10's neighbour-feature task, s12's non-SMILES branch and s13's retrieval follow
  `KIRBY_DISTANCE_METRIC`, which defaults to manhattan.
- The importance power splits three ways. 1 under manhattan inside the noise suite and in s02 and
  s08 (`preprocessing.weight_exponent`). 0.5 whatever the distance through
  `_shared.importance_feature_weights`, used by s05, s06, s10, s13. Hardcoded 0.5 in s04, which
  ignores the environment variable.
- The weight rescaling splits three ways. Mean 1 in s02, s08 and s10 by default. Sum 1 in s13.
  None in s04, s05, s06.
- How much a neighbour counts splits six ways. Equally in label_smooth and knn_smooth (n07:995,
  n07:1015) and in three of the four neighbour scores in `noise_estimation.py`. One over distance
  in distance_weighted (`noise_estimation.py:487-488`). Read from an environment variable,
  defaulting to equal, in s05 and s10. Left to sklearn, so equal, in s04. Equal in one n40 arm
  while another lets sklearn pick.

So a number from s05 and a number from s02 on the same dataset are not measuring the same
neighbourhood.

## 3. What each framework experiment covers

| experiment | what it measures | axes it covers | datasets | state 2026-10-05 |
|---|---|---|---|---|
| f34_neighbour_weighting_cross | predicting a held-out molecule's activity from its neighbours | 4, 5, 6 | herg_ki, caco2, adme_mdr1; 7 representations | done |
| f34_full_sweep | the same, on each dataset's full representation pool plus 2 hybrids | 1, 4, 5, 6 | all four including herg; 110-120 task files each, complete | done |
| f34_importance_sweep | the same, with importance weighting added and neighbour counts to 200 | 2, 3, 4, 5, 6 | all four; 745 task files | done |
| f35_scorer_path_check | whether the repo's three ways of computing a kNN score give the same number | none; it checks plumbing | herg_ki, caco2 scored; adme_mdr1 skipped 6 of its 7 representations | rerun 2026-10-05, see section 7 |
| f36_label_smoothing_portfolio | which label smoothings the noise suite must carry | none directly | Adelaide analyses | open |
| f37_label_damage_map | which kinds of label damage act alike, and at which levels | none; sets how many levels multiply every refit count | Adelaide analyses | open |
| f38_neighbour_screen | finding and fixing corrupted labels, with no refits | 1, 2, 3, 4, 5, 6, 7 | herg_ki, caco2, adme_mdr1 complete (80 of 80 tasks each); herg 0 of 60 finished, each part-way at 192-402 rows of about 500 | pulled to /Volumes/seagate/f38_neighbour_screen |

**The split that matters.** f34 and f35 measure predicting activity. f38 measures finding and
fixing bad labels. The two rankings differ, so f34 cannot set the axes for the noise experiments.
Evidence for the difference: on herg, f34 put one-over-position first among the
neighbour-counting schemes and gaussian nowhere, while f38 puts gaussian first on all 6 damage
kinds.

## 4. Where the evidence runs out

- **Nothing has yet compared the two neighbour lists for axis 7**, with and without a molecule's
  own scaffold group. f38 computes both; the comparison is not written.
- **Nothing measures neighbour counts above 20 for cleaning.** f34 reached 200 for prediction.
  f38 stopped at 20.
- **Nothing covers what the similarity experiments do beyond prediction:** applicability domain,
  retrieval, clustering, activity-cliff finding.
- **herg is the only classification dataset** in f34 and f38, and in f38 it is part-way.
- **The hybrid spaces in f38 are test spaces**, rebuilt from the old whole-dataset searches. The
  per-fold precompute now being redone will give different ones (`F38_ANALYSIS_HANDOFF.md`
  caveat 1, `HANDOFF_PER_FOLD_ALIGNMENT.md`). So no cut to hybrid, hybrid__general2 or
  hybrid_similarity rests on f38's recipes.

## 5. The order agreed (Adelaide, 2026-10-05)

1. Which features the distance is measured on.
2. Which distance formula.
3. Importance weighting and its power, together, because they interact.
4. How many neighbours.
5. How much each neighbour counts.
6. Whether the scaffold group is excluded.

## 6. The censoring finding, written into the thesis

Under censoring, every training label above a cut-point is replaced by that cut-point. It is the
only damage kind whose corrupted molecules are chosen by the label's own value
(`noise_helpers.py:1062`).

Scoring a molecule by the gap between its label and its 10 nearest neighbours' mean label, with
every neighbour counted equally, ranks the clipped molecules as the LEAST suspicious:

| dataset | detection AUROC | rows measured | rows above 0.5 |
|---|---|---|---|
| herg_ki | 0.2768 | 2550 | 0 |
| caco2 | 0.3151 | 2580 | 8 |
| adme_mdr1 | 0.3936 | 2520 | 200 |

One row is one combination of feature space, fold and model. Taking the median of the neighbours'
labels instead of the mean moves it further from 0.5: 0.1622, 0.1418, 0.1855. More neighbours
moves it toward 0.5 without reaching it; on herg_ki, 0.2463 at 5 neighbours to 0.3246 at 20. The
forest-proximity space is the most inverted, 0.1439 on herg_ki and 0.0700 on adme_mdr1.

Replacing a clipped label with its neighbours' mean was worse than leaving it alone in every row:
2550 of 2550 on herg_ki, 2580 of 2580 on caco2, 2520 of 2520 on adme_mdr1. herg_ki: 1.1312 rmse
to the clean labels before, 1.2022 after. caco2: 0.6171 before, 0.6474 after. adme_mdr1: 0.9725
before, 1.0399 after.

Written up in `thesis.tex`, section 4.4.2, paragraph "Censoring inverts the detection score",
with four `TODO(delve)` items in the comment above it. Bullet added to `THESIS_OUTLINE.md` 4.4.2.

## 7. f35 rerun, 2026-10-05

Run locally on all three datasets with `KIRBY_CACHE_DIR=/Volumes/seagate/kirby_simrun_cache`.
Result recorded here when it finishes.

Known limit of f35, read in the code 2026-10-05: it scores f34's single best cell across every
representation (`f35_scorer_path_check.py:95-112`). On adme_mdr1 that cell used minkowski at
p=0.5, which `f18.coefficients_for` defines only for continuous features, so the six fingerprint
representations had no stored score to compare against and were skipped. The skip is structural,
not a crash. Comparing the two coefficient-honouring paths on a fingerprint representation needs
f35 to take a cell per representation kind, which it does not do today.

---

# Axis 1: which representation the distance is measured in

## The ten candidates

`tests/experiments/noise/similarity_space.py:120-131`. Numbers 1 to 7 each take an importance
weighting and a distance, so each is 8 settings. Numbers 8, 9 and 10 are fixed, so each is 1.

| # | code name | what it is |
|---|---|---|
| 1 | base_1 | the model's best single representation, from its stored search |
| 2 | base_2 | the model's second-best single representation |
| 3 | base_1_aug_uncapped | number 1 with c21's augmentation features added, uncapped |
| 4 | base_2_aug_uncapped | number 2 with the same |
| 5 | hybrid | best predictive hybrid |
| 6 | hybrid__general2 | second-best predictive hybrid |
| 7 | hybrid_similarity | best hybrid built under the similarity objective |
| 8 | tanimoto_ecfp4 | 2048 Morgan bits at radius 2, scipy jaccard. Needs SMILES |
| 9 | forest_proximity | 1 minus the share of trees putting two molecules in the same leaf. Forest fitted on the noisy training labels |
| 10 | model_input | the matrix the model is trained on, process-wide distance |

Capped augmented spaces were cut 2026-10-03 (Adelaide, `similarity_space.py:115-119`).

Provenance: 9 and 10 both arrived in commit `a07a612`, when an earlier chat built the similarity
axis. Neither is in `DECISIONS.md` and no sign-off was found. 10 is the control for what the
noise code did before that commit.

The representation pools were already pruned to 10 per dataset (`DATASET_REPS`). The noise
experiments pick 2 of those 10 per model; the similarity experiments use all 10.

## Evidence

f38 detection AUROC, one fixed setting: no importance weighting, 10 neighbours, every neighbour
counted equally. A cell is one (damage kind, fold, model). Censoring left out because its score
is inverted (section 6). Each entry is cells won of cells compared.

| comparison | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| 2 beat 1 | 14 of 70 | 35 of 70 | 29 of 70 | 30 of 60 |
| 3 beat 1 | 52 of 70 | 45 of 70 | 51 of 70 | 41 of 60 |
| 4 beat 2 | 57 of 70 | 46 of 70 | 48 of 70 | 38 of 53 |
| 6 beat 5 | 39 of 70 | 37 of 70 | 26 of 70 | 3 of 4 |
| 5 beat 1 | 43 of 70 | 49 of 70 | 58 of 70 | 27 of 33 |
| 5 beat 2 | 61 of 70 | 52 of 70 | 61 of 70 | 25 of 33 |
| 7 beat 5 | 38 of 70 | 36 of 70 | 49 of 70 | no rows |
| 8 beat 1 | 50 of 70 | 36 of 70 | 11 of 70 | 7 of 60 |
| 9 beat 1 | 18 of 70 | 40 of 70 | 25 of 70 | no rows |
| 9 beat 5 | 18 of 70 | 20 of 70 | 11 of 70 | no rows |
| 10 beat 1 | 0 of 70 | 29 of 70 | 0 of 70 | no rows |

Largest single win: 8 over 1 is +0.0022 AUROC on herg; 9 over 5 is +0.1120 on adme_mdr1; 10 over
1 is +0.0000 on herg_ki and adme_mdr1 and +0.0022 on caco2.

f34_full_sweep, predicting a held-out molecule's activity, best setting per dataset in the
dataset's own metric. A hybrid is first on all four:

| dataset | best hybrid | best single representation | fold spread of the winner |
|---|---|---|---|
| herg_ki | 0.5604 rmse | 0.5948 rmse (atom_pair) | 0.0472 |
| caco2 | 0.3137 rmse | 0.3213 rmse (pdv) | 0.0224 |
| adme_mdr1 | 0.4856 rmse | 0.5165 rmse (pdv) | 0.0127 |
| herg | 0.8490 auroc | 0.8282 auroc (graph_kernel) | 0.0120 |

Neighbour overlap, from f38's overlap files: share of a molecule's 10 nearest neighbours two
spaces agree on, for pairs differing in the space alone. 1.000 means identical lists.

| pair | herg_ki | caco2 | adme_mdr1 |
|---|---|---|---|
| raw bits under tanimoto vs 8 | 1.000 | 1.000 | not scored |
| 5 vs 6 | 0.972 | 0.899 | 0.805 |
| 1 vs 2 | 0.623 | 0.348 | 0.248 |

herg has no overlap rows; its tasks have not finished.

## DECISION, axis 1 (Adelaide, 2026-10-05)

The ten candidates are numbered as in the list above:
1 best single representation, 2 second-best single representation, 3 number 1 with augmentation
features, 4 number 2 with augmentation features, 5 best predictive hybrid, 6 second-best
predictive hybrid, 7 best hybrid under the similarity objective, 8 raw ECFP4 bits with Tanimoto,
9 forest leaves, 10 the model's own matrix.

**Two dataset groups, split on molecule count.**

- `full_knn`: 5000 molecules or fewer.
- `fast_knn`: more than 5000 molecules. Eleven datasets: logd (5028), herg (7772), qm9 (9978),
  kiba (30000), bindingdb_kd (30000), nt_H3K27me3, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3,
  nt_promoter_tata, gb_human_enhancers_cohn (18000 to 33000). Counts from
  `REPRODUCIBILITY_LEDGER.md:180` and `TIME_ESTIMATION_METHOD.md:166`, both dated claims.
  Chosen because pair work grows with the square of the molecule count, and it jumps from 13
  times herg_ki's at logd to 30 times at herg.

**Kept per group.**

| group | kept | cut |
|---|---|---|
| full_knn | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 10 |
| fast_knn | 1, 3, 5, 6, 7, 9 | 2, 4, 8, 10 |

**Rules that override the group.**

- A dataset with no SMILES never uses 8, in either group. Reason: undefined. `similarity_space.py:602-604`
  returns the space unavailable when no SMILES are passed. This covers the five nt_ sets and
  gb_human_enhancers_cohn.
- 7 does not exist on tcga_brca_pam50 or tz_her2_binary (`similarity_space.py:217`), so it is
  not available there whatever the group says.

**Reasons recorded for each cut.**

- 10 cut everywhere. Degenerate: it is the same features as whichever space the model trains on.
  It beat the best base representation in 0 of 70 cells on herg_ki and adme_mdr1, biggest
  difference 0.0000; on caco2 the biggest difference is 0.0022 AUROC.
- 8 cut on fast_knn, and on any dataset without SMILES in either group. Undefined without SMILES.
  On herg, the only fast_knn dataset f38 has rows for, it beat the best base representation in 7
  of 60 cells, biggest win 0.0022 AUROC.
- 2 cut on fast_knn. Adelaide, 2026-10-05, for compute. On herg it beat the best base
  representation in exactly 30 of 60 cells, and herg's two models rank the same two
  representations in opposite order, so 1 and 2 cover one pair there.
- 4 cut on fast_knn. Adelaide, 2026-10-05, for compute. Against it: on herg the augmented second
  base beat its own base in 38 of 53 cells.
- 6 kept in both groups. It loses to 5 in the majority of cells on all three finished datasets
  (39 of 70 herg_ki, 37 of 70 caco2, 26 of 70 adme_mdr1), but f38 built it as the next-best
  stored recipe rather than by the rank-2 rule, so no cut rests on that.

**Settings per group**, counting a weighted space as 4 importance weightings x 2 distances and a
fixed space as 1:

- full_knn: 7 weighted spaces x 8, plus 8 and 9, = 58.
- fast_knn: 4 weighted spaces x 8, plus 9, = 33.
- Both before any cut on axes 2 to 6.

**Open, not decided here.** Whether the similarity experiments keep all 10 pruned representations
or drop to this list. The representation pools were already pruned to 10 per dataset
(`DATASET_REPS`), so that is a choice between 10 and this list, not an open set.

**Evidence that is missing.** herg's f38 tasks are part-way: 0 of 60 finished. It has no rows at
all for 7, 9 or 10. herg is the only fast_knn dataset in f38, so every fast_knn cut above rests
on herg alone or on compute.

---

# Axis 2: which distance formula

## What the code allows today

The noise suite refuses anything except manhattan and euclidean on a feature space.
`_CDIST = {'manhattan': 'cityblock', 'euclidean': 'euclidean'}` (`similarity_space.py:146`) and
`SimilaritySetting.__post_init__` raises on any other name (`similarity_space.py:184-185`).
Adding a name costs two lines in `_CDIST` plus a rule in `preprocessing.weight_exponent`, which
returns 1 for manhattan, 0.5 for euclidean and None for anything else
(`preprocessing.py:342-361`).

The similarity experiments split. s04, s05, s08's kNN arm and s10's prediction task pass no
distance, so sklearn uses euclidean. s01, s02, s06, s08's distance matrices, s10's
neighbour-feature task, s12's non-SMILES branch and s13's retrieval read
`KIRBY_DISTANCE_METRIC`, default manhattan (`distance_config.py:68`).

Numbers 8 and 9 on axis 1 are unaffected: Tanimoto bits and forest leaves each carry their own
fixed distance.

## Evidence

**f18_distance_metric_sweep.** Predicting a held-out molecule's activity from its 5 nearest
neighbours, scored as Spearman against the true activity. herg_ki (1412 molecules) and logd
(5028), 7 representations each, 5 folds. One block is one (dataset, representation); 14 in all.

| distance | blocks won, herg_ki | blocks won, logd |
|---|---|---|
| cosine | 2 of 7 | 2 of 7 |
| manhattan | 2 of 7 | 1 of 7 |
| dice | 0 of 7 | 2 of 7 |
| tversky | 1 of 7 | 1 of 7 |
| soergel | 1 of 7 | 1 of 7 |
| minkowski p=0.5 | 1 of 7 | 0 of 7 |
| euclidean | 0 of 7 | 0 of 7 |

Euclidean ranked LAST among the coefficients defined on that representation in 11 of the 14
blocks, and second-last in 2. It was never better than fourth.

**f34_neighbour_weighting_cross.** Same question, 3 datasets, 7 representations, 5 folds, rmse.
A group is one (representation, neighbour-weighting scheme, neighbour count).

| distance | herg_ki | caco2 | adme_mdr1 |
|---|---|---|---|
| manhattan | 48 of 210 | 21 of 210 | 7 of 210 |
| cosine | 43 of 210 | 47 of 210 | 1 of 210 |
| euclidean | 22 of 210 | 7 of 210 | 5 of 210 |
| correlation | 0 of 30 | 1 of 30 | 0 of 30 |
| chebyshev | 0 of 30 | 0 of 30 | 0 of 30 |
| mahalanobis | 0 of 30 | 0 of 30 | 0 of 30 |

**f34_full_sweep.** Winning distance on each hybrid space, in the dataset's own metric:

| dataset | hybrid 1 | hybrid 2 |
|---|---|---|
| herg_ki | minkowski p=0.5 | manhattan |
| caco2 | cosine | cosine |
| adme_mdr1 | manhattan | cosine |
| herg | minkowski p=0.3 | minkowski p=0.5 |

Euclidean wins 0 of the 8. Neither currently allowed distance wins on 5 of the 8.

**f38_neighbour_screen.** Finding corrupted labels. Cells won against manhattan on the best
predictive hybrid. A cell is one (damage kind, fold, model). Censoring excluded because its score
is inverted (section 6). No importance weighting, 10 neighbours, every neighbour counted equally.

| distance | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| euclidean | 30 of 70 | 44 of 70 | 32 of 70 | 9 of 23 |
| cosine | 34 of 70 | 38 of 70 | 28 of 70 | 14 of 19 |
| correlation | 27 of 70 | 34 of 70 | 21 of 70 | 13 of 19 |
| minkowski p=0.5 | 30 of 70 | 30 of 70 | 32 of 70 | 8 of 15 |

Largest single win over manhattan: 0.0684 AUROC. For cleaning the distance is close to
indifferent; for predicting activity it is not.

**Duplicates.** Share of a molecule's 10 nearest neighbours two distances agree on. Same
representation, same importance weighting, only the distance differs. Kept representations only.

| pair | herg_ki | caco2 | adme_mdr1 |
|---|---|---|---|
| correlation vs cosine | 0.983 | 0.970 | 0.928 |
| manhattan vs minkowski p=0.5 | 0.934 | 0.894 | 0.861 |
| every other pair | 0.81 to 0.90 | 0.74 to 0.83 | 0.55 to 0.79 |

herg has no overlap rows; its tasks have not finished.

## Other experiments that touch the distance, and why they are not used above

- f34_importance_sweep has a distance axis but a reduced coefficient list with no euclidean on
  continuous features, so its tally is not comparable.
- f23_noise_cleaning_metric has one distance pair, 0.6370 against 0.6475 detection AUROC, but one
  side is raw counts and the other z-scored (`f23_noise_cleaning_metric.py:129,137`). It differs
  in two ways at once.
- similarity_metrics/structural_rerank correlates a pairwise distance with activity difference,
  not neighbour prediction. Whole dataset, no folds.
- similarity_metrics/base_rep_comparison compares cosine against euclidean on one antibody
  representation across 4 datasets; cosine ahead on 3.
- f22 hardcodes euclidean (`f22_similarity_feature_sources.py:326,405,439,514`). c04 passes no
  metric (`c04_importance_as_metric.py:116,122`). f17 and f20 have no distance axis. f19 has no
  result files.

## Settings cost

| distances kept | full_knn | fast_knn |
|---|---|---|
| 2 | 58 | 41 |
| 3 | 86 | 61 |
| 4 | 114 | 81 |

The fractional Minkowskis and soergel are the dear ones per setting. f34 measured soergel on a
count fingerprint at 80 minutes on one core, and `pdv` under the two fractional Minkowskis took
two and a half hours of f34's 5 hour 40 minute total.
`SIMILARITY_AUTORESEARCH_STATUS.md:1713-1715` times scipy's cdist at 39 seconds under manhattan
against over 7 minutes under minkowski, on 1412 by 1412 molecules at 2048 features. NOT verified
in this session.

## DECISION, axis 2 (Adelaide, 2026-10-05)

| group | kept | cut |
|---|---|---|
| full_knn | manhattan, cosine, minkowski p=0.5 | euclidean, correlation |
| fast_knn | manhattan, cosine | euclidean, correlation, minkowski p=0.5 |

Settings: full_knn 86, fast_knn 41. fast_knn is unchanged from today's count.

Reasons recorded:

- euclidean cut in both groups. It wins 0 of f18's 14 blocks and ranks last in 11 of them, wins 0
  of f34_full_sweep's 8 hybrid spaces, and takes 5 to 22 of 210 groups in f34_cross.
- correlation cut in both groups. Duplicate of cosine at 0.928 to 0.983 shared neighbours, and it
  wins 0 of 30 groups on herg_ki and adme_mdr1 and 1 of 30 on caco2 in f34_cross.
- minkowski p=0.5 cut on fast_knn, for compute. Near-duplicate of manhattan at 0.861 to 0.934
  shared neighbours, and the dearest of the three per setting. This is a compute cut, not an
  evidence cut: it wins 1 of f18's 14 blocks and is the winner on 3 of f34_full_sweep's 8 hybrid
  spaces.

Carried with an asterisk: herg has 0 of 60 f38 tasks finished and no overlap rows, so every
fast_knn judgement rests on f18, f34 and compute rather than on herg's cleaning results. Revisit
when herg lands.

## Not reached by this decision

The only distance gaps anywhere that exceed their own fold spread are on fingerprint
representations, and they go to the set-overlap family against euclidean. From f34_full_sweep,
fingerprint representations only:

| dataset | representation | best | euclidean | gap | fold spread |
|---|---|---|---|---|---|
| adme_mdr1 | atom_pair | 0.5401 (dice) | 0.6324 | 0.0923 | 0.0200 |
| adme_mdr1 | ecfp4 | 0.5823 (tanimoto) | 0.6685 | 0.0862 | 0.0170 |
| adme_mdr1 | avalon | 0.5871 (dice) | 0.6167 | 0.0297 | 0.0164 |
| herg_ki | avalon | 0.6367 (tanimoto) | 0.6555 | 0.0188 | 0.0179 |

Four of 19 fingerprint blocks. Everywhere else the gaps sit inside the fold spread.

The noise suite cannot reach these by adding dice or tanimoto to its feature-space distances,
because it z-scores every matrix first (`n07_remediation.py:1629-1631`, and
`similarity_space.py:662-667` requires it). A z-scored bit is a signed number, so there is no set
to intersect. The only place set-overlap survives there is axis 1's fixed Tanimoto space, which is
always ECFP4 rather than whichever fingerprint `base_1` happens to be — ecfp4 on caco2, and on
herg_ki's svm. OPEN, not decided: whether a base representation should ever be passed raw.

---

# Axis 3: whether features are weighted by importance before the distance

The question: before measuring distance, is each feature multiplied by how important it is for
predicting the label? Four values today: no weighting, lightgbm_gain, treeshap, deeplift
(`similarity_space.py:133`).

## Evidence

**Did weighting help?** f34_importance_sweep. Best score with any weighting against best with
none, per representation, 5 folds, each dataset's own metric. 12 representations per dataset.

| dataset | representations where it helped | largest gain | gains beyond that representation's fold spread |
|---|---|---|---|
| herg_ki | 9 of 12 | 0.0265 rmse (mordred) | 0 of 12 |
| caco2 | 9 of 12 | 0.0092 rmse (graph_kernel) | 0 of 12 |
| adme_mdr1 | 8 of 12 | 0.0467 rmse (grover) | 1 of 12 |
| herg | 8 of 12 | 0.0235 auc (mordred) | 0 of 12 |

One representation of 48 gains more than its own fold-to-fold range: grover on adme_mdr1, 0.0467
rmse against a spread of 0.0462.

**On the hybrid spaces it does not help.** These are where the noise experiments mostly run.

| dataset | space | best | no weighting |
|---|---|---|---|
| herg_ki | hybrid_general | 0.5795 (lightgbm_gain) | 0.5798 |
| herg_ki | hybrid_general_distance | 0.5620 (none) | 0.5620 |
| caco2 | hybrid_general | 0.3137 (none) | 0.3137 |
| caco2 | hybrid_general_distance | 0.3152 (deeplift) | 0.3173 |
| adme_mdr1 | hybrid_general | 0.4862 (none) | 0.4862 |
| adme_mdr1 | hybrid_general_distance | 0.4856 (none) | 0.4856 |
| herg | hybrid_general | 0.8528 (lightgbm_gain) | 0.8501 |
| herg | hybrid_general_distance | 0.8501 (lightgbm_gain) | 0.8478 |

No weighting is first on 4 of the 8. The largest gain on any of the 8 is 0.0027 auc.

**Why the hybrids gain nothing.** The hybrid already selected its features, so there is little
left to reweight. Features given zero importance, so dropped from the distance, on herg_ki:
grover 4217 of 4800 under lightgbm_gain, ecfp4 1868 of 2048, but hybrid_general only 10 of 100.

**Is "match" a usable rule?** No. Using the importance method the hybrid itself used for feature
selection, and ranking it against the other three plus no weighting:

| dataset | space | recipe's method | its rank of 4 |
|---|---|---|---|
| herg_ki | hybrid_general | treeshap | 3rd |
| herg_ki | hybrid_general_distance | treeshap | 3rd |
| caco2 | hybrid_general | lightgbm_gain | 4th |
| caco2 | hybrid_general_distance | lightgbm_gain | 3rd |
| adme_mdr1 | hybrid_general | treeshap | 4th |
| adme_mdr1 | hybrid_general_distance | treeshap | 3rd |
| herg | hybrid_general | lightgbm_gain | 1st |
| herg | hybrid_general_distance | lightgbm_gain | 1st |

Worst or next to worst on all three regression datasets, best on both herg spaces.

**The three methods are not duplicates of each other.** Spearman between the feature rankings two
methods give, computed this session from the cached vectors in
`f34_importance_sweep/<dataset>/importance/`. One number per (representation, fold), mean over 5
folds.

| pair | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| lightgbm_gain / treeshap | +0.28 to +0.83 | +0.34 to +0.88 | +0.39 to +0.93 | +0.41 to +0.66 |
| lightgbm_gain / deeplift | +0.03 to +0.52 | +0.07 to +0.64 | +0.05 to +0.86 | -0.00 to +0.57 |
| treeshap / deeplift | -0.04 to +0.80 | +0.00 to +0.88 | -0.01 to +0.87 | -0.13 to +0.87 |

Highest single value anywhere is +0.933, on adme_mdr1 maccs at 167 features. So no method could
have been cut as a duplicate; the axis is cut because the whole axis buys nothing.

## DECISION, axis 3 (Adelaide, 2026-10-05)

**No importance weighting, in both groups. The axis is cut entirely.**

Reason: of 48 (dataset, representation) pairs, 1 gains more than its own fold-to-fold range. On
the 8 hybrid spaces, no weighting is first on 4 and the largest gain on any is 0.0027 auc.

Settings after this cut, with axis 1 and axis 2 as decided:

| group | weighted representations | distances | fixed spaces | settings |
|---|---|---|---|---|
| full_knn | 7 | 3 | 8 and 9 | 23 |
| fast_knn | 5 | 2 | 9 | 11 |

Down from 86 and 41.

**The power question disappears with this cut.** `preprocessing.weight_exponent` ties the power to
the distance (0.5 under euclidean and cosine, 1.0 under manhattan, 2.0 at minkowski p=0.5), and
s04, s05, s06, s10 and s13 disagreed with the noise suite about it. With no weighting there is no
power to set. The three-way split recorded in section 2 is closed by this decision for the noise
suite. STILL OPEN for the similarity experiments, which weight features for their own tasks.

**Not tested, so not a reason for anything:** power 2.0 was never run for predicting activity;
f34's grid held only 0.5 and 1.0. f34's correlation-with-importance numbers also fold in an
exponent that `preprocessing.py` says correlation cannot take.

---

# Axis 4: how many neighbours

Today the noise suite sweeps 5, 9, 10 and 20 (`SWEPT_KS`, `similarity_space.py:141`). The count
is not part of a setting name: one stably sorted list of 20 serves every count
(`_stable_topk`, `similarity_space.py:373-393`), and `neighbours_at` raises rather than trimming a
longer list (`:738-771`).

## Evidence

Only f34 and f38 swept this. Every other framework experiment fixes it: f18 and f19 at 5
(`f18_distance_metric_sweep.py:132`, `f19_objective_scoring.py:122`), f30 at 10
(`f30_scorers.py:684`), f29 at the square root of the sample count capped at 5
(`f29_model_and_tuning_survey.py:145`), f22 at a single passed value. So every f18 number used for
axis 2 was measured at 5 neighbours only.

`_shared.py:3202-3207` holds a kNN capacity family at 20, 10, 5 and 1. Those counts ARE what it
measures, so it is not evidence here.

**The two jobs want different counts.**

Predicting activity, f34_full_sweep, kept distances only, best over the weighting. 12
representations per dataset:

| dataset | most common best count |
|---|---|
| herg_ki | 10, on 7 of 12 |
| caco2 | 20, on 7 of 12 |
| adme_mdr1 | 40, on 7 of 12 |
| herg | 40, on 6 of 12 |

Finding corrupted labels, f38, no importance weighting, kept distances only, every neighbour
counted equally. A cell is one (kept representation, damage kind, fold, model). Censoring left
out:

| dataset | cells | best count |
|---|---|---|
| herg_ki | 490 | 5 on 203, 20 on 109, 10 on 91, 9 on 87 |
| caco2 | 490 | 5 on 223, 9 on 90, 20 on 95, 10 on 82 |
| adme_mdr1 | 490 | 20 on 143, 5 on 131, 10 on 119, 9 on 97 |
| herg | 157 | 9 on 81, 10 on 43, 5 on 31, 20 on 2 |

Prediction improves up to 40. Cleaning wants few, and 5 is the single most common winner on
herg_ki and caco2.

**What a shorter list costs.** Largest gap between the best at any count and the best within the
list.

Predicting activity, in each dataset's own metric:

| list | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| 5, 10, 20 | 0.0021 | 0.0014 | 0.0024 | 0.0013 |
| 10, 20 | 0.0021 | 0.0014 | 0.0024 | 0.0013 |
| 20 alone | 0.0064 | 0.0020 | 0.0024 | 0.0013 |
| 10 alone | 0.0021 | 0.0035 | 0.0107 | 0.0041 |

All inside the fold spreads, which run 0.012 to 0.047.

Finding corrupted labels, in detection AUROC:

| list | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| 5, 10, 20 | 0.0148 | 0.0200 | 0.0106 | 0.0052 |
| 10, 20 | 0.0539 | 0.0353 | 0.0567 | 0.0124 |
| 20 alone | 0.0976 | 0.0632 | 0.0567 | 0.0293 |
| 10 alone | 0.0679 | 0.0601 | 0.0656 | 0.0124 |

Dropping 5 is what hurts: the largest loss rises from 0.0200 to 0.0567 AUROC, and the fold range
on this score is about 0.03.

## DECISION, axis 4 (Adelaide, 2026-10-05)

**Keep 5, 10, 20. Three counts, both groups. Cut 9.**

Settings with axes 1 to 3 as decided:

| group | settings before this axis | neighbour counts | settings |
|---|---|---|---|
| full_knn | 23 | 3 | 69 |
| fast_knn | 11 | 3 | 33 |

Reason for cutting 9: the list of 5, 10, 20 loses at most 0.0200 AUROC against all four counts,
on every dataset, and 10 sits beside it.

Carried with an asterisk: 9 is the best count on 81 of herg's 157 cells, more than any other. herg
has 0 of 60 f38 tasks finished. Revisit when herg lands.

Not covered: nothing measured a count above 20 for cleaning, and prediction was still improving at
40 on adme_mdr1 and herg.

---

# Axis 5: how much each neighbour counts

The question: when the k neighbours' labels are averaged, does each count the same, or does a
closer one count more? Five ways, defined in `src/kirby/similarity/neighbour_weighting.py`:

| name in the code | in words |
|---|---|
| uniform | all k count the same |
| inverse | twice as far, half the say |
| inverse_square | twice as far, a quarter of the say |
| gaussian | the say falls off smoothly, scaled to how far this molecule's own neighbours sit |
| rank | closest counts 1, second a half, third a third; only the order matters, not the distance |

**Not swept today.** Equal counting in label_smooth and knn_smooth (`n07_remediation.py:995,
1015`) and in three of the four neighbour scores in `noise_estimation.py`. One over distance in
distance_weighted only (`noise_estimation.py:487-488`). s05 and s10 read an environment variable
defaulting to equal; s04 passes nothing, so sklearn counts equally.

## Evidence

f34: equal counting wins no group on any dataset, where a group is one (representation, distance,
neighbour count): 0 of 688 on herg_ki, 0 of 680 on caco2, 0 of 680 on adme_mdr1, 0 of 688 on herg.

f38, with axes 1 to 4 as decided. A cell is one (kept representation, damage kind, fold, model,
neighbour count). No importance weighting, kept distances only, censoring left out.

Finding corrupted labels, cells won:

| dataset | cells | uniform | inverse | inverse_square | gaussian | rank |
|---|---|---|---|---|---|---|
| herg_ki | 1470 | 252 | 235 | 414 | 133 | 436 |
| caco2 | 1470 | 233 | 159 | 365 | 209 | 504 |
| adme_mdr1 | 1470 | 126 | 128 | 617 | 108 | 491 |
| herg | 471 | 2 | 11 | 0 | 288 | 170 |

Fixing corrupted labels, same cells:

| dataset | uniform | inverse | inverse_square | gaussian | rank |
|---|---|---|---|---|---|
| herg_ki | 142 | 208 | 269 | 209 | 642 |
| caco2 | 200 | 193 | 303 | 320 | 454 |
| adme_mdr1 | 143 | 259 | 570 | 221 | 277 |
| herg | 66 | 62 | 49 | 83 | 211 |

Biggest single loss in detection AUROC if that scheme is dropped:

| dataset | uniform | inverse | inverse_square | gaussian | rank |
|---|---|---|---|---|---|
| herg_ki | 0.0253 | 0.0109 | 0.0323 | 0.0137 | 0.0528 |
| caco2 | 0.0365 | 0.0095 | 0.0424 | 0.0115 | 0.0309 |
| adme_mdr1 | 0.0133 | 0.0046 | 0.0204 | 0.0048 | 0.0261 |
| herg | 0.0012 | 0.0051 | none won | 0.0107 | 0.0146 |

rank and inverse_square together win 850 of 1470 cells on herg_ki, 869 on caco2 and 1108 on
adme_mdr1. On herg, inverse_square wins nothing and gaussian wins 288 of 471.

## DECISION, axis 5 (Adelaide, 2026-10-05)

**Keep two: rank and inverse_square. Both groups.**

| group | settings before this axis | schemes | settings |
|---|---|---|---|
| full_knn | 69 | 2 | 138 |
| fast_knn | 33 | 2 | 66 |

Reasons:

- uniform cut. Wins 0 of 688 groups on every f34 dataset, 2 of herg's 471 f38 cells, and is never
  first on any dataset. This is what the code does today, so the control goes with it.
- inverse cut. Smallest cost to lose of the five: biggest single loss 0.0109 AUROC on herg_ki,
  0.0095 on caco2, 0.0046 on adme_mdr1.
- gaussian cut. Biggest loss 0.0137 on the three regression datasets.

Carried with an asterisk, and it is larger here than on the earlier axes: gaussian is the best of
the five on herg, winning 288 of 471 cells, and cutting it costs 0.0107 AUROC there. herg is the
only classification dataset and has 0 of 60 f38 tasks finished. Keeping gaussian as a third would
be 207 and 99 settings. Revisit when herg lands.

This axis is free on the experiments that only score molecules and never refit (n01, n14, n22,
n32, n37, n40, n41): f38 computed all five schemes in one pass. It multiplies refits only for
label_smooth, knn_smooth and the sample-weighting methods.

---

# Axis 6: whether a molecule's own scaffold group is excluded

The question: when finding a molecule's nearest neighbours, are molecules from its own scaffold
group allowed, or skipped?

The argument for skipping: scaffold-mates are near-copies, so their labels look alike whatever the
label is, and the comparison then measures the scaffold rather than the label. The argument
against: skipping throws away the closest molecules, which are the ones whose labels should agree
most.

**What the code does now, and it is split rather than chosen.** `exclude_groups` sets every
same-group column to positive infinity (`similarity_space.py:433-437`), and the neighbour count
falls to min(20, training rows minus the largest group) (`:396-408`). n07's cleanup methods use the
full list; its detector grading excludes the scaffold group (`n07_remediation.py:2160-2172`). n42
uses the excluded list for detector scores and the full list for smoothing targets
(`n42:420-437, 521-561`). n18 and n24 and n25 split the same way. Everything else passes nothing,
so no exclusion. `cached_detector_scores` refuses a list that still holds a same-group neighbour
when groups are passed (`noise_cache.py:595-606`).

## Evidence

Detection AUROC, with axes 1 to 5 as decided. A cell is one (kept representation, damage kind,
fold, model, neighbour count, counting scheme), scored both ways. No importance weighting, kept
distances only, censoring left out.

| dataset | paired cells | excluding helped | median change | range |
|---|---|---|---|---|
| herg_ki | 2940 | 749 of 2940 | -0.0078 | -0.0855 to +0.0936 |
| caco2 | 2940 | 575 of 2940 | -0.0081 | -0.0479 to +0.0830 |
| adme_mdr1 | 2940 | 889 of 2940 | -0.0021 | -0.0228 to +0.0274 |
| herg | 942 | 0 of 942 | -0.0407 | -0.0739 to -0.0112 |

Excluding the scaffold group makes finding corrupted labels worse on all four datasets. On herg it
is worse in every one of 942 cells.

## DECISION, axis 6 (Adelaide, 2026-10-05)

**Keep one neighbour list, the full one. Do not exclude the scaffold group.**

| group | settings |
|---|---|
| full_knn | 138 |
| fast_knn | 66 |

Unchanged from axis 5: dropping the second list is what keeps the count from doubling to 276 and
132.

Reason: excluding is worse on all four datasets, and worse in every cell on herg. Median cost
0.0021 to 0.0407 AUROC.

**Implementation consequence to settle before coding.** n07's detector grading, n18, n24, n25 and
n42's detector scores currently read the excluded list. Keeping one list means those switch to the
full list, which moves their numbers. `cached_detector_scores` also refuses a list holding a
same-group neighbour when groups are passed (`noise_cache.py:595-606`), so that guard has to be
relaxed or the groups must stop being passed. Not decided here.

---

# The decided search space, all six axes

As of 2026-10-05. Numbers from the DECISION blocks above.

| axis | full_knn | fast_knn |
|---|---|---|
| 1. representation | best single, second-best single, both with augmentation, best hybrid, second-best hybrid, similarity hybrid, Tanimoto bits, forest leaves | best single, best single with augmentation, best hybrid, second-best hybrid, similarity hybrid, forest leaves |
| 2. distance | manhattan, cosine, minkowski p=0.5 | manhattan, cosine |
| 3. importance weighting | none | none |
| 4. neighbour count | 5, 10, 20 | 5, 10, 20 |
| 5. how much each neighbour counts | by position, one over distance squared | by position, one over distance squared |
| 6. scaffold group | not excluded | not excluded |
| **settings x neighbour counts x schemes** | **138** | **66** |

Group membership: fast_knn is any dataset over 5000 molecules, which is logd, herg, qm9, kiba,
bindingdb_kd, the five nt_ sets and gb_human_enhancers_cohn. full_knn is the rest.

Overriding rules: a dataset with no SMILES never uses the Tanimoto bits space, in either group.
The similarity hybrid does not exist on tcga_brca_pam50 or tz_her2_binary.

Against today's 59 settings x 4 neighbour counts = 236, full_knn is 138 and fast_knn is 66.

## Every asterisk, in one place

Each of these rests on herg, which has 0 of 60 f38 tasks finished and no overlap rows. Revisit all
of them together when herg lands.

- Axis 1: cutting the second base representation and the second augmented space on fast_knn was a
  compute decision, not an evidence one. herg has no f38 rows for the similarity hybrid, forest
  leaves or the model's own matrix.
- Axis 2: cutting minkowski p=0.5 on fast_knn was a compute decision. herg has no overlap rows, so
  no duplicate check ran there.
- Axis 4: 9 neighbours is the best count on 81 of herg's 157 cells, more than any other, and it
  was cut.
- Axis 5: gaussian is the best scheme on herg, winning 288 of 471 cells, and it was cut.

Also open, and not reached by any of the six decisions:

- The only distance gaps that beat their own fold spread are on fingerprint representations and go
  to the set-overlap family. The noise suite cannot use them because it z-scores every matrix
  first. Whether a base representation should ever be passed raw is undecided.
- Whether the similarity experiments keep all 10 pruned representations or move to axis 1's list.
- The importance power for the similarity experiments, which still splits three ways.
- Nothing measured a neighbour count above 20 for cleaning, and prediction was still improving at
  40 on adme_mdr1 and herg.
- The hybrid spaces in f38 are test spaces from the old whole-dataset searches, so no cut to the
  hybrid, second hybrid or similarity hybrid rests on f38's recipes.

---

# The noise grid: which kinds of label damage and which levels (f37)

Decided 2026-10-05. Recounted this session from `scratchpad/f37_results` (herg_ki, caco2, logd;
4 model-and-representation pairs each; 3 folds; 3 levels; 2 draws).

**How a kind was judged different.** Its rmse drop at the highest level against gaussian's, on each
model-and-representation pair, compared with how far apart two draws of the SAME kind land. A kind
counts as different on a pair when the gap beats that draw spread. Four pairs per dataset.

**Nine kinds are gaussian under another name**, 0 to 2 of 4 pairs apart on every dataset: laplace,
student_t_nu3, student_t_nu5, student_t_nu10, outlier_p01, outlier_p05, outlier_p10,
outlier_only_p05, grouped_wider.

**Nine are genuinely different**, 3 or 4 of 4 apart on every dataset: censoring, shift_only_p05,
shift_only_p10, sparse_dprime_p02, sparse_dprime_p05, sparse_dprime_p10, sparse_dprime_up_p02,
sparse_dprime_up_p05, sparse_dprime_up_p10.

⛔ The p02, p05, p10 suffix is the SHARE of molecules corrupted, which is a dose. The level axis
already sweeps dose. So one share per family is enough, and a family member differing from its
siblings is not a reason to keep it.

Two pairs that turned out to be the same thing: `sparse_dprime_p10` against
`sparse_dprime_up_p10` is 1, 2 and 4 of 4 across the three datasets; `outlier_p05` against
`outlier_only_p05` is 1 of 4 on all three. One pair genuinely separate: `grouped_shifted` against
`shift_only_p10` is 4 of 4 on all three.

## DECISION, noise grid (Adelaide, 2026-10-05): keep six kinds

| kind | why | levels |
|---|---|---|
| gaussian | the reference. Stands for all nine spread-shaped kinds | 0, 0.5, 1.0, 1.5 |
| outlier_p05 | same behaviour, but only a subset is corrupted, so detection is defined. Nothing else in that group of nine gives you that | 0, 0.5, 1.0, 1.5 |
| grouped_shifted | 4 of 4 pairs apart from shift_only on all three datasets | 0, 0.5, 1.0, 1.5 |
| censoring | 3 to 4 of 4 apart from gaussian everywhere, and the kind where the whole neighbour approach inverts | 0, 0.1, 0.25, 0.4 (fraction clipped) |
| shift_only_p10 | 4 of 4 apart from gaussian on all three | 0, 0.5, 1.0, 1.5 |
| sparse_dprime_up_p10 | 3 to 4 of 4 apart from gaussian on all three | 0, 0.5, 1.0, 1.5 |

**Cut sixteen.** Nine are gaussian repeated. Five are a second or third share inside a family that
keeps one member: shift_only_p05, sparse_dprime_up_p02, sparse_dprime_up_p05, sparse_dprime_p02,
sparse_dprime_p05. Two are the same as a kept kind on two datasets of three: sparse_dprime_p10 and
outlier_only_p05.

**Four levels per kind, including level 0.** The retention score (`auc_norm`,
`src/kirby/noise_robustness.py`) needs at least four points including level 0 and returns NaN below
that. Three levels plus 0 is the minimum, so 0, 0.5, 1.0, 1.5 is the floor, not a preference.

**Not measured.** f37 ran levels 0.3, 0.75 and 1.5. Levels 0.5 and 1.0 have never been run, so the
six kinds rest on the levels f37 used and the chosen levels have no f37 evidence behind them.
Nothing says whether the draw spread of `auc_norm` is smaller than the gaps between kinds; only the
rmse spread was measured.

---

# Screening the kNN settings per dataset instead of refitting all of them

Raised by Adelaide 2026-10-05. NOT yet a decision; the numbers are here so it can become one.

**The idea.** Inside each experiment, for each dataset and fold, score all 138 settings with no
model fits, keep the best few, and refit only those.

**What it cannot screen on.** f38's score is AUROC against which molecules were actually corrupted.
Inside an experiment that is the unknown, so screening on it would leak.

**What it can screen on.** How well a setting's neighbours predict a held-out molecule's activity.
That needs no knowledge of the damage, and it is f34's score.

**Does screening on prediction cost cleaning performance?** Measured this session on the settings
both experiments cover: base_1, base_2 and the hybrid, all kept distances, counting schemes and
neighbour counts. Cleaning is f38 detection AUROC, censoring excluded. The number is how much
detection AUROC is given up by refitting only the top N by prediction.

| dataset | settings | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|---|
| herg_ki | 36 | 0.0023 | 0.0022 | 0.0014 | 0.0009 |
| caco2 | 18 | 0.0047 | 0.0000 | 0.0000 | 0.0000 |
| adme_mdr1 | 18 | 0.0014 | 0.0000 | 0.0000 | 0.0000 |

Against a fold-to-fold range of about 0.03 on this score. Counting name overlap instead gives a
pessimistic picture — herg_ki shares 0 of the top 5 and 4 of the top 10 — but the settings that
differ score almost the same, which is what matters.

**Not covered.** The augmented and similarity-hybrid spaces are not in the comparison, because f34
never measured them. herg is not in it at all, so classification is unchecked.

**What it would save.** Refitting 5 settings instead of 138 is 28 times fewer fits, on top of
everything above.

---

# Compute, after every decision above

Refits per dataset for the methods that refit a model once per neighbour setting. A neighbour
configuration already includes the neighbour count and the counting scheme, so it is 236 before and
138 or 66 after. The (level, draw) count is 266 before and 60 after, with level 0 run once.

| experiment | method options | before | after, full_knn | after, fast_knn |
|---|---|---|---|---|
| n07 | 4 | 2,511,040 | 331,200 | 158,400 |
| n12 | 2 | 1,255,520 | 165,600 | 79,200 |
| n02 | 2 | 1,255,520 | 165,600 | 79,200 |
| n24 | 14 | 8,788,640 | 1,159,200 | 554,400 |
| n11 | 180 | 112,996,800 | 14,904,000 | 7,128,000 |
| n25 | 186 | 116,763,360 | 15,400,800 | 7,365,600 |

7.6 times fewer on full_knn, 15.9 on fast_knn. The noise grid gives 4.4 of that; the kNN space
gives 1.7 on full_knn and 3.6 on fast_knn.

**What still dominates, and nothing above touches it.** n11 and n25 each try about 180 cleanup
recipes per neighbour setting. For n25 that is 10 cleanup methods x 3 doses x 3 arms (plain,
cliff-exempt, random-exempt). For n11 it is 3 estimators x 60 options, the 60 being 4 strengths x 4
methods x 3 targets. Setting that multiplier to 1 leaves n25 at 82,800 refits; setting the
neighbour configurations to 1 leaves it at 111,600. So the cleanup recipes are the larger multiplier
and they have never been examined. The 10 methods do three different things (delete the molecule,
lower its weight, rewrite its label) and are genuinely distinct; the 3 doses within a method and the
3 arms have not been measured against each other. OPEN, and the next conversation.
