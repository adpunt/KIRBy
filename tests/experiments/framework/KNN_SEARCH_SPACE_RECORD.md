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

The similarity experiments split. s04, s05, s08's kNN arm and s10's prediction task pass no
distance, so sklearn uses euclidean. s01, s02, s06, s08's distance matrices, s10's
neighbour-feature task, s12's non-SMILES branch and s13's retrieval read
`KIRBY_DISTANCE_METRIC`, which defaults to manhattan (`distance_config.py:68`).

`distance_config.py:66` defines four names: manhattan, euclidean, cosine, minkowski_p0.5.
Adding a name to the noise suite needs two lines in `_CDIST` plus a rule in
`preprocessing.weight_exponent`, which returns 1 for manhattan, 0.5 for euclidean and None for
anything else (`preprocessing.py:342-361`).

Numbers 8 and 9 are not affected: Tanimoto bits and forest leaves each carry their own fixed
distance.

## Evidence

f38 detection AUROC against manhattan. No importance weighting, 10 neighbours, every neighbour
counted equally. A cell is one (damage kind, fold, model). Censoring left out. Each entry is
cells won of cells compared.

On number 1, the best single representation:

| distance | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| euclidean | 30 of 70 | 35 of 70 | 32 of 70 | 10 of 60 |
| cosine | 32 of 70 | 35 of 70 | 37 of 70 | 33 of 60 |
| correlation | 38 of 70 | 31 of 70 | 33 of 70 | 32 of 60 |
| minkowski_p0.5 | 44 of 70 | 42 of 70 | 29 of 70 | 33 of 60 |

On number 5, the best predictive hybrid:

| distance | herg_ki | caco2 | adme_mdr1 | herg |
|---|---|---|---|---|
| euclidean | 30 of 70 | 44 of 70 | 32 of 70 | 9 of 23 |
| cosine | 34 of 70 | 38 of 70 | 28 of 70 | 14 of 19 |
| correlation | 27 of 70 | 34 of 70 | 21 of 70 | 13 of 19 |
| minkowski_p0.5 | 30 of 70 | 30 of 70 | 32 of 70 | 8 of 15 |

Largest single win over manhattan: +0.0684 AUROC (cosine, caco2, number 1).

f34_full_sweep, predicting a held-out molecule's activity. Winning distance per
(dataset, representation), over every distance f18 defines for that representation's kind:

| dataset | winner on the two hybrid spaces | winner on the single representations |
|---|---|---|
| herg_ki | minkowski_p0.5, manhattan | cosine 1, tanimoto 2, soergel 1, euclidean 1, correlation 2, minkowski_p0.3 1, manhattan 1 |
| caco2 | cosine, cosine | tanimoto 4, cosine 2, correlation 1, dice 1, minkowski_p0.3 1 |
| adme_mdr1 | manhattan, cosine | dice 4, soergel 1, cosine 2, correlation 1, minkowski_p0.5 1 |
| herg | minkowski_p0.3, minkowski_p0.5 | euclidean 2, manhattan 3, dice 1, cosine 3, correlation 1 |

So on the hybrid spaces, which are the ones the noise experiments mostly use, the winner was
neither manhattan nor euclidean on 5 of the 8 (dataset, hybrid) pairs.

f34_neighbour_weighting_cross, same question on 7 representations and 3 datasets. Winner per
(dataset, representation) by coefficient, over 21 pairs: dice 7, cosine 5, tanimoto 4,
euclidean 1, tversky 1, soergel 1, minkowski_p0.3 1, minkowski_p0.5 1. chebyshev won 0 of 90
groups it was defined in, mahalanobis 0 of 90, correlation 1 of 90.

## What cannot travel as a plain sklearn argument

Checked against `sklearn.neighbors.VALID_METRICS` on 2026-10-01 (f34 write-up result 7).
sklearn accepts `dice` and `jaccard`. It has no name for `tanimoto`, `tversky` or `soergel`, and
none of these names works on a count fingerprint. Those need a precomputed distance matrix, which
is the path `simeval.py:161` already uses. On real MACCS bits scipy's `jaccard` equals simeval's
tanimoto to 1.110e-16.

## Cost

A distance is one setting per feature space, so each name added or removed moves the count by the
number of weighted feature spaces kept on axis 1: 7 on full_knn, 4 on fast_knn, times the 4
importance weightings.

| distances kept | full_knn settings | fast_knn settings |
|---|---|---|
| 1 | 30 | 17 |
| 2 (today) | 58 | 33 |
| 3 | 86 | 49 |
| 4 | 114 | 65 |

Counts include numbers 8 and 9 where they are kept, at 1 setting each.

Note on cost per distance, not just per setting: the fractional Minkowskis and soergel are the
dear ones. f34 measured soergel on a count fingerprint at 80 minutes on one core, and `pdv` under
the two fractional Minkowskis took two and a half hours of its 5 hour 40 minute total.

DECISION:

---

# Axis 3: importance weighting and its power

Not started.

DECISION:

---

# Axis 4: how many neighbours

Not started.

DECISION:

---

# Axis 5: how much each neighbour counts

Numbers brought to Adelaide 2026-10-05; no decision yet. Summary of what was counted:

Counting every neighbour equally wins no group in f34, where a group is one combination of
representation, distance and neighbour count: 0 of 688 on herg_ki, 0 of 680 on caco2, 0 of 680 on
adme_mdr1, 0 of 688 on herg.

In f38, counting equally is best at finding corrupted labels on 1 of herg_ki's 7 damage kinds, 0
of caco2's 7, 0 of adme_mdr1's 7, 0 of herg's 6. Best at fixing on 0 of 7, 1 of 7, 0 of 7 and 1
of 6. Censoring excluded throughout.

gaussian is best at finding on all 6 of herg's damage kinds, by 0.0170 to 0.0185 AUROC against a
median fold range of 0.0110 to 0.0145. On the three regression datasets the gaps between schemes
are smaller than the fold range.

DECISION:

---

# Axis 6: whether the scaffold group is excluded

Not started. f38 computes both lists; the comparison is not written.

DECISION:
