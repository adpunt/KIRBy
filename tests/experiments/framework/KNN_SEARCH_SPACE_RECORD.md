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

# Axis 1: which features the distance is measured on

## What the noise suite offers today

`tests/experiments/noise/similarity_space.py:120-131`. Ten values, of which seven take a
weighting and a distance and three are fixed.

| value | what it is | set at |
|---|---|---|
| base_1 | the model's best single representation, from its stored search | similarity_space.py:122 |
| base_2 | the model's second-best single representation | similarity_space.py:122 |
| base_1_aug_uncapped | base_1 with c21's augmentation features added, uncapped | similarity_space.py:119-121 |
| base_2_aug_uncapped | base_2 with the same | similarity_space.py:119-121 |
| hybrid | predictive general hybrid, rank 1 | similarity_space.py:126 |
| hybrid__general2 | predictive general hybrid, rank 2 | similarity_space.py:126 |
| hybrid_similarity | similarity general hybrid, rank 1 | similarity_space.py:126 |
| tanimoto_ecfp4 | 2048 Morgan bits at radius 2, scipy jaccard. No weighting, no distance choice | similarity_space.py:171-176 |
| forest_proximity | 1 minus the share of trees putting two molecules in the same leaf. Forest fitted on the noisy training labels | similarity_space.py:177-180 |
| model_input | today's behaviour: the model's own matrix, process-wide distance | similarity_space.py:201 |

The capped augmented spaces were cut on 2026-10-03 (similarity_space.py:115-119, Adelaide).

## What base_1 and base_2 actually are, per dataset and model

Read from f38's own rows, `base_reps_source` = "cached 5-fold baselines":

| dataset | model | base_1 | base_2 |
|---|---|---|---|
| herg_ki | dnn | mhg_gnn | grover |
| herg_ki | svm | mhg_gnn | ecfp4 |
| caco2 | dnn | ecfp4 | pdv |
| caco2 | svm | pdv | ecfp4 |
| adme_mdr1 | dnn | pdv | mhg_gnn |
| adme_mdr1 | svm | smited | pdv |
| herg | lightgbm | mhg_gnn | mordred |
| herg | xgboost | mordred | mhg_gnn |

On caco2 and herg the two chosen models rank the same two representations in opposite order. So
base_1 and base_2 together cover one pair of representations on those datasets, and any number
pooled over both models is the same for base_1 and base_2. That is why they tie in the tables
below. It is not a defect.

## Finding and fixing corrupted labels (f38)

Best over every importance weighting, distance, importance power and neighbour-counting scheme,
at 10 neighbours. Censoring left out, because its score is inverted. One cell is one combination
of damage kind, fold and model.

**Detection AUROC, higher is better. "Fix" is rmse to the clean label for the three regression
datasets, lower better; for herg it is the share of corrupted molecules put back in the clean
class, higher better.**

herg_ki, 7 damage kinds, 70 cells per space:

| space | detect | median fold range | fix |
|---|---|---|---|
| hybrid | 0.8142 | 0.0352 | 0.4848 |
| hybrid_similarity | 0.8142 | 0.0292 | 0.4829 |
| hybrid__general2 | 0.8134 | 0.0318 | 0.4837 |
| base_2_aug_uncapped | 0.8116 | 0.0392 | 0.5034 |
| base_1_aug_uncapped | 0.8109 | 0.0298 | 0.4908 |
| base_2 | 0.8096 | 0.0362 | 0.5144 |
| base_1 | 0.8074 | 0.0339 | 0.4984 |
| model_input | 0.8025 | 0.0320 | 0.5264 |
| base_2_raw | 0.8023 | 0.0351 | 0.5383 |
| tanimoto_ecfp4 | 0.7993 | 0.0369 | 0.5436 |
| forest_proximity | 0.7969 | 0.0364 | 0.6164 |

caco2, 7 damage kinds, 70 cells per space:

| space | detect | median fold range | fix |
|---|---|---|---|
| hybrid | 0.7805 | 0.0317 | 0.2574 |
| hybrid_similarity | 0.7803 | 0.0362 | 0.2538 |
| base_2_aug_uncapped | 0.7795 | 0.0376 | 0.2548 |
| hybrid__general2 | 0.7792 | 0.0302 | 0.2581 |
| base_2 | 0.7783 | 0.0401 | 0.2670 |
| base_1 | 0.7783 | 0.0401 | 0.2670 |
| base_1_aug_uncapped | 0.7782 | 0.0397 | 0.2543 |
| forest_proximity | 0.7710 | 0.0337 | 0.2660 |
| model_input | 0.7688 | 0.0292 | 0.2813 |
| base_2_raw | 0.7589 | 0.0459 | 0.2957 |
| base_1_raw | 0.7589 | 0.0459 | 0.2957 |
| tanimoto_ecfp4 | 0.7570 | 0.0456 | 0.2979 |

adme_mdr1, 7 damage kinds, 70 cells per space:

| space | detect | median fold range | fix |
|---|---|---|---|
| hybrid | 0.7739 | 0.0302 | 0.4363 |
| hybrid_similarity | 0.7738 | 0.0291 | 0.4289 |
| hybrid__general2 | 0.7737 | 0.0255 | 0.4357 |
| base_2_aug_uncapped | 0.7683 | 0.0299 | 0.4515 |
| base_1_aug_uncapped | 0.7662 | 0.0219 | 0.4409 |
| base_2 | 0.7593 | 0.0287 | 0.4730 |
| base_1 | 0.7578 | 0.0278 | 0.4646 |
| model_input | 0.7523 | 0.0309 | 0.4827 |
| forest_proximity | 0.7455 | 0.0326 | 0.5400 |
| tanimoto_ecfp4 | 0.7245 | 0.0304 | 0.5717 |

herg, 6 damage kinds, PART-WAY tasks, so the cell counts differ per space:

| space | detect | median fold range | fix | cells |
|---|---|---|---|---|
| hybrid__general2 | 0.8591 | not enough folds yet | 0.7875 | 4 |
| hybrid | 0.8569 | 0.0102 | 0.7873 | 33 |
| base_1_aug_uncapped | 0.8526 | 0.0116 | 0.7836 | 60 |
| base_2_aug_uncapped | 0.8523 | 0.0105 | 0.7821 | 53 |
| base_2 | 0.8515 | 0.0138 | 0.7826 | 60 |
| base_1 | 0.8515 | 0.0138 | 0.7826 | 60 |
| tanimoto_ecfp4 | 0.8260 | 0.0127 | 0.7539 | 60 |

hybrid_similarity, model_input and forest_proximity have no herg rows yet.

**Does the choice beat fold wobble?** Best space minus worst space, against the median
fold-to-fold range:

| dataset | best minus worst | median fold range |
|---|---|---|
| herg_ki | 0.0173 | about 0.034 |
| caco2 | 0.0235 | about 0.037 |
| adme_mdr1 | 0.0494 | about 0.029 |
| herg (part-way) | 0.0331 | about 0.012 |

## Predicting a held-out molecule's activity (f34_full_sweep)

Best setting on each dataset, over every distance, neighbour count and neighbour-counting scheme.
rmse for the three regression datasets, AUROC for herg.

| dataset | best hybrid | best single representation | gap | fold spread of the winner |
|---|---|---|---|---|
| herg_ki | 0.5604 (hybrid_general_distance) | 0.5948 (atom_pair) | 0.0344 | 0.0472 |
| caco2 | 0.3137 (hybrid_general) | 0.3213 (pdv) | 0.0076 | 0.0224 |
| adme_mdr1 | 0.4856 (hybrid_general_distance) | 0.5165 (pdv) | 0.0309 | 0.0127 |
| herg | 0.8490 (hybrid_general) | 0.8282 (graph_kernel) | 0.0208 | 0.0120 |

A hybrid is first on all four datasets for prediction, and on all four for detection. The two
questions agree on this axis, unlike axis 6.

## Which spaces give the same neighbours (f38 overlap files)

Share of a molecule's 10 nearest neighbours that two spaces agree on, for pairs differing in the
space alone. 1.000 means identical lists.

| pair | herg_ki | caco2 | adme_mdr1 |
|---|---|---|---|
| base_2_raw vs tanimoto_ecfp4 | 1.000 | 1.000 | not scored |
| base_1_raw vs tanimoto_ecfp4 | not scored | 1.000 | not scored |
| hybrid vs hybrid__general2 | 0.972 | 0.899 | 0.805 |
| base_1 vs base_1_aug_uncapped | 0.873 | 0.608 | 0.686 |
| base_2 vs base_2_aug_uncapped | 0.703 | 0.619 | 0.656 |
| hybrid vs hybrid_similarity | 0.710 | 0.534 | 0.385 |
| base_1 vs base_2 | 0.623 | 0.348 | 0.248 |

herg has no overlap rows yet; its tasks have not finished.

## Claude's reading of axis 1 — check it

- **base_1_raw and base_2_raw under tanimoto are the same list as tanimoto_ecfp4**, share 1.000
  on both datasets scored. Reason: base_1 or base_2 is ecfp4 on those datasets, and raw ecfp4
  bits under tanimoto is what tanimoto_ecfp4 computes. This is degenerate, in the sense the rules
  allow for a cut. Note these raw spaces are f38's own extra settings, not members of the noise
  suite's swept list, so cutting them changes f38's analysis and not the noise suite.
- **hybrid and hybrid__general2 share 0.805 to 0.972 of their neighbours.** This CANNOT carry a
  cut: f38 built hybrid__general2 as the next-best stored recipe that differs from the first,
  which is not the noise suite's rank-2 rule, and the per-fold precompute will give different
  recipes.
- **base_1 and base_2 are far apart**, sharing 0.248 to 0.623 of their neighbours. Both earn
  their place on this evidence.
- **tanimoto_ecfp4 and forest_proximity are last or near-last** on every dataset for both finding
  and fixing. They are also the two with no weighting and no distance choice, so each is one
  setting rather than eight.
- **model_input is below every base and hybrid space** on all three finished datasets for both
  finding and fixing. It is the one value that represents today's behaviour, so dropping it
  removes the control.

## Options for axis 1, one per line

Counts are settings, before multiplying by neighbour counts. Today: 7 spaces x 4 weightings x 2
distances + 2 fixed + model_input = 59.

- Keep all ten values. 59 settings. No cut.
- Cut forest_proximity. 58 settings. Reason to state: last or next-to-last on every dataset for
  both finding and fixing; its forest is fitted on the noisy labels, which is a different
  dependence from every other space.
- Cut tanimoto_ecfp4. 58 settings. Reason to state: last on caco2 and adme_mdr1 for both finding
  and fixing; identical to raw ecfp4 bits where ecfp4 is a base representation.
- Cut both fixed spaces. 57 settings.
- Cut the two augmented spaces. 51 settings. Reason to state: they sit between their base spaces
  and the hybrids on every dataset and never lead. Warning: n23 and n32 are built to compare
  augmented against plain, so this cut changes what they measure.
- Cut model_input. 58 settings. Reason to state: below every base and hybrid space on all three
  finished datasets. It is the control for today's behaviour.
- Cut base_1 and base_2, keep the hybrids only. 27 settings. Reason: hybrids lead on all four
  datasets. Warning: the handoff's rule 5 says the noise experiments use the 2 best base
  representations and 2 best hybrids per model, 8 configurations
  (`HANDOFF_PER_FOLD_ALIGNMENT.md` section 1 item 5), which this contradicts.
- Cut hybrid_similarity. 51 settings. Warning: six similarity experiments read the distance
  search, and the distance precompute is predictive-only
  (`project_similarity_hybrid_set_2026_10_02`). Not supported by f38, whose recipes are test
  spaces.

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

Not started. Numbers recounted and held: the noise suite allows manhattan and euclidean only
(`similarity_space.py:146, 184-185`), while f34's winning distance on the two 100-feature hybrid
spaces was cosine, minkowski p=0.5 or minkowski p=0.3 on 5 of the 8 (dataset, hybrid) pairs.

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
