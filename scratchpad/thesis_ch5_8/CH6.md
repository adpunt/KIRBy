# Chapter 6 (similarity in a hybrid): facts for the rewrite

Written by Claude on 2026-10-09 from code read in this session and from the raw result files under
`/Volumes/seagate/kirby_results_2026-10-09/similarity/` and `.../complexity/c04_importance_as_metric/`.
Nothing here was taken from the hand-written summaries of the answer pages.

Conventions in this file:
- R = `/Volumes/seagate/kirby_results_2026-10-09/similarity/`.
- VERIFIED = I read it in a raw result file or in code in this session. UNVERIFIED = only seen in a doc.
- Every number is a mean over 5 outer folds unless a fold count is given. All result files below have
  `n_outer_folds = 5`.
- "Small molecules" here means caco2, herg, herg_ki, logd, qm9. buchwald_hartwig is a reaction set.
  The antibody sets are abagym_EGFR, abcov_kd, cdr_flexibility, ddg_combined, flab_expression_koenig,
  flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_pr_cho, ginkgo_titer,
  humanness_therapeutic. tcga_brca_pam50 is gene expression.
- Counting by dataset puts 11 antibody sets against 5 small-molecule sets. Every "on N datasets" claim
  in the outline is therefore antibody-weighted unless split by domain. I split them below.

---

## 0. The changes the outline needs, in one list

1. 6.2's list of feature spaces is the noise suite's list, not Chapter 6's. Forest proximity, the
   base representations with augmented features, and "the model's own input matrix" appear in no
   similarity experiment. They come from `tests/experiments/noise/similarity_space.py:11-59`. The model's
   own input matrix is cut there too (line 142). Chapter 6's spaces are listed in section A.1.
2. No Chapter 6 experiment excludes a molecule's own group from its neighbours. Every result file
   stores `scaffold_group_excluded: false`. The one place a group is excluded is inside the similarity
   objective's own score (section A.2).
3. The importance weighting is not one formula. Three different powers are in use, depending on the
   experiment (section A.4). The exponent is swept only in f38 and n12, not in any Chapter 6
   experiment.
4. The similarity objective is undefined on more datasets than 6.2 says. tcga_brca_pam50 (4 classes)
   is missing from the outline. The code lists eight ginkgo sets, not five; four of them ran.
5. 6.3, "rises more steeply in the hybrid built for distance on every dataset that ran". abcov_kd is an
   exception: 0.0901 to 0.0376.
6. 6.3, coincidence of the best numbers of features. abcov_kd with LightGBM also coincides, at 500
   features under Manhattan. The two grids only overlap at 500 features on caco2, logd, buchwald and
   ginkgo_colloidal. So "prediction peaks at more features" is partly built into the grids.
7. 6.4, fixed setting. The hybrid loses on three datasets, not one. buchwald_hartwig goes 14.4572 to
   16.1811 RMSE and ginkgo_acsins 0.3912 to 0.3884 Spearman, besides ddg_combined. ddg_combined's
   number changed from 0.0864 to 0.0899.
8. 6.4, re-picked setting. The hybrid loses on seven datasets, not four. All five small-molecule sets
   win on both comparisons.
9. 6.4, Bayesian optimisation. It ran on two datasets, not one. On buchwald_hartwig the GP with the
   RBF kernel in the hybrid finds less than random picking. hERG Ki numbers moved slightly.
10. 6.5, CDR flexibility. "The same model goes 0.8393 to 0.8827" is XGBoost, not LightGBM. LightGBM is
    worse on the non-cliff molecules too: 0.8811 to 0.8665.
11. 6.5, "the gain is larger on the cliff molecules". Not true on the example given. On LogD with the
    network the gain is 0.0423 RMSE on cliff molecules and 0.0476 on the rest. QM9 is worse on cliffs.
12. 6.5, the no-data list misses abcov_kd.
13. 6.6, "the hybrid tracks error better except on three". It is worse on six datasets for both models.
14. 6.6, the narrower interval is not unique to hERG Ki. buchwald_hartwig also qualifies, with both
    models. More important: the interval has one width for every molecule in a fold
    (`s06_applicability_domain.py:73-75`). It measures the model's error in each space. It does not use
    distance.
15. 6.6, clustering. It flips between cluster counts on seven datasets, not five. SMAC colloidal
    stability does not lose both: it wins the re-picked retrieval comparison. abcov_kd loses both.
    Retrieval does not beat the principal-component control on four antibody sets.
16. 6.6, s03 has results now. The kept features carry no more label information than a random subset
    of the same size on most datasets. They carry less than principal components of the same size on
    most.
17. thesis.tex Chapter 6's similarity results are three-fold local runs on hERG Ki only. All are
    superseded. Its statement that the similarity hybrid is skipped on two of 16 datasets is wrong
    now. Its weighting formula matches only two of the experiments.
18. Results are missing for flab_expression_koenig in eight experiments, and for qm9 in three.

---

## A. METHODS, from code

### A.1 Feature spaces Chapter 6 actually compares

Read from `tests/experiments/similarity/similarity_helpers.py:321-363` (`fold_similarity_selection`) and
each script's docstring. Everything is chosen per outer fold from searches on that fold's training
molecules only. VERIFIED.

| Space | What it is | Used in |
|---|---|---|
| kNN's best and second-best single representation | ranked by the per-fold distance search (the hybrid search with kNN as the model) | s02, s04, s05, s06, s08, s10, s12, s13 |
| Each decided model's best and second-best single representation | ranked by that model's per-fold search | s02, s04, s08, s12 |
| Each decided model's two best hybrids | two models per dataset, so four hybrids | all s-experiments except s01 |
| kNN's top hybrid of the distance search | the "distance hybrid" | s02, s04, s05, s06, s08, s10, s12, s13 |
| Top hybrid of the similarity-objective search | the "similarity hybrid" | same as above, where defined |
| Principal components of all representations | at 100 features and at each hybrid's width | s13, s11 (50 components) |
| Random Gaussian projection to 50 features | null compression | s11 |
| Raw ECFP4 bits, RBF kernel and Tanimoto kernel | GP surrogates | s11 |
| All ten representations of the dataset, each alone | no selection | s01 |
| All representations joined side by side | no hybrid | c04, s06 (error model only) |

ECFP4 under Tanimoto appears in two other places. It decides which molecules are activity cliffs in s04
and s12 (Tanimoto at or above 0.55). It is not a feature space a distance is scored in there.

### A.2 Per experiment

All experiments: 5 outer folds, scaffold-group-aware (`sim_outer_folds`). Features scaled with
`FeatureScaler` on the training molecules: 0/1 features left unscaled, the rest standardised, held-out
values cut to the training range (`similarity_helpers.py:176-190`). VERIFIED.

The kNN settings swept everywhere (`similarity_helpers.py:153-173`, `noise/similarity_space.py:228-231`).
VERIFIED:
- Distances: Manhattan, cosine, Minkowski p = 0.5 on datasets of 5,000 molecules or fewer. Manhattan
  and cosine only above 5,000 (herg, logd, qm9).
- Neighbour counts: 5, 10, 20.
- How much each neighbour counts: 1/rank, or 1/d². Gaussian and 1/d are not used in any s-experiment.
- A molecule's own scaffold group is never excluded. The molecule itself is excluded by index.

| Exp. | Question (from its docstring) | Score | Notes |
|---|---|---|---|
| s01 | How poorly do standard distances follow activity? | Spearman between pair distance and absolute activity difference, 50,000 held-out pairs per fold; also share of neighbours with a large activity gap; SALI | No model, no selection, all 10 representations. No importance weighting. |
| s02 | Does distance in a hybrid follow activity better than in the best single representation? | Same Spearman | Swept over the decided budgets. Weighted and unweighted. |
| s03 | Do the kept features concentrate label information? | Sum over at most 50 principal components of one-feature Kraskov estimates, minus a 3-permutation baseline, in nats | Not the joint mutual information; the docstring says so (`s03:26-38`). Compared with a random subset and principal components of the same size. |
| s04 | Are cliff molecules predicted better from a hybrid? | Dataset's own metric on cliff and non-cliff held-out molecules | ECFP4 Tanimoto ≥ 0.55. Molecules with SMILES only; 13 of 18 files are stubs pointing to s12. |
| s05 | Is prediction from nearest neighbours better in a hybrid's features? | Dataset's own metric | One setting fixed before the run: distance hybrid against kNN's best single representation, unweighted, Manhattan, 5 neighbours, 1/rank (`s05:111-119`). Also a re-picked comparison. |
| s06 | Does distance to the nearest training molecules track prediction error? | Spearman between mean distance to k neighbours and absolute error | Errors from each model fitted on all representations joined, or on its own hybrid. Plus split conformal intervals. |
| s08 | Is the best number of features the same for prediction and for distance? | Dataset metric (prediction side); distance Spearman (distance side) | Prediction swept over the general budgets, distance over the distance budgets (`s08:13-15`). Also Levina-Bickel intrinsic dimension. |
| s10 | Does multiplying features by importance help four models that read a distance? | Dataset metric | kNN, GP with RBF kernel, SVR with RBF kernel (regression only, GP skipped above 5,000 molecules), q-RASAR. |
| s11 | Is Bayesian optimisation more sample-efficient in a hybrid? | Area under the best-found-so-far curve over 100 picks, in label units | 3 seed sets, not folds. Skipped on antibodies, tcga and classification. |
| s12 | Same as s04 for any domain | As s04 | Without SMILES, structural similarity is 1/(1+d) on all representations joined, top 1% of pairs. |
| s13 | Better clusters and retrieval in a hybrid? | k-means label coherence (higher better); label gap to nearest neighbours (lower better) | k-means at 3, 5, 10 clusters, squared Euclidean. Principal-component controls. |
| c04 | Does importance work as a distance weighting? | R² and dataset metric | All representations joined. 5, 10, 20 neighbours, equal vote; plus 10 neighbours by 1/d. Regression only. |

### A.3 The similarity objective (`objective='similarity'`)

`src/kirby/find_best_hybrid.py:2128-2238`, `_readacross_objective_score`. VERIFIED.
- Scored on the held-out molecules of each inner fold of the search. No downstream model is fitted.
- Their groups are split by `GroupKFold` into up to 5 parts. Each molecule is predicted from its 5
  nearest neighbours in the other parts. So inside the objective a molecule's own group IS excluded.
- Neighbours weighted by 1/d (`KNeighborsRegressor(weights='distance')`).
- Distance: the process-wide one, Manhattan by default (`distance_config.py:63`).
- No importance weighting in this distance (`noise_mitigation_methods.py:313`).
- Regression: Spearman between predicted and true labels. Binary: 2 × (AUROC − 0.5).
- Returns undefined (NaN) with fewer than 10 molecules, fewer than 2 groups, constant labels or
  predictions, or more than 2 classes.
- The docstring names MODI (Golbraikh et al. 2014) as the published relative and says why it was not
  used.

### A.4 Importance weighting inside a distance: three formulas

VERIFIED from code.
- `compute_distance_matrix` (`src/kirby/utils/noise_mitigation_methods.py:89-104`), used by s02 and
  s08. Weights rescaled to mean 1, then raised to a power set by the distance: 1 under Manhattan, 0.5
  under cosine and Euclidean, 2 under Minkowski p = 0.5.
- `importance_feature_weights` (`tests/experiments/_shared.py:3390-3415`), used by s04, s05, s06, s10,
  s13. Power 0.5 under every distance, then rescaled to mean 1. The power reads
  `KIRBY_IMPORTANCE_EXPONENT`; results store 0.5 (`R/s10.../logd.json results.weighting_settings`).
- c04 (`c04_importance_as_metric.py:137-145`). Square root, rescaled to sum 1.
- The noise suite's similarity axis uses no weighting since 2026-10-05
  (`noise/similarity_space.py:41-43`).
- The power is swept only in `framework/f38_neighbour_screen.py` and `noise/n12_*`. The outline's "the
  exponent is swept" is not true of any Chapter 6 experiment.
- thesis.tex line ~3734 says "x ⊙ w under Manhattan, x ⊙ √w under Euclidean". That matches s02 and s08
  only.

### A.5 Where the similarity objective is undefined, and why

`tests/experiments/dataset_policy.py:619-678` and `similarity_objective_undefined` (816-854). VERIFIED.
- tcga_brca_pam50 and tz_her2_multiclass: more than 2 classes, and the binary score is AUROC. Missing
  from the outline.
- tz_her2_binary: folds are cut on edit-distance shells from the wild-type CDR-H3. A held-out block
  can hold one shell.
- Eight ginkgo sets (hic, tm2, titer, pr_cho, acsins, purity, aggregation, colloidal). Each uses the
  competition's 5 folds as its 5 groups. Inside an outer fold each inner held-out block is one group,
  so no molecule has a neighbour outside its group. Added 2026-10-06. Four ran in the similarity
  experiments: acsins, colloidal, pr_cho, titer.
- ddg_combined (SKEMPI plus AB-Bind binding free energy). One complex holds 69 of 429 rows. On outer
  folds 1-4 it falls alone in an inner held-out block, so every configuration is refused. Added
  2026-10-08. The code skips it by name on all 5 folds, including fold 0, which could have been
  scored. Confirmed in `R/s02.../ddg_combined.json folds[*].selection.unavailable`.
- The outline's "all five Ginkgo developability sets" should read "the Ginkgo sets (eight in the code;
  four in these runs)".

### A.6 The 6.2 CITATIONS block against code and refs.bib

| Claim in outline | Code | refs.bib |
|---|---|---|
| kNN regression; Zheng2000, Itskowitz2005 | Fine | Both present (lines 3316, 4005) |
| 1/d is Dudani1976 | Used only inside the similarity objective (A.3), not in s-experiments | Present (3305) |
| Inverse powers: shepard1968two | 1/d² is one of the two schemes in every s-experiment | Present (4419) |
| 1/rank and Gaussian have no source; `neighbour_weighting.py:71-72, 259-272` | 1/rank is used in every s-experiment. Gaussian is used in no s-experiment (F34 only). The line numbers point at the name list and `weight_array`. The two schemes are at lines 200-212 (Gaussian) and 214 onward (1/rank). | — |
| Forest proximity, Breiman2001, `noise_estimation.py:2234-2277` | Lines match (2232-2276). The forest is fitted on the training molecules' labels, all trees counted, 100 trees, 5 per leaf. Used in no Chapter 6 experiment: only noise experiments and f36/f38. Move to Chapter 7. | Breiman2001 present (1313). Li et al. 2005 not in refs.bib. |
| Importance-weighted kNN: Wettschereck, Aha and Mohri 1997 | Fine, but "the exponent is swept" is wrong for Chapter 6 (A.4) | Not in refs.bib |
| Excluding the scaffold group, `noise_estimation.py:204-262` | That function excludes repeat measurements of the same pair (BindingDB drug-target), via `exclude_groups`. No Chapter 6 experiment excludes anything. The similarity objective does, through GroupKFold. | — |

Sources the Chapter 6 code relies on that the block does not name:
- MODI, Golbraikh2014: in refs.bib (1776).
- Cliff threshold "Stumpfe & Bajorath 2019" in the s04 result files' `cliff_definition`. refs.bib has
  Stumpfe2012 only.
- SALI, Guha2008 (1800). van Tilborg 2022 for cliffs (1816).
- Beyer1999 for distance concentration, cited by s11 (1958).
- Split conformal: Vovk2005 (991), Shafer2008 (1000), Lei2018 (4552).
- Kraskov2004 for the s03 estimator (3162).
- Not in refs.bib: Levina and Bickel 2005 (intrinsic dimension, s08); Snoek 2012 (s11); a q-RASAR
  source (s10); a k-means source (s13).

---

## B. RESULTS against the outline

### 6.3 Does distance in a hybrid track activity better

**Bullet 1: standard representations (s01).** Source: `R/s01_similarity_baseline/<ds>.json`,
`by_rep[<rep>][<distance>].sac_rho_mean`, 5 folds. All 18 datasets ran. VERIFIED.
- Caco-2 highest single representation 0.2292 (top_torsion, Minkowski p = 0.5). Outline 0.2292. SAME.
- SMAC colloidal stability (ginkgo_colloidal) highest 0.0155 (ablang2, Minkowski p = 0.5). SAME.
- "On five datasets some representations sit at or below zero." Current: eight datasets have at least
  one representation and distance at or below zero. They are abagym_EGFR (2 of 30 pairs),
  cdr_flexibility (14 of 30), flab_expression_koenig (2), ginkgo_acsins (6), ginkgo_colloidal (15),
  ginkgo_pr_cho (5), humanness_therapeutic (2) and tcga_brca_pam50 (7). CHANGED.
- No small-molecule dataset has any value at or below zero. All eight are antibody sets plus tcga.
- Small molecules alone, highest per dataset: caco2 0.2292, qm9 0.2789, logd 0.1470, herg_ki 0.1170,
  herg 0.0412. herg, logd and qm9 have two distances only.
- Fold spread is large. herg_ki mhg_gnn under Manhattan runs 0.241, 0.111, 0.046, 0.020, 0.048 across
  the five folds.

**Bullet 2: the distance hybrid (s02, s08).** Source: `R/s02_hybrid_similarity/<ds>.json`,
`folds[i].comparisons["knn distance hybrid - knn base rank 1"].rows`, Manhattan, unweighted, no budget
limit. Mean of per-fold `single_sac_rho` and `hybrid_sac_rho`, 5 folds each. VERIFIED.
- hERG Ki 0.0729 to 0.2625. SAME.
- "On every dataset that ran". CHANGED. abcov_kd goes 0.0901 to 0.0376 unweighted and 0.0715 to
  0.0621 weighted. Every other dataset of the 17 that ran goes up.
- Small molecules: qm9 0.1298 to 0.3779, logd 0.0765 to 0.2687, herg_ki 0.0729 to 0.2625, caco2 0.1517
  to 0.1969, herg 0.0257 to 0.0552.
- "The models' own predictive hybrids do not always do this." Still true. Losses for at least one model
  hybrid: abagym_EGFR (svm), abcov_kd (all), buchwald_hartwig (rf), caco2 (svm hybrid 1, 0.1843 to
  0.1794), ginkgo_colloidal (dnn), ginkgo_pr_cho (all four), ginkgo_titer (dnn hybrid 1). herg LightGBM
  hybrid 1 ties at 0.0244.
- Caveat 1. The single representation compared is kNN's top-ranked one for prediction. It is not the
  one whose distance best follows activity. Against s01's best representation (picked with hindsight
  on the held-out molecules, Manhattan), the distance hybrid is below on caco2 (0.2292 against
  0.1969), abcov_kd and ginkgo_titer. The similarity hybrid on caco2 is above, at 0.2769.
- Caveat 2. Importance weighting closes much of the gap on its own. On logd the weighted single
  representation scores 0.2762, above the unweighted hybrid's 0.2687. The weighted hybrid scores
  0.3094. Same pattern on qm9: weighted single 0.3286, unweighted hybrid 0.3779, weighted hybrid
  0.4596.
- DEFECT (files out of step with the code). Five files are unweighted only. They are caco2,
  ginkgo_colloidal, ginkgo_titer, herg and herg_ki, stamped `weighting: "unweighted only (decided
  2026-10-06)"`. They were written before the weighted distances were restored on 2026-10-07. The
  other twelve carry both. A weighted comparison on these five needs the same command run again.
- NOT YET RUN: flab_expression_koenig (`R/_precompute_errors/flab_expression_koenig.json`
  `result_status: empty`; LEDGER.md row 19 shows 4 of 11 similarity experiments done).

**Bullet 3: best number of features (s08).** Source: `R/s08_optimal_dimension/<ds>.json`,
`optima_compare[model][distance][sac_unweighted|sac_weighted]`. 16 datasets. VERIFIED.
- Caco-2 SVR at 500 and CDR flexibility LightGBM at 50: both coincide under every distance and both
  weightings. SAME.
- CHANGED: also abcov_kd LightGBM at 500 (Manhattan and Minkowski, unweighted). Partial coincidences:
  abagym_EGFR dnn at 250 (cosine), ddg_combined dnn at 100 (cosine), ddg_combined svm at 500
  (Manhattan and Minkowski, weighted), flab_thermo_tresanco both models at 250 (cosine, weighted).
- Over all 184 model, distance and weighting combinations: prediction peaks at more features in 158,
  at the same number in 22, at fewer in 4.
- Caveat (built into the design). The two grids differ (`s08:13-15`). On caco2, logd, buchwald_hartwig
  and ginkgo_colloidal the prediction grid is 500, 1000, 2000 features and the distance grid is 100,
  250, 500. They share only 500. Prediction cannot peak below 500 there and distance cannot peak above
  it. On herg_ki the grids share 250 and 500.
- Caveat. Many prediction optima are ties. caco2 dnn scores 0.2785 RMSE at 1000 features and 0.2783
  at 2000. Fold values run 0.24 to 0.32.
- NOT YET RUN: qm9 (partial file only; LEDGER.md row 288 shows it running), flab_expression_koenig.

### 6.4 Prediction from nearest neighbours

**Bullet 1: fixed setting (s05).** Source: `R/s05_knn_prediction/<ds>.json`, `summary.fixed_ref`
(`single_score_mean`, `hybrid_score_mean`, `n_paired_folds` = 5). Distance hybrid against kNN's best
single representation, unweighted, Manhattan, 5 neighbours, 1/rank. VERIFIED.
- CHO polyreactivity (ginkgo_pr_cho) 0.2059 to 0.3591 Spearman. SAME.
- Binding free energy (ddg_combined) 0.1424 to 0.0899. Outline 0.0864. CHANGED (number).
- "Better on every dataset except binding free energy". CHANGED. Also worse:
  - buchwald_hartwig: 14.4572 to 16.1811 RMSE.
  - ginkgo_acsins: 0.3912 to 0.3884 Spearman.
- Small molecules, all better: caco2 0.3386 to 0.3246 RMSE, herg_ki 0.6280 to 0.5822 RMSE, logd 0.6766
  to 0.5831 RMSE, qm9 0.6692 to 0.4205 RMSE, herg 0.8143 to 0.8335 AUROC.
- Largest antibody gain: ginkgo_colloidal 0.1268 to 0.2418.

**Bullet 2: re-picked setting (s05).** Source: `fold_reselection_result.knn` (`best_single_reselected`,
`hybrid_reselected`), 5 folds. VERIFIED.
- "The hybrid loses on four datasets". CHANGED to seven:
  - abcov_kd 0.2897 to 0.1638
  - buchwald_hartwig 13.8442 to 14.0299 RMSE
  - cdr_flexibility 0.8734 to 0.8553 AUROC
  - flab_thermo_tresanco 0.5717 to 0.5605
  - ginkgo_acsins 0.4559 to 0.4410
  - ginkgo_pr_cho 0.3501 to 0.2600
  - ginkgo_titer 0.3221 to 0.2482
- All seven are antibody sets or the reaction set. All five small-molecule sets win, for example logd
  0.6384 to 0.5648 and qm9 0.6167 to 0.4221.
- "Picks from a different set of settings on each side". Confirmed in `fold_reselection_status`: 216
  hybrid settings against 36 single-representation settings on full-group datasets. On herg, logd and
  qm9 it is 144 against 24. Where the similarity hybrid is undefined it is 180 against 36.
- ginkgo_pr_cho's sign flips between the fixed and the re-picked comparison: +0.1532 fixed, −0.0901
  re-picked.

**Bullet 3: importance weighting (s10, c04).** Source: `R/s10_importance_downstream/<ds>.json`,
`results.summary[<space>][recipe|all]`. kNN: `weighted_knn.cells[*].weighted_vs_unweighted.delta_mean`.
SVR: `weighted_svr.delta.delta_mean`. GP: `weighted_gp[fixed|median].delta.delta_mean`. Positive means
weighting helped, oriented by the dataset's direction. Units are space × setting, 5 folds each. 16
datasets. VERIFIED.
- kNN: weighting better in 1,409 of 2,100. GP: 41 of 145. SVR: 24 of 91. Direction SAME as the outline.
- kNN exceptions where weighting mostly hurts: ginkgo_titer 41 of 126, humanness_therapeutic 40 of
  144, abcov_kd 67 of 144.
- q-RASAR, not in the outline: 558 of 1,212. About even.
- LogD 0.6512 to 0.5911 RMSE. SAME. It is `single_knn1/all/weighted_knn/cells/"cosine|k10|inverse_square"`,
  one of 12 settings. The pre-fixed setting (Manhattan, 5 neighbours, 1/rank) gives 0.6766 to 0.6065.
- GP and SVR run on regression only. GP is skipped on logd (5,028 molecules, over the 5,000 cap).
  Neither runs on cdr_flexibility, herg, humanness_therapeutic or tcga_brca_pam50.
- c04 (`R/../complexity/c04_importance_as_metric/<ds>.json`, `methods[m].knn["10"]`, unweighted against
  weighted dataset metric, 5 folds). All representations joined, 10 neighbours. Weighting helps for at
  least one importance method on all 20 regression datasets that ran. It loses for some method on five:
  abagym_Ang2 (treeshap), abcov_kd (treeshap), caco2 (deeplift), ginkgo_colloidal (lightgbm_gain 0.1337
  to 0.1206, and treeshap) and ginkgo_hic (deeplift, treeshap). Classification datasets are skipped
  (`classification_not_supported`). Small
  molecules: logd 0.6962 to 0.6693, qm9 0.5749 to 0.4841, herg_ki 0.6422 to 0.6198, caco2 0.3425 to
  0.3365 RMSE.
- NOT YET RUN in s10: qm9 (LEDGER.md row 287, running), flab_expression_koenig.

**Bullet 4: Bayesian optimisation (s11).** Source: `R/s11_bayesian_optimization/<ds>.json`,
`results.conditions[<cond>].metrics.budget_100.auc_mean`. 3 seed sets (`auc_n_seeds`; svm_h2 on hERG
Ki has 2). Not outer folds. VERIFIED.
- hERG Ki, lower is better, hunting low pKi (`LABEL_BETTER_WHEN`, `s11:636-637`):
  - second network hybrid rebuilt every 25 picks (`hybrid_50d_refit__dnn_h2`): 410.59. Outline 410.63.
  - raw ECFP4 under the Tanimoto kernel: 426.25. Outline 426.63.
  - Status: CHANGED, small.
  - Others: raw ECFP4 under the RBF kernel 428.03, random picking 431.70, principal components
    415.00, random projection 417.34.
  - The first network hybrid with rebuilding scores 417.55, worse than principal components.
  - Bayesian-network and NGBoost surrogates on hybrids are at or worse than random picking (up to
    435.83).
- CHANGED: it also ran on buchwald_hartwig, where higher is better (hunting high yield):
  - Best is the Gauche GP on LightGBM hybrid 2: 9992.31.
  - The RBF GP on LightGBM hybrid 1 scores 9871.61, below random picking at 9901.52.
  - Random projection scores 9849.84.
  - The Tanimoto condition was skipped (`no_ecfp4_rep_for_tanimoto`).
- Skipped: caco2 and logd ("labels are clipped at the 99th percentile, so the top 1% of molecules
  share one value"); herg (classification); all antibody sets and tcga (domain). qm9 is running.
- `config.rebuild_interval` is stored as 0. The every-25 rebuild is the `hybrid_50d_refit` condition
  itself (`s11:1690-1694`).

### 6.5 Activity cliffs

Sources: `R/s04_activity_cliffs/<ds>.json` `model_summary[model][base1|hybrid1]` (`mean_cliff`,
`mean_noncliff`) and `R/s12_cross_domain_cliffs/<ds>.json`
`summary.models[model][partition].score_cliff|score_noncliff.{base1,hybrid1}.mean`. VERIFIED.

s04 has real results on five datasets only: caco2, herg, herg_ki, logd, qm9. The other 13 files are
stubs ("molecular-only experiment"). They name s12 as the replacement with
`replacement_verified: false`. s12 ran on all 18.

- LogD with the network: cliff 0.4299 to 0.3876, non-cliff 0.4650 to 0.4174 RMSE, 5 folds. SAME.
- "The gain is larger on the cliff molecules". CHANGED for this example. The gain is 0.0423 on cliff
  molecules and 0.0476 on the rest. Per dataset:
  - caco2: larger on cliffs (dnn 0.3396 to 0.2881 cliff, 0.2890 to 0.2720 rest).
  - herg_ki: larger on cliffs (dnn 0.5696 to 0.5266, 0.6095 to 0.5788).
  - herg: larger on cliffs (LightGBM 0.7930 to 0.8082, 0.8883 to 0.8954 AUROC).
  - logd: smaller on cliffs.
  - qm9: worse on cliffs. In s12, dnn 0.2422 to 0.2798 and LightGBM 0.2612 to 0.2666 RMSE, on 2 of 5
    folds only. 48 cliff molecules held out in total.
- "Where cliff molecules could be scored, the hybrid beats the single representation on them". Other
  counter-examples:
  - buchwald_hartwig with random forest: worse on both cliff and non-cliff molecules.
  - humanness_therapeutic XGBoost: worse on cliffs under Manhattan and cosine.
  - tcga_brca_pam50 SVR: worse on cliffs under all three distances, 1 to 3 folds.
  - flab_expression_koenig network: worse on cliffs under Manhattan and Minkowski.
  - abagym_EGFR: worse on cliffs, 1 fold only.
- CDR flexibility (concatenation partition, Manhattan, 5 folds):
  - Cliff molecules: XGBoost 0.8642 to 0.8467, LightGBM 0.8437 to 0.8332 AUROC. SAME.
  - Non-cliff molecules: XGBoost 0.8393 to 0.8827, LightGBM 0.8811 to 0.8665. CHANGED. The outline
    pairs XGBoost's non-cliff numbers with LightGBM. For LightGBM the hybrid is worse on both kinds of
    molecule.
  - "None of 18 settings on cliff molecules, 9 of 18 on the rest". This is the kNN count (18 = 3
    distances × 3 neighbour counts × 2 schemes), as on the s12 answer page line 110. Not re-derived by
    me. UNVERIFIED.
- "No data" list: ddg_combined, flab_thermo_tresanco (Manhattan; cosine has 1 fold with 3 cliff
  molecules) and the four ginkgo sets. SAME. CHANGED: abcov_kd also has no data (10 cliff molecules
  over 141 held out).
- Antibody gain worth naming: flab_expression_koenig with SVR, cliff 0.4884 to 0.6020, non-cliff
  0.7123 to 0.7866 Spearman (Manhattan).
- DEFECT. The cliff rule aims at 15% of training molecules. Held-out molecules come out at about 40%.
  logd folds run 39.1% to 44.5%, herg 38% to 46%, caco2 28% to 33%, herg_ki 19% to 27%
  (`cliff_definition.by_fold[i].pct_heldout_cliff`). The held-out rule is "at least two similar
  training molecules differ by more than the gap". It is looser than the training rule. So "cliff
  molecule" means something different in training and held-out molecules.
- DEFECT, minor. On the concatenation branch, s12 standardises features "over every row"
  (`cliff_info.structural_source`). Held-out features reach the structural cutoff. Labels do not.

### 6.6 Other uses of a distance

**Bullet 1: distance against error (s06).** Source: `R/s06_applicability_domain/<ds>.json`,
`distance_error_correlation.manhattan.k5["<model>_all_representations"].per_fold`, `knn_base1` against
`hybrid` (the distance hybrid), mean of 5 folds. Errors from each model fitted on all representations
joined. VERIFIED.
- Caco-2 with SVR errors: 0.1241. SAME. Other small molecules are higher: herg_ki SVR 0.1996, logd SVR
  0.2163.
- "Does not rise at all on three datasets". SAME: buchwald_hartwig, ddg_combined and ginkgo_titer are
  at or below zero for both models.
- "The hybrid tracks error better except on three". CHANGED. Lower for both models on six:
  - abcov_kd: XGBoost 0.1865 to 0.0723
  - buchwald_hartwig
  - ginkgo_colloidal
  - herg: LightGBM 0.1390 to 0.0185
  - humanness_therapeutic
  - tcga_brca_pam50: dnn 0.3253 to 0.1254
  - Mixed on abagym_EGFR.
- Small molecules: better on caco2, herg_ki, logd and qm9; worse on herg.
- Interval, hERG Ki with the network: width 2.1086 to 1.9476, coverage 0.9044 to 0.9129. SAME. Source:
  `conformal.by_model.dnn.arms.{knn_base1,hybrid}.{mpiw,picp}`.
- "Appears on hERG Ki with the network alone". CHANGED. Under the same rule (narrower, coverage at
  least 0.9), buchwald_hartwig qualifies with both models: rf 63.16 to 54.80 at 0.9066, LightGBM 65.85
  to 60.87 at 0.9226.
- DEFECT in what the interval means. The nonconformity is the plain absolute residual. Its width is the
  same for every held-out molecule in a fold (`s06_applicability_domain.py:73-75`, decided
  2026-10-06). Each space refits the model. So a narrower interval means smaller model errors in that
  space. It says nothing about distance as an applicability domain. The bullet should move to
  prediction, or the interval should be scaled by distance.
- Classification datasets have no interval: cdr_flexibility, herg, humanness_therapeutic,
  tcga_brca_pam50.

**Bullet 2: retrieval and clustering (s13).** Source: `R/s13_clustering_retrieval/<ds>.json`.
Clustering: `clustering_delta_by_cluster_count.clustering[3|5|10]`. Re-picked:
`fold_reselection_clustering_result.kmeans`. Retrieval: `retrieval_delta_by_setting` (count of
settings with delta > 0) and `fold_reselection_retrieval_unweighted_result.knn`. Principal components:
`*_vs_pca_result`. 5 folds. 17 datasets. VERIFIED.
- LogD wins both: coherence 0.1002 to 0.2252 at 10 clusters. SAME. Retrieval wins 12 of 12 settings;
  re-picked cost 0.5932 to 0.5434.
- "Retrieves with closer labels than the best single representation". CHANGED. The re-picked retrieval
  cost rises (worse) on:
  - abcov_kd 1.0933 to 1.2040
  - buchwald_hartwig 12.013 to 12.149
  - cdr_flexibility 0.2039 to 0.2082
  - ddg_combined 1.4386 to 1.5072
  - By setting: abcov_kd wins 2 of 18, ddg_combined 1 of 18, ginkgo_colloidal 5 of 18.
- "Beats the principal-component control". CHANGED. The control is better than the hybrid on
  abagym_EGFR (0.2610 against 0.2696), abcov_kd (1.0865 against 1.2040) and cdr_flexibility (0.1831
  against 0.2082). It ties on ginkgo_pr_cho (0.1546).
- "Clustering flips between cluster counts on five datasets". CHANGED to seven: buchwald_hartwig,
  caco2, ddg_combined, ginkgo_acsins, ginkgo_titer, herg_ki, and humanness_therapeutic (by 0.0002 at 3
  clusters).
- "SMAC colloidal stability loses both". Partly CHANGED. Clustering loses at all three cluster counts
  (10 clusters 0.5369 to 0.4608). Retrieval loses 13 of 18 settings. But the re-picked retrieval
  comparison wins, 0.4586 to 0.4241. abcov_kd is the cleaner "loses both" case: clustering loses at
  all three counts, retrieval at 16 of 18 settings.
- Small molecules: retrieval wins all settings on caco2, herg, herg_ki, logd, qm9. Clustering wins all
  counts on herg, logd, qm9. It flips on caco2 and herg_ki.
- Clustering is k-means under squared Euclidean, not the distances swept elsewhere
  (`config.clustering_distance`). The matched clusterers are off (`clustering_matched_note`).

**Bullet 3: label information in the kept features (s03).** The outline has no number. Source:
`R/s03_feature_concentration/<ds>.json`, `folds[i].models[m].hybrids[*]` with slot `top1`,
`per_rep[rep].summed_component_mi_{selected,random,pca}.held_out`. Counted over fold × model ×
representation units. 18 datasets. VERIFIED.
- Selected better than a random subset of the same size, by dataset:
  - About half on most: caco2 52/100, logd 53/100, herg 20/51, ginkgo_acsins 24/77, ginkgo_colloidal
    17/62.
  - Clearly above half only on qm9 50/58, herg_ki 72/99, flab_expression_koenig 62/98 and
    flab_thermo_tresanco 44/70.
- Selected better than principal components of the same size: below half on 15 of 18. The three at
  or above half are buchwald_hartwig 15/20, qm9 40/58 and flab_expression_koenig 53/98.
- The estimator is a sum of one-component estimates, not the mutual information (`s03:26-38`).
- If kept, this result goes against the outline's framing that the kept features concentrate label
  information. It should be stated as found.

---

## C. thesis.tex Chapter 6 (`\chapter{Hybrid Representations for Similarity and Label Noise}`, 3603-4739)

The outline's line ranges for the noise half (3705-3975 and 4102-4667) are stale. Current lines:

| Lines | Section | Words outside comments | Similarity or noise | Status |
|---|---|---|---|---|
| 3603-3614 | takeaways comment | — | both | Rewrite. Takeaways 3-6 are noise. |
| 3615-3659 | Introduction | ~540 (with Methods intro) | both | RQ 2 and 3 and the status paragraph are noise, move to Ch7. RQ 1 and the first paragraph are raw material. The status sentence's hERG Ki result is superseded. |
| 3660-3699 | Methods preamble | | mostly noise | The "three tasks kept apart" paragraph and the search-settings paragraph go to Ch7. |
| 3700-3737 | Similarity in a Hybrid | 626 | similarity | Keep as raw material, with fixes below. |
| 3738-4012 | Noise Conditions, Noise Mitigation Methods (with tab:noise_estimators, tab:noise_corrections), Detection Scores | 1,865 | noise | Moves to Ch7. |
| 4013-4138 | Results: Similarity (tab:ch6prelim_knn, _knn_weighted, _neighbor_gap) | 1,078 | similarity | All numbers superseded (see below). |
| 4139-4704 | Results: noise detection, LNL by noise type, real labels, comparison | 4,826 | noise | Moves to Ch7. |
| 4705-4739 | Conclusions | 158 | both | Only the first sentence is similarity, and it is superseded. |

What in 3700-3737 is still correct (checked against code this session):
- Two uses of a hybrid for similarity; the similarity hybrid ranks by 5-neighbour prediction with no
  model fitted. Correct.
- 1/d weighting of the 5 neighbours, groups split into up to 5 parts, neighbours drawn from other
  parts, Spearman or 2(AUROC − 0.5), undefined below 10 molecules or 2 groups. Correct
  (`find_best_hybrid.py:2193-2237`).
- "The neighbor score applies no importance weighting". Correct.
- "Manhattan distance on features z-scored as in Chapter 5". Now 0/1 features are left unscaled and
  held-out values are cut to the training range. Update.
- "Not searched on two of the 16 datasets (PAM50, trastuzumab-HER2)". WRONG now. Add the eight ginkgo
  sets (added 2026-10-06) and ddg_combined (added 2026-10-08). The same claim is repeated in the
  Conclusions notes (Paragraph 5).
- Importance-weighting paragraph ("x ⊙ w under Manhattan, x ⊙ √w under Euclidean"). Matches s02 and
  s08 only. s04, s05, s06, s10 and s13 use √w under every distance; c04 uses √w rescaled to sum 1.
- The paragraph defining "mean absolute difference to five nearest training molecules" is the s13
  retrieval cost. Keep the definition if s13 stays.

What in 4013-4138 is superseded:
- Every number. They come from local three-fold runs on hERG Ki with fixed hybrid settings (250
  features, linear allocation), Euclidean distance and equal neighbour votes. The single
  representations were chosen using all molecules. The text says so itself (line ~4060).
- The comment notes for this section describe the old design: Euclidean, budgets 10-500, "top three
  similarity hybrids". The current design is the one in section A.
- The c04 paragraph (R² 0.4638 to 0.5235) is from a dirty local run (comment cites
  `/Volumes/seagate/dirty_local_2026_09_28/...`). Current c04 on herg_ki, 5 folds, 10 neighbours:
  RMSE 0.6422 to 0.6198 (lightgbm_gain). Its no-shuffled-weights caveat still holds: no experiment
  has that control.

---

## D. Datasets missing results, per experiment

From `ls R/<exp>/*.json` against s01's 18 datasets. VERIFIED.

| Experiment | Missing | Status |
|---|---|---|
| s02, s05, s06, s13 | flab_expression_koenig | NOT YET RUN |
| s08, s10, s11 | flab_expression_koenig, qm9 | qm9 running (LEDGER.md rows 287-289) |
| s04 | 13 of 18 are stubs; real results on caco2, herg, herg_ki, logd, qm9 only | by design (molecules with SMILES) |
| s11 | ran on herg_ki and buchwald_hartwig only | caco2 and logd skipped for clipped labels |
| s01, s03, s12 | none | — |

Datasets not in the similarity runs at all, though c04 has them: abagym_Ang2, abagym_VEGF, adme_mdr1,
adme_rlm, bh_doyle_plates, ginkgo_hic, tz_her2_binary, tz_her2_multiclass.

## E. Antibody datasets where the hybrid does far better than elsewhere (for the domain discussion)

Candidates only. The author decides which to name.
- humanness_therapeutic: highest single-representation correlation of any dataset except tcga (0.3885,
  s01). Distance hybrid 0.2902 to 0.4666, similarity hybrid 0.5118 (s02).
- ginkgo_colloidal (SMAC colloidal stability): 5-neighbour prediction 0.1268 to 0.2418 fixed, 0.0918
  to 0.2906 re-picked (s05). It is the near-zero case in s01 (0.0155) and loses clustering (s13).
- ginkgo_pr_cho (CHO polyreactivity): clustering 0.0762 to 0.3266 at 10 clusters (s13). Fixed
  5-neighbour prediction 0.2059 to 0.3591. But the re-picked comparison loses (s05).
- flab_expression_koenig: cliff molecules with SVR 0.4884 to 0.6020 (s12). Missing from s02, s05, s06,
  s08, s10, s13.
- abcov_kd is the opposite case. It loses in s02, s05 (re-picked), s06 and s13, on 141 molecules.
