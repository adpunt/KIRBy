# Conditional patterns across Chapters 5 to 8 (exploratory)

Written by Claude on 2026-10-09 for the thesis fact-gathering. Read-only analysis. Nothing here is a
measured verdict for the thesis. It is a set of conditional statements, each with its raw numbers,
for Adelaide to choose from.

## How to read this file

- **Two kinds of source.**
  - VERIFIED (raw) means I read the number in a raw result file under
    `/Volumes/seagate/kirby_results_2026-10-09/` in this session, with a script listed at the end.
  - PAGE means the number comes from a generated table in
    `/Volumes/seagate/claude_analysis_2026-10-08/answers/`. Those tables were rebuilt on 9 Oct
    from the raw files. I parsed the tables but did not re-read every raw file behind them. I spot-checked
    one: n18 SMAC colloidal stability, support vector regressor, untreated 0.27404934 and cleaned 0.40336057
    Spearman, in `noise/n18_clean_the_real_labels/ginkgo_colloidal.json` (`no_cleaning_mean`, `cleaned_mean`).
    It matched, and that file has `folds_run: [0]`.
  - None of the hand-written summaries at the top of the answer pages was used.
- **Folds.**
  - The hybrid-against-single results (c02, c22, c01) cover 5 outer folds.
  - Every noise result covers fold 0 only.
- **Every noise number leans optimistic.** Each noise table picks the best setting of a method on the
  held-out molecules. The pages say so themselves.
- **Domain balance.** The run has 9 small-molecule datasets and 15 antibody datasets. Every pattern below is
  stated per domain. Where a pattern holds only because antibodies outnumber small molecules, I say so.
- **No significance tests.** Counts are what was counted, with the unit stated.
- **Forking paths.** I sliced these results about 20 ways: domain, size, noise type, model, action, detector,
  neighbour space, distance scaling, budget, allocation method, importance method, internal evaluator,
  representation count, metric, and representation family. With that many slices, some patterns will look
  real by chance. The ones I would trust are marked STRONG. They hold on every dataset in a domain, or
  they have a control.

---

## 1. The patterns, short list

STRONG means it holds on every dataset in its domain that ran, or it beats a control. Each one points to a section below.

1. **STRONG. On small molecules the hybrid beats the single representation on almost every fold.**
   - The count: 78 of 80 fold-and-model pairs, across 8 datasets and 5 folds (VERIFIED raw, c02). Both losses are on hERG.
   - It also beats the representation that scored best on the held-out molecules, a single representation
     that has seen the answer, on all 16 dataset-and-model pairs.
   - On antibodies the hybrid beats the selected single representation on 83 of 140 fold-and-model pairs.
     It beats the held-out-best single representation on only 6 of 28 dataset-and-model pairs. §2.
2. **STRONG. On antibodies the search often picks the wrong single representation.**
   - Picking on training molecules found the held-out-best representation on every fold for 4 of 28
     antibody dataset-and-model pairs.
   - On small molecules it did so for 8 of 16.
   - Much of the antibody "hybrid beats single" result is the hybrid beating a badly chosen single
     representation, not beating the best one. §2.3.
3. **STRONG, but it contradicts the antibody hypothesis as the pools stand.**
   - Small-molecule hybrids draw on many representations of different kinds.
     - About 5.5 to 7.2 representations carry the permutation importance, on caco2, herg_ki and the ADME sets.
     - Fingerprints, descriptors, pretrained graph models and SMILES models each give 10 to 47% of the features.
   - Antibody hybrids are mostly language-model embeddings of the same sequence. Language models give 69 to 100% of the features, and at least 90% of the permutation drop on 11 of 14 sets.
     - 1.4 to 5.4 representations carry the permutation importance. Most sets are at 2 to 4.
   - So antibodies are not where the hybrid finds more distinct features. They are where the pool offers less. §3.
4. **STRONG. A clipped label is fixed when the clip is flagged, and rarely otherwise.**
   - On small molecules under 25% clipping, moving the flagged labels helped both the single representation
     and the hybrid in 38 of 38 rows.
   - Detector-chosen actions under the same clipping helped in 51 of 168 rows. Noise-robust training helped in 25 of 87.
   - Pooled over every method, clipping is one of the least-fixed damage kinds. §4.1.
5. **STRONG on small molecules. The neural network gains from noise handling more often than the support vector regressor.**
   - Paired on the same method, dataset and damage: 421 against 260 of 711 conditions, small molecules.
   - It holds on caco2, herg_ki and logd separately.
   - On antibodies it does not hold. On binding free energy the regressor wins, 50 against 41 of 75. §4.3.
6. **On the labels as delivered, detector-chosen cleaning beats its control more often on antibodies than on small molecules.**
   - Antibodies: 22 to 51% of model-and-configuration pairs.
   - Small molecules: 7 to 21%.
   - Domain and dataset size are confounded. Every antibody set has 141 to 672 molecules. Every
     small-molecule set has 1,412 to 9,978. §4.5.
7. **Noise-robust training helps the network on the regression datasets and almost never on the classification ones.**
   - hERG: 3 of 70 rows. Antibody humanness: 0 of 30 rows.
   - LogD: 91 of 158 rows. caco2: 94 of 202 rows.
   - On the classification sets those rows are a neural network. Neither of the two models decided for those datasets is a neural network. §4.2.
8. **On hERG under label flips, down-weighting the molecules a detector flags helped in 93 of 104 rows.**
   - Deleting them helped in 58 of 153 rows.
   - On the regression small molecules no action clearly leads. §4.2.
9. **Finding neighbours in a hybrid helps down-weighting more often than finding them in a single representation.**
   - This holds on all 4 small-molecule datasets that ran: caco2, herg, herg_ki, logd.
   - The comparison is unequal. The hybrid-space rows cover 8 model-and-configuration pairs. The single-space rows cover 2. §4.6.
10. **Properties of the chosen hybrid do not predict whether it wins, on small molecules.**
    - Nothing varies: the hybrid wins whatever the budget, allocation method, importance method or internal evaluator.
    - On antibodies some properties line up with winning, but each one is confounded with which dataset it is. §2.4.

---

## 2. Hybrid against single representation, by domain

Source: `complexity/c02_complexity_matching_grid/<ds>.json`, keys
`standard_selection_summary[model][base1|hybrid1].per_fold[*].heldout_score` and `.name`. VERIFIED raw.
5 outer folds. The single representation (`base1`) and the hybrid (`hybrid1`) are each the first on that
fold's training molecules. Hybrid name format: allocation method / importance method / budget /
internal evaluator / step / representation count / subset.

herg_ic50 and flab_binding_koenig have no c02 summary file (flab_binding_koenig has only a `.partial.jsonl`).

### 2.1 Small molecules: the hybrid wins on 78 of 80 fold-and-model pairs

One row is one dataset and model. The scores are held-out means over 5 folds, single → hybrid.

| dataset | model | metric | single → hybrid | folds the hybrid is better |
|---|---|---|---|---|
| qm9 | dnn | RMSE | 0.3619 → 0.3155 | 5 of 5 |
| qm9 | lightgbm | RMSE | 0.3720 → 0.3463 | 5 of 5 |
| logd | dnn | RMSE | 0.4497 → 0.4089 | 5 of 5 |
| logd | svm | RMSE | 0.4930 → 0.4302 | 5 of 5 |
| caco2 | dnn | RMSE | 0.3042 → 0.2775 | 5 of 5 |
| caco2 | svm | RMSE | 0.3036 → 0.2862 | 5 of 5 |
| herg_ki | dnn | RMSE | 0.6031 → 0.5691 | 5 of 5 |
| herg_ki | svm | RMSE | 0.5881 → 0.5590 | 5 of 5 |
| adme_hlm | dnn | RMSE | 0.4660 → 0.4273 | 5 of 5 |
| adme_hlm | svm | RMSE | 0.4574 → 0.4267 | 5 of 5 |
| adme_rlm | dnn | RMSE | 0.5349 → 0.4987 | 5 of 5 |
| adme_rlm | svm | RMSE | 0.5209 → 0.4818 | 5 of 5 |
| adme_mdr1 | dnn | RMSE | 0.4732 → 0.4438 | 5 of 5 |
| adme_mdr1 | svm | RMSE | 0.4678 → 0.4370 | 5 of 5 |
| herg | lightgbm | AUROC | 0.8555 → 0.8666 | 4 of 5 |
| herg | xgboost | AUROC | 0.8532 → 0.8644 | 4 of 5 |

### 2.2 Antibodies: the hybrid wins on 83 of 140 fold-and-model pairs

The same layout. The metric is Spearman unless marked.

| dataset | molecules | model | single → hybrid | folds the hybrid is better |
|---|---|---|---|---|
| flab_expression_koenig | 4,275 | svm | 0.7029 → 0.7729 | 5 of 5 |
| flab_expression_koenig | 4,275 | dnn | 0.7163 → 0.7352 | 4 of 5 |
| ddg_combined | 429 | dnn | 0.1317 → 0.3542 | 4 of 5 |
| ddg_combined | 429 | svm | 0.0819 → 0.2178 | 4 of 5 |
| abagym_VEGF | 418 | gp | 0.1981 → 0.3017 | 4 of 5 |
| abagym_VEGF | 418 | svm | 0.4317 → 0.4737 | 3 of 5 |
| abagym_Ang2 | 397 | gp | 0.4550 → 0.5049 | 3 of 5 |
| abagym_Ang2 | 397 | svm | 0.4543 → 0.5216 | 3 of 5 |
| abagym_EGFR | 659 | dnn | 0.1810 → 0.2160 | 2 of 5 |
| abagym_EGFR | 659 | svm | 0.2821 → 0.3138 | 2 of 5 |
| ginkgo_acsins | 242 | gp | 0.5357 → 0.5547 | 3 of 5 |
| ginkgo_acsins | 242 | lightgbm | 0.4631 → 0.5242 | 3 of 5 |
| ginkgo_titer | 239 | dnn | 0.3083 → 0.3668 | 4 of 5 |
| ginkgo_titer | 239 | rf | 0.2504 → 0.2706 | 3 of 5 |
| ginkgo_pr_cho | 197 | dnn | 0.4003 → 0.4133 | 4 of 5 |
| ginkgo_pr_cho | 197 | gp | 0.5009 → 0.4419 | 2 of 5 |
| ginkgo_colloidal | 242 | dnn | 0.2766 → 0.2677 | 2 of 5 |
| ginkgo_colloidal | 242 | svm | 0.3485 → 0.3480 | 3 of 5 |
| ginkgo_hic | 242 | dnn | 0.5311 → 0.5293 | 3 of 5 |
| ginkgo_hic | 242 | gp | 0.6000 → 0.5515 | 2 of 5 |
| flab_thermo_tresanco | 672 | dnn | 0.6939 → 0.6764 | 2 of 5 |
| flab_thermo_tresanco | 672 | svm | 0.6901 → 0.6899 | 2 of 5 |
| abcov_kd | 141 | lightgbm | 0.2494 → 0.1654 | 2 of 5 |
| abcov_kd | 141 | xgboost | 0.2925 → 0.3050 | 3 of 5 |
| cdr_flexibility | 328 | lightgbm (AUROC) | 0.8713 → 0.8669 | 3 of 5 |
| cdr_flexibility | 328 | xgboost (AUROC) | 0.8527 → 0.8689 | 4 of 5 |
| humanness_therapeutic | 564 | rf (AUROC) | 0.9659 → 0.9614 | 1 of 5 |
| humanness_therapeutic | 564 | xgboost (AUROC) | 0.9511 → 0.9555 | 3 of 5 |

Other domains:
- buchwald_hartwig, RMSE in yield %:
  - rf: 14.4056 → 15.2383, hybrid better on 2 of 5 folds.
  - lightgbm: 16.1679 → 15.6211, hybrid better on 4 of 5 folds.
- tcga_brca_pam50, weighted F1:
  - dnn: 0.8528 → 0.9011, hybrid better on 4 of 5 folds.
  - svm: 0.8808 → 0.8950, hybrid better on 4 of 5 folds.

Two antibody losses are tiny:
- On SMAC colloidal stability with the regressor, the gap is 0.0005 Spearman.
- On nanobody thermostability with the regressor, it is 0.0002.

The outline calls both "loses for both models". The direction is right. But at those sizes, a loss and a tie cannot be told apart on 5 folds of about 50 and 135 antibodies.

### 2.3 Against the single representation that scored best on the held-out molecules

c01 (`complexity/c01_rep_characterization/<ds>.json`, `<rep>.baselines[model].score_mean`, 5 folds,
VERIFIED raw) scores every representation in the pool on the same aligned folds. Its best representation
by held-out mean is an upper bound for "the single representation". It is not a fair opponent, because it was picked on the
held-out molecules. The c02 single and the c01 single agree exactly where the search picked the same
representation, for example logd svm 0.4930 in both. So the folds line up.

| domain | dataset-and-model pairs | hybrid beats the selected single | hybrid beats the held-out-best single | training molecules picked the held-out-best single on every fold |
|---|---|---|---|---|
| small molecules | 16 | 16 | 16 | 8 |
| antibodies | 28 | 18 | 6 | 4 |
| reaction | 2 | 1 | 0 | 0 |
| gene expression | 2 | 2 | 2 | 1 |

The 6 antibody pairs are:
- ddg_combined, dnn: 0.2215 → 0.3542.
- ddg_combined, svm: 0.2034 → 0.2178.
- flab_expression_koenig, dnn: 0.7243 → 0.7352.
- flab_expression_koenig, svm: 0.7249 → 0.7729.
- ginkgo_acsins, gp: 0.5419 → 0.5547.
- ginkgo_acsins, lightgbm: 0.4722 → 0.5242.

Each pair reads held-out-best single → hybrid.

Example of the selection failing:
- On abagym_VEGF with gp, the selected single averages 0.1981 Spearman.
- esmc, the best on the held-out molecules, averages 0.4467.
- The hybrid's 0.3017 beats the first and not the second.

What this means for Chapter 5: on antibodies, "the hybrid beats the single representation" mostly
measures that a search on 150 to 500 training antibodies cannot tell which representation is best. On small molecules
the hybrid wins against both opponents.

### 2.4 Do properties of the chosen hybrid line up with winning?

The unit is one fold and one model. The source is the `hybrid1` name on each fold, VERIFIED raw.

**Small molecules.** The hybrid won 78 of 80 times, so no property can separate wins from losses. What the
search chose is still worth a sentence:
- Budget: 1,000 or 2,000 features in 65 of 80.
- Representation count: all of them (no top-N cut) in 66 of 80.
- Importance method: lightgbm_gain in 50 of 80. deeplift in 20 of 80, mostly for the network on the ADME sets.
- Allocation method: every one of the five appears.
- Internal evaluator: lightgbm or rf in 58 of 80.

**Antibodies.** Hybrid better, by property of the chosen hybrid. "Pairs" are fold-and-model pairs.

| property | value | hybrid better / pairs | datasets |
|---|---|---|---|
| importance method | treeshap | 44 / 60 | 14 |
| importance method | lightgbm_gain | 20 / 41 | 11 |
| importance method | deeplift | 19 / 39 | 10 |
| allocation method | power_2 | 17 / 23 | 12 |
| allocation method | rank_exp_d0.3 | 11 / 17 | 9 |
| allocation method | softmax_T1 | 25 / 40 | 14 |
| allocation method | wglobal_power2 | 21 / 40 | 11 |
| allocation method | performance_weighted | 9 / 20 | 11 |
| budget against training molecules | at or below | 45 / 68 | 10 |
| budget against training molecules | above | 38 / 72 | 11 |
| representation count | all | 47 / 71 | 13 |
| representation count | top 5 | 12 / 30 | 13 |
| internal evaluator same as downstream model | no | 54 / 85 | 13 |
| internal evaluator same as downstream model | yes | 29 / 55 | 14 |

None of these is clean:
- Each value clusters on particular datasets. The losing datasets (SMAC, HIC, thermostability, humanness)
  pull down whatever they happened to choose.
- treeshap's 44 of 60 is the largest gap. It has a stated mechanism nowhere in the code, and I would not report it.
- "Budget above training molecules" winning less often (38 of 72 against 45 of 68) is the only one with a
  mechanism: more features than molecules.
  - c33 already asks this directly (PAGE c33.md). Hybrids wider than the training molecules did better on
    ginkgo_titer and ginkgo_colloidal for both models, and worse on herg_ki and logd for both.
  - That matches the outline. Status SAME.
  - So the per-fold pattern above does not agree in direction with c33 on the two Ginkgo sets. Leave it out.

**The gap among single representations.**
- On small molecules the top three single representations sit close together. In c01, best against third
  best is within 0.004 to 0.049 RMSE, except qm9 lightgbm at 0.139.
  - Example: caco2 dnn 0.2980 against 0.3024.
- On antibodies the gap is 0.009 to 0.22 Spearman.
- Within antibodies a small gap does not predict a win:
  - flab_expression_koenig svm: gap 0.0086, hybrid better on 5 of 5 folds.
  - humanness rf: gap 0.0120, hybrid better on 1 of 5 folds.
- Source: VERIFIED raw, script `gap_vs_win.py`.

### 2.5 Related complexity results by domain (PAGE tables, rebuilt 9 Oct, not re-read raw)

- c17, the hybrid against PCA, mRMR and recursive elimination at the same number of features.
  - Small molecules: yes for both models on all 5 sets with a result (adme_mdr1, caco2, herg_ki, logd, qm9).
  - Antibodies: mixed, 9 of 18 dataset-and-model pairs yes.
- c11, the hybrid against an ensemble.
  - Small molecules: mixed, 7 of 12 pairs yes.
  - Antibodies: the ensemble usually wins, 7 of 24 pairs yes for the hybrid.
  - "One model per representation beats the hybrid": yes on 18 of 24 antibody pairs, and on 5 of 12 small-molecule pairs.
- c18, the budget split over the top 3 representations against all from the top 1.
  - Splitting wins in both domains: 14 of 14 small-molecule pairs, 24 of 30 antibody pairs.
  - The split is not a domain pattern. It is close to general.
  - The exception is humanness_therapeutic, where it loses for both models.

---

## 3. What the hybrids draw from: the antibody hypothesis

The hypothesis: antibodies are larger molecules, so there is more room for distinct featurisation, so
hybrids should help more.

Source: `complexity/c22_explainability/<ds>.json`, `per_fold[*].models[model].hybrids[slot=hybrid1]`.
Fields used:
- `allocation_built`: features taken per representation.
- `perm_share_by_rep`: share of the held-out score drop when a representation's features are permuted.

VERIFIED raw, 5 folds. "Effective number of representations" is 1 / sum(share squared). It is 10 when ten
representations contribute equally and 1 when one contributes everything.

c22 has no logd or adme_rlm file (only `.partial.jsonl`), and no flab_expression_koenig file.

| domain | dataset | model | representations given features (of 10) | effective number, by features | effective number, by permutation | top representation's share of permutation drop | hybrid better (folds) |
|---|---|---|---|---|---|---|---|
| small mol. | herg_ki | svm | 10.0 | 9.94 | 7.17 | 0.19 | 5 of 5 |
| small mol. | herg_ki | dnn | 9.8 | 9.48 | 6.74 | 0.20 | 5 of 5 |
| small mol. | adme_hlm | dnn | 10.0 | 8.70 | 6.53 | 0.25 | 5 of 5 |
| small mol. | adme_hlm | svm | 9.8 | 8.31 | 6.20 | 0.26 | 5 of 5 |
| small mol. | caco2 | svm | 10.0 | 9.92 | 6.08 | 0.22 | 5 of 5 |
| small mol. | caco2 | dnn | 10.0 | 8.85 | 5.60 | 0.26 | 5 of 5 |
| small mol. | adme_mdr1 | dnn | 10.0 | 9.02 | 5.79 | 0.26 | 5 of 5 |
| small mol. | adme_mdr1 | svm | 10.0 | 9.63 | 5.49 | 0.29 | 5 of 5 |
| small mol. | qm9 | dnn | 8.6 | 5.46 | 3.86 | 0.47 | 5 of 5 |
| small mol. | herg | xgboost | 5.8 | 4.90 | 2.77 | 0.49 | 4 of 5 |
| small mol. | herg | lightgbm | 4.4 | 3.41 | 2.28 | 0.56 | 4 of 5 |
| small mol. | qm9 | lightgbm | 3.0 | 2.72 | 1.27 | 0.88 | 5 of 5 |
| antibody | flab_binding_koenig | svm | 10.0 | 6.08 | 5.35 | 0.32 | 5 of 5 |
| antibody | flab_binding_koenig | rf | 10.0 | 8.14 | 4.28 | 0.37 | 5 of 5 |
| antibody | abagym_VEGF | svm | 9.0 | 8.40 | 4.66 | 0.38 | 3 of 5 |
| antibody | ginkgo_hic | dnn | 10.0 | 8.03 | 4.53 | 0.34 | 3 of 5 |
| antibody | ginkgo_pr_cho | dnn | 9.0 | 8.50 | 4.15 | 0.43 | 4 of 5 |
| antibody | abagym_Ang2 | svm | 7.6 | 6.57 | 4.03 | 0.41 | 3 of 5 |
| antibody | abagym_EGFR | svm | 7.6 | 5.58 | 3.87 | 0.47 | 2 of 5 |
| antibody | ginkgo_titer | dnn | 5.8 | 4.42 | 3.40 | 0.46 | 4 of 5 |
| antibody | ginkgo_acsins | gp | 7.4 | 6.17 | 3.36 | 0.47 | 3 of 5 |
| antibody | flab_thermo_tresanco | svm | 7.4 | 6.08 | 3.13 | 0.45 | 2 of 5 |
| antibody | ginkgo_colloidal | dnn | 7.8 | 6.59 | 2.89 | 0.52 | 2 of 5 |
| antibody | ddg_combined | svm | 6.4 | 5.81 | 2.32 | 0.65 | 3 of 5 |
| antibody | cdr_flexibility | xgboost | 3.0 | 2.97 | 2.24 | 0.60 | 4 of 5 |
| antibody | ginkgo_colloidal | svm | 4.6 | 3.56 | 2.01 | 0.65 | 3 of 5 |
| antibody | humanness_therapeutic | rf | 5.6 | 5.50 | 1.95 | 0.70 | 1 of 5 |
| antibody | abcov_kd | lightgbm | 3.4 | 2.63 | 1.57 | 0.78 | 2 of 5 |
| antibody | humanness_therapeutic | xgboost | 3.2 | 3.09 | 1.41 | 0.84 | 3 of 5 |
| reaction | buchwald_hartwig | lightgbm | 2.0 | 1.90 | 1.78 | 0.64 | 3 of 5 |
| gene expr. | tcga_brca_pam50 | svm | 3.8 | 2.29 | 1.34 | 0.86 | 4 of 5 |

The remaining antibody model rows sit between 1.6 and 4.0 by permutation. The full list is the output of `c22_blue.py`.

**Representation family.** These are shares of the hybrid's features, then shares of the permutation drop, mean over folds and models. VERIFIED raw, `c22_family.py`.
- **Small molecules: four families, none above about half.**
  - caco2 features: fingerprints and graph kernel 0.47, pretrained graph 0.25, SMILES models 0.17, descriptors 0.10.
  - caco2 permutation drop: pretrained graph 0.38, SMILES models 0.23, fingerprints 0.20, descriptors 0.18.
  - herg_ki permutation drop: pretrained graph 0.34, SMILES 0.25, descriptors 0.23, fingerprints 0.19.
- **Antibodies: language models of the sequence give 69 to 100% of the features, and at least 90% of the permutation drop on 11 of 14 sets.**
  - The three below 90% are abcov_kd 0.70, ddg_combined 0.72 and abagym_VEGF 0.78.
  - Protein plus antibody language models, by permutation drop:
    - thermostability 1.00.
    - humanness 1.00 (antibody language model alone 0.98).
    - SMAC 1.00.
    - HIC 0.98.
    - Ang2 0.97.
  - Hand-built sequence features (one-hot, BLOSUM, amino-acid composition, mutation features) carry 0.00
    to 0.30 of the permutation drop.
  - Their largest shares are abcov_kd 0.30, ddg_combined 0.28 and abagym_VEGF 0.22.
- **The antibody pools are themselves narrow** (pool lists in `_rep_dims.per_rep` of each c22 file).
  - Most hold 2 to 4 variants of ESM-2 (esm2_concat, esm2_early, esm2_midlayers, esm2_layer24,
    esm2_entropy_top20, esm2_cdr, esm2_per_region, esm2_wt_delta).
  - They hold at most 2 hand-built representations. 4 pools hold none.
  - The small-molecule pools hold 3 to 7 hand-built representations (fingerprints and descriptors) and no
    near-variants of one model. c01 file, `_meta.representations`.

**Intrinsic size of the representations** (c01, `<rep>.intrinsic_dim.pca_90`: features needed for 90% of
the variance; median over the pool; VERIFIED raw):
- Small molecules: 50.5 (herg_ki) to 106.5 (herg).
- Antibodies: 11.0 (ddg_combined) to 66.0 (abagym_EGFR). flab_*_koenig is the exception at 129.0.
- This measure is capped by the number of molecules, so it is low on the 242-antibody sets for that reason
  alone. It does not show the antibody representations carrying more independent directions.

**What this says about the hypothesis.**
- **Contradicts it, as the pools stand.** The antibody hybrids draw on fewer effective representations than the
  small-molecule hybrids, not more. They draw almost entirely on language models of one sequence.
- **Many antibody sets are deep mutational scans of one parent antibody:** koenig G6, the abagym sets, and
  the mutation sets inside ddg_combined.
  - There the antibodies differ by one or a few residues, so the large molecule mostly does not vary.
  - The variation a representation can pick up is small, not large.
  - Source: the dataset descriptions in THESIS_OUTLINE.md 5.3.1. The loaders were not re-read.
- **What might support a revised form.** The antibody sets where the hybrid beats even the held-out-best
  single representation each have something a single language model lacks:
  - **ddg_combined.** Its hybrids lean on a hand-built mutation representation.
    - svm fold 0: mutation_features takes 0.878 of the permutation drop with 35 of 100 features.
    - dnn fold 0: esm2_wt_delta, the language-model difference between mutant and wild type, takes it all.
  - **flab_expression_koenig.** The largest set, 4,275 antibodies.
  - **ginkgo_acsins.** It mixes two antibody language models, igt5 and ablang2, about evenly.
    gp fold 0: 0.445 and 0.555 of the permutation drop.
  - The data are consistent with "the hybrid helps where the pool holds representations of different kinds,
    or there are many molecules". That is the small-molecule picture too. It is a reading of 3 datasets, not a measured result.
- **flab_binding_koenig** (4,275 antibodies) has the most even antibody hybrid: 5.35 and 4.28 effective
  representations by permutation, 10 of 10 representations given features. Its hybrid is better on
  5 of 5 folds for both models in c22.
  - rf: 0.4325 → 0.5112.
  - svm: 0.4538 → 0.5211.
  - It has no c02 summary file yet. If c02 confirms it, it is the antibody set that looks most like the small molecules.

---

## 4. Noise handling: conditional patterns

Source for §4.1 to §4.4: PAGE `answers/NOISE_BY_GROUP.md`.
- I parsed it with `parse_noise.py` into 4,006 rows. One row is one method, dataset, damage condition and model.
- Each row has the untreated → treated score for the best single representation and the best hybrid.
- Fold 0. The best setting is picked on the held-out molecules.
- "Helped both" means both cells carry ✓. That is the page's "does it work?" without the control, which only some rows have.
- The groups and actions are the page's, taken from `_scripts/method_groups.json`. That is a dated reading of the code by Claude, not a fact.

### 4.1 By damage kind, small molecules only

"Helped both" over rows, all methods pooled.

| damage | helped both / rows | datasets |
|---|---|---|
| shift of a whole scaffold group | 150 / 213 | 4 |
| 5% outliers | 158 / 265 | 3 |
| Gaussian on every label | 128 / 233 | 3 |
| hERG label flips, 10% | 144 / 284 | 1 |
| 10% of labels shifted | 106 / 240 | 3 |
| clipped at the assay limit, 25% | 120 / 317 | 4 |
| simulated sources | 46 / 120 | 3 |
| labels as delivered | 136 / 457 | 5 |

Split by what says which molecules to act on. This is the split that changes the reading:
- **Clipped, 25%.**
  - A known flag, move the label: 38 of 38.
  - A detector, delete: 25 of 78. A detector, count less: 16 of 44. A detector, move the label: 10 of 46.
  - Noise-robust training: 25 of 87.
- **Scaffold-group shift.**
  - A known flag (the group), move the label: 18 of 18.
  - A detector, move the label: 15 of 18. A detector, count less: 18 of 24. A detector, delete: 40 of 65.
  - Noise-robust training: 50 of 72.
- **Labels as delivered.** Every route is at or below half:
  - A detector, delete: 32 of 104.
  - A detector, move: 11 of 70.
  - A known flag, move: 17 of 54.
  - Noise-robust training: 27 of 83.

Status against the outline:
- 7.7 says clipping is the most fixable and that Gaussian damage and outliers were fixed less often.
- Current rows, pooled over every method: CHANGED. Gaussian (128 of 233) and outliers (158 of 265) are
  helped in a larger share of rows than clipping (120 of 317).
- Restricted to flag-driven methods: SAME, and stronger. 38 of 38.
- So the statement must name the selection. It agrees with the outline's own paragraph "WHY THIS IS NOT JUST VOCABULARY".
- No control is counted in these rows.
- The QM9 rows with no noise added for the flag-driven lifts are n/a: QM9 has no clipped labels.

### 4.2 By action, per dataset ("helped both" / rows)

Read per dataset. Pooled, hERG dominates the "count less" count.

| dataset | a detector, delete | a detector, count less | a detector, move | a known flag, move | noise-robust training |
|---|---|---|---|---|---|
| caco2 | 86/175 | 50/106 | 37/108 | 37/64 | 94/202 |
| herg_ki | 78/154 | 39/94 | 39/96 | 22/56 | 82/173 |
| logd | 55/128 | 36/78 | 18/58 | 27/52 | 91/158 |
| herg (AUROC) | 58/153 | 93/104 | 11/16 | — | 3/70 |
| qm9 | 0/11 | 1/2 | 0/2 | 8/8 | 7/21 |
| ginkgo_colloidal | 27/53 | 29/58 | 32/74 | 12/48 | 53/88 |
| ginkgo_titer | 19/32 | 25/38 | 32/56 | 2/2 | 26/51 |
| ddg_combined | 13/49 | 41/58 | 31/48 | 0/12 | 27/88 |
| cdr_flexibility (AUROC) | 19/65 | 54/72 | 2/16 | — | 23/70 |
| ginkgo_acsins | 3/20 | 12/13 | 8/8 | — | 16/37 |
| humanness (AUROC) | 12/38 | 6/20 | 12/16 | — | 0/30 |
| buchwald_hartwig | 34/55 | 42/74 | 28/72 | 25/38 | 48/94 |

Patterns:
- **Classification datasets: noise-robust training almost never helps.**
  - hERG 3 of 70. Humanness 0 of 30. CDR flexibility 23 of 70.
  - These rows are the five network training methods of the twelve-method sweep (n09/n13), on dnn. Neither
    of the two models decided for these datasets is a neural network:
    - hERG: lightgbm and xgboost.
    - Humanness: xgboost and rf.
    - CDR flexibility: xgboost and lightgbm.
  - Example: meta_weight_net on hERG with labels as delivered. Single 0.831 → 0.811 AUROC, hybrid 0.842 → 0.846.
- **hERG under flips: down-weighting the molecules a detector flags (n21, n12) helped in 93 of 104 rows.**
  - Deleting them helped in 58 of 153.
  - Example: "best distance" with lightgbm, flips that depend on class. Single 0.822 → 0.840, hybrid 0.834 → 0.849 AUROC.
- **Regression small molecules: no action leads on all three of caco2, herg_ki and logd.**
  - Moving the label by a detector is the lowest on all three: 37 of 108, 39 of 96, 18 of 58.
- **Antibodies: down-weighting leads on ddg_combined, cdr_flexibility and ginkgo_acsins.**
  - Deleting is the weakest on those three: 13 of 49, 19 of 65, 3 of 20.
- **QM9 with labels that carry no assay error: detector deletion helped in 0 of 11 rows.**
  - That is the clean control behaving as it should.
  - The flag-driven moves' 8 of 8 on QM9 are under added clipping, not on clean labels.

The outline's open point: down-weighting has no home under "modifying or removing the label". In
these rows it is the action that most often helps on 4 of the 12 datasets above. That is a reason to give it a home.

### 4.3 By model: paired on the same method, dataset and damage

Same source. Pairs are (method, section, dataset, damage) where both models ran. "Works" here is
"helped both representations".

| domain | comparison | paired conditions | works, first model | works, second model |
|---|---|---|---|---|
| small molecules | dnn against svm | 711 (caco2, herg_ic50, herg_ki, logd) | 421 | 260 |
| — caco2 | | 273 | 171 | 95 |
| — herg_ki | | 242 | 136 | 88 |
| — logd | | 183 | 110 | 71 |
| — herg_ic50 | | 13 | 4 | 6 |
| antibodies | dnn against svm | 227 (ddg, thermostability, SMAC) | 122 | 110 |
| — ddg_combined | | 75 | 41 | 50 |
| — flab_thermo_tresanco | | 30 | 20 | 15 |
| — ginkgo_colloidal | | 122 | 61 | 45 |
| antibodies | dnn against rf | 62 (ginkgo_titer) | 30 | 51 |

Status against outline 7.11: "the network benefits from cleaning far more than the support vector regressor":
- SAME on caco2, herg_ki and logd.
- On herg_ic50, ddg_combined and ginkgo_titer the other model leads.
- The statement needs "on the small-molecule regression sets".

A caveat that may explain part of it:
- The regressor is deterministic, so a method with no effect leaves its score unchanged, which counts as "not helped".
- The network's untreated score moves between jobs. On QM9's second hybrid it is 0.3484, 0.3568 or 0.3618
  depending on the job. Source: PAGE QM9_CONTROL.md, also quoted in outline 7.4a. Not re-read raw.
- So some network "helps" may be the network's own run-to-run spread.
- The random-deletion control for the network on QM9 helped in only 9 of 90 settings (same page). So the
  spread does not explain most of it. I cannot separate the two from these files.

### 4.4 Does noise handling help the hybrid or the single representation more?

Same source. Rows where both cells have numbers.

| domain | rows | helped hybrid only | helped single only | hybrid ahead before / after treatment |
|---|---|---|---|---|
| QM9 control | 27 | 5 | 2 | 27 / 27 |
| small molecules | 2,245 | 281 | 249 | 2,033 / 2,100 |
| antibodies | 1,323 | 228 | 242 | 696 / 675 |
| reaction | 337 | 59 | 35 | 198 / 227 |

- On small molecules the hybrid is ahead before treatment in 2,033 of 2,245 rows and stays ahead.
- On antibodies it is ahead in about half before and after. That matches §2.
- Per dataset, the hybrid gains more than the single representation in a majority of rows on:
  - herg_ki: 317 of 619.
  - logd: 275 of 502.
  - ginkgo_titer: 131 of 177.
  - flab_thermo_tresanco: 39 of 54.
  - humanness: 63 of 102.
  - buchwald_hartwig: 193 of 337.
- In a minority on:
  - caco2: 281 of 707.
  - ginkgo_acsins: 26 of 77.
  - ginkgo_pr_cho: 20 of 56.
  - ddg_combined: 90 of 251.
- No domain-level pattern.

### 4.5 With a control: by dataset, labels as delivered against damage

Source: PAGE `answers/DETECTOR_BY_MITIGATION.md`, parsed by `parse_dbm.py` into 5,878
detector-and-mitigation combinations.
- Each combination reports "k/N improved" at its best setting. N is the model-and-configuration pairs, usually 8.
- Where a control exists, it also reports "beat <control> k/N".
- Controls: shuffled detector scores (n18), random deletion (n24), random molecules put back (n25).
- Fold 0.

Pairs that improved and also beat the control:

| domain | dataset | molecules | labels as delivered | under damage |
|---|---|---|---|---|
| small mol. | logd | 5,028 | 8 / 112 | 17 / 96 |
| small mol. | caco2 | 2,158 | 12 / 112 | 22 / 128 |
| small mol. | herg_ki | 1,412 | 58 / 288 | 114 / 456 |
| small mol. | herg (AUROC) | 7,772 | 17 / 80 | 160 / 192 |
| antibody | cdr_flexibility (AUROC) | 328 | 10 / 46 | — |
| antibody | humanness (AUROC) | 564 | 11 / 48 | — |
| antibody | abcov_kd | 141 | 27 / 96 | — |
| antibody | ginkgo_acsins | 242 | 27 / 88 | — |
| antibody | ddg_combined | 429 | 32 / 96 | — |
| antibody | ginkgo_titer | 239 | 38 / 96 | — |
| antibody | flab_thermo_tresanco | 672 | 42 / 96 | — |
| antibody | ginkgo_colloidal | 242 | 49 / 96 | — |
| reaction | buchwald_hartwig | 3,955 | 40 / 112 | 70 / 96 |

Patterns:
- **Labels as delivered.**
  - Small molecules: 7 to 21% of pairs.
  - Antibodies: 22 to 51%.
  - Within the small-molecule regression sets the smallest set has the highest share. herg_ki (1,412) is at
    20%, caco2 (2,158) at 11% and logd (5,028) at 7%.
  - Within antibodies there is no size order. abcov_kd, the smallest at 141, is at 28%. SMAC at 242 is at 51%.
  - The two antibody classification sets are lowest, 22% and 23%.
  - buchwald_hartwig is large (3,955) and still at 36%. So size alone does not carry it.
- **Under damage, the clearest controlled success is hERG flips: 160 of 192 pairs.**
- **No damage results with a control exist for any antibody set.**
- **Sampling resolution on the Ginkgo sets** (my arithmetic, not a test):
  - Fold 0 of ginkgo_colloidal holds 53 antibodies (c01 `held_out_rows_per_fold` = [53, 49, 48, 48, 44], VERIFIED raw).
  - A Spearman on about 50 molecules has a sampling spread of roughly 0.15. That is 1/sqrt(n-3) on the Fisher scale.
  - SMAC's 0.2740 → 0.4034 (n18, svm, base_1) is about one such spread, on one fold.
  - The shuffled control (0.2782) is the right answer to this. The full folds are the other answer.

### 4.6 Detector family

- **On SMAC and titer every detector works, so the detector is not what matters there.**
  - In n11, run only on ginkgo_colloidal and ginkgo_titer, all 14 detectors give at least half their pairs
    improved in 74 to 80 of their 75 to 80 combinations.
  - That includes conformal, ngboost, quantile forest, deep ensemble and the plain out-of-fold forest residual (cross_val).
  - n11 has no control. A result where every detector works equally is also what a detector carrying no information about the damage would give.
  - PAGE DETECTOR_BY_MITIGATION.md.
- **Small molecules, with a control.** Pairs improved and beat the control:

| experiment | detector | beat the control |
|---|---|---|
| n25 | neighbour gap damped by the neighbours' spread (activity_cliff_aware) | 64 / 168 |
| n25 | neighbour-label gap (neighbor_residual) | 51 / 168 |
| n25 | out-of-fold forest residual (cross_val) | 21 / 168 |
| n24 | neighbour-label gap | 155 / 480 |
| n24 | out-of-fold forest residual | 53 / 160 |
| n18 | neighbour-label gap | 28 / 144 |
| n18 | out-of-fold forest residual | 18 / 144 |

  Status against outline 7.6, "the out-of-fold forest residual and the neighbour-label gap drive almost every success":
  - Both are in nearly every experiment, so they would.
  - Where both ran side by side with a control on small molecules, the forest residual beat it less often
    in all three experiments.
  - The neighbour gap damped by the neighbours' spread did best in n25.
  - Status: CHANGED in emphasis. Pair the neighbour gap with the outline's claim. The forest residual is not its equal on small molecules.

- **Where neighbours are found** (n12, down-weighting). Pairs improved, at each combination's best setting. No control.

| dataset | neighbours in the pair's own single representation | neighbours in the first hybrid |
|---|---|---|
| caco2 | 40 / 112 | 220 / 530 |
| herg_ki | 19 / 54 | 149 / 278 |
| logd | 16 / 48 | 127 / 238 |
| herg | 42 / 52 | 226 / 246 |
| buchwald_hartwig | 37 / 54 | 223 / 296 |
| ginkgo_colloidal | 18 / 56 | 111 / 236 |
| flab_thermo_tresanco | 19 / 42 | 130 / 188 |
| abcov_kd | 33 / 64 | 144 / 320 |
| ddg_combined | 29 / 30 | 92 / 102 |
| ginkgo_titer | 28 / 30 | 115 / 145 |

  - The hybrid space gives the larger share on all 4 small-molecule sets.
  - Antibodies are split.
    - Hybrid space larger on ginkgo_colloidal, flab_thermo_tresanco and ginkgo_acsins.
    - About equal on cdr_flexibility, 86% against 87%.
    - Single space larger on ddg_combined (97% against 90%), ginkgo_titer (93% against 79%) and abcov_kd (52% against 45%).
  - Unequal comparison: hybrid-space rows apply to all 8 model-and-configuration pairs, and single-space rows only to that single representation's 2 pairs.
  - This is the nearest thing in the noise files to outline step 8, "a hybrid makes the noise methods better", tested on the detector rather than the model. It is worth a controlled re-run.
  - Scaling the distance by feature importance, against unscaled:
    - Small molecules: 814 of 1,358 against 884 of 1,538 pairs (60% against 57%).
    - Antibodies: 669 of 1,029 against 774 of 1,193 (65% against 65%).
    - No pattern.

---

## 5. The antibody exception: which datasets

Asked for in WHICH DATASETS ARE SHOWN (TODO: identify them). From the tables above.

**Labels as delivered, against a control** (§4.5):
- ginkgo_colloidal (SMAC): 49 of 96 pairs.
- flab_thermo_tresanco: 42 of 96.
- ginkgo_titer: 38 of 96.
- ddg_combined: 32 of 96.

**Without a control** (§4.2):
- ginkgo_pr_cho: noise-robust training helped both in 22 of 35 rows.
- ginkgo_acsins: down-weighting helped both in 12 of 13 rows.

**What they share:**
- Under 700 antibodies.
- Spearman metric.
- Folds of 44 to 135 held-out antibodies.
- The untreated fold-0 score is weak on several:
  - SMAC: median untreated single 0.097 Spearman across 86 rows.
  - ddg_combined: 0.341 across 64 rows.

**What argues against "the methods work better on antibodies" as a general claim:**
- The two antibody classification sets are among the lowest: cdr_flexibility 10 of 46, humanness 11 of 48.
- In n11, every detector works on SMAC and titer, which a detector with no information would also give.
- SMAC and HIC hold the same 242 antibodies. HIC has no cleaning results at all. NOISE_BY_GROUP has no
  ginkgo_hic rows, and DETECTOR_BY_MITIGATION lists no ginkgo_hic matrix. So "same molecules, opposite
  answer" (outline 7.8, 8.2a) currently rests on HIC's hybrid result, not on a cleaning result for HIC.
  Status for the cleaning half: NOT YET RUN.

---

## 6. Outline statements checked here

| outline | says | current | status |
|---|---|---|---|
| 5.4 | hybrid wins on Caco-2, LogD, hERG Ki, hERG, three ADME sets, both models | 78 of 80 fold-and-model pairs; table §2.1 (c02 raw, 5 folds) | SAME |
| 5.4 | LogD network 0.4497 → 0.4089; regressor 0.4930 → 0.4302 RMSE | same values, c02 `standard_selection_summary` (raw, 5 folds) | SAME |
| 5.4 | loses for both models on thermostability, SMAC, HIC | all six are losses; two by 0.0002 and 0.0005 Spearman | SAME, but two are ties at this resolution |
| 5.4 | hybrid beats both ensembles on HLM, hERG Ki svm, PAM50; both beat it on Ang2, abcov, PR_CHO | no adme_hlm or abagym_Ang2 file in `complexity/c11_ensemble_vs_hybrid/`; herg_ki svm, PAM50, abcov, PR_CHO agree (PAGE c11.md) | HLM and Ang2 NOT FOUND; rest SAME |
| 5.5 | beats PCA, mRMR, elimination on HLM, RLM, Caco-2, hERG Ki, LogD; none on Ang2, Buchwald, thermostability, PR_CHO | no adme_hlm, adme_rlm or abagym_Ang2 file in `complexity/c17_*`; Caco-2, hERG Ki, LogD yes; Buchwald, thermostability, PR_CHO no (PAGE c17.md) | HLM, RLM, Ang2 NOT FOUND; rest SAME |
| 5.4 | wider-than-training hybrids better on titer and SMAC, worse on hERG Ki and LogD | PAGE c33.md: titer yes/yes, colloidal yes/yes, herg_ki no/no, logd no/no | SAME |
| 7.7 | clipping most fixable; Gaussian and outliers fixed less often | pooled over every method, clipping is fixed less often than Gaussian or outliers; flag-driven only, 38 of 38 | CHANGED unless "by its flag" is said |
| 7.6 | forest residual and neighbour gap drive almost every success | with a control on small molecules the forest residual trails the neighbour gap in n18, n24, n25 | CHANGED in emphasis |
| 7.11 | network benefits far more than the regressor | paired: 421 vs 260 of 711 on small molecules; reversed on ddg_combined, herg_ic50; rf beats dnn on titer | SAME on small-molecule regression sets only |
| 7.8 | SMAC standout, 0.2740 → 0.4034 against 0.2782 shuffled | start and end values found in n18 raw, `folds_run: [0]`; shuffled value not re-read | SAME, fold 0 only |
| 7.8 / 8.2a | SMAC and HIC, same antibodies, opposite answer | HIC has no cleaning rows in either page | cleaning half NOT YET RUN |
| 5.3.1 / WHICH DATASETS | antibody exception datasets to identify | SMAC, thermostability, titer, ddg (with control); PR_CHO, AC-SINS (without) | NEW |

---

## 7. Speculation, kept apart

None of this is measured. Each item is a reading of the numbers above that the full folds could test.

- The hybrid wins where the pool holds near-equal representations of different kinds: small molecules, and
  ddg_combined with its mutation representation. It loses or ties where one language model dominates and the
  rest are its variants: thermostability (saprot), HIC (esm2_concat), SMAC (prot_t5). Adelaide's
  "more room" idea might hold if the antibody pools held distinct kinds: structure, physics-based or
  developability descriptors. Then it would be a statement about the pool, not the molecule.
- The antibody noise "exception" may be partly resolution. Small folds and weak untreated scores leave
  room for large gains in Spearman, and also for large chance gains. The shuffled-score control handles the
  second only partly, because it beats every one of 3 shuffles on one fold.
- The network's advantage over the regressor under cleaning may be partly the network's run-to-run spread (§4.3).
- Noise-robust training failing on the classification sets may reflect that it ran on a network those
  datasets do not otherwise use, rather than anything about classification.

## 8. What is missing for these patterns

- c02 summary for flab_binding_koenig and herg_ic50. c22 for logd, adme_rlm and flab_expression_koenig.
  c11 and c17 for adme_hlm (and adme_rlm, abagym_Ang2).
- Any noise result on adme_hlm, adme_rlm, adme_mdr1, abagym_*, flab_*_koenig, ginkgo_hic.
  So the noise patterns rest on 4 regression small-molecule sets, 1 classification small-molecule set, and 9 antibody sets.
- A control for any antibody damage condition.
- Folds 1 to 4 for every noise result.

## 9. Scripts (read-only on data)

All in `/Users/apunt/repos/KIRBy/scratchpad/thesis_ch5_8/patterns_scripts/`. Intermediate JSON in `/private/tmp/claude-501/`.
- `common.py`: domain list, from THESIS_OUTLINE.md 5.3.1, and the cut list.
- `c02_configs.py`, `c02_tab.py`, `c02_perds.py`: hybrid against single per fold, with the chosen configuration.
- `c01_pool.py`, `gap_vs_win.py`, `oracle.py`: pool properties, the gap among single representations, and the held-out-best single representation.
- `c22_blue.py`, `c22_family.py`: what the hybrids draw from.
- `parse_noise.py`, `noise_tab.py`, `noise_hyb.py`, `noise_start.py`, `action_ds.py`: NOISE_BY_GROUP rows.
- `parse_dbm.py`, `dbm_tab.py`, `dbm_tab2.py`, `n12_space.py`: DETECTOR_BY_MITIGATION combinations.
