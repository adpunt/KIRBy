# Ginkgo GDPa1 developability: sanity check for five datasets

Written 2026-10-09 by Claude (AI). This is a recommendation, not a measured result.

Datasets: ginkgo_colloidal (SMAC), ginkgo_hic, ginkgo_acsins, ginkgo_titer, ginkgo_pr_cho.

## Common to all five

### What the data is
- Source: Ginkgo Datapoints GDPa1 v1.2, file `GDPa1_v1.2_20250814.csv`, loaded by `src/kirby/datasets/ginkgo_dev.py`.
- The file has 246 rows. Each row is one therapeutic IgG: 106 approved, the rest in clinical trials or discontinued (read from the file's `highest_clinical_trial_asof_feb2025` column).
- One row is one distinct antibody against its own target. There are no mutants and no shared parent antibody.
- Her models see the heavy and light variable-domain protein sequences (`vh_protein_sequence`, `vl_protein_sequence`).
- Every assay was run on one platform (Ginkgo PROPHET-Ab). The CSV value is the median over replicates of the first production batch (Hugging Face dataset card, opened).
- Units are not stated in the CSV or on the dataset card. The units below come from the assay names and are unverified.

### How her folds are grouped
- Column `hierarchical_cluster_IgG_isotype_stratified_fold`: 5 folds of whole sequence clusters, balanced for IgG subclass.
- These are the competition's own folds. The Hugging Face card says to report on this column. The Ginkgo tutorial blog uses it for cross-validation.
- So her split is the split the leaderboard cross-validation numbers below were computed on.
- Her score is the mean of 5 per-fold Spearman values. The competition computes its cross-validation score the same way (tutorial blog, opened).
- Fold sizes for 242 antibodies: 53, 48, 48, 44, 49.

### Her scores: where they come from
- File: `/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid/<dataset>.json`, key `standard_selection_summary`.
- "Best single" is `base1`: the single representation picked on training antibodies in each fold. "Hybrid" is `hybrid1`. These match `CH5_RESULTS.md` §2.
- Spread is the sample standard deviation over 5 folds, plus the lowest and highest fold.

### Literature, shared by all five
| Source | What | Split | Supervised? | Status |
|---|---|---|---|---|
| van Niekerk et al. 2026, mAbs 18(1):2634216, doi:10.1080/19420862.2026.2634216 (PMC12928636) | Competition outcome. Best cross-validation score per property, and best score on a new blind set of 80 antibodies (GDPa3) | GDPa1 cluster folds (cross-validation); GDPa3 (blind test) | Supervised | OPENED |
| ginkgobioworks/abdev-benchmark README, github.com/ginkgobioworks/abdev-benchmark | 16 baseline models, cross-validation and blind test | "the defined folds in GDPa1" (column not named in the README) | Supervised, plus a random predictor | OPENED |
| Zhao, Moller, Quintero-Cadena, van Niekerk 2025, arXiv:2507.02670, Table 1 | ESM-2 embeddings (heavy and light, mean-pooled, concatenated) plus ridge regression, alpha 0.1 | 5 hierarchical-cluster folds by sequence identity | Supervised | OPENED |
| Channing, van Niekerk 2025, Hugging Face blog "How to Train an Antibody Developability Model" | p-IgGen embeddings plus ridge, HIC only | the same fold column | Supervised | OPENED |

Caveats on these numbers:
- The competition's top cross-validation scores come from teams that could see the folds. The paper says cross-validation scores "consistently exceeded held-out test performance". It also says the folds "did not appropriately emulate the differences between GDPa1 and GDPa3".
- The paper's abstract gives the blind-test winners as polyreactivity 0.356 and self-association 0.337. Its results text swaps these two numbers.
- The blind-test numbers are on 80 different antibodies. Her numbers are not comparable to them. Only the cross-validation column is comparable.
- The paper's conclusion: the models' predictive power "remains insufficient for reliable, decision-making fidelity". It calls HIC and Tm2 the more reliable properties.

### Red flags for all five
- 242 antibodies or fewer, so each held-out fold has 35 to 53 antibodies.
- With five groups, `objective='similarity'` (predicting each antibody from its five nearest neighbours) is undefined on these sets. They are listed in `SIMILARITY_UNDEFINED_OTHER` in `tests/experiments/dataset_policy.py`. c07 skips them for the same reason.
- `FAULTS.md` A2 (`/Volumes/seagate/claude_analysis_2026-10-08/answers/FAULTS.md`): every gp result on ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho comes from the old gp, which starts its kernel width at 1. The per-fold searches ran on 6 Oct. The fix is commit ffcd254, 8 Oct. I did not re-check the ledger dates.

---

## ginkgo_hic

### 1. What it is
- Endpoint: hydrophobic interaction chromatography retention time. A later retention means a more hydrophobic antibody, which is worse. Units probably minutes (unverified).
- n = 242 antibodies. Values run from 2.43 to 4.50; median 2.738. 139 distinct values.

### 2. Is the task useful
Hydrophobicity is screened early because hydrophobic antibodies tend to aggregate and to clear fast. Ranking candidates by predicted HIC before expressing them is a real triage use. HIC is one of the five scored competition properties. Both the competition paper and earlier work call it the most predictable of them.

### 3. Her scores (Spearman, 5 folds)
| Model | Best single, mean | per fold | SD | Hybrid, mean | per fold | SD | hybrid minus single |
|---|---|---|---|---|---|---|---|
| gp | 0.6000 | 0.544, 0.534, 0.634, 0.649, 0.639 | 0.056 | 0.5515 | 0.554, 0.534, 0.492, 0.505, 0.672 | 0.072 | -0.0486 |
| dnn | 0.5311 | 0.519, 0.498, 0.570, 0.606, 0.463 | 0.057 | 0.5293 | 0.502, 0.526, 0.400, 0.625, 0.594 | 0.088 | -0.0018 |

The best single representation is esm2_concat in all 5 folds, for both models.

### 4. Literature (Spearman, GDPa1 cluster folds, supervised)
| Source| Model | Spearman | What |
|---|---|---|
| Competition best, cross-validation | 0.758 | best of 113 teams, mAbs 2026 |
| moe_baseline | 0.656 | abdev-benchmark README |
| deepsp_ridge | 0.531 | abdev-benchmark README |
| ablang2_elastic_net | 0.461 | abdev-benchmark README |
| esm2_tap_ridge | 0.420 | abdev-benchmark README |
| esm2_ridge | 0.416 | abdev-benchmark README |
| ESM-2 plus ridge | 0.42 plus or minus 0.09 | arXiv:2507.02670 Table 1 |
| aggrescan3d | 0.404 | abdev-benchmark README |
| p-IgGen plus ridge | 0.324 | Hugging Face tutorial blog |
| onehot_ridge | 0.233 | abdev-benchmark README |
| random_predictor | 0.002 | abdev-benchmark README |
| Competition best, blind GDPa3 | 0.708 | not comparable: 80 different antibodies |

Her best single representation with gp, 0.600, is above every published baseline except moe_baseline (0.656), and below the best competition cross-validation score (0.758), on the same folds and the same metric.

### 5. Red flags
- 242 antibodies, 44 to 53 per held-out fold.
- The gp numbers are from the old gp (FAULTS A2). They are the stronger of her two models here.
- The hybrid loses to the single best representation with both models (-0.0486 gp, -0.0018 dnn).
- The best single representation is the same one (esm2_concat) in all 5 folds, so there is no representation-choice story here.
- `objective='similarity'` is undefined on this set.

### 6. Verdict
REPORT. Recognised benchmark, her split is the benchmark's own, her number sits inside the published range on a comparable split, and the hybrid losing with both models is a usable negative.

---

## ginkgo_acsins

### 1. What it is
- Endpoint: AC-SINS at pH 7.4, a self-association measure. Read as a plasmon wavelength shift in nm; a larger shift means more self-association, which is worse. Units unverified.
- n = 242. Values run from -1.875 to 29.5; median 1.625; 76 distinct values among 242 rows. 30 antibodies share the value -0.375.
- The dataset card changelog says AC-SINS was re-fit with better curve fitting in a later version.

### 2. Is the task useful
Self-association predicts high-concentration viscosity problems, which decide whether a subcutaneous formulation is possible. Ranking candidates on it before making concentrated material is a real decision. It is one of the five competition properties.

### 3. Her scores (Spearman, 5 folds)
| Model | Best single, mean | per fold | SD | Hybrid, mean | per fold | SD | hybrid minus single |
|---|---|---|---|---|---|---|---|
| gp | 0.5357 | 0.527, 0.545, 0.586, 0.335, 0.686 | 0.128 | 0.5547 | 0.593, 0.587, 0.576, 0.316, 0.702 | 0.143 | +0.0190 |
| lightgbm | 0.4631 | 0.437, 0.537, 0.439, 0.284, 0.618 | 0.125 | 0.5242 | 0.563, 0.686, 0.557, 0.263, 0.553 | 0.156 | +0.0611 |

Best single representation: ablang2 in 4 folds and igt5 in 1 (gp); igt5 in 4 and ablang2 in 1 (lightgbm).

### 4. Literature (Spearman, GDPa1 cluster folds, supervised)
| Model | Spearman | What |
|---|---|---|
| Competition best, cross-validation | 0.653 | mAbs 2026 |
| ablang2_elastic_net | 0.509 | abdev-benchmark README |
| ESM-2 plus ridge | 0.49 plus or minus 0.09 | arXiv:2507.02670 Table 1 |
| esm2_tap_ridge | 0.480 | abdev-benchmark README |
| moe_baseline | 0.424 | abdev-benchmark README |
| esm2_ridge | 0.420 | abdev-benchmark README |
| piggen | 0.388 | abdev-benchmark README |
| onehot_ridge | 0.230 | abdev-benchmark README |
| random_predictor | -0.026 | abdev-benchmark README |
| Competition best, blind GDPa3 | 0.337 or 0.356 | not comparable: 80 different antibodies |

Her hybrid with gp, 0.5547, is above every published baseline and below the best competition cross-validation score (0.653), on the same folds and metric.

### 5. Red flags
- The gp numbers, her stronger model, are from the old gp (FAULTS A2).
- Fold 3 is much lower than the other four in every row above (0.26 to 0.34 against 0.53 to 0.70). The mean rests on four folds.
- 76 distinct label values among 242 antibodies, so many ties; 30 antibodies share one value. Spearman handles ties, but the ceiling is lowered.
- `objective='similarity'` is undefined on this set.

### 6. Verdict
REPORT WITH CAVEAT. The hybrid wins with both models and the numbers are inside the published range, but the gp half has to be re-run under the new gp before the number is printed.

---

## ginkgo_titer

### 1. What it is
- Endpoint: expression titer, measured by Valita assay. Larger is better. Units probably mg/L (unverified).
- n = 239 (3 of 242 have no titer value). Values run from 34.26 to 781.93; median 224.95; 148 distinct values.

### 2. Is the task useful
Titer decides whether a candidate can be made at all, and at what cost. A model that ranks candidates by expected titer from sequence alone would be used, but titer depends heavily on the cell line and process, not only on the sequence. It is one of the five competition properties, and the one the competition paper's own models did worst on in the blind test.

### 3. Her scores (Spearman, 5 folds)
| Model | Best single, mean | per fold | SD | Hybrid, mean | per fold | SD | hybrid minus single |
|---|---|---|---|---|---|---|---|
| dnn | 0.3083 | 0.386, 0.202, 0.501, 0.281, 0.170 | 0.136 | 0.3668 | 0.226, 0.420, 0.524, 0.404, 0.260 | 0.123 | +0.0585 |
| rf | 0.2504 | 0.309, 0.288, 0.430, 0.021, 0.204 | 0.151 | 0.2706 | 0.262, 0.247, 0.521, 0.100, 0.224 | 0.154 | +0.0202 |

Best single representation (dnn): ablang2, igt5, ablang2, esm3, esm2_entropy_top20 across the 5 folds. A different representation wins in several folds.

### 4. Literature (Spearman, GDPa1 cluster folds, supervised)
| Model | Spearman | What |
|---|---|---|
| Competition best, cross-validation | 0.640 | mAbs 2026 |
| ablang2_elastic_net | 0.356 | abdev-benchmark README |
| piggen | 0.238 | abdev-benchmark README |
| esm2_tap_rf | 0.223 | abdev-benchmark README |
| esm2_tap_ridge | 0.221 | abdev-benchmark README |
| antifold | 0.194 | abdev-benchmark README |
| onehot_ridge | 0.193 | abdev-benchmark README |
| moe_baseline | 0.184 | abdev-benchmark README |
| esm2_ridge | 0.180 | abdev-benchmark README |
| ESM-2 plus ridge | 0.17 plus or minus 0.14 | arXiv:2507.02670 Table 1 |
| random_predictor | 0.068 | abdev-benchmark README |
| Competition best, blind GDPa3 | 0.310 | not comparable: 80 different antibodies |

Her hybrid with dnn, 0.3668, is above every published baseline and far below the best competition cross-validation score (0.640), on the same folds and metric.

### 5. Red flags
- rf fold 3 is 0.021 for the single best representation and 0.100 for the hybrid: that fold carries almost no signal.
- The spread across folds (SD 0.12 to 0.15) is about a third of the mean.
- No gp here, so FAULTS A2 does not apply.
- `objective='similarity'` is undefined on this set.

### 6. Verdict
REPORT. The largest hybrid win in her antibody panel (+0.0585 dnn, 4 of 5 folds), the single best representation changes between folds, and no gp fault touches it. The low absolute level is normal for titer in this benchmark.

---

## ginkgo_pr_cho

### 1. What it is
- Endpoint: polyreactivity score against CHO soluble membrane protein. Larger means more non-specific binding, which is worse. Dimensionless score (unverified).
- n = 197 of 246 rows have a value. Values run from 0.000 to 0.547; median 0.124.
- 58 of the 197 values are exactly 0.000, the floor of the assay. That is 29 percent of the rows tied at the bottom.
- Fold sizes: 45, 40, 36, 35, 41.
- The dataset card changelog says the polyreactivity score calculation was corrected in a later version.

### 2. Is the task useful
Polyreactivity predicts off-target binding, fast clearance and safety findings, and it is screened before candidate selection. Ranking candidates on it from sequence would be used. It is one of the five competition properties.

### 3. Her scores (Spearman, 5 folds)
| Model | Best single, mean | per fold | SD | Hybrid, mean | per fold | SD | hybrid minus single |
|---|---|---|---|---|---|---|---|
| gp | 0.5009 | 0.447, 0.575, 0.529, 0.502, 0.451 | 0.054 | 0.4419 | 0.420, 0.432, 0.532, 0.509, 0.315 | 0.086 | -0.0590 |
| dnn | 0.4003 | 0.389, 0.445, 0.485, 0.382, 0.301 | 0.070 | 0.4133 | 0.423, 0.457, 0.520, 0.351, 0.316 | 0.082 | +0.0130 |

Best single representation (dnn): prot_t5 in 3 folds, onehot in 1, esmc in 1. One fold is won by one-hot encoding.

### 4. Literature (Spearman, GDPa1 cluster folds, supervised)
| Model | Spearman | What |
|---|---|---|
| Competition best, cross-validation | 0.559 | mAbs 2026 |
| piggen | 0.424 | abdev-benchmark README |
| esm2_ridge | 0.420 | abdev-benchmark README |
| esm2_tap_ridge | 0.413 | abdev-benchmark README |
| ESM-2 plus ridge | 0.41 plus or minus 0.09 | arXiv:2507.02670 Table 1, CHO PR score |
| ablang2_elastic_net | 0.362 | abdev-benchmark README |
| moe_baseline | 0.353 | abdev-benchmark README |
| saprot_vh | 0.289 | abdev-benchmark README |
| onehot_ridge | 0.204 | abdev-benchmark README |
| aggrescan3d | 0.112 | abdev-benchmark README |
| random_predictor | -0.081 | abdev-benchmark README |
| Competition best, blind GDPa3 | 0.356 or 0.337 | not comparable: 80 different antibodies |

Her best single representation with gp, 0.5009, is above every published baseline and below the best competition cross-validation score (0.559). Her dnn number, 0.4003, is in line with the ESM-2-plus-ridge baselines (0.41 to 0.42).

### 5. Red flags
- The gp numbers are from the old gp (FAULTS A2), and gp is the stronger model here.
- The hybrid loses with gp (-0.0590) and wins slightly with dnn (+0.0130), so the direction depends on the model.
- 197 antibodies, 35 to 45 per held-out fold: the smallest of the five.
- 58 of 197 labels are exactly at the assay floor.
- One fold is won by one-hot encoding, which says the language-model representations add nothing there.
- `objective='similarity'` is undefined on this set.

### 6. Verdict
REPORT WITH CAVEAT. The number is competitive on the benchmark's own folds, but re-run the gp half first, and state the 58 tied-at-zero labels and the smaller n.

---

## ginkgo_colloidal (SMAC)

### 1. What it is
- Endpoint: standup monolayer adsorption chromatography retention time, a colloidal-stability measure. A later retention means worse colloidal behaviour. Units probably minutes (unverified).
- n = 242. Values run from 2.615 to 10.0; median 2.75. The 10th to 90th percentile band is 2.70 to 3.21, so almost all antibodies sit in a 0.5-wide window with a few far above it. 86 distinct values among 242 rows; 15 antibodies share 2.715.
- The maximum, 10.0, is held by 2 antibodies and looks like a cap rather than a measurement. Unverified.

### 2. Is the task useful
SMAC is a colloidal-stability screen used as an early flag for aggregation and viscosity risk. The use would be the same as AC-SINS. SMAC was not one of the five scored competition properties, so the field has little published prediction work on it, and the one published cross-validation number on it is near zero (below).

### 3. Her scores (Spearman, 5 folds)
| Model | Best single, mean | per fold | SD | Hybrid, mean | per fold | SD | hybrid minus single |
|---|---|---|---|---|---|---|---|
| svm | 0.3485 | 0.270, 0.338, 0.390, 0.509, 0.236 | 0.108 | 0.3480 | 0.171, 0.317, 0.413, 0.545, 0.295 | 0.140 | -0.0005 |
| dnn | 0.2766 | 0.067, 0.197, 0.249, 0.442, 0.428 | 0.159 | 0.2677 | 0.291, 0.027, 0.407, 0.358, 0.255 | 0.147 | -0.0088 |

Best single representation (svm): prot_t5, prot_t5, esm2_early, esm2_concat, esm2_concat.

### 4. Literature (Spearman, GDPa1 cluster folds, supervised)
| Model | Spearman | What |
|---|---|---|
| ESM-2 plus ridge | 0.18 plus or minus 0.17, Pearson 0.00 plus or minus 0.14 | arXiv:2507.02670 Table 1, SMAC RT |

No other comparable number exists: the competition did not score SMAC, so the mAbs paper and the abdev-benchmark README have no SMAC column.

Her svm number, 0.3485, is above the one comparable published number (0.18 plus or minus 0.17), on 5 cluster folds and the same metric. One number from one preprint is a thin anchor.

### 5. Red flags
- One comparable literature number only, and its own spread (plus or minus 0.17) covers her value.
- dnn fold 0 is 0.067 for the single best representation and 0.027 for the hybrid: that fold is near zero.
- Labels are compressed: 80 percent of antibodies lie in a 0.5-wide band, with a handful of outliers that will dominate a rank correlation.
- The hybrid loses with both models, though by less than 0.01 in both.
- Her svm fold spread is 0.236 to 0.509 around a mean of 0.3485.
- `objective='similarity'` is undefined on this set.
- This set carries several of her other findings: it is one of two datasets where c21 ran extra features, and the dataset where scoring candidates with the downstream model itself wins (c23), with the largest c24 effect.

### 6. Verdict
REPORT WITH CAVEAT. The endpoint is real and her number is above the only published number, but SMAC is outside the competition's scored five, so the comparison rests on one preprint row. If it is reported, say that.
