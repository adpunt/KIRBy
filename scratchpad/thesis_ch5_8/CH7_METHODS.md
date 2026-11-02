# Chapter 7 methods: what the code does (outline 7.2, 7.3, 7.4, 7.4a, and NAMES FOR THE NOISE METHODS)

Written by Claude on 2026-10-09 from the code and the raw result files. A reading, not a review by the author.

Conventions used below.
- VERIFIED means I read it in this session, in the code or in a raw result file, at the path given.
- UNVERIFIED means I saw it only in a document (the outline, an answers page, METHOD_GROUPS.md, METHOD_PROVENANCE.md).
- Every result number is outer fold 0 unless the line says otherwise.
- Experiment codes appear only as file pointers. The thesis uses none of them.
- Paths are shortened: `NE` = src/kirby/utils/noise_estimation.py; `NS` = src/kirby/noise_spec.py; `NH` = tests/experiments/noise/noise_helpers.py; `L` = src/kirby/utils/lnl_methods.py; `NI` = ~/repos/NoiseInject/noiseInject/core.py; `NIR` = ~/repos/NoiseInject/noiseInject/relative.py; `nXX` = tests/experiments/noise/nXX_*.py; `R` = /Volumes/seagate/kirby_results_2026-10-09.

---------------------------------------------------------------------------------------------------

## 0. What changes the outline, in order of how much it matters

1. **7.2 lists the wrong damage kinds.** The outline names seven regression kinds. The code's decided grid has six, and two of the outline's seven are described wrongly. (Section 1.1.)
   - "Outliers in 5 per cent of molecules" is wrong. Under `outlier_p05` EVERY training label gets Gaussian error; 5 per cent get it at 3 times the scale. (NI:65-67, NI:491-502.)
   - "A sparse spread increase in 10 per cent" is wrong. `sparse_dprime_up_p10` moves 10 per cent of labels UP by one fixed amount. The amount is the level times the spread of a forest model's out-of-fold residual. (NIR:16-18, NIR:176-190.) It is a shift, like `shift_only_p10`, in a different unit.
   - "A wider spread per scaffold group" (`grouped_wider`) is not in the decided grid at all (NH:2118-2120). No result file in R/noise has it as a condition.
2. **Three more damage kinds appear in the results and the outline never names them.** `outlier_only_p05` (n31), `dense_oofmse` (n40), and simulated multi-source labels with correlated errors (n39). Plus three no-injection regimes: labels as delivered, real repeat disagreement (herg_ic50), and no damage at all. (Section 1.4.)
3. **The per-fold search that picks the base representations and the hybrid recipe sees the CLEAN training labels in most experiments.** Only the hybrid's features are then rebuilt on the damaged labels. So the outline's caveat "the hybrid is rebuilt from that draw's damaged labels" is half true. The choice of which representations and which recipe is made with clean labels. (NH:2459-2461, n07:3155-3156, n07:3230-3235.) n09 and n20 are the exceptions.
4. **No generic control has run anywhere.** The random and shuffled twins in tests/experiments/noise/controls.py are switched off unless KIRBY_NOISE_CONTROLS=1 (controls.py:48-49). Every JSON that records it says `controls_run: false` (108 files across 13 experiments). The controls that DID run are the four built into single experiments. (Section 3.4.)
5. **One damage draw can never succeed.** KIRBy turns noiseInject's dose warning into an error (NH:476-486). The seed is fixed per condition and draw, so a retry draws the same labels and fails again. This killed flab_thermo_tresanco n12 under Gaussian damage twice: delivered 9.380351 against a target of 10.247953 (R/LEDGER.md, "Failed units"). noiseInject itself says about 1 per cent of draws land outside its band by chance (NI:665-700).
6. **METHOD_GROUPS.md has wrong rows and is incomplete.** Confident learning does not use out-of-fold probabilities. The n23 rows say "delete" but n23 also moves labels. The n20 row says "delete" but n20 runs nine cleanups. DivideMix, prox_smooth and at least 25 other methods with results are missing. (Section 3.2.)
7. **QM9_CONTROL.md files n27's removal_cost as "a detector, delete".** It is random deletion with no detector (n27:17-24). The fault is in answers/_scripts/qm9_control.py:52. (Section 5.)
8. **Several outline methods have no results.** `quantile_clip` (cut 2026-10-05), `prox_smooth` in n07, `huber_weight` in n07, and every robust loss in n33 except the network's Huber loss. Tukey, Student-t (except one NGBoost pair), expectile, fair and absolute-error losses ran on no dataset. (Section 4.2.)
9. **QM9 with no noise added: almost every method makes the plain model worse.** 10 of 12 noise-robust training methods are worse than the plain network on all five feature spaces in one QM9 job (section 5.3). The outline's QM9 counts in 7.4a are SAME.
10. **The "every number is fold 0" caveat is out of date.** Eight experiments now have 3 or 5 outer folds on some datasets, and promoted runs have folds 1 and 2 for parts of n07, n20, n23, n24, n25 and n30. (Section 6.)

---------------------------------------------------------------------------------------------------

## 1. Outline 7.2: what the noise is

### 1.1 The damage kinds the code sweeps

The decided list is `DECIDED_REGRESSION_CONDITIONS` at NH:2118-2120 (VERIFIED):
`gaussian`, `outlier_p05`, `grouped_shifted`, `censoring`, `shift_only_p10`, `sparse_dprime_up_p10`.
The one-pass screen leaves out `sparse_dprime_up_p10` (NH:2147). A promoted run sweeps all six (NH:1103-1105).
A classification dataset gets the six flip kinds instead (NH:2186-2191; NS:134-135).

One row per kind. "Level" is what the number on the axis means. ε is the error added to one training label. s is the scale the injector solves for so that the root-mean-square of ε over the training molecules equals the target (NS:560-580). SD_tr is the standard deviation of the fold's clean training labels (NS:546-556). All VERIFIED in code.

| kind (code name) | what is done to the training labels | level means | share of training molecules whose label moves | where |
|---|---|---|---|---|
| Gaussian (`gaussian`) | ε_i ~ N(0, s²) for every molecule | target RMS of ε, as a fraction of SD_tr | 100% | NI:58, NI:476-484, NI:659 |
| Outliers in 5% (`outlier_p05`) | every molecule gets N(0, s²); a random 5% get N(0, (3s)²) instead. RMS of ε over all molecules = level × SD_tr, so s = level × SD_tr / sqrt(0.05×9 + 0.95) | fraction of SD_tr | 100% move; about 5% at 3× the scale. The 5% carry 0.45/1.40 = 32% of the squared error | NI:66, NI:491-502 |
| Offset per scaffold group (`grouped_shifted`) | ε_i = s·(sqrt(0.62)·b_g + sqrt(0.38)·e_i). b_g is one N(0,1) draw per Murcko scaffold group; e_i is one per molecule. Offsets are not centred | fraction of SD_tr | 100% | NI:64, NI:711-734 |
| Clipping at an assay limit (`censoring`) | every training label above the (1 − level) quantile of the clean training labels is set to that quantile | share of training labels clipped | about the level; coarse labels with ties at the limit can land a few points off (LogD at 10% asked: 8.57% moved) | NI:736-781, NI:782-800, NS:538-546 |
| Shift in 10% (`shift_only_p10`) | 10% of molecules, drawn over the whole dataset, have the label raised by level × SD_tr. Up only | the shift, in units of SD_tr | about 10% of the training molecules | NH:564-565, NH:629-675 |
| Shift in 10%, sized by model error (`sparse_dprime_up_p10`) | 10% of molecules have the label raised by level × σ_r. σ_r is the SD of a LightGBM out-of-fold residual on the clean training labels, on the dataset's first representation. Up only | the shift, in units of σ_r | about 10% | NIR:16-18, NIR:176-190, NH:690-711, NH:2253-2268 |

Scaffold groups for the grouped kind come from `assign_noise_groups` (NS:300-341): Murcko scaffolds, with every acyclic molecule its own group. A dataset with no SMILES (antibodies, proteins, reactions, expression) has no groups, and `grouped_shifted` is dropped there (tests/experiments/dataset_policy.py:531-552). VERIFIED.

### 1.2 Classification flip kinds (herg, humanness_therapeutic, cdr_flexibility, tcga_brca_pam50)

The level is the target share of training labels flipped. A flip probability is solved on the training labels to hit it (NS:630-636; noiseInject `calibrate_flip_probability`). At level 0 nothing is flipped; the labels are as delivered (NS:624-633, decided 2026-10-05). VERIFIED in NI:1055-1258.

| flip kind | rule |
|---|---|
| `uniform` | every label flips with probability p to a random other class |
| `class_imbalance` | classes rarer than the threshold flip at 2p, commoner ones at 0.5p |
| `binary_asymmetric` | class 0 → 1 at 1.5p, class 1 → 0 at 0.5p. Raises on more than two classes |
| `instance_noise` | each molecule's p is multiplied by a N(1, 0.3) draw clipped to 0.1-3.0 |
| `class_dependent` | each class flips at p × (its frequency / the largest frequency) × 2. So the commonest class flips most |
| `confusion_directed` | with no confusion matrix given, the default is uniform off-diagonal: the same distribution as `uniform` |

Two consequences for the text.
- On a two-class dataset, `confusion_directed` with its default matrix is the same mechanism as `uniform`, drawn separately. The outline counts "six label-flip kinds"; five are distinct. VERIFIED (NI:1229-1247).
- `binary_asymmetric` cannot run on tcga_brca_pam50, which has more than two classes (NI:1123-1125). VERIFIED in code; I did not check what the pam50 result files record for it.

### 1.3 Levels and draws

Grid in code (NH:2129-2154, VERIFIED):
- Regression, promoted run: 0, 0.5, 1.0, 1.5 on datasets of 5,000 molecules or fewer; 0 and 1.0 above. Censoring: 0, 0.10, 0.25, 0.40, or 0 and 0.25.
- Regression, screen: 0 and 1.0; censoring 0 and 0.25.
- Classification: screen 0 and 0.10; promoted 0, 0.05, 0.10, 0.20, 0.30, 0.40 (dataset_policy.py:471).
- `sparse_dprime_up_p10` in n38 uses its own grid, 1, 2 and 4 residual SDs (NH:585-586).
- QM9 has 9,978 molecules, so it is in the larger group and gets 0 and 1.0 only, even when promoted (R/noise/n30_censored_likelihood/qm9.json, field `knn_group`: `fast_knn`).

Levels actually in the result files are mixed. caco2, herg_ki and herg carry the full grid in n07, n23, n24, n25, n30 and n31. Most other datasets carry 0 and 1.0. Some files carry one non-zero level only (n13 everywhere; n09 buchwald_hartwig has 1.0 and no level 0). VERIFIED by reading the `level` field of every row file under R/noise.

Draws (VERIFIED):
- Three draws per non-zero level, seeds 0, 1, 2 (NH:236, n07:173, n09:117).
- One draw under censoring, because the clip reads no seed (NH:2561-2572).
- Level 0 runs once.
- The seed is (42 × 1000003 + crc32(kind) + draw × 15485863) mod 2^31 (NH:294-309). It holds neither the level nor the fold.
- Because the level is not in the seed, levels nest: the error at level 1.0 is exactly twice the error at 0.5 on every molecule (NH:363-388).
- Which molecules are hit is drawn over the whole dataset, then the training rows are taken. A molecule has the same damage in every fold it trains in (NS:493-501).

### 1.4 Damage kinds in the results that 7.2 does not list

Read from the `noise_condition` field of every row file in R/noise (VERIFIED):

| condition | experiment | what it is |
|---|---|---|
| `outlier_only_p05` | n31 | the 5% get 3× the scale and every other label is left alone (NI:85-87) |
| `dense_oofmse` | n40 | every label gets N(0, level × σ_r²); n40 levels 0, 0.1, 0.3, 1.0 (n40:13-21; NIR:14-15). JSON only, in R/noise/n40_model_side_regularisation/*__dense_oofmse.json |
| `herg_ic50|rho…|tau…`, `uniform_1_to_4|rho…|tau…` | n39 | simulated several sources per molecule with correlated errors (rho) and a bias per source (tau) |
| `as_they_come` | n18, n20, n23 | labels as delivered, nothing injected (n07:427-436) |
| `real_replicate_disagreement` | n17, n19, n36 | herg_ic50's real repeat measurements |
| `none` | n02, n27, n29, n34 | no damage |

### 1.5 Only training labels are damaged: VERIFIED, with one qualification

- `inject_train_labels` returns training rows only (NH:399-497). Held-out rows are never read.
- n07 grades every fit against `labels[val_idx]`, the delivered labels (n07:3164).
- n27 says the same in its header (n27:51-53).
- Qualification. Where a per-fold search runs ON damaged labels (n09, n15, n20), the inner folds of that search are scored on damaged labels too. n09 says so explicitly (n09:125-133). The outer held-out fold stays clean.
- `TrainingNoise.held_out` exists to damage validation labels (NS:648-728). Its only caller is `find_best_hybrid._noisy_held_out` (find_best_hybrid.py:505-514). I found no experiment that scores the outer fold on damaged labels.

### 1.6 Is the damage applied before the features are chosen?

Three different answers, by experiment. VERIFIED in code.

1. **Most experiments (n07, n11, n12, n13, n18, n21, n23, n24, n25, n27, n30–n41, c10).** The per-fold search runs on CLEAN training labels: `fold_configurations` is called with `label_noise=None` (NH:2459-2461, marked "OPEN (handoff 2.12)"). It picks the two base representations and the two hybrid recipes. The hybrid's features are then re-ranked and the budget re-split on the DAMAGED labels (n07:3230-3235, `y_train=y_noisy`). So the hybrid's features move with the damage, but the representation choice and the recipe were made with clean labels.
2. **n09.** One per-fold search per (kind, level, draw, fold), trained and scored on the damaged labels (n09:118-140). Nothing clean is in view. Its two neighbour spaces still come from the clean search (n09:545-547).
3. **n15 and n20.** The comparison is the point: clean-searched against damaged-searched recipe (n15), and cleaned labels before against after the search (n20, four arms, n20:12-19).

n07 also keeps an oracle copy of each hybrid built from clean-label features, labelled `@clean_label_features` in the rows (R/noise/n07_remediation/qm9__*.rows.jsonl, field `model_input`).

---------------------------------------------------------------------------------------------------

## 2. Outline 7.3: the detectors

### 2.1 What each detector reads and computes

"Reads own label" means the molecule's own label enters its own score. Every class is in NE unless noted. VERIFIED by reading each `estimate` method.

| detector (code name) | where | score | reads own label? | what is swept |
|---|---|---|---|---|
| Out-of-fold forest residual (`cross_val`) | NE:635-725 | regression: abs(y − out-of-fold random-forest prediction) / SD(y). Classification: 1 − out-of-fold probability of the recorded class. 100 trees, 5 scaffold-grouped folds | yes | nothing. The default detector of n07, n20 and half of n18 and n24 (n07:550, n20:483) |
| Neighbour-label gap (`neighbor_residual`) | NE:361-422 | abs(y − mean of the k nearest neighbours' labels) / SD(y). Classification: share of neighbours with another class | yes | k 5, 10 or 20; rank or 1/d² weighting; neighbour space. Picked per fold by a screen (n07:1906-1981) |
| Gap divided by the neighbours' own spread (`local_variance_ratio`) | NE:425-497 | abs(y − neighbour mean) / max(neighbour SD, 0.01 SD(y)) | yes | as above |
| Cliff-aware gap (`activity_cliff_aware`) | NE:557-628 | (abs(y − neighbour mean)/SD(y)) × 1/(1 + neighbour SD/SD(y)) | yes | as above |
| Out-of-bag residual (`leave_one_out`) | NE:780-843 | abs(y − out-of-bag forest prediction) / SD(y), 200 trees | yes | nothing |
| Gaussian process (`gaussian_process`) | NE:914-1080 | out-of-fold posterior SD + abs(out-of-fold residual). Each inner training fold capped at 1,000 molecules by random subsample (NE:986-993) | yes (through the residual) | kernel RBF or Matern; not swept in the runs I read |
| Heteroscedastic Gaussian process (`hetero_gp`) | NE:1838-2021 | the label-noise half of the predicted spread, out of fold | no; predicted from features | nothing |
| Variance-head network (`heteroscedastic_nn`) | NE:1083-1201 | predicted SD, β-NLL loss, out of fold | no | nothing |
| Natural-gradient boosting (`ngboost`) | NE:1704-1834 | predicted SD, 50 stages | no | nothing |
| Shapley value to a neighbour predictor (`knn_shapley`, n32) | n32:995-1075 | exact recursion; utility = minus the mean squared label gap over the k nearest | yes | neighbour space and k |
| Conformal p-values with Benjamini-Hochberg (n41) | n41:248-273, n41:275-290 | score = abs(y − p)/σ̂, σ̂ a second out-of-fold LightGBM fitted to abs(y − p). p-values against a calibration set; flags at q 0.05, 0.1, 0.2 | yes | calibration share 0.2 or 0.5; random, ORACLE or repeat-based calibration (n41:292-349) |
| Isolation forest, LOF (n41) | n41:388-400 | on the features joined to the z-scored residual times sqrt(number of features) | yes, through the appended residual | — |
| cleanlab regression score (n32, n41) | NH:762-779 | cleanlab package; its internal folds are random, not scaffold-grouped | yes | — |
| Labels at the assay limit (n30) | n30:293-300 | the largest training label, if at least 2 molecules and 1% of them sit on it | yes, but reads no model and no neighbour | — |

Which read only the features. Three: the heteroscedastic Gaussian process, the variance-head network and natural-gradient boosting. Each predicts a spread from a molecule's features with a model that never saw that molecule. The code says they can only rank molecules when the damage is tied to structure (n23:42-56). Under `grouped_shifted` every molecule moves, so no detection number exists there either. The n23 caco2 JSON records an empty `detection` block and says so (R/noise/n23_uncertainty_cleaning/caco2__grouped_shifted.json, field `sources_defined_note`). So 7.6's "work only when the damage is tied to structure, which is the per-scaffold-group offset" must mean the cleaning result, not detection.

### 2.2 The outline's CITATIONS block, line by line

Status: CORRECT means the code matches what the outline says. Line numbers checked this session.

| outline line | status | what the code shows |
|---|---|---|
| "Every prediction a detector reads is out of fold, with scaffold groups kept together" (NE:141-197; core.py:135-186) | **WRONG FOR FOUR** | Out-of-bag residual: plain out-of-bag, no scaffold grouping. n07 hands neighbour detectors `exclude_groups=None` (n07:606-607), so `leave_one_out` takes the out-of-bag path (NE:826-835). cleanlab: random folds (NH:762-779). Confident learning and small-loss selection in n09: in-sample predictions (L:688-735, L:600-636). Gaussian process: each inner fold subsampled to 1,000 at random (NE:986-993) |
| Neighbour detectors work in a screened space, k 5/10/20, rank or 1/d² (n07:1906-1981) | CORRECT, with an exception | Rows carry k 5, 10, 20 and both schemes in n02, n12, n18, n21, n24, n25. n32's Shapley value is outside the weighting axis by decision (n32:1115-1125). n07's default detector is the forest residual, not a neighbour detector, so the screen picks the neighbours for the label moves, not for the detector |
| Out-of-fold forest residual (NE:635-725), Brodley1999 / Northcutt2021 | CORRECT | lines match; no citation in code |
| Out-of-bag residual (NE:780-843), "the code calls it leave-one-out" | CORRECT | registry name `leave_one_out` (NE:2040) |
| Neighbour-label gap (NE:361-497) | CORRECT | — |
| Cliff-aware: "docstring formula at :565 differs from the code at :621" | CORRECT, line off by 2 | docstring NE:565 says deviation × (1 − local variance); code NE:623 computes deviation × 1/(1 + neighbour SD/SD(y)) |
| GP (NE:914-1080): "score is the posterior spread plus the absolute residual" | CORRECT | NE:1069-1070. Add the 1,000-molecule cap |
| Heteroscedastic GP (NE:1838-2021) ported from qsar_qm_models | CORRECT | — |
| Variance-head network: Seitzer2022 β-NLL; NGBoost at 50 stages | CORRECT | NE:1775 `n_estimators=50` |
| Conformal + BH (n41:248-360); calibration on 3+ consistent repeats (n41:322-348) | CORRECT | Also: an ORACLE calibration drawn only from molecules the injection did not move (n41:306-319). Its rows are named `remove_oracle_clean_*` and `winsorise_oracle_clean_*`. Not a method a user can run; must not be reported as one |
| "ACS Omega 2024" at NE:1213 | CORRECT | NE:1213 |
| Shapley (n32:995-1075): "comment at n32:1012-1015 wrongly says Jia's recursion needs that" | CORRECT | n32:1011-1016 says the mean's squared error "is not a sum over neighbours, and Jia et al.'s exact recursion needs a sum" |
| cleanlab (NH:762-779) | CORRECT | — |
| Assay limit (n30:293-311) | CORRECT | rule at n30:290-300 |

---------------------------------------------------------------------------------------------------

## 3. Outline 7.4: the noise mitigation methods

### 3.1 Every method with results, as detector × action × target

The outline's scheme: what says which molecules (detector, known flag, every molecule), which action (delete, count less, move the label), and for a move, the target and how far. Built from the code and from the `method` field of every row file in R/noise. VERIFIED unless marked.

**Delete**

| method name in files | what selects | rule | experiments with rows |
|---|---|---|---|
| `remove` | detector (forest residual by default; neighbour gap in n18, n24) | drop the top 5, 10 or 20% of suspicion scores (n07:271, n07:957-964) | n07, n17, n18, n20, n24, n25, n31, n42 |
| `drop@share` | detector (a model's predicted spread, or the cliff-aware gap) | drop the top share | n23, n42 |
| `drop_clipped` | known flag (label at the training maximum) | drop those molecules | n30 |
| `small_loss` | detector (training loss after 10 warm-up epochs, in sample) | keep the 70% with the smallest loss, refit once | n09, n13 |
| `confident_learning` | detector (3-network ensemble, in sample) | drop molecules with error above the 85th percentile AND ensemble SD below the 25th | n09, n13 |
| `distance_selection` | detector (neighbour gap) | drop the top 15% | n09, n13 |
| `multistage_cleaning` | detector (loss) | drop the worst 10%, three rounds | n09, n13 |
| `two_learner_share*` | detector (two models each flag their worst share) | keep molecules both keep | n32 |
| `remove_*` in n41 | conformal flags | drop flagged | n41 |

**Count less**

| method | what selects | rule |
|---|---|---|
| `downweight` | detector | weight 1/(1 + strength × score rescaled to 0-1), mean 1; strength 1 default, swept 0.25-9 in n24, n25 (n07:966-971; NE:2178-2206) |
| `huber_weight` | detector | min(1, 1.345 × 1.4826 × median(score) / score), mean 1 (n07:973-998). Rows in n24, n25, n42 only |
| `reliability_weight`, `cliff_aware_weight` | detector (neighbour gap; cliff-aware gap) | weight from the score (n02) |
| `neighbor_residual_weight`, `local_variance_ratio_weight` | detector, several distances | n21 |
| `neighbor_residual`, `local_variance_ratio` (as weights) | detector, importance-scaled features | n12, n13 |
| `downweight_pairs` | detector (near-duplicate pairs that disagree) | weight 0.25 (n35) |
| `invvar_ngboost` | detector (NGBoost's predicted SD) | weight 1/σ², mean 1 (n33) |
| `count_weighted`, `spread_weighted`, `precision_weighted`, `invvar_weighted`, `spread_pooled_weighted` | known flag (measurement count or repeat spread) | inverse-variance weights (n19, herg_ic50 only) |
| `weights_*` | known flag (source) | n39 |

**Move the label** (target, then how far)

| method | what selects | target | how far | where |
|---|---|---|---|---|
| `label_smooth` | detector | weighted mean of 9 neighbours (regression) | 0.25 × score rescaled to 0-1 (n07:214, NE:2105-2108) | n07, n18, n20, n24, n25, n42 |
| `knn_smooth` | detector, top 5% only on regression | mean of 9 neighbours | all the way (replaced) (n07:1139-1168) | same |
| `oof_smooth` | detector | out-of-fold LightGBM prediction | 0.25 × rescaled score | n07, n18, n20, n24, n25 |
| `gp_smooth` | detector | out-of-fold GP posterior mean | 0.25 × rescaled score. Refused above 1,000 training molecules (n07:388), so NaN on QM9 | n07, n18, n20 |
| `js_shrink` | detector | out-of-fold prediction | 0.5s/(1 + 0.5s), s the rescaled score (n07:1092-1099) | n07, n20, n25 |
| `prox_smooth` | EVERY MOLECULE on regression (no detector read) | mean of the 5 forest-proximity neighbours | 0.25 of the way (n07:220-228, n07:1170-1237) | n20, n25 only |
| `constant_lift` | known flag: training labels on the training maximum, if 2+ molecules and 1%+ | limit + mean Tobit lift | to that value | n07, n17, n18, n20, n25, n30 |
| `per_molecule_lift` | same flag | limit + SD × inverse Mills ratio, per molecule | — | n30 |
| `truncated_normal_mean` | same flag | p + SD × inverse Mills ratio, floored at the limit | — | n30 |
| `censored_em` | same flag | the same, iterated 3 rounds with the model's own out-of-fold prediction | — | n30 |
| `winsor_c1`, `c1.25`, `c1.5` | residual threshold (a detector in effect) | p + clip(y − p, ±c × 1.4826 × median abs residual) | onto the band | n30 |
| `move_confidence`, `move_proportional` | detector (a model's predicted spread) | the model's out-of-fold prediction | strength 0.25, 0.5 or 1 | n23, n42 |
| `rewrite`, `bayes_shrink` | detector | out-of-fold prediction | all the way; or a fitted weight | n31 |
| `shrink_flagged_*` | detector, share 5/10/20% | prediction | one weight, or one per molecule | n32 |
| `group_center`, `group_center_3plus`, `random_intercept` | known flag (scaffold group) | label minus the group's out-of-fold residual mean; the last shrunk by group size | — | n34 |
| `offset_removed*` | known flag (source) | label minus the source offset | — | n39 |
| `shrunk`, `calibrated` | known flag (measurement spread) | out-of-fold prediction, further for less precise labels | — | n19 |
| `move_toward_oof` | EVERY MOLECULE | out-of-fold LightGBM | alpha 0.25, 0.5, 1.0 | n27 |
| `huber_teacher_blend` | EVERY MOLECULE | 0.5y + 0.5 × out-of-fold Huber LightGBM | half | n33 |
| `yeo_johnson` | EVERY MOLECULE | power transform and back | — | n33 |
| `document_quantile_map`, `mean_label_bias_corrected`, `per_record_bias`, `per_record_random_effect` | known flag (ChEMBL document) | per-document offset removed | — | n29, herg_ic50 only |

Methods named in the outline that have NO rows in R/noise (VERIFIED by grep of every rows file):
- `quantile_clip`: zero files. Cut from n07 on 2026-10-05 (n07:182-186).
- `prox_smooth`, `huber_weight`, `invvar_weight` in n07: n07's own run list leaves them out by decision 2026-10-06 (n07:3096-3098). They have rows only in n20, n25, n24, n42, n19.

### 3.2 METHOD_GROUPS.md: the three CHECK rows, and faults found

The three CHECK rows (all VERIFIED in code):

1. **Constant lift in n18 (CHECK).** Selects by a rule on the labels, not by a detector: training labels equal to the training maximum, when at least 2 molecules and 1% sit there (n30:293-300, called at n07:1356). No suspicion score is read (n07:254-256). The size of the lift uses an out-of-fold prediction. "Known flag, move the label" is right. But the flag is read off the labels, not recorded in the data. The outline defines a known flag as "recorded in the data and nothing is inferred". This one is inferred from the labels' own top value.
2. **Constant lift in n07 (CHECK).** Same code, same answer. On QM9 with no damage it declines on every fold: all 6 rows are NaN (R/noise/n07_remediation/qm9__*.rows.jsonl, `method: constant_lift`, level 0).
3. **invvar_and_count_feature (n19) (CHECK).** It does two things: an inverse-variance weight AND log(count) as an extra input feature (n19:96, n19:282-286). The weight half is "known flag, count less", which is mitigation. METHOD_GROUPS files the whole row as noise-robust training. Under the outline's own test it should be mitigation. The same test also puts `count_feature` and `ngboost_count_feature` (known flag, no action) in no group: they change the features, which none of the three actions covers.

Faults in other rows (VERIFIED):
- `confident_learning`: "score is the out-of-fold predicted class probability". Wrong. The ensemble is trained and predicts on the same molecules (L:688-711), and flags need high error AND low ensemble spread (L:722-724).
- n23 rows (`gaussian_process`, `hetero_gp`, `heteroscedastic_nn`, `ngboost`, `activity_cliff_aware`): all filed "delete". n23 runs both delete and move (n23:9-18; rows hold `drop`, `move_confidence`, `move_proportional`).
- "search and fit on cleaned labels" (n20): filed "delete". n20 ran nine cleanups, seven of which move a label (R/noise/n20_clean_before_search rows: constant_lift, downweight, gp_smooth, js_shrink, knn_smooth, label_smooth, oof_smooth, prox_smooth, remove).
- `cliff_put_back` (n25): filed "a detector, delete". It undoes part of another cleanup. It is a modifier on all nine cleanups, not a deletion.
- `two_learner` (n32): filed noise-robust training. Two models each flag their worst share and the fit keeps the agreed molecules (n32:1244-1270). That is a detector plus delete.
- Missing entirely: `dividemix` (one of the twelve; n09 rows carry it), `prox_smooth`, `js_shrink` in n20 and n25, `huber_weight` in n25, n23's move rules, all n29 methods, n19's `spread_weighted`, `precision_weighted`, `invvar_weighted`, `gp_per_molecule_noise`, n38's `robust_loss`/`winsorise`/`itlm`, n41's flags, n33's loss changes, n40's GP noise-term arms, n30's `xgboost_aft` and `lightgbm_huber`, n17's arms. "85 methods" is a count of the analysis page's sections, not of the methods with results.

### 3.3 Down-weighting and the feature search

n07's docstring says `sample_weight` appears zero times in src/kirby/hybrid.py and src/kirby/find_best_hybrid.py (n07:891-893, checked 2026-09-17). Re-checked this session: zero in find_best_hybrid.py, two in hybrid.py (hybrid.py:71, :77). Both are the fit signature of the support vector regressor wrapper `SVRScaledTarget`, which passes the weight to the model. Neither reaches feature ranking or the budget split. So no count-less method can reach the feature choice. 7.9's sentence is CORRECT.

### 3.4 Controls: which experiment runs which

| control | experiment | code | ran? |
|---|---|---|---|
| Matched random deletion, same count, 5 seeds | n24 (`delete_at_random`) | n24:310-330 | yes; 40,110 rows in R/noise/n24_delete_or_keep |
| Random trimming / random two-learner / random winsorising, same count, 3 seeds | n32, n38, n41 (`*_random/seed0-2`) | n32 and n41 rows | yes |
| Detector scores shuffled, cleanup re-run, 3 seeds | n18 (`*__score_shuffled_seed0-2`) | n18:1062-1070 | yes; 14 datasets have n18 results (abcov_kd, buchwald_hartwig, caco2, cdr_flexibility, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, humanness_therapeutic, logd, qm9; JSON-only for six of them) |
| Random molecules put back instead of cliff molecules, same count | n25 (`arm=random_exempt`) | n25:838-865, seed 20260923 + 1009 × fold + seed index | yes |
| Per-molecule lifts dealt to the wrong molecules, 3 seeds | n30 (`per_molecule_lift_shuffled`) | n30:356-361 | yes |
| Generic random twin and shuffled twin for any cleanup | controls.py, imported by 19 experiments | controls.py:114-200 | **no**. Gated by KIRBY_NOISE_CONTROLS=1; 108 JSONs say `controls_run: false` |

Controls the outline lists that match the code: all four in its STANDING CAVEATS. SAME.
One gap: n25's random arm can be smaller than the cliff arm when most molecules are cliffs (83% on buchwald_hartwig). Those passes are dropped and counted (n25:844-856).

---------------------------------------------------------------------------------------------------

## 4. Noise-robust training methods

### 4.1 The twelve published methods (n09, src/kirby/utils/lnl_methods.py)

All train the same network as the plain model: 256-128-64, dropout 0.2, learning rate 0.001, batch 256, 200 epochs, patience 20 (L:427-436). Nothing is swept except distance-based selection's neighbour setting (n09:43-50). VERIFIED.

| method | class, line | settings | group under the outline's test | citation in code | status of that citation (outline 7.11) |
|---|---|---|---|---|---|
| small-loss selection | L:568 | keep 70% after 10 warm-up epochs | mitigation (detector + delete) | Han 2018 | CORRECT: not co-teaching |
| confident learning | L:659 | 3 networks, error > 85th pct and spread < 25th | mitigation | Northcutt 2021 | CORRECT: in-sample, no confident joint |
| distance-based selection | L:748 | k 10 default, drop top 15%; neighbour setting screened | mitigation | none | CORRECT |
| multistage cleaning | L:837 | 3 rounds × 10% | mitigation | Zhao 2025 | CORRECT: wrong paper |
| early learning | L:926 | γ = 0.95 | robust training | Zhang 2025 | CORRECT: wrong paper |
| uncertainty curriculum | L:1075 | 20 warm-up epochs; stages 25/50/75/100% | robust training | Bengio 2009 | CORRECT: ranks by own loss |
| DivideMix | L:1239 | 10 warm-up epochs; two networks and a 2-component Gaussian mixture on squared errors | robust training | Li 2020 | CORRECT |
| Meta-Weight-Net | L:1437 | meta interval 5, meta lr 0.001 | robust training | Shu 2019 | CORRECT |
| MentorNet | L:1628 | k 10 neighbours | robust training | Jiang 2018 | CORRECT |
| contrast-to-divide | L:1836 | 5 perturbations, σ 0.01 | robust training | Zheltonozhskii 2021 | CORRECT: wrong description |
| heteroscedastic GP | L:2013 | RBF | robust training; regression only | Liu 2020 | CORRECT: wrong paper |
| evidential kernel | L:2149 | NIG coefficient 0.01; regression only | robust training | Amini 2020 | CORRECT |

`DATA_LEVEL_METHODS` = the first four (L:2329-2330). This matches the outline's split. SAME.
The code docstring of n09 still says "Learning with Noisy Labels" and "LNL" throughout (n09:1-12). Fine for code; does not reach the thesis.

### 4.2 Robust losses and label transforms (n33)

What the code offers (n33:46-104, VERIFIED): Yeo-Johnson transform, NGBoost inverse-variance weights, C-Mixup (bandwidth 0.2 label SD, Beta(2,2)), target-noise bagging (5 fits, noise SD 0.5 × residual SD), Huber teacher blend; and loss changes for LightGBM (Huber at c × s, fair, median, Tukey at 4.685 s, expectile), XGBoost (pseudo-Huber, absolute), random forest (absolute-error splits), the network (Huber at c × s, c in 0.5, 1, 1.5, 2) and NGBoost (Student-t, 3 degrees of freedom). s = 1.4826 × median abs out-of-fold residual.

What ran (VERIFIED in R/noise/n33_robust_training):
- Datasets with results: buchwald_hartwig (rows, gaussian and outlier_p05), caco2, herg_ki, logd. herg skipped.
- Decided models are dnn and svm on caco2, herg_ki and logd (fields `models`), so only the network's Huber loss ran among the loss changes, plus the NGBoost normal-against-Student-t pair hosted in svm's features (`ngboost_host: svm`, herg_ki__gaussian.json).
- On buchwald_hartwig the decided models are lightgbm and rf, but its rows hold only the five label arms. No LightGBM, XGBoost or forest loss change is stored on any dataset.
- So Tukey, expectile, fair, median and pseudo-Huber losses have **no results**. The outline's 7.4 citations for Tukey (beaton1974fitting), Student-t (lange1989robust) and expectiles (newey1987asymmetric) cite methods that did not run, except the one NGBoost Student-t pair.
- `controls_run: false` in every n33 JSON.

### 4.3 Model-side arms (n40)

Arms (n40:30-80, VERIFIED): retuned on damaged labels, settings tuned on clean labels and reused, one seed with subsampling, ten seeds averaged (`SEED_AVG_N`), 1,000-tree forest, a non-negative stack of the decided models, and GP noise-term arms (fitted, fixed from a Gamma-test estimate, or chosen from multiples).
Damage: `gaussian` plus its own `dense_oofmse` (n40:13-21).
Results: caco2, herg_ki, logd, JSON only (R/noise/n40_model_side_regularisation). herg skipped.
Target-noise bagging is in n33, not n40. The outline's 7.4 line "(n33, n40)" for "ten seeds, stacking, target-noise bagging" is right only if read as split between the two.

### 4.4 Others that change training, not data

- Iterative trimmed loss (`itlm_a0.8/0.9/0.95`, n32:1214-1241): the trimming learner is LightGBM, not the model being scored. Rows in n32, n38.
- Network Huber loss under damage sized by model error (`robust_loss`, n38).

---------------------------------------------------------------------------------------------------

## 5. Outline 7.4a: QM9 as the clean control

### 5.1 What has run on QM9

QM9 rows: R/LEDGER.md, last written 9 Oct 13:10. Ledger row: "qm9 … noise 36/121 running c10, n07__gaussian 47.3h". VERIFIED.

| state | units |
|---|---|
| finished with results (11) | n01, n02, n09 censoring, n09 grouped_shifted, n13 censoring, n16 × 5 damage kinds, n30 |
| finished as "not applicable" (3, ledger says ok) | n22 (no curator flag), n29 (no ChEMBL documents), n36 (no repeat measurements). Each JSON has `result_status: skipped` |
| skipped (5) | n17, n19, n26 (no repeat measurements); n32 gaussian and grouped_shifted (`skipped: detection_undefined`) |
| running at 13:10 (47) | includes n07 × 5, n09 gaussian/outlier/shift, n11 × 5, n12 × 5, n13 × 4, n14, n15, n18, n20 × 5, n21 × 4, n24 × 3, n27, n32 × 3, n33 gaussian, n35, n38, n39, c10 |

Outline says "14 noise jobs had finished and 47 were running". SAME, if the three not-applicable units are counted as finished.
Partial row files already exist for the running units n07, n12, n18, n21, n24, n27 (R/noise/*/qm9__*.rows.jsonl). n27 QM9 has fold 0 only so far.

Not running on QM9 and never will under the current code: anything needing repeat measurements, documents or curator flags (n17, n19, n22, n26, n29, n36), and n32 under Gaussian and grouped damage.

### 5.2 Outline 7.4a numbers against the raw rows

All QM9, no noise added (level 0), fold 0, held-out RMSE. Settings are deduplicated across the five damage-kind files, which each store their own level-0 copy.

| outline says | current results | source | status |
|---|---|---|---|
| deleting flagged molecules never helped: 0 of 12 network settings on the best hybrid, 0 of 6 LightGBM | 0 of 12 (dnn: hybrid + hybrid__general2); 0 of 6 (lightgbm) | R/noise/n07_remediation/qm9__*__fold_aligned_2026-10-06.rows.jsonl, `method: remove`, `level: 0.0`, `score_D` vs `score_Dprime` | SAME, VERIFIED |
| moving toward neighbours' mean helped the network in 10 of 12, best 0.3568 → 0.3395 | 10 of 12; best 0.356812 → 0.339488 | same files, `method: label_smooth`, `model_input: dnn:hybrid__general2` | SAME, VERIFIED |
| random deletion helped the network on the best hybrid in 9 of 90, best 0.3568 → 0.3440 | 9 of 90; best 0.356812 → 0.344028. LightGBM: 0 of 90 | R/noise/n24_delete_or_keep/qm9__*__fold_aligned_screened_inner_2026-10-06.rows.jsonl, `method: delete_at_random`, level 0 | SAME, VERIFIED |
| network's untreated score on the second hybrid is 0.3484, 0.3568 or 0.3618 depending on the job | n07: 0.3568 in the censoring job, 0.3618 in the other four. n09: 0.3484 in the grouped_shifted job, 0.3618 in the censoring job. First hybrid: 0.3491 (n07 censoring) or 0.3550 (other n07 jobs) | n07 rows `score_D` by file; R/noise/n09_lnl_survey/qm9__{censoring,grouped_shifted}.passes.jsonl, `value.results.hybrid__general2_no_lnl` | SAME, VERIFIED |
| LightGBM's is the same in every job (0.3759) | 0.3759 on hybrid__general2 in all five n07 jobs; 0.3719 on the first hybrid in all five | n07 rows `score_D` | SAME, but add the first hybrid's 0.3719 |

The single-representation untreated score moves too: dnn base_1 is 0.3962 in four n07 jobs and 0.3999 in the censoring job. The same labels go in each time, so the network's fit differs between jobs.

### 5.3 QM9, the twelve published methods, no noise added (new; not in the outline)

R/noise/n09_lnl_survey/qm9__censoring.passes.jsonl and qm9__grouped_shifted.passes.jsonl, record with `ckpt_key [0.0, 0, 0]`, field `value.results`. Network, held-out RMSE, fold 0. A method "helps" on a feature space if its RMSE is below the plain network's on that space. Five feature spaces: base_1, base_2, hybrid, hybrid__general2, weighted_hybrid.

| method | feature spaces where it beat the plain network, censoring job | same, grouped_shifted job |
|---|---|---|
| small-loss selection | 0 of 5 | 0 of 5 |
| confident learning | 1 of 5 (base_2) | 4 of 5 |
| distance-based selection | 1 of 5 (base_2) | 1 of 5 (base_2) |
| multistage cleaning | 0 | 0 |
| early learning | 0 | 0 |
| uncertainty curriculum | 0 | 0 |
| DivideMix | 0 | 0 |
| Meta-Weight-Net | 0 | 0 |
| MentorNet | 0 | 1 (base_2) |
| contrast-to-divide | 0 | 0 |
| heteroscedastic GP | 0 | 0 |
| evidential kernel | 1 (base_2) | 0 |

Plain network, censoring job: base_1 0.3962, base_2 0.4949, hybrid 0.3550, hybrid__general2 0.3618, weighted_hybrid 0.3583.
Plain network, grouped_shifted job: 0.3999, 0.4867, 0.3568, 0.3484, 0.3545.

Two things to say.
- On clean labels, 10 of 12 methods are never better than the plain network on the hybrid. DivideMix is the worst: hybrid 0.4517 against 0.3550.
- Confident learning beats the plain network on 4 of 5 spaces in one job and 1 of 5 in the other. The labels are the same. The jobs differ in the per-fold search, which n09 runs per job (n09:118-140). That is the job-to-job spread the outline's 7.4a warns about, inside a single method.

### 5.4 QM9_CONTROL.md: one fault

answers/_scripts/qm9_control.py:52 files `('n27', 'removal_cost')` as "a detector, delete". n27's removal_cost trains on a random share of molecules, 20 draws per share, labels correct (n27:17-24). It has no detector. Its 800 QM9 rows are random deletion. The page's "removal_cost" rows belong beside the random-deletion control, not among detector-driven methods.

---------------------------------------------------------------------------------------------------

## 6. Every noise experiment, one line each

Group: D = detection, M = noise mitigation, T = noise-robust training, X = descriptive (fits no method, or measures the noise itself). "Datasets with results" counts datasets with a JSON whose `result_status` is not `skipped`, or a rows file, in R/noise. Folds: fold 0 unless stated. Outline section: where the outline uses it, from its text or its file pointers. VERIFIED from the files unless marked.

| exp | question (from the file header) | group | datasets with results | outline uses it | status / note |
|---|---|---|---|---|---|
| c10 | how far each configuration's held-out score falls when the training labels are damaged | X | 13, folds 3-5; logd and buchwald_hartwig `empty` | **not used** | descriptive; classification sets skipped |
| n01 | prediction error on activity-cliff molecules against the rest | X | 12 incl. qm9 | **not used** | qm9 finished |
| n02 | down-weighting molecules that disagree with their neighbours, importance-scaled features | M | 16 incl. qm9 | not by name (7.4a via QM9_CONTROL) | no damage injected (`none`) |
| n06 | does a detector's suspicion match real disagreement on BindingDB | D | **none**; no result folder | not used | bindingdb_kd is cut (LEDGER.md header) |
| n07 | after detecting, what to do: 8 cleanups × detector | M | 9 incl. qm9 | 7.3, 7.4, 7.4a, 7.5 | promoted folds 0-1 on caco2 censoring (R/noise_full) |
| n09 | do the twelve published methods change rank between single representations and hybrids | M + T | 12 incl. qm9, tcga_brca_pam50 | 4.4.2, 7.11 | outline 7.11 says "all 9 datasets that ran"; now 12 have files. CHANGED |
| n11 | detector × action grid | D + M | 6; often level 0 only | **not used** | many units still running |
| n12 | neighbour detectors in importance-scaled features | D + M | 13 incl. qm9 | not used | one failed unit (flab_thermo_tresanco, dose error) |
| n13 | sensitivity across damage kinds | M | 9 incl. qm9; one non-zero level per file | not used | — |
| n14 | activity-cliff landscape; fits no model | X | caco2, herg_ki, logd, qm9 | not used | 5 folds |
| n15 | clean-searched against noise-searched hybrid | X | 6, folds 3-5 | not used | relevant to 7.9 |
| n16 | does feature importance move when labels get noisier | X | 7 incl. qm9, 5 folds | not used | qm9 finished, all 5 kinds |
| n17 | does cleaning help when the noise is real (herg_ic50 repeats) | M | herg_ic50 only | not used | — |
| n18 | does cleaning help on the labels as delivered; shuffled-score control | M | 14 incl. qm9 | 7.8 (SMAC, by description) | rows file for 8; JSON only for caco2, ginkgo_colloidal, ginkgo_titer, herg, herg_ki, logd |
| n19 | what a measurement count and spread are worth | M + T | herg_ic50 only | 7.4, 7.10 | — |
| n20 | must cleaning come before the feature search | M | 8 | 7.4, 7.9 | 7.9 names 5 datasets; files exist for 8 (adds cdr_flexibility, herg, logd). No control |
| n21 | which distance the neighbours are found in | D + M | 9 incl. qm9 | not used | — |
| n22 | do detectors find records ChEMBL curators flagged | D | **none**; every file `skipped` | not used | herg_ic50 unit still running |
| n23 | can a model's own per-molecule spread clean labels | D + M | caco2, herg, herg_ki, logd | 7.6 | promoted folds 1-2 in R/noise_full |
| n24 | delete or keep flagged molecules; random-deletion control | M | 6 incl. qm9 | 7.4a, 7.5 | promoted folds up to 0-4 on herg_ki censoring |
| n25 | does putting activity-cliff molecules back help; random put-back control | M | caco2, herg, herg_ki, logd | 7.4, 7.8 | — |
| n26 | how large the label noise is, from repeats | X | herg_ic50 only, 3 folds | not used | — |
| n27 | what removing molecules costs; how much label error reaches test error | X | 6; **5 folds** on buchwald_hartwig, caco2, ginkgo_colloidal, herg_ki, logd | standing caveat ("all five folds on Caco-2 and hERG Ki") | caveat CHANGED: five datasets have 5 folds, not two |
| n29 | does a per-document bias change what the model learns | M | herg_ic50 only | 7.4 | — |
| n30 | clipped labels: lift, drop, or censored likelihood; shuffled-lift control | M | 8 incl. qm9 | 7.3, 7.4, 7.7 | promoted logd 5 folds |
| n31 | how good a residual detector is and needs to be | D + M | 5 | 7.4 (pointer) | — |
| n32 | trimmed refits, two learners, Shapley and other detectors | D + M + T | 8 | 7.3, 7.4, 7.6 | qm9 skipped under two kinds |
| n33 | robust losses, transforms, weights, augmentation | T (+ M for invvar_ngboost, teacher blend) | buchwald_hartwig, caco2, herg_ki, logd | 7.4, 7.10 | most loss changes never ran (4.2) |
| n34 | offset per scaffold group: subtract or random intercept | M | caco2, herg_ki, logd | 7.4, 7.7 | — |
| n35 | how far apart the labels of two near-identical molecules are | X + M | 8 | **not used** | — |
| n36 | which molecules to measure again | X | herg_ic50 only | not used | — |
| n37 | how much model error is label noise | X | 5, 5 folds | **not used** | — |
| n38 | noise sized by the model's own error | M + T | 9 | not used | — |
| n39 | several sources with correlated errors and source bias | M | 5 | 7.10 (by description) | simulated sources |
| n40 | does changing the model help where changing labels did not | T | caco2, herg_ki, logd | 7.4, 7.10 | JSON only |
| n41 | flags with a controlled false-flag rate, then act | D + M | buchwald_hartwig, caco2, herg_ki, logd | 7.3, 7.6 | includes ORACLE calibration arms |
| n42 | every data-changing method × every similarity setting × every model input | M | caco2, cdr_flexibility, herg, herg_ki, logd, tcga_brca_pam50 | **not used** | — |
| noise_ceiling | extrapolate each noise sweep back to zero error | X | 17 files; qm9 has 0 curves | **not used** | reads other results |

Experiments with results that the outline does not use: c10, n01, n11, n12, n13, n14, n15, n16, n17, n21, n26, n35, n36, n37, n38, n42, noise_ceiling. n15 bears directly on 7.9. n37 and n26 measure the size of the noise, which 7.14's "assay error floor" needs.

---------------------------------------------------------------------------------------------------

## 7. Outline statements in 7.2-7.4a, checked

| outline says | current code or results | status |
|---|---|---|
| 7.2 damage kinds: Gaussian, outliers in 5%, shift in 10%, sparse spread increase in 10%, offset per group, wider spread per group, clipping, six flips | six regression kinds; "outliers in 5%" moves every label; "sparse spread increase" is an upward shift; "wider spread per group" is not in the grid | CHANGED |
| 7.2 level is a fraction of the spread of the clean training labels | true for Gaussian, outliers, group offset; the shift kinds use SD_tr or σ_r as the unit of a fixed shift; flips use a share | CHANGED (add the units) |
| 7.2 censoring is a share of labels clipped | yes; limit = quantile of the clean training labels | SAME |
| 7.2 only training labels are damaged | yes for the outer held-out fold | SAME |
| Standing caveat: every number is fold 0 except n27 on two datasets | n27 has 5 folds on five datasets; c10, n14, n15, n16, n26, n37 have 3-5; promoted folds exist for six experiments | CHANGED |
| Standing caveat: a hybrid's clean-label score moves with the damage, because it is rebuilt from damaged labels | the features are rebuilt; the representations and recipe are chosen on clean labels except in n09, n15, n20 | CHANGED (half true) |
| Controls exist for deletion, shuffled scores, cliff put-back, lifts | yes, all four run; no other control has run | SAME |
| 7.3 all predictions out of fold with scaffold groups | not for out-of-bag, cleanlab, two of the twelve, and the GP subsample | CHANGED |
| 7.4 "clipped onto a leaf quantile range" | quantile_clip has no results | NOT YET RUN (cut) |
| 7.4 "toward a forest-proximity neighbours' mean" | prox_smooth moves every label 0.25 of the way, no detector; rows only in n20, n25 | CHANGED (it is "every molecule") |
| 7.4 robust losses: Huber, Tukey, Student-t, expectiles, C-Mixup | only the network's Huber, C-Mixup and one NGBoost Student-t pair have results | CHANGED |
| 7.4a QM9 counts | as in 5.2 | SAME |
| 7.4a "14 finished, 47 running" | 14 ok (3 of them not-applicable records), 5 skipped, 47 running at 13:10 | SAME |
| NAMES: 13 names move a label | yes for 12 that ran; quantile_clip never ran; prox_smooth selects every molecule | SAME, with the two notes |
| NAMES: "n07_remediation.py:362 keeps huber_weight out of the label-repair list" | n07:362-363 says invvar_weight is kept out of REGRESSION_ONLY_CLEANUPS "for the reason huber_weight is not: it sets a row weight and never touches the label". The list itself starts at n07:368. The quoted words are at :363, about invvar_weight as well as huber_weight | SAME (line 363, not 362) |
| NAMES: 42 detector / 22 known flag / 6 every molecule of 85 | a count of METHOD_GROUPS rows, which has wrong and missing rows (3.2) | UNVERIFIED as a count of methods |
