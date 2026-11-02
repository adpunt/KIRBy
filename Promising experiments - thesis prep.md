# Promising experiments: thesis prep

Written by Claude on 2026-09-30, for Adelaide to edit.
It draws on two sources: the claims in "Promising experiments", taken as given, and the dirty local run of 28–30 Sep (`/Volumes/seagate/dirty_local_2026_09_28/REPORT.md`).
The dirty run is herg_ki only for Chapter 5, with 3 folds.
All RMSE is in pKi, lower is better. Spearman, MCC, R² and AUC are higher is better.

Ten leads, then the cut list.

---

## Chapter 5

### 1. The hybrid beats the best single representation
Thesis: "Results: when the hybrid helps"; Conclusions note "the hybrid against the best single representation".

herg_ki, dirty run. This compares the best two hybrids with the best two single representations, all on the same 3 folds:

| model | hybrid #1 | hybrid #2 | single #1 | single #2 |
|---|---|---|---|---|
| svm | 0.5663 | 0.5664 | 0.5979 mhg_gnn | 0.6122 ecfp4 |
| lightgbm | 0.5773 | 0.5803 | 0.6015 mhg_gnn | 0.6087 ecfp4 |
| rf | 0.5958 | 0.5976 | 0.6032 ecfp4 | 0.6103 mhg_gnn |

- **Where the hybrids come from:** the 48 allocation settings of c32, and s08 for rf at 100 features.
- **Per fold:** with svm, the best hybrid is below mhg_gnn on all 3 folds.
- **How both sides were picked:** the hybrids are the best of about 60 variants, the singles the best of 5. Both were picked on the held-out folds. The full runs pick inside each fold.

opig_blueprint_transfer, herg_ki, from the existing file. A recipe found on logd is re-run on herg_ki. It scores below herg_ki's own best single for three models:
- svm: 0.5678 against 0.6616.
- dnn: 0.5943 against 0.6594.
- lightgbm: 0.6019 against 0.6318.

For the full runs: this table, per dataset.

### 2. The lead grows with the size of the search
Thesis: Conclusions note "whether the lead grows with the size of the search".

From the existing file, one search (48 configurations) against another (2–4 configurations):
- c11, hybrid against ensemble: 0.6246 against 0.6023 with the larger search, 0.5535 against 0.6323 with the smaller.
- c05, hybrid against best single: ahead at 3 of 8 points with the larger search, behind at 7 of 8 with the smaller.

herg_ki, dirty run, RMSE:

| model | one configuration (c02) | best of 48 settings (c32) |
|---|---|---|
| svm | 0.5784 | 0.5663 |
| lightgbm | 0.5870 | 0.5773 |
| rf | 0.6038 | 0.5976 |

The existing file's guard still applies: the comparison methods get no search of their own.

### 3. The hybrid against PCA, recursive feature elimination (RFE) and random features, at the same number of features
Thesis: Conclusions note "the hybrid against PCA, mRMR, recursive feature elimination and random selection".

flab_thermo_tresanco, spearman, from the existing file:

| features | hybrid | PCA | RFE | random |
|---|---|---|---|---|
| 10 | 0.5945 | 0.4860 | 0.4682 | 0.2145 |
| 50 | 0.6131 | 0.5225 | 0.5689 | 0.5094 |
| 250 | 0.6029 | 0.4824 | 0.6004 | 0.5636 |

herg_ki, RMSE:
- At 250 features (dirty run, rf): hybrid 0.5992, PCA 0.6553, RFE 0.5922, random 0.6404.
- At 10 features (existing file): hybrid 0.6304 against PCA 0.6776.

The hybrid is ahead of PCA and random features at every number of features tested. It is ahead of RFE below 250 features. At 250 features RFE is level with it on flab and ahead by 0.0070 RMSE on herg_ki.

mRMR has no score at 250 features: its ranking stops at 200.

### 4. The recipe transfers to a smaller dataset; the exact features do not
Thesis: Methods note "whether the recipe or the exact features transfer better"; Limitations note "how far a recipe carries".

From the existing file (logd to herg_ki), against herg_ki's own best single:
- The re-run recipe is ahead for 3 of 5 models: svm, dnn and lightgbm.
- The copied feature list is ahead for 2 of 5: svm and lightgbm.
- With dnn, the copied features score 0.6815 against the recipe's 0.5943.

One dataset pair.

### 5. The best number of features differs by model on the same dataset
Thesis: "The search space and why each axis is swept", note "the best budget differs by model on the same dataset".

herg_ki, dirty run, c09, RMSE. The hybrid recipe is fixed and only the number of features changes:

| model | best number of features | score |
|---|---|---|
| rf | 100 | 0.5978 |
| lightgbm | 250 | 0.5870 |
| svm | 1,000 | 0.5733 |

s08 gives the same order on the same folds: rf at 100 features, lightgbm at 250 and svm at 500. svm peaking at a larger budget than the tree ensembles is the thesis note's own example.

### 6. Federated: choosing features across sites helps when the molecules stay at their sites
Thesis: "Results: federated"; note "whether keeping molecules at their sites and selecting features beats keeping molecules at their sites with all features".

fed03 Part A, three real hERG sources, MCC, with the molecules kept at their sources:
- Existing file: 0.4711 with all features, 0.4981 with the selected features.
- Dirty run: 0.4690 with FCFP4 bits, 0.5264 with the selected features, ahead on all 3 folds.

The dirty run also changes the representation, from FCFP4 to the five pool representations.

---

## Chapter 6

### 7. Correcting labels clipped at a detection limit
Thesis: "Censoring: correcting labels clipped at a detection limit".

herg_ki, rf on ecfp4, 10% of training labels clipped, RMSE:

| training labels | existing file | dirty run (n30) |
|---|---|---|
| real labels | 0.6032 | 0.6032 |
| clipped | 0.6588 | 0.6558 |
| clipped, then corrected | 0.6118 | 0.6139 |
| clipped molecules deleted | 0.7706 | 0.7721 |

From the existing file: the corrected labels beat the clipped labels in 48 of 48 comparisons, across 25% and 40% clipping and caco2. In the dirty run only the 10% level finished before the 3-hour cap.

### 8. The input matters more than the noise-robust training method
Thesis: "Gaussian noise: learning-with-noisy-labels methods"; note "whether the input moves the score more than the method".

flab_thermo_tresanco, Spearman, from the existing file:
- At 1.5× noise, the best of 12 methods on onehot (small_loss) scores 0.3568.
- The plain network on the hybrid scores 0.4859.

The comparison input here is onehot. The Methods note says the final comparison uses each model's top two single representations. So in the full run, onehot must be replaced by the top two.

### 9. Predicting each molecule's activity from its 5 nearest neighbours works better in the hybrid
Thesis: "Similarity in a hybrid"; Chapter 7 "Simple is often better".

herg_ki, dirty run:
- **s05, RMSE.** Similarity hybrid 0.6128 against the best single representation, weighted graph_kernel, at 0.6246. Against the single representation picked on each fold, the default hybrid is lower on all 3 folds.
- **s13.** The mean pKi gap to the 5 nearest training molecules is 0.5602 for the similarity hybrid against 0.5811 for pdv, lower on all 3 folds.

Weighting features by importance, with 10 neighbours, R²:
- From the existing file (c04): the gain is +0.2164, +0.0546 and +0.0487 on three datasets.
- Dirty run, herg_ki: 0.4638 to 0.5235.

Weighting the similarity hybrid itself goes the other way: 0.6128 to 0.6167 RMSE (s10).

### 10. Noise mitigation and noise-robust training on the recorded labels (level 0)
Thesis: "Real labels, nothing injected: cleaning"; "Gaussian noise: learning-with-noisy-labels methods".

Dirty run. Each line gives no mitigation → mitigated, with the per-fold scores in brackets.

**herg_ki, RMSE, lower is better. Every result here is on a base representation, not a hybrid:**
- **C-Mixup, svm on mhg_gnn (n33).** 0.6095 → 0.5885 (0.6053 → 0.5971, 0.6137 → 0.5798). 2 of 3 folds ran, and both are better.
- **C-Mixup, svm on ecfp4 (n33).** 0.6230 → 0.6033 (0.6174 → 0.6108, 0.6287 → 0.5959). 2 of 3 folds ran, and both are better.
- **Target-noise bagging, lightgbm on mhg_gnn (n33).** 0.6200 → 0.5942 (0.6159 → 0.6039, 0.6241 → 0.5845). 2 of 3 folds ran, and both are better.
- **Weight 0.25 on close pairs whose labels disagree, lightgbm on mhg_gnn (n35).** 0.6079 → 0.5966, better on 3 of 3 folds. With Gaussian noise injected, it is also better on 3 of 3 folds:
  - level 0.15: 0.6090 → 0.5917.
  - level 0.3: 0.6129 → 0.5979.
- **Downweighting suspect molecules (n42).** Only fold 0 ran.
  - lightgbm on mhg_gnn: 0.6255 → 0.5957, and 11 of 11 settings are better.
  - rf on mhg_gnn: 0.6211 → 0.6060, and 10 of 11 settings are better.

**humanness_therapeutic, AUC, higher is better (n20), mean of 3 folds. These are on the hybrid.** Cleaning is done before the hybrid's features are chosen, and the model is fitted on that hybrid:

| cleanup | rf | svm | lightgbm |
|---|---|---|---|
| remove | 0.9564 → 0.9606 | 0.9560 → 0.9619 | 0.9521 → 0.9618 |
| label smoothing | 0.9564 → 0.9639 | 0.9560 → 0.9640 | 0.9521 → 0.9656 |
| proximity smoothing | 0.9564 → 0.9615 | 0.9560 → 0.9605 | 0.9521 → 0.9615 |

- Proximity smoothing with lightgbm is better on every fold, whichever representation the cleaning is done in.
- The other eight cells are better on average but not on every fold.
- At level 0 the injector flips about 1.3% of class labels.

The n33 hybrid columns are left out: that experiment's hybrid saw the held-out labels.

---

## Cut, and why

- **c19, importance weighting during training.** The existing entry says weighting cannot move a tree model. In the dirty run rf moved from 0.6038 to 0.6084. The stated mechanism does not hold.
- **c21, extra features added to one representation.** More than half of the admitted features were decoys, and the admission gate was off.
- **c23, evaluator matching, as its own lead.** It is a null (8 of 18 on herg_ki; 3 of 6 in the dirty run). It survives only inside lead 2.
- **n18, cleaning real labels.** On herg_ki and caco2, no cleaning method lowers RMSE by more than 0.0001. flab gains on the hybrid but loses on onehot.
- **n21, finding flipped labels with the hybrid distance.** Average precision 0.5916 against 0.5086 and 0.5343, but at one noise level on one dataset. Also, class labels are flipped even at level 0.0.
- **fed01 and fed04.** In fed01 most of the gap is pooling the molecules, not the features. fed04 compares two hybrids.
- **c01, c22, c24, n14.** These give single numbers for Table 5.2, the blueprint and stability notes, and the activity-cliff counts. They come from the full runs, not from leads.
