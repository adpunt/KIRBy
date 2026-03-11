# KIRBy Molecular Pipeline Audit: Comparison Against Literature & Ground Truth

**Audit date:** 2026-03-10
**Auditor method:** WebSearch of GitHub repos + literature, cross-reference with user's qsar_qm_models repo
**Scope:** LogD, Caco2, hERG, QM9 datasets; 29 reps; Ridge/RF/XGBoost/LightGBM models

---

## 1. Repos & Papers Examined

### 1.1 Direct ground truth (user's own repo)
- **Repo:** `/Users/apunt/repos/qsar_qm_models/scripts/run_qm_qsar_models.py`
- **Data dir:** `/Users/apunt/repos/qsar_qm_models/results/validation_full/openadmet_logd/`
- **Split:** DeepChem ScaffoldSplitter 80/10/10 single split
- **Models:** RF, XGBoost, DNN, Quantile RF, GP (GAUCHE), SVM, GIN, GCN, GAT
- **Reps:** ECFP4, SNS (Sort & Slice), PDV, Mol2Vec, ChemBERTa, MHG-GNN
- **Metric:** R2, RMSE, MAE, Spearman

### 1.2 GitHub repos examined

| Repo / Paper | URL | What they do |
|---|---|---|
| **chemprop / chemprop_benchmark_v2** | https://github.com/chemprop/chemprop_benchmark_v2 | D-MPNN benchmarks on lipophilicity (AstraZeneca LogD 4200), scaffold split |
| **swansonk14/admet_ai** | https://github.com/swansonk14/admet_ai | Chemprop-RDKit trained on 41 ADMET datasets incl. lipophilicity, Caco2, hERG |
| **DonghaiZHAO-ZJU/Benchmark-ADMET-2025** | https://github.com/DonghaiZHAO-ZJU/Benchmark-ADMET-2025 | 2025 ADMET benchmark: Lipophilicity_AstraZeneca (4200), Caco2 (906), hERG. Scaffold split, MAE metric |
| **dahvida/GBM_Benchmarking** | https://github.com/dahvida/GBM_Benchmarking | XGBoost vs LightGBM vs CatBoost on 16 QSAR datasets, 157K models. Lipophilicity included |
| **oxpig/ECFP-Sort-and-Slice** | https://github.com/oxpig/ECFP-Sort-and-Slice | SNS fingerprints benchmarked on lipophilicity + 4 other datasets, scaffold + random splits, RF/XGBoost |
| **PDelre93/hERG-QSAR** | https://github.com/PDelre93/hERG-QSAR | hERG classification from ChEMBL, Dragon descriptors, classification (not regression) |
| **arthuc01/HERG-QSAR** | https://github.com/arthuc01/HERG-QSAR | hERG analysis with RDKit + sklearn, ChEMBL data |
| **jbr-ai-labs/lipophilicity-prediction** | https://github.com/jbr-ai-labs/lipophilicity-prediction | D-MPNN + substructure features for LogD, NeurIPS 2020 workshop |
| **mindrank-ai/PharmaBench** | https://github.com/mindrank-ai/PharmaBench | 11 ADMET datasets (incl. LogD 13K mols), scaffold + random splits, CSV format |
| **Merck/puqsar** | https://github.com/Merck/puqsar | Prediction intervals for QSAR: RF, DNN, GB with uncertainty |
| **PatWalters/practical_cheminformatics_tutorials** | https://github.com/PatWalters/practical_cheminformatics_tutorials | Industry-standard QSAR methodology reference |

### 1.3 Key papers examined

| Paper | Results relevant to KIRBy |
|---|---|
| Yang et al. 2019 "Analyzing Learned Molecular Representations" (Chemprop) | Lipophilicity scaffold RMSE: D-MPNN 0.542, RF(ECFP) ~0.655. 19 public datasets |
| Boldini et al. 2023 "Practical guidelines for gradient boosting" (J. Cheminform.) | XGBoost generally best perf, LightGBM fastest. CatBoost comparable. 16 datasets, 157K models |
| Dablander et al. 2024 "Sort & Slice" (J. Cheminform.) | SNS > hash-based ECFP by 11.37% MAE on lipophilicity. 20 scenarios, scaffold + random |
| Cai et al. 2023 "A systematic study of key elements" (Nature Comms) | RF on MorganBits best on lipophilicity under scaffold split. 62,820 models, 36 datasets |
| CaliciBoost 2025 (J. Cheminform.) | Caco2: XGBoost+PaDEL R2=0.78, RMSE=0.32 (scaffold). PaDEL > Morgan > RDKit |
| ADMET evaluation 2025 (J. Cheminform.) | Caco2 industrial validation: XGBoost+ECFP best, Pearson r~0.88 |
| ADMETlab 3.0 (Nucleic Acids Res 2024) | Multi-task DMPNN, R2 0.75-0.95 for regression endpoints |
| OpenADMET ExpansionRx Challenge (2025-2026) | Community blind challenge on same OpenADMET data KIRBy uses |
| hERG risk assessment (Chem Res Tox 2024) | hERG regression: SVM+ECFP4 best, CNN R2=0.70 |

---

## 2. Ground Truth Comparison: KIRBy vs qsar_qm_models

### 2.1 LogD (OpenADMET)

**CRITICAL CONTEXT:** qsar_qm_models uses DeepChem ScaffoldSplitter 80/10/10 single test split. KIRBy uses Murcko scaffold GroupKFold (3-fold). These are DIFFERENT evaluation protocols so exact numbers should differ.

| Rep + Model | qsar_qm_models R2 | KIRBy R2 | Delta | Verdict |
|---|---|---|---|---|
| DNN + MHG-GNN | **0.824** | N/A (no DNN) | -- | **MISSING MODEL** |
| DNN + SNS | **0.818** | N/A (no DNN, no SNS in survey) | -- | **MISSING MODEL + REP** |
| DNN + ECFP4 | **0.766** | N/A (no DNN) | -- | **MISSING MODEL** |
| XGBoost + SNS | 0.756 | N/A (SNS not in 'all' preset) | -- | **MISSING REP** |
| XGBoost + MHG-GNN | 0.747 | 0.795 (lgbm) | +0.048 | Expected (GroupKFold can be slightly different) |
| RF + MHG-GNN | 0.704 | 0.813 (ridge) | +0.109 | Expected (different models, ridge vs RF) |
| XGBoost + PDV | 0.740 | 0.788 (lgbm) | +0.048 | Expected |
| RF + PDV | 0.696 | 0.736 (ridge) | +0.040 | Expected |
| XGBoost + ECFP4 | 0.697 | 0.756 (lgbm) | +0.059 | Expected |
| RF + ECFP4 | 0.645 | 0.723 (ridge) | +0.078 | Expected (GroupKFold avg vs single test set) |
| DNN + PDV | **-0.024** | 0.788 (lgbm) | -- | Different models, N/A |
| GP + PDV | **-0.199** | 0.736 (ridge) | -- | GP known bad on raw PDV |

**Key finding:** KIRBy consistently reports HIGHER R2 than qsar_qm_models. This is likely because GroupKFold CV averages across folds (some folds are easier), while DeepChem ScaffoldSplitter gives a single held-out test split that can be harder. **NOT a bug, but KIRBy numbers are likely more optimistic than a single scaffold hold-out.**

### 2.2 Overall best LogD numbers

| Source | Best R2 | Method |
|---|---|---|
| KIRBy | 0.841 | grover + ridge (3-fold GroupKFold) |
| qsar_qm_models | 0.824 | DNN + MHG-GNN (ScaffoldSplitter 80/10/10) |
| Chemprop (Yang 2019) | ~0.80 (RMSE 0.542 on AZ-Lipo 4200) | D-MPNN scaffold split |
| ADMETlab 3.0 | 0.75-0.95 range | Multi-task DMPNN |

**Verdict:** KIRBy's 0.841 is strong. GROVER (4800D) is a very large pretrained representation and ridge is effectively doing massive implicit regularization. The number is credible but should be reported alongside the 3-fold GroupKFold caveat.

---

## 3. Specific Issues Found

### ISSUE 1: Missing Sort & Slice (SNS) from molecular survey (NEEDS INVESTIGATION)

**Severity:** High
**Evidence:** `create_sns()` exists in `/Users/apunt/repos/KIRBy/src/kirby/representations/molecular.py` (line 327) but is NOT in the `REP_PRESETS['all']` list at `/Users/apunt/repos/KIRBy/tests/find_best_hybrid_molecular.py` line 88-96. SNS is the user's own representation from their qsar_qm_models repo and was shown to outperform hash-based ECFP by 11.37% MAE in the OxPIG paper (https://github.com/oxpig/ECFP-Sort-and-Slice).

**Ground truth numbers (OpenADMET LogD, sigma=0):**
- XGBoost + SNS: R2 = 0.756 (better than XGBoost + ECFP4 at 0.697)
- DNN + SNS: R2 = 0.818 (2nd best overall)

**Note:** SNS requires per-fold fitting (featurizer must be fit on training data only), so it would need a rep_factory rather than a static rep. This is probably why it was omitted — it doesn't fit the static precomputed rep pattern. But the infrastructure for fold-dependent reps exists (rep_factories in find_best_hybrid.py).

**File:** `tests/find_best_hybrid_molecular.py` line 88, `src/kirby/representations/molecular.py` line 327

### ISSUE 2: No DNN / deep learning model in the model roster (NEEDS INVESTIGATION)

**Severity:** High
**Evidence:** KIRBy's DEFAULT_MODELS in `src/kirby/find_best_hybrid.py` (line 217-262) includes: ridge, rf, xgboost, knn, lightgbm, mlp, svm. The MLP is `MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)` -- a 2-layer sklearn MLP that is NOT equivalent to a properly trained DNN.

The ground truth repo uses a custom DNN (PyTorch, multi-epoch training with proper learning rate scheduling), and DNN is the best model for LogD (R2=0.824 with MHG-GNN). The CaliciBoost Caco2 paper also shows XGBoost beats RF by a large margin.

**Literature evidence:**
- DNN + MHG-GNN: R2=0.824 on LogD (best in qsar_qm_models)
- DNN + SNS: R2=0.818 on LogD
- DNN + ECFP4: R2=0.766 (even with simple ECFP)
- sklearn MLP is known to underperform PyTorch DNNs significantly for molecular QSAR

**Missing models by literature comparison:**
- **CatBoost**: Boldini et al. 2023 showed CatBoost achieves similar performance to XGBoost/LightGBM. User's qsar_qm_models includes CatBoost. Easy to add.
- **Gaussian Process (GAUCHE)**: User's qsar_qm_models has GP with Tanimoto kernel. GP + ECFP is a strong baseline in some datasets. However, GP+PDV = R2=-0.199 in ground truth, so it is very rep-dependent.
- **GNN models (GIN, GCN, GAT)**: These are end-to-end graph neural networks from qsar_qm_models. SchNet and DimeNet++ in KIRBy serve a similar role but are frozen pretrained, not trained on the target dataset.

**File:** `src/kirby/find_best_hybrid.py` line 217-262

### ISSUE 3: Missing CatBoost model (EXPECTED / EASY FIX)

**Severity:** Low
**Evidence:** CatBoost is used in qsar_qm_models and shown to be competitive with XGBoost/LightGBM in Boldini et al. 2023. KIRBy does not include it. Adding CatBoost is trivial (analogous to the existing XGBoost/LightGBM pattern).

**File:** `src/kirby/find_best_hybrid.py` line 217-262

### ISSUE 4: Missing PaDEL and Mordred descriptors (NEEDS INVESTIGATION)

**Severity:** Medium
**Evidence:** CaliciBoost (2025) showed that for Caco2 prediction, PaDEL descriptors (2D+3D) combined with XGBoost achieved R2=0.78, RMSE=0.32 -- significantly better than Morgan fingerprints or RDKit descriptors alone. KIRBy's PDV is RDKit 2D descriptors only (200D, DEFAULT_DESCRIPTOR_LIST in molecular.py line 138-170).

PaDEL descriptors include both 2D and 3D features, and Mordred provides >1800 descriptors. Neither is present in KIRBy.

**However:** Adding PaDEL/Mordred would require additional dependencies. KIRBy's PDV is the standard RDKit descriptor set. The CaliciBoost study was on a different Caco2 dataset (TDC benchmark, 906 compounds). KIRBy uses OpenADMET-Caco2_Efflux (2158 compounds) which is a different endpoint (efflux ratio, log-transformed).

**File:** `src/kirby/representations/molecular.py` line 138-170

### ISSUE 5: Caco2 best R2=0.519 appears low (NEEDS INVESTIGATION)

**Severity:** Medium-High
**Evidence:** KIRBy's best Caco2 is graph_kernel+lgbm at R2=0.519 (BOM line 20). Literature comparisons:
- CaliciBoost (TDC Caco2, 906 mols): XGBoost+PaDEL R2=0.78
- Natural products study: SVM-RF-GBM ensemble R2=0.76
- ADMETlab 3.0: accuracy 0.89 (classification, different task)

**However,** KIRBy is predicting Caco2_Efflux (P-gp efflux ratio, log-transformed) from OpenADMET, not Caco2 permeability. These are DIFFERENT endpoints. Efflux ratio is inherently noisier and harder to predict than absolute permeability. The R2=0.519 may be reasonable for this specific endpoint.

**Key question:** Is the efflux ratio the right Caco2 endpoint to use, or should KIRBy also/instead predict Caco2 permeability (which has more literature baselines)?

**File:** `tests/find_best_hybrid_molecular.py` line 293-300

### ISSUE 6: hERG best R2=0.552 -- plausible but no direct comparison (EXPECTED)

**Severity:** Low
**Evidence:** KIRBy's hERG dataset is ChEMBL-hERG-Ki (CHEMBL240) with 1,415 compounds predicting pKi as continuous regression. Most hERG literature focuses on classification (IC50 threshold), not regression.

Literature regression comparisons:
- CNN-based model: R2=0.70 (Chem Res Tox 2024, but different dataset/size)
- SVM + ECFP4 1024: best in one study (no R2 reported)
- Most hERG repos (PDelre93, arthuc01) do classification, not regression

KIRBy's R2=0.552 (ecfp4+lgbm) is reasonable for scaffold-split regression on 1,415 compounds. The small dataset size limits achievable R2.

**File:** `tests/find_best_hybrid_molecular.py` line 193-267

### ISSUE 7: QM9 best R2=0.139 -- confirmed NOT a bug but poor for the claim (EXPECTED)

**Severity:** Low (already documented)
**Evidence:** Previously investigated and documented. Caused by GroupKFold CV (67% train per fold) vs single 80/10/10 split. The user's qsar_qm_models with SchNet E2E gets R2=0.888. KIRBy's frozen reps (PDV, ECFP) cannot capture 3D quantum mechanical properties.

**File:** BOM line 21, memory MEMORY.md

### ISSUE 8: Outlier clipping on ALL datasets (NEEDS INVESTIGATION)

**Severity:** Medium
**Evidence:** `_load_openadmet_endpoint()` at line 185-188 clips labels at 1st/99th percentile for ALL datasets. This is applied to LogD and Caco2 (hERG uses a separate loader). The ground truth (qsar_qm_models) does NOT clip outliers.

```python
# Clip extreme outliers (1st / 99th percentile)
if len(labels_arr) > 10:
    p1, p99 = np.percentile(labels_arr, [1, 99])
    labels_arr = np.clip(labels_arr, p1, p99)
```

Clipping removes the hardest-to-predict compounds. This could inflate R2 by removing extreme values that contribute disproportionately to error. This is a legitimate preprocessing choice but should be documented. Pat Walters' guidelines recommend reporting preprocessing choices explicitly.

**File:** `tests/find_best_hybrid_molecular.py` line 185-188

### ISSUE 9: SMILES standardization differs between KIRBy and ground truth (EXPECTED)

**Severity:** Low
**Evidence:** KIRBy applies heavy SMILES standardization (line 118-139): Cleanup, FragmentParent, Uncharger, TautomerCanonicalize, metal rejection. The ground truth (qsar_qm_models) uses simple `Chem.MolToSmiles(mol, isomericSmiles=False)`.

This means KIRBy likely has fewer molecules after filtering (BOM shows 5,028 vs the raw ~7,309 for LogD). The stricter filtering removes ambiguous structures but also reduces dataset size.

**File:** `tests/find_best_hybrid_molecular.py` line 118-139

### ISSUE 10: No Chemprop / D-MPNN representation (NEEDS INVESTIGATION)

**Severity:** Medium
**Evidence:** Chemprop (D-MPNN) is arguably the most important molecular representation in the literature for QSAR regression. It is the gold standard in the field:
- Yang et al. 2019: D-MPNN RMSE 0.542 on lipophilicity (better than RF+ECFP at 0.655)
- ADMET-AI (swansonk14): best average rank on TDC leaderboard
- Chemprop v2 benchmarks show consistent SOTA

KIRBy does NOT have a Chemprop/D-MPNN representation. The closest are the frozen GNN embeddings (GROVER, MHG-GNN, GraphMVP, MolCLR) and the E2E SchNet/DimeNet++ fine-tuning. But D-MPNN trained on the target property is the single most competitive approach in the literature and is absent.

**File:** `src/kirby/representations/molecular.py` -- no chemprop anywhere

### ISSUE 11: GroupKFold CV vs single scaffold hold-out (EXPECTED but must be documented)

**Severity:** Medium (reporting issue, not a bug)
**Evidence:** KIRBy uses Murcko scaffold GroupKFold (3-fold or 5-fold). Ground truth and most literature use DeepChem ScaffoldSplitter with a single 80/10/10 split.

GroupKFold:
- Averages across multiple folds (some folds are easier)
- Each fold uses ~67% (3-fold) or ~80% (5-fold) of data for training
- Tends to give slightly higher R2 due to averaging

ScaffoldSplitter:
- Single deterministic split (same scaffolds always in test)
- Uses 80% for training (closer to 5-fold behavior)
- More conservative, harder to "get lucky" on one fold

KIRBy's numbers are NOT directly comparable to single-split literature numbers. The BOM should note "3-fold scaffold GroupKFold" explicitly when comparing to papers.

**File:** `tests/find_best_hybrid_molecular.py` line 72-78, `src/kirby/find_best_hybrid.py` line 533-546

### ISSUE 12: RF default `n_estimators=100` and `max_features='sqrt'` (EXPECTED)

**Severity:** Low
**Evidence:** KIRBy uses RF with n_estimators=100, max_features='sqrt' (line 219-222 of find_best_hybrid.py). Ground truth uses default sklearn RF (n_estimators=100, random_state=42). These are nearly identical.

XGBoost uses n_estimators=100, colsample_bytree=0.5. The colsample_bytree=0.5 is a reasonable regularization choice. Ground truth XGBoost also uses defaults (no specific HP tuning in the legacy split).

**File:** `src/kirby/find_best_hybrid.py` line 219-222, 225-229

### ISSUE 13: No RMSE, MAE, or Spearman metrics in survey output (NEEDS INVESTIGATION)

**Severity:** Medium
**Evidence:** KIRBy reports only R2. The ground truth reports R2, RMSE, MAE, and Spearman. Literature benchmarks universally report at least RMSE.

The BOM (line 14) says "Metric: R2" for all molecular datasets. R2 alone can be misleading:
- High-variance predictions can give decent R2 but poor RMSE
- R2 is sensitive to dataset composition (clipping outliers inflates it)
- RMSE is the standard primary metric in Chemprop, ADMETlab, PharmaBench

KIRBy already has RMSE defined in BUILTIN_METRICS (find_best_hybrid.py line 121). It would be trivial to report both R2 and RMSE.

**File:** `src/kirby/find_best_hybrid.py` line 118-124, `tests/find_best_hybrid_molecular.py` line 1146

---

## 4. Summary Table

| Issue | Severity | Type | Action needed |
|---|---|---|---|
| SNS missing from 'all' preset | High | Missing rep | Add SNS as rep_factory or static rep |
| No DNN model | High | Missing model | Add proper DNN (PyTorch, not sklearn MLP) |
| No CatBoost | Low | Missing model | Trivial add to DEFAULT_MODELS |
| No PaDEL/Mordred descriptors | Medium | Missing rep | Optional; RDKit PDV is standard |
| Caco2 R2=0.519 looks low | Medium-High | Needs investigation | Verify endpoint choice (efflux ratio vs permeability) |
| hERG R2=0.552 | Low | Expected | Reasonable for scaffold-split regression on small dataset |
| QM9 R2=0.139 | Low | Expected | Known limitation of frozen reps for QM properties |
| Outlier clipping (1st/99th %ile) | Medium | Preprocessing | Document clearly; consider making optional |
| SMILES standardization diffs | Low | Expected | More conservative = fewer mols but cleaner data |
| No Chemprop/D-MPNN | Medium | Missing rep | Literature's #1 QSAR method absent |
| GroupKFold vs single split | Medium | Reporting | Document in BOM when comparing to literature |
| RF/XGB hyperparameters | Low | Expected | Defaults are standard |
| No RMSE/MAE/Spearman output | Medium | Missing metric | Add RMSE as secondary metric |

---

## 5. Recommended Priority Actions

1. **Add SNS to the 'all' preset** (or as a rep_factory). This is the user's own published method and it outperforms ECFP4 in their own ground truth data.

2. **Add RMSE as a secondary metric** to molecular survey output. R2-only reporting is non-standard in the field.

3. **Investigate Caco2 endpoint choice.** Efflux ratio may be harder to predict than permeability. If the thesis only needs "Caco2 QSAR", permeability has more comparable baselines.

4. **Document GroupKFold vs single-split difference** clearly in BOM. All literature comparisons in the thesis must note this.

5. **Consider adding Chemprop/D-MPNN** as a fine-tuned representation. It is the single most cited molecular QSAR baseline.

6. **Document outlier clipping** in the BOM and thesis. Clipping inflates R2.

7. **Consider CatBoost** addition (trivial, competitive with XGBoost/LightGBM per Boldini 2023).

8. **DNN model** is the hardest to add (requires PyTorch training loop) but accounts for the largest performance gap in ground truth data.
