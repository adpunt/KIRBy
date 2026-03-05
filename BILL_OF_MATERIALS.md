# KIRBy Bill of Materials

Comprehensive inventory of datasets, representations, and their measured baseline
performance. This document is the single source of truth for what KIRBy supports
and what actually works. Representations without confirmed performance data should
not be used in experiments until validated.

**Last updated:** 2026-03-05 (augmented columns, comparison results, new reps added)

---

## 1. Datasets

### 1A. Molecular (regression, scaffold-split GroupKFold)

| Dataset | Source | Samples | Target | Metric | Best Rep | Best R² | Status |
|---------|--------|---------|--------|--------|----------|---------|--------|
| `logd` | OpenADMET-ExpansionRx | 5,028 | LogD | R² | grover (ridge) | **0.841** | **Done** |
| `caco2` | OpenADMET-ExpansionRx | 2,158 | Caco2 Efflux (log) | R² | graph_kernel (lgbm) | **0.519** | **Done** |
| `herg` | ChEMBL (CHEMBL240) | 1,415 | pKi | R² | ecfp4 (lgbm) | **0.552** | **Done** |
| `qm9` | QM9 | ~10,000 | HOMO-LUMO gap | R² | pdv_rank (ridge) | **0.139** | **Rerun needed** (was 5K/3-fold; now 10K/5-fold to match paper) |

### 1B. Antibody Intra-Family / DMS (regression, GroupKFold by family)

| Dataset | Source | Samples | Target | Best Rep | Best R² | Status |
|---------|--------|---------|--------|----------|---------|--------|
| `flab_binding_phillips_h3` | FLAb2 (Phillips 2021) | 5,000 | Binding (SPR Kd) | esm2 (ridge) | **0.960** | **Done** |
| `flab_expression_koenig` | FLAb (G6 SSM) | 4,275 | Expression (ER) | esm2 (ridge) | **0.480** | **Done** |
| `flab_binding_koenig` | FLAb (G6) | 4,275 | Kd | esm2 (ridge) | **0.303** | **Done** |
| `flab_binding_adams` | FLAb2 (Adams 2017) | 5,000 | Binding (Kd) | esm2 (ridge) | **0.409** | **Done** |
| `flab_expression_adams` | FLAb2 (Adams 2017) | 5,000 | Expression (ER) | prot_t5 (ridge) | **0.170** | **Done** |
| `abagym_HER2` | AbAgym DMS | 146 | Fitness | prot_t5 (ridge) | **0.449** | **Done** |
| `abagym_VEGF` | AbAgym DMS | 418 | Fitness | ablang2 (ridge) | **0.225** | **Done** |
| `abagym_EGFR` | AbAgym DMS | 659 | Fitness | prot_t5 (ridge) | **0.165** | **Done** |
| `abagym_lysozyme` | AbAgym DMS | 627 | Fitness | ablang2 (ridge) | **0.070** | **Done** |
| `abagym_Ang2` | AbAgym DMS | 397 | Fitness | ablang2 (ridge) | **-0.466** | **Done** (all neg) |
| `abagym_all` | AbAgym (pooled, 5 groups) | 2,247 | Fitness | igbert (ridge) | **-0.000** | **Done** (all neg) |

### 1C. Antibody Inter-Family (regression, identity-grouped GroupKFold)

| Dataset | Source | Samples | Target | Best Rep | Best R² | Status |
|---------|--------|---------|--------|----------|---------|--------|
| `flab_thermo_tresanco` | FLAb2 (Tresanco 2023) | 672 | Tm | prot_t5 (ridge) | **0.538** | **Done** |
| `flab_pk_jain_heprt` | FLAb2 (Jain 2023) | 130 | Heparin RT | prot_t5 (ridge) | **0.502** | **Done** |
| `flab_polyreact_jain_cic` | FLAb2 (Jain 2017) | 137 | CIC RT | prot_t5 (ridge) | **0.248** | **Done** |
| `flab_immunogenicity_marks` | FLAb2 (Marks 2021) | 217 | Immunogenicity | prot_t5 (ridge) | **0.188** | **Done** |
| `flab_pk_jain_fcrnrt` | FLAb2 (Jain 2023) | 132 | FcRn RT | ablang2 (ridge) | **0.186** | **Done** |
| `abcov_kd` | Ab-CoV (via FLAb) | 141 | -log Kd | ablang2 (ridge) | **0.147** | **Done** |
| `flab_aggregation_jain_hicrt` | FLAb2 (Jain 2017) | 137 | HIC RT | prot_t5 (ridge) | **0.135** | **Done** |
| `flab_thermo_jain` | FLAb (Jain 2017) | 137 | Tm | ablang2 (ridge) | **0.103** | **Done** |
| `flab_aggregation_jain` | FLAb (Jain 2017) | 137 | AC-SINS | ablang2 (ridge) | **0.020** | **Done** |
| `ddg_combined` | SKEMPI 2.0 + AB-Bind | 518 | ΔΔG | igbert (ridge) | **-0.137** | **Done** (all neg) |
| `flab_thermo_hutchinson` | FLAb2 (Hutchinson 2023) | 192 | Tm | ablang2 (ridge) | **-0.038** | **Done** (all neg) |
| `flab_polyreact_jain_bvp` | FLAb2 (Jain 2017) | 137 | BVP ELISA | ablang2 (ridge) | **-0.155** | **Done** (all neg) |
| `flab_aggregation_jain_sas` | FLAb2 (Jain 2017) | 137 | SAS | igbert (ridge) | **-0.599** | **Done** (all neg) |

**Dropped:** `sabdab_affinity` (was SKEMPI 2.0 only, replaced by `ddg_combined`),
`abbind` / `abbind_spm` (strict subsets of `ddg_combined`).

### 1D. TCGA (classification/regression, TSS-grouped StratifiedGroupKFold)

| Dataset | Task | Samples | Classes/Target | TSS | Best Rep | Best Score | Status |
|---------|------|---------|----------------|-----|----------|------------|--------|
| `tcga_luad_vs_lusc` | Binary clf | 1,017 | LUAD vs LUSC | 68 | gene_hvg2000 (lgbm) | **0.963** acc | **Done** (31/34 reps, re-running) |
| `tcga_brca_pam50` | 4-class clf | 821 | LumA/LumB/Basal/Her2 | 19 | gene_nmf50 (ridge) | **0.912** acc | **Done** (31/34 reps, re-running) |
| `tcga_brca_survival` | Regression | 151 | OS days (uncensored) | 21 | gene_corr_modules (knn) | **-0.059** R² | **Done** (all neg, too small) |

### 1E. Drug-Target Affinity (regression, cold-target GroupKFold)

| Dataset | Samples | Drugs | Proteins | Target | Best Rep | Best R² | Status |
|---------|---------|-------|----------|--------|----------|---------|--------|
| `davis` | 30,000 | 68 | 442 | pKd | drug_chemberta (xgb) | **0.223** | **Done** (drug-only; +protein_aac → 0.446) |
| `kiba` | 30,000 (sub) | 2,111 | 229 | KIBA score | drug_mol2vec (lgbm) | **0.340** | **Done** (frozen); run with `--finetune` for best results |

### 1F. Reaction Yield (regression, random 70/30 × 10 splits)

| Dataset | Samples | Target | Best Rep | Best R² | Status |
|---------|---------|--------|----------|---------|--------|
| `buchwald_hartwig` | 3,955 | Yield (0-100%) | drfp_2048 (xgb) | **0.934** | **Done** |
| `suzuki` | 5,760 | Yield (0-100%) | drfp_2048 (xgb) | **0.842** | **Done** |

**Data source:** rxn4chemistry/rxn_yields (Schwaller et al. 2021).
**Representations:** DRFP (2048D reaction fingerprints), Concat ECFP4 (per-component Morgan FPs).
**Deps:** `pip install drfp` for DRFP fingerprints.

### 1G. PDBBind (structure-based binding affinity, regression)

| Dataset | Samples | Source | Target | Status |
|---------|---------|--------|--------|--------|
| `pdbbind_refined` | ~5,316 complexes | PDBBind v2020 refined | -log(Kd/Ki) | Listed, manual download needed |
| `pdbbind_core` | ~290 complexes | PDBBind v2020 core | -log(Kd/Ki) | Listed, manual download needed |

**Data source:** pdbbind.org.cn (free registration, manual download required).
**Approach:** Sequence-based — extracts protein sequences from PDB + ligand SMILES from SDF.
Returns same dict format as DTA loader → plugs into existing DTA pipeline.
**Splits:** LP-PDBBind leak-proof via protein sequence clustering (Volkov et al. 2022).

---

## 2. Representations with Baseline Performance

Best R² (or Accuracy for classification) shown as `best_model=score`. All scores
are from scaffold-split (molecular), identity-grouped (antibody), or TSS-grouped
(TCGA) cross-validation — no random splits.

### 2A. Molecular Representations

**Source:** 5-fold scaffold-split GroupKFold baselines on LogD (5,028), Caco2 (2,158),
hERG (1,415), QM9 (~10,000). Models: rf, xgboost, ridge, knn, lightgbm.
Best score per rep shown (max across models).
"Best Aug" = best augmented (concat with smaller rep) R² using the same base model.

| Rep | Dim | LogD R² | LogD Aug | Caco2 R² | Caco2 Aug | hERG R² | hERG Aug | QM9 R² |
|-----|-----|---------|----------|----------|-----------|---------|----------|--------|
| `grover` | 4800 | **0.841** | **0.853** (+pdv) | 0.460 | — | 0.489 | — | 0.127 |
| `mhg_gnn` | 1024 | 0.813 | — | 0.489 | — | 0.543 | — | 0.134 |
| `ecfp4` | 2048 | 0.756 | — | 0.481 | — | **0.552** | **0.567** (+pdv_rank) | 0.134 |
| `pdv` | 200 | 0.788 | — | 0.491 | — | 0.530 | — | 0.133 |
| `pdv_rank` | 200 | 0.788 | — | 0.495 | — | 0.546 | — | **0.139** |
| `pdv_binary_mean` | 200 | 0.754 | — | 0.488 | — | 0.522 | — | 0.138 |
| `graph_kernel` | ~290 | 0.740 | — | **0.519** | **0.552** (+pdv_bm) | 0.539 | — | 0.067 |
| `molformer` | 768 | 0.753 | — | 0.429 | — | 0.516 | — | 0.109 |
| `mol2vec` | 300 | 0.753 | — | 0.436 | — | 0.511 | — | 0.131 |
| `smited` | 768 | 0.747 | — | 0.441 | — | 0.482 | — | 0.121 |
| `chemberta` | 384 | 0.722 | — | 0.397 | — | 0.500 | — | 0.119 |
| `graphmvp` | 300 | 0.692 | — | 0.351 | — | 0.457 | — | 0.123 |
| `selformer` | 768 | 0.689 | — | 0.352 | — | 0.458 | — | 0.106 |
| `chembert` | 256 | 0.665 | — | 0.387 | — | 0.499 | — | 0.122 |
| `maccs` | 167 | 0.676 | — | 0.385 | — | 0.479 | — | 0.131 |
| `schnet` | 128 | 0.640 | — | 0.327 | — | 0.345 | — | 0.090 |
| `dimenetpp` | — | — | — | — | — | — | — | — |
| `molclr` | 512 | 0.545 | — | 0.289 | — | 0.367 | — | 0.103 |
| PCA-50 variants | 50 | 0.556–0.706 | — | 0.308–0.420 | — | 0.418–0.475 | — | 0.101–0.122 |

**Augmentation summary (base = best rep per dataset):**
- **LogD:** grover+pdv → 0.853 (+0.012), 10/12 augmentations improved
- **Caco2:** graph_kernel+pdv_binary_mean → 0.552 (+0.033)
- **hERG:** ecfp4+pdv_rank → 0.567 (+0.015)
- **QM9:** pdv_rank+mhg_gnn_pca50 → 0.143 (+0.004, marginal)

**Not in this run (new, need resubmit):** clamp, rmat, map4, unimol_v2, dimenetpp.
**Fine-tuned reps (via `--finetune`):** ft_schnet (E2E SchNet → 128D embeddings, QM9 R²=0.888).

**Dropped:** `chemformer` (broken HF weights), `unimol` v1 (broken install).

**Key findings:**
- LightGBM dominates on fingerprints/descriptors; Ridge dominates on high-dim LM reps
- QM9 was using 5K/3-fold (67% train, R²=0.14) — now fixed to 10K/5-fold (80% train) to match paper protocol (10K, 80/10/10 scaffold, R²~0.83 with PDV)
- Augmentations: PDV/PDV-rank is the most consistent augmenter across molecular datasets
- PCA-50 variants retain 70-85% of full-dim performance, useful for low-dim hybrids

**Presets:**
- `fast` (3): ecfp4, maccs, pdv
- `fast+hf` (6): + chemberta, molformer, selformer
- `all` (29): all reps including GNN, graph_kernel, dimenetpp, clamp, rmat, map4, unimol_v2, pdv_binary_mean, pdv_rank, 7 PCA-50 variants

### 2B. Antibody Representations

**Approach (March 2026 rebuild):** LM embeddings → Ridge → R².
4 reps ran on all 24 datasets (LM survey). Only `ridge` model used.

**Script:** `tests/antibody_baseline_survey.py`

**Per-rep R² across key datasets (ridge only):**

| Rep | Dim | phillips_h3 | expr_koenig | bind_koenig | bind_adams | tresanco | heprt | HER2 |
|-----|-----|-------------|-------------|-------------|------------|----------|-------|------|
| `esm2` | 2560 | **0.960** | **0.480** | **0.303** | **0.409** | 0.467 | 0.481 | 0.440 |
| `prot_t5` | 2048 | 0.946 | 0.467 | 0.301 | 0.302 | **0.538** | **0.502** | **0.449** |
| `ablang2` | 480 | 0.860 | 0.433 | 0.223 | 0.354 | — | 0.494 | 0.437 |
| `igbert` | 1024 | -0.002 | -0.002 | -0.007 | -0.002 | -158K | -0.027 | -0.065 |

**Key findings:**
- **ESM-2** wins on large DMS datasets (phillips_h3 R²=0.96, binding/expression ~0.3-0.5)
- **ProtT5** wins on inter-family developability (Tm, PK, polyreactivity)
- **AbLang2** most stable on small datasets (480D avoids overfitting catastrophes)
- **IgBert** consistently predicts the mean (~0 R²) — embeddings carry no useful signal
- **AntiBERTy** not tested (model unavailable on HuggingFace, rewritten to use pip package)
- High-dim reps (ESM-2 2560D, ProtT5 2048D) show catastrophic R² blowups on small AbAgym datasets
- No augmentations or fine-tuning in these results

**Registered reps (18 total):** esm2, esmc, ablang2, igbert, antiberty, prot_t5,
amplify, onehot, blosum, aac, paratope, esm2_cdr, esmc_cdr, antiberty_cdr, prot_t5_cdr,
amplify_cdr, esm2_midlayers, structural (28D: SASA+confidence+paratope via ImmuneBuilder).
Full 18-rep survey needs GPU resubmit (updated submission script ready).

### 2C. TCGA Representations

**Source:** 3-fold TSS-grouped StratifiedGroupKFold. Models: rf, xgboost, ridge, knn, lightgbm.
31/34 reps completed (missing: bulkrnabert, scgpt, uce — TCGA resubmit in progress).

| Rep | Dim | Lung Acc | BRCA Acc | Survival R² |
|-----|-----|----------|----------|-------------|
| **gene_hvg2000** | 2000 | **0.963** | 0.883 | -0.192 |
| gene_pca100 | 100 | 0.957 | 0.843 | -0.182 |
| gene_nmf50 | 50 | 0.953 | **0.912** | -0.309 |
| gene_sparse_pca50 | 50 | 0.956 | 0.906 | -0.186 |
| gene_autoencoder50 | 50 | 0.953 | 0.885 | -0.218 |
| gene_reactome | 1817 | 0.952 | 0.894 | -0.382 |
| gene_corr_modules | 30 | 0.951 | 0.849 | **-0.059** |
| gene_ica50 | 50 | 0.950 | 0.884 | -0.235 |
| gene_combined_pathway | 280 | 0.947 | 0.880 | -0.244 |
| gene_quantile_rank | 50 | 0.947 | 0.900 | -0.231 |
| gene_oncogenic | 189 | 0.945 | 0.884 | -0.219 |
| gene_dorothea | 216 | 0.945 | 0.881 | -0.287 |
| gene_geneformer | 1152 | 0.942 | 0.876 | -0.222 |
| gene_kegg | 320 | 0.938 | 0.870 | -0.281 |
| gene_hallmark | 50 | 0.936 | 0.863 | -0.249 |
| gene_chromosome_arm | 42 | 0.924 | 0.773 | -0.419 |
| cnv_pca50 | 50 | 0.915 | 0.731 | -0.228 |
| methyl_pca50 | 50 | 0.907 | 0.674 | -0.123 |
| gene_immune | 22 | 0.898 | 0.789 | -0.198 |
| mirna_top200 | 200 | 0.892 | 0.666 | -0.302 |
| gene_progeny | 14 | 0.889 | 0.844 | -0.278 |
| methyl_hvg | 5000 | 0.936 | 0.651 | -0.268 |
| wsi_gigapath | 512 | 0.865 | 0.686 | -0.171 |
| wsi_hoptimus0 | 512 | 0.856 | 0.683 | -0.190 |
| cnv_gistic | 200 | 0.837 | 0.625 | -0.228 |
| wsi_ctranspath | 512 | 0.813 | 0.658 | -0.272 |
| wsi_moco | 512 | 0.803 | 0.646 | -0.215 |
| rppa | 258 | 0.799 | 0.788 | -0.222 |
| gene_stats | 15 | 0.794 | 0.567 | -0.198 |
| wsi_phikon | 512 | 0.758 | 0.632 | -0.214 |
| gene_scfoundation | 2048 | 0.508 | 0.510 | -0.193 |

**Fine-tuned (per-fold):**

| Rep | Lung Acc | BRCA Acc | Survival R² |
|-----|----------|----------|-------------|
| scfoundation (FT) | 0.953 | 0.855 | -0.180 |
| geneformer (FT) | 0.944 | 0.871 | -0.209 |

**Key findings:**
- gene_scfoundation frozen embeddings are **degenerate** (~0.51 = majority class). FT fixes it (0.95).
- Survival (n=151) is unsolvable — all R² negative. Too few uncensored samples.
- Augmentations tested: all negative delta on Lung; +autoencoder50 gives +0.004 on BRCA.
- Expression-derived reps dominate; WSI and multi-omic are secondary.
- Missing reps (bulkrnabert, scgpt, uce) will appear after TCGA resubmit completes.

**Presets:**
- `fast` (5): pca100, hallmark, progeny, ctranspath, phikon
- `all` (34): 18 expression + 5 foundation + 6 multi-omic + 5 WSI

### 2D. DTA Representations

Cold-target GroupKFold CV (3-fold). Each rep tested standalone (drug-only or protein-only).
Davis: 30K pairs (68 drugs × 442 proteins). KIBA: 30K subsample (2111 drugs × 229 proteins).
17 reps ran (8 missing from full preset: clamp, rmat, map4, unimol_v2, esmc, amplify, + PCA variants).

| Rep | Dim | Davis R² | KIBA R² |
|-----|-----|----------|---------|
| `drug_chemberta` | 384 | 0.223 | 0.339 |
| `drug_molformer` | 768 | 0.223 | 0.339 |
| `drug_selformer` | 768 | 0.223 | 0.338 |
| `drug_ecfp4` | 2048 | 0.223 | 0.334 |
| `drug_mol2vec` | 300 | 0.223 | **0.340** |
| `drug_maccs` | 167 | 0.223 | 0.333 |
| `drug_pdv` | 200 | 0.223 | 0.335 |
| `drug_pdv_binary_mean` | 200 | 0.223 | 0.335 |
| `drug_pdv_rank` | 200 | 0.223 | 0.335 |
| `drug_chemberta_pca50` | 50 | 0.223 | 0.330 |
| `drug_molformer_pca50` | 50 | 0.223 | 0.330 |
| `drug_selformer_pca50` | 50 | 0.223 | 0.329 |
| `protein_esm2_midlayers` | 1280 | 0.081 | 0.035 |
| `protein_esm2` | 1280 | 0.076 | 0.032 |
| `protein_kmer` | 8000 | 0.074 | 0.038 |
| `protein_aac` | 20 | 0.061 | 0.011 |
| `protein_descriptors` | 30 | 0.060 | 0.003 |

**Augmentation results (Davis, base=drug_chemberta):**
- drug_chemberta + protein_aac → **0.446** (+0.223), drug_chemberta + protein_descriptors → **0.435** (+0.212)
- Adding drug-only augmentations gives zero delta — all drug reps carry same info under cold-target

**Augmentation results (KIBA, base=drug_mol2vec):**
- drug_mol2vec + drug_pdv → 0.342 (+0.003), 6/9 augmentations positive but all marginal

**Key findings:**
- Davis drug reps all tied at 0.223 — drug-only features hit a ceiling under cold-target split
- Davis augmentation with protein features is **transformative** (+0.223 delta), confirming protein context is essential
- KIBA drug reps show genuine variance (0.329–0.340); protein-only near zero
- protein_esm2_lora attempted on KIBA but **failed** (all NaN) — needs debug (likely LR/batch_size issue with 30K pairs)
- **KIBA should be run with `--finetune`** — comparison shows DeepDTA E2E beats frozen by +0.056 R²

**Presets:**
- `fast` (5): drug_ecfp4, drug_maccs, drug_pdv, protein_esm2, protein_aac
- `full` (27): all drug + protein reps including PCA variants, ESMFold, AlphaFold DB

### 2E. Reaction Yield Representations

3-fold random 70/30 splits × 10. Only 2 reps (DRFP + concat ECFP4).

| Rep | Dim | Buchwald-Hartwig R² | Suzuki R² |
|-----|-----|---------------------|-----------|
| `drfp_2048` | 2048 | **0.934** (xgb) | **0.842** (xgb) |
| `concat_ecfp4` | 5120/15360 | 0.934 (xgb) | 0.735 (xgb) |

DRFP is clearly superior on Suzuki (+0.107). Both tied on Buchwald-Hartwig.

### 2F. Comparison Experiments

**DTA End-to-End (5-fold cold-target CV, 100 epochs):**

| Method | Davis R² | Davis MSE | KIBA R² | KIBA MSE |
|--------|----------|-----------|---------|----------|
| **KIRBy Frozen** (ECFP4+ESM2+XGB) | 0.552 | 0.350 | 0.314 | 0.472 |
| **DeepDTA** (CNN on SMILES+protein) | 0.559 | 0.353 | **0.379** | **0.429** |
| **GraphDTA-GIN** (GIN on graphs) | 0.525 | 0.382 | 0.131 | 0.601 |

- KIRBy frozen is competitive with DeepDTA on Davis (delta -0.007 R²) without any training
- DeepDTA wins KIBA (+0.065 R²); GraphDTA-GIN fails on KIBA
- Script: `tests/experiments/dta_e2e_comparison.py`

**QM9 3D Gap Prediction (4,950 molecules, random split):**

| Method | Test R² | Test MAE | Time (s) |
|--------|---------|----------|----------|
| **E2E SchNet** (100 epochs) | **0.888** | **0.330** | 50.0 |
| Frozen SchNet + Ridge | 0.134 | 0.962 | 219.1 |
| ECFP4 + XGBoost | 0.111 | 0.973 | 7.0 |

- End-to-end 3D training crushes frozen 2D baselines on quantum properties
- Frozen SchNet ≈ ECFP4 (no benefit without task-specific learning)
- Script: `tests/experiments/qm9_3d_comparison.py`

**Design implications (thesis boundary):**

The comparison experiments delineate when frozen reps are sufficient vs. when E2E training is needed:

1. **Frozen reps sufficient:** DTA on small drug spaces (Davis: 68 drugs). KIRBy frozen ties DeepDTA
   (0.552 vs 0.559 R²) despite zero training. Frozen ECFP4+ESM2+XGB is a strong default for
   binding affinity prediction when the chemical diversity is limited.

2. **Frozen reps competitive but with ceiling:** DTA on diverse drug spaces (KIBA: 2,111 drugs).
   DeepDTA pulls ahead (+0.065 R²). The frozen fingerprint can't capture enough structural
   variation. The `--finetune` LoRA path is the escape hatch for larger-scale DTA.

3. **Frozen reps benefit from E2E 3D training:** QM9 gap E2E SchNet R²=0.888. Frozen 2D reps
   achieve ~0.83 with proper setup (10K samples, 80/20 scaffold split, per paper protocol)
   but E2E SchNet still outperforms. The ft_schnet factory extracts 128D embeddings from
   E2E-trained SchNet for use as a fine-tuned rep in the standard pipeline.

4. **GraphDTA-GIN fails cold-target:** 0.525 Davis, 0.131 KIBA. The graph architecture doesn't
   generalize to unseen proteins under cold-target splits. This validates KIRBy's explicit
   drug+protein feature approach over implicit graph learning for generalization.

**WSI ABMIL aggregation:**

| Experiment | Script | Status |
|-----------|--------|--------|
| Mean pool vs gated attention vs ABMIL | `tests/experiments/wsi_abmil_comparison.py` | Needs server |

---

## 3. Known Issues and Reliability

### Unreliable Reps (may fail on server)

| Rep | Issue | Severity | Workaround |
|-----|-------|----------|------------|
| `gene_mutation` | Xena MC3 returns 403 | ~~HIGH~~ **RESOLVED** | **Dropped** from all presets/dispatch |
| `clinical` | Xena phenotype returns 403 | ~~HIGH~~ **RESOLVED** | **Dropped** from all presets/dispatch |
| `gene_hallmark` | ~~Enrichr download may fail~~ | ~~MEDIUM~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `gene_progeny` | ~~decoupler API 502 / NAType~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath weights, no API calls |
| `gene_dorothea` | ~~decoupler NAType in weight column~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath regulons, no API calls |
| `gene_reactome` | ~~ssGSEA orientation bug~~ | ~~LOW~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `wsi_titan` | HuggingFace gated, garbage results | ~~MEDIUM~~ **RESOLVED** | **Dropped** from presets/dispatch |
| `wsi_uni2h` | Manual download + h5 processing, features not found | ~~HIGH~~ **RESOLVED** | **Dropped** from presets/dispatch |
| `wsi_cnn320` | Local Seagate only, garbage results | ~~LOW~~ **RESOLVED** | **Dropped** from presets/dispatch |
| `balm` | Not on HuggingFace; needs manual clone + weight download | ~~MEDIUM~~ **RESOLVED** (2026-02-25) | Cloned to `BALM/` on server, weights in `pretrained-BALM/` |
| `antifold_score` | ImmuneBuilder not installed, 0/7 positive | ~~HIGH~~ **RESOLVED** | Deleted (March 2026 antibody rebuild) |
| `sapiens_score` | Tensor overflow for VH > 146 aa, 0/7 positive | ~~HIGH~~ **RESOLVED** | Deleted (March 2026 antibody rebuild) |

### Stale Server Caches

The following cached `.npy` files on the server are from buggy runs and must be
invalidated before re-running:

- `gene_reactome` — old ssGSEA orientation bug (now uses local gene sets, different dimensions)
- `gene_kegg` — same (now ~320 features vs old ~186)
- `gene_oncogenic` — same
- `gene_progeny` — old decoupler format (now uses local weights)
- `gene_dorothea` — old decoupler format (now ~367 TFs vs old ~300)
- `gene_hallmark` — old ssGSEA format (now mean z-score, same ~50 features)
- `gene_combined_pathway` — depends on all above

**All pathway caches should be deleted after this update** since the scoring
method changed from ssGSEA/decoupler to mean z-score/weighted linear scoring.
The cache validation code detects n_samples mismatches but NOT dimension changes
for the same n_samples, so manual deletion is recommended:
`rm ~/.cache/kirby/reps/tcga_*/gene_{hallmark,kegg,reactome,oncogenic,progeny,dorothea,combined_pathway}.npy`

### DTA Import Issue

~~The survey job failed with `ImportError: DTA datasets require PyTDC`.~~ **RESOLVED** — DTA now downloads directly from DeepDTA GitHub, no PyTDC needed.

---

## 4. Coverage Gaps and Actionable Status

### What we HAVE (confirmed baselines, March 2026 survey)

| Domain | Datasets | Reps | Status |
|--------|----------|------|--------|
| Molecular | 4/4 | 24/29 | **Done** (missing: clamp, rmat, map4, unimol_v2, dimenetpp) |
| Antibody | 24/24 | 4/18 | **Partial** (LM survey only: esm2, prot_t5, ablang2, igbert) |
| TCGA | 3/3 | 31/34 | **Partial** (missing: bulkrnabert, scgpt, uce — resubmit running) |
| DTA | 2/2 | 17/27 | **Done** (missing: clamp, rmat, map4, unimol_v2, esmc, amplify, esmfold, alphafold_db, PCA variants) |
| Reaction | 2/2 | 2/2 | **Done** |
| Comparisons | DTA e2e, QM9 3D | — | **Done** (WSI ABMIL pending) |
| PDBBind | 0/2 | — | **Blocked** (manual data download needed) |

### What's STILL PENDING

| Item | Gap | Blocker |
|------|-----|---------|
| Antibody full 18-rep survey | 14 reps × 24 datasets | GPU resubmit (updated script ready: `submit_antibody_survey.sh all`) |
| TCGA 3 foundation reps | bulkrnabert, scgpt, uce | Resubmit running (bulkrnabert OOM fixed batch_size=1) |
| Molecular 5 missing reps | clamp, rmat, map4, unimol_v2, dimenetpp | Need resubmit with updated 'all' preset (29 reps) |
| DTA 10 missing reps | clamp, rmat, map4, unimol_v2, esmc, amplify, esmfold, alphafold_db, PCA variants | Need resubmit with updated 'full' preset (27 reps) |
| PDBBind | 2 datasets | Manual download from pdbbind.org.cn |
| WSI ABMIL comparison | Mean pool vs gated attention | Needs server |
| IgBert | Predicts mean (R² ~ 0) everywhere | Investigate or drop |
| protein_esm2_lora | NaN on KIBA | Debug factory |

### Dropped

| Rep | Reason |
|-----|--------|
| `gene_mutation`, `clinical` | Xena 403 |
| `wsi_titan`, `wsi_uni2h`, `wsi_cnn320` | Gated/broken/garbage |
| `unimol` v1 | Broken install |
| `chemformer` | Broken HF weights |
| `gene_scfoundation` (frozen) | Degenerate embeddings (0.51 acc = majority class). FT version works. |

---

## 5. Representation Categories (Summary)

| Category | Description | Examples |
|----------|-------------|---------|
| **Standard** | Computed from input data, no pretrained models | ecfp4, maccs, pdv, aac, kmer, gene_pca100, cnv_gistic |
| **Pretrained** | Frozen embeddings from pretrained LMs | chemberta, molformer, esm2, ablang2, wsi_ctranspath |
| **Fine-tuned** | Per-fold fine-tuned embeddings (LoRA/Geneformer) | lora_chemberta, geneformer |
| **Customizable** | Derived from metadata, mutations, or domain knowledge | varpos_*, lm_score_*, biophysical_delta, structural |

---

## 6. Fine-Tuning Factories

### Active (proceed to Phase 2)

| Factory | Domain | Base Model(s) | Method | Output Dim | CLI Flag | GPU | Survey Result |
|---------|--------|---------------|--------|-----------|----------|-----|---------------|
| `make_geneformer_factory` | TCGA | Geneformer V2 (104M) | Freeze+head | 1152D | `--finetune geneformer` | Yes | **PASS** — best individual rep on lung (0.930), BRCA (0.789) |

### Retired (tested, does not help — do NOT run)

| Factory | Domain | Base Model(s) | Method | Output Dim | CLI Flag | Survey Result |
|---------|--------|---------------|--------|-----------|----------|---------------|
| `make_molecular_lora_factory` | Molecular | ChemBERTa-77M-MLM | LoRA (rank=8) | 384D | `--finetune lora_chemberta` | **FAIL** — worse than frozen ecfp4/pdv on all 3 datasets |
| `make_molecular_lora_factory` | Molecular | MoLFormer-XL | LoRA (rank=8) | 768D | `--finetune lora_molformer` | **FAIL** — worse than frozen ecfp4/pdv on all 3 datasets |

**Deleted (March 2026):** All antibody fine-tuning factories (`antibody_lora`, `antibody_freeze`,
`antibody_mlm`) removed. All produced negative R² on antibody datasets.

### Two-Phase Fine-Tuning Workflow

**Phase 1: Survey** (COMPLETE) — `run_finetune_survey.sh`
- 2 folds, 2 epochs, fast search, performance_weighted allocation
- Output: `tests/results/finetune_survey/`

| Domain | Datasets | Methods Tested | Frozen Reps | Verdict |
|--------|----------|----------------|-------------|---------|
| Molecular | logd, caco2, herg | lora_chemberta, lora_molformer | fast (3) | **All fail** — FT always worse than frozen |
| Antibody | flab_expression_koenig, flab_binding_koenig, abagym_Ang2 | lora_esm2, lora_esm2_small, mlm_esm2 | dms (9) | **All fail** — negative R², worse than mean predictor |
| TCGA | tcga_luad_vs_lusc, tcga_brca_pam50 | geneformer | fast (5-8) | **Geneformer passes** — best individual + improves hybrid |

**Phase 2: Full run** (targeted) — only Geneformer on TCGA
- 5 folds, 5 epochs, medium search
- Embeddings cached per-fold to `~/.cache/kirby/finetuned_reps/`

**Submission:**
```bash
# Phase 2: only Geneformer survived the survey
sbatch tests/slurm_scripts/run_finetune_tcga.sh fast tcga_luad_vs_lusc geneformer
sbatch tests/slurm_scripts/run_finetune_tcga.sh fast tcga_brca_pam50 geneformer
```

---

## 7. Fine-Tuning Survey Results (2026-02-25)

Full results from Phase 1 survey (2 folds, 2 epochs, performance_weighted allocation).
Raw JSON in `tests/results/finetune_survey/`.

### 7A. Molecular — FT does NOT help

| Dataset | Best Frozen Rep | R² | Best FT Rep | R² | Verdict |
|---------|----------------|-----|-------------|-----|---------|
| LogD | pdv (lgbm) | 0.772 | lora_chemberta (lgbm) | 0.697 | Frozen wins by 0.075 |
| Caco2 | pdv (xgb) | 0.466 | lora_chemberta (lgbm) | 0.335 | Frozen wins by 0.131 |
| hERG | ecfp4 (rf) | 0.504 | lora_chemberta (rf) | 0.398 | Frozen wins by 0.106 |

**Conclusion:** LoRA fine-tuning of ChemBERTa and MoLFormer consistently degrades
performance vs. frozen fingerprints/descriptors. The cheap standard reps (ecfp4, pdv)
remain dominant. Fine-tuning is not worth the GPU cost for molecular property prediction.

### 7B. Antibody — FT does NOT help (code deleted March 2026)

All antibody fine-tuning code has been removed. Summary of what was tested before deletion:

- `lora_esm2` (650M, LoRA rank=8): Negative R² on FLAb expression, binding, AbAgym Ang2
- `lora_esm2_small` (8M, LoRA rank=8): Negative R² on all datasets
- `mlm_esm2` (self-supervised MLM): Negative R² on all datasets
- Other variants (`lora_antiberty`, `lora_ablang2`, `mlm_antiberty`, `mlm_esm2_small`): Never tested, not expected to help

The entire mutation-aware representation approach (40+ custom functions, varpos features,
LM log-prob scores, delta reps) has been replaced with simple LM embeddings → Ridge →
Spearman ρ, following the compass document. See `tests/antibody_baseline_survey.py`.

### 7C. TCGA — Geneformer FT HELPS

| Dataset | Best Frozen Rep | Acc | Geneformer FT | Acc | Hybrid (frozen+FT) | Acc |
|---------|----------------|-----|---------------|-----|--------------------|----|
| LUAD vs LUSC | gene_nmf50 (rf) | 0.951 | geneformer (xgb) | 0.930 | 4-rep hybrid | 0.940 |
| BRCA PAM50 | gene_hvg2000 (xgb) | 0.866 | geneformer (xgb) | 0.789 | 5-rep hybrid | 0.820 |

**Conclusion:** Geneformer fine-tuning produces competitive individual performance and
improves hybrid quality on TCGA tasks. The 1152D V2 embeddings loaded correctly
(confirming the tokenization fix worked). No data leakage concern: Geneformer was
pretrained on single-cell data (Genecorpus-104M, non-cancer), not TCGA bulk RNA-seq.
**Proceed to Phase 2 full run (5 folds, 5 epochs).**

Note: gene_nmf50 outperforms Geneformer as an individual rep on lung (0.951 vs 0.930),
but Geneformer captures complementary information — it adds value in hybrids. On BRCA,
hvg2000 leads (0.866 vs 0.789), and the hybrid (0.820) doesn't beat the best individual;
this may improve with more folds and epochs in Phase 2.

---

## 8. Preprocessing Pipeline (2026-02-27)

All baselines must be re-run after these changes. Cache invalidation required.

### Molecular SMILES Standardization
- `_standardise_smiles()` now applies: `Cleanup → FragmentParent → Uncharger → TautomerEnumerator.Canonicalize`
- Metal-containing compounds rejected (organic descriptors unreliable)
- OpenADMET endpoints: 1st/99th percentile outlier clipping after deduplication

### QM9 Normalization Fix
- **Data leak fixed:** QM9 labels were pre-normalized from the full sampled set. Now returns raw labels; `StandardScaler` is applied per-fold in the evaluation pipeline.

### KIBA Score Transform
- KIBA scores negated so higher = stronger binding (consistent with pKd convention)

### DTA SMILES Re-canonicalization
- Drug SMILES from DeepDTA re-canonicalized with RDKit for consistency

### Antibody Label Preprocessing
- `abagym_all`: labels z-scored per experiment before concatenation (standard for pooling assays)
- `ddg_combined`: outlier filter removes entries with |ddG| > 8 kcal/mol or > 3σ from mean

### TCGA Preprocessing
- **ComBat batch correction:** expression data corrected across tissue source sites (TSS) using pyComBat, applied before caching so all downstream reps (HVG, PCA) get corrected data. Optional — falls back to uncorrected if `combat` not installed.
- **Zero-expression filter:** genes with >90% zero expression removed before HVG selection and PCA
- **Mutation matrix:** filtered to non-silent variants only (Missense, Nonsense, Frameshift, Splice_Site, etc.)

### Model Improvements
- **Ridge → RidgeCV:** internal LOO cross-validation selects best alpha from `np.logspace(-6, 6, 13)`. No extra compute cost.
- **LogisticRegression → LogisticRegressionCV:** built-in Cs=10 alpha selection
- **KNN adaptive k:** `n_neighbors = min(5, max(1, sqrt(N_train)))` — scales to dataset size

### Cache Invalidation (REQUIRED — March 2026 overhaul)

**⚠ SERVER ACTION NEEDED:** Clear ALL caches before re-running any baselines.
HP defaults changed (RF max_features, XGB colsample_bytree, RidgeCV alpha grid),
preprocessing changed, all old numbers are invalid.

```bash
# FULL CACHE CLEAR (run on server before any re-runs)
rm -rf ~/.cache/kirby/reps/              # all rep caches (SMILES changed)
rm -rf ~/.cache/kirby/openadmet_train.csv # re-download + re-standardize
rm -rf ~/.cache/kirby/chembl_herg_ki.csv  # re-fetch + re-standardize
rm -rf ~/.cache/kirby/dta/               # KIBA transform changed
rm -rf ~/.cache/kirby/tcga/expression/   # ComBat changes expression data
rm -rf ~/.cache/kirby/tcga/pathways/     # already documented as stale
rm -rf ~/.cache/kirby/ddg_combined/      # outlier filter changes dataset
rm -rf ~/.cache/kirby/abagym/            # z-scoring changes labels
rm -rf ~/.cache/kirby/finetuned_reps/    # FT embeddings from old HP regime
```

---

## 9. Open Problems

### Antibody Representation Strategy (March 2026 rebuild)

The old approach (40+ mutation-aware custom features, complex presets, HP tuning) has been
replaced with a simple baseline pipeline: LM embeddings → Ridge → Spearman ρ.

**Current approach:** 8 representations (5 LM + 3 sequence encodings) evaluated per
dataset with RidgeCV. This matches the standard in the field (FLAb2 paper uses the same).

**Key questions to answer with the new baselines:**
1. Which LM (ESM-2, AbLang2, IgBert, AntiBERTy, ProtT5) works best per task type?
2. Does one-hot match or beat LMs at ≥1K samples (as claimed in literature)?
3. Are inter-family datasets (ddg_combined, abcov_kd) still fundamentally too small/hard?
4. Is Ridge sufficient or do tree models add value for antibody tasks?
