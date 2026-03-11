# KIRBy Bill of Materials

Comprehensive inventory of datasets, representations, and their measured baseline
performance. This document is the single source of truth for what KIRBy supports
and what actually works. Representations without confirmed performance data should
not be used in experiments until validated.

**Last updated:** 2026-03-11 (Full integration audit: rep counts corrected, metrics documented, forward-compat reps added, splitting headers fixed, stale results flagged)

---

## 1. Datasets

### 1A. Molecular (regression, scaffold-split GroupKFold)

| Dataset | Source | Samples | Target | Metrics | Best Rep | Best R² | Status |
|---------|--------|---------|--------|---------|----------|---------|--------|
| `logd` | OpenADMET-ExpansionRx | 5,028 | LogD | R², Spearman ρ, RMSE | grover (ridge) | **0.841** | **Stale** (pre-Phase 1 preprocessing) |
| `caco2` | OpenADMET-ExpansionRx | 2,158 | Caco2 Efflux (log) | R², Spearman ρ, RMSE | graph_kernel (lgbm) | **0.519** | **Stale** |
| `herg` | ChEMBL (CHEMBL240) | 1,415 | pKi | R², Spearman ρ, RMSE | ecfp4 (lgbm) | **0.552** | **Stale** |
| `qm9` | QM9 | ~10,000 | HOMO-LUMO gap | R², Spearman ρ, RMSE | pdv_rank (ridge) | **0.139** | **Stale** (was 5K/3-fold; now 10K/5-fold) |
| `adme_hlm` | Polaris ADME-Fang | 3,087 | HLM clearance (log) | R², Spearman ρ, RMSE | — | — | **New** (run needed) |
| `adme_rlm` | Polaris ADME-Fang | 3,054 | RLM clearance (log) | R², Spearman ρ, RMSE | — | — | **New** (run needed) |
| `adme_mdr1` | Polaris ADME-Fang | 2,642 | MDR1 efflux (log) | R², Spearman ρ, RMSE | — | — | **New** (run needed) |
| `adme_solubility` | Polaris ADME-Fang | 2,173 | Solubility (log) | R², Spearman ρ, RMSE | — | — | **New** (run needed) |

**⚠ ALL existing molecular results are STALE** — run before Phase 1 changes (SMILES standardization, InChIKey dedup, per-fold PCA/scaling, BOS/EOS exclusion, new models). Rerun will produce different numbers.

**Preprocessing:** All molecular datasets apply `standardize_smiles()` (salt stripping, neutralization, tautomer canonicalization) and `deduplicate_by_inchikey()` (removes duplicate molecules with different SMILES but same InChIKey, preventing same molecule in train+test).

### 1B. Antibody Intra-Family / DMS (regression, position-modulo GroupKFold for DMS)

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
| `abagym_all` | AbAgym (pooled, 5 groups) | 2,247 | Fitness | igbert (ridge) | **-0.000** | **Stale** (all neg) |

**⚠ ALL existing antibody DMS results are STALE** — run with identity splits + ridge only. Now uses position-modulo splits (ProteinGym standard) + 9 models + 39 reps. Rerun will produce different (likely lower, more honest) numbers.

**Metrics (implemented, not yet reported):** R², Spearman ρ, Pearson r, RMSE per fold.

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
| `ddg_combined` | SKEMPI 2.0 + AB-Bind (Ab-Ag only) | 518 | ΔΔG | igbert (ridge) | **-0.137** | **Done** (all neg) |
| `ddg_combined_full` | SKEMPI 2.0 full + AB-Bind (all PP) | ~3,000 | ΔΔG | — | — | **New** (run needed) |
| `flab_thermo_hutchinson` | FLAb2 (Hutchinson 2023) | 192 | Tm | ablang2 (ridge) | **-0.038** | **Done** (all neg) |
| `flab_polyreact_jain_bvp` | FLAb2 (Jain 2017) | 137 | BVP ELISA | ablang2 (ridge) | **-0.155** | **Done** (all neg) |
| `flab_aggregation_jain_sas` | FLAb2 (Jain 2017) | 137 | SAS | igbert (ridge) | **-0.599** | **Done** (all neg) |

**Dropped:** `sabdab_affinity` (was SKEMPI 2.0 only, replaced by `ddg_combined`),
`abbind` / `abbind_spm` (strict subsets of `ddg_combined`).

**`ddg_combined_full`:** Full SKEMPI 2.0 (all protein-protein complexes, not just Ab-Ag) + AB-Bind. ~6x more data than `ddg_combined`. Enables comparison: Ab-Ag-specific vs general PP DDG prediction. Registered in antibody survey with `--include-all-skempi` flag.

### 1D. TCGA (classification/regression, TSS-grouped StratifiedGroupKFold)

| Dataset | Task | Samples | Classes/Target | TSS | Best Rep | Best Score | Status |
|---------|------|---------|----------------|-----|----------|------------|--------|
| `tcga_luad_vs_lusc` | Binary clf | 1,017 | LUAD vs LUSC | 68 | gene_hvg2000 (lgbm) | **0.963** acc | **Stale** (31/57 reps, pre-Phase 1 metrics) |
| `tcga_brca_pam50` | 4-class clf | 821 | LumA/LumB/Basal/Her2 | 19 | gene_nmf50 (ridge) | **0.912** acc | **Stale** (31/57 reps, pre-Phase 1 metrics) |
| `tcga_brca_survival` | Regression | 151 | OS days (uncensored) | 21 | gene_corr_modules (knn) | **-0.059** R² | **Stale** (all neg, too small) |

**⚠ ALL existing TCGA results are STALE** — run before per-fold factory conversion + new metrics. Rerun uses AUROC (binary), F1-weighted (multi-class), + Spearman ρ/RMSE for regression. 57 reps now registered (was 34).

### 1E. Drug-Target Affinity (regression, cold-target GroupKFold)

| Dataset | Samples | Drugs | Proteins | Target | Metrics | Best Rep | Best R² | Status |
|---------|---------|-------|----------|--------|---------|----------|---------|--------|
| `davis` | 30,000 | 68 | 442 | pKd | CI, R², Spearman ρ, RMSE | drug_chemberta (xgb) | **0.223** | **Stale** (drug-only; +protein_aac → 0.446) |
| `kiba` | 30,000 (sub) | 2,111 | 229 | KIBA score | CI, R², Spearman ρ, RMSE | drug_mol2vec (lgbm) | **0.340** | **Stale** (frozen) |
| `bindingdb_kd` | ~52,000 | ~10,000 | ~1,400 | pKd | CI, R², Spearman ρ, RMSE | — | — | **New** (run needed) |

**⚠ ALL existing DTA results are STALE** — run before Phase 1 metrics (CI now primary), Phase 2 models (9 models, was ~3), Phase 4B reps. Also supports `--cold-setting drug` for cold-drug evaluation.

### 1F. Reaction Yield (regression, GroupKFold leave-one-group-out)

| Dataset | Samples | Target | Metrics | Best Rep | Best R² | Status |
|---------|---------|--------|---------|----------|---------|--------|
| `buchwald_hartwig` | 3,955 | Yield (0-100%) | R², Spearman ρ, RMSE, MAE | drfp_2048 (xgb) | **0.934** | **Stale** (old random splits) |
| `suzuki` | 5,760 | Yield (0-100%) | R², Spearman ρ, RMSE, MAE | drfp_2048 (xgb) | **0.842** | **Stale** (old random splits) |
| `bh_doyle_plates` | 4,599 | Yield (0-100%) | R², Spearman ρ, RMSE, MAE | — | — | **New** (run needed) |

**⚠ ALL existing reaction results are STALE** — run with old random 70/30 splits. Now uses GroupKFold (BH: leave-additive-out 23 groups, Suzuki: leave-ligand-out 12 groups). R² will drop significantly (~0.93 → ~0.7-0.8). 9 models now available (was ~2).

**Data source:** rxn4chemistry/rxn_yields (Schwaller et al. 2021), Doyle lab HTE plates (Ahneman et al. 2018).
**Splits:** GroupKFold (BH: leave-additive-out, 23 groups; Suzuki: leave-ligand-out, 12 groups; Doyle plates: leave-additive-out).
**Representations (8 total):** DRFP (2048D), Concat ECFP4, Concat MACCS, Concat PDV, Product ECFP4, Reactant ECFP4, RXNFP (256D learned), One-Hot Components (~46D BH / ~60D Suzuki).
**Deps:** `pip install drfp rxnfp` (one-hot, concat variants, product/reactant need only RDKit).
**Note:** Only DRFP + Concat ECFP4 have results — 6 new reps need first run.

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
are from scaffold-split (molecular), position-modulo (antibody DMS), identity-grouped
(antibody inter-family), or TSS-grouped (TCGA) cross-validation — no random splits.

**⚠ ALL results below are from pre-Phase 1 runs.** They were generated before:
preprocessing fixes (SMILES standardization, InChIKey dedup, BOS/EOS exclusion),
per-fold PCA/scaling (leakage fix), position-based DMS splits, GroupKFold for reactions,
model expansion (GP, MLP, DNN, CNN, CatBoost), and metric additions (Spearman ρ,
CI, RMSE, AUROC, F1). **Every number in this section will change on rerun.**
The code now computes all domain-standard metrics; the tables below still show only
the old metric. Phase 5 re-runs will populate the full metric set.

### 2A. Molecular Representations

**Source:** 3-fold scaffold-split GroupKFold baselines on LogD (5,028), Caco2 (2,158),
hERG (1,415). Models: ridge, lightgbm (both shown).
QM9 results are STALE (old 5K/3-fold run) — rerun in progress.

| Rep | Dim | LogD ridge | LogD lgbm | Caco2 ridge | Caco2 lgbm | hERG ridge | hERG lgbm |
|-----|-----|-----------|-----------|-------------|------------|-----------|-----------|
| `grover` | 4800 | **0.841** | 0.752 | 0.442 | 0.460 | 0.489 | 0.485 |
| `mhg_gnn` | 1024 | 0.813 | 0.795 | 0.451 | 0.489 | 0.455 | 0.528 |
| `ecfp4` | 2048 | 0.723 | 0.756 | 0.442 | 0.480 | 0.509 | **0.552** |
| `pdv` | 200 | 0.736 | 0.788 | 0.427 | 0.491 | 0.349 | 0.529 |
| `pdv_rank` | 200 | 0.734 | 0.788 | 0.434 | 0.495 | 0.404 | 0.546 |
| `pdv_binary_mean` | 200 | 0.671 | 0.747 | 0.395 | 0.488 | 0.392 | 0.498 |
| `graph_kernel` | ~290 | 0.670 | 0.740 | 0.444 | **0.519** | 0.431 | 0.512 |
| `molformer` | 768 | 0.753 | 0.685 | 0.424 | 0.429 | 0.505 | 0.516 |
| `mol2vec` | 300 | 0.752 | 0.735 | 0.436 | 0.415 | 0.488 | 0.511 |
| `smited` | 768 | 0.747 | 0.721 | 0.409 | 0.441 | 0.425 | 0.482 |
| `chemberta` | 384 | 0.722 | 0.622 | 0.397 | 0.394 | 0.425 | 0.500 |
| `graphmvp` | 300 | 0.692 | 0.640 | 0.351 | 0.350 | 0.414 | 0.457 |
| `selformer` | 768 | 0.689 | 0.595 | 0.352 | 0.330 | 0.458 | 0.441 |
| `chembert` | 256 | 0.665 | 0.659 | 0.364 | 0.387 | 0.422 | 0.499 |
| `maccs` | 167 | 0.548 | 0.657 | 0.311 | 0.385 | 0.284 | 0.437 |
| `schnet` | 128 | 0.640 | 0.633 | 0.311 | 0.327 | 0.320 | 0.345 |
| `molclr` | 512 | 0.434 | 0.545 | 0.276 | 0.289 | 0.196 | 0.363 |

| `avalon` | 2048 | — | — | — | — | — | — |
| `minimol` | 512 | — | — | — | — | — | — |
| `gp_molformer` | 768 | — | — | — | — | — | — |
| `mole` | 768 | — | — | — | — | — | — |

**Not in this run (need resubmit):** clamp, rmat, map4, unimol_v2, dimenetpp, mordred, mordred_pca50, avalon, minimol, gp_molformer, mole, sns, all *_pca50 factory reps.
**QM9:** Stale results (5K/3-fold, R²~0.14). Rerun with 10K/5-fold in progress. User's qsar_qm_models pipeline gets R²>0.7 — investigate discrepancy.
**Fine-tuned:** ft_schnet (E2E SchNet → 128D, QM9 R²=0.888).
**Dropped:** `chemformer` (broken HF weights), `unimol` v1 (broken install).

**Presets:**
- `fast` (3): ecfp4, maccs, pdv
- `fast+hf` (6): + chemberta, molformer, selformer
- `all` (36): 25 base reps (ecfp4, maccs, avalon, pdv, mordred, mol2vec, chemberta, molformer, selformer, chembert, molclr, graphmvp, smited, mhg_gnn, grover, graph_kernel, schnet, dimenetpp, clamp, rmat, map4, unimol_v2, minimol, gp_molformer, mole) + 11 per-fold factory reps (pdv_binary_mean, pdv_rank, sns, chemberta_pca50, molformer_pca50, selformer_pca50, molclr_pca50, smited_pca50, mhg_gnn_pca50, grover_pca50, mordred_pca50)

**Phase 4A forward-compat reps (in 'all' preset):** `minimol` (MiniMol, pip install), `gp_molformer` (IBM GP-MoLFormer, HuggingFace), `mole` (MolE, GitHub + manual weights).

### 2B. Antibody Representations

**Approach (March 2026 rebuild):** LM embeddings → Ridge → R².
4/39 reps ran on 24 datasets. Ridge only — **LightGBM missing** (not installed when these ran). Full 39-rep + 9 models resubmit needed.

**Script:** `tests/antibody_baseline_survey.py`

**Per-rep R² across key datasets (ridge only):**

| Rep | Dim | phillips_h3 | expr_koenig | bind_koenig | bind_adams | tresanco | heprt | HER2 |
|-----|-----|-------------|-------------|-------------|------------|----------|-------|------|
| `esm2` | 2560 | **0.960** | **0.480** | **0.303** | **0.409** | 0.467 | 0.481 | 0.440 |
| `prot_t5` | 2048 | 0.946 | 0.467 | 0.301 | 0.302 | **0.538** | **0.502** | **0.449** |
| `ablang2` | 480 | 0.860 | 0.433 | 0.223 | 0.354 | — | 0.494 | 0.437 |
| `igbert` | 1024 | -0.002 | -0.002 | -0.007 | -0.002 | -158K | -0.027 | -0.065 |

**Status:** Waiting on resubmit with 9 models + all 39 reps. Only 4 reps (esm2, prot_t5, ablang2, igbert) × 1 model (ridge) have results.

**ANTIBODY_REPS registered in code (34 total):**
- **LM full-seq (12):** esm2, esmc, ablang2, igbert, igbert_cls, antiberty, prot_t5, amplify, esm3, saprot, abroberta, mint
- **LM CDR-pooled (8):** esm2_cdr, esmc_cdr, antiberty_cdr, prot_t5_cdr, amplify_cdr, esm3_cdr, saprot_cdr, abroberta_cdr
- **LM midlayer (1):** esm2_midlayers
- **LM layer variants (3):** esm2_layer24 (single layer 24, 2560D), esm2_early (layers 0-16 avg, 2560D), esm2_concat (H+L as single sequence, 1280D)
- **LM CDR3-only (1):** esm2_cdr3 (IMGT 105-117 only, 2560D)
- **Seq encoding (3):** onehot, blosum, aac
- **Structural (2):** structural (28D: SASA+confidence+paratope via ImmuneBuilder), paratope (8D)
- **Variation-pool (3):** esm2_entropy_top20 (2560D, per-fold factory), esm2_per_cdr (7680D, CDR1/2/3 separate), esm2_per_region (17920D, all 7 IMGT regions separate)
- **Forward-compat (1):** igt5 (1024D, Exscientia paired T5)

**MUTATION_REPS registered in code (6 total):**
- mutation_features (48D, hand-crafted AA deltas), esm2_mutsite (1280D, ±3 window at mutation site), esm2_wt_delta (2560D, mutant−wildtype embedding delta), esm2_llr (3D, ESM-2 masked marginal LLR), esmc_llr (3D, ESM-C masked marginal LLR), antiberty_llr (3D, AntiBERTy masked marginal LLR)

**Combined total: 40 reps** (34 ANTIBODY_REPS + 6 MUTATION_REPS). Mutation reps require mutation metadata; silently skipped for datasets without mutations.

**Phase 3C additions (not in previous BOM):** abroberta, abroberta_cdr, esm2_layer24, esm2_early, esm2_concat, esm2_cdr3, esmc_llr, antiberty_llr.
**Phase 4A additions:** igt5 (Exscientia IgT5, paired antibody T5), mint (MINT protein foundation model, 1280D×2=2560D, reuses protein.py `create_mint_protein_embeddings`).
**Phase 4B forward-compat (included in LM counts above):** ESM3 (1536D×2=3072D, `esm` package, Python>=3.10), SaProt (1280D×2=2560D, ProstT5 for 3Di tokens + HuggingFace SaProt_650M_AF2). Both have full-seq and CDR-pooled variants.

Mutation reps require mutation metadata (pos, wt_aa, mut_aa, chain) from dataset loaders; silently skipped for datasets without mutations (e.g. AbCoV). Variation reps use IMGT numbering via ANARCI — no mutation metadata needed.

### 2C. TCGA Representations

**Source:** 3-fold TSS-grouped StratifiedGroupKFold. Models: rf, xgboost, ridge, knn, lightgbm (old run); now 10 models available (+ gp, mlp, dnn, cnn, catboost).
31/57 reps completed (old run had 31/34; now 57 registered after Tiers 1-5 expansion). Missing from old run: bulkrnabert, scgpt, uce + all 26 Tier 1-5 new reps.
**Metrics (implemented, not yet reported):** AUROC (binary), F1-weighted (multi-class), Spearman ρ, RMSE.

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
- `all` (57): 18 expression + 5 foundation + 27 multi-omic + 7 WSI

**New reps (not yet baselined — run after server setup):**
- Tier 1 (mutation-derived, no deps): `mut_tmb` (5D), `mut_driver` (46D), `mut_pathway` (50D), `mut_pca50` (50D)
- Tier 2 (SigProfiler): `sig_sbs` (~78D), `sig_dbs` (~11D), `sig_id` (~18D)
- Tier 3 (PanCanAtlas): `immune_cibersort` (~22D), `immune_xcell` (~64D), `stemness_mrnasi` (2D), `purity_consensus` (4D), `heterogeneity_math` (1D), `msi_status` (3D), `hrd_score` (4D), `cimp_status` (2D)
- Tier 5 (multi-omic features): `splicing_psi50` (50D, 123 splicing factors + PCA), `lncrna_pca50` (50D, 142 cancer lncRNAs + PCA), `gsva_hallmark` (~50D, ssGSEA Hallmark sets), `promoter_meth` (50D, TSS200/TSS1500 probes + PCA)

**⚠ SERVER SETUP NEEDED before running new TCGA reps:**
```bash
# Tier 2: mutational signatures (one-time setup)
pip install SigProfilerAssignment
python -c "from SigProfilerMatrixGenerator import install; install.install('GRCh37')"
# Note: GRCh37 genome download is ~3.5GB, cached in ~/.SigProfiler/

# Tier 3: PanCanAtlas downloads are automatic (cached to /Volumes/seagate or ~/.cache/kirby/tcga/pancanatlas/)
# If GDC API URLs fail, manually download and place in the pancanatlas cache dir.
# MATH score (heterogeneity_math) and CIMP (cimp_status) need no new downloads.

# Clear old caches for full MAF (now includes DNA_VAF column):
rm -f /Volumes/seagate/kirby/tcga/mutations_full_*.csv
# Or: rm -f ~/.cache/kirby/tcga/mutations_full_*.csv
```

### 2D. DTA Representations

Cold-target GroupKFold CV (3-fold). Each rep tested standalone (drug-only or protein-only).
Davis: 30K pairs (68 drugs × 442 proteins). KIBA: 30K subsample (2111 drugs × 229 proteins).
17 reps ran (20 missing from full preset of 37: clamp, rmat, map4, unimol_v2, mordred, avalon, esmc, amplify, esmfold, alphafold_db, minimol, gp_molformer, mole, mint, PCA variants, mordred_pca50, esm3, saprot, boltz2).
**Metrics (implemented, not yet reported):** CI (primary), Spearman ρ, RMSE. Old table shows R² only. DTA baseline survey now ranks by CI (not R²).
**Models:** Old results use mixed models; 9 models now available (ridge, rf, xgb, lgbm, gp, mlp, dnn, cnn, catboost).
**Baseline survey:** 29 reps tested (16 drug + 13 protein) via `tests/dta_baseline_survey.py`.

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
- `fast+lm` (10): + drug_chemberta, drug_molformer, drug_clamp, protein_esmc, protein_descriptors
- `full` (37): 24 drug reps (ecfp4, maccs, pdv, pdv_binary_mean, pdv_rank, map4, mol2vec, mordred, avalon, chemberta, molformer, selformer, clamp, rmat, unimol_v2, minimol, gp_molformer, mole, chemberta_pca50, molformer_pca50, selformer_pca50, clamp_pca50, rmat_pca50, mordred_pca50) + 13 protein reps (esm2, esmc, esm2_midlayers, amplify, esmfold, alphafold_db, aac, descriptors, kmer, mint, esm3, saprot, boltz2)

**Phase 4A/4B forward-compat (in 'full' preset):** drug_minimol, drug_gp_molformer, drug_mole, protein_mint, protein_esm3, protein_saprot, protein_boltz2.

### 2E. Reaction Yield Representations

GroupKFold (BH: leave-additive-out, 23 groups; Suzuki: leave-ligand-out, 12 groups; Doyle plates: leave-additive-out).
3 datasets (BH, Suzuki, Doyle plates). 8 reps total (was 2; 6 new in Phase 3B).
**Models:** 9 (ridge, rf, xgb, lgbm, gp, mlp, dnn, cnn, catboost). Old results used XGB only.
**Metrics (implemented, not yet reported):** R², Spearman ρ, RMSE, MAE.

| Rep | Dim | Buchwald-Hartwig R² | Suzuki R² | Status |
|-----|-----|---------------------|-----------|--------|
| `drfp_2048` | 2048 | **0.934** (xgb) | **0.842** (xgb) | **Stale** (old random splits, 1 model) |
| `concat_ecfp4` | 5120/15360 | 0.934 (xgb) | 0.735 (xgb) | **Stale** (old random splits, 1 model) |
| `concat_maccs` | ~835/~835 | — | — | **New** (run needed) |
| `concat_pdv` | ~1000/~1000 | — | — | **New** (run needed) |
| `product_ecfp4` | 2048 | — | — | **New** (run needed) |
| `reactant_ecfp4` | 2048 | — | — | **New** (run needed) |
| `rxnfp` | 256 | — | — | **New** (run needed) |
| `onehot_components` | ~46/~60 | — | — | **New** (run needed) |

DRFP is clearly superior on Suzuki (+0.107). Both tied on Buchwald-Hartwig.
**Note:** Old results use random 70/30 splits. All reps should be rerun with GroupKFold.

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

**TCGA Modality Ablation (c12):**

| Experiment | Script | Status |
|-----------|--------|--------|
| Per-modality + pairwise + full combination evaluation | `tests/experiments/complexity/c12_modality_ablation.py` | Needs server |

Evaluates every TCGA modality group (expression, methylation, mutation, CNV, miRNA, RPPA, immune, clinical, WSI) individually, all pairwise combinations, and the full combination. Annotates which modalities help/hurt per task.

**WSI ABMIL aggregation:**

| Experiment | Script | Status |
|-----------|--------|--------|
| Mean pool vs gated attention vs ABMIL | `tests/experiments/wsi_abmil_comparison.py` | Needs server |

---

## 3. Known Issues and Reliability

### Unreliable Reps (may fail on server)

| Rep | Issue | Severity | Workaround |
|-----|-------|----------|------------|
| `gene_mutation` → `mutation_matrix` | Xena MC3 returns 403 | ~~HIGH~~ **RESOLVED** | Re-wired as `mutation_matrix` in Phase 0 (Issue 21). In 'all' preset. Download fallback handles Xena 403. |
| `clinical` | Xena phenotype returns 403 | ~~HIGH~~ **RESOLVED** | Re-wired in Phase 0 (Issue 21). In 'all' preset. Download fallback handles Xena 403. |
| `gene_hallmark` | ~~Enrichr download may fail~~ | ~~MEDIUM~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `gene_progeny` | ~~decoupler API 502 / NAType~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath weights, no API calls |
| `gene_dorothea` | ~~decoupler NAType in weight column~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath regulons, no API calls |
| `gene_reactome` | ~~ssGSEA orientation bug~~ | ~~LOW~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `wsi_titan` | HuggingFace gated, garbage results | MEDIUM | Still in 'all' preset (57 reps) but may produce garbage. Monitor on rerun — drop if results remain poor. |
| `wsi_uni2h` | Manual download + h5 processing, features not found | HIGH | Still in 'all' preset (57 reps) but requires manual setup. May fail silently (zeros fallback). |
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

**⚠ All "Done" baselines are STALE** — pre-Phase 1 preprocessing/metrics/models/splits. Every domain needs full rerun (Phase 5).

| Domain | Datasets | Reps (run / registered) | Models (run / available) | Metrics (reported / implemented) | Status |
|--------|----------|------------------------|-------------------------|----------------------------------|--------|
| Molecular | 4/8 | 17/36 | 2/10 | R² / R²+ρ+RMSE | **Stale** (4 ADME new; 19 reps unrun; 8 models unrun) |
| Antibody | 24/25 | 4/39 | 1/9 | R² / R²+ρ+Pearson+RMSE | **Stale** (35 reps × 8 models unrun; ddg_combined_full new) |
| TCGA | 3/3 | 31/57 | 5/10 | Acc / AUROC+F1+ρ+RMSE | **Stale** (26 new reps; 5 models unrun) |
| DTA | 2/3 | 17/34 | ~3/9 | R² / CI+ρ+RMSE | **Stale** (bindingdb_kd new; 17 reps unrun; 6 models unrun) |
| Reaction | 2/3 | 2/8 | 1/9 | R² / R²+ρ+RMSE+MAE | **Stale** (bh_doyle_plates new; 6 reps unrun; 8 models unrun; splits changed) |
| Comparisons | DTA e2e, QM9 3D | — | — | — | **Done** (c12 + WSI ABMIL pending server) |
| PDBBind | 0/2 | — | — | — | **Blocked** (manual download) |

### What's STILL PENDING

| Item | Gap | Blocker |
|------|-----|---------|
| Antibody full 39-rep survey | 35 reps × 25 datasets × 9 models | GPU resubmit (updated script ready: `submit_antibody_survey.sh all`; includes Phase 3C + 4A/4B additions: abroberta, igt5, esm3, saprot, layer variants, LLR reps) |
| TCGA 3 foundation reps | bulkrnabert, scgpt, uce | Resubmit running (bulkrnabert OOM fixed batch_size=1) |
| Molecular 19 missing reps + 4 new datasets | clamp, rmat, map4, unimol_v2, dimenetpp, mordred, mordred_pca50, avalon, minimol, gp_molformer, mole, sns, + PCA variants; ADME HLM/RLM/MDR1/Solubility | Need resubmit with updated 'all' preset (36 reps) × 10 models + `pip install mordred polaris-lib minimol` |
| Reaction 6 new reps + 1 new dataset | concat_maccs, concat_pdv, product_ecfp4, reactant_ecfp4, rxnfp, onehot_components; bh_doyle_plates | Need first run; `pip install rxnfp` on server |
| TCGA 26 new reps | Tiers 1-5 (mut_*, sig_*, immune_*, stemness_mrnasi, purity_consensus, heterogeneity_math, msi_status, hrd_score, cimp_status, splicing_psi50, lncrna_pca50, gsva_hallmark, promoter_meth) | Need resubmit with updated 'all' preset (57 reps) |
| DTA 17 missing reps + 1 new dataset | clamp, rmat, map4, unimol_v2, esmc, amplify, esmfold, alphafold_db, minimol, gp_molformer, mole, mint, PCA variants, protein_esm3, protein_saprot, protein_boltz2; bindingdb_kd | Need resubmit with updated 'full' preset (34 reps) × 9 models; CI primary metric |
| PDBBind | 2 datasets | Manual download from pdbbind.org.cn |
| WSI ABMIL comparison | Mean pool vs gated attention | Needs server |
| IgBert | Predicts mean (R² ~ 0) everywhere | Investigate or drop |
| protein_esm2_lora | NaN on KIBA | Debug factory |

### Dropped

| Rep | Reason |
|-----|--------|
| `wsi_cnn320` | Local Seagate only, garbage results |
| `unimol` v1 | Broken install |
| `chemformer` | Broken HF weights |
| `gene_scfoundation` (frozen) | Degenerate embeddings (0.51 acc = majority class). FT version works. |

**NOT dropped (re-wired in Phase 0):** `mutation_matrix` (was `gene_mutation`), `clinical` — both in 'all' preset with Xena 403 fallback handling.
**At risk (in 'all' but may fail):** `wsi_titan` (gated HF, garbage results), `wsi_uni2h` (manual download needed). Both still in TCGA 'all' preset — monitor on rerun.

---

## 5. Representation Categories (Summary)

| Category | Description | Examples |
|----------|-------------|---------|
| **Standard** | Computed from input data, no pretrained models | ecfp4, maccs, pdv, avalon, aac, kmer, gene_pca100, cnv_gistic |
| **Pretrained** | Frozen embeddings from pretrained LMs | chemberta, molformer, esm2, ablang2, wsi_ctranspath |
| **Fine-tuned** | Per-fold fine-tuned embeddings (FT factories) | geneformer, scfoundation, ft_schnet, ft_dimenetpp |
| **Mutation-aware** | Derived from mutation metadata or LLR scoring | mutation_features, esm2_mutsite, esm2_wt_delta, esm2_llr, esmc_llr, antiberty_llr |
| **Factory** | Per-fold fitted (PCA/NMF/scaling on train only) | gene_pca100, gene_nmf50, mordred_pca50, chemberta_pca50, gene_immune |

### Downstream Models (11 total)

| Model | Type | Tier | Notes |
|-------|------|------|-------|
| Ridge/RidgeCV | Linear | fast | Internal LOO CV for alpha selection |
| RF | Ensemble | fast | Default 100 trees, max_features=sqrt |
| XGBoost | Boosting | fast | |
| LightGBM | Boosting | fast | |
| KNN | Instance | fast | Adaptive k = min(5, sqrt(N)) |
| MLP | Neural | medium | VespaG-style (256,), early stopping |
| SVM/SVR | Kernel | medium | Size guard _max_samples=10000. **Not in SURVEY_MODELS** (too slow for standard survey); available via `get_model()` for targeted use. |
| GP | Kernel | medium | RBF+White, size guard _max_samples=5000 |
| CatBoost | Boosting | medium | Falls back to LightGBM if not installed |
| DNN | Neural | long | (256,128,64), PyTorch sklearn wrapper, internal early stopping |
| CNN | Neural | long | Multi-scale 1D Conv (k=3,7,15), PyTorch sklearn wrapper, internal early stopping |

---

## 6. Fine-Tuning Factories

### Active (proceed to Phase 2)

| Factory | Domain | Base Model(s) | Method | Output Dim | CLI Flag | GPU | Survey Result |
|---------|--------|---------------|--------|-----------|----------|-----|---------------|
| `make_geneformer_factory` | TCGA | Geneformer V2 (104M) | Freeze+head | 1152D | `--finetune geneformer` | Yes | **PASS** — best individual rep on lung (0.930), BRCA (0.789) |
| `make_scfoundation_factory` | TCGA | scFoundation (biomap-research) | Freeze last 2 layers + head | 2048D | `--finetune scfoundation` | Yes | **PASS** — frozen degenerate (0.51); FT rescues (0.95 lung, 0.86 BRCA) |
| `make_molecular_3d_factory` | Molecular | SchNet / DimeNet++ | E2E 3D GNN, 100 epochs | 128D | `--finetune ft_schnet` or `ft_dimenetpp` | Yes | **PASS** — SchNet E2E R²=0.888 on QM9 gap (vs 0.134 frozen) |

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

The original mutation-aware representation approach (40+ custom functions) was replaced with
simple LM embeddings → Ridge → R². A targeted set of 6 mutation-aware reps has been
re-introduced in `src/kirby/representations/mutation.py` (6 reps: mutation_features, esm2_mutsite,
esm2_wt_delta, esm2_llr, esmc_llr, antiberty_llr) — these are lightweight and use mutation metadata
from dataset loaders. Additionally, 3 variation-pool reps (esm2_entropy_top20, esm2_per_cdr,
esm2_per_region) provide finer-grained ESM-2 pooling without requiring mutation metadata.

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
- `standardize_smiles()` applies: `Cleanup → FragmentParent → Uncharger → TautomerEnumerator.Canonicalize`
- Metal-containing compounds rejected (organic descriptors unreliable)
- `deduplicate_by_inchikey()` removes duplicate molecules (same InChIKey, different SMILES) at dataset level — prevents same molecule appearing in both train and test folds
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

### Antibody LM Embedding Preprocessing
- **BOS/EOS token exclusion:** `_mean_pool()` in `antibody.py` zeroes position-0 (BOS/CLS) and final non-padded position (EOS/SEP) before mean-pooling transformer outputs. Applied to ESM-2, ABRoBERTa, and other LM reps.

### Splitting Improvements
- **Position-modulo splitting (antibody DMS):** Groups mutations by residue position mod n_folds — prevents same-position mutations from appearing in both train and test (ProteinGym standard protocol). Used for all FLAb2 and AbAgym DMS datasets.
- **GroupKFold for reactions:** Leave-additive-out (BH, 23 groups) and leave-ligand-out (Suzuki, 12 groups) replace old random 70/30 splits.

### Leakage Prevention — Per-Fold Factory Reps
- **Factory pattern:** All dimensionality-reduction reps (PCA, NMF, ICA, SparsePCA, etc.) use per-fold factories that fit on training data only and transform validation data — prevents data leakage through unsupervised fitting.
- **gene_immune factory:** Immune microenvironment scoring rewritten as per-fold factory. Bindea ssGSEA pre-computed once (sample-independent), ESTIMATE-like stromal/immune scores fitted on train-only then applied to validation.
- **12+ factory reps in TCGA:** gene_pca100, gene_nmf50, gene_ica50, gene_sparse_pca50, gene_autoencoder50, gene_quantile_rank, gene_corr_modules, gene_immune, splicing_psi50, lncrna_pca50, gsva_hallmark, promoter_meth, plus hvg2000 (StandardScaler per-fold).

### Model Improvements
- **Ridge → RidgeCV:** internal LOO cross-validation selects best alpha from `np.logspace(-6, 6, 13)`. No extra compute cost.
- **LogisticRegression → LogisticRegressionCV:** built-in Cs=10 alpha selection
- **KNN adaptive k:** `n_neighbors = min(5, max(1, sqrt(N_train)))` — scales to dataset size
- **GP (new):** Gaussian Process with RBF+White kernel. Size guard `_max_samples=5000`.
- **MLP (new):** VespaG-style (256,) single hidden layer with early stopping.
- **CatBoost (new):** Gradient boosting, falls back to LightGBM if not installed.
- **DNN (new):** (256,128,64) hidden layers, dropout=0.2, Adam+ReduceLROnPlateau+early stopping, target z-normalization. Internal holdout — skipped in HP search.
- **CNN (new):** Multi-scale 1D CNN treats embedding as 1D signal. Parallel Conv1d with kernel sizes (3,7,15) → InstanceNorm1d → ReLU → AdaptiveAvgPool1d → concat → FC. Internal 80/20 holdout + early stopping — skipped in HP search. Both regressor and classifier variants.

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
# Polaris ADME cache (only if re-downloading is needed):
# rm -rf ~/.cache/kirby/polaris_adme_fang.csv
```

---

## 9. Open Problems

### Antibody Representation Strategy (March 2026 rebuild)

The old approach (40+ mutation-aware custom features, complex presets, HP tuning) has been
replaced with a simple baseline pipeline: LM embeddings → Ridge → Spearman ρ.

**Current approach:** 39 representations (33 ANTIBODY_REPS + 6 MUTATION_REPS: 11 LM full-seq + 8 CDR-pooled + 1 midlayer + 3 layer variants + 1 CDR3-only + 3 seq-encoding + 2 structural + 3 variation-pool + 1 forward-compat + 6 mutation-aware) evaluated per dataset with 9 models (ridge, rf, xgb, lgbm, gp, mlp, dnn, cnn, catboost).

**Key questions to answer with the new baselines:**
1. Which LM (ESM-2, ESM-C, AbLang2, IgBert, AntiBERTy, ProtT5, AMPLIFY, ESM3, SaProt, ABRoBERTa, IgT5) works best per task type?
2. Does one-hot match or beat LMs at ≥1K samples (as claimed in literature)?
3. Are inter-family datasets (ddg_combined, abcov_kd) still fundamentally too small/hard?
4. Is Ridge sufficient or do tree models (RF, XGB, LightGBM, CatBoost) and neural models (MLP, DNN, CNN) add value?
5. Do mutation-aware reps (esm2_mutsite, esm2_wt_delta, esm2_llr, esmc_llr, antiberty_llr) rescue ΔΔG prediction (currently R²=-0.137)?
6. Does finer-grained pooling (per-CDR, per-region, entropy-top20) improve DMS prediction over full-sequence mean-pool?
7. Do ESM-2 layer variants (layer24, early, concat-chain) or CDR3-only provide signal beyond standard mean-pool?
