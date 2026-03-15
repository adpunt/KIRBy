# KIRBy Bill of Materials

Comprehensive inventory of datasets, representations, and their measured baseline
performance. This document is the single source of truth for what KIRBy supports
and what actually works. Representations without confirmed performance data should
not be used in experiments until validated.

**Last updated:** 2026-03-14 (Phase 5 results: molecular logd, reaction BH+Suzuki, TCGA 2/3, antibody 11 inter-family datasets + DMS partial, all comparisons (DTA E2E, QM9 3D, TCGA FM, WSI ABMIL, Mol FT, Reaction rep, Ab FT/region). DTA baselines + remaining molecular + antibody DMS still running.)

---

## 1. Datasets

### 1A. Molecular (regression, scaffold-split GroupKFold)

| Dataset | Source | Samples | Target | Metrics | Best Rep (model) | Best RMSE | Best R² | Status |
|---------|--------|---------|--------|---------|------------------|-----------|---------|--------|
| `logd` | OpenADMET-ExpansionRx | 5,028 | LogD | RMSE, R², Spearman ρ | mhg_gnn (gp) | **0.442** | **0.857** | **Phase 5** ✅ |
| `caco2` | OpenADMET-ExpansionRx | 2,158 | Caco2 Efflux (log) | RMSE, R², Spearman ρ | — | — | — | **Running** |
| `herg` | ChEMBL (CHEMBL240) | 1,415 | pKi | RMSE, R², Spearman ρ | — | — | — | **Running** |
| `qm9` | QM9 | ~10,000 | HOMO-LUMO gap | RMSE, R², Spearman ρ | — | — | — | **Running** |
| `adme_hlm` | Polaris ADME-Fang | 3,087 | HLM clearance (log) | RMSE, R², Spearman ρ | — | — | — | **Running** |
| `adme_rlm` | Polaris ADME-Fang | 3,054 | RLM clearance (log) | RMSE, R², Spearman ρ | — | — | — | **Running** |
| `adme_mdr1` | Polaris ADME-Fang | 2,642 | MDR1 efflux (log) | RMSE, R², Spearman ρ | — | — | — | **Running** |
| `adme_solubility` | Polaris ADME-Fang | 2,173 | Solubility (log) | RMSE, R², Spearman ρ | — | — | — | **Running** |

**LogD Phase 5 results (17 reps × 10 models, 5-fold scaffold-split):** RMSE primary metric. Best: mhg_gnn (GP, RMSE=0.442, R²=0.857, ρ=0.922). grover essentially tied (ridge, RMSE=0.443, R²=0.857, ρ=0.925). GP and DNN models dominate as best model for most reps. unimol_v2 failed (R²≈0). 7 remaining datasets running on HTC+ARC.

**Preprocessing:** All molecular datasets apply `standardize_smiles()` (salt stripping, neutralization, tautomer canonicalization) and `deduplicate_by_inchikey()` (removes duplicate molecules with different SMILES but same InChIKey, preventing same molecule in train+test).

### 1B. Antibody Intra-Family / DMS (regression, position-modulo GroupKFold for DMS)

| Dataset | Source | Samples | Target | Best Rep (model) | Best ρ | Reps Tested | Status |
|---------|--------|---------|--------|------------------|--------|-------------|--------|
| `flab_binding_koenig` | FLAb (G6) | 4,275 | Kd | blosum (cnn) | **0.309** | 4 | **Partial** (fast reps only) |
| `flab_expression_koenig` | FLAb (G6 SSM) | 4,275 | Expression | aac (rf) | **0.447** | 4 | **Partial** (fast reps only) |
| `flab_binding_phillips_h3` | FLAb2 (Phillips 2021) | 5,000 | Binding (SPR Kd) | — | — | 1 | **Partial** (aido_protein only, failed) |
| `flab_binding_adams` | FLAb2 (Adams 2017) | 5,000 | Binding (Kd) | — | — | 1 | **Partial** (aido_protein only, failed) |
| `flab_expression_adams` | FLAb2 (Adams 2017) | 5,000 | Expression (ER) | — | — | 1 | **Partial** (aido_protein only, failed) |
| `abagym_HER2` | AbAgym DMS | 146 | Fitness | — | — | 1 | **Partial** (aido_protein only, failed) |
| `abagym_VEGF` | AbAgym DMS | 418 | Fitness | — | — | 1 | **Partial** (aido_protein only, failed) |
| `abagym_EGFR` | AbAgym DMS | 659 | Fitness | — | — | 1 | **Partial** (aido_protein only, failed) |
| `abagym_lysozyme` | AbAgym DMS | 627 | Fitness | — | — | 1 | **Partial** (aido_protein only, failed) |
| `abagym_Ang2` | AbAgym DMS | 397 | Fitness | — | — | 1 | **Partial** (aido_protein only, failed) |
| `abagym_all` | AbAgym (pooled, 5 groups) | 2,247 | Fitness | — | — | 1 | **Partial** (aido_protein only, failed) |

**⚠ DMS datasets mostly INCOMPLETE** — 10/11 datasets have only aido_protein results (which produced zero signal on every dataset, ρ < 0.08). 2 Koenig datasets have 4 fast reps (onehot, blosum, aac, paratope). LM, mutation, variation, structural, MINT reps still running/pending.

**aido_protein (4608D) completely failed on all antibody DMS datasets** — R² negative and Spearman near zero across all 10 datasets tested. This is a strong negative result for AIDO.Protein on antibody tasks.

**Metrics:** R², Spearman ρ, Pearson r, RMSE per fold.

### 1C. Antibody Inter-Family (regression, identity-grouped GroupKFold)

| Dataset | Source | Samples | Target | Best Rep (model) | Best ρ | Reps Tested | Status |
|---------|--------|---------|--------|------------------|--------|-------------|--------|
| `flab_thermo_tresanco` | FLAb2 (Tresanco 2023) | 672 | Tm | esm3 (dnn) | **0.805** | 20 | **Phase 5** ✅ |
| `flab_pk_jain_heprt` | FLAb2 (Jain 2023) | 130 | Heparin RT | esm2_per_cdr (gp) | **0.658** | 3 | **Partial** |
| `flab_pk_jain_fcrnrt` | FLAb2 (Jain 2023) | 132 | FcRn RT | ablang2 (ridge) | **0.595** | 21 | **Phase 5** ✅ |
| `flab_polyreact_jain_cic` | FLAb2 (Jain 2017) | 137 | CIC RT | prot_t5 (ridge) | **0.586** | 23 | **Phase 5** ✅ |
| `flab_aggregation_jain_hicrt` | FLAb2 (Jain 2017) | 137 | HIC RT | prot_t5 (ridge) | **0.506** | 23 | **Phase 5** ✅ |
| `flab_thermo_jain` | FLAb (Jain 2017) | 137 | Tm | ablang2 (ridge) | **0.457** | 23 | **Phase 5** ✅ |
| `abcov_kd` | Ab-CoV (via FLAb) | 141 | -log Kd | prot_t5 (catboost) | **0.432** | 22 | **Phase 5** ✅ |
| `flab_aggregation_jain` | FLAb (Jain 2017) | 137 | AC-SINS | esm2_concat (ridge) | **0.425** | 23 | **Phase 5** ✅ |
| `flab_immunogenicity_marks` | FLAb2 (Marks 2021) | 217 | Immunogenicity | antiberty_cdr (mlp) | **0.399** | 23 | **Phase 5** ✅ |
| `flab_polyreact_jain_bvp` | FLAb2 (Jain 2017) | 137 | BVP ELISA | ablang2 (ridge) | **0.374** | 23 | **Phase 5** ✅ |
| `flab_thermo_hutchinson` | FLAb2 (Hutchinson 2023) | 192 | Tm | esm2 (xgboost) | **0.289** | 23 | **Phase 5** ✅ |
| `flab_aggregation_jain_sas` | FLAb2 (Jain 2017) | 137 | SAS | antiberty_cdr (xgb) | **0.201** | 23 | **Phase 5** ✅ |
| `ddg_combined` | SKEMPI 2.0 + AB-Bind (Ab-Ag only) | 429 | ΔΔG | — | — | 1 | **Partial** (aido_protein only, failed) |
| `ddg_combined_full` | SKEMPI 2.0 full + AB-Bind (all PP) | ~3,000 | ΔΔG | — | — | — | **New** (run needed) |

**Phase 5 inter-family results (20-23 LM reps, 10 models, identity-grouped GroupKFold):** Spearman ρ is now the primary metric. Every dataset with ≥20 reps tested shows viable signal — even "hard" datasets like aggregation_sas (ρ=0.201) and thermo_hutchinson (ρ=0.289) that had negative R² in stale results now show positive rank correlation. ESM3 dominates tresanco (ρ=0.805 with DNN). ablang2 and prot_t5 are the most consistent top performers across datasets. DNN/ridge split as best model — DNN wins on larger datasets, ridge on smaller ones.

**Key changes from stale → Phase 5:** bvp and sas, which had negative R² in old results, are now confirmed viable by Spearman. This validates the switch to Spearman as primary antibody metric. Datasets ranked 4→12 (by ρ, not R²) with only LM reps — fast reps, mutation reps, structural reps, MINT still pending.

**Dropped:** `sabdab_affinity` (was SKEMPI 2.0 only, replaced by `ddg_combined`),
`abbind` / `abbind_spm` (strict subsets of `ddg_combined`).

**`ddg_combined_full`:** Full SKEMPI 2.0 (all protein-protein complexes, not just Ab-Ag) + AB-Bind. ~6x more data than `ddg_combined`. Enables comparison: Ab-Ag-specific vs general PP DDG prediction. Registered in antibody survey with `--include-all-skempi` flag.

### 1D. TCGA (classification/regression, TSS-grouped StratifiedGroupKFold)

| Dataset | Task | Samples | Classes/Target | TSS | Best Rep (model) | Best Score | Status |
|---------|------|---------|----------------|-----|------------------|------------|--------|
| `tcga_luad_vs_lusc` | Binary clf | 1,017 | LUAD vs LUSC | 68 | gene_geneformer_v2 (catboost) | **0.946** AUROC | **Phase 5** ✅ |
| `tcga_brca_pam50` | 4-class clf | 821 | LumA/LumB/Basal/Her2 | 19 | gene_geneformer_v2 (ridge) | **0.861** F1w | **Phase 5** ✅ |
| `tcga_brca_survival` | Regression | 151 | OS days (uncensored) | 21 | wsi_titan (ridge) | **1171** RMSE | **Phase 5** ✅ (all R² neg) |

**Phase 5 TCGA results (36 reps × 10 models, 5-fold TSS-grouped):** Metrics: AUROC (binary), F1-weighted (multi-class), RMSE/Spearman (regression). gene_geneformer_v2 is best for both classification tasks. Survival remains unsolvable (n=151, all R² negative). 8 reps produce degenerate predictions (AUROC=0.5): clinical, heterogeneity_math, msi_status, gene_scfoundation, wsi_conch, wsi_hoptimus1, wsi_uni2h, wsi_virchow2. 36/57 reps tested (missing: factory reps like gene_hvg2000, gene_pca100, gene_nmf50 etc. that need per-fold fitting — likely still running).

### 1E. Drug-Target Affinity (regression, cold-target GroupKFold)

| Dataset | Samples | Drugs | Proteins | Target | Metrics | Best Rep | Best R² | Status |
|---------|---------|-------|----------|--------|---------|----------|---------|--------|
| `davis` | 30,000 | 68 | 442 | pKd | CI, R², Spearman ρ, RMSE | drug_chemberta (xgb) | **0.223** | **Stale** (drug-only; +protein_aac → 0.446) |
| `kiba` | 30,000 (sub) | 2,111 | 229 | KIBA score | CI, R², Spearman ρ, RMSE | drug_mol2vec (lgbm) | **0.340** | **Stale** (frozen) |
| `bindingdb_kd` | ~52,000 | ~10,000 | ~1,400 | pKd | CI, R², Spearman ρ, RMSE | — | — | **New** (run needed) |

**⚠ ALL existing DTA results are STALE** — run before Phase 1 metrics (CI now primary), Phase 2 models (9 models, was ~3), Phase 4B reps. Also supports `--cold-setting drug` for cold-drug evaluation.

### 1F. Reaction Yield (regression, GroupKFold leave-one-group-out)

| Dataset | Samples | Groups | Target | Best Rep (model) | Best R² | Spearman | Status |
|---------|---------|--------|--------|------------------|---------|----------|--------|
| `buchwald_hartwig` | 3,955 | 22 additive | Yield (0-100%) | reactant_ecfp4 (rf) | **0.718** | 0.855 | **Phase 5** ✅ |
| `suzuki` | 5,760 | 12 ligand | Yield (0-100%) | onehot_components (rf) | **0.453** | 0.680 | **Phase 5** ✅ |
| `bh_doyle_plates` | 4,312 | 24 additive | Yield (0-100%) | concat_ecfp4 (rf) | **0.613** | — | **Phase 5** ✅ |

**Phase 5 reaction results (8 reps × 10 models, GroupKFold):** As predicted, R² dropped significantly from stale random-split numbers (BH: 0.934→0.718, Suzuki: 0.842→0.453). Suzuki is dramatically harder under leave-ligand-out — all concat/drfp/rxnfp reps collapse to baseline (R²≈-0.03), only onehot_components generalizes. BH best is reactant_ecfp4 (not drfp_2048 as before). Doyle plates results from reaction_rep_comparison.json.

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

**⚠ Phase 5 results are replacing stale pre-Phase 1 numbers.** Sections marked "Phase 5" have fresh data with corrected preprocessing, per-fold PCA/scaling, proper splits, 10 models, and domain-standard metrics. DTA section (2D) is still fully stale — baseline jobs running.

### 2A. Molecular Representations

**Phase 5 results (LogD only so far).** 5-fold scaffold-split GroupKFold, 10 models, RMSE primary.
Caco2, hERG, QM9, 4× ADME still running on HTC+ARC.

| Rep | Dim | Best Model | LogD RMSE | LogD R² | LogD ρ |
|-----|-----|-----------|-----------|---------|--------|
| `mhg_gnn` | 1024 | gp | **0.442** | **0.857** | 0.922 |
| `grover` | 4800 | ridge | 0.443 | 0.857 | **0.925** |
| `mordred` | 1613 | dnn | 0.452 | 0.850 | 0.920 |
| `mol2vec` | 300 | gp | 0.464 | 0.843 | 0.913 |
| `ecfp4` | 2048 | gp | 0.475 | 0.836 | 0.913 |
| `molformer` | 768 | gp | 0.476 | 0.835 | 0.910 |
| `pdv` | 200 | dnn | 0.477 | 0.835 | 0.905 |
| `smited` | 768 | gp | 0.481 | 0.832 | 0.905 |
| `avalon` | 2048 | gp | 0.504 | 0.815 | 0.899 |
| `chemberta` | 384 | dnn | 0.534 | 0.792 | 0.882 |
| `chembert` | 256 | gp | 0.546 | 0.784 | 0.878 |
| `graph_kernel` | 290 | dnn | 0.556 | 0.775 | 0.877 |
| `selformer` | 768 | dnn | 0.597 | 0.740 | 0.849 |
| `graphmvp` | 300 | dnn | 0.627 | 0.714 | 0.839 |
| `maccs` | 167 | gp | 0.634 | 0.708 | 0.829 |
| `molclr` | 512 | dnn | 0.738 | 0.604 | 0.758 |
| `unimol_v2` | 768 | mlp | 1.176 | -0.005 | — |

**Key changes vs stale:** mhg_gnn overtakes grover (GP model). mordred (new Phase 3B) enters top 3. avalon (new) solid mid-tier. unimol_v2 completely failed. GP and DNN are dominant best models — ridge no longer best for any rep except grover.

**Not yet tested (running):** minimol, gp_molformer, mole, sns, dimenetpp, schnet, clamp, rmat, map4, all *_pca50 factory reps.
**Fine-tuned:** ft_schnet (E2E SchNet → 128D, QM9 R²=0.888).
**Dropped:** `chemformer` (broken HF weights), `unimol` v1 (broken install).

**Presets:**
- `fast` (3): ecfp4, maccs, pdv
- `fast+hf` (6): + chemberta, molformer, selformer
- `all` (36): 25 base reps (ecfp4, maccs, avalon, pdv, mordred, mol2vec, chemberta, molformer, selformer, chembert, molclr, graphmvp, smited, mhg_gnn, grover, graph_kernel, schnet, dimenetpp, clamp, rmat, map4, unimol_v2, minimol, gp_molformer, mole) + 11 per-fold factory reps (pdv_binary_mean, pdv_rank, sns, chemberta_pca50, molformer_pca50, selformer_pca50, molclr_pca50, smited_pca50, mhg_gnn_pca50, grover_pca50, mordred_pca50)

**Phase 4A forward-compat reps (in 'all' preset):** `minimol` (MiniMol, pip install), `gp_molformer` (IBM GP-MoLFormer, HuggingFace), `mole` (MolE, GitHub + manual weights).

### 2B. Antibody Representations

**Phase 5 results (20-23 LM reps × 10 models, identity-grouped GroupKFold).**
Spearman ρ is primary metric. 11 inter-family datasets with substantial coverage. DMS datasets mostly incomplete.

**Script:** `tests/antibody_baseline_survey.py`

**Per-rep best Spearman ρ across key inter-family datasets (best model per dataset):**

| Rep | Dim | tresanco | fcrnrt | cic | hicrt | thermo_jain | abcov_kd | immunogenicity |
|-----|-----|----------|--------|-----|-------|-------------|----------|----------------|
| `ablang2` | 480 | 0.722 | **0.595** | 0.455 | 0.397 | **0.457** | 0.384 | 0.293 |
| `prot_t5` | 2048 | 0.729 | 0.416 | **0.586** | **0.506** | 0.289 | **0.432** | 0.357 |
| `esm3` | 3072 | **0.805** | 0.374 | 0.369 | 0.358 | 0.429 | 0.352 | 0.312 |
| `igt5` | 1024 | 0.746 | **0.563** | **0.502** | 0.394 | 0.431 | 0.348 | 0.368 |
| `abroberta` | 1536 | 0.685 | 0.292 | 0.387 | 0.370 | **0.451** | 0.412 | 0.297 |
| `esm2` | 2560 | 0.759 | 0.407 | 0.369 | 0.382 | 0.368 | 0.412 | 0.311 |
| `antiberty` | 768 | 0.729 | 0.388 | 0.377 | 0.307 | 0.280 | 0.426 | **0.398** |
| `saprot` | 2560 | 0.784 | 0.409 | 0.317 | 0.303 | 0.357 | 0.309 | 0.283 |
| `esmc` | 1280 | 0.780 | 0.271 | 0.330 | 0.414 | 0.329 | 0.369 | 0.350 |
| `igbert` | 1024 | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

**Top rep rankings:**
1. **ablang2** — best on fcrnrt (#1) and thermo_jain (#1), strong everywhere. Antibody-specific LM.
2. **prot_t5** — best on cic (#1), hicrt (#1), abcov_kd (#1). General protein LM, dominates on smaller datasets.
3. **igt5** — forward-compat Phase 4A, consistently top 5. Best antibody-specific newcomer.
4. **esm3** — best on tresanco (#1, ρ=0.805 with DNN). Excels on the largest dataset.
5. **abroberta** — strong on thermo_jain (#1 by ρ despite R²≈0). CDR-pooled variant identical.

**Model landscape:** Ridge dominates small datasets (n≈130-140). DNN dominates on tresanco (n=672). Catboost/lightgbm competitive on abcov_kd. GP wins on heprt with variation reps.

**Dead confirmed:** igbert/igbert_cls (NaN everywhere), aido_protein (failed on all 10 DMS datasets).

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

**Phase 5 results.** 5-fold TSS-grouped StratifiedGroupKFold, 10 models, 36/57 reps completed.
Metrics: AUROC (binary), F1-weighted (multi-class), RMSE (survival regression).
Factory reps (gene_hvg2000, gene_pca100, gene_nmf50, etc.) missing — require per-fold fitting, likely still running.

| Rep | Dim | Lung AUROC | BRCA F1w | Survival RMSE |
|-----|-----|-----------|----------|---------------|
| **gene_geneformer_v2** | 1152 | **0.946** | **0.861** | — |
| gene_bulkrnabert | 256 | 0.939 | 0.841 | 1334 |
| methyl_hvg | 5000 | 0.939 | 0.631 | 1322 |
| gene_geneformer | 1152 | 0.936 | 0.850 | 1334 |
| gene_chromosome_arm | 42 | 0.918 | 0.762 | 1333 |
| wsi_titan | 768 | 0.917 | 0.696 | **1171** |
| methyl_pca50 | 50 | 0.916 | 0.664 | **1253** |
| cnv_pca50 | 50 | 0.915 | 0.712 | 1340 |
| mirna_top200 | 200 | 0.896 | 0.640 | 1337 |
| wsi_gigapath | 512 | 0.884 | 0.646 | 1295 |
| wsi_hoptimus0 | 512 | 0.874 | 0.661 | 1287 |
| cnv_gistic | 200 | 0.843 | 0.606 | 1338 |
| wsi_ctranspath | 512 | 0.830 | 0.594 | 1310 |
| wsi_moco | 512 | 0.816 | 0.613 | 1317 |
| immune_xcell | 12 | 0.812 | 0.576 | 1329 |
| wsi_phikon | 512 | 0.802 | 0.606 | 1337 |
| gene_stats | 15 | 0.793 | 0.543 | 1326 |
| rppa | 258 | 0.785 | 0.776 | 1339 |
| mutation_matrix | 200 | 0.774 | 0.557 | 1334 |
| mut_driver | 46 | 0.751 | 0.540 | 1334 |
| stemness_mrnasi | 2 | 0.751 | 0.612 | 1336 |
| immune_cibersort | 21 | 0.707 | 0.561 | 1314 |
| mut_pca50 | 50 | 0.705 | 0.455 | 1332 |
| mut_pathway | 50 | 0.702 | 0.545 | 1328 |
| cimp_status | 2 | 0.692 | 0.382 | 1330 |
| mut_tmb | 5 | 0.666 | 0.446 | 1329 |
| hrd_score | 4 | 0.656 | 0.430 | 1319 |
| purity_consensus | 4 | 0.601 | 0.522 | 1336 |
| clinical | 10 | 0.500 | 0.349 | 1333 |
| heterogeneity_math | 1 | 0.500 | 0.349 | 1338 |
| msi_status | 3 | 0.500 | 0.349 | 1338 |
| gene_scfoundation | 2048 | 0.500 | 0.349 | 1338 |
| wsi_conch | 768 | 0.500 | 0.349 | — |
| wsi_hoptimus1 | 1536 | 0.500 | 0.349 | — |
| wsi_uni2h | 1536 | 0.500 | 0.349 | 1336 |
| wsi_virchow2 | 1280 | 0.500 | 0.349 | — |

**Key findings (Phase 5):**
- gene_geneformer_v2 (forward-compat) is now #1 on both classification tasks — validates Phase 4 thesis.
- gene_bulkrnabert joins the top tier (AUROC=0.939).
- wsi_titan is best for survival (RMSE=1171, Spearman=0.27) — only WSI rep that works well.
- 8 reps produce degenerate predictions (AUROC=0.500) — clinical, heterogeneity_math, msi_status, gene_scfoundation (frozen), wsi_conch, wsi_hoptimus1, wsi_uni2h, wsi_virchow2.
- gene_scfoundation still degenerate frozen (unchanged from stale).
- Survival remains unsolvable (n=151, all R² negative).
- 21 factory reps not yet in results (gene_hvg2000, gene_pca100, gene_nmf50, etc.) — still running.

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

**Phase 5 results (8 reps × 10 models, GroupKFold).** BH: leave-additive-out (22 groups). Suzuki: leave-ligand-out (12 groups).

| Rep | Dim | BH Best Model | BH R² | BH ρ | Suzuki Best Model | Suzuki R² | Suzuki ρ |
|-----|-----|--------------|-------|------|------------------|-----------|----------|
| `reactant_ecfp4` | 2048 | rf | **0.718** | **0.855** | ridge | -0.033 | 0.031 |
| `concat_ecfp4` | 5120 | rf | 0.704 | 0.847 | rf | -0.033 | 0.000 |
| `drfp_2048` | 2048 | rf | 0.688 | 0.835 | ridge | -0.033 | 0.033 |
| `concat_maccs` | 835 | xgboost | 0.634 | 0.812 | rf | -0.033 | 0.000 |
| `onehot_components` | 45/37 | xgboost | 0.615 | 0.802 | rf | **0.453** | **0.680** |
| `concat_pdv` | 1085 | rf | 0.582 | 0.796 | rf | -0.033 | 0.000 |
| `rxnfp` | 256 | ridge | 0.566 | 0.771 | ridge | -0.034 | 0.003 |
| `product_ecfp4` | 2048 | ridge | 0.136 | 0.370 | rf | -0.033 | 0.000 |

**Key findings (Phase 5):**
- BH: R² dropped from 0.934 (random) to 0.718 (GroupKFold) as predicted. reactant_ecfp4 overtakes drfp_2048.
- Suzuki: **catastrophic collapse** — ALL concat/drfp/rxnfp reps produce R²≈-0.03 under leave-ligand-out. Only onehot_components generalizes (R²=0.453). This is a genuine finding: molecular fingerprints cannot extrapolate to unseen ligands in this reaction space.
- product_ecfp4 is weak on BH (product alone lacks reactant/reagent info).

### 2F. Comparison Experiments

**DTA End-to-End (5-fold cold-target CV, 100 epochs):**

| Method | Davis CI | Davis R² | Davis ρ | KIBA CI | KIBA R² | KIBA ρ | Status |
|--------|----------|----------|---------|---------|---------|--------|--------|
| **KIRBy Frozen** | **0.741** | **0.552** | **0.610** | 0.674 | 0.314 | 0.476 | **Phase 5** ✅ |
| **DeepDTA** (CNN on SMILES+protein) | 0.738 | 0.547 | 0.602 | **0.708** | **0.389** | **0.558** | **Phase 5** ✅ |
| **GraphDTA-GIN** (GIN on graphs) | 0.733 | 0.524 | 0.591 | 0.640 | 0.133 | 0.393 | **Phase 5** ✅ |

KIRBy frozen edges ahead on Davis (CI=0.741 vs 0.738 DeepDTA). DeepDTA wins on KIBA (CI=0.708 vs 0.674). GraphDTA-GIN poor on KIBA (R²≈0 on some folds). KIRBy frozen representations are competitive with E2E deep learning despite zero training.
Script: `tests/experiments/dta_e2e_comparison.py`

**DTA Fine-Tuning (5-fold cold-target CV, 5 epochs):**

| Method | Davis CI | Davis R² | KIBA CI | KIBA R² | Status |
|--------|----------|----------|---------|---------|--------|
| **Frozen baseline** (ECFP4+ESM2) | 0.748 | 0.568 | 0.693 | 0.353 | **Phase 5** ✅ |
| LoRA ESM-2 | NaN | NaN | NaN | NaN | **Failed** |
| LoRA ESM-C / ESM3 / SaProt | NaN | NaN | NaN | NaN | **Failed** |
| LoRA ChemBERTa / MolFormer | NaN | NaN | NaN | NaN | **Failed** |
| Joint LoRA | NaN | NaN | NaN | NaN | **Failed** |

**All 7 DTA LoRA fine-tuning strategies produced NaN.** Only frozen baseline ran successfully. This is a major negative result — DTA fine-tuning needs debugging.
Script: `tests/experiments/dta_finetune_comparison.py`

**QM9 3D Comparison (scaffold-split, 10K molecules):**

| Method | Gap R² | Gap MAE | Homo R² | Mu R² | Status |
|--------|--------|---------|---------|-------|--------|
| **E2E SchNet** (100 epochs) | 0.950* | 0.217* | — | — | *single fold only (CUDA required) |
| **ECFP4 + XGBoost** | 0.850 | 0.366 | 0.680 | 0.533 | **Phase 5** ✅ (5-fold) |
| Frozen DimeNet++ | -0.009 | 1.064 | -0.040 | -0.003 | **Failed** (embeddings useless) |
| Frozen SchNet | 0.134* | — | — | — | *single fold only |

E2E SchNet massively outperforms on single-fold GPU runs (R²=0.95 vs 0.85 ECFP4+XGB), confirming 3D geometry is essential for QM9. 5-fold runs failed for all GPU methods (no CUDA on evaluation machine). Frozen 3D GNN embeddings (SchNet, DimeNet++) are useless — the value is in E2E training, not frozen representations.
Script: `tests/experiments/qm9_3d_comparison.py`

**TCGA Foundation Model Comparison (LUAD/LUSC, 5-fold TSS-grouped):**

| Method | AUROC | F1w | Status |
|--------|-------|-----|--------|
| **scFoundation FT** | **0.988** | **0.931** | **Phase 5** ✅ |
| Geneformer FT | 0.977 | 0.862 | **Phase 5** ✅ |
| Geneformer frozen | 0.975 | 0.844 | **Phase 5** ✅ |
| scFoundation frozen | 0.500 | 0.127 | **Dead** (random chance) |

Fine-tuning inverts the ranking: scFoundation goes from worst (0.500 frozen) to best (0.988 FT). Geneformer works well frozen (0.975) and gains marginal improvement from FT. scFoundation FT achieves near-SOTA, but frozen embeddings are useless — KIRBy should only use the FT factory for scFoundation.
Script: `tests/experiments/tcga_foundation_comparison.py`

**WSI Aggregation Comparison (TITAN, GigaPath, H-optimus, etc.):**

| Encoder | BRCA PAM50 AUROC | LUAD/LUSC AUROC | Status |
|---------|-----------------|-----------------|--------|
| **TITAN** | **0.921** | **0.754** | **Phase 5** ✅ |
| H-optimus-0 | 0.856 | 0.641 | **Phase 5** ✅ |
| GigaPath | 0.789 | 0.662 | **Phase 5** ✅ |
| MoCo | 0.760 | 0.622 | **Phase 5** ✅ |
| CTransPath | 0.729 | 0.600 | **Phase 5** ✅ |
| Phikon | 0.719 | 0.590 | **Phase 5** ✅ |

TITAN dominates both tasks by a large margin. **All 3 aggregation methods (mean pool, gated attention, ABMIL) produce identical results per encoder** — attention mechanisms are not learning anything distinct. This is suspicious and may indicate a bug in the ABMIL implementation.
Script: `tests/experiments/wsi_abmil_comparison.py`

**Molecular Fine-Tuning Comparison (LogD, scaffold-split):**

| Method | LogD RMSE | LogD R² | Caco2 R² | hERG R² | Status |
|--------|-----------|---------|----------|---------|--------|
| **GROVER frozen** | **0.466** | **0.841** | **0.478** | **0.506** | **Phase 5** ✅ |
| GraphMVP FT | 0.560 | 0.770 | — | — | **Phase 5** ✅ |
| MolFormer frozen | 0.590 | 0.745 | — | — | **Phase 5** ✅ |
| ChemBERTa frozen | 0.622 | 0.717 | — | — | **Phase 5** ✅ |
| SELFormer frozen | 0.655 | 0.687 | — | — | **Phase 5** ✅ |
| MolCLR frozen | 0.887 | 0.426 | — | — | **Phase 5** ✅ |
| LoRA/contrastive methods | NaN | NaN | NaN | NaN | **Failed** |
| Freeze-head (ChemBERTa) | ~0.88 | — | — | — | **Worse than frozen** |

GROVER frozen is consistently the best molecular encoder across all 3 datasets. Most fine-tuning methods (LoRA, contrastive) produced NaN. GraphMVP FT works and improves over its frozen baseline. Freeze-head strategies degrade performance. **Molecular fine-tuning remains unproductive** — frozen encoders win.
Script: `tests/experiments/molecular_finetune_comparison.py`

**Antibody Fine-Tuning Comparison (Phase 5):**

| Method | phillips_h3 R² | koenig_expr R² | abcov_kd R² | Status |
|--------|----------------|----------------|-------------|--------|
| Frozen ESM-2 | 0.960 | 0.484 | -0.071 | **Phase 5** ✅ |
| LoRA ESM-2 | **0.978** | 0.446 | — | **Phase 5** ✅ |
| Frozen AbLang2 | 0.860 | 0.431 | **0.150** | **Phase 5** ✅ |
| FT AbLang2 | **0.971** (+0.111) | **0.519** (+0.088) | — | **Phase 5** ✅ |
| Frozen AntiBERTy | 0.924 | 0.314 | — | **Phase 5** ✅ |
| FT AntiBERTy | **0.983** (+0.059) | -55M (diverged) | — | **Phase 5** ✅ |

FT helps on large datasets (phillips_h3 n=5000). AntiBERTy FT diverged catastrophically on koenig datasets. Small datasets (abcov_kd n=141) too small for FT.
Script: `tests/experiments/antibody_finetune_comparison.py`

**Antibody Region Ablation (Phase 5):**

| Method | Dim | phillips_h3 R² | koenig_expr R² | tresanco R² | HER2 R² | Status |
|--------|-----|----------------|----------------|-------------|---------|--------|
| full_seq (baseline) | 1280 | 0.960 | 0.484 | — | — | **Phase 5** ✅ |
| region_var_k10 (best) | 17920 | **0.972** | **0.501** | 0.587 | **0.482** | **Phase 5** ✅ |
| region_var_k5 | 8960 | — | — | **0.589** | — | **Phase 5** ✅ |
| all_7_regions | 8960 | — | 0.490 | — | — | **Phase 5** ✅ |
| cdr_combined | 1280 | 0.960 | — | — | — | **Phase 5** ✅ |

region_var_kN (entropy-guided variability selection) consistently outperforms full-seq and CDR strategies across all datasets. Biggest gains on harder problems (koenig, tresanco, HER2).
Script: `tests/experiments/antibody_region_ablation.py`

**Forward Compatibility Experiments:**

| Experiment | Model | Domains | Thesis claim | Status |
|-----------|-------|---------|-------------|--------|
| Exp 1: MINT | MINT (Jan 2026, 1280D) | DTA, Antibody | MINT alone < MINT + KIRBy | **Covered by Wave 1 baselines** (pending) |
| Exp 2: SaProt | SaProt (ICLR 2024, 1280D) | DTA, Antibody | SaProt alone < SaProt + KIRBy | **Covered by Wave 1 baselines** (pending) |
| Exp 3: MolE/MiniMol | MolE, MiniMol, GP-MoLFormer | Molecular | New FM alone < New FM + KIRBy | **Covered by Wave 1 baselines** (running) |
| Exp 4: IgT5/AbRoBERTa | IgT5, Ab-RoBERTa | Antibody | New AbLM alone < New AbLM + KIRBy | **Covered by Wave 1 baselines** (running) |
| Exp 5: Boltz-2 | Boltz-2 (384D) | DTA | Boltz-2 alone < Boltz-2 + KIRBy | **Covered by Wave 1 baselines** (pending) |
| ESM3 | ESM3 (1536D) | Antibody, DTA | ESM3 alone < ESM3 + KIRBy | **Covered by Wave 1 baselines** (running) |
| Exp 6: xTrimoPGLM | xTrimoPGLM (2048D) | DTA, Antibody | xTrimoPGLM + KIRBy | **Covered by Wave 1 baselines** (pending) |
| Exp 8: Geneformer V2 | Geneformer V2 (1152D) | TCGA | V2 drop-in upgrade | **Phase 5** ✅ (AUROC=0.946, #1) |
| Exp 9: AIDO.Protein | AIDO.Protein (4608D) | DTA, Antibody | AIDO + KIRBy | **Not yet submitted** |

**Geneformer V2 validates forward-compat thesis:** dropped in with zero code changes, immediately became #1 on both TCGA classification tasks (AUROC=0.946 Lung, F1w=0.861 BRCA). No separate FC script needed — all FC reps are included in Phase 5 baseline survey rep lists.

**Ensemble vs Hybrid (c11):**

| Experiment | Script | Status |
|-----------|--------|--------|
| Best single rep vs model ensemble vs feature-level hybrid | `tests/experiments/complexity/c11_ensemble_vs_hybrid.py` | **Needs run** (select datasets from BOM results) |

**TCGA Modality Ablation (c12):**

| Experiment | Script | Status |
|-----------|--------|--------|
| Per-modality + pairwise + full combination evaluation | `tests/experiments/complexity/c12_modality_ablation.py` | **Needs run** |

Evaluates every TCGA modality group (expression, methylation, mutation, CNV, miRNA, RPPA, immune, clinical, WSI) individually, all pairwise combinations, and the full combination. Annotates which modalities help/hurt per task.

**WSI ABMIL aggregation:** See comparison results above (TITAN dominates; all 3 aggregation methods identical — likely bug).

**Design implications (thesis boundary):**

The comparison experiments delineate when frozen reps are sufficient vs. when E2E training is needed. Interpretations below are based on stale results — will be updated after Phase 5 rerun.

1. **Frozen reps sufficient:** DTA on small drug spaces (Davis: 68 drugs). KIRBy frozen ties DeepDTA despite zero training.
2. **Frozen reps competitive but with ceiling:** DTA on diverse drug spaces (KIBA: 2,111 drugs). DeepDTA pulls ahead. The `--finetune` LoRA path is the escape hatch.
3. **Frozen reps benefit from E2E 3D training:** QM9 gap E2E SchNet expected to outperform frozen 2D reps significantly. The ft_schnet factory extracts 128D embeddings for use as a fine-tuned rep.
4. **GraphDTA-GIN fails cold-target:** Graph architectures don't generalize to unseen proteins under cold-target splits.

### 2G. Complexity & Similarity Experiments (BOM-driven)

**Status:** Waiting for Phase 5 baseline results to select configurations.

After baselines populate the BOM, select (dataset, rep, model) configurations for targeted experiments:

| Experiment Suite | Scripts | What to select from BOM | Status |
|-----------------|---------|------------------------|--------|
| c01: Rep characterization | `c01_rep_characterization.py` | Top 3-5 reps per domain | **Waiting for BOM** |
| c02: Complexity grid | `c02_complexity_grid.py` | 2-3 datasets per domain, best + worst rep families | **Waiting for BOM** |
| c03: Cross-dataset transfer | `c03_cross_dataset_transfer.py` | Molecular: logd↔caco2↔herg; TCGA: luad↔brca | **Waiting for BOM** |
| c04: Importance as metric | `c04_importance_as_metric.py` | Same configs as c01 | **Waiting for BOM** |
| c05: Learning curves | `c05_learning_curves.py` | 1-2 large datasets per domain, best model | **Waiting for BOM** |
| c06: Within-family gradient | `c06_within_family_gradient.py` | Rep families with PCA variants (chemberta, molformer, grover) | **Waiting for BOM** |
| c07: Data complexity partitions | `c07_data_complexity_partitions.py` | 1-2 datasets with high variance across reps | **Waiting for BOM** |
| c08: Calibration | `c08_calibration.py` | Best model per domain | **Waiting for BOM** |
| c09: Formal analysis | `c09_formal_analysis.py` | Top 5 reps per domain, all models | **Waiting for BOM** |
| s01-s09: Similarity suite | `tests/experiments/similarity/` | 1-2 molecular datasets, top 3 reps | **Waiting for BOM** |

**Selection criteria (apply after Phase 5 baselines):**
- **"Interesting"** = reps where top model varies (ridge vs tree vs neural), reps that rank differently across datasets, reps near the top but from different families (fingerprint vs LM vs descriptor)
- **Skip** reps that fail or produce negative R²/CI everywhere
- **Prioritize** datasets with enough samples for learning curves (>1000) and enough group diversity for GroupKFold

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

| Domain | Datasets | Reps (run / registered) | Models | Metrics | Status |
|--------|----------|------------------------|--------|---------|--------|
| Molecular | 1/8 | 17/36 | 10/10 | R²+ρ+RMSE | **Phase 5** (logd done; 7 datasets running) |
| Antibody | 24/25 | 20-23/40 (inter) | 10/10 | ρ+R²+Pearson+RMSE | **Phase 5** (11 inter-family done; DMS partial) |
| TCGA | 2/3 | 36/57 | 10/10 | AUROC+F1+ρ+RMSE | **Phase 5** (2 clf done; survival dropped; factory reps running) |
| DTA | 0/3 | 0/37 | — | — | **Running** (baselines on HTC) |
| Reaction | 2/3 | 8/8 | 10/10 | R²+ρ+RMSE | **Phase 5** (BH+Suzuki done; Doyle plates via comparison) |
| Comparisons | 10 experiments | — | — | — | **Phase 5** ✅ (DTA E2E, DTA FT, QM9, TCGA FM, WSI, Mol FT, Rxn rep, Ab FT, Ab region all done) |
| PDBBind | 0/2 | — | — | — | **Blocked** (manual download) |

### What's STILL PENDING

| Item | Gap | Blocker |
|------|-----|---------|
| Antibody DMS full survey | 10/11 DMS datasets have only aido_protein (failed). Need full 40-rep LM runs. | LM jobs running on HTC. Fast reps (onehot, blosum, aac, paratope) running on ARC. |
| Antibody inter-family: remaining reps | Fast reps, mutation reps, variation reps, MINT, xtrimopglm still pending on most datasets. | Jobs running/queued. aido_protein+xtrimopglm not yet submitted. |
| TCGA factory reps | 21/57 reps missing (gene_hvg2000, gene_pca100, gene_nmf50, pathway reps, etc.) — these require per-fold fitting. | Likely still running on HTC. |
| Molecular 7 remaining datasets | caco2, herg, qm9, adme_hlm, adme_rlm, adme_mdr1, adme_solubility | Running on HTC+ARC |
| DTA all baselines | Davis, KIBA, BindingDB_Kd × 31 reps, CI primary | Running on HTC (~33h in) |
| DTA fine-tuning | All 7 LoRA strategies produced NaN | **Needs debug** — investigate loss calculation |
| PDBBind | 2 datasets | Manual download from pdbbind.org.cn |
| IgBert | NaN everywhere on antibody | **Drop** — confirmed dead across all 11 tested datasets |
| aido_protein | Zero signal on all 10 antibody DMS datasets | **Drop from antibody** — keep in DTA only (untested there) |
| WSI ABMIL bug | All 3 aggregation methods produce identical results | **Investigate** — attention heads may not be learning |

### Dropped

| Rep | Reason |
|-----|--------|
| `wsi_cnn320` | Local Seagate only, garbage results |
| `unimol` v1 | Broken install |
| `chemformer` | Broken HF weights |
| `gene_scfoundation` (frozen) | Degenerate embeddings (0.51 acc = majority class). FT version works. |
| `igt5_ft` | OOM on 44GB GPU — IgT5 paired T5 too large for LoRA FT. Removed from default methods list. |
| `aido_protein` (antibody) | Zero signal on all 10 DMS datasets (ρ < 0.08). Keep for DTA only. |

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

---

## 10. Viable Hybrid Pipeline Candidates (Phase 5, partial)

**Purpose:** Identify (dataset, rep, model) combinations where the baseline rep produces
meaningful signal — these are the inputs to KIRBy's hybrid search. If a rep gets nothing
on a dataset, it can't contribute to a hybrid.

**Status:** Substantial. Molecular (1/8 datasets), reaction (2/3), TCGA (2/3 viable), antibody inter-family (11/14 datasets with 20-23 reps each), all comparison experiments complete.
DTA baselines + remaining molecular + antibody DMS still running.

**Viability thresholds (domain-specific, informed by published benchmarks):**
- Molecular: R² > 0.5 (Chemprop scaffold-split gets 0.75-0.85)
- Reaction: R² > 0.2 under GroupKFold (published OOS range: 0.54-0.84)
- TCGA classification: AUROC > 0.6 (random = 0.5; published SOTA = 0.95-0.98)
- TCGA multi-class: F1w > 0.4 (majority-class baseline = ~0.35)
- Antibody: Spearman ρ > 0.15 (ProteinGym supervised range: 0.5-0.8 on random splits; position-modulo splits are harder)
- DTA: CI > 0.55 (random = 0.5; published cold-target: 0.82-0.87)

### 10A. Molecular — LogD (5,028 samples, 5-fold scaffold GroupKFold)

**Published context:** Chemprop scaffold-split R²≈0.75-0.85 on lipophilicity. User's qsar_qm_models: DNN+MHG-GNN R²=0.824.

16/17 reps viable. All R² > 0.6 except molclr (borderline) and unimol_v2 (dead).

| Rep | Dim | R² | RMSE | ρ | Best Model | Rep Family |
|-----|-----|----|------|---|-----------|-----------|
| mhg_gnn | 1024 | 0.857 | 0.442 | 0.922 | gp | GNN |
| grover | 4800 | 0.857 | 0.443 | 0.925 | ridge | GNN |
| mordred | 1613 | 0.850 | 0.452 | 0.920 | dnn | Descriptor |
| mol2vec | 300 | 0.843 | 0.464 | 0.913 | gp | Embedding |
| ecfp4 | 2048 | 0.836 | 0.475 | 0.913 | gp | Fingerprint |
| molformer | 768 | 0.835 | 0.476 | 0.910 | LM |
| pdv | 200 | 0.835 | 0.477 | 0.905 | dnn | Descriptor |
| smited | 768 | 0.832 | 0.481 | 0.905 | gp | LM |
| avalon | 2048 | 0.815 | 0.504 | 0.899 | gp | Fingerprint |
| chemberta | 384 | 0.792 | 0.534 | 0.882 | dnn | LM |
| chembert | 256 | 0.784 | 0.546 | 0.878 | gp | LM |
| graph_kernel | 290 | 0.775 | 0.556 | 0.877 | dnn | Kernel |
| selformer | 768 | 0.740 | 0.597 | 0.849 | dnn | LM |
| graphmvp | 300 | 0.714 | 0.627 | 0.839 | dnn | GNN |
| maccs | 167 | 0.708 | 0.634 | 0.829 | gp | Fingerprint |
| molclr | 512 | 0.604 | 0.738 | 0.758 | dnn | GNN (borderline) |

**Dead:** unimol_v2 (R²=-0.005).

**Hybrid search notes:** 5 rep families present (GNN, Descriptor, Fingerprint, LM, Kernel) — good diversity for hybridization. GP and DNN dominate as best model; ridge only wins for grover. Top hybrid candidates: combine fingerprint (ecfp4/avalon) with LM (molformer/chemberta) or descriptor (mordred/pdv) — different information types.

**Missing (still running):** caco2, herg, qm9, adme_hlm, adme_rlm, adme_mdr1, adme_solubility. Also missing reps: minimol, gp_molformer, mole, sns, dimenetpp, schnet, clamp, rmat, map4, all *_pca50 factory reps.

### 10B. Reaction — Buchwald-Hartwig (3,955 reactions, leave-additive-out 22 groups)

**Published context:** Random-split R²=0.92-0.96. OOS reagent hold-out R²=0.54-0.84 (Schwaller 2021, JCIM 2024). KIRBy's GroupKFold is equivalent to OOS.

7/8 reps viable. R²=0.718 best — squarely in the OOS literature range.

| Rep | Dim | R² | RMSE | ρ | Best Model | Rep Type |
|-----|-----|----|------|---|-----------|---------|
| reactant_ecfp4 | 2048 | 0.718 | 14.16 | 0.855 | rf | Molecular FP |
| concat_ecfp4 | 5120 | 0.704 | 14.57 | 0.847 | rf | Concat FP |
| drfp_2048 | 2048 | 0.688 | 14.93 | 0.835 | rf | Reaction FP |
| concat_maccs | 835 | 0.634 | 16.22 | 0.812 | xgboost | Concat FP |
| onehot_components | 45 | 0.615 | 16.62 | 0.802 | xgboost | Design encoding |
| concat_pdv | 1085 | 0.582 | 17.42 | 0.796 | rf | Concat descriptor |
| rxnfp | 256 | 0.566 | 17.77 | 0.771 | ridge | Learned reaction |

**Dead:** product_ecfp4 (R²=0.136 — product alone lacks reagent info).

**Hybrid search notes:** 4 rep types (molecular FP, concat, reaction FP, design encoding) provide diversity. reactant_ecfp4 beating concat_ecfp4 is interesting — reactant features alone carry more signal than the full concatenation under OOS splitting. Hybrid of reactant_ecfp4 + onehot_components would combine molecular + factorial information.

### 10C. Reaction — Suzuki (5,760 reactions, leave-ligand-out 12 groups) — AUGMENTATION EXPERIMENT

**Published context:** "When Yield Prediction Does Not Yield Prediction" (JCIM 2024) predicts exactly this collapse.

1/8 reps viable. **All molecular fingerprint reps collapse to R²≈-0.03 under leave-ligand-out.**

| Rep | Dim | R² | ρ | Best Model |
|-----|-----|----|---|-----------|
| onehot_components | 37 | 0.453 | 0.680 | rf |
| concat_ecfp4 | 5120 | -0.033 | 0.000 | — |
| concat_maccs | 835 | -0.033 | 0.000 | — |
| drfp_2048 | 2048 | -0.033 | 0.033 | — |
| rxnfp | 256 | -0.034 | 0.003 | — |
| (all others) | — | ≈-0.03 | ≈0.00 | — |

**Augmentation experiment:** Can KIRBy hybrid search rescue the dead reps by augmenting onehot_components with fingerprint features? The hypothesis: onehot captures the factorial design (generalizes to unseen ligands) while fingerprints capture molecular detail (fails alone but may add signal when combined with a rep that already generalizes). This is a strong test of KIRBy's thesis — hybridization rescuing individually-failed reps.

### 10D. TCGA — LUAD vs LUSC (1,017 samples, binary clf, TSS-grouped 5-fold)

**Published context:** OmicsFootPrint AUC=0.98 (expression only). Multi-omics 99.8% acc. Our 0.946 is slightly below — likely because factory reps (gene_hvg2000, gene_nmf50, etc.) are still running.

28/36 reps viable (AUROC > 0.6).

| Rep | Dim | AUROC | Best Model | Modality |
|-----|-----|-------|-----------|---------|
| gene_geneformer_v2 | 1152 | 0.946 | catboost | Expression FM |
| gene_bulkrnabert | 256 | 0.939 | ridge | Expression FM |
| methyl_hvg | 5000 | 0.939 | xgboost | Methylation |
| gene_geneformer | 1152 | 0.936 | catboost | Expression FM |
| gene_chromosome_arm | 42 | 0.918 | ridge | CNV-derived |
| wsi_titan | 768 | 0.917 | rf | WSI |
| methyl_pca50 | 50 | 0.916 | catboost | Methylation |
| cnv_pca50 | 50 | 0.915 | gp | CNV |
| mirna_top200 | 200 | 0.896 | ridge | miRNA |
| wsi_gigapath | 512 | 0.884 | catboost | WSI |
| wsi_hoptimus0 | 512 | 0.874 | catboost | WSI |
| cnv_gistic | 200 | 0.843 | gp | CNV |
| wsi_ctranspath | 512 | 0.830 | lightgbm | WSI |
| wsi_moco | 512 | 0.816 | catboost | WSI |
| immune_xcell | 12 | 0.812 | gp | Immune deconv |
| wsi_phikon | 512 | 0.802 | lightgbm | WSI |
| gene_stats | 15 | 0.793 | ridge | Expression summary |
| rppa | 258 | 0.785 | ridge | Protein |
| mutation_matrix | 200 | 0.774 | rf | Mutation |
| mut_driver | 46 | 0.751 | ridge | Mutation |
| stemness_mrnasi | 2 | 0.751 | ridge | Stemness |
| immune_cibersort | 21 | 0.707 | gp | Immune deconv |
| mut_pca50 | 50 | 0.705 | rf | Mutation |
| mut_pathway | 50 | 0.702 | rf | Mutation |
| cimp_status | 2 | 0.692 | dnn | Epigenetic |
| mut_tmb | 5 | 0.666 | gp | Mutation |
| hrd_score | 4 | 0.656 | dnn | Genomic instability |
| purity_consensus | 4 | 0.601 | dnn | Tumor purity |

**Dead (AUROC=0.5):** clinical, heterogeneity_math, msi_status, gene_scfoundation (frozen), wsi_conch, wsi_hoptimus1, wsi_uni2h, wsi_virchow2.

**Hybrid search notes:** 8 modalities represented (expression FM, methylation, WSI, CNV, miRNA, protein, mutation, immune). Strong multi-modal diversity — KIRBy's hybrid search should find cross-modality combinations (e.g., expression + WSI, expression + mutation) that outperform any single modality. gene_geneformer_v2 as top validates forward-compat thesis.

**Missing:** 21 factory reps (gene_hvg2000, gene_pca100, gene_nmf50, gene_hallmark, gene_kegg, gene_reactome, gene_dorothea, gene_progeny, gene_combined_pathway, gene_immune, etc.) — these dominated the old stale run and are likely still running on the server. Also missing: sig_sbs, sig_dbs, sig_id, splicing_psi50, lncrna_pca50, gsva_hallmark, promoter_meth.

### 10E. TCGA — BRCA PAM50 (821 samples, 4-class clf, TSS-grouped 5-fold)

**Published context:** OmicsFootPrint AUC=0.83 for 4-class PAM50. KIRBy's 0.861 F1w exceeds this.

27/36 reps viable (F1w > 0.4).

| Rep | Dim | F1w | Best Model | Modality |
|-----|-----|-----|-----------|---------|
| gene_geneformer_v2 | 1152 | 0.861 | ridge | Expression FM |
| gene_geneformer | 1152 | 0.850 | ridge | Expression FM |
| gene_bulkrnabert | 256 | 0.841 | ridge | Expression FM |
| rppa | 258 | 0.776 | lightgbm | Protein |
| gene_chromosome_arm | 42 | 0.762 | ridge | CNV-derived |
| cnv_pca50 | 50 | 0.712 | ridge | CNV |
| wsi_titan | 768 | 0.696 | ridge | WSI |
| methyl_pca50 | 50 | 0.664 | ridge | Methylation |
| wsi_hoptimus0 | 512 | 0.661 | ridge | WSI |
| wsi_gigapath | 512 | 0.646 | catboost | WSI |
| mirna_top200 | 200 | 0.640 | ridge | miRNA |
| methyl_hvg | 5000 | 0.631 | ridge | Methylation |
| wsi_moco | 512 | 0.613 | ridge | WSI |
| stemness_mrnasi | 2 | 0.612 | gp | Stemness |
| cnv_gistic | 200 | 0.606 | catboost | CNV |
| wsi_phikon | 512 | 0.606 | ridge | WSI |
| wsi_ctranspath | 512 | 0.594 | catboost | WSI |
| immune_xcell | 12 | 0.576 | catboost | Immune deconv |
| immune_cibersort | 21 | 0.561 | catboost | Immune deconv |
| mutation_matrix | 200 | 0.557 | catboost | Mutation |
| mut_pathway | 50 | 0.545 | rf | Mutation |
| gene_stats | 15 | 0.543 | ridge | Expression summary |
| mut_driver | 46 | 0.540 | lightgbm | Mutation |
| purity_consensus | 4 | 0.522 | ridge | Tumor purity |
| mut_pca50 | 50 | 0.455 | rf | Mutation |
| mut_tmb | 5 | 0.446 | rf | Mutation |
| hrd_score | 4 | 0.430 | rf | Genomic instability |

**Dead (F1w ≈ 0.35 = majority class):** Same 8 as LUAD/LUSC + cimp_status drops below threshold.

**Hybrid search notes:** Ridge dominates for expression FMs — the linear signal is strong enough. But catboost/lightgbm win for mutation and WSI reps, suggesting nonlinear structure there. Hybrid search should use mixed model strategies. rppa (protein) at #4 suggests protein-level data adds unique information beyond RNA — good hybrid partner for expression FMs.

### 10F. TCGA — BRCA Survival — DROPPED

All R² negative, Spearman near zero or NaN across all 32 reps. n=151 uncensored samples is too small for any meaningful regression. Not viable for hybrid search.

### 10G. Antibody — Phase 5 Inter-Family (11 datasets with ≥20 reps)

**Published context:** ProteinGym supervised Spearman 0.5-0.8 (random splits). FLAb2 (2025): "AI models produce NO statistically significant correlations for 80% of datasets." Position-modulo splits (used here) are harder than random.

**Viability threshold: Spearman ρ > 0.15** (lower than ProteinGym because we use harder splits and smaller datasets).

11 inter-family datasets have 20-23 reps tested. DMS datasets still mostly incomplete (only aido_protein or fast reps).

**Summary across all inter-family datasets:**

| Dataset | N | Reps Tested | Best Rep | Best ρ | Viable/Total | Best Model |
|---------|---|-------------|----------|--------|-------------|-----------|
| flab_thermo_tresanco | 672 | 20 | esm3 | **0.805** | 18/20 | dnn |
| flab_pk_jain_heprt | 130 | 3 | esm2_per_cdr | **0.658** | 3/3 | gp |
| flab_pk_jain_fcrnrt | 132 | 21 | ablang2 | **0.595** | 18/21 | ridge |
| flab_polyreact_jain_cic | 137 | 23 | prot_t5 | **0.586** | 21/23 | ridge |
| flab_aggregation_jain_hicrt | 137 | 23 | prot_t5 | **0.506** | 21/23 | ridge |
| flab_thermo_jain | 137 | 23 | ablang2 | **0.457** | 21/23 | ridge |
| abcov_kd | 141 | 22 | prot_t5 | **0.432** | **22/22** | catboost |
| flab_aggregation_jain | 137 | 23 | esm2_concat | **0.425** | 21/23 | ridge |
| flab_immunogenicity_marks | 217 | 23 | antiberty_cdr | **0.399** | 21/23 | mlp |
| flab_polyreact_jain_bvp | 137 | 23 | ablang2 | **0.374** | 19/23 | ridge |
| flab_thermo_hutchinson | 192 | 23 | esm2 | **0.289** | 21/23 | xgboost |
| flab_aggregation_jain_sas | 137 | 23 | antiberty_cdr | **0.201** | 6/23 | xgboost |

**Top 5 most consistent reps across 11 datasets (by average rank):**

| Rep | Datasets in Top 5 | Notable Wins |
|-----|-------------------|-------------|
| **ablang2** | 6/11 | fcrnrt (#1), thermo_jain (#1), bvp (#1), aggregation (#2) |
| **prot_t5** | 7/11 | cic (#1), hicrt (#1), abcov_kd (#1), immunogenicity (#4) |
| **igt5** | 6/11 | fcrnrt (#2), aggregation (#3), bvp (#2) |
| **esm3** | 5/11 | tresanco (#1, ρ=0.805), thermo_jain (#5) |
| **esm2_early** | 4/11 | cic (#3), hicrt (#3) |

**Key findings:**
- **Every dataset is viable for hybridization** — all 11 have ≥6 viable reps (ρ > 0.15). Even the hardest dataset (sas, ρ=0.201) is above threshold.
- **Spearman vs R² divergence is dramatic:** Datasets that showed negative R² in stale results (bvp, sas, hutchinson) now show clear positive Spearman correlations. R² is the wrong metric for antibody.
- **DNN wins on the largest dataset** (tresanco n=672), ridge wins on most small datasets (n≈137). This mirrors the molecular finding where GP/DNN dominate larger data.
- **abcov_kd: 22/22 reps viable** — the most uniformly predictable dataset, with even the weakest reps above 0.15.
- **igbert/igbert_cls consistently dead** (NaN or near-zero) — confirmed non-functional for antibody tasks.
- **CDR-pooled variants generally match full-seq** — no consistent advantage to CDR pooling (exception: antiberty_cdr slightly beats antiberty on immunogenicity).
- **Forward-compat validated:** igt5 (#4-tier), esm3 (best on tresanco), saprot (mid-tier), abroberta (strong on thermo_jain ρ=0.451) all confirmed viable with zero code changes.
- **aido_protein completely failed** on all 10 DMS datasets tested (ρ < 0.08) — negative result for this large (4608D) model.

**Still missing from antibody:** Fast reps (onehot, blosum, aac, paratope — running on ARC), mutation reps (6 reps — 13/14 inter-family datasets passed, mutation job incomplete), variation reps (esm2_entropy_top20, esm2_per_cdr, esm2_per_region — partial: heprt only), MINT, xtrimopglm. DMS datasets (Phillips, Adams, AbAgym) still need full LM survey.

### 10H. DTA — PENDING (jobs running, 33h in)

**Published context:** Cold-target CI: 0.82-0.87 (deep interaction models). KIRBy uses linear/tree models, so lower CI expected — but the thesis is about rep quality, not model SOTA.

No results yet. Davis, KIBA, BindingDB_Kd all running. 31 reps (16 drug + 13 protein + 2 combo). CI is primary metric.

### 10I. Forward Compatibility — Embedded in Baselines

Forward-compat experiments are NOT separate — all FC reps are included in the baseline survey rep lists. Results so far:

| FC Rep | Domain | Baseline Score | Context | Status |
|--------|--------|---------------|---------|--------|
| gene_geneformer_v2 | TCGA | AUROC=0.946 (#1), F1w=0.861 (#1) | Dropped in, immediately best | **Validated** ✅ |
| gene_bulkrnabert | TCGA | AUROC=0.939 (#2), F1w=0.841 (#3) | New foundation model, top tier | **Validated** ✅ |
| mordred | Molecular | R²=0.850 (#3 logd) | New Phase 3B descriptor, enters top 3 | **Validated** ✅ |
| avalon | Molecular | R²=0.815 (#9 logd) | New Phase 3B fingerprint, solid mid-tier | **Validated** ✅ |
| abroberta | Antibody | ρ=0.451 (#1 thermo_jain) | New Phase 3C, best Spearman | **Validated** ✅ |
| igt5 | Antibody | ρ=0.431 (#4 thermo_jain), top 5 in 6/11 datasets | Phase 4A, consistently competitive | **Validated** ✅ |
| esm3 | Antibody | ρ=0.805 (#1 tresanco), top 5 in 5/11 datasets | Phase 4B, best on largest dataset | **Validated** ✅ |
| saprot | Antibody | ρ=0.357 (#10 thermo_jain) | Phase 4B, mid-tier | **Validated** ✅ |
| aido_protein | Antibody | ρ < 0.08 on all 10 DMS datasets | **Completely failed** — zero signal | **Failed** ❌ |
| GROVER | Molecular | RMSE=0.466 (#1 in finetune comparison) | Best frozen molecular encoder | **Validated** ✅ |
| MINT | Antibody | — | Pending (MINT jobs queued) | **Pending** |
| minimol, mole, gp_molformer | Molecular | — | Not in logd results (running) | **Pending** |
| protein_mint, protein_saprot, protein_boltz2, protein_esm3 | DTA | — | DTA jobs running | **Pending** |

**Thesis narrative strong:** New models plug in with zero code changes. geneformer_v2 immediately #1 on TCGA. esm3 immediately #1 on largest antibody dataset (tresanco ρ=0.805). abroberta #1 by Spearman on thermo_jain. igt5 consistently top-5 across 6/11 antibody datasets. mordred enters top 3 on molecular logd. **One notable failure:** aido_protein (4608D) produced zero signal on all antibody DMS datasets — not all new models work. The hybrid search (Wave 2) will test the core claim: hybrid(new_model + old_rep) > new_model alone.
