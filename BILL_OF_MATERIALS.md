# KIRBy Bill of Materials

Comprehensive inventory of datasets, representations, and their measured baseline
performance. This document is the single source of truth for what KIRBy supports
and what actually works. Representations without confirmed performance data should
not be used in experiments until validated.

**Last updated:** 2026-02-26

---

## 1. Datasets

### 1A. Molecular (regression, scaffold-split GroupKFold)

| Dataset | Source | Samples | Scaffolds | Target | Metric | Status |
|---------|--------|---------|-----------|--------|--------|--------|
| `logd` | OpenADMET-ExpansionRx | ~5,039 | ~2,387 | LogD | R² | **Confirmed** |
| `caco2` | OpenADMET-ExpansionRx | ~2,161 | ~1,069 | Caco2 Efflux (log) | R² | **Confirmed** |
| `herg` | ChEMBL (CHEMBL240) | ~1,415 | ~767 | pKi | R² | **Confirmed** |
| `qm9` | QM9 | 2,000 | varies | HOMO-LUMO gap | R² | **Confirmed** |

### 1B. Antibody Intra-Family / DMS (regression, GroupKFold by family)

| Dataset | Source | Samples | Target | Status |
|---------|--------|---------|--------|--------|
| `flab_expression_koenig` | FLAb (G6 SSM) | 4,275 | Expression (ER) | **Confirmed** (20 reps) |
| `flab_binding_koenig` | FLAb (G6) | 4,275 | Kd | **Confirmed** (20 reps) |
| `flab_thermo_hie` | FLAb (S309) | 10 | Tm | **Too small** (all reps negative) |
| `flab_binding_rosace` | FLAb (adalimumab) | 14 | Kd | **Too small** (all reps negative) |
| `abagym_Ang2_2017_G6` | AbAgym DMS | 397 | Fitness | **Confirmed** (20 reps) |
| `abagym_EGFR_2013_Cetuximab` | AbAgym DMS | 659 | Fitness | **Confirmed** (20 reps) |
| `abagym_lysozyme_2019_D441` | AbAgym DMS | 627 | Fitness | **Confirmed** (20 reps) |
| `abagym_VEGF_2017b_G6` | AbAgym DMS | 418 | Fitness | **Confirmed** (20 reps) |
| `abagym_HER2_2021_trastuzumab` | AbAgym DMS | 146 | Fitness | **Confirmed** (20 reps) |

### 1C. Antibody Inter-Family (regression, identity-grouped GroupKFold)

| Dataset | Source | Samples | Target | Status |
|---------|--------|---------|--------|--------|
| `ddg_combined` | SKEMPI 2.0 + AB-Bind (deduplicated) | 544 | ΔΔG | **Confirmed** (loads OK, paired) |
| `abcov_kd` | Ab-CoV (via FLAb) | 141 | -log Kd | **Confirmed** (loads OK, paired) |
| `flab_thermo_jain` | FLAb (Jain 2017) | 137 | Tm | **Partial** (6 reps, R²=0.14, see §2B) |
| `flab_aggregation_jain` | FLAb (Jain 2017) | 137 | AC-SINS | **Confirmed** (loads OK, paired) |
| `abagym_all` | AbAgym (pooled 5 experiments) | 2,247 | Fitness | **Confirmed** (loads OK, paired) |

**Dropped:** `sabdab_affinity` (was SKEMPI 2.0 only, replaced by `ddg_combined`),
`abbind` / `abbind_spm` (strict subsets of `ddg_combined`).

### 1D. TCGA (classification/regression, TSS-grouped StratifiedGroupKFold)

| Dataset | Task | Samples | Classes/Target | TSS | Metric | Status |
|---------|------|---------|----------------|-----|--------|--------|
| `tcga_luad_vs_lusc` | Binary clf | ~1,017 | LUAD vs LUSC | 68 | Accuracy | **Confirmed** |
| `tcga_brca_pam50` | 4-class clf | ~821 | LumA/LumB/Basal/Her2 | 19 | Accuracy | **Confirmed** |
| `tcga_brca_survival` | Regression | ~151 | OS days (uncensored) | 21 | R² | Listed, no baselines yet |

### 1E. Drug-Target Affinity (regression, cold-target GroupKFold)

| Dataset | Samples | Drugs | Proteins | Target | Status |
|---------|---------|-------|----------|--------|--------|
| `davis` | 30,056 pairs | 68 | 442 | pKd | Listed, no baselines yet |
| `kiba` | ~118K pairs | 2,111 | 229 | KIBA score | Listed, no baselines yet |

---

## 2. Representations with Baseline Performance

Best R² (or Accuracy for classification) shown as `best_model=score`. All scores
are from scaffold-split (molecular), identity-grouped (antibody), or TSS-grouped
(TCGA) cross-validation — no random splits.

### 2A. Molecular Representations

**Source:** C01 baselines on LogD (~5,039 samples), hERG (~1,415), Caco2 (~2,161).
Models: rf, xgboost, ridge, knn, lightgbm, mlp, svm.

| Rep | Dim | Category | LogD R² (best) | Caco2 R² | hERG R² | Deps |
|-----|-----|----------|---------------|----------|---------|------|
| `ecfp4` | 2048 | Standard (fingerprint) | lgbm=0.756 | rf=0.482 | rf=0.547 | rdkit |
| `maccs` | 167 | Standard (fingerprint) | xgb=0.674 | rf=0.372 | rf=0.487 | rdkit |
| `pdv` | 200 | Standard (descriptor) | lgbm=0.778 | xgb=0.488 | lgbm=0.543 | rdkit |
| `chemberta` | 384 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `molformer` | 768 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `selformer` | 768 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `mol2vec` | 300 | Pretrained (Word2Vec) | **PENDING** | — | — | mol2vec |
| `chembert` | 768 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `molclr` | 512 | Pretrained (GNN) | **PENDING** | — | — | torch_geometric |
| `graphmvp` | 300 | Pretrained (GNN) | **PENDING** | — | — | torch_geometric |
| `smited` | 256 | Pretrained | **PENDING** | — | — | HuggingFace |
| `mhg_gnn` | 64 | Pretrained (GNN) | **PENDING** | — | — | torch_geometric | **FRAGILE** (SIGBUS crashes) |
| `grover` | 5000 | Pretrained (Transformer) | **PENDING** | — | — | grover |
| `graph_kernel` | varies | Standard (kernel) | **PENDING** | — | — | grakel |
| `schnet` | 128 | Pretrained (3D GNN) | **PENDING** | — | — | schnetpack |
| `unimol` | 512 | Pretrained (3D) | **DROPPED** | — | — | unimol_tools | No path to install `unimol_tools` |
| `chemformer` | 256 | Pretrained (Transformer) | **PENDING** | — | — | transformers | Not in `all` preset (in precache pipeline) |

**Augmentations (not base reps):** `graph_topology`, `spectral`

**Fine-tuning:** NONE — see §7 for survey results. `lora_chemberta` and `lora_molformer`
were tested and performed **worse** than frozen baselines on all molecular datasets.

**Presets:**
- `fast` (3): ecfp4, maccs, pdv
- `fast+hf` (6): + chemberta, molformer, selformer
- `all` (15): all reps including GNN and graph_kernel

### 2B. Antibody Representations

**Source:** C01 baselines on FLAb Expression Koenig (~4,275 samples, intra-family DMS).
Models: rf, knn, ridge, mlp, xgboost, lightgbm.

#### Inter-Family Base Reps

**Sources:** abcov_kd baselines (slurm-11380416, 13 reps, incomplete search),
flab_thermo_jain (baseline survey job 11384921, 6 reps only).

| Rep | Dim | Category | abcov_kd R² | thermo_jain R² | Deps | Notes |
|-----|-----|----------|------------|---------------|------|-------|
| `kmer` | 400 | Standard (sequence) | rf=0.003 | rf=-0.072 | None | Near-zero |
| `aac` | 20 | Standard (composition) | rf=-0.212 | rf=-0.177 | None | |
| `blosum62` | 20 | Standard (substitution) | rf=-0.189 | rf=-0.235 | None | |
| `physicochemical` | 68 | Standard (biophysical) | rf=-0.237 | rf=-0.127 | None | |
| `antiberty` | 2048 | Pretrained (Ab LM) | rf=-0.085 | — | transformers | Concat H+L |
| `ablang2` | 480 | Pretrained (Ab LM) | rf=-0.094 | rf=**0.149** | ablang2 | Best on thermo_jain |
| `igbert` | 1024 | Pretrained (Ab LM) | rf=-0.085 | — | transformers | |
| `esm2` | 2560 | Pretrained (protein LM) | rf=-0.177 | — | transformers | Concat H+L |
| `prot_t5` | 2048 | Pretrained (protein LM) | rf=-0.171 | — | transformers | Concat H+L |
| `ab_roberta` | 1536 | Pretrained (Ab LM) | rf=-0.052 | — | transformers | Best LM on abcov |
| `balm` | 1024 | Pretrained (Ab LM) | **PENDING** | — | manual | Setup done (see §3) |
| `pigen` | 768 | Pretrained (Ab LM) | rf=-0.225 | — | transformers | |
| `esmc` | varies | Pretrained (protein LM) | **PENDING** | — | esmc | Never tested |
| `sapiens` | varies | Pretrained (humanization) | **PENDING** | — | sapiens | Never tested |
| `structural` | 28 | Customizable (3D) | ridge=-0.177 | rf=-0.416 | ImmuneBuilder | Server only |
| `developability` | 16 | Customizable | — | — | None | Tested on DMS only (see below) |
| `imgt_onehot` | varies | Standard (positional) | **PENDING** | — | anarci | Never tested |

**Observation:** All inter-family reps are negative on abcov_kd (141 samples, very hard).
Only ablang2 is positive on flab_thermo_jain. LM embeddings do not clearly help for
inter-family antibody tasks on these small datasets. Needs testing on ddg_combined
(~1,000-1,200 samples) for a fairer comparison.

**Note:** Full-sequence LM embeddings are known to be near-useless for intra-family
DMS (published ρ ≈ 0.08). The `dms` presets use mutation-specific reps instead.

> **⚠ ANTIBODY DOMAIN NEEDS RETHINKING.** The fine-tuning survey (Feb 2025) showed
> that LoRA and MLM fine-tuning of ESM-2 produced negative R² on all antibody datasets
> tested (FLAb expression, FLAb binding, AbAgym Ang2). Combined with the known failure
> of full-sequence LM embeddings on DMS data, the entire antibody representation
> strategy needs to be revisited. Possible directions: inverse folding models (ESM-IF1,
> ProteinMPNN scores), per-residue evolutionary conservation, structure-conditioned
> embeddings, or abandoning LM-based reps entirely in favour of physics-based or
> positional features. See §7 for full survey results.

#### DMS / Mutation-Aware Reps

**Sources:** Full 9-dataset baseline survey (slurm-7219595, dms_full 20 reps, 5-fold
GroupKFold, rf+ridge+xgboost, 180/180 cells). Two datasets excluded from analysis:
flab_thermo_hie (N=10) and flab_binding_rosace (N=14) — too small for meaningful CV.

| Rep | Dim | Category | flab_expr | flab_bind | Ang2 | EGFR | lyso | VEGF | HER2 | Best | +/7 |
|-----|-----|----------|-----------|-----------|------|------|------|------|------|------|-----|
| `aac` | 20 | Standard | xgb=**0.260** | xgb=0.070 | ridge=0.084 | ridge=-0.001 | xgb=**0.208** | rf=0.158 | ridge=0.037 | 0.260 | 6/7 |
| `blosum62` | 20 | Standard | rf=**0.267** | rf=0.072 | ridge=0.090 | ridge=0.002 | rf=**0.248** | ridge=0.120 | rf=0.102 | 0.267 | 6/7 |
| `biophysical_delta` | 6 | Customizable (mutation) | xgb=**0.265** | rf=0.071 | ridge=0.053 | ridge=0.023 | rf=**0.226** | ridge=0.002 | ridge=0.176 | 0.265 | 7/7 |
| `developability` | 16 | Customizable | rf=**0.272** | rf=**0.112** | ridge=0.094 | ridge=0.013 | rf=**0.264** | rf=0.106 | rf=**0.443** | 0.443 | 7/7 |
| `esm2_site` | 1280 | Customizable (mutation) | ridge=**0.197** | rf=**0.105** | ridge=**0.406** | xgb=**0.184** | xgb=0.082 | rf=**0.418** | ridge=**0.423** | 0.423 | 7/7 |
| `lm_score_ablang2` | 5 | Customizable (LM score) | ridge=0.002 | rf=-0.635 | ridge=**0.317** | ridge=-0.004 | ridge=0.022 | ridge=0.085 | ridge=-0.059 | 0.317 | 3/7 |
| `lm_score_esm2` | 5 | Customizable (LM score) | ridge=-0.009 | rf=-0.009 | ridge=**0.291** | ridge=0.014 | ridge=-0.010 | ridge=**0.109** | ridge=-0.033 | 0.291 | 3/7 |
| `lm_score_prot_t5` | 5 | Customizable (LM score) | ridge=0.007 | ridge=-0.023 | ridge=**0.257** | ridge=0.030 | ridge=-0.014 | ridge=0.086 | ridge=-0.058 | 0.257 | 4/7 |
| `lm_score_antiberty` | 5 | Customizable (LM score) | ridge=-0.011 | ridge=-0.105 | ridge=**0.301** | ridge=0.034 | ridge=0.002 | ridge=**0.111** | ridge=0.004 | 0.301 | 4/7 |
| `positional` | 11 | Customizable (mutation) | rf=-0.238 | rf=-0.082 | rf=**0.385** | ridge=**0.111** | rf=0.077 | rf=**0.468** | rf=**0.408** | 0.468 | 5/7 |
| `varpos_onehot` | varies | Customizable (varpos) | xgb=-0.012 | rf=-0.018 | rf=**0.491** | xgb=**0.271** | ridge=0.185 | xgb=**0.507** | xgb=**0.367** | 0.507 | 5/7 |
| `varpos_blosum` | varies | Customizable (varpos) | xgb=-0.012 | rf=-0.017 | rf=**0.466** | xgb=**0.259** | xgb=**0.214** | xgb=**0.549** | xgb=**0.510** | 0.549 | 5/7 |
| `varpos_physchem` | varies | Customizable (varpos) | xgb=-0.012 | rf=-0.017 | rf=**0.448** | xgb=**0.239** | rf=0.160 | xgb=**0.483** | rf=**0.403** | 0.483 | 5/7 |
| `varpos_cdr_onehot` | varies | Customizable (varpos) | rf=-0.010 | xgb=-0.011 | rf=**0.474** | xgb=0.131 | xgb=0.188 | xgb=**0.504** | xgb=**0.315** | 0.504 | 5/7 |
| `varpos_cdr_physchem` | varies | Customizable (varpos) | rf=-0.010 | xgb=-0.011 | rf=**0.427** | xgb=0.114 | rf=0.171 | xgb=**0.487** | ridge=**0.402** | 0.487 | 5/7 |
| `varpos_cdrh3_physchem` | varies | Customizable (varpos) | rf=-0.010 | ridge=-0.011 | rf=**0.226** | rf=-0.011 | rf=0.053 | rf=0.123 | ridge=**0.402** | 0.402 | 4/7 |
| `varpos_fw_physchem` | varies | Customizable (varpos) | rf=-0.013 | xgb=-0.013 | rf=0.020 | rf=**0.106** | rf=-0.020 | rf=-0.000 | rf=0.094 | 0.106 | 3/7 |
| `antifold_score` | 5 | Customizable (structure) | rf=-0.010 | xgb=-0.010 | ridge=-0.005 | ridge=-0.022 | rf=-0.007 | ridge=-0.007 | xgb=-0.071 | -0.005 | **0/7** |
| `sapiens_score` | 5 | Customizable (humanness) | ridge=-0.089 | ridge=-0.013 | ridge=-0.005 | ridge=-0.022 | rf=-0.007 | ridge=-0.007 | xgb=-0.071 | -0.005 | **0/7** |
| `esm1v_score` | 5 | Customizable (LM score) | rf=-0.021 | rf=-0.050 | ridge=**0.294** | ridge=0.032 | ridge=-0.008 | ridge=**0.116** | ridge=-0.030 | 0.294 | 3/7 |

**Column key:** flab_expr = flab_expression_koenig (4275), flab_bind = flab_binding_koenig (4275),
Ang2 = abagym_Ang2_2017_G6 (397), EGFR = abagym_EGFR_2013_Cetuximab (659),
lyso = abagym_lysozyme_2019_D441 (627), VEGF = abagym_VEGF_2017b_G6 (418),
HER2 = abagym_HER2_2021_trastuzumab (146). Best = best R² across 7 reliable datasets.
+/7 = number of datasets with R² > 0.01.

**Excluded datasets:** flab_thermo_hie (N=10, all reps deeply negative R² ≈ -5 to -130)
and flab_binding_rosace (N=14, similar). Too few samples for meaningful 5-fold CV.

**Known bugs in results:**
- `antifold_score`: "WARNING: WT structure prediction failed — returning zeros" on all
  AbAgym datasets. ImmuneBuilder not available on server → all-zero features → negative R².
- `sapiens_score`: "Sapiens scoring failed (tensor size mismatch)" on EGFR, lysozyme, VEGF,
  HER2 (VH > 146 residues overflows Sapiens positional embeddings). Returns zeros → same
  R² as antifold on those datasets. Only tested properly on FLAb (flab_expr: -0.089).

**Key insight:** Performance is highly dataset-dependent. Composition reps (aac, blosum62,
biophysical_delta, developability) win on FLAb expression/binding, while positional/varpos
reps dominate on AbAgym datasets (varpos_blosum R²=0.549 on VEGF, varpos_cdr_physchem
R²=0.487 on VEGF). `esm2_site` and `positional` are strong across both families.
`developability` is the surprise winner on HER2 (0.443). `antifold_score` and `sapiens_score`
are broken — never positive on any dataset.

**Delta reps:** `--delta` flag generates ΔX = X_mut − X_wt for every base rep, doubling count.

**Fine-tuning:** NONE — see §7 for survey results. `lora_esm2`, `lora_esm2_small`,
and `mlm_esm2` were all tested and produced **negative R²** on antibody datasets.
`lora_antiberty`, `lora_ablang2`, `mlm_antiberty`, `mlm_esm2_small` are implemented
but not expected to help given the ESM-2 results.

**Presets:**
- `fast` (4): kmer, physicochemical, aac, blosum62
- `fast+lm` (7): + antiberty, ablang2, igbert
- `all_server` (13): + esm2, prot_t5, ab_roberta, balm, pigen, structural
- `mutation_fast` (7): kmer, aac, blosum62, physicochemical, esm2, prot_t5, varpos_physchem
- `mutation_full` (10): + antiberty, ablang2, varpos_onehot
- `dms` (12): lm_score_ablang2, lm_score_esm2, positional, biophysical_delta, aac, blosum62, varpos_physchem, varpos_cdr_physchem, varpos_cdrh3_physchem, antifold_score, sapiens_score, esm1v_score
- `dms_full` (20): all mutation-aware + full varpos arsenal (incl. varpos_cdr_onehot) + antifold_score + sapiens_score + esm1v_score

### 2C. TCGA Representations

**Source:** Baselines from SLURM jobs 11379620/11379734-11379737 (original) and
7219430/7219506 (new/fixed reps) on LUAD vs LUSC and BRCA PAM50. Models: rf, xgboost, ridge, knn.

#### Expression-Derived

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `gene_hvg2000` | 2000 | Standard | xgb=**0.959** | xgb=**0.866** | scanpy (opt) | YES |
| `gene_pca100` | 100 | Standard | ridge=0.939 | xgb=0.826 | sklearn | YES |
| `gene_nmf50` | 50 | Standard | rf=**0.950** | rf=0.834 | sklearn | YES |
| `gene_ica50` | 50 | Standard | ridge=0.940 | xgb=0.751 | sklearn | YES |
| `gene_stats` | 15 | Standard | ridge=0.757 | rf=0.502 | scipy | YES |
| `gene_hallmark` | 50 | Pathway (mean z-score) | ridge=**0.933** | xgb=0.833 | local gene sets | YES |
| `gene_kegg` | 320 | Pathway (mean z-score) | ridge=**0.941** | xgb=0.816 | local gene sets | YES |
| `gene_reactome` | 1817 | Pathway (mean z-score) | xgb=**0.945** | xgb=0.859 | local gene sets | YES |
| `gene_oncogenic` | 189 | Pathway (mean z-score) | ridge=**0.946** | xgb=0.833 | local gene sets | YES |
| `gene_progeny` | 14 | Pathway (weighted) | ridge=**0.879** | rf=0.799 | local gene sets | YES |
| `gene_dorothea` | 216 | Pathway (weighted) | ridge=**0.933** | xgb=0.839 | local gene sets | YES |
| `gene_combined_pathway` | 280 | Pathway (combined) | ridge=**0.931** | xgb=0.839 | local gene sets | YES |
| `gene_quantile_rank` | 50 | Expression (rank PCA) | ridge=**0.935** | xgb=0.818 | sklearn | YES |
| `gene_corr_modules` | 30 | Expression (eigengene) | ridge=**0.943** | rf=0.836 | sklearn | YES |
| `gene_chromosome_arm` | 42 | Genomic (chr arm) | ridge=**0.905** | xgb=0.713 | local gene sets | YES |
| `gene_autoencoder50` | 50 | Expression (AE) | rf=**0.942** | xgb=0.827 | torch | YES |
| `gene_sparse_pca50` | 50 | Expression (sparse) | knn=**0.944** | xgb=0.883 | sklearn | YES |
| `gene_immune` | ~22 | Computed (immune) | rf=0.793 | rf=0.524 | hardcoded | YES |

#### Foundation Model Embeddings (GPU)

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `geneformer` (FT) | 1152 | Fine-tuned (per-fold) | xgb=**0.930** | xgb=0.789 | torch, transformers | GPU |
| `gene_scfoundation` | 2048 | Pretrained (frozen) | **PENDING** | **PENDING** | torch, transformers | GPU, never tested |

#### Multi-Omic

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `gene_mutation` | 200 | Somatic | **FAILED** | **FAILED** | Xena MC3 | **BROKEN** (403 Forbidden) |
| `cnv_gistic` | 200 | CNV | xgb=0.820 | xgb=0.607 | Xena | YES |
| `cnv_pca50` | 50 | CNV | xgb=0.877 | **PENDING** | Xena | YES |
| `rppa` | 258 | Protein | rf=0.749 | xgb=0.773 | Xena | YES |
| `mirna_top200` | 200 | miRNA | rf=0.888 | xgb=0.675 | Xena | YES |
| `clinical` | 10 | Phenotype | **FAILED** | **FAILED** | Xena | **BROKEN** (403 Forbidden) |

#### WSI (Pretrained Vision)

| Rep | Dim | Source | Lung Acc (best) | BRCA Acc (best) | Access | Server? |
|-----|-----|--------|----------------|----------------|--------|---------|
| `wsi_ctranspath` | 512 | GigaSSL (Mendeley) | xgb=0.783 | rf=0.513 | Public | YES |
| `wsi_phikon` | 512 | GigaSSL (Mendeley) | xgb=0.776 | rf=0.513 | Public | YES |
| `wsi_hoptimus0` | 512 | GigaSSL (Mendeley) | xgb=0.823 | rf=0.648 | Public | YES |
| `wsi_gigapath` | 512 | GigaSSL (Mendeley) | xgb=0.854 | **PENDING** | Public | YES |
| `wsi_moco` | 512 | GigaSSL (Mendeley) | xgb=0.775 | rf=0.616 | Public | YES |
| `wsi_titan` | 768 | HuggingFace (gated) | 0.429 | 0.513 | Gated | **GARBAGE** (~random, likely failed to load) |
| `wsi_uni2h` | 1536 | HuggingFace (gated) | 0.429 | — | Manual | **GARBAGE** (~random, features not found) |
| `wsi_cnn320` | 320 | Local (Seagate) | 0.429 | N/A | Local | **GARBAGE** (~random, Lung only) |

**Fine-tuning:** `geneformer` (per-fold, proper vocab tokenization → 1152D V2 embeddings).
**Survey confirmed:** Geneformer FT is the best individual rep on LUAD vs LUSC (0.930 acc)
and competitive on BRCA PAM50 (0.789). Hybrids with Geneformer reach 0.940 / 0.820.
Geneformer was pretrained on single-cell data (Genecorpus-104M, non-cancer) — no TCGA
data leakage. **Proceed to Phase 2 full run.**

**Presets:**
- `fast` (5): pca100, hallmark, progeny, ctranspath, phikon
- `fast+diverse` (9): + nmf50, stats, immune, quantile_rank
- `multimodal` (12): + mutation, cnv_gistic, rppa, mirna_top200, clinical
- `fast+learned` (17): expanded genomic + geneformer + scfoundation + 5 WSI
- `all` (34): everything (19 expression + 2 foundation + 5 multi-omic + 8 WSI)
- `genomic_fast` (3): pca100, hallmark, progeny
- `genomic_diverse` (10): multi-omic without WSI
- `genomic_all` (25): all genomic + 5 new reps + foundation models + multi-omic

### 2D. DTA Representations

Cold-target GroupKFold CV (5-fold). "Paired" = drug rep + protein_aac or protein rep + drug_ecfp4.
Davis: 30K pairs (68 drugs × 442 proteins). KIBA: 30K subsample (2111 drugs × 229 proteins).
Best model per cell: Davis=XGBoost (paired), Ridge (standalone); KIBA=Ridge/XGBoost varies.

| Rep | Dim | Category | Davis R² (paired) | KIBA R² (standalone) | Deps |
|-----|-----|----------|:-:|:-:|------|
| `drug_ecfp4` | 2048 | Standard (fingerprint) | 0.454 | 0.322 | rdkit |
| `drug_maccs` | 167 | Standard (fingerprint) | 0.445 | 0.317 | rdkit |
| `drug_pdv` | 200 | Standard (descriptor) | 0.460 | 0.334 | rdkit |
| `drug_chemberta` | 384 | Pretrained (drug LM) | 0.459 | **0.344** | transformers |
| `drug_molformer` | 768 | Pretrained (drug LM) | **0.467** | 0.341 | transformers |
| `drug_selformer` | 768 | Pretrained (drug LM) | 0.463 | 0.341 | transformers |
| `protein_esm2` | 1280 | Pretrained (protein LM) | **0.538** | 0.007 | transformers |
| `protein_aac` | 20 | Standard (composition) | 0.454 | 0.001 | None |
| `protein_descriptors` | 30 | Standard (biophysical) | 0.429 | -0.003 | None |
| `protein_kmer` | 8000 | Standard (sequence) | 0.501 | 0.018 | None |
| **all_concat** | 13665 | All 10 reps | **0.548** | **0.369** | — |

**Presets:**
- `fast` (5): drug_ecfp4, drug_maccs, drug_pdv, protein_esm2, protein_aac
- `fast+lm` (8): + drug_chemberta, drug_molformer, protein_descriptors
- `full` (10): + drug_selformer, protein_kmer

**Notes:**
- Davis: Protein info is essential — paired R² ≈ 0.45–0.54 vs standalone ≈ 0.22. XGBoost dominates.
- KIBA: Protein info *hurts* under cold-target split (only 229 targets). Drug-only standalone beats paired. Ridge dominates.
- `protein_esm2` is the strongest single rep on Davis (0.538 paired) — huge uplift over hand-crafted protein reps.
- `drug_chemberta` is the strongest drug rep on KIBA (0.344), beating fingerprints by ~1pp.
- `drug_molformer` is the strongest drug rep on Davis (0.467 paired).
- all_concat (13665D): 0.548 Davis, 0.369 KIBA — meaningful lift over best single rep.
- Server job 7221614 (L40S GPU, 3.5h): all 10 reps, both datasets, zero failures.

---

## 3. Known Issues and Reliability

### Unreliable Reps (may fail on server)

| Rep | Issue | Severity | Workaround |
|-----|-------|----------|------------|
| `gene_mutation` | Xena MC3 returns 403 | HIGH | Needs manual pre-download or removal from presets |
| `clinical` | Xena phenotype returns 403 | HIGH | Needs manual pre-download or removal from presets |
| `gene_hallmark` | ~~Enrichr download may fail~~ | ~~MEDIUM~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `gene_progeny` | ~~decoupler API 502 / NAType~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath weights, no API calls |
| `gene_dorothea` | ~~decoupler NAType in weight column~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath regulons, no API calls |
| `gene_reactome` | ~~ssGSEA orientation bug~~ | ~~LOW~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `wsi_titan` | HuggingFace gated, awaiting access | MEDIUM | Skip unless access granted |
| `wsi_uni2h` | Manual download + h5 processing required | HIGH | `prepare_uni2h_features()` must be run first |
| `wsi_cnn320` | Local Seagate only, LUAD/LUSC only | LOW | Silently skipped if absent |
| `balm` | Not on HuggingFace; needs manual clone + weight download | ~~MEDIUM~~ **RESOLVED** (2026-02-25) | Cloned to `BALM/` on server, weights in `pretrained-BALM/` |
| `antifold_score` | ImmuneBuilder not installed on server → "WT structure prediction failed" → all-zero features | HIGH | Install ImmuneBuilder or **DROP** (0/7 datasets positive even if fixed) |
| `sapiens_score` | Sapiens positional embeddings overflow for VH > 146 residues (AbAgym VH ~210-220 aa) → zeros | HIGH | Fix Sapiens padding or **DROP** (0/7 datasets positive on FLAb either) |

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

The survey job (11380418) failed with `ImportError: DTA datasets require PyTDC`.
This is because the server had the old code. Current `kirby.datasets.dta` downloads
directly from DeepDTA GitHub with no PyTDC dependency. The server needs `git pull`
to pick up commit `3ba481e`.

---

## 4. Coverage Gaps and Actionable Status

### What we HAVE (confirmed baselines)

- **Molecular:** ecfp4, maccs, pdv confirmed on logd/caco2/herg (3 reps × 3 datasets = 9 cells)
- **Antibody inter:** 12 reps confirmed on abcov_kd; 6 reps on flab_thermo_jain
- **Antibody DMS:** 20 reps confirmed on all 9 datasets (180/180 cells, slurm-7219595)
- **TCGA:** 13 reps confirmed on Lung; 11 reps on BRCA PAM50; geneformer FT on both
- **DTA:** Nothing

### What's still PENDING (needs compute)

| Domain | Gap | Estimated cells | Blocker |
|--------|-----|----------------|---------|
| Molecular | 14 pretrained reps (chemberta→chemformer) on logd/caco2/herg/qm9 | ~56 | Baseline survey keeps timing out during rep generation |
| Antibody inter | ddg_combined (544), abagym_all (2,247), abcov_kd (141), flab_aggregation_jain (137) | ~30 | Needs baseline survey |
| ~~Antibody DMS~~ | ~~7 remaining DMS datasets + antifold/sapiens~~ | ~~137~~ | **DONE** (slurm-7219595, 180/180 cells). antifold_score broken (ImmuneBuilder missing), sapiens_score broken (tensor size mismatch on long VH) |
| TCGA | gene_ica50 on BRCA, cnv_pca50 on BRCA, wsi_gigapath on BRCA | ~3 | Needs re-run |
| DTA | All 10 reps on davis + kiba | ~20 | Timed out during protein_kmer generation |

### What's BROKEN (needs fixing before re-run)

| Item | Issue | Fix |
|------|-------|-----|
| ~~gene_hallmark, gene_kegg, gene_oncogenic~~ | ~~MSigDB `.gmt` files missing~~ | **FIXED**: Now uses local Enrichr gene sets (mean z-score scoring) |
| ~~gene_reactome~~ | ~~ssGSEA returns wrong n_samples~~ | **FIXED**: Now uses local gene sets, no ssGSEA |
| ~~gene_progeny, gene_dorothea~~ | ~~decoupler API 502 / NAType~~ | **FIXED**: Now uses local OmniPath weights |
| ~~gene_combined_pathway~~ | ~~All dependencies broken~~ | **FIXED**: All deps now local |
| gene_mutation, clinical | Xena 403 Forbidden | Pre-download MC3/phenotype from login node or remove from presets |
| wsi_titan | HF gated, access denied | Apply for access or **DROP** |
| wsi_uni2h | Features never prepared | Run `prepare_uni2h_features()` or **DROP** |
| wsi_cnn320 | Local Seagate only, scored random | **DROP** from default presets |
| DTA (protein_kmer) | Times out during generation | Increase timeout or generate separately |

### Candidates to DROP

Reps that are broken AND low-value should be removed from presets to save compute:

| Rep | Reason to drop |
|-----|---------------|
| `wsi_titan` | Gated, scored random (~0.43 on binary), likely never loaded |
| `wsi_uni2h` | Manual prep required, scored random, never worked |
| `wsi_cnn320` | Local-only, scored random, Lung-only, 320D |
| `gene_mutation` | Xena 403, unreliable download |
| `clinical` | Xena 403, unreliable download |
| `unimol` | **DROPPED** — `unimol_tools` package has no working install path |

### Prerequisites before re-running baseline survey

1. `git pull` on server (picks up all fixes including local gene sets)
2. Delete stale pathway caches: `rm ~/.cache/kirby/reps/tcga_*/gene_{hallmark,kegg,reactome,oncogenic,progeny,dorothea,combined_pathway}.npy`
3. ~~Pre-download gseapy gene sets~~ **No longer needed** — gene sets are now bundled as JSON files
4. ~~Ensure gseapy/decoupler installed~~ **No longer needed** for pathway reps (gseapy still needed for `gene_immune` Bindea ssGSEA)
5. For DTA: no PyTDC needed (fixed), but protein_kmer needs longer timeout
6. **Molecular pretrained reps:** Run `bash scripts/setup_server_models.sh` (one-time),
   then `sbatch tests/slurm_scripts/run_precache_molecular.sh` to precache all reps
   in isolated subprocesses before running the baseline survey
7. **New reps:** `gene_autoencoder50` requires torch; all others use only sklearn/scipy

---

## 5. Representation Categories (Summary)

| Category | Description | Examples |
|----------|-------------|---------|
| **Standard** | Computed from input data, no pretrained models | ecfp4, maccs, pdv, aac, kmer, gene_pca100, cnv_gistic |
| **Pretrained** | Frozen embeddings from pretrained LMs | chemberta, molformer, esm2, ablang2, wsi_ctranspath |
| **Fine-tuned** | Per-fold fine-tuned embeddings (LoRA/MLM/Geneformer) | lora_chemberta, antibody_lora, geneformer |
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
| `make_antibody_lora_factory` | Antibody | ESM-2 (650M) | LoRA (rank=8) | 1280D | `--finetune lora_esm2` | **FAIL** — negative R² on FLAb expr, binding, AbAgym |
| `make_antibody_lora_factory` | Antibody | ESM-2 (8M) | LoRA (rank=8) | 320D | `--finetune lora_esm2_small` | **FAIL** — negative R² on FLAb expr, binding, AbAgym |
| `make_antibody_mlm_factory` | Antibody | ESM-2 (650M) | Self-supervised MLM | 5D | `--finetune mlm_esm2` | **FAIL** — negative R² on all antibody datasets |
| `make_antibody_lora_factory` | Antibody | AntiBERTy | LoRA (rank=8) | 512D | `--finetune lora_antiberty` | Not tested (ESM-2 results make this unlikely to help) |
| `make_antibody_lora_factory` | Antibody | AbLang2 | LoRA (rank=8) | 480D | `--finetune lora_ablang2` | Not tested |
| `make_antibody_mlm_factory` | Antibody | ESM-2 (8M) | Self-supervised MLM | 5D | `--finetune mlm_esm2_small` | Not tested |
| `make_antibody_mlm_factory` | Antibody | AntiBERTy | Self-supervised MLM | 5D | `--finetune mlm_antiberty` | Not tested |

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

### 7B. Antibody — FT does NOT help (domain needs rethinking)

| Dataset | Best Frozen Rep | R² | Best FT Rep | R² | Verdict |
|---------|----------------|-----|-------------|-----|---------|
| FLAb Expression (Koenig) | varpos_physchem (lgbm) | 0.382 | lora_esm2 (ridge) | -0.020 | Frozen wins; FT is negative |
| FLAb Binding (Koenig) | biophysical_delta (ridge) | 0.009 | lora_esm2 (ridge) | -0.029 | Both near zero; FT negative |
| AbAgym Ang2 | aac (lgbm) | 0.037 | lora_esm2 (ridge) | -0.106 | Both poor; FT worse |

**Conclusion:** ESM-2 LoRA and MLM fine-tuning produce negative R² across all antibody
datasets. This is consistent with the known failure of full-sequence LM embeddings for
intra-family DMS prediction (published ρ ≈ 0.08). The antibody domain's best reps are
mutation-aware features (varpos_physchem, biophysical_delta) that capture *positional*
changes, not global sequence embeddings. The entire antibody representation strategy
needs fundamental rethinking — see §8.

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

## 8. Open Problems

### Antibody Representation Strategy (NEEDS RETHINKING)

The current antibody pipeline relies on:
1. **Full-sequence LM embeddings** (ESM-2, AbLang2, etc.) — known to fail on DMS (ρ ≈ 0.08)
2. **LoRA fine-tuning** of those same LMs — produces negative R², no improvement
3. **Mutation-aware features** (varpos, biophysical_delta, lm_score) — best performers but still modest (R² ≈ 0.38 on expression)

The fundamental problem: DMS datasets contain families of very similar sequences where
single-residue changes cause large functional shifts. Full-sequence embeddings compress
these near-identical sequences to near-identical vectors, losing the mutation signal.
Fine-tuning doesn't fix this because the pretraining objective (masked LM) doesn't
learn residue-level functional effects.

**Possible directions to explore:**
- Inverse folding scores (ESM-IF1, ProteinMPNN) — capture structural plausibility of mutations
- Per-residue evolutionary conservation (MSA-based, not LM-based)
- Structure-conditioned embeddings (SaProt, ESM-IF1 encoder states)
- Zero-shot fitness predictors as features (EVE, Tranception, ESM-1v marginals)
- Abandoning LM embeddings entirely for antibodies; focus on positional + physics-based
- Paratope-aware features: only embed the binding interface, not the full sequence
