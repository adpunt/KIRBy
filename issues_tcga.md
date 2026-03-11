# KIRBy TCGA Pipeline Audit: Missing Features and Representations

**Date:** 2026-03-10
**Scope:** Comprehensive gap analysis of KIRBy's 34-rep TCGA pipeline vs. published multi-omics cancer benchmarks.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Published Benchmark Repos and What They Use](#2-published-benchmark-repos)
3. [Critical Gap: Somatic Mutation Features (ZERO in KIRBy)](#3-somatic-mutation-features)
4. [Critical Gap: Immune Deconvolution Beyond Bindea/ESTIMATE](#4-immune-deconvolution)
5. [Critical Gap: Mutational Signatures (COSMIC SBS)](#5-mutational-signatures)
6. [High-Priority: Cancer Stemness Index (mRNAsi)](#6-cancer-stemness)
7. [High-Priority: Tumor Heterogeneity and Purity Scores](#7-tumor-heterogeneity)
8. [High-Priority: Missing WSI Foundation Models](#8-wsi-models)
9. [Medium-Priority: Splicing / lncRNA / Gene Fusions](#9-splicing-lncrna-fusions)
10. [Medium-Priority: HRD Score and DNA Repair Deficiency](#10-hrd-score)
11. [Medium-Priority: Clinical Composite Features](#11-clinical-features)
12. [Medium-Priority: MOJO Joint Embedding](#12-mojo)
13. [Lower-Priority: CPTAC Proteomics / Additional Protein Features](#13-cptac-proteomics)
14. [Lower-Priority: Pan-Cancer Foundation Model Embeddings](#14-pan-cancer-fm)
15. [Existing Code Never Wired: create_mutation_matrix](#15-dead-code)
16. [Reference Accuracy Numbers](#16-reference-accuracy)
17. [Full Gap Summary Table](#17-gap-summary)

---

## 1. Executive Summary

KIRBy's TCGA pipeline has 34 representations across expression (18), foundation models (5), multi-omic (6), and WSI (5). While gene expression coverage is strong, there are **glaring omissions** that every serious multi-omics cancer benchmark includes:

**CRITICAL (every competitive benchmark uses these):**
- Somatic mutation features (binary matrix, TMB, pathway-projected mutations) -- **ZERO** in KIRBy despite `create_mutation_matrix()` existing in code but never wired
- COSMIC mutational signatures (SBS/DBS/ID decomposition)
- Proper immune deconvolution (CIBERSORTx, xCell, quanTIseq) -- KIRBy only has a simplified Bindea ssGSEA + ESTIMATE approximation

**HIGH PRIORITY:**
- Cancer stemness indices (mRNAsi, mDNAsi)
- Tumor heterogeneity (MATH score, clonality)
- Newer WSI foundation models (Virchow/Virchow2, UNI/UNI2, CONCH, TITAN)
- TMB and MSI status features

**MEDIUM PRIORITY:**
- Alternative splicing PSI values
- lncRNA expression features
- Gene fusion binary features
- HRD score (LOH + TAI + LST)
- CIMP status
- MOJO joint multi-omic embeddings
- Clinical composite features (TNM, grade, stage as numeric features)

---

## 2. Published Benchmark Repos

### 2.1 MLOmics / Cancer-Multi-Omics-Benchmark
- **URL:** https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark
- **Data types:** mRNA, miRNA, DNA methylation, CNV -- 4 omics across 32 cancer types, 8,314 patients
- **Feature engineering:** Three feature scales (Original, Top via ANOVA, Aligned), stratified features
- **Models:** Extensive baselines across all omics combinations
- **What KIRBy is missing:** Somatic mutation data completely absent from KIRBy. MLOmics has pre-processed, aligned multi-omic matrices ready for ML. KIRBy has no standardized mutation features.

### 2.2 Shamir-Lab Multi-Omics-Cancer-Benchmark
- **URL:** https://github.com/Shamir-Lab/Multi-Omics-Cancer-Benchmark
- **Data types:** mRNA, miRNA, DNA methylation, CNV -- all 11 combinations of 4 omics tested
- **Methods:** iClusterBayes, LRAcluster, SNF, PFA, moCluster, NEMO, CIMLR, MultiNMF, PINS, Subtype-GAN
- **Key finding:** mRNA + miRNA + CNV and mRNA + methylation are the most effective combinations. More omics does NOT always help.
- **What KIRBy is missing:** KIRBy never evaluates which omics combinations actually help vs. hurt. No systematic ablation across modalities.

### 2.3 GaoLabXDU/MultiOmicsIntegrationStudy
- **URL:** https://github.com/GaoLabXDU/MultiOmicsIntegrationStudy
- **Data types:** mRNA, miRNA, DNA methylation, CNV across 9 cancers (including LUAD, LUSC, BRCA)
- **Methods:** 10 integration methods benchmarked on all 11 omics combinations
- **Key finding:** Integrating more omics data can negatively impact performance. Effective combos: mRNA + miRNA + CNV, mRNA + methylation.
- **What KIRBy is missing:** Same as above. KIRBy treats all reps independently; never does modality-level ablation.

### 2.4 MOGONET
- **URL:** https://github.com/txWang/MOGONET
- **Data types:** mRNA, miRNA, DNA methylation (3 omics)
- **Models:** Multi-omics Graph cOnvolutional NETworks + VCDN integration
- **Accuracy:** BRCA PAM50 = 82.9% (using GCN multi-omics integration)
- **What KIRBy is missing:** Graph-based patient similarity networks. KIRBy uses flat feature vectors only.

### 2.5 MOGDx
- **URL:** https://github.com/Barry8197/MOGDx (also https://github.com/biomedicalinformaticsgroup/MOGDx)
- **Data types:** Multi-omics with graph-based integration
- **Benchmarked on:** BRCA TCGA dataset
- **What KIRBy is missing:** Graph-based multi-omics integration approach.

### 2.6 OmiVAE
- **URL:** https://github.com/zhangxiaoyu11/OmiVAE
- **Data types:** Multi-omics (gene expression + others)
- **Models:** Variational autoencoder, end-to-end
- **Accuracy:** Pan-cancer 33-type classification precision 97.49%
- **What KIRBy is missing:** VAE-based latent features. KIRBy has `gene_autoencoder50` (a regular AE) but not a VAE trained on pan-cancer data.

### 2.7 Tybalt (greenelab)
- **URL:** https://github.com/greenelab/tybalt
- **Data types:** Gene expression (top 5000 MAD genes)
- **Models:** VAE with 100D latent space
- **What KIRBy is missing:** VAE-based pan-cancer expression embedding. Tybalt captures nonlinear biological patterns that PCA misses.

### 2.8 greenelab/pancancer
- **URL:** https://github.com/greenelab/pancancer
- **Data types:** Gene expression, mutation, copy number
- **Feature engineering:** Binary mutation matrix + copy number gains/losses, pathway-level mutation aggregation
- **Key method:** Classifiers detect pathway activation from mutation + expression jointly
- **What KIRBy is missing:** Mutation-based pathway activation scores. KIRBy has pathway scores from expression only, never from mutations.

### 2.9 greenelab/mpmp (Multimodal Pan-cancer Mutation Prediction)
- **URL:** https://github.com/greenelab/mpmp
- **Data types:** Gene expression, mutation, copy number
- **What KIRBy is missing:** Multimodal mutation prediction features.

### 2.10 cognoma/cancer-data
- **URL:** https://github.com/cognoma/cancer-data
- **Data types:** Gene expression (20,530 genes), mutation (21,940 genes binary matrix)
- **Format:** Sample x gene matrices (expression-matrix.tsv, mutation-matrix.tsv)
- **What KIRBy is missing:** A properly processed binary mutation matrix. KIRBy has `create_mutation_matrix()` but it is NEVER CALLED in the survey.

### 2.11 DeepProg
- **URL:** https://github.com/lanagarmire/DeepProg
- **Data types:** RNA-Seq, miRNA, DNA methylation
- **Models:** Autoencoders + CoxPH regression
- **What KIRBy is missing:** Multi-omic autoencoder latent features for survival prediction.

### 2.12 AstraZeneca/Multimodal_NSCLC
- **URL:** https://github.com/AstraZeneca/Multimodal_NSCLC
- **Data types:** miRNA, mRNA, DNA methylation, lncRNA, clinical data (5 modalities)
- **Models:** Denoising autoencoder + Elastic Net Cox
- **Key insight:** lncRNA is a separate modality! 732 NSCLC patients from TCGA.
- **What KIRBy is missing:** lncRNA expression as a separate feature set. KIRBy only uses protein-coding mRNA.

### 2.13 IntegrativeVAEs (CancerAI-CL)
- **URL:** https://github.com/CancerAI-CL/IntegrativeVAEs
- **Data types:** mRNA, CNV, DNA methylation, miRNA
- **Models:** Several VAE architectures (JMVAE, MVAE, MMVAE, MoPoE-VAE, etc.)
- **What KIRBy is missing:** Multi-modal VAE joint latent features.

### 2.14 moBRCA-net
- **URL:** https://github.com/cbi-bioinfo/moBRCA-net
- **Data types:** Gene expression, DNA methylation, miRNA
- **Models:** Multi-omics attention neural networks
- **Accuracy:** BRCA PAM50 F1=0.887 with omics-level attention
- **What KIRBy is missing:** Attention-weighted multi-omics integration.

### 2.15 PathomicFusion (mahmoodlab)
- **URL:** https://github.com/mahmoodlab/PathomicFusion
- **Data types:** Histology images + genomics
- **Models:** Attention gating + tensor fusion
- **What KIRBy is missing:** Formal WSI-genomic multimodal fusion. KIRBy treats WSI and genomic as independent reps.

### 2.16 PathOmics
- **URL:** https://github.com/Cassie07/PathOmics
- **Data types:** Histopathology images + genomics
- **Models:** Multimodal transformer for survival prediction (MICCAI 2023 Oral)
- **What KIRBy is missing:** Joint pathology-genomics embedding.

### 2.17 MOVICS
- **URL:** https://github.com/xlucpu/MOVICS
- **Data types:** Up to 6 omics datasets
- **Models:** 10 clustering algorithms unified interface
- **What KIRBy is missing:** Consensus clustering-derived subtype features.

### 2.18 instadeepai/multiomics-open-research (BulkRNABert + MOJO)
- **URL:** https://github.com/instadeepai/multiomics-open-research
- **Data types:** Bulk RNA-seq + DNA methylation (joint)
- **Models:** MOJO learns joint representations; BulkRNABert for single-modality
- **What KIRBy is missing:** KIRBy already has BulkRNABert embeddings but NOT MOJO joint embeddings that combine expression + methylation.

### 2.19 SigProfiler Suite (AlexandrovLab)
- **URL:** https://github.com/AlexandrovLab/SigProfilerAssignment
- **URL:** https://github.com/AlexandrovLab/SigProfilerExtractor
- **Data types:** Somatic mutations -> COSMIC v3.4 mutational signatures (SBS, DBS, ID)
- **What KIRBy is missing:** Mutational signature decomposition. This is a standard feature in cancer genomics that KIRBy completely lacks.

### 2.20 leofabrizio/somatic_mutation_ml
- **URL:** https://github.com/leofabrizio/somatic_mutation_ml
- **Data types:** Somatic mutation profiles for cancer type classification
- **What KIRBy is missing:** Mutation-based cancer type classification features.

### 2.21 hwanglab/tcga_tmb_prediction
- **URL:** https://github.com/hwanglab/tcga_tmb_prediction
- **Data types:** Whole slide images -> TMB prediction
- **Models:** Deep learning TMB prediction from histology
- **What KIRBy is missing:** TMB as a feature, and WSI-based TMB prediction.

### 2.22 HerrMo/multi-omics_benchmark_study
- **URL:** https://github.com/HerrMo/multi-omics_benchmark_study
- **Data types:** Multi-omics survival prediction
- **Models:** Large-scale benchmark of survival prediction methods
- **What KIRBy is missing:** Proper survival analysis benchmarking methodology.

### 2.23 VAECox
- **URL:** https://github.com/dmis-lab/VAECox
- **Data types:** Multi-omics pan-cancer
- **Models:** VAE + Cox regression for survival
- **What KIRBy is missing:** VAE-based survival-aware latent features.

### 2.24 BiomedSciAI/biomed-multi-omic (BMFM-RNA)
- **URL:** https://github.com/BiomedSciAI/biomed-multi-omic
- **Data types:** RNA/DNA
- **Models:** Transcriptomics Foundation Model (TFM) with Whole Cell Expression Decoder (WCED)
- **What KIRBy is missing:** BMFM-RNA embeddings. A newer foundation model for bulk RNA that may outperform BulkRNABert/Geneformer.

### 2.25 BoevaLab/CancerFoundation
- **URL:** https://github.com/BoevaLab/CancerFoundation
- **Data types:** scRNA-seq (cancer-specific)
- **Models:** Cancer-specific foundation model
- **What KIRBy is missing:** Cancer-specific foundation model embeddings.

---

## 3. Critical Gap: Somatic Mutation Features (ZERO in KIRBy)

**Severity: CRITICAL**

KIRBy has `create_mutation_matrix()` implemented in `genomic.py` (line 534) that creates a binary matrix of top-200 mutated genes. However, this function is **NEVER registered** in `_REP_DISPATCH` in `find_best_hybrid_tcga.py` and is therefore **never called** during any survey run. Every serious TCGA multi-omics benchmark includes mutation data.

### What should be added:

| Feature | Dim | Description | Difficulty |
|---------|-----|-------------|------------|
| `mut_binary200` | 200 | Binary top-200 mutated gene matrix (ALREADY CODED, just wire it) | Trivial |
| `mut_pca50` | 50 | PCA of binary mutation matrix | Easy |
| `mut_tmb` | 1-5 | Tumor mutational burden (total non-silent mutations, log-scaled, per Mb) | Easy |
| `mut_pathway` | ~50 | Pathway-level mutation aggregation (how many genes in each pathway are mutated) | Medium |
| `mut_oncogene_ts` | ~40 | Oncogene/tumor suppressor mutation status for key driver genes (TP53, KRAS, EGFR, PIK3CA, etc.) | Easy |
| `mut_network_propagated` | 50-100 | Network-propagated mutation features (de-sparsification via PPI network) | Hard |

### Evidence:
- cognoma/cancer-data provides pre-built mutation-matrix.tsv (21,940 genes x 7,306 samples)
- greenelab/pancancer uses mutation + CNV + expression jointly
- leofabrizio/somatic_mutation_ml classifies cancer types purely from somatic mutations
- MOGONET, MLOmics, and every major benchmark include mutation data
- KIRBy's MC3 MAF data is already downloaded and cached -- the infrastructure exists

---

## 4. Critical Gap: Immune Deconvolution Beyond Bindea/ESTIMATE

**Severity: CRITICAL**

KIRBy's `create_immune_scores()` (genomic.py line 723) uses:
- Bindea et al. 2013 ssGSEA (24 cell types) -- a simplified marker-gene approach
- ESTIMATE algorithm (stromal + immune + purity) -- an approximation

This is a **toy implementation** compared to what the field uses. Published TCGA analyses routinely use:

| Method | Cell Types | Approach | Status in KIRBy |
|--------|-----------|----------|-----------------|
| CIBERSORTx | 22 (LM22 matrix) | SVR deconvolution | MISSING |
| xCell | 64 cell types | Enrichment-based | MISSING |
| MCP-counter | 10 cell types | Marker gene quantification | MISSING |
| TIMER | 6 immune cell types | Cancer-type specific deconvolution | MISSING |
| quanTIseq | 10 cell types | Constrained least squares | MISSING |
| EPIC | 8 cell types | Weighted least squares | MISSING |
| Bindea ssGSEA | 24 cell types | ssGSEA of marker genes | Present (simplified) |
| ESTIMATE | 3 scores | Mean expression of gene lists | Present |

### What should be added:
- `immune_cibersort` (~22D): CIBERSORTx LM22 fractions -- the gold standard
- `immune_xcell` (~64D): xCell scores -- broadest cell type coverage
- `immune_mcpcounter` (~10D): MCP-counter absolute abundances
- `immune_quant` (~10D): quanTIseq fractions
- `immune_consensus` (~10D): Consensus of multiple methods (median of normalized scores)

### Tools:
- Python: `immunedeconv` R package (wrappable via rpy2), or CIBERSORTx web API
- Pre-computed: TIMER2.0 database has pre-computed scores for all TCGA samples
- xCell pre-computed scores available for 9,947 TCGA samples

---

## 5. Critical Gap: Mutational Signatures (COSMIC SBS)

**Severity: CRITICAL**

COSMIC mutational signatures are a standard feature in cancer genomics. Every tumor has a unique combination of ~70+ single-base substitution (SBS) signatures that encode the mutational processes active in that tumor (aging, smoking, APOBEC, UV, defective DNA repair, etc.).

### What should be added:
- `sig_sbs` (~70D): COSMIC v3.4 SBS signature activities per sample
- `sig_dbs` (~10D): Doublet-base substitution signatures
- `sig_id` (~17D): Small insertion/deletion signatures
- `sig_combined_pca` (~20D): PCA of all signature types

### Tools:
- **SigProfilerAssignment** (https://github.com/AlexandrovLab/SigProfilerAssignment): Python, assigns COSMIC v3.4 signatures to individual samples from VCF/MAF
- **SigProfilerExtractor** (https://github.com/AlexandrovLab/SigProfilerExtractor): De novo extraction
- KIRBy already downloads MC3 MAF data -- SigProfiler can consume this directly

### Importance:
- Mutational signatures are among the most biologically informative features
- They capture the **cause** of mutations (smoking vs. APOBEC vs. aging), not just presence/absence
- Used clinically for treatment selection (e.g., SBS3 = HRD -> PARP inhibitor sensitivity)

---

## 6. High-Priority: Cancer Stemness Index (mRNAsi)

**Severity: HIGH**

The mRNA expression-based stemness index (mRNAsi) from Malta et al. (Cell, 2018) is a widely used feature in TCGA analysis. It quantifies how "stem-cell-like" a tumor sample is using a one-class logistic regression (OCLR) model trained on stem cell vs. differentiated cell expression profiles.

### What should be added:
- `stemness_mrnasi` (1D): mRNAsi score
- `stemness_mdnasi` (1D): DNA methylation-based stemness index
- `stemness_ereg` (1D): Epigenetic regulation-based mRNAsi

### Source:
- Pre-computed scores available from GDC PanCanStemness-2018 publication
- URL: https://gdc.cancer.gov/about-data/publications/PanCanStemness-2018
- Can also be recomputed using the OCLR model weights

### Importance:
- Correlates with tumor dedifferentiation, immune checkpoint expression, prognosis
- Used in >200 TCGA papers since 2018
- 1-3 features, trivially easy to add

---

## 7. High-Priority: Tumor Heterogeneity and Purity Scores

**Severity: HIGH**

### What should be added:
| Feature | Dim | Description |
|---------|-----|-------------|
| `purity_estimate` | 3 | ESTIMATE stromal + immune + purity (ALREADY partially implemented) |
| `purity_absolute` | 2 | ABSOLUTE tumor purity + ploidy (pre-computed for TCGA) |
| `purity_consensus` | 5 | Consensus purity from 4 methods (CPE, ABSOLUTE, ESTIMATE, LUMP, IHC) |
| `heterogeneity_math` | 1 | Mutant-Allele Tumor Heterogeneity score from MAF |
| `heterogeneity_depth` | 1 | mRNA-based intratumor heterogeneity |

### Source:
- TCGA PanCanAtlas provides pre-computed purity/ploidy/MATH for all samples
- TCGAbiolinks R package has `Tumor.purity` table with 5 estimation methods
- MATH score computable from MAF file (already downloaded in KIRBy)

---

## 8. High-Priority: Missing WSI Foundation Models

**Severity: HIGH**

KIRBy has 5 WSI models from GigaSSL (CTransPath, Phikon, H-optimus-0, GigaPath, MoCo). The field has moved far beyond these. State-of-the-art models released in 2024-2025:

| Model | Params | Training Data | Dim | Status in KIRBy | Source |
|-------|--------|--------------|-----|-----------------|--------|
| CTransPath | 28M | 15K WSIs | 768 | Present | GigaSSL |
| Phikon | 300M | 6K WSIs | 768 | Present | GigaSSL |
| H-optimus-0 | 300M | 500K WSIs | 768 | Present | GigaSSL |
| GigaPath | 1.1B | 170K WSIs | 768 | Present | GigaSSL |
| MoCo | - | - | 768 | Present | GigaSSL |
| **Virchow** | 632M | 1.5M WSIs | 1280 | **MISSING** | https://huggingface.co/paige-ai/Virchow |
| **Virchow2** | 632M | 3.1M WSIs | 1280 | **MISSING** | https://huggingface.co/paige-ai/Virchow2 |
| **UNI** | 300M | - | 1024 | **MISSING** | https://github.com/mahmoodlab/UNI |
| **UNI2** | 1.1B | 200M+ images | 1536 | **MISSING** | https://github.com/mahmoodlab/UNI |
| **CONCH** | - | 1.17M image-caption pairs | 512 | **MISSING** | https://github.com/mahmoodlab/CONCH |
| **TITAN** | - | 335K WSIs | 768 | **MISSING** | https://github.com/mahmoodlab/TITAN |

### Notes:
- Virchow2 was trained on 3.1M WSIs (2x Virchow, 18x GigaPath)
- UNI2 trained on 200M+ images, state-of-the-art on Patho-Bench (95 tasks)
- CONCH is vision-language (can do zero-shot classification)
- TITAN is slide-level (no need for tile aggregation) and was NOT trained on TCGA (no data contamination)
- TRIDENT toolkit (https://github.com/mahmoodlab/TRIDENT) can extract embeddings from 20+ foundation models

### Difficulty: MEDIUM-HARD
- Requires downloading model weights (some gated on HuggingFace)
- Requires tile-level extraction + aggregation for patch-level models (Virchow, UNI)
- TITAN provides slide-level embeddings directly
- KIRBy already has WSI infrastructure; extending to new models is feasible

---

## 9. Medium-Priority: Splicing / lncRNA / Gene Fusions

### 9.1 Alternative Splicing PSI Values

**Severity: MEDIUM**

TCGA SpliceSeq provides Percent Spliced In (PSI) values for all splice events across 33 tumor types. Alternative splicing is a recognized cancer hallmark.

- `splicing_psi_pca50` (~50D): PCA of top variable PSI values
- `splicing_psi_hvg` (~200D): Most variable splicing events
- Source: https://bioinformatics.mdanderson.org/public-software/tcgaspliceseq/
- Also: AS-CMC pan-cancer database of alternative splicing for molecular classification

### 9.2 lncRNA Expression

**Severity: MEDIUM**

KIRBy only uses protein-coding mRNA. Long non-coding RNAs are highly cancer-type-specific and informative:
- `lncrna_pca50` (~50D): PCA of lncRNA expression
- `lncrna_hvg200` (~200D): Top variable lncRNAs
- AstraZeneca/Multimodal_NSCLC uses lncRNA as a separate modality
- TCGA pan-cancer lncRNA atlas: 5,037 samples, 13 cancer types

### 9.3 Gene Fusion Features

**Severity: MEDIUM-LOW**

- `fusion_binary` (~50-100D): Binary matrix of recurrent gene fusions
- Source: TumorFusions portal (9,966 TCGA samples, 33 cancer types via PRADA pipeline)
- Fusions are rare events (sparse), but some are diagnostic (EML4-ALK in LUAD, BCR-ABL in CML)

---

## 10. Medium-Priority: HRD Score and DNA Repair Deficiency

**Severity: MEDIUM**

Homologous Recombination Deficiency (HRD) is a clinically actionable feature:
- `hrd_score` (3-4D): LOH + TAI + LST + composite HRD score
- Pre-computed for 8,847 TCGA tumors (PanCanAtlas)
- Clinically relevant: predicts PARP inhibitor sensitivity
- Source: Knijnenburg et al. (Cell Reports, 2018)

---

## 11. Medium-Priority: Clinical Composite Features

**Severity: MEDIUM**

KIRBy has `create_clinical()` in genomic.py but it is NOT in `_REP_DISPATCH`. Clinical features (stage, grade, TNM) are important confounders and can be predictive:
- `clinical_features` (~15-20D): Encoded stage, grade, TNM, age, gender
- Note: KIRBy code exists (line 1355) but is dead code, never called in survey

---

## 12. Medium-Priority: MOJO Joint Embedding

**Severity: MEDIUM**

MOJO (from InstaDeep, same group as BulkRNABert) learns joint representations of bulk RNA-seq + DNA methylation. KIRBy already has BulkRNABert but not MOJO:
- `mojo_joint` (~256D or similar): Joint RNA + methylation embedding
- Source: https://github.com/instadeepai/multiomics-open-research
- Pre-trained models available on HuggingFace

---

## 13. Lower-Priority: CPTAC Proteomics / Additional Protein Features

**Severity: LOW-MEDIUM**

KIRBy has RPPA (~260 proteins). CPTAC provides mass-spec-based proteomics with much broader coverage:
- `cptac_proteomics_pca50` (~50D): PCA of CPTAC proteomics
- Caveat: CPTAC only covers subset of TCGA samples (BRCA, OV, CRC initially)
- Not available for LUAD/LUSC via CPTAC in sufficient overlap with TCGA expression samples

---

## 14. Lower-Priority: Pan-Cancer Foundation Model Embeddings

**Severity: LOW-MEDIUM**

Newer foundation models specifically for bulk RNA-seq:
- `bmfm_rna` (variable): BMFM-RNA from IBM BiomedSciAI (https://github.com/BiomedSciAI/biomed-multi-omic)
- `cancer_foundation` (variable): CancerFoundation scRNA-seq FM fine-tuned on malignant cells (https://github.com/BoevaLab/CancerFoundation)
- `genebag` (variable): GeneBag -- pre-trained on scRNA, fine-tuned on bulk, ~96.2% cancer vs. normal accuracy

---

## 15. Existing Code Never Wired: create_mutation_matrix and create_clinical

**Severity: TRIVIAL TO FIX (but critical impact)**

Two functions exist in `src/kirby/representations/genomic.py` but are **never registered** in the `_REP_DISPATCH` table in `tests/find_best_hybrid_tcga.py`:

1. **`create_mutation_matrix()`** (line 534): Binary top-200 mutated gene matrix. Fully functional. Just needs a `_REP_DISPATCH` entry.

2. **`create_clinical()`** (line 1355): Clinical feature encoding (stage, grade, TNM, etc.). Fully functional. Just needs a `_REP_DISPATCH` entry.

### Fix:
Add to `_REP_DISPATCH`:
```
'mut_binary200':     ('create_mutation_matrix', 'mutation'),
'clinical':          ('create_clinical',        'mutation'),
```
And add to the `'all'` preset. This is a 2-line fix that adds 2 representations.

---

## 16. Reference Accuracy Numbers

### LUAD vs LUSC Classification
| Method | Data Type | Accuracy/AUC | Source |
|--------|-----------|-------------|--------|
| XGBoost + LASSO | Gene expression | 99.1% (LUAD), 98.6% (LUSC) | Gene expression ML studies |
| LightGBM | RNA-seq | ~97.1% acc, ~97% AUC | TCGA benchmarks |
| CNN | RNA-seq | 97% accuracy | Deep learning on TCGA |
| DNN | DNA methylation | 96.92% acc, AUC 0.998 | Multi-cohort validation |
| Histopathology DL | WSI | AUC 0.97 | Nature Medicine 2018 |

**KIRBy context:** LUAD vs LUSC is considered an "easy" task for gene expression. If KIRBy is below ~95% accuracy with TSS-grouped CV, something may be wrong. Most methods saturate near 97-99%.

### BRCA PAM50 Subtype Classification
| Method | Data Type | Accuracy/F1 | Source |
|--------|-----------|-------------|--------|
| HyperCLSA | Multi-omics | 90.1% accuracy | State-of-art (2025) |
| moBRCA-net | Expression + methylation + miRNA | F1=0.887 | BMC Bioinformatics 2023 |
| MOGONET | Expression + methylation + miRNA | 82.9% accuracy | Nature Communications 2021 |
| SNF graph integration | Multi-omics | NMI=0.495 vs PAM50 | arXiv 2025 |

**KIRBy context:** BRCA PAM50 with 4 classes is harder. 85-90% accuracy is competitive. Below 80% suggests the representation is inadequate.

### BRCA Survival Prediction
| Method | Data Type | C-index | Source |
|--------|-----------|---------|--------|
| SNF graph | Multi-omics | 0.681 | arXiv 2025 |
| DeepProg | RNA + miRNA + methylation | Variable | lanagarmire/DeepProg |
| Elastic Net Cox | 5 modalities | Variable | AstraZeneca NSCLC |

---

## 17. Full Gap Summary Table

| Gap | Severity | # Features | Difficulty | Data Available | Implementation |
|-----|----------|-----------|------------|----------------|----------------|
| Wire `mut_binary200` (dead code) | CRITICAL | 200 | Trivial | Yes (MC3 MAF) | Add 1 line to dispatch |
| Wire `clinical` (dead code) | CRITICAL | ~15 | Trivial | Yes | Add 1 line to dispatch |
| TMB (tumor mutational burden) | CRITICAL | 1-5 | Easy | Yes (MC3 MAF) | Count mutations per sample |
| `mut_pca50` | CRITICAL | 50 | Easy | Yes | PCA of binary mutation matrix |
| `mut_pathway` (pathway-aggregated mutations) | CRITICAL | ~50 | Medium | Yes | Count mutated genes per pathway |
| `mut_driver` (key driver gene status) | CRITICAL | ~40 | Easy | Yes | Binary status of known drivers |
| COSMIC mutational signatures (SBS) | CRITICAL | ~70 | Medium | Yes (MC3 MAF) | SigProfilerAssignment |
| CIBERSORTx immune fractions | CRITICAL | 22 | Medium | Pre-computed available | Download or compute via API |
| xCell scores | CRITICAL | 64 | Medium | Pre-computed available | Download pre-computed |
| Cancer stemness (mRNAsi) | HIGH | 1-3 | Easy | Pre-computed | Download from GDC |
| Tumor purity (ABSOLUTE) | HIGH | 2 | Easy | Pre-computed | Download from PanCanAtlas |
| Heterogeneity (MATH) | HIGH | 1 | Easy | Computable from MAF | Simple calculation |
| Virchow/Virchow2 WSI | HIGH | 1280 | Hard | HuggingFace (gated) | Tile extraction + pooling |
| UNI/UNI2 WSI | HIGH | 1024-1536 | Hard | HuggingFace (gated) | Tile extraction + pooling |
| CONCH WSI | HIGH | 512 | Hard | HuggingFace (gated) | Tile extraction + pooling |
| TITAN WSI (slide-level) | HIGH | 768 | Medium | HuggingFace (gated) | Direct slide embedding |
| MSI status | HIGH | 1-3 | Easy | Pre-computed | Download |
| CIMP status | MEDIUM | 1-3 | Easy | Pre-computed | Download |
| HRD score (LOH+TAI+LST) | MEDIUM | 3-4 | Easy | Pre-computed | Download from PanCanAtlas |
| Splicing PSI values | MEDIUM | 50-200 | Medium | TCGA SpliceSeq | Download + PCA |
| lncRNA expression | MEDIUM | 50-200 | Medium | TCGA Xena | Filter non-coding + PCA |
| MOJO joint embedding | MEDIUM | ~256 | Medium | HuggingFace | Inference with pre-trained |
| Gene fusion binary | MEDIUM-LOW | 50-100 | Medium | TumorFusions | Download + process |
| BMFM-RNA embedding | LOW-MEDIUM | Variable | Medium | GitHub | New FM, untested |
| VAE latent (Tybalt-style) | LOW | 100 | Medium | Code available | Train or use pre-trained |

### Totals:
- **Currently in KIRBy:** 34 representations
- **Trivial fixes (wire dead code):** +2 reps
- **Easy additions (pre-computed downloads):** +10-15 reps
- **Medium difficulty (computation needed):** +8-12 reps
- **Hard (new model inference):** +4-6 reps (WSI models)
- **Potential total:** ~60-70 representations

---

## Key Takeaway

The single most embarrassing gap is that KIRBy has `create_mutation_matrix()` implemented and functional but **never calls it**. Somatic mutations are arguably the most important data type in cancer genomics after gene expression. Every benchmark (MOGONET, MLOmics, Shamir-Lab, greenelab/pancancer, cognoma) includes mutation data. KIRBy's omission of ALL mutation-derived features is a serious weakness that undermines the claim of comprehensive TCGA representation benchmarking.

The second most important gap is proper immune deconvolution. KIRBy's Bindea ssGSEA approach is a simplified approximation. CIBERSORTx with the LM22 signature matrix is the field standard, and pre-computed xCell scores covering 64 cell types are freely available for all TCGA samples.

Third, COSMIC mutational signatures capture biologically meaningful mutational processes and are standard in cancer genomics papers. SigProfilerAssignment is a Python package that can consume the MC3 MAF data KIRBy already downloads.

The WSI gap (missing Virchow, UNI, CONCH, TITAN) is real but harder to fix due to computational requirements for tile extraction. TITAN slide-level embeddings would be the easiest addition.
