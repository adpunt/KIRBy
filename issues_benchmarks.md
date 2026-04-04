# KIRBy Benchmarking Issues & Standards Reference

> Research-only document. Generated 2026-03-10.
> Purpose: define the standard benchmarks, protocols, preprocessing, metrics, and SOTA for every KIRBy domain so the framework can build on top of real research rather than replace it.
> Constraints: NO random splitting, NO TDC, NO MoleculeNet.

---

## Table of Contents

1. [Molecular QSAR](#1-molecular-qsar)
2. [Antibody Fitness / DMS](#2-antibody-fitness--dms)
3. [DDG (Binding Free Energy)](#3-ddg-binding-free-energy)
4. [DTA (Drug-Target Affinity)](#4-dta-drug-target-affinity)
5. [TCGA Cancer Genomics](#5-tcga-cancer-genomics)
6. [Reaction Yields](#6-reaction-yields)

---

## 1. Molecular QSAR

### 1.1 Standard Benchmark Datasets (NOT MoleculeNet / TDC)

| Dataset | Source | Size | Task | Notes |
|---------|--------|------|------|-------|
| **Polaris Hub benchmarks** | [polarishub.io](https://polarishub.io/) | Varies per endpoint | Regression / Classification | Emerging gold standard. Immutable datasets, locked scaffold splits, public leaderboards. ASAP-Discovery + OpenADMET challenge data included. |
| **PharmaBench** | [github.com/mindrank-ai/PharmaBench](https://github.com/mindrank-ai/PharmaBench) | 52,482 entries across 11 ADMET endpoints | Regression / Classification | LLM-curated from 14,401 bioassays. Provides both scaffold and random splits at 80/20 ratio. Standardized SMILES, deduplicated. |
| **Benchmark-ADMET-2025** | [github.com/DonghaiZHAO-ZJU/Benchmark-ADMET-2025](https://github.com/DonghaiZHAO-ZJU/Benchmark-ADMET-2025) | Multiple endpoints, cleaned + deduplicated | Regression / Classification | Scaffold + UMAP + Butina splits. Includes Roughness Index (MODI, SARI, ROGI) for task difficulty estimation. |
| **OpenADMET** | [openadmet.org](https://openadmet.org/) | Blind challenge datasets | Regression | ExpansionRx blind challenge (Oct 2025). Temporal splits preferred by top performers. |
| **ChEMBL-derived** | [ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl/) | 2,219 protein targets, 30-6000 cpds each | Regression | Gold standard source for QSAR data. Requires manual curation. |
| **AqSolDB** | Public | ~10,000 | Regression (logS) | Curated aqueous solubility from 9 open sources. Reliability groups assigned. |
| **hERG-DB** | ChEMBL v30 | 14,364 compounds | Classification / Regression | Cardiotoxicity. IC50-based. Structural diversity. |
| **Papyrus** | Public | Large-scale | Multiple | Combines ChEMBL + ExCAPE-DB. Good for multi-task QSAR. |
| **BigSolDB 2.0** | [nature.com/articles/s41597-025-05559-8](https://www.nature.com/articles/s41597-025-05559-8) | 103,944 measurements | Regression | Multi-solvent solubility. |
| **Inductive Bio benchmarks** | [inductive.bio/blog/building-better-benchmarks-for-adme-optimization](https://www.inductive.bio/blog/building-better-benchmarks-for-adme-optimization) | Endpoint-specific | ADME endpoints | Specifies test data but NOT training data -- training set design is part of the research. |

### 1.2 Standard Evaluation Protocol

**Pat Walters' recommendations (Practical Cheminformatics, Nov 2024):**

- **Temporal split is ideal** when dates are available: train on historical data, test on future molecules. This mirrors real drug discovery.
- **Scaffold split is a proxy** for temporal split when dates are unavailable. Known limitations:
  - Molecules can differ by a single atom yet have different scaffolds (Tanimoto similarity 0.66), making scaffold split "leaky".
  - Scaffold splits **overestimate** virtual screening performance (Guo et al., ICANN 2024).
- **UMAP split is more rigorous**: Morgan fingerprints projected to 2D via UMAP, then agglomerative clustering into 7 clusters. Produces significantly lower R^2 than scaffold split -- more realistic.
- **Butina split**: RDKit Butina clustering on Morgan fingerprints. Intermediate difficulty between scaffold and UMAP.
- **Random split is forbidden** in this lab group.

**Standard split ratio:** 80/10/10 (train/val/test) for single-split evaluation, or 5-fold GroupKFold with scaffold groups.

**Polaris protocol:** Scaffold split, 80/20, immutable indices stored per benchmark. No data drift -- exact same test set for all models.

**QSARtuna (AstraZeneca):** Optuna-based HPO with scaffold-based testing. Automated comparison of molecular representations and ML models. Published in JCIM 2024.

### 1.3 Standard Preprocessing

The **QSAR-Ready Workflow** (Mansouri et al., J. Cheminformatics 2024) defines the standard:

1. **Salt stripping** -- remove counterions, keep largest fragment
2. **Neutralization** -- remove formal charges where possible
3. **Tautomer standardization** -- canonical tautomer via RDKit
4. **Stereochemistry stripping** -- for 2D-only representations
5. **Valence correction** -- fix RDKit valence errors
6. **Duplicate removal** -- by InChIKey; check for tautomeric duplicates
7. **Inorganic/organometallic removal** -- keep only drug-like organic molecules
8. **Activity cliff flagging** -- SALI (Structure-Activity Landscape Index) to identify compounds with high similarity but large activity differences. These should be flagged, not removed.

**ChEMBL curation:**
- Multiple manual + automated steps
- Maximize data accessibility and comparability
- Flag outliers, ambiguities, potential errors
- Add curated annotations

### 1.4 Standard Metrics

| Task | Primary Metric | Secondary Metrics | Notes |
|------|---------------|-------------------|-------|
| Regression | **RMSE** | R^2, MAE, Pearson r | "Beware of R^2" (Tropsha, JCIM 2015): RMSE is more informative. Rule of thumb: test RMSE < 10% of target range. |
| Classification | **AUROC** | Balanced accuracy, MCC, Cohen's kappa | MCC preferred for imbalanced datasets. |

### 1.5 SOTA Under Scaffold Splits

| Endpoint | Model | RMSE (scaffold) | R^2 (scaffold) | Source |
|----------|-------|-----------------|-----------------|--------|
| Solubility (logS) | Chemprop-RDKit | ~0.9-1.1 | ~0.75-0.85 | ADMET-AI |
| Lipophilicity (logD) | MolGPS / CheMeleon | -- | -- | Polaris leaderboard |
| hERG | Random Forest + ECFP | AUROC ~0.85 | -- | ChEMBL v30 benchmarks |
| Caco-2 | Chemprop-RDKit | -- | -- | ADMET-AI |
| General ADMET | CheMeleon | 79% win rate across Polaris tasks | -- | Polaris 2025 |

**Key insight:** Under scaffold splits, ECFP4 + Random Forest or XGBoost often matches or outperforms deep learning methods. Foundation models (MolGPS, CheMeleon) show marginal improvement on Polaris tasks but not universally.

### 1.6 Key Repos

- [PatWalters/practical_cheminformatics_tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials) -- QSAR best practices
- [PatWalters/qsar](https://github.com/PatWalters/qsar) -- Simple ML QSAR
- [MolecularAI/QSARtuna](https://github.com/MolecularAI/QSARtuna) -- AstraZeneca automated QSAR
- [chemprop/chemprop](https://github.com/chemprop/chemprop) -- GNN for molecular property prediction
- [swansonk14/admet_ai](https://github.com/swansonk14/admet_ai) -- ADMET-AI training/prediction
- [DonghaiZHAO-ZJU/Benchmark-ADMET-2025](https://github.com/DonghaiZHAO-ZJU/Benchmark-ADMET-2025) -- Comprehensive ADMET benchmark
- [mindrank-ai/PharmaBench](https://github.com/mindrank-ai/PharmaBench) -- LLM-enhanced ADMET benchmarks

### 1.7 What KIRBy Does vs. What It SHOULD Do

| Aspect | KIRBy Current | Standard Practice | Gap |
|--------|--------------|-------------------|-----|
| **Datasets** | ESOL (via DeepChem/MoleculeNet), hERG, Cox2, DRD2, LogD, Caco2, QM9 | Polaris, PharmaBench, Benchmark-ADMET-2025, ChEMBL-derived, OpenADMET | ESOL is loaded via DeepChem's MoleculeNet loader. Should switch to independent data sources. |
| **Splitting** | Scaffold GroupKFold (5-fold) | Scaffold split (80/10/10 or 5-fold) is acceptable; UMAP split is more rigorous | Scaffold GroupKFold is reasonable but should acknowledge scaffold split limitations. |
| **Preprocessing** | SMILES canonicalization via RDKit | Full QSAR-Ready workflow: salt strip, neutralize, tautomer standardize, deduplicate, filter inorganics | Missing salt stripping, neutralization, tautomer standardization, duplicate removal by InChIKey. |
| **Metrics** | R^2 (primary) | RMSE (primary), R^2 (secondary) | Should report RMSE as primary. "Beware of R^2" is well-established advice. |
| **Activity cliffs** | Not handled | SALI flagging, awareness of cliff pairs | Should flag but this is minor for representation comparison. |
| **Benchmarking against SOTA** | Internal baselines only | Polaris leaderboard, PharmaBench, published Chemprop numbers | Need external validation numbers for context. |

---

## 2. Antibody Fitness / DMS

### 2.1 Standard Benchmark Datasets

| Benchmark | Size | Properties | Splits | Notes |
|-----------|------|------------|--------|-------|
| **ProteinGym** | 217 DMS assays, ~2.7M substitutions (single-site) | Fitness (various phenotypes) | Random, Contiguous, Modulo (5-fold each) | Gold standard. NeurIPS 2023. Per-assay evaluation. |
| **FLIP** | 7 protein landscapes | Binding, stability, activity | Custom splits (domain-specific) | NeurIPS 2021. Tests generalization. |
| **FLIP2** | 7 new datasets, 16 splits across 5 categories | Enzymes, PPIs, light-sensitive proteins | Number, Position, Mutation, Fitness, Wild-type splits | Harder than FLIP. Feb 2026. Simpler models often match fine-tuned pLMs. |
| **FLAb** | 6 Ab properties (expression, Tm, immunogenicity, aggregation, polyreactivity, binding) | Therapeutic antibody fitness | Per-property evaluation | Graylab. Largest therapeutic Ab benchmark. |
| **FLAb2** | 32 studies, >4M antibodies | Expanded to include pharmacokinetics | Per-property evaluation | 2025 expansion. |
| **AbBiBench** | 184,500 measurements, 14 Abs, 9 Ags | Binding affinity (DDG-like) | Ab-Ag complex as fundamental unit | June 2025. Prevents data leakage by using unique mutant complexes. |
| **AbRank** | 380,000+ binding assays | Pairwise affinity ranking | Standardized splits with increasing distribution shift | Reframes prediction as ranking. |

### 2.2 Standard Evaluation Protocol

**ProteinGym supervised benchmark:**
- 5-fold cross-validation per assay
- Three CV schemes of increasing difficulty:
  1. **Random fold**: mutations randomly distributed across 5 folds (easiest)
  2. **Modulo fold**: positions assigned by modulo-5 (every 5th position in same fold)
  3. **Contiguous fold**: protein split into 5 equal-length contiguous segments (hardest)
- First 4 folds for training, 5th for testing. Report mean across all assays.

**FLIP/FLIP2:**
- Spearman rank correlation and NDCG on held-out test sets
- 16 distinct train-test splits simulating real protein engineering scenarios
- Zero-shot, ridge regression, and fine-tuned pLM baselines

**FLAb:**
- Per-property evaluation across datasets
- Zero-shot pLM likelihoods vs. supervised predictors vs. Rosetta energy

### 2.3 Standard Metrics

| Metric | Usage | Notes |
|--------|-------|-------|
| **Spearman rho** | Primary for all DMS/fitness benchmarks | Rank correlation. Robust to non-linear relationships, noise. |
| **MSE** | Secondary in ProteinGym | Measures absolute prediction quality. |
| **NDCG** | Used in FLIP2 | Normalized Discounted Cumulative Gain. Useful for protein engineering where top variants matter most. |
| **R^2** | NOT standard for DMS | User's group uses R^2 for antibody. This is non-standard -- the entire field uses Spearman. |

### 2.4 Preprocessing

- **Per-protein normalization**: DMS scores are typically normalized per assay (z-scoring within each experiment) before cross-assay comparison.
- **Multi-mutant handling**: Combinatorial libraries are parsed by position. Higher-order mutants are typically excluded from standard single-site benchmarks (ProteinGym) or handled separately.
- **Fitness score standardization**: ProteinGym standardizes enrichment ratios, binding scores, etc., into a unified fitness value per assay.
- **Wild-type normalization**: Many assays report fitness relative to wild-type (log enrichment ratio, deltaKd, etc.).

### 2.5 SOTA (Supervised, ProteinGym)

| Model | Spearman (mean, random CV) | Notes |
|-------|---------------------------|-------|
| **ProteinNPT** | Best across all CV schemes | Semi-supervised; joint sequence-label attention |
| **Ridge + ESM-1v embeddings** | Strong baseline | Often competitive with fine-tuned models |
| **Ridge + Tranception embeddings** | Strong baseline | MSA-augmented |
| **Ridge + ESM-2 embeddings** | Competitive | Simple but effective |

Key finding from FLIP2: **simpler models (ridge regression on pLM embeddings) often match or outperform fine-tuned protein language models**, challenging the utility of existing transfer learning techniques.

### 2.6 Key Repos

- [OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym) -- Official benchmark
- [Graylab/FLAb](https://github.com/Graylab/FLAb) -- Antibody fitness benchmark
- [FLIP2](https://flip.protein.properties/) -- Expanded fitness landscapes

### 2.7 What KIRBy Does vs. What It SHOULD Do

| Aspect | KIRBy Current | Standard Practice | Gap |
|--------|--------------|-------------------|-----|
| **Datasets** | FLAb (18 datasets), SAbDab, AbCoV, AbAgym, AB-Bind | FLAb + ProteinGym (217 assays) + FLIP2 + AbBiBench | ProteinGym is the gold standard and is missing. |
| **Splitting** | GroupKFold by unique sequence (5-fold) | ProteinGym: Random/Contiguous/Modulo 5-fold | KIRBy's random GroupKFold is equivalent to ProteinGym's "Random" fold, the easiest scheme. Should implement Contiguous and Modulo. |
| **Metrics** | R^2 only (user directive) | **Spearman rho** (primary everywhere) | Major gap. The entire antibody/DMS field uses Spearman. R^2 is non-standard. At minimum, report Spearman alongside R^2. |
| **Preprocessing** | Raw fitness values from FLAb CSVs | Per-assay normalization, multi-mutant filtering | Should verify whether FLAb data is pre-normalized or needs z-scoring. |
| **Baselines** | Ridge, RF, XGBoost on KIRBy reps | Ridge on ESM-1v/ESM-2/Tranception embeddings (ProteinGym standard) | Missing ProteinGym standard baselines for comparison. |

---

## 3. DDG (Binding Free Energy)

### 3.1 Standard Benchmark Datasets

| Benchmark | Size | Scope | Notes |
|-----------|------|-------|-------|
| **SKEMPI 2.0** | 7,085 mutations, 348 complexes | General protein-protein DDG | Gold standard. Manually curated. Includes kinetics (1,844) and thermo (443). |
| **AB-Bind** | 1,101 mutations, 27 antibodies + 6 proteins | Antibody-specific DDG | Direct DDG measurements. 32 Ab-Ag complexes. |
| **CATH-ddG splits** | 813 mutations in test (30% of CATH superfamilies) | Leak-free SKEMPI subset | Addresses data leakage by splitting on CATH homologous superfamilies. |
| **DACUM** | Database of Affinity Change Upon Mutations | PPI mutations | From HADDOCK group. |
| **AbBiBench** | 184,500 measurements | Ab-Ag affinity maturation | New (2025). Unique mutant complexes prevent data leakage. |

### 3.2 Standard Evaluation Protocol

**Critical issue: Data leakage in SKEMPI v2.0**

Three types of leakage identified (CATH-ddG paper, 2025):
1. Test set PPIs contain the same mutations as training set
2. Data splitting violates SKEMPI v2.0's proposed held-out proteins
3. Homologous PPIs appear in both training and test sets

**Proper split (CATH-ddG):**
- Split by CATH homologous superfamilies
- Greedy strategy ensures no leaking between train/test
- Held-out test: 54 CATH superfamilies, 53 PPIs, 813 mutations (12.12% of SKEMPI v2.0)
- Under this clean split, deep learning performance drops substantially

**Standard practice:**
- Leave-one-complex-out CV (older, simpler)
- CATH-superfamily split (newer, leak-free)
- Report forward AND reverse mutation performance

### 3.3 Standard Metrics

| Metric | Usage | Notes |
|--------|-------|-------|
| **Pearson r** | Primary | Standard for DDG. Measures linear correlation. |
| **RMSE** | Primary | kcal/mol units. Important for absolute accuracy. |
| **Spearman rho** | Secondary | Rank correlation. |
| **R^2** | NOT standard | Same as antibody -- field uses Pearson r, not R^2. |

### 3.4 Preprocessing

- **DDG computation from Kd**: DDG = RT * ln(Kd_mut / Kd_wt), where R = 1.9872036e-3 kcal/(mol*K), T typically 298.15K
- **Reverse mutation handling**: Antisymmetric property: DDG(M->W) = -DDG(W->M). Models should enforce or be tested for this.
- **Reverse mutation augmentation**: Effectively doubles training data while enforcing physical constraint. Standard practice.
- **Outlier removal**: Remove entries with DDG > 5 kcal/mol or DDG < -5 kcal/mol (some protocols).
- **Multi-mutation parsing**: SKEMPI format "YH27A,SH30A" means multiple simultaneous mutations. Parse each individually.
- **PDB quality**: Use highest-resolution wildtype structure when multiple are available.

### 3.5 SOTA

| Model | Pearson r (SKEMPI standard split) | Pearson r (CATH-ddG clean split) | Notes |
|-------|-----------------------------------|----------------------------------|-------|
| **FoldX** | ~0.5-0.6 | -- | Physics-based. Fast. |
| **Flex ddG** | ~0.55-0.65 | -- | Rosetta-based. Outperforms many DL methods when leakage is corrected. |
| **CATH-ddG** | -- | **0.615** | SOTA on leak-free split |
| **RDE-Network** | -- | 0.459 | Previous SOTA, drops on clean split |
| **Graphinity** | up to 0.87 (likely leaky) | -- | GNN on Ab-Ag structures |
| **CSM-AB** | 0.61-0.64 | -- | Single/multi-point mutations |

**Key insight**: When data leakage is corrected, deep learning predictors lag behind physics-based methods like Flex ddG. Clean evaluation matters enormously.

### 3.6 Key Repos

- [SKEMPI 2.0](https://life.bsc.es/pid/skempi2/) -- Database
- [ak422/CATH-ddG](https://github.com/ak422/CATH-ddG) -- Leak-free split + model
- [haddocking/DACUM](https://github.com/haddocking/DACUM) -- PPI mutation database

### 3.7 What KIRBy Does vs. What It SHOULD Do

| Aspect | KIRBy Current | Standard Practice | Gap |
|--------|--------------|-------------------|-----|
| **Datasets** | SKEMPI 2.0 + AB-Bind, deduplicated, Ab-Ag filtered | SKEMPI 2.0 with CATH-ddG clean splits | Using SKEMPI + AB-Bind is good, but split strategy may have leakage. |
| **Splitting** | GroupKFold by PDB complex | CATH homologous superfamily split | Current complex-based split may leak via homologous structures. |
| **DDG computation** | RT * ln(Kd_mut/Kd_wt) | Same formula | Correct. |
| **Metrics** | R^2 | Pearson r (primary), RMSE (primary) | Should report Pearson r and RMSE, not R^2. |
| **Reverse mutations** | Not handled | Antisymmetric augmentation standard | Should augment with DDG(rev) = -DDG(fwd) or at least test antisymmetry. |
| **Preprocessing** | Ab-Ag filtering by keyword matching | Standard practice | Good approach, but keyword matching may miss some complexes. |

---

## 4. DTA (Drug-Target Affinity)

### 4.1 Standard Benchmark Datasets

| Dataset | Size | Metric | Transform | Notes |
|---------|------|--------|-----------|-------|
| **Davis** | 68 drugs x 442 kinases = 30,056 pairs | Kd (nM) | pKd = -log10(Kd/1e9) | Kinase selectivity. Dense matrix (no missing). |
| **KIBA** | 2,111 drugs x 229 proteins, ~118K valid pairs | KIBA score | Negated (higher = stronger) | Integrated from Ki, Kd, IC50. Sparse (~75% missing). |
| **BindingDB** | >2.6M data points | Various (Ki, Kd, IC50, EC50) | Depends on assay | Much larger but noisier. |
| **PDBbind** | 23,496 complexes (19,443 protein-ligand) | Experimental binding affinity | -log(Kd or Ki) | 3D structure-based. CASF-2016 test set (285 complexes). Major leakage issues. |

### 4.2 Standard Evaluation Protocol

**Split types (in order of importance):**

1. **Warm start** (standard random): Both drugs and targets can appear in train and test. Easiest. Not recommended for final evaluation.
2. **Cold drug**: Test drugs are unseen in training. Tests drug generalization.
3. **Cold target**: Test proteins are unseen in training. Tests protein generalization.
4. **Cold both (cold pair)**: Both drug and target are unseen. Hardest. Most realistic for novel target prediction.

**Standard split ratio:** 5:1 (train:test), following DeepDTA/GraphDTA convention.

**Important:** The DTA field has NOT standardized on a single split strategy. Many papers use warm-start random splits (which are trivially easy and should be avoided). Cold-target or cold-drug splits are more meaningful.

**PDBbind-specific:** The CASF-2016 benchmark had significant train-test data leakage. PDBbind CleanSplit (2023) addresses this but performance drops substantially when leakage is removed.

### 4.3 Standard Preprocessing

- **pKd transform**: Davis Kd values (nM) -> pKd = -log10(Kd / 1e9). Standard since DeepDTA (2018).
- **KIBA score negation**: Negate so higher = stronger binding (consistent direction with pKd).
- **SMILES canonicalization**: RDKit canonical SMILES for all drug structures.
- **Protein sequence length**: Sequences range from 400-2549 AA in Davis. Many models truncate to max 1000 or 1200 AA for computational efficiency. No universal standard.
- **Missing value handling**: KIBA has ~75% missing entries. Standard: only use observed (non-NaN) pairs.
- **Duplicate removal**: Remove exact duplicates by (drug, protein) pair.

### 4.4 Standard Metrics

| Metric | Importance | Notes |
|--------|-----------|-------|
| **CI (Concordance Index)** | Primary | Measures ranking ability. CI > 0.5 = better than random. |
| **MSE** | Primary | Measures absolute prediction error. |
| **R^2** | Secondary | Less common in DTA than CI. |
| **Pearson r** | Secondary | Linear correlation. |
| **Spearman rho** | Secondary | Rank correlation. |
| **Rm^2** | Used in some papers | Modified R^2 for external validation. |

### 4.5 SOTA (Davis and KIBA, warm-start split)

| Model | Davis CI | KIBA CI | Year | Architecture |
|-------|----------|---------|------|-------------|
| **GS-DTA** | 0.903 | 0.895 | 2024 | GAT + Transformer |
| **DeepH-DTA** | 0.924 | 0.927 | 2024 | Hierarchical features |
| **3DProtDTA** | 0.891 | -- | 2024 | 3D structure-based |
| **FusionDTA** | -- | 0.898 | 2023 | Multi-head attention |
| **GramSeq-DTA** | ~0.89 | ~0.90 | 2024 | Grammar-based |
| **GraphDTA** | 0.878 | 0.878 | 2020 | GNN baseline |
| **DeepDTA** | 0.878 | 0.858 | 2018 | CNN baseline |

**WARNING:** These numbers are almost all on warm-start (random) splits. Cold-split performance is significantly lower. The field has an inflation problem from easy splits.

### 4.6 Key Repos

- [hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA) -- Original data source and CNN baseline
- [thinng/GraphDTA](https://github.com/thinng/GraphDTA) -- GNN baseline
- [dingyan20/Davis-Dataset-for-DTA-Prediction](https://github.com/dingyan20/Davis-Dataset-for-DTA-Prediction) -- Cleaned Davis data

### 4.7 What KIRBy Does vs. What It SHOULD Do

| Aspect | KIRBy Current | Standard Practice | Gap |
|--------|--------------|-------------------|-----|
| **Datasets** | Davis and KIBA from DeepDTA GitHub | Davis, KIBA, BindingDB | Good. Could add BindingDB for scale. |
| **Splitting** | Cold-target GroupKFold (5-fold) | Cold-target or cold-drug; warm-start is too easy | KIRBy is actually ahead here -- cold-target is the right approach. Should also evaluate cold-drug. |
| **pKd transform** | -log10(Kd/1e9) for Davis | Same | Correct. |
| **KIBA transform** | Negate | Standard is to negate | Correct. |
| **Metrics** | R^2 | **CI (primary)**, MSE | Major gap. Should report CI as the primary metric for DTA. R^2 is secondary. |
| **Protein preprocessing** | No truncation | Many models truncate to 1000-1200 AA | Not critical for representation comparison but worth noting. |
| **Cold-drug split** | Implemented (assign_cold_drug_groups) but not used in survey | Should evaluate both cold-target and cold-drug | Should run both. |
| **Comparison to SOTA** | Internal baselines only | DeepDTA CI ~0.878 as reference | Should report CI alongside published baselines. |

---

## 5. TCGA Cancer Genomics

### 5.1 Standard Classification Benchmarks

| Task | Classes | Samples | Difficulty | Notes |
|------|---------|---------|------------|-------|
| **LUAD vs LUSC** | 2 (adenocarcinoma vs squamous) | ~682 (359 LUAD + 323 LUSC) | Easy with gene expression | Near-perfect with deep learning (AUC >0.98). |
| **BRCA PAM50** | 4 (LumA, LumB, Basal, HER2) | ~836-900 | Moderate | HER2 and Normal subtypes hardest to classify. |
| **BRCA survival** | Regression (OS) or binary | Variable (depends on censoring) | Hard | Uncensored subset is small. |
| **Pan-cancer type** | 33 cancer types | ~10,000+ | Variable | Multi-class classification. |
| **Pan-cancer subtype** | 106 subtypes across 26 cohorts | ~8,791 | Hard | Recent work (2025) achieves compact feature sets. |

### 5.2 Standard Evaluation Protocol

**CRITICAL: Site-preserved cross-validation (Howard et al., Nature Comms 2021)**

- TCGA samples come from multiple tissue submitting sites (TSS)
- Deep learning models can identify submitting sites from histology alone
- If TSS correlates with labels, models learn site artifacts instead of biology
- **Standard practice**: Site-preserved CV ensures all samples from one TSS are in the same fold
- Without site preservation, accuracy is inflated by site-specific batch effects
- Effect size is small for features with clear histologic basis (subtype, grade) but significant for demographic/genetic features

**Standard CV:** 5-fold stratified CV with site preservation. Some studies use 80/10/10 single split.

### 5.3 Standard Preprocessing

1. **Expression normalization**: log2(TPM + 0.001) or log2(TPM + 1). UCSC Xena Toil recount2 provides pre-computed log2(TPM+1).
2. **Batch correction**: ComBat (standard). Improves weighted F1-score for tissue-of-origin classification.
3. **Gene filtering**: Remove low-expression genes (< n counts in < p% of samples, user-defined n, p). Common: keep protein-coding genes only (GENCODE annotation).
4. **Sample filtering**: Remove low-purity tumor samples using pre-computed purity values.
5. **Feature selection**: Most studies use top N most variable genes (N = 2000-5000) or PAM50 gene set for BRCA.

**CAUTION:** Preprocessing can worsen classification when test data comes from independent studies (ICGC, GEO). Preprocessing choices are not universally beneficial.

### 5.4 Standard Metrics

| Task | Primary Metric | Secondary | Notes |
|------|---------------|-----------|-------|
| Binary classification | **AUROC** | Balanced accuracy, F1 | LUAD vs LUSC |
| Multi-class | **Weighted F1** or **macro AUROC** | Accuracy | PAM50, pan-cancer |
| Survival | **C-index** | Log-rank p-value | Concordance for survival |

### 5.5 SOTA

| Task | SOTA Model | Performance | Notes |
|------|-----------|-------------|-------|
| **LUAD vs LUSC** | OmicsFootPrint (EfficientNetV2) | AUC 0.98 +/- 0.02 | Gene expression only |
| **LUAD vs LUSC** | Multi-omics (RNA-seq + methylation) | Accuracy ~99.8% | Fusing two modalities |
| **BRCA PAM50** | OmicsFootPrint | AUC 0.83 +/- 0.07 | 4-class |
| **BRCA PAM50** | Multi-omics deep learning | 91% accuracy (external test) | Training 98%, validation 97% |
| **Pan-cancer (23 types)** | GA + RNAseq | 98.81% average accuracy | Feature selection approach |
| **BRCA survival** | PRISM (multi-omics) | C-index 0.774 | DNA methylation + miRNA + CNV |

### 5.6 Multi-Omics Integration Benchmarks

| Benchmark | Scope | Modalities | Notes |
|-----------|-------|------------|-------|
| **MLOmics** | 8,314 patients, 32 cancer types | mRNA, miRNA, methylation, CNV | Open database. 6-10 baseline methods per task. Uniformly processed. |
| **TCGA Pan-Cancer Atlas** | All TCGA data | All available modalities | Standard source. Batch-corrected matrices available. |

### 5.7 Key Repos

- [MLOmics](https://arxiv.org/html/2409.02143v2) -- ML benchmark for cancer multi-omics
- [UCSC Xena](https://xenabrowser.net/) -- Standardized TCGA data access

### 5.8 What KIRBy Does vs. What It SHOULD Do

| Aspect | KIRBy Current | Standard Practice | Gap |
|--------|--------------|-------------------|-----|
| **Datasets** | LUAD vs LUSC, BRCA PAM50, BRCA survival | Same + pan-cancer | Core tasks are correct. |
| **Preprocessing** | log2(TPM+1) from UCSC Xena Toil recount2 | Same | Correct. |
| **Batch correction** | Not implemented | ComBat standard | Should add ComBat batch correction. |
| **Gene filtering** | Protein-coding only (GENCODE) | Same + variance filtering | Correct base approach. Could add variance filtering. |
| **Splitting** | GroupKFold by site (TSS-aware) | Site-preserved CV (Howard 2021) | Need to verify KIRBy uses TSS-preserved splits. If grouping by something else, this is a gap. |
| **Metrics** | R^2 for survival, accuracy for classification | AUROC (binary), weighted F1 (multi-class), C-index (survival) | Should report AUROC for classification tasks. |
| **Multi-omics** | Gene expression only (+ WSI, methylation separate) | Fused multi-omics (mRNA + methylation + miRNA + CNV) | Multi-omics integration would be a natural extension. |

---

## 6. Reaction Yields

### 6.1 Standard HTE Datasets

| Dataset | Reactions | Reaction Type | Reference | Notes |
|---------|-----------|---------------|-----------|-------|
| **Buchwald-Hartwig (BH)** | 3,955 | C-N cross-coupling | Ahneman et al., Science 2018 | 1 amine, 3 bases, 4 ligands, 15 aryl halides, 23 isoxazole additives |
| **Suzuki-Miyaura** | 5,760 | C-C cross-coupling | Perera et al., Science 2018 | Automated flow synthesis, LC-MS analysis |
| **Deoxyfluorination** | 740 | Alcohol deoxyfluorination | Literature (various) | Smaller. Used for interaction effects modeling. |
| **USPTO yield** | Large | Various | Schwaller et al. | Sub-gram and gram subsets. Heterogeneous quality. |
| **ORDerly** | Variable | Various | From Open Reaction Database | Benchmark for forward prediction, retrosynthesis, conditions |

### 6.2 Standard Evaluation Protocol

**CRITICAL ISSUE: Random splits are misleading for HTE data**

"When Yield Prediction Does Not Yield Prediction" (Beker et al., JCIM 2024):
- Models perform well on HTE random splits (R^2 > 0.82)
- Same models perform poorly on real-world ELN data (R^2 = 0.266)
- Random splits within an HTE plate don't test generalization to new conditions
- The HTE combinatorial structure means random splits leak information about reagent effects

**Proper evaluation splits for BH:**
1. **Random 70/30**: Standard baseline. Matches Sandfort et al. and Schwaller et al. Easy.
2. **Out-of-sample (isoxazole additives)**: 4 test sets defined by holding out isoxazole additives. Much harder. Tests extrapolation to unseen reagents.
3. **Leave-one-reagent-out**: Hold out all reactions with one specific reagent (ligand, base, additive). Most challenging. Reveals true generalization.

**Proper evaluation splits for Suzuki:**
- Similar: random splits vs. leave-one-catalyst-out or leave-one-substrate-out.

### 6.3 Standard Preprocessing

- **Yield normalization**: BH yields 0-100 (percentage). Suzuki yields originally 0-1, typically scaled to 0-100.
- **Reaction SMILES construction**: Reactants.Reagents>>Products. Include catalyst, ligand, base, additive, solvent.
- **Component encoding**: One-hot encoding of categorical reagents (simplest), DFT descriptors (Ahneman), Morgan fingerprints (DRFP), or reaction SMILES (Yield-BERT).
- **Missing yield handling**: HTE data is usually complete (combinatorial design), but very low yields (<1%) may be noise.

### 6.4 Standard Metrics

| Split Type | Primary Metric | Notes |
|------------|---------------|-------|
| Random 70/30 | R^2, RMSE | Easy baseline |
| Out-of-sample (OOS) | R^2, RMSE per OOS test set | Mean across 4 OOS test sets |
| Leave-one-reagent-out | R^2, RMSE per held-out reagent | Most informative |

### 6.5 SOTA

**Buchwald-Hartwig (Random 70/30):**

| Model | R^2 | RMSE | Notes |
|-------|-----|------|-------|
| Yield-BERT (augmented) | ~0.95 | ~7 | BERT on reaction SMILES |
| DRFP + RF | ~0.93 | ~8 | Differential reaction fingerprint |
| RXNFP | ~0.95 | ~7 | Reaction fingerprint from BERT |
| DFT descriptors + RF | ~0.92 | -- | Ahneman original |

**Buchwald-Hartwig (Out-of-sample, mean across 4 OOS):**

| Model | Mean R^2 (OOS) | Notes |
|-------|---------------|-------|
| DRFP + RF | Best on average across all 11 tests | Outperforms augmented Yield-BERT on OOS |
| Egret | Best in 3/4 OOS tasks | 2024 |
| DFT + RF | Competitive | Ahneman baseline |

**Suzuki-Miyaura (Random 70/30):**

| Model | R^2 | Notes |
|-------|-----|-------|
| Yield-BERT | ~0.85 | Lower than BH |
| DRFP + RF | ~0.82 | Competitive |
| GPs with ARD | Superior to NNs on Perera data | Non-parametric |

**Key insight:** Random-split performance is unrealistically high. Out-of-sample and leave-one-reagent-out performance drops substantially and is more informative.

### 6.6 Key Repos

- [rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields) -- Schwaller et al. data + RXNFP/DRFP
- [ORDerly](https://github.com/ORDerly) -- Open Reaction Database benchmarks

### 6.7 What KIRBy Does vs. What It SHOULD Do

| Aspect | KIRBy Current | Standard Practice | Gap |
|--------|--------------|-------------------|-----|
| **Datasets** | BH (3,955) and Suzuki (5,760) | Same | Correct. Could add deoxyfluorination. |
| **Splitting** | Random 70/30 (10 splits) | Random 70/30 PLUS out-of-sample (isoxazole) splits | **Major gap.** Random-only splits are misleading. Must implement OOS splits for BH. |
| **Preprocessing** | Full reaction SMILES with components parsed | Same | Correct. |
| **Metrics** | R^2, RMSE, MAE | Same | Correct for reaction yields. |
| **Evaluation** | 10 random splits, averaged | Should also report 4 OOS test sets | Missing the most informative evaluation. |
| **Comparison** | Internal baselines only | DRFP R^2 ~0.93 (random), much lower OOS | Need published reference numbers. |

---

## Cross-Cutting Issues

### Issue 1: Metric Mismatch

KIRBy uses R^2 as the primary metric everywhere. This is non-standard in most domains:

| Domain | KIRBy Metric | Standard Primary Metric |
|--------|-------------|------------------------|
| Molecular QSAR | R^2 | RMSE |
| Antibody/DMS | R^2 | Spearman rho |
| DDG | R^2 | Pearson r + RMSE |
| DTA | R^2 | CI (Concordance Index) |
| TCGA Classification | accuracy | AUROC |
| Reaction Yields | R^2 | R^2 (one domain where it's fine) |

**Recommendation:** Add domain-specific metrics. R^2 can remain for internal comparison but each domain should report its standard metric for external comparability.

### Issue 2: Splitting Rigor

KIRBy uses scaffold GroupKFold throughout, which is good. But:

1. **DTA**: Cold-target GroupKFold is correct and ahead of many published papers.
2. **Molecular**: Scaffold GroupKFold is standard but UMAP splits are more rigorous.
3. **Antibody/DMS**: Random GroupKFold is the easiest ProteinGym scheme. Should test contiguous/modulo.
4. **DDG**: Complex-based GroupKFold may leak via homologous structures. Should use CATH-superfamily splits.
5. **TCGA**: Must verify site-preserved (TSS-based) grouping.
6. **Reaction yields**: Random-only is insufficient. Must add out-of-sample reagent splits.

### Issue 3: Preprocessing Gaps

1. **Molecular**: Missing salt stripping, neutralization, tautomer standardization, InChIKey deduplication.
2. **DDG**: Missing reverse mutation augmentation.
3. **TCGA**: Missing ComBat batch correction.
4. **DMS/Antibody**: Should verify per-assay normalization.

### Issue 4: External Validation

KIRBy only reports internal baselines. Each domain should include published SOTA numbers for context:
- Molecular: Chemprop, ADMET-AI, Polaris leaderboard numbers
- Antibody: ProteinGym supervised leaderboard
- DDG: FoldX, Flex ddG, CATH-ddG numbers
- DTA: DeepDTA, GraphDTA CI numbers
- Reaction yields: DRFP/RXNFP published R^2 values

### Issue 5: Dataset Sources

ESOL is currently loaded via DeepChem's MoleculeNet loader, which contradicts the MoleculeNet ban. Should source ESOL data independently (the original Delaney 2004 CSV is publicly available without MoleculeNet).

---

## Sources

### Molecular QSAR
- [Polaris Hub](https://polarishub.io/)
- [OpenADMET](https://openadmet.org/)
- [Pat Walters - Some Thoughts on Splitting Chemical Datasets (Nov 2024)](http://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html)
- [Pat Walters - We Need Better Benchmarks for ML in Drug Discovery](http://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html)
- [Inductive Bio - Building Better Benchmarks for ADME Optimization](https://www.inductive.bio/blog/building-better-benchmarks-for-adme-optimization)
- [Scaffold Splits Overestimate Virtual Screening Performance (ICANN 2024)](https://arxiv.org/abs/2406.00873)
- [PharmaBench (Scientific Data 2024)](https://www.nature.com/articles/s41597-024-03793-0)
- [Benchmark-ADMET-2025](https://github.com/DonghaiZHAO-ZJU/Benchmark-ADMET-2025)
- [QSARtuna (JCIM 2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00457)
- [ADMET-AI (Bioinformatics 2024)](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030)
- [QSAR-Ready Workflow (J. Cheminformatics 2024)](https://link.springer.com/article/10.1186/s13321-024-00814-3)
- [Beware of R^2 (JCIM 2015)](https://pubs.acs.org/doi/10.1021/acs.jcim.5b00206)
- [Lessons from the Polaris ADMET competition](https://www.inductive.bio/blog/lessons-from-the-polaris-admet-competition)

### Antibody / DMS
- [ProteinGym (NeurIPS 2023)](https://proteingym.org/)
- [ProteinGym GitHub](https://github.com/OATML-Markslab/ProteinGym)
- [FLIP (NeurIPS 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/2b44928ae11fb9384c4cf38708677c48-Paper-round2.pdf)
- [FLIP2 (2026)](https://flip.protein.properties/)
- [FLAb (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.01.13.575504v1)
- [FLAb GitHub](https://github.com/Graylab/FLAb)
- [AbBiBench (arXiv 2025)](https://arxiv.org/abs/2506.04235)
- [ProteinNPT (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/file/6a4d5d85f7a52f062d23d98d544a5578-Paper-Conference.pdf)

### DDG
- [SKEMPI 2.0 (Bioinformatics 2019)](https://academic.oup.com/bioinformatics/article/35/3/462/5055583)
- [AB-Bind (PMC 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815335/)
- [CATH-ddG (Bioinformatics 2025)](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i362/8199350)
- [Resolving data bias in binding affinity (Nature Machine Intelligence 2025)](https://www.nature.com/articles/s42256-025-01124-5)
- [Antibody-antigen DDG volume and diversity (Nature Comp. Sci. 2025)](https://www.nature.com/articles/s43588-025-00823-8)

### DTA
- [DeepDTA (Bioinformatics 2018)](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245)
- [GraphDTA (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970)
- [DTA Benchmark Survey (Briefings in Bioinformatics 2025)](https://academic.oup.com/bib/article/26/5/bbaf491/8260789)
- [DTI Benchmark from Structure Perspective (2024)](https://arxiv.org/html/2407.04055)
- [PDBbind CleanSplit](https://arxiv.org/html/2308.09639v2)

### TCGA
- [Howard et al. - Site-specific digital histology signatures (Nature Comms 2021)](https://www.nature.com/articles/s41467-021-24698-1)
- [MLOmics (Scientific Data 2025)](https://www.nature.com/articles/s41597-025-05235-x)
- [OmicsFootPrint (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.03.21.586001v1.full)
- [RNA-Seq preprocessing comparison (BMC Bioinformatics 2024)](https://link.springer.com/article/10.1186/s12859-024-05801-x)

### Reaction Yields
- [When Yield Prediction Does Not Yield Prediction (JCIM 2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01524)
- [DRFP (Digital Discovery 2022)](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c)
- [rxn_yields GitHub](https://github.com/rxn4chemistry/rxn_yields)
- [Ahneman et al. (Science 2018)](https://doyle.princeton.edu/wp-content/uploads/2020/07/43-Predicting-Reaction-Performance-in-C-N-Cross-Coupling-Using-Machine-Learning.pdf)
- [Perera et al. (Science 2018) - Suzuki flow synthesis](https://www.science.org/doi/10.1126/science.adc8743)
- [Real-world datasets for reaction yield (Doyle lab)](https://doyle.chem.ucla.edu/wp-content/uploads/2023/03/76.pdf)
