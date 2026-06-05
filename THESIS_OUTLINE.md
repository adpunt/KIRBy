# DPhil Thesis Outline

**Working title:** *Hybrid Representation Learning for Drug Discovery: Compatibility, Evaluation, and Noise Robustness Across Molecular, Antibody, and Multi-Omics Domains*

**Candidate:** Adelaide Punt
**Department:** Department of Statistics, University of Oxford
**Group affiliation:** OPIG (Oxford Protein Informatics Group)
**Format:** Integrated thesis (MPLS-permitted; 3–5 paper-equivalent contribution chapters bracketed by commentary). See [MPLS integrated thesis guidance](https://www.mpls.ox.ac.uk/graduate-school/information-for-postgraduate-research-students/submitting-your-thesis/guidance-for-submitting-an-integrated-thesis).

---

## Thesis statement

Across drug discovery domains, the **input representation matters more than the downstream model architecture**, and *principled hybridisation of multiple learned representations*, combined with *matched internal evaluators*, produces measurable and consistent gains across molecular property prediction, antibody developability, drug–target affinity, and multi-omics cancer phenotype prediction. The framework is forward-compatible with new foundation models without code changes, and its behaviour can be characterised — and predicted — through diagnostics on data complexity, similarity structure, and label noise.

---

## Structural template

Body layout follows the OPIG/Morris-group convention of **3–5 substantive contribution chapters**, each ≈ one paper, with a per-chapter **Preamble** declaring publication status and collaborator attribution (Moesser-style). Introduction is a short Dablander-style technical primer with "Critical View" subsections per representation family. No spinoff work (kirby-av, kirby-finance, kirby-rogii, ANNITIA Trustii challenge) is included anywhere — they are out of scope by candidate decision.

---

# Chapter 1 — Introduction

*Target length: ≈ 8–12 pages. Modelled on Dablander's compact §1 (4 pp.) but slightly extended to handle the multi-domain framing.*

## 1.1 Why machine learning for drug discovery

- Eroom's law framing: doubling cost per approved drug, 95% attrition, ML as cost-reducer
- Two operative obstacles for ML in this space: **data scarcity** (small bioactivity datasets, expensive labels) and **representation choice** (no single featurisation dominates across tasks)

## 1.2 The representation problem

- Molecular, antibody, protein, gene-expression, structural, reaction, and nucleotide modalities each have their own preferred representation families
- Hand-engineered descriptors (ECFP, Mordred), learned embeddings (graph NNs, ESM, AntiBERTy), and structural reps coexist without consensus
- The "best" representation depends on task, dataset size, model, and noise level — and no published benchmark covers all four simultaneously

## 1.3 Why hybrids

- Combination is not concatenation: high-dimensional concatenation overfits in small-data regimes
- Feature-importance-weighted allocation as a principled alternative
- The interaction between *internal evaluator* (the model used during the hybrid search) and *downstream model* (the final reported model) is itself a research object

## 1.4 Thesis statement and outline of contributions

A per-chapter bulleted walkthrough (Dablander §1.1 style). Each bullet states what the chapter does, the headline finding, and the publication/dissemination status. Draft bullets:

- **Chapter 3** — *NoiseInject*: a benchmarking framework and survey of noise robustness and uncertainty quantification in QSAR models across 8 representations × 14 models × 6 noise injection strategies. *Status: paper draft `paper.tex` in `qsar_qm_models/`, target venue TBD; supplementary PDV case study `pdv_case_study_draft.tex`.*
- **Chapter 4** — *KIRBy*: a framework for hybrid molecular and biological representation combination via importance-weighted feature allocation, with matched internal-evaluator/downstream-model selection. Federated extension included as §4.7. *Status: code complete (`src/kirby/find_best_hybrid.py`), paper in preparation.*
- **Chapter 5** — *Cross-domain hybrid surveys*: KIRBy-augmented baselines across 8 molecular, ~24 antibody, 3 TCGA, and 3 DTA datasets, with domain-aware metrics and splitting. *Status: surveys complete (Phase 5 in `PIPELINE_READINESS_TODO.md`).*
- **Chapter 6** — *When and why hybrids work*: diagnostic experiments on data complexity (C01–C24), similarity structure (S01–S13), and label noise (N01–N13). *Status: code complete; results in flight on ARC.*
- **Chapter 7** — *Forward compatibility and collaboration*: demonstrating KIRBy's compatibility with newly released foundation models (MINT, SaProt, ESM-C, Boltz-2) and with OPIG collaborators' new datasets (Trastuzumab-HER2, Graphinity ΔΔG, Humanness, Ginkgo developability). *Status: implementations complete; analysis underway.*

---

# Chapter 2 — Background

*Target length: ≈ 30–40 pages. Dablander-style structured primer: each subsection ends with a "Critical View" paragraph rather than just exposition. The depth aligns with MPLS integrated-thesis requirement for a "full literature review" (stand-alone or distributed).*

## 2.1 Drug discovery pipeline and ML's role

- Hit discovery, lead optimisation, ADMET screening
- Where in the pipeline learned models currently sit (virtual screening, property prediction, affinity prediction)

## 2.2 Molecular representations

- **2.2.1** Fingerprints: ECFP/ECFP4, MACCS, atom-pair, Avalon, Mordred PDV
- **2.2.2** Learned: graph kernels, Weisfeiler-Lehman, GROVER, Chemprop, MHG-GNN, UniMol
- **2.2.3** Sort & Slice (SNS) as a substructure-pooling improvement over hashing (Dablander 2024, *Journal of Cheminformatics*)
- **2.2.4** Critical view: each family has known failure modes (size bias for fingerprints, scaffold-hopping failures for GNNs, dead features in PDV)
- Source: `src/kirby/representations/molecular.py`

## 2.3 Antibody representations

- **2.3.1** Sequence-only language models: AbLang v1/v2, AntiBERTy, IgBERT, ESM-2 variants, ESM-C, Ab-RoBERTa, MINT
- **2.3.2** CDR-specific extractions, ANARCI numbering, region ablation
- **2.3.3** Physicochemical descriptors, BLOSUM substitution, developability features (CamSol etc.)
- **2.3.4** Mutation encoding for DMS tasks: position + WT/MUT AA + local context
- **2.3.5** Critical view: full-chain vs CDR-only, frozen vs fine-tuned, concat vs paired
- Source: `src/kirby/representations/antibody.py`, `mutation.py`

## 2.4 Protein, structural, and DTA representations

- **2.4.1** ESM-2 layer variants, SaProt (with ProstT5 3Di tokens), Boltz-2
- **2.4.2** PDB-derived geometry; structural fingerprints
- **2.4.3** DTA-specific: cold-target evaluation, drug Sort & Slice
- Source: `src/kirby/representations/protein.py`, `structural.py`

## 2.5 Multi-omics (TCGA) representations

- **2.5.1** Genomic foundation models: GeneFormer v1/v2, scFoundation
- **2.5.2** WSI vision transformers: Titan, Uni2H, CONCH, Hoptimus1, Virchow2
- **2.5.3** Pre-computed PanCanAtlas reps: CIBERSORT, xCell, mRNAsi, MATH, HRD, MSI, CIMP
- **2.5.4** Critical view: which modalities are needed for which task (modality ablation feeds forward to Ch 6 §6.3)
- Source: `src/kirby/representations/genomic.py`, `structural.py`; `BILL_OF_MATERIALS.md` Sections 4–6

## 2.6 Reaction representations

- RxnFP, CGR (condensed graph of reaction), one-hot components, SMILES baselines
- Source: `src/kirby/representations/reaction.py`

## 2.7 Hybrid representation combination — prior work

- Multi-view learning, late vs early fusion
- Stacked ensembles vs feature-level fusion
- Why concatenation is not enough: dimensionality, redundancy, feature-importance dilution
- Federated knowledge distillation (Lhasa FLuID, *Nature MI* 2025) as a related but orthogonal direction

## 2.8 Noise and uncertainty in QSAR

- Sources of noise: experimental measurement error, label discretisation, assay drift
- Uncertainty taxonomy: aleatoric vs epistemic
- Quantification methods: Gaussian Processes, NGBoost, Bayesian NNs, Variational Bayesian Last Layer (VBLL)
- Calibration metrics: ECE, coverage at 1σ/2σ, Spearman ρ on residuals
- Feeds Chapter 3

## 2.9 Splitting and evaluation in drug discovery

- Why random splits fail: scaffold leakage, replicate leakage, cold-target leakage, cohort leakage
- Domain-aware protocols: scaffold GroupKFold, position-modulo GroupKFold (antibody DMS), identity-grouped GroupKFold (antibody inter-family), TSS StratifiedGroupKFold (TCGA), cold-target GroupKFold (DTA)
- Metric choice per domain (RMSE for molecular regression, ρ for antibody, AUROC/F1w for classification, CI for DTA)
- Source: `issues_splitting.md` (477 lines, 15+ refs); `BILL_OF_MATERIALS.md`

## 2.10 Critical overview

A Dablander §2.7-style synthesis table: strengths/weaknesses of representation families across the modalities the thesis touches. Motivates the hybrid approach in Ch 4.

---

# Chapter 3 — NoiseInject: A Survey of Noise Robustness and Uncertainty Quantification in QSAR Models

*Target length: ≈ 35–45 pages. The qsar_qm_models paper as a contribution chapter. Moesser-style Preamble at the top.*

**Preamble** — *"This chapter is based on the manuscript drafted in `qsar_qm_models/paper.tex` (NoiseInject framework). The supplementary case study on Gaussian Process kernel–representation interaction was extracted to `qsar_qm_models/pdv_case_study_draft.tex` for revision. All experiments, code, and analysis were carried out by the author. Submission status: in preparation."*

## 3.1 Introduction

- Why noise robustness matters in QSAR: real bioactivity data is noisy
- Gap in existing work: prior surveys vary one axis (model OR representation OR noise type) — none cross all three
- Stated contribution (paraphrasing `paper.tex` line 338): investigate "the contributions of molecular representation and model architecture to both overall prediction performance and, more specifically, noise robustness"

## 3.2 Datasets

- **Primary:** QM9 (HOMO-LUMO gap), 10 000-molecule subsets × 10 replicates (`paper.tex` line 170)
- **Validation:** LogD (N=5039), Caco-2 efflux (N=2161), hERG Ki (N=1482) — OpenADMET + ChEMBL sources (line 175)

## 3.3 Representations (8 families)

ECFP4, Sort & Slice fingerprint, 200-D PDVs (continuous and binarised), one-hot-encoded SMILES, mol2vec, MHG-GNN graph embeddings. Per-rep dimensionality table.

## 3.4 Models (14 architectures)

Random Forests, Quantile Regression Forests, XGBoost, LightGBM, NGBoost, SVMs (RBF / poly-2), Gaussian Processes (Tanimoto kernel for binary, RBF for continuous), NN-α [128, 64], NN-β [128, 128], full Bayesian NN, Variational Bayesian Last Layer.

*Note: supervisor edit `paper_edits_supervisor.md` §1b — fix the paper's incorrect claim that SVMs use a Tanimoto kernel; code audit confirms SVMs use RBF (ECFP4) or polynomial-2 (SNS, PDV).*

## 3.5 Noise injection strategies (6)

Gaussian, Outlier (3σ/0.1σ), Quantile (2σ/0.1σ), Threshold (2σ/0.1σ), Value-proportional σ(1+0.1|y|), Heteroscedastic σ√(0.1+0.05|y|). Mathematical definitions per `paper.tex` lines 301–306.

## 3.6 Metrics

- **Performance:** RMSE, R²
- **Noise robustness:** Noise Degradation Slope (NDS) — slope of R² vs noise level
- **Variance decomposition:** two-way ANOVA η² over (model × representation × noise)
- **Uncertainty calibration:** Spearman ρ on residuals, ECE, coverage at 1σ/2σ
- **Rank stability across noise strategies:** Kendall's W
- A consolidated metrics table should be added (supervisor request, `paper_edits_supervisor.md` §2)

## 3.7 Results

- **Headline:** "Model architecture was the dominant factor in variance in degradation slope … NGBoost and SVMs showed the strongest robustness to noise" (`paper.tex` line 128)
- **ANOVA outcome:** For NDS, model architecture dominates variance (line 340) — unlike *baseline* performance, where the model × rep interaction term dominates
- **Cross-strategy ranking:** Kendall's W = 0.92 (p = 2.7 × 10⁻⁸) — a model's relative noise robustness is largely strategy-independent (line 392)

## 3.8 Case study: GP kernel × representation interaction

The PDV case study (extracted as `pdv_case_study_draft.tex`):

- Tanimoto kernel on binary PDV: R² = 0.52 at σ=0.5, 0.33 at σ=0.9 — graceful degradation
- RBF kernel on the same binary PDV: R² = 0.58 at σ=0.4 → collapse to ≈ 0 at σ=0.5 — cliff failure
- NDS, being a linear slope, misses this cliff — motivates a complementary cliff-detection metric

## 3.9 Investigation: PDV binarisation bug

From `scripts/pdv_investigation/REPORT.md`:

- For ≈ 1 year, the PDV had been silently binarised by `(pdv > 0).astype(np.uint8)` at line 1230 of `process_and_train.py`
- Investigation showed the bug worked because of feature degeneracy: 13 always-1, 30 always-0, 29 mostly-1, 52 mostly-0, 76 informative — binarisation effectively turns the 200-D PDV into a functional-group + coarse-surface fingerprint
- Multi-model rerun (RF, XGBoost, SVM, BNN Full, VBLL): XGBoost has the highest baseline R² (0.895) but catastrophic NDS (−27 to −47); BNN Full and VBLL have the most robust NDS (−0.5 to −0.8) — a real performance/robustness tradeoff

## 3.10 Discussion

- Take-aways for practitioners (model > representation for noise robustness)
- Why XGBoost's tree splits fail under noise; why VBLL/BNN succeed
- Limits: single primary dataset (QM9); validation tasks are smaller

## 3.11 Open items (from `paper_edits_supervisor.md`)

To be addressed before the chapter is final:
- §1: clarify two-kernel GP setup up front (lines 19–31)
- §1b: fix SVM kernel claim (lines 50–66)
- §2: add consolidated metrics table (lines 76–120)
- §5: reorder Hetero last in noise-strategy table (lines 169–183)
- §7: fix per-panel histogram bin recomputation bug (lines 237–255)
- §8: BNN paragraph reorganisation (lines 281–314)
- §9: epistemic/aleatoric definitions on first use (lines 324–339)

---

# Chapter 4 — KIRBy: A Framework for Hybrid Representation Combination with Matched Evaluator–Model Selection

*Target length: ≈ 30–40 pages. Core methodology chapter. The "what KIRBy actually is" chapter.*

**Preamble** — *"KIRBy is implemented in the `KIRBy` repository (`src/kirby/`). The framework, allocation algorithms, internal-evaluator architecture, and federated extension were designed and implemented by the author. Manuscript in preparation."*

## 4.1 Introduction and motivation

- The empirical question: given N candidate representations for a task, what's the best way to combine them?
- Why concatenation underperforms in small-data regimes
- Prior work in multi-view fusion (cited from Ch 2 §2.7)

## 4.2 Framework overview

- Two stages: **screening** (pre-filter weak reps via quick 3-fold CV) and **hybrid search** (importance-weighted greedy allocation under a budget)
- Source: `src/kirby/find_best_hybrid.py` (~1800 lines), `src/kirby/screen.py`

## 4.3 Allocation strategies

Seven strategies implemented and characterised:
- Greedy (default)
- Performance-weighted
- Two-stage
- Dominant-greedy
- Fixed
- Stability selection
- Greedy-feature

Validation: framework experiments F01–F03, F11, F12 in `tests/experiments/framework/`.

## 4.4 Importance methods

Eight methods supported:
- TreeSHAP, XGBoost gain, LightGBM gain
- DeepLIFT, KernelSHAP, integrated gradients, GradientSHAP
- Permutation importance

Correlation analysis between methods: framework experiment F05.

## 4.5 Budget and augmentation

- Budget as **input** to `find_best_hybrid`, not as a search dimension (architectural decision documented in `project_experiment_hybrid_architecture.md`)
- Augmentation strategies: uncapped (default), capped, none — see `project_augmentation_decisions.md`
- Cache-collision fix for `aug_strategy` (commits a13e0f4, 390f607, documented in `project_cache_collision_fix.md`)

## 4.6 Matched evaluator–model selection

- The core methodological contribution beyond hybrid search itself
- Problem: when the internal evaluator (model used during the hybrid search) differs from the final downstream model, the search overfits to evaluator-specific feature preferences
- Solution: kNN distance evaluator + matched downstream model set (BNN, CatBoost, MLP)
- Validation: framework experiments F12 (allocation v2) + F13 (evaluator–model matching). Results final 2026-03-29, see `project_f12_f13_results.md`.
- Removed: legacy `best_overall` selection (matched evaluators enforced; documented in `project_model_matching.md`)

## 4.7 Federated extension (Lhasa/FLuID integration)

A subsection rather than a chapter (candidate decision). Frames federated as **one mode** of KIRBy operation:

- **§4.7.1** Motivation: Lhasa's FLuID (Nature MI 2025) used knowledge distillation across 7 pharma partners; student models underperformed the teacher MCC but expanded applicability domain by 15–30%
- **§4.7.2** KIRBy's hypothesis: feature-level fusion avoids prediction-level noise that hurt FLuID students
- **§4.7.3** Five federated experiment configurations (`tests/experiments/federated/`):
  - FED01 — noise heterogeneity across silos
  - FED02 — rep diversity across silos
  - FED03 — synthetic private data
  - FED04 — cluster-aware importance
  - FED05 — antibody federated
- **§4.7.4** Reference roadmap: `FEDERATED_EXPERIMENT_ROADMAP.md`

## 4.8 Implementation details

- Caching architecture: per-fold rep caching with `_cache_meta` per file
- Downstream model registry: 7 standard models (`DOWNSTREAM_MODELS`), plus noise-experiment-specific `DOWNSTREAM_MODELS_NOISE` (+ NGBoost), plus BNN
- Internal evaluator set: `['match', 'rf', 'svm', 'lightgbm', 'mlp']` (`project_internal_evaluators.md`)

## 4.9 Validation experiments

Reference forward to Ch 6 for the diagnostic experiments (complexity/similarity/noise) that interrogate *why* hybrids work — this chapter establishes that they *do*.

## 4.10 Discussion

- Limitations: greedy allocation is a heuristic; no theoretical optimality guarantees
- Cost: hybrid search is more expensive than single-rep training (mitigated by caching)
- Where KIRBy does not help: single-modality datasets where one rep is dominant (foreshadow Ch 6 §6.3)

---

# Chapter 5 — Cross-Domain Hybrid Surveys

*Target length: ≈ 40–50 pages. Domain-by-domain application of KIRBy. This is the largest chapter and the empirical heart of the thesis.*

**Preamble** — *"All surveys were designed and run by the author. Datasets are public or licensed; provenance is documented in `PIPELINE_READINESS_TODO.md` and `BILL_OF_MATERIALS.md`. ChEMBL bulk loaders for the Imrie transfer learning experiment were built in collaboration with OPIG (see Ch 7). Manuscript in preparation."*

## 5.1 Methodology

- Standard survey design: per-dataset, run all candidate reps under KIRBy hybrid search with matched evaluator–model selection
- Domain-aware splits and metrics from Ch 2 §2.9 applied
- Statistical comparison: paired CV folds, Wilcoxon signed-rank for hybrid-vs-best-single-rep

## 5.2 Molecular property prediction

- **5.2.1** ADME regression: LogD, Caco-2, AdmeT-HLM/RLM/MDR1/Solubility (Polaris ADME-Fang) (`src/kirby/datasets/`)
- **5.2.2** Quantum chemistry: QM9 HOMO-LUMO (10K, 5-fold scaffold)
- **5.2.3** hERG classification: Lhasa/TDC, 10K molecules, binary (`herg.py`)
- **5.2.4** Inhibition: COX-2, DRD2 from ChEMBL
- Headline result format: table per dataset, primary metric, best single rep vs best hybrid, Δ
- Source datasets verified in agent inventory; results pending Phase 5 rerun

## 5.3 Antibody developability and binding

- **5.3.1** Intra-family DMS: AbAgym (HER2, VEGF, EGFR, Lysozyme, Ang2) (`abagym.py`)
- **5.3.2** Inter-family binding/expression: FLAb sub-datasets (Koenig, Phillips, Adams variants) (`flab.py`)
- **5.3.3** Binding affinity: Ab-CoV spike binding (`abcov.py`); SAbDab CDR + ANARCI (`sabdab.py`)
- **5.3.4** ΔΔG: AB-Bind + SKEMPI 2.0 (`abbind.py`)
- **5.3.5** Classification: Trastuzumab-HER2 binding (`trastuzumab.py`) — discussed in Ch 7 as OPIG collaboration
- Region ablation results (`antibody_region_ablation.py`): CDR vs FR contribution

## 5.4 Drug–target affinity

- **5.4.1** Davis, KIBA, BindingDB_Kd (`dta.py`)
- **5.4.2** Cold-target evaluation protocol (`--cold-setting drug`)
- **5.4.3** SOTA CI gap analysis
- **5.4.4** Sort & Slice on the drug side: `drug_sns` factory with per-fold fitting on training-set drugs

## 5.5 TCGA multi-omics

- **5.5.1** Tasks: LUAD vs LUSC binary, PAM50 multiclass, survival regression
- **5.5.2** 57 representations across 5 modalities (mutations, gene expression, CNV, DNAm, immune deconvolution, WSI)
- **5.5.3** PanCanAtlas pre-computed reps integrated (CIBERSORT, xCell, mRNAsi, MATH, HRD, MSI, CIMP)
- **5.5.4** WSI: Titan + Uni2H wired; Virchow2, CONCH deferred (tile extraction overhead)

## 5.6 Reaction yield prediction

- **5.6.1** USPTO + HTE (Doyle plates)
- **5.6.2** Reaction-group leave-one-out splitting (not random 70/30)
- **5.6.3** RxnFP, CGR, one-hot components — survey status: implemented, Phase 5 pending

## 5.7 Synthesis

- Cross-domain table: KIRBy hybrid Δ over best single rep, per domain
- Where does the hybrid help most? (Foreshadow Ch 6 diagnostics)
- Where doesn't it? (Foreshadow Ch 6 §6.3)

---

# Chapter 6 — When and Why Hybrids Work: Complexity, Similarity, and Noise Diagnostics

*Target length: ≈ 35–45 pages. The diagnostic chapter — the "interpretability and understanding" companion to Ch 5's "does it work?".*

**Preamble** — *"All diagnostic experiments were designed, implemented, and run by the author. Roadmaps `NOISE_EXPERIMENT_ROADMAP.md` and the complexity/similarity protocols in `tests/experiments/{complexity,similarity,noise}/` document the design. Manuscript in preparation."*

## 6.1 Data complexity diagnostics (C-series)

`tests/experiments/complexity/c01_*.py` through `c24_*.py`. Subset selected for the chapter:

- **6.1.1 C01 Rep characterisation** — bias-variance-complexity profile per rep
- **6.1.2 C02 Complexity grid** — optimal model–budget pairing across capped/uncapped/top-tier conditions (the core diagnostic; bug fix documented in `EXPERIMENT_BUGS.md`)
- **6.1.3 C03 Blueprint transfer** — does an allocation found on dataset A transfer to dataset B?
- **6.1.4 C05 Learning curves** — sample efficiency
- **6.1.5 C07 Scaffold-partitioned complexity** — optimal allocation varies by subpopulation
- **6.1.6 C09 Formal analysis** — ANOVA decomposition, information retention, effective rank, bias-variance
- **6.1.7 C12 Modality ablation** — TCGA omics combinations (which modalities help?)
- **6.1.8 C16 Feature selection as regularisation** — hybrid budget vs model-level regularisation
- **6.1.9 C18 Diversity vs depth** — number of reps vs quality of each

## 6.2 Similarity structure diagnostics (S-series)

`tests/experiments/similarity/s01_*.py` through `s13_*.py`. Subset:

- **6.2.1 S01 Similarity baselines** — Tanimoto, Euclidean, cosine per rep
- **6.2.2 S02 Hybrid similarity** — is the hybrid's similarity structure better-conditioned than its components'?
- **6.2.3 S04 Activity cliffs** — SAR discontinuities under different rep metrics
- **6.2.4 S05 kNN as rep quality proxy**
- **6.2.5 S06 Applicability domain** — convex hull / Lp-norm definitions
- **6.2.6 S07 Noise bridge** — connects S- and N-series (similarity assumptions underpin LNL methods)
- **6.2.7 S08 Intrinsic dimensionality**
- **6.2.8 S12 Cross-domain cliffs** — generalisation of cliff phenomena across molecular + antibody + TCGA

## 6.3 Noise diagnostics (N-series)

`tests/experiments/noise/n01_*.py` through `n13_*.py`. This complements Ch 3 by examining noise *in real data*, not injected.

- **6.3.1 N01 Activity-cliff framework** — Dablander/SALI landscape
- **6.3.2 N02 Sample weighting** — importance-weighted samples (cliff-aware)
- **6.3.3 N04 GP-LOO detection** — leave-one-out GP residuals as a noise signal
- **6.3.4 N05 Noise-injection benchmark** — 15 binary noise detectors across σ levels
- **6.3.5 N06 BindingDB ground truth** — replicate disagreement as ground-truth noise (real data!)
- **6.3.6 N07 Remediation** — 6 post-detection strategies
- **6.3.7 N08 Estimation survey** — 15 magnitude estimators, cliff discrimination
- **6.3.8 N09 LNL survey** — 12 Learning-from-Noisy-Labels methods (DivideMix etc.)
- **6.3.9 N10 Hybrid neighbourhood LNL** — using KIRBy hybrid distance inside LNL similarity computations
- **6.3.10 N11 Estimation × remediation grid** — cross-product of 15 × 6
- **6.3.11 N12 Importance-weighted distance** — feature-importance weighting in distance metrics

## 6.4 Synthesis: a predictive model of when KIRBy helps

- Pull C/S/N findings together into a predictor of hybrid improvement magnitude
- Cross-domain summary (C14) and cross-domain cliffs (S12)
- Where hybrids do not help: single-modality datasets, very small n, dominant-rep regimes — backed by C18, C16

---

# Chapter 7 — Forward Compatibility and OPIG Collaboration

*Target length: ≈ 20–30 pages. Demonstrates framework generality across newly released foundation models and new datasets contributed by collaborators.*

**Preamble** — *"This chapter integrates work contributed by, or in collaboration with, members of OPIG. Specific contributions are itemised below. Datasets B1–B5 were identified through OPIG group meetings; the Imrie pre-trained DNN approach (C4) is a re-implementation of Fergus Imrie's prior published method (DPhil thesis, 2021) extended to evaluate whether transfer learning changes representation ranking under KIRBy. All implementation, integration, and analysis was carried out by the author. Roadmap: `OPIG_COLLAB_ROADMAP.md`."*

## 7.1 Forward compatibility as a thesis claim

- "Plug a new foundation model in, get a hybrid out" — zero code change required
- The validation question: do new SOTA reps actually help KIRBy hybrids, or are they redundant with existing reps?

## 7.2 New foundation models tested

Phase 3+4 additions, all wired into the rep factory:

- **7.2.1 MINT** (antibody language model)
- **7.2.2 SaProt** (protein + antibody, 1280-D via ProstT5 3Di tokens)
- **7.2.3 ESM-3** (protein + antibody, 1536-D)
- **7.2.4 ESM-C LLR** (3-D log-likelihood ratio rep)
- **7.2.5 Boltz-2** (protein/DTA only, 384-D CLI wrapper)
- **7.2.6 Ab-RoBERTa** (`abroberta`, `abroberta_cdr`, 1536-D)
- **7.2.7 ESM-2 layer + concat + CDR3-only variants**
- **7.2.8 AntiBERTy LLR**
- Source: `src/kirby/representations/antibody.py`, `protein.py`; corresponding C13 forward-compatibility experiment

## 7.3 OPIG collaboration: new datasets (Category B)

- **7.3.1 B1 Trastuzumab-HER2 binding** — first antibody classification task in KIRBy. 524K → 5K stratified subsample. `trastuzumab.py`. Initial result: BLOSUM RF 84.9% / one-hot RF 84.9% (binary, 5K, 5-fold).
- **7.3.2 B2 Graphinity ΔΔG** — synthetic FoldX/Rosetta ΔΔG variants from Zenodo 15384945. `graphinity.py`. ≈ 943K → 50K subsample.
- **7.3.3 B3 Humanness** — binary human/non-human classifier. `species` preset from OAS FASTA (Zenodo 10562968); `therapeutic` preset from IMGT/mAb-DB via BioPhi. `humanness.py`.
- **7.3.4 B5 Ginkgo developability** — 9 regression properties from GDPa1 (242 antibodies). Pre-computed 5-fold hierarchical cluster splits. `ginkgo_dev.py`. Verified end-to-end (ginkgo_hic ridge R² = 0.08 / ρ = 0.39).

## 7.4 OPIG collaboration: methodological items (Category C)

- **7.4.1 C3 DDG expansion** — SKEMPI 2.0 full + AB-Bind merged (1536 samples)
- **7.4.2 C4 Imrie transfer learning** — does pre-training on ChEMBL change rep ranking? Self-contained comparison (not a core pipeline change). Bulk ChEMBL IC50 loader for top-200 targets (`chembl_bioactivity.py`). Sklearn-compatible DNN with pre-trained weights + freeze (`dnn_pretrained.py`). 5 conditions × 6 reps × 3 datasets. SLURM scripts: `imrie_pretrain.sh`, `imrie_eval.sh`. *Status: implemented; analysis underway.*
- **7.4.3 C5 ChEMBL QSAR activity-cliff stratification** — `opig_activity_cliff_qsar.py`
- **7.4.4 C6 King-Smith component splits** — leave-one-component-out on BH reaction datasets (aryl_halide, ligand, base, additive). `opig_kingsmith_component_splits.py`. *Status: written, not yet run.*

## 7.5 OPIG collaboration: new representations (Category A)

- **7.5.1 A1 Sort & Slice cross-domain wiring** — `drug_sns` added to DTA baseline survey (factory rep with per-fold fitting on unique training drugs) and to hybrid search (40 reps in full preset). Reaction SNS *skipped* — fingerprints collapse under leave-ligand-out evaluation.
- **7.5.2 A2 CaLM (codon LM)** — investigated and deferred. Antibody datasets lack nucleotide data (PDB-derived only); DTA feasible but low ROI. Documented in `OPIG_COLLAB_ROADMAP.md`.

## 7.6 Cross-Pretraining: a meta-experiment

- **7.6.1 Pre-train the hybrid itself.** `PRETRAIN_REP_ROADMAP.md`. Pre-train the hybrid representation on larger related data (C4 corpus, Wikipedia) for small-n fine-tuning targets. `pretrain_rep_experiment.py`, `pretrain_rep_model_transfer.py`. *Status: code complete, awaiting analysis review.*

## 7.7 Discussion

- Does new ≠ better? Identify cases where newer foundation models are redundant with existing reps
- Plug-and-play validation: did any new rep require non-trivial framework changes? (Document in narrative)
- Collaboration model: how the OPIG group meetings shaped which datasets/reps to test

---

# Chapter 8 — Discussion

*Target length: ≈ 12–18 pages. Synthesis across Chapters 3–7 plus an honest accounting of limits.*

## 8.1 Cross-chapter findings

- Representation matters more than model architecture (Ch 3 ANOVA; Ch 5 cross-domain survey; Ch 6 C09 formal analysis)
- Matched evaluator–model selection eliminates a class of overfitting (Ch 4 F12+F13; Ch 6 C23)
- Hybrid help is *predictable*: small dominant-rep regimes don't benefit; multi-modal regimes do (Ch 6)
- Noise robustness is largely strategy-independent (Ch 3 Kendall's W = 0.92) — a model's NDS rank carries across noise types

## 8.2 Where KIRBy does not help

- Single-modality datasets where one rep dominates (C18)
- Very small n (n < 200): hybrid greedy underperforms because internal val split is too noisy (F10 small-dataset experiments)
- Datasets where representation diversity is low (multiple antibody LMs that all encode the same information)

## 8.3 Methodological limits

- Greedy allocation: heuristic, no optimality guarantee
- Importance methods disagree (Ch 4 §4.4, F05) — chapter-level results are conditioned on the importance method used
- Cold-target / scaffold splits are protective but not perfect; some leakage paths remain open (`issues_splitting.md`)
- Foundation models are released faster than they can be evaluated — the "frozen" state of this thesis is itself a snapshot

## 8.4 Practical recommendations

- For a practitioner with a new dataset: which checks to run first to know whether KIRBy will help
- Default settings: uncapped augmentation, step_size = 5, patience = 20 (`project_augmentation_decisions.md`); internal evaluator = kNN distance (`project_internal_evaluators.md`); downstream model set per domain

---

# Chapter 9 — Conclusions and Future Directions

*Target length: ≈ 6–10 pages. Dablander-style two-part: forward-looking subsection + reflective subsection.*

## 9.1 Summary of contributions

A consolidated bulleted list (Dablander §8 style):
1. NoiseInject framework: open-source benchmarking suite for noise robustness × UQ × QSAR
2. KIRBy: importance-weighted hybrid representation framework with matched evaluator–model selection
3. The largest reported cross-domain hybrid survey (molecular, antibody, DTA, TCGA, reaction): X datasets, Y reps, Z models
4. Diagnostic suite (C/S/N experiments) characterising *when* hybrids help
5. Forward-compatibility evidence across MINT, SaProt, ESM-C, Boltz-2 and OPIG-contributed datasets (Trastuzumab, Graphinity, Humanness, Ginkgo)
6. Federated extension validating KIRBy as a feature-fusion alternative to FLuID-style prediction-fusion

## 9.2 Future directions

- Trainable allocation (replace greedy with a learned policy)
- Bayesian-optimised allocation budgets per dataset
- Online / incremental hybrid updates as new foundation models release
- Per-sample hybrid blueprints rather than global ones (the "S07 noise bridge" thread)
- Theoretical work: when does importance-weighted greedy match optimal allocation?

## 9.3 Final remarks

Short, reflective closing tying back to §1.2's representation problem.

---

# Appendices

- **A** — Full Bill of Materials: every (dataset, rep, model, fold, metric) tuple from Ch 5 surveys. Generated from `BILL_OF_MATERIALS.md`.
- **B** — NoiseInject supplementary results: per-noise-strategy tables, full ANOVA tables, calibration curves. Generated from `qsar_qm_models/results/`.
- **C** — KIRBy framework experiment tables: F00–F16 with hyperparameter sweeps.
- **D** — Complexity/Similarity/Noise full results: experiments C01–C24, S01–S13, N01–N13 in full.
- **E** — Splitting protocols per domain — detailed audit from `issues_splitting.md`.
- **F** — Dataset provenance, licensing, and preprocessing — Phase 5 status from `PIPELINE_READINESS_TODO.md`.
- **G** — Statement of Authorship forms for any co-authored papers presented as integrated-thesis chapters (per MPLS integrated-thesis requirement).

---

# Items deliberately excluded

- **kirby-av** (AV2 2026 Scenario Mining Challenge), **kirby-finance** (options + mean-reversion trading), **kirby-rogii** (wellbore Kaggle), **ANNITIA Trustii MASLD challenge**. Out of scope by candidate decision: these are independent applications of the framework outside the drug-discovery thesis arc. They will not be referenced in the thesis.

# Verification notes

- Transfer-of-status report at `~/Downloads/Transfer_of_Status-1 2022-08-19 07_21_19.pdf` lists **Leo J. Klarner** as author, not the candidate. It was not used as a source for this outline. If a different transfer report exists, supply the path and the historical framing in Ch 1 / Ch 2 can be revisited.
- Oxford/MPLS word count for Statistics DPhil — public-facing MPLS page defers to departmental handbook; the GSO.20a primary returned 403 during research. Verify with the Department of Statistics graduate handbook before drafting.
- OPIG peer thesis TOCs (Imrie 2021, Boyles 2020, Schneider 2021, etc.) could not be retrieved from ORA's HTML landing pages — only abstracts. Their TOCs would refine the per-chapter target lengths in this outline. Available via PDF download from ORA.

# Sources used (all verified file paths)

- `/Users/apunt/repos/qsar_qm_models/paper.tex`, `pdv_case_study_draft.tex`, `README.md`, `paper_edits_supervisor.md`, `scripts/pdv_investigation/REPORT.md`
- `/Users/apunt/repos/KIRBy/` — `ISSUES.md`, `OPIG_COLLAB_ROADMAP.md`, `NOISE_EXPERIMENT_ROADMAP.md`, `FEDERATED_EXPERIMENT_ROADMAP.md`, `CROSS_SPECIES_ROADMAP.md`, `PRETRAIN_REP_ROADMAP.md`, `BILL_OF_MATERIALS.md`, `PIPELINE_READINESS_TODO.md`, `EXPERIMENT_BUGS.md`
- `/Users/apunt/repos/KIRBy/src/kirby/representations/` (8 files)
- `/Users/apunt/repos/KIRBy/src/kirby/datasets/` (31+ files)
- `/Users/apunt/repos/KIRBy/src/kirby/find_best_hybrid.py`
- `/Users/apunt/repos/KIRBy/tests/experiments/{complexity,similarity,noise,framework,federated,opig,pretrain_rep,cross_species}/`
- Markus Dablander DPhil thesis (Oxford, 2023) — `~/Downloads/DPhil_Thesis_Markus_Dablander--Submitted.pdf`
- Marc Moesser DPhil thesis (Oxford, 2022) — `~/Downloads/PhD_Thesis_Marc_Moesser.pdf`
- MPLS integrated-thesis guidance and OPIG theses index (URLs in §0 above)
