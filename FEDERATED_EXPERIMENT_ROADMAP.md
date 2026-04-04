# Federated Learning Experiment Roadmap

**Created:** 2026-03-26
**Updated:** 2026-03-27 (incorporated deep FLuID codebase + publication analysis)
**Context:** Extending FLuID (Lhasa Limited) federated framework into KIRBy as its own experiment category. Funded in part by Lhasa — more work on their pipeline is valuable.

**Experiment categories (5 total):** complexity, similarity, noise, OPIG, **federated**

---

## Background: What FLuID Actually Is

### Published System (Nature Machine Intelligence, March 2025)

FLuID = **Federated Learning Using Information Distillation**. It is a knowledge distillation approach, NOT traditional federated learning (no gradient sharing). Published in *Nature Machine Intelligence* 7, 423–436 (2025) by Hanser, Ahlberg, Amberg et al.

**Published pipeline:**
1. **Teachers**: Each pharma company trains a local model on private data
2. **Public pool**: Lhasa curates ~300K unlabeled public molecules as a shared dataset
3. **Annotation**: Each teacher predicts labels on the public pool
4. **Aggregation**: Lhasa (honest broker) consolidates predictions → filters by "Reliability" (≥0.005) and "Decidability" (confidence) → top 10K positive + 10K negative → balanced 20K federated training set
5. **Student**: Trained on the 20K federated labels (MLP or SOHN architecture)
6. **Hybrid refinement**: Student optionally fine-tuned on each partner's private data

**Published results (Effiris hackathon, 7 companies, 5 endpoints):**

| Endpoint | Public set size | Federated MCC | Best internal MCC | Winner |
|----------|----------------|---------------|-------------------|--------|
| hERG | 8,298 | 0.55 (MLP refined) | 0.57 (RF) | Internal |
| GABAA | 2,433 | 0.60 (MLP refined) | 0.68 (SVC) | Internal |
| COX2 | 3,228 | 0.29 (MLP refined) | 0.30 (XGBoost) | Tied |
| AChM1R | 2,191 | 0.56 (MLP refined) | 0.54 (RF) | Federated |
| 5-HT2B | 1,506 | 0.32 (MLP/SOHN) | 0.32 (XGBoost) | Tied |

**Key published finding:** Student models generally **underperformed** internal teacher models on MCC, but expanded the **applicability domain by 15-30%**. The paper explicitly notes: "predictions rather than experimental labels introduce another level of uncertainty."

**Implication for us:** This is exactly the gap KIRBy can address. If feature-level fusion (sharing feature indices/weights) avoids the prediction noise that hurt FLuID's student models, that's a publishable result directly extending their Nature MI paper.

### FLuID_POC Codebase (what we have locally)

The code at `/Users/apunt/repos/lhasa/FLuID_POC/preprocessing/FLuID.py` (1,974 lines) implements an **earlier, simulation-based version** of FLuID that differs from the published system:

| Aspect | FLuID_POC Code | Published System |
|--------|----------------|------------------|
| **Teacher creation** | K-Means on 2048-bit Morgan FPs → K teachers from SAME dataset | Each pharma company IS a teacher (real silos) |
| **Smooth factor** | 5% random reassignment of cluster boundaries | Not mentioned in papers |
| **Teacher weighting** | Similarity-based: 8-NN distances → quadratic decay → normalized per sample | Filtered by Reliability/Decidability, no per-sample weighting |
| **Federation** | Weighted average of teacher probability distributions | Top 10K pos + 10K neg by confidence → balanced set |
| **Student training** | Trains on federated labels, balanced/random/best/worst selection | MLP pretrained on public → refined on member data |
| **Class balancing** | Random Under-Sampling (RUS) | Imbalanced federated sets noted as problem |
| **Supported models** | RF (150 trees, depth 15), NB, SVM, XGBoost, KNN, GP, DT, MLP, AdaBoost | RF, SVM at partners; MLP/SOHN at Lhasa |
| **Metrics** | BAC, MCC, ACC, F1, SENS, SPEC, PPV, NPV, AUC | BAC, MCC, AUC, sensitivity, specificity |

**Data format expected by FLuID_POC:**
- `MOLECULE` column: RDKit Mol objects
- `FP` column: 2048-bit Morgan fingerprint arrays (radius=2, useFeatures=True)
- `CLASS` column: 0 (Inactive) or 1 (Active)
- `ACTIVITY` column: 'Active' or 'Inactive'
- `INCHI` column: InChI Key
- `SOURCE` column: data source name
- `CLUSTER` column: assigned by `cluster_data_space()` (1 to K)

**Key functions we will wrap:**
- `cluster_data_space(data, name, prefix, k, smooth_factor, params)` — K-Means clustering + boundary smoothing
- `build_teacher_models(teacher_data, params)` — trains K+1 classifiers (one per cluster + full dataset)
- `create_trained_classifier(algorithm, data, balance=True)` — RF/SVM/etc. with optional RUS
- `annotate_transfer_data(transfer, teachers, teacher_data, params)` — 8-NN similarity weighting per teacher
- `federate_labels(federated_data, teachers)` — weighted probability averaging across teachers
- `compute_validation_row(table, y_true, y_pred, title, size)` — BAC/MCC/AUC metrics

---

## Overview

FLuID clusters molecules by fingerprint similarity into K "teachers", trains independent models per cluster, then ensembles predictions. KIRBy combines multiple representations via importance-weighted feature selection. These are complementary:

- **FLuID** = prediction-level fusion (ensemble teachers)
- **KIRBy** = feature-level fusion (hybrid reps)

The experiments below test where each approach wins, how noise heterogeneity affects them, and how they can be combined. The strongest publication angle is **"Federated Feature Selection"** — KIRBy's importance allocation as a privacy-preserving alternative to sharing predictions that avoids the noise amplification problem documented in Lhasa's own Nature MI paper.

---

## Phase F1: Heterogeneous Noise Across Silos + Federated Noise Detection

*Merged from original F1 and F5 — noise heterogeneity and noise detection are the same experiment with different analysis focuses.*

### Question
When different data silos have different noise levels, does feature-level fusion (KIRBy) handle it better than prediction-level fusion (FLuID)? Can KIRBy's noise detectors identify which silo is noisy without seeing raw data?

### Design

**Dataset:** hERG (binary classification, already in both codebases — 8,298 public compounds in Lhasa's set)

**Setup:**
1. Use FLuID's `cluster_data_space()` to split training data into K=5 clusters (K-means on 2048-bit Morgan FPs, radius=2)
2. Assign different noise levels per cluster:
   - Teacher 0 (full dataset): no injection (natural noise only)
   - Teacher 1: clean (0% injected)
   - Teacher 2: 5% label flip
   - Teacher 3: 10% label flip
   - Teacher 4: 20% label flip
   - Teacher 5: 30% label flip

**Noise types to test (using KIRBy's `_inject_noise()` pipeline):**
- Label flip noise (classification: swap active/inactive)
- Gaussian noise on continuous IC50 values (then re-threshold)
- Systematic bias (shift all IC50 values by a factor per silo, simulating different assay protocols — this mirrors the hackathon's real problem where "predictions rather than experimental labels introduce uncertainty")

**Conditions compared:**

| Condition | How it works | What it tests |
|-----------|-------------|---------------|
| Naive pool | Merge all silo data, train single model | Baseline — noise contaminates everything |
| FLuID teacher ensemble (equal) | One teacher per cluster, weighted by 8-NN similarity (code default) | Prediction-level fusion with similarity weighting |
| FLuID teacher ensemble (quality) | Weight teachers by cross-val MCC | Prediction-level fusion, quality-aware |
| FLuID → student (published pipeline) | Federate labels via confidence filtering → train student on top 10K/10K | Replicates published approach |
| KIRBy hybrid (per-cluster reps) | Each cluster's features treated as a separate rep → importance allocation | Feature-level fusion, implicit noise detection |
| KIRBy hybrid + N05 detectors | Run all 15 N05 noise detectors on hybrid features, downweight flagged samples | Combined approach |

**Noise detection analysis (integrated from original F5):**
- Run KIRBy's 15 N05 detectors (knn_k*, lof, distance_weighted, mahalanobis, cv_disagreement, bootstrap_ensemble, co_teaching, dividemix, zscore, prediction_error) on:
  - Full pooled data (centralized — unrealistic but ideal baseline)
  - Per-silo predictions only (realistic federated)
  - KIRBy hybrid features (per-silo feature contributions)
- Test whether KIRBy importance scores correlate with silo noise level (importance_silo_i vs injected_noise_silo_i)
- Measure noise detection precision/recall per detector in centralized vs federated settings

**Metrics:** BAC, MCC, AUROC, F1, plus noise detection precision/recall/F1

**Extension — noise injection sweep:**
For each condition, sweep noise fraction [0, 0.05, 0.1, 0.15, 0.2, 0.3] on the noisiest silo while keeping others clean. This produces a degradation curve showing each method's robustness.

**Extension — smooth_factor interaction:**
Sweep smooth_factor [0, 0.01, 0.05, 0.1, 0.2] across noise levels. Hypothesis: boundary mixing acts as regularization — makes ensemble MORE robust to noise at cluster boundaries but LESS able to isolate noisy clusters. This is a unique variable only in the code (not published).

### What this shows for Lhasa
- Whether FLuID's teacher ensemble naturally resists noisy silos (published results suggest it doesn't — student underperformed)
- Whether quality-weighted ensembling or the 8-NN similarity weighting (from code) recovers signal
- Whether KIRBy's importance allocation implicitly detects and down-weights noisy data — if so, this directly addresses the prediction noise problem from the Nature MI paper
- Whether the 15 N05 detectors can identify noisy silos from hybrid features alone (privacy-preserving noise detection)
- Practical guidance: when should a federated pipeline use FLuID-style ensembling vs KIRBy-style feature fusion?

### Comparison to published results
The hackathon achieved MCC 0.55 for federated hERG (vs 0.57 internal). If KIRBy hybrid achieves higher MCC under heterogeneous noise, that's a direct improvement over the published system.

---

## Phase F2: Representation Diversity as Federated Silos

### Question
When different organizations contribute different molecular representations (not different data), does KIRBy's feature-level hybrid outperform FLuID's prediction-level ensemble?

### Design

**Dataset:** hERG (primary), extend to logd, caco2 (regression)

**Setup:**
Each "silo" has the same molecules but a different representation:
- Silo A: ECFP4 (structural fingerprints, 2048D)
- Silo B: Mordred (physicochemical descriptors, ~1600D)
- Silo C: MolFormer (foundation model embeddings, 768D)
- Silo D: PDV (RDKit descriptors, 200D)
- Silo E: ChemBERTa (learned SMILES embeddings, 768D)

**Conditions compared:**

| Condition | How | What it tests |
|-----------|-----|---------------|
| Best single silo | Best individual rep (standard baseline survey) | Single-rep baseline |
| FLuID teacher ensemble (equal) | One teacher per silo, 8-NN similarity weighting | Prediction-level fusion with code's weighting |
| FLuID teacher ensemble (quality) | Weight by silo-specific cross-val MCC | Prediction-level fusion, quality-aware |
| KIRBy hybrid | Feature-level fusion across all silo reps | Standard KIRBy allocation |
| KIRBy top-tier hybrid | Feature-level fusion on top 3 reps only | Budget-constrained KIRBy |
| FLuID → KIRBy pipeline | FLuID clustering WITHIN each silo's rep space, then KIRBy across silos | Novel: intra-silo structure + inter-silo fusion |

**Key insight:** FLuID was designed for silos with different DATA (different molecules), not different REPRESENTATIONS of the same molecules. When silos differ by representation, prediction-level averaging makes less sense because each teacher sees the same molecules from a different lens. KIRBy's feature-level fusion is the natural approach here. The FLuID → KIRBy pipeline condition is genuinely novel — it uses FLuID's clustering to find structure WITHIN each representation, then KIRBy's allocation to combine ACROSS representations.

**Publication framing:** In real federated drug discovery, organizations often have different computational chemistry pipelines producing different molecular descriptors. They can't share raw data but could share model predictions (FLuID) or selected features (KIRBy). This experiment directly tests which produces better downstream models.

### Regression extension
For logd/caco2 (regression tasks):
- FLuID's teacher ensemble → mean of teacher predictions (not majority vote)
- FLuID's weighted ensemble → quality-weighted average by per-silo cross-val RMSE
- Requires regression adaptation in `federated_helpers.py`: replace `create_trained_classifier()` with regressor, replace `federate_labels()` probability averaging with weighted mean of continuous predictions, replace BAC/MCC with RMSE/Spearman

---

## Phase F3: Synthetic Private Datasets

*Lead experiment: Strategy 2 (source-based hERG) — most aligned with Lhasa's published hackathon setup.*

### Question
Can we simulate realistic federated drug discovery scenarios where each organization has partial, overlapping datasets?

### Design

**Creating synthetic private datasets from public data:**

Take a large public molecular dataset and partition it to simulate organizations with different:
- **Chemical space coverage**: cluster by scaffold, assign clusters to orgs
- **Data volume**: org A has 10K compounds, org B has 500
- **Label quality**: org A has IC50 values (precise), org B has active/inactive (binary)
- **Assay protocol**: org A's threshold is 1 μM, org B's is 10 μM
- **Temporal splits**: org A has compounds from 2010-2020, org B from 2020-2025

**Source datasets to partition:**

| Source | Size | Task | Why |
|--------|------|------|-----|
| ChEMBL hERG | ~30K | clf | Large, classification, already in KIRBy + Lhasa's hackathon endpoint |
| LogD (OpenADMET) | ~7K | reg | Regression, good KIRBy results |
| Caco2 (OpenADMET) | ~3.7K | reg | Regression, moderate size |
| ADME HLM/RLM | ~3K | reg | ADME endpoints, pharma-relevant |
| Buchwald-Hartwig | ~4K | reg | Reaction yields, very different domain |

**Partition strategies:**

**Strategy 1: Scaffold-based silos**
- Run Murcko scaffold decomposition
- K-means on scaffold fingerprints → K silos
- Each silo has structurally distinct compounds
- Tests: can federated methods generalize across chemical series?

**Strategy 2: Source-based silos (for hERG) — LEAD EXPERIMENT**
- Silo A: ChEMBL hERG data (large, noisy, heterogeneous)
- Silo B: Lhasa FLuID data (curated, smaller, higher quality)
- Silo C: TDC hERG data (medium, pre-processed)
- This uses REAL data source heterogeneity, not synthetic
- Directly mirrors the hackathon setup (7 companies with different data quality/volume)
- **Published comparison point:** hackathon federated hERG MCC = 0.55, best internal = 0.57
- The published paper found federated sets can be imbalanced (e.g., GABAA: 14,915 neg / 5,085 pos from insufficient confident positive predictions). KIRBy's importance allocation may handle this better than FLuID's simple top-N selection.

**Strategy 3: Temporal silos**
- Split by publication year (available in ChEMBL metadata)
- Silo 1: pre-2015, Silo 2: 2015-2020, Silo 3: 2020+
- Tests: does knowledge from older data help newer predictions?

**Strategy 4: Asymmetric information — STRONGEST PUBLICATION ANGLE**
- Silo A: has structural reps (ECFP4, graph kernel)
- Silo B: has physicochemical reps (Mordred, PDV)
- Silo C: has learned reps (MolFormer, ChemBERTa)
- Each silo contributes what it can compute — mirrors real organizations
- Directly addresses the published paper's limitation: "predictions rather than experimental labels introduce uncertainty." Sharing feature indices and weights (KIRBy's allocation) avoids prediction noise entirely. This is genuinely novel.

### How FLuID + KIRBy work together here
1. FLuID clusters WITHIN each silo (intra-org structure)
2. KIRBy combines ACROSS silos (inter-org fusion)
3. Noise levels vary per silo (realistic quality heterogeneity)
4. Importance allocation reveals which silos/reps contribute most

---

## Phase F4: Cluster-Aware Importance

### Question
Does the optimal KIRBy allocation vary across chemical space regions? Do activity cliff regions need different reps than smooth SAR regions?

### Design

1. Use FLuID's `cluster_data_space()` with K-means on hERG (K=5, Morgan FPs)
2. For each cluster, run KIRBy importance independently
3. Compare per-cluster importance profiles:
   - Do structural FPs (ECFP4) dominate near activity cliffs?
   - Do foundation model embeddings dominate in smooth regions?
   - Does optimal budget change by cluster?
4. Build cluster-conditional hybrid: allocate differently per chemical space region

**Smooth factor sweep:**
Test smooth_factor [0, 0.01, 0.05, 0.1, 0.2] and measure whether cluster-conditional importance profiles become more or less stable. Hypothesis: higher smooth_factor → more homogeneous importance profiles across clusters (boundary mixing reduces cluster specificity).

**Connection to existing experiments:**
- Extends C07 (data complexity partitions) with FLuID's clustering
- Extends N01 (activity cliff framework) with per-cluster analysis
- Extends S04 (activity cliffs) with cluster-aware similarity

---

## Phase F6: Antibody Domain Extension (Stretch Goal)

*Deprioritized. FLuID is built around Morgan fingerprints and K-Means on small molecules. Antibody data doesn't have Morgan FPs — would require replacing the clustering entirely (e.g., K-Means on ESM2 embeddings or sequence edit distance). The "natural silos" framing is really just dataset categories, not federated silos in FLuID's sense.*

### Question
Does the federated framework apply to antibody property prediction?

### Design

Antibody data doesn't come from multiple organizations in the same way as small molecules, but there IS natural data heterogeneity:

**Silo structure for antibodies:**
- Silo A: DMS data (abagym — single-family, mutation-focused)
- Silo B: Inter-family data (FLAb — diverse antibodies, developability properties)
- Silo C: Clinical data (Trastuzumab, humanness — therapeutic candidates)
- Silo D: Synthetic data (Graphinity DDG — computational predictions)

Each silo has different:
- Sequence diversity (single family vs diverse)
- Property focus (fitness vs binding vs stability)
- Data quality (experimental vs computational)
- Available representations (mutation-aware vs full-sequence LMs)

**Adaptation required:** Replace FLuID's Morgan FP K-Means with clustering on ESM2 embeddings or CDR sequence edit distance. This is a significant departure from the core FLuID architecture and may not be worth the effort unless F1-F4 produce strong results.

**What this tests:**
- Does hybrid across DMS + inter-family data help either domain?
- Can KIRBy's importance allocation discover that mutation_features matter for DMS but not inter-family?
- Does adding synthetic ΔΔG data (Graphinity) help or hurt experimental predictions?

---

## Implementation Order

| Phase | Priority | Blocked on | Est. experiments | Rationale |
|-------|----------|-----------|-----------------|-----------|
| **F1** (noise + detection) | HIGH — core Lhasa value | Nothing | 1 experiment, ~6 conditions × noise sweep × smooth_factor sweep | Merged F1+F5. Directly addresses Nature MI paper's noise limitation. |
| **F2** (rep diversity silos) | HIGH — core KIRBy thesis | Nothing | 1 experiment, ~6 conditions × 3 datasets | FLuID → KIRBy pipeline is genuinely novel. |
| **F3** (synthetic private data) | HIGH — Strategy 2 is lead | F1, F2 working | 3-4 experiments (one per strategy) | Strategy 2 (source-based hERG) directly comparable to hackathon results. Strategy 4 strongest for publication. |
| **F4** (cluster-aware importance) | MEDIUM — connects to existing | F1 | 1 experiment | Interesting but incremental over C07. |
| **F6** (antibody extension) | LOW — stretch goal | F1-F4 | 1-2 experiments | Requires major FLuID adaptation. |

**Total:** 7-9 experiments (down from 8-11 after F1+F5 merge and F6 deprioritization).

---

## File structure

```
tests/experiments/
  federated/                        # 5th experiment category (alongside complexity/
                                    #   similarity/noise/opig)
    federated_helpers.py            # FLuID bridge (see API spec below)
    fed01_noise_heterogeneity.py    # Phase F1 (includes noise detection from old F5)
    fed02_rep_diversity_silos.py    # Phase F2
    fed03_synthetic_private.py      # Phase F3 (one file, multiple partition strategies)
    fed04_cluster_aware_importance.py  # Phase F4
    fed05_antibody_federated.py     # Phase F5 (antibody extension)
    herg.py                         # Per-dataset wrapper (same pattern as other categories)
    logd.py                         # Per-dataset wrapper
    caco2.py                        # Per-dataset wrapper
    ...

tests/slurm_scripts/
    submit_federated.sh             # SLURM submission for federated experiments
```

**Naming convention:** `fed01`-`fed05` (not `f1`-`f5`) to avoid collision with framework experiments `f00`-`f13`.

---

## federated_helpers.py — API Specification

### Format Bridge

KIRBy and FLuID use different data formats. The bridge must convert between them:

```
KIRBy world:  SMILES list → rep_dict[rep_name] = numpy array (n_samples × d_features)
FLuID world:  DataFrame with columns [MOLECULE (Mol), FP (2048-bit Morgan), CLASS (0/1),
              ACTIVITY (Active/Inactive), INCHI, SOURCE, CLUSTER, CLUSTER_ID]
```

### Core Functions

```python
# === Format Bridge ===

def smiles_to_fluid_df(smiles_list, labels, source_name="KIRBy"):
    """Convert KIRBy SMILES + labels → FLuID-format DataFrame.
    Computes Morgan FPs (radius=2, 2048-bit, useFeatures=True),
    RDKit Mol objects, InChI keys. Returns DataFrame with all
    required FLuID columns."""

def fluid_clusters_to_indices(fluid_df):
    """Extract cluster assignments from FLuID DataFrame → dict of
    {cluster_id: numpy array of sample indices}. Maps back to
    KIRBy's index space."""

# === FLuID Wrappers ===

def cluster_data_space(smiles_list, labels, k=5, smooth_factor=0.05, seed=42):
    """Bridge: SMILES → FLuID DataFrame → cluster_data_space() → silo indices.
    Returns (silo_indices_dict, fluid_df) where silo_indices_dict maps
    cluster_id → numpy array of row indices."""

def build_teacher_ensemble(smiles_list, labels, silo_indices, algorithm='rf'):
    """Train one FLuID classifier per silo. Returns list of
    (classifier, silo_id, training_size) tuples."""

def predict_teacher_ensemble(teachers, smiles_list, weighting='similarity'):
    """Get ensemble predictions from all teachers.
    weighting='similarity': 8-NN distance weighting (code default)
    weighting='quality': weight by cross-val MCC
    weighting='equal': uniform weights
    Returns (predictions, teacher_weights_per_sample)."""

def federate_labels_published(teachers, smiles_list, n_per_class=10000):
    """Replicate published pipeline: filter by confidence, take
    top n_per_class per class → balanced federated set.
    Returns (federated_indices, federated_labels)."""

# === Noise Per Silo ===

def inject_noise_per_silo(labels, silo_indices, noise_levels_dict, noise_type='label_flip'):
    """Heterogeneous noise injection across silos.
    noise_levels_dict: {silo_id: noise_fraction}
    noise_type: 'label_flip' | 'gaussian' | 'systematic_bias'
    Returns (labels_noisy, noise_mask_per_silo)."""

# === KIRBy Hybrid as Federated Fusion ===

def create_silo_hybrid(rep_dict_per_silo, labels, silo_indices,
                       allocation_method='performance_weighted'):
    """Treat each silo's reps as a named rep in KIRBy's allocation.
    Returns (X_hybrid, silo_allocation, silo_importance)."""

def federated_noise_detection(X_hybrid, labels, noise_mask_true, detector_names=None):
    """Run N05 detectors on silo hybrid features.
    Returns per-detector: {detector: (precision, recall, f1)}."""

# === Partition Strategies (for F3) ===

def scaffold_partition(smiles_list, k=5):
    """Murcko scaffold decomposition → K-means on scaffold FPs → silo indices."""

def temporal_partition(dates, k=3):
    """Split by publication year → silo indices."""

def source_partition(source_labels):
    """Group by data source column → silo indices."""

def asymmetric_rep_partition(rep_dict, silo_rep_assignments):
    """Assign different reps to different silos.
    silo_rep_assignments: {silo_id: [rep_names]}
    Returns rep_dict_per_silo."""

# === Cluster-Aware Analysis (for F4) ===

def per_cluster_importance(rep_dict, labels, silo_indices, method='treeshap'):
    """Run KIRBy importance independently per FLuID cluster.
    Returns {cluster_id: importance_profile_dict}."""

# === Regression Adaptation ===

def build_teacher_ensemble_regression(smiles_list, labels, silo_indices, algorithm='rf'):
    """Regression variant: RF regressor per silo instead of classifier."""

def predict_teacher_ensemble_regression(teachers, smiles_list, weighting='quality'):
    """Regression: weighted mean of continuous predictions (not majority vote)."""
```

### Infrastructure Reuse

| Component | Source | Import path |
|-----------|--------|-------------|
| `create_hybrid()` | hybrid.py | `from kirby.hybrid import create_hybrid` |
| `apply_feature_selection()` | hybrid.py | `from kirby.hybrid import apply_feature_selection` |
| `compute_feature_importance()` | hybrid.py | `from kirby.hybrid import compute_feature_importance` |
| `_inject_noise()` | n05_noise_injection_benchmark.py | local import |
| `get_mitigation_method()` | noise_mitigation_methods.py | local import |
| `load_data()` | _shared.py | `from tests.experiments._shared import load_data` |
| `cv_split()` | _shared.py | `from tests.experiments._shared import cv_split` |
| `get_domain_metric()` | _shared.py | `from tests.experiments._shared import get_domain_metric` |
| `DATASET_REPS` | complexity_helpers.py | `from tests.experiments.complexity.complexity_helpers import DATASET_REPS` |
| FLuID clustering | FLuID.py | `sys.path` insert for `/Users/apunt/repos/lhasa/FLuID_POC/preprocessing` |

---

## Connection to existing infrastructure

| Federated component | KIRBy equivalent | How they connect |
|---------------------|-----------------|------------------|
| FLuID `cluster_data_space()` (K-Means on Morgan FPs) | KIRBy scaffold groups (Murcko decomposition) | Both partition chemical space; FLuID by FP K-means, KIRBy by Murcko scaffold |
| FLuID teacher ensemble (8-NN similarity weighting) | KIRBy `find_best_hybrid` (importance allocation) | Prediction-level vs feature-level fusion |
| FLuID smooth_factor (5% boundary permutation) | No direct equivalent | Unique to code; acts as cluster regularization |
| FLuID class balancing (RUS) | KIRBy `get_model_factories` | Both handle imbalanced data |
| FLuID validation metrics (BAC, MCC, AUC) | KIRBy `get_domain_metric` | FLuID's metric table maps to KIRBy's domain-aware dispatch |
| FLuID Reliability/Decidability (published) | No direct equivalent | Published label filtering; could add as condition |
| FLuID student refinement on private data (published) | KIRBy per-fold hybrid creation | Both adapt global knowledge to local data |

---

## Open questions (with answers from research)

1. **K selection:** FLuID_POC defaults to K=8 (in params dict). The roadmap originally said K=5. The hackathon used 7 real companies. **Answer:** Sweep K=[3,5,7,10] as planned. Use K=5 as default for comparability with roadmap; K=7 for hackathon comparison.

2. **Smooth factor:** FLuID_POC uses 5% (0.05), NOT 10% as originally stated. It randomly reassigns `int(smooth_factor * n_samples)` molecules to random clusters. **Answer:** This is a regularization on cluster boundaries — unique to the code, not published. Sweep [0, 0.01, 0.05, 0.1, 0.2] in F1 to test noise interaction. Hypothesis: higher smooth_factor → more robust to noise at boundaries, less able to isolate noisy clusters.

3. **Regression adaptation:** FLuID_POC is classification-only (`predict_proba()`, `CLASS` column, RUS balancing). **Answer:** For regression datasets (logd, caco2), `federated_helpers.py` needs: replace `create_trained_classifier()` with `RandomForestRegressor`, replace `federate_labels()` probability averaging with weighted mean of continuous predictions, replace BAC/MCC with RMSE/Spearman. This is a meaningful code change.

4. **Cross-domain transfer:** Speculative (hERG + logd in same federation). **Answer:** Skip for now. Focus on within-endpoint federation first.

5. **Publication framing:** "Federated Feature Selection" — KIRBy's importance allocation as a privacy-preserving alternative to sharing predictions. **Answer:** This is the strongest angle. The published paper's explicit limitation is prediction noise. Sharing feature indices and weights avoids that entirely. Each silo computes importance locally, shares only feature indices and weights. This is genuinely novel and directly extends the Nature MI paper.

---

## Known Issues (from 2026-03-28 audit)

All federated experiments (fed01-fed05) were implemented in a single session and then audited. The following issues were found. Bugs marked [FIXED] were patched during the session. Issues marked [OPEN] need discussion before implementation.

### Splitting and Evaluation

**ISSUE S1 [FIXED]: All federated experiments use single 80/20 ShuffleSplit.**
Every experiment did one random 80/20 split with `StratifiedShuffleSplit` or `GroupShuffleSplit`, ignoring scaffold groups.
- **Fixed (2026-03-28):** All 5 experiments now use `cv_split(groups, 1, ...)` from `_shared.py` which is scaffold-group-aware. Fed01/fed04 also restructured to split-first-then-cluster (matching FLuID's pattern where training/test separation precedes clustering). Multi-fold CV (n_splits=3+) is a future enhancement for confidence intervals.
- **Scope:** fed01, fed03 (Part A + Part C), fed04, fed05 (C5 pooled binding).

**ISSUE S2 [CLOSED — by design]: RF is the only model everywhere.**
All conditions use `RandomForestClassifier/Regressor(n_estimators=150, max_depth=15)`, matching FLuID's published `create_classifier('rf')` defaults exactly. FLuID's teacher_algorithm and student_algorithm are both 'rf'. Using the same model family ensures KIRBy's federated results are directly comparable to FLuID's published hackathon results (MCC 0.55).

### Experimental Design Gaps

**ISSUE D1 [FIXED]: Fed01 has no centralized KIRBy baseline.**
Fed01's "naive pool" (C1) trains a single RF on Morgan FPs from all silos merged. But there is no condition where all data is merged and `create_hybrid` is run on all 15+ reps centrally (no federation, no silos). Without this upper-bound baseline, you cannot measure the cost of the federated protocol — if federated KIRBy (C5) underperforms, you don't know whether it's because federation hurts or because the small per-silo sample sizes are the bottleneck.
- **Fix:** Add a C0 or C7 condition: pool all training data, run `create_hybrid(all_reps, all_labels, budget=100)`, train RF on hybrid, evaluate on test.

**ISSUE D2 [FIXED]: Fed03 Part A has a feature-space confound.**
FLuID conditions (C2, C3) train teachers using only Morgan fingerprints (one 2048-bit representation). KIRBy condition (C4) uses 15+ representations (ECFP4, Mordred, MolFormer, PDV, ChemBERTa, etc.). If KIRBy wins, the result is ambiguous: is it because federated feature selection is a better fusion strategy, or simply because KIRBy has 15x more input information? The comparison is not apples-to-apples.
- **Fix option A:** Add FLuID conditions that use all reps (train per-silo teachers on multi-rep feature vectors, ensemble predictions). This tests fusion strategy with equal input information.
- **Fix option B:** Add a KIRBy condition that uses only Morgan FPs (same input as FLuID). This tests fusion strategy with equal input information.
- **Fix option C (simplest):** Both A and B. Two extra conditions to isolate the effect.

**ISSUE D3 [FIXED]: Fed06 never runs FLuID's teacher ensemble on antibody data.**
Fed06 has 6 conditions but only C6 (rep diversity) uses a FLuID-style ensemble — and only on one silo's data. Conditions C1-C4 use only KIRBy's hybrid pipeline. So the central claim "KIRBy outperforms FLuID for antibodies" is untested. Fed01 and Fed02 both run FLuID conditions head-to-head with KIRBy conditions — Fed06 should too.
- **Fix:** Add FLuID similarity + quality ensemble conditions to Fed06, using ESM2 embeddings for the similarity weighting (since antibodies don't have Morgan FPs). This requires adapting the ensemble helpers to work with protein embedding distances.

### Privacy Issues

**ISSUE P1 [FIXED]: Fed01 C4 student trained on test molecule fingerprints.**
`federate_labels_and_train_student` predicted on the full `fluid_df` including test molecules. The student's training pool included test FPs. Fixed by adding `train_indices` parameter to exclude test indices from the public pool.

**ISSUE P2 [FIXED]: Fed02 C6 virtual silos could access all reps (cross-silo).**
`fluid_within_kirby_across` passed the full `rep_dict_train` (all reps) to `federated_feature_selection`, so a virtual silo from rep A's clustering could select features from rep B. Fixed by restricting each virtual silo to only its parent rep.

**ISSUE P3 [ACKNOWLEDGED]: `federated_feature_selection` Phase 3 uses full rep_dict.**
`apply_feature_selection(rep_dict, trimmed_fi)` operates on the full training array (all samples from all silos), not silo-local data. In a real federated system, no single party would hold all rows. This is a simulation-level concern — the feature SELECTION is done locally (Phase 1), but the hybrid ASSEMBLY is centralized (Phase 3). Acceptable for a paper if explicitly stated as simulation.

**ISSUE P4 [ACKNOWLEDGED]: Fed01 C2 similarity weighting accesses silo training FPs at coordinator.**
The 8-NN distance computation in `predict_fluid_similarity_ensemble` samples 20% of each silo's training FPs to build a KNN index at the coordinator. In a true federated deployment, the KNN lookup would happen silo-side. This is faithful to how Lhasa actually runs FLuID (co-located data in a notebook), but the privacy claim in comments should say "predictions + similarity weights" not just "predictions only."

### Bugs Fixed During Session

| Issue | File | Fix |
|-------|------|-----|
| `get_molecular_representations` wrong import path | fed03 | Corrected to `find_best_hybrid_molecular.get_representations` |
| ChEMBL rep array not filtered after dedup | fed03 Part A | Added InChIKey keep mask to filter `reps_chembl` |
| `load_data('herg_asymmetric')` crash | fed03 | Redesigned dispatcher to use `--strategy` arg |
| `pred_probs` = max prob not P(positive) | helpers | Changed to `fed_probs[:, 1]` in all 3 ensemble functions |
| Missing `figure_font` in FLuID params | helpers | Added to params dict |
| `builtins.display` patch didn't affect FLuID | helpers | Added `fluid.display = lambda ...` |
| Global normalization in regression ensemble | helpers | Changed to per-silo normalization |
| `np.searchsorted` on unsorted train indices | fed04, fed05 | Replaced with explicit lookup or train-only clustering |
| Gaussian noise used global label range | helpers | Changed to silo-local range and MAD |
| `n_flip` gate blocked Gaussian noise for tiny silos | helpers | Separated gate to only apply to label_flip path |
| Fed05 C4 pooled labels without z-score normalization | fed05 | Added per-silo z-score before pooling |
| Fed01 C6 trained final model on Morgan FPs not hybrid | fed01 | Rebuilds KIRBy hybrid on cleaned data |
| Fed05 C5 `zscore` NaN on zero-variance | fed05 | Added variance guard |
| Various dedup, fallback, and edge case fixes | multiple | See git diff |

---

### OPIG Experiment Issues (from same audit)

The same audit checked all 6 OPIG experiment files. Data loading integrity is clean (all from original repos, correct sources, correct splits). But infrastructure issues overlap with federated:

**ISSUE O1 [OPEN]: All 6 OPIG experiments hardcode models instead of using config/catalog.**
Every experiment defines its own `MODELS = {'ridge': Ridge(alpha=1.0), 'rf': RF(n_estimators=500)}`. KIRBy's centralized model catalog uses `RidgeCV(alphas=logspace(-4,4,50))` (auto-tuned) and `RF(n_estimators=100)`. The hardcoded `alpha=1.0` means no regularization tuning. Results may differ from survey baselines that use the catalog.
- Files: all `opig_*.py`
- **Fix:** Replace hardcoded models with `get_model_factories(config['models'])` from `_shared.py`.

**ISSUE O2 [OPEN]: All 6 OPIG experiments reimplement evaluation loops.**
Each experiment has its own `evaluate_rep_clf` or `evaluate_rep_reg` function (~60 lines each) instead of calling `antibody_baseline_survey.evaluate_rep()` which already handles StratifiedGroupKFold, binary rep detection, scaling, and all metrics. This is duplicated code that can diverge from bug fixes in the survey.
- **Fix:** Refactor to call the survey's existing evaluate_rep function.

**ISSUE O3 [OPEN]: Config dict ignored in all OPIG experiments.**
Same as federated ISSUE S3. Hardcoded budgets, allocation methods, importance methods, search speed. No way to override without editing source.

**ISSUE O4 [OPEN]: Trastuzumab experiment changes methodology from paper.**
The original paper (Chinery et al. 2024) used zero-shot LLR ranking — no classifiers were trained. The OPIG experiment converts this to supervised classification (train RF/Ridge, rank by predict_proba). This is a methodological deviation, not just a hyperparameter choice. The paper's pipeline was rank-by-score, not train-then-rank.
- **Fix:** Add a zero-shot ranking condition that uses raw rep values (e.g., ESM2 LLR scores) directly for enrichment, matching the paper's approach. Keep the supervised condition as a separate comparison.

**ISSUE O5 [ALREADY FIXED]: Transfer learning feature scaling mismatch.**
`pretrain_chembl_dnn.py` applies StandardScaler during pre-training (line 101). `dnn_pretrained.py` does NOT apply StandardScaler during fine-tuning (line 113). Pre-trained weights expect scaled inputs but receive raw features. This could silently degrade transfer learning results.
- Files: `src/kirby/models/pretrain_chembl_dnn.py`, `src/kirby/models/dnn_pretrained.py`
- **Fix:** Apply the same StandardScaler in `PretrainedDNNRegressor.fit()`, or save/load the scaler from pre-training.

**ISSUE O6 [OPEN]: Graphinity `augmented` condition described but not implemented.**
The docstring for `opig_graphinity_ddg.py` describes an "augmented" condition (synthetic + experimental transfer) but it's missing from `CONDITION_DATASETS`. The condition is unreachable.

**ISSUE O7 [OPEN]: MEMORY.md claims humanness has "Thera-SAbDab fallback" — does not exist.**
The memory entry was written incorrectly. The therapeutic preset only loads from BioPhi IMGT/mAb-DB. No Thera-SAbDab code path exists. Memory needs correction.

**NOT an issue (confirmed clean):**
- All 7 dataset loaders download from correct original sources
- All unit conversions (pIC50, pKi, pActivity) verified correct
- All OPIG experiments use grouped splits (GroupKFold, StratifiedGroupKFold, pre-computed folds) — no random splits
- Trastuzumab framework sequences character-identical to source repo
- Ginkgo fold assignments use the paper's hierarchical cluster splits

---

## FLuID Cross-Reference (2026-03-28)

Systematic comparison of FLuID's actual code (`FLuID_POC/preprocessing/FLuID.py`) vs KIRBy's federated experiments. Fixed items marked ✓.

| # | FLuID Pattern | KIRBy Before | Status |
|---|--------------|-------------|--------|
| 1 | `cluster_data_space(trainingData)` — clusters training only | Clustered ALL data, then split | ✓ FIXED: split-first-then-cluster in fed01/fed04 |
| 2 | `create_trained_classifier(algo, data, False)` — balance=False for teachers | `class_weight='balanced'` on RF | Documented deviation (both valid) |
| 3 | 50K external ChEMBL transfer pool for student | Training molecules as public pool | Documented simulation compromise |
| 4 | `federate_labels()` — similarity-weighted vote (W-T columns) | Simple average (`all_preds /= k`) | ✓ FIXED: similarity-weighted in student pipeline |
| 5 | `select_student_data(mode='balanced')` — sort by confidence, pick top N/2 per class | Filter by threshold 0.7, random subsample | ✓ FIXED: matches FLuID's balanced mode |
| 6 | `cv=5` cross-validation in teachers | `cv=min(5, min_class_count)` | Documented (adapts to small silos) |
| 7 | K=8 (7 hackathon companies + smoothing) | K=5 | Documented (smaller datasets need fewer clusters) |
| 8 | `n_estimators=150, max_depth=15, class_weight='balanced'` | Same | ✓ Matches |
| 9 | 8-NN with 20% sample, `(1-d)^2` normalization | Same | ✓ Matches |
| 10 | `replicate_count=3` for CI | Single run | TODO: add multi-fold CV for CIs |

---

## References

- Hanser, T., Ahlberg, E., Amberg, A. et al. "Data-driven federated learning in drug discovery with knowledge distillation." *Nature Machine Intelligence* 7, 423–436 (2025). https://www.nature.com/articles/s42256-025-00991-2
- "Federated Learning in Computational Toxicology: An Industrial Perspective on the Effiris Hackathon." *Chemical Research in Toxicology* (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10523574/
- Hanser, T. "Federated Learning for Molecular Discovery." *Current Opinion in Structural Biology* (2023). https://www.sciencedirect.com/science/article/abs/pii/S0959440X23000192
- Lhasa Limited. "Advancing Drug Discovery Through Federated Learning." Blog post. https://www.lhasalimited.org/blog/federated-learning-drug-discovery/
- Lhasa Limited. "Effiris: A Secondary Pharmacology Model Suite." Blog post. https://www.lhasalimited.org/blog/effiris-a-secondary-pharmacology-model-suite-powered-by-privacy-preserving-data-sharing/
