# Experiment Refactor Roadmap

**Purpose:** Map all experiments to three thesis chapters (Complexity, Similarity,
Noise Mitigation), identify gaps, connect OPIG work, and align with the 4-step workflow.

**Date:** 2026-03-15

---

## The 4-Step Workflow

| Step | What | Status |
|------|------|--------|
| **1. Baselines** | Full landscape in the BOM — every rep × model × dataset | **In progress** (molecular 1/8, antibody 24/41, TCGA 2/3, DTA 0/3, reaction 2/3) |
| **2. Selection** | Use BOM to select viable tasks (data + target + preprocessing + models + base reps) | **Blocked** on Step 1 completion |
| **3. Framework** | Test hybrid search machinery, finalize the algorithm | **Code written** (f00–f04), **needs rewrite** (f01–f04), **unwritten** (f05–f08) |
| **4. Thesis experiments** | Complexity + Similarity + Noise Mitigation + Forward Compatibility | **Code written** (c01–c15, s01–s11, n01–n03), **remaining gaps identified below** |

Steps 1-2 feed into Steps 3-4. Framework (Step 3) produces the finalized hybrid search
settings. Thesis experiments (Step 4) use those settings to produce chapter results.

---

## Current Experiment Inventory (41 scripts)

### Complexity (24 scripts — `tests/experiments/complexity/`)

| ID | Script | What it tests | Chapter claim |
|----|--------|---------------|---------------|
| **c01** | `c01_rep_characterization.py` | Intrinsic dim, MI, effective rank, per-model baselines | Reps vary widely; no single best |
| **c02** | `c02_complexity_grid.py` | Model × budget × allocation × importance grid | Optimal pairing exists and is dataset-dependent |
| **c03** | `c03_cross_dataset_transfer.py` | Do optimal configs transfer? Complexity profile correlation | Configs partially transfer within domains |
| **c04** | `c04_importance_as_metric.py` | Feature importance as kNN weights + similarity retrieval | Importance scores are a learned metric |
| **c05** | `c05_learning_curves.py` | Sample efficiency at 10-100% fractions | Hybrids help more in low-data regimes |
| **c06** | `c06_within_family_gradient.py` | Capacity gradient within model families | Budget should match model capacity |
| **c07** | `c07_data_complexity_partitions.py` | Easy vs hard subpopulation behavior | Hybrids help most on hard partitions |
| **c08** | `c08_calibration.py` | Conformal prediction at optimal vs suboptimal configs | Optimal complexity → better calibration |
| **c09** | `c09_formal_analysis.py` | ANOVA, MI retention, effective rank, bias-variance | Theoretical grounding (4 sub-analyses) |
| **c10** | `c10_noise_robustness.py` | Noise Degradation Slope (NDS) | Hybrids are more noise-robust **(-> Noise chapter too)** |
| **c11** | `c11_ensemble_vs_hybrid.py` | Model ensemble vs feature-level hybrid | Rep-level fusion >= model-level fusion, cheaper |
| **c12** | `c12_modality_ablation.py` | TCGA modality combinations (9 omics types) | Multi-modal hybrids capture complementary info |
| **c13** | `c13_forward_compatibility.py` | 5-condition ablation for new reps (A-E), controls | New reps benefit from and improve hybrids |
| **c14** | `c14_cross_domain_summary.py` | KIRBy hybrid vs best single rep across domains, Wilcoxon | Consistent improvement across molecular/antibody/TCGA |
| **c15** | `c15_ml_theory_depth.py` | Distance concentration, kernel degradation + RBF gamma sensitivity, double descent, Rademacher | Theoretical depth across 4 ML theory levels |
| **c16** | `c16_feature_selection_as_regularization.py` | Feature selection as implicit regularizer | Budget = regularization strength, not just compression |
| **c17** | `c17_information_theoretic_decomposition.py` | MI decomposition of hybrid vs individual reps | Hybrid captures complementary information |
| **c18** | `c18_diversity_vs_depth.py` | Rep diversity vs feature depth tradeoff | Many shallow reps > few deep reps (given budget) |
| **c19** | `c19_importance_weighted_training.py` | Importance-weighted sample/feature training | Feature importance as training signal |
| **c21** | `c21_single_rep_augmentation.py` | Augmentation on individual reps (merged C20) | Augmentation helps even without hybridization |
| **c22** | `c22_explainability.py` | SHAP/importance explanations of hybrid decisions | Hybrid feature selection is interpretable |
| **c23** | `c23_evaluator_model_matching.py` | Internal evaluator vs downstream model matching | Matched evaluators improve hybrid quality |
| **c24** | `c24_blueprint_transfer.py` | Cross-fold blueprint generalization + Nogueira stability | Feature selection is not overfit to training folds |

**Dispatch notes:**
- c01-c24: Per-dataset experiments, triggered via `run_all()` in `complexity_helpers.py`.
- c03, c14: Cross-dataset/cross-domain, standalone dispatch only.
- Cross-dataset blueprint transfer is now OPIG C5 (`opig_blueprint_transfer.py`), dispatched via `submit_opig_experiments.sh`.

### Similarity (11 scripts — `tests/experiments/similarity/`)

| ID | Script | What it tests | Chapter claim |
|----|--------|---------------|---------------|
| **s01** | `s01_similarity_baseline.py` | SAC, violation rate, SALI per rep | The problem: distances don't track activity |
| **s02** | `s02_hybrid_similarity.py` | Hybrid distance (weighted + unweighted) | Hybrids improve similarity-activity correlation |
| **s03** | `s03_feature_concentration.py` | MI density, effective rank, Gini of importance | Why: information concentration in fewer dimensions |
| **s04** | `s04_activity_cliffs.py` | Prediction accuracy on cliff vs non-cliff compounds | Hybrids reduce cliff prediction error **(→ Noise chapter too)** |
| **s05** | `s05_knn_prediction.py` | kNN with hybrid distance, multiple k | Direct downstream benefit of better similarity |
| **s06** | `s06_applicability_domain.py` | Distance-error correlation, conformal AD | Hybrid distance tracks prediction reliability |
| **s07** | `s07_noise_bridge.py` | Noise detector false positives (cliffs flagged as noise) | Bridge to Noise chapter **(→ Noise chapter)** |
| **s08** | `s08_optimal_dimension.py` | Optimal budget for similarity vs prediction | Different objectives → different optimal dimensions |
| **s09** | `s09_cross_dataset_summary.py` | Aggregate s02–s08 across datasets, Wilcoxon test | Generalization: improvements hold across tasks |
| **s10** | `s10_importance_downstream.py` | Importance-weighted GP/SVR/kNN/q-RASAR | Importance weights improve distance-dependent methods |
| **s11** | `s11_bayesian_optimization.py` | GP-BO in 6+ conditions (ecfp4/pca/hybrid/random) + **per-rep BO** (`--per-rep-bo`) + **multi-objective BO** (`--multi-objective`, label+QED geometric mean per Moesser 2023 Ch.2) | Compact hybrid space enables efficient BO; multi-obj also benefits |

### Framework (5 scripts — `tests/experiments/framework/`)

| ID | Script | What it tests | Status |
|----|--------|---------------|--------|
| **f00** | `f00_rep_screening.py` | Where should rep filtering go? 4 options | Written, needs testing |
| **f01** | `f01_allocation_strategies.py` | 17 allocation strategies | **Needs rewrite** (hardcoded R², missing 4 families) |
| **f02** | `f02_adaptive_pruning.py` | Pruning simulation | **Needs rewrite** (directional bug, incomplete) |
| **f03** | `f03_greedy_step_size.py` | Step size × patience | **Needs rewrite** (hardcoded R², greedy-only) |
| **f04** | `f04_two_phase_search.py` | Full grid vs two-phase | **Needs rewrite** (hardcoded R², model mismatch) |

### Domain-Specific Comparisons (11 scripts — `tests/experiments/`)

| Script | What it tests | Which theme? |
|--------|---------------|--------------|
| `dta_e2e_comparison.py` | Frozen KIRBy vs DeepDTA vs GraphDTA | **Complexity** (frozen vs E2E) |
| `molecular_finetune_comparison.py` | Frozen vs LoRA vs contrastive LMs | **Complexity** (FT doesn't help → frozen wins) |
| `dta_finetune_comparison.py` | Protein/drug LoRA on DTA | **Complexity** (all NaN — negative result) |
| `antibody_finetune_comparison.py` | AbLang2/ESM-2/AntiBERTy FT | **Complexity** (FT helps on large DMS only) |
| `qm9_3d_comparison.py` | When frozen embeddings break down | **Complexity** (3D geometry = exception) |
| `tcga_foundation_comparison.py` | Genomic foundation models | **Complexity** (Geneformer V2 = forward compat) |
| `wsi_abmil_comparison.py` | WSI aggregation methods | **Complexity** (attention not learning) |
| `reaction_rep_comparison.py` | Reaction representation ranking | **Complexity** (baseline) |
| `antibody_region_ablation.py` | CDR vs framework vs entropy pooling | **Complexity** (region-level complexity) |
| `covariate_shift.py` | Scaffold vs MW vs spectral splits | **Complexity** (robustness under shift) |
| `imrie_transfer_learning.py` | Pre-train on ChEMBL, fine-tune on target | **Complexity** (model improvement orthogonal to rep choice?) |

---

## Theme 1: COMPLEXITY (Main Hybrid Paper)

### Core argument
Hybrids are a domain-agnostic, forward-compatible, computationally efficient method
for handling multi-modal input data. The input representation matters more than the
model architecture. Controlling representation complexity (dimension, information
density) is the key degree of freedom.

### What you have (solid)

**Characterization & profiling:** c01, c09a (ANOVA)
- Intrinsic dimensionality landscape, MI with labels, effective rank
- Variance decomposition: how much does model vs budget vs interaction explain?

**Optimal matching:** c02, c06, c09d (bias-variance)
- Model-budget grids, within-family capacity gradients
- The "Goldilocks" claim: underfitting and overfitting in representation space

**Generalization:** c03, c12 (modality ablation)
- Cross-dataset transfer of optimal configs
- Multi-modal (TCGA 9 omics) combination analysis

**Efficiency claims:** c05 (learning curves), c08 (calibration)
- Sample efficiency of hybrids, especially in low-data regimes
- Calibrated uncertainty at optimal complexity

**Head-to-head:** c11 (ensemble vs hybrid)
- Feature-level fusion (hybrid) vs model-level fusion (ensemble)
- Timing comparison — hybrid should be cheaper

**Forward compatibility:** Geneformer V2 baseline result (AUROC=0.946, #1 immediately)
- New reps drop in with zero code changes and get hybridized

### What's missing (gaps to fill)

#### GAP C-A: Forward Compatibility Experiment (HIGH PRIORITY)
**The thesis claim:** New representations released after KIRBy was built can be dropped
in and immediately benefit from hybridization. This is NOT just a baseline — it needs
a controlled experiment.

**What to test:**
For each forward-compat rep (MINT, SaProt, IgT5, MolE, MiniMol, GP-MoLFormer,
ESM3, Boltz-2, Geneformer V2):
1. **Standalone performance** — rep + best model (from survey)
2. **Hybrid performance** — rep added to existing rep pool, `find_best_hybrid`
3. **Hybrid without new rep** — existing rep pool only
4. **Delta** — does the new rep improve the hybrid? Does the hybrid improve the new rep?

**Key metric:** Does hybridization improve the new rep more than the new rep improves
the hybrid? (Tests whether KIRBy enhances new reps, not just absorbs them.)

**Per domain:**
- Molecular: MiniMol, GP-MoLFormer, MolE → logd, caco2, herg
- Antibody: MINT, SaProt, IgT5, ESM3 → tresanco, fcrnrt, cic
- DTA: MINT, SaProt, ESM3, Boltz-2 → davis, kiba
- TCGA: Geneformer V2 (already validated), BulkRNABERT → luad_vs_lusc, brca_pam50

**Script needed:** `tests/experiments/complexity/c13_forward_compatibility.py`
(NOT the same as running baselines — this is a controlled A/B with the hybrid)

**OPIG connection:** IgT5 (Exscientia), SaProt (ICLR 2024), ESM3 are all new
models from the community. Demonstrating that KIRBy future-proofs against model
churn is a core selling point.

#### GAP C-B: Computational Cost Analysis (MEDIUM)
**The claim:** Hybrids compress representations, reducing computational cost for
downstream models without sacrificing performance.

**What to test:**
1. Wall time: model training + inference on full-dim rep vs hybrid at various budgets
2. Memory footprint during training
3. Inference latency at deployment scale (batch prediction on 10K compounds)
4. Compare: PCA to same dim vs hybrid to same dim (is the cost of hybrid construction
   amortized by cheaper downstream compute?)

**Should be woven into c02 or c11**, not a standalone script. Add timing columns.

#### GAP C-C: Theoretical Connection to ML Theory (LOW — mostly narrative)
**What to formalize:** c09 has the 4 formal sub-analyses but they're empirical.
The paper needs to connect to:
- Bias-variance tradeoff (c09d is the proxy)
- Approximation theory: PAC-learning bounds on VC dimension of compressed inputs
- Rate-distortion theory: information retention vs compression ratio (c09b)
- Double descent: do hybrids avoid the interpolation threshold?

**No new code needed.** This is analysis + writing on top of c09 results. But you
should ensure c09 produces enough data points (budgets at fine granularity near
the optimal) to support the theoretical claims.

#### GAP C-D: Domain-Agnostic Demonstration (HIGH)
**The claim:** KIRBy works across all domains with the same algorithm.

**What's needed:** A single summary figure/table showing hybrid improvement over
best single rep across ALL domains (molecular, antibody, TCGA, DTA, reaction).
c09/s09 cross-dataset summary gets at this but only within a domain.

**Script needed:** `tests/experiments/complexity/c14_cross_domain_summary.py`
- Consume BOM + hybrid results from all domains
- For each task: best_single_rep, best_hybrid, delta, % improvement
- Statistical test: is hybrid significantly better across domains? (paired Wilcoxon)

### Existing experiments → chapter mapping

| Chapter section | Primary experiments | Supporting evidence |
|----------------|--------------------|--------------------|
| "Reps vary widely" | c01 | BOM tables |
| "Optimal complexity exists" | c02, c06, c09d | c07 (partition analysis) |
| "Complexity transfers" | c03, c24, c25 | c12 (modality ablation) |
| "Information concentration" | c09b (MI retention), c09c (effective rank), c17 | s03 (from Similarity) |
| "Feature selection as regularization" | c16 | c09d (bias-variance) |
| "Diversity vs depth" | c18 | c01 (characterization) |
| "Hybrids vs ensembles" | c11 | Timing data |
| "Forward compatible" | c13 | BOM forward-compat baselines |
| "Calibrated uncertainty" | c08 | s06 (AD from Similarity) |
| "Sample efficient" | c05 | c07 (hard partitions) |
| "Domain agnostic" | c14 | All domain BOM results |
| "Noise robust" | c10 | s04, s07 (from Similarity/Noise) |
| "Distance concentration" | c15a | Beyer (1999), hubness (Radovanovic 2010) |
| "Kernel degradation" | c15b | Bengio (2005), kernel-target alignment |
| "Double descent avoidance" | c15c | Budget always << n, no interpolation threshold |
| "Generalization bounds" | c15d | Rademacher complexity, sqrt(d/n) scaling |
| "Explainability" | c22 | SHAP feature attribution |
| "Evaluator-model matching" | c23 | c02 grid (allocation matters) |
| "Pretrained hybrids" | OPIG C5 (`opig_blueprint_transfer.py`) | c24 (cross-fold stability), c03 (within-domain transfer) |

---

## Theme 2: SIMILARITY

### Core argument
Many ML methods assume similar inputs have similar outputs. This assumption fails in
molecular/biological data (activity cliffs, measurement noise, representation inadequacy).
Hybrids can fix this: a compact, task-aware representation makes similarity meaningful
again. This has concrete downstream benefits beyond prediction accuracy.

### What you have (solid)

**The problem statement:** s01 (SAC baseline)
- Per-rep SAC, violation rate, SALI
- Shows that standard reps have poor similarity-activity correlation

**The solution:** s02 (hybrid similarity), s03 (why it works)
- Hybrid distance improves SAC
- Feature concentration: MI density in fewer dimensions

**Activity cliffs:** s04
- Cliff-stratified prediction (cliffs vs non-cliffs)
- kNN + RF on cliffs with hybrid distance

**Downstream tasks:** s05 (kNN), s06 (applicability domain)
- kNN prediction with hybrid distance
- Distance-error correlation, conformal AD

**Bridge to noise:** s07
- Noise detector false positives that are actually cliffs

**Optimal dimension:** s08
- Different optimal budgets for similarity vs prediction tasks

**Generalization:** s09
- Cross-dataset summary with Wilcoxon test

### What's missing (gaps to fill)

#### GAP S-A: Formal Problem Statement (HIGH PRIORITY)
**You said:** "I need to formally show the problem — many ML methods rely on the
assumption that similar data will exhibit similar values. The metrics I've used to
examine molecular similarity are dismal."

s01 gets at this but needs to be more rigorous. The formal claim is:

**Proposition:** For representation r with distance function d_r, the Similarity
Assumption holds if Spearman(d_r(x_i, x_j), |y_i - y_j|) is significantly positive.
For standard molecular representations, this correlation is weak (ρ < 0.3) and
the violation rate (fraction of k-NN pairs with |Δy| > median) exceeds 40%.

**What to add to s01:**
1. Statistical significance testing (permutation test, not just p-value)
2. Comparison across representation families (fingerprint, descriptor, LM, graph)
3. Comparison across domains (molecular, antibody, DTA) — does the problem exist
   everywhere, or is it molecular-specific?
4. Connection to LNL failures: "every LNL method that relied on neighbor scores
   failed" — can you show that the violation rate predicts LNL failure?

**Script modification:** Extend `s01_similarity_baseline.py` with cross-domain runs
and LNL failure correlation. Or create `s00_formal_problem.py` as a precursor.

#### GAP S-B: Feature Importance as Downstream Weights (HIGH PRIORITY)
**You said:** "I want to use feature importance scores as weights in downstream tasks
where applicable. I've had particular success with this in noise mitigation research."

c04 partially covers this (importance as kNN weights), but you need a broader set of
downstream tasks where importance weights directly improve performance:

**What to test:**
1. **Weighted kNN regression** — importance scores as feature weights (c04/s05 have this)
2. **Weighted distance for GP kernel** — RBF kernel with importance-weighted distances.
   This is similarity-adjacent: k(x,x') = exp(-γ Σ_i w_i (x_i - x'_i)²).
   You already have GP + TanimotoKernel (C1). Add importance-weighted RBF.
3. **Weighted SVR with RBF** — same as GP but for SVR. You already noted this.
4. **Sample weighting** — use per-sample importance-based reliability scores as
   sample weights in model training (connects to Noise theme)
5. **Feature-weighted clustering** — importance weights in k-means/DBSCAN distance

**Script needed:** `tests/experiments/similarity/s10_importance_downstream.py`
- For each downstream task: unweighted vs importance-weighted, hybrid vs base reps
- Models: kNN, GP, SVR (all distance-dependent)
- Compare importance methods: treeshap, lgbm_gain, permutation

#### GAP S-C: Similarity-Adjacent Downstream Tasks (MEDIUM-HIGH)
**Beyond kNN and AD, what tasks benefit from better similarity?**

From `experiment_extension.txt` (your earlier brainstorm):

1. **Bayesian Optimization / Active Learning** (highest impact)
   - Simulate active learning loop: seed set → GP in hybrid space → EI acquisition → iterate
   - Compare sample efficiency: hybrid (50D) vs ECFP4 (2048D) vs PCA (50D)
   - Metric: area under best-found-so-far curve
   - Datasets: logd, herg, BindingDB_Kd (large enough for meaningful BO)
   - **Script needed:** `tests/experiments/similarity/s11_bayesian_optimization.py`
   - **OPIG connection:** GAUCHE (C1) provides the Tanimoto kernel for GP. Leo Klarner's
     work on BO in molecular space is directly relevant.

2. **Anomaly / Novelty Detection**
   - Isolation Forest, Local Outlier Factor, One-Class SVM in hybrid vs full-dim space
   - Can you detect genuinely novel compounds better?
   - **Script needed:** `tests/experiments/similarity/s12_novelty_detection.py`

3. **Clustering Quality**
   - k-means/DBSCAN on hybrid vs ECFP4. Measure: Silhouette score, Calinski-Harabasz,
     but also chemical meaningfulness (do clusters correspond to known scaffolds/MoA?)
   - **Script needed:** `tests/experiments/similarity/s13_clustering.py`

4. **Retrieval-Augmented Prediction**
   - For each test compound, retrieve k nearest neighbors in hybrid space from training
     set, use their properties as additional features for a model
   - Tests whether hybrid distance is good enough to find "useful" neighbors
   - Could be integrated into s05 as an extension

5. **Read-Across (regulatory toxicology)**
   - Given a query compound, retrieve structurally similar compounds with known tox
   - Evaluate whether hybrid-retrieved analogs have more consistent tox profiles
   - Domain: hERG (ion channel safety), ADME datasets
   - Publishable angle: "task-aware read-across via hybrid representations"
   - **Lower priority** but unique regulatory angle

#### GAP S-D: Cross-Domain Similarity (MEDIUM)
**The problem might not be molecular-specific.** Does the similarity assumption
also fail for antibody tasks? DTA? TCGA?

**What to test:**
- Run s01 on antibody datasets (FLAb inter-family, trastuzumab)
- Run s01 on DTA (davis, kiba — but what is "similarity" for drug-target pairs?)
- Run s01 on TCGA (what does "similar" mean for multi-omic profiles?)

If the problem exists across domains, the Similarity chapter becomes stronger.
If it's molecular-specific, that's still interesting but narrows the scope.

s09 already aggregates across datasets but only if you run s01-s08 on non-molecular
tasks. The infrastructure supports it (domain-aware metrics in `_shared.py`).

#### GAP S-E: Controlled Dimension Sweep for Similarity (LOW)
s08 has this but could be more systematic. The key figure is:

```
Budget: 10  25  50  100  200  500  full_dim
SAC:    ?   ?   ?   ?    ?    ?    baseline
kNN:    ?   ?   ?   ?    ?    ?    baseline
GP:     ?   ?   ?   ?    ?    ?    baseline
RF:     ?   ?   ?   ?    ?    ?    baseline
```

Show that similarity-dependent methods (kNN, GP, SVR) peak at lower budgets than
similarity-independent methods (RF, XGBoost). This would be a clean figure.

### OPIG connections to Similarity

| OPIG item | Connection | Priority |
|-----------|------------|----------|
| **Activity cliffs (#30, Dablander)** | Central to s04. Dablander's ECFP Sort & Slice (A1) already in KIRBy. His activity cliff research provides the formal framework. | **HIGH** |
| **GAUCHE Tanimoto kernel (C1)** | TanimotoKernel in GP HP grid. For s11 (BO), use Tanimoto GP in hybrid space. | **MEDIUM** |
| **Q-SAVI covariate shift (C2)** | MW/spectral splits test whether similarity improvements hold under distribution shift. | **MEDIUM** |
| **Ginkgo developability (B5)** | Small-sample regime (94-242 samples). Similarity methods should shine here. | **LOW** |

### Existing experiments → chapter mapping

| Chapter section | Primary experiments | Gaps to fill |
|----------------|--------------------|----|
| "The similarity assumption fails" | s01 | **S-A** (formal problem, cross-domain) |
| "Hybrids fix it" | s02, s03 | — (solid) |
| "Activity cliffs" | s04 | Dablander framework (OPIG #30) |
| "Downstream: kNN" | s05 | — (solid) |
| "Downstream: applicability domain" | s06 | — (could deepen) |
| "Downstream: GP/SVR/weighted" | — | **S-B** (importance as weights) |
| "Downstream: BO/active learning" | — | **S-C** (new experiment) |
| "Optimal dimension for similarity" | s08 | **S-E** (controlled sweep) |
| "Bridge to noise" | s07 | — (connects to Noise theme) |
| "Generalization" | s09 | **S-D** (cross-domain) |

---

## Theme 3: NOISE MITIGATION

> **Detailed roadmap:** See `NOISE_EXPERIMENT_ROADMAP.md` for the full noise experiment
> plan including N09-N13 (LNL survey, hybrid neighborhood, estimation grid, importance
> weighting, noise distributions). The roadmap incorporates confirmation report research
> on 12 LNL methods, similarity-assumption failures, and noise estimation/smoothing.

### Core argument
Improving ML in noisy data settings. Ideally removing or smoothing noise; at minimum,
proving methods that work despite noise. Activity cliff awareness is central.
Feature importance weighting for sample selection is a key mechanism.

### What you have

**Noise robustness profiling:** c10
- NDS (Noise Degradation Slope) per rep and per budget
- Feature stability (Jaccard overlap) under noise

**Activity cliff connection:** s04, s07
- s04: cliff-stratified prediction error
- s07: noise detection false positives that are actually cliffs

**Activity cliff framework:** n01 ✅ DONE
- Consensus cliff identification (Jaccard overlap across reps)
- SALI landscape smoothing (hybrid vs per-rep)
- Cliff mitigation by hybridization (cliff pair count reduction)
- Cliff-stratified prediction error (RF, hybrid vs per-rep)

**Sample weighting:** n02 ✅ DONE
- 4 conditions: base ± weights, hybrid ± weights
- Reliability score from importance-weighted neighborhood consistency
- Tests if hybrids already handle noise implicitly

**SAS Landscape Analysis:** n03 ✅ DONE (Moesser 2023 / Shanmugasundaram & Maggiora 2001)
- 4-quadrant SAS maps per representation (Zone I–IV: cliffs, nondescript, scaffold hops, smooth)
- Canonical ECFP4 Tanimoto baseline (molecular) + per-rep cosine similarity
- Hybrid zone population shift (Zone I reduction, Zone IV increase)
- Cross-rep landscape comparison (zone populations vs dim/MI/erank)
- **SAS-BO bridge**: Zone I fraction predicts GP quality — explains why compact hybrid helps BO (links to S11)

### What's missing (remaining gaps)

#### GAP N-D: GP-Based Noise Detection in Hybrid Space (HIGH)
**From experiment_extension.txt:** The GP's leave-one-out predictive distribution
in hybrid space can distinguish noise from activity cliffs.

**What to test:**
1. Fit GP in hybrid space (budget=50)
2. Compute LOO predictive probabilities for each sample
3. Flag samples below threshold as suspected noise
4. Validate against: known activity cliffs (should NOT be flagged), synthetic
   noise injection (SHOULD be flagged)
5. Compare: GP-LOO in hybrid space vs GP-LOO in full ECFP4 space vs small-loss
   selection vs CNLCU

**Script needed:** `tests/experiments/noise/n04_gp_noise_detection.py`

**This is the "killer experiment"** for the noise chapter — it connects the
Similarity theme (hybrid distance) to the Noise theme (noise detection) and
uses the Activity Cliff framework (N-A) for validation. The narrative:
"Hybrid representations enable noise-aware training that respects activity cliffs."

#### GAP N-E: Noise Injection Study (MEDIUM)
c10 does this for NDS (how quickly R² degrades with noise), but a deeper study would:

1. **Inject calibrated noise** at multiple levels (5%, 10%, 20%, 30% of label range)
2. **Compare noise detection methods** at each level:
   - GP-LOO (N-C)
   - Small-loss selection (train model, flag high-loss samples)
   - kNN consistency (flag samples that disagree with k nearest neighbors)
   - CNLCU-style temporal averaging
3. **Evaluate on downstream performance after noise removal**
4. **Cross-reference with activity cliffs** — false positive rate (cliffs wrongly flagged)

**Script needed:** `tests/experiments/noise/n05_noise_injection.py`

#### GAP N-F: BindingDB Ground-Truth Noise (MEDIUM)
**From experiment_extension.txt:** BindingDB has compound-target pairs measured by
multiple labs. Disagreements between measurements = ground-truth noise estimate.

**What to test:**
1. Identify BindingDB entries with replicate measurements (same InChIKey + same target)
2. Quantify inter-lab disagreement as noise ground truth
3. Test noise detection methods against this ground truth
4. Show that hybrid-space GP-LOO detects genuinely noisy measurements better than
   alternatives

**Script needed:** `tests/experiments/noise/n06_bindingdb_noise.py`

**This would be a genuinely novel contribution** — most noise mitigation papers use
synthetic noise on CIFAR/MNIST. Ground-truth biological noise is much more convincing.

#### GAP N-G: Noise-Robust Model Training (LOW-MEDIUM)
After noise detection, what do you do with the flagged samples?

Options to compare:
1. **Remove** — drop flagged samples, retrain
2. **Relabel** — replace flagged labels with GP posterior mean
3. **Downweight** — use (1 - noise_score) as sample weight
4. **Robust loss** — Huber loss, trimmed loss, etc. (model-side, not data-side)
5. **Smoothed labels** — kNN-smoothed labels in hybrid space

**Script needed:** `tests/experiments/noise/n07_noise_remediation.py`

### OPIG connections to Noise Mitigation

| OPIG item | Connection | Priority |
|-----------|------------|----------|
| **Dablander activity cliffs (#30)** | Foundation of N-A. His cliff definitions, SALI, and Sort & Slice are already integrated. | **HIGH** |
| **Graphinity synthetic DDG (B2)** | ~943K synthetic labels vs ~429 experimental. Can noise detection flag synthetic labels that disagree with experimental? | **MEDIUM** |
| **Ginkgo developability (B5)** | 242 antibodies — small enough that individual noisy points have outsized impact. Noise mitigation could matter here. | **LOW** |

### Existing experiments → chapter mapping

| Chapter section | Primary experiments | Gaps to fill |
|----------------|--------------------|----|
| "Noise degrades ML" | c10 (NDS) | **N-D** (deeper injection study) |
| "Activity cliffs ≠ noise" | s04, s07 | **N-A** (formal Dablander framework) |
| "Detection: GP in hybrid space" | — | **N-C** (GP-LOO vs alternatives) |
| "Detection: sample weighting" | — | **N-B** (importance-weighted reliability) |
| "Ground-truth noise" | — | **N-E** (BindingDB replicates) |
| "Remediation" | — | **N-F** (remove, relabel, downweight, robust loss) |

---

## Framework Experiments (Step 3)

The FRAMEWORK_EXPERIMENT_ROADMAP.md already covers this in detail. Summary of what
needs to happen and how it feeds into the thesis experiments:

| Phase | What | Feeds into |
|-------|------|-----------|
| Phase 0 | `_shared.py` rewrite (domain-aware metrics, all 11 models) | **DONE** (mostly) |
| Phase 1 | f00 rep screening | All thesis experiments (should bad reps enter the search?) |
| Phase 2 | f01-f04 rewrites (allocation, step size, pruning, two-phase) | Complexity c02 (grid uses finalized search settings) |
| Phase 3 | f05-f07 (importance correlation, augmentation, internal evaluator) | Similarity S-B (which importance method to use as weights?) |
| Phase 4 | f08 cross-domain analysis | Complexity C-D (domain-agnostic claim) |
| Phase 5 | Algorithm refinement (new SPEED_TIERS) | **All** thesis experiments use finalized search |
| Phase 6 | A/B validation | Complexity c13 (forward compat uses validated search) |

**Critical dependency:** Framework Phase 5 (finalized search settings) must happen
BEFORE running thesis experiments at scale. Otherwise you'd need to re-run everything
after changing the algorithm.

**Exception:** Similarity experiments (s01-s09) don't depend on `find_best_hybrid`
for most of their analysis (they use `create_hybrid` directly with explicit settings).
These can run in parallel with framework experiments.

---

## Task Selection (Step 2)

Based on BOM results so far, these are the viable tasks per domain. This will expand
as more baselines land.

### Confirmed viable (Phase 5 results exist, signal confirmed)

| Domain | Task | Why selected | Primary metric |
|--------|------|-------------|----------------|
| **Molecular** | `logd` (5,028) | Best coverage, 17 reps tested, strong signal | RMSE |
| **Antibody** | `flab_thermo_tresanco` (672) | Largest inter-family, ρ=0.805 | Spearman |
| **Antibody** | `flab_pk_jain_fcrnrt` (132) | Small but strong signal, ρ=0.595 | Spearman |
| **Antibody** | `flab_polyreact_jain_cic` (137) | Strong signal, ρ=0.586 | Spearman |
| **TCGA** | `tcga_luad_vs_lusc` (1,017) | Best classification task, AUROC=0.946 | AUROC |
| **TCGA** | `tcga_brca_pam50` (821) | 4-class, F1w=0.861 | F1-weighted |
| **Reaction** | `buchwald_hartwig` (3,955) | Strong signal, R²=0.718 | RMSE |
| **Reaction** | `suzuki` (5,760) | Interesting failure mode, only onehot works | RMSE |

### Likely viable (baselines running, expect good signal)

| Domain | Task | Why expected viable |
|--------|------|---------------------|
| **Molecular** | `caco2`, `herg` | Standard benchmarks, OpenADMET/ChEMBL data |
| **Molecular** | `adme_hlm`, `adme_rlm` | Polaris ADME, 3K samples |
| **Antibody** | `tz_her2_binary` (5K) | Fast reps show 84.9% acc, LMs should improve |
| **DTA** | `davis` (30K) | Frozen KIRBy already ties DeepDTA (CI=0.741) |
| **DTA** | `kiba` (30K) | Diverse drug space, genuine rep variance |

### To verify (need baseline results)

| Domain | Task | Concern |
|--------|------|---------|
| **Molecular** | `qm9` | 3D tasks may not suit frozen 2D reps |
| **Molecular** | `adme_mdr1`, `adme_solubility` | Unknown signal quality |
| **Antibody** | `flab_binding_koenig` (DMS) | Only 4 fast reps tested, low ρ |
| **Antibody** | `humanness_species/therapeutic` | New, untested |
| **Antibody** | 9× `ginkgo_*` | Very small (94-242), may be too noisy |
| **Antibody** | 3× `graphinity_*` | New, untested |
| **DTA** | `bindingdb_kd` (52K) | New, untested |

### Dropped / deferred

| Task | Reason |
|------|--------|
| `tcga_brca_survival` (151) | All R² negative. Unsolvable with current features. |
| `ddg_combined` | Only aido_protein tested (failed). May revive with LM reps. |
| `pdbbind_refined/core` | Blocked on manual download |
| All DMS datasets (10/11) | Only aido_protein tested (failed). Revive when LM baselines land. |

### Per-theme task allocation

**Complexity experiments:** Run on ALL confirmed viable tasks (8+). Need cross-domain
coverage. Use `all` rep presets.

**Similarity experiments:** Focus on molecular (logd, caco2, herg) + 1-2 antibody
(tresanco, tz_her2_binary). Similarity metrics are best understood for molecular data.
Expand to other domains via GAP S-D.

**Noise experiments:** Focus on molecular (logd, herg — known noisy assays) + DTA
(BindingDB_Kd — replicate measurements for ground truth). Add antibody if Ginkgo
results are noisy.

**Framework experiments:** Use the standard task set from FRAMEWORK_EXPERIMENT_ROADMAP.md
(logd, tresanco, luad_vs_lusc, buchwald_hartwig, davis).

---

## Execution Order

### Phase A: Complete Baselines (Step 1) — NOW
- Wait for running HTC/ARC jobs to finish
- Populate BOM with remaining results
- Drop dead reps/tasks

### Phase B: Select Tasks (Step 2) — AFTER PHASE A
- Apply selection criteria from §Task Selection above
- Verify selected tasks are "worth exploring with hybrid functionality"
- Deep dive: is each task used in industry? Is the protocol standard?
- Document in BOM §2G

### Phase C: Framework Experiments (Step 3) — AFTER PHASE B
- Rewrite f01-f04 (domain-aware metrics, all 11 models)
- Run f00-f04 on selected tasks
- Write f05-f08
- Synthesize: finalize SPEED_TIERS and algorithm settings
- A/B validate

### Phase D: Thesis Experiments (Step 4) — AFTER PHASE C

Run order within Phase D:

1. **Complexity c01-c12** (existing scripts, selected tasks, finalized search)
   - Plus c13 (forward compat) and c14 (cross-domain summary)
   - These don't depend on similarity/noise results

2. **Similarity s01-s09** (existing scripts, selected tasks)
   - Plus s10 (importance downstream), s11 (BO), s12-s13 (novelty, clustering)
   - s01 can run independently of everything else
   - s02+ need hybrid results from Step 3

3. **Noise n01-n07** (n01-n03 written, n04-n07 planned)
   - n01 (cliff framework) ✅, n02 (sample weighting) ✅, n03 (SAS landscape) ✅
   - n04 (GP noise detection) is the next core experiment
   - n05-n07 are depth experiments, can wait for initial results

### Parallelism

- Complexity and Similarity are mostly independent — run in parallel
- Noise depends on Similarity results (s04, s07) — run after Similarity
- Framework must complete before full-scale runs of any theme
- Within each theme, experiments are numbered in dependency order

---

## New Scripts Needed (Summary)

| ID | Script | Theme | Priority | Depends on |
|----|--------|-------|----------|------------|
| **c13** | `c13_forward_compatibility.py` | Complexity | HIGH | BOM baselines |
| **c14** | `c14_cross_domain_summary.py` | Complexity | HIGH | All domain results |
| **c15** | `c15_ml_theory_depth.py` | Complexity | HIGH | **WRITTEN** — 4 sub-analyses |
| **s10** | `s10_importance_downstream.py` | Similarity | HIGH | c04 design |
| **s11** | `s11_bayesian_optimization.py` | Similarity | MEDIUM-HIGH | GP infrastructure |
| **s12** | `s12_novelty_detection.py` | Similarity | MEDIUM | — |
| **s13** | `s13_clustering.py` | Similarity | MEDIUM | — |
| **n01** | `n01_activity_cliff_framework.py` | Noise | HIGH | s04, Dablander |
| **n02** | `n02_sample_weighting.py` | Noise | HIGH | — |
| **n03** | `n03_gp_noise_detection.py` | Noise | HIGH | n01 (cliff validation) |
| **n04** | `n04_noise_injection.py` | Noise | MEDIUM | n03 |
| **n05** | `n05_bindingdb_noise.py` | Noise | MEDIUM | BindingDB_Kd baselines |
| **n06** | `n06_noise_remediation.py` | Noise | LOW-MEDIUM | n03, n04 |
| **f05** | `f05_importance_correlation.py` | Framework | MEDIUM | — |
| **f06** | `f06_augmentation_test.py` | Framework | MEDIUM | — |
| **f07** | `f07_internal_evaluator.py` | Framework | MEDIUM | — |
| **f08** | `f08_cross_domain_analysis.py` | Framework | HIGH | f00-f07 outputs |

**Total: 17 new scripts** (6 Noise, 4 Similarity, 3 Complexity, 4 Framework)
**c15 already written.**

---

## OPIG Connections Summary

| OPIG Item | Complexity | Similarity | Noise | Status |
|-----------|-----------|------------|-------|--------|
| A1 Sort & Slice (Dablander) | — | Baseline rep | Foundation | **DONE** |
| B1 Trastuzumab | Forward compat task | — | — | **DONE** (loader) |
| B2 Graphinity DDG | Augmentation testing | — | Synth vs exp noise | **DONE** (loader) |
| B3 Humanness | Classification task | — | — | **DONE** (loader) |
| B5 Ginkgo | Small-sample regime | Small-sample similarity | Noisy small samples | **DONE** (loader) |
| C1 GAUCHE kernel | GP model improvement | BO input kernel | — | **DONE** (kernel) |
| C2 Q-SAVI shifts | Robustness under shift | Similarity under shift | — | **DONE** (script) |
| C4 Imrie TL | Model vs rep importance | — | — | **DONE** (script) |
| **#30 Activity Cliffs** | — | s04 foundation | **N-A foundation** | **TBD** |

**Key unfinished OPIG connection:** Dablander's activity cliff research (#30) is
the theoretical foundation for BOTH the Similarity chapter (s04) and the Noise
chapter (n01). This is the highest-priority OPIG integration that hasn't been
done yet. His formal definitions, cliff quantification methods, and the connection
between Sort & Slice and activity landscape analysis should be the backbone of
the cliff-related experiments.

---

## Thesis Chapter → Experiment Flow

```
COMPLEXITY                    SIMILARITY                  NOISE
    │                             │                         │
    ├─ c01 rep characterize       ├─ s01 problem statement  ├─ n01 cliff framework
    ├─ c02 complexity grid        ├─ s02 hybrid similarity  │   (uses s04, Dablander)
    ├─ c03 cross-dataset          ├─ s03 why it works       │
    ├─ c05 learning curves        ├─ s04 activity cliffs ───┤
    ├─ c06 capacity gradient      ├─ s05 kNN prediction     ├─ n02 sample weighting
    ├─ c07 partitions             ├─ s06 applicability dom  ├─ n03 GP noise detection
    ├─ c08 calibration            ├─ s07 noise bridge ──────┤   (uses hybrid space)
    ├─ c09 formal analysis        ├─ s08 optimal dimension  │
    ├─ c10 noise robustness ──────┤                         ├─ n04 noise injection
    ├─ c11 ensemble vs hybrid     ├─ s09 cross-dataset      ├─ n05 BindingDB ground truth
    ├─ c12 modality ablation      ├─ *s10 importance wts*   ├─ n06 remediation
    ├─ c15 ML theory depth        ├─ *s11 BO/active learn*  │
    ├─ *c13 forward compat*       ├─ *s12 novelty detect*   │
    ├─ *c14 cross-domain*         ├─ *s13 clustering*       │
    │                             │                         │
    └──────────────┬──────────────┴──────────┬──────────────┘
                   │                         │
              FRAMEWORK (f00-f08)       BOM (baselines)
              finalizes search          selects tasks
```

*Italicized = new scripts needed*

---

## APPENDIX: Research-Backed Experiment Designs

Research conducted 2026-03-16 across 5 parallel literature reviews. Each gap now has
concrete methods, baselines, metrics, and references.

---

### A1. c13 Forward Compatibility — Detailed Design

**Closest prior work:** Ko, Parkinson & Wang (Briefings in Bioinformatics, Jan 2026)
"Scalable Embedding Fusion" — greedy forward selection for protein embedding subsets.
Benchmark against their method. Also cite MULTIBENCH (NeurIPS 2021) evaluation axes.

**Novel contribution:** No existing paper formalizes KIRBy-style forward compatibility
(post-hoc feature fusion of frozen embeddings from arbitrary models). Apple's FCT
(CVPR 2022) and backward-compatible training (Shen et al., CVPR 2020) address
different problems (embedding retrieval, not feature fusion).

**5-condition ablation (per new rep r, per dataset d):**

| Condition | What | Tests |
|-----------|------|-------|
| A | New rep r + ridge | Raw embedding quality (linear) |
| B | New rep r + best model | Best-case single-rep |
| C | Best hybrid WITHOUT r | Current KIRBy ceiling |
| D | Best hybrid WITH r | Does r improve the hybrid? (D - C) |
| E | r + existing reps in hybrid | Does hybrid improve r? (E - B) |

**Controls (per new rep r):**
- Random Gaussian matrix (same dims as r) — permutation null
- Row-shuffled r — tests content vs dimensionality
- PCA of existing reps to same dims — tests redundancy vs novelty

**Statistical framework (Demsar, JMLR 2006):**
- Per-dataset: Nadeau-Bengio corrected t-test (5-fold CV)
- Cross-dataset within domain: Wilcoxon signed-rank (paired)
- All conditions across datasets: Friedman + Nemenyi post-hoc
- Effect sizes: Kendall's W for Friedman, rank-biserial for Wilcoxon
- **Minimum 5 datasets per domain** for Friedman power

**Key metrics to report:**
1. Delta performance: D - C (does new rep help hybrid?)
2. Augmentation gain: E - B (does hybrid help new rep?)
3. Allocation share: features_from_r / total_budget
4. Complementarity: 1 - max_correlation(r_features, existing_selected_features)
5. Win rate: fraction of datasets where condition improves

**Critical implementation detail:** Condition C and D must re-run `find_best_hybrid`
from scratch — don't just add r to existing best hybrid. The allocation may change.

---

### A2. s01/S-A Formal Problem Statement — Theoretical Foundation

**Distance concentration theorem (Beyer et al., ICDT 1999; Aggarwal et al., ICDT 2001):**
For iid features, the ratio `(max_dist - min_dist) / min_dist → 0` as dimensionality
increases. All pairwise distances become indistinguishable.

**Measurable:** Compute this ratio for ECFP4 (2048D) vs hybrid (50D) on each dataset.
Plot distributions of pairwise distances. Show hybrid distances have larger relative
spread. This is the theoretical justification for everything.

**Hubness problem (Radovanovic et al., JMLR 2010):**
In high dims, some points become "hubs" (appearing as k-NN of disproportionately many
others). Measurable: k-occurrence distribution skewness. Lower skewness in hybrid
space = less hubness = better kNN. Compare N_k skewness for ECFP4 vs hybrid.

**RBF kernel degradation (Bengio et al., 2005):**
Local kernel methods (GP, SVM with RBF) require exponentially more data in high dims
because `k(x,x') → constant` for all pairs. Scikit-learn docs explicitly state:
"GPs lose efficiency when features exceed a few dozen."

**Representation-dependent cliffs (Bajorath et al.):**
Only 20% of activity cliffs detected by fingerprint similarity are also detected by
MMP relationships. Different fingerprints identify different compounds as cliffs.
This means "is this compound a cliff?" depends on which representation you use.
**KIRBy's argument:** A hybrid combining multiple views produces more "consensus"
activity landscapes where true cliffs are consistently identified.

**Add to s01:**
1. Distance concentration ratio per rep and per dataset
2. Hubness (k-occurrence skewness) per rep
3. Cross-domain runs (antibody, TCGA) — does the problem exist everywhere?
4. All of these re-computed on hybrid reps for the s02 comparison

---

### A3. s10 Importance-Weighted Downstream Tasks — Methods

**RASAR / q-RASAR (Luechtefeld 2018; Banerjee & Roy 2022):**
Compute "similarity descriptors" from k nearest neighbors and use them AS FEATURES.
- Simple RASAR: kNN-weighted property transfer
- q-RASAR: compute concordance measures (gm, sm1, sm2) from neighbors, append as features
- Compare ECFP4-derived vs hybrid-derived similarity descriptors
- q-RASAR consistently outperforms standard QSAR (R²=0.85 vs lower)
- **Datasets:** All molecular regression (ADME-Fang, hERG, logd)

**eMOSAIC (Nature Machine Intelligence, 2025):**
Mahalanobis distance between query embedding and training clusters → features for
uncertainty prediction. Covariance estimation breaks in 2048D (ill-conditioned) but
works in 50D. Direct, clean demonstration of compact rep advantage.
- **Datasets:** DTA (KIBA, BindingDB_Kd), ADME datasets

**Virtual screening enrichment:**
Pure similarity ranking task. Given known active, rank library by similarity, compute
enrichment factor and AUROC. Tests whether hybrid similarity is better at identifying
actives.
- **Datasets:** hERG (binary), trastuzumab (binder/non-binder)

**Diversity selection quality (MaxMin algorithm):**
In high-D, MaxMin selections become near-random. In compact hybrid space, MaxMin
should produce genuinely diverse subsets that span activity space.
- Measure: activity-space coverage of selected subsets
- **Datasets:** ADME-Fang (2000-3000 compounds)

---

### A4. s11 Bayesian Optimization — 10-Condition Design

**10 conditions (representations × kernels):**

| # | Representation | Kernel | Dim | Rationale |
|---|---------------|--------|-----|-----------|
| 1 | ECFP4 | Tanimoto | 2048 | GAUCHE baseline |
| 2 | ECFP4 | RBF | 2048 | Shows RBF failure in high-D |
| 3 | Fragprints (FP+RDKit frag) | Tanimoto | ~2200 | GAUCHE best |
| 4 | Mordred | RBF | ~1613 | High-D continuous descriptors |
| 5 | Mordred PCA50 | RBF | 50 | Unsupervised dim reduction |
| 6 | **KIRBy hybrid** | **RBF** | **50** | **Main condition** |
| 7 | KIRBy hybrid | Tanimoto | 50 | Hybrid + domain kernel |
| 8 | Random projection | RBF | 50 | REMBO-style uninformed baseline |
| 9 | SAASBO on ECFP4 | auto | 2048 | HDBO state-of-art (BoTorch) |
| 10 | ECFP4 + DSP | RBF | 2048 | Hvarfner ICML 2024 fix |

**Key competitors from literature:**
- **MolDAIS** (Sorourifar et al., Digital Discovery 2025) — adaptive subspace selection
  DURING BO. KIRBy does it offline. The comparison tests pre-computed vs adaptive.
- **SAASBO** (Eriksson & Jankowiak, UAI 2021) — sparse subspace BO via half-Cauchy
  priors. Automatic implicit dimensionality reduction within GP. Available in BoTorch.
- **LOL-BO** (Maus et al., NeurIPS 2022) — LSBO state-of-art. VAE latent space +
  trust regions. KIRBy advantage: no VAE training, interpretable features.
- **Hvarfner et al. (ICML 2024)** — counterargument: vanilla BO works in high-D with
  dimensionality-scaled priors. Must include as baseline (condition 10).

**Protocol:**
1. Hold out 80% as "unlabeled library"
2. Initialize with 5% (seed set)
3. Build KIRBy hybrid on seed set (for conditions 6-7)
4. BO loop: GP fit → EI acquisition → reveal label → repeat
5. Budgets: 25, 50, 100, 200 oracle calls
6. Optionally rebuild hybrid every K iterations (unique to KIRBy)

**Metrics:**
- Primary: AUC of top-10 average property vs oracle calls (PMO-style)
- Secondary: #calls to find top-1% molecule, GP calibration (NLPD), final regret

**Datasets:** logd, herg, ADME datasets (1K-5K, good for BO scale)

**Key insight from literature:** KIRBy's advantage over LSBO is that hybrid space is
optimized for *prediction* (supervised feature selection), not *reconstruction*
(VAE objective). This avoids the surrogate-objective mismatch pathology.

---

### A5. n01 Activity Cliff Framework — Dablander Integration

**Dablander's key paper:** "Exploring QSAR Models for Activity-Cliff Prediction"
(J. Cheminformatics 15:47, 2023). Code: github.com/MarkusFerdinandDablander/
QSAR-activity-cliff-experiments

**Key findings to build on:**
1. QSAR models are weak at classifying ACs unless one compound's activity is known
2. GINs competitive with ECFPs for AC classification but NOT for general QSAR
3. Three-target study: D2, Factor Xa, SARS-CoV-2 Mpro (clean MMP-curated cliffs)

**MoleculeACE benchmark (van Tilborg et al., JCIM 2022):**
24 models × 30 targets. Introduces RMSE_cliff metric. All methods struggle on cliffs.
Descriptor-based ML outperforms GNNs on cliff compounds. Code: github.com/molML/MoleculeACE

**Experiment components for n01:**
1. **Consensus cliff identification:** Identify cliffs under multiple rep distance
   metrics (ECFP4, MACCS, PDV, learned LM). Show that hybrid distance produces cliff
   identifications that are more stable across underlying fingerprints than any single
   fingerprint. This addresses Bajorath's finding that only 20% of fingerprint-detected
   cliffs overlap with MMP-detected cliffs.
2. **SALI landscape smoothing:** Compute pairwise SALI under each rep AND under hybrid
   distance. Show that hybrid SALI landscapes are smoother (fewer spurious cliffs) while
   preserving true cliffs. Use `iCliff` (Lopez-Perez & Miranda-Quintana, JCIM 2025)
   for the set-level metric.
3. **Cliff prediction from hybrid features:** Given a compound, predict whether it will
   form an activity cliff with any other compound — using only features (not knowing
   the activity of the paired compound). Test whether hybrid features are more predictive
   of cliff-forming tendency than individual reps.
4. **Hybrid cliff mitigation:** Show that in hybrid space, fewer compound pairs qualify
   as cliffs (because the hybrid separates them). Count cliff pairs before and after
   hybridization at the same similarity threshold.

**Dablander's datasets (available on GitHub):** D2, Factor Xa, SARS-CoV-2 Mpro with
curated MMP-cliffs — use as ground-truth validation alongside KIRBy's molecular datasets.

---

### A6. n03 GP Noise Detection — 5-Method Comparison

**5 conditions ordered by complexity:**

| # | Method | Implementation | Compute |
|---|--------|----------------|---------|
| 1 | **RF OOB residuals** | Fit RF, flag |OOB residual| > 2σ | O(N·trees) |
| 2 | **GP-LOO analytical** | Rasmussen-Williams Eq. 5.12: μ_i = y_i - α_i/K⁻¹_ii | O(N³), free after matrix inversion |
| 3 | **Heteroscedastic GP-LOO** | Immer et al. (2021): per-sample noise variance via MLE multiplicative updates | O(N³) |
| 4 | **Cleanlab regression** | pip install cleanlab; cross-validated label quality scores | Cost of 5-fold CV |
| 5 | **KNN-Shapley** | Exact Shapley values for kNN in O(N log N). Negative values = harmful samples | O(N log N) |

**All conditions run in hybrid space (50D) AND in ECFP4 space (2048D).** The key
test is whether hybrid space separates cliffs from noise that ECFP4 space cannot.

**GP-LOO key reference:** Rasmussen & Williams Chapter 5, Eq. 5.12. LOO predictions
come "for free" once the kernel matrix is inverted — no retraining. Flag samples where
standardized residual |y_i - μ_i| / σ_i exceeds threshold (2-3σ).

**Robust GP alternatives:**
- Student-t likelihood GP (Jylanki et al., JMLR 2011) — heavier tails, less sensitive
  to outliers. Available as `StudentTLikelihood` in GPyTorch.
- Robust GP via iterative trimming (Li et al. 2020) — pip install robustgp
- Sparse variational contaminated noise GP (2024) — for BindingDB_Kd scale (52K)

**Validation strategy:**
1. **Synthetic noise:** Inject controlled noise (5/10/20/30% of samples, 1 log unit
   perturbation) into clean datasets. Measure precision/recall of each method.
2. **Activity cliff cross-reference:** True cliffs (from n01) should NOT be flagged.
   Compute false positive rate: cliffs wrongly flagged as noise.
3. **BindingDB ground truth (n05):** Replicate measurements as noise ground truth.
4. **Downstream benefit:** Remove flagged samples, retrain, measure R² improvement.

**The "killer comparison":** In ECFP4 space, GP-LOO flags activity cliffs as noise
(they look identical — similar structure, different activity). In hybrid space, GP-LOO
correctly identifies cliffs as non-anomalous (they're separated) and only flags
genuine label errors. This validates the thesis: "hybrid representations enable
noise-aware training that respects activity cliffs."

---

### A7. n05 BindingDB Ground-Truth Noise — Quantification

**Key reference:** Landrum & Riniker (JCIM 2024), "Combining IC50 or Ki Values from
Different Sources Is a Source of Significant Noise." Code: github.com/rinikerlab/
overlapping_assays

**Ground-truth noise levels (from Landrum & Riniker):**
- Minimally curated: 65% differ by >0.3 log units, 27% by >1 log unit, τ = 0.51
- Maximally curated: 48% differ by >0.3 log units, 13% by >1 log unit, τ = 0.71
- Individual measurement σ: ~0.62 pIC50, ~0.49 pKi (from Kramer et al. 2012)

**The 0.3 log unit threshold** is the standard boundary between "acceptable agreement"
and "discrepant" in cheminformatics.

**Experiment design for n05:**
1. In KIRBy's BindingDB_Kd dataset, identify compound-target pairs with replicate
   measurements (same InChIKey + same UniProt target, different assay entries)
2. Compute per-pair standard deviation as ground-truth noise
3. For each pair, take the measurement that deviates most from the mean → "noisy sample"
4. Run all 5 noise detection methods (from n03) on the full dataset
5. Compute AUROC: can each method identify the truly noisy samples?
6. Compare hybrid space vs ECFP4 space

**Theoretical R² ceiling (Wallach & Heifets, J. Cheminf. 2021):**
There is a theoretical R² upper bound as a function of noise variance and label variance.
Models CAN outperform their noisy training data (LOO predictions average over noise).
Compute this ceiling per dataset using replicate-estimated noise, then show that
noise-aware GP training approaches it.

---

### A8. s12 Novelty Detection — eMOSAIC-Inspired

**eMOSAIC (Nature Machine Intelligence, 2025):**
Cluster training embeddings, compute Mahalanobis distance from test to each cluster.
These distances become features for predicting absolute residuals (uncertainty).
Covariance estimation fails in 2048D, works in 50D.

**Experiment design:**
1. Cluster training set (k-means, k=5-10) in ECFP4 vs hybrid space
2. For test compounds: compute Mahalanobis distance to each cluster centroid
3. Measure correlation between Mahalanobis distance and |prediction error|
4. Compare: which representation space produces a more informative novelty score?
5. Also: Isolation Forest, Local Outlier Factor at various dimensions

**Scaffold splits as natural OOD:** Scaffold-split test compounds are inherently
scaffold-novel. Measure whether OOD detectors correctly flag them as novel, and
whether this correlates with prediction degradation.

---

### A9. Key References (Consolidated)

**Similarity / Distance Theory:**
- Beyer et al. (ICDT 1999) — "When Is Nearest Neighbor Meaningful?"
- Aggarwal et al. (ICDT 2001) — distance concentration in high dimensions
- Radovanovic et al. (JMLR 2010) — hubness problem
- Bengio et al. (2005) — curse of dimensionality for local kernel machines

**Activity Cliffs:**
- Guha & Van Drie (JCIM 2008) — SALI original paper
- Dablander et al. (J. Cheminf. 2023) — QSAR vs AC prediction
- van Tilborg et al. (JCIM 2022) — MoleculeACE benchmark
- Stumpfe & Bajorath (ACS Omega 2019) — evolving concept of activity cliffs
- Lopez-Perez & Miranda-Quintana (JCIM 2025) — iCliff (fixes SALI singularity)

**Noise Detection:**
- Immer et al. (2021) — GP-LOO with heteroscedastic noise
- Rasmussen & Williams Ch. 5 — analytical GP-LOO
- Northcutt et al. (JAIR 2021) — Confident Learning / Cleanlab
- Ghorbani & Zou (ICML 2019) — Data Shapley / KNN-Shapley
- Landrum & Riniker (JCIM 2024) — IC50 noise quantification
- Kramer et al. (PLOS ONE 2012) — mixed IC50 comparability

**Bayesian Optimization:**
- Griffiths et al. (NeurIPS 2023) — GAUCHE
- Eriksson & Jankowiak (UAI 2021) — SAASBO
- Sorourifar et al. (Digital Discovery 2025) — MolDAIS
- Hvarfner et al. (ICML 2024) — vanilla BO in high dimensions
- Maus et al. (NeurIPS 2022) — LOL-BO (LSBO state-of-art)
- Gao et al. (NeurIPS 2022) — PMO benchmark

**Forward Compatibility:**
- Ko et al. (Briefings in Bioinformatics 2026) — embedding fusion
- Demsar (JMLR 2006) — statistical comparison of classifiers
- Ponzoni et al. (Scientific Reports 2017) — hybridizing feature selection + learning
- Ramanujan et al. (CVPR 2022) — Apple forward compatible training

**Downstream Applications:**
- Luechtefeld et al. (Tox Sci 2018) — RASAR
- Banerjee & Roy (JCIM 2022) — q-RASAR
- Sahlin et al. (J. Cheminf. 2017) — reliability-density AD
- Alvarsson et al. (J. Pharm. Sci. 2021) — conformal prediction in drug discovery
