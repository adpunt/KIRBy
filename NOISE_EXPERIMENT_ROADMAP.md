# Noise Experiment Roadmap

**Purpose:** Comprehensive plan for all noise mitigation experiments, incorporating the
confirmation report LNL research, similarity-assumption insights, and noise estimation
work into the existing N01-N08 framework.

**Date:** 2026-03-27

---

## Background

Three bodies of prior work feed into this roadmap:

1. **Confirmation report LNL survey** — 12 LNL methods tested on QM9 (10K, scaffold
   split, Gaussian noise sigma 0.0-1.0). Key findings: distance-augmented methods
   uniformly fail; DivideMix (86.7% retention), small-loss selection (84.2%), uncertainty
   curriculum (82.7%), and HetGP (79.0%) were most promising. Representation choice
   explained less variance under LNL than in standard QSAR.

2. **Confirmation report similarity-assumption insight** — LNL methods that relied on
   neighbor similarity failed catastrophically (all 20 worst configs used distance
   metrics). Hypothesis: this is because standard representations have poor
   similarity-activity correlation. Hybrids should fix this — providing a stronger
   definition of "neighboring" for methods that need it.

3. **Confirmation report noise estimation & smoothing** — Extensive grid of estimators
   × smoothing/weighting methods × alpha × distance metrics × representations. Only 8/860
   configs produced positive recovery at both noise levels. Best robust config:
   PDV/euclidean/local_variance_ratio/exponential/alpha=0.1 (+14.3% avg recovery).
   Conservative alpha (0.1) massively outperforms aggressive values. Feature importance
   weighting for distance computation showed promise.

---

## Existing Experiments (N01-N08)

| ID | Script | What it does | Status |
|----|--------|-------------|--------|
| **N01** | `n01_activity_cliff_framework.py` | Dablander cliff analysis: consensus cliff ID, SALI landscape, cliff mitigation, cliff-stratified prediction | Written |
| **N02** | `n02_sample_weighting.py` | Importance-weighted sample weights: 5 conditions (base ±weights, hybrid ±weights, cliff-aware) | Written |
| **N03** | `n03_sas_landscape.py` | SAS 4-quadrant maps: zone shift under hybridization, SAS-BO bridge | Written |
| **N04** | `n04_gp_loo_detection.py` | GP-LOO noise detection in hybrid vs concat vs PCA: cliff FP rate comparison | Written |
| **N05** | `n05_noise_injection_benchmark.py` | 15 binary detectors × sigma levels × concat/hybrid distance | Written |
| **N06** | `n06_bindingdb_ground_truth.py` | BindingDB replicate disagreement as ground-truth noise | Written |
| **N07** | `n07_remediation.py` | 6 remediation strategies post-detection | Written |
| **N08** | `n08_estimation_survey.py` | 15 magnitude estimators, cliff discrimination, cross-correlation | Written |

### What N01-N08 Cover Well
- Activity cliff framework and its connection to noise (N01, N03, N04)
- Binary noise detection at various sigma levels (N05)
- Ground-truth noise validation (N06)
- Basic remediation comparison (N07)
- Estimator survey with cliff discrimination (N08)
- Hybrid vs concat distance for detection (N04, N05)
- Importance-weighted sample weighting (N02)

### What N01-N08 Do NOT Cover
- The 12 LNL training-loop methods (Meta Weight Net, etc.) — only simple detectors exist
- Hybrid distance for LNL methods' internal similarity computation
- The full estimator × smoothing/weighting × alpha × distance × rep grid
- Feature importance-weighted distance for noise estimation
- Recovery rate evaluation protocol (clean → noisy → denoised)
- Multiple noise distributions
- Varying hybrid budgets for noise-related similarity

---

## New Experiments

### N09: LNL Method Survey with Hybrids

**Motivation:** The confirmation report tested 12 LNL methods on QM9 using individual
representations. These methods modify the TRAINING PROCESS (meta-learning, curriculum,
iterative refinement) rather than just detecting noise post-hoc. The key question:
do hybrids change which LNL methods work best?

**Methods to implement (12):**

| Method | Category | Key mechanism |
|--------|----------|--------------|
| Meta Weight Net | Meta-learning | Validation-based weight updates every 5 epochs |
| Small-Loss Selection | Sample selection | One-time selection of 70% smallest MSE samples |
| Confident Learning | Sample selection | High error + low uncertainty flagging |
| Distance-Based Selection | Sample selection | Outlier detection via molecular similarity |
| MentorNet | Curriculum | Online mentor; 7D input: loss + history + uncertainty + distance |
| Uncertainty Curriculum | Curriculum | Staged inclusion: 25% → 50% → 75% → 100% |
| DivideMix | Iterative | GMM separation, co-teaching (regression adaptation) |
| Multistage Cleaning | Iterative | Progressive 10%/stage × 3 stages removal |
| Early Learning | Iterative | Cross-epoch reweighting with geometric decay |
| Contrast-to-Divide | Iterative | Consistency via prediction variance under perturbation |
| Heteroscedastic GP | GP extension | GP for mean + NN for noise variance |
| Evidential Kernel | GP extension | Multitask GP predicting NIG parameters |

**Conditions (per method):**
1. Base rep (ECFP4) + LNL method (confirmation report baseline)
2. Hybrid rep + LNL method (does the hybrid change which methods win?)
3. No LNL (pure baseline for that rep)

**Design:**
- Datasets: logd, caco2, herg, qm9 (molecular focus — matches confirmation report)
- Noise: Gaussian at sigma = {0.0, 0.1, 0.3, 0.6, 1.0} (confirmation report levels)
- Reps for condition 1: ECFP4, PDV, SNS (match confirmation report)
- Hybrid for condition 2: greedy, budget=100, standard config
- Model: DNN (MLP) as base architecture (LNL methods need gradient-based training)
- Metric: retention = R²(noisy) / R²(clean), absolute R²
- Scaffold split, 5-fold CV

**Key questions:**
- Does the ranking of LNL methods change with hybrid input?
- Do methods that failed with individual reps succeed with hybrids?
- Does representation choice still explain less variance under LNL?

**Dependencies:** None (self-contained). LNL methods need new implementations in
`src/kirby/utils/lnl_methods.py` (they are training-loop modifications, not just
post-hoc detectors like the existing noise_mitigation_methods.py).

**Script:** `tests/experiments/noise/n09_lnl_survey.py`

---

### N10: Hybrid Neighborhood for Similarity-Dependent LNL

**Motivation:** The confirmation report's most striking finding: ALL 20 worst-performing
LNL configurations used distance metrics. Not a single distance-augmented config
achieved both high baseline (R² > 0.6) and high noise retention (> 80%). The
hypothesis: standard representations have poor similarity-activity correlation
(shown in S01), so using them for neighbor-based reasoning in LNL methods is
fundamentally flawed. Hybrids should fix this.

**The key distinction:** N09 tests hybrids as INPUT to LNL methods. N10 tests hybrids
as the SIMILARITY SPACE for LNL methods that need neighborhood computation. These are
different — a method might use ECFP4 as input features but compute neighbors in hybrid
space.

**Methods that rely on similarity (from the confirmation report):**
- Distance-Based Selection (outlier detection via molecular similarity)
- MentorNet (10-NN distance as input feature)
- Small-Loss Selection (optional distance-based outlier removal)
- Confident Learning (optional distance filtering)
- All neighbor-based noise estimators (neighbor_residual, local_variance_ratio, AC-aware)
- kNN-based label smoothing
- kNN-based sample weighting

**Conditions:**

For each similarity-dependent method:
1. Standard distance (ECFP4 Tanimoto / Euclidean on raw rep) — confirmation report baseline
2. Hybrid distance, small budget (25) — cheap, tests if even minimal hybrid helps
3. Hybrid distance, medium budget (100) — standard config
4. Hybrid distance, large budget (200) — tests if more budget helps similarity
5. Importance-weighted distance on base rep — does weighting alone help?
6. Importance-weighted distance on hybrid — combined approach

**Design:**
- Datasets: logd, caco2, herg (molecular, matches confirmation report)
- Noise: sigma = {0.3, 0.6} (the two levels from the estimation experiments)
- For each condition: compute distances, pass to method, evaluate downstream
- Metric: retention, recovery rate, and direct R² comparison
- Standard ECFP4 + RF for downstream evaluation (matches confirmation report protocol)

**Key questions:**
- Does hybrid distance rescue distance-dependent LNL methods?
- Is there a budget threshold where distance methods start working?
- Does importance weighting help independently of hybridization?
- Can any distance-dependent method achieve R² > 0.6 AND retention > 80% with hybrid distance?

**Confirms or refutes:** "Similar molecules have similar properties" is only true
when similarity is measured in hybrid space.

**Dependencies:** N09 (for LNL method implementations). Can share infra.

**Script:** `tests/experiments/noise/n10_hybrid_neighborhood_lnl.py`

---

### N11: Noise Estimation × Remediation Grid

**Motivation:** The confirmation report tested 860 configurations of estimator ×
smoothing/weighting × alpha × distance × representation, finding only 8 robust configs.
The existing N07 tests 6 remediation strategies with a SINGLE estimator
(ActivityCliffAwareEstimator) and single noise fraction. N08 surveys estimators but
doesn't test downstream smoothing/weighting. This experiment fills the gap: the FULL
grid from the confirmation report, extended with hybrid representations.

**Estimators (from confirmation report + N08):**

Neighbor-based:
- neighbor_residual
- local_variance_ratio
- activity_cliff_aware (AC-aware)

Model-based:
- cross_val_residual
- ensemble_disagreement

Uncertainty-based:
- gaussian_process
- heteroscedastic_nn
- conformal
- deep_ensemble
- bayesian_rf
- quantile_regression
- ngboost

**Smoothing methods (from confirmation report):**
- proportional: y_smooth = y + alpha × score_norm × (target - y)
- threshold: only correct samples above (1-alpha) percentile
- soft_threshold: sigmoid-like correction strength
- confidence: alpha × sigma / (1 + alpha × sigma) correction

**Weighting methods (from confirmation report):**
- inverse: w = 1 / (1 + alpha × score_norm)
- exponential: w = exp(-alpha × score_norm)
- soft: w = 1 - alpha × score_norm², clipped [0.1, 1.0]

**Alpha values:** {0.1, 0.3, 0.5, 0.7}

**Distance metrics for neighbor-based estimators:** euclidean, manhattan, cosine

**Representations:**
- ECFP4 (baseline)
- PDV (best single rep from confirmation report)
- Hybrid (budget=100)
- Feature importance-weighted variants of the above

**Design:**
- Datasets: logd, caco2, herg, qm9
- Noise: sigma = {0.3, 0.6} (match confirmation report)
- Evaluation: RF on ECFP4 trained on clean → noisy → denoised
- Metric: recovery rate = (R²_denoised - R²_noisy) / (R²_clean - R²_noisy)
- Report: % positive recovery, mean recovery, best recovery per condition

**Key questions:**
- Does the best robust config (PDV/euclidean/local_variance_ratio/exp/alpha=0.1) replicate?
- Does hybrid outperform PDV for noise estimation?
- Does feature importance weighting improve estimation?
- Which estimator × smoothing/weighting combos are robust across datasets (not just QM9)?
- Can we push recovery above the +14.3% avg from the confirmation report?

**Dependencies:** N08 (estimator implementations already exist). Smoothing/weighting
functions partially exist in noise_estimation.py (smooth_by_scores, weights_from_scores)
but need extending to match all confirmation report variants.

**Script:** `tests/experiments/noise/n11_estimation_remediation_grid.py`

---

### N12: Feature Importance-Weighted Distance for Noise Estimation

**Motivation:** The confirmation report investigated whether feature importance scores
improve distance computation for noise detection. The idea: scale features by
sqrt(importance) before computing distances, so task-relevant features dominate
neighborhood structure. This was partially explored but deserves systematic treatment,
especially combined with hybrids.

**Conditions:**
1. Unweighted distance on base rep (baseline)
2. Importance-weighted distance on base rep
3. Unweighted distance on hybrid
4. Importance-weighted distance on hybrid
5. Importance-weighted distance on hybrid (hybrid's own feature importance, not base rep's)

**Estimators tested per condition:**
- neighbor_residual (k=5, 10, 20)
- local_variance_ratio (k=5, 10, 20)
- activity_cliff_aware (k=5, 10, 20)

**Importance methods:** lightgbm_gain, permutation, treeshap (if available)

**Design:**
- Datasets: logd, caco2, herg
- Noise: sigma = {0.3, 0.6}
- Feature weighting: x' = x * sqrt(max(w, 0)), normalized
- Evaluation: same recovery rate protocol as N11

**Key questions:**
- Does importance weighting improve neighbor-based noise estimation?
- Is hybrid + importance weighting redundant? (hybrids already select important features)
- Which importance method produces the best noise-estimation distance?

**Script:** `tests/experiments/noise/n12_importance_weighted_distance.py`

---

### N13: Alternative Noise Distributions

**Motivation:** The confirmation report used only Gaussian noise injection. Real
measurement noise in QSAR may follow different distributions (e.g., multiplicative
noise from assay variability, heavy-tailed noise from outlier measurements, uniform
noise from binned measurements). Different noise distributions may favor different
detection/estimation methods.

**Noise types:**
- Gaussian (baseline, sigma = {0.1, 0.3, 0.6} × label range)
- Uniform (width = {0.1, 0.3, 0.6} × label range)
- Multiplicative (relative noise: y_noisy = y × (1 + N(0, sigma)))
- Heavy-tailed (Student-t, df=3, scaled to match Gaussian sigma)
- Structured / partial (inject noise into random 10%/30%/50% of samples only)

**Methods tested:**
- Top methods from N09 (DivideMix, uncertainty curriculum, HetGP, small-loss)
- Top estimators from N11 (local_variance_ratio, ngboost, gaussian_process)
- Binary detection methods from N05 (for comparison)

**Representations:** ECFP4, hybrid (budget=100)

**Design:**
- Datasets: logd, qm9
- Evaluation: retention and recovery rate
- Key analysis: method × noise distribution interaction

**Key question:** Are the methods that work for Gaussian noise also robust to other
noise distributions? Does the hybrid advantage persist?

**Script:** `tests/experiments/noise/n13_noise_distributions.py`

---

## Integration with Existing Framework

### How new experiments connect to N01-N08

```
N01 (cliff framework)──────────────────────────────── Cliff definitions used by N10
  │
N02 (sample weighting) ─── extended by ──── N11 (full smoothing/weighting grid)
  │                                           N12 (importance-weighted distance)
N03 (SAS landscape)────────────────────────── Context: zone analysis explains N10 results
  │
N04 (GP-LOO detection)─── feeds into ────── N09 (HetGP method), N10 (GP in hybrid space)
  │
N05 (15 detectors)──────── extended by ──── N09 (12 LNL methods), N13 (noise distributions)
  │
N06 (BindingDB ground truth)────────────── Validate N09, N10, N11 on real noise
  │
N07 (remediation)───────── replaced by ──── N11 (full grid with all estimators)
  │
N08 (estimator survey)──── feeds into ────── N11 (estimators + downstream evaluation)
                                              N12 (importance-weighted variants)
```

### Connection to Similarity Chapter

N10 is the bridge: it directly tests whether the similarity chapter's main claim
(hybrids improve similarity-activity correlation) has practical consequences for noise
mitigation. The narrative: S01 shows the problem (bad similarity), S02 shows hybrids
fix it, N10 shows this fix enables noise mitigation methods that previously failed.

### Connection to Complexity Chapter

N09 tests whether the complexity chapter's main claim (representation choice matters
more than model choice) holds under LNL methods. The confirmation report found that
representation choice explains LESS variance under LNL — does this change with hybrids?

---

## Implementation Status

**ALL DONE (2026-03-28).** N09-N13 implemented and verified.

| Experiment | Status | Script | Lines |
|-----------|--------|--------|-------|
| N09 (LNL survey) | **DONE** | `n09_lnl_survey.py` | ~290 |
| N10 (hybrid neighborhood) | **DONE** | `n10_hybrid_neighborhood_lnl.py` | ~340 |
| N11 (estimation grid) | **DONE** | `n11_estimation_remediation_grid.py` | ~310 |
| N12 (importance-weighted) | **DONE** | `n12_importance_weighted_distance.py` | ~280 |
| N13 (noise distributions) | **DONE** | `n13_noise_distributions.py` | ~300 |

New utility: `src/kirby/utils/lnl_methods.py` (~830 lines, 12 LNL methods + registry).

---

## Utility Code (DONE)

### `src/kirby/utils/lnl_methods.py` — IMPLEMENTED

12 LNL methods with shared DNN infrastructure. API:

```python
from kirby.utils.lnl_methods import get_lnl_method, LNL_REGISTRY

method = get_lnl_method('dividemix', epochs=200)
result = method.train_and_predict(X_train, y_train, X_val, y_val)
# result = {'y_pred': np.ndarray, 'metadata': dict}
```

Methods: small_loss, confident_learning, distance_selection, multistage_cleaning,
early_learning, uncertainty_curriculum, dividemix, meta_weight_net, mentornet,
contrast_to_divide, heteroscedastic_gp, evidential_kernel.

### `src/kirby/utils/noise_estimation.py` — NO CHANGES NEEDED

All smoothing (proportional, threshold, soft_threshold, confidence) and weighting
(inverse, exponential, soft) already existed in `smooth_by_scores` and
`weights_from_scores`. N11 uses them directly.

### Distance computation — NO CHANGES NEEDED

`compute_distance_matrix` already supports `feature_weights` parameter. N10 and N12
build weighted spaces externally and pass to estimators.

---

## Datasets per Experiment

| Experiment | Molecular | Antibody | DTA | Notes |
|-----------|-----------|----------|-----|-------|
| N09 | logd, caco2, herg, qm9 | | | Match confirmation report |
| N10 | logd, caco2, herg | | | Focus: molecular (similarity well-defined) |
| N11 | logd, caco2, herg, qm9 | | | Extend confirmation report |
| N12 | logd, caco2, herg | | | Targeted molecular |
| N13 | logd, qm9 | | | Two datasets sufficient |

Antibody and DTA extensions can follow once molecular experiments establish the
methodology. N06 (BindingDB ground truth) provides the DTA validation path.

---

## Confirmation Report Results to Replicate

These results from the confirmation report should be reproduced (or improved) by
the new experiments:

1. **DivideMix 86.7% retention** (N09) — does this hold with hybrids?
2. **Distance methods uniformly fail** (N10) — does hybrid distance rescue them?
3. **PDV/euclidean/local_variance_ratio/exp/alpha=0.1 = +14.3% avg recovery** (N11) — does hybrid beat PDV?
4. **Alpha=0.1 dominates** (N11) — does this hold across datasets?
5. **Only 8/860 configs robust across both noise levels** (N11) — can hybrids increase this count?
6. **HetGP 79% retention** (N09) — stable across kernels, does hybrid input help?
7. **Label smoothing fails >95% of configs** (N11) — is this fixable with better similarity?
8. **Hybrid recovery rate 27.8% positive at sigma=0.6** (N11) — can we improve this?

---

## SLURM Considerations

N09 is the most expensive (12 methods × conditions × datasets × noise levels × folds).
Estimate: ~24h per dataset with DNN training. Run molecular datasets in parallel.

N11 has the most configurations (12 estimators × 7 smoothing/weighting × 4 alpha × 3
distance × 4 reps = 4,032 per sigma per dataset). But each config is cheap (just
estimation + RF retraining). Estimate: ~2h per dataset.

N10 is moderate (7 methods × 6 conditions × 3 datasets × 2 sigma). ~4h per dataset.

N12 and N13 are small. ~1h each.
