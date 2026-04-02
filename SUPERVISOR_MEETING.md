# Supervisor Meeting — Presentation Notes

Run `python tests/supervisor_presentation.py | tee presentation_output.txt` on the server
to generate the per-dataset results tables, hybrid heatmaps, and evidence section.

This document covers everything else needed for the meeting.

---

## 1. Allocation Strategies

KIRBy's hybrid search distributes a feature budget across multiple representations.
The allocation strategy decides HOW that budget gets split.

### Fast Tier (3 strategies — used for screening)
These were selected by F12 as the minimum set that captures most of the performance:

| Strategy | Description |
|----------|-------------|
| `entropy_1.0` | Performance-weighted with diversity penalty. Scores each rep by its predictive performance, then penalizes reps that are informationally redundant (high mutual information with already-selected reps). Diversity penalty strength = 1.0. Avoids allocating budget to reps that provide similar signal. |
| `top_3` | Allocates budget to only the 3 best-performing reps, proportional to their scores. All other reps get zero budget. Simple and aggressive — works when a few reps dominate. |
| `greedy_fill_s25` | Greedy forward selection (step size 25 features) with patience-based early stopping, then fills remaining budget from top reps. Iteratively tries adding 25 features from the rep that improves validation score most. Stops when no improvement for 3 rounds, then distributes leftover budget proportionally to the best reps found so far. |

### Full Tier (31 strategies — used for publication)

**Performance-Weighted Family** (22 strategies):
All start from the same idea: score each rep by its individual predictive performance, then convert scores to budget fractions using a weighting scheme.

| Strategy | How it converts scores to budget shares |
|----------|----------------------------------------|
| `performance_weighted` | Linear proportional to raw scores |
| `softmax_T0.5` | Softmax with temperature 0.5 — very peaked, top rep gets most budget |
| `softmax_T2` | Softmax T=2 — moderate spread |
| `softmax_T10` | Softmax T=10 — nearly uniform across reps |
| `power_2` | Scores raised to power 2 before normalizing — squares the gap between good and bad reps |
| `power_3` | Scores raised to power 3 — even more aggressive concentration |
| `thresh_0.3` | Zero out reps scoring below 30th percentile, then proportional |
| `thresh_0.5` | Zero out reps below median score |
| `thresh_0.7` | Zero out reps below 70th percentile — only top 30% get budget |
| `top_2` / `top_3` / `top_5` | Only top-k reps get budget, rest get zero |
| `rank_linear` | Budget proportional to rank (best=N, worst=1) instead of raw score |
| `rank_exp_d0.1` / `d0.3` / `d0.5` | Exponentially decaying rank weights. d=0.1 is nearly uniform, d=0.5 drops fast |
| `score_gap_0.2` to `0.7` | Only reps within X of the best rep's score get budget. gap=0.2 is tight (only near-best reps), gap=0.7 is loose |
| `entropy_0.5` / `1.0` / `2.0` | Performance-weighted with diversity penalty (see fast tier). Higher values penalize redundancy more aggressively |

**Greedy Family** (9 strategies):
Iterative forward selection — actually builds the hybrid incrementally and measures which features help.

| Strategy | How it works |
|----------|-------------|
| `greedy_forced_s5` / `s10` / `s25` | No patience (no early stopping). Keeps adding features until budget is full. Step size = how many features to add per round. Smaller steps = finer search but slower |
| `greedy_fill_s5` / `s10` / `s25` | With patience-based early stopping + fills remaining budget. More efficient — stops when gains plateau, then distributes leftover |
| `dom_greedy_f0.7` / `f0.9` | Dominant greedy. Allocates fraction f of budget to the single best rep, then greedy-searches the remaining (1-f) across all reps. f=0.9 means 90% goes to the best rep, only 10% is searched |

---

## 2. Experiments Testing Hybrids (Pending / Blocked by Hybrid Searches + Caching)

### Complexity Experiments
- **C02 Complexity Grid**: Tests which (model, budget, allocation, importance) combination is optimal per dataset. The core experiment — every dataset gets a full grid search.
- **C03 Blueprint Transfer**: Takes the winning hybrid config from one dataset and applies it to others. Tests whether the optimal allocation/budget generalizes or is dataset-specific.
- **C05 Learning Curves**: Varies training set size and measures how hybrid improvement scales. Identifies the minimum sample count where hybrids start helping.
- **C06 Within-Family Gradient**: Compares hybrid performance across model variants within the same family (e.g., RF with 100 vs 500 vs 1000 trees). Tests whether optimal budget changes with model capacity.
- **C07 Data Complexity Partitions**: Splits data into easy/hard subsets and checks if optimal hybrid config differs by subpopulation difficulty.
- **C08 Calibration**: Measures prediction uncertainty (calibration curves, expected calibration error) at optimal vs suboptimal budgets. Tests whether hybrid selection introduces miscalibration.
- **C09 Formal Analysis**: ANOVA decomposition of variance across (model, budget, allocation, importance). Information retention analysis. Effective rank of hybrid feature matrices. Bias-variance decomposition.
- **C10 Noise Robustness**: Injects label noise at increasing levels and measures whether hybrid allocators remain stable or degrade. Computes Noise Degradation Score (NDS).
- **C11 Ensemble vs Hybrid**: Head-to-head comparison of feature-level fusion (hybrid) vs model-level fusion (ensemble of single-rep models). Tests whether combining features beats combining predictions.
- **C12 Modality Ablation (TCGA only)**: Systematically removes omics modalities (gene expression, methylation, CNV, WSI, mutations) and measures which combinations benefit from hybridization.
- **C13 Forward Compatibility**: Tests whether newly added representations (ESM3, SaProt, Boltz-2, etc.) integrate seamlessly into hybrid search or break existing rankings.
- **C15 ML Theory Depth**: Concentration of measure in high-dimensional hybrid spaces. Kernel degradation analysis. Double descent curves for hybrid features.
- **C16 Feature Selection as Regularization**: Tests whether hybrid budget (feature count limit) makes explicit model regularization (L1, L2, dropout) redundant.
- **C17 Information-Theoretic Decomposition**: Compares importance methods (mutual information, MRMR, RFE) for feature selection within hybrid allocation.
- **C18 Diversity vs Depth**: Fixed budget tradeoff — is it better to take a few features from many reps (diversity) or many features from few reps (depth)?
- **C19 Importance-Weighted Training**: Uses feature importance scores as sample weights during downstream model training.
- **C21 Single-Rep Augmentation**: Runs find_best_hybrid on individual reps to discover which augmentation features help each rep independently.
- **C22 Explainability**: SHAP/TreeSHAP attribution on hybrid features to understand which original reps contribute most to predictions.
- **C23 Evaluator-Model Matching**: Validates whether matching the internal evaluator to the downstream model (e.g., RF evaluator for RF downstream) outperforms mismatched pairs.

### Similarity Experiments
- **S02 Property-Aware Similarity**: Finds optimal distance-space weights using hybrid importance. Instead of Euclidean distance in raw rep space, weights features by their predictive importance.
- **S03 Feature Concentration**: Analyzes WHY hybrid similarity works — measures mutual information density in weighted vs unweighted feature spaces.
- **S04 Activity Cliffs**: Measures RMSE specifically on activity cliff pairs (structurally similar but different activity). Tests whether hybrid distance better identifies these.
- **S05 kNN Prediction**: Uses hybrid-weighted distance for k-nearest-neighbor prediction. Tests whether the distance metric itself (not just the features) improves predictions.
- **S06 Applicability Domain**: Tests whether hybrid distance from training data correlates with prediction uncertainty. Goal: reliable confidence estimation.
- **S07 Noise Bridge**: Distinguishes real activity cliffs from noisy measurements using hybrid distance + importance.
- **S08 Optimal Dimension**: Tests whether the optimal budget for distance-based tasks (kNN, similarity search) differs from prediction tasks (RF, LightGBM).
- **S10 Importance as Distance Weights**: Feature importance scores directly as distance metric weights — validated against S02 results.
- **S11 Bayesian Optimization**: Uses hybrid feature space for Bayesian optimization with GP surrogate. Tests whether compact hybrid space improves BO convergence.
- **S12 Cross-Domain Cliffs**: Generic activity cliff detection across molecules, antibodies, and genomics.
- **S13 Clustering and Retrieval**: k-means clustering quality and nearest-neighbor retrieval precision in hybrid space vs raw rep space.

### Noise Experiments
- **N01 Activity Cliff Framework**: Implements Dablander-inspired cliff analysis — classifies each pair as cliff/non-cliff based on similarity threshold and activity difference. Measures how hybrid distance changes cliff boundaries.
- **N02 Sample Weighting**: Uses hybrid importance scores as sample weights during training. Down-weights samples that look noisy (high residual in hybrid space).
- **N03 SAS Landscape**: 4-quadrant maps plotting similarity vs activity change. Visualizes where hybrids move samples compared to single-rep spaces.
- **N04 GP LOO Detection**: Gaussian Process leave-one-out cross-validation for noise detection. Tests GP performance in hybrid feature space vs raw.
- **N05 Noise Injection Benchmark**: 15 binary noise detectors (Isolation Forest, LOF, Mahalanobis, etc.) tested at 5 noise levels (0%, 5%, 10%, 20%, 30%) in both raw and hybrid distance spaces. Full combinatorial benchmark.
- **N07 Remediation**: Given detected noisy samples, tests 4 strategies: removal, reweighting, relabeling (kNN imputation), and robust loss functions. Measures which strategy recovers the most performance.
- **N08 Estimation Survey**: Benchmarks 15 noise level estimators (variance-based, entropy-based, density-based, ensemble disagreement) on hybrid vs raw features.
- **N09 LNL Survey**: Tests 12 learning-with-noisy-labels methods (SELF, Co-teaching, Meta-Weight-Net, DivideMix, etc.) on hybrid features. Measures whether hybrid features make LNL methods more or less effective.
- **N10 Hybrid Neighborhood LNL**: Uses hybrid distance to define neighborhoods for LNL sample selection (which samples to trust, which to re-label). Tests whether hybrid neighborhoods improve clean sample identification.
- **N11 Estimation x Remediation Grid**: Full grid of (15 estimators x 4 remediation strategies) on all datasets. Finds the best combination per domain.
- **N12 Importance-Weighted Distance**: Weights distance metrics by feature importance for noise-aware similarity. Tests whether importance-weighted distance better separates clean from noisy samples.
- **N13 Alternative Noise Distributions**: Tests hybrid robustness to non-uniform noise: class-dependent noise, feature-dependent noise, structured outliers. Not just random label flips.

### Federated Experiments
- **Fed01 Noise Heterogeneity**: Each silo has different noise levels/types. Tests per-silo hybrid search with heterogeneous noise, then aggregates. Can federated hybrids handle non-IID noise?
- **Fed02 Rep Diversity Silos**: Each silo has access to different representations (e.g., silo A has ESM2, silo B has ProtT5). Tests whether federated hybrid search discovers complementary signal across silos.
- **Fed03 Synthetic Private Data**: Blends synthetic data with real private data. Tests hybrid search on synthetic+real mixtures, measuring whether hybrids degrade gracefully with synthetic contamination.
- **Fed04 Cluster-Aware Importance**: Computes importance scores within data clusters rather than globally. Tests whether cluster-aware weighting improves federated aggregation.
- **Fed05 Antibody Federated**: Per-silo DMS datasets + inter-family datasets. Tests cross-silo hybrid transfer in the antibody domain.

### Forward Compatibility (standalone)
- **C13 Forward Compatibility**: Tests whether new foundation model reps (ESM3, SaProt, Boltz-2, MolE, etc.) integrate into existing hybrid searches. Measures whether adding a new rep improves or disrupts existing hybrid rankings.

---

## 3. Overfitting, Selection Bias, and Nested CV

### The Problem
When we search over many hybrid configurations (allocation x importance x budget x evaluator)
and pick the best one, the winning score is **optimistically biased**. This is standard
hyperparameter selection bias — the more configs we try, the luckier the best one gets.

### Why It's More Manageable Than Model Overfitting
1. **Hybrids modify the INPUT, not the model.** The downstream model (RF, LightGBM, etc.) is
   fixed. The search only decides which features to include and how many. This limits the
   degrees of freedom — a 27-rep search has ~27 allocation parameters, not thousands of weights.

2. **Natural ceiling.** Feature selection can't extract signal that doesn't exist in the
   representations. Unlike a deep net that can memorize noise, hybrid search can only
   recombine existing features.

3. **Low-parameter strategies.** Allocation strategies decide per-rep budgets based on
   importance scores. Most strategies have 0-2 hyperparameters (temperature, threshold, k).

### What We Do About It — Nested CV
- **Outer CV** (5-fold): Holds out honest test data that the search NEVER sees
- **Inner search**: Runs find_best_hybrid() on outer-train only, searching all configs
- **Honest evaluation**: Takes the winning config from inner search, builds hybrid on
  outer-train, scores on held-out outer-test
- Reports `bias = search_score - honest_score` to quantify selection optimism
- **Materialized fold indices**: All configs are compared on identical CV splits.
  No accidental fold variation inflating winners.

### Evidence That Hybrids Are Not One-Offs
(See the EVIDENCE section in `presentation_output.txt` for actual numbers)

The script reports, for each dataset:
- **beat/tested**: How many hybrid configs beat the best single-rep baseline.
  When many configs exceed baseline, the improvement is structural, not a lucky draw.
- **% beating baseline**: Aggregated across datasets. If 30-60% of randomly sampled
  configs beat baseline, the hybrid feature space is genuinely better — not one config
  getting lucky.
- **Cross-budget consistency**: The heatmap shows improvement across budgets and models.
  A one-off result would spike at one point; structural improvement shows a plateau or
  consistent gain across the grid.
- **Multiple allocation strategies succeed**: Different strategies find different good
  hybrids. The improvement comes from combining reps, not from one strategy overfitting.

---

## 4. Hybrid Configuration Spec

Each hybrid config in the search is fully specified by:

| Field | Description |
|-------|-------------|
| **Internal evaluator** | Model used INSIDE the search to score feature subsets during allocation (match, rf, svm, lightgbm, mlp). "match" = same as downstream model. |
| **Allocation strategy** | How budget is distributed across reps (see Section 1) |
| **Base reps** | Which rep set was used — "general" (all viable reps) or "top_tier" (best N only) |
| **Augmentation** | none, uncapped (add aug features without budget limit), or capped (aug features count against budget) |
| **Budget** | Total number of features in the hybrid. Screening budgets: 50, 500, 1000, 2000, 3000, 4000. |
| **Importance method** | How individual features are scored within each rep (treeshap, lightgbm_gain, deeplift). Fast tier uses treeshap + lightgbm_gain. |
