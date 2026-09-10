# Feature importance scoring methods — what f30 tests, and why everything else was dropped

**Date:** 2026-09-03 · **Branch:** `similarity-metrics-study` · **Owner:** Adelaide
**Source:** the literature survey at `~/Downloads/compass_artifact_wf-1275ec2e-*.md`
**Experiment:** `tests/experiments/framework/f30_importance_methods.py`
**Status of every number below:** MEASURED on the local cache unless the line says
otherwise. Nothing here is a result about which method wins — that experiment has not
been run yet.


> **Line numbers verified 2026-09-03 against the working tree.** `src/kirby/hybrid.py`
> has uncommitted changes from a parallel session, so these shifted once already and may
> shift again. Every reference below also names the function or the exact code, which is
> the durable anchor — trust the name over the number.

---

## 1. Why a filter can be tested at all

`compute_feature_importance` (`hybrid.py:3292`) returns one float per feature per
representation. The **only** thing that reads it is
`np.argsort(importances)[-n_select:][::-1]` — `hybrid.py:1981`, `:4048`, `:1760`, `:1777`.
`concatenate_features` raises if the vector length does not match the representation's
width (`hybrid.py:4054`).

So a scoring method has to supply **an ordering of that representation's features, and a
vector of the right length**. Magnitude, sign, scale and normalisation are never used.

That is why "some of these are filters, not scoring methods" is not an obstacle. Three
conversion rules cover every entry in the survey:

| rule | applies to | conversion |
|---|---|---|
| **S1** | already a per-feature number | use it |
| **S2** | greedy / nested order (mRMR, JMI, CMIM, matching pursuit, RFE) | score = `K − step` for the feature picked at `step`, 0 for never-picked. Already the convention at `selector_prediction.py:177` |
| **S3** | a subset or a mask (Boruta, knockoffs, stability selection, lasso) | use the **continuous statistic the method thresholds**, never the boolean: Boruta's shadow hit count, knockoffs' `W_j`, selection frequency, `\|coef\|` |

A method is only untestable if it produces neither a per-feature number nor an ordering
long enough to fill a representation's quota.

---

## 2. Facts about this data that decide several entries

Measured from the cached matrices, 2026-09-03.

| dataset | n | task | binary representations | continuous |
|---|---|---|---|---|
| caco2 | 2158 | regression (rmse) | ecfp4, avalon, atom_pair, top_torsion — all {0,1} | pdv, grover, mhg_gnn, molformer, smited |
| buchwald_hartwig | 3955 | regression (rmse) | cgr_morgan, onehot_components {0,1}; diff_avalon, diff_maccs {−1,0,1} | the other 6 |
| flab_expression_koenig | 4275 | regression (spearman) | **none** | all 10 |
| tcga_brca_pam50 | 821 | classification, 5 class (f1_weighted) | **none** | all 10 |

Consequences:

- **Only tcga is a classification problem**, and every one of its representations contains
  negative values (measured minima: gene_geneformer_v2 −13.6, rppa −9.2, cnv_pca50 −154,
  mirna_top200 −1.2, wsi_titan −2.5).
- **Two of four datasets contain no binary features at all.**
- Any filter that needs discretised inputs is an approximation on three of four datasets.

---

## 3. What is being tested — 31 arms

9 incumbents, 19 new, 3 controls. 12 of the 31 are redundancy-aware.

### Incumbents (already branches in `hybrid.py`)
`lightgbm_gain` · `lightgbm_gain_nocap` · `xgboost_gain` · `treeshap` · `kernelshap` ·
`gradientshap` · `deeplift` · `integrated_gradients` · `permutation`

`lightgbm_gain_nocap` is the same method with the 2,000-feature pre-screen lifted — see
`HYBRID_IMPORTANCE_DEFECTS_2026_09_03.md`.

### New
| arm | what it computes | redundancy-aware | reference followed |
|---|---|---|---|
| `catboost_importance` | PredictionValuesChange | no | catboost docs `concepts/fstr` |
| `rf_mdi` | RandomForest Gini importance — **has no branch today**, `method='random_forest'` raises at `hybrid.py:4008` | no | sklearn `_forest.py` |
| `treeshap_interventional` | TreeSHAP with `feature_perturbation='interventional'` | no | Chen, Janizek, Lundberg & Lee 2020, arXiv:2006.16234 |
| `f_test` | ANOVA F / f_regression | no | sklearn `_univariate_selection.py` |
| `spearman_marginal` | \|Spearman rho\| per feature | no | monotone counterpart to the F-test |
| `mutual_info` | Kraskov k-NN mutual information | no | sklearn `_mutual_info.py` |
| `mrmr` | relevance ÷ mean \|corr\| to the selected set | **yes** | smazzanti/mrmr `mrmr/pandas.py` (FCQ) |
| `jmi` | `sum_s I(x_k, x_s ; y)` | **yes** | scikit-feature `JMI.py`; Brown et al. 2012 JMLR 13:27-66 |
| `cmim` | `min_s I(x_k ; y \| x_s)` | **yes** | scikit-feature `CMIM.py` (Fleuret 2004) |
| `hsic_lasso` | kernel relevance minus kernel redundancy, non-negative lasso | **yes** | riken-aip/pyHSICLasso `hsic_lasso.py` + `nlars.py` |
| `dcor` | distance correlation per feature | no | vnmabus/dcor `_dcor.py` |
| `relieff` | ReliefF / RReliefF | **yes** | skrebate `relieff.py` |
| `multisurf` | MultiSURF | **yes** | skrebate `multisurf.py`; Urbanowicz et al. 2018 |
| `conditional_permutation` | permute the residual of `x_j` given the others | **yes** | hidimstat `conditional_feature_importance.py` (Chamma et al. 2023) |
| `boruta_hits` | fraction of iterations the feature beat the best shadow | no | Kursa & Rudnicki 2010; scikit-learn-contrib/boruta_py |
| `stability_selection` | selection frequency over 20 half-sample bootstraps | no | Meinshausen & Bühlmann 2010 |
| `lasso` | `\|coef\|` | **yes** | sklearn LassoCV |
| `elastic_net` | `\|coef\|`, grouping effect | **yes** | sklearn ElasticNetCV |
| `matching_pursuit` | greedy on the residual of y | **yes** | already at `selector_prediction.py:135` |

### Controls
| arm | why it is there |
|---|---|
| `variance` | ranks features by their own variance, ignoring the labels. A method that cannot beat this is not selecting. |
| `laplacian` | unsupervised, never sees the labels. With `variance` it brackets how much comes from label information at all. |
| `random` | uniform random, fixed seed. The floor. |

### Blends
The survey names three aggregation families (§7). All three already exist in this repo, so
nothing is invented: rank averaging / Borda (`rankers.py:146 combine_ranks`), reciprocal
rank fusion (`aggregate.py:90`), and bootstrap ensembling (= `stability_selection`).

**Which arms get blended is decided from the agreement matrix after the first pass**, not
chosen in advance. Blending two methods that already rank features the same way does
nothing.

---

## 4. What was dropped, and the reason for each

Nothing was dropped for being unlikely to win. Every reason below is one of: it cannot run
on this data, it produces no usable ordering, no working implementation exists, or the
compute is arithmetic rather than preference.

### Cannot run on this data
| method | reason |
|---|---|
| **Chi-square** | Needs a categorical target AND non-negative features. Three datasets are regression; tcga has negative values in all ten representations. `sklearn.feature_selection.chi2` raises. Runs on **none** of the four. |
| **Mann-Whitney U / Kruskal-Wallis** | Needs a categorical target — tcga only. One dataset cannot rank a method against the others. Held for a second pass if tcga stays ambiguous. |
| **Kolmogorov-Smirnov** | Same restriction, and a strictly weaker two-sample statistic than the rank test above. |
| **Fisher score** | Between-class over within-class variance — needs classes. tcga only. |
| **Group lasso / sparse group lasso** | Selects predefined *groups* of features. There is no group structure inside a representation. Undefined here, not expensive. |

### The same quantity as an arm already running
| method | reason |
|---|---|
| **Pearson correlation** | For regression `F = r²(n−2)/(1−r²)`, strictly increasing in `\|r\|`. Identical ranking to `f_test`. The experiment asserts rank-identity on fold 0 rather than taking this on trust. |
| **Point-biserial** | Pearson with one binary variable. Same identity. |
| **Information gain / gain ratio** | Information gain about the target **is** mutual information. Gain ratio rescales by the feature's own entropy, constant for a standardised embedding. |
| **SPEC** | The Laplacian score is SPEC's second ranking function. |
| **MIFS** | mRMR is MIFS with `β = 1/\|S\|`. Running both varies a constant, not a method. |
| **Extra Trees** | Same MDI and permutation statistics from a differently randomised forest. Varies the estimator, not the importance. |
| **DeepSHAP** | DeepLIFT with SHAP's baseline distribution. Both DeepLIFT and GradientSHAP already run on the same MLP. |
| **LRP** | For a ReLU network with zero biases, LRP-ε equals Gradient × Input (Ancona et al. 2018). The MLP at `hybrid.py:3536` is that shape. |
| **Model Reliance** | Permutation importance for a single model. |
| **Feature ablation / occlusion** | Permutation importance with a constant replacement. |
| **Shapley effects** | Variance-based rather than loss-based Shapley; the loss question is SAGE's, which is held for pass 2. |
| **Banzhaf / Owen values** | Banzhaf reweights the same coalitions to a near-identical ranking. Owen values need feature groups, which do not exist inside a representation. |
| **Benjamini-Hochberg** | Monotone in the p-value, so it cannot change any ranking. It changes how many features you keep, and the budget is fixed at 100. |

### No usable per-feature ordering
| method | reason |
|---|---|
| **CFS** | Evaluates whole subsets, not features. Getting a per-feature order means running it as a sequential search, which is the matching-pursuit arm. Weka-only. |
| **Anchors** | Outputs if-then rules. No published conversion to a per-feature score. |
| **Counterfactual / DiCE** | A minimal input change per instance, not a ranking. |
| **Sparse PCA** | A sparsity pattern per component, with no standard rule for combining components into one per-feature score. |
| **UMAP / clustering selection** | No per-feature score; it is the Laplacian score's objective computed less directly with more knobs. |
| **Friedman H-statistic** | A `d × d` interaction matrix. Reducing it to a per-feature number measures how *interactive* a feature is, not how useful — a feature can be strongly interactive and useless. |
| **Shapley-Taylor interaction indices** | Also `d × d`; 23 million entries per representation at `d = 4800`. Summing a row returns the plain Shapley value. |
| **Nogueira stability** | Not a per-feature score at all — one scalar per method. It is used in this experiment, as an **analysis** statistic, not as an arm. |
| **RBO** | Also a pairwise scalar, not a score. Not used: a package exists, but it has never been used in this project, and four agreement statistics already in the repo are enough. |

### Compute is arithmetic, not preference
| method | the arithmetic |
|---|---|
| **Drop-column / LOCO** | One full model refit per feature. 4,800 features × 200 scoring calls = 960,000 refits. Permutation importance is its cheap approximation and is an arm. |
| **SFS / SBS / SFFS** | One model fit per candidate feature per step. One 2,048-feature representation selecting 50 features ≈ 100,000 fits, for one representation on one fold; the grid needs 200. Infeasible on the cluster too, not just locally. Its closed-form substitute, matching pursuit, **is** an arm. |
| **Exhaustive search** | 2^2048 subsets for one representation. |
| **Genetic algorithms / PSO / simulated annealing** | A model fit per evaluation, no per-feature score at the end, and population and generation budgets that would need tuning per representation. Dominated by stability selection, which is an arm. |
| **Sobol** | `(d+2) × N` model evaluations, `N` in the thousands — ~10⁷ predictions per representation per fold. Also assumes independent inputs, which is false here. |
| **Morris** | Perturbs inputs along a trajectory through the input hypercube, which for a standardised protein embedding lands nowhere near a real molecule, and needs declared input ranges. Permutation importance is the same one-at-a-time idea on real data. |
| **CRT / HRT / LOCO-CRT** | A conditional model per feature, then resampling — ~10⁸ model evaluations. Their output is a p-value, and under a fixed model the p-value ordering is monotone in the statistic permutation importance already gives. What they buy is false-discovery control, and this experiment has no false-discovery criterion — it has a prediction criterion at a fixed budget of 100 features. |
| **LOCO inference** | The point estimate is drop-column importance; the addition is a confidence interval nothing downstream reads. |
| **Conformal feature importance** | A wrapper that adds intervals to an importance computed by some other method. No ranking of its own. |
| **Markov blanket (GS, IAMB, MMPC, HITON)** | A conditional-independence test per feature per step. On continuous features that is either partial correlation — which reduces to the lasso arm — or a kernel test at `O(n²)` each, the same order as Sobol. Output is also a set of a few dozen features with everything else tied. `pyCausalFS` is unmaintained. |
| **LIME** | Local by construction. Going global needs SP-LIME's submodular pick, fitting a local surrogate per explained instance — each itself a 4,800-feature lasso. Lundberg & Lee 2017 show LIME is the SHAP kernel with different weights, so KernelSHAP answers the same question with the right weights. |
| **NAM** | One neural shape function per feature — 4,800 subnetworks per representation per fold. EBM is the same additive idea with a maintained implementation and is held for pass 2. |
| **TabNet** | Per-instance attention masks. The survey itself flags attention weights as unstable, and there is no trusted aggregation to a global per-feature importance. |
| **Concrete autoencoder** | Its selector is trained to reconstruct the inputs, not predict the label — a different objective from the one this experiment scores. It also picks exactly `k` features with no ordering, and `k` is fixed before training. The unsupervised control it would give is covered more cheaply by `laplacian`. |
| **Spike-and-slab / horseshoe / variational dropout** | MCMC or a variational approximation over a `d`-dimensional model space at `d` in the thousands, with no maintained scikit-compatible implementation. |
| **Deep Feature Selection / FSNet / input-gradient sparsity** | No canonical maintained implementation; input-gradient sparsity is Gradient × Input, which is covered. |

### Implementation does not exist or does not apply
| method | reason |
|---|---|
| **Model Class Reliance** | Needs the Rashomon set. The author's implementation is R (`aaronjfisher/mcr`). The one Python implementation, `mcrforest`, is git-only and works by exploiting the internals of **a modified random forest** — it does not apply to a gradient-boosted tree. Output is also an interval per feature; the contract reads a point ranking. |
| **Robust rank aggregation** | R-only (`cran/RobustRankAggreg`). Borda and RRF cover the blend question and are already in this repo. |

### Held for a second pass — not discarded
Run only if the family it belongs to places in the first pass:
`scad` / `mcp` (skglm has both), `adaptive_lasso`, `partition_shap`, `sage`,
`expected_gradients`, `saliency` / `gradient_x_input` / `smoothgrad` (near-zero marginal
cost on the MLP the Captum arms already train), `knockoffs`, `ebm`, `ard`, `pdp_variance`,
`ale`, `disr` / `cife` / `jmim`, `fcbf`, `surf` / `multisurf_star` / `turf`, `boruta_shap`,
`rulefit`, `pca_loadings`.

### Routed to the cluster rather than cut
`rfe` (one refit per feature at `step=1`; at a coarse step its output becomes tied blocks
rather than a per-feature order — the experiment measures the tie fraction so the
coarseness is a number), `lassonet`, `stg` (both train a network per representation per
fold).

---

## 5. One arm will not finish

`kernelshap` — MEASURED at 96–118 seconds on **60** features. Its cost is roughly
`2d + 2048` coalitions per explained sample, so at 2,048 features it is of order an hour
per representation per fold, and the grid has 200 of those per arm. It stays in the list
because it is an incumbent; expect that task to hit its wall rather than return.

---

## 6. How the results will be read

Three parts, **per dataset, never pooled** — rmse, spearman and f1_weighted are not
commensurable and two of the four run in opposite directions.

1. **Predictive.** Paired per-fold delta of the hybrid against the best single
   representation for that same downstream model, in that dataset's own metric,
   direction-corrected.
2. **Agreement.** Four statistics, all already in this repo and all previously used here:
   Spearman (`f05:161`), Kendall τ_b (`find_best_hybrid_molecular.py:1168`), top-k overlap
   (same file `:1179`), Nogueira Φ̂ (`c10:42`, `c24:50`). Each rank statistic is reported
   **twice** — over all features, and over the union of features both arms scored non-zero
   (see the defects document for why that second view is the honest one).
   Prediction agreement is separate: Spearman of the prediction vectors for the three
   regression datasets, Cohen's κ for tcga.
3. **Compute time** per arm per representation per fold, and relative to the cheapest arm
   on that dataset.

Decision rule: two arms that agree closely collapse to one and the cheaper is kept, applied
**per dataset** — a pair that agrees on caco2 need not agree on tcga.
