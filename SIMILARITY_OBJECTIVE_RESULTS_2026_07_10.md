# Replacing the search objective for similarity tasks — research + measured results

**Date:** 2026-07-10
**Scope:** herg_ki only (n=1415, 751 Murcko scaffolds), 5-fold scaffold-grouped CV, budget=100.
**Status:** evidence for a decision. NOT a code change. Nothing in `src/` was touched.

---

## 0. What SAC is, and where it came from

`similarity_activity_correlation_scalable` (`src/kirby/utils/noise_mitigation_methods.py:149`)
is the Spearman correlation between pairwise distance in the standardized feature
space and the pairwise activity gap |Δy|, over 5000 sampled pairs.

**It was invented for this project by Claude. It has no citation and no literature
analogue.** It was written into `find_best_hybrid` as `objective='similarity_sac'`
(`find_best_hybrid.py:1028`), into `hybrid.py:_evaluate_allocation`
(`hybrid.py:1182`), and into the standalone experiment, as the *only* similarity
objective the engine can express (`_VALID_OBJECTIVES`, `find_best_hybrid.py:1029`).

It has a structural defect: its maximiser is the embedding X = y, at which distance
equals |Δy| exactly and every activity cliff is destroyed by construction.

Meanwhile `tests/experiments/similarity/similarity_helpers.py` already contained
**real, cited** metrics that went unused as objectives: SALI at line 94 (Guha &
Van Drie 2008), the Stumpfe & Bajorath cliff definition at line 406, and the
neighbour violation rate at line 121.

---

## 1. Where the objective actually enters the search

Two places, and only one of them is currently swappable.

| Location | What it decides | Swappable today? |
|---|---|---|
| `hybrid.py:1143` `_evaluate_allocation` | the scalar that all five allocation methods maximize — i.e. **how many features to take from each representation** | yes: `predictive` (fit a model, score with `metric_fn`, line 1196) or `similarity_sac` (score the held-out geometry, line 1182) |
| `hybrid.py:1106` `np.argsort(importances)` | **which** features are taken from inside each representation | **no** — `objective` is never passed. All 8 methods in `compute_feature_importance` (treeshap, xgboost_gain, lightgbm_gain, deeplift, permutation, kernelshap, integrated_gradients, gradientshap) are model-fit criteria. |

---

## 2. Literature-verified alternatives

Two independent deep-research runs (216 agents, ~900 web calls, 50 sources fetched,
50 claims adversarially verified 3-vote; 0 refuted).

### Objectives that score a whole feature space

| Name | Citation | Uses y? | Collapse-prone? | Cost |
|---|---|---|---|---|
| Trustworthiness / Continuity | Venna & Kaski (ICANN 2001; JMLR 2010); unified by Lee & Verleysen, *Neurocomputing* 72:1431-1443, 2009 | **no** | **no** | O(n² log n) |
| Triplet correct-ordering | Coupry & Pogány, *J Cheminform* 14:12 (2022), `10.1186/s13321-022-00595-7` | **no** | **no** | O(#triplets · d) |
| MODI | Golbraikh, Muratov, Fourches & Tropsha, *JCIM* 2014, `10.1021/ci400572x` | yes (1-NN only) | weakly | O(n log n) |
| Neighborhood behaviour (trapezoidal) | Patterson, Cramer, Ferguson, Clark & Weinberger, *J Med Chem* 39(16):3049-3059, 1996, `10.1021/jm960290n` | yes, one-sided | no | O(pairs) |
| Distance correlation | Székely, Rizzo & Bakirov, *Ann. Statist.* 35(6):2769-2794, 2007 | yes, global | yes | O(n²) |
| Kernel–target alignment | Cristianini, Shawe-Taylor, Elisseeff & Kandola, NIPS 2002 (**uncentered**). Centered version: Cortes, Mohri & Rostamizadeh, *JMLR* 13:795-828, 2012 | yes, global | **yes** | O(n²) |
| SALI | Guha & Van Drie, *JCIM* 48(3):646-658, 2008, `10.1021/ci7004093` | yes | — | O(n²) |
| **SAC** | **none — invented here** | yes, global | **yes** | O(pairs) |

### Feature rankers (per-feature score → drop-in for the argsort)

| Name | Citation | Notes |
|---|---|---|
| RReliefF | Robnik-Šikonja & Kononenko, *Machine Learning* 53:23-69, 2003, `10.1023/A:1025667309714` | The **regression** Relief. `skrebate` does **not** implement it — it only has classification ReliefF. Implemented here from Algorithm 3. |
| Laplacian Score | He, Cai & Niyogi, NIPS 2005 | unsupervised, locality-preserving |
| HSIC-Lasso | Yamada, Jitkrittum, Sigal, Xing & Sugiyama, *Neural Computation* 26(1):185-207, 2014 | `pyHSICLasso`; selects a subset, not a full ranking |

### Rejected on derivation, before implementation

**iCliff** (*JCIM* 2025, `10.1021/acs.jcim.5c00506`) is real and cheap (O(n)), but

    iCliff = Var(P) · (1 + iT + iT² + iT³)/2 ,  iT = mean pairwise similarity

For a fixed dataset `Var(P)` is constant, so iCliff is a strictly increasing
function of mean pairwise similarity alone. **Maximizing it collapses every
molecule onto one point.** It is a valid dataset-roughness descriptor (its intended
use) and a degenerate feature-selection objective. Report it; never optimize it.

### Corrections to earlier claims made in this project
* Kernel–target alignment is **Cristianini et al. (NIPS 2002)**, not Cortes.
* BEDROC is **Truchon & Bayly** (*JCIM* 47(2):488-508, 2007), not Bajorath.
* **MHG-GNN is real** (Kishimoto, Kajino et al., IBM, arXiv:2309.16374, NeurIPS 2023
  AI4Mat workshop; checkpoint `ibm/materials.mhg-ged`), building on Kajino's ICML
  2019 hypergraph grammar (PMLR v97:3183-3191) — **but it is an autoencoder for
  property prediction and valid-molecule generation. It is not a similarity method,
  and its latent distance has never been evaluated against structural similarity.**

### The caution that governs everything
Praski, Adamczyk & Czech, arXiv:2508.06199 (2025, preprint): across 25 pretrained
molecular embedding models × 25 datasets, **nearly all neural embeddings show
negligible or no improvement over the plain ECFP fingerprint.** The only model that
is significantly better, CLAMP, is itself fingerprint-based. Any similarity-native
representation must be measured against a plain-fingerprint baseline or the result
is meaningless.

---

## 3. Experiment 1 — does a training-time objective predict held-out similarity utility?

140 candidate spaces (7 rankers × 4 budgets × 5 folds). Each objective computed on
the training fold; utility measured on the held-out fold. Spearman correlation
computed **within each budget separately**, so feature count cannot drive it.

Mean correlation across the four similarity utilities (read-across, structural
retrieval, cliff-detection AUROC, neighbour-lie rate):

| Objective | mean ρ with held-out utility |
|---|---|
| trustworthiness | **+0.681** |
| continuity | +0.665 |
| triplet_struct | +0.639 |
| struct_align | +0.604 |
| modi_binarized | +0.593 |
| triplet_activity | +0.482 |
| **predictive (KIRBy today)** | **+0.468** |
| patterson_bound | +0.353 |
| dcor | +0.333 |
| cka | +0.177 |
| **sac (invented)** | **+0.158** |
| kta | **−0.113** |

**SAC is a worse guide to held-out similarity performance than the predictive metric
it was meant to replace.** Kernel–target alignment is actively anti-correlated.

### Feature rankers, held-out utility, feature count held fixed (z-scored; `random` = null)

| Ranker | read-across | retrieval | cliffs | neighbour-lie |
|---|---|---|---|---|
| **lightgbm_gain (KIRBy today)** | **+1.02** | **+1.06** | **+0.55** | **+1.03** |
| rrelieff | +0.61 | +0.82 | +0.30 | +0.53 |
| struct_align | +0.53 | +0.80 | +0.19 | +0.39 |
| sac_perfeat | −0.09 | −0.60 | +0.20 | −0.03 |
| laplacian | −0.12 | −0.40 | −0.11 | −0.11 |
| random (null) | −0.19 | −0.01 | −0.12 | −0.25 |

The existing predictive ranker wins on **every** task, including structural
retrieval. The per-feature form of SAC is no better than random.
(`variance` scored −1.75 but is an invalid control: features are standardized before
ranking, so all variances are 1 and it sorts numerical noise.)

---

## 4. Experiment 2 — swap the objective inside the greedy allocation loop

Faithful standalone reproduction of `_evaluate_allocation` + greedy allocation.
Within-rep ranker held at `lightgbm_gain` (what KIRBy does today) so that **only**
the search objective varies. Budget 100. Baseline = best single representation at
full dimensionality.

Paired per fold, n=5. **The Wilcoxon floor at n=5 is p=0.0625**, so no result here can
reach p<0.05; the meaningful bar is winning 5/5 folds.

| Comparison | Δ | folds won |
|---|---|---|
| trustworthiness vs **sac** — read-across | +0.021 | **5/5** |
| trustworthiness vs **sac** — retrieval | +0.034 | **5/5** |
| trustworthiness vs **predictive** — retrieval | +0.021 | **5/5** |
| trustworthiness hybrid vs best single rep (ecfp6) — retrieval | +0.032 | **5/5** |
| lightgbm_gain vs rrelieff ranker — read-across | +0.044 | **5/5** |
| lightgbm_gain vs rrelieff ranker — retrieval | +0.033 | **5/5** |
| trustworthiness vs predictive — read-across | +0.002 | 4/5 |
| trustworthiness vs predictive — cliff AUROC | +0.001 | 2/5 |
| trustworthiness vs predictive — neighbour-lie | −0.001 | 4/5 |
| trustworthiness hybrid vs best single rep — read-across | +0.002 | 3/5 |
| trustworthiness hybrid vs best single rep — cliff AUROC | +0.003 | 1/5 |

### What this says

1. **Swapping the objective to trustworthiness beats SAC consistently** (5/5 on two
   of four tasks) and beats the predictive objective on **structural retrieval only**.
2. On read-across, cliff detection and neighbour-lie, the objective makes **no
   reliable difference**. Consistent with arXiv:2508.06199.
3. **The hybrid does not beat the best single fingerprint** except on retrieval
   (+0.032, 5/5). At 100 features vs ecfp6's 1024, it matches on everything else —
   a compression win, not an accuracy win.
4. **The within-rep ranker matters more than the objective.** `lightgbm_gain` beats
   RReliefF 5/5 on two tasks. The earlier hypothesis that the predictive ranker was
   crippling the similarity hybrid is **not supported by this data.**
5. The objective genuinely changes what gets selected. Summed over folds,
   trustworthiness allocated `{ecfp4:220, ecfp6:60, atompair:0, maccs:100, desc:120}`
   — dropping atom-pair entirely — where predictive allocated
   `{ecfp4:80, ecfp6:140, atompair:90, maccs:70, desc:120}`.

---

## 5. Limitations — read before acting

* **One dataset.** herg_ki, n=1415. Nothing here generalizes until it runs on caco2,
  logd, adme_hlm, adme_rlm, adme_mdr1, qm9.
* **One budget** (100 features) in experiment 2.
* **n=5 folds** ⇒ no result can be significant at p<0.05. "5/5 folds" is the bar used.
* The structural reference is **ECFP4 Tanimoto**, which is circular for the retrieval
  metric (we score a space by how well it reproduces ECFP4 neighbours, and ECFP4 is
  one of the input representations). The literature's answer (SimGNN, Bai et al.,
  WSDM 2019) is to target an **expensive** reference — graph edit distance or MCS —
  precisely to avoid this. **Any production version must switch the reference.**
* Trustworthiness at O(n² log n) is affordable at n≈1.4k. At qm9's n≈10k inside a
  search loop it is not, without subsampling.

---

## 6. Proposed next step (NOT taken — requires sign-off)

1. Extend `_VALID_OBJECTIVES` with `similarity_trustworthiness`, keeping
   `similarity_sac` present so it can be beaten on the record rather than deleted.
2. Leave `compute_feature_importance` alone. The data does not justify changing it.
3. Replace the ECFP4-Tanimoto reference with a non-circular one (MCS or graph edit
   distance on a sampled pair set, precomputed once per dataset).
4. Re-run experiments 1 and 2 across all six molecular datasets on ARC before any
   `src/` change lands.

Scratch code: `scratchpad/{build_bench,objectives,lit_objectives,run_bench,run_search}.py`
