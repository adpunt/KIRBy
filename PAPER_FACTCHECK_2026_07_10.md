# PAPER_DRAFT.md fact-check against the code — 2026-07-10

Method: 87 agents, each reading the actual source (no grep-only claims, no memory).
Every alleged error was independently re-checked by three adversarial verifiers
(correctness / charity / edit-safety) before any edit was applied. Two alleged
errors were rejected on verification and left alone.

---

## Applied (10 edits, facts only, no prose rewritten)

| Line | Was | Now | Code |
|---|---|---|---|
| 98 | C13 conditions A–E, Wilcoxon test | A/B/C/D/D_random, debiased effect sizes | `c13_forward_compatibility.py:13-45` ("Condition E is gone"; "No significance testing is reported") |
| 163 | ~15,000 configs, 7 models | ~7,300 configs, 8 models | code's own counter `find_best_hybrid.py:3349-3356` |
| 163 | caps at 2,000 or 3,000 features | caps at 500, 1,000, or 2,000 | `MODEL_MAX_BUDGET`, `find_best_hybrid.py:1938-1947` |
| 167 | AUROC for ... tumor subtype | ... tumor type | `tcga.py:1446-1447` (luad_vs_lusc = binary type; brca_pam50 = 4-class subtype) |
| 171 | genomics: six modalities | eight (adds microRNA, whole-slide histology) | `AUG_GROUPS['tcga']`, `complexity_helpers.py:890-904` |
| 175 | logistic regression for classification | a ridge classifier | `RidgeClassifierCV`, `find_best_hybrid.py:1422` |
| 175 | seven models (no Ridge) | eight models (adds Ridge) | `_resolve_models(None, speed='full')` → 8 |
| 175 | GP skipped above 6,000 samples | above 1,000 | `'gp': {'_max_samples': 1000}`, line 1458 |
| 179 | max input dim 2,048 (ECFP) | 17,920 (per-region antibody embeddings) | `esm2_per_region`, `antibody.py:1821`; in 15 dataset pools |
| 181 | 5-fold GroupKFold throughout | + StratifiedGroupKFold for classification | `find_best_hybrid.py:2117-2123` |
| 188 | reaction grouped by template | by component (additive / ligand) | `reaction_yields.py:185, 318` |
| 189 | genomics grouped by patient ID | by tissue source site | `_extract_tss_groups`, `tcga.py:1415-1435` |
| 217 | capped augmentation prunes at 25% | capped uses 15%; only standalone scripts use 25% | `PRUNING_MODES` both 0.15; `adaptive_pruning=0.25` only in `opig_*`/standalone |
| 217 | speed tiers (fast and full) | a single full tier | `SPEED_TIERS` has one key, `find_best_hybrid.py:1914-1932` |

---

## Not applied — needs your decision

### 1. Allocation strategies: the paper says 13, the code searches 9 — and they are not the families described

`SPEED_TIERS['full']['allocation_methods']` (`find_best_hybrid.py:1920-1928`) is:

```
top_2, top_3, top_5, performance_weighted, entropy_1.0,
power_2, rank_linear, entropy_2.0, rank_exp_d0.5
```

All nine map to `('performance_weighted', ...)` in `NAMED_ALLOCATIONS`. So the live
search space is **entirely score-based**. It contains no `min_floor` (the floor-based
variant) and no `greedy_*` (the greedy-forward variant) — both of which exist in
`hybrid.py` but are **not in the searched list**.

The paper (line 155) says the retained strategies "fall into three families",
naming score-based, floor-based, and greedy-forward. Two of those three families
are not actually searched.

`NAMED_ALLOCATIONS` holds 40 keys (37 unique + 3 legacy aliases), not 31.
"31 candidates → 13 retained" appears only in stale docs (`ANALYSIS.md:360`,
`PAPER_OUTLINE.md:182`, `PIPELINE_READINESS_TODO.md:119`).

**Two options:** (a) the tier list was over-trimmed and `min_floor` + a greedy
variant should go back in — a search-space change, needs your signoff and a rerun;
or (b) the prose drops the floor/greedy families and says 9. I did not guess.
Same issue repeats in Supplementary Methods (line 217: "All 13 allocation strategy
definitions ... 31 candidates ... 13 retained").

### 2. Statistical analysis section describes three tests that do not exist

Line 195 claims four procedures. Only one is implemented:

- **Wilcoxon signed-rank** — exists (`c14_cross_domain_summary.py:353`, `s09_cross_dataset_summary.py:249`)
- **Nadeau-Bengio corrected paired t-test** — does not exist. The only paired t-test in the repo (`utils/evaluation.py:271`) is plain `scipy.stats.ttest_rel`, uncorrected. No `nadeau` token anywhere.
- **Friedman + Nemenyi** — does not exist. The only mention (`c13_forward_compatibility.py:43-44`) is a comment saying it is deliberately *not* applied.
- **Effect sizes (rank-biserial correlation, Kendall's W)** — neither exists. The `kendall` hits are `kendalltau`, a different statistic.

Not edited because removing them is a three-sentence deletion that orphans the
`^[NadeauBengio2003]` setup sentence. Either implement the tests, or cut the claims.

### 3. Experiment C25 does not exist, and neither does a "fast" tier

Line 118 proposes "Fast vs full tier convergence (C25)". The complexity suite is
c01–c24 (c20 also absent), and `SPEED_TIERS` has a single `full` tier. This bullet
is dead as written — the reduced tier it would compare against was removed in favour
of the multi-stage adaptive-pruning schedule.

### 4. "up to 57 sources per domain" (abstract, line 25, line 171, line 179) — unverified

Measured from the actual per-dataset rep lists in `complexity_helpers.py`:

| domain | datasets | union of sources | max for one dataset |
|---|---|---|---|
| antibody | 23 | 76 | 29 |
| cancer genomics | 2 | 28 | 28 |
| molecular | 7 | 23 | 21 |
| drug-target | 3 | 20 | 19 |
| reaction | 1 | 8 | 8 |

No pool is 57. `genomic.py` does define 54 `create_*` functions, so "57" may be a
count of *available* genomic representations rather than any pool actually searched.
Left alone — you know which number you meant.

### 5. "Dataset sizes range from 94 to about 50,000 samples" (line 179) — unverified

n=94 is `ginkgo_heparin`, a **cut** dataset. Smallest in the run set is ~141
(`abcov_kd`). Largest in-scope is ~30,000–33,000 (`kiba`, `bindingdb_kd`,
`phillips_h3`), not 50,000. The final dataset list isn't pinned in code, so I did
not invent bounds.

### 6. "We use the full set for model-matching and noise experiments" (line 175)

Neither is the full 13. Model-matching (`c23_evaluator_model_matching.py:76-77`)
uses **11** (excludes BNN and NGBoost). Noise experiments iterate
`DOWNSTREAM_MODELS_NOISE` = **6** (rf, xgboost, ridge, svm, mlp, lightgbm;
`_shared.py:474`). Two different rosters; one sentence can't hold both without a
rewrite, so I left it for you.

### 7. Minor: the Introduction's search-axis list is incomplete

Line 25: "searching over budgets, allocation strategies, importance methods, and
downstream models". The internal evaluator is also a searched axis (5 options,
looped per config) and Methods line 163 lists it. Incomplete, not wrong — not edited.

---

## Checked and found correct (no change)

- Three importance methods (LightGBM gain, TreeSHAP, DeepLift) chosen from **eight** candidates — `compute_feature_importance` dispatches exactly 8 (`permutation, treeshap, integrated_gradients, kernelshap, xgboost_gain, lightgbm_gain, deeplift, gradientshap`); `SPEED_TIERS` retains 3.
- Five internal evaluators = `['match','rf','svm','lightgbm','mlp']`.
- **Ten** metrics: `BUILTIN_METRICS` has 11 keys but `sac` is a guard that raises, not a prediction metric.
- **Thirteen** models available in `DEFAULT_MODELS`. NGBoost present. MLP single-layer `(256,)`; DNN three-layer `(256,128,64)`; BNN is MC-Dropout; CNN is multi-scale 1D (kernels 3/7/15).
- Augmentation ranked by random forest importance (`candidate_ranking='rf_importance'`), appended only on internal-evaluator improvement. MACCS = 167 bits. ESM-2 LLR = 1–3 dims.
- **"No random splits are used" is true and enforced** — `_make_cv_splitter` *raises* rather than falling back to a random k-fold (`find_best_hybrid.py:2096-2116, 2135-2139`).
- Preprocessing is inside the fold: training-fold median imputation, all-NaN columns dropped, inf clipped, binary matrices skip standardization.
- The nested internal split is group-aware `GroupShuffleSplit` on the outer grouping variable (`hybrid.py:3488-3493`).
- Search result cached to disk; Blueprint contents; pipeline ordering (per-source baselines → per-fold importance on train only → allocation → CV scoring; best kept per model×budget).
- Splits for molecular (Bemis-Murcko), DMS (position mod 5), SKEMPI (PDB complex), Ginkgo (benchmark's precomputed clusters), antibody inter-family (identity clustering).
- Nogueira index, Jaccard stability and the Blueprint transfer test all exist (`c10_noise_robustness.py`, `c24_blueprint_transfer.py`).
- Adaptive pruning: fold 0, top-3, `final_strategy_threshold = 0.15` default.
- Augmentation defaults `step_size=5`, `patience=20`.

## Alleged wrong, then refuted on verification (left alone)

- **Blueprint fields** (line 163). Verifiers judged "all parameters (source names, allocation strategy, importance method, budget, evaluator, metric)" defensible as an illustrative list; the dataclass also carries `step_size`, `speed`, `n_folds`, `seed`, `task_type`, `augmentation_strategy`, `lower_is_better`, `aug_per_source_budget`, `objective`, `rep_scores`.
- **Pipeline ordering** (line 149). Matches the code.
