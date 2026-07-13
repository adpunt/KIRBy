# Complexity Experiments — Reading Guide

How to read the JSONs under `tests/results/complexity/<exp>/<dataset>.json` produced by the per-dataset complexity suite (`tests/experiments/complexity/c01_*` … `c24_*`).

Every file is the dict returned by the experiment's `run()`, plus a `timestamp` and `experiment` key prepended by `save_experiment_results` (`tests/experiments/_shared.py:2540`).

---

## Pull command (server → local, JSON-only)

```
rsync -av --include='*/' --include='*.json' --exclude='*' \
  scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/complexity/ \
  tests/results/complexity/
```

---

## Dependency graph

Several experiments cross-reference each other's JSONs. If a dependency is missing they fall back to a degraded run (printed as "No cNN results found …") rather than crashing.

| Exp | Reads JSONs from |
|-----|------------------|
| c03 | c02 (multi-dataset blueprint transfer) — optional, can run own searches |
| c07 | c02 (optional — for cross-reference panel) |
| c08 | c02 (optional — for "optimal vs sub-optimal config" pairing) |
| c09 | c02 (ANOVA over its grid), c06 (bias-variance) |
| c14 | none (self-contained, runs `find_best_hybrid` per task) |
| c15 | c01 (rep characterization for comparison), c02 (optional) |

**Practical rule:** if you re-run c02 standalone, you can optionally re-run c07/c08/c09/c15 to pick up the cross-reference panels — but their headline results are already valid without c02.

---

## Per-experiment reading guide

For each experiment: the **headline number** (the one to look at first), the **JSON top-level keys**, and **what to ignore unless you're digging deep**.

### c01 — Representation Characterization
- **Question:** profile each rep (dim, sparsity, kernel rank, intrinsic dimensionality).
- **Headline:** none — this is descriptive context for c15 and c17.
- **Keys:** built incrementally; per-rep stats keyed by rep name.
- **What to look at:** intrinsic dim ≪ raw dim → rep is compressible (matters for c15/c17).

### c02 — Complexity Matching Grid (CORE)
- **Question:** for each downstream model, what's the optimal (rep, allocation, importance, budget) combo? Does the best hybrid beat the best single rep?
- **Headline:** `best_by_model[model]['score_mean']` minus `baselines[model]['best_single_score']` → the per-model hybrid-vs-baseline delta.
- **Keys:** `configs`, `best_by_model`, `baselines`, `pca_baselines`, `heatmap`, `importance_wins`, `conditions`, `best_per_model_across_conditions`, `has_augmentation`, `top3_base_reps`.
- **`heatmap[model][budget]`** is your eyeball check — does score peak at a specific budget for each model? That's the complexity-matching claim.
- **`conditions`** = {`no_aug`, `uncapped`, `capped`, `top_tier_only`, `frozen_plus_ft`, `per_model_top_rep_dim`}: best config per condition. Compare to see if augmentation / top-tier / FT changes the winner.
- **Don't touch:** raw `configs` list — thousands of rows, only useful for re-aggregation.

### c03 — Cross-Dataset Blueprint Transfer
- **Question:** does the optimal config from dataset A still work on dataset B?
- **Keys:** per-dataset `best_by_model`, `_complexity_correlation`, `_budget_variance`, `performance_matrix`, `novel_models`.
- **Headline:** `_complexity_correlation` (rank correlation of optimal budgets across datasets). High → blueprint transfers; low → context-dependent.

### c04 — Importance as Learned Metric
- **Question:** does feature importance double as a kNN/similarity metric?
- **Keys:** `methods` (per-method `weighted_r2`, `unweighted_r2`, `delta_r2`), `cross_test`, `low_dim`, `best_knn_method`, `best_sim_method`.
- **Headline:** the per-method `delta_r2` (importance-weighted minus unweighted). Positive → importance acts as a learned metric.

### c05 — Learning Curves
- **Question:** are hybrids more sample-efficient than individual reps?
- **Keys:** `sample_sizes`, `budgets`, `models`, `search_speed`, plus per-(n, model, rep_or_hybrid) score curves.
- **Headline:** at the smallest n, does hybrid > best single rep? Cross-over point matters more than absolute scores.

### c06 — Within-Family Capacity Gradient
- **Question:** within a single model family (RF n_estimators, XGB depth, MLP width), does capacity track budget?
- **Keys:** `families` (per-family capacity sweep), `search_speed`.
- **Headline:** does best capacity rise monotonically with budget within each family? Confirms the matching hypothesis without a hybrid.

### c07 — Scaffold-Partition Data Complexity
- **Question:** does the optimal (model, rep, budget) shift across data subpopulations?
- **Keys:** `partitions` (per-subpop best hybrid + `best_by_model`), `search_speed`.
- **Headline:** are the per-partition winners different? If yes → context-dependent; if no → a global optimum exists.

### c08 — Calibration at Optimal vs Suboptimal Complexity
- **Question:** does complexity matching improve uncertainty calibration (ECE / NLL / Brier)?
- **Keys:** `conditions` (per-condition calibration metrics), `task_type`.
- **Headline:** ECE / NLL at `optimal` vs `suboptimal_low` and `suboptimal_high`. Optimal should be lowest.

### c09 — Formal Analysis (ANOVA + Information Retention + Effective Rank + Bias-Variance)
- **Question:** what fraction of score variance is explained by model vs budget vs interaction?
- **Keys:** `budgets`, `mi_xy` (rep→label MI), `mi_xx_full` (rep redundancy), `best_configs`. ANOVA / bias-variance fields populate when c02/c06 JSONs exist.
- **Headline:** the ANOVA target is **model 50%+, budget 25%+, interaction 10%+** of variance. If those are met, complexity matching is real.

### c10 — Noise Robustness
- **Question:** are hybrids more robust to label noise than individual reps?
- **Keys:** `sigmas`, `strategies`, `individual_reps`, `hybrids`, `feature_stability`.
- **Headline:** **NDS** (Noise Degradation Slope, score vs σ) — hybrids should be flatter (less negative) than individual reps.

### c11 — Ensemble vs Hybrid
- **Question:** an ensemble needs all K featurisers + K models at every inference, forever. The hybrid pays a one-off search, then deploys ONE model on a fused vector. Does the hybrid recover the ensemble's accuracy at a fraction of the standing cost?
- **★ It is a PROCESS comparison, and every process is replicated PER FOLD.** Each arm is a procedure you could actually run on a training set. It is replicated independently on each CV fold and may only ever see *that fold's training split*: it splits the training data with an **inner group-aware CV**, scores every (model, rep) there, and uses **only those inner scores** to rank reps / pick top-K / set weights / choose the aggregator. It then refits the chosen members on the full training split and touches the held-out split exactly once, to be scored. Nothing crosses a fold boundary. In particular the member ranking must **never** come from the other folds' scores — a model scored on fold *g* was trained on data containing fold *f*, so ranking fold *f*'s members by fold *g*'s score leaks fold *f*'s labels. An earlier version of this file did exactly that; `members_per_fold` is the audit trail that it no longer does.
- **Arms** (`results[].method`): `best_single`, `pca_matched`, `hybrid`, `ensemble_s1` (fixed model, one member per rep), `ensemble_s2` (each rep picks its own best model, per fold, on the inner CV), `ensemble_s1_auto` / `ensemble_s2_auto` (**the fair comparators** — self-contained processes that also choose their rep set and aggregator on the inner CV, exactly as the hybrid chooses its own recipe), `oracle_single` (**peeks at the held-out fold — an upper bound, not a fair arm**).
- **Rep sets** (`results[].rep_set`): `top` (top-3 by inner CV) and `all` (every tier≥2 rep). **Aggregators** (`results[].aggregator`): mean, median, trimmed20, iqr_filtered, mad_filtered, weighted, top3_mean, majority_vote (hard-label classification only).
- **Keys:** `results`, `n_reps`, `total_dim`, `rep_names`, `general_reps`, `rep_dims`, `n_inner_folds`, `honest_rescore` (hybrid vs single — unchanged contract), `honest_rescore_ensemble`, `head_to_head`, `break_even`, `search_cost_model_fits`. Ensemble rows also carry `members_per_fold` and (for `*_auto`) `auto_choice_per_fold`.
- **Headline:** `head_to_head[model].auto_minus_hybrid` (+ = ensemble more accurate), read *against* `break_even[model]`. Prefer `ensemble_auto` over `ensemble_honest`: the auto process is self-contained, so its plain fold-mean is already honest and needs no leave-one-fold-out reselection.
- **`cost.selection_fits`** — what the process itself burns to make its choices. A plain **mean over all reps is 0** — that zero-search ensemble needs no selection signal at all and is the real competitor to beat. `top`/`weighted`/`topk`/`auto` all pay an inner-CV cost, and so does the hybrid (its one-off search).
- **★ The number that decides the paper:** `break_even[model].hybrid_rep_sparsity` = |S|/K, the fraction of reps the blueprint actually draws a column from. Feature extraction dominates test-time cost (Xu et al., *Cost-Sensitive Tree of Classifiers*, ICML 2013), and a featuriser is paid for the first time ANY of its columns is used. So if sparsity = 1.0 the hybrid runs every encoder the ensemble runs, `featurisation_saving_exists` is False, and **there is no compute win to claim** — only the downstream forward pass, a rounding error next to a PLM. Check this before writing any efficiency claim.
- **Selection optimism:** the aggregator sweep is a *search*, so every (rep set × aggregator) rule is a candidate re-picked leave-one-fold-out by `honest_rescore`; rep-set membership and weights are chosen per held-out fold from the *other* folds. Same tax as the hybrid. Never read `score_mean` for a headline — read the honest block.
- **NOT a mean baseline.** This averages model *predictions* (an ensemble = a method). It never averages rep *scores* into a composite baseline — that remains banned.

### c13 — Forward Compatibility
- **Question:** does adding a new representation lift the existing hybrid (and vice versa)?
- **Keys:** structured output via `make_structured_output` — per-rep before/after deltas + `statistic`/`p_value` from Wilcoxon.
- **Headline:** delta when each candidate new rep is included in the search pool; sign + significance.

### c14 — Cross-Domain Summary
- **Question:** does the hybrid beat the best single rep consistently across domains?
- **Keys:** per-task records (best single rep, hybrid score, delta, pct_improvement), aggregate Wilcoxon test, win rate, per-domain breakdown.
- **Headline:** **win rate** (fraction of tasks where hybrid > best single) and Wilcoxon p-value across all tasks. This is the paper's headline claim.

### c15 — ML Theory Depth (concentration of measure, hubness, kernel degradation)
- **Question:** do compact hybrid reps avoid high-D pathologies?
- **Keys:** `per_rep`, `per_budget`, `concat`. Each has `beyer_ratio`, `relative_contrast`, `distance_cv`, `distance_iqr_ratio`, `hubness_skewness`, `hubness_robin_hood`.
- **Headline:** at the budget c02 picks as optimal, are these metrics better than the full-concat versions? E.g., higher `relative_contrast`, lower `hubness_skewness`.

### c16 — Feature Selection as Regularization
- **Question:** does the hybrid's budget subsume L2 / shallow trees / SVM-C regularization?
- **Keys:** `metric`, `lower_is_better`, `budgets`, `total_dim`, `families` (per-family score over (regularization × budget) grid + `Winner` + per-budget `Sensitivity`).
- **Headline:** for each model family, `Winner` should be `kirby_budget` if budget regularizes effectively. `tuned_full_dim` winning means tuned regularization is enough — that's a (mild) negative result for KIRBy on this dataset.

### c17 — Information-Theoretic Decomposition
- **Question:** mRMR vs hybrid vs PCA at matched budget — which retains the most predictive information?
- **Keys:** `metric`, `budgets`, `total_dim`, `strategies` (e.g. `hybrid`, `mrmr`, `pca`), `per_budget` (per-strategy `mi`, `mi_density`, `effective_rank`, `downstream_r2`, `n_features`).
- **Headline:** at each budget, `downstream_r2` for hybrid vs mRMR vs PCA.

### c18 — Diversity vs Depth
- **Question:** when budgeting K dimensions, is it better to spread across many reps (diversity) or concentrate on few (depth)?
- **Keys:** `metric`, `budgets`, `n_reps`, `rep_ranking_full_data`, `per_budget`.
- **Headline:** at each budget, score vs number of reps included → curve shape (peak at small n_reps = depth wins; peak at large = diversity wins).

### c19 — Importance Weights as Training Signal
- **Question:** does using importance weights at training time (not just selection) help?
- **Keys:** `metric`, `lower_is_better`, `budget`, `conditions`, `per_space`.
- **Headline:** weighted vs unweighted training delta in `conditions`.

### c21 — Single-Rep Augmentation Test
- **Question:** does uncapped augmentation lift individual reps even without hybridization?
- **Keys:** `metric`, `lower_is_better`, `n_folds`, `n_samples`, `aug_reps`, `aug_dims`, `aug_groups`, `rep_results`, `summary`.
- **Headline:** per-base-rep delta (uncapped vs baseline) in `summary`.

### c22 — Hybrid Explainability
- **Question:** can we trace every selected feature back to its source rep?
- **Keys:** per-fold blueprint contents — rep attribution table, per-feature sources.
- **Headline:** dominant rep(s) in the blueprint and what fraction of the budget they occupy.

### c23 — Evaluator–Downstream Model Matching
- **Question:** does using the same model family for the internal evaluator and the downstream model help?
- **Keys:** matched-vs-mismatched score per downstream model + `score_mean`, `score_std`, `n_features`, `budget`, `allocation_method`, `importance_method`.
- **Headline:** matched - mismatched delta.

### c24 — Blueprint Transfer Across Folds
- **Question:** do feature-selection blueprints discovered on one fold generalize to other folds (defense against feature-selection overfitting)?
- **Keys:** `n_folds`, `total_features`, `metric`, `lower_is_better`, `transfer_matrix`, `per_target`, `aggregate`, `stability`, `fold_blueprints`.
- **Headline:** `aggregate` mean off-diagonal score in `transfer_matrix` (fold i blueprint applied to fold j) ≈ on-diagonal. Big drop → blueprint overfits.

---

## How to spot a "this dataset failed" run

In any JSON, presence of a top-level `'error'` key means the experiment caught its own failure and wrote a stub. Treat as missing data.

For c02 specifically: if the file is missing entirely (the `[saved …]` line never printed in the slurm log), the experiment crashed before save. Fixed in commit after 2026-04-28 — the prior bug was a `KeyError: 'gp'` from iterating internal-evaluator extras (knn, gp) that had no baseline entry.

---

## Reading order for a new dataset

1. **c14** (if it includes this dataset) — does hybrid beat best single rep at all? If no, the rest is mostly negative results.
2. **c02** `best_by_model` and the per-model delta vs `baselines` — which models benefit, by how much.
3. **c02** `heatmap` — eyeball the model × budget grid for the matching pattern.
4. **c09** ANOVA percentages — confirms (or doesn't) that model/budget interaction is real.
5. **c16** `Winner` per family — is the budget really doing regularization work, or is tuned-L2 enough?
6. Everything else is supporting evidence for specific claims.
