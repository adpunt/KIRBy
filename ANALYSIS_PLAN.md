# KIRBy Results Analysis Plan

Goal: read ALL experiment results (complexity / similarity / noise) the way the
user actually wants — **individual datasets, domain-segmented, NO cross-dataset
or cross-domain averages as headlines** — without one git-pull/run/paste cycle
per experiment.

Core question every experiment ultimately serves: **does a mixed-representation
"hybrid" beat the single best representation on the deployed model?** User's
primary domain = small-molecule QM/ADMET (caco2, herg_ki, adme_*). They ran
everything seriously but care most about molecular, and about *individual*
well-studied datasets with noteworthy results — not roll-ups.

## Hard constraints (do not relitigate)
- NO averages as the headline. NO domain averages. Individual datasets first;
  counts (+/=/−) and ranges over means. Domain segmentation is allowed
  (molecular / antibody / tcga / genomic); pooling across domains is not.
- Metric DIRECTION must be corrected: lower_is_better (RMSE/MAE) for molecular &
  reaction regression; everything else higher-better. `improvement()` already
  handles this in `tests/analysis/quick_hybrid_summary.py`.
- Same model on both sides of every hybrid-vs-base comparison. Compare the model
  you'd actually deploy (best single-rep score), not best-model-per-side.
- Read-only. User runs on the server; Claude has NO server access (pastes all
  output). Build readers from experiment SOURCE (local) + schema recon keys.

## Execution model (fixes the round-trip problem)
ONE analyzer, `tests/analysis/quick_hybrid_summary.py`, with a reader function
per experiment, all invoked in a single `__main__`. User does ONE
`git pull && python tests/analysis/quick_hybrid_summary.py tests/results` and
pastes one output. To write the ~35 new readers efficiently: fan out agents,
each assigned a handful of experiments — agent reads the experiment `.py` +
records the exact result-JSON paths it writes, returns a reader spec (which keys
hold the answer to the core question, per-dataset, direction-aware). We assemble
the readers, user runs once.

## Experiment inventory & tiering
(n = datasets with result files, from schema recon 2026-06-29)

### TIER 1 — directly test the core claim (hybrid vs best single rep)
| exp | n | what it answers |
|-----|---|-----------------|
| c02 complexity_matching_grid | 39 | **DONE** — top-2 deployed model, base vs best hybrid, by-model×domain, hybrid size |
| c11 ensemble_vs_hybrid | 11 | does feature-level mixing beat just ensembling the models? |
| c16 feature_selection_as_regularization | 9 | is the hybrid's budget cap doing the work (does selection alone explain lift)? |
| c17 information_theoretic_decomposition | 8 | does combining reps add *information* or just dimensions? |
| c18 diversity_vs_depth | 8 | wide-diverse vs deep-single-family hybrids |
| s02 hybrid_similarity | 26 | **DONE** — hybrid distance vs best single rep (SAC) |
| s05 knn_prediction | 23 | hybrid-space kNN vs single-rep kNN |
| s07 noise_bridge | 22 | hybrid under label noise vs single rep |

### TIER 2 — when / why it works (conditions on the core result)
| exp | n | what it answers |
|-----|---|-----------------|
| c05 learning_curves | 22 | does the hybrid advantage grow or shrink with sample size? |
| c06 within_family_gradient | 22 | how lift varies across a rep family |
| c07 data_complexity_partitions | 18 | lift on easy vs hard data partitions |
| c08 calibration | 14 | does the hybrid stay calibrated? |
| c09 formal_analysis | 12 | anova / effective-rank / bias-variance of the lift |
| c10 noise_robustness | 12 | hybrid vs single rep across noise sigmas |
| c13 forward_compatibility | 10 | does adding a *new* rep change the ranking (core thesis) |
| s06 applicability_domain | 23 | distance-error correlation, reliability |
| s08 optimal_dimension | 23 | the budget/dim sweet spot |
| s12 cross_domain_cliffs | 23 | cliff behaviour across domains |
| s13 clustering_retrieval | 23 | hybrid space for retrieval/clustering |
| c04 importance_as_metric | 39 | **DONE** — importance-weighted kNN/retrieval |
| s01 similarity_baseline | 37 | **DONE** — best-rep SAC per dataset |

### TIER 3 — diagnostics / properties (light touch; report only if noteworthy)
c01 rep_characterization (42), c15 ml_theory_depth (10), c19 importance_weighted_training (8),
c21 single_rep_augmentation (7), c22 explainability (7), c23 evaluator_model_matching (8),
c24 blueprint_transfer (8), s03 feature_concentration (25), s10 importance_downstream (23),
all noise n01–n13.

### EMPTY / THIN — confirm then skip
- s04 activity_cliffs (26) — `skipped` on every dataset.
- s11 bayesian_optimization (22) — `skipped` on every dataset.
- noise n05–n13 — only 2 datasets each (barely ran); n01–n04 have 9.
- c03 cross_dataset_transfer — 0 results (single combined job, depends on c02; not yet submitted).

## Phased order
1. Tier 1 readers (the claim) → one run → read individual datasets, molecular first.
2. Tier 2 readers (the conditions) → one run.
3. Tier 3 + decide which thin/empty ones to bother with.
4. Submit c03 on the server when the cross-dataset meta-analysis is wanted.

## Open decisions for the user (engineer owns methodology)
- Confirm the Tier 1/2/3 split — anything mis-tiered? (e.g. is forward-compat c13
  Tier 1 for the thesis rather than Tier 2?)
- OK to fan out agents to write the readers, or write them inline one tier at a time?
- Any experiments to drop entirely vs. the "report only if noteworthy" treatment?
