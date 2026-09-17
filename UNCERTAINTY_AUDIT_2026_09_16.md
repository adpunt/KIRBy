# Uncertainty-display audit — 2026-09-16

Every .py file in `tests/experiments/{similarity,similarity_metrics,complexity,noise,federated}/`
plus the top-level `tests/experiments/opig_*.py` files (140 files total) was read in full by
8 parallel agents — not grepped. Each agent recorded every place the code computes or displays
a standard deviation, standard error, confidence interval, bootstrap interval, p-value, or
residual spread, what sample size (n) it's computed over, what that n counts (replicates,
CV folds, seeds, silos, datasets), and whether the surrounding code states any reason that
n is enough. No judgment is attached to any row — these are facts, not a verdict.

A row means: one place in the code where an uncertainty/significance number is computed and
either printed, plotted, or written to a result file.

Gotcha checked in every file: "CI" is also used in this codebase as an abbreviation for
Concordance Index (a predictive-performance metric), unrelated to a confidence interval.
Agents read context on every occurrence; false positives from this are noted per section
where relevant.

---

## Cross-cutting pattern

The dominant construct across the whole suite is `mean ± std` (or a bare `std`/`sd` field
written to a result JSON) computed across k cross-validation folds, where k comes from
`config['n_splits']`, defaulting to `DEFAULT_N_FOLDS = 5`. This appears in nearly every
experiment file that runs cross-validation. In the large majority of instances, there is no
comment near the computation stating why 5 folds is an adequate sample for that statistic.

A second recurring construct is `mean ± std` across a small number of repeated
noise-injection draws or random seeds — typically 3 (e.g. `NOISE_SEEDS = (0,1,2)` in
similarity_metrics/aggregate.py, `N_REPLICATES = 3` in noise/n16, `FED_SEEDS = [0,1,2]` in
federated_helpers.py). Whether these are justified varies sharply by file — see below.

## Already-correct examples (worth reading before deciding anything)

These files already state, in their own comments, a reason for the sample size behind their
uncertainty display — several of them state almost exactly the rule you just gave:

- **`tests/experiments/noise/n19_measurement_count.py:806-810`** — docstring says: "at three
  draws over five folds there is no number of replicates that would support a confidence
  interval, and the suite's standing rule is raw spread only."
- **`tests/experiments/noise/n18_clean_the_real_labels.py:33-37`** — "THE HEADLINE IS A COUNT
  OF FOLDS, NOT A MEAN... the mean is never the headline."
- **`tests/experiments/noise/n16_analyse.py:210-245`** — explicitly refuses to read anything
  from a single noise draw ("one noise draw only, so there is nothing for the drift above to
  clear").
- **`tests/experiments/noise/n17_replicate_removal.py:527-530`** — "the spread across draws
  is the spread over WHICH measurement became the training label. A difference between arms
  smaller than this is not a difference between arms."
- **`tests/experiments/noise/n12_importance_weighted_distance.py:11-16, 929-934`** — "treat
  any condition ordering as suggestive only unless the gap clearly exceeds that spread. Do
  not read the boolean 'helps'/'redundant' flags as significance tests."
- **`tests/experiments/complexity/c13_forward_compatibility.py:62-64`** — "No significance
  testing is reported. Wilcoxon on <=5 non-independent CV folds is underpowered... present
  effect sizes with fold spread, not p-values."
- **`tests/experiments/complexity/c33_budget_overfitting.py:248-255`** — "nothing is called
  double descent unless BOTH the rise... and the fall... are larger than the folds' own
  spread... A shape smaller than the fold spread is reported as inconclusive."
- **`tests/experiments/complexity/c31_perf_weighted_global_selection.py:220-221`** /
  **`c32_allocation_strategy_cross.py:149-150`** — "Only NFOLDS=5 paired folds, so any
  per-fold sign/Wilcoxon test is underpowered — the bootstrap over stored predictions is used
  instead"; "the deliverable is per-cell WIN-COUNTS across the 60 cells, NOT per-cell
  significance."
- **`tests/experiments/complexity/c09_formal_analysis.py:144-156`** — deliberately counts
  each (model, budget) configuration once in its ANOVA, not once per fold, because "counting
  each of them inflated the observation count fivefold and the p-values with it."
- **`tests/experiments/similarity_metrics/base_rep_comparison.py`** (module docstring) —
  "Every number carries a 95% bootstrap confidence interval, so 'wins' means the intervals
  actually separate, not a third-decimal difference." Classification libraries get no
  interval; a resample whose own value is undefined is dropped and counted.
- **`tests/experiments/opig_graphinity_ddg.py:336-344`** — the one place that explains why
  NO spread is shown: "Single train/test pair — no fold variance to report."
- **`tests/experiments/opig_trastuzumab_enrichment.py:99-101`** — flags a ~10-molecule
  top-1% enrichment estimate as high-variance and recommends a more stable alternative
  (E@5%/E@10%) instead of suppressing it silently.

## Highest-leverage single finding

**`tests/experiments/federated/federated_helpers.py:329-330, 345-376, 2832-2866`** —
`FED_SEEDS = [0, 1, 2]` and `aggregate_seed_runs`/`run_all` wrap *every* federated experiment
(fed01 through fed05) in a 3-seed mean±std, unconditionally. The only justification given
anywhere is a citation, "3-seed sweep for error bars (user decision 2026-06-04)" — no
statistical reasoning for why 3 is enough. Because this is centralized in one helper, it is
the single place where a decision would change the most downstream results at once, unlike
every other finding below, which is local to one file.

## Inconsistency noted

`tests/experiments/noise/_smoke_n06.py` and `n06_bindingdb_ground_truth.py`'s underlying
`_ground_truth_providers.py` compute a replicate group's spread with `ddof=0` (population
std), while `n17_replicate_removal.py` and `n19_measurement_count.py` explicitly use
`ddof=1` and document why. Neither is in a shared helper, so the same "spread of repeat
measurements" quantity is computed two different ways depending on which file you're in.

---

## similarity/ (16 files, agent: a8ad844857b5feefa)

`DEFAULT_N_FOLDS = 5` (`src/kirby/find_best_hybrid.py:3444`) feeds every s0x file's fold
count via `config['n_splits']`.

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| s11_bayesian_optimization.py:860,1656 | `auc_std` across BO replicate seeds | n=3 seeds (kwarg/CLI default) | none stated |
| s11_bayesian_optimization.py:826-835,1374 | same std, per budget checkpoint/iteration | n=3 seeds | none stated |
| s11_bayesian_optimization.py:1424-1425,1455-1459 | printed "AUC=X+/-Y" + summary Std column | n=3 seeds | none stated |
| s09_cross_dataset_summary.py:668-694 | Wilcoxon p-value on per-dataset deltas | n=datasets, gated `>=3` (`MIN_DATASETS_FOR_WILCOXON`) | mechanical floor only, no power argument |
| s05_knn_prediction.py:560-561,599-601 | `score_std` of a hybrid cell's CV score | n≤5 folds | none stated |
| s05_knn_prediction.py:696,713-716 | `best_k_std` of best-k kNN baseline | n≤5 folds | none stated |
| s05_knn_prediction.py:775,786-790 | `score_std` of pool's best-single-rep score | n≤5 folds | none stated |
| s05_knn_prediction.py:900,921-923 | `score_std` of (rep, model) reference score | n≤5 folds | none stated |
| s06_applicability_domain.py:668-736 | conformal PICP/MPIW averaged over folds | n=5 folds, n_cal≥10 per fold | is the experiment's subject; no n-adequacy argument given |
| s06_applicability_domain.py:676,683-684 | interval-width construction (tree std → interval) | std over RF's n_estimators (not this file's scope) | mechanism, not a reported confidence claim |

Lower priority (large n, not a small-sample concern): s01/similarity_helpers SALI std (over
all molecules), s01 permutation-test p (n_perm=200), s01 Spearman p (n up to 5000 pairs),
s10 std-as-feature (q-RASAR input, not a displayed uncertainty), s13 std-as-metric-formula.

Files read in full, nothing found: `__init__.py`, `_run_similarity_locally.py`,
`s02_hybrid_similarity.py`, `s03_feature_concentration.py`, `s04_activity_cliffs.py`,
`s08_optimal_dimension.py`, `s12_cross_domain_cliffs.py`. No plotting code anywhere in this
directory (no matplotlib/seaborn found). No "CI" abbreviation occurs in this directory at all.

## similarity_metrics/ (30 files, agent: aab06af58daf43f16)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| triplet_similarity.py:313-368 | mean±std of readacross_rho/struct_mcs_rho across folds | n=`n_splits`, 5 (4 in `--quick`) | none stated |
| aggregate.py:146-181 | mean/sd of detection-AUROC/recovery-corr per (sigma,rate) cell | n=3 injection seeds (`NOISE_SEEDS=(0,1,2)`) | stated (thin): justifies sweeping doses, not why 3 seeds specifically |
| aggregate.py:196-220,264-272 | sd of det/rec across the 9 dose×rate grid cells | n=9 cells, each an n=3-seed mean | none stated for this specific sd |
| base_rep_comparison.py:156-175 | `boot_ci()` — 95% bootstrap CI on Spearman rho | n_boot=1000, resample n=held-out variants; undefined resamples dropped | explicitly stated (see "already-correct" above) |
| base_rep_comparison.py:178-207 | `paired_boot_diff()` — 95% CI on the difference between two reps | same n_boot=1000 | explicitly stated: marginal-interval overlap is rejected as a tie test |
| base_rep_comparison.py:328-402 | `[ci_lo, ci_hi]` printed beside every rho; verdict read off whether the diff interval crosses zero | same as above | explicitly gated: classification libraries get no interval at all |
| scaffold_ddg_experiment.py:159-187 | `sd_over_complexes` — spread of per-complex Spearman | n=complexes scored (≥25 variants each, count varies) | cites a 2026-09-11 user decision for the *statistic choice*, not n-adequacy |
| run_battery_moleculeace.py:148-162 | paired Wilcoxon, per-metric rho vs ECFP4:Tanimoto | n=targets with finite pairs, gated `>=6` | mechanical floor only |
| run_battery_moleculeace.py:184-197 | same Wilcoxon, per noise-sweep setting | same gate | none stated |

Descriptive only, not a confidence claim (excluded from concern per agent's own read):
mcs_cache.py:162 (`mcs mean/std`, n=8000 sampled pairs), scaffold_experiment.py:348-350
(ground-truth pair-distance spread, n=8000 pairs).

Files read in full, nothing found: `_data.py`, `_heldout.py`, `_nanrank.py`,
`aug3d_similarity.py`, `augmented_descriptor_reps.py`, `build_local_datasets.py`,
`build_moleculeace.py`, `creative_hybrids.py`, `creative_hybrids_v2.py`,
`extra_fingerprints.py`, `feature_ablation.py`, `feature_type_study.py`,
`learned_embeddings.py`, `nary_similarity.py`, `rankers.py`, `scaffold_feature_isolation.py`,
`scaffold_reps.py`, `shape_similarity.py`, `simeval.py`, `similarity_noise_cleaning.py`,
`structural_embeddings.py`, `structural_yardstick.py`, `winning_recipe_honest.py`,
`winning_recipe_vs_standard.py`. No "CI" false positives found. No plotting code anywhere in
this directory.

## complexity/ part A: c01–c16 (18 files, agent: a5bc0975eec090a5a)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| c09_formal_analysis.py:38-138 | ICC(1,1) reliability between model/budget pairs | n=subjects, floor 2 | none stated beyond mechanical floor |
| c09_formal_analysis.py:265-306 | ANOVA F-stat/p-value on model/budget/interaction | n=(model,budget) configs, one obs each (not per-fold) | explicitly stated: deliberately avoids fold-count inflation |
| c14_cross_domain_summary.py:531-552,572-578 | Wilcoxon p-value (overall + per-domain) | n=tasks/datasets, gated `>=3` | mechanical floor only |
| c14_cross_domain_summary.py:479-484,500-516,584-585 | mean/median/std of per-task delta, printed with ± | n=tasks (per-domain case has no minimum-n gate at all — can be n=1) | none stated |
| c13_forward_compatibility.py:62-64,885-1197 | score_std / "±" fold spread for every condition | n≤5 folds | explicitly stated (see "already-correct" above) |
| c08_calibration.py:415-433,300-343 | conformal PICP/MPIW/cal_error, averaged over folds | n_cal≥20 per fold; averaged over n=5 folds | is the experiment's explicit subject; no n=5-adequacy argument |
| c08_calibration.py:360-362,437-441 | Spearman(uncertainty, error), p-value discarded | gated `>=3` validation samples | none stated; p-value computed then discarded |
| c11_ensemble_vs_hybrid.py:696,767,883,1131,1513 | score_std, every arm (best_single, pca_matched, hybrid, oracle, ensemble) | n≤5 folds | none stated |
| c12_modality_ablation.py:605-620,659-674,882,982 | score_std, per-rep and per-combination | n≤5 folds | none stated |
| c07_data_complexity_partitions.py:310 | score_std for best config per subpopulation | n=sub_folds, as low as 2 | none stated |
| c06_within_family_gradient.py:324-357 | val_std printed as "±" per cell | n≤5 folds | none stated |
| c06_within_family_gradient.py:393-426 | Spearman rho+p, capacity_index vs optimal_budget | n≥3 variants (gated) | gate is a data-validity check, not a power argument |
| c03_cross_dataset_transfer.py:456-491 | Spearman rho+p per (descriptor, model) | n=datasets with a winning budget, gated `>=2` (n=3 under file's own default) | mechanical floor only |
| c03_cross_dataset_transfer.py:579-587 | "Budget Std" across datasets | n=datasets (as few as 3) | none stated |
| c02_complexity_grid.py:230,250,510,542 | score_std, written to every result record | n≤5 folds | none stated (docstring steers readers to a paired delta instead, but doesn't say so of std itself) |
| c01_rep_characterization.py:106-116 | score_std, per (rep, model) baseline | n≤5 folds | none stated |
| c15_ml_theory_depth.py:741-777 | "instability_ratio" variance ratio, double-descent buckets | n=budget points per bucket, floor 2 | explains why n=1 is refused, not why n=2 is adequate |
| c16_feature_selection_as_regularization.py:505-511 | std(R²) across a regularization sweep | n=regularization settings (4-6), gated `>=2` | is the experiment's stated subject |

Files read in full, nothing found: `__init__.py`, `_cache_evidence_local.py`,
`complexity_helpers.py`, `c04_importance_as_metric.py`, `c05_learning_curves.py` (means over
3 subsampling seeds, but no spread ever attached).

## complexity/ part B: c17–c43 (14 files, agent: a043aafe822306205)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| c24_blueprint_transfer.py:50-114,485-491,520-526 | Nogueira stability SE + 95% CI | n=M folds that produced a scored blueprint, ≤5 | none stated for M≈5 adequacy; only guards M<2 |
| c31_perf_weighted_global_selection.py:426-458,840-841 | bootstrap 95% CI + p_boot on pooled-OOF R² delta | n_boot=1000 over groups; underlying OOF pooled from NFOLDS=5 | explicitly stated (see "already-correct" above) |
| c32_allocation_strategy_cross.py:556-584,429-430 | identical bootstrap CI/p_boot | same | explicitly stated: "deliverable is per-cell WIN-COUNTS, NOT per-cell significance" |
| c20_signal_saturation.py:26-28,120-150,318-370 | std(ddof=1) of 5 full-width fold scores → plateau tolerance | n=5 folds | statistic choice justified (2026-09-11 user decision); n=5 adequacy not argued |
| c33_budget_overfitting.py:152-158,288-290,342,349,567-569,601,615-627 | fold-delta std used as significance gate for "double descent" claims | n≤5 folds | explicitly stated (see "already-correct" above) |
| c21_single_rep_augmentation.py:404,790,793 | hybrid_std/base_std/aug_std | n≤5 folds | none stated |
| c22_explainability.py:326,330-332,387,408 | shap_by_rep_std, printed "±" | n=folds with successful SHAP, ≤5 | none stated |
| c23_evaluator_model_matching.py:146,338 | score_std surfaced from nested_cv_evaluate | n≤5 folds | none stated in this file |
| c31_perf_weighted_global_selection.py:661-670,723-732 | std over NFOLDS=5 per-fold weight/slot values | n=5 | descriptive purpose stated, not adequacy |

Files read in full, nothing found: `c17_information_theoretic_decomposition.py`,
`c18_diversity_vs_depth.py`, `c19_importance_weighted_training.py`,
`c43_buchwald_step3.py`, `c43_buchwald_signal.py` (only a full-dataset label std used to
normalize RMSE — not a small-sample statistic), `forward_compat_reps.py` (a static
literature-citation catalog; its "±" numbers are quoted from external papers, explicitly
labeled as not KIRBy-computed). `c21`'s "residual_correlation" is a feature-ranking method
name, not a statistical residual analysis — excluded as a name collision.

## noise/ part A: helpers + n01–n15 (17 files, agent: a79d4c03133754f5d)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| n15_clean_vs_noisy_hybrid.py:466-467 | `oracle_advantage_std_over_draws` | n≤3 draws (`NOISY_SEARCH_SEEDS=(0,1,2)`) | explicitly stated: used as a significance-style gate against level differences |
| n15_clean_vs_noisy_hybrid.py:421,423 | `clean_bp_score_std`/`noisy_bp_score_std` | n≤5 paired folds | none stated |
| c10_noise_robustness.py:733-748 | Nogueira 95% CI, printed | n=M folds at level 0, ≤5 | only mathematical validity guard (M>2), not adequacy |
| n06_bindingdb_ground_truth.py:548 | Spearman rho+p, n06 vs n11/n13 estimator orderings | n=common estimators, gated `>=5` (`MIN_ESTIMATORS_FOR_TRANSFER`) | states the floor exists, not why 5 is enough |
| _ground_truth_providers.py:216,263,723 | std of a replicate group's labels (the real ground-truth signal) | n=group size, floor 2 (`MIN_REPLICATES=2`) | only the ddof convention is justified, not n=2 adequacy |
| _probe_chembl_herg.py:217,257 | per-compound spread of ChEMBL records | n=records per compound, floor 2 (`n_records>1`) | only ddof convention justified |
| n02_sample_weighting.py:337,345,650,658,700 | unweighted_std/weighted_std/score_std | n≤5 folds | none stated |
| n09_lnl_survey.py:637,658 | std over combined (fold×draw) passes | n up to 5×3=15, or 5 at sigma=0 | averaging-over-draws is justified; the std itself is not |
| n12_importance_weighted_distance.py:861,830-833 | `per_condition_recovery_std` | n generally >5 (pooled) | explicitly and repeatedly stated (see "already-correct" above) |

Files read in full, nothing found: `__init__.py`, `noise_helpers.py`,
`_probe_estimator_distinctness.py`, `_similarity_hybrid_space.py`,
`n01_activity_cliff_framework.py`, `n07_remediation.py`, `n11_estimation_remediation_grid.py`
(recovery means over folds as low as 2, but no std/CI ever attached), `n13_noise_distributions.py`,
`n14_activity_cliff_landscape.py` (residual SD pooled over the full sample, not small-n). Every
"CI" occurrence in this scope is Concordance Index.

## noise/ part B: n16–n19 + smokes (19 files, agent: a766ef8d29cc1b27d)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| n17_replicate_removal.py:530 | `std_over_draws` | n=3 draws (`N_DRAWS`, as low as 2 in smoke) | explicitly stated (see "already-correct" above) |
| n17_replicate_removal.py:184,375-376 | `true_spread` per entity (repeat-measurement std) | n=entity's own measurement count, as low as 2 | ddof choice justified; n=2 adequacy not argued |
| n19_measurement_count.py:211 | `var_all` per-molecule variance, feeds fit weights | n=measurement count, as low as 2 | docstring (67-71) acknowledges the risk directly; is why the `calibrated` alternative exists |
| n19_measurement_count.py:806-810,835,843-845 | `spread_over_folds`, gates a "gain exceeds fold spread" flag | n≤3 draws×5 folds | explicitly stated (see "already-correct" above — near word-for-word your rule) |
| n18_clean_the_real_labels.py:258 | `spread_over_folds`, reported beside `folds_won` | n≤5 folds per cell | explicitly stated: headline is the fold-win count, never the mean |
| n16_analyse.py:210-245 | range (max−min) of per-draw drift medians | n=independent draws in the result file | explicitly stated, with an n<2 guard that refuses to say anything |
| n16_importance_under_noise.py:154-158 | `N_REPLICATES=3` default | n=3 draws | explicitly stated: "one draw gives no run-to-run spread" |
| _smoke_n06.py:75-114 | population std (ddof=0) of 2-3 synthetic measurements | n=2-3 | see "inconsistency noted" above |

Lower priority (descriptive ranges, not framed as CIs): n16_analyse.py:193-207,
n16_importance_under_noise.py:1043-1051,1073-1086 (median/min/max over cells, n generally
>5). Files read in full, nothing found: all 9 `_smoke_audit_*`/`_smoke_harness`/`_smoke_n0*`
files not listed above are pure assertion-based unit tests with no reported uncertainty. No
p-values, bootstrap, or plotting anywhere in this batch. No "CI" false positives.

## federated/ (12 files, agent: a6d9859a6db579032)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| federated_helpers.py:329-330,345-376,2832-2866 | `FED_SEEDS=[0,1,2]`, universal seed-level mean±std wrapper applied to every fed01-05 result | n=3 seeds | cites a user decision (2026-06-04), no statistical reasoning — see "highest-leverage finding" above |
| fed01_noise_heterogeneity.py:907-960,995-1008 | per-fold mean±std, all conditions × noise levels | n=5 folds | bug-fix history given, no n=5-adequacy argument |
| fed01_noise_heterogeneity.py:150-162 | docstring on the 3-seed error bar this file feeds | n=3 seeds | explains a past bug (seeds gave no real variance), not current adequacy |
| fed01_noise_heterogeneity.py:744-766 | Spearman rho+p, silo-noise vs silo-budget/score | n=5 (silos) | none stated |
| fed02_rep_diversity_silos.py:636-676,684-750,803-853 | mean±std, 6 conditions × several metrics | n=5 folds | bug-fix history given, no adequacy argument |
| fed02_rep_diversity_silos.py:78-119 | mean over inner folds, picks C5's top-3 reps | n=3 (`INNER_RANK_FOLDS`) | justified by compute cost, not statistics |
| fed03_synthetic_private.py:757-815,1221-1270 | mean±std tables, Parts A and B | n=5 folds | K=3 real data sources is justified; fold count is not |
| fed04_cluster_aware_importance.py:446-464,538-554 | mean±std, conditional/global MCC etc. | n=5 folds | none stated |
| fed04_cluster_aware_importance.py:289-291,401,466-470 | mean pairwise Spearman, cluster importance stability | n=K=5 silos/clusters | returns None below 3 reps (stated floor); n=5 adequacy not argued |
| fed05_antibody_federated.py:129-133,325,369,426,692 | every condition uses a single 80/20 split, no internal fold loop | n=1 split per silo | leakage-order fix justified; single-split design itself is not |
| fed05_antibody_federated.py:477-494 | mean pairwise Spearman stability within a silo | n=`K_INTRA=3` sub-clusters | none stated |
| fed05_antibody_federated.py:9-12,76-86 | comparisons across 3 real silos (ddg_combined, abagym_EGFR, flab_thermo_tresanco) | n=3 datasets | provenance for *which* 3 justified; not that 3 is enough to generalize |

Files read in full, nothing found: `__init__.py`, `_smoke_driver.py`, `caco2.py`, `herg.py`,
`logd.py` (thin CLI wrappers), `lhasa_herg_loader.py`. No "CI" abbreviation occurs in this
directory (metrics are MCC/BAC/AUROC/F1/RMSE/Spearman/Pearson/R²).

## opig_*.py (13 files, agent: a6818c3b460e3996b)

| file:line | what | n & what it counts | justification |
|---|---|---|---|
| opig_kingsmith_component_splits.py:369-387 | nanstd of r2/rmse/spearman across leave-one-group-out folds | n varies by split axis: additive=23, aryl_halide=15, ligand=4, base=3 | docstring flags the small splits as "hardest"/"extreme OOS," but doesn't argue the std is adequate at n=3-4 |
| opig_shared_vectors_census.py:302-318,438 | std of labels within a tied-representation group | n=group size, as low as 2 | none stated |
| opig_trastuzumab_enrichment.py:99-101 | top-1% enrichment cutoff | n≈10 molecules per fold | explicitly stated (see "already-correct" above) |
| opig_trastuzumab_enrichment.py:300-324,502-520 | mean±std of accuracy/AUROC/enrichment across folds | n≤5 folds | none stated |
| opig_graphinity_ddg.py:200 | std of pearson/spearman/r2/rmse across folds | n≤5 folds | none stated |
| opig_graphinity_ddg.py:336-344 | single train/test split, no std computed | n=1 | explicitly stated (see "already-correct" above) |
| opig_blueprint_transfer.py:576-579,631-632,663-678 | fold std / delta_std, printed with fold count alongside | n≤5 folds (1 under `--mode intermediary`) | shows count+spread together; win threshold (0.001 RMSE) stated, adequacy of n not argued |
| opig_shared_vectors_remedy.py:706-719 | outer-fold std, sets augmentation-margin thresholds | n≤5 folds | std-as-normalizer justified; n=5 adequacy not argued |
| opig_activity_cliff_qsar.py:182-186,391-397 | std of rmse/mae/r2/spearman across folds | n≤5 folds | none stated |
| opig_cdr_flexibility.py:203-210 | std across StratifiedGroupKFold folds | n≤5 folds | none stated |
| opig_cgr_comparison.py:306-311 | std across GroupKFold folds | n=min(5, n_groups) | none stated |
| opig_ginkgo_developability.py:207,227,256-266 | std across folds, baseline + hybrid arms | n=min(5, unique folds) | none stated |
| opig_humanness.py:229-234 | std across StratifiedGroupKFold folds | n≤5 folds | none stated |

Files read in full, nothing found: `opig_scaffold_hybrid.py` (reports per-fold delta list, no
spread statistic attached), `opig_scaffold_similarity.py` (every number is a Spearman
correlation against structural ground truth — "NO model is trained" per its own docstring).
No genuine confidence-interval code, bootstrap, t-test, or plotting anywhere in this batch. No
"CI" false positives.
