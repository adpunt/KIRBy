# Wiring the selection shape

Three counts over the 77 experiment files. 0 take the shape as a rename with every
reported number unchanged. 39 take it, and in each of those at least one reported number
moves or a new row appears. 38 do not take it at all.

The shape is: for each downstream model, the top 2 base representations and the top 2
hybrids, giving 4 rows per downstream model and 20 rows over 5 downstream models. Four
functions now compute it, in `tests/experiments/_shared.py`: `top_base_reps` at :6669,
`top_hybrids` at :6868, `rank_downstream_models_by_base_reps` at :7112,
`select_model_rep_pairs` at :7227. `top_base_reps` reads `search.baselines`. `top_hybrids`
reads `search.all_results`. Neither fits a model. `selection_diff` at :7317 reports what a
switch renamed. `tests/experiments/test_top_n_selection.py` holds 33 tests; 33 pass, 0
fail.

Two model lists exist and this document always means the first. The downstream model is
the one scored at the end: `DOWNSTREAM_MODELS` at `tests/experiments/_shared.py`:539,
holding rf, xgboost, svm, dnn and lightgbm. The internal evaluator is the model used
inside a search to pick features: `DEFAULT_INTERNAL_EVALUATORS` at
`src/kirby/find_best_hybrid.py`:3483, holding match, rf, lightgbm and dnn.

No experiment file has been changed. The selector was added to
`tests/experiments/_shared.py` and `tests/experiments/test_top_n_selection.py` is new.
`n07_remediation.py`, `n11_estimation_remediation_grid.py`, `_smoke_repair_targets.py` and
`submit_opig_experiments.sh` carry edits made before this work started.

## Applies cleanly -- a refactor, no number moves

None of the 77. Every experiment that can take the shape either gains rows or changes a
number it already reports.

Three come closest, and their entries are in the next section: c31, c32 and c33. In each
of the three, every number printed today keeps its value, and a second reference base
representation is added beside the first. They cost 0 new model fits.

## Adopting the shape would move the numbers

**c02** -- `c02_complexity_grid.py`:277-281 sorts base representations by score with no
tie-break and takes 2 at :336; :241 takes 1 hybrid per downstream model from
`search.best_by_model`. The shape takes 2 hybrids per downstream model, 10 hybrid rows
over 5 downstream models where there are 5 today. Baseline scores are rounded to 4
decimals at `find_best_hybrid.py`:5525, so exact ties exist; today a tie is broken by
representation build order and `top_base_reps` breaks it by representation name. Where a
tie flips the rank-2 representation, the per-model top-rep-dim score at :400-405 and
`reselected_delta_by_condition` for that condition both move. `importance_wins` at :452-455
counts 5 winners over 5 downstream models today and would count 10.

**c03** -- `c03_cross_dataset_transfer.py`:151 takes 1 hybrid per downstream model through
`get_best_search_for_downstream`; :362 dumps one reference model's whole `baselines` dict
unranked. The shape names 2 base representations per downstream model, which nothing in
the file names today, and 2 hybrids per downstream model. The rank-1 hybrid should hold,
because hybrid `score_mean` is not rounded. One direction rule differs:
`get_best_for_downstream` reads the metric name, `top_hybrids` reads the stored flag, and
under a blend objective containing prediction the two disagree and the winner flips.
Whether any submit script here passes a blend objective is unverified.

**c05** -- `c05_learning_curves.py`:282-284 keeps 1 base representation per downstream
model by strictly-better comparison; :249-251 keeps 1 hybrid per downstream model.
`results['top3_base_reps']` at :473 stops being 3 representations ranked by rf and becomes
2 representations for each of 5 downstream models. The third rf-ranked representation
stops being reported and 8 entries for xgboost, svm, dnn and lightgbm start being
reported. `best_single` and the same-fold delta at :417-419 move only where two
representations tie at 4 decimals.

**c07** -- `c07_data_complexity_partitions.py`:288 keeps 3 base representations ranked by
rf; :318-323 keeps 1 hybrid per downstream model. `top3_base_reps` becomes 2
representations for each of 5 downstream models. `best_by_model` gains a second, worse
hybrid per downstream model. Kept additive, `delta_budget` and `config_differs` at
:453-457 do not move, because they read the cross-model winner at :296-306.

**c11** -- `c11_ensemble_vs_hybrid.py`:708 hands every base representation's per-fold
scores to `fold_reselection` as the baseline arm. Restricting that arm to 2 base
representations makes `best_single_reselected` equal or worse on every held-out fold, never
better. `hybrid_minus_single` and `hybrid_minus_single_mean_of_folds` at :1226 and :1233
therefore grow without the hybrid improving. That is the arm c11 exists to test the hybrid
against. c11's own base scores come from its own fits under its own scaling at :305-319,
so `search.baselines` ranks on different numbers than c11 reports.

**c14** -- `c14_cross_domain_summary.py`:289 keeps 3 base representations ranked by rf.
The shape reports 2 representations for each of 5 downstream models, 10 rows, and the
third rf name drops. The scores behind those names do not change, because both read
`ncv.baselines`. The hybrid half cannot be filled from what c14 holds: `NestedCVResult`
carries one honest hybrid per downstream model, and its `.all_results` property at
`find_best_hybrid.py`:9722-9733 holds inner-search selection scores its own docstring bars
from being reported. Replacing section 1 at :128-191 as well would move
`best_single_score`, `delta_vs_best_single_common_folds`, `win_rate_common_folds`,
`mean_delta_common_folds` and `median_delta_common_folds`.

**c18** -- `c18_diversity_vs_depth.py`:51 ranks representations by the sum of their
feature importances, and :52 sorts that sum descending with no metric direction read
anywhere. The shape ranks by held-out accuracy instead. The importance sum is not divided
by how many features a representation has, so a 2048-feature representation accumulates
gain over 2048 features and a 100-feature one over 100. Ranking by accuracy removes that
width advantage, so narrow representations rise into depth_1 and depth_2 and wide ones
fall out. depth_1, depth_2 and depth_3 move at every budget. `rep_ranking_full_data` at
:167 changes from importance-sum units to the dataset's own metric. uniform_all does not
move.

**c31** -- `c31_perf_weighted_global_selection.py`:522 keeps 1 base representation per
downstream model, ranked on the dataset's native metric over the 5 folds it is then
compared on. Adding the rank-2 representation as a second reference gives each of the 360
weighted arms per downstream model a second delta. Every delta already reported keeps its
value. The second delta is not always the larger one: the ranking is on the native metric
and the headline delta is on R-squared.

**c32** -- `c32_allocation_strategy_cross.py`:391 keeps 1 base representation per fold per
downstream model, picked on that fold's training rows and scored on its held-out rows.
Adding `fold_ranked[i][1]` as a second reference gives each of the 64 strategy-scorings
per downstream model a second delta. Every delta already reported keeps its value.

**c33** -- `c33_budget_overfitting.py`:523 reads `best_single_fold_scores`, one base
representation per outer fold picked inside `nested_cv_evaluate`. Adding rank 2 puts a
second delta beside `delta_vs_best_single` in each of about 14 to 16 budgets for each of 6
downstream models. The existing column does not move. Rank 2 has to come from each outer
fold's inner search at `result.inner_searches[i]`, not from `baselines_by_model`, which
holds held-out outer-fold scores and would pick with the answer in view.

**c43_buchwald_step3** -- `c43_buchwald_step3.py`:83 hardcodes `BEST = "cgr_morgan"` and
never ranks it against the other 9 staged representations. Measuring the top 2 under
lightgbm moves `hybrid_minus_target_mean` for all 4 cells if a different representation
comes first. `slice_minus_full_mean`, `quota_best_mean` and the saturation curve would
then be about a different representation's features, not a shifted version of the same
number. Which representation lightgbm puts first is unmeasured in either c43 file.

**s02** -- `s02_hybrid_similarity.py`:381 keeps 3 base representations ranked by rf, while
the scored arm at :626-628 keeps 1 representation ranked by this file's own distance
statistic. `cell['top3_base_reps']` becomes 2 representations ranked for knn, so the names
change wherever knn and rf disagree. Each cell gains a second hybrid arm.
`cell['best_sac']` at :487 is a maximum over the hybrid arms. Over 2 arms it rises or
stays equal. s09 subtracts the best representation's number from it, so that cross-dataset
delta moves upward.

**s04** -- `s04_activity_cliffs.py`:57 sets `_N_POOL_HYBRIDS = 3`, giving 6 hybrid arms
over 2 search families, and :560-562 scores rank 1 of the base ranking only. At 2 per
family the arms fall from 6 to 4, so 2 of the arm rows and 2 lines of the cliff table at
:907-913 disappear. Every hybrid arm is a candidate in `reselection_candidates` at :727-734, so
removing the rank-3 arms lowers or leaves equal the hybrid side of the held-out cliff
delta. Adding a rank-2 base representation raises or leaves equal the best-single side.
Both push the reported hybrid-minus-single cliff delta downward. This file holds no
SearchResult, so the shape is adopted by changing the two counts inside
`get_fold_pool_recipes`.

**s05** -- `s05_knn_prediction.py`:864 keeps 1 base representation per downstream model,
over 6 downstream models; :94 sets `N_HYBRIDS_PER_FAMILY = 3`, giving 6 hybrid arms per
fold for knn. `summary.reference_by_model` grows from 6 rows to 12 and `delta_by_model`
emits 2 deltas per downstream model. The rank-1 numbers stay. Cutting hybrids from 3 ranks
per family to 2 lowers or leaves `summary.posthoc_max.delta_vs_best_rep`, because it is a
maximum over the cells. `fold_reselection_result.knn.delta` loses candidate cells and can
move either way; s09 pools that number across datasets and runs a signed-rank test on it.

**s06** -- `s06_applicability_domain.py`:233 keeps 1 comparator base representation,
ranked for knn with a fallback to rf; :203 keeps 1 hybrid. A second comparator
representation adds no new correlation, decile or interval, because every representation
already has all three. What changes is which arm is named: `results['comparator_rep']`,
the correlation keys spelled with that name at :460-461, and
`conformal['hybrid_genuinely_tighter']`. s09 reads that name at :501 to build
`ad_corr_delta`. The name can also move on a tie, because `extract_top3_base_reps` keeps
dict order and `top_base_reps` breaks a tie by representation name. A second hybrid adds a
whole second set of correlations, deciles and intervals.

**s08** -- `s08_optimal_dimension.py`:531-537 keeps 3 base representations per downstream
model. It substitutes 999 for a missing score at :530 and filters no NaN, so a
representation whose score is NaN can enter the top 3. The shape keeps 2 and adds the NaN
filter, which changes the list contents on any dataset where a downstream model lost every
fold on a representation. `results['top3_rf']` and `results['top3_knn']` change with it.
Keeping 2 budgets per downstream model at :573-577 adds an entry and leaves
`best_rf_budget` and `best_knn_budget` alone. The shared selector cannot read this table:
it holds s08's own full-width fits, not `search.baselines`.

**s10** -- `s10_importance_downstream.py` measures every representation in four
estimators, at :316, :513, :686 and :870, and sets `_N_POOL_HYBRIDS = 1` at :94. Under the
shape each estimator measures 2 base representations instead of 10 and 2 hybrid recipes
per fold instead of 1. `mean_delta` printed at :1544 and :1551 is the mean over the per-
representation entries. Over 2 representations instead of 10 it changes on every dataset.
The `best=` entry beside it becomes the best of 2 rather than the best of 10. The 8
representations that stop being measured are the file's evidence that importance weighting
is neutral-to-negative downstream.

**s12** -- `s12_cross_domain_cliffs.py`:411 scores all 10 representations for each of 5
downstream models, and :62 sets `_N_POOL_HYBRIDS = 3`. `results['per_rep']` shrinks from
10 representations by 5 downstream models to at most 2 by 5, and `pool_hybrid_ranks` loses
pool_rank3. `hybrid_cliff_delta_directed[model]` at :761-768 is built from the best cliff
score over the representations scored. The best over 2 can only equal or lose to the best
over 10. The directed delta rises or stays, with no hybrid having changed. The top 2 are
ranked in the dataset's own metric over all rows, while s12 scores on cliff rows only.

**c10** -- `c10_noise_robustness.py`:370-374 gives every representation an entry and keeps
1 hybrid per cell, re-picked per fold by `fold_reselection`; :308 passes `models=['rf']`, so
rf is the only downstream model present. The summary table at :798-825 would print 2 rows
per downstream model instead of one row per representation. Every hybrid auc_norm moves
unless the call passes `rule='other_folds', held_out_fold=0`, because a whole-dataset top
2 is not the per-fold re-pick. One decision is hers: auc_norm summarises a curve of score
against noise level for one fixed arm, and re-picking the top 2 inside each noise level
makes each curve a curve of different representations.

**n01** -- `n01_activity_cliff_framework.py`:668 scores every representation for each of 5
downstream models, and :521 keeps 1 hybrid per downstream model per fold. The printed
ranking at :575-577 is by rf whatever the downstream model, so svm's and dnn's blocks
would name representations they never named.
`stratified_by_model[model]['_comparison']['top3_smallest_gap']` at :721-726 is computed
over whatever representations are present, so it ranks over 2 instead of all and can name
different representations. `get_best_for_downstream` at :521 hands back another downstream
model's blueprint when the fold's search scored none under this name. `top_hybrids` has no
such fallback. That downstream model then loses its hybrid arm instead of reporting a
mislabelled one.

**n02** -- `n02_sample_weighting.py`:745-756 ranks base representations per downstream
model by this run's own unweighted cross-validation mean and keeps 3; :443-450 keeps 1
hybrid per downstream model. Each block in `per_model_top3` drops its third line and gains
a second hybrid line. If membership comes from `search.baselines` the representations
named can change, because the search's fold means are a different measurement from n02's
own unweighted fits. The scores printed beside each surviving representation stay.
`results['hybrid'][model]` gains a second entry, so the count at :737-738 is over 10
hybrid deltas instead of 5.

**n07** -- `n07_remediation.py`:487-488 ranks the 5 downstream models by each model's
single best representation and keeps 2. :500-502 keeps 2 representations per kept model.
:720 keeps 1 hybrid per downstream model. Her second selector ranks downstream models by
the mean of each model's top 3 representation scores, so the pair carried forward changes
wherever those two orderings differ. `per_model_per_top_rep`,
`per_model_per_top_rep_recovery_pct` and `best_strategy_per_model_per_top_rep` are then
reported for different cells, and oof_smooth, gp_smooth and quantile_clip appear under
different downstream models. A second hybrid per downstream model adds a second recovery
percentage per row. The first hybrid can also change: `get_best_for_downstream` borrows
another downstream model's blueprint at its last level and `top_hybrids` never leaves the
named model.

**n13** -- `n13_noise_distributions.py`:313 picks 1 base representation by name for all 5
downstream models, through `pick_noise_base_rep`, and :491-498 keeps 1 hybrid per
downstream model. Under the shape it reports 2 base representations and 2 hybrids per
downstream model, 20 rows, and stops reporting a single control representation.
`noisy_score_by_model`, `clean_score_by_model`, Track A retention, `delta_from_clean_dnn`,
the Track B recovery denominator, `hybrid_wins_by_noise_type` and
`hybrid_advantage_persists` all move. The base arm would be chosen by accuracy rather than
by name, so on the search's own ranking it cannot get worse: the clean and noisy baselines
rise, the clean-minus-noisy gap narrows and the hybrid's margin shrinks. That direction
follows from the selection rule and is not a measured result. The current base
representation is picked without looking at the labels; a top-2-by-accuracy arm is picked
from a clean-label search, which is the head start n15 exists to measure.

**n17** -- `n17_replicate_removal.py`:390-391 makes every loaded representation its own
space, and :470-477 takes 1 hybrid chosen for rf which then serves all 5 downstream
models. Under the shape each downstream model reports 2 representations instead of the 10
the cost line at :100-104 assumes, and 2 hybrids of its own. Every `gain_in_hybrid` and
`hybrid_minus_base` entry for xgboost, svm, dnn and lightgbm changes, because those four
are scored today on rf's recipe. Arm scores inside a surviving representation do not move.
This file removed a one-representation compute cut on 2026-09-17, recorded at :71-77, so
narrowing to 2 representations is a cut to the same axis and needs her stated reason.

**n18** -- `n18_clean_the_real_labels.py`:199 takes the alphabetically first
representation as the base space, and :259-266 takes 1 hybrid chosen for rf for all 5
downstream models. Under the shape the base space becomes each downstream model's top 2 by
accuracy, which are different matrices. `no_cleaning_mean`, `folds_won`, `folds_scored`,
`mean_gain` and `spread_over_folds` at :420-430 are recomputed, and
`settings_winning_most_folds` and `settings_winning_every_fold` move with them. The hybrid
side gains a second recipe and the four downstream models other than rf stop reading rf's
recipe. Nothing in the output records which representation a base number came from beyond
`base_rep` at :207.

**n19** -- `n19_measurement_count.py`:563 takes the alphabetically first representation,
and :514-521 takes 1 hybrid chosen for rf. Every base-space number moves for n18's reason:
`untouched_mean`, `folds_won`, `folds_lost`, `folds_tied`, `mean_gain`,
`spread_over_fold_gains`, `gain_exceeds_its_own_spread` at :831-858, and `gain_in_base`
and `hybrid_minus_base` at :878-885. `mean_tau2_over_folds` and
`mean_shrinkage_over_folds` at :723-728 also move, because the out-of-fold prediction at
:680 is computed inside the fold's own feature space.

**n21** -- `n21_neighbour_distance.py`:602 picks 1 base representation by name and
:613-620 picks 1 hybrid for rf, and those two serve all 5 downstream models across 13
distance spaces. The chosen representation is not only one row: the 8 importance-weighted
spaces are built from it at :494-509, the forest proximity is fitted on it at :515, and
every repaired fit is scored on it at :707-711. So if rf's top representation is not
ecfp4, every detection average precision and every recovery percent in the file changes,
the plain row included.

**n22** -- `n22_curator_flag.py`:458 ranks 3 base representations by rf and keeps the
first at :462; :414-427 keeps 1 hybrid per search objective family. With 2 representations
the file reports a second comparator carrying its own per-detector table over records,
over molecules and per flag text. `extract_top3_base_reps` leaves a tie in representation
build order and `top_base_reps` breaks it by name. Baseline scores are rounded to 4
decimals. On a tie at the top the comparator changes, and every average precision in the
file changes with it. Swapping the hybrid arms for `top_hybrids` would drop the
similarity-family arm whenever both top 2 come from the winning family.

**opig_activity_cliff_qsar** -- `opig_activity_cliff_qsar.py`:526 loops every
representation for each of 5 downstream models, and :346 keeps 1 hybrid per fold per
downstream model. The per-target table falls from 7 representations by 5 downstream models
plus 5 hybrid rows, to 2 by 5 plus 2 by 5. `best_mae` at :691-715 is a minimum over those
rows. Over at most 10 rows instead of 35 it rises, meaning worse MAE. That happens
whenever the current winner is a representation no downstream model ranks in its top 2.
`best_mcc` at :712 falls or stays for the same reason. That MCC is what this experiment
compares against Dablander's published value.

**opig_blueprint_transfer** -- `opig_blueprint_transfer.py`:286-302 keeps 1 base
representation per fold per downstream model, and :422 and :526 keep 1 hybrid each. Rank 1
holds on almost every fold, because the selector reads the same `fold_search.baselines`
numbers; only a 4-decimal tie can name a different representation. The hybrid side moves
more: `get_best_for_downstream` returns another downstream model's blueprint at its last
level, and `top_hybrids` returns nothing. A downstream model with no row of its own loses
its native and source hybrid numbers instead of carrying a borrowed recipe.
`delta_vs_base`, `delta_n_folds` and `delta_std` are differences against a base arm that
becomes two arms, and `n_recipe_win` and `n_full_win` at :689-690 drive the verdict line
at :698-707.

**opig_cdr_flexibility** -- `opig_cdr_flexibility.py`:358 loops all 10 representations for
each of 5 downstream models, and :307 ranks 3 by rf for printing only. The table falls
from 50 rows to 10 base rows and gains 10 hybrid rows. `best_roc` at :465-467 is a maximum
over those rows. Over 10 rows instead of 50 it falls or stays, and the printed comparison
against the ITsFlexible GNN's 0.82 falls with it. `best_single_reselected` at :450 is re-
picked per held-out fold over every representation today; from 2 representations it gets
worse, which widens the printed delta in the hybrid's favour.

**opig_cgr_comparison** -- `opig_cgr_comparison.py`:382 takes rf's single fitted hybrid
and :391-397 scores that same rf-chosen matrix with all 5 downstream models. Under the
shape each downstream model gets its own 2 hybrids, so the RMSEs printed at :775-785 for
the four downstream models other than rf move, generally improving. `delta_by_model` at
:666-690 subtracts the SMILES-only hybrid's held-out RMSE from the SMILES-plus-CGR
hybrid's per downstream model, so both pools move and the verdict line at :854-856 can
flip. `top3_base` at :434 becomes 2 names per downstream model; the two rf names hold
except on a 4-decimal tie.

**opig_ginkgo_developability** -- `opig_ginkgo_developability.py`:527-547 fits every
loaded representation, 10 general plus 2 augmentation per property, against each of 5
downstream models, and :353 trains all 5 on one rf-chosen feature set. `Best indiv` and
`Delta` at :660-663 change. The maximum on the base side is over 2 representations per
downstream model instead of 12. `Best indiv` falls or stays and `Delta` rises or stays.
The count `Hybrid wins: n of 5 properties` at :665 can rise. `hybrid_reselected` moves for
xgboost, svm, dnn and lightgbm on every property once each reads its own feature set.

**opig_graphinity_ddg** -- `opig_graphinity_ddg.py`:469 hands `fold_reselection` every fold-
scored hybrid row and every representation, per downstream model. Under the shape
`candidates_by_model` drops to 2 hybrids per downstream model and `baselines_by_model` to
2 representations. Then `hybrid_reselected`, `best_single_reselected`, `delta_mean_of_folds`,
`n_paired_folds` and both winners-per-fold lists move on every downstream model row. The
`Best rep` column at :568-583 is taken over 2 representations instead of 9, so the gap to
the structure-based EGNN at :600-610 grows or stays. The transfer_foldx condition has no
SearchResult at all and cannot take the shape without a new search on the synthetic
training rows.

**opig_humanness** -- `opig_humanness.py`:331 hands `fold_reselection` every hybrid row and
all 11 representations per downstream model. Under the shape that becomes 2 hybrids and 2
representations per downstream model, so `hybrid_reselected`, `best_single_reselected` and
`delta_mean_of_folds` move on every row of the table printed at :489-503. `Best individual
rep` at :437-442 becomes a maximum over 2 representations per downstream model instead of
11, so it falls or stays. Each of the 8 deltas against published humanness methods at
:444-448 becomes more negative or stays.

**opig_kingsmith_component_splits** -- `opig_kingsmith_component_splits.py`:707-731 fits
all 14 representations for each of 5 downstream models on every held-out component, and
:497 gives all 5 downstream models one rf-chosen feature set. `BestBase RMSE` in the table
at :833-864 rises, meaning worse, because the per-component re-pick draws from 2
representations instead of 13, so `Delta RMSE` rises. `Hybrid RMSE` moves for xgboost,
svm, dnn and lightgbm. One complication belongs to this file: the fold count is the group
count, 23 additives, 15 aryl halides, 4 ligands and 3 bases, and the search is rebuilt
inside every fold. A per-fold top 2 would leave one representation scored over 2
components and another over 22. So the top 2 has to be fixed once from fold 0, and the
file has to say so.

**opig_scaffold_hybrid** -- `opig_scaffold_hybrid.py`:157-158 keeps 1 base representation
per downstream model out of 12, excluding the 2 scaffold ones, with direction from the raw
stored flag at :133. Reporting 2 gives 10 base rows over 5 downstream models instead of 5.
Rank 1 does not move on these 4 antibody datasets, which are scored in spearman or auroc,
both higher-is-better, so the raw flag and `object_direction` agree. A two-representation
tie now prints a different `best_base_rep`, because the tie-break moves to representation
name. `scaffold_rank` at :165-169 ranks each scaffold representation against all 12 and
stays. The hybrid half cannot go to 2 from stored rows, so this file reports 2 base
representations and 1 hybrid per downstream model.

**opig_trastuzumab_enrichment** -- `opig_trastuzumab_enrichment.py`:759-778 scores every
representation against every one of 5 downstream models, and :431-432 keeps 1 fitted
hybrid per downstream model per fold. `Best individual` at :885-890 is the largest mean
enrichment at the top 1 percent. It is taken over up to 13 representations by 5 downstream
models, 65 numbers. Over at most 10 numbers it can only fall or stay. It will usually
fall, because the top 2 are chosen by the search's AUC baselines while the table reports
enrichment. The comparison table at :861-877 shrinks from up to 65 base rows to at most
10. The AUROC half of `reselected_comparison` can take a second hybrid as arithmetic; the
enrichment half needs a second fitted recipe per fold.

**fed02** -- `fed02_rep_diversity_silos.py`:78-119 ranks the 5 silo representations. It
fits each on an inner 3-fold split cut inside that outer fold's training rows. :540 cuts
that ranking to 3 for condition C5. Under the shape C5 is built on 2 representations
instead of 3. That is a different feature set at the same 100-feature budget. Its MCC on
herg and its RMSE on logd and caco2 change. Which representations C5 picks changes on most
folds. C5's score also becomes selection-optimistic against C4 and C6 under
`rule='whole_dataset'`. The search's own mean covers the molecules C5 is then scored on.
The comment at :530-539 records that leak having been removed. `rule='other_folds'` would
avoid it, but the search's fold numbering is not fed02's own. C1's honest row moves too,
because it reads `silo_ranked[0][0]` at :401. This file scores rf and only rf:
`fed_to_kirby_model` maps rf, xgb and dnn, and `fed_model` is never set anywhere in the
repository.

## The shape does not apply

One row is one experiment file whose comparison is not base representations against
hybrids per downstream model. The second column says what it compares instead.

| Experiment | What it compares instead |
| --- | --- |
| c01 | Profiles every representation one at a time: intrinsic dimensionality, mutual information with the labels, effective rank, one score per downstream model. No hybrid in the file. |
| c04 | Predicting each molecule's activity from its 10 nearest training molecules, run weighted and unweighted over the concatenation of all representations. |
| c06 | Within one model family, whether a variant's capacity index and the budget at which its score peaks move together. |
| c08 | Conformal interval coverage and width for one hybrid recipe against three deliberately mis-set complexity conditions. All four arms are hybrids. |
| c09 | One hybrid's selected features against a PCA rotation and a random feature subset, on mutual information and effective rank. |
| c12 | One omics assay's representation against two assays' representations side by side at full width. No hybrid is built. |
| c13 | One named embedding model against a hybrid over a set including it, and against a random-Gaussian matrix of the same width. |
| c15 | The geometry of a feature space: distance spread, kernel conditioning, error at the feature count that reaches the row count, absorbed label noise. |
| c16 | Regularisation strength against hybrid budget inside one model family. Both arms of the comparison block are hybrids. |
| c17 | Five ways of choosing features out of the concatenation of every representation, scored by mutual information per feature on held-out molecules. |
| c19 | Five ways of feeding importance scores into training, on one fixed hybrid feature space. |
| c20 | One base representation against itself at 10, 25, 50, 100, 200, 300, 500 and 1000 features, then its full native width. |
| c21 | One base representation alone against the same representation plus extra features, and one fixed hybrid against itself plus extra features. |
| c22 | Explains one hybrid: which representation each hybrid feature came from, and each representation's share of the SHAP weight. |
| c23 | A hybrid built under an internal evaluator matching the downstream model against the same model's hybrid built under a mismatched one. |
| c24 | A feature selection built on one set of folds against one built on a different set of folds, both scored on the same held-out molecules. |
| c43_buchwald_signal | One named representation against predicting the training-fold mean, to decide whether buchwald_hartwig carries signal. |
| s01 | Pairwise distances between molecules against the activity differences of those same pairs, per representation. |
| s03 | Mutual information of the selected features against the full representation, the same count of random features, and the same count of PCA components. |
| s09 | Loads the stored JSON of s02, s04, s05, s06 and s08 and runs a signed-rank test down a column of one number per dataset. |
| s11 | Eleven feature spaces in a Bayesian-optimization screening loop, scored by the area under the best-found-so-far curve. |
| s13 | Each space by k-means label coherence and by nearest-neighbour retrieval cost on held-out rows. No downstream model is fitted. |
| n06 | Each noise estimator's per-molecule score against real inter-lab disagreement between repeat measurements of the same drug and protein. |
| n09 | Twelve learning-with-noisy-labels methods against each other, under one fixed base representation and one DNN. |
| n11 | Which features the suspicion score is computed in. All 5 downstream models are fitted on the same single base representation. |
| n12 | Five distance geometries used to find suspect labels. Every downstream model is fitted on one fixed base representation. |
| n14 | Counts activity cliffs: molecules in a pair that is structurally similar and more than 2 log units apart in activity. |
| n15 | The recipe a search found on clean labels against the recipe a search found on noisy labels. No base-representation arm. |
| n16 | Ranks individual features by importance and measures how far that ordering moves as the labels get noisier. |
| n20 | Four arms differing only in which label column the search and the feature ranking saw. Every arm is a hybrid. |
| n23 | Cleaned training sets against the uncleaned one, counted per fold. The 2 feature spaces are a fixed substrate shared by every arm. |
| opig_scaffold_similarity | How far apart a representation puts two variants of one antibody against an independent substitution matrix, over 32,000 sampled pairs. No model is fitted. |
| opig_shared_vectors_census | How many molecules get a byte-for-byte identical row from a representation, split by cause. No model is fitted. |
| opig_shared_vectors_remedy | One representation chosen by how many molecules it merges, then whether hybridising or augmenting it moves its held-out score and its tie count. |
| fed01 | Seven federated training protocols at six label-noise levels on the same molecules. No arm is a base representation. |
| fed03 | Three curated data sources against a naive pool and two prediction-level ensembles; and three representation families, each a concatenation. |
| fed04 | Whether the same hybrid recipe splits its 100-feature budget differently in different regions of chemical space. |
| fed05 | Eight federated conditions across three antibody silos holding kcal/mol, a normalised deep-mutational-scanning score and degrees Celsius. |

## What it costs

Choosing costs 0 new model fits in all 39 experiments that can take the shape.
`top_base_reps` reads `search.baselines` and `top_hybrids` reads `search.all_results`, and
both dicts are already in memory wherever a search is in hand. No experiment needs a
search it does not already run.

Nine of the 39 hold no SearchResult to read, so the shape is adopted there by changing a
count in the file's own ranking rather than by calling the selector: c31, c32,
c43_buchwald_step3, s04, s05, s08, s12 and, for the hybrid half only, c14 and
opig_scaffold_hybrid. c14's `NestedCVResult` carries one honest hybrid per downstream
model and nothing more; opig_scaffold_hybrid's carries the same.

The re-measurement around the choosing is not 0, and it runs both ways. These are each
experiment's own counts, per dataset unless named otherwise, and none is a measured run.

Downstream model fits removed:

- opig_kingsmith_component_splits: about 5,400 across buchwald_hartwig and bh_doyle_plates
- n17_replicate_removal: 3,150
- opig_ginkgo_developability: at least 1,250 across the 5 properties
- n02_sample_weighting: 600, or 400 if n02 keeps its own unweighted ranking
- s10_importance_downstream: 520 on the base side
- opig_cdr_flexibility: 400 across CDRH3 and CDRL3
- opig_activity_cliff_qsar: 375 across the 3 targets
- opig_trastuzumab_enrichment: 275
- opig_humanness: at least 225
- s12_cross_domain_cliffs: 200
- opig_graphinity_ddg: at least 175 on the experimental condition
- n01_activity_cliff_framework: 200 on the representation arm, 175 net
- fed02_rep_diversity_silos: 75
- c18_diversity_vs_depth: 410 LightGBM importance fits, which are not downstream fits

Downstream model fits added:

- n21_neighbour_distance: 29,250 if the 5 downstream models pick 2 distinct base representations and 2 distinct hybrids between them, 263,250 if they pick 10 and 10
- n13_noise_distributions: up to 9,900, plus up to 23,760 DNN trainings if the extra spaces reach Track A
- n07_remediation: about 2,200 for the second hybrid per downstream model, about 2,100 more if the base half widens from 2 kept downstream models to 5
- n19_measurement_count: 1,800, plus 2,700 inner fits in `oof_predict`
- n18_clean_the_real_labels: 1,250
- s10_importance_downstream: 585 on the hybrid side, so s10 nets to about 65 added
- opig_kingsmith_component_splits: 450, with 450 hybrid rebuilds
- opig_cgr_comparison: 150 across the 3 default datasets, with 30 hybrid builds
- opig_ginkgo_developability: 125, with 125 hybrid rebuilds
- opig_blueprint_transfer: 100 per source-to-target pair, with 50 hybrid builds
- c43_buchwald_step3: 45 to rank the 10 staged representations under lightgbm, 35 to measure the second one
- opig_trastuzumab_enrichment: 25, against 275 removed
- s06_applicability_domain: 10, with 10 hybrid builds
- n22_curator_flag: about 50 detector fits, no downstream fits, because n22 fits no downstream model

Two of the 39 are re-runs rather than refactors, and their arithmetic sits apart from the
lists above. c10_noise_robustness passes `models=['rf']` at :308, so the other 4
downstream models have no baseline score and no stored row. Getting them means re-running
56 searches per dataset at about 5 times their current fits. n13_noise_distributions runs
66 condition-and-level cells across 5 folds and 3 noise seeds, 990 passes per dataset, and
the added arms multiply through every one of them.

Ten experiments cost 0 new model fits and save none, because the rows the shape reads are
stored and the arms they feed are already fitted: c03, c05, c07, c11, c14 on its base
half, c31, c32, c33, s08, and opig_scaffold_hybrid on its base half. c02 is 0 as well,
unless a 4-decimal tie flips its rank-2 representation. c02's tie case costs one cold per-
model pass: 5 allocation methods by 3 importance methods by 4 internal evaluators, 60
configurations, across 5 folds, 300 fits for 1 downstream model, before adaptive pruning
drops any.

Two experiments save fold-level hybrid builds rather than model fits. s04_activity_cliffs
at 2 hybrids per search family saves 20 build-and-fit pairs and spends 10 kNN fits on the
second base representation, netting 10 saved. s05_knn_prediction at 2 ranks per family
saves 240 kNN fits out of 720, and 120 fold hybrid builds out of 360. Reporting its top 2
hybrids under the 5 downstream models instead would cost 300 new fits, because s05 fits
only kNN today.

The per-dataset counts cannot be summed into one number for the suite. Each experiment
runs on its own dataset list, and those lists differ.
