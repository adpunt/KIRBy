# Closed — findings a commit fixed in code

**The rule.** A finding leaves `EXPERIMENTS_FINALIZED.md` only when the code change exists on
disk. A decision that has been made is still open until it is implemented. Nothing else moves.

104 items, all labelled **Fixed**. The 47 items previously here under **Changed shape** — where a
commit moved the situation without closing the finding — went back into
`EXPERIMENTS_FINALIZED.md` on 2026-09-08, under each experiment's *Still open*, and the six with
no experiment to sit under went into a new *Engine — open* section in the front matter.

**Not independently verified.** These 104 came from a re-check workflow's reports, not from me
reading each one in the code. One claim from that same workflow was wrong — a metric_fn refusal
reported as committed in `60ea2ec` was working-tree only. Treat the Fixed label as unverified
until re-read.

## augmented_descriptor_reps
- **Fixed** (47e4b9e) — An undefined structural fidelity was stored as 0.0; it is now NaN with a defined flag beside it.
  Same repair as extra_fingerprints: NaN instead of 0.0, struct_fidelity_defined added, both winner lists guarded and counted at lines 251-262.

## c03
- **Fixed** (1e74b25) — create_hybrid is called with no groups, so the internal ranking split is a random 20% carve and the budget is allocated under scaffold leakage.
  The call now passes groups=data['groups'][train_idx], matching every other caller. One added line, no other change to the file in that commit.

## c06
- **Fixed** (a09ac08) — The reused importance scores were fetched at the engine's default fold count while the variant searches ran at a different one.
  get_precomputed_search is now called with n_folds=n_folds and seed=42, and n_folds reaches find_best_hybrid through **kwargs (_shared.py:4708 call_kwargs.update(kwargs)). The fold indices in the returned cache are then checked against set(range(n_folds)) and the cache is discarded with a printed reason if they differ.

## c08
- **Fixed** (1e74b25) — create_hybrid is called with no groups, so the internal holdout that admits features is a random 20% carve inside a group-aware outer fold.
  Now create_hybrid(train_reps, y_train, groups=groups[train_idx], ...). With groups, internal_holdout_split takes the first fold of GroupKFold (hybrid.py:4443-4459) instead of the RandomState(42) permutation.
- **Fixed** (1e74b25) — The conformal calibration set is carved out of training rows by a random permutation, so calibration and proper-train share scaffolds and coverage is biased low.
  A new _cal_split uses make_cv_splitter(1, groups=train_groups, seed=42+fold_i), which returns GroupShuffleSplit(n_splits=1, test_size=0.2) (_shared.py:1647-1648). Both the classification and regression branches now call it.

## c09
- **Fixed** (a09ac08) — The information curves are estimated on the same training rows the selection was fitted on, so they rise partly by overfitting.
  The features chosen on the training rows are now applied to the held-out rows with apply_feature_selection, the scaler and PCA are fitted on the training rows and transformed onto the held-out rows, the neighbour information estimate reads y_val, and the per-principal-component ridge is scored on X_val_concat. The payload gains measured_on='held-out rows' at line 307.

## c14
- **Fixed** (a09ac08) — hybrid_score is the search's own winner-of-the-grid score on the rows that selected it, while best_single_score is an unsearched CV mean; every headline number is built on that difference.
  get_best_search_for_downstream is gone from the imports. nested_cv_evaluate runs at 257-268 with n_outer_folds, the dataset's metric, DOWNSTREAM_MODELS and config['budgets'].
- **Fixed** (a09ac08) — mode='intermediary' gave n_splits=1 for the single-representation arm while the hybrid arm's score came from the search's own 5-fold internal CV, so the two arms were scored on different splits.
  n_outer_folds=n_folds, so at n_splits=1 nested_cv_evaluate takes one group-aware 80/20 outer split (find_best_hybrid._make_cv_splitter, the n_folds==1 branch) and scores both the hybrid and the best single representation on that same held-out fold. The reported delta is now one paired difference on one split rather than two numbers from two different splits.

## c19
- **Fixed** (a09ac08) — get_fitted_pool_hybrid builds the feature matrix over every row, then the fold loop splits that already-selected matrix, so feature selection has seen the validation labels.
  get_fitted_pool_hybrid is gone. Only the recipe is fetched now, at line 152, and the features are selected inside the fold loop by build_fold_hybrid on labels[train_idx].
- **Fixed** (a09ac08) — sample_X = X_full is assigned and never read.
  _sample_X is now the first representation and is passed to cv_split on line 188.

## c22
- **Fixed** (a09ac08) — feature_info is taken from the last fold while avg_shap is averaged over every fold, so SHAP mass is attributed to whichever representation occupies those positions on fold 4.
  aggregate_importance_by_rep is now called inside the fold loop with that fold's own feature_info and that fold's own SHAP vector. shap_by_rep averages those per-fold results.
- **Fixed** (a09ac08) — The comment claims feature_info is deterministic given the blueprint, so all folds produce identical structure.
  The comment now says the opposite and explains why. allocation_moved_between_folds is computed at line 259 and written to the JSON.
- **Fixed** (a09ac08) — Only the last fold's allocation reaches the JSON, with no per-fold record.
  allocation_by_fold now holds every fold's budget split, and allocation_is_from_fold marks the single-fold number as the last fold's.
- **Fixed** (a09ac08) — search_score and the per-fold held-out model scores sit in the same JSON with no marker separating them.
  Renamed to recipe_selection_score with recipe_selected_on='all rows' beside it. The old files on disk still use the key search_score.

## c30
- **Fixed** (38bf326) — Imports round2_harness from a hardcoded /private/tmp scratchpad directory that no longer exists. ImportError on any run.
  The path is now derived from the file: _REPO = Path(__file__).resolve().parents[3], then `from experiments import _alloc_harness as H`. Every H.* name the file uses (load_full, make_folds, score_matrix, rank_columns, _clean, assert_metric_direction, EXPECTED_DIRECTION) exists in _alloc_harness.
- **Fixed** (38bf326) — Writes expB_{ds}.json into the same missing scratchpad directory, and under the exact filename exp_b_balanced_global.py also writes, so a repair to a real directory would make them overwrite each other.
  c30 now writes tests/results/complexity/c30_balanced_global.json (override C30_OUT). exp_b writes tests/results/complexity/exp_b/expB_{ds}.json (override EXPB_OUT).
- **Fixed** (38bf326) — best_rep is ranked on whole-data fold scores, then used as the paired target on those same folds. The baseline is selected on its own test data.
  Each outer fold now ranks all 10 representations by a 3-fold group-aware inner CV on its own training rows (c30:169-189). That ranking picks the fold's target and its pool5/pool10 membership.
- **Fixed** (38bf326) — apply_feature_selection is imported and never used.
  The only kirby.hybrid import is create_hybrid, which is used by _validate_against_engine.

## c31
- **Fixed** (38bf326) — best_scheme_by_cell reported a delta and a bootstrap interval for a weighting method chosen on the folds it was scored on.
  r2_delta_mean became r2_delta_max_over_schemes, the interval was removed, and scheme_fold_wins_vs_target out of scheme_fold_pairs was added, which does not depend on picking a winner. A selection_note in the record says the number is the largest of three methods picked on its own folds.

## c32
- **Fixed** (38bf326) — The caveat claims no whole-data leak in the sizing, but pool membership, the N-subsets and the target are all whole-data ranked on the same folds used for the paired delta.
  One inner group-aware CV per outer fold, on that fold's training rows only, over every staged representation (c32:262-271). It now does three jobs: fold_ranked picks the fold's target (c32:293), fold_pool_full[i][:N] picks the fold's N-subset (c32:374), and fold_hib sizes the slots.
- **Fixed** (38bf326) — The target's per-fold arrays are never written, so only the deltas survive in the JSON and an arm's absolute per-fold score cannot be reconstructed.
  mout['target'] now writes native_per_fold and r2_per_fold alongside the means, plus best_rep_per_fold and width_per_fold. With the per-strategy r2_delta_per_fold already stored, an arm's absolute per-fold score is now recoverable from the JSON.

## caco2_runner
- **Fixed** (45b0125) — Running the default set dispatches fed04, which binarizes caco2's continuous labels at a hardcoded 6.0 and runs on a single-class target.
  Fixed in the dependency, not here. binarise_labels now gives caco2 its own cutoff, log10(2) on the log10 efflux ratio, which is the FDA in vitro drug interaction P-gp substrate call (federated_helpers.py:235).

## engine
- **Fixed** (c74905b) — The scores that split the budget were measured on a GroupShuffleSplit carve while create_hybrid carved with GroupKFold, under a comment saying the two matched.
  One function now. create_hybrid calls it at hybrid.py:4752 and find_best_hybrid's importance and allocation precomputes call the same function at 5133.
- **Fixed** (c74905b) — performance_weighted and fixed computed feature importance over every row handed to create_hybrid, held-out part included.
  Both branches now call compute_feature_importance(rep_dict_train, train_labels, ...). This changes which features every performance_weighted, top_k and fixed recipe selects, on every dataset.
- **Fixed** (c74905b) — A create_hybrid call outside a search fitted the downstream model to size each representation's share, while a search fitted the internal evaluator, so one recipe split its budget two ways.
  The budget split is now fitted with internal_evaluator and read off with the dataset's metric on both paths. The downstream model still scores greedy allocation and every augmentation keep-or-drop.
- **Fixed** (c74905b) — Every fold rebuild through apply_blueprint split the budget by blueprint.rep_scores, a number measured over every row the search ran on, including the fold's own held-out rows.
  rep_scores are injected only when labels is None. Same rule in tests/experiments/_shared.py:4241 build_fold_hybrid, so the two rebuild paths agree.
- **Fixed** (c74905b) — create_hybrid accepted model= with no metric_fn; every representation scored minus infinity and the budget fell through to an equal split under the requested allocation name.
  _evaluate_allocation now raises. The eight federated places named in the audit raise instead of equal-splitting, and bd72b82 passed each one its dataset's metric.
- **Fixed** (c74905b) — A blend component that cannot be ranked for any representation is dropped from the blend in silence.
  Raises naming the component and the number of representations.
- **Fixed** (c74905b) — The out-of-fold residuals that rank candidate augmentation features were computed on a shuffled KFold, so a molecule's scaffold-mates sat in the rows its own residual was predicted from.
  GroupKFold, StratifiedGroupKFold for classification, falling back to the shuffled split when there are fewer distinct groups than folds.
- **Fixed** (c74905b) — result_cache row identity ignored the scoring metric, so a run asking for rmse could be handed a row scored with spearman.
  metric and task_type are identity columns. This widens the primary key and _ensure_schema (result_cache.py:192-198) now raises naming the file.
- **Fixed** (c74905b) — condition_suffix reached the search-pickle filename and the search metadata but neither the result database's filename nor its row identity, so two arms differing only by it shared rows.
  It rides on tier, which is IDENTITY_COLS[0]. A run passing no condition_suffix leaves tier byte-identical.
- **Fixed** (5ebaf17) — tune is not in any validated cache key, so a tuned run and an untuned run land on one path.
  Both the baseline and the hybrid-eval filenames carry __tuned<n_iter>. _OPT_IN_META_KEYS also rejects a stored file carrying an opt-in the caller did not ask for, which was the audit's separate 304-313 finding.
- **Fixed** (c74905b) — screen_threshold is not in any validated cache key.
  In the search metadata. feature_blacklist is applied at 6631-6656 before the search, so it changes rep_shapes, which is in the metadata and is not ignored on read.
- **Fixed** (1e74b25) — The MCS pair set under struct_fid is not in any validated cache key.
  A run with pairs stores an extra struct_fidelity metric per fold; the pair set is now part of the hybrid-eval identity.
- **Fixed** (1e74b25) — Augmentation matrix widths are not in any validated cache key, so a width-reduced run and a full-width run over the same representation names produced identical metadata.
  Emitted only when supplied. Widths are reproducible integers, unlike the neural representation hashes that are on the ignore list.
- **Fixed** (47e4b9e) — nested_cv_evaluate's baseline arm swallows every exception with no message, so a dead comparator reads as delta = NaN.
  Every failure is recorded and printed, and a representation that never scored on any outer fold is named. The per-budget rebuild at 8884-8889 was a bare pass and now prints too.
- **Fixed** (47e4b9e) — Under a geometry or blend objective the hybrid arm averages folds with np.mean and the baseline arm with a NaN-skipping mean.
  score_mean uses _geometry_fold_mean in geometry mode, the same device the baseline arm always used. Predictive runs are unaffected: over finite values _geometry_fold_mean is np.mean.
- **Fixed** (47e4b9e) — Under a blend on an RMSE dataset the best-single-representation comparator is picked backwards; the worst RMSE wins the max.
  _baselines_lower_is_better takes the side metric's direction under a blend and the search metric's otherwise.
- **Fixed** (c74905b) — apply_blueprint never passes is_clf, so a classification recipe is rebuilt with regression scoring.
  is_clf is a parameter of apply_blueprint and is forwarded.
- **Fixed** (c01132d) — distance_config was imported by committed code but never committed, so every checkout but one raised ModuleNotFoundError.
  Committed. tests/experiments/_shared.py:29-33 imports it and get_config at 5974-5976 calls set_distance_metric process-wide.
- **Fixed** (c74905b) — The 642 uncommitted engine lines recorded on 2026-09-03 are still uncommitted.
  git diff HEAD is empty for all five. Over the 34 commits the five files took 2701 insertions and 320 deletions: find_best_hybrid 1640, _shared 728, hybrid 438, distance_config 189, result_cache 26.

## exp_b
- **Fixed** (38bf326) — Imports round2_harness from a hardcoded /private/tmp scratchpad path that no longer exists.
  _REPO is derived from the file's own position and the import is `from experiments import _alloc_harness as H`.
- **Fixed** (38bf326) — H._label_signal_ladder is not defined in the surviving harness, so the blocking direction gate raises before any measurement.
  direction_gate now calls H.assert_metric_direction(verbose=True, datasets=[ds]), which exists at _alloc_harness.py:168. It still raises on a mismatch, so it is still blocking.
- **Fixed** (38bf326) — OUTDIR is the deleted scratchpad directory, and the filename expB_{ds}.json collides with the one c30 writes.
  OUTDIR is tests/results/complexity/exp_b (override EXPB_OUT). c30 writes a single file one directory up.
- **Fixed** (38bf326) — The target's downstream model is chosen by its score on the same held-out folds it is then compared on.
  The comparator model is now picked from the inner group-aware CV on the outer fold's training rows, on the representation those same rows put on top. Every arm in that fold then uses that fold's model (exp_b:272, passed to _fit_predict at 296/321/341), so the pairing holds.

## extra_fingerprints
- **Fixed** (47e4b9e) — An undefined structural fidelity was stored as 0.0; it is now NaN with a defined flag beside it.
  structural_fidelity returns NaN instead of 0.0, score_metric adds struct_fidelity_defined, and the two winner lists at lines 251-262 now skip non-finite values and report how many metrics each comparison was made on.

## fed02
- **Fixed** (45b0125) — C5's three representations were chosen by their score on the fold C5 was then measured on.
  _rank_reps_train_only fits each of the five representations on a 3-fold group-aware split inside the training rows and ranks on the mean inner MCC or RMSE. It raises rather than falling back to a random split when the training rows have too few distinct scaffold groups.
- **Fixed** (1e74b25) — C4's and C5's create_hybrid calls pass no groups, so the internal split that sets the budget share is random.
  Both now pass groups[train_idx], so internal_holdout_split takes the GroupKFold branch at hybrid.py:4460. C6 passes groups through fluid_within_kirby_across at fed02:542 and the helper forwards them at federated_helpers.py:2020.

## fed04
- **Fixed** (45b0125) — Labels are binarized at a hardcoded 6.0, which collapses caco2 and logd to one class.
  PKI_THRESHOLD is gone. binarise_labels reads CLASSIFICATION_CUTOFFS (federated_helpers.py:230): herg_ki at pKi 6.0, caco2 at log10(2) on the log10 efflux ratio, logd at LogD 3.0.
- **Fixed** (45b0125) — get_best_search_for_downstream is handed data=, so the in-memory store answers a classification request with a regression search.
  data=None. The request falls through to the on-disk cache, whose identity carries labels_hash (find_best_hybrid.py:549).
- **Fixed** (45b0125) — metric_name comes from get_domain_metric, so a regression metric name is passed with task_type='classification'.
  metric='auc' is now written literally. get_domain_metric is no longer called anywhere in the file.
- **Fixed** (1e74b25) — create_hybrid is called without groups=, so the internal accept/reject carve is a random permutation.
  All three calls now pass the training fold's scaffold groups. The per-cluster call passes train_groups[idx].

## logd_runner
- **Fixed** (45b0125) — Running the default set dispatches fed04, which binarizes logD at a hardcoded 6.0 and leaves a near-constant target.
  Fixed in the dependency. logd now has its own cutoff, LogD(7.4) >= 3.0, the lipophilicity line from Hughes 2008 and Gleeson 2008 (federated_helpers.py:238).

## n01
- **Fixed** (1e74b25) — make_cv_splitter(n_folds, groups) is called without task_type, so a classification dataset gets GroupKFold instead of StratifiedGroupKFold.
  task_type=data['task_type'] is now passed, and make_cv_splitter returns StratifiedGroupKFold for classification (_shared.py:1649-1653). This was one of the six unstratified splits named in the commit message.

## n06
- **Fixed** (6761eae) — 'concat', the hstack of every representation, is used as the comparison space against the hybrid, and individual base representations are never evaluated as spaces.
  The hstack is gone. The comparator is one representation: the best by RF off the same search that built the hybrid (n06:340), falling back to pick_noise_base_rep when the search gives no ranking (n06:346).
- **Fixed** (9b288a0) — Rows in the same replicate group are feature-identical, so a neighbour-based estimator's nearest neighbour is its own replicate and the group standard deviation is recovered close to by construction.
  The (drug, target) pair ids are passed as exclude_groups to every estimator whose __init__ takes it (n06:446). _knn_indices drops every same-group row before taking the k nearest (noise_estimation.py:212-238), covering neighbor_residual, local_variance_ratio, distance_weighted_residual, activity_cliff_aware and leave_one_out.

## n07
- **Fixed** (9b288a0) — build_fold_hybrid is handed the clean labels, so the feature space is chosen before the noise is injected.
  Each (fold, noise seed) now builds its own space from that draw's noisy training labels at line 593, and every noisy arm is scored in it. The clean-label space survives as a named oracle arm, per_model_per_strategy_oracle_features.
- **Fixed** (9b288a0) — Docstring claims six models including ridge; the list holds five and no ridge.
  Docstring now says five: random forest, xgboost, support vector, DNN, lightgbm. That matches DOWNSTREAM_MODELS_NOISE, which is DOWNSTREAM_MODELS = ['rf','xgboost','svm','dnn','lightgbm'] at _shared.py:387 and 538 after 0ce7ab0 replaced mlp with dnn.

## n12
- **Fixed** (1e74b25) — create_hybrid is called with groups=None, so the internal admission holdout is a random 20% cut on a scaffold-grouped dataset.
  Line 148 now reads `train_reps_for_bp, y_train, groups=groups`, and run() passes groups[train_idx] at line 412-414. The admission holdout is the first fold of a GroupKFold.

## n15
- **Fixed** (c74905b) — The noisy arm was selected by scoring on clean validation labels, so the docstring's 'what an honest practitioner could have picked' was false.
  `**({'score_on_noisy_labels': True} if spec is not None else {})`. find_best_hybrid.py:3623 sets the validation label source to the noisy column, so the winning configuration is picked with no clean label anywhere.

## opig_activity_cliff_qsar
- **Fixed** (6b37acd) — The kirby_hybrid representation is built from search.fitted_results['rf'].selected_indices, fitted on all rows including every validation fold, then scored by GroupKFold on those same rows.
  The representation is no longer added to `reps`. Lines 372-378 are a comment recording why.

## opig_cgr_comparison
- **Fixed** (6b37acd) — delta_by_model subtracted two RMSEs that both came from a hybrid selected on all rows and re-scored on those same folds, with unequal pool sizes.
  Both sides now read reselected_comparison['hybrid_reselected'], the leave-one-fold-out re-score, and the printed table gained two LOFO rows above the Delta at :796-806.

## opig_graphinity_ddg
- **Fixed** (1e74b25) — The PDB overlap filter never ran, because load_dataset dropped the complex identifiers, so FoldX rows sharing a complex with the SKEMPI test set stayed in training.
  tests/antibody_baseline_survey.py:236-240 now carries pdb_ids through, and ddg_combined.py:974 supplies them. The file raises at :243 if the test side has no identifiers and at :251 if the training side has none, so silence can no longer mean the filter did nothing.
- **Fixed** (6b37acd) — Both submit scripts passed --max-samples 5000, which argparse rejects, so the job produced nothing.
  Both now pass --max-foldx 5000, which the parser defines at :675.

## opig_humanness
- **Fixed** (6b37acd) — The kirby_hybrid representation was refit on every antibody's label, then re-scored under the same folds and written into results alongside the honestly-CV'd single representations.
  The apply_feature_selection import, the search.fitted_results['rf'] pull and the reps['kirby_hybrid'] assignment are all deleted. A comment in their place records the removal and the 0.9229 rf AUROC it produced.

## opig_kingsmith_component_splits
- **Fixed** (6b37acd) — The reported delta was the largest R2 over 14 representations x 5 models minus the largest over 5 hybrid models, both maxima taken on the folds that produced them.
  _paired_hybrid_delta hands the hybrid's per-fold scores and every base representation's per-fold scores to fold_reselection. On each held-out component the base representation is re-picked from the other components' scores only, and the subtraction is fold-index paired.
- **Fixed** (6b37acd) — A fold whose find_best_hybrid raised was dropped from the hybrid arm while the baselines still averaged over all folds, and the two means were then subtracted.
  _pad_fold writes NaN for every model on any path that abandons a fold, and evaluate_rep does the same for a failed base fit at lines 322-324. fold_reselection then only pairs folds where both arms have a score and reports n_paired_folds, printed in the Folds column.
- **Fixed** (6b37acd) — _summarize_scores emitted mean, std and a fold count only, so no paired delta could be recomputed from the saved file.
  r2_per_fold and rmse_per_fold are now written per representation and per model. spearman_per_fold is not, so a per-component Spearman comparison still cannot be rebuilt from the file.

## opig_scaffold_similarity
- **Fixed** (6b37acd) — greedy_forward chose its feature subset on the same pairs its score was reported on
  6b37acd assigns each variant to half 0 or half 1 at :163 and keeps a pair only when both its variants are in the same half (:165, and :256-259 per complex on the multi-scaffold path). mode_features chooses the greedy subset, the top-k ordering and the truncation peak on half 0 (:351-355, :361, :365) and reports greedy_score, topk_score, all_features_score, raw_units_score and peak_score from half 1 (:357-362, :366-368).

## opig_trastuzumab_enrichment
- **Fixed** (6b37acd) — hybrid_search.rf_cv_auc was the winning recipe's mean over the folds that chose it, presented in the comment and the printed line as the honest hybrid number, and it was the only hybrid number the experiment produced.
  Renamed search_best_cv_auc, kept as a search diagnostic with a note saying what it is. The reported hybrid number is now reselection_result, a leave-one-fold-out re-score against the same baseline representations, computed at line 373 and printed at 549-566.
- **Fixed** (6b37acd) — The returned dict had no hybrid entry and no per-fold scores anywhere, so the hybrid question could not be answered from the saved file.
  The file now carries fold_reselection (the per-fold scores for every searched recipe and every baseline representation, per model) and reselected_comparison (the computed leave-one-fold-out delta). The hybrid number is AUROC, not enrichment, so question 3 as the docstring states it - does a hybrid beat the best single representation at binder enrichment - is still unanswered.

## ranker_experiment
- **Fixed** (4d16dbb) — Feature selection saw the labels of every molecule, then the selected space was scored by a GroupKFold five-nearest-neighbour prediction over those same molecules; the activity-target arms were graded on folds whose labels chose their own features and the structural arms were not.
  readacross_foldwise re-runs the selector inside each of the five scaffold folds. make_ctx at line 239 builds one context per training fold: the pooled matrix is z-scored from training rows (zscore_fit), and only cached pairs with both endpoints among the training molecules survive (pairs_within_mask).
- **Fixed** (4d16dbb) — The z-score mean and standard deviation were computed over all rows, including rows later held out by the scoring step.
  build_pool now takes fit_rows and calls _heldout.zscore_fit, which takes the per-feature mean and spread from the fit rows alone. Every context passes its own rows.

## s04
- **Fixed** (1e74b25) — make_cv_splitter is called without task_type, so a classification dataset gets plain GroupKFold
  1e74b25 added task_type=task_type to the make_cv_splitter call; 0ce7ab0 then replaced the splitter entirely with nested_cv_outer_folds(reps, labels, groups, n_folds, 42, task_type), which routes through _make_cv_splitter and returns StratifiedGroupKFold when task_type == 'classification' (find_best_hybrid.py:3138-3141).

## s05
- **Fixed** (f86d9a4) — The hybrid is rebuilt from scratch for hybrid_weighted with identical arguments; 60 of 120 create_hybrid calls are duplicates
  _fold_build_cache is keyed (budget, k, fold_i) and _build returns the stored (X_train, X_val, feat_info, blueprint). The weighting is applied afterwards at s05:249-256.
- **Fixed** (f86d9a4) — get_fitted_pool_hybrid runs a full-data create_hybrid whose matrix is discarded
  get_fitted_pool_hybrid is no longer called anywhere in the file. The only whole-data-ish call is _fold_recipe(0, folds_b[0][0], None) at s05:125, which reports fold 0's top base representations; get_fold_best_search memoises on the fold, not the budget, so it costs no extra search.
- **Fixed** (1e74b25) — make_cv_splitter called without task_type, so a classification dataset gets GroupKFold instead of StratifiedGroupKFold
  Folds now come from nested_cv_outer_folds(reps, labels, groups, n_folds, 42, task_type), which reaches _make_cv_splitter at find_best_hybrid.py:3138-3140 and returns StratifiedGroupKFold on a classification dataset. make_cv_splitter is still imported at s05:42 and never called.

## s06
- **Fixed** (6761eae) — X_concat, the concatenation of every representation, is the reference arm for the correlation block and the only comparator in the conformal block
  The comparator is now one base representation, cmp_rep, ranked by knn from the search's own baselines with rf as fallback, and the file raises rather than proceeding when no ranking exists (s06:176-179). It is written to results['comparator_rep'] and the correlation keys carry that name.

## s08
- **Fixed** (1e74b25) — bp is the best config across all budgets, so the entire cross-validated half of a dimension sweep ran at one fixed dimension
  build_fold_hybrid is now called with budget_override=budget, and build_fold_hybrid honours it over blueprint.budget (_shared.py:4172). The six budget rows therefore give six different widths in the CV arm, so model_scores, sac_fold_scores and sac_fold_scores_unweighted move with the budget.
- **Fixed** (0ce7ab0) — DOWNSTREAM_MODELS has no 'knn', so knn_score is NaN at every budget and s09's optimal_dim_ratio can never be computed
  model_factories = get_model_factories(DISTANCE_MODELS, ...) and DISTANCE_MODELS = DOWNSTREAM_MODELS + ['knn'] (_shared.py:390). 'knn' is in DEFAULT_MODELS at find_best_hybrid.py:2278, so get_model_factories returns a factory for it rather than warning and skipping.
- **Fixed** (0ce7ab0) — model_scores filled from the search's cached fold_scores when the budget matched: selection-optimistic numbers on different folds, averaged in beside freshly fitted ones
  refit_models = dict(model_factories) with no cache branch, and the comment records what the shortcut did: five of six budgets scored identically and the sixth was the search's own number.

## s09
- **Fixed** (6761eae) — The applicability-domain delta is the hybrid minus 'concat', a concatenation of every representation, which the project does not allow as a baseline.
  s06 now ranks the base representations by the model its search used and writes the winner's name to results['comparator_rep'] (s06_applicability_domain.py:174-181). s09 reads that name at :489 and only falls back to the string 'concat' when the key is absent, so an old result file still parses.

## s11
- **Fixed** (f86d9a4) — The hybrid's recipe is tuned on labels the loop is meant to be discovering; get_best_search_for_downstream is handed the full labels and groups arrays.
  _seed_blueprint now calls get_fold_best_search with train_idx=seed_idx, so the five settings are searched on that seed's 5% seed rows only. cache_namespace='bo_seed{i}_s{seed}' at :905 keeps one seed's search from being served from another's or from a cross-validation fold's.
- **Fixed** (f86d9a4) — create_hybrid is called with no groups, so its internal accept/reject holdout falls to np.random.RandomState(42).permutation on a scaffold-grouped dataset.
  groups is now sliced and passed at all three build sites: :698 in build_hybrid_features, :1491 for the initial build in _run_bo_with_rebuild, and :1530-1531 for each rebuild. hybrid.py:4440 takes the GroupKFold path when groups are present.

## s12
- **Fixed** (f86d9a4) — cv_split is called with mode=mode, which _shared ignores while raising a DeprecationWarning on every call.
  The folds now come from nested_cv_outer_folds(reps, labels, groups, n_folds, 42, task_type). No cv_split call remains anywhere in the file.
- **Fixed** (f86d9a4) — The winning hybrid recipe is never written to the JSON, so a later reader cannot say what won.
  Each fold's allocation method, importance method, internal evaluator, budget and augmentation strategy are collected per fold and written under hybrid[model]['fold_recipes'].
- **Fixed** (f86d9a4) — The honest number rests on a recipe the precompute chose over every row, so only the single-representation side paid for held-out selection.
  _fold_blueprint calls get_fold_best_search with train_idx=this fold's training rows, and the recipe is cached per (model, fold). get_fold_best_search stamps selection_scope='fold_train_rows' (_shared.py:1056) and s12 writes recipe_selection='fold_train_rows' into the honest block.

## s13
- **Fixed** (f86d9a4) — get_fitted_pool_hybrid builds a whole-dataset hybrid matrix that is bound to _X_unused and thrown away, one 336 s create_hybrid per run.
  No get_fitted_pool_hybrid call remains. The only create_hybrid work is the five per-fold build_fold_hybrid calls, and each of those output matrices is scored.
- **Fixed** (f86d9a4) — The blueprint's budget, allocation method and importance method are printed but never written to the JSON.
  results['fold_recipes'] now carries allocation, importance, evaluator, budget and augmentation_strategy for every fold that produced a hybrid.
- **Fixed** (f86d9a4) — The recipe was searched once over every row by a kNN score close to the retrieval cost this file reports.
  _fold_blueprint calls get_fold_best_search with train_idx=tr, so fold i's recipe was scored on fold i's training rows only. recipe_selection='fold_train_rows' is written into both honest blocks.

## scaffold_experiment
- **Fixed** (4d16dbb) — The folds group variants by their first mutated position, so on a combinatorial library the held-out variant shares most of its mutations with training variants, and readacross_rho was written anyway.
  score_space now sets rho to NaN whenever frac_single < FRAC_SINGLE_MIN = 0.99, the threshold position_gated was already refused on. frac_single, n_position_groups, n_variants and grouped_by='first_mutated_position' go into every row.

## sim_hybrid_space
- **Fixed** (9b288a0) — n11 and n13 print the hybrid_sim refusal reason and throw it away, so an absent arm reads later as a null result.
  Both now write results['hybrid_sim'] with arm_on, the availability dict, selection_rows and a per_fold map of every fold's skip reason. A new function similarity_hybrid_available (_similarity_hybrid_space.py:306) runs the caches, the 3D shape block and the coverage check and stops before the descriptor drop, so the ON/OFF line costs nothing.
- **Fixed** (9b288a0) — The descriptor drop, the fine-2D pick and the z-score statistics are computed over every row of the study cache, then sliced per fold.
  build_similarity_hybrid now takes selection_rows. Only structural pairs with both molecules in those rows survive (:280), and _zscore_fit (:90-100) takes the mean and standard deviation from those rows.

## sm_aggregate
- **Fixed** (4d16dbb) — a6_dca ranks the six metrics against the activity gap and is then scored on those same labels by scaffold-grouped kNN
  The keep order is now |Spearman(pair distance, MCS distance)|, which carries no label. pairs['dy'] no longer enters the ranking.

## sm_base_rep_comparison
- **Fixed** (4d16dbb) — onehot reproducing the CONSTANT control, and dphyschem_pos drawing the same five neighbours, are now recorded rather than read as measurements
  degenerate_readacross returns two flags per row: every train-test distance identical, and every held-out variant drawing the same five training rows. Both print beside the score at :245-250 and go into the JSON.

## sm_feature_ablation
- **Fixed** (4d16dbb) — select_fine2D ranks 2048 features on the cached MCS pairs and the fine2D block's structural fidelity is then measured on those same pairs
  select_fine2D takes fit_rows and ranks only on molecules in the fit half. On herg_ki that is 709 of 1415 molecules and 1970 of the 8000 cached pairs; the reported number comes from the other 706 molecules and their 2005 pairs.
- **Fixed** (4d16dbb) — desc_drop_one_greedy maximises structural fidelity on the same 8000 pairs it reports, 143 drops deep
  The drop sequence is chosen on the fit half's 1970 pairs, and after every drop the same descriptor set is re-scored on the score half's 2005 pairs. final_struct is the held-out number, final_struct_fit the search maximum beside it.
- **Fixed** (4d16dbb) — rmcs = rankdata(mcs) computed and never used
  The line and the comment above it are gone from select_fine2D.

## structural_yardstick
- **Fixed** (47e4b9e) — An uncomputable structural fidelity could sit at the top of the ranked table
  The sort key is now rank_desc, which maps a non-finite value to -inf inside the key only (_nanrank.py:36-44), so an undefined metric goes to the bottom. Each row also carries struct_fidelity_defined and activity_fidelity_defined (lines 112, 114), and line 132 prints how many of the 47 metrics the correlation was computed for.

## triplet_similarity
- **Fixed** (47e4b9e) — An undefined read-across Spearman was recorded as 0.0; it is now NaN, with defined flags and per-fold defined lists.
  heldout_readacross returns NaN, agg at lines 206-212 averages through _nanrank.nanmean, and the summary table at lines 244-248 prints how many folds each mean was taken over.

## winning_recipe_reselected
- **Fixed** (4d16dbb) — The 50/50 split is a permutation of the 8000 pairs, not of the molecules, so the selection has seen every molecule it is scored on.
  sel_idx, sco_idx = scaffold_halves(groups, seed=SPLIT_SEED) splits the molecules by whole Bemis-Murcko scaffold group (_heldout.py:41-60), and pairs_within (_heldout.py:63-71) keeps a cached pair only when both endpoints land on the same side. Straddling pairs are dropped and counted at :94.
- **Fixed** (4d16dbb) — make_D is imported unchanged, so the z-score mean and standard deviation still come from all molecules.
  make_D_fit replaces it and calls zscore_fit(X, fit_idx) at :55, taking the mean and sd from the select rows only. The /sqrt(width) block scaling at :56 and the per-block multiplier at :57 are unchanged.

## winning_recipe_vs_standard
- **Fixed** (4d16dbb) — desc_blacklist_greedy and select_fine2d are fitted on the full pair set and every struct_fidelity is graded on those same pairs, while the docstring calls both yardsticks honest.
  scaffold_halves splits the molecules by whole scaffold group at :128, pairs_within cuts the cached pairs to each side at :129-130, the blacklist is fitted on desc[sel_idx] at :136 and the 100 fine-2D features on X_fine_all[sel_idx] at :138. score_space_split at :105 reports struct_fidelity on the score half and :106 keeps struct_fidelity_insample beside it.
- **Fixed** (4d16dbb) — make_D is imported unchanged, so the per-feature mean and standard deviation come from all molecules including the scaffold folds the activity score holds out.
  make_D_fit calls zscore_fit(X, fit_idx) at :79 with fit_idx = sel_idx. It is used for the z-scored Euclidean comparator at :163 and for the winning space at :167.

## distance sweep

- **Fixed** (2026-09-09, uncommitted) — `input_fingerprint` did not record the pairwise distance coefficient, so a run under a second coefficient into the normal results tree read the first one's files, saw no change and skipped every experiment.
  `tests/experiments/_shared.py:5638` now stores `distance_metric` from `get_distance_metric()`, and `fingerprint_mismatch` compares it. Verified in this session: a manhattan fingerprint checked against a cosine run returns `distance_metric changed: 'manhattan' -> 'cosine'`, and against itself returns None.
- **Fixed** (2026-09-09, uncommitted) — three numbers in the `distance_config.py` module docstring did not match the f18 result files: euclidean last in "13 of 14" blocks, "manhattan is never last", and two rank triples.
  Corrected in place to 11 of 14, last in 2 of the 14 budget-split blocks, and the full four-by-three rank table read off `tests/results/f18_distance_metric_sweep/{herg_ki,logd}.json`. A dated line records what the paragraph used to say.
