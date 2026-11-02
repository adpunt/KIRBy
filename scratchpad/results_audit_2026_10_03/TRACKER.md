# Server-results audit — every item (written by Claude, 2026-10-05)

Source: four read-only audits of the results pulled 2026-10-03 to
/Volumes/seagate/kirby_server_results_2026-10-01/ (server code 165d250).
Status values: FIX (this pass), DECIDE (Adelaide), DONE-ALREADY (fixed in a later commit),
NO-ACTION (by design / non-issue, with her word where she gave it), SUPERSEDED (the
experiment-alignment pass in HANDOFF_EXPERIMENT_ALIGNMENT.md / HANDOFF_PER_FOLD_ALIGNMENT.md
replaces it), DEFERRED (file is being edited by another session right now).
Never delete a row. Update its status and add the commit.

## Adelaide's answers, 2026-10-05
- 10 (width below budget): store the width actually built. Not a big deal.
- 11 (similarity-objective precompute): precompute is not happening the same way as before.
- 15 (models dropped for wall time): checked against DECISIONS.md Step 1 — no drop removes a decided model.
- 16 (adme_hlm/lhasa_herg precompute): all precompute is redone. Non-issue.
- 17 (remedy chemberta cache): give server commands.
- 12, 13, 14: she asked for explanation (below, in the reply of 2026-10-05).

## Global
| id | item | status |
|---|---|---|
| G1 | Server ran 165d250; HEAD is later. Deploy HEAD. | NO-ACTION (alignment pass deploys) |
| G2 | June/July files sit beside current ones; use a new output folder | NO-ACTION (runbook uses a new folder) |
| G3 | noise_prelim_sweep folder was not pulled | CLOSED 2026-10-05: pulled to /Volumes/seagate/kirby_server_results_2026-10-01/noise_prelim_sweep (prelim run = debug, not audited) |

## Federated / opig / noise
| id | item | status |
|---|---|---|
| F1 | fed02 crashes at 165d250 | DONE-ALREADY 5b01654 |
| F2 | fed05 runs once per federated dataset (x3); herg copy stamped auc | DONE e58106f (runs in lhasa_herg job only, stamped rmse) |
| F3 | fed05 transfer: budgets above 100 identical (source hybrid at default budget 100) | DONE e58106f (source rebuilt at each transfer budget) |
| F4 | fed05 C1 rebuilds hybrid from part of the recipe (no budget) | DONE e58106f (C1, C4, C6 built with full recipe + budget; aug none as searched) |
| F5 | fed05 C4/C5: no representation shared by the 3 antibody datasets | DELEGATED 2026-10-05: keep; C4/C5 record why they cannot run |
| F6 | fed05 C6: sequence_lm group missing on tresanco | DELEGATED 2026-10-05: keep C6 groups; comment says why |
| F7 | fed01/03/04 trials timed out, no files | NO-ACTION |
| O1 | shared_vectors_remedy: herg_ki chemberta cache stale | server commands (item 17) |
| O2 | blueprint_transfer trial failed, no log | CLOSED 2026-10-05: not a bug; refused to build logd general search from scratch (0 of 8160 stored). Per-fold design builds it in the experiment. |
| O3 | census: hybrid failure only printed, not in JSON | DEFERRED (file dirty) |
| O4 | census: error entry has no dataset name (:560) | DEFERRED (file dirty) |
| O5 | census: qm9 chemberta cache 200/200 rows differ | server commands (with item 17) |
| O6 | humanness: hybrid failure reason not in JSON | DEFERRED (file dirty) |
| O7 | scaffold_similarity: undefined Spearman stored as 0.0 | DONE e58106f |
| O8 | scaffold_similarity: two koenig datasets give the same numbers (same variants, label unused) | NO-ACTION (expected) |
| N1 | noise files predate 1d84221 | NO-ACTION (re-run) |
| N2 | noise no_blueprint / no_single_representation from missing precompute | NO-ACTION (item 16) |
| N3 | n16 skips classification by design | NO-ACTION |

## Complexity
| id | item | status |
|---|---|---|
| C11a | c11 search inside each fold refused -> no hybrid score | DONE e86df5c (also already unrefused at HEAD via fb1df54) ; grid question SUPERSEDED (decided_grid.py, uncommitted, applies one grid to every search) |
| C11b | 7 averaging rules give 3 results with 3 members | DELEGATED 2026-10-05: keep (3 members) |
| C11c | PCA rows above training-row count are copies | DONE e86df5c |
| C11d | tz_her2_binary fold with a handful of one class | see S10d |
| C11e | "upper bound" column is not one; best-single picked on held-out mean | DONE e86df5c |
| W | hybrid width below budget: c08 c15 c16 c17 c18 c21 c33 s02 s08 | FIX (store width built, item 10) |
| C13a | old files could not search svm/xgboost | DONE-ALREADY e56c20d |
| C13b | side-1 win rate counts unscored models as losses | DONE e86df5c |
| C13c | D_established - C_established mixes two changes on 1-rep datasets | DELEGATED 2026-10-05: keep; documented |
| C13d | one random seed; 3 candidates load nowhere; cdr gives 0 features | DELEGATED 2026-10-05: keep one seed; record unloadable candidates |
| C15a | random features capped at row count, never cross it | DONE e86df5c |
| C15b | instability ratio meaningless | DELEGATED 2026-10-05: remove ratio (82ef2a1 or 6056c8d) |
| C15c | fixed gamma grid gives identity kernel | DELEGATED 2026-10-05: gamma relative to 1/(features x variance), same number of settings (82ef2a1 or 6056c8d) |
| C15d | "Beyer ratio" is not Beyer's per-molecule ratio | DELEGATED 2026-10-05: per-molecule farthest/nearest ratio (82ef2a1 or 6056c8d) |
| C15e | 15c/15d store no width or per-fold scores | DONE e86df5c |
| C15f | Ang2 all NaN (old code) | DONE-ALREADY 01e309a |
| C16a | identical grid rows across budgets | DONE e86df5c |
| C16b | "equivalent alpha" has no tolerance | DELEGATED 2026-10-05: keep; SUPERSEDED by chat 3 of the alignment (ridge family removed with ridge everywhere; uncommitted, her review pending) |
| C16c | classification recipe searched for ridge, scored with LogisticRegression | DELEGATED 2026-10-05: keep; SUPERSEDED by chat 3 of the alignment (ridge and logistic families removed; uncommitted, her review pending) |
| C16d | tuned setting at grid edge; SVR C identical; "capped full" label | DONE e86df5c (label); DELEGATED 2026-10-05: extend grid edges, 4 settings (82ef2a1 or 6056c8d) |
| C17a | "best" = method with fewest features | DELEGATED 2026-10-05: remove best label; report MI and width (82ef2a1 or 6056c8d) |
| C17b | MI saturates at a cap set by held-out rows | DELEGATED 2026-10-05: keep; documented |
| C17c | non-predictive hybrid renamed out of run list (unverified) | DONE e86df5c (only under a similarity pool subset; now recorded) |
| C18a | natural hybrid narrower than other sets | DONE e86df5c |
| C18b | depth_1 ranked by summed importance, not best score | DELEGATED 2026-10-05: keep (depth = importance by design) |
| C18c | depth_2/3 not forced to 2/3 representations | DELEGATED 2026-10-05: keep |
| C18d | uniform_all width stored as [] | DONE e86df5c |
| C18e | rebuilt natural hybrid does not reproduce search score | DONE 4415a50: c18 forests used every feature per split, the search uses sqrt; same fixed in c17, c24 |
| C19a | feature weighting cannot change a tree model | DELEGATED 2026-10-05: note in output; no new models |
| C19b | budget fixed at 100 | DELEGATED 2026-10-05: keep budget 100 (sweep = 8x compute) |
| C19c | no per-fold scores | DONE e86df5c |
| C21a | augmented sets lose base features; n_added wrong | DONE e86df5c (counts); DELEGATED 2026-10-05: keep as finding |
| C21b | capped == uncapped identical | DELEGATED 2026-10-05: keep as finding |
| C21c | H0 budgets barely change H0 | DONE e86df5c |
| C21d | DEFAULT_TOP_N comment wrong | DONE e86df5c |
| C21e | decoy features admitted | DELEGATED 2026-10-05: keep as finding |
| C22a | SHAP shares mix two recipes | DELEGATED 2026-10-05: explain both models every fold (82ef2a1 or 6056c8d) |
| C22b | top-level blueprint from last fold | DONE e86df5c |
| C23a | recipe from fold 0, width mean over folds | DONE e86df5c |
| C23b | identical fold scores across evaluators | DONE e86df5c (per-fold recipe now stored) |
| C24a | nothing is transferred between folds | DELEGATED 2026-10-05: describe as fresh search with one more fold out (82ef2a1 or 6056c8d) |
| C24b | phi compares different recipes/budgets | DELEGATED 2026-10-05: keep phi with per-fold inputs |
| C24c | ci_95 stored and printed | DONE e86df5c (nogueira_se still stored) |
| C24d | fold split not told task type | DONE e86df5c |
| C33a | saturation check uses pool width | DONE e86df5c |
| C33b | files from three code versions | NO-ACTION (re-run) |
| C33c | inner bias opposite directions; distance repeats general | DELEGATED 2026-10-05: keep as finding; duplicate part = S03d rule |
| C10 | c10 moved to noise; complexity/c10 files are June leftovers | NO-ACTION |
| C01a | dnn on adme_hlm rmse 72,237 / 2,339 | DONE eda240c (owner OK 2026-10-05); eda240c clipped in evaluate_single_rep_cv, which c01 no longer calls, so chat 3 of the alignment added the same clip in c01 (uncommitted) |
| C01b | MI above ln 2 on binary labels (sum over components) | DONE ccab773 (components stored); DELEGATED 2026-10-05: keep estimator |
| C01c | effective rank computed two ways by width | DONE ccab773 |
| C01d | no per-fold scores | DONE ccab773 |
| C02a | per_model_top_rep_dim rank 2 overwrites rank 1 | DONE ccab773 (rank 2 = per_model_top_rep_dim_rank2) |
| C02b | models dropped on herg/nt | NO-ACTION (item 15) |
| C02c | PCA baseline width not stored, repeats | DONE ccab773 |
| C02d | uncapped == none on datasets without augmentation | NO-ACTION (intended) |
| C02e | tz_her2_binary fold with few pairs | see S10d |
| C02f | PCA split not told task type | DONE ccab773 |
| C04a | files say spearman, values are R2 | DONE ccab773 |
| C04b | kNN vs 10-neighbour differ in 3 ways | DELEGATED 2026-10-05: both manhattan, both per-fold mean (82ef2a1 or 6056c8d) |
| C05a | 4 stale files | NO-ACTION (re-run) |
| C05b | size asked != rows drawn | DONE ccab773 (record); rule: DELEGATED 2026-10-05: skip <10 groups (82ef2a1 or 6056c8d) |
| C05c | 1-group draws skipped silently | DONE ccab773 |
| C05d | 2-group draws crash | DONE-ALREADY (caught, recorded in search_failures) |
| C05e | c05 drops budgets above training rows | DONE ccab773 (both arms) |
| C05f | reselected_curve never filled | DONE ccab773 (filled) |
| C05g | tz_her2_binary hybrid NaN at 500 rows | DONE ccab773 |
| C06a | kNN k overwritten by engine | DONE ccab773 (verified k 20/10/5/1 kept) |
| C06b | GP length scale re-fitted | DONE ccab773 (verified 4 length scales kept) |
| C06c | Lasso NaN, no error saved | DONE ccab773 (reason counts; Lasso cause unverified) |
| C06d | val_r2 key holds spearman | DONE ccab773 |
| C06e | p-values stored | DONE ccab773 |
| C06f | regression_only reason outdated | DONE ccab773; DELEGATED 2026-10-05: keep regression-only (wiring = new compute) |
| C07a | too few groups to split (item 14) | DELEGATED 2026-10-05: skip with reason when <10 groups (82ef2a1 or 6056c8d) |
| C07b | partitions {} with no reason | DONE ccab773 |
| C07c | general search error not saved | DONE ccab773 |
| C07d | one-class folds on tz_her2 halves | see S10d |
| C07e | all-rows re-score compares budget with rows; except pass | DONE ccab773 (+ AUROC from graded scores in that pass) |
| C07f | same rows, different answer | CLOSED: one configuration on one fold differs (rf-based budget split); ginkgo_hic no longer runs c07 (<10 groups, 82ef2a1). Cause of rf difference unverified. |
| C08a | overfit asks all, builds 3-15% | DONE ccab773 (7f50d5e had most) |
| C08b | underfit wider than optimal on cdr | DELEGATED 2026-10-05: underfit below optimal (82ef2a1 or 6056c8d) |
| C08c | optimal recipe is not c02's | DELEGATED 2026-10-05: keep; alignment pass replaces where recipes come from |
| C08d | uncertainty vs error needs rf, never on regression | DELEGATED 2026-10-05: keep NaN on regression (no new model) |
| C08e | val_r2 key | DONE ccab773 |
| C09a | PCA effective rank = width (unit-variance scaling) | DELEGATED 2026-10-05: PCA rank without unit-variance rescale (82ef2a1 or 6056c8d) |
| C09b | inflection budget after the peak | DELEGATED 2026-10-05: report peak budget too (82ef2a1 or 6056c8d) |
| C09c | xgboost fallback on nt | SUPERSEDED (decided models) |
| C09d | fold spread discarded | PARTLY ccab773; 9d per-fold needs _shared.evaluate_cv_with_train (DEFERRED) |
| C09e | MI comment wrong | DONE ccab773 |
| C09f | F and p stored | DONE ccab773 |

## Similarity
| id | item | status |
|---|---|---|
| S00 | s04/s11/nt skips intended; nt skip from f6ff799 (2026-06-10) | NO-ACTION |
| S00b | nt precompute runs before the skip | SUPERSEDED |
| S01a | SALI dominated by identical-feature pairs (item 13) | DONE eda240c: identical pairs left out and counted (owner OK 2026-10-05) |
| S01b | permutation p-values and sac_significant stored | DONE e7d9ae2 |
| S01c | sequence-cliff block counts 100% on Ang2 | DELEGATED 2026-10-05: keep; documented |
| S02a | similarity family missing (item 11) | SUPERSEDED |
| S02b | distance cells at 1000/2000 hold 500/50-feature hybrids | DONE e7d9ae2 (None + recipe_budget + reason) |
| S02c | general and distance same recipe (pre-HEAD) | NO-ACTION (HEAD stores tier) |
| S02d | budget != features scored | DONE e7d9ae2 |
| S02e | family picked on same held-out folds | DONE e7d9ae2 |
| S02f | metric stamp auc/f1 on a Spearman | DONE e7d9ae2 |
| S03a | ratio undefined when full-set MI clamps to 0 | DELEGATED 2026-10-05: selected minus random as headline (82ef2a1 or 6056c8d) |
| S03b | selected and full set summed differently | DELEGATED 2026-10-05: keep; documented |
| S03c | rebuilt from all reps | DONE-ALREADY 7e17900 |
| S03d | distance repeats general (item 12) | DELEGATED 2026-10-05: take next distinct recipe; belongs to the alignment pass hybrid selection (top_hybrids/get_rule_hybrids in _shared.py) |
| S04a | similarity hybrids absent silently | DONE e7d9ae2 |
| S04b | 15% cliff target skipped on classification | DELEGATED 2026-10-05: keep; documented |
| S04c | kNN predictive general slots may be empty | CLOSED 4415a50 (delegated): predictive general kNN never stored; s04 runs 4 arms per fold, documented |
| S05a | similarity family missing silently | DONE e7d9ae2 |
| S05b | says manhattan, computes Euclidean | DONE e7d9ae2 |
| S05c | kNN reference on 5-group datasets inflated | DELEGATED 2026-10-05: keep; held-out difference does not use the reference; fixed_ref is secondary |
| S05d | duplicate recipes in slots (item 12) | DELEGATED 2026-10-05: same as S03d |
| S05e | equal votes vs distance-weighted | SUPERSEDED (axis 5) |
| S06a | comparator best of 20 on same folds | DONE e7d9ae2; DELEGATED 2026-10-05: accept s09 reading delta vs kNN pick |
| S06b | conformal flag vs earlier comparator | DONE e7d9ae2 |
| S06c | weighting power 0.5 vs 1 | SUPERSEDED (Part 2 of alignment) |
| S06d | similarity hybrid absent | SUPERSEDED |
| S08a | similarity never built | SUPERSEDED |
| S08b | mixed recipes per row | DONE-ALREADY (HEAD) |
| S08c | width not stored | DONE e7d9ae2 |
| S08d | best width for a model with no scores | DONE e7d9ae2 |
| S08e | trustworthy with 2 widths | DONE e7d9ae2 (MIN_RF_SCORED_WIDTHS=3, her number) |
| S10a | AUC from 0/1 predictions | DONE e7d9ae2 |
| S10b | GP constant; mean over remaining folds | DONE e7d9ae2 (record); DELEGATED 2026-10-05: GP normalize_y on by default (82ef2a1 or 6056c8d) |
| S10c | duplicate slots (item 12) | DELEGATED 2026-10-05: same as S03d |
| S10d | tz_her2 grouping: 13-molecule held-out fold | DELEGATED 2026-10-05: keep tz_her2 grouping as is (changing groups changes every experiment on 2 datasets) |
| S10e | 9 widths reuse the 100-feature recipe | DELEGATED 2026-10-05: keep; documented |
| S10f | config misstates what was searched; k_values | DONE e7d9ae2 |
| S12a | cliff Spearman on 2 antibodies (item 14) | DONE e7d9ae2 (counts); DELEGATED 2026-10-05: minimum 10 cliff rows per fold (82ef2a1 or 6056c8d) |
| S12b | 15% target skipped on classification | DELEGATED 2026-10-05: keep; documented |
| S12c | distance repeats general (item 12) | DELEGATED 2026-10-05: same as S03d |
| S12d | two recipes share one name | CLOSED 2026-10-06: the label is built from _HYBRID_IDENTITY_FIELDS, which includes rep_count |
| S13a | fold 0 dropped everywhere | DONE e7d9ae2 |
| S13b | retrieval: hybrids weighted, singles not | DELEGATED 2026-10-05: unweighted-only headline (82ef2a1 or 6056c8d) |
| S13c | each arm picks its own cluster count | DELEGATED 2026-10-05: report per cluster count (82ef2a1 or 6056c8d) |
| S13d | duplicate slots (item 12) | DELEGATED 2026-10-05: same as S03d |
| S13e | similarity only on herg | SUPERSEDED |
| S14 | esm3 on abcov_kd 11.3% NaN filled with 0 | CLOSED 2026-10-05: 32 of 141 antibodies have no light chain; esm3 averages an empty sequence to NaN for the light half, filled with 0 = what the other representations give for a missing light chain. No effect. |
| S15 | nt_H3K4me3 refusal message names the wrong condition | DEFERRED |

## Found while fixing (2026-10-05)
| id | item | status |
|---|---|---|
| X1 | c05 all-rows pass skipped budgets above training rows | DONE ccab773 (recorded, not skipped) |
| X2 | c07 all-rows pass scored AUROC from 0/1 labels | DONE ccab773 |
| X3 | unbounded validation scaling (C01a cause) likely also in c02 PCA dnn, c05/c07 second passes, c08 | DELEGATED 2026-10-05: cap validation features in c02/c05/c07/c08 (82ef2a1 or 6056c8d) |
| X4 | per-fold width of searches not returned by find_best_hybrid (only mean, :5464) | DEFERRED (engine) |
| X5 | c19 sample_weighted fold refitted unweighted when fit rejects sample_weight | DONE e86df5c (left NaN, listed) |
| X6 | c24 STABLE/SOME DRIFT/UNSTABLE label from fixed cut-offs | DONE e86df5c (removed) |
| X7 | c18 natural sets rebuilt at recipe's own budget, depth sets at the loop's budget (may explain C18e) | DELEGATED 2026-10-05: natural built at loop budget (82ef2a1 or 6056c8d) |
| X8 | c24 still stores nogueira_se | DELEGATED 2026-10-05: remove nogueira_se (82ef2a1 or 6056c8d) |
| X9 | AUC from 0/1 predictions also in s05, s08, s04, s12 | DONE 473855a |
| X10 | s04 treats fold 0 as pruned always (same as S13a) | DONE 473855a |
| X11 | s01 stamped with dataset metric (same as S02f) | DONE 473855a |
| X12 | s01 stores sac_p (scipy p-value) | DONE 473855a |
| X13 | s02 per-fold hybrid list loses fold positions after a failed build | DONE 473855a |
| X14 | c04 kNN regressor (equal weights, k 5/10/20) vs 10-neighbour prediction (1/distance, k 10) still differ in 2 ways | KEEP, documented |
| X15 | uncommitted own_search_kwargs in _shared.py raises without dataset_name; c05/c07/c15/c33 don't pass it | FOR OTHER CHAT (its edit) |
