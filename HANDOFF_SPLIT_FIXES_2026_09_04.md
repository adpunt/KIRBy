# Handoff prompts — the 86 open split and leakage findings

Written 2026-09-04. Every open finding from `SPLIT_AUDIT_2026_09_03.md` is in exactly one of
the nine prompts below; nothing is left over. Paste one into a fresh session.

Packages 1 to 7 touch disjoint files and can run in parallel. Package 8 edits the engine and
package 9 edits shared helpers, so run those two on their own.

---

## The preamble — put this at the top of whichever prompt you use

Read `CLAUDE.md` at the repo root first and follow it for everything you write to me.
The short version, and I mean it:

- Do not explain chemistry, statistics or machine learning to me. I have a PhD in this. Tell
  me what my code does that I cannot see.
- No sentence over 30 words. Name the quantity that is computed, not the function it lives
  in. Say representations, folds, budgets, allocation methods, objectives, datasets,
  features. Never "call site", "branch", "guard", "pool", "arm", "cell", "knob", "columns".
- Never write "read-across". Say: predicting each molecule's activity from its five nearest
  neighbours.
- Never introduce a thing with "the" that I have not been told about. Do not refer back to
  your own numbered or lettered lists.
- No appraisal words. Do not tell me something is important, critical or concerning.

Context you need: `SPLIT_AUDIT_2026_09_03.md` at the repo root is an audit of splits, folds
and leakage across find_best_hybrid and the complexity, noise, similarity, OPIG and
federated experiments. 159 defects were raised, each was re-checked by a second reader whose
job was to refute it, 26 fell, 133 stood. 47 are already fixed. Your package is listed
below with file, line and what is wrong. Read the matching entry in that document for the
full detail before you touch anything — it also records what was checked and found correct,
and what was refuted, so you do not re-raise something already settled.

Two standing project rules. Random splits are forbidden anywhere: group-aware
cross-validation is mandatory, using scaffold groups for molecules, sequence identity for
antibodies, PDB complex for binding-affinity change, cold target for drug-target affinity,
tissue source site for TCGA, reaction component for yields. And never cut a dataset,
budget, allocation method or objective out of a search to save compute — a cut needs a
stated reason: degenerate, broken, or undefined.

Note: another session has been editing `src/kirby/find_best_hybrid.py` and
`src/kirby/hybrid.py`. Re-read the exact lines immediately before editing either, and check
the file still parses after every edit.

**How I want you to work.**

Finish what you start. Do not hand me a plan, a summary, or a list of what you would do
next. If a fix has five call sites, change all five. If changing one file means a caller has
to change too, change the caller. Before you tell me you are done, confirm every file you
touched still parses and still imports, and say so.

Come to me only for the decisions listed at the end of your package, and when you do: state
the choice in one sentence, what changes either way in one sentence each, and your
recommendation. Do not ask me to approve work you can just do.

When you are done: commit with a message that says what changed and why, push, and edit
`SPLIT_AUDIT_2026_09_03.md` to mark your entries fixed with a one-line note each. Do not
write a verdict or an opinion into any of my other documents.

---

## P1-noise — The noise suite  (10 findings)
Every one of these builds the hybrid feature space, or picks a noise estimator, using
information the experiment is supposed to be measuring. The commonest shape: the feature
space is built from the CLEAN labels and only then is noise added, so the space already
knows the answer before the experiment corrupts it. I am stuck on the noise experiments, so start here.

### What to fix

**`tests/experiments/noise/_similarity_hybrid_space.py:149`**  
hybrid_sim's features are picked using every molecule, not just each fold's training rows.  
*Wrong because:* The feature choice sees each fold's held-out molecules, which the codebase elsewhere treats as a leak. The comparator arm isn't fully per-fold either: only its feature computation is, and its recipe is chosen once on the whole dataset using clean labels.  
*Fix:* Restrict the 8000 structural pairs to each fold's training rows before the two feature-selection steps, rebuilding the feature choice per fold; if that's too costly, label the run selection_rows='all'.

**`tests/experiments/noise/n01_activity_cliff_framework.py:258`**  
n01's cliff reference on reaction, TCGA, and DTA is built from the representations it grades  
*Wrong because:* The cliff set used to grade a representation is built from that same set of representations. Whichever contributes the most features — methyl_hvg on TCGA, the three fingerprints on buchwald — defines most of the cliffs it's later graded on.  
*Fix:* Give reaction, TCGA, and DTA datasets a cliff reference built outside the representations being compared, or drop the non-circularity claim for them before the next run.

**`tests/experiments/noise/n01_activity_cliff_framework.py:398`**  
Each fold's hybrid budget split is set by scores fit on that fold's held-out rows too  
*Wrong because:* Those scores are fit on all rows of a dataset, including the rows held out in that fold. So each fold's hybrid is built using numbers that already saw its own held-out rows.  
*Fix:* Score representations inside each fold's own training rows (nested cross-validation), or label the hybrid rows as selected on all rows, like n11 and n12 already do.

**`tests/experiments/noise/n06_bindingdb_ground_truth.py:295`**  
Six of n06's ten noise estimators skip the split-by-protein grouping the other four use.  
*Wrong because:* Six of the ten estimators never receive the split-by-protein grouping the other four use, so a repeated row's label can leak into its duplicate's score. That leak inflates the Spearman correlation used to pick the best one.  
*Fix:* Route cross_val_residual through the same protein-grouped split as the other estimators, and either do the same for leave_one_out or drop it from screening.

**`tests/experiments/noise/n06_bindingdb_ground_truth.py:300`**  
n06 scores six of ten noise estimators using their own replicate labels  
*Wrong because:* Six of ten estimators include a row's own replicate rows when computing its score, so part of the correlation is arithmetic, not signal. The other four group folds by protein target, so replicate rows stay in the same fold together and their scores aren't contaminated.  
*Fix:* Score each row using only rows from a different drug-target pair, and rebuild the hybrid space on the training side of that split.

**`tests/experiments/noise/n07_remediation.py:502`**  
n07 picks hybrid features using clean labels, before the labels it studies are corrupted  
*Wrong because:* That baseline is scored on features chosen using clean labels, before noise is added, which favors all four remediation strategies over it in noise detection. One produced file already has a negative denominator, making its percentages uninterpretable for a separate reason.  
*Fix:* Pick features from each fold's noisy labels, not clean ones, and fetch the noise-aware search instead. Keep a clean-label row too, but label it an oracle, as n15 does.

**`tests/experiments/noise/n11_estimation_remediation_grid.py:697`**  
n11's best_config is the top of ~4,800 configs, scored on the same folds it reports on  
*Wrong because:* Every config is scored on the same folds used to report its recovery, so the winning score is partly luck. That printed number is not a fair estimate of how well that specific config recovers out of sample.  
*Fix:* Pre-register one config before running, split the folds in two to score the winner separately, or relabel these three fields as a search result, not a measured recovery.

**`tests/experiments/noise/n13_noise_distributions.py:817`**  
n13 tags a fixed recipe as `fold_reselection`; the per-fold reselection it promises never runs.  
*Wrong because:* The `fold_reselection` key promises a recipe re-picked per held-out fold, but n13 gives it one recipe, so re-selection never happens. That recipe was chosen on all folds with clean labels, true of every recipe in the suite, and it does not inflate the delta.  
*Fix:* Either rename the key to drop its per-fold-selection promise, or feed every searched recipe so the re-selection is real — costing one refit per noise_type x level x fold x seed.

**`tests/experiments/noise/n14_activity_cliff_landscape.py:342`**  
n14's residual-SD outputs are in label-SD units, but compared to a fixed pKi noise floor  
*Wrong because:* The Kramer 0.54 pKi noise floor only matches these values when a dataset's label SD is 1.0, which is not true in general. On herg_ki, label SD ~1.2, a reported 0.45 is actually 0.54 pKi — right at the floor, but reads as comfortably below it.  
*Fix:* Keep the params from_linear returns, multiply the four residual fields by the saved label SD, and record which scale the 0.54 pKi floor uses.

**`tests/experiments/noise/n15_clean_vs_noisy_hybrid.py:184`**  
n15 picks its noisy-label blueprint by scoring candidates against clean labels  
*Wrong because:* Both blueprints are chosen using the true labels, so the reported gap only measures worse fitting, not worse selection. A practitioner working with noisy labels has no clean labels to select a configuration with, so a small gap cannot clear n11 and n12's warning.  
*Fix:* Score the search's inner validation folds on the noisy labels too when label_noise is set, and keep the clean labels only for n15's own final held-out scoring.

### Bring these to me before doing them

- n15 exists to measure how much of a head start the clean-label search gives. Fixing n11
  and n07 to search on noisy labels changes what n15 is comparing. Ask me which I want measured before you change both.
- n06 scores each estimator against the replicate spread of the same labels the estimator
  saw. Making it honest changes which estimator wins. Tell me that before doing it.

---

## P2-complexity — The complexity experiments c05-c24  (18 findings)
These report a hybrid-versus-best-single-representation number whose winning recipe was
chosen by its score on the same folds the number is measured on. Most need the
leave-one-fold-out re-pick that `fold_reselection` in tests/experiments/_shared.py already
does; the per-fold scores it needs are usually already saved, so it costs no compute. Two
(c06, c07, c09, c17) are a different shape: a quantity computed over every row enters a
held-out score.

### What to fix

**`/Users/apunt/repos/KIRBy/tests/experiments/complexity/c05_learning_curves.py:110`**  
c05/c07 call a blueprint "no leakage" though it's tuned on the same rows it's scored on  
*Wrong because:* The comment says this blueprint is re-fit on the subsample only, with no leakage. Only feature selection is re-fit; the allocation method, importance method, evaluator, and budget were chosen using every row the subsample later holds out.  
*Fix:* Rewrite the c05:110-111 comment and rename fulldata_bp so it reads as a blueprint chosen with every label, not as a leak-free re-fit.

**`tests/experiments/complexity/c05_learning_curves.py:242`**  
c05 picks its hybrid winner from far more candidates than its representation winner.  
*Wrong because:* The hybrid winner is chosen from about ten times more candidates than the representation winner. Picking a max over more candidates on the same folds inflates the hybrid score most when folds are small.  
*Fix:* Rerun c05 and read the curve from fold_reselection_by_n, which it already writes and which scores every candidate on folds that did not pick it.

**`/Users/apunt/repos/KIRBy/tests/experiments/complexity/c06_within_family_gradient.py:186`**  
c06's default mode fits importance scores on rows it later validates on  
*Wrong because:* This only happens when you run c06 with no --mode flag, its default single 80/20 split. It reuses importance scores from a 5-fold split, so most of the 80/20 test rows already sat in the training data behind those scores.  
*Fix:* Pass the fold count into get_precomputed_search at line 126, or skip the cached importance scores when the fold counts don't match.

**`/Users/apunt/repos/KIRBy/tests/experiments/complexity/c07_data_complexity_partitions.py:156`**  
c07's target_variance split decides easy/hard using the same labels it later scores.  
*Wrong because:* No label reaches the fitted model, scaler, or importance ranking, since each fold holds the whole group out together. But easy/hard membership needs the label to assign, and that same label variance is also part of what the two halves are compared on.  
*Fix:* Drop target_variance as a partition axis, replace it with a label-independent proxy, or compute each fold's variance cut from its training rows only.

**`/Users/apunt/repos/KIRBy/tests/experiments/complexity/c07_data_complexity_partitions.py:338`**  
c07 and c03 pick hybrid features on a different holdout than the blueprint used  
*Wrong because:* c07 and c03 pick features for the hybrid using a random 20% holdout instead of the fold split used elsewhere. That random split falls inside the outer fold's training rows, so scores stay honest, but the features picked don't match the blueprint's or c05's rebuild.  
*Fix:* Pass groups (and model, metric_fn) into c07's build_fold_hybrid call and into c03's create_hybrid call, matching c05's rebuild.

**`tests/experiments/complexity/c09_formal_analysis.py:371`**  
c09 measures retained label information on the same rows that picked the features  
*Wrong because:* Features are chosen because they match those labels, so the measured amount is inflated by that same match. Whether the inflation grows or shrinks with the budget is not established.  
*Fix:* Keep feature selection on each fold's training rows, but measure the information on that fold's held-out validation rows instead.

**`tests/experiments/complexity/c09_formal_analysis.py:607`**  
c09's fallback picks its blueprint from all rows, then scores it on folds of the same rows  
*Wrong because:* Allocation method, importance method, internal evaluator and augmentation strategy are all chosen by score_mean computed over every row, including rows later scored as validation. So val_score is optimistic and gap is understated, by the spread across those four choices, at every budget alike.  
*Fix:* Write a flag in the results JSON saying this fallback's blueprint was picked using all rows, so val_score is never read as held out.

**`tests/experiments/complexity/c11_ensemble_vs_hybrid.py:419`**  
c11's hybrid recipe and budget split are chosen using every fold's held-out labels  
*Wrong because:* The comparison assumes both arms see only their fold's training split, but the hybrid's recipe and per-representation scores come from every row, held-out rows included. fold_reselection only re-picks the budget itself; it doesn't redo the recipe or the per-representation scores.  
*Fix:* Either search for the recipe inside each fold's training split like nested_cv_evaluate does, or keep the full-dataset search and relabel hybrid_reselected as recipe-selected on all folds.

**`tests/experiments/complexity/c12_modality_ablation.py:497`**  
c12's HELPS/HURTS verdict compares two best scores fit and scored on the same folds  
*Wrong because:* The single representations entering a pair, and the single-modality winner it's compared against, are all picked by taking the best score on the same folds they are then scored on. Neither side is a held-out score, so the difference has no honest direction — not a known bias against the pair.  
*Fix:* Use the leave-one-fold-out check already in the file: pick each modality's winning representation on the other folds, then score it only on the held-out fold.

**`tests/experiments/complexity/c14_cross_domain_summary.py:279`**  
c14's hybrid score comes from the same folds that picked the winning recipe.  
*Wrong because:* The folds that pick the winning recipe are the same folds that score it, so the number is optimistic. The single-representation score is also a maximum, but over about 50 combinations, not several hundred, so the two sides are not comparable.  
*Fix:* Replace the recipe lookup with nested_cv_evaluate, so each fold's winning recipe is chosen only from that fold's training rows, and report delta_mean_of_folds.

**`tests/experiments/complexity/c15_ml_theory_depth.py:291`**  
c15b's Align column scores a hybrid against the same labels used to build it  
*Wrong because:* The hybrid's features are picked to fit these same labels, so its alignment score reflects that fit, not a compact space's shape. The representations beside it in the table were never fit to the labels, so the comparison favors the hybrid unfairly.  
*Fix:* Split into train and test rows first, build the hybrid on train rows only, then compute alignment on the held-out test rows.

**`tests/experiments/complexity/c16_feature_selection_as_regularization.py:365`**  
c16's budget allocation uses whole-dataset scores inside every fold's hybrid  
*Wrong because:* Every budget's representation ranking is scored using rows from the fold it will later be tested on, not just its own training rows. That leaks into the heatmap, the sensitivity curve, and the L2-equivalence output — not only the hybrid_wins count.  
*Fix:* Decide whether per-budget representation searches should run inside each outer training fold instead of once over the whole dataset, then re-run c16.

**`tests/experiments/complexity/c17_information_theoretic_decomposition.py:297`**  
c17: three of five feature-selection strategies see validation-row labels before scoring  
*Wrong because:* Hybrid, mRMR, and recursive elimination pick features using every row's label, including validation rows; PCA and the random baseline pick features with no labels at all. mi_density is also measured in-sample for all five strategies alike, so best_strategy favors whichever strategy was built to maximize that same number — not necessarily mRMR, which actually scores 0.0 at every budget in abcov_kd.json.  
*Fix:* Move feature selection for hybrid, mRMR, and recursive elimination inside each fold: select on training rows only, apply the chosen features to validation rows, then measure mi_density and downstream_r2 there. PCA and random need no change.

**`tests/experiments/complexity/c19_importance_weighted_training.py:143`**  
c19 picks the hybrid features and recipe using validation labels, reports it as held-out.  
*Wrong because:* get_fitted_pool_hybrid picks the winning allocation/importance/budget recipe and the hybrid's features once, using every row's label, including validation rows. The cross-validation loop at line 178 then only refits the scaler and importance weights inside that fixed feature set.  
*Fix:* Fetch the recipe with get_best_search_for_downstream, then call build_fold_hybrid inside the fold loop on the training rows' labels only, as c18 does at line 243.

**`tests/experiments/complexity/c22_explainability.py:104`**  
c22's search_score is a selection score over all rows, not held-out performance  
*Wrong because:* The allocation method, importance method, evaluator, and budget it uses were picked by comparing scores computed on every row, including each fold's validation rows. search_score is that whole-dataset selection score, not a held-out performance number.  
*Fix:* Label search_score as a selection score — rename it or set recipe_source='whole-data search' in get_best_search_for_downstream — so no report treats it as performance.

**`tests/experiments/complexity/c22_explainability.py:231`**  
c22 averages SHAP scores across folds but labels them with the last fold's features  
*Wrong because:* Each fold selects its own features from its own training rows, so position 37 can name a different feature in each fold. Averaging SHAP scores across folds and labeling them with one fold's feature names mixes different features together.  
*Fix:* Compare each fold's selected features per representation, not just the total feature count, and average SHAP scores within matching (representation, feature) pairs instead of by index.

**`tests/experiments/complexity/c23_evaluator_model_matching.py:102`**  
c23's evaluator-match deltas score the winning configuration on the folds that picked it  
*Wrong because:* The configuration is chosen by maximizing score_mean on those same folds, so score_mean is inflated before the matched-minus-mismatched delta is even computed. Condition E searches a superset of condition A's evaluators for rf, lightgbm and xgboost, so its delta can only be zero or negative.  
*Fix:* Score each model-evaluator pair with nested_cv_evaluate on held-out outer folds and report delta_mean_of_folds instead of score_mean.

**`tests/experiments/complexity/c24_blueprint_transfer.py:181`**  
c24's blueprint-transfer score is biased by an unfair recipe search, not by leaked features  
*Wrong because:* The recipe for each cross-fold cell is picked using training rows that include that fold's own validation rows. Only five recipe settings cross folds — features get re-selected fresh on each target fold, so the delta cannot test whether feature selection generalizes.  
*Fix:* Search each source fold's recipe on rows that exclude every target fold's validation rows, or relabel the off-diagonal as recipe sensitivity, not transfer.

### Bring these to me before doing them

- c11's honest field is still contaminated even after the leave-one-fold-out pass, because
  that pass only re-picks the budget, not the recipe. Fixing it properly means searching
  inside each fold, which costs five searches instead of one. Tell me the cost.
- c22 averages feature-attribution scores across folds and labels them with the last fold's
  feature map. Ask whether I want them matched per (representation, feature) or dropped.

---

## P3-alloc-sweeps — The allocation sweeps c30, c31, c32, exp_b  (6 findings)
All four pick the thing they compare against — the target representation, the top-N
shortlist, the comparator model, the pool of five — by its score on the same folds every
method is then scored on. c31 additionally reports the best of three weighting schemes per
cell with a confidence interval that ignores the three-way choice.

### What to fix

**`tests/experiments/complexity/c30_balanced_global_selection.py:121`**  
c30 picks the target representation using the folds it also scores every method on  
*Wrong because:* The target and the shortlist are both picked from the same fold scores used to grade all three methods. That makes every delta reported against the target biased in an unknown direction, though comparisons between the three methods themselves stay valid.  
*Fix:* Rank representations per fold so the target and shortlist are chosen per fold, as c31 does, or state the leak and stop quoting target deltas as clean.

**`tests/experiments/complexity/c31_perf_weighted_global_selection.py:428`**  
c31 and c32 pick their baseline representation using the same folds it is later scored on  
*Wrong because:* That baseline is chosen from the same five folds it is later scored against, so its score is inflated. Because the baseline is the inflated side, every reported delta understates the true gap, not overstates it.  
*Fix:* Add this caveat to c32, matching c31; optionally add a per-fold baseline computed from the already-saved fold_hib scores, reported as a separate line.

**`tests/experiments/complexity/c31_perf_weighted_global_selection.py:711`**  
best_scheme_by_cell's delta is the best of three weighting methods, reported as one  
*Wrong because:* The five folds that pick the winning method are the same five folds that score it, so the reported delta runs high. Its confidence interval, built at line 335 for one fixed method, does not cover choosing the best of three, so it reads narrower than it should.  
*Fix:* Label best_scheme_by_cell's delta as a maximum across three methods, or replace it with a win count across the three methods and five folds.

**`tests/experiments/complexity/c32_allocation_strategy_cross.py:204`**  
c32 leaks one oracle ranking into its N=2/3/5 picks and its comparator column.  
*Wrong because:* Picking the target from the same five folds it's scored against cancels out when ranking strategies against each other. But the set of 2, 3, or 5 representations each strategy draws also comes from that same ranking, and that does not cancel.  
*Fix:* Move both the target pick and the top-N representation selection inside each fold's training split, using the inner group-aware CV already built at line 245.

**`tests/experiments/complexity/c32_allocation_strategy_cross.py:309`**  
c32's N=2/3/5 shortlists are picked using validation-fold scores, not train-only  
*Wrong because:* Fold i's candidate representations were chosen partly from fold i's own held-out labels, so a win at N=2 or N=3 cannot be told apart from picking winners after seeing the answer. N=10 uses all ten representations, so it is not affected.  
*Fix:* Rank representations inside each fold using fold_hib[i], already computed and train-only, and take the top N of that instead of the whole-data ranking.

**`tests/experiments/complexity/exp_b_balanced_global.py:142`**  
c30_balanced_global_selection.py picks pool5 and comparator model on folds it scores them on  
*Wrong because:* Pool5 and the comparator model are chosen by ranking representations on the same five folds they are later scored against. So the pool5-vs-pool10 comparison is not fair, because pool5 was picked using the scores that grade it.  
*Fix:* In c30_balanced_global_selection.py, rank representations and pick pool5 and the comparator model from each fold's training rows only, not from the held-out scores.

### Bring these to me before doing them

- Ranking representations inside each fold instead of once over all of them changes every
  number these four produce. Confirm with me before starting, and say plainly that the
  current numbers do not survive it.

---

## P4-similarity — The similarity experiments s02-s13  (8 findings)
Every one picks the hybrid recipe or its features once, over all rows, then reports a
number it calls held-out. Several already compute the honest version and write the other
one beside it.

### What to fix

**`tests/experiments/similarity/s02_hybrid_similarity.py:170`**  
s02 picks the hybrid's features using rows it later scores as "held-out"  
*Wrong because:* The single representations it's compared against get no such selection, so the comparison isn't fair. The reported gap for the hybrid is biased upward by that mismatch.  
*Fix:* Either rebuild each fold's hybrid representation from its recipe using training rows only, like s04/s05 do, or rename the output and add a caveat.

**`tests/experiments/similarity/s05_knn_prediction.py:144`**  
s05 and s04 pick their best hybrid recipe on the same folds they score it on  
*Wrong because:* The recipe is chosen and scored on the same 5 folds, so the score is the best of 75 combinations measured on itself. Neither the fixed_ref number nor the leave-one-fold-out re-score fixes this, because re-scoring only re-picks budget, condition, and k, not the recipe.  
*Fix:* Choose the allocation method, importance method, and evaluator inside each evaluation fold's own training rows, instead of once on the folds used to score results.

**`tests/experiments/similarity/s05_knn_prediction.py:397`**  
s05's "pre-registered" baseline in fixed_ref is picked from the same folds it's scored on  
*Wrong because:* The hybrid score is fixed before folds are scored. The baseline score is chosen afterward, from the best representation and best k on the same folds, so it is inflated.  
*Fix:* Have firstpass_notready.py's read_s05 use the leak-free knn_delta_heldout number s05 already computes and s09 already stores, instead of knn_delta_fixed_ref.

**`tests/experiments/similarity/s07_noise_bridge.py:390`**  
s07 scores hybrid noise contamination on features selected using every fold's labels  
*Wrong because:* The features in that matrix were picked using every point's label, including the labels each fold later holds out. Fitting the scaler and noise threshold per fold does not undo that earlier choice.  
*Fix:* Rebuild the hybrid per fold from that fold's train labels, and also rerun the search that picks which representations combine — it too uses every fold's labels.

**`tests/experiments/similarity/s08_optimal_dimension.py:217`**  
s08's per-model budget picks aren't a real sweep, and never reach s09's optimal_dim_ratio  
*Wrong because:* At every budget except the one already picked by the search, s08 rebuilds using the budget the knn search picked, not that loop's budget. So five of six scores are identical, and the sixth is just the search's own winning score, copied in.  
*Fix:* Score every model at its own loop budget using build_fold_hybrid with a budget_override, and drop the cache shortcut entirely so no entry gets copied in.

**`tests/experiments/similarity/s11_bayesian_optimization.py:883`**  
s11 picks the hybrid representation's settings using pool labels the search should reveal only later  
*Wrong because:* Those rows include the pool rows the search loop is meant to reveal one at a time. So the hybrid representation enters the comparison already knowing labels the other representations never see.  
*Fix:* Choose the hybrid representation's five settings from seed rows only, inside the loop that runs each seed, or use the fixed defaults already coded at lines 682-689.

**`tests/experiments/similarity/s12_cross_domain_cliffs.py:273`**  
s12 says it re-picks the hybrid per fold; the recipe is picked once, on all rows  
*Wrong because:* The hybrid's features are refit inside each fold, but its recipe — allocation method, budget, importance, evaluator, augmentation — is chosen once, using every row. Single representations are re-picked fold by fold, so only that side pays for held-out selection, not the hybrid.  
*Fix:* Either pick the hybrid's recipe inside each fold using only its training rows, or give held-out re-selection several hybrid recipes per fold to choose among.

**`tests/experiments/similarity/s13_clustering_retrieval.py:185`**  
s13 picks its hybrid recipe on all rows, but its docstring claims otherwise  
*Wrong because:* The recipe is picked with a kNN score close to the retrieval score s13 reports, so it sees rows it is later scored on. Everything else — the KMeans fit, kNN index, scaler, and column selection — is redone train-only inside each fold.  
*Fix:* Either move the recipe search inside each fold, or keep it as is but fix the docstring and log which rows chose the recipe.

### Bring these to me before doing them

- s11 currently chooses the hybrid's five settings using labels from the pool the Bayesian
  search is supposed to discover. The alternative is the fixed defaults already coded in the
  file. Ask which I want; it changes what the experiment demonstrates.

---

## P5-simmetrics — The similarity-metric scripts  (9 findings)
These choose features, descriptor blacklists or metric orderings using the exact molecule
pairs or activity gaps they are then graded on. Three are grouping problems instead: folds
that hold out whole mutated positions, or that split pairs rather than molecules.

### What to fix

**`/Users/apunt/repos/KIRBy/tests/experiments/similarity_metrics/aggregate.py:108`**  
A6_dca's metric order comes from labels it's scored on, though nothing is ever dropped by it  
*Wrong because:* The keep order comes from the same activity gaps A6_dca is later scored on. That never matters here: metric correlations top out at 0.724 and 0.717, well under the 0.9 threshold, so all six are always kept.  
*Fix:* Order the six metrics by their correlation with MCS distance, already stored in the same file, instead of with the activity gap.

**`/Users/apunt/repos/KIRBy/tests/experiments/similarity_metrics/base_rep_comparison.py:116`**  
koenig folds hold out whole positions, so two positional reps score near zero by arithmetic  
*Wrong because:* Onehot marks every position, but two single mutants always differ at exactly two entries, so every train-test distance is 2.0, a constant. dphyschem_pos shares no features between train and test positions, so the same five neighbours are picked for every query.  
*Fix:* Drop onehot and dphyschem_pos from the koenig family verdict, or add a second fold split that keeps each position present in both train and test.

**`/Users/apunt/repos/KIRBy/tests/experiments/similarity_metrics/feature_ablation.py:96`**  
The fine2D block is picked to match the MCS score it is later graded on  
*Wrong because:* Features chosen to match MCS values will score well on those same values, so fine2D's fidelity number is inflated by an unmeasured amount. A similar check elsewhere, splitting pairs in half, moved one such score from 0.449 to 0.410 on herg_ki.  
*Fix:* Pick the fine2D features on one held-out half of molecules and score them on the other half, as similarity_noise_cleaning.py does at line 177.

**`/Users/apunt/repos/KIRBy/tests/experiments/similarity_metrics/feature_ablation.py:186`**  
feature_ablation.py reports a search maximum, not a held-out structural-fidelity score  
*Wrong because:* The reported correlation is the best of 143 drops, chosen on the same pairs it is scored on. No molecule or pair is held out, so the number is a search maximum, not a measured fidelity.  
*Fix:* Split molecules by scaffold group into a fit half and a score half, run the drop sequence on the fit half's pairs, and report correlation only on the other half, as similarity_noise_cleaning.py already does.

**`tests/experiments/similarity_metrics/ranker_experiment.py:207`**  
combo_r1+r2, combo_r2|r1, and r9_greedy structural are scored on the same pairs they select on  
*Wrong because:* Baseline rows are never fit to MCS, so their struct_fidelity score is fair while these three rows score themselves on data they were picked with. The roadmap's structural-vs-activity comparison uses these same three rows, so part of that reported gain may be this same effect.  
*Fix:* Split the pairs once, rank features on one half and score struct_fidelity on the other, for selector rows and baselines alike.

**`tests/experiments/similarity_metrics/ranker_experiment.py:260`**  
Activity feature selection sees the test-fold labels it's later scored against  
*Wrong because:* Every activity-target selector and r5_rrelieff pick features using labels from rows later used as the test fold, inflating their five-nearest-neighbour accuracy score. The structural selector table is untouched — it's scored on MCS distances, which carry no labels.  
*Fix:* Move feature selection inside each fold: score and allocate features only on training-fold pairs, then score five-nearest-neighbour accuracy on the held-out fold; if too costly, report that accuracy only for the structural selectors.

**`tests/experiments/similarity_metrics/scaffold_experiment.py:78`**  
No frac_single gate on the five-nearest-neighbour metric; the grouping goes unrecorded  
*Wrong because:* The position_gated metric checks frac_single is at least 0.99 before running; this metric runs with no such check. The written JSON stores no frac_single and no group count, so a later reader can't tell the split was this uneven.  
*Fix:* Gate this metric on frac_single like position_gated is gated, in both scaffold_experiment.py and base_rep_comparison.py, and write frac_single plus group counts into every output row.

**`tests/experiments/similarity_metrics/winning_recipe_reselected.py:37`**  
winning_recipe_reselected.py splits pairs, not molecules, and normalizes features on both halves  
*Wrong because:* Each molecule appears in about eleven pairs, so nearly every molecule sits on both the selection half and the scoring half. The feature z-scoring also uses all molecules, not just the selection half, so splitting the pairs by molecule would not fix it alone.  
*Fix:* Split molecules by scaffold group first, keep only pairs with both molecules on one side, and fit the z-score stats on the selection half only.

**`tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:73`**  
Winning representation's structural fidelity score is fit to the same pairs it is tested on  
*Wrong because:* Its descriptor columns and 100 fingerprint columns are chosen by fitting to those same pairs, then fidelity is scored on them again. The three comparator representations select no columns, so only the winning score is inflated this way.  
*Fix:* Score struct_fidelity on pairs held out from the ones used to pick descriptor columns and fine-2D columns, for the winning representation and comparators alike.

### Bring these to me before doing them

- base_rep_comparison holds out whole mutated positions, which makes two position-indexed
  representations score near zero by arithmetic rather than by merit. Either drop those two
  from that comparison or add a second split. My call.

---

## P6-opig — The OPIG experiments  (10 findings)
These face an external collaborator, so a defect here is publicly visible. Most report the
search's own best cross-validation score as if it were held-out. Two conditions of the
binding-affinity-change experiment are skipped on every run.

### What to fix

**`tests/experiments/opig_activity_cliff_qsar.py:375`**  
kirby_hybrid features are selected with each fold's own labels, then tested on those folds  
*Wrong because:* Every held-out fold's labels helped pick which features kirby_hybrid uses, so no fold is scored on features chosen without seeing its own answers. The MAE, RMSE, R2, and AC-MCC printed for kirby_hybrid are measured on data that helped build it.  
*Fix:* Drop the kirby_hybrid row from the per-representation table and the Hybrid MAE and Delta columns from the cross-target summary; report only the honest version already built.

**`tests/experiments/opig_blueprint_transfer.py:184`**  
Native arm's score is measured on the same folds that picked its recipe  
*Wrong because:* Under --mode full the search folds and the scoring folds are the identical 5-fold split, so the target-side comparisons come out biased conservative, not optimistic. The source-dataset check near line 406 picks its winner from a larger search than its baseline, inflating toward the paper's claim instead.  
*Fix:* Score the native arm, the base_best baseline, and the source-side check with nested_cv_evaluate, searching and rebuilding the recipe inside each held-out fold.

**`tests/experiments/opig_blueprint_transfer.py:215`**  
Blueprint transfer picks the recipe and best representation on the folds it scores them on  
*Wrong because:* The evaluation folds are identical to the folds the search already scored, because GroupKFold is deterministic on the same groups. The recipe and the single representation chosen are the ones already seen during scoring, not held out from it.  
*Fix:* Re-select the recipe and the best single representation inside each held-out fold, using training rows only, or fold_reselection's per-fold argmax that excludes the scored fold.

**`tests/experiments/opig_cdr_flexibility.py:297`**  
opig_cdr_flexibility reports a search maximum as the hybrid AUC, not a held-out score  
*Wrong because:* That mean is a maximum picked from hundreds of configurations on those same folds, not a held-out score like the per-representation AUCs beside it. The two numbers share the same metric, so they read as directly comparable when they are not.  
*Fix:* Replace it with a leave-one-fold-out re-score, as opig_graphinity_ddg.py does, or label it a search maximum, not a results-table score.

**`tests/experiments/opig_cgr_comparison.py:379`**  
The CGR-helps-hybrid verdict is biased toward saying yes, not measuring it  
*Wrong because:* Both RMSEs already use feature indices chosen with all labels, so both are optimistic. SMILES+CGR has more representations to pick from, so it gets more of that same freedom, and that biases the delta toward showing CGR helped.  
*Fix:* Compute the CGR delta from the held-out scores already stored per representation set, not from the full-data hybrid scoring.

**`tests/experiments/opig_graphinity_ddg.py:714`**  
within_foldx and within_rosetta are skipped every run, though the code under them works  
*Wrong because:* graphinity_foldx and graphinity_rosetta are not in that list, so within_foldx and within_rosetta are skipped every run. evaluate_condition loads both datasets directly and never needs that list, so the check blocks a path that works.  
*Fix:* Re-register graphinity_foldx and graphinity_rosetta, or drop the two conditions from the file — both datasets were cut as label-unfixable on 2026-07-09, so this needs her decision.

**`/Users/apunt/repos/KIRBy/tests/experiments/opig_humanness.py:320`**  
kirby_hybrid's recipe and features both see held-out labels before being scored on them  
*Wrong because:* The blueprint behind kirby_hybrid is chosen by a search ranked over every fold's labels, not held out from them. Its features then come from refitting that same blueprint on the whole dataset, so the folds test on labels already used twice.  
*Fix:* Save the honest comparison the script already prints at lines 455-477 into that results file, and drop the kirby_hybrid rows built from leaked features.

**`/Users/apunt/repos/KIRBy/tests/experiments/opig_kingsmith_component_splits.py:674`**  
The hybrid-vs-base delta pits 5 hybrid models against 70 base draws.  
*Wrong because:* Both numbers are maxima taken over the same folds they're measured on, so both run optimistic. The base side picks the best of 70 representation-model pairs, the hybrid side the best of only 5.  
*Fix:* For each model, subtract the base representation's held-out fold score from the hybrid's, using the same arithmetic fold_reselection already runs, then rerun the experiment keeping per-fold scores.

**`/Users/apunt/repos/KIRBy/tests/experiments/opig_scaffold_similarity.py:283`**  
mode_features picks its feature subset and scores it on the same pairs it picked from  
*Wrong because:* greedy_forward keeps adding features only while the score on that pair sample improves, so its subset is tuned to those exact pairs. Top-k and all-14 stay fixed, so only greedy's score is tuned to the pairs it is reported on.  
*Fix:* Split each dataset's pair sample into two halves by variant, choose the feature subset on one half, and report its score from the other.

**`/Users/apunt/repos/KIRBy/tests/experiments/opig_trastuzumab_enrichment.py:361`**  
The printed hybrid AUC is the search's own best CV score, not a held-out score  
*Wrong because:* That score is chosen on the same folds it's measured on, so it's inflated by picking the best of many settings. Every single representation printed next to it is instead a mean AUROC over held-out folds.  
*Fix:* Score the hybrid the same way as the representations: build_fold_reselection_input(search) then fold_reselection(...), the two calls opig_humanness.py already makes.

### Bring these to me before doing them

- opig_scaffold_similarity chooses and scores a feature subset on the same 8000 pairs.
  Splitting them by variant halves the pairs available. Confirm before doing it.
- opig_graphinity's within_foldx and within_rosetta conditions are skipped every run because
  their datasets are not registered. Ask me whether to register them or delete the conditions.

---

## P7-federated — The federated experiments  (7 findings)
Each silo's hybrid recipe comes from a search over all of that silo's rows, including the
ones that later become its test set. fed04 crashes outright on two datasets, and fed01's
condition 6 differs from condition 5 in four ways at once, so their gap does not isolate
what the experiment claims.

### What to fix

**`tests/experiments/federated/fed01_noise_heterogeneity.py:143`**  
fed01's settings come from labels it later corrupts; four conditions never see that search.  
*Wrong because:* Those four settings are chosen once, using the same labels fed01 later corrupts, and used in every fold for three conditions only (C0, C5, C6). The other four conditions (C1-C4) run fixed settings and never see that search.  
*Fix:* Pass label_noise so the search runs on each fold's noisy training labels only, not the whole dataset's clean ones.

**`tests/experiments/federated/fed01_noise_heterogeneity.py:523`**  
fed01 condition 6 differs from condition 5 four ways, so their MCC gap isn't detection alone  
*Wrong because:* Condition 6 pools every silo's cleaned rows into one feature selection, the same call condition 0 uses, not per-silo like condition 5. It also differs in augmented features, in whether a model is passed, and in how the 100-feature budget is split across silos.  
*Fix:* Rebuild condition 6 with per-silo feature selection on the cleaned rows, like condition 5, and check its silent fallback to raw Morgan fingerprints.

**`tests/experiments/federated/fed02_rep_diversity_silos.py:413`**  
fed02's C5 selects its three representations using the test fold it is scored on  
*Wrong because:* The three representations entering C5 are picked by their score on the same test rows C5 is then measured on, every fold. That makes C5 look better than the best single representation and better than using all five, for a reason that isn't real.  
*Fix:* Rank each silo representation inside the training rows only, using a group-aware inner split, then build C5 from the three that win there.

**`tests/experiments/federated/fed04_cluster_aware_importance.py:146`**  
fed04 crashes on caco2 and logd before writing any result, not a corrupted metric  
*Wrong because:* caco2's label tops out at a log efflux ratio of 1.678 and logd's at 4.500, so every row lands in one class. RandomUnderSampler then raises on that single class before any metric is computed, and run_all records only an error for both datasets.  
*Fix:* Give caco2 and logd their own classification cutoff, or have fed04 refuse to run on a dataset with none defined, instead of defaulting to the hERG pKi threshold.

**`tests/experiments/federated/fed04_cluster_aware_importance.py:185`**  
fed04's in-memory search cache returns a regression search when asked for classification  
*Wrong because:* fed04 asks for a classification search after binarising caco2 and logd's labels. This path returns a regression search built on the continuous labels instead, with its allocation, importance and evaluator settings unchanged.  
*Fix:* Pass data=None from fed04 so it falls through to the on-disk cache, or key the in-memory store by labels_hash and task_type.

**`tests/experiments/federated/fed05_antibody_federated.py:199`**  
fed05 picks its hybrid recipe using rows that later become the held-out test set  
*Wrong because:* The rows used to test the hybrid recipe are the same rows used to pick it. In C6 that recipe is tuned on test rows while the quality-weighted ensemble it's compared against gets no tuning at all.  
*Fix:* Run find_best_hybrid on the training rows only, after the split, or fix the allocation method, importance method and evaluator the same way for both sides before splitting.

**`tests/experiments/federated/federated_helpers.py:1100`**  
silo_scores measures a feature set on the labels used to pick it  
*Wrong because:* Feature selection sees the validation rows' labels before those rows are scored, inflating silo_scores for every silo. Which silo gets inflated more is not established — in the one recorded run the noisy silo scored 0.0 and got the smallest budget.  
*Fix:* Carve each silo's validation rows out before create_hybrid, select features on the remaining rows only, then score on the held-out rows; pass groups through from fed05 C4 and from federated_noise_detection (which has no groups parameter yet) so the 80/20 split respects folds, not a random shuffle.

### Bring these to me before doing them

- Searching each silo's recipe inside its training rows costs a search per silo per fold.
  Give me the number before doing it.
- fed01's condition 6 is a centralised hybrid over pooled cleaned rows, not condition 5 plus
  detection. Ask me what condition 6 should be.

---

## P8-engine — find_best_hybrid and hybrid.py  (10 findings)
These are inside the engine rather than any one experiment, so a fix here changes every
experiment at once. Three concern the budget split being taken from a score measured over
the whole dataset. Be careful: another session has been editing find_best_hybrid.py
heavily, so re-read every line immediately before you change it and check the file parses
after each edit.

### What to fix

**`src/kirby/find_best_hybrid.py:3401`**  
n15's oracle_advantage only isolates the fitting head start, not the selection head start.  
*Wrong because:* Both searches pick their winning configuration using the same clean validation labels; only training differs. So oracle_advantage only measures the head start from fitting on clean labels, not from picking with them.  
*Fix:* Add a mode where find_best_hybrid scores inner folds on the noisy training labels too, so selection gets no clean-label access, and run n15's noisy search under it.

**`src/kirby/find_best_hybrid.py:4528`**  
The cached budget-split score doesn't match what create_hybrid computes on its own.  
*Wrong because:* The held-out rows and the model used to score each representation are both different from what create_hybrid uses when you call it directly. So a hybrid built outside a search gets a different budget split than the one the search reported.  
*Fix:* Score representations on the same split create_hybrid uses per fold, and either add the model to the cache key or skip the cache when a model is given.

**`src/kirby/find_best_hybrid.py:4529`**  
Precompute sets budgets using a split create_hybrid stopped using  
*Wrong because:* A comment there says this split matches create_hybrid's own split; it no longer does. create_hybrid switched to a different split on 20 August, but every default allocation method still takes its scores from the old one.  
*Fix:* Build the same split create_hybrid uses now inside the precompute too, or have it import that split, so the two can't drift apart again.

**`src/kirby/find_best_hybrid.py:5387`**  
Interval coverage passes by construction; feature selection leaks calibration labels  
*Wrong because:* calibration_coverage passes by construction: it is measured on the same molecules whose residuals set the interval width. Those molecules' labels also picked which features feed the model, so real interval widths, and the flags built from them, run narrower than they should.  
*Fix:* Hold the calibration fold out before running find_best_hybrid, select features and fit only on the training rows, then calibrate on the fold that was held out.

**`src/kirby/find_best_hybrid.py:6504`**  
Hybrid budget split uses a whole-dataset score, not each fold's own training rows  
*Wrong because:* That score decides how many features each representation gets, but it never changes per fold. A fold's own training rows never set its own budget split.  
*Fix:* Stop injecting rep_scores into fold rebuilds unless its rows are disjoint from the search's rows, or use each fold's own score from create_hybrid's internal split instead.

**`src/kirby/find_best_hybrid.py:6746`**  
Outer-fold rebuild splits the budget by a different score than the search used  
*Wrong because:* The inner search split the budget using scores computed on each inner fold's own training rows. The rebuild instead uses one mean score averaged across the whole inner search, and under top_3 can pick a different set of representations.  
*Fix:* Decide which per-representation score should split the outer-fold budget: the mean inner-CV score now used, or a live score recomputed on each outer-train fold like the search did.

**`src/kirby/hybrid.py:1315`**  
One held-out group makes f26's similarity allocation methods collapse to a default  
*Wrong because:* On tz_her2_binary, that held-out block always contains one CDRH3 edit-distance group, so every representation scores exactly 0.0. Allocation methods then fall back to an equal split of the budget across representations, or to ranking representations by their order in the roster, not by any real comparison.  
*Fix:* Make the per-representation score return no-score, not 0.0, when the held-out block is degenerate, and have the allocator raise instead of silently defaulting.

**`src/kirby/hybrid.py:2699`**  
Candidate-feature ranker splits training rows by random shuffle, not by scaffold group  
*Wrong because:* The shuffled split lets a molecule's scaffold-mates land in its own training rows, so its residuals shrink toward the fit rather than measuring real error. That is not leakage — no held-out row reaches this step — but it weakens the residual signal the ranking depends on.  
*Fix:* Add a groups argument to the ranking function, pass in the scaffold groups for that fold's training rows, and use GroupKFold instead of the shuffled KFold when groups are given.

**`src/kirby/hybrid.py:4580`**  
Base features are picked using the same rows that judge whether extra features get kept.  
*Wrong because:* That validation part then decides whether adding extra features beats the base, so the base looks stronger there than it actually is. Extra features are scored from the training part only, so the comparison isn't fair.  
*Fix:* For the performance_weighted and fixed allocation methods, compute importance from the training part only, matching how weighted_global already works; or subset the passed-in importance to that part.

**`src/kirby/result_cache.py:45`**  
Search-result cache matches rows without checking metric or task_type.  
*Wrong because:* task_type picks the fold splitter — classification gets StratifiedGroupKFold, everything else gets GroupKFold — so the same folds split differently. metric sets what the stored score means, but a fetched row's ranking_metric is dropped on read and never checked.  
*Fix:* Add metric and task_type to the cache match, or reject a fetched row whose stored ranking_metric doesn't match the metric you requested.

### Bring these to me before doing them

- Removing the whole-dataset representation scores from the engine's own fold rebuild means
  every fold re-scores its representations. I have already accepted that cost on the
  experiment-facing path — it is just preventing leakage. Confirm it applies here too, then
  do it.
- The conformal interval selects features on every row and calibrates on a subset of the
  same rows, so its coverage passes by construction. Fixing it changes every reported
  coverage number.

---

## P9-data-and-plumbing — Dataset grouping, batch correction, resume, shared helpers  (8 findings)
A mixed set: two group vectors that are not really groups, batch correction that uses the
class label, a resume check that skips staleness, and three shared helpers that hand
callers a hybrid fitted on every row.

### What to fix

**`tests/experiments/_shared.py:952`**  
Fold 0's held-out hybrid score picks from candidates fold 0 already helped filter.  
*Wrong because:* Cluster submitters default to a prune that keeps only the top half of each allocation-importance pair's configs, ranked by fold-0 score. Fold 0's held-out hybrid score then gets picked from those survivors, while the single-representation score never goes through that cut.  
*Fix:* Either drop fold 0's term when computing the delta from a pruned search, or run with a schedule that prunes nothing (checkpoints=[], tail_drop=0, threshold=0) so nothing is fold-dependent.

**`tests/experiments/_shared.py:3257`**  
f27 never builds the similarity recipe representations for its 'recipes' and 'both' pools  
*Wrong because:* load_data has no path to build the eleven similarity recipe representations; get_rep_factories, which builds them, is never called from here. So pool='recipes' always errors; pool='both' errors too if the general representations are already cached, or otherwise silently returns just those general representations mislabeled 'both'.  
*Fix:* Add a path in load_data that calls get_rep_factories for the sim_* names and passes the result to run_nested_cv; until then, make pool='recipes'/'both' refuse outright instead of silently skipping or duplicating.

**`tests/experiments/_shared.py:4513`**  
A hybrid recipe is picked using every row, then s02 reports its score as held-out.  
*Wrong because:* In s02, that same score is then reported as the held-out delta, so the recipe was chosen using rows the delta claims to exclude. s05 and s06 rebuild features per fold from training rows only, so they keep the recipe choice, not contaminated feature weights.  
*Fix:* Route these experiments through nested_cv_evaluate, or have precompute_hybrid_searches (similarity_helpers.py:945-958) stamp its output as whole-dataset-selected so the delta is never read as held-out.

**`tests/experiments/_shared.py:4600`**  
s07's held-out noise score is fit on the same labels it is later scored against  
*Wrong because:* s07_noise_bridge scores that one matrix per held-out fold against single-rep baselines that never saw any labels. The hybrid side already used a fold's own labels to pick its features before being scored on that fold.  
*Fix:* Rebuild the hybrid feature matrix per fold in s07 with build_fold_hybrid, as s08 already does, or drop the held-out label on this number.

**`tests/antibody_baseline_survey.py:82`**  
Jain identity split gives one group per antibody, so folds aren't grouped by homology  
*Wrong because:* Each antibody lands in its own group — measured 136/137, 129/130, 207/217 groups to samples. GroupKFold then splits like plain KFold over distinct antibodies, not by homology.  
*Fix:* Decide whether you want unseen-antibody generalisation (current split) or unseen-germline-family generalisation (V-gene grouping via ANARCI, already in cross_species_ness.py).

**`src/kirby/datasets/nucleotide.py:366`**  
gb_* genomics folds are set by an MD5 hash of the sequence, not a real group  
*Wrong because:* A hash of the sequence carries no biology, so the five folds are close to a random split of the rows. The hash only keeps byte-identical duplicate sequences together — homologous or overlapping sequences still land in different folds.  
*Fix:* Carry chromosome position through where a data mirror has it; for the two demo datasets, which have no coordinate at all, choose sequence-identity clustering or drop the grouped-split claim.

**`tests/experiments/run_dataset.py:126`**  
Resumed runs skip stale checks: file existence only, never folds or representations.  
*Wrong because:* The check only asks whether the output file exists, or, for noise and similarity, whether it holds a measurement. It never asks whether the fold grouping, labels, or representation list changed since that file was written.  
*Fix:* Save the fold grouping, labels, split count, and sorted representation list in each result file, and compare them before skipping a recompute.

**`src/kirby/datasets/tcga.py:295`**  
TCGA batch correction is fit on held-out samples, and separately per class for LUAD vs LUSC  
*Wrong because:* The correction for a site is fit on all of that site's samples, including the held-out ones a fold later tests on. For LUAD vs LUSC, the correction for each sample is fit using only samples with its own class label.  
*Fix:* Fit the batch correction inside each fold, on training rows only, and correct both LUAD and LUSC samples together in one matrix.

### Bring these to me before doing them

- The four gb_* genomics datasets assign folds by hashing the sequence. Genomic Benchmarks
  ships its own train/test split which the loader reads and then throws away. Proposing that
  split is the concrete option; k-mer clustering is the other. Bring me a real suggestion or drop it; do not
  restate the problem at me.
- The seven Jain endpoints give one antibody per group. This is NOT leakage — nearest
  neighbour identity in that panel was measured at 0.823 median. It only means the claim is
  'generalises to an unseen antibody', not 'to an unseen germline family'. The decision is which claim I want to make.
- TCGA batch correction is fit per cohort, and cohort is the class label for
  tcga_luad_vs_lusc. It cannot run today (the package is not installed and the cached
  matrices predate the code) but will fire on the next cold rebuild.
