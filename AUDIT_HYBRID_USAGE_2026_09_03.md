# Hybrid usage audit — find_best_hybrid and the five experiment suites

**Date:** 2026-09-03
**Scope:** `src/kirby/find_best_hybrid.py`, `src/kirby/hybrid.py`, `src/kirby/result_cache.py`,
`src/kirby/registry.py`, `src/kirby/noise_spec.py`, `src/kirby/screen.py`,
`tests/experiments/_shared.py`, and every experiment in complexity, noise, similarity, OPIG and
federated.
**Out of scope by instruction:** `tests/experiments/framework/` (f-numbered).

**What this file is.** Every line below came from reading the code on 2026-09-03. Nothing here is
a measured result. Each item was written by one reader and then re-checked by a second reader who
was asked to refute it. 351 items were raised, 42 were refuted, 309 survived. What survived is a
claim about what the code does, not a claim about what any number means.

A 12 percent refutation rate is low. Treat it as one re-read each, not as proof. The items marked
*small* in §3 were re-read once and no further; the six in §4 I opened and checked myself.

**Counts.** 309 items survived: 151 marked *large* (changes a reported number, or needs your
decision), 158 marked *small*.

---

## 1. What you have to decide

Six things. Everything else in this file is downstream of them.

### 1.1 The similarity suite never asks for the similarity objective

Not one similarity experiment passes `objective='similarity'`. Every hybrid the suite scores as a
similarity space was selected for model accuracy instead.

Sites: `s02:324-328`, `s03:126-130`, `s04:145-151`, `s05`, `s06`, `s07`, `s08`, `s10`, `s12`,
`s13`. The engine supports the objective and `get_precomputed_search` refuses to serve a geometry
request from the predictive store, so the plumbing is there and unused.

The choice: re-run the similarity suite under the geometry objective, or state in the write-up
that these are predictive hybrids measured on geometry. Cost of the first is a full similarity
re-run on every dataset.

### 1.2 Every federated allocation is an equal split

`create_hybrid` accepts `model=` with no `metric_fn`. Every representation then scores minus
infinity, and the allocation falls through to an equal split under whatever name the run is filed
under.

Eight sites do this: `fed02:370-378`, `fed02:421-429`, `fed03:462-468`, `fed03:750-757`,
`fed03:813-818`, `fed03:1091-1098`, `fed03:12xx`, `federated_helpers:1100-1107`.

`build_fold_hybrid` already refuses this combination (`_shared.py:3676-3684`). `create_hybrid`
does not. Adding the same refusal makes all eight raise, so each needs the dataset's metric passed
alongside its model. Every federated number is affected.

### 1.3 Nine experiments report the inner search's own score as the hybrid's number

The search's `score_mean` is its own cross-validation score on the rows it selected from. Put
beside a baseline computed in a separate loop, it is not a comparison.

`c14:247-293`, `opig_cdr_flexibility:275-311`, `opig_trastuzumab_enrichment:343-374`,
`s08:213-223`, `opig_cgr_comparison:373-399`, `opig_ginkgo_developability:576-601`,
`c02:236-278`, `c11:1035-1037`, `similarity_helpers:945-956`.

`nested_cv_evaluate` exists for exactly this and returns `delta_mean_of_folds`. The fix per
experiment is to route the headline through it. The cost is one nested-CV run per dataset per
experiment.

### 1.4 Configurations are ranked against each other on different numbers of folds

`find_best_hybrid.py:3597-3600` — a fold that raises is skipped and the configuration is still
scored. `score_mean` is the mean over whatever folds survived. Only all-folds-failing raises.

The fold list is stored, so the count is recoverable. Nothing checks it. The same shape appears in
`n01:292-298`, `n07:509-510`, `n15:212-247`, `s13:270-299`, `c11`, and in `_shared.py:1689` and
`1771`.

Two policies, your pick: refuse a configuration that did not score every fold, or keep it and
record `n_folds_scored` so ranking can exclude it. Either changes which configuration wins some
searches.

### 1.5 The result database ignores the metric and the condition suffix

`result_cache.py:45-48` — the row identity is tier, labels hash, groups hash, fold count, seed,
splitter id. The scoring metric is stored on the row and never compared on read. `condition_suffix`
reaches the checkpoint filename and the search metadata, and reaches neither the database filename
nor the row identity.

Live consequence, read off the code: c13's with-representation and without-representation arms
share rows (`c13:498`, `c13:535`), and c02's native-dimension pass fetches c02's whole main grid
(`c02:321-343`). f26's allocation arms are the same shape.

`condition_suffix` can be folded into the `tier` string with no schema change and no wipe — the
noise spec and the objective already ride there. Putting the metric in the key needs every
`result_db/*.db` deleted, because `_ensure_schema` has no migration. That wipe is your call.

### 1.6 Four files cannot run at all

`c20_signal_saturation.py:50-54`, `c30_balanced_global_selection.py:28-30`,
`exp_b_balanced_global.py`, `c43_buchwald_signal.py` and `c43_buchwald_step3.py:19-25` all import
their harness from a `/private/tmp` scratchpad that no longer exists. c30 and exp_b also wrote
their results there.

Whether the harness is recoverable, or those experiments are rewritten against `_shared.py`, is
your call. I have not looked for a copy.

---

## 2. The recurring patterns

Most of the 309 items are the same handful of mistakes repeated. Naming them once is more useful
than listing them 309 times. Section 3 has the per-experiment detail.

### 2.1 A bare `create_hybrid` where `build_fold_hybrid` belongs

`build_fold_hybrid` forwards the internal evaluator, the augmentation strategy and margin and
fraction, the candidate ranking rule, the augmentation budgets, and the representation scores the
search chose. A bare `create_hybrid` forwards none of them and takes the defaults instead.

15 experiments: `c03:160-168`, `c08:220-234`, `c09:352-359`, `c09:504-511`, `c10:375-392`,
`c10:497-509`, `c24:219-227`, `c24:302-310`, `n11:199-207`, `n12:146-154`, `n13:184-193`,
`s11:696-705`, `s11:1447-1456`, `opig_blueprint_transfer:249-261`, and every federated file.

Note n13 builds it correctly 400 lines later in its honest arm, so the two arms of one file
disagree.

### 2.2 `groups` not passed, so the internal split is a random 20 percent

With no groups, `create_hybrid` carves a `RandomState(42)` 20 percent block (`hybrid.py:4448-4453`).
On a scaffold-split dataset that holdout shares scaffolds with its own training rows. It is what
ranks the representations against each other, and what every augmentation strategy other than
`'none'` scores keep-or-drop on.

Every site in 2.1, plus `c07:337-342`, `create_hybrid_cached` (`_shared.py:78-85`, no groups
parameter at all), `fed01:312-318`, `fed04:89-97`, `fed05:144-150`, `s06:489-498`,
`federated_helpers:1533-1539`.

### 2.3 Selected on all rows, then scored on folds of those same rows

The recipe is searched once on the whole dataset and then applied per fold. Each fold's test rows
helped choose the recipe.

`c11:414-432`, `c12:492-509`, `c17:297-301`, `c19:143-150`, `c32:202-209`, `s07:348-353`,
`s11:883-890`, `s13`, `similarity_helpers:945-956`, `fed01:143-152`, `fed02:117-126`,
`fed04:185-194`, `opig_cgr_comparison:651-665`.

c17 is the sharpest case: the hybrid, mRMR and RFE all select on all rows, while PCA and random
cannot leak — so the comparison is between methods that saw the test rows and methods that did not.

### 2.4 A concatenation of every representation as the only comparator

Concatenation baselines are banned. Five experiments have no individual-representation comparator
at all.

`n06:208-211` (hybrid versus concat, and only the hybrid was built with the labels),
`s06:500-503`, `s09:479-492`, `fed03:308-318` (all three parts), `fed05:115-169` (C1's
"independent baseline" is itself a hybrid).

fed03 already fits per-representation models in Part A and throws them away.

### 2.5 One model's blueprint fed to every model

The internal evaluator and the downstream model are search axes. A recipe chosen for random forest
is not the other models' recipe.

`c16:243-252` (every model family on a hybrid searched for random forest), `opig_cgr:355-366`,
`opig_activity_cliff_qsar`, `opig_kingsmith:450-451` (one RF blueprint under all six models),
`c13:206-239`, `fed01:171` and `fed04:203` (`'match'` hand-resolved to `'rf'` even when the swept
model is xgboost or mlp), `fed05:382-389` (first silo's blueprint used for every silo).

### 2.6 Axes cut without a stated reason

`c31:128-138` cuts 12 of 15 weighting schemes for wall time — and the weighting family is the
question c31 asks. `n11:75` is missing manhattan, two of three documented distances.
`n12:63` sweeps two importance methods where the engine searches three.
`opig_graphinity:73-77` uses a hand-picked representation list that omits the mutation-aware ones.
`opig_scaffold_similarity:228-236` tests two of five representations and one of two distances.
Every federated condition runs at a single budget of 100 with no sweep (`fed01:146`, `fed02:71`,
`fed04:187`).

In the framework: `MODEL_MAX_BUDGET` (`find_best_hybrid.py:2665-2676`) removes budgets per model,
so a budget column compares unequal sets of models, and can leave a model with zero configurations
and no message. The registered budget policy drops budgets silently
(`find_best_hybrid.py:2216-2225`). Fold-0 pruning deletes whole allocation methods from the grid
by default, and they leave no row anywhere (`find_best_hybrid.py:4908-4933`).

### 2.7 A failure swallowed so a broken run reads as a null result

`nested_cv_evaluate`'s baseline arm swallows every exception with no message and has no
all-folds guard (`find_best_hybrid.py:7649-7664`). A dead baseline arm then reads as
`delta = NaN`, and `best_single_reps_per_fold` still names a representation, so the output looks
like a real comparison. The per-budget rebuild swallows everything silently too (`7747`).

Same shape at `n07:509-510`, `n13:654-664`, `noise n01:292-298`, `_shared.py:1689`, `1771`,
`c15`, `s11`.

### 2.8 Cache identity gaps in the framework

Beyond §1.5. Not in any validated key: `tune` (`5569-5572`), `screen_threshold` and
`feature_blacklist` (`5588-5597`), the MCS pair set under `struct_fid` (`6072-6085`), augmentation
matrix widths (`256-258`), `cv_splitter_id` and `task_type` in the per-representation baseline
cache (`144-171`). `cache_dir` reroots only the checkpoint and the database — baselines, folds,
per-fold hybrid evaluations and fine-tuned embeddings still read and write the shared root
(`5698-5700`). Cache validation iterates the expected metadata only, so a default-valued run is
served a file written under a non-default setting (`304-313`).

---

## 3. Per-experiment detail

Severity: **L** = changes a reported number or needs your decision. **S** = local fix.
Everything here is OPEN unless §4 says I fixed it.

### 3.1 Framework core — `src/kirby/`

**Search and ranking**
- **L** `3597-3600` partial-fold ranking. See §1.4.
- **L** `5259-5288` a configuration scoring NaN can be returned as a model's best, because the
  all-NaN guard filters a list nothing then uses.
- **L** `3798` vs `3996`/`4107` under a geometry or blend objective the hybrid arm averages folds
  with `np.mean` and the baseline arm with a NaN-skipping mean.
- **L** `4994-5002` the database seeds results with every stored row for the identity, including
  configurations this run did not ask for.
- **L** `5064-5071` resuming from a checkpoint evaluates a different configuration set than an
  uninterrupted run, under a schedule with tail dropping.
- **L** `2978-2991` a per-fold representation factory that raises leaves that fold with fewer
  representations, and the mean mixes folds built from different pools.
- **L** `3052-3060` a representation wider than 5000 features is scored on 500 and recorded at its
  full dimension, contradicting the full-dimension baseline rule.
- **S** `4850-4854` an exception in the fold-0 pruning pass is swallowed even under verbose.
- **S** `5190-5215` refinement rows are computed under different settings and then ranked beside
  phase-1 rows.
- **S** `2764-2820` mid-flight tail pruning ranks entries from several budgets together.

**Honest evaluator**
- **L** `7649-7664` silent baseline arm. See §2.7.
- **L** `3983-3999` under `struct_fid` the per-representation baselines hold nearest-neighbour
  scores labelled as MCS fidelity, so the two arms are not the same quantity.
- **L** `4016-4023` under a blend on an RMSE dataset the best-single-representation comparator is
  picked backwards — the worst RMSE wins the max.
- **L** `5822-5842` the at-n boundary budget takes a different value in each outer fold, so its
  honest mean is over one fold and sits beside five-fold means.
- **L** `7299-7302` the documented inner-fold leak in `outer_rep_builders` also decides the
  baseline arm's champion. Direction of the effect is not established by the code.
- **S** `7768-7776` the hybrid-arm guard tests `is not None`, but an unscoreable geometry fold
  stores NaN.
- **S** `1507-1567` `_readacross_objective_score` accepts `is_clf` and never reads it, so a
  classification run is a nearest-neighbour regression on class codes.
- **S** `6975` `best_by_model` reports `n_features` as the requested budget, not the width the
  hybrid had.

**Rebuild fidelity**
- **L** `6744-6746` every rebuild through `apply_blueprint` splits the budget by a different
  per-representation score than the search used. The injected `rep_scores` is the best baseline
  across models at full dimension; the search used the internal evaluator's score on the fold's
  internal carve. A top_k rebuild can therefore select a different set.
- **L** `4529-4541` the scores that split the budget on the default predictive path are computed
  on a `GroupShuffleSplit` carve, while `create_hybrid` carves with `GroupKFold`. No leak — both
  live inside the fold's training rows — but the two are different holdouts and the comment says
  they match.
- **S** `6677-6742` `apply_blueprint` never passes `is_clf`, so a classification recipe is rebuilt
  with regression scoring.
- **S** `6453-6470` the full-data refit behind `return_fitted` rebuilds a different recipe.

**Allocators and augmentation — `src/kirby/hybrid.py`**
- **L** `4431`, `4448-4453` the random 20 percent fallback. See §2.2.
- **L** `2101-2113` model without metric. See §1.2.
- **L** `1170-1210` a blended objective weights representations by rank while a plain objective
  weights them by score, so f28's simplex corners are not the limit of its interior.
- **L** `216-222`, `1315-1321` when a geometry score degenerates every representation gets the same
  number and the split silently becomes an equal split under the requested name.
- **L** `3205-3218` vs `3244-3255` `augment_uncapped` scores the extras in rank order and ships
  them in donor order, so the block size was chosen on a matrix that was never built.
  `augment_capped` was written to avoid this; uncapped is the default.
- **L** `4883-4885` `create_hybrid` writes demoted representations into the caller's extras
  dictionary and never removes them, so later rebuilds in the same fold depend on model ordering.
- **L** `find_best_hybrid.py:3584-3593` an auto-demoted representation is trained on its
  top-importance features and validated on the first features of the same representation. Widths
  match, so nothing raises.
- **L** `find_best_hybrid.py:7486-7487` the honest evaluator cannot score an augmented top_k
  blueprint — the fold is dropped from the hybrid arm and kept in the baseline arm.
- **L** `4565-4582` base feature importance is fitted on all rows including the internal validation
  rows the augmentation accept-or-reject test then scores on. Deliberate and commented; it biases
  which extra blocks are accepted, not the outer number.
- **S** `1274-1310` `min_floor` can spend more than the total budget and the trim loop cannot take
  it back.
- **S** `79-86` an unrecognised objective string — the cut `similarity_sac` included — is silently
  scored as prediction instead of raising.
- **S** `1331-1342` top_k's width-capped fill can pull features from representations outside the
  top k.

**Support modules**
- **L** `noise_spec.py:14-26` one `NoiseSpec` yields two different noisy label columns depending on
  the entry point, and both carry the same `spec_id`. No experiment currently mixes the two.
- **L** `check_precompute_status.py:314-333` a precompute that died after one configuration reads
  as done, because the database stores no completeness marker.
- **S** `registry.py:9-12` the registry documents a guard that does not exist; the metric defaults
  to r2.
- **S** `screen.py:68-73` `screen_reps(metric='auroc')` silently scores accuracy.

### 3.2 `tests/experiments/_shared.py`

- **L** `1752-1763` `evaluate_cv_with_train` rebuilds the hybrid from four recipe fields only, so
  the train/validation gap c09 reports is a different hybrid from the one the search chose.
- **L** `1811-1838` `evaluate_single_rep_cv` silently compresses with PCA any representation wider
  than the training fold, while the hybrid arm it is compared against is not compressed.
- **S** `58-75` `create_hybrid_cached`'s cache key omits the labels and most of the recipe.
- **S** `4680-4682` `get_fitted_hybrid` puts the objective into the search but not into its
  full-data rebuild.
- **S** `3913-3918` the distance-tier precompute runs a `'capped'` search its own comment says it
  does not, and no consumer reads.

### 3.3 Complexity

- **L** c02 `236-278` headline delta is an argmax-over-grid hybrid against an argmax-over-reps
  baseline, with no honest counterpart for the extra across-conditions selection.
- **L** c03 `101-118` the per-dataset "optimal configuration" is SVM's best, not the dataset's best
  across models.
- **L** c05 `253-270` the baseline arm is scored through a different path — hard-label AUROC on
  classification datasets while the hybrid uses probabilities.
- **L** c08 `76-77`, `179-180` defaults to and hardcodes the cut `greedy` allocation.
- **L** c08 `249-262` the conformal calibration set is a random split of a scaffold-grouped fold.
- **L** c09 `342-378` mutual information is estimated on the same rows and labels that drove the
  feature selection. `336-339` falls back to the cut `greedy` allocation in all three
  sub-analyses. `165-247` the ANOVA treats allocation, importance and evaluator variation as
  within-cell replicates.
- **L** c10 `152-176` the recipe is chosen by a clean-label search, in the one experiment that
  corrupts the labels. Its docstring promises a config-under-noise result the code never computes.
- **L** c12 `72-158` run through the suite c12 only sees the 10 pruned representations, so three of
  its nine modalities are absent.
- **L** c14 `74-75`, `136`, `145`, `418-433` two arms on different splits, hard-label AUROC on one
  arm, and a headline Wilcoxon pooling deltas in incommensurable units.
- **L** c15 `482-486` 15c samples only the p≈n band so its instability ratio cannot be computed;
  `485-486` an unexplained 2048 cap deletes the sweep on datasets above 2560 samples; `504-519`
  namespaces per-budget searches by mangling `dataset_name`, which the engine forbids because it
  bypasses the budget and model-skip policies.
- **L** c16 `243-252` every model family evaluated on a hybrid searched for random forest.
- **L** c17 `297-301`, `319-330` selection leak, and at a matched budget the hybrid arm can carry
  more features than the competitors and draw them from representations the competitors cannot see.
- **L** c19 `119` the budget is fixed at 100 and cannot be changed from config or command line.
- **L** c21 `184-189` the noise decoy is sized from the donors' full widths, but conditions C and D
  cap those donors to 5 features each.
- **L** c22 `227-231` SHAP is averaged across folds but mapped back through only the last fold's
  feature info.
- **L** c23 `229-277` Condition E searches a superset of Condition A's grid, so its delta is
  non-positive by construction for rf, lightgbm and xgboost.
- **L** c24 `285-338` the transfer matrix re-selects the features on the target fold, so nothing
  about the feature selection is actually transferred.
- **L** c32 `320-336` the quota arm does not hold the budget fixed — it drops the width-capped
  remainder, and the engine it claims to reproduce does not.
- **S** c05/c07/c23 namespace their caches by mangling `dataset_name` (`c05:226`, `c23:100-101`),
  which silently drops the per-dataset budget cap and model-skip policy. `cache_namespace` is the
  supported way.
- **S** c13 `428-445` condition A records itself as `'ridge'` while running random forest.
- **S** c18 `133-134` above the sample count the hybrid arm is rebuilt at a different budget from
  the arms it is compared against; `44-57` the depth and uniform arms rank with `lightgbm_gain`
  while the natural arm uses the searched method.
- **S** complexity_helpers `869-876` `run_all` builds a config with `allocations` and
  `importance_methods` that the precompute never forwards to `find_best_hybrid`.
- **S** c30/exp_b `159-192` compare arms at unequal feature counts when the per-representation cap
  times the pool size is below the budget.
- **S** c43_step3 `94-97` reports a lightgbm hybrid minus a hardcoded random-forest number from a
  different script run as a paired delta.
- **S** c30, c31, c32 and exp_b are allocation bake-offs living in `complexity/` rather than
  `tests/experiments/framework/`.

### 3.4 Noise

- **L** n06 `208-211`, `213-225`, `268-322` concat-only comparator; hybrid versus concat confounded
  because only the hybrid was built with the labels; and replicate rows are identical in feature
  space, so a neighbour-based estimator's score for a ground-truth row comes from that row's own
  replicate group's labels.
- **L** n06 `288-300` estimators whose constructor has no groups parameter run on non-group-aware
  internal splits, silently.
- **L** n07 `502-508` builds the hybrid feature space from the clean labels it then corrupts, and
  records nothing about it.
- **L** n11 `199-207` bare `create_hybrid` — no groups, seven recipe fields dropped.
- **L** n12 `146-154` same; `162-168` conditions 4 and 5 are the same construction under two
  importance methods and collapse to identical arrays whenever the blueprint's method is one of the
  two swept.
- **L** n13 `184-193` Track A/B builds the hybrid with a bare `create_hybrid` while the honest arm
  400 lines later builds it correctly; `939-952` the similarity-objective hybrid is counted as a
  baseline in the hybrid-advantage tally; `806-828` the per-fold reselection arm has one candidate and
  one baseline, so the leave-one-fold-out selection it claims to perform does nothing.
- **L** n15 `18-37` the noise-searched arm still picked its blueprint using clean validation labels.
- **L** noise_helpers `64-90` the noisy-search levels do not overlap the levels the rest of the
  suite runs.
- **L** `_similarity_hybrid_space.py:9-12` the equal-block scaling and the 0.5 count down-weight are
  cancelled by the callers' `StandardScaler`.
- **S** n01 `202-213` an empty blueprint dict is not None, so a failed hybrid pool wipes out the
  per-representation baselines too; `416-426` representations are ranked by absolute gap while the
  primary metric is the signed gap with `lower_is_better=True`.
- **S** n02 `174-178` `results['summary']` is written empty and the base-versus-hybrid comparison
  the docstring names is never computed; `53-54` svm is dropped under a reason that does not hold.
- **S** n07 `413-421` the saved result says the injection was Gaussian; the code injects outlier
  contamination. `276-281` a comment attributes a per-dataset assay error to a function that
  returns a flat 0.5.
- **S** n09 `311` the per-fold noisy re-search inherits its augmentation condition from the
  clean-label pool.
- **S** n13/n12/n15 call `cv_split` without `task_type`, so classification folds fall back to the
  label-dtype heuristic and differ from the search's folds.

### 3.5 Similarity

- **L** the objective. See §1.1.
- **L** s05 `356-362` the "fixed_ref, HONEST" delta subtracts a baseline chosen in-sample over
  representation and k.
- **L** s08 `200`, `250-253` the budget sweep rebuilds the same hybrid at every budget; `213-223`
  inner-search fold scores are mixed into the per-budget prediction curve at one budget only;
  `116-118` the kNN arm the experiment is built around is never scored; `234-284` the held-out arms
  rebuild a recipe chosen with the held-out rows in view.
- **L** s06 `500-503` the conformal comparison has only a concatenation baseline.
- **L** s07 `348-353` the held-out contamination arm scores a hybrid whose features and weights were
  fitted on the whole dataset.
- **L** s09 `479-492` the cross-dataset applicability-domain delta is hybrid minus concat, read off
  the tautological pair; `586-610` the dimension ratio compares two prediction models and is dropped
  silently when it cannot be computed.
- **L** s11 `682-705` the hybrid conditions fall back to a hand-picked `entropy_1.0` /
  `lightgbm_gain` recipe and the output never says which was used; `883-890` the recipe is chosen
  with the pool labels the loop is meant to be discovering, while its comparators are not.
- **L** s13 `306-320` importance weighting is given to the hybrid retrieval arm only, then the
  weighted arm competes against unweighted representations; `171`, `259` the `pca_matched` space is
  matched to a fixed 100 rather than the hybrid's width, and never enters either comparison.
- **L** similarity_helpers `945-956` every blueprint in the group is searched on the whole dataset
  and then scored on folds of that same dataset.
- **L** `tests/slurm_scripts/run_similarity_experiments.sh:42-63` the shipped submit script runs
  older duplicates of s06, s07, s08 and s09, not these modules.
- **S** s05 `144-158` labels each per-budget cell with the budget it asked for, not the budget the
  returned blueprint carries.
- **S** s06 `136-145` asks the pool for budget 100 and rebuilds at a different width; `203-204` the
  weighted comparison arms use a hardcoded importance method while the hybrid arm uses the
  blueprint's; `489-498` random calibration split when a fold's groups are unusable.
- **S** s09 `240-246` NaN values enter the aggregated series and the Wilcoxon.
- **S** s10 `956` asks for a budget-10 hybrid under `run_all` where the file's own default is 100.
- **S** similarity_helpers `829-841` the config keys for allocations, importance methods,
  evaluators, budgets and models never reach the search.

### 3.6 OPIG

- **L** opig_cgr `373-399` the pool verdict comes from two full-data hybrids re-scored on their own
  folds; `355-366` one hybrid searched for rf is fed to every model.
- **L** opig_cdr_flexibility `275-311` the hybrid's number is the search's own score printed beside
  plain per-representation scores.
- **L** opig_graphinity `104-109` two of four conditions are skipped at startup because
  `graphinity_foldx` and `graphinity_rosetta` are not registry keys; `216` the PDB leakage filter on
  the synthetic-to-experimental transfer never runs because the test dict has no `pdb_ids`;
  `73-77` the representation pool is hand-picked and omits the mutation-aware representations.
- **L** opig_trastuzumab `343-374` the hybrid is reported as the inner-search score, on a different
  metric and a different split from the representations it is compared with.
- **L** opig_kingsmith `422-442` the per-fold blueprint is chosen by r2, not the registered rmse;
  `450-451` one RF blueprint is scored under all six models.
- **L** opig_scaffold_similarity `228-236` on the multi-scaffold dataset only two of five
  representations and one of two distances are tested.
- **S** opig_humanness `81` the default run exits before doing anything — `humanness_species` is not
  a registry key; `293-307` no `condition_suffix`, so this search and the dataset's shared
  precompute evict each other.
- **S** opig_ginkgo `300-316` the per-fold search optimises r2 while the experiment and the
  competition both rank by Spearman; `576-601` the cross-property "Hybrid wins N/9" is a same-fold
  argmax on both arms, and the honest block it stores is never printed.
- **S** opig_blueprint_transfer `573-574` `--max-folds` is parsed and never read; `215-220`
  `_cv_split` is called without `task_type`.
- **S** opig_kingsmith `656-680` the headline maxes the baseline over about twenty representations
  by five models but the hybrid over five models, biasing the delta against the hybrid.
- **S** opig_scaffold_hybrid `165-167` three CLI flags are parsed into config and never reach the
  run.

### 3.7 Federated

- **L** the equal-split defect. See §1.2.
- **L** fed01 `225-227` the FLuID arms see one 2048-bit Morgan fingerprint while the KIRBy arms see
  all ten representations; `143-152` the KIRBy conditions inherit a recipe chosen on clean labels
  while the FLuID conditions only see the noisy ones; `506-530` C6 is a centralized hybrid, not a
  federated one, so it differs from C5 in two ways at once.
- **L** fed01/fed04 `55`, `44` the logd and caco2 wrappers threshold their labels at pKi 6.0.
- **L** fed02 `259-272` C5 picks its top-3 representations using the test fold's own scores.
- **L** federated_helpers `1864-1894` `fluid_within_kirby_across` computes a score-proportional
  budget per virtual silo and never applies it, so fed02 C6 is an equal split with duplicated
  features; `1096-1107` the coordinator's allocation can only shrink a silo below `budget//k`, never
  grow it, and ties break by silo number.
- **L** fed03 `320-322` the single-silo baseline is chosen per fold by its own test score, and that
  oracle choice is saved as the headline; `203-217` Part A rebuilds the keep-mask by set membership,
  keeping both copies of a repeated InChIKey and misaligning representations against labels;
  `1193-1275` Part C C2 scores on test folds containing synthetic rows; `706-888` Part B has no
  prediction-level fusion arm, so Strategy 4 cannot be answered.
- **L** fed04 `279-328` the cluster-conditional versus global comparison confounds per-region
  allocation with per-region training size.
- **L** fed05 `108-112` every number comes from one 80/20 draw — no folds, no seeds, no error bar;
  `199-206` the recipe is searched on the silo's full representation set and applied to the shared
  subset; `413-507` C4 has no clean control; `509-603` C5 reports an RMSE with no comparator.
- **S** fed05 `199-206` per-silo `find_best_hybrid` runs with caching off; `199-216` the searched
  budget is discarded and the rebuild runs at `create_hybrid`'s default 100; `205`
  `adaptive_pruning=0.25` prunes more than the engine default with no reason recorded.
- **S** federated_helpers `1322-1326` noise-detection setting 3 hardcodes
  `performance_weighted`/`treeshap`/`rf` instead of the blueprint fed01 already resolved.

---

## 4. What I fixed

Six things, all local, none of them changing a stored number.

1. `_shared.py:4274`, `4287`, `4476` — `get_best_precomputed_search` and
   `get_best_search_for_downstream` captured `return_fitted` and `weight_method` by name and then
   dropped them. Every `SearchResult` from the pool came back with `fitted_results` empty.
2. `s03_feature_concentration.py:126-160` — read the selected features off the `feature_info` the
   full-data rebuild returns, not off `search.fitted_results`. s03's whole
   selected-versus-random-versus-PCA comparison had been skipped for every representation, budget
   and model, with no error.
3. `n13_noise_distributions.py:527` — defined `_seed_combined`. It was referenced at line 656 and
   never defined, so every Track A method raised `NameError` into a bare except and Track A
   produced nothing.
4. `n13_noise_distributions.py:977` — iterate the noise types the run produced instead of the
   module-level `NOISE_TYPES`, which is `None` until `run()` resolves it. Iterating it raised
   `TypeError` and killed the summary.
5. `opig_scaffold_hybrid.py:91-125` — read `delta_mean_of_folds` and `blueprints_per_fold` from
   `by_model`, where `to_dict()` puts them. Both of the file's answers printed empty.
6. `s08_optimal_dimension.py:796` — removed the duplicate `--budgets` argument. It collided with
   the shared parser's, so s08 raised `ArgumentError` and could not be launched from its own
   command line. Also honoured `--budget` in `n06_bindingdb_ground_truth.py:220`, which was parsed
   and then discarded; its default is now None, so the existing behaviour is unchanged.

All six files parse. Nothing was run.

---

## 5. What I did not do

- Did not run anything. No training, no cluster, no imports beyond a syntax check and one
  argument-parser check on s08.
- Did not touch `tests/experiments/framework/`.
- Did not change any cache key, any ranking rule, or any allocation.
- Did not look for a copy of the deleted `/private/tmp` harness that c20, c30, c43 and exp_b import.
- Did not check whether the 42 refuted items were refuted correctly.
