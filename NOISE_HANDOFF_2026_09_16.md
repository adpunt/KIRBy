# Handoff prompts for the rest of the noise plan

Written 2026-09-16. The plan these come from is
`~/.claude/plans/read-through-users-apunt-downloads-compa-stateful-quasar.md`.
n19 is built. Everything below is not.

Each section is a prompt to paste into a fresh session. They are written to be
self-contained: each one names what to read first, what was already decided and
why, and what the smoke has to show. Nothing below re-opens a decision that is
already recorded in `NOISE_AUDIT_2026_09_10.md` or in `CLAUDE.md`.

**Three rulings already given, 2026-09-16. Do not re-ask them.**

1. The hybrid engine does NOT get a per-molecule weight in this round. Build
   without it and record the limit on every weighted row.
2. n22 waits behind a count. Run the ChEMBL probe first and decide after.
3. Order: n19, n20, then n17/n18 gaining a hybrid, then n21, then n22.

**Two things every one of these must do, because the rest of the suite does.**

- The recipe comes from `get_best_search_for_downstream(..., pool_subset=
  'predictive_and_similarity')` and the features are rebuilt per fold through
  `build_fold_hybrid`. n02, n06, n07, n11, n12, n13 and n19 all do it that way;
  copy `n19_measurement_count._hybrid_for_fold`.
- Register the id in three places in `tests/experiments/noise/noise_helpers.py`:
  `SINGLE_DATASET_EXPERIMENTS` (:54), `_import_experiment`'s map (:517) and
  `_experiment_full_name`'s map (:540). Missing the third writes the result into
  a directory named after the short id instead of the experiment.
- `save_experiment_results(results, output_dir, experiment_id, dataset)`, in
  that order. n17 and n18 had the arguments rotated and their files landed under
  the working directory where the resume probe could never find them; both were
  fixed 2026-09-16.

---

## Prompt 1 — n20, cleaning before the recipe search

> Read `NOISE_AUDIT_2026_09_10.md` in full, then
> `tests/experiments/noise/n07_remediation.py`,
> `tests/experiments/noise/n18_clean_the_real_labels.py` and
> `tests/experiments/noise/n19_measurement_count.py` in full. Then read
> `src/kirby/find_best_hybrid.py` around the `label_noise` block at :8155-8215
> and the `_run_search` signature at :4150-4170.
>
> Build `tests/experiments/noise/n20_clean_before_search.py` with
> `_smoke_n20.py` beside it.
>
> THE QUESTION. Every cleanup in this suite is applied after the recipe is
> fixed, so the cleaning never reaches the feature selection. `create_hybrid`
> ranks features and splits the budget against whatever labels it is handed, and
> it is always handed the uncleaned ones. Nothing has measured what that
> ordering costs.
>
> THE ENGINE CHANGE IT NEEDS, and it is small. `_run_search` already takes
> `train_labels`, a full-length label column the model trains on while it is
> scored on the clean one (`find_best_hybrid.py:4153-4162`). The top-level
> `find_best_hybrid` will only build that column from a `NoiseSpec`
> (`:8166-8182`). Add a top-level `train_labels=` that reaches it. Refuse it
> together with `label_noise`, and refuse it together with `rep_factories` for
> the reason already written at `:8168-8176` — fine-tuned embeddings would still
> be fitted on the untouched labels. The cache identity already hashes the
> actual column (`_label_col_id` at `:8193`), so two label columns cannot read
> each other's stored rows and no wipe is needed.
>
> FOUR WAYS OF RUNNING IT, and the point is the shape:
>
> | the recipe was searched on | the model was trained on |
> |---|---|
> | the labels as they are | the labels as they are |
> | the labels as they are | the cleaned labels |
> | the cleaned labels | the cleaned labels |
> | the cleaned labels | the labels as they are |
>
> The second row is what the whole suite does today. The difference between the
> second and the third is what the ordering costs. The fourth separates the
> feature choice from the fit: a gain there is the cleaning helping the feature
> choice with no cleaned label ever reaching a model.
>
> THE CLEANUP. n07's four, called rather than copied, the way n18 calls
> `n07_remediation._evaluate_fold`. Sweep the share of rows flagged over n18's
> `FLAG_RATES`.
>
> WHERE IT RUNS. Both regimes, because they answer different questions. Under
> injected noise on the shared grid (0.0, 0.15, 0.3, 0.5, 0.75, 1.0 of the clean
> label spread) under `outlier_p10`, where a corrupted subset exists so
> `noise_helpers.detection_scores` is defined. And on the labels as they come,
> n18's regime, where nothing was corrupted. Put n20 in
> `HONOURS_NOISE_CONDITION`.
>
> COST. One search per fold per cleaned label column. Use `speed='fast'` with
> `tune=True` and `tune_knn_k=True`, matching what n15 was set to on 2026-09-12
> and for the same reason — n15 also searches inside the experiment.
>
> THE SMOKE MUST SHOW. At level 0.0 nothing is corrupted, so the four ways are
> handed identical labels and their four scores must agree to within the spread
> across noise draws. A gap there is the two searches choosing from lists of
> different lengths, which is the defect n15 carries at audit section 4, n15
> defect 3 — if it appears, report it rather than widening the tolerance. Also
> check that `find_best_hybrid(train_labels=..., label_noise=...)` raises, that
> `train_labels` of the wrong length raises, and that two different label
> columns land on two different cache identities.

---

## Prompt 2 — n17 and n18 gain a hybrid

> Read `tests/experiments/noise/n17_replicate_removal.py`,
> `n18_clean_the_real_labels.py` and `n19_measurement_count.py` in full, and
> `NOISE_AUDIT_2026_09_10.md` sections 14 and 15.
>
> Both files read `sorted(reps)[0]` — `n17:334` and `n18:149`. They are the only
> two experiments in the directory that fit models and have no hybrid, and they
> are also the only two where the noise is real rather than injected.
>
> Give each of them the two-space treatment n19 has: a base representation and a
> hybrid, every condition run in both. Copy `n19._hybrid_for_fold` rather than
> writing a third version of it — if it is being copied a third time, move it
> into `noise_helpers.py` and have n19 import it from there.
>
> Two things to carry over from n19 and write onto every result row.
> `recipe_saw_clean_labels` is False: nothing is corrupted here and there is no
> clean column, so the recipe saw the same labels the experiment trains on, and
> the oracle caveat n11, n12 and n13 carry does not apply. And on n17, the
> hybrid must be built on the training molecules of a fold only, from their
> single-measurement labels, never from the consensus labels the held-out
> molecules are graded against.
>
> ALSO IN n17, and these are cuts for compute that audit section 14.7 records as
> having no stated reason behind them: it ran `rf` alone and `drug_maccs` alone.
> Give it the roster of five and the pruned representations. Read
> `feedback_do_not_prune_search_space.md` first.
>
> THE SMOKE MUST SHOW. `_smoke_n17.py` and `_smoke_n18.py` already separate a
> dataset whose labels are genuinely wrong from one where nothing is wrong.
> Extend both so that separation is checked in the hybrid space as well as the
> base one, and so a run where the hybrid recipe is unavailable still produces
> every base-space number rather than dropping the run.

---

## Prompt 3 — n21, the distance the neighbours are found in

> Read `tests/experiments/noise/n12_importance_weighted_distance.py` in full,
> `src/kirby/utils/noise_estimation.py` (the neighbour estimators at :239-495
> and `_knn_indices` at :178-236), and `NOISE_AUDIT_2026_09_10.md` section 4
> under n12 and sections 6.4 and 6.9.
>
> Build `tests/experiments/noise/n21_neighbour_distance.py` with
> `_smoke_n21.py`.
>
> THE QUESTION. Seven of the detection methods judge a molecule against its
> nearest neighbours. Which distance those neighbours are found in has never
> been swept properly.
>
> FOUR DISTANCES:
>
> | distance | where it comes from |
> |---|---|
> | plain | the base representation as it is |
> | importance-weighted | each feature multiplied by the square root of its importance — what n12 does |
> | forest proximity | how often two molecules land in the same leaf of a fitted forest |
> | the hybrid's own | the feature space the recipe built |
>
> Forest proximity is absent from the repository — `.apply()` is called nowhere
> in `src/kirby/utils/`. It is `rf.apply(X)` giving one leaf id per tree per
> molecule, and the proximity of two molecules is the share of trees where the
> two ids match. Turn it into a distance as one minus that share. The forest
> must be fitted on the fold's training molecules only.
>
> WHAT IS SCORED. `noise_helpers.detection_scores` against the molecules whose
> label actually moved — it is defined under `outlier_p01`, `outlier_p05`,
> `outlier_p10`, `grouped_wider` and `censoring`, and `DETECTION_IS_DEFINED`
> names them. Average precision is the headline, because it needs no threshold
> and so does not inherit the defect in audit section 6.9 where a filter's ruler
> grows with the amount of noise. Beside it, the share of the clean-to-corrupted
> gap that repairing on each distance puts back.
>
> ONE DEFECT IN n12 IT SETTLES. n12 multiplies features by the square root of
> their importance and then compares them under a Manhattan distance (audit
> section 4, n12 defect 1). The square root is the Euclidean convention; under
> Manhattan a feature's weight in the distance is its multiplier, not the square
> of it. Run both conventions against both metrics so the scaling that is right
> for the metric is the one reported. Do not quietly fix n12 — n21 measures which
> is right and n12 changes after.
>
> THE SMOKE MUST SHOW. A synthetic set where two of twenty features carry the
> whole signal and the other eighteen are noise. The importance-weighted and
> forest-proximity distances must find the corrupted molecules better than the
> plain one there, and a second set where every feature matters equally must show
> no such gap.

---

## Prompt 4 — n22, detection against a curator's judgement

> RUN THE COUNT FIRST. `tests/experiments/noise/_probe_chembl_herg.py` already
> pulls CHEMBL240. Add a count of how many records carry a
> `data_validity_comment`, run it, and report the number before building
> anything. If it is under fifty the experiment does not run, and that count is
> a fact about ChEMBL worth writing into `NOISE_AUDIT_2026_09_10.md` either way.
>
> Then, if the count supports it:
>
> Read `NOISE_AUDIT_2026_09_10.md` sections 11.1 and 11.4,
> `tests/alternative_data_noise_robustness.py` around the fetch at :990-1055,
> `src/kirby/datasets/herg.py:490-560`, and
> `tests/experiments/noise/noise_helpers.py:441-513`.
>
> THE QUESTION. Every detection number in the suite is scored against a list the
> same script drew. This one is scored against a list a person made. ChEMBL
> marks records its own curators found suspect. The hERG fetch excludes them —
> `"data_validity_comment__isnull": "true"` at
> `tests/alternative_data_noise_robustness.py:997`. Keep them, with the flag as
> a column.
>
> Build `tests/experiments/noise/n22_curator_flag.py` with `_smoke_n22.py`.
> Precision, recall and average precision of each detector against the flag, in
> the base representation and in the hybrid, through the same
> `noise_helpers.detection_scores` that grades the injected runs — so the two are
> readable side by side: the same detectors, the same number, one graded against
> a distribution and one against a person.
>
> The detector list is `n11_estimation_remediation_grid`'s, imported rather than
> copied, for the reason recorded at `n06_bindingdb_ground_truth.py:330-334` —
> two hand-kept lists drift and then the two files cannot be joined.
>
> THE SMOKE MUST SHOW. A synthetic set where a known subset of molecules carries
> a flag AND a wrong label, and a second where the flag is attached at random.
> Average precision must be high on the first and near the flagged share on the
> second.

---

## Prompt 5 — the repair rules n07 and n11 lack

> Read `src/kirby/utils/noise_estimation.py` in full, and
> `tests/experiments/noise/n07_remediation.py` and
> `n11_estimation_remediation_grid.py` in full.
>
> `smooth_by_scores` (`noise_estimation.py:1434-1485`) moves a label toward a
> target, and every caller in the repository passes neighbour means. Three
> targets the menu names are already computed somewhere and thrown away:
>
> - a group-aware out-of-fold prediction — `oof_predict`
>   (`src/kirby/utils/noisy_labels/core.py:98`), which n19 already uses;
> - a Gaussian-process posterior mean — `GaussianProcessEstimator.predictions_`
>   (`noise_estimation.py:805`), computed on every call and discarded;
> - a conditional quantile range to clip into —
>   `BayesianRFEstimator(method='quantile')` (`noise_estimation.py:1193-1196`)
>   computes the two quantiles and nothing clips. `quantile_forest` 1.4.1 is
>   installed if a proper quantile regression forest is wanted instead.
>
> The first two are caller changes and need no new function. The third needs a
> clipping smoother beside `smooth_by_scores`, because clipping into a range is
> not moving toward a target.
>
> Add all three as repair rules in n07 and n11 beside the ones they have. Do not
> drop any existing rule.
>
> THE SMOKE MUST SHOW. On a set with a known corrupted subset, each of the three
> recovers some of the clean-to-corrupted gap, and the quantile clip changes only
> the labels that sat outside the range.

---

## Prompt 6 — the fifth model takes a weight

> `_standard_dnn_train_predict(..., sample_weights=...)` at
> `src/kirby/utils/lnl_methods.py:96-100` is a written weighted-MSE training
> loop, and the comment at `:28` records its architecture as matching
> `DNNRegressor._build_net()` exactly. Its only caller is `HeteroscedasticGP`.
>
> `DNNRegressor.fit` (`src/kirby/models/dnn.py:47`) is
> `def fit(self, X, y, groups=None)` and raises
> `TypeError: unexpected keyword argument 'sample_weight'`, measured 2026-09-12.
> That one TypeError is why `dnn` is absent from `n02.WEIGHTED_MODELS`
> (`n02_sample_weighting.py:101`), from n13's track B
> (`n13_noise_distributions.py:163`) and from n19's three weighted conditions.
>
> Thread a `sample_weight` through `DNNRegressor.fit` and `DNNClassifier.fit`,
> then add `dnn` to `n02.WEIGHTED_MODELS`. n11, n12, n13 and n19 all discover the
> refusal at run time rather than from a list, so they need no change.
>
> THE SMOKE MUST SHOW. Fitting with every weight equal to one gives the same
> predictions as fitting with no weight at all, to within the network's own
> run-to-run spread. And a weight of zero on a block of rows gives the same
> predictions as deleting those rows.

---

## Prompt 7 — extrapolating the level sweep back to zero error

> Read `src/noise_inject/noiseInject/metrics.py` (`_retention_weibull` at :57
> and `_curve_robustness` at :78), `tests/experiments/noise/noise_helpers.py`
> `_already_there` at :380-423, and `tests/experiments/dataset_policy.py`
> :527-548.
>
> This is a READING of numbers the suite already writes, not a run. Every
> injecting experiment produces a score at six levels of added error. Fit that
> curve and extrapolate to the point where the added error is zero: that
> estimates what the score would be if the assay were exact.
>
> It gives a measured version of the ceiling `_already_there` currently computes
> from a published figure. `REFERENCE_ERROR_LABEL_UNITS` holds a published
> figure for three datasets — herg_ki, caco2 and logd — out of forty-two, so
> every other dataset has no ceiling at all today.
>
> Error adds in quadrature, so a score measured at level L on labels already
> carrying b of their own spread sits at sqrt(b² + L²). The extrapolation
> therefore runs back to the total, not to L = 0, and the difference between
> those two is the thing being estimated. Be explicit about which one every
> number is.
>
> Put it in an analysis script, not in an experiment — nothing here fits a model.
> `_curve_robustness` needs four levels and the shared grid has six.
>
> THE SMOKE MUST SHOW. On synthetic scores generated from a known ceiling and a
> known baseline error, the extrapolation recovers the ceiling. And on a dataset
> whose published error is registered, the extrapolated ceiling and the
> published one are reported side by side rather than one being used to check
> the other.

---

## Prompt 8 — the cheap defects

> Five, none of them connected to the above. Each was read out of the code on
> 2026-09-16.
>
> 1. `NoiseInjectorClassification.get_per_class_flip_rates` builds its dictionary
>    and the file ends before the return statement
>    (`src/noise_inject/noiseInject/core.py:1213-1223`). It returns None. The
>    vendored copy at `src/noise_inject/` is meant to be byte-identical to
>    `~/repos/NoiseInject/`, so fix it there and re-vendor, not here.
> 2. `calculate_noise_metrics` asks `get_effective_noise` for
>    `method='std_normalized'` (`src/noise_inject/noiseInject/metrics.py:163`);
>    the three it accepts are `rms`, `rms_normalized` and `range_normalized`
>    (`core.py:941-949`). Passing `y_noisy_dict` raises. Same repository.
> 3. `tests/experiments/run_dataset.py` has no `--noise-condition` flag
>    (`:174-179`), so the censoring runs decided in audit section 13.1 cannot be
>    started through the normal route — only by calling `run_all` directly.
> 4. `tests/slurm_scripts/run_smoke_gate.sh:167` loops over `n02 n03 n08 n16`.
>    n03 and n08 were cut from the suite and are absent from
>    `_import_experiment` (`noise_helpers.py:88-90`).
> 5. `tests/audit_full_wave.py:46-47` lists twelve noise experiments, five of
>    which are cut, and omits c10, n06 and n14 through n19.
