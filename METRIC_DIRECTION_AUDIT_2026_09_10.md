# Metric direction — audit of `find_best_hybrid` and the experimental suite

2026-09-10. Scope: `src/kirby/` and `tests/experiments/`, excluding
`tests/experiments/framework/` and `tests/experiments/annitia/`.

The question this asks of every line: a score is compared, ranked or subtracted here —
does the code know whether a bigger number is better on this dataset? `rmse` and `mae`
are lower-is-better. `r2`, `accuracy`, `auc`, `spearman`, `pearson`, `ci`, `f1_weighted`,
`mcc`, `sac`, `readacross` and `struct_fidelity` are higher-is-better
(`src/kirby/find_best_hybrid.py`, `BUILTIN_METRICS`). `logd`, `caco2`, `herg_ki` and
`buchwald_hartwig` run under rmse; the antibody sets under spearman; `herg` under auc;
the nucleotide sets under mcc; DTA under ci.

---

## 1. Fixed. Seventeen changes across fourteen files, all in the working tree

Each one is a code fact I read in this session, not a claim from a document.

**`src/kirby/find_best_hybrid.py:2167`** — a feature count was being ranked as if it were
a score. `_evaluate_baselines` writes `baselines[rep]['dimension_scored'] = 500` (200
under 1000 molecules) for any representation wider than 5000 features. Three separate
filters read that dict and excluded only `'dimension'` and `'_secondary'`, so 500 passed
through as a score. On a higher-is-better dataset `max()` returned 500 and that
representation outranked every other one; on rmse `min()` ignored it. One shared exclusion
tuple now covers all three sites (lines 879, 2192, 7960).

**`src/kirby/hybrid.py:1974` — the worst one, and it changes which features get picked.**
`allocate_weighted_global` gave a representation missing from `precomputed_scores` the
literal 0.0. Every real score arriving there has already been turned so that higher is
better (`_score_rep_for_budget`, line 2734), so on an rmse dataset every real score is
negative and an invented 0.0 is the *largest* of them. Measured by calling the weighting
functions directly, three representations with real scores of −0.30 and −0.35 and one
unscored: the unscored representation took a weight of 0.35 against 0.05 and 0.00 for the
two that were actually measured — 87 of every 100 features, to the one representation
nobody could score. It is 0.00 now. `allocate_performance_weighted` had already been
corrected to NaN, with a comment saying exactly why 0.0 is wrong; its twin was missed. The
`warn_once` that names the missing representations was copied across too, so if this ever
happens the run says so.

**`src/kirby/find_best_hybrid.py:7643`** — the metric's own direction is now kept across
the blend flip. Under a blend objective containing prediction, `lower_is_better` is set to
False at line 7652 because the winner is scored on a correlation. `SearchResult.baselines`
is still filled with the dataset's own metric, so ranking the representations with the
flipped flag took `max()` over a set of RMSEs and named the worst representation.
`nested_cv_evaluate` has carried this guard since 2026-09-04 (line 9063); the search did
not. The two consumers now read the new flag: the pre-search screen at 7760 and
`Blueprint.rep_scores` at 7963.

**`tests/experiments/_shared.py:2847`** — `fit_learning_curve` was fitting a rising curve
with the asymptote bounded to `[0, 1]`. On an rmse dataset whose error exceeds 1 — logd
sits near 1.17 — the starting guess was outside its own bound, `curve_fit` raised on every
call, and the `except` branch returned `max(scores)`, which is the *worst* score in the
sweep, reported as the asymptote. It now fits on the score turned so it rises, bounds the
asymptote by the observed range, and returns `min(scores)` on failure for a lower-is-better
metric. Measured on a synthetic RMSE curve whose true asymptote is 1.2000: it now returns
1.2000 with a fit quality of 1.00, where before it returned 1.2795 with a fit quality of
0.00. `c05_learning_curves.py:430` and `:437` pass the direction.

**`tests/experiments/_shared.py:6063`** — `make_structured_output` now resolves the
top-level `lower_is_better` from the loaded dataset instead of taking the caller's word.
An experiment whose headline quantity runs the other way gets that recorded separately as
`primary_metric_lower_is_better`. This matters because `save_experiment_results` fills the
top-level key from the registry only when it is `None`, so one hardcoded literal hid the
registry's answer for a whole file.

**`tests/experiments/noise/n01_activity_cliff_framework.py:742`** — the one caller that did
that. It passed `lower_is_better=True`, which describes the cliff-versus-non-cliff gap, next
to a `metric_name` the registry fills in as spearman or ci. Checked after the fix: on
`flab_thermo_tresanco` the file now records `lower_is_better: False` for the dataset and
`primary_metric_lower_is_better: True` for the gap.

**`tests/experiments/complexity/c02_complexity_grid.py:355` and `:434`** — two printed
deltas were hardcoded hybrid-minus-baseline, so on an rmse dataset a hybrid that won printed
a negative number. Sign-normalised. Print-only; nothing parses this stdout.

**`tests/experiments/complexity/c09_formal_analysis.py:267`** — the console line said
`Grand mean R²` for a number that holds whatever metric c02 ran under. The stored key was
renamed to `grand_mean_score` on an earlier pass and this print was left behind. It now
names the metric.

**`tests/experiments/complexity/c12_modality_ablation.py:252`** — deleted
`higher_is_better(metric)`, which returned True for any metric name handed to it and had no
caller. c12's own scores come from its local `score_predictions`, which is r2 / auroc /
f1_weighted only, so c12's `lower_is_better=False` constants are correct; the dead helper
was a trap for the next caller.

**`tests/experiments/complexity/c16_feature_selection_as_regularization.py:366`** — the
train-minus-validation gap had a fixed sign. On rmse the training error is the smaller
number, so overfitting produced a negative gap. Now positive means overfitting under both
directions, matching `c09:625` and c33. The consumer at 445-460 matches on
`abs(gap - hybrid_gap)`, so flipping both sides consistently leaves its answer unchanged.

**`tests/experiments/complexity/c17_information_theoretic_decomposition.py:352`** — the
file now stores `lower_is_better` beside `metric`, and the console line names the metric
instead of printing `R²=` for a number that holds RMSE on logd, caco2 and herg_ki. The
stored key is still called `downstream_r2` — see §2.

**`tests/experiments/complexity/c21_single_rep_augmentation.py:800`** — the head-to-head
table's header claimed `higher = better` above three columns of raw scores. The rows are
picked direction-correctly; only the header was wrong. It now names the metric and its
direction.

**`tests/experiments/noise/n07_remediation.py:745`** — `summary.noise_gap` was
clean-minus-noisy with a fixed sign, so on rmse it went negative exactly when the injected
noise did damage. The recovery denominator twenty lines below already knew this and tests
`(ref_gap < 0) if lower_is_better`. The gap now matches: positive always means the noise
cost something.

**`src/kirby/find_best_hybrid.py:8182`** — `apply_blueprint` took the direction from
`getattr(blueprint, 'lower_is_better', False)`. It accepts a plain dict, and
`Blueprint.from_dict` drops any key the dataclass does not declare, so a blueprint written
to disk without that key arrived as False and inverted every accept in the budget split
with nothing raising. It now resolves from the blueprint's own `metric` name through
`BUILTIN_METRICS` and falls back to the flag only when the name is not a builtin. Checked:
a dict carrying `metric='rmse'` and no direction key used to answer False and now answers
True.

**`tests/experiments/_shared.py:702`** — `extract_top3_base_reps` sorted the
per-representation baselines by `search.lower_is_better`. That is not the direction the
baselines were measured in: under a blend containing prediction it has been overwritten to
False while the baselines stay in the dataset's own metric, so on an rmse dataset the sort
would put the largest error first. It resolves from `search.metric_name` first now. Thirty
files call this, and `s04_activity_cliffs.py:463` fits a model on whichever representation
comes out first, so it is not print-only. No dispatched run passes an objective today, so
this was latent rather than live.

**`tests/experiments/opig_humanness.py:369` and `:461`, and
`tests/experiments/opig_graphinity_ddg.py:641`** — three reads of a stored honest-rescore
block that took `block.get('lower_is_better', False)`. `hr_direction` exists for exactly
this and refuses to guess; the saved path in these files had been converted to it and the
console paths had not. All three now call it.

---

## 2. Open — these need your decision

### 2.1 Two stored deltas run in the dataset's own direction, with no corrected twin beside them

*Corrected 2026-09-10 after reading each file. An earlier draft of this section named eleven
files. That list came from a pattern search, not from reading them, and it was wrong — most
of those files are fine, for the reasons below.*

Two fields hold a difference of two scores in the dataset's own metric with a fixed sign,
so a positive number means the arm won on spearman and lost on rmse:

- `tests/experiments/complexity/c19_importance_weighted_training.py:295` — `deltas[cond]`,
  each weighting treatment minus the unweighted baseline. Three lines below it,
  `_best_fn = min if lower_is_better else max` picks the winner correctly, so the choice is
  right and only the stored number is raw.
- `tests/experiments/noise/n13_noise_distributions.py:810` — `delta_from_clean_dnn`. Nothing
  in the repo reads it.

Everything else in the earlier list turned out to be sound:

- `c21:718` stores a raw `delta` and a sign-normalised `relative_improvement` in the same
  dict, and the only reader (`firstpass_notready.py:1099`) takes the corrected one.
- `n09:565` does the same: `delta_from_clean` raw, `score_ratio_pct` direction-branched at
  `n09:548`.
- `c32:323-324` computes `dR2` off `r2_score`, which is higher-is-better on every dataset,
  and applies `gsign(lower_is_better)` to `dNat`, which is the dataset's own metric. Both
  are oriented.
- `c04:129-131` uses its own `mae` and `r2`, and orients each one for itself.
- `c12` scores with its own `r2` / `auroc` / `f1_weighted` and never reads the registry.
- `s07:713` subtracts two contamination rates, which are counts over counts.

- **Fix both.** They are one line each, and the raw operands are stored beside them, so no
  re-run: `c19` keeps `scores` and `n13` keeps `mean_s` and `clean_mean_dnn`.
- **Leave them and rely on the `lower_is_better` key each file already writes.**

**My recommendation: fix both, and match what the file next to them already does** — c21
and n09 both keep the raw difference and add a direction-normalised sibling, which loses
nothing and lets a reader take either. That is the convention to spread, not a rewrite.

**>> (a) fix both, or leave them:**

### 2.2 A field named `r2` that holds something else

Twenty-two files in the suite carry a score field whose name says `r2` while the value is
whatever metric the dataset runs under. The two that a reader is most likely to hit:
`c17`'s `downstream_r2` (the answer to c17's second question) and `c31`/`c32`'s
`r2_delta_mean` and `dR2`.

Renaming a stored key changes the results-file schema, so I did not do it. I fixed the
console lines and added the direction to the file instead.

- **Rename the keys.** New files read correctly on their own. Old files on disk keep the old
  name, so any reader has to know both. Nothing in the repo reads `downstream_r2` today.
- **Leave the names and rely on the stored `metric` and `lower_is_better` keys.** No schema
  change, and the direction is already there to be read.

**My recommendation: rename, and only in the files whose number is quoted** — c17, c31, c32.
The others are internal.

**>> (b) rename the r2-named keys, or leave them:**

### 2.3 `c10`'s robustness gate admits every representation on an rmse dataset

`c10_noise_robustness.py:311` and `:412` pass `lower_is_better` into
`compute_noise_robustness` but not `baseline_threshold`, so it takes its default of 0.3
(`src/kirby/noise_robustness.py:25`). The gate inside is applied on one side of the
direction only: for a lower-is-better metric the only requirement is a clean-label score
above zero, so every representation is admitted. For a higher-is-better metric any
representation whose clean-label score is under 0.3 is dropped and returns NaN.

So the rmse datasets and the spearman/auc datasets are not admitting representations on the
same rule, and which representations c10 ranks depends on which.

This changes what c10 measures, so it is yours.

- **Turn the gate off** (`baseline_threshold=None` at both call sites). Every representation
  is ranked on every dataset, including ones whose clean-label score is near chance.
- **A per-metric floor**, read from the registry the way the budget-split rule in
  `FRAMEWORK_FINALIZATION.md` §3.5 is. Costs one line per metric.

Either way the run has to be repeated on the higher-is-better datasets, because the current
gate has already dropped representations there.

**>> (c) turn c10's baseline gate off, or give it a per-metric floor:**

### 2.4 Under a blend objective, the hybrid and its comparator are scored on two different quantities

`nested_cv_evaluate`, `src/kirby/find_best_hybrid.py:9363`. Both arms are scored on the
blend on each outer fold. But the representation the hybrid is measured *against* is picked
from the inner search's `baselines`, and under a blend those hold the dataset's own metric,
not the blend. So the comparator is chosen on one quantity and scored on another, and
`delta_mean_of_folds` — the number the thesis quotes — inherits that.

The direction handling here is right; the quantity is not. This only fires under a dict
blend or `objective='predictive_corr'`, which today means f28.

The mechanism if you want it: `blended_rep_scores` (`src/kirby/hybrid.py:326`) already
produces one blend score per representation on each inner fold, and `hybrid.py:1395`
already calls it to split the budget. Returning that map out of the search alongside
`SearchResult.baselines` would let the comparator be picked on the same quantity it is
scored on.

**>> (d) pick the blend comparator on the blend, or leave it on the dataset's metric:**

---

## 3. Checked and found right

Worth recording, so none of it is re-derived.

**The selection path is direction-correct throughout.** `_run_search`'s pruning, the
mid-flight tail drops, the end-of-fold-0 strategy prune, `_aggregate_results`,
`get_best_for_downstream`, `get_best_search_for_downstream`'s pool sort, `honest_rescore`
and `_hr_lofo_score` all branch on the direction they are given, and they are given the
right one.

**`hr_direction` refuses to guess** (`tests/experiments/_shared.py:1245`). A stored block
with neither `metric_name` nor `lower_is_better` raises rather than defaulting to
higher-is-better, and a block whose stored flag contradicts `BUILTIN_METRICS` raises too.

**Every allocator's scores are sign-normalised before they are weighted.**
`_score_rep_for_budget` ends `return -raw if lower_is_better else raw`
(`src/kirby/hybrid.py:2734`), so every weighting scheme downstream — linear, softmax, rank,
`thresholded`, `score_gap`, `sqrt` — sees higher-is-better on every metric. No double
negation anywhere on that path.

**The six findings from the 2026-06-29 audit are all fixed.** s11's hardcoded
`lower_is_better=False`, s11's gain sign, c09's bias-variance proxy, c10's NDS
interpretation, c10's summary header, and `inspect_results.py`'s selection. I re-read each
one in the code.

**`_screen_reps` is dead.** It runs only when `screen_threshold` is not None, which defaults
to None and which no caller in the repo passes. Fixed anyway.

**The hand-rolled running-best loops are clean.** Fifty-six variables initialised to a
`±inf` sentinel. Fifty-two of them hold a quantity with a fixed direction — a training loss,
a validation loss, a distance, a correlation, a recovery percentage. The four that hold a
dataset-metric-dependent score are fed values that were sign-normalised upstream, and I
found the line that does it in each case (`s08:571`, `hybrid.py:2734` for the two greedy
allocators, `s11:610`'s scorer is `scoring='r2'` on every arm).

---

## 4. Method, and what the coverage numbers are

Two passes. Pass one searched for metric-direction defects by reading each file against the
question its `EXPERIMENTS_FINALIZED.md` entry says it answers: 38 regions covering the
whole of `src/kirby` and all 117 documented suite files, 174 candidate findings, each one
then handed to a separate agent told to refute it. 156 were refuted, 13 confirmed, 5 left
plausible but unreachable. The refutations are as useful as the confirmations and are kept
in the session record.

Pass two is mechanical and does not use names at all — §5 has its numbers.

**Tests.** `test_metric_direction_persisted.py`, `test_metric_direction_end_to_end.py`,
`test_budget_split_arithmetic.py` and `test_pairdist_redundancy.py` — which between them
cover the persisted direction, the end-to-end delta sign, the budget-split arithmetic and
the allocator — pass against every change here: 27 passed, 1 xfailed, and the xfail
predates this work. Run with `PYTHONPATH=tests:src python3 -m pytest`.

One failure elsewhere, and it is not from this audit:
`test_x_mol_rd_analysis.py::test_search_cache_path_matches_t_mol_4_layout` expects a
`__full` speed-tier suffix in the cache filename that commit `9bd48ac` removed. It fails
identically with every change here stashed, and `tests/experiments/cross_species/` is
outside the five categories this audit covered.

**What this audit did not cover.** `tests/analysis/`, `tests/experiments/annitia/`,
`tests/experiments/cross_species/`, `tests/experiments/pretrain_rep/` and the loose
comparison scripts at the top of `tests/experiments/` are outside the five categories you
named. The AST catalogue reaches them; no agent read them. Seventeen defaulted-direction
reads sit in that set — §5 lists them.

---

## 5. Pass two — every ordering operation, classified

This pass does not use names at all. An AST scan catalogued every ordering operation in
`src/kirby` and `tests/experiments` in every spelling it knows: `max`, `min`, `sorted`,
`.sort`, `sort_values`, `argmax`, `argmin`, `argsort`, `idxmax`, `idxmin`, `nlargest`,
`nsmallest`, `partition`, `argpartition`, `cummax`, `cummin`, `rank`, `rankdata`,
`percentile`, `quantile`, `clip`, `most_common`, `searchsorted`, `reverse=`, the four
comparison operators, and `a if a > b else b`. That is 7041 sites. 3648 survive after
dropping the ones whose operands provably cannot be a score — loop bounds, array shapes,
string compares, `None` guards, clamps against an integer literal.

Twenty-nine agents classified those 3648, one file at a time, each required to account for
every site in its list. **All 3648 are accounted for. Nothing was left unreached.**

| class | count | what it means |
|---|---|---|
| A | 2280 | not an ordering of any decision quantity |
| B | 1091 | orders a quantity whose direction is fixed and intrinsic — a count, a width, a distance, a variance, a p-value, a similarity coefficient, an importance score, a training loss |
| C | 259 | orders a quantity whose direction depends on the dataset's metric, and the direction is consulted correctly |
| D | 18 | orders a quantity whose direction depends on the dataset's metric, and it is not consulted |

The agents also found **179 hand-rolled orderings the AST scan cannot see** — running-best
loops, boolean-mask selection of an extreme, indexing after an ordering, early-stopping
tolerances, a threshold with a fixed polarity applied to a score. Those were classified the
same way and are included in the 23 findings below.

Twenty-three findings came out of the 18 class-D sites and the 179 extras. Each went to a
separate agent told to refute it. **Two are real. Both are fixed** — the
`allocate_weighted_global` 0.0 and `extract_top3_base_reps`, both in §1. Twenty-one were
refuted, and the refutations are worth trusting: they were specific about which line the
quantity came from and where its direction was set.

### The one class worth acting on, and it is not a bug today

Twenty-two sites in thirteen files read the direction as
`getattr(obj, 'lower_is_better', False)` or `.get('lower_is_better', False)`. Every one was
refuted individually, and correctly: the field is always present on the object each of them
actually receives, so the default never fires.

But that is the pattern `hr_direction` was written to refuse, and its docstring says why —
"The historic sign bug came from readers doing `bool(block.get('lower_is_better'))`, which
turns a MISSING key into False = higher is better and silently inverts every RMSE/MAE
conclusion." A default of False is not a direction. It is a guess that happens to be right
on 46 of the 70 registry entries.

Five of the twenty-two are fixed in §1 — the two in `opig_humanness`, the one in
`opig_graphinity_ddg`, `extract_top3_base_reps`, and `apply_blueprint`. The remaining
seventeen are:

| file | lines |
|---|---|
| `tests/experiments/_shared.py` | 4332, 5012, 5023, 5538, 5559 |
| `tests/analysis/blueprint_trace.py` | 237, 255, 392 |
| `tests/analysis/quick_hybrid_summary.py` | 45 |
| `tests/analysis/firstpass_notready.py` | 67 |
| `tests/analysis/f15_pruning_analysis.py` | 81 |
| `tests/experiments/noise/n13_noise_distributions.py` | 991 |
| `tests/experiments/pretrain_rep/pretrain_rep_experiment.py` | 103 |
| `tests/experiments/cross_species/x_ab_therapeutic_eval.py` | 168 |
| `tests/experiments/cross_species/x_mol_rd_analysis.py` | 140, 204 |
| `tests/experiments/framework/f02_adaptive_pruning.py` | 108 |

Two notes on that list. `_shared.py:5012` and `:5023` are inside
`get_best_precomputed_search`, which has zero callers anywhere in the repo — it was
superseded by `get_best_search_for_downstream`. Deleting it is the cleanest fix, and it is
the one place in the list where the default genuinely inverts the answer: when a search is
`None` the sentinel is `-inf`, which on a lower-is-better metric is the best possible value,
so the missing condition wins. The `tests/analysis/` and `cross_species/` files are outside
the five categories you named, so I left them.

- **Convert all seventeen.** Resolve from the metric name through `BUILTIN_METRICS` first,
  fall back to the stored flag, and raise when neither is there — the contract
  `hr_direction` already has. Nothing changes in any current run, because none of the
  defaults fire; what changes is that the next result file written without a direction key
  stops the run instead of inverting quietly.
- **Convert the five in the suite and delete the dead function.** Leaves
  `tests/analysis/` alone.
- **Leave them.** They are correct today.

A grep for the pattern returns nineteen hits, not seventeen. The two extra are the
fallback arms of the fixes in §1 — `find_best_hybrid.py:8186` and `_shared.py:706` — which
now fire only when the metric name is not a builtin, and in that case the stored flag does
describe the quantity being ordered. They are correct as they stand.

**My recommendation: convert all seventeen, and delete `get_best_precomputed_search`.** The
cost is an afternoon and no re-runs. The thing it buys is that this class cannot come back,
which it has once already.

**>> (e) convert all seventeen, convert the five in the suite, or leave them:**


---

## 6. Pass three — the framework's decisions, traced one at a time

Passes one and two gave `find_best_hybrid.py` and `hybrid.py` two agents between them,
across 15,687 lines that carry every selection the pipeline makes. That was too thin, and
it was the wrong shape of pass: both defects found there were invisible at the sort. This
pass traces sixteen decisions instead, one agent each — naming the quantity, finding the
line that computes it, following it through every return, dict key and cache round trip,
and stating what direction the comparison at the far end assumes.

Sixteen decisions traced. **None came back clean.** 98 findings raised, each handed to a
separate agent told to refute it: 13 confirmed, 7 left plausible, 75 refuted.

### 6.1 The budget split runs on a different rule depending on the metric

`_sanitize_baseline_scores` (`src/kirby/hybrid.py:897`) shifts every representation's score
up by the worst one, but only when the worst is below zero:
`shift = -min_val if min_val < 0 else 0.0`.

On an rmse dataset every score is a negated error, so every score is negative, the shift
always fires, and the worst representation lands exactly on zero. On a dataset scored by
spearman, auc or mcc every score is already positive, so nothing shifts and the worst
representation keeps a share proportional to its raw score.

Measured by calling the two functions directly, four representations under `linear`, which
is the default weighting:

| metric | scores in | features per 100, best to worst |
|---|---|---|
| rmse | −0.30, −0.32, −0.35, −0.40 | 43, 35, 22, **0** |
| spearman | 0.70, 0.68, 0.65, 0.60 | 27, 26, 25, **23** |

The worst representation gets no features on one metric and nearly a quarter of them on the
other, from the same allocation method on the same spread of scores. Every proportional
scheme inherits it: `linear`, `power_law`, `softmax`, `thresholded`, `score_gap`, `entropy`.

This is the same origin problem `FRAMEWORK_FINALIZATION.md` §3.5 is about, and §3.5's rule —
one normalisation per metric — is the answer to it. §3.5 is recorded there as answered and
not built.

### 6.2 The unscored-representation fix reaches two of the five weighting schemes

`allocate_performance_weighted` and `allocate_weighted_global` now put `float('nan')` in for
a representation nobody could score (`hybrid.py:1387` and `:1974`, fixed earlier today).
`_sanitize_baseline_scores` then converts every non-finite score back to `0.0` at
`hybrid.py:898`.

Its docstring says that assigns weight zero. That is true only where the weight is the score
itself. Measured on two scored representations at 0.70 and 0.68 and one unscored:

| scheme | features per 100 to the representation nobody could score |
|---|---|
| linear | 0 |
| min_floor | 0 |
| rank_linear | 17 |
| softmax | 20 |
| rank_exp | 24 |

So the fix holds for two of the five and not for the other three.

### 6.3 The rest, named

Confirmed, and each changes which configuration is chosen:

- **Hyperparameters on an auc dataset are chosen on hard 0/1 class labels.**
  `_scorer_for_metric_fn` (`find_best_hybrid.py:3722`) builds the scorer with no
  `response_method`, and sklearn 1.6.1 resolves that to `predict`. Every reported auc number
  instead goes through `_predict_for_metric`, which returns `predict_proba[:, 1]`. Two
  different quantities, and the search never sees the one that gets reported.
- **A tuned search and an untuned one share every result-database row.** `tune` is assigned
  onto the row at `find_best_hybrid.py:5230` but is in neither `IDENTITY_COLS` nor
  `CONFIG_COLS` nor `ALL_COLS` (`result_cache.py:46-94`), so `upsert_rows` discards it.
  Whichever ran first answers for both.
- **`augment_uncapped` scores one block of donor features and rebuilds another.** It scores
  the candidates in rank order (`hybrid.py:3836`) and records them grouped by donor
  (`:3881`); `apply_augmentation_selection` rebuilds them grouped. `augment_capped` closes
  exactly this gap with `_selection_for`/`_cand_block`; `augment_uncapped` has no equivalent.
- **The prediction term of a blended budget split is not an AUROC on a classification set.**
  `_score_component` (`hybrid.py:274`) receives `m.predict(X_val_scaled)`, which is hard
  class labels, so the term carries about half the signal of the AUROC it is named for.
- **`allocate_weighted_global` takes exactly k features per representation whatever they
  scored** (`hybrid.py:2050`), while `concatenate_features` stops at the last feature scoring
  above zero (`:4709`). The rule was added at one of the two selection points.
- **`ConfidentLearning` decides which molecules are mislabelled from a training residual.**
  All three ensemble members are fitted on `X_t` and then predict `X_t`
  (`utils/lnl_methods.py:677`).
- **`HeteroscedasticGP` is the only method in `lnl_methods.py` that does not z-normalise its
  labels** (`:2037`), so how much it downweights depends on the label's spread.
- **`find_best_hybrid_dta.py:933`** tunes on every row and then cross-validates those same
  rows, and tunes on r2 while the run is reported on concordance index.
- **`complexity_theory_experiments.py:717`** reads every `score_mean` as an R² and renames
  the field to `r2_mean`, on datasets the same file searched under rmse.

Plausible, reachability unproven:

- The per-fold factory-representation baselines are fitted untuned while the base-rep
  baselines are tuned, and both land in one `baselines` dict that one argmax runs over
  (`find_best_hybrid.py:5142`).
- Fold-0 pruning can seed one configuration from a tuned fit read off disk and fit its
  competitor untuned (`find_best_hybrid.py:6047`) — same root as the missing `tune` column.
- `_evaluate_hybrid_config` ranks the baselines with the blend-flipped flag
  (`find_best_hybrid.py:904`), the same shape as the fix at 7643 but on a path that fix did
  not reach.
- The frozen geometry branch of `_evaluate_baselines` is the only one of the three that does
  not call `_impute_nan` (`find_best_hybrid.py:4984`), so a representation carrying NaN drops
  out of the comparator pool silently.
- `_tune_on_training_fold` returns the first randomly sampled combination when every
  candidate scores NaN (`find_best_hybrid.py:3783`).
- The weighted arm of `MahalanobisOutlierDetector` flags the same molecules as the unweighted
  arm (`utils/noise_mitigation_methods.py:780`).


---

## 7. Everything fixed — 2026-09-10

You said fix everything. This is what was done, on top of §1's nineteen.

### 7.1 The budget-split origin rule — §3.5's Shape B, built

`src/kirby/hybrid.py`, the block above `_sanitize_baseline_scores`. One
normalisation per metric, from `BUDGET_SPLIT_NORMALIZATION_HANDOFF_2026_09_09.md`:
`spearman`, `pearson`, `mcc` and `r2` clip at zero; `auc` and `ci` become
2·(score − 0.5); `rmse` and `mae` become a skill score against predicting the
training mean or median; `f1_weighted` and `accuracy` against always predicting
the majority training class. A metric the rule does not cover keeps what ran
before, so nothing is silently reshaped.

Measured through the real allocator, five representations, budget 100, `linear`:

| metric | before | after |
|---|---|---|
| spearman 0.700 … 0.693 | 20 · 20 · 20 · 20 · 20 | 20 · 20 · 20 · 20 · 20 |
| −RMSE −0.300 … −0.307 | 44 · 31 · 19 · 6 · 0 | 20 · 20 · 20 · 20 · 20 |
| AUROC 0.90 0.70 0.55 0.52 0.50 | 28 · 22 · 17 · 16 · 16 | 60 · 30 · 7 · 3 · 0 |

Five near-identical representations now split evenly whatever the metric, and the
representation sitting at chance gets nothing instead of 16 features per 100.

**Two sub-answers were needed and I took the ones your own documents record.**
§3.7 (a): a negative spearman or mcc **clips to zero** — what §3.5's rule says and
what f32 measured. §3.7 (b): a null model with zero error **warns and falls back
to the old shift for that vector**, naming the metric, rather than raising, because
§3.6 is still open on whether a lost fold costs the configuration. Both are single
module-level constants, `ORIGIN_NEGATIVE_CORRELATION` and
`ORIGIN_NULL_MODEL_PERFECT`, so either flips in one line.

### 7.2 The other engine fixes

- `hybrid.py` — a representation nobody could score now weighs zero under **every**
  weighting scheme. The NaN in §1 was turned back into 0.0 by the sanitiser, and
  0.0 is weight zero only where the weight is the score itself. Measured: it used
  to take 17, 20 and 24 features per 100 under `rank_linear`, `softmax` and
  `rank_exp`. All five schemes now give it 0.
- `hybrid.py` — `allocate_weighted_global` stops at the last feature with a
  positive importance score, at both of its selection points, matching
  `concatenate_features`.
- `hybrid.py` — `augment_uncapped` now scores the donor block in the order
  `apply_augmentation_selection` rebuilds it. It scored pooled rank order and
  recorded donor-grouped order.
- `hybrid.py` — the prediction term of a blended budget split gets a real AUROC.
  It was reading hard class labels; measured on a synthetic set, 0.7340 on labels
  against 0.7767 on the ranking. The accuracy score every other caller reads is
  unchanged — verified identical.
- `find_best_hybrid.py` — hyperparameters on an `auc` dataset are tuned on the
  probability AUROC, matching what gets reported. Verified: the scorer returns
  0.7626, the probability AUROC, where it used to return 0.7828 off hard labels.
- `find_best_hybrid.py` — the per-fold label-fitted representations are tuned on
  the same terms as the base representations, so one argmax no longer runs over a
  tuned half and an untuned half.
- `find_best_hybrid.py` — the frozen geometry baseline imputes, like its two
  siblings. A representation carrying NaN used to drop out of the comparator pool
  silently.
- `find_best_hybrid.py` — `_tune_on_training_fold` keeps the untuned estimator and
  says so when every sampled combination scores NaN, instead of returning the
  first one sampled.
- `find_best_hybrid.py` — the baselines' own direction is threaded through
  `_run_search`, `_run_refinement_phase` and `_evaluate_hybrid_config` as
  `baselines_lower_is_better`, so the top-k subset prediction ranks them on their
  own scale under a blend.
- `find_best_hybrid.py` — `phase1_step` is `_DEFAULT_STEP_SIZE`, not the literal 10.

### 7.3 Under a blend, the comparator is now picked on the blend

`nested_cv_evaluate`. Both arms are scored on the blend, but the representation
the hybrid was measured against came from the inner search's baselines, which hold
the dataset's own metric. It is now picked from each representation's outer-fold
score under the run's own objective, by its mean over the **other** folds — the
same leave-one-fold-out device `_hr_lofo_score` uses, so fold f's own score never
enters fold f's choice and the pick is not an oracle. What the old rule would have
picked is kept beside it as `best_single_reps_per_fold_metric_pick`, and the run
says once that the rule changed.

### 7.4 The defaulted-direction reads: 22 down to 2

Routed through one resolver — metric name first, stored flag second, raise when
neither — `object_direction` in `tests/experiments/_shared.py`, mirrored as
`_direction_of` in `tests/analysis/blueprint_trace.py`.

`get_best_precomputed_search` was **deleted** rather than fixed. It had zero
callers anywhere in the repository, and it carried the one defaulted read that
genuinely inverted: when a search was `None` its sentinel was `-inf`, which on a
lower-is-better metric is the best possible value, so the missing condition won.

The two that remain are both correct: `find_best_hybrid.py:8571` is the fallback
arm of `apply_blueprint`'s own fix, reached only when the metric name is not a
builtin, where the stored flag does describe the quantity; and
`tests/experiments/framework/f02_adaptive_pruning.py:108` is a framework
experiment, which you put out of scope.

### 7.5 The rest

- `c10`'s robustness gate is off on every metric (`baseline_threshold=None`). The
  default of 0.3 was applied on one side of the direction only, so the rmse
  datasets admitted every representation while the spearman and auc ones dropped
  any representation under 0.3. **This one needs a re-run on the higher-is-better
  datasets** — the old gate has already removed representations there.
- `c17`'s `downstream_r2` is `downstream_score`. Nothing read the old name.
  `c31` and `c32` were left alone: their `r2_*` fields genuinely hold `r2_score`,
  and the dataset's own metric is in their `native_*` fields, correctly oriented.
- `c19` and `n13` store a sign-normalised sibling beside the raw delta.
- `complexity_theory_experiments.py` picks the best representation direction-aware,
  and a representation missing a score for a model is skipped rather than winning
  on a `-inf` default.
- `find_best_hybrid_dta.py` no longer tunes on every row before cross-validating
  those same rows. It passes `tune=True` into `find_best_hybrid`, which tunes per
  training fold — the path `tune_models`' own docstring calls the supported one.
- `lnl_methods.py` — `HeteroscedasticGP` divides its residuals by the training
  labels' spread, like every other method in the file.
- `noise_mitigation_methods.py` / `noise_mitigation_advanced.py` — the Mahalanobis
  weighted arm is dropped from the sweep and logged, because a per-feature weight
  cannot survive the standardise-then-re-estimate-covariance it goes through.

### 7.6 Three findings were already fixed in your tree

`tune` and `rep_pool` are in `IDENTITY_COLS` and `KEY_COLS` in `result_cache.py`,
as an uncommitted change dated 2026-09-10 that is not mine. That closes the
"a tuned and an untuned search share every row" finding and both of the fold-0
pruning findings that shared its root.

### 7.7 What this costs

Every change in §7.1 and §7.2 changes what `create_hybrid` builds, so the stored
searches are invalid. One wipe and one re-run covers all of them together. Your
memory note says fourteen `result_db` files go first, and the cache audit says
commit before wiping.


---

## 8. Pass four — the drivers and the reading scripts

Passes one to three swept `src/kirby` and `tests/experiments`. They never read
`tests/*.py` or `tests/analysis/*.py`: the CLI drivers that launch every search,
`precompute_hybrids.py`, the standalone `similarity_experiments.py`, the phase
analyses, the baseline surveys, and the scripts you read results with. That is 89
files carrying 1424 ordering sites.

Fifteen agents classified every one. **All 1424 accounted for, none unreached.**

| class | count |
|---|---|
| A — not a decision quantity | 758 |
| B — direction fixed and intrinsic | 426 |
| C — direction consulted correctly | 215 |
| D — direction not consulted | 25 |

163 further hand-rolled orderings were found beyond the catalogue. 32 findings went
to a verifier each: 5 confirmed, 8 plausible, 19 refuted.

### 8.1 The confirmed cluster is all in one file, and it is dispatched

`tests/complexity_theory_experiments.py`, run by
`tests/slurm_scripts/run_complexity_experiments.sh`. Three bare `max()` calls over
a score that holds the dataset's own metric — rmse on logd, caco2, herg_ki and
qm9, every dataset `main()` ships:

- **Experiment 2's heatmap.** Each `heatmap[model][budget]` cell named the
  configuration with the highest rmse, i.e. the worst at that model and budget.
  `_lower` was already in scope eight lines above, from the baselines block fixed
  earlier today.
- **Experiment 6's per-budget pick.** The allocation and importance method of the
  worst row at each budget are what `evaluate_cv_with_train` is then re-run with,
  so the stored train-versus-held-out gap described the worst recipe. Experiment
  9d reads that heatmap for its bias-variance analysis.
- **Experiment 8's "optimal configuration".** It takes the best model out of
  Experiment 2's results and uses that recipe for everything downstream. On an rmse
  dataset it picked the worst model. It now reads the direction out of
  `exp2_results['baselines']`, which has carried it since this morning, and raises
  rather than guessing if it is absent.

Two more in the same file:

- **The Experiment 2 headline delta** was hardcoded hybrid-minus-baseline and
  printed as `R²=`. Sign-normalised, and it names the metric now.
- **Experiment 5's two arms were in different metrics.** The hybrid arm is
  `find_best_hybrid`'s `score_mean`, in the dataset's own metric. The
  single-representation arm came from a local `evaluate_single_rep_cv` that scored
  with `r2_score` whatever the dataset was. So `delta = h - s` subtracted an R²
  from an RMSE — not a sign error, two different quantities. Both arms now use the
  dataset's metric, the best-of is direction-aware, and the delta is
  sign-normalised.
- **The file carried its own byte-identical copy of `fit_learning_curve`**, with
  the same `[0, 1]` asymptote defect fixed in `_shared.py` this morning. The copy
  is deleted; it delegates to the one definition.

### 8.2 The reading scripts

- `tests/quick_summary.py` hardcoded `lower=False` in four places under a comment
  saying higher-is-better was "true for everything in this run". It is not true for
  logd, caco2, herg_ki or buchwald_hartwig. Every best-of picked the worst row and
  every delta printed backwards, which then fed the win count and the per-model and
  per-domain pools. It reads the registry per dataset now, and the delta is
  sign-normalised where it is built so every reader below is right without
  branching.
- `tests/supervisor_presentation.py` computed `delta = hybrid_score - best_val` with
  a fixed sign while `higher` was in scope two lines above, and then counted
  `delta > 0` as datasets where the hybrid won.
- `tests/analysis/quick_hybrid_summary.py` treated an importable-but-empty registry
  as an empty world and locked in a hardcoded name list that covers only the rmse
  molecular sets. It keeps trying the other import paths, says so on stderr if it
  ends up with nothing, and raises for a dataset the loaded registry does not know
  rather than guessing from a name.

### 8.3 Checked and found correct

`tests/similarity_experiments.py` ranks on `r2_mean`, and that field really does
hold `r2_score` on every path that fills it — the name is accurate and `max()` is
right. `tests/analysis/blueprint_trace.py` and `tests/quick_summary.py`'s
`best_score` both take the direction as an argument and apply it correctly.
