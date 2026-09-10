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
