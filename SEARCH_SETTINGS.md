# Search settings, and how to make a search cheaper

What `find_best_hybrid` varies, and the three supported ways to run it faster.
Written 2026-10-26. Every line number below was read in the file on that date.

This exists because a local check was run with one entry per axis — one
allocation method, one importance method, one internal evaluator, one
representation count — and reported as a hybrid search. It was not a search. It
was a single recipe built at three widths.

---

## The six axes

These are the six the decided grid names (`tests/experiments/decided_grid.py:25`).

| axis | default | where |
|---|---|---|
| downstream models | `['rf', 'xgboost', 'svm', 'dnn', 'lightgbm']` | `tests/experiments/_shared.py:570` |
| budgets (features) | `[10, 25, 50, 100, 250, 500, 1000, 2000]` | `tests/experiments/_shared.py:495` |
| importance methods | `['deeplift', 'lightgbm_gain', 'treeshap']` | `src/kirby/find_best_hybrid.py:3710` |
| allocation methods | `['performance_weighted', 'power_2', 'softmax_T1', 'rank_exp_d0.3', 'wglobal_power2']` | `src/kirby/find_best_hybrid.py:3719` |
| internal evaluators | `['match', 'rf', 'lightgbm', 'dnn']` | `src/kirby/find_best_hybrid.py:3740` |
| representation counts | `[2, 3, 5, None]` | `src/kirby/find_best_hybrid.py:3105` |

Recipes per fold on the defaults, before budgets: 3 × 5 × 4 × 4 = 240.

Notes on three of them.

**Representation counts.** How many of the representations in the pool one
recipe may use at all. `None` is the whole pool. The whole grid runs once per
entry, each time restricted to the N representations that fold's allocation
scores rank highest. A searched axis since 2026-09-12. It replaced the retired
`top_2`/`top_3`/`top_5` allocation names, which welded a count into the
weighting. Passing `rep_counts=[None]` means every recipe spends the budget
over every representation and a recipe using 2 or 3 of them is never tried.

**Allocation methods.** The five defaults are one weighting-by-mechanism pair
each, cut from the c32 cross sweep on 2026-09-01. The registry behind them,
`NAMED_ALLOCATIONS` (`src/kirby/hybrid.py:517`), holds many more: softmax at
five temperatures, power 2 and 3, thresholded, min_floor, top_k, rank_linear,
rank_exp at three decays, score_gap at four, entropy at three, pairwise-distance
redundancy at three plus three score-based, and three weighted-global variants.
`EXTRA_ALLOCATION_METHODS = ['weighted_global']` is opt-in.

⛔ The whole greedy family is cut and must not be used — `greedy`,
`greedy_forced_s*`, `greedy_fill_s*`, `dominant_greedy`, `dom_greedy_f*`. Closed
2026-04-09, re-confirmed 2026-08-25 and 2026-08-26. The names still resolve so
old configs do not crash. Nothing selects them. See the block above
`NAMED_ALLOCATIONS`.

**Importance methods.** Eight are retained in the engine
(`src/kirby/hybrid.py:9-12`): treeshap, xgboost_gain, lightgbm_gain and deeplift
are the fast ones; permutation and kernelshap are slower; integrated_gradients
and gradientshap are for a long search only. Three are in the default list.

Allocation methods that are insensitive to the importance method are evaluated
once per budget and model rather than once per method
(`src/kirby/hybrid.py:15-22`).

---

## The budget is a maximum, not a target

Stated in `src/kirby/hybrid.py:24-29` and again at
`src/kirby/find_best_hybrid.py:8242`. A hybrid that scores better on fewer
features is the better answer and comes back narrower. It is compared against a
base representation cut to the same budget, not that representation at its full
native width.

The budget is not an axis inside the search. It is explored by running the
search at a range of budgets and reading the range
(`src/kirby/find_best_hybrid.py:11319`). So report one row per budget. Ranking
the budgets against each other and keeping the winner collapses the axis.

`budget=None` ranks over every budget the search ran
(`tests/experiments/_shared.py:1718`). Pass a budget to get that budget's best.

---

## Three ways to run it faster

### 1. Adaptive pruning

Off by default since 2026-09-13, owner's decision
(`src/kirby/find_best_hybrid.py:8315`). Pass it explicitly.

It does not shrink the grid up front. It starts the full grid and deletes
members as evidence arrives, two ways.

**Axis votes** (`DEFAULT_PRUNING_SCHEDULE`,
`src/kirby/find_best_hybrid.py:3824`). After `vote_folds: 2` folds, members of
one axis are grouped by every other axis — so they compete only against each
other for the same model at the same budget — ranked on the mean of those two
folds, and a member placing in the best `top_n` in fewer than `threshold` of the
groups is deleted.

| axis | rule | what it deleted on the f33 record |
|---|---|---|
| representation count | best-of-1 at 10% | N=5 on logd, N=2 on tcga_brca_pam50 |
| importance method | best-of-2 at 20% | deeplift on flab_thermo_tresanco and herg_ki, treeshap on logd |
| internal evaluator | best-of-3 at 10% | svm as an evaluator on logd |
| allocation method | best-of-2 at 20% (`final_strategy_threshold`) | softmax_T1 on logd |

Each rule was calibrated on the f33 record — five datasets, 25 model-by-budget
conditions each — against the standard that every condition's best five-fold
score is unchanged. The comments in the file record the damage at each
neighbouring setting. Two things from them worth carrying:

- After one fold alone, no setting is safe on any axis. `vote_folds: 2` is why.
- The internal-evaluator rule is the narrowest. Only two settings in the whole
  sweep cut without damage, and one cut-off above either, flab_thermo_tresanco
  loses conditions at 0.0219 to 0.0269 spearman.

**Mid-flight drops.** At `checkpoints: [100]`, the bottom
`tail_drop_per_checkpoint: 0.5` of allocation-and-importance pairs are dropped.
This prunes configurations inside a condition and never removes a whole axis
member.

Two limits:

- `strategy_top_n` is 2 of 5 allocation methods. With three or fewer allocation
  methods searched, every method places every time and nothing is dropped
  (`src/kirby/find_best_hybrid.py:6295`). So pruning is a no-op on a fast
  search of three methods and live on a full search of five.
- `min_configs_per_model: 1` is the floor, so `best_by_model[m]` always exists.

Per-fold searches run unpruned by default (Adelaide, 2026-10-05, handoff 2.1).
`search_on_fold` prints a line and ignores a schedule passed to it
(`tests/experiments/_shared.py:1419-1432`).

### 2. Fast tier

`speed='fast'`. Valid values are `'fast'` and `'full'`; anything else raises
(`src/kirby/find_best_hybrid.py:8703`).

| axis | full | fast |
|---|---|---|
| importance methods | 3 | 1 (`lightgbm_gain`) |
| allocation methods | 5 | 3 (`performance_weighted`, `power_2`, `rank_exp_d0.3`) |
| representation counts | 4 (`[2,3,5,None]`) | 2 (`[3, None]`) |
| downstream models | 5 | the 3 with `tier == 'fast'`: rf, lightgbm, dnn |
| internal evaluators | 4 | the same 4 |

Recipes per fold before budgets: 72 against 240.

⛔ Every one of those lists is in the search cache key (`_make_search_meta`), so
a fast run files separately from a full one. It cannot be topped up into a full
result later. It is a different search, not a subset of one. The ⛔ block saying
so is at `src/kirby/find_best_hybrid.py:3094`.

A list passed explicitly wins over `speed`
(`src/kirby/find_best_hybrid.py:8694-8713`), so `speed='fast'` with
`allocation_methods=[...]` uses the list, not the fast three.

### 3. Hand-coded smaller grid

Pass the axis lists yourself. Nothing checks that what you passed is a search.
One entry per axis gives one recipe, and the result reports it as a winner with
no warning. If you do this, say in the write-up which axes you collapsed and to
what — the numbers are about one recipe, not about the search.

---

## Things that are not axes but change the recipe

- `objective`: `'predictive'` (default) or `'similarity'`
  (`src/kirby/find_best_hybrid.py:1844`). Under `'similarity'` no downstream
  model is fit; each row is scored from its nearest neighbours on the held-out
  fold, and the search finds a different hybrid.
- `augmentation_strategy`: `'none'`, `'all'`, `'greedy_ablation'`, `'capped'`,
  `'uncapped'`, with `candidate_ranking` choosing how a candidate feature is
  ranked. Six rankings in `CANDIDATE_RANKINGS` (`src/kirby/hybrid.py:3535`):
  four against the labels (`rf_importance`, `lightgbm_gain`, `mutual_info`,
  `deeplift`) and two against what the base representation cannot explain
  (`residual_correlation`, `residual_deeplift`).
- `two_phase` with `refinement_top_n` and `refinement_step_sizes`, `step_size`,
  `tune`.

---

## Local runs

- A local run is a debug run. "Does it run" means normal settings, time-boxed,
  with the time recorded and anything slow named. Not a result.
- `DEFAULT_N_FOLDS` is 5 (`src/kirby/find_best_hybrid.py:3741`).
- Set `KIRBY_LGBM_THREADS=1` on macOS. LightGBM and torch link different OpenMP
  runtimes, and a multi-threaded LightGBM fit after torch has started its own
  thread pool segfaults the process with no traceback. The default importance
  list includes `deeplift`, which runs torch, so every full search hits this.
  See `lgbm_n_jobs` (`src/kirby/hybrid.py:84-120`).
- A per-fold search needs a `DECISIONS.md` row for the dataset, or
  `decided_grid.UndecidedSearch` is raised — deliberately, with no fall back to
  the engine defaults. For a dataset with no row, `search_on_fold` takes
  `fixed_grid=` (`src/kirby/find_best_hybrid.py` callers; the escape hatch is
  named at `tests/experiments/decided_grid.py:144`). `fold_searches` and
  `fold_standard_selection` do not forward it.

## Honest comparison

A search run once over all folds has seen the held-out rows when it chose its
recipe. The per-fold path is `search_on_fold` → `top_base_reps` /
`top_hybrids` → `build_fold_hybrid`, one search per outer fold on that fold's
training rows only. `recipe_search` raises unless it is `'per_fold'`
(`tests/experiments/complexity/c11_ensemble_vs_hybrid.py:458-461`); the
whole-dataset recipe search was removed 2026-10-05.

Use `kirby.find_best_hybrid.nested_cv_outer_folds` to build the folds, so the
search runs on exactly the training subsets the experiment later asks for.

Scaling on both sides is `FeatureScaler` (`src/kirby/scaling.py`): a feature
whose training values are all 0 or 1 passes through unscaled, every other
feature is standardised. Handoff 2.3 item 17, 2026-10-05.
