# Experiment fixes surfaced by the analysis (2026-07-07)

These are changes to the **experiments themselves** that came up while doing the
first-pass "does the hybrid beat the best existing model+rep" analysis. Each one
has been checked against the current code so we know whether it's still needed
before the big re-run.

Scope note: this list is meant to grow *from what the analysis turns up*, not from
old planning docs. (I briefly pulled candidate items from `REPRODUCIBILITY_LEDGER.md`
etc. — that was the wrong source and I've dropped it; those are a separate track.)

---

## 1. "Which direction is better" not recorded — ALREADY FIXED, no action

The problem: results files didn't record whether a higher or lower score is better,
so the analysis had to guess it from a registry (the source of the recurring
sign-flip on RMSE datasets).

Checked the code: `save_experiment_results` already stamps both `lower_is_better`
and `metric_name` on every file, looked up from the dataset registry
(`tests/experiments/_shared.py:3795-3832`). The old files on disk lack it only
because they were written before that fix. **Your re-run will include it.**

One thing to confirm, not fix: the stamp is skipped silently if the dataset isn't
in the registry (`registry_lookup` throws → nothing stamped). Worth a one-line
check that every dataset you care about is registered, or that file quietly loses
its direction again.

---

## 2. A failed model gets written as a real "-inf best score" — STILL A BUG

The problem: in the nucleotide datasets, some models (e.g. xgboost) showed up with
`best_rep = None` and `score = -inf`, which polluted the "top existing combos"
ranking.

Checked the code (`c02_complexity_grid.py:290-301`): when a model has no finite
score on *any* representation, `best_single_score` is left at its `-inf` starting
value and `best_single_rep` at `None`, and that gets written to the file verbatim.

Two parts:
- **Fix (small, in the experiment):** when the best score is non-finite, store
  `None`/skip it and record *why* (all reps failed), instead of writing `-inf` as
  if it were a real result.
- **Investigate (needs server data):** find out *why* those models produce no valid
  score on the nucleotide sets — is it a genuine model failure, or all-NaN reps?
  The code already notes GP/KNN go all-NaN at high dimensions
  (`c02_complexity_grid.py:322-324`), so this may be the same class of problem.
  Can't tell locally — needs the per-model baseline numbers from the server.

The analysis reader already filters these out, so they won't corrupt tomorrow's
tables — but the underlying experiment still writes bad data.

---

## 3. Can't show "hybrid wins with fewer features" — STILL MISSING, small add

The problem: to say "the hybrid beats the single representation while using fewer
features," we need each representation's native width in the file. It isn't there,
so that column comes up blank.

Checked the code: the `run()` return dict (`c02_complexity_grid.py:551-562`) doesn't
include representation widths, and nothing downstream adds them. But the widths are
known during the run (the per-model logic already uses each rep's dimension). So
this is a small, safe add — it only affects what's written, not what's computed, so
it reuses all the precomputed work.

---

## 4. Can't compute proper error bars on hybrid-vs-baseline — STILL MISSING, bigger change

The problem: to say a hybrid's win is statistically real, we'd compare the two
across the 5 cross-validation folds. The hybrid's per-fold scores are saved, but the
baseline's are not — so we can only do the weaker "is the baseline outside the
hybrid's error bar" check, not a proper paired test.

Checked the code (`c02_complexity_grid.py:285-300`): the baseline is already reduced
to a single averaged number *before* the experiment writes it — the per-fold values
are gone by then, inside the search library. So this isn't a one-line add to the
experiment; it needs the search library to expose per-fold baseline scores.

For tomorrow the weaker check is fine (and it's what the reader does). Flag this as a
later change, not a pre-re-run must-do.

---

## 5. Allocation set cut to five — DONE (2026-09-01)

The problem: the default allocation set carried nine methods, several of which
never placed near the top and one (`fixed`, from f01) that only looked good
because it was silently given many more features than its rivals. The c32 cross
sweep scored each allocation as a full **weighting × mechanism** pair (the
number-of-reps knob held as a separate axis, not baked into the allocation) over
9 datasets × 3 models × 2 budgets. Counting how often each pair landed in the
top three of its cell, five stood clear and the rest were consistently lower.

The change (`src/kirby/find_best_hybrid.py`, `DEFAULT_ALLOCATION_METHODS`):

**Kept (the five):**

| allocation (weighting \| mechanism) | code name |
|---|---|
| linear \| quota | `performance_weighted` |
| power_law² \| quota | `power_2` |
| softmax T=1 \| quota | `softmax_T1` |
| rank_exp d=0.3 \| quota | `rank_exp_d0.3` |
| power_law² \| global | `wglobal_power2` |

**Dropped from the default set:** `top_2`, `top_3`, `top_5` (these welded a
rep-count into the weighting slot — the number-of-reps knob is now a separate
axis), `entropy_1.0`, `entropy_2.0`, `rank_linear`, and `rank_exp_d0.5` (the
chosen rank_exp decay is 0.3, not 0.5). None are deleted from the engine; they
remain reachable by name for old configs.

`wglobal_power2` is the one global-pooling arm promoted into the default set;
the other `weighted_global` variants stay opt-in in `EXTRA_ALLOCATION_METHODS`.
Note it bypasses the per-fold persistent cache by design, so its global arm
recomputes each fold (documented at the `EXTRA_ALLOCATION_METHODS` comment).

Evidence and per-N tables: the c32 result files
(`c32_allocation_cross/*.json`, qm9 excluded) and memory
`project_c32_allocation_cross_2026_08_23`.

---

## Summary

| # | Change | Verdict |
|---|--------|---------|
| 1 | Record metric direction | **Already fixed** — re-run will have it |
| 2 | Don't write `-inf` for failed models | **Fix before re-run** (small) + investigate cause (needs server data) |
| 3 | Record representation widths | **Worth adding** before re-run (small, reuses precompute) |
| 4 | Save per-fold baseline scores for real error bars | **Defer** — bigger change, not needed for tomorrow |
| 5 | Allocation set cut to five | **Done** (2026-09-01) — `DEFAULT_ALLOCATION_METHODS` now holds the c32-chosen five |
