# f30 local run — what completed, what was cancelled, and why

**Date:** 2026-09-05 · **Dataset:** caco2 · **Branch:** `similarity-metrics-study`
**Settings:** 3 outer folds, 2 inner folds, internal evaluator `rf`, downstream
`lightgbm` + `rf`, allocations `power_2` + `rank_exp_d0.3`, budget 100, pruned 10
representations. `xgboost_gain` dropped by user decision 2026-09-03.

**28 of 31 methods have results. 3 were cancelled on measured cost.** None was
cancelled because it looked unlikely to win.

---

## Cancelled, with the measurement that justified it

| method | how far it got | measured cost | why |
|---|---|---|---|
| `conditional_permutation` | fold 0 of 3, in 3h31m | ~10.5 h projected for one method | Implementation fits a ridge regression of each feature on a random subset of the others — 4,800 ridges per representation per fold, 30 of those per method. |
| `boruta_hits` | did not finish fold 0 in 9h50m | > 30 h projected | Implementation runs 40 random-forest fits on a **doubled-width** matrix per representation per fold. On `grover` that is a 9,600-feature forest, forty times, thirty times over. |
| `kernelshap` | not started | 96–118 s on **60** features (measured on synthetic data 2026-09-03) | Cost is ~`(2d + 2048)` coalitions per explained sample. At caco2's 4,800-wide representations that is of order an hour per representation per fold, ~150 h for the method. |

**The reason in every case is my implementation's cost on one machine, not the
method being unusable.** All three are affordable as their own cluster tasks at 16
cores with a 12-hour wall, which is exactly the shape
`tests/slurm_scripts/run_f30_importance_methods.sh` submits. They are ROUTED, not
dropped: they stay in `f30_scorers.SCORERS` and in the SLURM arm list.

### If they are wanted locally later, these are the levers
- `conditional_permutation`: `n_cond` (currently 64 conditioning features) and
  `n_repeats` (3). The ridge conditioner could also be fit once per representation
  rather than once per feature.
- `boruta_hits`: `max_iter` (currently 40). Boruta's own convergence check would
  stop most features far earlier than 40 rounds.
- `kernelshap`: `nsamples` and `max_eval_samples` are already kwargs on
  `compute_feature_importance`; production leaves them at `'auto'` / 100.

---

## What the local run cost in total

The three cancelled methods consumed roughly **23 hours of wall clock between them
and produced nothing**. The 25 methods that completed took about 12.5 hours
together, and the cheapest 13 of those took under 20 minutes combined.

This is the clearest argument in the run for putting f30 on the cluster: the cost is
concentrated in a handful of methods, and on a cluster each one is its own task and
cannot block the others. Locally they run in sequence and one bad method stalls
everything behind it.

---

## Ordering lesson for the next dataset

Arms were run cheapest-first, which was right — it meant 25 results existed before
the expensive tail stalled. Two things to change:

1. **Put a per-method wall clock in the driver.** A method that exceeds it should
   raise, record `timed_out`, and let the run continue. Right now a slow method
   blocks every method behind it, which is how `deeplift` — a production default —
   ended up unmeasured after 23 hours of waiting.
2. **Do not split one dataset across several invocations.** Each one re-loads the
   data (~15 min on caco2, rdkit scaffolds) and, before it was cached, re-ran the
   engine-equivalence check (~1 h). Both are now avoided: the engine check caches to
   `<ds>__engine_check.json`, and the arms run in a single invocation.
