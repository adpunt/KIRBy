# Handoff: re-measure the adaptive pruning rules on the current search space

Written 2026-09-02. The task is to re-establish which adaptive-pruning settings are safe,
because the search space they were fitted to no longer exists.

## Read these before doing anything

The numbers in this document are from 2026-09-02 and the search space has been changing.
Re-read all of these and work from what they say now, not from what is quoted here:

- `src/kirby/find_best_hybrid.py` — `DEFAULT_ALLOCATION_METHODS`, `DEFAULT_IMPORTANCE_METHODS`,
  `DEFAULT_INTERNAL_EVALUATORS`, `DEFAULT_N_FOLDS`, `DEFAULT_PRUNING_SCHEDULE`,
  `MODEL_MAX_BUDGET`, `EXTRA_ALLOCATION_METHODS`.
- `src/kirby/hybrid.py` — `NAMED_ALLOCATIONS`, and the ⛔ block above it recording that the
  whole greedy family is cut.
- `tests/experiments/_shared.py` — `PRUNING_MODES` and the calibration comment above it.
- `tests/analysis/f15_pruning_analysis.py` — the analysis that produced the shipped rule.
- `tests/experiments/framework/f15_adaptive_pruning_validation.py` and
  `f16_multistage_pruning.py` — the two experiments being superseded.

## What the shipped pruning rules are

Two prunes run, both inside `_run_search` in `find_best_hybrid.py`.

**End-of-fold-0 strategy prune.** After fold 0 every configuration has a score. They are
grouped, one group per (importance method, evaluator, model, budget). Inside a group the
allocation methods are ranked and the best `strategy_top_n` each get a point. An allocation
method is dropped from the remaining folds when it scored a point in fewer than
`final_strategy_threshold` of the groups. Defaults: `strategy_top_n=3`,
`final_strategy_threshold=0.15`.

**Mid-flight tail drop.** With `checkpoints` and `tail_drop_per_checkpoint` both set, fold 0
stops when it crosses out of a budget that exactly matches a checkpoint value, and drops the
worst fraction of (allocation, importance) pairs per (model, evaluator) group.
`min_configs_per_model` is the floor. Off by default.

Nothing prunes internal evaluators, models or budgets — those are group keys in both rules.
That exclusion was itself a measured decision; see the last section.

## Why the existing calibration cannot be reused

**The allocation set was replaced.** F15 recorded 13 allocation methods:
entropy_1.0, entropy_2.0, greedy_fill_s5, min_floor, performance_weighted, power_2,
rank_exp_d0.5, rank_linear, score_gap_0.2, thresh_0.3, top_2, top_3, top_5.

`DEFAULT_ALLOCATION_METHODS` is now five: performance_weighted, power_2, softmax_T1,
rank_exp_d0.3, wglobal_power2 (cut to five on 2026-09-01 from the c32 cross sweep).

Only **performance_weighted and power_2** appear in both. softmax_T1, rank_exp_d0.3 and
wglobal_power2 are absent from every F15 result file, so no re-analysis of the F15 data can
say anything about them. wglobal_power2 matters most here: it is the only global feature-pooling
arm in the default set, it allocates by a different mechanism from the other four, and it has
never been through any pruning validation.

The F15 set also contains greedy_fill_s5, which is a cut arm.

**The threshold is a fraction of a count that changed.** Both rules count places among N
allocation methods. At N=13 a method that places in 15% of groups is doing something; at N=5,
with 3 places available per group, the arithmetic is entirely different. At N=3 with
`strategy_top_n=3` nothing can ever be dropped, because every method places every time.

**F16's calibration is one dataset.** `tests/results/framework/f16_multistage_pruning/calibration_summary.json`
has `datasets_evaluated: ["caco2"]`. Its recommendation of the `very_aggressive` schedule rests
on that single dataset, and that schedule's own row shows top-3 overlap dropping to 0.67 and a
per-model change of 2.1% against a stated acceptance limit of 2%.

**The representations moved too.** F15 ran on April rep pools, before the pool prune to ten per
dataset and before the speed/tier removal. Group counts and therefore both thresholds depend on
how many representations are in play.

**F15 was not nested CV.** The script calls no `nested_cv_evaluate`; it is a flat 5-fold search.
Every pruning number in circulation was measured on the un-nested evaluator.

## An open discrepancy to settle

`PAPER_DRAFT.md:217` states that the pruning is "F15-validated: pruned strategies had zero
best-wins across all test datasets", and that capped-augmentation and standalone searches use a
more aggressive 25% threshold "without affecting the quality of the surviving strategies".

Running `python tests/analysis/f15_pruning_analysis.py` on 2026-09-02 over the result files in
`tests/results/framework/f15_adaptive_pruning/` produced the opposite: at top-3 in <15% the
winning strategy is lost on some folds on five of six datasets, and at 25% on all six. On
buchwald_hartwig the mean regret at 15% is 0.531 RMSE.

Establish which is right before the draft goes out. The likely explanations are that the April
analysis used a different rule definition, or a different subset of the result files, or that
the files on disk changed after the sentence was written. Do not assume the script is correct
just because it runs today.

## What to produce

A new framework experiment. Next free number is f29 — check the directory first, more may have
landed. Editing f00–f15 is banned; f15 and f16 stay on disk as the record of what was measured
in April.

Answer these:

1. For the allocation set as it stands, what does each combination of `strategy_top_n`
   (1, 2, 3, 4) and `final_strategy_threshold` (0.05 through 0.5) prune, and does the true
   winner survive? Report per dataset, per model, per fold — never a mean across datasets.
2. Same question for `checkpoints` × `tail_drop_per_checkpoint`, with checkpoint values taken
   from the budget list actually being run. A checkpoint that is not exactly one of the budgets
   never fires (`find_best_hybrid.py`, the `cur_b in _all_checkpoints` test).
3. What does each setting cost and save in wall time, measured, not derived from config counts.
4. Whether `PRUNING_MODES` in `tests/experiments/_shared.py` should change, and whether
   `DEFAULT_PRUNING_SCHEDULE` should change. Both are user decisions — bring the numbers, do not
   edit the defaults.
5. The paper discrepancy above, settled with a stated cause.

Report per-cell win counts, not mean ranks.

## Constraints

- **Do not cut the search space for compute.** Every allocation method, importance method,
  evaluator, budget, model and dataset in the current defaults stays in. A cut needs a stated
  reason: degenerate (assert it), broken (fix it), or undefined. A theory about why something
  will lose is not a reason. Split the work by dataset, never by axis.
- **Run on real data on day one.** Scope the first real run to minutes, not a whole grid. Reading
  and synthetic data have repeatedly missed defects that one real run found.
- Anything estimated over three hours locally becomes a server job. State the wall estimate before
  starting.
- The greedy family is cut and stays cut. Do not add any greedy name back to a roster.
- Do not write a verdict of your own into any repo document. Later it is indistinguishable from a
  measured result.

## Known blockers to check before running

`PIPELINE_READINESS_TODO.md:1523` lists 8 bugs in the F15 script: missing `_impute_nan` in the
allocation cache, missing `MODEL_MAX_BUDGET` caps, a bare except, non-deterministic `set()`,
wrong `n_folds`, an unnecessary import, no matched flag, and an unverified `_resolve_metric`
import. Check which are still present before reusing any of that scaffolding.

Locally, LightGBM segfaults after the RDKit loader has imported unless `KIRBY_LGBM_THREADS=1`
is set; the cause is deeplift plus LightGBM `n_jobs=-1` pulling in two OpenMP runtimes.

## Why internal evaluators are not pruned, for context

This was decided, not overlooked. `PIPELINE_READINESS_TODO.md:1520` records the F12 preliminary
finding: "Strategies are the only prunable dimension — evaluators and importance methods always
produce winners." The same sentence heads `tests/analysis/f15_pruning_analysis.py`.

The supporting evidence: F07 found no evaluator dominating across domains; F11 found the
evaluator condition irrelevant, max delta 0.017 across 5 antibody datasets; a local measurement on
2026-07-16 put the evaluator spread at ~0.003 against 0.009–0.018 for allocation.

Two gaps. The F12 table behind that sentence was not found in the repo — the claim survives only
as prose in two places, both tracing to the same analysis. And the same doc warns the F12 rules
were fitted on 24 groups per dataset against ~1,170 in the full precompute, so they "need
re-validation" — which never happened for evaluators.

The user refused collapsing the evaluator axis on 2026-08-27 and that decision stands. This
handoff does not reopen it. If the new experiment produces evidence on evaluator pruning, report
it as a finding; do not act on it.
