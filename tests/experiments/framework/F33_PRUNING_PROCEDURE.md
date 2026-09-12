# How the pruning settings were established, and how to establish them again

Written 2026-09-12. This is the procedure, not the result. It exists because the
analysis keeps getting shortened: a sweep gets collapsed to the shipped value, a
per-dataset table gets averaged, a fold gets skipped. Run all five steps. Do not
substitute a subset.

What was done the first time: f02 simulated pruning on stored results; a
preliminary read on f12's data, 6 datasets at 24 groups each, said allocation
methods were the only prunable dimension; f15 then ran the full grid unpruned on
6 datasets and stored every fold; `f15_pruning_analysis.py` replayed rule
families off that record; f16 swept the mid-flight schedule on caco2 alone and
took the most aggressive setting that stayed within 2% of unpruned. Steps 1 to 5
below are that, minus the superseded simulations, plus the fold count as a swept
setting rather than a fixed 1.

## Step 1 — record the grid, unpruned. The only step that costs compute.

One run per dataset, `adaptive_pruning=False`, `two_phase=False`, no augmentation.

Grid: 5 allocation methods x 3 importance methods x 5 budgets (25, 50, 100, 150,
300) x 17 model-and-evaluator pairs = 1,275 configurations. One configuration is
one allocation method, one importance method, one budget, one model, one internal
evaluator. The 17 pairs come from 5 models and the evaluator list `match, rf,
lightgbm, dnn`, where `match` resolves to the model's own type and duplicates
collapse. Budgets 25 and 100 are not in the default search list; they are there
because a mid-flight checkpoint fires only on an exact budget match, so without
them two thirds of the shipped schedule cannot be tested.

5 folds, seed 42, the dataset's own grouping. Score in the dataset's registry
metric: rmse, spearman, auc, mcc or f1_weighted.

Store per configuration: the list of 5 fold scores, their mean, and the seconds
the configuration took across all 5 folds.

Nothing after this step refits anything.

## Step 2 — establish what an unpruned search returns.

For each of the 25 combinations of a model at a budget, the winning allocation
method is the one whose mean of 5 fold scores is best — lowest for rmse, highest
for spearman, auc, mcc, f1_weighted. Those 25 are what every rule below has to
keep reachable. Record which method wins how many of the 25, per dataset.

## Step 3 — sweep the end-of-fold-0 allocation vote.

Group every configuration by its importance method, internal evaluator, model and
budget: 3 x 17 x 5 = 255 groups, each holding the 5 allocation methods.

In each group, rank the 5 methods on their score **on the voting fold** and give
one point to the best N. A non-finite score sorts to last, so a method whose
configurations failed loses those places. Sum each method's points over the 255
groups. Delete a method whose count is below `threshold x 255`.

Sweep all of:
- N in 1, 2, 3, 4. Shipped value 3.
- threshold in 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50. Shipped value 0.15,
  which at 255 groups deletes a method below 38.2 groups.
- voting fold = fold 0; then fold 1 alone, fold 2 alone, fold 3 alone, fold 4
  alone. The four single-fold repeats are the test of whether fold 0's verdict is
  a property of the data or of that fold.
- two folds instead of one, both forms: rank on the mean of folds 0 and 1 with the
  threshold applied once; and rank folds 0 and 1 separately, add the points, and
  double the threshold.

Report per setting, per dataset, never averaged across datasets:
- which allocation methods it deletes, by name;
- how many of the 25 winners from step 2 are no longer reachable;
- for each one lost, the winner's 5-fold mean and the best surviving method's
  5-fold mean, both in that dataset's metric, then the difference;
- seconds saved: the seconds of every deleted configuration, times
  (5 - number of voting folds) / 5.

## Step 4 — sweep the mid-flight drop, per model.

A checkpoint fires only if it is exactly one of the budgets run. At checkpoint
budget b, inside each model-and-evaluator group, take every configuration at
budget b or lower, rank on fold-0 score, keep the best `round(n x (1 - f))`. The
rest are dropped as allocation-and-importance pairs and never run at any higher
budget in that group. Then run the step-3 vote on what survives.

Sweep checkpoint lists [25], [50], [100], [25,50], [25,50,100], [10,25,50] against
f in 0.0 to 0.8 in steps of 0.1. Shipped: [25,50,100] at 0.5. Include f = 0.0: at
zero no drop runs, so that row isolates what the vote alone costs.

Report per model, per checkpoint list, per f, per dataset:
- relative loss: the model's best 5-fold mean over survivors against its best over
  everything, as a share of the unpruned number, signed so positive is worse;
- share of fold evaluations not run;
- a trivial marker when the model's unpruned best sat at a budget at or below the
  last checkpoint — the drop had no chance to delete it, so that cell proves
  nothing;
- how many of the unpruned grid's best 3 and best 10 configurations survive.

A cell passes at relative loss 2% or below and not trivial.

Also run each schedule applied to one model only, not all of them: svm alone, dnn
alone, both. That is how a per-model setting gets evidence.

## Step 5 — read the answer off the sweep.

Per model and checkpoint list, the largest f that passes on every non-trivial
dataset. Then the (checkpoint list, f) combinations where every model passes on
every dataset, ordered by what they save. Tell "fails at every value" apart from
"never had a chance to fail" — they print the same otherwise and mean opposite
things.

If one model fails everywhere while the others pass high, that model takes its own
setting rather than holding the rest down.

## Rules that are not optional

- Every dataset with a complete representation pool goes in. Cutting one needs a
  stated reason: degenerate, broken, or undefined.
- Never report a mean across datasets. Per dataset, per model, per fold.
- Always print the raw pair of scores in the dataset's own metric before any
  derived percentage.
- Count how many groups the voting fold can actually separate. A fold whose
  scores are nearly all tied ranks by list order, not by score, and that looks
  identical to a rule being unsafe.
