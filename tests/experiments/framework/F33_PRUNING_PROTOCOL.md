# The pruning analysis protocol

Agreed 2026-09-16. This is how every pruning question gets answered from here on.
It exists because the analysis kept changing shape between questions — a rule
family would get swept, then quietly swapped for another, then a fold choice
would get merged away, and the tables could not be compared to each other.

Nothing here is a result. Results go in `FRAMEWORK_FINALIZATION.md` §7.

## 1. What can be pruned

Four axes. Model and budget are the question the search answers — the result is
reported per (model, budget) condition — so pruning them means declining to
answer, and they are not candidates.

| axis | members | |
|---|--:|---|
| representation count | 4 | 2, 3, 5, all |
| allocation method | 5 | performance_weighted, power_2, softmax_T1, rank_exp_d0.3, wglobal_power2 |
| importance method | 3 | deeplift, lightgbm_gain, treeshap |
| internal evaluator | 4 | match, rf, lightgbm, dnn |

## 2. What SAFE means

For every (model, budget) condition — 25 per dataset — the best 5-fold mean
score after pruning is **the same number** as before pruning.

Not "the winning member is still reachable". Not "within 2%". The same number.
A member can survive a prune while the configuration that made it win is
dropped, and that is a loss.

Where it is not safe, report how many of the 25 conditions got worse and by how
much, in that dataset's own metric. Per dataset. **Never averaged across
datasets.**

## 3. The rule shapes

**Shape A — the places vote.** Group by every axis except the one being pruned,
so members compete only against each other for the same model at the same
budget. Rank the members inside each group on the voting fold. A point to the
best `top_n`. Delete a member scoring in fewer than `threshold` of the groups.
This is what `_run_search` does to allocation methods today.

**Shape B — the checkpoint drop.** At a checkpoint budget, inside each (model,
evaluator) group, rank every configuration at or below that budget on the
ranking fold and drop the worst `fraction`. Dropped recipes never run at any
higher budget. Shape B drops configurations, not axis members, so its tables are
not per-axis.

**Shape C — B then A**, the order `_run_search` runs them in. Only the
combinations that were safe alone are worth running together.

**Excluded: percentage-behind.** Cut a member whose best score is more than X%
behind the best score anywhere in the search. Excluded because it compares a
member against a configuration from a different model and budget: it deleted
N=2 on logd, which was 2.95% behind the best `dnn`-at-budget-300 configuration
while itself winning 17 of the 25 conditions. Stated here so the exclusion has a
reason on record and can be overruled.

## 4. The parameters swept

```
voting / ranking fold   fold 0 | fold 1 | folds 0 and 1 combined
                        THREE SEPARATE TABLES. Never merged into one number.

top_n      (shape A)    1, 2, 3, up to members-1
                        (1,2,3 for a 4- or 5-member axis; 1,2 for importance)

threshold  (shape A)    5%, 10%, 15%, 20%, 25%, 30%, 40%
                        (15% is what ships today)

checkpoints (shape B)   [25] | [50] | [100] | [25,50] | [25,50,100]

fraction   (shape B)    0.0 to 0.8 in steps of 0.1
                        (0.0 isolates what shape A costs on its own)
```

## 5. The tables

One table per (axis, rule shape, fold choice). Rows are the threshold or the
fraction. Columns are the datasets. Each cell reads exactly one of:

- `cuts nothing`
- `SAFE — cuts <member names>`
- `n/25, gap X.XXXX`  — conditions damaged, and the mean amount in that metric

Under each row, name the members cut on each dataset.

Shape A: 4 axes x 3 fold choices x 3 values of `top_n` = 36 tables.
Shape B: 3 fold choices x 5 checkpoint lists = 15 tables.

**No hours. No savings. No percentages of wall time.** Adaptive pruning is for
finding the best answer; compute is the speed tier's job. What a rule costs in
time is not part of this decision.

## 6. What the protocol does not cover

- Augmentation. Every f33 record was made with `augmentation_strategy='none'`.
- Four of the nine datasets with a complete representation pool:
  `buchwald_hartwig`, `caco2`, `flab_expression_koenig`, `nt_promoter_tata`.
  Every table is five datasets wide, not nine. Not cut — not yet recorded.
