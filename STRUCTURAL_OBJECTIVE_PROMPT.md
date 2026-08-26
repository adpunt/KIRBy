# New-chat prompt — should the search be able to optimise structural fidelity?

Paste everything below the line into a fresh chat.

---

## The situation

KIRBy's search picks a feature subset by maximising **one number**. Which number is a
free choice, and today `find_best_hybrid` accepts three:

| name in the code | what it maximises |
|---|---|
| `predictive` | a model is fitted on the subset and scored on held-out rows |
| `similarity_sac` | Spearman between pairwise distance and absolute activity difference |
| `similarity` | Spearman between kNN-predicted activity and true activity (read-across) |

There is a fourth quantity the code computes but cannot select on: **MCS structural
fidelity**, the Spearman between pairwise distance and maximum-common-substructure
distance. It asks whether two molecules the space calls close really do share more atoms
and bonds. It is one of the similarity study's two grading yardsticks.

Until 26 August it could only be computed at the very end, on the outer test fold in
`nested_cv_evaluate`. `compute_mcs_pairs` stores each pair as two row numbers into the
whole dataset. The search only ever saw one fold's rows, renumbered from zero, and
nothing carried the original numbers in, so a stored pair could not be located.
Demonstrated: on a 100-row fold of a 200-row set, the fidelity is 0.9978 with the
original row numbers and −0.0671 with rows renumbered from zero.

**That is now fixed.** `find_best_hybrid`, `_run_search` and `_evaluate_hybrid_config`
take `row_index` (whole-dataset row number of each row, in order) and `structural_pairs`
(`{'I','J','mcs'}`). When both are given, every configuration's result carries a
`struct_fidelity` entry in its metrics, per fold. `nested_cv_evaluate` fills `row_index`
with the outer-training row numbers automatically. Both the per-configuration cache and
the search cache carry a hash of the pair set, so a run that records structural fidelity
cannot be served a cached run that did not.

Recording it does **not** change which configuration wins. Verified: the ranking score is
identical to 1e-12 with and without the pairs. Adding a fourth objective is the separate
decision, and it is the one you are here to settle.

## What is already measured — read this before designing anything

`tests/experiments/framework/f19_objective_scoring.py` ran on herg_ki and logd. It builds
a pool of 60 candidate subsets, has each objective pick its own argmax on the training
half, and scores every pick on the held-out half under all four yardsticks. Results are in
`tests/results/f19_objective_scoring/`. All four columns are higher-is-better; the
`predictive` column is held-out Spearman rho, **not** the `metric_name: rmse` /
`lower_is_better: true` the JSON also carries — those two fields describe the dataset's
domain metric and mislabel that column. Fixing that label is a fair first task.

Held-out scores, by which objective did the picking:

| picked by | herg_ki predictive / readacross / struct | logd predictive / readacross / struct |
|---|---|---|
| predictive | 0.7198 / 0.3708 / 0.1857 | 0.8815 / 0.5928 / 0.3000 |
| sac | 0.4659 / 0.2482 / 0.1745 | 0.7686 / 0.6018 / 0.2650 |
| readacross | 0.7408 / 0.4303 / 0.2903 | 0.8899 / 0.6896 / 0.4100 |
| struct_fid | 0.7228 / 0.3752 / 0.2750 | 0.8878 / 0.6935 / 0.3964 |
| readacross 0.75 + struct_fid 0.25 | 0.7455 / 0.4452 / 0.3134 | 0.8899 / 0.6896 / 0.4100 |

Three things that table says, both datasets agreeing:

1. **Picking by read-across costs nothing on the predictive yardstick and buys a lot of
   structure.** herg_ki 0.7408 against predictive's own 0.7198; logd 0.8899 against
   0.8815. Structure goes 0.1857 to 0.2903 and 0.3000 to 0.4100.
2. **Picking by structural fidelity is close behind and also costs nothing predictively**
   (0.7228 and 0.8878). So a structural objective is not obviously worth its own slot — a
   read-across objective may already deliver most of it.
3. **Picking by SAC is the worst option on every yardstick** (herg_ki predictive 0.4659).
   That matters beyond this experiment: under BOTH similarity objectives the allocator
   ranks representations by SAC when it splits the budget, in `_evaluate_allocation`.

Those numbers come from a 60-candidate pool, one train/test half per fold, not from the
real search and not from `nested_cv_evaluate`. Treat them as the hypothesis to test, not
as the answer.

## What you have to answer

**Is a structural-fidelity objective worth adding to the search grid, or does one of the
existing three already win?**

Concretely, three sub-questions:

1. Does picking by structural fidelity beat picking by read-across, inside the real
   search, under honest nested CV? f19 says no by a small margin on two datasets. Test it
   properly.
2. Do blends beat singles? f19's best cell on herg_ki was three-quarters read-across plus
   one-quarter structure, which beat every single objective on all three yardsticks. The
   engine cannot express a blend today, so building one is part of the work if the answer
   is yes.
3. Should the budget split stop ranking representations by SAC? f19 says SAC is the worst
   selection rule of the four, and it is the rule the allocator uses under every geometry
   objective. Measure it; do not assume it.

## How to build it

A **framework** experiment. It decides how KIRBy is built, so it goes in
`tests/experiments/framework/` with the next free f-number. Not the noise suite, not the
similarity suite. Do not edit f00 through f15.

Use the real search, not a candidate pool — that is the whole point of building it after
f19. `find_best_hybrid(..., objective=..., row_index=..., structural_pairs=...)` now
records `struct_fidelity` per configuration, so every arm can be graded on all four
yardsticks whichever objective picked it.

Adding a fourth objective means:

- a new constant beside `OBJECTIVE_PREDICTIVE`, `OBJECTIVE_SIMILARITY_SAC` and
  `OBJECTIVE_SIMILARITY` in `find_best_hybrid.py`, added to `_VALID_OBJECTIVES` and
  `_is_geometry_objective`
- a mirror of the same constant in `hybrid.py`, which keeps its own copy so it need not
  import `find_best_hybrid`. **Miss this one and every fold fails with "All folds
  failed"** — the mirrored list is a known trap.
- a branch in the geometry scoring path that returns `_structural_fidelity_score` instead
  of read-across
- `_evaluate_allocation` in `hybrid.py` returns a SAC score under every geometry
  objective. A structural objective that leaves that alone would pick its budget split by
  SAC and its winner by structure, which is sub-question 3.

## Coverage

Every axis in full. Server time is free and an under-explored grid costs more than a long
one. Objective (all four, plus blends if you build them) crossed with the full budget
list, over the molecular roster, split by dataset. Report per dataset and never average
across datasets. Grade every arm on all four yardsticks, not just the one that picked it —
an objective that wins only on its own yardstick has proved nothing.

If a cell is skipped, log what was skipped in the output. A truncated run that says
nothing reads as a complete one.

Estimate wall time before starting. Over three hours locally becomes a server job. That is
a routing decision, never a reason to shrink the grid.

## House rules

- Report `delta_mean_of_folds` from `nested_cv_evaluate`. Never `search_mean`, never
  `.all_results`.
- No random splits. Scaffold-group-aware folds throughout.
- Baselines are individual representations, reported separately. No averaged baselines.
- Greedy allocators are CUT. If greedy comes up, say so in one sentence and move on.
- Never `ssh` or `scp` to the server. The user submits jobs.
- Read `CLAUDE.md` at the repo root before writing any message.

## Definition of done

A recommendation with numbers behind it: add a structural objective, or do not, and if
blends win then which blend at which weight. Per dataset. Then update the entry in
`SIMILARITY_METRICS_ROADMAP.md` that says structural fidelity cannot be an optimisation
target, because after this chat it either can be or has been decided against.
