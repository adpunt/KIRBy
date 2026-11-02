# Handoff: c03, and the other cross-dataset rows

Written 2026-10-30. Everything below was read in this session; nothing is from memory.

## What c03 is

`tests/experiments/complexity/c03_cross_dataset_transfer.py`, 800 lines. It asks whether
the budget that wins on a dataset tracks measurable properties of that dataset. Per
dataset, per held-out fold, per decided model, it runs one search on the fold's training
rows and scores the two best single representations and the two best hybrids on the
held-out rows. Then, across datasets, it computes one Spearman coefficient per
(complexity descriptor, downstream model), and splits budget variance between datasets
and between each dataset's own two models. The descriptors are those of Lorena et al.
2018, via `compute_dataset_complexity`.

The winning budget for a (dataset, model) is the median over folds of each fold's top
hybrid's budget. Every fold's budget is written to the result file.

c03 is not blueprint transfer. c24 is.

## Why it matters

`thesis.tex` line 2490 declines exactly this question: "We did not test whether these
picks follow measurable properties of a dataset, so we only describe them." c03 is the
run that would turn that sentence into a result. It is listed in `PAPER_2_OUTLINE.md`
as a potential point, not a result.

## State: never run

No `c03_cross_dataset_transfer` folder exists in any of the seven result runs on
`/Volumes/seagate`: `kirby_results_2026-10-08`, `-10-08b`, `-10-08b_noise_view`,
`-10-09`, `-10-09_noise_view`, `-10-10`, `-10-10b`.

`/Volumes/seagate/kirby_results_2026-10-10b/LEDGER.md` line 39:

    | cross:c03 | big | pending | 5 stages: abagym_Ang2/complexity, adme_mdr1/complexity,
      adme_rlm/complexity, flab_expression_koenig/complexity ... | |

c03 is a cross-dataset row, so it waits for each listed dataset's complexity stage to
finish before it starts. Those stages did not all finish, so it never started.

Two other cross-dataset rows in the same table:

    | cross:c14 | small | skipped | ready | |
    | cross:s09 | small | skipped | 5 stages: abagym_Ang2/similarity, adme_rlm/similarity,
      flab_binding_koenig/similarity, flab_expression_koenig/similarity ... | |

`cross:c14` says `ready` under "waits for" and is still `skipped`. That is not explained
by the wait, and it has not been looked into.

## The first job in the new chat: choosing which datasets c03 runs across

c03's result is a correlation across datasets, so the set of datasets is not a detail of
the run — it is the experiment. A descriptor that tracks the winning budget across six
small-molecule datasets and a descriptor that tracks it across a mixed set of molecules,
antibodies and reactions are different claims, and a set that is mostly one domain can
produce a coefficient that is really a domain difference.

Adelaide chooses the groups and which ones get tested, and that planning is done together
in the new chat before anything is launched. Do not pick a set, and do not present one as
settled. Bring what is needed to decide: which datasets have a finished complexity stage,
how many samples and groups each has, which two models each has decided, which domain
each belongs to, and what each descriptor is computed from. Then ask.

Open questions for that conversation, not to be answered alone:

- One set of all available datasets, or several groups run separately, for example one per
  domain and one mixed?
- Does a group need a minimum number of datasets before its correlation is reported?
- Are antibodies grouped with small molecules or kept apart, given the antibody datasets
  are much smaller?
- Do the reaction datasets go in at all, given there are two of them?

## What the next session has to establish, before proposing anything

1. Which five datasets `cross:c03` is waiting on. The ledger line is truncated at four
   plus an ellipsis. The full list is in the orchestrate plan, not the ledger.
2. Whether those datasets' complexity stages have since finished. Read the result
   folders, not the ledger, and not `analysis/*.md` (truncated at 400 lines, written
   2026-10-08).
3. Why `cross:c14` is skipped while its waits read `ready`.
4. Whether the file still matches the current engine. Its docstring records alignment
   decisions of 2026-09-12, 2026-10-05 and 2026-10-06, all before the 2026-10-26 and
   2026-10-29 code changes. It has not been run since any of them, so nothing has
   exercised those paths.
5. How many datasets c03 needs for a Spearman across datasets to mean anything. Its own
   docstring says no p-values. The number of datasets is the sample size of that
   correlation, and nothing in the file states a minimum.

## Rules that apply

- No axis, arm, budget or dataset may be cut for compute. A cut needs a stated reason:
  degenerate, broken, or undefined.
- The author decides what runs. Bring the numbers and the options; do not pick the
  datasets or the descriptor set.
- Her runs go local or through `srun`, never `sbatch`.
- Patient data (TCGA, ANNITIA, NHANES) is excluded everywhere, including from any
  cross-dataset correlation and any dataset count.
