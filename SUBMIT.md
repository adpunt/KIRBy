# How to submit. One page.

Written 2026-09-17 because we were not working from the same picture.

---

## The one command

```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin similarity-metrics-study
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh

bash tests/slurm_scripts/submit_everything.sh --dry-run   # look
bash tests/slurm_scripts/submit_everything.sh             # then type y
```

That queues, for every dataset: four 12-hour precompute jobs chained one after
another, then one job per experiment category depending on the last of them.
41 datasets, 287 jobs.

Nothing else needs submitting. Everything below is why, and what to change.

---

## The three rules

**1. A precompute job that runs out of wall has not failed.** Rows are written
to `result_db` every 25 configurations, and the next run skips everything
already stored. That is why the chain exists: link 1 gets as far as it gets,
link 2 resumes exactly there, and a link with nothing left exits in minutes.
Over-chaining costs a queue slot. Under-chaining leaves work undone.

**2. The experiment jobs read your code when they start, not when you submit.**
They are queued without naming experiments, so each runs whatever
`SINGLE_DATASET_EXPERIMENTS` holds at the moment it starts —
`noise_helpers.py`, `complexity_helpers.py`, `similarity_helpers.py`. Add an
experiment to that list, delete one, or edit any experiment file, and every
queued job picks the change up. **No resubmission.**

**3. Do not change what identifies a stored result while jobs are queued.** The
metric, the fold split, the seed, the representation pool, `n_folds`. Those are
the columns the result database keys on, so changing one orphans every row the
precompute already wrote and every queued experiment then finds nothing. Adding
or removing an experiment is safe. Changing how a score is computed is not.

---

## What is infeasible at full search

Every dataset, in a single 12-hour job. Measured on the f33 record and scaled to
the eight budgets the precompute uses:

| dataset | full search | with calibrated pruning |
|---|--:|--:|
| logd | 79 h | 44 h |
| herg_ki | 42 h | 35 h |
| tcga_brca_pam50 | 42 h | 30 h |
| flab_thermo_tresanco | 32 h | 24 h |
| humanness_therapeutic | 15 h | 13 h |

The other 36 datasets have no measured timing and I will not invent one.

**This is what the chain is for.** Four links is 48 hours, which covers
everything in that table with pruning on, and covers logd unpruned only just.
If a dataset is still short after four links, add more — the jobs resume, so
there is nothing to redo.

**Both halves prune the same way.** The precompute and the experiment jobs are
both given `calibrated`. They have to match: if the experiments pruned
differently they would ask for configurations the precompute deleted, and
rebuild them. `aggressive_pruning`, which the experiment jobs used to default
to, is the April setting — it drops at budgets 25 and 50 as well as 100, and on
the f33 record it deletes the two allocation methods that win most of
tcga_brca_pam50's conditions.

**Pruning before fast.** `--pruning calibrated` is the default and it is free:
measured across five datasets, every one of their 25 model-and-budget conditions
returned the *identical* best five-fold score as the unpruned grid, for 13% to
44% less time. `--speed fast` is roughly twenty times less work but searches
shorter lists, so it changes the answer. Use fast only where pruning is not
enough, and say so in the writeup.

---

## Changing things after submission

| what you want to do | how |
|---|---|
| add or drop an experiment | edit `SINGLE_DATASET_EXPERIMENTS`. Nothing to resubmit. |
| change an experiment's code | edit the file. Nothing to resubmit. |
| add more chain links to one dataset | `sbatch --dependency=afterany:<last> ... run_precompute_dataset.sh <ds>` |
| stop a dataset entirely | `scancel --name=pc_<ds>_1` … and its experiment jobs |
| run something by hand meanwhile | see below — but never on a dataset whose job is running |

**Never run a dataset by hand while its queued job is running.** Both write
`result_db/<dataset>.db`; SQLite waits 120 seconds for the other writer and then
the search dies, after paying for its baselines. One process per dataset.

---

## By hand, in an interactive session

```bash
srun -p interactive --cpus-per-task=8 --mem=32G --time=4:00:00 --pty /bin/bash
cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
```

One dataset, every condition, resumable:

```bash
DS=herg_ki; for C in general_none general_uncapped general_capped; do python -u tests/precompute_hybrids.py $DS --condition $C --speed full --pruning-mode calibrated --objectives predictive,similarity; done; for C in general_distance_none general_distance_uncapped; do python -u tests/precompute_hybrids.py $DS --condition $C --speed full --pruning-mode calibrated --distance; done
```

Change `DS` and paste. Skip any dataset whose queued job is running.

Worth the session, in order — by how many experiment files name the dataset,
how well its base representations predict, and how large it is:

`herg_ki` (45 files, +0.758) · `flab_thermo_tresanco` (10, +0.615) ·
`humanness_therapeutic` (6, +0.805, two conditions only) ·
`bh_doyle_plates` (4, +0.808) · `herg` (9, federated) ·
`tcga_luad_vs_lusc` (3, +0.881)

Not worth it: the five ginkgo sets (one experiment each, +0.190 to +0.518),
`ddg_combined` (+0.154), `flab_binding_adams` (+0.075).

---

## Checking on it

```bash
squeue -u $USER -o '%.10i %.26j %.20P %.2t %.10M %.10L %R' | head -40
python tests/precompute_coverage.py --trial      # in an srun
sacct -u $USER --starttime today -o JobID,JobName%30,State,MaxRSS,Elapsed | grep -E 'pc_|x_'
```

`TIMEOUT` on a `pc_` job is expected and fine. `OUT_OF_MEMORY` is not — raise
that dataset's memory and add a link.

---

## Blocked, and why

| dataset | why | fix |
|---|---|---|
| `suzuki` | `cgr_morgan` not built | `python tests/precompute_reps.py suzuki` |
| `kiba`, `bindingdb_kd` | no representations at all | `bash tests/slurm_scripts/submit_precompute_reps.sh dta` — htc, GPU |
| `davis` | parked, per-drug-mean degeneracy | your call |
| `chembl_qsar` | `load_data` has no path to it | builds its own inside `opig_activity_cliff_qsar` |
| `qm9` | 4 augmentation representations absent | `none` condition only |
