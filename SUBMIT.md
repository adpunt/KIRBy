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

**13 of the 41 datasets are on the short search**, chosen by size and listed with
their estimates at the top of `run_precompute_dataset.sh`: the five nucleotide
and genomic sets (18,000 to 33,000 rows — past any full grid), `qm9` (~103 h),
`flab_expression_koenig` (~64 h), `logd` (~62 h), `nt_promoter_tata` (~39 h),
`buchwald_hartwig` (~36 h), `herg`, and the two `tz_her2` sets. The other 28 run
the full grid. Move a dataset between them by editing that one list.

The cost is driven by ROWS, not features. It is **not** driven by svm — measured
over five datasets from the per-configuration seconds the engine stores,
`xgboost` or `lightgbm` is the dearest model every time and svm is 6% to 24%,
the cheapest of the five on two of them. Capping svm lower would buy little and
would give up the model that wins `herg_ki` outright. The short search is the
lever, and it drops xgboost and svm together.

**Pruning before fast.** `--pruning calibrated` is the default and it is free:
measured across five datasets, every one of their 25 model-and-budget conditions
returned the *identical* best five-fold score as the unpruned grid, for 13% to
44% less time. `--speed fast` is roughly twenty times less work but searches
shorter lists, so it changes the answer. Use fast only where pruning is not
enough, and say so in the writeup.

---

## Existing jobs are never cancelled

A dataset that already has jobs queued keeps them. Its first new link waits for
them with `afterany`, and its experiment jobs wait behind that.

**Why not cancel and resubmit.** A queued job accrues age, worth up to 1000
priority points. On 2026-09-18 the 55 jobs queued three days earlier were at 235
to 344 age against jobs getting slots at 1088 to 1541 total. A resubmitted job
starts at zero. That age is the margin.

**Why wait rather than run alongside.** Two processes on one dataset both write
`result_db/<dataset>.db`. SQLite waits 120 seconds for the other writer, and if
it does not get the lock the search dies after paying for its baselines and its
per-fold scores. Waiting costs nothing — the new link would have sat in the
queue anyway, and the old jobs are doing that dataset's work meanwhile.

Only datasets with old jobs wait. The rest start as soon as the scheduler
reaches them.

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

**One command. It is the same script the queue runs**, so it picks the same
speed, the same pruning and the same passes, in the same order, and it resumes
whatever the queue already did.

```bash
srun -p interactive --cpus-per-task=8 --mem=32G --time=4:00:00 --pty /bin/bash
cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh

bash tests/slurm_scripts/run_precompute_dataset.sh herg_ki
```

Change the last word. That is the whole thing — no flags to remember, no
question about whether you did the similarity or knn pass, because it does all
four in the right order and prints its class, speed and pruning in the first
three lines.

Only two rules: **never on a dataset whose queued job is running**, and finish
the small ones first.

To do only part of it, e.g. skip the two single-experiment passes:

```bash
KIRBY_PASSES="sim dist" bash tests/slurm_scripts/run_precompute_dataset.sh herg_ki
```

---

## The three classes

Set by size, from a calibrated run (caco2: 1,811 of 8,160 configurations in four
hours) scaled by rows^1.5 × features. Edit the two lists at the top of
`run_precompute_dataset.sh` to move a dataset.

**SMALL — full grid, pruning OFF.** 15 datasets. They finish in about 1 to 9
hours, so there is no reason to accept even the calibrated schedule's risk.

`tcga_brca_pam50` ~1 h · `flab_thermo_tresanco` ~3 h ·
`humanness_therapeutic` ~3 h · `herg_ki` ~9 h · the five ginkgo sets ·
`abcov_kd` · the three abagym sets · `ddg_combined` · `cdr_flexibility`

**MEDIUM — full grid, calibrated pruning.** 14 datasets. 10 to 40 hours, which
four 12-hour links cover. Pruning is measured to lose no winner on any of the
five recorded datasets while saving 13% to 44%. This is also the default for
every dataset whose size is unknown.

`caco2` ~18 h · `buchwald_hartwig` ~36 h · `tcga_luad_vs_lusc` ·
`bh_doyle_plates` · the three `adme_` sets · `flab_binding_koenig` ·
`flab_binding_adams` · `flab_expression_adams` · `lhasa_herg` · `suzuki` ·
`kiba` · `bindingdb_kd`

**LARGE — fast tier.** 12 datasets. Past what four links reach at full, or past
10,000 rows where no full grid is sensible.

`nt_H3K27me3` 33,000 rows · `nt_H3K9me3` 28,288 ·
`gb_human_enhancers_cohn` 27,791 · `nt_H3K9ac` 24,278 · `nt_H3K4me3` 18,244 ·
`qm9` ~103 h · `flab_expression_koenig` ~64 h · `logd` ~62 h ·
`nt_promoter_tata` ~39 h · `herg` 7,772 rows · `tz_her2_binary` ·
`tz_her2_multiclass`

**svm is not banned anywhere, and should not be.** It is already skipped above
10,000 rows, which covers the five nucleotide and genomic sets. Below that,
measured over five datasets: `xgboost` or `lightgbm` is the dearest model every
time at 23.8% to 56.5%, and svm is 5.8% to 23.8% — the cheapest of the five on
two of them. It also wins `herg_ki` outright. The fast tier drops xgboost and
svm together, which is the lever that works.

---

## Which passes a dataset needs

All four, on every dataset. The script does them in this order so that a chain
that never finishes still leaves the most useful things done:

| pass | what reads it |
|---|---|
| `general_none` predictive + similarity | every complexity, noise and opig experiment, first |
| `general_distance_none` | c03, s04, s05, s06, s08, s10, s13 |
| `general_uncapped`, `general_capped` | the augmentation arms; skipped in seconds when a dataset has no augmentation representations |
| `general_distance_uncapped` | as above, distance lineage |
| knn | s02 only |
| ridge | c16 only |

`KIRBY_PASSES="sim dist"` drops the last two if you are short of time — nothing
but s02 and c16 notices.

---

## Checking on it## Checking on it

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
