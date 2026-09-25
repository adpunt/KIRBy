# How to submit. One page.

Written 2026-09-17 because we were not working from the same picture.

**2026-09-25: the four stages (prelim noise, trials, full suite, full noise)
are in `tests/slurm_scripts/runbook/README.md`, with one script each.** This
page is still how the precompute chain and `submit_everything.sh` work.

---

## Which cluster and which account

Fair share is kept per cluster AND per account. Measured 2026-09-24, higher
starts sooner:

| cluster | account | FairShare |
|---|---|--:|
| htc | stat-cadd | 0.0491 |
| htc | stat-ecr | 0.0360 |
| arc | stat-cadd | 0.0326 |
| arc | stat-ecr | 0.0195 |

The `/data/stat-ecr` paths are a filesystem and have nothing to do with which
account is billed. `submit_dataset.sh` no longer names an account in its
header; until 2026-09-25 it named stat-ecr, the lowest of the four.

`submit_everything.sh` run with no flags submits to the login node's cluster
and bills the account `where_to_submit.sh --emit` picks, which compares the
accounts on that one cluster only. To use the best pair across both clusters:

```bash
sshare -M arc,htc -u $USER -o Cluster,Account,User,FairShare     # re-measure
read CL ACCT < <(bash tests/slurm_scripts/where_to_submit.sh --best-cluster-account)
bash tests/slurm_scripts/submit_everything.sh --clusters "$CL" --account "$ACCT" --dry-run
```

A dependency cannot cross clusters: a job on htc cannot wait for a precompute
job on arc. `submit_everything.sh` skips a dataset whose precompute is queued on
the other cluster when it queues precompute; `runbook/stage3_full.sh` gates the
experiment-only submission on `tests/tools/precompute_status.py` instead.

---

## The one command

```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin similarity-metrics-study
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh

bash tests/slurm_scripts/submit_everything.sh --dry-run   # look
bash tests/slurm_scripts/submit_everything.sh             # then type y
```

That queues five families, all dependency-linked so nothing runs before the
searches it reads exist:

| family | jobs | waits for |
|---|--:|---|
| precompute | 4 chained links per dataset | the previous link, and any job already queued for that dataset |
| complexity, similarity, noise | 1 each per dataset | that dataset's last link |
| federated | 1 each on `lhasa_herg`, `caco2`, `logd` | the same |
| opig | 13, one per script | the last link of every dataset that script reads |

41 datasets, 288 jobs. Framework and `similarity_metrics/` are deliberately not
in it — they are not part of this run.

Nothing else needs submitting. Everything below is why, and what to change.

---

## The three rules

**1. A precompute job that runs out of wall has not failed.** Rows are written
to `result_db` every 25 configurations, and the next run skips everything
already stored. That is why the chain exists: link 1 gets as far as it gets,
link 2 resumes exactly there, and a link with nothing left exits in minutes.
Over-chaining costs a queue slot. Under-chaining leaves work undone.

**2. Python is read when the job starts. The shell scripts are not.**

`sbatch` copies the shell script into SLURM's own store at submit time. So
editing `run_precompute_dataset.sh` or `submit_dataset.sh` changes nothing for a
job already queued — it will run the copy it was submitted with. See exactly
what a queued job will run:

```bash
scontrol write batch_script <jobid> -
```

Everything that copy then *invokes* is read from disk at run time. So:

| edit | reaches queued jobs? |
|---|---|
| `SINGLE_DATASET_EXPERIMENTS` in the three `*_helpers.py` | **yes** |
| any experiment file (`n07.py`, `c11.py`, …) | **yes** |
| `precompute_hybrids.py`, `find_best_hybrid.py`, `_shared.py` | **yes** |
| `run_precompute_dataset.sh`, `submit_dataset.sh` | **no** — resubmit to change those |

`submit_dataset.sh` changed on 2026-09-25: each experiment runs in its own
python process, so one segfault or out-of-memory kill no longer ends the rest;
each experiment's exit code goes to `<output>/_job_summaries/<jobid>.tsv`; the
defaults are 8 CPUs, 64G, 47:59:00, partition medium (were 32, 128G, 7 days,
long); `KIRBY_SKIP_INSTALLS=1` and `KIRBY_LGBM_THREADS=1` by default. Jobs
queued before that run the old copy.

Adding or dropping an experiment therefore needs no resubmission. Changing which
passes or which speed a precompute job uses does.

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

**Nucleotide datasets drop the knn pass automatically.** Every `s01`–`s13`
experiment is skipped on that domain — `_shared.NUCLEOTIDE_SKIP_EXPERIMENTS`,
on the stated ground that the similarity premise is unreliable for sequence
embeddings — and s02 is the knn pass's only reader. The job says so when it
does it.

The other three passes stay, each for a reason worth knowing: `n13` sets
`POOL_SUBSET='predictive_and_similarity'` and is *not* in that skip list, so the
similarity searches are read; `c03` reads the distance pool; `c16` reads the
ridge pass; and all 20 complexity experiments run on nucleotide.

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
