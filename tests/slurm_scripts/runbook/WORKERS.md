# Running the experiments on the server: one job script, one ledger

Since 2026-10-06. Every server job is `tests/slurm_scripts/worker.sh`. It reads one
ledger, takes the next dataset nobody holds, runs that dataset's current stage one
experiment at a time, writes the time and status of each one back, closes the stage,
and takes the next dataset. No dependencies, no per-dataset scripts.

The ledger lives in the run root, `tests/results/<run>/`:

| file | what |
|---|---|
| `ledger.json` | the state; every change goes through a lock on `ledger.lock` |
| `LEDGER.md` | the document: one row per dataset, one cell per stage, rewritten after every change |
| `events.jsonl` | one line per change, append-only |
| `logs/<dataset>__<stage>__<unit>.log` | the output of each experiment |
| `complexity/ similarity/ noise/ noise_full/ federated/ opig/` | the result files, in the layout `run_dataset.py` has always written |

`tests/results/CURRENT_RUN` (one line, untracked) names the live run. `worker.py`,
`ledger.py` and `run_dataset.py` read it, so nothing has to be passed around.

## Stages per dataset, in order

| stage | what runs | on which datasets |
|---|---|---|
| complexity | `search_f0` .. `search_f4` (the per-fold searches, one fold per unit, built before anything else), then c06 c19 c22 c07 c18 c01 c02 c08 c15 c04 c13 c11 (c12 on tcga) c16 c09 c24 c05 c23 c33 c17 c21 | all 41 |
| federated | fed02 fed03 fed04 fed05 fed01 | lhasa_herg, caco2, logd |
| similarity | s04 s03 s01 s12 s06 s13 s02 s10 s05 s08 s11 | every non-nucleotide dataset |
| noise | one unit per (experiment, noise condition); a many-method experiment runs fold 0 only (the one-fold screen, `noise_helpers.one_fold_screen_config`); c10 n01 n14 n15 n16 n26 n27 n37 run every fold | all 41, nucleotide without n01 n02 n11 n12 n14; herg_ic50 runs n17 n19 n22 n26 n29 n36 only |
| noise_full | empty until you promote experiments (below); a promoted unit runs every fold | |

The 11 OPIG scripts and the three cross-dataset scripts (s09, c03, c14) are rows of
their own (`opig:ginkgo`, `cross:s09`, ...). Each starts when the complexity stage
(similarity for s09) is closed on every dataset it reads.

A unit is `pending`, `running`, `ok`, `failed` or `skipped`. A unit that exits non-zero
is run once more by the next job, then left `failed`; the stage closes as
`done_with_failures` and the next stage still runs. A unit whose job died (wall time,
OOM) runs again, and its checkpoints resume. A stage closes when every unit is terminal.

## Memory and CPUs

A job's memory decides which datasets it may take. `nt_*`, `gb_*`, `tcga_*`, kiba,
bindingdb_kd and qm9 are the `big` class and need 128G; everything else runs in 64G,
and a 64G job never takes a big dataset. The cluster bills memory per core (arc 8 GB
per core, htc 10 GB per core), so ask for the cores that memory implies:

| class | htc | arc |
|---|---|---|
| small | `--cpus-per-task=8 --mem=64G` (the default) | `--cpus-per-task=8 --mem=64G` |
| big | `--cpus-per-task=12 --mem=120G` | `--cpus-per-task=16 --mem=128G` |

Run `bash tests/slurm_scripts/where_to_submit.sh --fits 8 64G` first. A GPU buys
nothing today: `src/kirby/models/dnn.py` pins the device to cpu.

## The four commands

1. Start a run (once; it also writes `CURRENT_RUN`):
```
python tests/experiments/orchestrate/ledger.py init --run tests/results/run_2026_10_06
```
2. Submit workers. The first argument is how many; after `--` come the sbatch flags
(cluster, partition, memory, CPUs; time defaults to 47:59:00). Each job sleeps 90 s
longer than the one before it. `DRY=1` prints the lines and submits nothing.
```
bash tests/slurm_scripts/submit_workers.sh 10 -- --clusters=htc -p medium,long
bash tests/slurm_scripts/submit_workers.sh 4 -- --clusters=htc -p medium,long --cpus-per-task=12 --mem=120G
bash tests/slurm_scripts/submit_workers.sh 3 --dataset qm9 -- --clusters=htc -p medium,long --cpus-per-task=12 --mem=120G
```
A job given `--dataset` goes to that dataset even when another job holds it; the two
take different experiments. That is how a big dataset gets more jobs.
3. Read the document: `tests/results/run_2026_10_06/LEDGER.md`, or
```
python tests/experiments/orchestrate/ledger.py status | less
python tests/experiments/orchestrate/ledger.py show herg_ki
```
4. Promote noise experiments to all folds (adds their units to `noise_full` on every
dataset, or `--datasets a b`):
```
python tests/experiments/orchestrate/ledger.py promote n20,n25
```

## By hand, in an srun

The same worker, or `run_dataset.py` as before; both record into the ledger.
```
srun --clusters=htc -p interactive --cpus-per-task=8 --mem=64G --time=04:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/experiments/worker.py --dataset herg_ki --experiment c07
python tests/experiments/worker.py --dataset herg_ki --stage noise
python tests/experiments/run_dataset.py noise herg_ki --experiments n16 --noise-condition gaussian
```
`--experiment` resets a failed unit and runs just that one. `run_dataset.py` with one
experiment finds the run through `CURRENT_RUN`, writes under the run root, skips a unit
the ledger already has as ok, and records its time. `--dry-run` on the worker prints
what would run.

## When something fails

`LEDGER.md` has a "Failed units" table with the attempts, exit code, job and log, and
the first error line. Fix the code, pull on the login node, then either let the next
worker retry it (one retry is automatic) or run it by hand with `--experiment`. To
clear a unit's record: `ledger.py reset herg_ki c07`. To free a dataset a dead job
still holds (workers do this themselves when the job is gone from squeue):
`ledger.py release-claim herg_ki`. `ledger.py sync` marks units ok from result files
written outside the ledger.

## Jobs queued before 2026-10-06 keep their place

Every job submitted before the ledger existed sources `setup.sh` at run time from a
frozen copy of its script. `setup.sh` now ends by checking the job's name: a job named
`x_<category>_<dataset>`, `opig_<name>` or `pc_<dataset>_x<N>` runs the worker instead
of its own body, with `--from-job-name` (the dataset or OPIG row the job was meant for,
first) and `--then-any` (then any dataset with work, until the wall). The job keeps its
queue position, its account, its 2 days and its memory; a 128G job may take the big
datasets. Nothing has to be resubmitted. An srun shell has job name `bash` and is never
redirected; `KIRBY_NO_REDIRECT=1` turns the redirect off for one job.

Held jobs (`JobHeldUser`) still have to be released by hand, as many at a time as you
want running: `scontrol -M arc release <id> <id> ...`. Jobs waiting on a dependency
(`afterany` on a running precompute link) start on their own when it ends and are
redirected the same way. A precompute job that is already running is past `setup.sh`
and finishes its whole-dataset search as before.

## What the old launchers are now

`submit_dataset.sh`, `submit_everything.sh`, `runbook/submit_all_stages.sh`,
`opig_one.sh` and `cross_one.sh` are not used to submit. The worker runs
`opig_one.sh` and `cross_one.sh` for the OPIG and cross rows (their argument tables),
with `KIRBY_OUTPUT_DIR` under the run root.
