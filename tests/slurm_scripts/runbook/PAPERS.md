# Running Papers 2 and 3 on the server: one job script, one ledger

Since 2026-11-03. Same design as the experiments' ledger (`WORKERS.md`) and the screening
ledger (`SCREEN.md`), built on the same code: every job is
`tests/slurm_scripts/papers_worker.sh`. It takes the next dataset that has work and is not
held, runs one unit at a time, records each one, and moves on. Queued jobs and sruns
record into the same file.

## Files in the run folder

| file | what |
|---|---|
| `RUN_SETTINGS.json` | every setting the run depends on, and the gate. Written once by `init`; after that only you edit it |
| `ledger.json` | the state; every change goes through a lock on `ledger.lock` |
| `PAPERS.md` | the document: one row per dataset, one cell per step, and why a step is held |
| `events.jsonl` | one line per change |
| `logs/<dataset>__<step>__<unit>.log` | each unit's output |
| `cache/` | this run's own `KIRBY_CACHE_DIR` (below) |
| `complexity/ similarity/ noise/` | the experiments' result files |

`tests/results/CURRENT_PAPERS` names the live run. The old `CURRENT_RUN` and the screen's
`CURRENT_SCREEN` are never read or written.

## The run's own cache

Every unit runs with `KIRBY_CACHE_DIR=<run>/cache`. In that folder:
- **These start empty:** search results (`result_db`, `searches`), baselines, fold scores and
  checkpoints. The list is `FRESH_CACHE_ENTRIES` in `tests/experiments/papers/ledger.py`.
- **These are links into the thesis cache** (`/data/stat-ecr/scat9264/.cache/kirby`):
  everything else, meaning the representation files and downloaded data.
- **These are not linked at all:** patient data, and leftovers no search reads (old checkpoints
  at the top of the cache, stale or trashed copies).
- **The thesis results are never touched.** The one write that reaches the thesis cache is a
  representation build adding a file that was missing.

## Steps per dataset, in order

| step | a unit is | starts when |
|---|---|---|
| reps | `tests/precompute_reps.py DATASET --include-augmentation`: builds the representation files that are missing and checks the rest | at once |
| search | one outer fold: `orchestrate/search_unit.py DATASET --fold k`, the three per-fold searches every experiment reads back (the standard search, the distance search and the similarity search) | reps is closed, `open.search` is true, and the dataset's two models are recorded |
| check | Step 3: fast tier and full search without pruning, at the dataset's best budget and model | not written yet; `open.check` |
| paper2 | one experiment of the Paper 2 run list | every search fold is ok, and `open.paper2` is true |
| paper3 | one experiment (or experiment x noise type) of the Paper 3 run list | every search fold is ok, and `open.paper3` is true |

**The search space**, from `RUN_SETTINGS.json`:
- **Mode:** `search_space: full` means only the two models come from DECISIONS.md, through
  `decided_grid.json`. Every other axis is the code's complete list, the same for every
  dataset. The lists are in `decided_grid.full_space_lists`.
- **Standard search:**
  - budgets of 10, 25, 50, 100, 250, 500, 1000 and 2000 features;
  - feature ranking by deeplift, lightgbm_gain and treeshap;
  - allocation methods performance_weighted, power_2, softmax_T1, rank_exp_d0.3 and
    wglobal_power2;
  - internal evaluators match, rf, lightgbm and dnn;
  - 2, 3, 5 or all representations.
- **Distance search:** budgets of 10 to 500 features, evaluators knn, rf, dnn, lightgbm and
  svm, and knn added to the two models.
- **Pruning:** `pruning: calibrated` runs `DEFAULT_PRUNING_SCHEDULE`. What it dropped, the
  folds it voted on, and the configurations before and after are written to
  `cache/searches/<search name>.pruning.json`.
- **The thesis file:** `decided_grid.json` is not read for anything but the models.

## The gate and the holds

`RUN_SETTINGS.json` has an `open` block: `search`, `check`, `paper2`, `paper3`. A step that is
not open never starts, on any dataset, even when named by hand with `--unit`. Set one to
`true` and the next claim starts it; nothing has to be restarted.

`future_work` lists datasets taken out at the gate. Their units not yet ok are held, not
deleted.

A dataset with no row in `decided_grid.json` is held at every step after reps. That means its
two models are not recorded. Rebuild the file (`python tests/experiments/build_decided_grid.py`)
and the next claim releases it.

## The commands

1. Start the run (once). Login node, after `git pull` and `. ./setup.sh`:
```
python tests/experiments/papers/ledger.py init --run /data/stat-ecr/scat9264/papers/run_2026_11_03
```
2. Read the state: `<run>/PAPERS.md`, or
```
python tests/experiments/papers/ledger.py status | less
python tests/experiments/papers/ledger.py show herg_ki
python tests/experiments/papers/ledger.py settings
python tests/experiments/papers/worker.py --dry-run
```
3. Queued jobs. The first argument is how many. After `--` come the sbatch flags. These are
the defaults when left out: 47:59:00, 8 CPUs, 64G, and `--exclude=htc-g011` on htc. `DRY=1`
prints the lines and submits nothing.
```
bash tests/slurm_scripts/submit_papers_workers.sh 10 -- --clusters=htc -p medium,long
bash tests/slurm_scripts/submit_papers_workers.sh 2 --dataset qm9 -- --clusters=htc -p medium,long
```
Jobs are named `kp` or `kp_<dataset>`, which `setup.sh` never redirects.

4. By hand, in an srun:
```
srun --clusters=htc --exclude=htc-g011 -p interactive --cpus-per-task=8 --mem=64G --time=04:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/experiments/papers/worker.py
python tests/experiments/papers/worker.py --dataset herg_ki --unit search_f2
```
5. Rerun one unit without touching the others: `ledger.py reset herg_ki search_f2`.
`ledger.py release-claim herg_ki` frees a dataset a dead job still holds. `ledger.py sync`
marks units ok from result files written outside the ledger. `ledger.py extend --datasets NAME`
adds a dataset once its loader and `DATASET_REPS` entry exist.

## When something fails

`PAPERS.md` has a "Failed units" table: attempts, exit code, job, first error line, log. A unit
that exits non-zero runs once more. A unit whose job died (wall time, memory) runs again and
resumes from its checkpoint.
