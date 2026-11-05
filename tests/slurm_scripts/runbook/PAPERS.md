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
| `complexity/ similarity/ noise/ meta_learner/` | the experiments' result files |

`tests/results/CURRENT_PAPERS` names the live run. The old `CURRENT_RUN` and the screen's
`CURRENT_SCREEN` are never read or written.

## The caches

Each of the three searches has its own cache and starts with an empty score store, so no
search reuses another's scores and the hours recorded per fold are that search's real time:
`<run>/cache_pruned`, `<run>/cache_fast`, `<run>/cache_full`. `<run>/cache` is what the
representation step uses.
- **These start empty in each:** search results (`result_db`, `searches`), baselines, fold
  scores and checkpoints. The list is `FRESH_CACHE_ENTRIES` in `tests/experiments/papers/ledger.py`.
- **These are links into the thesis cache** (`/data/stat-ecr/scat9264/.cache/kirby`):
  representation files and downloaded data.
- **These are not linked at all:** patient data, and leftovers no search reads (old checkpoints
  at the top of the cache, stale or trashed copies).
- **The thesis results are never touched.** The one write that reaches the thesis cache is a
  representation build adding a file that was missing.

## Steps per dataset, in order

| step | a unit is | starts when |
|---|---|---|
| reps | `tests/precompute_reps.py DATASET --include-augmentation`: builds the missing representation files and checks the rest | at once |
| search_pruned | one outer fold, full search space, calibrated pruning | reps finished without failure, the step is open, and the dataset's two models are recorded |
| search_fast | one outer fold, the engine's fast tier | same |
| search_full | one outer fold, full search space, no pruning | same |
| paper2 | one experiment of the Paper 2 run list. The meta-learner has one unit per outer fold, `meta_f0` to `meta_f4`, each also held until the dataset is named (below) | the step is open, `experiments_search` is set, and that search's folds are ok on the dataset |
| paper3 | one experiment (or experiment x noise type) of the Paper 3 run list | same |

A search unit runs `orchestrate/search_unit.py DATASET --fold k`, which runs the three kinds
of search the experiments read back: predicting the label, the distance-based evaluators, and
the five nearest neighbours.

**What each search holds** (`decided_grid.full_space_lists`, `fast_space_lists`). The dataset's
two models come from DECISIONS.md through `decided_grid.json`; nothing else is read from that file.

| axis | full and pruned | fast |
|---|---|---|
| budgets, in features | 10, 25, 50, 100, 250, 500, 1000, 2000 | same |
| feature ranking | deeplift, lightgbm_gain, treeshap | lightgbm_gain |
| allocation methods | performance_weighted, power_2, softmax_T1, rank_exp_d0.3, wglobal_power2 | performance_weighted, power_2, rank_exp_d0.3 |
| internal evaluators | match, rf, lightgbm, dnn | same |
| number of representations | 2, 3, 5, all | 3, all |

The distance-based search runs budgets of 10 to 500 features and evaluators knn, rf, dnn,
lightgbm and svm in all three, with knn added to the two models.

What pruning dropped, the folds it voted on, and the configurations before and after are in
`cache_pruned/searches/<search name>.pruning.json`.

## The gate and the holds

`RUN_SETTINGS.json` has an `open` block with one entry per step past reps. A step that is not
open never starts, on any dataset, even when named by hand with `--unit`. Set one to `true` and
the next claim starts it; nothing has to be restarted. `experiments_search` is `pruned`, `fast`
or `full`: the search the experiments read, chosen once you have compared them.

`future_work` lists datasets taken out at the gate. Their units not yet ok are held, not deleted.

A dataset with no row in `decided_grid.json` is held at every step after reps, because its two
models are not recorded. Rebuild the file (`python tests/experiments/build_decided_grid.py`) and
the next claim releases it. A dataset whose representation build failed for good is also held.

## The meta-learner in the paper2 step

Paper 2's meta-learner over models with their own hybrids, `tests/experiments/papers/meta_learner_own_hybrids.py`.
Opening paper2 does not start it: its units also wait for the `meta_learner` block in
`RUN_SETTINGS.json`, and run only on datasets you name there.

| key | what | starts as |
|---|---|---|
| `datasets` | the datasets it runs on, chosen from results and runtime | empty |
| `candidate_models` | every model searched, per outer fold; the top 5 are taken on the fold's training rows | rf, dnn, lightgbm, svm, gp, xgboost, ngboost |
| `net_sizes` | nets added to the candidates: `dnn_small` is (64, 32), `dnn_wide` is (512, 256, 128) | both |
| `inner_folds` | folds inside the training folds, for the predictions the meta-learner trains on; every fit uses the model's one fixed feature set | 5 |
| `min_rows` | a named dataset with fewer rows writes a skipped result and fits nothing | 500 |

Add a name to `datasets` and the next claim starts it; nothing has to be restarted. To time one
fold before committing to a dataset, name it, then run
`python tests/experiments/papers/worker.py --dataset herg_ki --unit meta_f0`. The unit's hours are
in `PAPERS.md`, and each candidate's search seconds are in its result file.

Because a dataset you do not name keeps these units held, its paper2 cell never reaches all units
finished: it reads 5 short, and that is not a failure.

Results: `<run>/meta_learner/<dataset>__meta_f<k>.json`: each candidate's search score and seconds,
the top models, each model's held-out score, and for every meta-learner its held-out score and its
cross-validated score on the training rows. The script does not choose a meta-learner. Predictions
are in `<dataset>__meta_f<k>.predictions/`, so a killed job resumes, and adding a meta-learner fits
no downstream model again (reset the unit to rerun it).

Candidates outside the dataset's two recorded models have their own search files, named
`meta_<model>`, so they do not overwrite the two-model search the experiments read. The two recorded
models read the ledger's search. Under `pruned`, each extra candidate's pruning votes on that model
alone, while the two recorded models' pruning voted on both.

A run started before 2026-11-05 has none of this. `ledger.py extend --all` adds the units to its
paper2 step and changes no setting and no finished unit. Then add the `meta_learner` block to that
run's `RUN_SETTINGS.json` by hand. Do not use `replan` for this: it closes every step past reps.

## The commands

1. Start the run (once). Login node, after `git pull` and `. ./setup.sh`:
```
python tests/experiments/papers/ledger.py init --run /data/stat-ecr/scat9264/papers/run_2026_11_03
```
2. Read the state: `<run>/PAPERS.md`, or
```
python tests/experiments/papers/ledger.py progress
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
