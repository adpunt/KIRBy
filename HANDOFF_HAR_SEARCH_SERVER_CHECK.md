# Handoff: first server run of the hybrid search on UCI HAR, with the dnn on a GPU

Written 2026-10-28. Paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`.

---

> Read `~/repos/KIRBy/CLAUDE.md` first and follow it in every reply. The author is the engineer
> and decides. This is a crash check, time-boxed. It produces no result that goes in a document.

## What this is for

The large-row benchmarks (So2Sat, BreizhCrops, RadioML) are trained by minibatch gradient
descent: one batch of rows per gradient step, early stopping on a held-out slice, each step on a
GPU. The search's `dnn` model already trains that way (`src/kirby/models/dnn.py`, batch 256,
up to 200 passes, patience 20). Until 2026-10-28 it was pinned to CPU. Now `_pick_device()`
in `dnn.py` picks `KIRBY_DNN_DEVICE` if set (`cpu`, `cuda`, `mps`), else cuda, else mps, else
cpu. Both fits, regression and classification, use it.

This handoff checks three things on the server, on UCI HAR, which is the one of the new
datasets already downloaded and already reproduced (NEW_DATASETS.md, section 1):

1. The dnn picks the GPU and one fit runs on it.
2. The search runs on the 18 HAR feature sets with `rf`, `lightgbm` and `dnn` as the
   models, groups = person, and finishes.
3. The numbers are in the range measured locally (table below).

## Measured locally on 2026-10-28 (CPU, laptop)

Script: `tests/experiments/benchmarks/har_search_smoke.py`. Log and json under
`/Volumes/seagate/benchmarks_protocol/uci_har/hybrid_smoke/`.

Run with `KIRBY_LGBM_THREADS=1 OMP_NUM_THREADS=1` (see the crash note below), so every
model fit used one core. Search settings: budgets 25, 50, 100, 250; models rf, lightgbm, dnn;
allocation `performance_weighted`; importance `lightgbm_gain`; internal evaluator `match`;
all 18 representations; 3 folds grouped by person; no cache.

One row is one measurement. Accuracy is the search's own 3-fold validation accuracy on the
7,352 training windows, 6 activities; it is not the published test accuracy (0.9637).

| Measurement | Local value |
|---|---|
| dnn device picked | cpu (this torch build has no GPU) |
| one dnn fit, 7,352 windows x 561 features, early stopping | 9.9 s |
| one pass of the dnn over 350,000 rows x 72 features, batch 256 | 2.6 s |
| whole search, 12 configurations + 54 baseline fits, 3 folds | 9.3 min |
| best single representation, any model | tGravityAcc, lightgbm, 0.8432 |
| hybrid, dnn, budgets 25 / 50 / 100 / 250 | 0.8797 / 0.8868 / 0.8804 / 0.9011 |
| hybrid, lightgbm, budgets 25 / 50 / 100 / 250 | 0.8802 / 0.8953 / 0.9151 / 0.9213 |
| hybrid, rf, budgets 25 / 50 / 100 / 250 | 0.8945 / 0.9010 / 0.9070 / 0.9093 |

The 18 single-representation baselines (rf, lightgbm, dnn) are in the json. They run from
0.4363 (tBodyGyroJerkMag, lightgbm) to 0.8432 (tGravityAcc, lightgbm).


## Known crash and what it means for the server

Locally the search died with exit 139 in the baseline evaluation until it was run with
`KIRBY_LGBM_THREADS=1 OMP_NUM_THREADS=1`. That is the known macOS crash: torch and LightGBM
load two OpenMP runtimes (memory file `project_fold_hook_and_lgbm_segfault_2026_09_02`). On the
server `setup.sh` sets `OMP_NUM_THREADS` to the core count and every search runs with `dnn`
and `lightgbm` together, so it is not expected there. If the server run dies with exit 139 and
no traceback, that is the signature; rerun with `OMP_NUM_THREADS=1` to confirm, and report it.

## Steps on the server

The HAR data is not on the server yet. The author copies files; do not ssh or scp.

1. Push the branch; pull on the server.
2. Put the UCI HAR zip on the server and unzip it, for example under
   `/data/stat-ecr/scat9264/benchmarks_protocol/uci_har/`. The outer zip is
   `human+activity+recognition+using+smartphones.zip` (md5 `96924e20fe5b8c9ffc2196eec7d5048f`),
   from `https://archive.ics.uci.edu/static/public/240/`. Inside it is `UCI HAR Dataset.zip`
   (md5 `53e099237392e0b9602f8c38f578bd8f`), which unzips to the `UCI HAR Dataset` folder.
   Check both md5s before unzipping.
3. One line for a GPU shell on htc (GPU jobs go to htc, memory file
   `project_arc_vs_htc_cluster`; the interactive partition name for GPUs is to be checked
   with `where_to_submit.sh` first):

   srun --clusters=htc -p interactive --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty /bin/bash

   Wait for the prompt. Then, inside that shell, one command per line:

   cd /data/stat-ecr/scat9264/KIRBy
   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
   export KIRBY_HAR_DIR="/data/stat-ecr/scat9264/benchmarks_protocol/uci_har/UCI HAR Dataset"
   python -c "import torch; print(torch.cuda.is_available())"
   python tests/experiments/benchmarks/har_search_smoke.py --out /data/stat-ecr/scat9264/benchmarks_protocol/uci_har/hybrid_smoke

4. What to read in the output, in order:
   - the line `dnn device:` must say `cuda`. If it says `cpu`, stop: torch on the server does
     not see the GPU, and the rest of the run says nothing about the GPU.
   - `A(b)`: seconds per pass over 350,000 rows on the GPU. Compare with the local CPU number
     in the table above. If the GPU is not faster, say so; it is a real finding, not a failure.
   - `B search done in N min`.
   - the per-model, per-budget accuracies and the 18 baselines, against the local table.
     Differences of a few thousandths are expected (different device, different thread count).
     A model at 0.17 (chance over 6 activities) or a nan is a defect; report it.

5. Report back as one table: local versus server, one row per number above. Nothing else.
   Do not run the published protocol (`b01`) and do not run any other dataset.

## Not in scope

- No change to `find_best_hybrid` or to any experiment file.
- No run on So2Sat, BreizhCrops, Kepler or Ninapro. Those have no loader yet.
- No entry in `NEW_DATASETS.md`; the author decides what goes there.
