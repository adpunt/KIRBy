# Screening candidate datasets on the server: one job script, one ledger

Since 2026-11-01. A screen asks, for one candidate dataset: how good is each view
(representation) on its own, under the models the dataset's papers used, and how much
do the views overlap. Its result is a `summary.md` per dataset, which feeds the choice
of up to ten views in `NEW_DATASETS.md`.

It runs the same way as the experiments (`WORKERS.md`), and it is built on the same
code: `tests/experiments/screen/ledger.py` is the experiments' ledger with a different
plan, so the lock, the claims, the liveness check (one `squeue` call; a job that is gone
is stale and its unit runs again), the retry rules and the heartbeats are the ones
already in use. Queued jobs and interactive sruns record into the same file and can
pick up each other's work at any point.

| file | what |
|---|---|
| `<run>/ledger.json` | the state; every change goes through a lock on `ledger.lock` |
| `<run>/SCREEN.md` | the document: one row per dataset, one row per view, failed units; rewritten after every change |
| `<run>/events.jsonl` | one line per change, append-only |
| `<run>/logs/<dataset>__<stage>__<unit>.log` | the output of each unit |
| `<run>/<dataset>/` | the results; `<run>/dronerf/b06_views/summary.md`, `<run>/matbench/b03_views/summary.md`, `<run>/reactions/b10_views/summary.md` |

`tests/results/CURRENT_SCREEN` (one line, untracked) names the live screen run. It is a
different file from the experiments' `CURRENT_RUN`, so the two never touch.

## Stages and units

Every dataset row has three stages. A job takes runnable evaluate units first, then
build units, then the summary: views that are ready get scored before more are built.

| stage | a unit is | waits for |
|---|---|---|
| build | one view's feature file; a view trained per fold (the two DroneRF CNNs; chemprop_cgr on reactions) is one unit per task (and split) and fold | nothing (DroneRF builds wait for `extract`) |
| evaluate | one view x one model x one task (x one split on DroneRF): every outer fold of the published protocol. Reactions: two per view, model and dataset, the view alone and with cgr_bond_changes appended | its view's build unit(s), and cgr_bond_changes' build for the second |
| summarise | the dataset's `summary.md`: scores, error overlap and linear CKA between views | every evaluate unit closed |

A unit is `pending`, `running`, `ok`, `failed` or `skipped`. A unit is `ok` only when its
file is there (an evaluate file must say `result_status: ok` and not be a `--debug`
run). A unit that exits non-zero runs once more, then stays failed. A unit whose job died
(wall time, OOM, an srun closed) runs again and resumes: evaluate units at the next fold,
image builds at the next 100 chunks.

## What is in the rows (2026-11-01)

| dataset | views | models | units |
|---|---|---|---|
| dronerf | spectrum, spectrum_low, spectrum_high, welch_psd, spectral_scalars, stats (built by `extract`); psd_resnet50, spectrogram_resnet50, histogram_resnet50, gaf_resnet50 (one build each); rfuavnet_emb, alemadi_emb (60 builds each) | knn, rf, logreg, xgboost, dense | 125 build, 360 evaluate, 1 summary |
| matbench | magpie, stoichiometry, valence_orbital, ion_property, sine_coulomb, soap_avg, site_stats_fingerprint, density_features, rdf, xrd, global_symmetry, chgnet, mace_mp (expt_gap: the first four) | rf | 75 build, 69 evaluate, 1 summary |
| reactions (tasks buchwald_hartwig, bh_doyle_plates, suzuki) | each dataset's pool of 10 (`complexity_helpers.DATASET_REPS[ds]['general']`, cgr_morgan among them); cgr_smiles_lm, chytorch_graphormer, cgr_atompair, cgr_torsion, cgr_bond_changes (one build each); chemprop_cgr (5 builds, one per outer fold) | `decided_models(ds, 'general')`: rf, lightgbm on all three | 60 build, 186 evaluate, 1 summary |

Two models are planned but not in the rows:
- DroneRF `svm`: Kılıç's grid over C and gamma (2^-15 to 2^15) with inner CV is about
  460,000 fits. Not added until a smaller grid is chosen.
- Matbench `modnet`: needs MODNet's own python 3.8 environment, not built on the server.
Either is added to a live run with `extend` (below); finished units are kept.

## Once: copy the data and write the interpreter file

On the server, login node, first:
```
mkdir -p /data/stat-ecr/scat9264/dronerf /data/stat-ecr/scat9264/matbench/protocol /data/stat-ecr/scat9264/screen
```
Then on the Mac (you run scp):
```
Reactions need nothing copied for the views that are built on the server: the reaction
files download into `$KIRBY_CACHE_DIR/reaction_yields/` on first load, and the pool
views are read from `$KIRBY_CACHE_DIR/rep_cache/<dataset>/` (built there if missing,
through `_shared._load_data_impl`). cgr_smiles_lm needs the CGR-pretrained checkpoint,
already on the server at `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint`;
without it the unit fails rather than score base ChemBERTa. chytorch_graphormer and
cgr_smiles_lm are built into the run's own folder and never read from the rep cache
(the cached buchwald_hartwig chytorch_graphormer of 2026-05-01 is all zeros).
DOPtools' cgr_circus and cgr_chyline were dropped 2026-11-02: doptools needs
chython>=1.78 and chytorch needs the installed 1.68. The reaction libraries are not
installed by `setup.sh`; inside an srun, after sourcing it:
`python -m pip install -c constraints.txt chemprop==1.6.1` (chython, chytorch and
rxnmapper were already in py311-kirby on 2026-11-02). Run the import check in an srun,
not on the login node: there it died with "Illegal instruction".
```
scp -r /Volumes/seagate/benchmarks_protocol/dronerf/views scat9264@arc-login.arc.ox.ac.uk:/data/stat-ecr/scat9264/dronerf/
scp -r /Volumes/seagate/matbench/reps /Volumes/seagate/matbench/data scat9264@arc-login.arc.ox.ac.uk:/data/stat-ecr/scat9264/matbench/
scp -r /Volumes/seagate/benchmarks_protocol/matbench/rf /Volumes/seagate/benchmarks_protocol/matbench/modnet_v0.1.12 scat9264@arc-login.arc.ox.ac.uk:/data/stat-ecr/scat9264/matbench/protocol/
```
Sizes: DroneRF views 9.0 GB (the 22,700 chunks as hand-computed views plus the int16
frames every other view is built from; the raw CSVs are not needed), Matbench caches
1.8 GB, Matbench data 13 MB, leaderboard files 17 MB (only for the published means in
the summary).

On the server, login node:
```
cd /data/stat-ecr/scat9264/KIRBy
git pull
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python -c "import torch, torchvision, sklearn, xgboost, scipy, matplotlib, PIL; print('dronerf ok')"
python -c "import matminer, matbench, pymatgen; print('matbench ok')"
python -c "import chython, chytorch, rxnmapper, chemprop, lightgbm; print('reactions ok')"
```
If the second line fails, the Matbench units need another interpreter: put its path in
the interpreter file. The file is sourced by every screen job:
```
cat > tests/results/SCREEN_ENV <<'EOF'
export KIRBY_SCREEN_PY_MATBENCH=/path/to/python/with/matminer
EOF
```
Leave the file out when the py311-kirby venv has everything.

## The commands

1. Start a run (once; it also writes `CURRENT_SCREEN`). The paths are stamped into the
rows here, so later jobs need no environment of their own. Files already on disk (the
DroneRF views, the 60 Matbench caches) are marked ok at once.
```
export KIRBY_DRONERF_DIR=/data/stat-ecr/scat9264/dronerf
export KIRBY_MATBENCH_REPS=/data/stat-ecr/scat9264/matbench/reps
export MATMINER_DATA=/data/stat-ecr/scat9264/matbench/data
export KIRBY_MATBENCH_PROTOCOL=/data/stat-ecr/scat9264/matbench/protocol
python tests/experiments/screen/ledger.py init --run /data/stat-ecr/scat9264/screen/run_2026_11_01 --datasets dronerf matbench
```

2. Queued jobs. The first argument is how many; after `--` come the sbatch flags (time
defaults to 47:59:00, 8 CPUs, 64G). `DRY=1` prints the lines and submits nothing.
```
bash tests/slurm_scripts/submit_screen_workers.sh 6 -- --clusters=htc -p medium,long
bash tests/slurm_scripts/submit_screen_workers.sh 4 --dataset dronerf -- --clusters=arc -p medium
```
Jobs are named `ks` / `ks_<dataset>`, which `setup.sh` never redirects.

3. By hand, in an srun. Same worker, same ledger; when the session ends, a running unit
is picked up again by the next job or session.
```
srun --clusters=htc -p interactive --cpus-per-task=8 --mem=64G --time=03:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
[ -f tests/results/SCREEN_ENV ] && . tests/results/SCREEN_ENV
python tests/experiments/screen/worker.py
python tests/experiments/screen/worker.py --dataset matbench
python tests/experiments/screen/worker.py --dataset dronerf --unit spectrum__rf__10__grouped
python tests/experiments/screen/worker.py --dry-run
```
`--unit` resets a failed unit and runs only that one. `--stage build` keeps a session on
builds.

4. Read the state: `<run>/SCREEN.md`, or
```
python tests/experiments/screen/ledger.py status | less
python tests/experiments/screen/ledger.py views dronerf
python tests/experiments/screen/ledger.py show dronerf
```

5. Add a model, or a dataset once its spec exists, to the live run:
```
python tests/experiments/screen/ledger.py extend --datasets dronerf --models dronerf=svm
python tests/experiments/screen/ledger.py extend --datasets matbench --models matbench=modnet
```

Reactions, added to the live run (KIRBY_CACHE_DIR is set by `setup.sh`;
KIRBY_CGR_CHECKPOINT may be left unset, the checkpoint is found at the stat-cadd path):
```
export KIRBY_CGR_CHECKPOINT=/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint
python tests/experiments/screen/ledger.py extend --datasets reactions
```
Both paths are stamped into the row at that moment.

6. Take a view out everywhere, keeping its finished results:
```
python tests/experiments/screen/ledger.py skip-view alemadi_emb --reason "cost" --dry-run
python tests/experiments/screen/ledger.py skip-view alemadi_emb --reason "cost"
```

## When something fails

`SCREEN.md` has a "Failed units" table with the attempts, exit code, job, first error
line and log. Fix the code, `git pull` on the login node, then let the next job retry it
(one retry is automatic) or run it with `--unit`. `ledger.py reset DATASET UNIT` clears
a unit's record; `ledger.py release-claim DATASET` frees a dataset a dead job still
holds (jobs do this themselves when the holder is gone from squeue); `ledger.py sync`
marks units ok from files written outside the ledger.

## Adding a dataset

1. A screen script under `tests/experiments/benchmarks/` that follows the contract in
   `tests/experiments/screen/datasets.py`: one command per unit, the result file
   written last, evaluate files with `result_status: ok`, resumable.
2. A spec class in `datasets.py` (its units, commands, files and dependencies) added to
   `SPECS`.
3. `ledger.py extend --datasets <name>` on the live run, and a section in
   `NEW_DATASETS.md` that names the run's `summary.md`.

## Times measured so far (laptop, 2026-10-30 to 11-01)

These are the laptop's; the server's will differ. Per fold unless said.

| unit | time |
|---|---|
| DroneRF `extract` (once, from the 454 CSVs) | 2.8 h; not needed on the server (views copied) |
| DroneRF knn on the spectrum, 10 folds | 155 to 170 s per unit |
| DroneRF rf, xgboost, logreg on a 2,000-feature view | 85 to 160 s per fold |
| DroneRF dense, 20 epochs (debug) | about 100 s; the protocol is 200 epochs, not timed |
| DroneRF image view, rendering and ResNet50 | not timed on all chunks; 0.1 to 0.6 s per chunk to render |
| DroneRF RF-UAVNet build | about 40 min per epoch estimated from one batch; 5 epochs per fold |
| DroneRF Al-Emadi build | not timed; 350 epochs per fold |
| Matbench rf on magpie, jdft2d | 23 s per unit (5 folds) |
| Reactions: atom mapping, buchwald_hartwig (once per dataset, cached) | 22 min for 3,955 reactions |
| Reactions: cgr_atompair, cgr_torsion builds after mapping | 106 s, 57 s per dataset |
| Reactions: chytorch_graphormer build, buchwald_hartwig | not finished; about 2 h estimated from 200 reactions in 10 min, one thread |
| Reactions: chemprop_cgr build | about 4 min per epoch, one thread; about 2 h per fold at chemprop's 30 epochs; not finished |
| Reactions: rf evaluate on a 2,078-feature view | about 1 s per fold |
