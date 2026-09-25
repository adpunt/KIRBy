# Noise prelim: sample HTC scripts

**2026-09-25: the four-stage runbook is `../runbook/README.md`.** Stage 1 there
(`stage1_prelim_noise.py`) is the sweep: every noise experiment, every
condition, full folds and levels, one array task per experiment and condition.
The files here remain the one-fold, one-level check that a run is repeatable
and reads the precompute (`check_one.sbatch`), which the runbook uses as its
first step.

Samples for plan section 7, item 2. Run them your own way; nothing here is
wired into `submit_*.sh`, `run_dataset.py` or the precompute. Written
2026-09-25. None has been run on the cluster yet.

## The files

| File | What it is for |
|---|---|
| `run_one.py` | Runs one noise experiment on one dataset through `noise_helpers.run_all`, cut to one fold, one condition and one level. Writes `_run_one_manifest.json` saying what it cut. |
| `check_one.sbatch` | Batch job on htc. Runs `run_one.py` twice with identical settings, each with its own `KIRBY_NOISE_CACHE_REPORT`, then runs the checker. Exits 1 on any FAIL. |
| `check_one_srun.sh` | The same job as an `srun` on the `interactive` partition. |
| `prelim_array.sbatch` | Array job. One task per line of a text file: `<dataset> <experiment> [condition]`. |
| `prelim_tasks.example.txt` | An example task file (caco2, five lines). |
| `../../experiments/noise/check_noise_run.py` | The checker. |

## Why `run_one.py` exists

`run_dataset.py` can narrow a noise run to one condition with
`--noise-condition`. It has no flag for one fold or one level, and none was
added to it.

Two environment variables look like they would do it. Neither is safe for this
check:

- `KIRBY_LOCAL_FOLDS` also sets the fold count of the hybrid search
  (`_local_grid.py`). The search would then need a 3-fold search the
  precompute never stored. With `KIRBY_NO_COLD_SEARCH=1` it refuses; without it,
  it searches from nothing.
- `KIRBY_SMOKE_N` caps the number of molecules (`_shared.load_data`).

So `run_one.py` does the cutting itself, in its own process, before any
experiment module is imported:

- **Fold.** A split over every molecule of the dataset, with 2 or more folds,
  keeps fold 0 only. It is fold 0 of the same 5-fold split the full run makes. A
  split inside a training fold is left alone. Each cut is printed with the file
  and line that asked for it. The hybrid search makes its own folds and is not
  touched.
- **Level.** The level grids in `dataset_policy` are replaced by one level. The
  default is 0.5 of the label SD (`reference_noise_level`), 0.25 for
  censoring, and a 0.2 flip rate for classification. A level written inside an
  experiment's own file is not reached.
- **Condition.** Passed to `run_all` the same way `run_dataset.py` passes it. An
  experiment that names its own condition runs without one.

The result files still say `n_splits=5` in their `input_fingerprint`, because the
configuration is the full run's. So everything writes under
`tests/results/noise_prelim/`, never `tests/results/noise/`. A full run
resumes on any file it finds there.

`run_all` catches an experiment's exception and carries on, so a crashed
experiment used to exit 0. `run_one.py` exits 1 when that happens.

## What the checker prints

`check_noise_run.py` reads the two runs and prints three sections. It exits 1
on any FAIL.

- **(a)** Whether both runs finished, from the exit codes and each run's
  manifest. Whether rows (`d_vs_dprime.ROW_FIELDS`, in `*.jsonl` under the output
  directory) exist for every fold that ran. Whether run 2's rows equal run 1's,
  score for score.
- **(b)** Hits, misses and bypassed calls per cache kind, for run 1 and run 2. Run 2
  must be all hits. A run with no cache calls at all is a FAIL too, because it
  cannot show run 2 was all hits.
- **(c)** What the result file records: metric and direction, representations
  with their feature counts, tier, augmentation condition, budget, recipe,
  pruning mode, and the top single representations. Beside it, what the
  precompute stored for that dataset.

The stored side uses the same calls `run_all` and the experiments make:
`precompute_hybrid_searches`, then `get_best_search_for_downstream` through
`pool_recipe_per_downstream_model`, then `noise_base_reps_by_model`. It reads
them read-only:

- Four functions that fit a model or score features are replaced by ones that
  raise: `_evaluate_single`, `_evaluate_hybrid_config`,
  `compute_feature_importance` and `_evaluate_allocation`.
- `KIRBY_NO_COLD_SEARCH=1.0` refuses any search with fewer than every
  configuration stored.

Any attempt to compute is counted and printed as a FAIL. It means the stored
search does not hold everything the experiment asks for.

`--no-stored` skips the stored side and loads no data.

Two things the checker cannot see:

- Output rows exist only once an experiment is wired to `d_vs_dprime`.
- Cache counts exist only once it calls `noise_cache`.

Until then (a) and (b) FAIL for that experiment, and the message says which.

## HTC facts, and where each came from

| Fact | Source |
|---|---|
| Select htc with `#SBATCH --clusters=htc` | [job-scheduling.html](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html) |
| Or log in to `htc-login.arc.ox.ac.uk` (arc is `arc-login.arc.ox.ac.uk`) | [connecting-to-arc.html](https://arc-user-guide.readthedocs.io/en/latest/connecting-to-arc.html) |
| Partitions: `short` (1 h default, 12 h maximum), `medium` (12 h default, 48 h maximum), `long` (24 h default, 30 days maximum), `devel` (10 min maximum, "for batch job testing only"), `interactive` (24 h maximum, "can oversubscribe, for pre/post-processing and building software") | [job-scheduling.html](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html) |
| Default memory is 8000 MB per CPU | [job-scheduling.html](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html) |
| htc compute nodes "typically have 16, 20 or 48 cores"; arc nodes typically 48 | [job-scheduling.html](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html) |
| htc has 124 worker nodes, 51 of them GPU nodes. CPU node types listed: Intel Cascade Lake 48 cores / 384 GB; AMD EPYC 64 to 168 cores / 1.1 to 2.3 TB; other Intel generations 8 to 96 cores / 128 GB to 1.5 TB; one Grace Hopper node, 72 cores / 580 GB | [arc-systems.html](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html) |
| Two high-memory htc nodes, requested with `--mem` between 400G and 3000G | [arc-systems.html](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html), [job-scheduling.html](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html) |
| GPUs exist only on htc; co-investment GPU nodes are limited to `short` (12 h) | [job-scheduling.html](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html) |
| Arrays: `#SBATCH --array=1-4`, `SLURM_ARRAY_TASK_ID`, `%A_%a` in file names. The page states no limit on array size or on jobs per user. | [slurm-reference.html](https://arc-user-guide.readthedocs.io/en/latest/slurm-reference.html) |
| On the free Basic service: 24 h maximum per job, and only one job running at a time | [slurm-faq.html](https://arc-user-guide.readthedocs.io/en/latest/slurm-faq.html) |

Not from the ARC pages; read these from the cluster:

- Real array limit: `scontrol show config | grep -i MaxArraySize`
- Live wall limits on htc: `sinfo -M htc -o "%P %l %D %c %m"`
- Your jobs on htc: `squeue -M htc -u $USER`

`where_to_submit.sh` reports on the cluster of the login node you run it from,
except `--best-cluster-account`, which reads arc and htc both.

Repo conventions these scripts follow:

- No `--account` in any header. Fair share is per cluster and per account
  (measured 2026-09-24: htc stat-cadd 0.0491, htc stat-ecr 0.0360, arc
  stat-cadd 0.0326, arc stat-ecr 0.0195), so the account is added on the sbatch
  line when wanted: `bash tests/slurm_scripts/where_to_submit.sh --best-cluster-account htc`
  prints the cluster and account. Without one SLURM bills the default
  association.
- `KIRBY_SKIP_INSTALLS=1` before `. setup.sh` (full mode re-installs into the
  shared venv).
- No `--output`, so logs are SLURM's default `slurm-<jobid>.out`, or
  `slurm-<arrayid>_<task>.out` for the array.
- `. setup.sh` from the repo root.
- `KIRBY_NO_COLD_SEARCH=1`.
- `--pruning-mode calibrated`, the value `queue_noise_conditions.py` passes.
- `KIRBY_LGBM_THREADS=1`.
- `OMP_NUM_THREADS` stays at setup.sh's value (the CPU count).

The search speed for each dataset comes from `dataset_policy.FAST_SPEED_DATASETS`.
Nothing here sets it; `_shared` applies it to every search read, and the checker
prints it.

`KIRBY_LGBM_THREADS` is part of every `noise_cache` key, as `lgbm_n_jobs`. A
later run shares these cache entries only if it uses the same value.

## Commands

From the repo root on the server (`/data/stat-ecr/scat9264/KIRBy`), after
`git pull`:

```bash
# one check, batch
sbatch tests/slurm_scripts/noise_prelim/check_one.sbatch caco2 n07 outlier_p10
sbatch tests/slurm_scripts/noise_prelim/check_one.sbatch caco2 n18
sbatch --mem=128G tests/slurm_scripts/noise_prelim/check_one.sbatch qm9 n07 gaussian

# the same, interactive
bash tests/slurm_scripts/noise_prelim/check_one_srun.sh caco2 n07 outlier_p10
# which runs:
srun --clusters=htc -p interactive --cpus-per-task=8 --mem=64G --time=12:00:00 \
     --job-name=np_check_i bash tests/slurm_scripts/noise_prelim/check_one.sbatch caco2 n07 outlier_p10

# array, one task per line
TASKS=tests/slurm_scripts/noise_prelim/prelim_tasks.example.txt
N=$(grep -cvE '^\s*(#|$)' "$TASKS")
sbatch --array=1-${N}%20 --export=ALL,TASKS=$TASKS \
       tests/slurm_scripts/noise_prelim/prelim_array.sbatch
# every fold and level instead of one of each:
sbatch --array=1-${N}%20 --partition=medium --time=47:59:00 \
       --export=ALL,TASKS=$TASKS,PRELIM_CUT=none \
       tests/slurm_scripts/noise_prelim/prelim_array.sbatch

# the checker alone, on two runs already on disk
python tests/experiments/noise/check_noise_run.py --dataset caco2 --experiment n07 \
    --condition outlier_p10 --out1 <dir>/run1 --out2 <dir>/run2 \
    --report1 <dir>/cache_run1.json --report2 <dir>/cache_run2.json --pruning-mode calibrated
```

Output lands in `tests/results/noise_prelim/<dataset>__<experiment>__<condition>__<jobid>/`.
It holds `run1/`, `run2/`, `cache_run1.json`, `cache_run2.json` and
`check.txt`. Array output goes to `tests/results/noise_prelim/array_<id>/`
instead.

Memory: `queue_noise_conditions.py` gives 128G to `nt_*`, `gb_*`, `tcga_*`,
`kiba`, `bindingdb_kd` and `qm9`. The samples ask for 64G, which is 8 CPUs at
ARC's 8 GB each.

n15 runs its own hybrid searches on noisy labels. `KIRBY_NO_COLD_SEARCH` does
not refuse those, so an n15 check pays for them inside the job.
