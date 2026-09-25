# Runbook: the four stages

Written 2026-09-25. Nothing here has been run on the cluster yet. Every script
below was checked locally with `bash -n`, a `--dry-run`, and stand-in `sbatch`,
`squeue`, `sshare` and `sacctmgr` commands.

| Stage | What runs | Script | Results go to |
|---|---|---|---|
| 1 | every noise experiment, every condition, on the trial datasets | `stage1_prelim_noise.py` | `tests/results/noise_prelim_sweep/` |
| 2 | complexity, similarity, federated and the 11 OPIG scripts, on two trial datasets | `stage2_trials.sh` | `tests/results/trial_<category>/` |
| 3 | complexity, similarity, federated and OPIG on every dataset whose precompute is finished | `stage3_full.sh` | the real tree, `tests/results/<category>/` |
| 4 | every noise experiment on every dataset whose precompute is finished, with the random and shuffled control twins | `stage4_noise.sh` | the real tree, `tests/results/noise/` |

Stages 1 and 2 never write to the real results tree. Every script takes its
datasets as an argument. `herg_ki` (regression, 1,412 molecules) and `herg`
(classification, 7,772 molecules) appear in the examples. No small
classification dataset is eligible for every category. Federated has silo data
only for `lhasa_herg`, `caco2` and `logd`.

---

## Before any stage

```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin similarity-metrics-study
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh

bash tests/slurm_scripts/where_to_submit.sh --best-cluster-account   # prints e.g. "htc stat-cadd"
python tests/tools/precompute_status.py herg_ki herg                 # DONE / NOT DONE per dataset
```

Pull once, by hand. Every job the runbook submits sets `KIRBY_NO_PULL=1`,
because hundreds of array tasks starting together would each fetch into one
`.git`. Queued jobs read the Python files when they start, so a pull before
they start reaches them. The shell scripts are copied at submit time, so a pull
does not reach those (`SUBMIT.md`, rule 2).

One check before stage 1: run one experiment twice on one fold and confirm the
second run gets every noise-cache entry from the first
(`../noise_prelim/README.md`):

```bash
sbatch tests/slurm_scripts/noise_prelim/check_one.sbatch herg_ki n07 outlier_p10
```

---

## Stage 1: prelim noise experiments on the trial datasets

```bash
python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --dry-run
python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --max-concurrent 40
bash   tests/slurm_scripts/runbook/progress.sh tests/results/noise_prelim_sweep logs/noise_prelim
```

**What one task is.** One noise experiment, on one dataset, under one noise
condition, in its own Python process. It calls `submit_dataset.sh noise
<dataset> <experiment>` with `NOISE_CONDITION` set. It runs every fold and every
level.

**Which tasks.** Every experiment in `noise_helpers.SINGLE_DATASET_EXPERIMENTS`:

- An experiment that takes a condition gets one task per condition: 14 shapes
  for a regression dataset, 6 flip models for a classification dataset
  (`noise_helpers.conditions_for_task`).
- n21 gets only the conditions in `DETECTION_IS_DEFINED`, because it raises
  under the others (`n21_neighbour_distance.py:671-677`).
- An experiment that names its own condition gets one task with no condition.
- A second set of arrays starts after the first set has ended:
  - n06, because it reads the rankings n11 and n13 write.
  - n21, n23, n32, n40 and n41 with no condition. Each of these also sweeps
    conditions that `--noise-condition` cannot pass: `shift_only_*`,
    `sparse_dprime_*` and `dense_oofmse`. A condition whose file stage 1 already
    wrote is skipped, so these tasks compute only the extra conditions.

The dry run on 2026-09-25 counted 257 tasks for herg_ki and 121 for herg.
The count follows the experiment list at submit time. n42 is registered in
`SINGLE_DATASET_EXPERIMENTS`, but on 2026-09-25 its file did not exist yet.
Until it lands, every n42 task fails at import.

**Resources.** 8 CPUs, 47:59:00, partition medium, one task per array element.
Memory is 64G per task, or 128G for `nt_*`, `gb_*`, `tcga_*`, `kiba`,
`bindingdb_kd` and `qm9`, the same split as `queue_noise_conditions.py:80`.
Each (phase, memory) pair gets its own array, split into arrays of at most
1,000 tasks. SLURM's default `MaxArraySize` is 1001. ARC's real value comes from
`scontrol show config | grep -i MaxArraySize`. `--max-concurrent N` caps the
tasks running at once in each array, not across all of them.

**Cluster and account.** htc by default. The account defaults to the one with
the best fair share on htc. `--clusters best` picks the cluster as well. An
explicit `--account` wins. `--account none` passes none.

**What it writes.**

| Path | What |
|---|---|
| `tests/results/noise_prelim_sweep/<experiment dir>/` | the result files |
| `tests/results/noise_prelim_sweep/_job_summaries/<array>_<task>.tsv` | one row per experiment: exit code, status, seconds, log |
| `tests/results/noise_prelim_sweep/_job_summaries/<array>_<task>.running` | the experiment a task is inside now |
| `tests/results/noise_prelim_sweep/_cache_reports/<ds>__<exp>__<cond>__<array>_<task>.json` | noise-cache hits and misses for that task |
| `tests/results/noise_prelim_sweep/_tasks/ns1_<time>/` | the task files and `env.sh` the arrays read |
| `logs/noise_prelim/noise_<ds>_<exp>__<cond>.log` | that experiment's own output |
| `slurm-<array>_<task>.out` in the directory you submitted from | SLURM's log of the whole task |

**Rerun.** Run the same command again. `run_dataset.py` skips any experiment
whose result file is already on disk with the same input fingerprint. A task
that finished costs one data load and one read of the stored searches.

---

## Stage 2: proof of concept for complexity, similarity, federated and OPIG

```bash
# prints one srun line per (category, dataset) and per OPIG script; paste each into its own tmux window
bash tests/slurm_scripts/runbook/stage2_trials.sh --datasets "herg_ki herg" --fed-datasets "caco2"

# or as batch jobs: print, then submit
bash tests/slurm_scripts/runbook/stage2_trials.sh --datasets "herg_ki herg" --fed-datasets "caco2" --sbatch
bash tests/slurm_scripts/runbook/stage2_trials.sh --datasets "herg_ki herg" --fed-datasets "caco2" --sbatch --go

# which experiments are running, finished, or failed
bash tests/slurm_scripts/runbook/stage2_trials.sh --status
```

- complexity and similarity: `submit_dataset.sh <category> <dataset>`. Each
  experiment runs in its own Python process, one after another.
- federated: the same, on `--fed-datasets`. It must be one or more of
  `lhasa_herg`, `caco2` and `logd`.
- OPIG: `opig_one.sh <name>` for each of the 11 names in `OPIG_JOBS`
  (`submit_everything.sh:127-139`), with that table's flags. Each script reads
  its own datasets (`bash tests/slurm_scripts/runbook/opig_one.sh --list`).
  `--datasets` does not change them.

srun mode: partition interactive, 12 h. sbatch mode: partition short, 11:59:00.
A whole category on one dataset may need longer. `--partition medium --wall
47:59:00` gives it two days. Everything runs on htc by default, billed to the
account with the best fair share there.

If `srun` on the arc login node refuses `--clusters=htc`, log in to
`htc-login.arc.ox.ac.uk` and pass `--clusters ""`.

**What it writes.** Results go to `tests/results/trial_<category>/`. Exit codes
go to `tests/results/trial_<category>/_job_summaries/`. Each experiment's log is
`logs/trial/<category>_<dataset>_<experiment>.log`, and each OPIG script's is
`logs/trial/opig_<name>.log`.

---

## Stage 3: the full suite except noise

```bash
bash tests/slurm_scripts/runbook/stage3_full.sh --dry-run
bash tests/slurm_scripts/runbook/stage3_full.sh
bash tests/slurm_scripts/runbook/stage3_full.sh --opig-only        # later, once the OPIG datasets pass
```

1. Cluster and account: the best fair share over arc and htc.
2. Precompute gate. A dataset goes forward only when
   `tests/tools/precompute_status.py` calls it DONE and no `pc_` job is
   still queued or running for it on either cluster. Every dataset held back is
   printed with its reason.
3. `submit_everything.sh --experiments-only --categories "complexity similarity"`
   runs on the datasets that passed. Federated jobs are added for `lhasa_herg`,
   `caco2` and `logd` when they are in that list.

**Why a gate and not a dependency.** A SLURM dependency cannot name a job on
the other cluster. `submit_everything.sh` looks for queued precompute jobs only
on the cluster it submits to. Without the gate, an experiment job on htc could
start while a precompute job on arc is still writing that dataset's searches.

**Memory.** `submit_everything.sh` passes no `--mem` for experiment jobs, so
they get `submit_dataset.sh`'s 64G. The six large dataset families go in a
second call with `SBATCH_MEM_PER_NODE=128G`.

**OPIG.** `submit_everything.sh` queues all 11 OPIG scripts or none. They are
queued only when every dataset any of them reads has passed the gate. Otherwise
the script prints which OPIG script waits for which dataset. Run again with
`--opig-only` later. `--include-opig` queues them anyway.

---

## Stage 4: the full noise suite

```bash
bash tests/slurm_scripts/runbook/stage4_noise.sh --dry-run
bash tests/slurm_scripts/runbook/stage4_noise.sh --max-concurrent 200
bash tests/slurm_scripts/runbook/progress.sh tests/results/noise logs/noise_full
```

These are stage 1's arrays, on every dataset that passes the stage-3 gate (at
most 41: `DATASET_REPS` minus `chembl_qsar` and `davis`). They run with
`KIRBY_NOISE_CONTROLS=1`, write to `tests/results/noise/`, log to
`logs/noise_full/`, and have job names `ns4_*`. A dry run over all 41 datasets
with the gate off counted 8,497 tasks in 11 arrays on 2026-09-25.

The noise cache is keyed on a hash of the label vector, not on the output
directory. Stage 4 therefore reuses what stage 1 computed on the trial datasets,
provided `KIRBY_LGBM_THREADS` matches. It is 1 in every job here.

---

## Server checks: every experiment, briefly, many at once

Added 2026-09-25. Not run on the cluster yet. Both scripts pass `bash -n`;
nothing else was tried.

These are dirty checks, not results. Each experiment gets one short srun. The
time limit may cut it off. A check that was cut off with no error in its log
counts as progressing, and the last line of its log shows how far it got.

### The commands, in order, on arc-login

```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin similarity-metrics-study      # the script pulls again; this one gets the script itself

# 1. print every srun line and the count; launches nothing
bash tests/slurm_scripts/runbook/server_checks.sh --dry-run

# 2. launch; returns at once, and a background dispatcher keeps 20 running
bash tests/slurm_scripts/runbook/server_checks.sh

# 3. the table, as often as you like; also after logging back in
bash tests/slurm_scripts/runbook/server_checks_status.sh
bash tests/slurm_scripts/runbook/server_checks_status.sh --not-passed
bash tests/slurm_scripts/runbook/server_checks_status.sh --tail noise_herg_ki_n07
```

The defaults are `--datasets "herg_ki herg"`, `--time 02:00:00`, `--partition
short`, `--max-parallel 20`, `--clusters htc` and the account with the best fair
share on htc. When `where_to_submit.sh` cannot say which account that is, the
account is `stat-cadd`. `--help` lists every option.

Before launching, the script runs one 1-minute srun with the same cluster,
account and partition. If srun on arc-login rejects `--clusters=htc`, nothing is
launched. Log in to `htc-login.arc.ox.ac.uk` and add `--clusters ""`.

### What one check runs

| Category | One srun per | What runs | Results |
|---|---|---|---|
| noise | dataset x experiment, every experiment in `noise_helpers.SINGLE_DATASET_EXPERIMENTS`, n42 included | `noise_prelim/run_one.py` on fold 0, one condition and one level, run twice, then `check_noise_run.py` | `tests/results/server_checks/noise/<dataset>__<experiment>/` |
| similarity | dataset x experiment in `similarity_helpers.SINGLE_DATASET_EXPERIMENTS` | `submit_dataset.sh similarity <dataset> <experiment>`, which runs `run_dataset.py --experiments <experiment>` | `tests/results/trial_similarity/` |
| federated | the same, on `--fed-datasets` | `submit_dataset.sh federated <dataset> <experiment>` | `tests/results/trial_federated/` |
| opig | row of `OPIG_JOBS` (`submit_everything.sh:127-139`) | `opig_one.sh <name>`, with that row's flags and memory | `tests/results/trial_opig/` |

Each srun sources `setup.sh` with `KIRBY_SKIP_INSTALLS=1`. So every check reads
the one cache at `/data/stat-ecr/scat9264/.cache/kirby`, whichever account pays.
Every check also sets `KIRBY_NO_COLD_SEARCH=1` and `KIRBY_NO_PULL=1`.

- The noise condition is `outlier_p10` on a regression dataset. On a
  classification dataset it is the first classification condition. Only
  experiments that take a condition get one. Change it with `--cond-reg` and
  `--cond-clf`.
- n21 and n25 run only under the conditions in `CONDITION_LIMITED_TO`.
  `outlier_p10` is one of them. On herg (classification) none of them is.
  The comment at `noise_helpers.py:1507-1515` says run_all then writes a skip
  instead of running. That has not been run. A skip shows as passed, with
  "skipped" in the detail.
- `KIRBY_SIMILARITY_SETTINGS` is left unset, so the noise experiments sweep every
  similarity setting. Within 2 hours most of them will not finish run 1, so
  the cache check will not be reached. `--similarity-settings model_input` runs
  only today's neighbours; the choice is yours.
- n06 reads the result files n11 and n13 write in the same output directory
  (`n06_bindingdb_ground_truth.py:545`). Each check has its own directory, so
  n06 finds neither, and that half of n06 is not exercised here.
- federated: only `lhasa_herg`, `caco2` and `logd` have silo data. When none of
  `--datasets` is one of them, `caco2` is used, and the launch says so.
- similarity, federated and opig run with `KIRBY_FORCE_RERUN=1`. Without it, a
  result file from stage 2 in `trial_<category>/` would be skipped, not rerun.
  `--no-force` turns it off.
- complexity is not in the default set. `--categories "noise similarity
  federated opig complexity"` adds it.

### Where it writes

- `logs/server_checks/<category>_<dataset>_<experiment>.log` is the whole srun
  output. OPIG rows use `opig_own_<name>`.
- `logs/server_checks/<...>.rc` holds the srun's exit code; there is no `.rc`
  until it ends.
- `logs/server_checks/<...>.steps` (noise only) holds the exit codes of run 1,
  run 2 and the check.
- `logs/server_checks/commands.tsv` lists every planned command.
  `_dispatcher.log` and `_dispatcher.pid` belong to the background dispatcher.

A new launch first moves the previous `logs/server_checks/` and
`tests/results/server_checks/` to `..._previous/<time>/`. Nothing is deleted. A
launch refuses to start while the previous dispatcher is still alive. To stop
it: `kill $(cat logs/server_checks/_dispatcher.pid)`. srun commands it already
started keep running.

### What "passed" means

For a noise experiment, "passed" needs all four of these:

1. Neither run crashed. Both `run_one.py` runs exit 0.
2. Rows are on disk. `check_noise_run.py` (a) finds the result file and the
   per-fold rows for fold 0, and run 2's rows equal run 1's.
3. The second run was all cache hits. (b) finds no miss and no bypass in run
   2's `KIRBY_NOISE_CACHE_REPORT`. The status table shows this as `cache`.
4. The inputs it read match the precompute. (c) compares the recipes,
   representations, budget, pruning mode, metric and direction in the result
   file with what the precompute stored. That comparison is read-only, and any
   model fit it attempts is counted as a failure.

`check` in the table is PASS, or FAIL with a count per section (`a`, `b`, `c`).

For similarity, federated and OPIG, "passed" means only one thing: exit 0 with
no `_errors/` file for that dataset and experiment. `submit_dataset.sh` counts
an `_errors/` file as a failure itself. There is no second run and no cache
check for these.

The `state` column in the table takes these values:

| state | meaning |
|---|---|
| waiting | not started; at most `--max-parallel` run at once |
| running | srun alive; detail = minutes since the log last changed, and its last line |
| passed | as above |
| cut_off | stopped by the time limit; detail says "progressing, no error in log" with the last line, or "error before the cut" with the error |
| failed | any other non-zero exit; detail = the last error line of the log (for noise, which of run 1, run 2 or the check) |

---

## Checking progress and failures

```bash
bash tests/slurm_scripts/runbook/progress.sh <results dir> [<results dir> ...] <log dir>
```

For each results directory it prints the status of each experiment's latest
run, then every failure with its exit code and log. It also prints the
experiment each running job is inside, with that log's last line, and any file
in `_errors/`. For a log directory it prints each log written in the last hour
with its last line. Finally it prints your job counts on arc and htc.

Exit codes in the summaries:

| Exit | Status | Meaning |
|---|---|---|
| 0 | `ok` | finished |
| 1 | `failed` or `error_file:<name>` | Python error, or `run_dataset.py` wrote an `_errors/` file for this dataset and experiment |
| 124 | `timeout` | hit `EXP_TIMEOUT` |
| 137 | `killed_9_oom_or_wall` | killed with signal 9, usually out of memory |
| 139 | `segfault` | segmentation fault; LightGBM and torch in one process at more than one LightGBM thread (`hybrid.py:57-73`) |
| `-` | `not_applicable` | not run under this condition, with the reason |

```bash
sacct -M htc -u $USER --starttime today -o JobID%20,JobName%24,State,ExitCode,MaxRSS,Elapsed | grep -E 'ns1_|ns4_|t2_'
squeue -M arc,htc -u $USER -o '%.18i %.24j %.9P %.2t %.10M %.10L %R'
```

---

## `submit_dataset.sh`, changed 2026-09-25

- **One process per experiment.** Before, one crash in one experiment killed
  every later experiment in the same job: a segfault (exit 139) or running out
  of memory. Now it kills only the experiment it happens in. Each process
  reloads the dataset and re-reads the stored searches. How long that takes has
  not been measured.
- **Exit codes.** Each experiment's exit code goes to
  `<output>/_job_summaries/<jobid>.tsv`. The job exits non-zero when any
  experiment failed.
- **Resources.** The defaults are 8 CPUs, 64G, 47:59:00, partition medium. They
  were 32 CPUs, 128G, 7 days, partition long, which fits neither medium's 48 h
  limit nor htc's 16- and 20-core nodes.
- **Account.** There is no account in the header. The old header named
  stat-ecr, which had the lowest fair share of the four on 2026-09-24.
- **Environment defaults.** `KIRBY_SKIP_INSTALLS=1` and `KIRBY_LGBM_THREADS=1`.
  The file gives the reasons for each.
- **New options.** `KIRBY_NO_PULL=1` skips the git pull. `KIRBY_OUTPUT_DIR`
  sets where results go. `KIRBY_EXP_LOG_DIR` writes one log per experiment.
  `EXP_TIMEOUT` sets a wall for each experiment.

`KIRBY_LGBM_THREADS=1` is a change for every category. Before, the LightGBM fits
inside `hybrid.py` used every core (-1) in these jobs. Whether that changes any
score has not been measured. No LightGBM call in `src/kirby/hybrid.py` sets
`deterministic`.

---

## Fair share: which cluster and account

Fair share is kept per cluster and per account. It was measured on 2026-09-24;
a higher number starts sooner:

| Cluster | Account | FairShare |
|---|---|--:|
| htc | stat-cadd | 0.0491 |
| htc | stat-ecr | 0.0360 |
| arc | stat-cadd | 0.0326 |
| arc | stat-ecr | 0.0195 |

Re-measure before a large submission:

```bash
sshare -M arc,htc -u $USER -o Cluster,Account,User,FairShare
bash tests/slurm_scripts/where_to_submit.sh --best-cluster-account
```

The account has nothing to do with the `/data/stat-ecr` paths. Those are a
filesystem, and every job reads `KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby`
whichever account pays. `where_to_submit.sh --emit` reads the login node's
cluster only. `--best-cluster-account` reads both clusters.

htc has CPU nodes: 124 worker nodes, 51 of which are GPU nodes (ARC docs,
`arc-systems.html`). The runbook sends every CPU job to htc by default.

---

## GPU: what would move, and what it would change

Findings, each from the line cited:

- **The dnn model is CPU only.** `src/kirby/models/dnn.py:160` and `:317` set
  `device = torch.device('cpu')`.
- **The label-noise DNN methods are CPU only.**
  `src/kirby/utils/lnl_methods.py:138` and `:252` set
  `device = torch.device('cpu')`.
- **deeplift runs on the CPU.** Its tensors in `src/kirby/hybrid.py:3636-3760`
  are created with no device, which torch puts on the CPU.
- **xgboost and lightgbm name no device.** None is set in the model factories
  (`tests/experiments/_shared.py:3088-3091`) or in the fits inside
  `src/kirby/hybrid.py` (`:2909-2928`, `:5068-5071`, `:5136-5139`). They run on
  the CPU.
- **svm, rf and treeshap have no GPU path here.** svm is scikit-learn's
  SVR/SVC and rf is scikit-learn's random forest; scikit-learn has no GPU path
  for either. treeshap uses shap's tree explainer on the CPU.
- **Which of these takes the time in the noise jobs has not been measured.**
  svm, rf, tuned xgboost, treeshap and deeplift were named in the handoff for
  this work; none was timed in this session. For the precompute search,
  `SUBMIT.md` records xgboost or lightgbm as the most expensive model on every
  one of five datasets (23.8% to 56.5% of the time).
- **A GPU would change the numbers.** The precompute's stored scores were
  computed on the CPU. XGBoost's documentation does not promise that its CUDA
  build gives the same numbers as its CPU build. A dnn trained on CUDA would
  not repeat the CPU run's floating-point results either. Moving either to a
  GPU would make every score differ from the scores cached by the precompute.
  Not tested here.

So a job does not move to a GPU by itself. The runbook runs the CPU jobs on
htc's CPU nodes under the account with the best fair share there.
