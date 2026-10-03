# Submitting complexity, similarity, federated and OPIG on 2026-10-03

Written 2026-10-03. Nothing here has been run on the cluster. The commands were dry-run on the laptop, with stand-in `squeue` and `sbatch` commands.

What the submit command does, in four lines:
- It sends one job per (category, dataset), plus one job per OPIG script, plus one job each for s09, c03 and c14, which read several datasets.
- Each job waits for that dataset's `pc_` jobs to end, on whichever cluster they are on.
- It bills stat-ecr.
- It writes to a new folder you name, so no result from older code is reused.

---

## 1. Before submitting, once

The changes this doc relies on are in commit `31715d4`, pushed 2026-10-03. Later commits on the branch include it.

Then, on the htc login node:

```bash
cd /data/stat-ecr/scat9264/KIRBy && git pull --ff-only origin similarity-metrics-study
```
```bash
git log -1 --oneline
```
The second line must print `31715d4` or a later commit.

```bash
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
```
```bash
echo $KIRBY_CACHE_DIR
```
That must print `/data/stat-ecr/scat9264/.cache/kirby`.

Optional: which datasets have finished their precompute, and which `pc_` jobs are still queued on each cluster.

```bash
python tests/tools/precompute_status.py
```

---

## 2. The commands

Pick the folder name once. These commands use `tests/results/run_2026_10_03`. Use the same name every time you resubmit this run.

**Dry run first.** It prints every `sbatch` line and submits nothing.

```bash
bash tests/slurm_scripts/runbook/submit_all_stages.sh --stages 3 --no-hold-stage3 --account stat-ecr --cluster htc --output-root tests/results/run_2026_10_03 --dry-run
```

Read three things in its output:
- `dataset order` lists the datasets.
- Each job line ends with the cluster and the `pc_` job numbers it waits for.
- A line ending in `HELD` means that dataset has `pc_` jobs on both clusters (see below).

**Then submit.** It is the same line without `--dry-run`.

```bash
bash tests/slurm_scripts/runbook/submit_all_stages.sh --stages 3 --no-hold-stage3 --account stat-ecr --cluster htc --output-root tests/results/run_2026_10_03
```

It submits 99 jobs when all 41 datasets are in the roster:
- 41 similarity jobs;
- 3 federated jobs, for lhasa_herg, caco2 and logd;
- 41 complexity jobs;
- 11 OPIG jobs;
- 3 jobs that read several datasets: s09, c03 and c14 (section 2a).

The job numbers are saved to `logs/submit_all_stages_<date>_<time>.txt`.

**One category at a time instead.** Add `--categories` with one of `complexity`, `similarity`, `federated`, `opig` or `cross`. `cross` is the three jobs of section 2a; `s09`, `c03` or `c14` submits one of them. For example:

```bash
bash tests/slurm_scripts/runbook/submit_all_stages.sh --stages 3 --no-hold-stage3 --account stat-ecr --cluster htc --output-root tests/results/run_2026_10_03 --categories complexity
```

### What each job gets

| Setting | Value |
|---|---|
| account | stat-ecr |
| partition | medium,long |
| wall | 47 h 59 min |
| CPUs | 8 |
| memory | 64 GB; 128 GB for nt_*, gb_*, tcga_*, kiba, bindingdb_kd, qm9 and the ginkgo and graphinity OPIG scripts |
| waits for | every `pc_` job of its datasets still queued or running, with `afterany` |
| results | `tests/results/run_2026_10_03/<category>/`, and `.../opig/` |
| one log per experiment | `logs/run_2026_10_03/<category>_<dataset>_<experiment>.log`, and `opig_<name>.log` |
| SLURM's own log | `logs/run_2026_10_03/slurm/<job name>_<job number>.out` |
| job comment | `run_2026_10_03`, the folder's name |

`afterany` means the job starts when the `pc_` jobs end, whether they passed or failed.

### 2a. The three jobs that read several datasets

s09, c03 and c14 are in no per-dataset job. `complexity_helpers.py` and `similarity_helpers.py` leave them out of their experiment lists, so `run_dataset.py` refuses them by name. Stage 3 now sends one job for each, after the OPIG jobs. Each runs `runbook/cross_one.sh`.

One row is one job:

| Job | What it reads | Waits for (`afterany`) | Datasets | Partition, wall | CPUs, memory | Writes |
|---|---|---|---|---|---|---|
| `x_cross_s09` | the s02, s04, s05, s06 and s08 result files in `<output-root>/similarity/` | every `x_similarity_*` job of this `--output-root`: the ones this command submits, and any still queued from an earlier paste | all 41, the similarity roster | short, 2 h | 1, 16 GB | `similarity/s09_cross_dataset_summary/results.json` |
| `x_cross_c03` | the precompute's stored searches | every `pc_` job of all 41 datasets | all 41, the complexity roster | long, 7 days | 8, 256 GB | `complexity/c03_cross_dataset_transfer/multi.json` |
| `x_cross_c14` | nothing; it runs its own searches | nothing | logd, caco2, herg, flab_thermo_tresanco, tcga_luad_vs_lusc | long, 29 days 23 h | 8, 128 GB | `complexity/c14_cross_domain_summary/results.json` |

Where each choice comes from:
- **s09's datasets.** Its code reads whichever of the five files exist for each dataset named, and skips a dataset with none. It restricts nothing, so it gets every dataset the similarity jobs cover.
- **c03's datasets.** Its code and docstring give no fixed list (its default is logd, caco2 and herg_ki). So it gets every dataset the complexity jobs cover. It loads all of them into memory before it starts, hence 256 GB. That figure is not measured.
- **c03's wall.** It fits SVR and ElasticNet on 5 folds of every regression dataset, kiba and bindingdb_kd included, and saves nothing until the end. Its time is not measured, so it goes to long rather than risk medium's 48 h.
- **c14's datasets.** Its `--tasks` has no default. The five are the union of the two usage lines in its docstring, which name the three domains it compares: molecular, antibody and TCGA. Change them with `--c14-tasks "A B ..."`.
- **c14's wall.** It runs two nested searches per dataset, neither cached, and writes one file at the very end. A job cut off by its wall keeps nothing. The 2026-09-18 compute audit counted about 1.84 million model fits per dataset for the first nested search alone (`EXPERIMENT_COMPUTE_AUDIT_2026_09_18.md`). Its time is not measured; long allows 30 days at most.
- **c14 and `KIRBY_NO_COLD_SEARCH=1`.** Every stage-3 job carries it. It does not stop c14: c14 passes no cache setting, `find_best_hybrid` defaults to no cache, and the refusal is only checked when the cache is on (`find_best_hybrid.py`, the `if cache and not _own` line).
- **c03 sees no augmented searches.** It loads each dataset without its augmentation representations, so it reads only the stored search without augmentation. The per-dataset complexity jobs load them. That is how `c03_cross_dataset_transfer.py` is written; it is not changed here.

s09 and c03 can wait for jobs on both clusters. The same rule as below applies: such a job goes to htc, HELD, and its line names the arc jobs.

**Submitting just these three later**, into the same folder:

```bash
bash tests/slurm_scripts/runbook/submit_all_stages.sh --stages 3 --no-hold-stage3 --account stat-ecr --cluster htc --output-root tests/results/run_2026_10_03 --categories cross
```

s09 then waits for the `x_similarity_*` jobs of that folder still queued or running. If none are left, it starts at once and reads what is there.

### Which cluster each job goes to

A SLURM job can only wait for jobs on its own cluster. So the script reads the `pc_` jobs on both arc and htc:
- A dataset with `pc_` jobs on one cluster goes to that cluster.
- A dataset with no `pc_` jobs left goes to htc, and may start at once.
- A dataset with `pc_` jobs on both clusters goes to htc, HELD. Its line names the arc jobs. Release it once they have ended, with the command the script prints at the end.

An OPIG script that reads several datasets follows the same rule over all of them.

### When a dataset's precompute did not finish

The job still runs. An experiment that needs a search the precompute did not store builds it itself if at most 500 configurations are missing. Above that it refuses in minutes, and the refusal is written to `_errors/`.

### Jobs from earlier submissions

Old held `x_*` and `opig_*` jobs do not stop these. The script only refuses to submit a job whose name and comment match one still queued or running. Old jobs have no comment. What to do with the old held jobs is your call. If one is released, it writes to its own old folder, not to this one.

---

## 3. Checking progress

```bash
bash tests/slurm_scripts/runbook/progress.sh tests/results/run_2026_10_03/similarity tests/results/run_2026_10_03/federated tests/results/run_2026_10_03/complexity tests/results/run_2026_10_03/opig logs/run_2026_10_03
```

For each results folder it prints:
- how many experiments finished, by status;
- every failure, with its exit code and log;
- the experiment each running job is inside now, with the last line of its log;
- the files in `_errors/`.

For the log folder it prints each log written in the last hour, with its last line. At the end it counts your jobs on arc and on htc.

The queue itself:

```bash
squeue -M arc,htc -u $USER -o '%.18i %.30j %.9P %.2t %.10M %.10L %R'
```

Jobs that have ended today, with their memory:

```bash
sacct -M arc,htc -u $USER --starttime today -o JobID%20,JobName%30,State,ExitCode,MaxRSS,Elapsed
```

---

## 4. When jobs run out of time

Paste the same submit command again, with the same `--output-root`, after the first jobs have ended. Nothing else changes.

What happens then:
- A job that is still queued or running is not submitted twice. Its line says "already queued or running".
- In each new job, an experiment that finished in an earlier job is skipped without starting Python. Its summary row says `done_earlier`.
- An experiment that was cut off part-way starts again. How much it keeps depends on the experiment. The table below says which.

| Experiments | What a resubmitted job keeps |
|---|---|
| similarity s01, s02, s03, s04, s05, s06, s08, s10, s12, s13 | every finished unit, from `<experiment>/<dataset>.partial.jsonl` (`tests/experiments/_checkpoint.py`). The unit is a representation, budget, fold or condition, depending on the experiment. |
| similarity s11 | every finished condition, from `.s11_checkpoint_<dataset>.json` in the results folder |
| federated fed01 to fed05 | every finished (FLuID model, seed) pair, from `.partial.jsonl` (`_checkpoint.resumable_run`) |
| all 11 OPIG scripts | every finished item, from `<output>.partial.jsonl` (`_checkpoint.script_checkpoint`). The item is a target, dataset, property, condition or model, depending on the script. |
| every complexity experiment | nothing inside the experiment. No complexity file saves partial results. A cut-off complexity experiment starts from the beginning. Finished ones are skipped. |
| c03 and c14 | the same: nothing inside the run; skipped once finished (`cross_one.sh`) |
| s09 | never skipped. It runs again on every paste, because it takes minutes and the similarity files it reads may have grown since |

No finished experiment runs again, as long as the folder is the same.

Two ways to make a job redo finished work:
- `KIRBY_FORCE_RERUN=1` in the environment turns every skip off.
- A new `--output-root` starts the run again from nothing, apart from the precompute's stored searches.

**Use a new folder whenever the code changes.** The skip trusts the folder. It checks that the data is the same, not that the code is.

One more thing is reused whatever the folder: the precompute's stored searches (`result_db`). That is intended.

---

## 5. What a failure looks like, and where it is written

One failure does not stop anything else:
- Each experiment runs in its own Python process, one after another.
- A Python error, a crash, or running out of memory ends only that experiment. The job goes on to the next one.
- When the job hits its 48-hour wall, the rest of that job's experiments do not run. Resubmitting continues from there (section 4).
- Only s09 waits for other experiment jobs, the similarity ones. Every other job waits only for `pc_` jobs, or for nothing. `afterany` starts a job whether what it waits for passed or failed.

Where each failure is written:

| Where | What |
|---|---|
| `<results>/_job_summaries/<job number>.tsv` | one row per experiment: exit code, status, seconds, log path |
| `<results>/_errors/<dataset>__<experiment>.json` | the Python traceback of a failed experiment |
| `logs/run_2026_10_03/<category>_<dataset>_<experiment>.log` | that experiment's whole output |
| SLURM's state for the job | FAILED when any experiment in it failed; the exit code is the worst one |
| email | SLURM mails adelaide.punt@stcatz.ox.ac.uk when a job fails |

The exit codes in the summaries:

| Exit | Status | Meaning |
|---|---|---|
| 0 | `ok` | finished |
| 0 | `done_earlier` | finished in an earlier job; skipped |
| 1 | `failed` | Python error |
| 1 | `error_file:<name>` | Python exited 0 but wrote an `_errors/` file |
| 124 | `timeout` | hit `EXP_TIMEOUT`, which is unset here |
| 137 | `killed_9_oom_or_wall` | killed, usually out of memory |
| 139 | `segfault` | segmentation fault |

---

## 6. The noise suite: your call

The prelim noise run is separate, and whether to run it tonight is your decision. It is stage 1 of the runbook (`README.md`, "Stage 1"). Replace `herg_ki herg` with your trial datasets.

Dry run:

```bash
python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --account stat-ecr --clusters htc --wait-for-precompute --output tests/results/noise_prelim_2026_10_03 --log-dir logs/noise_prelim_2026_10_03 --max-concurrent 40 --dry-run
```

Then the same line without `--dry-run`. Progress:

```bash
bash tests/slurm_scripts/runbook/progress.sh tests/results/noise_prelim_2026_10_03 logs/noise_prelim_2026_10_03
```

Three things differ from the other categories:
- It waits only for `pc_` jobs on the cluster in `--clusters`. A trial dataset whose `pc_` jobs are on arc is not waited for.
- It uses a new `--output` folder here for the same reason as above. The default, `tests/results/noise_prelim_sweep`, may hold rows from older code.
- `KIRBY_SIMILARITY_SETTINGS` is left unset, so every similarity setting runs (`NOISE_AND_QUEUE_REQUIREMENTS.md`, warning A).

---

## 7. What changed today, and why

**Experiment order inside each job.** Before today, each job ran its experiments in the order of the category's list in its `_helpers.py` file. Now complexity, similarity and federated run the quickest first. The order is `RUN_ORDER` in `submit_dataset.sh`. Noise keeps its list order.

Three experiments must follow others, because they read those experiments' result files:
- c08 reads c02's.
- c09 reads c02's and c06's.
- noise n06 reads n11's and n13's.

The order keeps each of them after its inputs. Any experiment not in `RUN_ORDER` still runs, after the named ones. `KIRBY_KEEP_LIST_ORDER=1` restores the old order.

c21 used to run third. A comment in `complexity_helpers.py` says it ran early because "its question gates the others". c21 reads no other experiment's file, so that is not a data dependency. It now runs last, as the slowest.

**Order between jobs.** Similarity jobs are submitted first, then federated, then complexity, then OPIG. Within each, datasets go in the precompute's size classes from `run_precompute_dataset.sh`: SMALL, then MEDIUM, then LARGE. OPIG scripts go in the order of the table in `submit_everything.sh`, which is now quickest first. Submission order only decides between jobs that become ready at the same time.

**What the order is based on.** Seconds unless marked. "Measured" means a finished run recorded it. Every measurement is from older code than tonight's.

Complexity: the middle value of five cluster runs in June 2026, on five antibody datasets of 137 to 418 sequences (`WAVE_STATUS_2026_06_04.md` §8). c08, c12 and c33 are not in those runs.

| Order | Experiment | Middle of five | Range | Kind |
|---|---|---|---|---|
| 1 | c06 | 12 | 6 to 43 | measured |
| 2 | c19 | 28 | 19 to 515 | measured |
| 3 | c22 | 59 | 12 to 127, three runs | measured |
| 4 | c07 | 159 | 0.5 to 3,055 | measured |
| 5 | c18 | 359 | 165 to 1,182 | measured |
| 6 | c01 | 478 | 88 to 3,862 | measured |
| 7 | c02 | 634 | 169 to 2,857 | measured |
| 8 | c08 | none | 72 to 615, four earlier runner logs (`slurm-120137*.out`) | measured elsewhere |
| 9 | c15 | 1,791 | 1,249 to 5,573 | measured |
| 10 | c04 | 1,805 | 653 to 9,263 | measured |
| 11 | c13 | 2,068 | 584 to 2,699 | measured |
| 12 | c11 | 2,118 | 843 to 6,824 | measured |
| 13 | c12 | about 2 hours | tcga_* only | guessed: stated, not measured (`TIME_ESTIMATION_METHOD.md`) |
| 14 | c16 | 6,152 | 1,547 to 8,891 | measured |
| 15 | c09 | 7,332 | 2,877 to 37,288 | measured |
| 16 | c24 | 7,784 | 1,167 to 12,044, three runs | measured |
| 17 | c05 | 8,421 | 2,894 to 24,646 | measured |
| 18 | c23 | 10,358 | 19 to 10,363, three runs | measured |
| 19 | c33 | none | about 58 minutes per fold on the laptop | guessed from the laptop note in `c33_budget_overfitting.py` |
| 20 | c17 | 32,709 | 24,248 to 192,233 | measured |
| 21 | c21 | 69,338 | 2,740 to 69,491, three runs; two other runs hung in it | measured |

Similarity: April to June 2026 runner logs on five datasets (`slurm-11595868.out` and four others).

| Order | Experiment | Seconds seen | Kind |
|---|---|---|---|
| 1 | s04 | 0.0 to 0.1 | guessed: that small a time probably means it ran nothing on those datasets |
| 2 | s03 | 0.3 to 48 | measured |
| 3 | s01 | 11 to 125 | measured |
| 4 | s12 | 3 to 189 | measured |
| 5 | s06 | 0.3 to 384 | measured |
| 6 | s13 | 72 to 112, two runs | measured |
| 7 | s02 | 6.5 to 4,079 | measured |
| 8 | s10 | 128 to 1,702 | measured |
| 9 | s05 | 7 to 44,363 | measured |
| 10 | s08 | 1,093 to 6,102 | measured |
| 11 | s11 | 54 to 674, two runs | guessed: placed last because it fits about 2.3 million models (`EXPERIMENT_COMPUTE_AUDIT_2026_09_18.md`) |

Federated: guessed from model-fit counts in `EXPERIMENT_COMPUTE_AUDIT_2026_09_18.md`. Only fed02 has a time.

| Order | Experiment | Evidence | Kind |
|---|---|---|---|
| 1 | fed02 | about 6,500 fits per run on caco2 or logd, about 49,400 on lhasa_herg; one part took 49 minutes on caco2 (memory note of 2026-09-23) | partly measured |
| 2 | fed03 | about 43,000 fits | guessed |
| 3 | fed04 | about 107,000 to 115,000 fits | guessed |
| 4 | fed05 | about 408,000 fits | guessed |
| 5 | fed01 | about 1.9 million fits on caco2 | guessed |

OPIG: `elapsed_seconds` in March and April result files, older code.

| Order | Script | Evidence | Kind |
|---|---|---|---|
| 1 | shared_vectors_census | no model fits | guessed |
| 2 | scaffold_similarity | about 1 minute on a small test | guessed: stated, not measured |
| 3 | humanness | 1,950 s | measured |
| 4 | graphinity | 6,683 to 45,980 s | measured |
| 5 | trastuzumab | 13,481 s | measured |
| 6 | cgr_comparison | about 82,000 fits per dataset | guessed |
| 7 | ginkgo | about 42,000 s over 9 properties; one run hit a 24-hour wall | measured |
| 8 | qsar | about 13.5 million fits | guessed |
| 9 | blueprint_transfer | hit a 24-hour wall | measured, lower bound |
| 10 | scaffold_hybrid | ran 1 day 12 hours, then hit its wall | measured, lower bound |
| 11 | shared_vectors_remedy | about 948 million fits per dataset | guessed |

shared_vectors_remedy will not finish in one 48-hour job. Each resubmission keeps every finished partner set (section 4).

**Resubmission.** A resubmitted job now skips finished experiments without starting Python (`submit_dataset.sh`, `opig_one.sh`). Before, each finished experiment still loaded the dataset and read the stored searches before being skipped.

**Failures are kept, one file per experiment.** `_errors/` used to hold one file per dataset. The next experiment on that dataset deleted it when it succeeded. Now each experiment has its own file (`run_dataset.py`).

**A complexity experiment that cannot be imported is a failure.** It used to print SKIP and exit 0, so the summary said `ok` (`complexity_helpers.py`).

**c12 runs on tcga_* datasets.** `complexity_helpers.run_all` adds c12 only when it builds its own list. `submit_dataset.sh` names every experiment itself, so c12 had never run under it.

**The submit script** (`submit_all_stages.sh`, stage 3) now:
- reads `pc_` jobs on both clusters and sends each job to the right one;
- requires `--output-root`, and writes results, experiment logs and SLURM logs under that name;
- refuses to submit a job twice into the same folder while the first is queued or running;
- takes `--cluster` and `--categories`;
- no longer needs `--trial` when only stage 3 is submitted.

---

## 8. Not checked

- Nothing here has run on the cluster.
- Whether `sbatch --clusters=arc` works from the htc login node.
- The caches other than the precompute's: whether any of them keeps a value computed by last week's code. Only the folder-level skip is covered by the new folder.
- Every time in section 7 comes from older code. The order may be wrong for large datasets.
- How long c03 and c14 take, and how much memory c03 needs with all 41 datasets loaded at once.
- `EXPERIMENTS_FINALIZED.md` line 317 records "c14 | delete". c14 is submitted here because you asked for it on 2026-10-03.
