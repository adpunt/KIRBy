# KIRBy re-run playbook — 2026-06-05

All commands you run **on the ARC login node**, from the repo root, after
`cd /data/stat-ecr/scat9264/KIRBy && git pull && . ./setup.sh`. Nothing here is
run for you — these are the commands.

Order: **0.5 cache hygiene → 1 check precompute → 2 (optional) clear the old queue
→ 3/4 submit the wave → 5 monitor → 6 standalone after precompute**.

---

## 0.5 — Cache hygiene (run once, before submitting)

The loader now **refuses to run** on a stale representation cache whose row count
doesn't match the labels (it raises and names the dataset + this command). Clear
the two known-stale caches first. Both tools are **dry-run by default** — inspect,
then re-run with `--delete`.

```bash
# Stale herg rep cache (1412 rows vs 7772 labels) — the silent-misalignment case.
python tests/purge_rep_cache.py herg                 # dry-run: shows what it would delete
python tests/purge_rep_cache.py herg --delete        # do it

# Graphinity corrupt cache (values-only change the meta check can't catch).
python tests/purge_graphinity_cache.py               # dry-run
python tests/purge_graphinity_cache.py --delete      # do it
```

Genomics/nucleotide NFS "Stale file handle" (ESTALE) fix — put HuggingFace's
cache/lock on node-local storage so the `datasets` FileLock doesn't sit on the
flaky shared NFS. Add to the job environment (or your `setup.sh`):

```bash
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"   # node-local; or /tmp/$USER/hf_datasets
```

---

## 1 — Which datasets have finished precompute (for BOTH complexity and similarity)

A dataset is ready for complexity **and** similarity only when its **distance**
conditions are also built (similarity needs them), so always pass `--distance`.

```bash
# Human-readable table:
python tests/check_precompute_status.py --done       --distance   # READY (both cx + sim)
python tests/check_precompute_status.py --incomplete --distance   # NOT ready (partial/missing)

# Machine-readable name lists (used by the wave submitter):
python tests/check_precompute_status.py --done       --distance --names-only
python tests/check_precompute_status.py --incomplete --distance --names-only
```

`DONE` = ready for the combined job (part 3). Everything else goes to the
precompute-building split (part 4). The wave submitter does this classification
for you — you don't have to copy lists by hand.

---

## 2 — Cancel the currently-queued KIRBy jobs (SAFE, list-first)

> I am deliberately **not** giving you a blanket `scancel -u $USER` or a piped
> `xargs scancel` — that's the exact pattern that cost you a week of queue
> priority. Review the list, then cancel the specific IDs you mean.

```bash
# 1) See everything you have queued/running, with job names:
squeue -u $USER -o "%.18i %.60j %.8T %.10M %.9P"

# 2) (optional) Narrow to the KIRBy experiment job names to eyeball them:
squeue -u $USER -o "%.18i %.60j %.8T" | grep -E "kirby_full|complexity|similarity|opig_|fc_|pc_"

# 3) Cancel the SPECIFIC ids you reviewed (space-separated):
scancel 12345678 12345679 12345680
```

If, after reviewing, you are certain you want every `kirby_full` job gone, the
name-scoped form exists (`scancel -u $USER --name=kirby_full`) — but confirm the
`squeue` list first; I'm not putting it in a runnable block here on purpose.

---

## 3 + 4 — Submit the experiment wave (precompute-aware)

One command. It classifies every dataset (part 1) and submits:
- **DONE** datasets → **1 job** each: `complexity similarity noise` (reuse cache).
- **NOT-DONE** datasets → **2 jobs** each: `complexity noise` (cx builds general/top
  precompute, noise reuses it) submitted **forward**, and `similarity` (builds
  distance) submitted **reverse** — so cx and sim don't cold-build the same
  dataset's cache at the same instant.

```bash
# Dry-run FIRST — prints the exact sbatch plan, submits nothing:
bash tests/slurm_scripts/submit_rerun_wave.sh

# Review the plan, then actually submit:
bash tests/slurm_scripts/submit_rerun_wave.sh --submit

# Restrict to specific datasets (dry-run, then --submit):
bash tests/slurm_scripts/submit_rerun_wave.sh logd caco2
bash tests/slurm_scripts/submit_rerun_wave.sh --submit logd caco2
```

Notes:
- "Same loaded reps" is achieved via the shared on-disk caches: the rep cache
  (`rep_cache/`) and the search cache (`searches/`) are reused across cx/sim/noise,
  so the expensive precompute is paid once per dataset, not three times.
- Residual race: if the queue happens to start a NOT-DONE dataset's cx and sim
  jobs at the exact same time, both cold-build general/top once. The forward/reverse
  ordering makes that unlikely. If you'd rather eliminate it entirely, make the sim
  job depend on the cx job per dataset (`--dependency=afterok:<cx_jobid>`) — say the
  word and I'll add a `--serialize` mode to the submitter.

---

## 5 — Monitor progress (both timing tools)

Two tools, both read-only:

```bash
# (a) Precompute timing — per-condition wall time, per-model share, write date:
python tests/check_precompute_status.py --timing            # all datasets
python tests/check_precompute_status.py --timing --done     # only finished ones
python tests/check_precompute_status.py --timing --verbose  # + per-condition detail

# (b) Wave snapshot — per-job health / completion / failure classification + ETA:
python tests/audit_full_wave.py --include-running --eta
```

Scheduling them so you can check in (ARC login node — pick one):

```bash
# Live refresh every 30 min (Ctrl-C to stop), good while you watch:
watch -n 1800 'python tests/check_precompute_status.py --distance; echo; python tests/audit_full_wave.py --include-running --eta'

# Or a detached loop that appends a timestamped snapshot every 30 min to a log:
nohup bash -c 'while true; do
  ts=$(date "+%Y-%m-%d_%H%M");
  { echo "===== $ts ====="; \
    python tests/check_precompute_status.py --timing --distance; \
    python tests/audit_full_wave.py --include-running --eta; } \
    >> monitor_${ts:0:10}.log 2>&1;
  sleep 1800;
done' >/dev/null 2>&1 &
# stop it later with:  kill %1   (or: pkill -f "check_precompute_status.py --timing")
```

(If the login node has cron enabled you can use a `*/30 * * * *` entry instead;
the loop above avoids depending on that.)

---

## 6 — Standalone experiments (run AFTER precompute is finished)

Gate first — only proceed once precompute is built (and the wave's precompute-
building jobs have drained):

```bash
python tests/check_precompute_status.py --done --distance   # should list the datasets you expect
squeue -u $USER --name=kirby_full                           # the cx/sim jobs that build precompute
```

Then submit the three standalone families:

```bash
# OPIG (own datasets/reps; uses its own submitter):
bash tests/slurm_scripts/submit_opig_experiments.sh all

# Forward-compatibility (new foundation-model reps; own submitter):
bash tests/slurm_scripts/submit_forward_compat.sh

# Federated — per-dataset, reuses the general/top precompute the wave built:
for DS in $(python tests/check_precompute_status.py --done --distance --names-only); do
  sbatch --export=DATASET=$DS,CATEGORIES="federated" tests/slurm_scripts/run_full_suite.sh
done
```

(`fed05` self-loads its three antibody silos; the others read the shared
precompute, so running them only after part 3/4 avoids paying for an uncached
search inside the experiment.)
