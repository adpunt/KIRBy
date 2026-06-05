# PRE-RERUN / PRE-PRECOMPUTE SERVER CHECKLIST (KIRBy, compiled 2026-06-05)

Definitive, deduplicated list of everything that must happen **on ARC**
(`scat9264@arc-login.arc.ox.ac.uk` : `/data/stat-ecr/scat9264/KIRBy`) **before**
the precompute / experiment re-run. Compiled from a full read of:
`RERUN_PLAYBOOK_2026_06_05.md`, `CACHE_OPERATIONS_2026_06_04.md`,
`CACHE_FIX_PLAN.md`, `NEXT_SESSION_PROMPT.md`, `POST_PRECOMPUTE_COMMANDS.md`,
`NOISE_AUDIT_STATUS_2026_06_04.md`, smoke `DECISIONS_FOR_USER.md`, `AGENT_BRIEF.md`,
`SETUP.md`.

Order is the recommended execution order. Every cache deletion is **dry-run first,
then `--delete`** — never blanket-delete. All commands run from the repo root.

---

## STEP 0 — Deploy code + environment (REQUIRED, do first)

**0.1 — Pull the fixed code and source setup.** REQUIRED.
The cache-drift fix, the experiment fixes, and the new purge scripts only become
active after a `git pull`. The running wave is OLD code (`374bb08`, Jun 1) and must
be restarted/resubmitted.
```bash
cd /data/stat-ecr/scat9264/KIRBy && git pull && . ./setup.sh
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:5`, `NEXT_SESSION_PROMPT.md:28`, `SETUP.md:686-696`.

**0.2 — Confirm `noiseInject` installed (for c10 noise-robustness).** REQUIRED if
running noise experiments. `setup.sh:155-173` auto-clones + installs the private
`noise_inject` package; verify it took. Without it, c10 returns an "not installed" stub.
```bash
python -c "import noiseInject"        # must succeed (no output = OK)
# if it failed, manual install:
#   git clone https://github.com/adpunt/noise_inject.git src/noise_inject
#   pip install --no-deps -e src/noise_inject
```
Source: `NEXT_SESSION_PROMPT.md:40` (CX-5), `SETUP.md:184-201`, `setup.sh:155-173`.

**0.3 — Genomics/nucleotide HuggingFace cache on node-local storage.** REQUIRED only
for nucleotide/genomics jobs (fixes the NFS "Stale file handle" / ESTALE FileLock crash).
Add to the job environment or `setup.sh`:
```bash
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"   # node-local; or /tmp/$USER/hf_datasets
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:29-35`.

**0.4 — mol2vec weights symlink in place (for OPIG qsar + imrie).** REQUIRED for those
jobs only — already verified present per the doc, just confirm `~/model_300dim.pkl` exists
(KIRBy looks for `model_300dim.pkl` relative to cwd; SETUP.md §3a).
```bash
ls -lh ~/model_300dim.pkl    # or repo-root model_300dim.pkl
```
Source: `POST_PRECOMPUTE_COMMANDS.md:57-59` ("symlink ~/model_300dim.pkl is in place, verified"),
`SETUP.md:206-213`.

---

## STEP 1 — CACHE PURGES / STALE / CORRUPT ENTRIES (the critical pre-rerun hygiene)

> These exist because the CACHE-1 anti-churn fix makes the cache **ignore representation
> content hashes**. An entry auto-invalidates only when shape / labels / n_samples /
> dataset_name / n_folds / metric change. A fix that changes a representation's **VALUES
> but not its shape** is **silently reused stale/corrupt** unless purged by hand.
> Source: `CACHE_OPERATIONS_2026_06_04.md:16-17`.

**1.1 — Purge the stale `herg` representation cache (silent row-count misalignment).** REQUIRED.
On-disk `herg` rep cache was **1412 rows vs 7772 labels** — a hard misalignment that corrupts
any downstream experiment. The loader now **refuses to run** on this mismatch (raises, names the
dataset + this exact command), so this purge is mandatory before any herg-touching job.
```bash
python tests/purge_rep_cache.py herg            # DRY-RUN: shows what it would delete
python tests/purge_rep_cache.py herg --delete   # after reviewing the list
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:19-27`, smoke `DECISIONS_FOR_USER.md` item 3.3
(`_shared.py:168-176`; observed herg 1412 vs 7772). Script verified to exist: `tests/purge_rep_cache.py`.

**1.2 — Purge the corrupt Graphinity cache (values-only change the meta check can't catch).** REQUIRED.
The Graphinity mutation-position parser fix changes embedding VALUES but not shape → the meta-check
cannot detect it → stale/corrupt embeddings are silently reused. Must purge by hand.
```bash
python tests/purge_graphinity_cache.py            # DRY-RUN: lists what it would delete
python tests/purge_graphinity_cache.py --delete   # after reviewing the list
```
The script walks only `$KIRBY_CACHE_DIR`, matches on filename + `_cache_meta.dataset_name` in
{graphinity_foldx, graphinity_rosetta}, and catches the content-hashed `reps/` entries a glob
would miss. Script verified to exist: `tests/purge_graphinity_cache.py`.
Source: `RERUN_PLAYBOOK_2026_06_05.md:24-27`, `CACHE_OPERATIONS_2026_06_04.md:19-26`,
`NEXT_SESSION_PROMPT.md:38-39` (MEMORY.md flags this as the "★ BEFORE RERUN" item).

**1.3 — (Conditional) Force-recompute any OTHER dataset whose loader VALUES changed.** OPTIONAL,
only if you changed a loader's values without changing shape. Review-first glob:
```bash
DS=<dataset>
ls -lh $KIRBY_CACHE_DIR/searches/${DS}__*.pkl*  2>/dev/null   # review FIRST
ls -lh $KIRBY_CACHE_DIR/baselines/${DS}__*.pkl  2>/dev/null
ls -lh $KIRBY_CACHE_DIR/folds/${DS}__*.pkl      2>/dev/null
# then, once confirmed:
rm $KIRBY_CACHE_DIR/searches/${DS}__*.pkl* $KIRBY_CACHE_DIR/baselines/${DS}__*.pkl $KIRBY_CACHE_DIR/folds/${DS}__*.pkl
python tests/purge_graphinity_cache.py --targets ${DS}   # for content-hashed reps/ entries
```
Source: `CACHE_OPERATIONS_2026_06_04.md:28-38`.

**Cases that AUTO-invalidate — NO action needed:** InChIKey dedup (row count changes),
task_type/metric changes, StratifiedGroupKFold change (fold structure changes → folds cache
rebuilds), and the SCREENING→FULL budget change (precompute always used FULL_BUDGETS, so
`searches/` files are unaffected). Sources: `CACHE_OPERATIONS_2026_06_04.md:17, 46-47`.

> NOTE — historical `CACHE_FIX_PLAN.md` (Apr) called for a full cache nuke
> (`mv ~/.cache/kirby ~/.cache/kirby.old`) due to the `_CACHE_VERSION` orphaning and the
> hash→filename redesign. That redesign is **done and committed** (filename-keyed cache,
> content validation on load). The current 2026-06 guidance is **targeted purges only**
> (1.1/1.2), NOT a blanket nuke. See "CONFLICTS" below.

---

## STEP 2 — Verify precompute readiness state (REQUIRED before submitting the wave)

A dataset is ready for complexity **and** similarity only when its **distance** conditions are
also built (similarity needs them) → always pass `--distance`.
```bash
python tests/check_precompute_status.py --done       --distance              # READY (cx + sim)
python tests/check_precompute_status.py --incomplete  --distance              # NOT ready
python tests/check_precompute_status.py --done       --distance --names-only  # machine-readable
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:39-52`.

`check_precompute_status.py` was rewritten to validate cache CONTENT on disk (file exists AND
`_cache_meta` validates AND `best_by_model`/`all_results` non-empty), not just filename presence.
Source: `CACHE_FIX_PLAN.md:208-212, 387`.

---

## STEP 3 — (Optional) Cancel the stale queued wave — LIST-FIRST, never blanket

> Standing rule: **NO** `scancel -u $USER` and **NO** piped `xargs scancel` — that pattern
> cost a week of queue priority. Review the list, cancel specific reviewed IDs only.
```bash
squeue -u $USER -o "%.18i %.60j %.8T %.10M %.9P"               # see everything
squeue -u $USER -o "%.18i %.60j %.8T" | grep -E "kirby_full|complexity|similarity|opig_|fc_|pc_"
scancel 12345678 12345679                                       # specific reviewed IDs only
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:60-79`, `CACHE_FIX_PLAN.md:413`,
user preference `feedback_destructive_commands`.

---

## STEP 4 — Submit the experiment wave (precompute-aware) — DRY-RUN FIRST

One submitter classifies every dataset (DONE → 1 reuse-cache job; NOT-DONE → cx-forward +
sim-reverse to avoid cold-building the same dataset's cache simultaneously).
```bash
bash tests/slurm_scripts/submit_rerun_wave.sh            # DRY-RUN: prints plan, submits nothing
bash tests/slurm_scripts/submit_rerun_wave.sh --submit   # actually submit after reviewing
# restrict to specific datasets:
bash tests/slurm_scripts/submit_rerun_wave.sh --submit logd caco2
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:83-113`.

---

## STEP 5 — Standalone experiments — run ONLY AFTER precompute is finished

Gate first (must list expected datasets, and the cx/sim precompute-building jobs must have drained):
```bash
python tests/check_precompute_status.py --done --distance
squeue -u $USER --name=kirby_full
```
Then the three standalone families (OPIG / forward-compat / federated). These are POST-precompute,
not pre-rerun, but listed so they aren't started early:
```bash
bash tests/slurm_scripts/submit_opig_experiments.sh all
bash tests/slurm_scripts/submit_forward_compat.sh
for DS in $(python tests/check_precompute_status.py --done --distance --names-only); do
  sbatch --export=DATASET=$DS,CATEGORIES="federated" tests/slurm_scripts/run_full_suite.sh
done
```
Source: `RERUN_PLAYBOOK_2026_06_05.md:153-176`, `POST_PRECOMPUTE_COMMANDS.md` (whole file is the
post-precompute queue).

---

## STEP 6 — Dataset verifications gated to specific jobs

**6.1 — Verify `metz` exists in the cluster's PyTDC BEFORE any metz job.** REQUIRED only if you
intend to use metz. The dev box's PyTDC shipped only 6 DTI sets and did NOT include Metz, so the
exact TDC name is UNCONFIRMED.
```bash
python -c "from tdc.utils import retrieve_dataset_names; n=retrieve_dataset_names('DTI'); print('DTI sets:', sorted(n)); print('metz present:', 'metz' in [x.lower() for x in n])"
# if present, confirm it loads:
python -c "from tdc.multi_pred import DTI; d=DTI(name='Metz'); df=d.get_data(); print('metz rows:', len(df)); print(df.columns.tolist())"
# if absent: pip install -U PyTDC  (then re-check). If still absent, skip metz — loader raises a clear error.
```
Source: `POST_PRECOMPUTE_COMMANDS.md:126-148`.

**6.2 — imrie transfer-learning needs server-side pretrained weights.** REQUIRED only for the
transfer-learning conditions (the scratch/ridge/hybrid path doesn't need them). Run the pretraining
first; `/Volumes/seagate/kirby/chembl_pretrain` won't exist on the cluster, so the TL conditions skip
until `kirby.models.pretrain_chembl_dnn` has run. Operational, no code change.
Source: smoke `DECISIONS_FOR_USER.md` item 4.4; `POST_PRECOMPUTE_COMMANDS.md:53-54`
(`imrie_eval.sh` is manual AFTER `imrie_pretrain` finishes).

---

## POST-RERUN (not pre-rerun, but recorded so they aren't forgotten)

These are tuning/design reviews to do AFTER results regenerate — NOT pre-rerun blockers:
- **Cliff prevalence knob** (n01/n04/n08, s07/s12): after rerun inspect `n_cliff_pairs` /
  `pct_cliff_compounds`; retune the 2% percentile if prevalence is ~0% or ~100%.
  Source: `NEXT_SESSION_PROMPT.md:42`, `NOISE_AUDIT_STATUS_2026_06_04.md:68-69`.
- **n05/n10 MAD low-sigma ground-truth threshold:** σ=0.05 marks ~0 noisy → F1≈0 there;
  decide `'mad'` vs `'all'` (KEPT as `'mad'` default for now). Source:
  `NOISE_AUDIT_STATUS_2026_06_04.md:60-67, 72-74`, `NEXT_SESSION_PROMPT.md:44`.
- Whole noise suite, complexity/similarity, and most experiment JSONs are STALE and must be
  regenerated on ARC (handled by Step 4 / Step 5). Source: `NOISE_AUDIT_STATUS_2026_06_04.md:59-61`,
  `NEXT_SESSION_PROMPT.md:36-43`.

---

## CONFLICTS / DUPLICATES BETWEEN DOCS (flagged)

1. **Full cache nuke vs targeted purge (RESOLVED in favour of targeted).**
   `CACHE_FIX_PLAN.md:243-250, 415-419` (April) instructs a blanket
   `mv ~/.cache/kirby ~/.cache/kirby.old_2026-04-15` as the cache deletion step, because of the
   `_CACHE_VERSION` orphaning + hash→filename redesign. That redesign is **complete and committed**
   (`CACHE_FIX_PLAN.md:384-390`). The current 2026-06 docs
   (`RERUN_PLAYBOOK_2026_06_05.md`, `CACHE_OPERATIONS_2026_06_04.md`) prescribe **targeted purges only**
   (herg + Graphinity). **Use the targeted purges (Step 1). Do NOT blanket-nuke** unless you are
   deliberately starting from an empty cache.

2. **Cache dir path drift.** `CACHE_FIX_PLAN.md` and `SETUP.md §7` reference the old
   `/data/stat-cadd/scat9264/...` location and `~/.cache/kirby`. Per MEMORY + the 2026-06 docs the
   server moved to `/data/stat-ecr/scat9264/KIRBy` and `$KIRBY_CACHE_DIR =
   /data/stat-ecr/scat9264/.cache/kirby` (`CACHE_OPERATIONS_2026_06_04.md:7`). Use the **stat-ecr**
   path; `. setup.sh` sets `$KIRBY_CACHE_DIR` correctly — always reference the env var, not a literal.

3. **Graphinity purge appears in 3 docs** (RERUN_PLAYBOOK 1.2, CACHE_OPERATIONS §1,
   NEXT_SESSION_PROMPT). All agree: dry-run then `--delete`. No conflict — same single action (1.2 above).

4. **herg purge:** smoke `DECISIONS_FOR_USER.md` 3.3 proposed it as a decision; `RERUN_PLAYBOOK`
   has it as a committed required step with a dedicated script (`purge_rep_cache.py`) + a hard loader
   guard. Treat as REQUIRED (1.1 above). The row-count guard from 3.3 is already applied
   (`_finalize_load_data` raises on mismatch).

---

## ONE-SCREEN REQUIRED-PATH SUMMARY

```bash
cd /data/stat-ecr/scat9264/KIRBy && git pull && . ./setup.sh   # 0.1
python -c "import noiseInject"                                  # 0.2 (noise jobs)
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"                  # 0.3 (nucleotide jobs)
python tests/purge_rep_cache.py herg                            # 1.1 dry-run
python tests/purge_rep_cache.py herg --delete                   # 1.1 do it
python tests/purge_graphinity_cache.py                          # 1.2 dry-run
python tests/purge_graphinity_cache.py --delete                 # 1.2 do it
python tests/check_precompute_status.py --done --distance       # 2
bash tests/slurm_scripts/submit_rerun_wave.sh                   # 4 dry-run
bash tests/slurm_scripts/submit_rerun_wave.sh --submit          # 4 submit
# metz only: verify PyTDC has it (Step 6.1) before any metz job
```
