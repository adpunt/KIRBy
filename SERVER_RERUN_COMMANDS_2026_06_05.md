# Server re-run commands — authoritative, 2026-06-05

Everything you run on ARC, in order, to get the precompute + experiment wave going.
All commands from `/data/stat-ecr/scat9264/KIRBy`. Cache deletions are review-first.
Grounded in: the 4-agent cache audit (`tests/results/_smoke_2026_06_05/cache_audit/`),
`CACHE_OPERATIONS_2026_06_04.md`, `CACHE_FIX_PLAN.md`, `POST_PRECOMPUTE_COMMANDS.md`,
and your 2026-06-05 `check_precompute_status --distance` snapshot (33 DONE / 31 partial / 9 missing).

---

## 0. One-time pre-rerun hygiene (BEFORE any experiment/precompute job)

### 0a. Environment
```bash
cd /data/stat-ecr/scat9264/KIRBy
git pull
. ./setup.sh                                  # activates py311 env + installs noiseInject (c10 needs it)
python -c "import noiseInject" && echo "noiseInject OK"
export KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby
```

### 0b. Cache purge — GRAPHINITY ONLY (the only proven-corrupt cache)
The mutation-position parser fix (commit 3cadb95, after the cache was built) changed Graphinity
rep **values** but not their shape — so the content-hash-ignoring cache would silently reuse the
corrupt embeddings. Graphinity has NO search cache (precompute missing 0/9), so only the rep
embeddings exist. Review, then remove:
```bash
ls -ld $KIRBY_CACHE_DIR/rep_cache/graphinity_* $KIRBY_CACHE_DIR/searches/graphinity_* $KIRBY_CACHE_DIR/baselines/graphinity_* 2>/dev/null
rm -rf $KIRBY_CACHE_DIR/rep_cache/graphinity_foldx $KIRBY_CACHE_DIR/rep_cache/graphinity_rosetta $KIRBY_CACHE_DIR/rep_cache/graphinity_augmented
```
**DO NOT delete anything else.** qm9 and every other molecular/antibody/TCGA/reaction cache is
VALID — the molecular rep fix (REP-1/2/3, commit 3236092 / PDV 61e5a87) predates the cache AND is
shape-changing (auto-invalidates), and git shows no other post-cache value-changing rep fix.
The old "`mv ~/.cache/kirby` nuke everything" instruction in CACHE_FIX_PLAN is SUPERSEDED.

### 0c. Other asset prep
```bash
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"     # nucleotide/genomic jobs: avoids NFS ESTALE on the HF lock
ls -l ~/model_300dim.pkl                            # mol2vec symlink (OPIG qsar/imrie) — must exist
```
(For already-queued jobs to get `HF_DATASETS_CACHE`, it must live in a file the job sources —
`setup.sh` or `run_full_suite.sh` — not your login shell. See note at the bottom.)

---

## 1. Verify the cache is healthy (answer "is it corrupted?" with real content)
```bash
python tests/check_precompute_status.py --distance --incomplete   # anything tagged corrupt/empty/stale (NOT 'missing') = real breakage
python tests/check_precompute_status.py --dataset qm9 --verbose    # spot-check any dataset's per-condition content
```
`missing` = just not built yet (fine). `corrupt`/`empty`/`stale` = a real broken entry → purge that
dataset's searches/baselines and rebuild.

---

## 2. What precompute is finished (ready for BOTH cx and sim → needs `--distance`)
```bash
python tests/check_precompute_status.py --done       --distance   # READY
python tests/check_precompute_status.py --incomplete --distance   # NOT ready (partial/missing)
```

---

## 3. Cancel queued jobs — list first, explicit IDs only (NEVER blanket)
```bash
squeue -u $USER -o "%.18i %.55j %.8T %.10M"
scancel <space-separated specific IDs you reviewed>
```
Do NOT `scancel -u $USER` reflexively — it kills in-flight precompute that hasn't written its last
condition (that is what drops the DONE count).

---

## 4. Diagnose failed jobs
```bash
sacct -X --starttime $(date +%Y-%m-%d) --format=JobID,JobName%18,State,ExitCode,Elapsed | grep -viE "RUNNING|PENDING|COMPLETED"
grep -iE "error|traceback|killed|oom|stale file handle|importerror|raise|no module" slurm-<FAILED_ID>.out | tail -40
```

---

## 5. The experiment wave — one `run_full_suite.sh` job per dataset
`run_full_suite.sh` runs the requested categories for one dataset; `complexity`/`similarity` build
their precompute tiers as they go (so this both builds precompute for not-done datasets AND runs
experiments). It self-`git pull`s + `. setup.sh` inside the job.

**DONE datasets (33) — one job each, all three categories (reuses cached precompute):**
```bash
CATEGORIES="complexity similarity noise" bash tests/slurm_scripts/submit_full_suite.sh \
  abagym_Ang2 abagym_EGFR abagym_HER2 abagym_VEGF abagym_lysozyme abcov_kd ddg_combined \
  flab_aggregation_jain flab_aggregation_jain_hicrt flab_binding_adams flab_expression_adams \
  flab_immunogenicity_marks flab_pk_jain_fcrnrt flab_pk_jain_heprt flab_polyreact_jain_bvp \
  flab_polyreact_jain_cic flab_thermo_jain flab_thermo_tresanco ginkgo_acsins ginkgo_colloidal \
  ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer humanness_therapeutic tz_her2_binary \
  tz_her2_multiclass adme_mdr1 caco2 herg_ki buchwald_hartwig tcga_brca_pam50 tcga_luad_vs_lusc
```

**NOT-DONE datasets (40) — complexity+noise FORWARD, similarity REVERSE** (cx builds general/top,
sim builds distance; opposite order staggers cold builds so they don't double-build the same dataset):
```bash
NOTDONE="adme_hlm adme_rlm bindingdb_kd kiba herg logd qm9 bh_doyle_plates \
flab_binding_koenig flab_binding_phillips_h3 flab_expression_koenig \
ginkgo_aggregation ginkgo_purity ginkgo_tm2 \
gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm gb_human_enhancers_cohn gb_human_nontata_promoters \
nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_H4K20me1 \
nt_enhancers nt_enhancers_types nt_promoter_all nt_promoter_no_tata nt_promoter_tata \
nt_splice_sites_acceptors nt_splice_sites_donors \
cdr_flexibility chembl_qsar graphinity_foldx graphinity_rosetta humanness_species"

CATEGORIES="complexity noise" bash tests/slurm_scripts/submit_full_suite.sh $NOTDONE
CATEGORIES="similarity"       bash tests/slurm_scripts/submit_full_suite.sh $(echo $NOTDONE | tr ' ' '\n' | tac | tr '\n' ' ')
```
The last 5 of `$NOTDONE` (cdr_flexibility, chembl_qsar, graphinity_foldx/rosetta, humanness_species)
are OPIG-only datasets — drop that line if you'd rather run those only in step 6.
(Re-verify the two lists against `check_precompute_status --done --distance` before submitting;
they are from the 2026-06-05 snapshot.)

---

## 6. Standalone families — AFTER their precompute is built
Gate: `python tests/check_precompute_status.py --done --distance` shows the datasets you need.
```bash
bash tests/slurm_scripts/submit_opig_experiments.sh        # OPIG find_best_hybrid comparisons
bash tests/slurm_scripts/submit_opig_surveys.sh            # OPIG surveys + imrie pretrain
sbatch tests/slurm_scripts/imrie_eval.sh                   # ONLY after imrie pretrain finishes
bash tests/slurm_scripts/submit_forward_compat.sh          # forward-compat new-model reps
for DS in $(python tests/check_precompute_status.py --done --distance --names-only); do
  sbatch --export=DATASET=$DS,CATEGORIES="federated" tests/slurm_scripts/run_full_suite.sh
done
```
`metz` (DTA): verify it exists in the cluster PyTDC before any metz job —
`python -c "from tdc.utils import retrieve_dataset_names; print('metz' in [x.lower() for x in retrieve_dataset_names('DTI')])"`;
if False, `pip install -U PyTDC`, else skip.

---

## 7. Monitor (both tools)
```bash
python tests/check_precompute_status.py --timing --distance     # per-condition wall time + per-model share
python tests/audit_full_wave.py --include-running --eta         # per-job health / completion / ETA
```
Detached 30-min snapshot loop (Ctrl-C / `kill %1` to stop):
```bash
nohup bash -c 'while true; do ts=$(date "+%Y-%m-%d_%H%M");
  { echo "=== $ts ==="; python tests/check_precompute_status.py --timing --distance;
    python tests/audit_full_wave.py --include-running --eta; } >> monitor_${ts:0:10}.log 2>&1;
  sleep 1800; done' >/dev/null 2>&1 &
```

---

## NOTE — known footguns (so they don't bite again)
- `purge_graphinity_cache.py` crashes on python <3.10 (`str | None`). Use the plain `rm` in §0b, or
  `purge_rep_cache.py <ds>` **without** `--all-caches` (with `--all-caches` it pickle-walks the whole
  cache and appears to hang for many minutes over NFS).
- `submit_rerun_wave.sh` has the same `--all-caches` slowness in its status step — use §5's explicit
  `submit_full_suite.sh` lists instead.
- `HF_DATASETS_CACHE` set in your login shell does NOT reach SLURM jobs — it must be in `setup.sh`/
  `run_full_suite.sh` (which jobs source). Ask me to push that one-line edit if you want queued
  nucleotide jobs to pick it up at their `git pull`.
