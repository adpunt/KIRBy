# server-wipe-procedure

**title:** The ordered server procedure: from 'many recent changes' to a cache that is provably correct

**area:** Cache WIPE and SERVER procedure (wipe_allocation_caches.sh, setup.sh, SLURM scripts, result_db schema guard, run_dataset.py resume)

**class:** C-stale-undetected

**severity as reported:** large

**location:** /Users/apunt/repos/KIRBy/tests/slurm_scripts/wipe_allocation_caches.sh:102; /Users/apunt/repos/KIRBy/setup.sh:55; /Users/apunt/repos/KIRBy/src/kirby/result_cache.py:203-208

## scenario as reported
This is the deliverable, not a defect. What follows is which layer each recent change invalidates and by what mechanism, then the commands. INVALIDATED AUTOMATICALLY, no action — (1) the 2026-09-10 objective change: `objective` is written unconditionally into all three metas (find_best_hybrid.py:597 search, :677 baseline, :908 hybrid_eval), and _load_cache_with_meta walks the EXPECTED meta, so every searches/, baselines/ and hybrid_evals/ entry lacking the key is a stale miss and rewrites in place; (2) the model-tier change: `model_names` is in the search meta (find_best_hybrid.py:605) and `model` is a CONFIG column (result_cache.py:76), so an old catboost/ridge row is simply never selected; (3) the budget cap 500 to 1000: `budgets` is in the search meta (:610) and `budget` is a CONFIG column, so 1000 is a new row and the old ones still hit — correctly, once the wipe below has removed the ones computed under the old carve; (4) the 2026-09-04 metric/task_type change and the 2026-09-10 objective change on result_db: not a miss but a REFUSAL, _ensure_schema raises and names the file. INVALIDATED BY A FILENAME CHANGE, orphaned disk only — fold_scores/alloc: _alloc_in_hash is built from the internal carve indices (find_best_hybrid.py:5481-5482), so the GroupShuffleSplit-to-GroupKFold move lands every file on a new name; baselines written under a non-predictive objective or with tune=True carry their marker in the filename (:207-219), so an old file is orphaned, not served. NOT INVALIDATED, MUST BE DELETED BY HAND — result_db/*.db (raises, so it stops the job rather than lying, but nothing deletes it for you); searches/, hybrid_evals/, hybrid_fold_cache/ against the GroupKFold internal-carve move, which appears in NO key (hybrid.py:4754-4781 internal_holdout_split, deterministic, never hashed); hybrid_fold_cache/ also against the create_hybrid default moving from top_3 to performance_weighted (hybrid.py:4793) and against candidate_ranking being pinned to residual_correlation, which is not in _hybrid_fold_cache_key at all (_shared.py:71-99); fold_scores/ against the 0.0-to-NaN failure encoding (see alloc-failure-encoding-changed). MUST NOT BE DELETED, this is most of the compute — rep_cache/, reps/, baselines/, folds/, ft_embeddings/, finetuned_reps/, esm2_per_residue/, protein_structures/, structures/, mcs_pairs/, cdr_indices/, antibody_reps/ and every raw-data directory.

## evidence as reported
wipe_allocation_caches.sh:102 TARGETS is missing fold_scores. setup.sh:55 `export KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby`. result_cache.py:203-208 raises and names the file. find_best_hybrid.py:6129 `print(f"  ResultCache: {len(_rc_done)} stored configs reused; "` and :7977 `print(f"[cache] {_cache_stats_summary()}")` are the two lines that prove cold-then-warm. check_precompute_status.py:296-300 prints `!! <path>: written under an older column list`. submit_dataset.sh:77-80 turns KIRBY_FORCE_RERUN=1 into --force-rerun; submit_all_datasets.sh:148 `EXPORT_FLAG="--export=ALL,PRUNING_MODE=${PRUNING_MODE}"` carries it into the job.

## fix as proposed
Run from the repo root on the server, in this order.

# 0 — get the code the wipe is for
cd /path/to/KIRBy && git checkout similarity-metrics-study && git pull
git log -1 --format='%h %ad %s' --date=short   # the 2026-09-10 commits must be here

# 1 — point at the real cache and PROVE it before deleting anything
. ./setup.sh
echo "$KIRBY_CACHE_DIR"                        # must print /data/stat-ecr/scat9264/.cache/kirby
ls -d "$KIRBY_CACHE_DIR"/baselines "$KIRBY_CACHE_DIR"/rep_cache   # both must exist

# 2 — inventory BEFORE, so step 7 has something to compare against
ls -1 "$KIRBY_CACHE_DIR"
ls -1 "$KIRBY_CACHE_DIR"/result_db 2>/dev/null | wc -l
ls -1 "$KIRBY_CACHE_DIR"/fold_scores 2>/dev/null | wc -l
find "$KIRBY_CACHE_DIR"/rep_cache -name '*.npy' | wc -l    # this number must NOT change

# 3 — add the missing layer to the wipe script (one word, line 102)
#     TARGETS=(result_db searches fold_scores hybrid_evals hybrid_fold_cache
#              create_hybrid_cache hybrid_search hybrid_registry.json)

# 4 — wipe, on a compute node (hybrid_evals alone is hundreds of thousands of files)
sbatch tests/slurm_scripts/wipe_allocation_caches.sh
#     If you would rather not edit the script, run it as-is and then, by hand:
#       mv "$KIRBY_CACHE_DIR"/fold_scores "$KIRBY_CACHE_DIR"/FS-DELETE && rm -rf "$KIRBY_CACHE_DIR"/FS-DELETE

# 5 — do NOT run either invalidate_*.sh (see invalidate-scripts-wipe-the-home-cache)

# 6 — the experiment JSONs survived the wipe; force past the resume
export KIRBY_FORCE_RERUN=1
python tests/preflight_cache_check.py --all       # row-count guard on the kept rep_cache
python tests/check_precompute_status.py           # expect 'No result cache at ...' or all-missing
bash tests/slurm_scripts/submit_all_datasets.sh <category> ...

# 7 — VERIFY. Three checks, all of which must pass.
# 7a — no old-schema database survived anywhere under the root
python - <<'PY'
import os, glob, sqlite3, sys
sys.path.insert(0, 'src')
from kirby.result_cache import ALL_COLS
root = os.environ['KIRBY_CACHE_DIR']
stale = 0
for p in sorted(glob.glob(os.path.join(root, 'result_db', '*.db'))):
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    have = {r[1] for r in c.execute('PRAGMA table_info(results)')}
    miss = [x for x in ALL_COLS if x not in have]
    n = 0 if miss else c.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    c.close()
    print(('STALE ' if miss else 'ok    ') + p + f'  rows={n} missing={miss}')
    stale += bool(miss)
print('stale files:', stale)   # must be 0
PY
# 7b — the first job's log must show COLD, and a repeat of the same job must show WARM
grep -E 'ResultCache: [0-9]+ stored configs reused|\[cache\] cache stats' slurm-<jobid>.out
#     first run:  '0 stored configs reused', and baseline miss far exceeding baseline hit
#     repeat run: reused > 0, and baseline hit far exceeding miss
#     'meta drift' or 'stale' climbing on the repeat run means a layer is still churning
# 7c — the expensive layers survived
find "$KIRBY_CACHE_DIR"/rep_cache -name '*.npy' | wc -l   # same count as step 2
ls -1 "$KIRBY_CACHE_DIR"/baselines | wc -l                 # non-zero

One thing this procedure does not cover, and I have not verified: tests/analysis/blueprints/<ds>_* on the server. Those are traced from the old best hybrid and sit inside the repo, not under the cache root, so the wipe script never sees them. invalidate_refolded.sh:64-67 deletes them per dataset. If that directory exists on the server it should go with the wipe.
