# invalidate-scripts-wipe-the-home-cache

**title:** The two invalidate_*.sh scripts never source setup.sh and silently target $HOME/.cache/kirby

**area:** Cache WIPE and SERVER procedure (wipe_allocation_caches.sh, setup.sh, SLURM scripts, result_db schema guard, run_dataset.py resume)

**class:** C-stale-undetected

**severity as reported:** large

**location:** /Users/apunt/repos/KIRBy/tests/slurm_scripts/invalidate_allocation_caches.sh:109 and :121; /Users/apunt/repos/KIRBy/tests/slurm_scripts/invalidate_refolded.sh:43

## scenario as reported
On ARC the real cache is /data/stat-ecr/scat9264/.cache/kirby, set by setup.sh:55. Neither invalidate script sources setup.sh. The operator opens a fresh login shell and runs `bash tests/slurm_scripts/invalidate_allocation_caches.sh --go`. KIRBY_CACHE_DIR is unset, so CACHE falls back to $HOME/.cache/kirby. The only guard is `[[ -d "$CACHE" ]]` at :121 — and that directory exists on the login node, because any run made without sourcing setup.sh created it. The script prints `cache root : /home/scat9264/.cache/kirby`, deletes whatever few stray files are there, and reports success. The real result_db, searches and hybrid_fold_cache under /data/stat-ecr are untouched. The next job hits every one of them and serves allocations computed before the GroupShuffleSplit-to-GroupKFold move — the silent-wrong-answer class those scripts were written to prevent. wipe_allocation_caches.sh does not have this hole: it sources setup.sh at :72, refuses on an empty KIRBY_CACHE_DIR at :76-79, and refuses unless baselines/ and rep_cache/ are both present at :86-90.

## evidence as reported
invalidate_allocation_caches.sh:109 `CACHE="${KIRBY_CACHE_DIR:-$HOME/.cache/kirby}"`; :121 `[[ -d "$CACHE" ]] || { echo "cache root does not exist: $CACHE"; exit 1; }` — that is the whole guard. invalidate_refolded.sh:43 `CACHE="${KIRBY_CACHE_DIR:-$HOME/.cache/kirby}"`. Neither file contains the string `setup.sh` (grep over both). Contrast wipe_allocation_caches.sh:72 `. ./setup.sh`, :74 `CACHE="${KIRBY_CACHE_DIR:-}"` (no home fallback), :86 `if [ ! -d "${CACHE}/baselines" ] || [ ! -d "${CACHE}/rep_cache" ]; then`.

## fix as proposed
Three scripts now delete overlapping sets of cache layers and disagree about which ones matter, so this needs your decision rather than a patch I pick. The two options, with what each costs: (a) delete invalidate_allocation_caches.sh and invalidate_refolded.sh and keep wipe_allocation_caches.sh as the one wipe — costs the per-dataset scoping (--datasets) and the dry-run that only the invalidate scripts have; (b) keep them and give each the three guards wipe_allocation_caches.sh already has: `. ./setup.sh`, no $HOME fallback (`CACHE="${KIRBY_CACHE_DIR:-}"` plus an abort on empty), and the baselines/+rep_cache/ presence check. Either way, until it is decided: do not run either invalidate script on the server, and if you must, run `. ./setup.sh` first and read the `cache root :` line it prints before typing --go.
