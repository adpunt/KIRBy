#!/bin/bash
# Size every top-level entry under the stat-cadd home, one at a time.
#
# Run it inside an interactive allocation, never on arc-login:
#   srun -p interactive --cpus-per-task=2 --mem=16G --time=4:00:00 --pty /bin/bash
#   bash tests/slurm_scripts/measure_stat_cadd.sh 2>&1 | tee ~/cadd_sizes.txt
#
# Read-only. Deletes nothing.
#
# Entries are ordered by how likely they are to be both large and disposable,
# so the interactive wall running out still leaves the answer on screen.
# Each entry gets 15 minutes; a slower tree prints TIMEOUT and the walk moves on.

ROOT=/data/stat-cadd/scat9264
PER_ENTRY_SECONDS=900

FIRST="conda_pkgs conda_envs .cache KIRBy qsar_qm_models kirby_data kirby_models
       pull_backup_2026-08-28 pull_backup_2026-09-01_2322 arc_untracked_2026-08-28
       results results_hybrid_experiments_20251211_024905
       py311-kirby .micromamba .local"

size_one() {
    local path="$1"
    [ -e "${path}" ] || return 0
    printf '%s  ' "$(date +%H:%M:%S)"
    timeout "${PER_ENTRY_SECONDS}" du -sh "${path}" 2>/dev/null \
        || echo "TIMEOUT (over ${PER_ENTRY_SECONDS}s)  ${path}"
}

echo "start: $(date)"
df -h /data/stat-cadd
echo "===== likely large, likely disposable ====="

seen=""
for name in ${FIRST}; do
    size_one "${ROOT}/${name}"
    seen="${seen} ${name}"
done

echo "===== everything else ====="
for entry in "${ROOT}"/* "${ROOT}"/.[!.]*; do
    [ -e "${entry}" ] || continue
    name=$(basename "${entry}")
    case " ${seen} " in *" ${name} "*) continue ;; esac
    size_one "${entry}"
done

echo "done: $(date)"
