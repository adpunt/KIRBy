#!/bin/bash
#SBATCH --job-name=models_migrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Migrate HF_HOME, TORCH_HOME, and KIRBY_MODELS_DIR off stat-cadd (100% full,
# causing SIGBUS on mmap reads) onto stat-ecr (4.7T avail).
# Runs on an HTC compute node — login node not involved.
# --ignore-existing: never overwrite anything already at stat-ecr.
# COPY only — does NOT delete source. Verify and delete source manually after.

set -e

migrate_dir() {
    local NAME=$1
    local SRC=$2
    local DST=$3

    echo "=========================================="
    echo "[${NAME}] start: $(date)"
    echo "[${NAME}] src: ${SRC}"
    echo "[${NAME}] dst: ${DST}"

    if [ ! -d "${SRC}" ]; then
        echo "[${NAME}] SOURCE DOES NOT EXIST — skipping"
        return 0
    fi

    mkdir -p "${DST}"

    # Top-level entry count only (no recursive scan — stat-cadd is full).
    echo "[${NAME}] before:  src top-level=$(ls -A ${SRC} 2>/dev/null | wc -l)  dst top-level=$(ls -A ${DST} 2>/dev/null | wc -l)"

    rsync -a --ignore-existing --info=progress2,stats2 "${SRC}/" "${DST}/"
    local RC=$?

    echo "[${NAME}] after:   src top-level=$(ls -A ${SRC} 2>/dev/null | wc -l)  dst top-level=$(ls -A ${DST} 2>/dev/null | wc -l)"
    echo "[${NAME}] rsync exit: ${RC}"
    echo "[${NAME}] done: $(date)"

    return ${RC}
}

migrate_dir "HF_HOME"           /data/stat-cadd/scat9264/.cache/huggingface  /data/stat-ecr/scat9264/.cache/huggingface
migrate_dir "TORCH_HOME"        /data/stat-cadd/scat9264/.cache/torch        /data/stat-ecr/scat9264/.cache/torch
migrate_dir "KIRBY_MODELS_DIR"  /data/stat-cadd/scat9264/kirby_models        /data/stat-ecr/scat9264/kirby_models

echo "=========================================="
echo "ALL MIGRATIONS COMPLETE: $(date)"
echo "Next: verify, then update setup.sh paths to stat-ecr, then delete sources from stat-cadd."
