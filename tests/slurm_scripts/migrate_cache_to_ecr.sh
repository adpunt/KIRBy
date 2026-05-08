#!/bin/bash
#SBATCH --job-name=cache_migrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# One-time cache migration: copy stat-cadd cache files to stat-ecr.
# Runs on an HTC compute node — login node not involved.
# --ignore-existing: never overwrite the 167 files already at stat-ecr.

SRC=/data/stat-cadd/scat9264/.cache/kirby/
DST=/data/stat-ecr/scat9264/.cache/kirby/

echo "start: $(date)"
echo "before:  src $(ls ${SRC}searches/ 2>/dev/null | wc -l)  dst $(ls ${DST}searches/ 2>/dev/null | wc -l)"

mkdir -p "${DST}"
rsync -a --ignore-existing --info=progress2 "${SRC}" "${DST}"
RC=$?

echo "after:   src $(ls ${SRC}searches/ 2>/dev/null | wc -l)  dst $(ls ${DST}searches/ 2>/dev/null | wc -l)"
echo "rsync exit: ${RC}"
echo "done: $(date)"
exit ${RC}
