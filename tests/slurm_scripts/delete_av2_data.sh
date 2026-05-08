#!/bin/bash
#SBATCH --job-name=av2_delete
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# DESTRUCTIVE: deletes ~793G of AV2 raw scenario mining data.
# Public dataset — re-downloadable from Argoverse 2 S3 if needed later.
# Runs on compute node so login node isn't burdened by the metadata-heavy walk.
#
# Two-stage delete:
#   1. mv to staging path (atomic, instant)
#   2. rm -rf the staging path (slow, but on compute node where it doesn't matter)
# This means if the rm fails halfway, the source is already gone from the
# original path — re-running this script just continues cleanup.

set -e

DATA_DIR=/data/stat-cadd/scat9264/av2
STAGING=/data/stat-cadd/scat9264/av2-DELETE-STAGING

echo "=========================================="
echo "AV2 DELETE: $(date)"
echo "  source:  ${DATA_DIR}"
echo "  staging: ${STAGING}"
echo "=========================================="

# Stage 1: rename if source exists
if [ -d "${DATA_DIR}" ]; then
    echo "[stage 1] moving ${DATA_DIR} -> ${STAGING}"
    mv "${DATA_DIR}" "${STAGING}"
    echo "[stage 1] renamed at $(date)"
elif [ -d "${STAGING}" ]; then
    echo "[stage 1] source already moved to staging (resuming previous delete)"
else
    echo "[stage 1] nothing to delete — both paths absent"
    exit 0
fi

# Stage 2: recursive delete on compute node
echo "[stage 2] rm -rf ${STAGING} starting at $(date)"
rm -rf "${STAGING}"
RC=$?
echo "[stage 2] rm exit: ${RC} at $(date)"

# Confirm both gone
if [ -e "${DATA_DIR}" ] || [ -e "${STAGING}" ]; then
    echo "WARNING: residue remains:"
    [ -e "${DATA_DIR}" ] && ls -ld "${DATA_DIR}"
    [ -e "${STAGING}" ] && ls -ld "${STAGING}"
    exit 1
fi

echo "=========================================="
echo "AV2 DELETE COMPLETE: $(date)"
echo "Reclaimed approx 793G on /data/stat-cadd/"
echo "=========================================="
