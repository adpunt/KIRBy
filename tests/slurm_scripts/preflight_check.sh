#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Stale-cache pre-flight on a COMPUTE node (proper env + real RAM).
# The login node can't run this: it lacks the venv deps (Levenshtein/torch)
# and the memory to load the big datasets. This short job does, and schedules
# fast — so you get a conclusive answer long before the 7-day jobs start.
#
# Submit:  sbatch tests/slurm_scripts/preflight_check.sh
# Then read: slurm-<jobid>.out  (look for the STALE / ALL CLEAR summary)

cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin main 2>&1 || echo "WARNING: git pull failed; using on-disk code"
. setup.sh

echo "=================================================================="
echo "PRE-FLIGHT stale-cache check (all datasets)"
echo "Node: $(hostname)   Start: $(date)"
echo "=================================================================="

python -u tests/preflight_cache_check.py --all

echo "=================================================================="
echo "Done: $(date)"
echo "=================================================================="
