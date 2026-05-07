#!/bin/bash
#SBATCH --job-name=annitia_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA full pipeline — LONG walltime (48h on `long` partition).
# Identical to short / medium; only --time and --partition differ.
# Resume cache means concurrent submissions cooperate.
#
# Pipeline: BE → CD → HB → BRANCHES → EXTERNAL → FUSION
# Includes every roadmap T1.x + T2.x condition.

source /data/stat-ecr/scat9264/KIRBy/tests/experiments/annitia/_slurm_setup.sh

bash tests/experiments/annitia/run_all_local.sh

echo "=== DONE — final master:"
ls -la tests/results/annitia/full/sub_master.csv 2>/dev/null \
    || echo "  (not produced this run — check /tmp/annitia_runs/*.log)"
