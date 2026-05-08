#!/bin/bash
# NHANES T2.x: external-data pretraining + transfer + 5-subset ablation.
# Per A17 audit, prior NHANES round was buggy (zero-shot eval no-op'd, subset
# env vars not forwarded). Both fixed in commits this session — re-run is mandatory.
#
# Coverage:
#   T2.0  NHANES preprocess + pretrain encoder per subset (5 subsets)
#   T2.1  Zero-shot gate eval per subset
#   T2.2  6 transfer recipes a–f per subset that passes T2.1 gate
#   T2.3  Full 5-subset ablation grid (60 cells = 5 subsets × 6 recipes × 2 endpoints)
#
# Usage:
#   bash tests/slurm_scripts/submit_nhanes_t2_htc.sh
#
#SBATCH --job-name=annitia_nhanes_t2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=medium
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=FAIL

set -euo pipefail
cd /data/stat-ecr/scat9264/KIRBy
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export ANNITIA_VAL_TEST=/data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv

# Clean stale dummies if present (per A17 audit recommendation)
if [ -f tests/results/annitia/external/zeroshot.json ]; then
    grep -q "n/a" tests/results/annitia/external/zeroshot.json 2>/dev/null && \
        echo "Clearing stale tests/results/annitia/external/ (broken zero-shot run)" && \
        rm -rf tests/results/annitia/external/encoder_*.pt tests/results/annitia/external/zeroshot.json || true
fi

# All 5 subsets in T2.3 — env-driven via experiment_external.py.
export ANNITIA_T23_SUBSETS="fib4_high,fib4_high_2007,fib4_high_2017,t2dm_only,no_viral"
# Don't gate on T2.1 zero-shot pass — run the full ablation regardless so we
# get the complete grid for the writeup.
export ANNITIA_IGNORE_T21_GATE=1

python -u tests/experiments/annitia/experiment_external.py

echo "DONE NHANES T2.x at $(date)"
