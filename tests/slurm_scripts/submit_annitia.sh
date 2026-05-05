#!/bin/bash
#SBATCH --job-name=annitia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA MASLD risk-stratification experiment (Trustii challenge).
#
# Runs the full hybrid search (63 view subsets x 3 models = 189 candidates per
# endpoint) over both endpoints, then writes a Trustii submission CSV from the
# winning models against val_test.csv.
#
# Expected runtime: 30-90 min depending on RSF speed. Memory is modest (the
# data is ~1.3 MB and feature views are <200 dims).
#
# Usage:
#   sbatch tests/slurm_scripts/submit_annitia.sh

cd /data/stat-cadd/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore
export ANNITIA_DATA_DIR=/data/stat-cadd/scat9264/KIRBy/data/annitia

echo "ANNITIA MASLD experiment"
echo "Start: $(date)"
echo "Data dir: $ANNITIA_DATA_DIR"
ls -la $ANNITIA_DATA_DIR

# 1. Hybrid search + refit per endpoint. Persists annitia_{hepatic,death}_model.pkl
#    and per-condition C-index tables under tests/results/annitia/.
python tests/experiments/annitia/run.py \
    --models cox,coxnet,rsf,gbsa \
    --prune-levels none,50 \
    --top-k-keep 3 \
    --endpoints hepatic,death \
    --n-folds 5 \
    --output-dir tests/results/annitia

# 2. Generate the Trustii submission CSV from the saved models.
python tests/experiments/annitia/submit.py \
    --validation-csv $ANNITIA_DATA_DIR/val_test.csv \
    --models-dir tests/results/annitia \
    --output tests/results/annitia/annitia_submission.csv

echo "Done: $(date)"
echo "Submission file:"
ls -la tests/results/annitia/annitia_submission.csv
echo "First 5 rows:"
head -6 tests/results/annitia/annitia_submission.csv
