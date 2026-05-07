#!/bin/bash
#SBATCH --job-name=annitia_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA full pipeline — SHORT walltime (1h on `short` partition).
# Identical to medium / long; only --time and --partition differ.
# Resume cache means a second submission picks up where this left off.
#
# Pipeline: BE → CD → HB → BRANCHES → EXTERNAL → FUSION
# Includes every roadmap T1.x + T2.x condition.

REPO=/data/stat-ecr/scat9264/KIRBy
cd $REPO
. ./setup.sh

export ANNITIA_VAL_TEST=${ANNITIA_VAL_TEST:-/data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv}
export ANNITIA_TSFM_BACKEND=${ANNITIA_TSFM_BACKEND:-fft}
export ANNITIA_K_DISC=${ANNITIA_K_DISC:-10}
export ANNITIA_N_SYNTH=${ANNITIA_N_SYNTH:-50,100,200,500}
export ANNITIA_T23_SUBSETS=${ANNITIA_T23_SUBSETS:-fib4_high,fib4_high_2007,fib4_high_2017,t2dm_only,no_viral}

bash tests/experiments/annitia/run_all_local.sh

echo "=== DONE — final master:"
ls -la tests/results/annitia/full/sub_master.csv 2>/dev/null \
    || echo "  (not produced — check earlier stage logs)"
