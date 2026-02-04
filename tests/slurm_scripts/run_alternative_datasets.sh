#!/bin/bash
#SBATCH --job-name=alt_datasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh
cd tests

# Run all three datasets: LogD, Caco2, hERG
# 6 strategies × 11 sigmas × ~8 models × 4 reps × 3 datasets
# Uncertainty values saved automatically

python alternative_data_noise_robustness.py --datasets all --results-root results/alternative_full
