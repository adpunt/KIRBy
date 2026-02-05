#!/bin/bash
#SBATCH --job-name=herg_noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh
cd tests

# Run hERG classification only
# 6 classification strategies × 11 sigmas × ~7 models × 4 reps
python alternative_data_noise_robustness.py --datasets herg --results-root results/alternative_full
