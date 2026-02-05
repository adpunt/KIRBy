#!/bin/bash
#SBATCH --job-name=validation_noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=71:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh
cd tests

# Validation Noise Robustness: 3 Regression Datasets with 5-fold Scaffold CV
# Datasets: OpenADMET-LogD, OpenADMET-Caco2_Efflux, ChEMBL-hERG-Ki
# Workload: 4 reps × 8 models × 6 strategies × 11 sigmas × 5 folds × 3 datasets
# Output: summary.csv + *_uncertainty_values.csv per dataset

python alternative_data_noise_robustness.py --datasets all --results-root results/validation
