#!/bin/bash
#SBATCH --job-name=importance_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh
cd tests

# Importance Method & Aggregation Benchmark
# Datasets: OpenADMET-LogD, QM9 HOMO-LUMO gap (5k samples)
# Models: RF, MLP, LightGBM
# Budgets: 50, 100, 200
# Tests: 16 importance methods, 32+ aggregation strategies, 7 allocation strategies
# Expected runtime: 3-5 hours

python benchmark_importance_aggregation.py --full --output results/importance_benchmark
