#!/bin/bash
#SBATCH --job-name=complexity_theory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh
cd tests

# Complexity Theory Experiments
# Tests: intrinsic dim, MI, learning curves, generalization gap, double descent, hybrid comparison
# Datasets: LogD, QM9
# Expected runtime: 1-2 hours

python complexity_theory_experiments.py --full --datasets logd,qm9 --output results/complexity_theory
