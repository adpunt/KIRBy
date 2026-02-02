#!/bin/bash
#SBATCH --job-name=hybrid_effectiveness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd KIRBy/tests/

# Run hybrid effectiveness analysis on OpenADMET Caco-2 Efflux with all representations
python hybrid_effectiveness_analysis.py --dataset openadmet_caco2_efflux --all-samples --budget 1000

# Optionally run on other datasets
# python hybrid_effectiveness_analysis.py --dataset openadmet_logd --all-samples --budget 1000
# python hybrid_effectiveness_analysis.py --dataset genentech_hlm --all-samples --budget 1000
# python hybrid_effectiveness_analysis.py --dataset herg --all-samples --budget 1000
