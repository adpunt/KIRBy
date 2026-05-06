#!/bin/bash
#SBATCH --job-name=ab_complexity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Antibody Complexity Theory Experiments (FLAb datasets)
# Uses the same experiment framework as molecular, with antibody-specific
# datasets and mutation-aware representations.
#
# Usage:
#   sbatch run_antibody_complexity.sh quick              # fast test
#   sbatch run_antibody_complexity.sh quick_delta         # with delta reps
#   sbatch run_antibody_complexity.sh full flab_expression_koenig
#   sbatch run_antibody_complexity.sh full_delta flab_binding_koenig

MODE=${1:-quick}
DATASET=${2:-flab_expression_koenig}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK: flab_expression_koenig, mutation_fast reps, Exp 1+2+6 ==="
    python tests/complexity_theory_experiments.py --mode quick \
        --reps mutation_fast \
        --datasets flab_expression_koenig \
        --experiments 1 2 6 \
        --output tests/results/ab_complexity_fast

elif [ "$MODE" = "quick_delta" ]; then
    echo "=== QUICK+DELTA: flab_expression_koenig, mutation_fast + delta, Exp 1+2+6 ==="
    python tests/complexity_theory_experiments.py --mode quick \
        --reps mutation_fast --delta \
        --datasets flab_expression_koenig \
        --experiments 1 2 6 \
        --output tests/results/ab_complexity_fast_delta

elif [ "$MODE" = "full" ]; then
    echo "=== FULL: ${DATASET}, mutation_full reps, all experiments ==="
    python tests/complexity_theory_experiments.py --mode full \
        --reps mutation_full \
        --datasets ${DATASET} \
        --experiments 1 2 3 4 5 6 7 8 9 10 \
        --output tests/results/ab_complexity_${DATASET}

elif [ "$MODE" = "full_delta" ]; then
    echo "=== FULL+DELTA: ${DATASET}, mutation_full + delta, all experiments ==="
    python tests/complexity_theory_experiments.py --mode full \
        --reps mutation_full --delta \
        --datasets ${DATASET} \
        --experiments 1 2 3 4 5 6 7 8 9 10 \
        --output tests/results/ab_complexity_${DATASET}_delta

else
    echo "Usage: sbatch run_antibody_complexity.sh [quick|quick_delta|full|full_delta] [dataset]"
    echo "Datasets: flab_expression_koenig, flab_binding_koenig, flab_thermo_jain,"
    echo "          flab_aggregation_jain, flab_thermo_hie, flab_binding_rosace,"
    echo "          abcov_kd, abcov_ic50, abagym_all, abagym_Ang2_2017_G6,"
    echo "          abagym_EGFR_2013_Cetuximab, abagym_lysozyme_2019_D441,"
    echo "          abagym_VEGF_2017b_G6, abagym_HER2_2021_trastuzumab, abbind"
    exit 1
fi
