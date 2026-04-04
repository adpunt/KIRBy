#!/bin/bash
# Submit F14 greedy tolerance experiment for herg_ki and tcga_brca_pam50.
#
# Usage:
#   bash tests/slurm_scripts/submit_framework_f14.sh
#   bash tests/slurm_scripts/submit_framework_f14.sh --dry-run

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/data/stat-cadd/scat9264/KIRBy"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

submit() {
    local name=$1
    local dataset=$2
    local mem=$3

    CMD="sbatch --parsable \
        --job-name=f14_${name} \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=11:59:00 --partition=short \
        --mem=${mem} \
        --wrap=\"cd ${PROJECT_DIR} && . ./setup.sh && python tests/experiments/framework/f14_greedy_tolerance.py --dataset ${dataset}\""

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $CMD"
    else
        echo "Submitting: f14_${name} (${mem})"
        eval $CMD
    fi
}

echo "F14: Greedy Tolerance Experiment"
echo "================================"
submit herg     herg_ki          64G
submit brca     tcga_brca_pam50  128G

echo ""
echo "Results will be in tests/results/framework/f14_greedy_tolerance_*.json"
