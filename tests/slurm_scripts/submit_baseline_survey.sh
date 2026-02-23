#!/bin/bash
# Submit the baseline survey as parallel SLURM jobs (one per domain).
# Run this on the server login node (NOT via sbatch).
#
# All runs use maximal presets with --adaptive-pruning --test-rep-subsets.
# Fine-tuned reps are NOT included (later task).
#
# Usage:
#   bash tests/slurm_scripts/submit_baseline_survey.sh        # all domains
#   bash tests/slurm_scripts/submit_baseline_survey.sh quick   # molecular + ab_intra only (fastest)

MODE=${1:-all}
SCRIPT=tests/slurm_scripts/run_baseline_survey.sh

echo "Submitting baseline survey jobs (COMPREHENSIVE)..."
echo "  All runs: maximal presets + --adaptive-pruning --test-rep-subsets"
echo ""

if [ "$MODE" = "quick" ]; then
    # Quick survey: molecular (4 runs) + antibody intra (9 runs)
    J1=$(sbatch --job-name=survey_mol   "$SCRIPT" molecular | awk '{print $4}')
    J2=$(sbatch --job-name=survey_ab_in "$SCRIPT" ab_intra  | awk '{print $4}')
    echo "Submitted:"
    echo "  $J1  molecular   (4 datasets × all preset = 4 runs, 15 reps each)"
    echo "  $J2  ab_intra    (9 datasets × dms_full = 9 runs, 17 reps each)"
    echo ""
    echo "Total: 2 jobs, 13 runs"

elif [ "$MODE" = "all" ]; then
    # Full survey: all domains in parallel
    J1=$(sbatch --job-name=survey_mol   "$SCRIPT" molecular | awk '{print $4}')
    J2=$(sbatch --job-name=survey_ab_in "$SCRIPT" ab_intra  | awk '{print $4}')
    J3=$(sbatch --job-name=survey_ab_ex "$SCRIPT" ab_inter  | awk '{print $4}')
    J4=$(sbatch --job-name=survey_tcga  --mem=96G "$SCRIPT" tcga | awk '{print $4}')
    J5=$(sbatch --job-name=survey_dta   --mem=96G "$SCRIPT" dta  | awk '{print $4}')
    echo "Submitted:"
    echo "  $J1  molecular   (4 datasets × all = 4 runs, 15 reps each)"
    echo "  $J2  ab_intra    (9 datasets × dms_full = 9 runs, 17 reps each)"
    echo "  $J3  ab_inter    (6 datasets × all_server = 6 runs, 13 reps each)"
    echo "  $J4  tcga        (3 datasets × all = 3 runs, 27 reps each, 96G mem)"
    echo "  $J5  dta         (2 datasets × full = 2 runs, 10 reps each, 96G mem)"
    echo ""
    echo "Total: 5 jobs, 24 runs"

else
    echo "Usage: bash submit_baseline_survey.sh [all|quick]"
    exit 1
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in:   tests/results/baseline_survey/"
