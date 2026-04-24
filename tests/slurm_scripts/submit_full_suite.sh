#!/bin/bash
# Submit run_full_suite.sh for a list of datasets, one SLURM job each.
# Run from the repo root OR the slurm_scripts/ dir — either works.
#
# Usage (antibody defaults):
#   bash tests/slurm_scripts/submit_full_suite.sh
#
# Usage (custom list):
#   bash tests/slurm_scripts/submit_full_suite.sh flab_thermo_tresanco humanness_therapeutic caco2
#
# Usage (only some categories):
#   CATEGORIES="complexity noise" bash tests/slurm_scripts/submit_full_suite.sh flab_binding_adams
#
# The script prints the SLURM job id for each submission so you can track.
# Kill all with: scancel -u $USER --name=kirby_full

set -u

# Default antibody + small-n set that's "DONE" per the latest diagnose run.
# Replace/extend as needed.
DEFAULT_DATASETS=(
    # Fully precomputed DONE list (2026-04-22 diagnose snapshot)
    abagym_Ang2
    abagym_EGFR
    abagym_HER2
    abagym_VEGF
    abcov_kd
    flab_immunogenicity_marks
    flab_polyreact_jain_bvp
    flab_polyreact_jain_cic
    humanness_therapeutic
)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Resolve the SLURM script path relative to this launcher.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SLURM_SCRIPT="$SCRIPT_DIR/run_full_suite.sh"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "ERROR: $SLURM_SCRIPT not found" >&2
    exit 2
fi

echo "Submitting ${#DATASETS[@]} dataset jobs..."
echo

EXPORT_EXTRA=""
if [[ -n "${CATEGORIES:-}" ]]; then
    EXPORT_EXTRA=",CATEGORIES=\"$CATEGORIES\""
fi

for DS in "${DATASETS[@]}"; do
    OUT=$(sbatch --export="DATASET=$DS${EXPORT_EXTRA}" "$SLURM_SCRIPT")
    echo "  $DS -> $OUT"
done

echo
echo "Track with: squeue -u \$USER --name=kirby_full"
echo "Logs:       slurm-<JOBID>.out in \$PWD on the server"
