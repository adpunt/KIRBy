#!/bin/bash
# Submit precompute_hybrids.py runs with multi-stage pruning, one SLURM job
# per (dataset, kind=standard|distance). Both `_all_standard` and
# `_all_distance` are submitted per dataset by default.
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute_pruned.sh DATASET [DATASET ...]
#
# Env-var overrides:
#   PRUNING_PRESET=baseline                 (default; current single-stage prune)
#   MODEL_OVERRIDE_PRESET=gp_aggressive     (default; aggressive on GP only)
#   KIND=both                               (both | standard | distance)
#
# Examples:
#
#   # Test on a single mid-size dataset to see if multi-stage works
#   bash tests/slurm_scripts/submit_precompute_pruned.sh caco2
#
#   # One dataset per domain (molecular + antibody + reaction)
#   bash tests/slurm_scripts/submit_precompute_pruned.sh \
#       caco2 flab_thermo_jain bh_doyle_plates
#
#   # Override the preset for an aggressive run
#   PRUNING_PRESET=medium MODEL_OVERRIDE_PRESET=gp_svm_aggressive \
#       bash tests/slurm_scripts/submit_precompute_pruned.sh caco2
#
#   # Just the standard bundle, skip distance
#   KIND=standard bash tests/slurm_scripts/submit_precompute_pruned.sh caco2
#
# Output is one job ID per submission, plus a final summary.

set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: $0 DATASET [DATASET ...]" >&2
    echo "  Env: PRUNING_PRESET, MODEL_OVERRIDE_PRESET, KIND" >&2
    exit 1
fi

PRUNING_PRESET="${PRUNING_PRESET:-baseline}"
# 'none' = no per-model override. Currently the only available value because
# named per-model presets were removed pending real f16 calibration.
MODEL_OVERRIDE_PRESET="${MODEL_OVERRIDE_PRESET:-none}"
KIND="${KIND:-both}"

case "$KIND" in
    both|standard|distance) ;;
    *) echo "KIND must be both|standard|distance, got '$KIND'" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_precompute_pruned.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "Missing: $RUN_SCRIPT" >&2; exit 1
fi

datasets=("$@")
echo "Submitting ${#datasets[@]} dataset(s) with:"
echo "  PRUNING_PRESET=${PRUNING_PRESET}"
echo "  MODEL_OVERRIDE_PRESET=${MODEL_OVERRIDE_PRESET}"
echo "  KIND=${KIND}"
echo ""

submitted=()
for ds in "${datasets[@]}"; do
    # 128G for nucleotide datasets, 64G otherwise
    mem="64G"
    [[ "$ds" == nt_* || "$ds" == gb_* ]] && mem="128G"

    kinds=()
    case "$KIND" in
        both)     kinds=(standard distance) ;;
        standard) kinds=(standard) ;;
        distance) kinds=(distance) ;;
    esac

    for k in "${kinds[@]}"; do
        export_str="DATASET=${ds},KIND=${k},PRUNING_PRESET=${PRUNING_PRESET},MODEL_OVERRIDE_PRESET=${MODEL_OVERRIDE_PRESET},MEM=${mem}"
        jobid=$(sbatch \
            --job-name="pc_${ds}_all_${k}" \
            --mem="${mem}" \
            --export="${export_str}" \
            "${RUN_SCRIPT}" \
            | awk '{print $4}')
        if [[ -n "$jobid" ]]; then
            echo "  SUBMIT  ${jobid}  pc_${ds}_all_${k}  (${mem})"
            submitted+=("$jobid")
        else
            echo "  FAILED  pc_${ds}_all_${k}"
        fi
    done
done

echo ""
echo "=========================================="
echo "  Submitted ${#submitted[@]} jobs"
if (( ${#submitted[@]} > 0 )); then
    first="${submitted[0]}"
    last="${submitted[-1]}"
    echo "  Range: ${first}-${last}"
fi
echo "=========================================="
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -o '%.10i %.8P %.40j %.2t %.10M %.10L' | grep pc_"
echo ""
echo "Status (once any complete):"
echo "  python tests/check_precompute_status.py --done"
