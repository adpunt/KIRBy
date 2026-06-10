#!/bin/bash
# Submit precompute_hybrids.py runs with adaptive pruning, one SLURM job
# per (dataset, kind=standard|distance). Both `_all_standard` and
# `_all_distance` are submitted per dataset by default.
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute_pruned.sh DATASET [DATASET ...]
#
# Env-var overrides:
#   PRUNING_MODE=default_pruning   (default_pruning | aggressive_pruning)
#   PRUNING_CHECKPOINTS=           override checkpoint budgets (e.g. "50")
#   PRUNING_TAIL_DROP=             override tail-drop fraction (e.g. 0.5)
#   PRUNING_MIN_SAMPLES_GATE=      override n_samples gate (e.g. 1000)
#   KIND=both                      both | standard | distance
#
# Examples:
#
#   # Test single mid-size with aggressive pruning on expensive models
#   PRUNING_MODE=aggressive_pruning \
#       bash tests/slurm_scripts/submit_precompute_pruned.sh caco2
#
#   # Default mode (no multi-stage), one per domain
#   bash tests/slurm_scripts/submit_precompute_pruned.sh \
#       caco2 flab_thermo_jain bh_doyle_plates
#
#   # Aggressive mode with manual gate override (be careful)
#   PRUNING_MODE=aggressive_pruning PRUNING_MIN_SAMPLES_GATE=500 \
#       bash tests/slurm_scripts/submit_precompute_pruned.sh ddg_combined
#
#   # Standard bundle only
#   KIND=standard bash tests/slurm_scripts/submit_precompute_pruned.sh caco2
#
# Output is one job ID per submission, plus a final summary.

set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: $0 DATASET [DATASET ...]" >&2
    echo "  Env: PRUNING_MODE, PRUNING_CHECKPOINTS, PRUNING_TAIL_DROP," >&2
    echo "       PRUNING_MIN_SAMPLES_GATE, KIND" >&2
    exit 1
fi

PRUNING_MODE="${PRUNING_MODE:-default_pruning}"
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
echo "  PRUNING_MODE=${PRUNING_MODE}"
[[ -n "${PRUNING_CHECKPOINTS:-}" ]]      && echo "  PRUNING_CHECKPOINTS=${PRUNING_CHECKPOINTS}"
[[ -n "${PRUNING_TAIL_DROP:-}" ]]        && echo "  PRUNING_TAIL_DROP=${PRUNING_TAIL_DROP}"
[[ -n "${PRUNING_MIN_SAMPLES_GATE:-}" ]] && echo "  PRUNING_MIN_SAMPLES_GATE=${PRUNING_MIN_SAMPLES_GATE}"
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

    # Build the --export string. Only include override env vars when set so
    # we don't shadow the run-script defaults with empty values.
    export_parts=("DATASET=${ds}" "KIND=${k}" "PRUNING_MODE=${PRUNING_MODE}" "MEM=${mem}")
    [[ -n "${PRUNING_CHECKPOINTS:-}" ]]              && export_parts+=("PRUNING_CHECKPOINTS=${PRUNING_CHECKPOINTS}")
    [[ -n "${PRUNING_TAIL_DROP:-}" ]]                && export_parts+=("PRUNING_TAIL_DROP=${PRUNING_TAIL_DROP}")
    [[ -n "${PRUNING_MIN_SAMPLES_GATE:-}" ]]         && export_parts+=("PRUNING_MIN_SAMPLES_GATE=${PRUNING_MIN_SAMPLES_GATE}")

    for k in "${kinds[@]}"; do
        # Per-iteration: replace the placeholder KIND in export_parts
        kind_export_parts=("${export_parts[@]}")
        for i in "${!kind_export_parts[@]}"; do
            [[ "${kind_export_parts[$i]}" == KIND=* ]] && kind_export_parts[$i]="KIND=${k}"
        done
        export_str=$(IFS=,; echo "${kind_export_parts[*]}")
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
