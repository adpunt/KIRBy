#!/bin/bash
# Resubmit precompute using the `_all_standard` / `_all_distance`
# convention: one SLURM job per (dataset, standard|distance) that internally
# loops over every tier × aug_strategy. Much lower queue pressure than the
# per-condition resubmit (resubmit_missing.sh).
#
# Flow:
#   1. Identify incomplete (dataset, tier, strat) from the same status-check
#      logic used elsewhere.
#   2. Group by (dataset, is_distance). Each (ds, non-distance) becomes an
#      `_all_standard` submission; each (ds, distance) becomes an
#      `_all_distance` submission.
#   3. Skip datasets already covered by a queued `_all_*` job or per-condition
#      job. A dataset still needs bundling only if at least one of its
#      (tier, strat) entries is missing AND not covered.
#   4. Show plan. Require `y`.
#
# Targets `long` (7d). 64G standard, 128G for nt_* / gb_* nucleotide loaders.

set -uo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
TIME_LIMIT="7-00:00:00"
PARTITION="long"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Pull incomplete (dataset, tier, strat) from check_precompute_status
#    helpers. Includes distance conditions unconditionally — that's the
#    whole point of the bundled script.
# ---------------------------------------------------------------------------
echo "Scanning cache for missing/corrupt precompute entries (incl. distance)..."
MISSING=()
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    MISSING+=("$line")
done < <(cd "$REPO_ROOT" && python3 - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
from experiments.complexity.complexity_helpers import DATASET_REPS
from check_precompute_status import (
    _searches_dir, expected_conditions, _check_file)

searches_dir = _searches_dir()
for ds in sorted(DATASET_REPS.keys()):
    cfg = DATASET_REPS[ds]
    for tier, strat in expected_conditions(cfg, include_distance=True):
        fname = f"{ds}__{tier}__{strat}__full.pkl"
        status, _ = _check_file(searches_dir / fname)
        if status != 'ok':
            print(f"{ds}|{tier}|{strat}|{status}")
PYEOF
)
echo "  Found ${#MISSING[@]} incomplete entries across all datasets."
if (( ${#MISSING[@]} == 0 )); then
    echo "Nothing missing. Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Snapshot current pc_ queue. Track three kinds of coverage:
#    - exact per-condition jobs: `{ds}_{suffix}`
#    - `_all_standard` jobs per dataset
#    - `_all_distance` jobs per dataset
# ---------------------------------------------------------------------------
declare -A QUEUED_PERCOND
declare -A QUEUED_ALL_STANDARD
declare -A QUEUED_ALL_DISTANCE
while IFS='|' read -r jobid name state; do
    [[ -z "$name" ]] && continue
    local_name=${name#pc_}
    if [[ "$local_name" == *_all_standard ]]; then
        QUEUED_ALL_STANDARD["${local_name%_all_standard}"]=1
    elif [[ "$local_name" == *_all_distance ]]; then
        QUEUED_ALL_DISTANCE["${local_name%_all_distance}"]=1
    else
        QUEUED_PERCOND["$local_name"]=1
    fi
done < <(squeue -u "$USER" -h -o "%i|%j|%t" | awk -F'|' '$2 ~ /^pc_/')

echo "  ${#QUEUED_PERCOND[@]} per-condition pc_* jobs queued."
echo "  ${#QUEUED_ALL_STANDARD[@]} datasets with pending/running _all_standard."
echo "  ${#QUEUED_ALL_DISTANCE[@]} datasets with pending/running _all_distance."

# ---------------------------------------------------------------------------
# 3. Group missing entries by (dataset, standard|distance). A dataset joins
#    NEEDS_STANDARD / NEEDS_DISTANCE if at least one (tier, strat) is:
#      - missing, AND
#      - not covered by a queued `_all_*` job of the right flavor, AND
#      - not covered by a queued per-condition job for that exact suffix.
# ---------------------------------------------------------------------------
cond_suffix() {
    local tier=$1 strat=$2
    case "${tier}" in
        general)           echo "general_${strat}" ;;
        top)               echo "top_${strat}" ;;
        general_distance)  echo "general_distance_${strat}" ;;
        top_distance)      echo "top_distance_${strat}" ;;
        *) echo "" ;;
    esac
}

declare -A NEEDS_STANDARD          # ds -> 1 (need _all_standard bundle)
declare -A NEEDS_DISTANCE          # ds -> 1 (need _all_distance bundle)
declare -A COVERED_BY_ALL_STD      # ds -> 1 (skipped because _all_standard queued)
declare -A COVERED_BY_ALL_DIST     # ds -> 1 (skipped because _all_distance queued)
declare -A COVERED_BY_PERCOND      # ds -> count of conditions already covered per-cond

for row in "${MISSING[@]}"; do
    IFS='|' read -r ds tier strat status <<< "$row"
    is_distance=0
    [[ "$tier" == *_distance ]] && is_distance=1

    if (( is_distance == 1 )); then
        if [[ -n "${QUEUED_ALL_DISTANCE[$ds]:-}" ]]; then
            COVERED_BY_ALL_DIST["$ds"]=1
            continue
        fi
    else
        if [[ -n "${QUEUED_ALL_STANDARD[$ds]:-}" ]]; then
            COVERED_BY_ALL_STD["$ds"]=1
            continue
        fi
    fi

    suffix=$(cond_suffix "$tier" "$strat")
    if [[ -n "$suffix" && -n "${QUEUED_PERCOND[${ds}_${suffix}]:-}" ]]; then
        COVERED_BY_PERCOND["$ds"]=$(( ${COVERED_BY_PERCOND[$ds]:-0} + 1 ))
        continue
    fi

    if (( is_distance == 1 )); then
        NEEDS_DISTANCE["$ds"]=1
    else
        NEEDS_STANDARD["$ds"]=1
    fi
done

# ---------------------------------------------------------------------------
# 4. Print plan
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Bundled resubmit plan -> long (7d)"
echo "  Each job = one dataset, all tier×strat conditions (standard OR distance)"
echo "=========================================="

if (( ${#COVERED_BY_ALL_STD[@]} > 0 )); then
    echo ""
    echo "Already covered by queued _all_standard (skipped):"
    for ds in "${!COVERED_BY_ALL_STD[@]}"; do
        printf "  %s\n" "$ds"
    done | sort
fi
if (( ${#COVERED_BY_ALL_DIST[@]} > 0 )); then
    echo ""
    echo "Already covered by queued _all_distance (skipped):"
    for ds in "${!COVERED_BY_ALL_DIST[@]}"; do
        printf "  %s\n" "$ds"
    done | sort
fi
if (( ${#COVERED_BY_PERCOND[@]} > 0 )); then
    echo ""
    echo "Partially covered by per-condition jobs (skipped those conditions):"
    for ds in "${!COVERED_BY_PERCOND[@]}"; do
        printf "  %-40s  %d conditions covered\n" "$ds" "${COVERED_BY_PERCOND[$ds]}"
    done | sort
fi

echo ""
echo "WILL SUBMIT _all_standard bundles:"
if (( ${#NEEDS_STANDARD[@]} == 0 )); then
    echo "  (none)"
else
    for ds in "${!NEEDS_STANDARD[@]}"; do
        mem="64G"
        [[ "$ds" == nt_* || "$ds" == gb_* ]] && mem="128G"
        printf "  %-40s  [%s]\n" "$ds" "$mem"
    done | sort
fi

echo ""
echo "WILL SUBMIT _all_distance bundles:"
if (( ${#NEEDS_DISTANCE[@]} == 0 )); then
    echo "  (none)"
else
    for ds in "${!NEEDS_DISTANCE[@]}"; do
        mem="64G"
        [[ "$ds" == nt_* || "$ds" == gb_* ]] && mem="128G"
        printf "  %-40s  [%s]\n" "$ds" "$mem"
    done | sort
fi

TOTAL=$(( ${#NEEDS_STANDARD[@]} + ${#NEEDS_DISTANCE[@]} ))
echo ""
echo "Summary:"
echo "  ${#NEEDS_STANDARD[@]} _all_standard bundles"
echo "  ${#NEEDS_DISTANCE[@]} _all_distance bundles"
echo "  ${TOTAL} total sbatch submissions"
echo "=========================================="

if (( TOTAL == 0 )); then
    echo "Nothing to submit."
    exit 0
fi

echo ""
read -rp "Proceed with sbatch to long? [y/N] " ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
    echo "Aborted. No submissions made."
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. Submit
# ---------------------------------------------------------------------------
echo ""
echo "Submitting..."

submit_bundle() {
    local ds=$1
    local kind=$2   # "standard" or "distance"
    local mem="64G"
    [[ "$ds" == nt_* || "$ds" == gb_* ]] && mem="128G"

    local dist_flag=""
    [[ "$kind" == "distance" ]] && dist_flag="--distance"

    local cmd="python tests/precompute_hybrids.py ${ds} --speed full ${dist_flag}"
    local new
    new=$(sbatch \
        --job-name="pc_${ds}_all_${kind}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${TIME_LIMIT}" --partition="${PARTITION}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" 2>&1 | awk '/Submitted/ {print $4}')
    if [[ -n "$new" ]]; then
        echo "  SUBMIT  $new  pc_${ds}_all_${kind} ($mem)"
    else
        echo "  FAILED  pc_${ds}_all_${kind}"
    fi
}

for ds in $(printf '%s\n' "${!NEEDS_STANDARD[@]}" | sort); do
    submit_bundle "$ds" "standard"
done
for ds in $(printf '%s\n' "${!NEEDS_DISTANCE[@]}" | sort); do
    submit_bundle "$ds" "distance"
done

echo ""
echo "Done. Monitor: squeue -u \$USER -o '%.10i %.8P %.40j %.2t %.10M %.10L' | grep pc_"
