#!/bin/bash
# Resubmit precompute jobs for (dataset, condition) pairs that are NOT DONE
# according to tests/check_precompute_status.py AND are not already queued.
#
# Targets `long` (7d). Uses 64G (128G for nt_*). Preserves any existing
# checkpoint (the new job picks it up on disk).
#
# Flow:
#   1. Walk DATASET_REPS, compute expected (tier, aug_strategy) per dataset
#      using the same logic as check_precompute_status.py, and mark every
#      entry whose cache file is missing / corrupt / empty / stale.
#   2. Snapshot currently-queued pc_ jobs (R + PD). Skip any (dataset, cond)
#      already covered by a live job — we don't want duplicates.
#   3. Print the plan. Require `y` before any sbatch.
#
# Flags:
#   --distance    include distance conditions (off by default)
#
# Safe to re-run: already-DONE and already-queued pairs are skipped.

set -uo pipefail

INCLUDE_DISTANCE=0
for arg in "$@"; do
    case "$arg" in
        --distance) INCLUDE_DISTANCE=1 ;;
    esac
done

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
TIME_LIMIT="7-00:00:00"
PARTITION="long"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Map "tier_augstrategy" -> SLURM job-name suffix (same convention as
# submit_precompute.sh).
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

# ---------------------------------------------------------------------------
# 1. Ask the status-check logic which (dataset, tier, strat) are NOT ok.
# ---------------------------------------------------------------------------
echo "Scanning cache for missing/corrupt precompute entries..."
MISSING=()
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    MISSING+=("$line")
done < <(cd "$REPO_ROOT" && INCLUDE_DISTANCE="$INCLUDE_DISTANCE" python3 - <<'PYEOF'
import os, sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
from experiments.complexity.complexity_helpers import DATASET_REPS
from check_precompute_status import (
    _searches_dir, expected_conditions, _check_file)

include_distance = os.environ.get('INCLUDE_DISTANCE') == '1'
searches_dir = _searches_dir()
for ds in sorted(DATASET_REPS.keys()):
    cfg = DATASET_REPS[ds]
    for tier, strat in expected_conditions(cfg, include_distance=include_distance):
        fname = f"{ds}__{tier}__{strat}__full.pkl"
        status, _ = _check_file(searches_dir / fname)
        if status != 'ok':
            print(f"{ds}|{tier}|{strat}|{status}")
PYEOF
)

echo "  Found ${#MISSING[@]} incomplete (dataset, tier, strat) entries."
if (( ${#MISSING[@]} == 0 )); then
    echo "Nothing missing. Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Snapshot currently queued pc_ jobs so we don't double-submit.
#    Track both per-condition names ({ds}_{cond}) AND the `_all_*`
#    one-job-per-dataset covers.
# ---------------------------------------------------------------------------
declare -A QUEUED               # exact `{ds}_{cond}` match
declare -A QUEUED_ALL_STANDARD  # dataset has a pending `_all_standard` job
declare -A QUEUED_ALL_DISTANCE  # dataset has a pending `_all_distance` job

while IFS='|' read -r jobid name state; do
    [[ -z "$name" ]] && continue
    local_name=${name#pc_}
    if [[ "$local_name" == *_all_standard ]]; then
        ds=${local_name%_all_standard}
        QUEUED_ALL_STANDARD["$ds"]=1
    elif [[ "$local_name" == *_all_distance ]]; then
        ds=${local_name%_all_distance}
        QUEUED_ALL_DISTANCE["$ds"]=1
    else
        QUEUED["$local_name"]=1
    fi
done < <(squeue -u "$USER" -h -o "%i|%j|%t" | awk -F'|' '$2 ~ /^pc_/')

echo "  ${#QUEUED[@]} per-condition pc_* jobs queued (R+PD)."
echo "  ${#QUEUED_ALL_STANDARD[@]} datasets have a pending _all_standard job."
echo "  ${#QUEUED_ALL_DISTANCE[@]} datasets have a pending _all_distance job."

# ---------------------------------------------------------------------------
# 3. Build action plan
# ---------------------------------------------------------------------------
ACTIONS=()        # ds|cond
SKIPPED_QUEUED=() # ds|cond|jobsuffix

for row in "${MISSING[@]}"; do
    IFS='|' read -r ds tier strat status <<< "$row"
    suffix=$(cond_suffix "$tier" "$strat")
    if [[ -z "$suffix" ]]; then
        echo "  WARN: unknown tier '$tier' for $ds, skipping"
        continue
    fi
    job_local="${ds}_${suffix}"
    # Covered by a queued per-condition job?
    if [[ -n "${QUEUED[$job_local]:-}" ]]; then
        SKIPPED_QUEUED+=("$ds|$suffix|per-condition job queued")
        continue
    fi
    # Covered by a queued `_all_*` job for this dataset?
    if [[ "$tier" == *_distance ]]; then
        if [[ -n "${QUEUED_ALL_DISTANCE[$ds]:-}" ]]; then
            SKIPPED_QUEUED+=("$ds|$suffix|_all_distance job queued")
            continue
        fi
    else
        if [[ -n "${QUEUED_ALL_STANDARD[$ds]:-}" ]]; then
            SKIPPED_QUEUED+=("$ds|$suffix|_all_standard job queued")
            continue
        fi
    fi
    ACTIONS+=("$ds|$suffix|$status")
done

# ---------------------------------------------------------------------------
# 4. Print plan
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Resubmit plan: incomplete -> long (7d)"
echo "=========================================="
echo ""
echo "SKIPPED (already queued, will finish on their own):"
if (( ${#SKIPPED_QUEUED[@]} == 0 )); then
    echo "  (none)"
else
    for row in "${SKIPPED_QUEUED[@]}"; do
        IFS='|' read -r ds cond status <<< "$row"
        printf "  %-40s  %-30s  [%s]\n" "$ds" "$cond" "$status"
    done
fi

echo ""
echo "WILL SUBMIT (dataset / condition / current status):"
if (( ${#ACTIONS[@]} == 0 )); then
    echo "  (none)"
else
    for row in "${ACTIONS[@]}"; do
        IFS='|' read -r ds cond status <<< "$row"
        mem="64G"
        [[ "$ds" == nt_* ]] && mem="128G"
        printf "  %-40s  %-30s  [%s] %s\n" "$ds" "$cond" "$mem" "$status"
    done
fi

echo ""
echo "Summary:"
echo "  ${#SKIPPED_QUEUED[@]} skipped (already queued)"
echo "  ${#ACTIONS[@]} will submit to long"
echo "=========================================="

if (( ${#ACTIONS[@]} == 0 )); then
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
for row in "${ACTIONS[@]}"; do
    IFS='|' read -r ds cond status <<< "$row"
    mem="64G"
    [[ "$ds" == nt_* ]] && mem="128G"
    pruning=""
    [[ "$cond" == *_capped ]] && pruning="--pruning 0.25"
    dist=""
    [[ "$cond" == *_distance_* ]] && dist="--distance"

    cmd="python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full ${pruning} ${dist}"
    new=$(sbatch \
        --job-name="pc_${ds}_${cond}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${TIME_LIMIT}" --partition="${PARTITION}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" 2>&1 | awk '/Submitted/ {print $4}')
    if [[ -n "$new" ]]; then
        echo "  SUBMIT  $new  $ds / $cond ($mem)"
    else
        echo "  FAILED to submit $ds / $cond"
    fi
done

echo ""
echo "Done. Monitor: squeue -u \$USER -o '%.10i %.8P %.40j %.2t %.10M %.10L' | grep pc_"
