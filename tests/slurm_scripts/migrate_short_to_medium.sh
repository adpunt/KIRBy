#!/bin/bash
# Move currently-queued precompute jobs from the `short` partition to `medium`.
#
# For every pc_* job on short (running or pending) this script:
#   1. Parses the job name into (dataset, condition).
#   2. Checks ~/.cache/kirby/hybrid_registry.json for an entry matching
#      that (dataset, tier/aug_strategy).
#   3. If already in the registry  -> cancel only (don't waste medium time).
#   4. If not in the registry      -> cancel + resubmit to medium.
#
# The hybrid search checkpoint (<cache_path>.ckpt.pkl) is preserved across
# cancel/resubmit, so resubmitted jobs resume from the last checkpointed
# fold-0 progress instead of starting over.
#
# Nothing happens until you review the plan and type `y`.

set -uo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
TIME_LIMIT="2-00:00:00"
PARTITION="medium"
REGISTRY_PATH="${KIRBY_CACHE_DIR:-$HOME/.cache/kirby}/hybrid_registry.json"

# Condition suffixes ordered longest-prefix-first so `general_distance_none`
# matches before `general_none`, etc.
CONDITIONS=(
    general_distance_uncapped
    general_distance_none
    top_distance_uncapped
    top_distance_none
    general_uncapped
    general_capped
    general_none
    top_uncapped
    top_capped
    top_none
)

parse_name() {
    local name=${1#pc_}
    DS=""
    COND=""
    for c in "${CONDITIONS[@]}"; do
        if [[ "$name" == *"_${c}" ]]; then
            COND="$c"
            DS="${name%_${c}}"
            return 0
        fi
    done
    return 1
}

cond_to_registry_key() {
    # Registry keys are "<tier>/<aug_strategy>" (see precompute_hybrids.py
    # _update_registry), while job conditions are "<tier>_<aug_strategy>".
    case "$1" in
        general_distance_none)      echo "general_distance/none" ;;
        general_distance_uncapped)  echo "general_distance/uncapped" ;;
        top_distance_none)          echo "top_distance/none" ;;
        top_distance_uncapped)      echo "top_distance/uncapped" ;;
        general_none)               echo "general/none" ;;
        general_uncapped)           echo "general/uncapped" ;;
        general_capped)             echo "general/capped" ;;
        top_none)                   echo "top/none" ;;
        top_uncapped)               echo "top/uncapped" ;;
        top_capped)                 echo "top/capped" ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# 1. Snapshot short-partition pc_ jobs
# ---------------------------------------------------------------------------
echo "Gathering short-partition pc_ jobs from squeue..."
RAW=()
while IFS= read -r line; do
    RAW+=("$line")
done < <(squeue -u "$USER" -h -o "%i|%j|%P" | awk -F'|' '$3=="short" && $2 ~ /^pc_/')

N_RAW=${#RAW[@]}
echo "  Found $N_RAW jobs on short."
if (( N_RAW == 0 )); then
    echo "Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Load the registry into a bash set for O(1) lookups
# ---------------------------------------------------------------------------
declare -A DONE
if [[ -f "$REGISTRY_PATH" ]]; then
    while IFS='|' read -r ds key; do
        [[ -z "$ds" ]] && continue
        DONE["$ds|$key"]=1
    done < <(python3 - "$REGISTRY_PATH" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for ds, entry in r.items():
    for key in entry.get('configs', {}):
        print(f"{ds}|{key}")
PYEOF
)
    echo "  Loaded ${#DONE[@]} registry entries from $REGISTRY_PATH"
else
    echo "  WARNING: registry not found at $REGISTRY_PATH"
    echo "  Every job will be treated as needs-work."
fi

# ---------------------------------------------------------------------------
# 3. Classify each short job
# ---------------------------------------------------------------------------
UNPARSED=()
ALREADY_DONE=()
NEEDS_WORK=()

for row in "${RAW[@]}"; do
    IFS='|' read -r jobid name part <<< "$row"
    if ! parse_name "$name"; then
        UNPARSED+=("$jobid|$name")
        continue
    fi
    reg_key=$(cond_to_registry_key "$COND")
    if [[ -z "$reg_key" ]]; then
        UNPARSED+=("$jobid|$name")
        continue
    fi
    if [[ -n "${DONE[$DS|$reg_key]:-}" ]]; then
        ALREADY_DONE+=("$jobid|$name|$DS|$COND")
    else
        NEEDS_WORK+=("$jobid|$name|$DS|$COND")
    fi
done

# ---------------------------------------------------------------------------
# 4. Show plan
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Migration plan: short -> medium"
echo "=========================================="

echo ""
echo "ALREADY IN REGISTRY (cancel only, no resubmit):"
if (( ${#ALREADY_DONE[@]} == 0 )); then
    echo "  (none)"
else
    for row in "${ALREADY_DONE[@]}"; do
        IFS='|' read -r jobid name ds cond <<< "$row"
        printf "  %-12s  %-44s  %s / %s\n" "$jobid" "$name" "$ds" "$cond"
    done
fi

echo ""
echo "NEEDS RESUBMISSION (cancel + sbatch on medium):"
if (( ${#NEEDS_WORK[@]} == 0 )); then
    echo "  (none)"
else
    for row in "${NEEDS_WORK[@]}"; do
        IFS='|' read -r jobid name ds cond <<< "$row"
        mem="64G"
        [[ "$ds" == nt_* ]] && mem="128G"
        printf "  %-12s  %-44s  %s / %s  [%s]\n" "$jobid" "$name" "$ds" "$cond" "$mem"
    done
fi

if (( ${#UNPARSED[@]} > 0 )); then
    echo ""
    echo "COULD NOT PARSE (left alone, not touched):"
    for row in "${UNPARSED[@]}"; do
        IFS='|' read -r jobid name <<< "$row"
        printf "  %-12s  %s\n" "$jobid" "$name"
    done
fi

echo ""
echo "Summary:"
echo "  ${#ALREADY_DONE[@]} cancel-only (already cached)"
echo "  ${#NEEDS_WORK[@]} cancel + resubmit on medium"
echo "  ${#UNPARSED[@]} left alone (unparseable name)"
echo "=========================================="

TOTAL_ACTION=$(( ${#ALREADY_DONE[@]} + ${#NEEDS_WORK[@]} ))
if (( TOTAL_ACTION == 0 )); then
    echo "Nothing to do."
    exit 0
fi

echo ""
read -rp "Proceed with cancel + resubmit? [y/N] " ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
    echo "Aborted. No changes made."
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. Execute
# ---------------------------------------------------------------------------
echo ""
echo "Executing..."

# Cancel the already-done ones (they're wasting cycles for nothing)
for row in "${ALREADY_DONE[@]}"; do
    IFS='|' read -r jobid name ds cond <<< "$row"
    if scancel "$jobid" 2>/dev/null; then
        echo "  CANCEL-ONLY  $jobid  ($ds / $cond) -- already cached"
    else
        echo "  CANCEL-ONLY  $jobid  FAILED (job already gone?)"
    fi
done

# Cancel + resubmit the needs-work ones
for row in "${NEEDS_WORK[@]}"; do
    IFS='|' read -r jobid name ds cond <<< "$row"
    scancel "$jobid" 2>/dev/null || true

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
        echo "  MIGRATE      $jobid -> $new  ($ds / $cond, $mem)"
    else
        echo "  MIGRATE      $jobid  FAILED to resubmit ($ds / $cond)"
    fi
done

echo ""
echo "Done. Monitor with: squeue -u \$USER -o '%.10i %.8P %.40j %.2t %.10M %.10L' | grep pc_"
