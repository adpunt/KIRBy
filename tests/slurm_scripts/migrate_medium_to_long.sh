#!/bin/bash
# Move currently-PENDING precompute jobs from the `medium` partition to `long`.
#
# Only touches jobs in state PD (pending). RUNNING jobs are left alone so
# checkpoints continue. For every pending pc_* job on medium this script:
#   1. Parses the job name into (dataset, condition).
#   2. Checks ~/.cache/kirby/hybrid_registry.json for an entry matching
#      that (dataset, tier/aug_strategy).
#   3. If already in the registry  -> cancel only (don't waste long time).
#   4. If not in the registry      -> cancel + resubmit to long (7d max).
#
# The hybrid search checkpoint (<cache_path>.ckpt.pkl) is preserved across
# cancel/resubmit, so resubmitted jobs resume from the last checkpointed
# fold-0 progress instead of starting over.
#
# Nothing happens until you review the plan and type `y`.

set -uo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
TIME_LIMIT="7-00:00:00"
PARTITION="long"
REGISTRY_PATH="${KIRBY_CACHE_DIR:-$HOME/.cache/kirby}/hybrid_registry.json"

# Condition suffixes ordered longest-prefix-first so `general_distance_none`
# matches before `general_none`, etc. `all_standard` / `all_distance` come from
# submit_precompute_new_nucleotide.sh — one-big-job-per-dataset convention,
# no per-condition split.
CONDITIONS=(
    all_standard
    all_distance
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

# A job name carries the dataset, the condition, and -- since 2026-09-12 -- which
# model list and objective the search was launched with. Before that suffix
# existed the name held only pc_<ds>_<cond>, so a knn, ridge or similarity pass
# that got migrated came back as the plain five-model predictive pass with no
# message. PASS is "" for the plain pass, so old queued names still parse.
# parse_name / pass_flags / pass_suffix live in ONE file, shared with
# submit_precompute_extra_passes.sh. This script held its own copies and two of
# them were WRONG: knn rebuilt without --objectives similarity, and dist was
# missing entirely so a migrated distance job lost --distance and wrote to the
# wrong tier. See the note at the top of that file.
. "$(dirname "$0")/_precompute_passes.sh"


# For per-condition jobs, map to a registry key so we can skip if already done.
# For `_all_*` jobs, we can't cleanly check "every inner condition is cached";
# since these are pending (no checkpoint), just always resubmit. Returns empty
# string to signal "skip registry check, always resubmit".
cond_to_registry_key() {
    case "$1" in
        all_standard|all_distance) echo "" ;;
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
        *) echo "UNKNOWN" ;;
    esac
}

# ---------------------------------------------------------------------------
# 1. Snapshot PENDING medium-partition pc_ jobs (running jobs are left alone)
# ---------------------------------------------------------------------------
echo "Gathering PENDING medium-partition pc_ jobs from squeue..."
RAW=()
while IFS= read -r line; do
    RAW+=("$line")
done < <(squeue -u "$USER" -h -o "%i|%j|%P|%t" | awk -F'|' '$3=="medium" && $4=="PD" && $2 ~ /^pc_/')

N_RAW=${#RAW[@]}
echo "  Found $N_RAW pending jobs on medium."
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
# 3. Classify each pending job
# ---------------------------------------------------------------------------
UNPARSED=()
ALREADY_DONE=()
NEEDS_WORK=()

for row in "${RAW[@]}"; do
    IFS='|' read -r jobid name part st <<< "$row"
    if ! parse_name "$name"; then
        UNPARSED+=("$jobid|$name")
        continue
    fi
    reg_key=$(cond_to_registry_key "$COND")
    if [[ "$reg_key" == "UNKNOWN" ]]; then
        UNPARSED+=("$jobid|$name")
        continue
    fi
    # Empty reg_key = `_all_*` job -> always resubmit (pending, no checkpoint).
    # The knn and ridge passes share a registry key with the plain pass -- they
    # differ only in the model list, which the registry does not record -- so the
    # plain pass's entry would mark a queued knn or ridge job as already cached
    # and cancel it. Only the plain pass may be cancelled on a registry hit.
    if [[ -z "$PASS" && -n "$reg_key" && -n "${DONE[$DS|$reg_key]:-}" ]]; then
        ALREADY_DONE+=("$jobid|$name|$DS|$COND|$PASS")
    else
        NEEDS_WORK+=("$jobid|$name|$DS|$COND|$PASS")
    fi
done

# ---------------------------------------------------------------------------
# 4. Show plan
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Migration plan: medium (PENDING only) -> long"
echo "  Running jobs on medium: LEFT ALONE"
echo "=========================================="

echo ""
echo "ALREADY IN REGISTRY (cancel only, no resubmit):"
if (( ${#ALREADY_DONE[@]} == 0 )); then
    echo "  (none)"
else
    for row in "${ALREADY_DONE[@]}"; do
        IFS='|' read -r jobid name ds cond pass <<< "$row"
        printf "  %-12s  %-44s  %s / %s\n" "$jobid" "$name" "$ds" "$cond"
    done
fi

echo ""
echo "NEEDS RESUBMISSION (cancel + sbatch on long, 7d):"
if (( ${#NEEDS_WORK[@]} == 0 )); then
    echo "  (none)"
else
    for row in "${NEEDS_WORK[@]}"; do
        IFS='|' read -r jobid name ds cond pass <<< "$row"
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
echo "  ${#NEEDS_WORK[@]} cancel + resubmit on long"
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

for row in "${ALREADY_DONE[@]}"; do
    IFS='|' read -r jobid name ds cond pass <<< "$row"
    if scancel "$jobid" 2>/dev/null; then
        echo "  CANCEL-ONLY  $jobid  ($ds / $cond) -- already cached"
    else
        echo "  CANCEL-ONLY  $jobid  FAILED (job already gone?)"
    fi
done

for row in "${NEEDS_WORK[@]}"; do
    IFS='|' read -r jobid name ds cond pass <<< "$row"
    scancel "$jobid" 2>/dev/null || true

    mem="64G"
    [[ "$ds" == nt_* ]] && mem="128G"

    # Build wrap command per job style:
    #   all_standard    -> run every non-distance condition for this dataset
    #   all_distance    -> run every distance condition for this dataset
    #   <tier>_<strat>  -> single condition via --condition
    if [[ "$cond" == "all_standard" ]]; then
        cmd="python tests/precompute_hybrids.py ${ds} --speed full $(pass_flags "$pass")"
    elif [[ "$cond" == "all_distance" ]]; then
        cmd="python tests/precompute_hybrids.py ${ds} --speed full --distance $(pass_flags "$pass")"
    else
        pruning=""
        [[ "$cond" == *_capped ]] && pruning="--pruning 0.25"
        dist=""
        [[ "$cond" == *_distance_* ]] && dist="--distance"
        pflags=$(pass_flags "$pass")
    cmd="python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full ${pruning} ${dist} ${pflags}"
    fi

    new=$(sbatch \
        --job-name="pc_${ds}_${cond}$(pass_suffix "$pass")" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${TIME_LIMIT}" --partition="${PARTITION}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" 2>&1 | awk '/Submitted/ {print $4}')
    if [[ -n "$new" ]]; then
        echo "  MIGRATE      $jobid -> $new  ($ds / $cond${pass:+ / $pass pass}, $mem)"
    else
        echo "  MIGRATE      $jobid  FAILED to resubmit ($ds / $cond${pass:+ / $pass pass})"
    fi
done

echo ""
echo "Done. Monitor with: squeue -u \$USER -o '%.10i %.8P %.40j %.2t %.10M %.10L' | grep pc_"
