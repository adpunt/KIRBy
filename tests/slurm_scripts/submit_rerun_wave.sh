#!/bin/bash
# Precompute-aware experiment re-run wave.
#
# Classifies every dataset by precompute status (check_precompute_status.py)
# and submits run_full_suite.sh jobs accordingly:
#
#   DONE  (general/top AND distance precompute already on disk):
#     1 job/dataset, CATEGORIES="complexity similarity noise".
#     All three reuse the existing precompute + on-disk rep cache, so the
#     expensive search is paid ZERO extra times.
#
#   NOT DONE (partial/missing precompute):
#     2 jobs/dataset:
#       (A) CATEGORIES="complexity noise"  — complexity builds the general/top
#           precompute as it runs; noise reuses it (noise appended after cx).
#       (B) CATEGORIES="similarity"        — builds the distance precompute
#           (and any missing general/top).
#     The cx+noise jobs are submitted iterating the NOT-DONE datasets FORWARD;
#     the similarity jobs iterating them in REVERSE. This staggers the cold
#     general/top builds so cx and sim are unlikely to be building the SAME
#     dataset's cache at the same instant (the only case that double-builds —
#     otherwise they share the cache by filename).
#
# Federated, OPIG, and forward-compat are NOT part of this wave — run them
# AFTER precompute is finished (see RERUN_PLAYBOOK_2026_06_05.md, part 6).
#
# DRY-RUN BY DEFAULT: prints the sbatch commands it WOULD run. Add --submit to
# actually submit. Review the printed plan first.
#
# Usage (login node, repo root, after `. ./setup.sh`):
#   bash tests/slurm_scripts/submit_rerun_wave.sh                 # dry-run, all datasets
#   bash tests/slurm_scripts/submit_rerun_wave.sh --submit        # submit, all datasets
#   bash tests/slurm_scripts/submit_rerun_wave.sh ds1 ds2 ...     # dry-run, only these
#   bash tests/slurm_scripts/submit_rerun_wave.sh --submit ds1 ds2

set -u

SUBMIT=0
RESTRICT=()
for a in "$@"; do
    if [[ "$a" == "--submit" ]]; then
        SUBMIT=1
    else
        RESTRICT+=("$a")
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUITE="$SCRIPT_DIR/run_full_suite.sh"
STATUS="$REPO_ROOT/tests/check_precompute_status.py"

[[ -f "$SUITE"  ]] || { echo "ERROR: $SUITE not found"  >&2; exit 2; }
[[ -f "$STATUS" ]] || { echo "ERROR: $STATUS not found" >&2; exit 2; }

# A dataset counts as ready for BOTH complexity and similarity only when its
# distance conditions are also present, so classify with --distance.
mapfile -t DONE     < <(python "$STATUS" --done       --distance --names-only)
mapfile -t NOTDONE  < <(python "$STATUS" --incomplete --distance --names-only)

# Optional restriction to a user-supplied dataset list (intersect).
in_restrict() {
    [[ ${#RESTRICT[@]} -eq 0 ]] && return 0
    local x; for x in "${RESTRICT[@]}"; do [[ "$x" == "$1" ]] && return 0; done
    return 1
}
filter() { local out=(); local d; for d in "$@"; do in_restrict "$d" && out+=("$d"); done; printf '%s\n' "${out[@]}"; }
if [[ ${#RESTRICT[@]} -gt 0 ]]; then
    mapfile -t DONE    < <(filter "${DONE[@]:-}")
    mapfile -t NOTDONE < <(filter "${NOTDONE[@]:-}")
fi

n_done=${#DONE[@]}; n_notdone=${#NOTDONE[@]}
echo "=================================================================="
echo "  RE-RUN WAVE PLAN   (mode: $([[ $SUBMIT -eq 1 ]] && echo SUBMIT || echo DRY-RUN))"
echo "  DONE (1 job each, cx+sim+noise):      $n_done datasets"
echo "  NOT DONE (2 jobs each, cx+noise / sim): $n_notdone datasets -> $((n_notdone*2)) jobs"
echo "  total jobs: $((n_done + n_notdone*2))"
echo "=================================================================="

do_sbatch() {  # $1 = DATASET, $2 = CATEGORIES
    local cmd=(sbatch --export="DATASET=$1,CATEGORIES=$2" "$SUITE")
    if [[ $SUBMIT -eq 1 ]]; then
        local out; out=$("${cmd[@]}")
        echo "  [$1] CATEGORIES=\"$2\" -> $out"
    else
        echo "  would: sbatch --export=DATASET=$1,CATEGORIES=\"$2\" run_full_suite.sh"
    fi
}

echo; echo "--- DONE datasets: cx + sim + noise (one job each) ---"
for ds in "${DONE[@]:-}"; do
    [[ -z "$ds" ]] && continue
    do_sbatch "$ds" "complexity similarity noise"
done

echo; echo "--- NOT-DONE datasets: complexity+noise (FORWARD order) ---"
for ds in "${NOTDONE[@]:-}"; do
    [[ -z "$ds" ]] && continue
    do_sbatch "$ds" "complexity noise"
done

echo; echo "--- NOT-DONE datasets: similarity (REVERSE order) ---"
for ((i=${#NOTDONE[@]}-1; i>=0; i--)); do
    ds="${NOTDONE[$i]}"
    [[ -z "$ds" ]] && continue
    do_sbatch "$ds" "similarity"
done

echo
if [[ $SUBMIT -eq 0 ]]; then
    echo "DRY-RUN only. Re-run with --submit to actually queue these jobs."
else
    echo "Submitted. Track: squeue -u \$USER --name=kirby_full"
fi
