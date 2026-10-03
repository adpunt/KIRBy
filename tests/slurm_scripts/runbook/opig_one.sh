#!/bin/bash
# Run ONE OPIG script, by its name in submit_everything.sh's OPIG_JOBS, in its
# own python process, with its output sent somewhere other than the real
# results tree. Written 2026-09-25 for runbook stage 2.
#
#   bash tests/slurm_scripts/runbook/opig_one.sh <name>
#   bash tests/slurm_scripts/runbook/opig_one.sh --list
#
# WHAT IT RUNS. The command line OPIG_JOBS gives that name
# (submit_everything.sh:127-139), with ' PM' replaced by
# '--pruning-mode $PRUNING_MODE' exactly as submit_everything.sh does it, plus
# an output argument. The arguments each script takes for its output were read
# from its argparse on 2026-09-25:
#   --output <file.json>  humanness, trastuzumab, ginkgo, graphinity, qsar,
#                         cgr_comparison (each defaults to
#                         tests/results/comparisons/opig_<x>.json),
#                         scaffold_similarity (opig_scaffold_similarity.py:549)
#   --output <dir>        blueprint_transfer (opig_blueprint_transfer.py:856,903)
#   --output-dir <dir>    shared_vectors_census (:774), shared_vectors_remedy
#                         (:642), scaffold_hybrid (:281)
# A name not in that table runs with no output argument and a warning, because
# it would write to its own default under tests/results.
#
# The datasets each script reads are its own; OPIG_JOBS's fourth column lists
# them. Stage 2's --datasets does not narrow them.
#
# ENVIRONMENT
#   KIRBY_OUTPUT_DIR   default tests/results/trial_opig
#   KIRBY_EXP_LOG_DIR  default logs/trial; log <dir>/opig_<name>.log
#   PRUNING_MODE       default calibrated (what submit_everything.sh passes)
#   KIRBY_NO_COLD_SEARCH, KIRBY_LGBM_THREADS (default 1), KIRBY_SKIP_INSTALLS
#   (default 1), KIRBY_REPO: as in submit_dataset.sh.
#
# Records the exit code in <KIRBY_OUTPUT_DIR>/_job_summaries/<jobid>.tsv, same
# columns as submit_dataset.sh, and exits with it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"
SE="$REPO/tests/slurm_scripts/submit_everything.sh"

opig_table() {
    # The quoted block between OPIG_JOBS=' and the closing ' line.
    awk "/^OPIG_JOBS='/{on=1; next} on && /^'/{exit} on && NF" "$SE"
}

if [ "${1:-}" = "--list" ]; then
    opig_table | awk -F'|' '{printf "%-24s %-5s reads: %s\n", $1, $2, $4}'
    exit 0
fi

NAME="${1:?usage: opig_one.sh <name> | --list}"
ROW=$(opig_table | awk -F'|' -v n="$NAME" '$1==n')
[ -n "$ROW" ] || { echo "opig_one: no '$NAME' in OPIG_JOBS of $SE (try --list)"; exit 2; }
IFS='|' read -r _N _MEM CMD DSLIST <<< "$ROW"

PRUNING_MODE="${PRUNING_MODE:-calibrated}"
OUTDIR="${KIRBY_OUTPUT_DIR:-tests/results/trial_opig}"
LOGDIR="${KIRBY_EXP_LOG_DIR:-logs/trial}"
CMD=${CMD/ PM/ --pruning-mode ${PRUNING_MODE}}

case "$NAME" in
    humanness|trastuzumab|ginkgo|graphinity|qsar|cgr_comparison|scaffold_similarity)
        OUT_ARGS=(--output "${OUTDIR}/${NAME}.json") ;;
    blueprint_transfer)
        OUT_ARGS=(--output "${OUTDIR}/${NAME}") ;;
    shared_vectors_census|shared_vectors_remedy|scaffold_hybrid)
        OUT_ARGS=(--output-dir "${OUTDIR}/${NAME}") ;;
    *)
        echo "WARNING: no output argument known for '$NAME'; it writes to its own default"
        OUT_ARGS=() ;;
esac

cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS="${KIRBY_SKIP_INSTALLS:-1}"
. ./setup.sh
# Same value and reasons as submit_dataset.sh.
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
export PYTHONWARNINGS=ignore
[ -n "${KIRBY_NO_COLD_SEARCH:-}" ] && export KIRBY_NO_COLD_SEARCH

JOB_TAG="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$OUTDIR/_job_summaries" "$LOGDIR" "$OUTDIR"
SUMMARY="$OUTDIR/_job_summaries/${JOB_TAG}.tsv"
RUNNING="$OUTDIR/_job_summaries/${JOB_TAG}.running"
[ -f "$SUMMARY" ] || printf 'job\tcategory\tdataset\texperiment\tcondition\texit_code\tstatus\tseconds\tstarted\tended\tlog\n' > "$SUMMARY"
LOGF="$LOGDIR/opig_${NAME}.log"

# FINISHED IN AN EARLIER JOB (added 2026-10-03, as in submit_dataset.sh): when
# the latest row for this script in $OUTDIR/_job_summaries/*.tsv says ok, a
# resubmitted job exits 0 without starting python. A script cut off part-way
# has no ok row; it reruns and reads back the units it finished
# (_checkpoint.script_checkpoint). KIRBY_FORCE_RERUN=1 turns this skip off.
if [ "${KIRBY_FORCE_RERUN:-0}" != "1" ]; then
    PREV=$(cat "$OUTDIR"/_job_summaries/*.tsv 2>/dev/null \
           | awk -F'\t' -v n="$NAME" '$1!="job" && $2=="opig" && $4==n' \
           | sort -t$'\t' -k10,10 | tail -n 1 | awk -F'\t' '{print $1, $7}')
    case "${PREV#* }" in
        ok|done_earlier)
            echo "=== opig ${NAME}: finished in job ${PREV%% *}; skipped"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" opig "${DSLIST:--}" "$NAME" "-" \
                0 done_earlier 0 "$(date -Iseconds)" "$(date -Iseconds)" "job ${PREV%% *}" >> "$SUMMARY"
            exit 0 ;;
    esac
fi

# CMD is word-split on purpose: it is a script name and its flags, as
# submit_everything.sh passes it to --wrap.
# shellcheck disable=SC2206
ARGV=($CMD)
echo "=== opig ${NAME}: python -u tests/experiments/${CMD} ${OUT_ARGS[*]}"
echo "  reads: ${DSLIST:-(no dataset named)}   log ${LOGF}   node $(hostname)"
printf '%s\t%s\t%s\t%s\t%s\n' "opig_${NAME}" "-" "$(date -Iseconds)" "$LOGF" "$(hostname)" > "$RUNNING"
T0=$(date +%s); STARTED=$(date -Iseconds)
python -u "tests/experiments/${ARGV[0]}" "${ARGV[@]:1}" "${OUT_ARGS[@]}" < /dev/null 2>&1 | tee "$LOGF"
RC=${PIPESTATUS[0]}
SECS=$(( $(date +%s) - T0 ))
case "$RC" in 0) STATUS=ok ;; 124) STATUS=timeout ;; 137) STATUS=killed_9_oom_or_wall ;; 139) STATUS=segfault ;; *) STATUS=failed ;; esac
rm -f "$RUNNING"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" opig "${DSLIST:--}" "$NAME" "-" \
    "$RC" "$STATUS" "$SECS" "$STARTED" "$(date -Iseconds)" "$LOGF" >> "$SUMMARY"
echo "=== opig ${NAME} exit=${RC} ${STATUS} ${SECS}s"
exit "$RC"
