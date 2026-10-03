#!/bin/bash
# Run ONE of the three experiments that work across datasets, in its own python
# process. Written 2026-10-03 for submit_all_stages.sh stage 3 (owner: submit
# them with the rest, with SLURM dependencies).
#
#   bash tests/slurm_scripts/runbook/cross_one.sh s09 <dataset> <dataset> ...
#   bash tests/slurm_scripts/runbook/cross_one.sh c03 <dataset> <dataset> ...
#   bash tests/slurm_scripts/runbook/cross_one.sh c14 <dataset> <dataset> ...
#
# None of the three is in a per-dataset job. complexity_helpers.py and
# similarity_helpers.py leave them out of SINGLE_DATASET_EXPERIMENTS and out of
# their module maps, so run_dataset.py refuses them by name.
#
# WHAT EACH ONE RUNS (argparse read 2026-10-03):
#   s09  similarity/s09_cross_dataset_summary.py --datasets ... --results-dir
#        $KIRBY_OUTPUT_DIR --output $KIRBY_OUTPUT_DIR. Reads
#        <dir>/s02_hybrid_similarity/<dataset>.json and the same for s04, s05,
#        s06 and s08, as the per-dataset similarity jobs write them. No model
#        fits. Writes <dir>/s09_cross_dataset_summary/results.json.
#   c03  complexity/c03_cross_dataset_transfer.py --mode full --datasets ...
#        --output $KIRBY_OUTPUT_DIR. Reads the precompute's stored searches for
#        each dataset; fits SVR and ElasticNet on regression datasets. Holds
#        every dataset in memory at once. Writes
#        <dir>/c03_cross_dataset_transfer/multi.json.
#   c14  complexity/c14_cross_domain_summary.py --mode full --tasks a,b,...
#        --output $KIRBY_OUTPUT_DIR. Runs its own nested searches, uncached.
#        Writes <dir>/c14_cross_domain_summary/results.json once, at the end.
#
# RESUBMITTING. c03 and c14 are skipped when their latest row in
# $KIRBY_OUTPUT_DIR/_job_summaries/*.tsv says ok, as opig_one.sh does. s09 is
# never skipped: it takes minutes, and the similarity files it reads may have
# grown since its last run. KIRBY_FORCE_RERUN=1 turns the skip off.
# Neither c03 nor c14 saves partial results: a cut-off run starts again.
#
# ENVIRONMENT
#   KIRBY_OUTPUT_DIR   required: <output-root>/similarity for s09,
#                      <output-root>/complexity for c03 and c14
#   KIRBY_EXP_LOG_DIR  default logs/cross; log <dir>/cross_<name>.log
#   KIRBY_NO_COLD_SEARCH, KIRBY_LGBM_THREADS (default 1), KIRBY_SKIP_INSTALLS
#   (default 1), KIRBY_REPO: as in submit_dataset.sh.
#
# Records the exit code in <KIRBY_OUTPUT_DIR>/_job_summaries/<jobid>.tsv, same
# columns as submit_dataset.sh (category 'cross', dataset 'multi'), and exits
# with it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"

NAME="${1:?usage: cross_one.sh s09|c03|c14 <dataset>...}"; shift
[ $# -gt 0 ] || { echo "cross_one: no datasets given"; exit 2; }
OUTDIR="${KIRBY_OUTPUT_DIR:?cross_one: set KIRBY_OUTPUT_DIR}"
LOGDIR="${KIRBY_EXP_LOG_DIR:-logs/cross}"

case "$NAME" in
    s09) ARGV=(tests/experiments/similarity/s09_cross_dataset_summary.py
               --datasets "$@" --results-dir "$OUTDIR" --output "$OUTDIR") ;;
    c03) ARGV=(tests/experiments/complexity/c03_cross_dataset_transfer.py
               --mode full --datasets "$@" --output "$OUTDIR") ;;
    c14) ARGV=(tests/experiments/complexity/c14_cross_domain_summary.py
               --mode full --tasks "$(echo "$@" | tr ' ' ',')" --output "$OUTDIR") ;;
    *)   echo "cross_one: unknown '$NAME' (s09, c03 or c14)"; exit 2 ;;
esac

cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS="${KIRBY_SKIP_INSTALLS:-1}"
. ./setup.sh
# Same value and reasons as submit_dataset.sh.
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
export PYTHONWARNINGS=ignore
[ -n "${KIRBY_NO_COLD_SEARCH:-}" ] && export KIRBY_NO_COLD_SEARCH

JOB_TAG="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$OUTDIR/_job_summaries" "$LOGDIR"
SUMMARY="$OUTDIR/_job_summaries/${JOB_TAG}.tsv"
RUNNING="$OUTDIR/_job_summaries/${JOB_TAG}.running"
[ -f "$SUMMARY" ] || printf 'job\tcategory\tdataset\texperiment\tcondition\texit_code\tstatus\tseconds\tstarted\tended\tlog\n' > "$SUMMARY"
LOGF="$LOGDIR/cross_${NAME}.log"

if [ "$NAME" != s09 ] && [ "${KIRBY_FORCE_RERUN:-0}" != "1" ]; then
    PREV=$(cat "$OUTDIR"/_job_summaries/*.tsv 2>/dev/null \
           | awk -F'\t' -v n="$NAME" '$1!="job" && $2=="cross" && $4==n' \
           | sort -t$'\t' -k10,10 | tail -n 1 | awk -F'\t' '{print $1, $7}')
    case "${PREV#* }" in
        ok|done_earlier)
            echo "=== cross ${NAME}: finished in job ${PREV%% *}; skipped"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" cross multi "$NAME" "-" \
                0 done_earlier 0 "$(date -Iseconds)" "$(date -Iseconds)" "job ${PREV%% *}" >> "$SUMMARY"
            exit 0 ;;
    esac
fi

echo "=== cross ${NAME}: python -u ${ARGV[*]}"
echo "  datasets ($#): $*   log ${LOGF}   node $(hostname)"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}  cold-search refusal: ${KIRBY_NO_COLD_SEARCH:-off}"
printf '%s\t%s\t%s\t%s\t%s\n' "cross_${NAME}" "-" "$(date -Iseconds)" "$LOGF" "$(hostname)" > "$RUNNING"
T0=$(date +%s); STARTED=$(date -Iseconds)
python -u "${ARGV[@]}" < /dev/null 2>&1 | tee "$LOGF"
RC=${PIPESTATUS[0]}
SECS=$(( $(date +%s) - T0 ))
case "$RC" in 0) STATUS=ok ;; 124) STATUS=timeout ;; 137) STATUS=killed_9_oom_or_wall ;; 139) STATUS=segfault ;; *) STATUS=failed ;; esac
rm -f "$RUNNING"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" cross multi "$NAME" "-" \
    "$RC" "$STATUS" "$SECS" "$STARTED" "$(date -Iseconds)" "$LOGF" >> "$SUMMARY"
echo "=== cross ${NAME} exit=${RC} ${STATUS} ${SECS}s"
exit "$RC"
