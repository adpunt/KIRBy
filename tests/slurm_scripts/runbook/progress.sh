#!/bin/bash
# Which experiments are running, finished, or failed. Read-only. Written
# 2026-09-25 for the runbook; works for every stage.
#
#   bash tests/slurm_scripts/runbook/progress.sh DIR [DIR ...]
#   bash tests/slurm_scripts/runbook/progress.sh tests/results/trial_complexity logs/trial
#   bash tests/slurm_scripts/runbook/progress.sh tests/results/noise_prelim_sweep logs/noise_prelim
#   RECENT_MIN=30 bash tests/slurm_scripts/runbook/progress.sh logs/trial
#
# A DIR holding _job_summaries/ is read as an output directory:
#   finished  one row per experiment run, from _job_summaries/*.tsv, written by
#             submit_dataset.sh and opig_one.sh when the experiment's process
#             ends. When the same (category, dataset, experiment, condition)
#             ran more than once, only its latest row counts, so a rerun that
#             succeeded clears an earlier failure.
#   running   _job_summaries/*.running, one per job, naming the experiment it
#             is inside now, with the last line of that experiment's log.
#             A .running file whose job was killed outright (wall, node
#             failure) is never removed; its age says so.
#   _errors/  every file there, newest first.
# Any other DIR is read as a log directory: the logs written in the last
# RECENT_MIN minutes (default 60), each with its last line.
# Then the user's jobs on arc and htc, if squeue is on PATH.
set -uo pipefail
RECENT_MIN="${RECENT_MIN:-60}"
[ $# -gt 0 ] || { sed -n '2,22p' "$0"; exit 2; }

last_line() { [ -f "$1" ] && tail -n 200 "$1" | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c1-150; }
age_min() {
    local now m
    now=$(date +%s)
    m=$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo "$now")
    echo $(( (now - m) / 60 ))
}

for D in "$@"; do
    if [ -d "$D/_job_summaries" ]; then
        echo "== output ${D}"
        # Latest row per (category, dataset, experiment, condition): column 10
        # is the ISO end time, which sorts as text.
        LATEST=$(cat "$D"/_job_summaries/*.tsv 2>/dev/null | awk -F'\t' '$1!="job"' \
                 | sort -t$'\t' -k10,10 \
                 | awk -F'\t' '{k=$2"|"$3"|"$4"|"$5; row[k]=$0} END {for (k in row) print row[k]}')
        if [ -z "$LATEST" ]; then
            echo "  no experiment has finished yet"
        else
            echo "  finished, by status (latest run of each experiment):"
            printf '%s\n' "$LATEST" | awk -F'\t' '{print $7}' | sed 's/:.*//' | sort | uniq -c | sed 's/^/    /'
            FAILED=$(printf '%s\n' "$LATEST" | awk -F'\t' '$6!="0" && $6!="-"')
            if [ -n "$FAILED" ]; then
                echo "  FAILED (dataset, experiment, condition, exit, status, seconds, log):"
                printf '%s\n' "$FAILED" | sort -t$'\t' -k3,3 -k4,4 \
                    | awk -F'\t' '{printf "    %-22s %-6s %-18s exit=%-4s %-26s %6ss  %s\n",$3,$4,$5,$6,$7,$8,$11}'
            fi
        fi
        RUN_FILES=$(ls "$D"/_job_summaries/*.running 2>/dev/null)
        if [ -n "$RUN_FILES" ]; then
            echo "  running now:"
            for f in $RUN_FILES; do
                IFS=$'\t' read -r EXP COND SINCE LOGF HOST < "$f"
                printf '    %-8s %-18s since %s on %s (file %s min old)\n' "$EXP" "$COND" "$SINCE" "$HOST" "$(age_min "$f")"
                [ "$LOGF" != "-" ] && echo "      last: $(last_line "$LOGF")"
            done
        else
            echo "  running now: none recorded"
        fi
        if [ -d "$D/_errors" ] && [ -n "$(ls -A "$D/_errors" 2>/dev/null)" ]; then
            echo "  _errors/ (newest first):"
            ls -t "$D/_errors" | head -20 | sed 's/^/    /'
        fi
    elif [ -d "$D" ]; then
        echo "== logs ${D}: written in the last ${RECENT_MIN} min"
        N=0
        for f in $(ls -t "$D"/*.log 2>/dev/null); do
            A=$(age_min "$f")
            [ "$A" -le "$RECENT_MIN" ] || break
            printf '    %-58s %4s min ago | %s\n' "$(basename "$f")" "$A" "$(last_line "$f")"
            N=$((N + 1))
        done
        [ "$N" -gt 0 ] || echo "    none"
    else
        echo "== ${D}: does not exist yet"
    fi
    echo
done

if command -v squeue >/dev/null 2>&1; then
    echo "== your jobs (arc and htc)"
    for C in arc htc; do
        echo "  -- ${C}: by state"
        squeue -M "$C" -u "$USER" -h -o '%T' 2>/dev/null | grep -v '^CLUSTER' | sort | uniq -c | sed 's/^/    /'
    done
fi
