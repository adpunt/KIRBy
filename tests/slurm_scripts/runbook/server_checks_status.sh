#!/bin/bash
# One table for the server checks launched by server_checks.sh. Read-only.
# Written 2026-09-25.
#
#   bash tests/slurm_scripts/runbook/server_checks_status.sh
#   bash tests/slurm_scripts/runbook/server_checks_status.sh --not-passed
#   bash tests/slurm_scripts/runbook/server_checks_status.sh --tail noise_herg_ki_n07
#
# One row per planned command (logs/server_checks/commands.tsv):
#   category, experiment, dataset, condition, state, cache, check, detail
# STATE
#   waiting      not started yet (the dispatcher holds at most --max-parallel)
#   running      srun alive; detail = minutes since the log last changed, and its last line
#   passed       exit 0. Noise: run 1 and run 2 exit 0 AND check_noise_run.py
#                printed RESULT: PASS (rows on disk, run 2 all cache hits,
#                inputs read match the precompute). A deliberate skip written
#                by the experiment shows as passed, with "skipped" in detail.
#   cut_off      srun ended by the time limit. detail says whether the log
#                holds an error before the cut; if not, it was progressing,
#                and detail is the log's last line.
#   failed       any other non-zero exit; detail = the last error line of the log
# CACHE (noise only): PASS when check_noise_run.py recorded no failure in its
#   section (b), run 2's cache hits; FAIL(n) otherwise; - when the check did not run.
# CHECK (noise only): PASS / FAIL with the count per section, a=finished and
#   rows, b=cache, c=inputs against the precompute.
# _errors/ files newer than the launch that name the dataset and experiment are
# counted in detail.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"
LOGD="${SC_LOGD:-$REPO/logs/server_checks}"
CMDS="$LOGD/commands.tsv"
ONLY_BAD=false

case "${1:-}" in
    --not-passed) ONLY_BAD=true ;;
    --tail)
        F="$LOGD/${2:?usage: --tail <category>_<dataset>_<experiment>}.log"
        [ -f "$F" ] || { echo "no log $F"; exit 1; }
        tail -n "${3:-60}" "$F"; exit 0 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac

[ -f "$CMDS" ] || { echo "No $CMDS: server_checks.sh has not launched anything (a --dry-run writes nothing)."; exit 1; }

age_min() {
    local now m
    now=$(date +%s)
    m=$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo "$now")
    echo $(( (now - m) / 60 ))
}
clean() { tr '\r' '\n' | grep -v '^[[:space:]]*$'; }
last_line() { [ -f "$1" ] && tail -n 300 "$1" | clean | grep -v 'slurmstepd\|^srun: ' | tail -n 1 | cut -c1-140; }
# The last line that names an error: a Python exception line first, then
# srun/slurm errors, then check FAIL lines.
last_error() {
    [ -f "$1" ] || return
    local L
    L=$(clean < "$1" | grep -E '^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Interrupt|Exit)(:|$)' | tail -n 1)
    [ -n "$L" ] || L=$(clean < "$1" | grep -E 'srun: error|slurmstepd: error|Killed|Segmentation|Bus error|oom' | tail -n 1)
    [ -n "$L" ] || L=$(clean < "$1" | grep -E '^\s*FAIL|Traceback|ERROR' | tail -n 1)
    echo "$L" | sed 's/^[[:space:]]*//' | cut -c1-140
}
count_errors() {  # DIR STARTEDFILE DS EXP -> number of _errors/ files naming both
    local n=0 f
    [ -d "$1" ] || { echo 0; return; }
    for f in "$1"/*; do
        [ -f "$f" ] && [ "$f" -nt "$2" ] || continue
        if { basename "$f" | grep -q -- "$3" || grep -q -- "$3" "$f"; } \
           && { basename "$f" | grep -q -- "$4" || grep -q -- "$4" "$f"; }; then
            n=$((n + 1))
        fi
    done
    echo "$n"
}

declare -A NSTATE
ROWS=()
while IFS=$'\t' read -r NAME CAT DS EXP COND OUT _CMD; do
    [ -n "${NAME:-}" ] || continue
    LOG="$LOGD/${NAME}.log"; RCF="$LOGD/${NAME}.rc"; ST="$LOGD/${NAME}.started"
    CACHE="-"; CHECK="-"; DETAIL=""
    if [ ! -f "$ST" ]; then
        STATE="waiting"
    elif [ ! -f "$RCF" ]; then
        STATE="running"
        DETAIL="log $(age_min "$LOG")m old | $(last_line "$LOG")"
    else
        RC=$(cat "$RCF")
        if grep -q 'DUE TO TIME LIMIT' "$LOG" 2>/dev/null; then
            STATE="cut_off"
            if clean < "$LOG" | grep -q 'Traceback'; then
                DETAIL="error before the cut: $(last_error "$LOG")"
            else
                DETAIL="progressing, no error in log | $(last_line "$LOG")"
            fi
        elif [ "$RC" = "0" ]; then
            STATE="passed"
        else
            STATE="failed"
            DETAIL="exit ${RC}: $(last_error "$LOG")"
        fi
    fi

    if [ "$CAT" = "noise" ] && [ -f "$ST" ]; then
        STEPS="$LOGD/${NAME}.steps"
        CT="$OUT/check.txt"
        if [ -f "$CT" ] && grep -q '^RESULT:' "$CT"; then
            NB=$(grep -c '^  - b:' "$CT"); NA=$(grep -c '^  - a:' "$CT"); NC=$(grep -c '^  - c:' "$CT")
            if [ "$NB" -eq 0 ]; then CACHE="PASS"; else CACHE="FAIL($NB)"; fi
            if grep -q '^RESULT: PASS' "$CT"; then CHECK="PASS"; else CHECK="FAIL a${NA} b${NB} c${NC}"; fi
            if [ "$STATE" = "failed" ] && [ "$CHECK" != "PASS" ]; then
                DETAIL="check: $(grep -m1 '^  - ' "$CT" | sed 's/^  - //' | cut -c1-130)"
            fi
        fi
        if [ -f "$STEPS" ] && [ "$STATE" = "failed" ]; then
            R1=$(awk '$1=="run1"{print $2}' "$STEPS"); R2=$(awk '$1=="run2"{print $2}' "$STEPS")
            if [ -n "$R1" ] && [ "$R1" != "0" ]; then
                DETAIL="run 1 exit ${R1}: $(last_error "$LOG")"
            elif [ -n "$R2" ] && [ "$R2" != "0" ]; then
                DETAIL="run 2 exit ${R2}: $(last_error "$LOG")"
            fi
        fi
        if [ -f "$OUT/run1/_run_one_manifest.json" ] \
           && grep -q '"status": "skipped"' "$OUT/run1/_run_one_manifest.json"; then
            DETAIL="skipped by the experiment (see log). ${DETAIL}"
        fi
        NE=$(( $(count_errors "$OUT/run1/_errors" "$ST" "$DS" "$EXP") + $(count_errors "$OUT/run2/_errors" "$ST" "$DS" "$EXP") ))
        [ "$NE" -gt 0 ] && DETAIL="_errors/:${NE} ${DETAIL}"
    elif [ "$CAT" != "opig" ] && [ -f "$ST" ]; then
        NE=$(count_errors "$OUT/_errors" "$ST" "$DS" "$EXP")
        [ "$NE" -gt 0 ] && DETAIL="_errors/:${NE} ${DETAIL}"
        S=$(grep -E "^<<< ${CAT} ${DS} ${EXP} exit=" "$LOG" 2>/dev/null | tail -n 1 | awk '{print $6}')
        [ -n "$S" ] && [ "$S" != "ok" ] && DETAIL="status ${S}. ${DETAIL}"
    elif [ "$CAT" = "opig" ] && [ -f "$ST" ]; then
        S=$(grep -E "^=== opig ${EXP} exit=" "$LOG" 2>/dev/null | tail -n 1 | awk '{print $5}')
        [ -n "$S" ] && [ "$S" != "ok" ] && DETAIL="status ${S}. ${DETAIL}"
    fi

    NSTATE[$STATE]=$(( ${NSTATE[$STATE]:-0} + 1 ))
    [ "$ONLY_BAD" = true ] && [ "$STATE" = "passed" ] && continue
    ROWS+=("$(printf '%-10s %-7s %-14s %-17s %-8s %-9s %-16s %s' \
        "$CAT" "$EXP" "$DS" "$COND" "$STATE" "$CACHE" "$CHECK" "$DETAIL")")
done < "$CMDS"

[ -f "$LOGD/launch.txt" ] && sed 's/^/# /' "$LOGD/launch.txt"
if [ -f "$LOGD/_dispatcher.pid" ] && kill -0 "$(cat "$LOGD/_dispatcher.pid")" 2>/dev/null; then
    echo "# dispatcher running (pid $(cat "$LOGD/_dispatcher.pid"))"
else
    echo "# dispatcher not running (finished, or stopped: see logs/server_checks/_dispatcher.log)"
fi
echo
printf '%-10s %-7s %-14s %-17s %-8s %-9s %-16s %s\n' category exp dataset condition state cache check detail
[ "${#ROWS[@]}" -gt 0 ] && printf '%s\n' "${ROWS[@]}" | sort -k1,1 -k3,3 -k2,2
echo
printf 'totals:'
for S in passed failed cut_off running waiting; do printf '  %s %s' "$S" "${NSTATE[$S]:-0}"; done
echo
