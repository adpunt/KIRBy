#!/bin/bash
# Read-only ARC state dump. Run on the server; paste the output to Claude.
#
# Gathers, in one go:
#   - Partition capacities (name, max time, CPU usage, node count, state)
#   - Your current queue (pending + running)
#   - Your last 14 days of completed jobs (runtime, mem, timeout/completed state)
#
# None of these commands write anything or touch caches.
#
# Usage:
#   bash tests/scripts/check_arc_state.sh             # defaults: 14-day window
#   bash tests/scripts/check_arc_state.sh 30          # 30-day window
#   bash tests/scripts/check_arc_state.sh 14 >state.txt && cat state.txt

set -euo pipefail

WINDOW_DAYS=${1:-14}
START=$(date -d "${WINDOW_DAYS} days ago" +%Y-%m-%d 2>/dev/null \
        || date -v-"${WINDOW_DAYS}"d +%Y-%m-%d)

echo "============================================================"
echo "ARC state dump for user: ${USER}"
echo "Generated: $(date)"
echo "History window: last ${WINDOW_DAYS} days (from ${START})"
echo "============================================================"

echo
echo "--- [1/5] Partition inventory (sinfo) ---"
echo "    cols: Partition  MaxTime  CPUsAllocIdleOther/Total  Nodes  State"
sinfo -o "%20P %12l %20C %6D %T" || true

echo
echo "--- [2/5] Per-partition summary (sinfo -s) ---"
sinfo -s || true

echo
echo "--- [3/5] Your running + pending queue (squeue) ---"
squeue -u "${USER}" -o "%.10i %20P %30j %2t %10M %10L %6D %R" || true
echo
echo "Counts:"
squeue -u "${USER}" -h -t RUNNING | wc -l | awk '{printf "  running: %s\n", $1}' || true
squeue -u "${USER}" -h -t PENDING | wc -l | awk '{printf "  pending: %s\n", $1}' || true

echo
echo "--- [4/5] Recent job history: completed + timed-out + failed (sacct) ---"
echo "    cols: JobID  JobName  Partition  State  Elapsed  TimeLimit  ReqMem  MaxRSS  NodeList"
sacct -u "${USER}" \
      --starttime="${START}" \
      --format=JobID%-16,JobName%-30,Partition%-10,State%-15,Elapsed%-10,Timelimit%-10,ReqMem%-10,MaxRSS%-10,NodeList%-15 \
      --noheader \
      | grep -Ev '\.batch|\.extern' \
      | head -200 || true

echo
echo "--- [5/5] Timeout hot-spots (jobs killed by TIMEOUT in window) ---"
echo "    counts of (JobName, Partition) pairs that hit timeout"
sacct -u "${USER}" \
      --starttime="${START}" \
      --state=TIMEOUT \
      --format=JobName%-40,Partition%-10 \
      --noheader \
      | grep -Ev '\.batch|\.extern' \
      | sort | uniq -c | sort -rn | head -30 || true

echo
echo "============================================================"
echo "Done. Paste everything above to Claude for analysis."
echo "============================================================"
