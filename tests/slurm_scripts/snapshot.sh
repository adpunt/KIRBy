#!/bin/bash
#SBATCH --job-name=kirby_snapshot
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#
# ONE-SHOT progress snapshot.
#   - FULL per-dataset status (check_precompute_status --distance) is appended
#     to monitor_<date>.log on disk.
#   - A CONCISE summary is emailed: jobs running/pending + one line per dataset
#     (precompute % and cx/sim/noise experiment fractions).
# Pre-schedule many of these at staggered --begin times with schedule_snapshots.sh.
#
# Email target: override with EMAIL=you@x sbatch ... ; default below.

REPO=/data/stat-ecr/scat9264/KIRBy
EMAIL="${EMAIL:-adelaide.punt@stcatz.ox.ac.uk}"
cd "$REPO"
. ./setup.sh          # check_precompute_status imports the kirby package

LOG="$REPO/monitor_$(date +%Y-%m-%d).log"
STAMP="$(date '+%Y-%m-%d %H:%M')"
TMP_STATUS="$(mktemp)"

# Per-dataset precompute + experiment status (no --timing: fast).
python tests/check_precompute_status.py --distance > "$TMP_STATUS" 2>&1

RUN=$(squeue -u "$USER" -h -t RUNNING | wc -l)
PEND=$(squeue -u "$USER" -h -t PENDING | wc -l)

# Simple per-dataset table from the status output's data rows:
#   $1 name  $2 DONE/partial/missing  $3 pre# (n/n)  $6 cx  $7 sim  $8 noise.
# precompute % = conditions present / total.
SIMPLE="$(awk '
  /^  [A-Za-z]/ && ($2=="DONE" || $2=="partial" || $2=="missing") {
    split($3, p, "/");
    pct = (p[2] > 0) ? int(p[1]*100/p[2] + 0.5) : 0;
    printf "  %-34s precompute %3d%%   cx %-7s sim %-7s noise %-7s\n", \
           $1, pct, $6, $7, $8;
  }' "$TMP_STATUS")"

# FULL detail -> on-disk log
{
  echo "================ $STAMP ================"
  echo "Jobs: running=$RUN  pending=$PEND"
  cat "$TMP_STATUS"
  echo
} >> "$LOG" 2>&1

# CONCISE summary -> email (best-effort; never fails the job)
if command -v mail >/dev/null 2>&1; then
  { echo "KIRBy progress — $STAMP"; echo;
    echo "Jobs: running=$RUN  pending=$PEND"; echo;
    echo "Per-dataset (precompute % | experiments done):"; echo "$SIMPLE"; } \
    | mail -s "KIRBy $STAMP — run=$RUN pend=$PEND" "$EMAIL" 2>/dev/null || true
fi

rm -f "$TMP_STATUS"
