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
#   - FULL detail (queue + check_precompute_status --timing + audit_full_wave)
#     is appended to monitor_<date>.log on disk.
#   - A CONCISE summary (queue counts + precompute totals) is emailed, so you
#     can track progress without logging in.
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

# Run the (slow) timing scan once; reuse for both log and email.
python tests/check_precompute_status.py --timing --distance > "$TMP_STATUS" 2>&1

QUEUE="$(squeue -u "$USER" -h -o '%T' | sort | uniq -c)"
TOTALS="$(grep -E 'Total datasets checked|Ready for experiments' "$TMP_STATUS")"

# FULL block -> on-disk log
{
  echo "================ $STAMP ================"
  echo "--- queue (state counts) ---"; echo "$QUEUE"
  echo "--- precompute timing ---"; cat "$TMP_STATUS"
  echo "--- wave snapshot (audit_full_wave) ---"
  python tests/audit_full_wave.py --include-running --eta
  echo
} >> "$LOG" 2>&1

# CONCISE summary -> email (best-effort; never fails the job)
if command -v mail >/dev/null 2>&1; then
  { echo "KIRBy wave snapshot — $STAMP"; echo;
    echo "Queue (state x count):"; echo "$QUEUE"; echo;
    echo "Precompute:"; echo "$TOTALS"; echo;
    echo "(full detail: $LOG on the server)"; } \
    | mail -s "KIRBy $STAMP — $(echo "$TOTALS" | tr '\n' ' ')" "$EMAIL" 2>/dev/null || true
fi

rm -f "$TMP_STATUS"
