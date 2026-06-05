#!/bin/bash
#SBATCH --job-name=kirby_monitor
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#
# Periodic progress snapshot for the KIRBy wave.
#
# Each run: appends a timestamped block (queue summary + check_precompute_status
# --timing + audit_full_wave snapshot) to monitor_<date>.log, then re-submits
# itself to run again in INTERVAL_MIN minutes — but ONLY while kirby_full jobs
# still exist, so it auto-stops when the wave finishes.
#
#   Start:  sbatch tests/slurm_scripts/monitor_progress.sh
#   Read:   tail -f /data/stat-ecr/scat9264/KIRBy/monitor_$(date +%Y-%m-%d).log
#   Stop:   scancel --name=kirby_monitor
#   Change cadence: edit INTERVAL_MIN below.

INTERVAL_MIN=30
REPO=/data/stat-ecr/scat9264/KIRBy
SELF="$REPO/tests/slurm_scripts/monitor_progress.sh"
LOG="$REPO/monitor_$(date +%Y-%m-%d).log"

cd "$REPO"
. ./setup.sh          # needed: check_precompute_status imports the kirby package

{
  echo "================ $(date '+%Y-%m-%d %H:%M') ================"
  echo "--- queue (state counts) ---"
  squeue -u "$USER" -h -o "%T" | sort | uniq -c
  echo
  echo "--- precompute timing (check_precompute_status --timing --distance) ---"
  python tests/check_precompute_status.py --timing --distance
  echo
  echo "--- wave snapshot (audit_full_wave) ---"
  python tests/audit_full_wave.py --include-running --eta
  echo
} >> "$LOG" 2>&1

# Re-arm only while the wave is still queued/running, so it stops on its own.
if squeue -u "$USER" --name=kirby_full -h -o "%i" | grep -q .; then
    sbatch --begin=now+${INTERVAL_MIN}minutes "$SELF"
fi
