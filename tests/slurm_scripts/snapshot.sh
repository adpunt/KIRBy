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
# ONE-SHOT progress snapshot: appends a timestamped block (queue counts +
# check_precompute_status --timing + audit_full_wave) to monitor_<date>.log,
# then exits. Pre-schedule many of these at staggered --begin times with
# schedule_snapshots.sh.

REPO=/data/stat-ecr/scat9264/KIRBy
cd "$REPO"
. ./setup.sh          # check_precompute_status imports the kirby package

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
} >> "$REPO/monitor_$(date +%Y-%m-%d).log" 2>&1
