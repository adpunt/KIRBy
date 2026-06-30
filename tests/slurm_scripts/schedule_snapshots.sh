#!/bin/bash
# Pre-schedule N one-shot progress snapshots using SLURM --begin.
# Every job is submitted NOW and sits PENDING with a future BeginTime, firing
# at its scheduled minute. No self-resubmission, no node held in between.
#
#   bash tests/slurm_scripts/schedule_snapshots.sh [INTERVAL_MIN] [COUNT]
#
# Defaults: every 4 hours (240 min), 84 snapshots (= 14 days of coverage).
# Override, e.g. every 2 h for 1 week:  schedule_snapshots.sh 120 84
#
#   See scheduled times:  squeue -u $USER --name=kirby_snapshot --start
#   Cancel all:           scancel --name=kirby_snapshot
#   Output log:           monitor_<date>.log  (each job appends one block)

set -u
INTERVAL_MIN=${1:-240}
COUNT=${2:-84}
SCRIPT=/data/stat-ecr/scat9264/KIRBy/tests/slurm_scripts/snapshot.sh

echo "Pre-scheduling $COUNT snapshots, one every $INTERVAL_MIN min:"
for i in $(seq 1 "$COUNT"); do
    OFFSET=$((i * INTERVAL_MIN))
    OUT=$(sbatch --begin=now+${OFFSET}minutes "$SCRIPT")
    echo "  +${OFFSET}min  ->  $OUT"
done
echo
echo "Scheduled times: squeue -u \$USER --name=kirby_snapshot --start"
echo "Cancel all:      scancel --name=kirby_snapshot"
