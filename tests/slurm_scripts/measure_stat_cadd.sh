#!/bin/bash
#SBATCH --job-name=cadd_size
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Size every top-level entry under the stat-cadd home, one at a time.
# The recursive walk runs on a compute node, never the login node — same
# reason delete_av2_data.sh runs as a job.
# Read-only. Deletes nothing.

ROOT=/data/stat-cadd/scat9264

echo "start: $(date)"
df -h /data/stat-cadd
echo "=========================================="

for entry in "${ROOT}"/* "${ROOT}"/.[!.]*; do
    [ -e "${entry}" ] || continue
    echo "--- $(date +%H:%M:%S) ${entry}"
    du -sh "${entry}" 2>/dev/null
done

echo "=========================================="
echo "done: $(date)"
