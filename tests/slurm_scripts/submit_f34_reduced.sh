#!/bin/bash
#SBATCH --job-name=f34red
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --mail-type=FAIL
#
# f34 reduced kNN sweep: every task in f34_tasks.txt, 32 at a time.
#
# A batch job rather than an interactive session because an interactive
# session dies when the ssh connection to the login node drops.
#
# Resumable. Each task writes one line per finished distance and a re-run
# skips those, so submitting this again finishes whatever the wall cut off.
#
#   sbatch --clusters=htc tests/slurm_scripts/submit_f34_reduced.sh

cd /data/stat-ecr/scat9264/KIRBy || exit 2
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh

F=tests/experiments/framework/f34_reduced_sweep.py
python $F --list-tasks f34_tasks.txt || exit 1
# 230 minutes, so the analysis below still runs inside the 4-hour wall.
timeout 230m xargs -P 32 -L 1 python $F --task < f34_tasks.txt
python tests/experiments/framework/f34_reduced_analyse.py
