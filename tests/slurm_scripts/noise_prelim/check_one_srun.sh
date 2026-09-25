#!/bin/bash
# SAMPLE. The same check as check_one.sbatch, in an interactive session on htc.
#
#   bash tests/slurm_scripts/noise_prelim/check_one_srun.sh <dataset> <experiment> [condition]
#
# It prints the srun line and then runs it. The job runs check_one.sbatch with
# bash: its #SBATCH lines are comments to bash, so the resources come from the
# srun flags below instead.
#
# PARTITION. 'interactive': "maximum run time 24hrs, can oversubscribe, for
# pre/post-processing and building software"; 'devel' is 10 minutes, batch
# testing only, too short for this.
#   https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html
# where_to_submit.sh section 6 says whether interactive can start a job right
# now; run it first. Its comment says interactive is 4 h; the ARC page says 24 h.
# `sinfo -M htc -p interactive` shows the live limit.
#
# CLUSTER. --clusters=htc is on the line. If srun on the arc login node refuses
# it, log in to htc-login.arc.ox.ac.uk (ARC docs, connecting-to-arc.html) and
# drop the flag: the login node picks the cluster. NEVER pass --account.
set -uo pipefail
DATASET="${1:?usage: check_one_srun.sh <dataset> <experiment> [condition]}"
EXP="${2:?usage: check_one_srun.sh <dataset> <experiment> [condition]}"
COND="${3:-}"
MEM="${MEM:-64G}"          # 8 CPUs x ARC's 8000 MB default per CPU
WALL="${WALL:-12:00:00}"   # under the 24 h interactive maximum

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CMD=(srun --clusters=htc -p interactive --cpus-per-task=8 --mem="$MEM"
     --time="$WALL" --job-name=np_check_i
     bash tests/slurm_scripts/noise_prelim/check_one.sbatch "$DATASET" "$EXP")
[ -n "$COND" ] && CMD+=("$COND")
echo "${CMD[*]}"
exec "${CMD[@]}"
