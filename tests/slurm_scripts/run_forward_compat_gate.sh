#!/bin/bash
#SBATCH --job-name=fc_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --partition=medium
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=tests/results/_fc_gate/gate_%j.log

# =============================================================================
# WHICH FORWARD-COMPATIBILITY CANDIDATES BUILD ON THE SERVER?
# =============================================================================
# Run this BEFORE any c13 grid. It calls each of the 56 candidate builders on a
# handful of real molecules / antibody sequences / DNA windows / reactions and
# prints one line per candidate: built, needs a package, needs a checkpoint, is
# a stub, or is a bug. It fits no model and touches no dataset, so it costs
# minutes plus whatever weights download the first time.
#
# The point is that finding a missing checkpoint inside a 30-hour grid job
# wastes the job.
#
# WHICH CLUSTER: htc. This is CPU-only with no GPU need for most candidates,
# which is an htc job by ARC's own division of labour (see
# run_f24_augmentation_cost.sh:27). The cluster is set by the LOGIN NODE, not by
# --partition (run_precompute_reps.sh:16), so submit from htc-login or pass
# --clusters=htc. Two candidates DO want a GPU -- ouroboros calls .cuda()
# unconditionally and raises on a CPU node, and chemfm is a 1B decoder -- and a
# --gres=gpu:1 request is rejected outright from arc-login.
#
# --partition is LIVE STATE. The value above is a placeholder: run
# where_to_submit.sh and override it.
#
#   bash tests/slurm_scripts/where_to_submit.sh
#   sbatch --clusters=htc --partition=<what it said> \
#          tests/slurm_scripts/run_forward_compat_gate.sh
#   sbatch --clusters=htc --partition=<...> --export=ALL,FC_DOMAIN=molecular \
#          tests/slurm_scripts/run_forward_compat_gate.sh
#
# Reads: nothing. Writes: tests/results/_fc_gate/.
#
# ON THREADS. setup.sh sets OMP_NUM_THREADS to the full CPU allocation, which on
# macOS makes a torch build followed by a LightGBM fit die at exit 139 with no
# traceback. The repo's position is that this is macOS-only — see the comment at
# tests/experiments/framework/f30_scorers.py:1018, "NOT measured on Linux and
# not expected there". This script therefore leaves setup.sh's threading alone.
# If a task here dies with exit 139 and no traceback, that position is wrong:
# re-run with OMP_NUM_THREADS=1 and say so, because it then also applies to any
# c13 run with a torch-backed candidate.
# =============================================================================
set -u

cd /data/stat-ecr/scat9264/KIRBy || { echo "repo not found"; exit 1; }
git pull 2>/dev/null
. ./setup.sh

OUT="$PWD/tests/results/_fc_gate"
mkdir -p "$OUT"

DOMAIN_ARG=""
[ -n "${FC_DOMAIN:-}" ] && DOMAIN_ARG="--domain $FC_DOMAIN"

# FC_TRACEBACK=1 makes an unclassified failure print its stack, so a genuine bug
# is debuggable from the log instead of just being labelled ERROR.
export FC_TRACEBACK=1

echo "=================================================================="
echo "FORWARD-COMPATIBILITY BUILDER GATE"
echo "  node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  OMP=${OMP_NUM_THREADS:-unset}"
echo "  domain=${FC_DOMAIN:-all}"
echo "=================================================================="

python3 -u tests/check_forward_compat_builders.py \
    $DOMAIN_ARG --json "$OUT/gate_${SLURM_JOB_ID:-local}.json"
RC=$?

echo
echo "=================================================================="
if [ $RC -eq 0 ]; then
    echo "NO BUGS. Every candidate either built or is waiting on an install or a"
    echo "checkpoint — read the table above for which."
else
    echo "SOME CANDIDATES ARE BROKEN IN CODE (BAD_SHAPE / ERROR / NOT_REGISTERED)."
    echo "Those are bugs, not missing installs. Fix them before running c13:"
    echo "a BAD_SHAPE builder returns rows that do not line up with the labels."
fi
echo
echo "Rows written to $OUT/gate_${SLURM_JOB_ID:-local}.json"
echo "Then run the experiment on whatever came back OK, e.g.:"
echo "  python3 tests/experiments/complexity/c13_forward_compatibility.py \\"
echo "      --dataset logd --reps all --mode full --new-reps <the OK names>"
echo "=================================================================="
exit $RC
