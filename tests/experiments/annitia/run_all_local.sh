#!/usr/bin/env bash
# ANNITIA local end-to-end runner.
#
# Runs all four phases sequentially with resume support. Total ~3-5h on a Mac
# with the optional deps already installed (catboost via conda, pycox/tabpfn/
# tabicl via pip, torch upgraded to 2.2.2).
#
# Each Python runner is RESUME-friendly — re-running this script is safe:
# completed (config_id, OOF + val .npy) tuples are skipped.
#
# Stages:
#   BE        — 6 view-sets × 8 algos × 2 endpoints × 5-fold     (~25 min)
#   CD        — Phase C framings + Phase D tricks + finegray + D.9 (~30 min)
#   HB        — RSF-1000, CatBoost, weighted/bagged/snapshot tricks (hep only) (~30 min)
#   BRANCHES  — DSM, NHANES-pretrained-MLP, tsfm view, TabPFN landmark (~60-90 min)
#   FUSION    — final cross-pool fusion across all four pools          (~5 min)
#
# Usage:
#   bash tests/experiments/annitia/run_all_local.sh
#
# Env overrides:
#   ANNITIA_VAL_TEST=/path/to/val_test.csv
#   ANNITIA_SKIP_TABPFN=1   # skip the slowest stage if you're tight on time
#   ANNITIA_TSFM_BACKEND=fft|chronos|timesfm|auto (default: fft for speed)

set -e

VAL_TEST=${ANNITIA_VAL_TEST:-/Volumes/seagate/kirby/annitia/val_test.csv}
TSFM_BACKEND=${ANNITIA_TSFM_BACKEND:-fft}
LOG_DIR=/tmp/annitia_runs
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")/../../.."   # repo root

echo "============================================================"
echo "  ANNITIA local end-to-end runner"
echo "  val_test: $VAL_TEST"
echo "  tsfm backend: $TSFM_BACKEND"
echo "  log dir: $LOG_DIR"
echo "============================================================"

run_stage() {
    local name=$1
    shift
    local log="$LOG_DIR/${name}.log"
    echo
    echo ">>> [$name] $*"
    echo ">>> log: $log"
    if python -u "$@" 2>&1 | tee "$log"; then
        echo "<<< [$name] OK"
    else
        echo "<<< [$name] FAILED — continuing to next stage"
    fi
}

# 1. BE
run_stage be tests/experiments/annitia/experiment_be.py \
    --validation-csv "$VAL_TEST"

# 2. CD
run_stage cd tests/experiments/annitia/experiment_cd.py \
    --validation-csv "$VAL_TEST"

# 3. HB (hepatic boost)
run_stage hep_boost tests/experiments/annitia/experiment_hep_boost.py \
    --validation-csv "$VAL_TEST"

# 4. BRANCHES (DSM, NHANES-pretrained MLP, TSFM view, TabPFN landmark)
TABPFN_FLAG=""
if [ "${ANNITIA_SKIP_TABPFN:-0}" = "1" ]; then TABPFN_FLAG="--skip-tabpfn"; fi
run_stage branches tests/experiments/annitia/experiment_branches.py \
    --validation-csv "$VAL_TEST" \
    --tsfm-backend "$TSFM_BACKEND" \
    $TABPFN_FLAG

# 5. FUSION (cross-pool: pools BE + CD + HB + BRANCHES)
run_stage fusion tests/experiments/annitia/experiment_full.py \
    --validation-csv "$VAL_TEST" \
    --skip-be --skip-cd

echo
echo "============================================================"
echo "  DONE — final master:"
ls -la tests/results/annitia/full/sub_master.csv 2>/dev/null \
    || echo '  (not produced — check fusion log)'
echo "============================================================"
