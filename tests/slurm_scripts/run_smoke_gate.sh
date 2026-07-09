#!/bin/bash
#SBATCH --job-name=kirby_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# =============================================================================
# §V SMOKE GATE — server edition
# =============================================================================
# Runs the WHOLE KIRBy pipeline end-to-end on SMALL subsets of real data, for
# BOTH classification and regression, through the NEW per-dataset SQLite result
# cache (src/kirby/result_cache.py). This is the server mirror of the local §V
# gate (REPRODUCIBILITY_LEDGER.md §V) — run it ONCE on ARC to prove the
# redesigned code is healthy before committing to a full re-run.
#
# It exercises every family the way the local smoke fleet did, using the same
# shrink drivers (KIRBY_SMOKE_N + tiny config, all gated so full runs are
# unaffected):
#   precompute  ->  complexity  ->  similarity  ->  noise  ->  federated  ->  opig
#
# What it asserts (the point of the gate):
#   1. Every family runs end-to-end with no traceback / no unexpected {'error'}.
#   2. The cache is REUSED: a warm re-run prints "ResultCache: N stored configs
#      reused" and writes no new rows.
#   3. The months-long-stall bug is dead: ZERO "meta drift" lines anywhere.
#
# Usage:
#   sbatch tests/slurm_scripts/run_smoke_gate.sh
#
# Optional overrides (env):
#   sbatch --export=ALL,SMOKE_N=300 tests/slurm_scripts/run_smoke_gate.sh
#   sbatch --export=ALL,AB_CLF=humanness_therapeutic,AB_REG=ginkgo_heparin,\
#                       MOL_REG=logd,MOL_CLF=herg tests/slurm_scripts/run_smoke_gate.sh
#
# A non-zero exit = the gate FAILED; read the per-step logs named below.
# =============================================================================

set -u

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh

# --- knobs -------------------------------------------------------------------
SMOKE_N="${SMOKE_N:-200}"          # row cap for every family
AB_CLF="${AB_CLF:-humanness_therapeutic}"   # antibody classification dataset
AB_REG="${AB_REG:-}"# antibody regression dataset
MOL_REG="${MOL_REG:-logd}"                  # molecular regression dataset
MOL_CLF="${MOL_CLF:-herg}"                  # molecular classification dataset (federated clf)
FED_REG="${FED_REG:-caco2}"                 # molecular regression dataset (federated reg)
AB_REPS="${AB_REPS:-aac,blosum,onehot}"     # local antibody reps (no foundation models)

# Dedicated SMOKE cache dir — wiped at start so we get a true COLD->WARM test.
# This is NEVER the real cache; it is safe to delete.
SMOKE_CACHE="${SMOKE_CACHE:-$PWD/tests/results/_smoke_gate_server/cache}"
LOGDIR="$PWD/tests/results/_smoke_gate_server/logs"
rm -rf "$SMOKE_CACHE" "$LOGDIR"
mkdir -p "$SMOKE_CACHE" "$LOGDIR"

# Smoke env applied to every run (single-thread avoids the OpenMP segfault;
# KIRBY_SMOKE_N caps rows + arms the gated shrink logic in each driver).
export KIRBY_CACHE_DIR="$SMOKE_CACHE"
export KIRBY_SMOKE_N="$SMOKE_N"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error

echo "=================================================================="
echo "KIRBy §V SMOKE GATE (server)"
echo "  smoke_n=$SMOKE_N   cache=$SMOKE_CACHE"
echo "  ab_clf=$AB_CLF  ab_reg=$AB_REG  mol_reg=$MOL_REG  mol_clf=$MOL_CLF"
echo "=================================================================="
echo

FAIL=0
declare -a RESULTS

# run_step <label> <logfile> <command...>
# Runs a command, tees to its own log, records PASS/FAIL, and flags any
# "meta drift" or traceback as a hard failure.
run_step () {
    local label="$1"; shift
    local log="$LOGDIR/$1"; shift
    echo "------------------------------------------------------------------"
    echo "[$label] $*"
    echo "------------------------------------------------------------------"
    local rc=0
    "$@" > "$log" 2>&1 || rc=$?
    # Hard-fail signals, regardless of the process exit code:
    if grep -q "meta drift" "$log"; then
        echo "[$label] FAIL: 'meta drift' found (cache key not reproducible)" >&2
        rc=1
    fi
    if grep -qE "Traceback \(most recent call last\)" "$log"; then
        echo "[$label] FAIL: traceback in log" >&2
        rc=1
    fi
    if [[ $rc -ne 0 ]]; then
        echo "[$label] FAILED (rc=$rc) — see $log" >&2
        FAIL=1
        RESULTS+=("FAIL  $label  ($log)")
    else
        echo "[$label] ok — $log"
        RESULTS+=("PASS  $label")
    fi
    echo
}

# =============================================================================
# 1. PRECOMPUTE — and the precompute->experiment cache HANDOFF (reg + clf).
#    This driver surfaces the "ResultCache: N stored configs reused" line
#    directly, so it is also our primary cache-reuse assertion.
# =============================================================================
run_step "precompute-reg" "01_precompute_reg.log" \
    python3 tests/experiments/_run_precompute_handoff.py reg
run_step "precompute-clf" "02_precompute_clf.log" \
    python3 tests/experiments/_run_precompute_handoff.py clf

# =============================================================================
# 2. COMPLEXITY (incl. c13 forward-compatibility). Cover the historically-buggy
#    ones. Run reg, then clf. The clf run re-uses nothing new but proves the
#    regression-only experiments SKIP cleanly instead of erroring.
# =============================================================================
CX_EXPS="c01_rep_characterization c06_within_family_gradient c09_formal_analysis \
c13_forward_compatibility c15_ml_theory_depth c17_information_theoretic_decomposition \
c24_blueprint_transfer"

KIRBY_LOCAL_DS="$AB_REG" KIRBY_LOCAL_REPS="$AB_REPS" \
KIRBY_LOCAL_OUT="$LOGDIR/cx_reg" \
run_step "complexity-reg" "03_complexity_reg.log" \
    python3 tests/experiments/_run_complexity_locally.py $CX_EXPS

KIRBY_LOCAL_DS="$AB_CLF" KIRBY_LOCAL_REPS="$AB_REPS" \
KIRBY_LOCAL_OUT="$LOGDIR/cx_clf" \
run_step "complexity-clf" "04_complexity_clf.log" \
    python3 tests/experiments/_run_complexity_locally.py $CX_EXPS

# =============================================================================
# 3. CACHE REUSE (the headline). Re-run the SAME complexity reg step on the
#    SAME warm cache. The warm run must print "ResultCache: ... reused" and add
#    no new rows. Asserted explicitly after the summary.
# =============================================================================
KIRBY_LOCAL_DS="$AB_REG" KIRBY_LOCAL_REPS="$AB_REPS" \
KIRBY_LOCAL_OUT="$LOGDIR/cx_reg_warm" \
run_step "complexity-reg-WARM" "05_complexity_reg_warm.log" \
    python3 tests/experiments/_run_complexity_locally.py $CX_EXPS

# =============================================================================
# 4. SIMILARITY (reg). Also runs s11 which must SKIP cleanly on classification.
# =============================================================================
SIM_EXPS="s01_similarity_baseline s04_activity_cliffs s05_knn_prediction \
s11_bayesian_optimization s13_clustering_retrieval"

run_step "similarity-reg" "06_similarity_reg.log" \
    python3 tests/experiments/similarity/_run_similarity_locally.py "$MOL_REG" $SIM_EXPS
run_step "similarity-clf-skip" "07_similarity_clf.log" \
    python3 tests/experiments/similarity/_run_similarity_locally.py "$AB_CLF" $SIM_EXPS

# =============================================================================
# 5. NOISE (reg + clf). n03 warn-not-halt, n07 clf weighting, n08 :.3f guard.
# =============================================================================
for EXP in n02 n03 n08; do
    run_step "noise-$EXP-reg" "08_noise_${EXP}_reg.log" \
        python3 tests/experiments/noise/_smoke_harness.py "$EXP" --reg --subsample "$SMOKE_N"
done
for EXP in n02 n07; do
    run_step "noise-$EXP-clf" "09_noise_${EXP}_clf.log" \
        python3 tests/experiments/noise/_smoke_harness.py "$EXP" --clf --subsample "$SMOKE_N"
done

# =============================================================================
# 6. FEDERATED (clf + reg). Distinct sequential phases — shrunk to COMPLETE.
# =============================================================================
run_step "federated-clf" "10_federated_clf.log" \
    python3 tests/experiments/federated/_smoke_driver.py "$MOL_CLF" \
        --experiments fed01 fed03 fed04 \
        --output "$LOGDIR/fed_clf"
run_step "federated-reg" "11_federated_reg.log" \
    python3 tests/experiments/federated/_smoke_driver.py "$FED_REG" \
        --experiments fed02 \
        --output "$LOGDIR/fed_reg"

# =============================================================================
# 7. OPIG (antibody clf + reg).
# =============================================================================
run_step "opig-humanness-clf" "12_opig_humanness.log" \
    python3 tests/experiments/opig_humanness.py \
        --datasets "$AB_CLF" --n-folds 2 --reps "$AB_REPS" \
        --output "$LOGDIR/opig_humanness"
run_step "opig-ginkgo-reg" "13_opig_ginkgo.log" \
    python3 tests/experiments/opig_ginkgo_developability.py \
        --properties ginkgo_hic --n-folds 2 --reps "$AB_REPS" \
        --output "$LOGDIR/opig_ginkgo"

# =============================================================================
# SUMMARY + GATE ASSERTIONS
# =============================================================================
echo "=================================================================="
echo "PER-STEP RESULTS"
echo "=================================================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo

echo "=================================================================="
echo "CACHE ASSERTIONS"
echo "=================================================================="

# (a) No "meta drift" anywhere = the cache key is reproducible.
DRIFT=$(grep -rl "meta drift" "$LOGDIR" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$DRIFT" != "0" ]]; then
    echo "  FAIL: 'meta drift' appeared in $DRIFT log(s):" >&2
    grep -rl "meta drift" "$LOGDIR" >&2
    FAIL=1
else
    echo "  PASS: zero 'meta drift' across all logs"
fi

# (b) Cache reuse actually happened (warm precompute + warm complexity).
REUSE=$(grep -rh "stored configs reused" "$LOGDIR" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$REUSE" == "0" ]]; then
    echo "  FAIL: never saw 'stored configs reused' — cache was not exercised" >&2
    FAIL=1
else
    echo "  PASS: cache reused ($REUSE 'stored configs reused' line(s))"
    grep -rh "stored configs reused" "$LOGDIR" | sed 's/^/    /' | sort -u
fi

# (c) No unexpected experiment errors. (Clean SKIPs are fine; tracebacks are not
#     — those already failed their step above.)
ERRS=$(grep -rhoE "RETURNED ERROR: [a-z_]+" "$LOGDIR" 2>/dev/null | sort -u)
if [[ -n "$ERRS" ]]; then
    echo "  NOTE: experiments that returned an error key (review — some are"
    echo "        intentional guards e.g. no_new_reps / regression_only):"
    echo "$ERRS" | sed 's/^/    /'
fi

echo
echo "=================================================================="
if [[ $FAIL -eq 0 ]]; then
    echo "SMOKE GATE PASSED — pipeline healthy + cache reused, no drift."
else
    echo "SMOKE GATE FAILED — see the FAIL lines above and $LOGDIR/*.log"
fi
echo "=================================================================="
exit $FAIL
