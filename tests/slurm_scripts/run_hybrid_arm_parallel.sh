#!/bin/bash
# The kirby_hybrid arm, eight datasets AT ONCE, inside one interactive srun.
#
#   bash tests/slurm_scripts/run_hybrid_arm_parallel.sh
#
# WHY PARALLEL. Scoring the arm honestly refits the feature selection inside
# every fold, so it runs a whole find_best_hybrid per fold per model -- ten
# searches per dataset where there used to be one. Run one after another that
# does not finish in a sitting. The eight datasets do not depend on each other,
# so they run side by side and the wall clock is the SLOWEST dataset, not the
# sum. Nothing about the search is reduced.
#
# Each dataset gets its own log and writes its own result file, so a dataset
# that finishes is banked even if the session ends. Re-running skips what is
# already written.
#
# WALKING AWAY. nohup only stops a hangup from the TERMINAL. It does NOT keep
# anything alive once the srun allocation ends -- SLURM kills the whole step,
# nohup or not. Start tmux (or screen) on the LOGIN node and run the srun
# INSIDE it:
#
#     tmux new -s hybrid
#     srun -p interactive --cpus-per-task=8 --mem=16G --time=10:00:00 --pty /bin/bash
#     cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh
#     bash tests/slurm_scripts/run_hybrid_arm_parallel.sh
#     # ctrl-b then d to detach; `tmux attach -t hybrid` to come back
#
# Detaching tmux leaves the srun running, so the allocation and everything in
# it stays up. Closing the terminal without tmux ends the allocation and the
# run dies with it.
#
# Whatever finished is still banked: each dataset writes its result file as
# soon as it is done, and --skip-existing means a restart continues rather
# than repeating.

set -u
cd /data/stat-ecr/scat9264/KIRBy || exit 1

OUTPUT="${OUTPUT:-tests/results/geometry_survey}"
LOGS="${LOGS:-logs/hybrid_arm}"
BASE_REPS="${BASE_REPS:-pruned}"
MODELS="${MODELS:-rf,dnn}"
DATASETS="${DATASETS:-abcov_kd ddg_combined flab_thermo_tresanco ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer}"

# One thread per process. Eight datasets on eight cores each trying to use all
# of them is slower than eight single-threaded ones.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"

git log -1 --format='[hybrid-arm] running commit %h %ad %s' --date=short
mkdir -p "$OUTPUT" "$LOGS"

pids=""
n_started=0
for ds in $DATASETS; do
    log="$LOGS/${ds}.log"
    nohup python -u tests/experiments/structural_geometry_survey.py "$ds" \
        --base-reps "$BASE_REPS" --models "$MODELS" \
        --hybrid-only --skip-existing \
        --output "$OUTPUT/" > "$log" 2>&1 &
    pids="$pids $!"
    n_started=$((n_started + 1))
    echo "  started $ds  (pid $!)  -> $log"
done

echo ""
echo "All ${n_started} started. Watch one with:   tail -f $LOGS/<dataset>.log"
echo "Progress across all:                    grep -c 'fold ' $LOGS/*.log"
echo ""

# Heartbeat. `wait` prints nothing until everything finishes, so a run that is
# working and a run that is hung look identical for hours. Report every few
# minutes which datasets have finished how many folds, and how many processes
# are still alive.
HEARTBEAT="${HEARTBEAT:-180}"
set +m          # no job-control notices when the heartbeat is killed below
(
    while :; do
        sleep "$HEARTBEAT"
        alive=0
        for p in $pids; do
            kill -0 "$p" 2>/dev/null && alive=$((alive + 1))
        done
        [ "$alive" -eq 0 ] && break
        line=""
        for ds in $DATASETS; do
            n=$(grep -c 'fold ' "$LOGS/${ds}.log" 2>/dev/null || echo 0)
            done_mark=""
            [ -f "$OUTPUT/hybridarm_${ds}.json" ] && done_mark="*"
            # Not ${ds%%_*} -- that prints all five ginkgo datasets as
            # "ginkgo". Last 13 characters keeps them apart.
            short=$(echo "$ds" | tail -c 14)
            line="$line ${short}:${n}${done_mark}"
        done
        echo "  [$(date +%H:%M)] $alive running |$line   (10 folds each; * = file written)"
    done
) &
heartbeat_pid=$!

fail=0
for p in $pids; do
    wait "$p" || fail=$((fail + 1))
done
kill "$heartbeat_pid" 2>/dev/null
wait "$heartbeat_pid" 2>/dev/null

echo ""
echo "==== done, $fail dataset(s) exited non-zero ===="
ls -lt "$OUTPUT"/hybridarm_*.json 2>/dev/null | head -20
echo ""
python tests/analysis/analyse_geometry_survey.py "$OUTPUT"
