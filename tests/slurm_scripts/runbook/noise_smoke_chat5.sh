#!/bin/bash
# Time-boxed check of every noise experiment on herg_ki (herg_ic50 for the six
# that need repeat measurements), inside an srun. One terminal per part:
#   bash tests/slurm_scripts/runbook/noise_smoke_chat5.sh 1
#   bash tests/slurm_scripts/runbook/noise_smoke_chat5.sh 2
#   bash tests/slurm_scripts/runbook/noise_smoke_chat5.sh 3
# Each experiment stops after BOX (default 20m). Prints every experiment's last
# log line once a minute, then the error lines. Deletes nothing; safe to re-run.
set -u
PART=${1:?give the part: 1, 2 or 3}
BOX=${BOX:-20m}
cd /data/stat-ecr/scat9264/KIRBy || exit 1
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh
export KIRBY_NO_COLD_SEARCH=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KIRBY_LGBM_THREADS=1 KIRBY_RF_JOBS=1
LOGS=logs/smoke_chat5c
OUT=tests/results/smoke_chat5c/noise
mkdir -p "$LOGS" "$OUT"

case "$PART" in
  1) KI="c10 n01 n02 n07 n09 n11 n12 n13 n14 n15 n16 n18"; IC="" ;;
  2) KI="n20 n21 n23 n24 n25 n27 n30 n31 n32 n33 n34 n35 n37"; IC="" ;;
  3) KI="n38 n39 n40 n41 n42 n06"; IC="n17 n19 n22 n26 n29 n36" ;;
  *) echo "part must be 1, 2 or 3"; exit 1 ;;
esac

echo "commit: $(git log --oneline -1)"
for e in $KI; do
  timeout "$BOX" python -u tests/experiments/run_dataset.py noise herg_ki \
    --experiments "$e" --output "$OUT" > "$LOGS/$e.log" 2>&1 &
done
for e in $IC; do
  timeout "$BOX" python -u -c "import sys; sys.path[:0]=['src','tests','tests/experiments']; from experiments.noise import noise_helpers as nh; nh.run_all('herg_ic50', '$OUT', experiments=['$e'])" \
    > "$LOGS/$e.log" 2>&1 &
done

while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  echo "=== $(date +%H:%M) running: $(jobs -r | wc -l)"
  for e in $KI $IC; do
    printf "%-4s " "$e"
    grep -v "budget is a MAXIMUM\|further occurrences" "$LOGS/$e.log" 2>/dev/null | tail -n 1 | cut -c1-110
  done
  sleep 60
done

echo "=== ALL STOPPED. Error lines:"
for e in $KI $IC; do
  echo "-- $e"
  grep -m3 -E "Traceback|Error" "$LOGS/$e.log" | cut -c1-200
done
echo "=== end"
