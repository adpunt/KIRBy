#!/bin/bash
# NHANES T2.x: external-data pretraining + transfer + 5-subset ablation.
# Per A17 audit, prior NHANES round was buggy (zero-shot eval no-op'd, subset
# env vars not forwarded). Both fixed in commits this session — re-run is mandatory.
#
# Coverage:
#   T2.0  NHANES preprocess + pretrain encoder per subset (5 subsets)
#   T2.1  Zero-shot gate eval per subset
#   T2.2  6 transfer recipes a–f per subset that passes T2.1 gate
#   T2.3  Full 5-subset ablation grid (60 cells = 5 subsets × 6 recipes × 2 endpoints)
#
# Usage:
#   bash tests/slurm_scripts/submit_nhanes_t2_htc.sh
#
#SBATCH --job-name=annitia_nhanes_t2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=medium
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=FAIL

set -uo pipefail   # NOT -e: keep going on individual install failures
cd /data/stat-ecr/scat9264/KIRBy
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
for cand in \
    /data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv \
    /data/stat-cadd/scat9264/KIRBy/data/annitia/val_test.csv \
    $TMPDIR/val_test.csv \
    /tmp/val_test.csv; do
    if [ -f "$cand" ]; then
        export ANNITIA_VAL_TEST="$cand"; echo "[run] val_test found: $cand"; break
    fi
done
if [ -z "${ANNITIA_VAL_TEST:-}" ] || [ ! -f "$ANNITIA_VAL_TEST" ]; then
    echo "[FATAL] val_test.csv not found in any candidate path"; exit 1
fi

# NHANES T2.x doesn't strictly need autogluon, but T2.2d uses TabPFN PEFT
# which needs tabpfn. Install if missing so that recipe doesn't fall through.
python -c "import tabpfn" 2>/dev/null || {
    echo "[install] tabpfn missing, installing..."
    pip install --quiet tabpfn 2>&1 | tail -3
}

# Clean stale dummies if present (per A17 audit recommendation)
if [ -f tests/results/annitia/external/zeroshot.json ]; then
    grep -q "n/a" tests/results/annitia/external/zeroshot.json 2>/dev/null && \
        echo "Clearing stale tests/results/annitia/external/ (broken zero-shot run)" && \
        rm -rf tests/results/annitia/external/encoder_*.pt tests/results/annitia/external/zeroshot.json || true
fi

# All 5 subsets in T2.3 — env-driven via experiment_external.py.
export ANNITIA_T23_SUBSETS="fib4_high,fib4_high_2007,fib4_high_2017,t2dm_only,no_viral"
# Don't gate on T2.1 zero-shot pass — run the full ablation regardless so we
# get the complete grid for the writeup.
export ANNITIA_IGNORE_T21_GATE=1

# Pass val-test path explicitly so it survives even if experiment_external.py
# is at an older revision that didn't read ANNITIA_VAL_TEST.
echo "[run] using val_test = $ANNITIA_VAL_TEST"
python -u tests/experiments/annitia/experiment_external.py \
    --validation-csv "$ANNITIA_VAL_TEST"

echo "DONE NHANES T2.x at $(date)"
