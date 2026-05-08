#!/bin/bash
# MITRA hep+death disc-survival on HTC. K=10 hep / K=20 death.
# Estimated <1h on a single GPU; short partition safe.
#
# Usage:
#   bash tests/slurm_scripts/submit_mitra_hep.sh
#
#SBATCH --job-name=mitra_hep
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=FAIL

set -uo pipefail   # NOT -e: we want to keep going if individual installs fail
cd /data/stat-ecr/scat9264/KIRBy
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# Force CPU if libcudnn.so.9 is missing on this compute node.
if ! ldconfig -p 2>/dev/null | grep -q libcudnn.so.9 && \
   ! find /usr/lib /usr/local/lib /usr/local/cuda -name 'libcudnn.so.9' 2>/dev/null | grep -q .; then
    echo "[cuda] libcudnn.so.9 not found; forcing CPU mode for MITRA"
    export CUDA_VISIBLE_DEVICES=""
fi
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

# Install autogluon[mitra] if missing (slurm-7631111 silently failed at this).
python -c "import autogluon" 2>/dev/null || {
    echo "[install] autogluon.tabular[mitra] missing, installing..."
    pip install --quiet "autogluon.tabular[mitra]" 2>&1 | tail -5
    python -c "import autogluon" 2>/dev/null && echo "[install] autogluon OK" || echo "[install] FAILED — MITRA disabled"
}

# autogluon torch<2.3 compat shim. CANNOT import autogluon (triggers the very
# class def we're patching). Use `find` to locate get_optimizer.py instead.
GETFILE=$(find $(python -c "import site; print(site.getsitepackages()[0])") \
    -path '*autogluon/tabular/models/mitra/_internal/core/get_optimizer.py' \
    2>/dev/null | head -1)
if [ -z "$GETFILE" ]; then
    GETFILE=$(find /data/stat-cadd/scat9264/py311-kirby/lib -path '*mitra/_internal/core/get_optimizer.py' 2>/dev/null | head -1)
fi
if [ -n "$GETFILE" ] && [ -f "$GETFILE" ]; then
    if grep -q 'torch<2.3 compat shim' "$GETFILE"; then
        echo "[shim] autogluon MITRA already patched at $GETFILE"
    else
        echo "[shim] patching autogluon MITRA at $GETFILE"
        python -c "
import pathlib
p = pathlib.Path('$GETFILE')
t = p.read_text()
shim = '''if not hasattr(torch.amp, \"GradScaler\"):
    _OldGradScaler = torch.cuda.amp.GradScaler
    class _CompatGradScaler(_OldGradScaler):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, **kwargs)
    torch.amp.GradScaler = _CompatGradScaler  # torch<2.3 compat shim
'''
needle = 'class GradScaler(torch.amp.GradScaler):'
if needle in t and 'torch<2.3 compat shim' not in t:
    t = t.replace(needle, shim + needle)
    p.write_text(t)
    print('[shim] PATCHED')
else:
    print('[shim] needle not found or already patched')
"
    fi
else
    echo "[shim] WARNING: autogluon get_optimizer.py not found — MITRA may crash"
fi

echo "[run] starting experiment_branches.py with MITRA-only flags at $(date)"
echo "[run] using val_test = $ANNITIA_VAL_TEST"
python -u tests/experiments/annitia/experiment_branches.py \
    --validation-csv "$ANNITIA_VAL_TEST" \
    --skip-tsfm --skip-tabpfn --skip-dsm --skip-nhanes \
    --skip-tabpfn25-disc --skip-tabsurv --skip-synthetic-aug \
    --skip-survtrace --skip-nhanes-transfer --skip-tabtune --skip-xfer \
    --K-disc 10

echo "DONE at $(date)"
