#!/bin/bash
# All foundation-model branches on hep + death (T1.1 / T1.2 / T1.3 / T1.4 / T1.5).
# Resumes — skips anything with existing oof_/val_ files.
#
# Coverage:
#   T1.1  MITRA disc-K10 hep + K20 death  (was failing on torch.amp.GradScaler — patched)
#   T1.1  TabPFN-2.5 disc K10 hep + K20 death
#   T1.3  TabSurv FT-Transformer K10 hep + K15 death
#   T1.4  Synthetic-positive aug (TabDDPM / TabSyn / TabEBM via synthcity, fallback to GMM)
#   T1.5  SurvTRACE competing-risks (joint hep+death) — skipped if pkg not installed
#   Plus: TSFM, DSM, NHANES-MLP, transfer-endpoint, tabpfn-landmark — anything cached locally is skipped, anything new runs.
#
# Usage:
#   bash tests/slurm_scripts/submit_foundations_htc.sh
#
#SBATCH --job-name=annitia_foundations
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=FAIL

set -uo pipefail   # NOT -e: keep going if individual installs fail
cd /data/stat-ecr/scat9264/KIRBy
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# If the compute node lacks libcudnn.so.9 (which autogluon[mitra] requires for
# its torch GPU init), fall back to CPU. Without this, MITRA crashes with
# "libcudnn.so.9: cannot open shared object file" — observed on 7631558.
if ! ldconfig -p 2>/dev/null | grep -q libcudnn.so.9 && \
   ! find /usr/lib /usr/local/lib /usr/local/cuda -name 'libcudnn.so.9' 2>/dev/null | grep -q .; then
    echo "[cuda] libcudnn.so.9 not found; forcing CPU mode for MITRA"
    export CUDA_VISIBLE_DEVICES=""
fi
# Pick whichever val_test.csv path is actually visible on this compute node.
# stat-ecr is the canonical post-2026-05-07 location but may not be mounted
# on all htc compute nodes; stat-cadd is the legacy location and is broadly
# mounted. Falls through to /tmp if a job-local copy was placed there.
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

# 0. Ensure foundation packages exist; install if missing. slurm-7631175
# silently failed because autogluon was absent in the server venv.
echo "[install] checking foundation packages..."
python -c "import autogluon" 2>/dev/null || {
    echo "[install] autogluon missing, installing..."
    pip install --quiet "autogluon.tabular[mitra]" 2>&1 | tail -5
    python -c "import autogluon" 2>/dev/null && echo "[install] autogluon OK" || echo "[install] autogluon FAIL — MITRA branch will be skipped"
}
python -c "import tabpfn" 2>/dev/null || {
    pip install --quiet tabpfn 2>&1 | tail -3
    python -c "import tabpfn" 2>/dev/null && echo "[install] tabpfn OK" || echo "[install] tabpfn FAIL — TabPFN branches skipped"
}
python -c "import synthcity" 2>/dev/null || {
    pip install --quiet synthcity 2>&1 | tail -3
    python -c "import synthcity" 2>/dev/null && echo "[install] synthcity OK" || echo "[install] synthcity FAIL — synaug TabDDPM/TabSyn fall back to GMM"
}
python -c "import tab_ebm" 2>/dev/null || {
    pip install --quiet tab-ebm 2>&1 | tail -3
    python -c "import tab_ebm" 2>/dev/null && echo "[install] tab-ebm OK" || echo "[install] tab-ebm FAIL"
}
python -c "import survtrace" 2>/dev/null || {
    pip install --quiet survtrace 2>&1 | tail -3 || \
    pip install --quiet "git+https://github.com/RyanWangZf/SurvTRACE.git" 2>&1 | tail -3
    python -c "import survtrace" 2>/dev/null && echo "[install] survtrace OK" || echo "[install] survtrace FAIL — T1.5 skipped"
}

# 1. autogluon torch<2.3 GradScaler shim. CANNOT import autogluon to find the
# file path — that triggers the very class definition we're patching, causing
# AttributeError at line 65. Use `find` to locate get_optimizer.py instead.
GETFILE=$(find $(python -c "import site; print(site.getsitepackages()[0])") \
    -path '*autogluon/tabular/models/mitra/_internal/core/get_optimizer.py' \
    2>/dev/null | head -1)
if [ -z "$GETFILE" ]; then
    GETFILE=$(find $HOME/.local/lib -path '*mitra/_internal/core/get_optimizer.py' 2>/dev/null | head -1)
fi
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

# 2. Run all foundation-model branches (no skips). Resume-friendly per-condition.
# Pass val-test path explicitly so it survives even if experiment_branches.py
# is at an older revision that didn't read ANNITIA_VAL_TEST.
VAL_CSV="${ANNITIA_VAL_TEST}"
echo "[run] using val_test = $VAL_CSV"
echo "[run] starting experiment_branches.py at $(date)"
python -u tests/experiments/annitia/experiment_branches.py \
    --validation-csv "$VAL_CSV" \
    --K-disc 10

echo "DONE foundations at $(date)"
