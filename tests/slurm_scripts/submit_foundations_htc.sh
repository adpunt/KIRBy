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
export ANNITIA_VAL_TEST=/data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv

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

# 1. autogluon torch<2.3 GradScaler shim (only if autogluon imported).
python -c "
try:
    import autogluon.tabular.models.mitra._internal.core.get_optimizer as g
except ImportError:
    print('autogluon not installed — skipping shim'); raise SystemExit(0)
import pathlib
src = pathlib.Path(g.__file__)
text = src.read_text()
if 'torch<2.3 compat shim' in text:
    print('autogluon MITRA already patched')
else:
    new_block = '''if not hasattr(torch.amp, \"GradScaler\"):
    _OldGradScaler = torch.cuda.amp.GradScaler
    class _CompatGradScaler(_OldGradScaler):
        def __init__(self, *args, device=None, **kwargs):
            super().__init__(*args, **kwargs)
    torch.amp.GradScaler = _CompatGradScaler  # torch<2.3 compat shim
import inspect as _inspect
_GS_PARAMS = set(_inspect.signature(torch.amp.GradScaler.__init__).parameters)
class GradScaler(torch.amp.GradScaler):'''
    src.write_text(text.replace('class GradScaler(torch.amp.GradScaler):', new_block))
    print('autogluon MITRA PATCHED')
"

# 2. Run all foundation-model branches (no skips). Resume-friendly per-condition.
echo "[run] starting experiment_branches.py at $(date)"
python -u tests/experiments/annitia/experiment_branches.py \
    --K-disc 10

echo "DONE foundations at $(date)"
