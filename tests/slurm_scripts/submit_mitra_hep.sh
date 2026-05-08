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
export ANNITIA_VAL_TEST=/data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv

# Install autogluon[mitra] if missing (slurm-7631111 silently failed at this).
python -c "import autogluon" 2>/dev/null || {
    echo "[install] autogluon.tabular[mitra] missing, installing..."
    pip install --quiet "autogluon.tabular[mitra]" 2>&1 | tail -5
    python -c "import autogluon" 2>/dev/null && echo "[install] autogluon OK" || echo "[install] FAILED — MITRA disabled"
}

# autogluon torch<2.3 compat shim (idempotent — skip if already patched, no-op if autogluon missing).
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

echo "[run] starting experiment_branches.py with MITRA-only flags at $(date)"
python -u tests/experiments/annitia/experiment_branches.py \
    --skip-tsfm --skip-tabpfn --skip-dsm --skip-nhanes \
    --skip-tabpfn25-disc --skip-tabsurv --skip-synthetic-aug \
    --skip-survtrace --skip-nhanes-transfer --skip-tabtune --skip-xfer \
    --K-disc 10

echo "DONE at $(date)"
