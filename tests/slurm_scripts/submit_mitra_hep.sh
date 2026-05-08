#!/bin/bash
# MITRA hep+death disc-survival — Oxford ARC htc partition with V100 GPU.
# K=10 hep / K=20 death. ~100h on Mac CPU; <30 min on V100.
#SBATCH --job-name=mitra_hep
#SBATCH --partition=htc
#SBATCH --clusters=htc
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=mitra_hep_%j.out
#SBATCH --error=mitra_hep_%j.err

set -euo pipefail
cd /data/stat-ecr/scat9264/KIRBy   # update if your storage path differs
source setup.sh

# torch<2.3 compat shim for autogluon MITRA (idempotent — skips if already done).
python -c "
import autogluon.tabular.models.mitra._internal.core.get_optimizer as g
import pathlib, torch
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

export ANNITIA_VAL_TEST=/data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# MITRA only — skip everything else; resumes if oof files already exist.
python -u tests/experiments/annitia/experiment_branches.py \
    --skip-tsfm --skip-tabpfn --skip-dsm --skip-nhanes \
    --skip-tabpfn25-disc --skip-tabsurv --skip-synthetic-aug \
    --skip-survtrace --skip-nhanes-transfer --skip-tabtune --skip-xfer \
    --K-disc 10

echo "DONE at $(date)"
