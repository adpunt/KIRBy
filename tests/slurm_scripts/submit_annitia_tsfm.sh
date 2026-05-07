#!/bin/bash
#SBATCH --job-name=annitia_tsfm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/tsfm-views — time-series-foundation-model embeddings
# of per-lab trajectories (Apache-2.0). Tries TimesFM (google-research),
# falls back to Chronos (amazon-science), then to a deterministic FFT
# signature if neither library is available.
#
# The view is added to VIEW_BUILDERS under the key 'tsfm'. We then run the
# existing search.py over a small set of view subsets that include 'tsfm',
# alongside the strongest hand-crafted views, to test whether trajectory-
# shape signal beyond hand-rolled FIB-4 / APRI slopes is present.

REPO=/data/stat-ecr/scat9264/KIRBy
BRANCH=feat/tsfm-views
LABEL=tsfm
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh

# Try TimesFM first; if install fails Chronos is the next preference.
pip install timesfm || true
pip install git+https://github.com/amazon-science/chronos-forecasting.git || true

mkdir -p $OUT
echo "$(date): launching $LABEL on $BRANCH"

DEVICE=cpu
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE=cuda
fi
echo "device=$DEVICE"

# Pre-compute the tsfm view once (cached); subsequent fold passes hit cache.
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views
data = load_annitia()
_ = build_views(data, view_names=['tsfm'])
print('tsfm view precomputed and cached')
"

# Then run the standard search with tsfm in the view list.
python tests/experiments/annitia/run.py \
    --views snapshot,slope,delta,nit,clinical,clinical_pro,tsfm \
    --models cox,coxnet,rsf,gbsa \
    --prune-levels none,50 \
    --max-subset-size 4 \
    --top-k-keep 5 \
    --n-folds 5 \
    --endpoints hepatic,death \
    --output-dir $OUT

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
