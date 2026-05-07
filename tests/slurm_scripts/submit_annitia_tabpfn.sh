#!/bin/bash
#SBATCH --job-name=annitia_tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/tabpfn-landmark — TabPFN-v2 ICL classifier under the
# landmark-binary survival framing. Per-horizon binary classifiers' predicted
# probabilities are rank-averaged into a single risk score per patient.
#
# License caveat: TabPFN-v2 is under Prior Labs research/non-commercial license.
# OK for the Trustii submission; revisit if the model is reused commercially.

REPO=/data/stat-ecr/scat9264/KIRBy
BRANCH=feat/tabpfn-landmark
LABEL=tabpfn
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh

# TabPFN install. Pin to a recent v2 release. CUDA build comes via torch
# already in setup.sh.
pip install tabpfn || true

mkdir -p $OUT
echo "$(date): launching $LABEL on $BRANCH (worktree $WORK)"

# Detect device: prefer cuda if visible.
DEVICE=cpu
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE=cuda
fi
echo "device=$DEVICE"

python tests/experiments/annitia/run_tabpfn.py \
    --views snapshot,slope,delta,nit,clinical,clinical_pro,meld_na \
    --endpoints hepatic,death \
    --modes drop,ipcw \
    --horizons q25,q50,q75 \
    --max-subset-size 4 \
    --max-features 100 \
    --device $DEVICE \
    --n-folds 5 \
    --output-dir $OUT

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
