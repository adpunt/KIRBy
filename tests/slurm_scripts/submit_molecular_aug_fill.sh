#!/bin/bash
# Fill missing augmentation data: usrcat, coulomb_matrix, usr.
# These 3D shape reps are inconsistent across datasets — need per-dataset
# testing before deciding domain-wide inclusion.
#
# Datasets with gaps (usrcat/coulomb/usr):
#   caco2     — missing all 3
#   logd      — missing coulomb_matrix, usr
#   qm9       — missing usr
#   adme_mdr1 — missing all 3
#
# Already complete: herg, adme_hlm, adme_rlm.
#
# --augment re-computes baselines (needed for top-2 base ranking),
# then tests ALL tier-1/2 reps as augments. Results merge with
# existing data via dict update (no overwrites of different-format data).
#
# GPU needed for base rep generation (chemberta, molformer, grover, mhg_gnn).
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_molecular_aug_fill.sh          # all 4 datasets
#   bash tests/slurm_scripts/submit_molecular_aug_fill.sh caco2    # single dataset

MODE=${1:-all}
OUTPUT="tests/results/molecular_aug_fill"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Screening: 3 models, 3 folds
MODELS="ridge,rf,xgboost"
FOLDS=3

submit() {
    local name=$1
    local dataset=$2
    local mem=${3:-128G}
    local wall=${4:-11:59:00}

    J=$(sbatch \
        --job-name="aug_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/run_baseline_survey.py ${dataset} --augment --models ${MODELS} --n-folds ${FOLDS} --output ${OUTPUT}/ ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  aug_${name}  (long, ${mem}, ${wall}, GPU, ${FOLDS} folds)"
}

echo "Submitting molecular augmentation fill..."
echo "  Output: ${OUTPUT}/"
echo "  Models: ${MODELS} (screening)"
echo "  Folds: ${FOLDS}"
echo "  Target augments: usrcat (60D), coulomb_matrix (50D), usr (12D)"
echo ""

case "$MODE" in
    all)
        submit caco2 caco2
        submit logd logd
        submit qm9 qm9 128G 23:59:00
        submit adme_mdr1 adme_mdr1
        echo ""
        echo "  Total: 4 jobs (baselines + augmentation per dataset)"
        ;;
    caco2|logd|qm9|adme_mdr1)
        local_wall=11:59:00
        [ "$MODE" = "qm9" ] && local_wall=23:59:00
        submit "$MODE" "$MODE" 128G "$local_wall"
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_molecular_aug_fill.sh [MODE]"
        echo ""
        echo "Modes: all (default), caco2, logd, qm9, adme_mdr1"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
echo ""
echo "Look at augmentations section: delta > 0 means augment helps."
echo "Resolves whether usrcat/coulomb/usr are domain-wide or dataset-specific."
