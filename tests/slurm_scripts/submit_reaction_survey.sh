#!/bin/bash
# Submit reaction baseline survey with expanded rep set (38 reps).
# Screening mode: 3 models × 3 folds (vs. full 5×5).
#
# 38 reps = 14 original + 24 molecular variants:
#   Original: drfp_2048, product/reactant/diff_ecfp4, product/reactant/diff_maccs,
#             product/reactant_pdv, rxnfp, reactiont5, onehot_components,
#             component_descriptors, ligand_descriptors
#   New:      {product,reactant,diff}_{mordred,mhg_gnn,grover,mol2vec,
#              chemberta,molformer,avalon,pharmacophore3d}
#
# Needs GPU for: chemberta, molformer, grover, mhg_gnn, rxnfp, reactiont5.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_reaction_survey.sh           # all 3 datasets
#   bash tests/slurm_scripts/submit_reaction_survey.sh bh        # buchwald_hartwig only
#   bash tests/slurm_scripts/submit_reaction_survey.sh suzuki    # suzuki only
#   bash tests/slurm_scripts/submit_reaction_survey.sh doyle     # bh_doyle_plates only

MODE=${1:-all}
OUTPUT="tests/results/reaction_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Screening: 3 fast models, 3 folds
MODELS="ridge,rf,xgboost"
FOLDS=3

submit() {
    local name=$1
    local dataset=$2

    J=$(sbatch \
        --job-name="rxn_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=23:59:00 --partition=long \
        --mem=256G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/run_baseline_survey.py ${dataset} --n-folds ${FOLDS} --models ${MODELS} --augment --output ${OUTPUT}/ ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  rxn_${name}  (long, 256G, 24h, GPU, ${FOLDS} folds, models=${MODELS})"
}

echo "Submitting reaction baseline survey (screening mode)..."
echo "  Output: ${OUTPUT}/"
echo "  Models: ${MODELS}"
echo "  Folds: ${FOLDS}"
echo "  Reps: 38 (14 original + 24 molecular variants)"
echo ""

case "$MODE" in
    all)
        submit bh buchwald_hartwig
        submit suzuki suzuki
        submit doyle bh_doyle_plates
        echo ""
        echo "  Total: 3 jobs (38 reps × 3 models × 3 folds per dataset)"
        ;;
    bh)
        submit bh buchwald_hartwig
        ;;
    suzuki)
        submit suzuki suzuki
        ;;
    doyle)
        submit doyle bh_doyle_plates
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_reaction_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all      All 3 reaction datasets (default)"
        echo "  bh       Buchwald-Hartwig only"
        echo "  suzuki   Suzuki only"
        echo "  doyle    BH Doyle plates only"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
