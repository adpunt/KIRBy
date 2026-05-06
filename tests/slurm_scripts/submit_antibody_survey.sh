#!/bin/bash
# Submit antibody baseline survey as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_antibody_survey.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_antibody_survey.sh           # all reps (42)
#   bash tests/slurm_scripts/submit_antibody_survey.sh dms       # DMS datasets only
#   bash tests/slurm_scripts/submit_antibody_survey.sh inter     # inter-family only
#   bash tests/slurm_scripts/submit_antibody_survey.sh all_ft    # all + ESM-2 LoRA

MODE=${1:-all}
OUTPUT="tests/results/antibody_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# All 42 reps in one list
ALL_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local dataset=$2
    local extra=$3

    J=$(sbatch \
        --job-name="ab_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem=256G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/antibody_baseline_survey.py ${dataset} --output ${OUTPUT}/ --n-folds 5 --augment ${extra} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (256G, 48h, GPU)"
}

echo "Submitting antibody baseline survey..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    dms)
        submit all_dms all_dms "--reps ${ALL_REPS}"
        echo "  Total: 1 job (42 reps, DMS datasets)"
        ;;
    inter)
        submit all_inter all_inter "--reps ${ALL_REPS}"
        echo "  Total: 1 job (42 reps, inter-family datasets)"
        ;;
    all)
        submit all all "--reps ${ALL_REPS} --skip-existing"
        echo "  Total: 1 job (42 reps, all datasets)"
        ;;
    all_ft)
        submit ft_dms all_dms "--reps ${ALL_REPS} --finetune"
        submit ft_inter all_inter "--reps ${ALL_REPS} --finetune"
        echo "  Total: 2 jobs (42 reps + ESM-2 LoRA FT)"
        ;;
    *)
        echo "Usage: bash submit_antibody_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all       All 42 reps, all datasets (2 jobs)"
        echo "  all_ft    All 42 reps + ESM-2 LoRA fine-tuning (2 jobs)"
        echo "  dms       DMS datasets only (1 job)"
        echo "  inter     Inter-family datasets only (1 job)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
