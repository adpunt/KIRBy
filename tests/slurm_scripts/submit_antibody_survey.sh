#!/bin/bash
# Submit antibody baseline survey as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_antibody_survey.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_antibody_survey.sh           # all reps (18)
#   bash tests/slurm_scripts/submit_antibody_survey.sh fast      # seq encodings only (CPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh lm        # LM reps only (GPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh struct    # structural reps (GPU/CPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh all_ft    # all + ESM-2 LoRA (GPU)

MODE=${1:-all}
OUTPUT="tests/results/antibody_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

# All 18 rep names
ALL_LM="esm2,esmc,ablang2,igbert,antiberty,prot_t5,amplify,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers"
ALL_FAST="onehot,blosum,aac,paratope"
ALL_STRUCT="structural"
ALL_REPS="${ALL_LM},${ALL_FAST},${ALL_STRUCT}"

submit() {
    local name=$1
    local dataset=$2
    local extra=$3
    local mem=${4:-64G}
    local wall=${5:-12:00:00}
    local gpu=${6:-}

    J=$(sbatch \
        --job-name="ab_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        ${gpu:+--gres=gpu:1} \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUTPUT} && python tests/antibody_baseline_survey.py ${dataset} --output ${OUTPUT}/ --n-folds 5 --augment ${extra}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu:+, GPU})"
}

echo "Submitting antibody baseline survey..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    fast)
        # Fast reps (CPU): onehot, blosum, aac, paratope
        submit fast_dms all_dms "--reps ${ALL_FAST}" 32G 02:00:00
        submit fast_inter all_inter "--reps ${ALL_FAST}" 32G 02:00:00
        echo ""
        echo "  Total: 2 jobs (4 fast reps, CPU only, with --augment)"
        ;;
    lm)
        # All LM reps (GPU): 13 reps
        submit lm_dms all_dms "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit lm_inter all_inter "--reps ${ALL_LM}" 96G 23:59:00 gpu
        echo ""
        echo "  Total: 2 jobs (13 LM reps, GPU, with --augment)"
        ;;
    struct)
        # Structural reps (needs ImmuneBuilder, CPU-heavy but benefits from GPU)
        submit struct_dms all_dms "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit struct_inter all_inter "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        echo ""
        echo "  Total: 2 jobs (structural reps, GPU, with --augment)"
        ;;
    dms)
        submit fast_dms all_dms "--reps ${ALL_FAST}" 32G 02:00:00
        submit lm_dms all_dms "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit struct_dms all_dms "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        echo ""
        echo "  Total: 3 jobs (DMS: fast + LM + structural, with --augment)"
        ;;
    inter)
        submit fast_inter all_inter "--reps ${ALL_FAST}" 32G 02:00:00
        submit lm_inter all_inter "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit struct_inter all_inter "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        echo ""
        echo "  Total: 3 jobs (inter: fast + LM + structural, with --augment)"
        ;;
    all)
        submit fast_dms all_dms "--reps ${ALL_FAST}" 32G 02:00:00
        submit fast_inter all_inter "--reps ${ALL_FAST}" 32G 02:00:00
        submit lm_dms all_dms "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit lm_inter all_inter "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit struct_dms all_dms "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit struct_inter all_inter "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        echo ""
        echo "  Total: 6 jobs (18 reps: 4 fast + 13 LM + 1 structural, with --augment)"
        ;;
    all_ft)
        submit fast_dms all_dms "--reps ${ALL_FAST}" 32G 02:00:00
        submit fast_inter all_inter "--reps ${ALL_FAST}" 32G 02:00:00
        submit lm_ft_dms all_dms "--reps ${ALL_LM} --finetune" 96G 23:59:00 gpu
        submit lm_ft_inter all_inter "--reps ${ALL_LM} --finetune" 96G 23:59:00 gpu
        submit struct_dms all_dms "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit struct_inter all_inter "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        echo ""
        echo "  Total: 6 jobs (18 reps + ESM-2 LoRA FT, with --augment, GPU)"
        ;;
    *)
        echo "Usage: bash submit_antibody_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all      All 18 reps, all datasets (6 jobs: 2 CPU + 4 GPU)"
        echo "  all_ft   All 18 reps + ESM-2 LoRA fine-tuning (6 jobs, GPU)"
        echo "  fast     Fast reps only (onehot, blosum, aac, paratope) — CPU"
        echo "  lm       LM reps (13: ESM-2, ESM-C, AbLang2, IgBert, AntiBERTy,"
        echo "           ProtT5, AMPLIFY, + CDR variants + midlayers) — GPU"
        echo "  struct   Structural reps (SASA + confidence + paratope, ImmuneBuilder)"
        echo "  dms      DMS datasets only (fast + LM + structural)"
        echo "  inter    Inter-family datasets only (fast + LM + structural)"
        echo ""
        echo "All modes include --augment (concat small reps with top base)."
        echo ""
        echo "Rep categories (18 total):"
        echo "  LM full-seq (7):  esm2, esmc, ablang2, igbert, antiberty, prot_t5, amplify"
        echo "  LM CDR-pooled (5): esm2_cdr, esmc_cdr, antiberty_cdr, prot_t5_cdr, amplify_cdr"
        echo "  LM midlayer (1):  esm2_midlayers"
        echo "  Seq encoding (4): onehot, blosum, aac, paratope"
        echo "  Structural (1):   structural (28D: SASA + confidence + paratope)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
