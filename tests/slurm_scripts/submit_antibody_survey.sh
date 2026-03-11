#!/bin/bash
# Submit antibody baseline survey as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_antibody_survey.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_antibody_survey.sh           # all reps (35)
#   bash tests/slurm_scripts/submit_antibody_survey.sh fast      # seq encodings only (CPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh lm        # LM reps only (GPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh struct    # structural reps (GPU/CPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh all_ft    # all + ESM-2 LoRA (GPU)

MODE=${1:-all}
OUTPUT="tests/results/antibody_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

# All 29 rep names
# LM reps: 9 full-seq + 7 CDR + 1 midlayer + 4 Phase 3C layer/concat + 2 Ab-RoBERTa = 23
ALL_LM="esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm3,esm3_cdr,saprot,saprot_cdr,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr"
ALL_FAST="onehot,blosum,aac,paratope"
ALL_STRUCT="structural"
# Mutation reps: 3 embedding-based + 3 LLR-based = 6
ALL_MUTATION="mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr"
ALL_VARIATION="esm2_entropy_top20,esm2_per_cdr,esm2_per_region"
ALL_REPS="${ALL_LM},${ALL_FAST},${ALL_STRUCT},${ALL_MUTATION},${ALL_VARIATION}"

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
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUTPUT} && python tests/antibody_baseline_survey.py ${dataset} --output ${OUTPUT}/ --n-folds 5 --augment ${extra} 2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)" \
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
        echo "  Total: 2 jobs (25 LM reps, GPU, with --augment)"
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
    mutation)
        # Mutation-aware reps (GPU): 4 reps, only useful for ΔΔG/DMS datasets with mutation metadata
        submit mut_dms all_dms "--reps ${ALL_MUTATION}" 96G 23:59:00 gpu
        submit mut_inter all_inter "--reps ${ALL_MUTATION}" 96G 23:59:00 gpu
        echo ""
        echo "  Total: 2 jobs (6 mutation reps, GPU, with --augment)"
        ;;
    variation)
        # Variation/entropy-aware pooling reps (GPU): 3 reps
        submit var_dms all_dms "--reps ${ALL_VARIATION}" 96G 23:59:00 gpu
        submit var_inter all_inter "--reps ${ALL_VARIATION}" 96G 23:59:00 gpu
        echo ""
        echo "  Total: 2 jobs (3 variation reps, GPU, with --augment)"
        ;;
    all)
        submit fast_dms all_dms "--reps ${ALL_FAST}" 32G 02:00:00
        submit fast_inter all_inter "--reps ${ALL_FAST}" 32G 02:00:00
        submit lm_dms all_dms "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit lm_inter all_inter "--reps ${ALL_LM}" 96G 23:59:00 gpu
        submit struct_dms all_dms "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit struct_inter all_inter "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit mut_dms all_dms "--reps ${ALL_MUTATION}" 96G 23:59:00 gpu
        submit mut_inter all_inter "--reps ${ALL_MUTATION}" 96G 23:59:00 gpu
        submit var_dms all_dms "--reps ${ALL_VARIATION}" 96G 23:59:00 gpu
        submit var_inter all_inter "--reps ${ALL_VARIATION}" 96G 23:59:00 gpu
        echo ""
        echo "  Total: 10 jobs (35 reps: 4 fast + 25 LM + 1 structural + 6 mutation + 3 variation, with --augment)"
        ;;
    all_ft)
        submit fast_dms all_dms "--reps ${ALL_FAST}" 32G 02:00:00
        submit fast_inter all_inter "--reps ${ALL_FAST}" 32G 02:00:00
        submit lm_ft_dms all_dms "--reps ${ALL_LM} --finetune" 96G 23:59:00 gpu
        submit lm_ft_inter all_inter "--reps ${ALL_LM} --finetune" 96G 23:59:00 gpu
        submit struct_dms all_dms "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit struct_inter all_inter "--reps ${ALL_STRUCT}" 64G 23:59:00 gpu
        submit mut_dms all_dms "--reps ${ALL_MUTATION}" 96G 23:59:00 gpu
        submit mut_inter all_inter "--reps ${ALL_MUTATION}" 96G 23:59:00 gpu
        submit var_dms all_dms "--reps ${ALL_VARIATION}" 96G 23:59:00 gpu
        submit var_inter all_inter "--reps ${ALL_VARIATION}" 96G 23:59:00 gpu
        echo ""
        echo "  Total: 10 jobs (35 reps + ESM-2 LoRA FT, with --augment, GPU)"
        ;;
    *)
        echo "Usage: bash submit_antibody_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all       All 35 reps, all datasets (10 jobs: 2 CPU + 8 GPU)"
        echo "  all_ft    All 35 reps + ESM-2 LoRA fine-tuning (10 jobs, GPU)"
        echo "  fast      Fast reps only (onehot, blosum, aac, paratope) — CPU"
        echo "  lm        LM reps (25: ESM-2, ESM-C, AbLang2, IgBert, IgBert_cls, AntiBERTy,"
        echo "            ProtT5, AMPLIFY, IgT5, ESM3, SaProt + CDR variants + midlayers"
        echo "            + layer24, early, concat, cdr3, abroberta) — GPU"
        echo "  struct    Structural reps (SASA + confidence + paratope, ImmuneBuilder)"
        echo "  mutation  Mutation-aware reps (6: mutation_features, esm2_mutsite,"
        echo "            esm2_wt_delta, esm2_llr, esmc_llr, antiberty_llr) — GPU"
        echo "  variation Variation/entropy-aware pooling (3: esm2_entropy_top20,"
        echo "            esm2_per_cdr, esm2_per_region) — GPU"
        echo "  dms       DMS datasets only (fast + LM + structural)"
        echo "  inter     Inter-family datasets only (fast + LM + structural)"
        echo ""
        echo "All modes include --augment (concat small reps with top base)."
        echo ""
        echo "Rep categories (35 total):"
        echo "  LM full-seq (11):   esm2, esmc, ablang2, igbert, igbert_cls, antiberty, prot_t5, amplify, igt5, esm3, saprot"
        echo "  LM CDR-pooled (7):  esm2_cdr, esmc_cdr, antiberty_cdr, prot_t5_cdr, amplify_cdr, esm3_cdr, saprot_cdr"
        echo "  LM midlayer (1):    esm2_midlayers"
        echo "  LM layer/concat (4):esm2_layer24, esm2_early, esm2_concat, esm2_cdr3"
        echo "  Ab-RoBERTa (2):     abroberta, abroberta_cdr"
        echo "  Seq encoding (4):   onehot, blosum, aac, paratope"
        echo "  Structural (1):     structural (28D: SASA + confidence + paratope)"
        echo "  Mutation (6):       mutation_features, esm2_mutsite, esm2_wt_delta, esm2_llr, esmc_llr, antiberty_llr"
        echo "  Variation (3):      esm2_entropy_top20, esm2_per_cdr, esm2_per_region"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
