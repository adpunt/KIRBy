#!/bin/bash
# Submit antibody baseline survey as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_antibody_survey.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_antibody_survey.sh           # all (fast + LM)
#   bash tests/slurm_scripts/submit_antibody_survey.sh fast      # one-hot, blosum, aac only (CPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh lm        # LM reps only (GPU)
#   bash tests/slurm_scripts/submit_antibody_survey.sh dms       # DMS datasets only
#   bash tests/slurm_scripts/submit_antibody_survey.sh inter     # inter-family only

MODE=${1:-all}
OUTPUT="tests/results/antibody_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

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
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python tests/antibody_baseline_survey.py ${dataset} --output ${OUTPUT}/ --n-folds 5 ${extra}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu:+, GPU})"
}

echo "Submitting antibody baseline survey..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    fast)
        # Fast reps (CPU): 12 datasets x 3 reps x 1 model
        submit fast_dms all_dms "--skip-lm" 32G 02:00:00
        submit fast_inter all_inter "--skip-lm" 32G 02:00:00
        echo ""
        echo "  Total: 2 jobs (fast reps, CPU only)"
        ;;
    lm)
        # LM reps (GPU): 12 datasets x 5 reps x 1 model
        submit lm_dms all_dms "--reps esm2,ablang2,igbert,antiberty,prot_t5" 64G 12:00:00 gpu
        submit lm_inter all_inter "--reps esm2,ablang2,igbert,antiberty,prot_t5" 64G 12:00:00 gpu
        echo ""
        echo "  Total: 2 jobs (LM reps, GPU)"
        ;;
    dms)
        submit fast_dms all_dms "--skip-lm" 32G 02:00:00
        submit lm_dms all_dms "--reps esm2,ablang2,igbert,antiberty,prot_t5" 64G 12:00:00 gpu
        echo ""
        echo "  Total: 2 jobs (DMS: fast + LM)"
        ;;
    inter)
        submit fast_inter all_inter "--skip-lm" 32G 02:00:00
        submit lm_inter all_inter "--reps esm2,ablang2,igbert,antiberty,prot_t5" 64G 12:00:00 gpu
        echo ""
        echo "  Total: 2 jobs (inter: fast + LM)"
        ;;
    all)
        submit fast_dms all_dms "--skip-lm" 32G 02:00:00
        submit fast_inter all_inter "--skip-lm" 32G 02:00:00
        submit lm_dms all_dms "--reps esm2,ablang2,igbert,antiberty,prot_t5" 64G 12:00:00 gpu
        submit lm_inter all_inter "--reps esm2,ablang2,igbert,antiberty,prot_t5" 64G 12:00:00 gpu
        echo ""
        echo "  Total: 4 jobs (12 datasets x 8 reps)"
        ;;
    *)
        echo "Usage: bash submit_antibody_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all    All reps, all datasets (4 jobs: 2 CPU + 2 GPU)"
        echo "  fast   Fast reps only (one-hot, blosum, aac) — CPU"
        echo "  lm     LM reps only (ESM-2, AbLang2, IgBert, AntiBERTy, ProtT5) — GPU"
        echo "  dms    DMS datasets only (fast + LM)"
        echo "  inter  Inter-family datasets only (fast + LM)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
