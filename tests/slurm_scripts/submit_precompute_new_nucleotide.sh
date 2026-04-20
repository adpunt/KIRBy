#!/bin/bash
# Precompute the 6 newly-confirmed nucleotide datasets on ARC.
#
# 4 nt_* datasets (n=33K) use default speed=full + 6 models
# 2 gb_demo_* datasets (n=100K) drop SVM (kernel cost too high at n=100K),
#   keep MLP + all other models, bump to 256G + 4-day wall.
#
# 12 sbatch calls total (2 per dataset: standard + distance).
#
# Usage (from /data/stat-cadd/scat9264/KIRBy on the ARC login node):
#   bash tests/slurm_scripts/submit_precompute_new_nucleotide.sh

set -euo pipefail

WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"

# Datasets at standard nucleotide scale (n=33K, default 6 models)
NT_DATASETS=(
    nt_enhancers
    nt_enhancers_types
    nt_splice_sites_acceptors
    nt_splice_sites_donors
)

# Datasets at large scale (n=100K, drop SVM, bump resources)
GB_DEMO_DATASETS=(
    gb_demo_coding_vs_intergenomic_seqs
    gb_demo_human_or_worm
)

# 5 models = drop svm; keep ridge,rf,xgboost,lightgbm,mlp
GB_MODELS="ridge,rf,xgboost,lightgbm,mlp"

submit_nt() {
    local ds=$1
    local distance_flag=$2  # "" or "--distance"
    local suffix=$3         # "standard" or "distance"

    JOB=$(sbatch --clusters=arc \
        --job-name="pc_${ds}_all_${suffix}" \
        --partition=long --mem=128G --time=2-00:00:00 --cpus-per-task=4 \
        --wrap="${WRAP_PREFIX} && python tests/precompute_hybrids.py ${ds} --speed full ${distance_flag}" \
        | awk '{print $4}')
    echo "  ${JOB}  pc_${ds}_all_${suffix}  (long, 128G, 2d)"
}

submit_gb() {
    local ds=$1
    local distance_flag=$2
    local suffix=$3

    JOB=$(sbatch --clusters=arc \
        --job-name="pc_${ds}_all_${suffix}" \
        --partition=long --mem=256G --time=4-00:00:00 --cpus-per-task=4 \
        --wrap="${WRAP_PREFIX} && python tests/precompute_hybrids.py ${ds} --speed full --models ${GB_MODELS} ${distance_flag}" \
        | awk '{print $4}')
    echo "  ${JOB}  pc_${ds}_all_${suffix}  (long, 256G, 4d, models=${GB_MODELS})"
}

echo "=== Submitting 12 precompute jobs (6 datasets × 2 conditions) ==="
echo ""

echo "-- 4 nt_* datasets (n=33K, default 6 models) --"
for ds in "${NT_DATASETS[@]}"; do
    submit_nt "$ds" ""           "standard"
    submit_nt "$ds" "--distance" "distance"
done

echo ""
echo "-- 2 gb_demo_* datasets (n=100K, no SVM, 256G, 4d wall) --"
for ds in "${GB_DEMO_DATASETS[@]}"; do
    submit_gb "$ds" ""           "standard"
    submit_gb "$ds" "--distance" "distance"
done

echo ""
echo "Total: 12 sbatch calls"
echo "Monitor: squeue --clusters=arc -u \$USER"
