#!/bin/bash
# Submit structural geometry augmentation survey as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_geometry_survey.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_geometry_survey.sh           # inter-family + ginkgo
#   bash tests/slurm_scripts/submit_geometry_survey.sh all       # all datasets
#   bash tests/slurm_scripts/submit_geometry_survey.sh dms       # DMS only
#   bash tests/slurm_scripts/submit_geometry_survey.sh inter     # inter-family only

MODE=${1:-default}
CLUSTERS="${CLUSTERS:-htc}"
OUTPUT="tests/results/geometry_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull --ff-only origin similarity-metrics-study 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && python tests/check_structure_prediction.py"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Base reps to augment with (must already be cached from antibody survey)
# Override with BASE_REPS=pruned to ask each dataset for the pool it was
# precomputed with, DATASET_REPS[ds]['general']. The nine below are a
# fixed list that is not any dataset's pruned pool — abcov_kd shares three
# names with it, ddg_combined one — so asking for them finds almost nothing
# in the precompute cache.
BASE_REPS="${BASE_REPS:-esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac}"

# check_structure_prediction.py runs first, on the compute node, and exits
# non-zero if antibody structure prediction is broken there. && then stops the
# job before it spends hours producing zeros. It costs about 40 seconds and it
# is the only place the check can run honestly: the login node has no CUDA, so
# torch does not import there and the check cannot reach a prediction at all.
#
# --clusters=htc, because this job asks for a GPU. Oxford ARC runs two clusters
# and the cluster is chosen by the LOGIN NODE, not by the partition: from
# arc-login every job lands on arc, where the GPU request is rejected with
# "Invalid feature specification" (both partitions, 2026-08-28). htc is where
# the GPU, high-memory and high-throughput nodes are. submit_phase5_htc.sh runs
# the same flags and is named for it. Override with CLUSTERS= if submitting from
# htc-login, where it is already implied.
#
# Keep every comment OUT of the sbatch call below. A comment between two
# backslash-continued lines swallows the rest of the command, --wrap included,
# and sbatch then blocks forever reading a batch script from stdin.
submit() {
    local name=$1
    local dataset=$2
    local extra=$3

    J=$(sbatch \
        --job-name="geo_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --clusters="${CLUSTERS}" \
        --time=47:59:00 --partition=long \
        --mem=256G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/structural_geometry_survey.py ${dataset} --base-reps ${BASE_REPS} --models rf,mlp --output ${OUTPUT}/ ${extra} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (256G, 48h, GPU)"
}

echo "Submitting structural geometry augmentation survey..."
echo "  Output: ${OUTPUT}/"
echo "  Base reps: ${BASE_REPS}"
echo ""

case "$MODE" in
    default)
        submit inter all_inter ""
        submit ginkgo all_ginkgo ""
        echo "  Total: 2 jobs (inter-family + Ginkgo developability)"
        ;;
    dms)
        submit dms all_dms ""
        echo "  Total: 1 job (DMS datasets)"
        ;;
    inter)
        submit inter all_inter ""
        echo "  Total: 1 job (inter-family datasets)"
        ;;
    all)
        submit dms all_dms ""
        submit inter all_inter ""
        submit clf all_clf ""
        submit ginkgo all_ginkgo ""
        submit ddg all_ddg ""
        echo "  Total: 5 jobs (all dataset groups)"
        ;;
    *)
        echo "Usage: bash submit_geometry_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  default   Inter-family + Ginkgo (2 jobs)"
        echo "  all       All dataset groups (5 jobs)"
        echo "  dms       DMS datasets only (1 job)"
        echo "  inter     Inter-family datasets only (1 job)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
