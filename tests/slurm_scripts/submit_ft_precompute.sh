#!/bin/bash
# Phase B FT precompute on HTC (GPU).
#
# Goal: populate the per-fold .npz cache at $KIRBY_CACHE_DIR/finetuned_reps/
# for every FT rep × dataset × fold combo, so downstream experiments can
# load them via load_or_compute() without re-running fine-tuning.
#
# Coverage:
#   - Molecular  → precompute_ft_reps.py --domain molecular --speed full
#                  Caches DATASET_REPS['ft_reps']:
#                    lora_molformer, contrastive_molformer, gnn_graphmvp
#                  NOTE: does NOT cover lora_chemberta or ft_schnet — those live
#                  in make_ft_factories() (run_baseline_survey.py --finetune).
#                  Run run_baseline_survey.py --finetune separately if needed.
#   - Antibody   → precompute_ft_reps.py --domain antibody --speed full
#                  Caches antibody_lora factories.
#   - DTA        → run_baseline_survey.py <ds> --finetune --n-folds 5
#                  No DTA support in precompute_ft_reps.py; survey populates
#                  cache as side effect of FT factory call.
#   - TCGA       → SKIPPED per user.
#
# Usage (from /data/stat-cadd/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_ft_precompute.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull --ff-only && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
DTA_OUT="tests/results/phase5_baselines/dta"

submit_gpu() {
    local name=$1
    local cmd=$2
    local mem=$3
    local wall=$4

    JOB=$(sbatch --parsable \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}")
    echo "  ${JOB}  ${name}  (long, ${mem}, ${wall}, GPU)"
}

echo "=== Phase B FT Precompute (HTC GPU) ==="
echo ""

echo "--- Molecular FT precompute (1 job, all 8 datasets) ---"
submit_gpu "ftpc_molecular" \
    "python tests/precompute_ft_reps.py --domain molecular --speed full" \
    "128G" "23:59:00"

echo ""
echo "--- Antibody FT precompute (1 job, all 29 datasets) ---"
submit_gpu "ftpc_antibody" \
    "python tests/precompute_ft_reps.py --domain antibody --speed full" \
    "128G" "3-23:59:00"

echo ""
echo "--- DTA FT survey (3 jobs, populates cache via --finetune) ---"
mkdir -p "${DTA_OUT}"
for ds in davis kiba bindingdb_kd; do
    submit_gpu "ftpc_dta_${ds}" \
        "mkdir -p ${DTA_OUT} && python tests/run_baseline_survey.py ${ds} --finetune --n-folds 5 --output ${DTA_OUT}/" \
        "256G" "3-23:59:00"
done

echo ""
echo "Total: 5 GPU jobs"
echo "Cache target: \$KIRBY_CACHE_DIR/finetuned_reps/ (= /data/stat-cadd/scat9264/.cache/kirby/finetuned_reps/)"
echo "Monitor: squeue -u \$USER"
echo ""
echo "⚠️  bindingdb_kd: job 7516299 is already running the hybrid resubmit. The new"
echo "    ftpc_dta_bindingdb_kd job will queue and may start while 7516299 is still"
echo "    going — both write to ${DTA_OUT}/bindingdb_kd.json. If concurrency causes"
echo "    issues, scancel the FT job and resubmit after 7516299 finishes."
