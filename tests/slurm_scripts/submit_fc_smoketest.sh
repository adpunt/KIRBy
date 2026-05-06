#!/bin/bash
# Forward Compatibility Smoke Test
#
# Lightweight test: can each new model load, embed, and score?
# 1 small dataset per model, 2 folds, only the specific rep.
# Should complete in <2 hours per job (vs 24-48h for full experiments).
#
# Usage:
#   bash tests/slurm_scripts/submit_fc_smoketest.sh

set -euo pipefail

BASE_OUTPUT="tests/results/forward_compat/smoketest"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-04:00:00}

    J=$(sbatch \
        --job-name="fc_smoke_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall})"
}

echo "============================================"
echo "Forward Compat Smoke Test"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo "  1 dataset per model, 2 folds, targeted reps"
echo ""

# --- Protein/Antibody models ---

# MINT (antibody, 1 dataset)
submit mint \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps mint"

# SaProt (antibody, 1 dataset)
submit saprot \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps saprot"

# ESM3 (antibody, 1 dataset)
submit esm3 \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps esm3"

# IgT5 (antibody, 1 dataset)
submit igt5 \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps igt5"

# Ab-RoBERTa (antibody, 1 dataset)
submit abroberta \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps abroberta"

# xTrimoPGLM (antibody, 1 dataset)
submit xtrimopglm \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps xtrimopglm"

# AIDO.Protein-16B (antibody, 1 dataset — needs more VRAM)
submit aido \
    "mkdir -p ${BASE_OUTPUT} && python tests/antibody_baseline_survey.py flab_binding_koenig --output ${BASE_OUTPUT}/ --n-folds 2 --reps aido_protein" \
    256G 04:00:00

# Boltz-2 (antibody — uses protein embedder, no antibody wrapper yet; test via DTA instead)
# Boltz-2 is currently broken (every sequence fails), skip for now.

# --- Molecular models ---

# MolE (molecular, logd)
submit mole \
    "mkdir -p ${BASE_OUTPUT} && python tests/run_baseline_survey.py logd --n-folds 2 --output ${BASE_OUTPUT}/ --reps drug_mole"

# MiniMol (molecular, logd)
submit minimol \
    "mkdir -p ${BASE_OUTPUT} && python tests/run_baseline_survey.py logd --n-folds 2 --output ${BASE_OUTPUT}/ --reps drug_minimol"

# GP-MoLFormer (molecular, logd)
submit gp_molformer \
    "mkdir -p ${BASE_OUTPUT} && python tests/run_baseline_survey.py logd --n-folds 2 --output ${BASE_OUTPUT}/ --reps drug_gp_molformer"

# --- Reaction models ---

# ReactionT5 (reaction, buchwald_hartwig)
submit reactiont5 \
    "mkdir -p ${BASE_OUTPUT} && python tests/run_baseline_survey.py buchwald_hartwig --n-folds 2 --output ${BASE_OUTPUT}/ --reps reactiont5"

# --- Genomic models ---

# Geneformer V2 (TCGA, luad_vs_lusc)
submit geneformer_v2 \
    "mkdir -p ${BASE_OUTPUT} && python tests/run_baseline_survey.py tcga_luad_vs_lusc --n-folds 2 --output ${BASE_OUTPUT}/ --reps gene_geneformer_v2" \
    256G 04:00:00

echo ""
echo "============================================"
echo "Total: 13 smoke test jobs"
echo "  7 protein/antibody (flab_binding_koenig, 2-fold)"
echo "  3 molecular (logd, 2-fold)"
echo "  1 reaction (buchwald_hartwig, 2-fold)"
echo "  1 genomic (tcga_luad_vs_lusc, 2-fold)"
echo ""
echo "Each tests: model loads → embeddings generated → Ridge scores"
echo "Expected: <2h per job"
echo "============================================"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
