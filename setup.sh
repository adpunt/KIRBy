#!/bin/bash
# Environment setup for KIRBy
# Called by SLURM scripts before running experiments

# Redirect all caches and user packages to /data/ (home quota is tiny on ARC)
# Load secrets (.env is gitignored)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

export HF_HOME=/data/stat-cadd/scat9264/.cache/huggingface
export TORCH_HOME=/data/stat-cadd/scat9264/.cache/torch
export PIP_CACHE_DIR=/data/stat-cadd/scat9264/.cache/pip
export PYTHONUSERBASE=/data/stat-cadd/scat9264/.local
export KIRBY_CACHE_DIR=/data/stat-cadd/scat9264/.cache/kirby
export KIRBY_MODELS_DIR=/data/stat-cadd/scat9264/kirby_models

# Install KIRBy in editable mode (picks up src/kirby)
pip install --quiet -e . 2>/dev/null

# Install any missing dependencies
pip install --quiet shap captum lime Boruta 2>/dev/null
pip install --quiet ImmuneBuilder antifold 2>/dev/null
