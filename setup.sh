#!/bin/bash
# Environment setup for KIRBy
# Called by SLURM scripts before running experiments

# Redirect all caches to /data/ (home quota is tiny on ARC)
export HF_HOME=/data/stat-cadd/scat9264/.cache/huggingface
export TORCH_HOME=/data/stat-cadd/scat9264/.cache/torch
export PIP_CACHE_DIR=/data/stat-cadd/scat9264/.cache/pip

# Install KIRBy in editable mode (picks up src/kirby)
pip install --quiet -e . 2>/dev/null

# Install any missing dependencies
pip install --quiet shap captum lime Boruta 2>/dev/null
