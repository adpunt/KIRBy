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

# Ensure user packages override system Anaconda packages (e.g., openpyxl 3.0.9)
USER_SITE=$(python -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ -n "$USER_SITE" ] && [ -d "$USER_SITE" ]; then
    export PYTHONPATH="$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

# Install KIRBy in editable mode (picks up src/kirby)
pip install --quiet -e . 2>/dev/null

# Install any missing dependencies
pip install --quiet shap captum lime Boruta 2>/dev/null
pip install --quiet ImmuneBuilder antifold antiberty 2>/dev/null
pip install --quiet 'openpyxl>=3.1.0' 2>/dev/null

# Molecular reps: CLAMP, R-MAT, Uni-Mol v2
CLAMP_DIR="$(dirname "$SCRIPT_DIR" 2>/dev/null)/src/clamp"
[ -z "$CLAMP_DIR" ] && CLAMP_DIR="$PWD/src/clamp"
if [ ! -d "$CLAMP_DIR" ]; then
    echo "Installing CLAMP..."
    git clone --quiet https://github.com/ml-jku/clamp.git "$CLAMP_DIR" 2>/dev/null
    pip install --quiet -e "$CLAMP_DIR" 2>/dev/null
    pip install --quiet mhnreact 2>/dev/null
fi
pip install --quiet "git+https://github.com/gmum/huggingmolecules.git#subdirectory=src" 2>/dev/null  # R-MAT
pip install --quiet unimol-tools 2>/dev/null  # Uni-Mol v2

# Protein/antibody LMs: ESM-C
pip install --quiet esm 2>/dev/null
