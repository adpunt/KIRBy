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

# constraints.txt locks numpy/pandas/scipy/sklearn/torch so NO pip install can upgrade them
CONSTRAINTS="-c ${SCRIPT_DIR}/constraints.txt"

# Core scientific stack — only install if broken
if ! python -c "import numpy, pandas, scipy, sklearn" 2>/dev/null; then
    echo "Fixing broken scientific stack..."
    pip install numpy==1.26.4 pandas==2.2.0 scipy==1.11.4 scikit-learn==1.4.0 pytz==2023.3.post1 \
        || { echo "FATAL: core scientific stack install failed"; return 1 2>/dev/null || exit 1; }
fi

PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

# Load CUDA module (ARC cluster) — must come first so real CUPTI libs take
# priority over PyTorch's stub libcupti.so
module load CUDA 2>/dev/null || true
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null)
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$TORCH_LIB"
fi

# Install kirby package (editable)
pip install --quiet $CONSTRAINTS -e . 2>/dev/null || true

# Only install packages if not already present — avoids breaking torch-cluster/PyG
_install_if_missing() {
    python -c "import $1" 2>/dev/null || pip install --quiet $CONSTRAINTS "$2" 2>/dev/null || true
}

_install_if_missing lightgbm lightgbm
if [ "${PY_MINOR:-9}" -ge 10 ]; then
    _install_if_missing xgboost xgboost
else
    _install_if_missing xgboost 'xgboost<3'
fi
_install_if_missing shap 'shap<0.50'
_install_if_missing captum captum
_install_if_missing lime lime
_install_if_missing boruta boruta
_install_if_missing ImmuneBuilder ImmuneBuilder
_install_if_missing antifold antifold
_install_if_missing antiberty antiberty
_install_if_missing xformers 'xformers==0.0.25'
_install_if_missing openpyxl 'openpyxl>=3.1.0'

# Molecular reps: CLAMP, R-MAT
CLAMP_DIR="${SCRIPT_DIR}/src/clamp"
if [ ! -d "$CLAMP_DIR" ]; then
    echo "Installing CLAMP..."
    git clone --quiet https://github.com/ml-jku/clamp.git "$CLAMP_DIR" 2>/dev/null || true
    pip install --quiet $CONSTRAINTS -e "$CLAMP_DIR" 2>/dev/null || true
    pip install --quiet $CONSTRAINTS mhnreact 2>/dev/null || true
fi

# huggingmolecules — only install if not present
_install_if_missing huggingmolecules "git+https://github.com/gmum/huggingmolecules.git#subdirectory=src"

# Packages requiring Python >=3.10 or numpy>=2 — skip on 3.9
if [ "${PY_MINOR:-9}" -ge 10 ]; then
    _install_if_missing unimol_tools unimol-tools
    _install_if_missing esm esm
fi

# Always succeed so && chains in SLURM wraps don't break
true
