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

# constraints.txt locks numpy/pandas/scipy/sklearn so NO pip install can upgrade them
CONSTRAINTS="-c ${SCRIPT_DIR}/constraints.txt"

# Core scientific stack — pinned to known-working versions (requirements-working.txt)
CORE_PKGS='numpy==1.26.4 pandas==2.2.0 scipy==1.11.4 scikit-learn==1.4.0 pytz==2023.3.post1'
if ! python -c "import numpy, pandas, scipy, sklearn" 2>/dev/null; then
    echo "Fixing broken scientific stack..."
    pip install $CORE_PKGS \
        || { echo "FATAL: core scientific stack install failed"; return 1 2>/dev/null || exit 1; }
else
    pip install --quiet $CORE_PKGS 2>/dev/null || true
fi

PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

# Load CUDA module (ARC cluster) — must come first so real CUPTI libs take
# priority over PyTorch's stub libcupti.so
module load CUDA 2>/dev/null || true
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null)
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$TORCH_LIB"
fi

# All subsequent pip installs use $CONSTRAINTS to prevent upgrading core stack
pip install --quiet $CONSTRAINTS -e . 2>/dev/null || true

# lightgbm + xgboost (used by all surveys); xgboost >=3 requires Python >=3.10
if [ "${PY_MINOR:-9}" -ge 10 ]; then
    pip install --quiet $CONSTRAINTS lightgbm xgboost 2>/dev/null || true
else
    pip install --quiet $CONSTRAINTS lightgbm 'xgboost<3' 2>/dev/null || true
fi
pip install --quiet $CONSTRAINTS 'shap<0.50' captum lime Boruta 2>/dev/null || true
pip install --quiet $CONSTRAINTS ImmuneBuilder antifold antiberty 2>/dev/null || true
pip install --quiet $CONSTRAINTS xformers 2>/dev/null || true  # required by AMPLIFY
pip install --quiet $CONSTRAINTS 'openpyxl>=3.1.0' 2>/dev/null || true

# Molecular reps: CLAMP, R-MAT
CLAMP_DIR="${SCRIPT_DIR}/src/clamp"
if [ ! -d "$CLAMP_DIR" ]; then
    echo "Installing CLAMP..."
    git clone --quiet https://github.com/ml-jku/clamp.git "$CLAMP_DIR" 2>/dev/null || true
    pip install --quiet $CONSTRAINTS -e "$CLAMP_DIR" 2>/dev/null || true
    pip install --quiet $CONSTRAINTS mhnreact 2>/dev/null || true
fi
pip install --quiet $CONSTRAINTS "git+https://github.com/gmum/huggingmolecules.git#subdirectory=src" 2>/dev/null || true

# Packages requiring Python >=3.10 or numpy>=2 — skip on 3.9
if [ "${PY_MINOR:-9}" -ge 10 ]; then
    pip install --quiet unimol-tools 2>/dev/null || true
    pip install --quiet esm 2>/dev/null || true
fi

# Always succeed so && chains in SLURM wraps don't break
true
