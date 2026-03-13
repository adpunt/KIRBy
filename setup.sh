#!/bin/bash
# Environment setup for KIRBy
# Called by SLURM scripts before running experiments

# Redirect all caches and user packages to /data/ (home quota is tiny on ARC)
# Load secrets (.env is gitignored)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# ---------------------------------------------------------------
# Activate venv if available. The py311-kirby venv has Python 3.11
# which is required for ESM-C, ESM3, SaProt, Boltz-2 (Phase 4B).
# If the venv doesn't exist, fall back to system Python.
# ---------------------------------------------------------------
VENV_DIR="/data/stat-cadd/scat9264/py311-kirby"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR ($(python --version 2>&1))"
else
    echo "No venv found at $VENV_DIR — using system Python ($(python --version 2>&1))"
    echo "  Phase 4B models (ESM3, SaProt, Boltz-2) require Python >=3.10."
    echo "  Create venv: python3.11 -m venv $VENV_DIR"
fi

export HF_HOME=/data/stat-cadd/scat9264/.cache/huggingface
export TORCH_HOME=/data/stat-cadd/scat9264/.cache/torch
export PIP_CACHE_DIR=/data/stat-cadd/scat9264/.cache/pip
export KIRBY_CACHE_DIR=/data/stat-cadd/scat9264/.cache/kirby
export KIRBY_MODELS_DIR=/data/stat-cadd/scat9264/kirby_models
# PYTHONUSERBASE only matters outside venv — inside venv, pip installs to venv
if [ -z "$VIRTUAL_ENV" ]; then
    export PYTHONUSERBASE=/data/stat-cadd/scat9264/.local
fi

# Suppress RDKit C++ deprecation warnings (GetValence etc.) and threadpoolctl
# ctypes crashes. Both are harmless stderr noise from C extensions that can't be
# caught by Python's warnings module or RDLogger.DisableLog.
export KIRBY_STDERR_FILTER="grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'AttributeError.*split'"

# Ensure user packages override system Anaconda packages (only outside venv)
if [ -z "$VIRTUAL_ENV" ]; then
    USER_SITE=$(python -c "import site; print(site.getusersitepackages())" 2>/dev/null)
    if [ -n "$USER_SITE" ] && [ -d "$USER_SITE" ]; then
        export PYTHONPATH="$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
    fi

    # Fix threadpoolctl crash from system Anaconda's old version (2022.05).
    pip install --user "threadpoolctl>=3.1" -q 2>/dev/null
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

# Install kirby package (editable) — core deps only.
# Domain extras (torch, transformers, scanpy, etc.) are assumed to already be
# in the venv from initial setup. We do NOT re-resolve extras on every job
# because that risks breaking a working environment.
pip install --quiet --no-deps -e . 2>/dev/null || true

# Only install packages if not already present — avoids breaking torch-cluster/PyG
_install_if_missing() {
    python -c "import $1" 2>/dev/null || pip install --quiet $CONSTRAINTS "$2" 2>/dev/null || true
}

# --- Core ML models (Phase 2) ---
_install_if_missing lightgbm lightgbm
if [ "${PY_MINOR:-9}" -ge 10 ]; then
    _install_if_missing xgboost xgboost
else
    _install_if_missing xgboost 'xgboost<3'
fi
_install_if_missing catboost catboost

# --- Feature importance / interpretability ---
_install_if_missing shap 'shap<0.50'
_install_if_missing captum captum
_install_if_missing lime lime
_install_if_missing boruta boruta

# --- Antibody ---
_install_if_missing ImmuneBuilder ImmuneBuilder
_install_if_missing antifold antifold
_install_if_missing antiberty antiberty
_install_if_missing ablang2 ablang2
_install_if_missing sentencepiece sentencepiece

# --- Misc ---
_install_if_missing xformers 'xformers==0.0.25'
_install_if_missing openpyxl 'openpyxl>=3.1.0'
_install_if_missing peft peft

# --- Phase 3B: Molecular descriptors + reaction reps ---
_install_if_missing mordred mordredcommunity
_install_if_missing drfp drfp
_install_if_missing rxnfp rxnfp
_install_if_missing polaris polaris-lib
_install_if_missing map4 map4

# --- Phase 4A: Forward compatibility models ---
_install_if_missing minimol minimol
# MINT and MolE are git clones — only run once, not on every job:
#   git clone https://github.com/VarunUllanat/mint.git src/MINT && pip install -e src/MINT
#   wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt -P $KIRBY_MODELS_DIR/mint/
#   git clone https://github.com/recursionpharma/mole_public.git src/mole && pip install -e src/mole

# Molecular reps: CLAMP, R-MAT
CLAMP_DIR="${SCRIPT_DIR}/src/clamp"
if [ ! -d "$CLAMP_DIR" ]; then
    echo "Installing CLAMP..."
    git clone --quiet https://github.com/ml-jku/clamp.git "$CLAMP_DIR" 2>/dev/null || true
    pip install --quiet $CONSTRAINTS -e "$CLAMP_DIR" 2>/dev/null || true
    pip install --quiet $CONSTRAINTS mhnreact 2>/dev/null || true
fi

# huggingmolecules (R-MAT) — clone locally, don't use git+https (hangs on auth)
HMOL_DIR="${SCRIPT_DIR}/src/huggingmolecules"
if ! python -c "import huggingmolecules" 2>/dev/null; then
    if [ -d "$HMOL_DIR" ]; then
        pip install --quiet $CONSTRAINTS -e "$HMOL_DIR/src" 2>/dev/null || true
    else
        echo "huggingmolecules not installed. Clone manually:"
        echo "  git clone https://github.com/gmum/huggingmolecules.git $HMOL_DIR"
        echo "  pip install -e $HMOL_DIR/src"
    fi
fi

# --- Packages requiring Python >=3.10 (Phase 4B) ---
if [ "${PY_MINOR:-9}" -ge 10 ]; then
    _install_if_missing unimol_tools unimol-tools
    _install_if_missing esm esm         # ESM-C, ESM3
    _install_if_missing boltz boltz     # Boltz-2
    # SaProt: uses transformers (HF) + ProstT5 — no extra install
fi

# PyG health check: detect broken C++ extensions (ABI mismatch from torch upgrades)
# _install_if_missing only checks Python import, which succeeds even with broken .so files.
# We test actual C++ ops — if they fail, warn but don't block (surveys work without PyG).
if python -c "import torch_cluster" 2>/dev/null; then
    if ! python -c "
import torch, torch_cluster
x = torch.randn(4, 3)
b = torch.zeros(4, dtype=torch.long)
torch_cluster.radius_graph(x, r=2.0, batch=b)
" 2>/dev/null; then
        echo "WARNING: PyG extensions have broken C++ binaries (ABI mismatch)."
        echo "  QM9 and DTA e2e comparison experiments will fail."
        echo "  Fix: bash fix_pyg.sh (from interactive GPU session)"
        echo "  Surveys and frozen reps still work fine."
    fi
fi

# Always succeed so && chains in SLURM wraps don't break
true
