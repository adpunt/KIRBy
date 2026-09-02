#!/bin/bash
# Environment setup for KIRBy
# Called by SLURM scripts before running experiments

# Force unbuffered Python output so SLURM logs show progress in real time
export PYTHONUNBUFFERED=1

# Native-library thread count — SPLIT to avoid nested oversubscription.
# The dominant cost is xgboost + lightgbm, which parallelise via OpenMP, so OMP
# gets the whole CPU allocation (capping it to 1 was a large regression for the
# exact models that dominate runtime). But sklearn models (RF/KNN) parallelise
# via joblib n_jobs=-1, and if BLAS/numba ALSO each spawn N threads underneath,
# you get N×N threads on N cores — the contention slowdown + the known
# numba+BLAS+joblib nesting crash. So BLAS/numba/veclib are pinned to 1: the
# heavy OpenMP models keep the full node, the nested libraries don't multiply.
# (Determinism of BLAS-computed reps is a free side benefit, not the goal — the
#  rep cache is name-keyed so it never drifts regardless.)
_KIRBY_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}"
export OMP_NUM_THREADS="$_KIRBY_THREADS"   # xgboost/lightgbm OpenMP — keep full
export OPENBLAS_NUM_THREADS=1              # prevent BLAS nesting under joblib
export MKL_NUM_THREADS=1                   # prevent MKL nesting under joblib
export NUMBA_NUM_THREADS=1                 # numba was named in the crash trigger
export VECLIB_MAXIMUM_THREADS=1            # macOS Accelerate, harmless on ARC

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

# 2026-05-07: stat-cadd group quota at 99.9%, repo migrated to stat-ecr.
# READ-ONLY caches (HF, torch, pip, models) stay on stat-cadd — no writes
# needed at job-time, just read from existing locations.
# WRITES go to stat-ecr: KIRBY_CACHE_DIR (search/baseline/folds caches),
# slurm logs (CWD-at-submit), tests/results/* (inside repo).
export HF_HOME=/data/stat-cadd/scat9264/.cache/huggingface
export TORCH_HOME=/data/stat-cadd/scat9264/.cache/torch
export PIP_CACHE_DIR=/data/stat-cadd/scat9264/.cache/pip
export KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby
export KIRBY_MODELS_DIR=/data/stat-cadd/scat9264/kirby_models
# PYTHONUSERBASE only matters outside venv — inside venv, pip installs to venv
if [ -z "$VIRTUAL_ENV" ]; then
    export PYTHONUSERBASE=/data/stat-cadd/scat9264/.local
fi

# Suppress RDKit C++ deprecation warnings (GetValence etc.) and threadpoolctl
# ctypes crashes. Both are harmless stderr noise from C extensions that can't be
# caught by Python's warnings module or RDLogger.DisableLog.
export KIRBY_STDERR_FILTER="grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'AttributeError.*split'"

# ---------------------------------------------------------------
# Non-pip binaries for the BCR-seq assembly pipelines (T-Ab.2):
# sra-tools, igblast, vsearch, seqkit, entrez-direct. Installed
# one-time via `scripts/install_bcr_binaries.sh` (matches the
# MolE / MINT precedent below — not run per-job). When $REPO/bin
# is absent the rest of the pipeline is unaffected.
# ---------------------------------------------------------------
if [ -d "$SCRIPT_DIR/bin" ]; then
    export PATH="$SCRIPT_DIR/bin:$PATH"
    if [ -f "$SCRIPT_DIR/bin/.igdata" ]; then
        IGDATA_PATH="$(cat "$SCRIPT_DIR/bin/.igdata")"
        [ -d "$IGDATA_PATH" ] && export IGDATA="$IGDATA_PATH"
    fi
fi

# Ensure user packages override system Anaconda packages (only outside venv)
if [ -z "$VIRTUAL_ENV" ]; then
    USER_SITE=$(python -c "import site; print(site.getusersitepackages())" 2>/dev/null)
    if [ -n "$USER_SITE" ] && [ -d "$USER_SITE" ]; then
        export PYTHONPATH="$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
    fi

    # Fix threadpoolctl crash from system Anaconda's old version (2022.05).
    pip install --user "threadpoolctl>=3.1" -q 2>/dev/null
else
    # Inside venv: clear any stale PYTHONPATH that points to user site-packages
    # (e.g. from a previous setup.sh run without the venv). This prevents
    # Python 3.9 packages from ~/.local leaking into the 3.11 venv.
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
fi

# Fast mode: skip all install checks (for precompute and other batch jobs
# where the venv is already set up). Set KIRBY_SKIP_INSTALLS=1 before sourcing.
if [ "${KIRBY_SKIP_INSTALLS:-0}" = "1" ]; then
    echo "[setup.sh] skip mode: loading CUDA module..."
    module load CUDA 2>/dev/null || true
    echo "[setup.sh] CUDA module done"
    # Set torch lib path without importing torch (avoids contention on shared nodes)
    _TORCH_DIR="${VENV_DIR}/lib/python3.11/site-packages/torch/lib"
    if [ -d "$_TORCH_DIR" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_TORCH_DIR"
    fi
    echo "[setup.sh] setup complete (skip mode)"
    return 0 2>/dev/null || exit 0
fi

# constraints.txt locks numpy/pandas/scipy/sklearn/torch so NO pip install can upgrade them
CONSTRAINTS="-c ${SCRIPT_DIR}/constraints.txt"

# Core scientific stack — only install if broken
if ! python -c "import numpy, pandas, scipy, sklearn" 2>/dev/null; then
    echo "Fixing broken scientific stack..."
    pip install numpy==1.26.4 pandas==2.2.0 scipy==1.11.4 scikit-learn==1.6.1 pytz==2023.3.post1 \
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
# pdbfixer FIRST, and it is not optional. ABodyBuilder2 imports its structure
# refinement step at module load, and that step imports pdbfixer, which imports
# openmm. With either missing, `import ImmuneBuilder` raises ModuleNotFoundError
# and every geometry feature is unavailable — that is what emptied the March
# structural geometry survey (15 datasets, no scores). It also defeats the
# _install_if_missing guard below, whose check is `import ImmuneBuilder`: the
# check fails on the missing pdbfixer, pip reports ImmuneBuilder already
# satisfied, and the run continues broken. Ordering the two lines fixes that.
# pdbfixer ships on conda-forge and GitHub; whether the PyPI name resolves
# varies by cluster, so try PyPI and fall back to the source repository.
_install_if_missing openmm openmm
if ! python -c "import pdbfixer" 2>/dev/null; then
    pip install --quiet $CONSTRAINTS pdbfixer 2>/dev/null || true
    python -c "import pdbfixer" 2>/dev/null \
        || pip install --quiet $CONSTRAINTS \
             'git+https://github.com/openmm/pdbfixer.git' 2>/dev/null || true
    python -c "import pdbfixer" 2>/dev/null \
        || echo "[setup.sh] WARNING: pdbfixer missing — ImmuneBuilder will not import, antibody geometry features unavailable"
fi
# ANARCI, same story one import further in: ABodyBuilder2 numbers every
# sequence before it predicts a structure, and that numbering is ANARCI.
# ANARCI itself shells out to hmmscan, so a missing HMMER binary breaks it even
# when the Python package installs. The check below names which of the two is
# absent instead of leaving a bare ModuleNotFoundError in the job log.
if ! python -c "import anarci" 2>/dev/null; then
    pip install --quiet $CONSTRAINTS anarci 2>/dev/null || true
    python -c "import anarci" 2>/dev/null \
        || pip install --quiet $CONSTRAINTS \
             'git+https://github.com/oxpig/ANARCI.git' 2>/dev/null || true
    if ! python -c "import anarci" 2>/dev/null; then
        if command -v hmmscan >/dev/null 2>&1; then
            echo "[setup.sh] WARNING: anarci missing (hmmscan is present) — ImmuneBuilder and antifold will not import, antibody geometry features unavailable"
        else
            echo "[setup.sh] WARNING: anarci missing AND no hmmscan on PATH — ANARCI needs HMMER; try 'module load HMMER' before setup.sh. ImmuneBuilder and antifold will not import."
        fi
    fi
fi
_install_if_missing ImmuneBuilder ImmuneBuilder

# `import anarci` succeeding proves nothing. ANARCI numbers a sequence by
# shelling out to the hmmscan binary, so with HMMER absent the import is clean,
# setup.sh prints "ImmuneBuilder OK", and every structure prediction then raises
# FileNotFoundError: 'hmmscan' in about 0.02 seconds. Jobs 8668608 and 8668609
# failed exactly that way: 3706 predictions in 8m39s, no structures, no reason
# in the log. So number a sequence here rather than importing one.
if python -c "import anarci" 2>/dev/null; then
    if ! python -c "
from anarci import number
r = number('EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS')
assert r and r[0], 'anarci returned no numbering'
" 2>/dev/null; then
        if command -v hmmscan >/dev/null 2>&1; then
            echo "[setup.sh] WARNING: anarci imports and hmmscan is on PATH, but numbering a known heavy chain failed — check the HMMER data files. Antibody structure prediction WILL fail on every sample."
        else
            echo "[setup.sh] WARNING: anarci imports but hmmscan is NOT on PATH, so numbering fails and antibody structure prediction WILL fail on every sample. Try 'module load HMMER' before setup.sh, or install HMMER into the environment."
        fi
    fi
fi
_install_if_missing antifold antifold
_install_if_missing antiberty antiberty
_install_if_missing ablang2 ablang2
_install_if_missing sentencepiece sentencepiece

# --- Misc ---
_install_if_missing xformers 'xformers==0.0.25'
_install_if_missing openpyxl 'openpyxl>=3.1.0'
_install_if_missing peft peft

# --- Label-noise injection (c10 noise-robustness) ---
# Private package (github.com/adpunt/noise_inject); not on PyPI, so the
# requirements pin alone never installs it on ARC. Clone + editable install,
# same precedent as CLAMP/MolE below. Runtime deps are only numpy/pandas/scipy
# (already locked), so --no-deps keeps the scientific stack untouched.
# The checkout is UPDATED on every setup, not only when the import fails. The
# earlier guard was `if ! python -c "import noiseInject"`, so once the package
# imported it was never touched again. That cost array 12906522 six of seven
# tasks on 2026-08-29: noise_spec.py had moved onto NoiseInjectorRegression.
# .from_condition (KIRBy c74aaf5, NoiseInject 1.0.0), the server's checkout
# still predated it, and every task died with AttributeError after building its
# representations.
NOISEINJECT_DIR="${SCRIPT_DIR}/src/noise_inject"
if [ ! -d "$NOISEINJECT_DIR" ]; then
    git clone --quiet https://github.com/adpunt/noise_inject.git "$NOISEINJECT_DIR" 2>/dev/null || true
fi
if [ -d "$NOISEINJECT_DIR/.git" ]; then
    git -C "$NOISEINJECT_DIR" pull --ff-only --quiet 2>/dev/null \
        || echo "[setup.sh] WARNING: could not fast-forward $NOISEINJECT_DIR — it may be stale"
fi
if [ -d "$NOISEINJECT_DIR" ]; then
    python -c "import noiseInject" 2>/dev/null \
        || pip install --quiet --no-deps $CONSTRAINTS -e "$NOISEINJECT_DIR" 2>/dev/null \
        || true
else
    echo "noiseInject not installed (label-noise experiments will fail). Clone manually:"
    echo "  git clone https://github.com/adpunt/noise_inject.git $NOISEINJECT_DIR"
    echo "  pip install --no-deps -e $NOISEINJECT_DIR"
fi
# Say WHICH copy every job is about to import and what version it is, so a stale
# install is visible at the top of the log instead of as a traceback hours in.
python - <<'PY' 2>/dev/null || echo "[setup.sh] WARNING: noiseInject does not import"
import noiseInject as n
has = hasattr(getattr(n, 'NoiseInjectorRegression', object), 'from_condition')
print(f"[setup.sh] noiseInject {getattr(n, '__version__', 'unknown')} "
      f"from {n.__file__}"
      + ("" if has else "  ** STALE: no NoiseInjectorRegression.from_condition **"))
PY

# --- Phase 3B: Molecular descriptors + reaction reps ---
_install_if_missing mordred mordredcommunity
_install_if_missing drfp drfp
_install_if_missing rxnfp rxnfp
_install_if_missing polaris polaris-lib
_install_if_missing map4 map4
_install_if_missing mol2vec mol2vec

# --- Phase 4A: Forward compatibility models ---
_install_if_missing minimol minimol
# MINT and MolE are git clones — only run once, not on every job:
#   git clone https://github.com/VarunUllanat/mint.git src/MINT && pip install -e src/MINT
#   wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt -P $KIRBY_MODELS_DIR/mint/
MOLE_DIR="${SCRIPT_DIR}/src/mole"
if ! python -c "import mole" 2>/dev/null; then
    if [ -d "$MOLE_DIR" ]; then
        pip install --quiet $CONSTRAINTS -e "$MOLE_DIR" 2>/dev/null || true
    else
        echo "MolE not installed. Clone manually:"
        echo "  git clone https://github.com/recursionpharma/mole_public.git $MOLE_DIR"
        echo "  pip install -e $MOLE_DIR"
        echo "  Download weights to \$KIRBY_MODELS_DIR/mole/ or ~/kirby_models/mole/"
    fi
fi

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
echo "[setup.sh] setup complete (full mode)"
true
