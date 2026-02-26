#!/bin/bash
# =============================================================================
# Setup pretrained molecular models on the server
#
# Idempotent — safe to re-run. Skips steps that are already done.
# Run from the login node (needs internet access for git clones and wget).
#
# NOTE: HF model downloads and model verification need torch/CUDA, so they
# are skipped on login nodes. The precache SLURM job will auto-download HF
# models on first use. Run verify_server_models.py from an interactive GPU
# session if you want to check: srun --gres=gpu:1 python scripts/verify_server_models.py
#
# Usage:
#   bash scripts/setup_server_models.sh
# =============================================================================

set -euo pipefail

# Load environment (KIRBY_MODELS_DIR, HF_HOME, etc.)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../setup.sh" 2>/dev/null || true

MODELS_DIR="${KIRBY_MODELS_DIR:-${HOME}/kirby_models}"
mkdir -p "$MODELS_DIR"

# Detect whether we can import torch (login nodes typically can't)
HAS_TORCH=false
if python -c "import torch" 2>/dev/null; then
    HAS_TORCH=true
fi

echo "============================================================"
echo "KIRBy Server Model Setup"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  HF_HOME:    ${HF_HOME:-~/.cache/huggingface}"
echo "  Has torch:  $HAS_TORCH"
echo "  Date:       $(date)"
echo "============================================================"

PASS=0
FAIL=0
SKIP=0

ok()   { echo "  [OK]   $1"; PASS=$((PASS + 1)); }
skip() { echo "  [SKIP] $1 (already exists)"; SKIP=$((SKIP + 1)); }
fail() { echo "  [FAIL] $1: $2"; FAIL=$((FAIL + 1)); }
warn() { echo "  [WARN] $1"; }

# =============================================================================
# 1. PIP DEPENDENCIES
# =============================================================================
echo ""
echo "--- Installing pip dependencies ---"

pip_install() {
    local pkg=$1
    if python -c "import $pkg" 2>/dev/null; then
        skip "pip: $pkg"
    else
        if pip install --quiet "$pkg" 2>/dev/null; then
            ok "pip: $pkg"
        else
            fail "pip: $pkg" "install failed"
        fi
    fi
}

# mol2vec needs gensim
pip_install gensim
pip_install mol2vec
pip_install selfies

# GNN deps (torch-geometric ecosystem)
echo "  Installing torch-geometric deps..."
pip install --quiet torch-geometric 2>/dev/null && ok "pip: torch-geometric" || fail "pip: torch-geometric" "install failed"
pip install --quiet ogb 2>/dev/null && ok "pip: ogb" || fail "pip: ogb" "install failed"

# SMI-TED needs pytorch-fast-transformers
pip install --quiet pytorch-fast-transformers 2>/dev/null && ok "pip: pytorch-fast-transformers" || warn "pytorch-fast-transformers failed (may need manual install)"

# Graph kernel
pip_install grakel

# SchNet
pip install --quiet schnetpack 2>/dev/null && ok "pip: schnetpack" || fail "pip: schnetpack" "install failed"

# =============================================================================
# 2. CLONE REPOSITORIES
# =============================================================================
echo ""
echo "--- Cloning model repositories ---"

clone_repo() {
    local name=$1
    local url=$2
    local target="${MODELS_DIR}/${name}"

    if [ -d "$target" ]; then
        skip "repo: $name"
    else
        echo "  Cloning $name..."
        if git clone --depth 1 "$url" "$target" 2>/dev/null; then
            ok "repo: $name"
        else
            fail "repo: $name" "git clone failed"
        fi
    fi
}

clone_repo "grover" "https://github.com/tencent-ailab/grover.git"
clone_repo "Chemformer" "https://github.com/MolecularAI/Chemformer.git"
clone_repo "MolCLR" "https://github.com/yuyangw/MolCLR.git"
clone_repo "GraphMVP" "https://github.com/chao1224/GraphMVP.git"

# IBM materials repo (SMI-TED + MHG-GNN)
clone_repo "materials" "https://github.com/IBM/materials.git"

# =============================================================================
# 3. DOWNLOAD WEIGHTS (non-torch downloads only)
# =============================================================================
echo ""
echo "--- Downloading model weights ---"

# mol2vec model_300dim.pkl (pure wget, no torch needed)
MOL2VEC_PKL="${MODELS_DIR}/model_300dim.pkl"
if [ -f "$MOL2VEC_PKL" ]; then
    skip "weights: mol2vec"
else
    echo "  Downloading mol2vec weights..."
    MOL2VEC_URL="https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
    if wget -q -O "$MOL2VEC_PKL" "$MOL2VEC_URL" 2>/dev/null || \
       curl -sL -o "$MOL2VEC_PKL" "$MOL2VEC_URL" 2>/dev/null; then
        ok "weights: mol2vec"
    else
        fail "weights: mol2vec" "download failed"
    fi
fi

# MolCLR checkpoint (Google Drive — may need manual download)
MOLCLR_CKPT="${MODELS_DIR}/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth"
if [ -f "$MOLCLR_CKPT" ]; then
    skip "weights: MolCLR"
else
    warn "MolCLR weights missing: $MOLCLR_CKPT"
    warn "  Google Drive auto-download often fails. Manual download:"
    warn "  https://drive.google.com/uc?export=download&id=1BwmVmBOxMDUmMlpAEB0RGjcnuW-vxhEv"
    warn "  Place at: $MOLCLR_CKPT"
fi

# GraphMVP weights (needs huggingface_hub which needs torch)
GRAPHMVP_CKPT="${MODELS_DIR}/GraphMVP/pretrained_models/GraphMVP.pth"
if [ -f "$GRAPHMVP_CKPT" ]; then
    skip "weights: GraphMVP"
elif [ "$HAS_TORCH" = true ]; then
    echo "  Downloading GraphMVP weights via HuggingFace..."
    mkdir -p "${MODELS_DIR}/GraphMVP/pretrained_models"
    if python -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download('chao1224/GraphMVP', 'pretraining_model/GraphMVP_C/model.pth',
                        local_dir='${MODELS_DIR}/GraphMVP/pretrained_models/..')
dest = '${GRAPHMVP_CKPT}'
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.move(path, dest)
print('Downloaded')
" 2>/dev/null; then
        ok "weights: GraphMVP"
    else
        fail "weights: GraphMVP" "HF download failed"
    fi
else
    warn "GraphMVP weights: skipped (needs torch). Will auto-download during precache job."
fi

# =============================================================================
# 4. PRE-DOWNLOAD HUGGINGFACE MODELS (torch required)
# =============================================================================
echo ""
if [ "$HAS_TORCH" = true ]; then
    echo "--- Pre-downloading HuggingFace models ---"

    hf_download() {
        local name=$1
        local model_id=$2

        if python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('${model_id}')
AutoModel.from_pretrained('${model_id}')
print('OK')
" 2>/dev/null; then
            ok "HF: $name ($model_id)"
        else
            fail "HF: $name" "download/load failed"
        fi
    }

    hf_download "ChemBERTa" "DeepChem/ChemBERTa-77M-MLM"
    hf_download "MolFormer" "ibm/MoLFormer-XL-both-10pct"
    hf_download "SELFormer" "HUBioDataLab/SELFormer"
    hf_download "ChemBERT" "jonghyunlee/ChemBERT_ChEMBL_pretrained"
else
    echo "--- Skipping HuggingFace model downloads (no torch on login node) ---"
    echo "  HF models will auto-download during the precache SLURM job."
    echo "  Or run this script from an interactive GPU session:"
    echo "    srun --gres=gpu:1 --mem=32G --time=01:00:00 bash scripts/setup_server_models.sh"
fi

# =============================================================================
# 5. MANUAL DOWNLOAD WARNINGS
# =============================================================================
echo ""
echo "--- Manual downloads needed ---"

GROVER_WEIGHTS="${MODELS_DIR}/grover/grover_large.pt"
if [ ! -f "$GROVER_WEIGHTS" ]; then
    warn "GROVER weights missing: $GROVER_WEIGHTS"
    warn "  Download: https://ai.tencent.com/ailab/ml/ml-data/grover-models/grover_large.tar.gz"
    warn "  Then: tar xzf grover_large.tar.gz -C $MODELS_DIR/grover/"
else
    skip "weights: GROVER (already present)"
fi

CHEMFORMER_WEIGHTS="${MODELS_DIR}/Chemformer/models/pre-trained/combined/step=1000000.ckpt"
ALT_CHEMFORMER="${MODELS_DIR}/Chemformer/step=1000000.ckpt"
if [ ! -f "$CHEMFORMER_WEIGHTS" ] && [ ! -f "$ALT_CHEMFORMER" ]; then
    warn "Chemformer weights missing"
    warn "  Download from Box: https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq"
    warn "  Place at: $CHEMFORMER_WEIGHTS"
else
    skip "weights: Chemformer (already present)"
fi

# =============================================================================
# 6. VERIFY (only if torch available)
# =============================================================================
echo ""
if [ "$HAS_TORCH" = true ]; then
    echo "--- Running verification ---"
    python "${SCRIPT_DIR}/verify_server_models.py" 2>&1 | head -40
else
    echo "--- Skipping model verification (no torch on login node) ---"
    echo "  Path check only:"
    for dir in grover Chemformer MolCLR GraphMVP materials; do
        if [ -d "${MODELS_DIR}/${dir}" ]; then
            echo "  [OK]   ${dir}/"
        else
            echo "  [MISS] ${dir}/"
        fi
    done
    for f in model_300dim.pkl grover/grover_large.pt; do
        if [ -f "${MODELS_DIR}/${f}" ]; then
            echo "  [OK]   ${f}"
        else
            echo "  [MISS] ${f}"
        fi
    done
    echo ""
    echo "  For full verification: srun --gres=gpu:1 python scripts/verify_server_models.py"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "  OK:      $PASS"
echo "  Skipped: $SKIP"
echo "  Failed:  $FAIL"
echo "  Date:    $(date)"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Some steps failed — check output above."
    echo "Non-critical failures (HF downloads, GraphMVP) will resolve"
    echo "automatically during the precache SLURM job."
fi
