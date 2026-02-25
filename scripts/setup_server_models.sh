#!/bin/bash
# =============================================================================
# Setup pretrained molecular models on the server
#
# Idempotent — safe to re-run. Skips steps that are already done.
# Run from the login node (needs internet access for downloads).
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

echo "============================================================"
echo "KIRBy Server Model Setup"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  HF_HOME:    ${HF_HOME:-~/.cache/huggingface}"
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
# 3. DOWNLOAD WEIGHTS
# =============================================================================
echo ""
echo "--- Downloading model weights ---"

# mol2vec model_300dim.pkl
MOL2VEC_PKL="${MODELS_DIR}/model_300dim.pkl"
if [ -f "$MOL2VEC_PKL" ]; then
    skip "weights: mol2vec"
else
    echo "  Downloading mol2vec weights..."
    if python -c "
from mol2vec.features import mol2alt_sentence, mol2sentence
from gensim.models import word2vec
import urllib.request, os
url = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
urllib.request.urlretrieve(url, '${MOL2VEC_PKL}')
print('Downloaded')
" 2>/dev/null; then
        ok "weights: mol2vec"
    else
        fail "weights: mol2vec" "download failed"
    fi
fi

# MolCLR checkpoint
MOLCLR_CKPT="${MODELS_DIR}/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth"
if [ -f "$MOLCLR_CKPT" ]; then
    skip "weights: MolCLR"
else
    echo "  Downloading MolCLR weights..."
    mkdir -p "$(dirname "$MOLCLR_CKPT")"
    if python -c "
import urllib.request
url = 'https://drive.google.com/uc?export=download&id=1BwmVmBOxMDUmMlpAEB0RGjcnuW-vxhEv'
urllib.request.urlretrieve(url, '${MOLCLR_CKPT}')
print('Downloaded')
" 2>/dev/null; then
        ok "weights: MolCLR"
    else
        warn "MolCLR weights may need manual download from Google Drive"
        fail "weights: MolCLR" "auto-download failed (try manual)"
    fi
fi

# GraphMVP via HuggingFace
GRAPHMVP_DIR="${MODELS_DIR}/GraphMVP/pretrained_models"
GRAPHMVP_CKPT="${GRAPHMVP_DIR}/GraphMVP.pth"
if [ -f "$GRAPHMVP_CKPT" ]; then
    skip "weights: GraphMVP"
else
    echo "  Downloading GraphMVP weights..."
    mkdir -p "$GRAPHMVP_DIR"
    if python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('chao1224/GraphMVP', 'pretraining_model/GraphMVP_C/model.pth',
                        local_dir='${GRAPHMVP_DIR}/..')
import shutil, os
os.makedirs('${GRAPHMVP_DIR}', exist_ok=True)
shutil.move(path, '${GRAPHMVP_CKPT}')
print('Downloaded')
" 2>/dev/null; then
        ok "weights: GraphMVP"
    else
        fail "weights: GraphMVP" "HF download failed"
    fi
fi

# =============================================================================
# 4. PRE-DOWNLOAD HUGGINGFACE MODELS
# =============================================================================
echo ""
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

# SMI-TED (ibm/materials.smi-ted)
echo "  Pre-downloading SMI-TED..."
python -c "
try:
    from huggingface_hub import snapshot_download
    snapshot_download('ibm/materials.smi-ted')
    print('OK')
except Exception as e:
    print(f'WARN: {e}')
" 2>/dev/null && ok "HF: SMI-TED" || warn "SMI-TED download may need manual setup"

# =============================================================================
# 5. MANUAL DOWNLOAD WARNINGS
# =============================================================================
echo ""
echo "--- Manual downloads needed ---"

GROVER_WEIGHTS="${MODELS_DIR}/grover/grover_large.pt"
if [ ! -f "$GROVER_WEIGHTS" ]; then
    warn "GROVER weights missing: $GROVER_WEIGHTS"
    warn "  Download from OneDrive: https://ai.tencent.com/ailab/ml/ml-data/grover-models/grover_large.tar.gz"
    warn "  Extract to: $MODELS_DIR/grover/"
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
# 6. VERIFY
# =============================================================================
echo ""
echo "--- Running verification ---"
python "${SCRIPT_DIR}/verify_server_models.py" 2>&1 | head -40

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
    echo "Some steps failed. Check output above and fix manually."
    exit 1
fi
