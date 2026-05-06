#!/bin/bash
# Nuclear fix for PyG (PyTorch Geometric) + torch extensions.
#
# ROOT CAUSE: xformers pulled in torch 2.8.0, overwriting torch 2.2.2.
# The PyG extensions (torch_cluster, torch_scatter, torch_sparse) have .so
# binaries linked against torch 2.8's C++ symbols. After torch was constrained
# back to 2.2.2, the .so files remain incompatible, causing:
#   OSError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs
#
# FIX: Complete uninstall of all PyG extensions, then reinstall from the
# correct wheel URL matching torch 2.2.2 + CUDA 12.1 + Python 3.9.
#
# MUST be run from a GPU interactive session (srun --gres=gpu:1 --pty bash)
# or a login node. NOT from a SLURM --wrap command.
#
# Usage:
#   srun --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
#   cd /data/stat-ecr/scat9264/KIRBy
#   . setup.sh   # sets up env vars, CONSTRAINTS, etc.
#   bash fix_pyg.sh

set -e

echo "=== PyG Nuclear Fix ==="
echo ""

# Step 0: Verify environment
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "MISSING")
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Python: ${PY_VER}"
echo "PyTorch: ${TORCH_VER}"
echo "CUDA: ${CUDA_VER}"
echo ""

if [ "$TORCH_VER" = "MISSING" ]; then
    echo "ERROR: PyTorch not installed. Installing torch 2.2.2..."
    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
fi

# If torch is wrong version, fix it — constraints.txt says 2.2.2 but something
# (usually xformers or another transitive dep) upgrades it silently.
if ! echo "$TORCH_VER" | grep -q '^2\.2'; then
    echo "ERROR: torch ${TORCH_VER} detected, expected 2.2.x"
    echo "  Something upgraded torch past constraints.txt pin."
    echo "  Downgrading to torch 2.2.2+cu121..."
    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
    echo "  Now: torch ${TORCH_VER}, CUDA ${CUDA_VER}"
    echo ""
fi

# Determine PyG wheel URL based on installed torch + CUDA
# Strip any +cu suffix from torch version for the URL
TORCH_BASE=$(echo "$TORCH_VER" | sed 's/+.*//')
CUDA_TAG=$(echo "$CUDA_VER" | sed 's/\.//')
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_BASE}+cu${CUDA_TAG}.html"
echo "PyG wheel URL: ${PYG_WHEEL_URL}"
echo ""

# Step 1: Nuclear uninstall — remove ALL PyG packages and their .so files
echo "Step 1: Removing ALL PyG packages..."
pip uninstall -y \
    torch-geometric \
    torch-scatter torch_scatter \
    torch-sparse torch_sparse \
    torch-cluster torch_cluster \
    torch-spline-conv torch_spline_conv \
    pyg-lib pyg_lib \
    2>/dev/null || true

# Also remove any leftover egg-info or .so files in site-packages
SITE_PKGS=$(python -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ -n "$SITE_PKGS" ] && [ -d "$SITE_PKGS" ]; then
    echo "  Cleaning leftover files in ${SITE_PKGS}..."
    rm -rf "${SITE_PKGS}"/torch_scatter* 2>/dev/null || true
    rm -rf "${SITE_PKGS}"/torch_sparse* 2>/dev/null || true
    rm -rf "${SITE_PKGS}"/torch_cluster* 2>/dev/null || true
    rm -rf "${SITE_PKGS}"/torch_spline_conv* 2>/dev/null || true
    rm -rf "${SITE_PKGS}"/pyg_lib* 2>/dev/null || true
fi

# Check system site-packages too
SYS_SITE=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)
if [ -n "$SYS_SITE" ] && [ -d "$SYS_SITE" ]; then
    echo "  Cleaning leftover files in ${SYS_SITE}..."
    rm -rf "${SYS_SITE}"/torch_scatter* 2>/dev/null || true
    rm -rf "${SYS_SITE}"/torch_sparse* 2>/dev/null || true
    rm -rf "${SYS_SITE}"/torch_cluster* 2>/dev/null || true
    rm -rf "${SYS_SITE}"/torch_spline_conv* 2>/dev/null || true
    rm -rf "${SYS_SITE}"/pyg_lib* 2>/dev/null || true
fi

echo "  Done."
echo ""

# Step 2: Verify torch is still intact after uninstalls
echo "Step 2: Verifying PyTorch..."
python -c "
import torch
assert torch.__version__.startswith('2.2'), f'Expected torch 2.2.x, got {torch.__version__}'
print(f'  torch {torch.__version__} OK (CUDA {torch.version.cuda})')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: No GPU detected (expected if on login node)')
"
echo ""

# Step 3: Install compiled extensions from correct wheel URL
# CRITICAL: --no-deps prevents numpy 2.x from being pulled in
echo "Step 3: Installing PyG extensions from ${PYG_WHEEL_URL}..."
pip install --no-deps \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f "${PYG_WHEEL_URL}"
echo ""

# Step 4: Install torch-geometric (pure Python)
# Pin to <2.7 because 2.7.0 requires Python >= 3.10
echo "Step 4: Installing torch-geometric..."
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
if [ "${PY_MINOR}" -lt 10 ]; then
    echo "  Python 3.${PY_MINOR} detected — pinning torch-geometric<2.7 (2.7+ needs >=3.10)"
    pip install --no-deps 'torch-geometric>=2.5,<2.7'
else
    pip install --no-deps 'torch-geometric>=2.5'
fi
echo ""

# Step 5: Verify EVERYTHING works
echo "Step 5: Verification..."
python -c "
import torch
print(f'torch:          {torch.__version__} (CUDA {torch.version.cuda})')

import torch_geometric
print(f'torch-geometric: {torch_geometric.__version__}')

import torch_scatter
print(f'torch-scatter:   {torch_scatter.__version__}')

import torch_sparse
print(f'torch-sparse:    {torch_sparse.__version__}')

import torch_cluster
print(f'torch-cluster:   {torch_cluster.__version__}')

import torch_spline_conv
print(f'torch-spline-conv: {torch_spline_conv.__version__}')

# Critical test: actually USE the C++ extensions (not just import)
import torch
x = torch.randn(10, 3)
batch = torch.zeros(10, dtype=torch.long)

# Test torch_scatter
src = torch.randn(10, 5)
index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
out = torch_scatter.scatter(src, index, dim=0, reduce='mean')
print(f'torch_scatter:   C++ ops OK (scatter_mean: {out.shape})')

# Test torch_cluster
from torch_cluster import radius_graph
edge_index = radius_graph(x, r=2.0, batch=batch)
print(f'torch_cluster:   C++ ops OK (radius_graph: {edge_index.shape})')

# Test torch_sparse
row = torch.tensor([0, 1, 2])
col = torch.tensor([1, 2, 0])
val = torch.ones(3)
adj = torch_sparse.SparseTensor(row=row, col=col, value=val, sparse_sizes=(3, 3))
print(f'torch_sparse:    C++ ops OK (SparseTensor: {adj.sizes()})')

print()
print('=== ALL PyG EXTENSIONS VERIFIED ===')
"

echo ""
echo "Step 6: Quick SchNet test..."
python -c "
from torch_geometric.nn.models import SchNet
model = SchNet(hidden_channels=128, num_filters=128, num_interactions=6)
print(f'SchNet model created: {sum(p.numel() for p in model.parameters())} params')
print('SchNet OK')
"

echo ""
echo "=== PyG Fix Complete ==="
echo ""
echo "Next steps:"
echo "  1. Resubmit QM9:     bash tests/slurm_scripts/submit_comparisons.sh qm9"
echo "  2. Resubmit DTA e2e: bash tests/slurm_scripts/submit_comparisons.sh dta"
echo "  3. Or submit all:    bash tests/slurm_scripts/submit_comparisons.sh all"
