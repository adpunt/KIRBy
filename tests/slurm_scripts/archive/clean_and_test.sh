#!/bin/bash
#SBATCH --job-name=kirby_clean_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# =============================================================================
# STEP 1: Clean stale TCGA rep caches
# =============================================================================
# These were generated with the ssGSEA orientation bug (reactome/kegg/oncogenic)
# or decoupler compat bug (progeny/dorothea). The cache validation code will
# auto-detect them, but cleaning explicitly is cleaner.

echo "============================================================"
echo "STEP 1: Cleaning stale TCGA rep caches"
echo "============================================================"

STALE_REPS="gene_reactome gene_kegg gene_oncogenic gene_progeny gene_dorothea"
CACHE_DIRS=$(find ~/.cache/kirby/reps -maxdepth 1 -name 'tcga_*' -type d 2>/dev/null)
SEAGATE_DIRS=$(find /Volumes/seagate/kirby/reps -maxdepth 1 -name 'tcga_*' -type d 2>/dev/null)

for dir in $CACHE_DIRS $SEAGATE_DIRS; do
    for rep in $STALE_REPS; do
        if [ -f "$dir/${rep}.npy" ]; then
            echo "  Removing stale cache: $dir/${rep}.npy"
            rm "$dir/${rep}.npy"
        fi
    done
done
echo "  Done."

# =============================================================================
# STEP 2: Test each potentially unreliable TCGA rep individually
# =============================================================================
# Tests gene_hallmark, gene_kegg, gene_reactome, gene_oncogenic, gene_progeny,
# gene_dorothea, gene_mutation, clinical on tcga_luad_vs_lusc.
# This confirms which reps actually work on the server before running the full survey.

echo ""
echo "============================================================"
echo "STEP 2: Testing individual TCGA reps"
echo "============================================================"

PASS=0
FAIL=0
SKIP=0

test_tcga_rep() {
    local rep=$1
    echo -n "  Testing $rep... "
    if python -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')

from find_best_hybrid_tcga import get_representations
from kirby.datasets.tcga import load_dataset

expression, labels, task_type, metadata = load_dataset('tcga_luad_vs_lusc')
gene_names = metadata['gene_names']
sample_ids = metadata['sample_ids']
cohort = metadata.get('cohort', 'LUAD')

reps = get_representations(
    expression, gene_names, sample_ids,
    wsi_dir=None, preset_or_list='$rep',
    cohort=cohort, cache=False
)
if not reps:
    print('NO OUTPUT', file=sys.stderr)
    sys.exit(1)
for name, X in reps.items():
    print(f'{name}: shape={X.shape}')
" 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
}

# Reps that should always work (no external downloads after first run)
for rep in gene_pca100 gene_nmf50 gene_ica50 gene_stats gene_immune; do
    test_tcga_rep $rep
done

# Reps that need gseapy (ssGSEA)
for rep in gene_hallmark gene_kegg gene_reactome gene_oncogenic; do
    test_tcga_rep $rep
done

# Reps that need decoupler
for rep in gene_progeny gene_dorothea gene_combined_pathway; do
    test_tcga_rep $rep
done

# Reps that need Xena downloads (potentially 403)
for rep in gene_mutation cnv_gistic cnv_pca50 rppa mirna_top200 clinical; do
    test_tcga_rep $rep
done

echo ""
echo "TCGA rep tests: $PASS passed, $FAIL failed"

# =============================================================================
# STEP 3: Test DTA dataset loading (was failing due to old PyTDC code)
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 3: Testing DTA dataset loading"
echo "============================================================"

echo -n "  Testing davis... "
if python -c "
import sys
sys.path.insert(0, 'src')
from kirby.datasets.dta import load_dataset
data = load_dataset('davis')
print(f'OK: {len(data[\"labels\"])} pairs')
" 2>&1; then
    echo "PASS"
else
    echo "FAIL"
fi

echo -n "  Testing kiba... "
if python -c "
import sys
sys.path.insert(0, 'src')
from kirby.datasets.dta import load_dataset
data = load_dataset('kiba')
print(f'OK: {len(data[\"labels\"])} pairs')
" 2>&1; then
    echo "PASS"
else
    echo "FAIL"
fi

# =============================================================================
# STEP 4: Test molecular dataset loading (should be fine)
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 4: Quick molecular + antibody sanity check"
echo "============================================================"

echo -n "  Testing logd dataset... "
if python -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
from find_best_hybrid_molecular import load_dataset
smiles, labels, task = load_dataset('logd')
print(f'OK: {len(smiles)} molecules, task={task}')
" 2>&1; then
    echo "PASS"
else
    echo "FAIL"
fi

echo -n "  Testing flab_expression_koenig dataset... "
if python -c "
import sys
sys.path.insert(0, 'src')
from kirby.datasets.flab import load_flab_dataset
data = load_flab_dataset('flab_expression_koenig')
print(f'OK: {len(data[\"heavy_sequences\"])} samples')
" 2>&1; then
    echo "PASS"
else
    echo "FAIL"
fi

echo ""
echo "============================================================"
echo "All tests complete. Review output above to decide which reps"
echo "to include/exclude before running the full baseline survey."
echo "============================================================"
