#!/bin/bash
#SBATCH --job-name=similarity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Similarity Experiments
#
# Quick mode: logd only, fast reps, subset of experiments
# Full mode: split into parts to fit within 48h wall time
#   Default reps: fast+hf (safe). Use 3rd arg to override (e.g. "all" after pre-caching).
#
# Usage:
#   sbatch run_similarity_experiments.sh quick
#
#   # Full mode — submit one job per dataset × part:
#   sbatch run_similarity_experiments.sh full_a logd           # fast+hf reps
#   sbatch run_similarity_experiments.sh full_b logd
#   sbatch run_similarity_experiments.sh full_a logd all       # all reps (needs pre-cache)
#   sbatch run_similarity_experiments.sh full_cross            # Exp 9 (cross-dataset)

MODE=${1:-quick}
DATASET=${2:-logd}
REPS=${3:-}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+4+5+8 ==="
    python tests/similarity_experiments.py --mode quick --reps fast \
        --experiments 1 2 4 5 8 --output tests/results/similarity_fast

    echo ""
    echo "=== QUICK 2: logd, all reps, Exp 1+2+5 ==="
    python tests/similarity_experiments.py --mode quick --reps all \
        --experiments 1 2 5 --output tests/results/similarity_all

elif [ "$MODE" = "full_a" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL PART A: ${DATASET}, ${REP_FLAG} reps, Exp 1+2+4+5+6+8 ==="
    python tests/similarity_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets ${DATASET} \
        --experiments 1 2 4 5 6 8 --output tests/results/similarity_${DATASET}_a

elif [ "$MODE" = "full_b" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL PART B: ${DATASET}, ${REP_FLAG} reps, Exp 3+7 ==="
    python tests/similarity_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets ${DATASET} \
        --experiments 3 7 --output tests/results/similarity_${DATASET}_b

elif [ "$MODE" = "full_cross" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL CROSS-DATASET: all 4 datasets, ${REP_FLAG} reps, Exp 9 ==="
    python tests/similarity_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets logd caco2 herg qm9 \
        --experiments 9 --output tests/results/similarity_cross

elif [ "$MODE" = "full_mol" ]; then
    # Molecular-only experiments. s04 (activity cliffs, ECFP4-Tanimoto) and
    # s11 (GP-BO on fingerprints) both gate on domain=='molecular' and were
    # skipped in the antibody/reaction/TCGA batch. Run them on the molecular
    # datasets so they actually produce results (user decision 2026-06-03).
    # Uses the canonical run_dataset.py runner (the s-module dispatch that
    # writes tests/results/similarity/s04_* and s11_*), one dataset at a time.
    echo "=== FULL MOLECULAR: logd caco2 herg qm9, s04 + s11 ==="
    rc=0
    for DS in logd caco2 herg qm9; do
        echo "--- $DS: s04 s11 ---"
        python tests/experiments/run_dataset.py similarity "$DS" \
            --experiments s04 s11 || rc=1
    done
    exit $rc

elif [ "$MODE" = "full_s09" ]; then
    # s09 reads the finished s02/s04-s08 result files and writes the
    # cross-dataset summary. `full_cross` above does NOT run it: that branch
    # runs tests/similarity_experiments.py's own experiment 9, which is the
    # legacy cross-dataset routine, not tests/experiments/similarity/
    # s09_cross_dataset_summary.py. This branch runs the s-module through the
    # canonical runner.
    #
    # s09 needs the per-experiment result files already on disk, and it needs
    # --output tests/results or it cannot see them. On one dataset it writes
    # n_datasets=1 and skips the paired test, and the file still reads as a
    # cross-dataset result later — so give it every dataset that has inputs.
    DS_LIST=${2:-"logd caco2 herg qm9"}
    echo "=== S09 CROSS-DATASET SUMMARY: ${DS_LIST} ==="
    rc=0
    for DS in ${DS_LIST}; do
        echo "--- $DS: s09 ---"
        python tests/experiments/run_dataset.py similarity "$DS" \
            --experiments s09 --output tests/results || rc=1
    done
    exit $rc

else
    echo "Usage: sbatch run_similarity_experiments.sh [quick|full_a|full_b|full_cross|full_mol|full_s09] [dataset] [reps]"
    echo "Datasets: logd, caco2, herg, qm9"
    echo "Reps: fast+hf (default), all (needs run_precache_reps.sh first)"
    echo ""
    echo "Full mode split:"
    echo "  full_a    = Exp 1,2,4,5,6,8 (core similarity experiments)"
    echo "  full_b    = Exp 3,7          (supplementary: neighbor preservation, noise FP)"
    echo "  full_cross = Exp 9           (cross-dataset summary, loads all datasets)"
    echo "  full_mol  = Exp 4,11         (molecular-only: activity cliffs + GP-BO; molecular datasets)"
    echo "  full_s09  = s09              (cross-dataset summary from the s-module, needs the other s-results on disk)"
    exit 1
fi
