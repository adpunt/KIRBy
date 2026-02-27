#!/bin/bash
#SBATCH --job-name=ab_full_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Comprehensive Antibody Hybrid Search
#
# Tests ALL antibody datasets with appropriate presets:
# - Intra-family DMS: dms/dms_full presets (varpos reps + mutation features)
#   NOTE: The 'dms' and 'dms_full' presets include region-filtered variable-
#   position reps (varpos_cdr_physchem, varpos_cdrh3_physchem, varpos_fw_physchem).
#   These use approximate Chothia CDR boundaries to isolate mutations in CDRs
#   vs framework. CDR-H3 (56D on flab_expr) is the most compact and focused.
#   For inter-family datasets, varpos reps are irrelevant (all positions vary).
# - Inter-family: inter_family preset (full-sequence LM embeddings)
#
# Usage:
#   sbatch run_antibody_full_search.sh                   # all datasets
#   sbatch run_antibody_full_search.sh intra              # intra-family only
#   sbatch run_antibody_full_search.sh inter              # inter-family only
#   sbatch run_antibody_full_search.sh single <dataset> <preset>  # one run

MODE=${1:-all}
SEARCH=${2:-medium}
OUTPUT="tests/results/ab_full_search"

mkdir -p "$OUTPUT"

echo "============================================================"
echo "Antibody Full Search"
echo "  Mode:   $MODE"
echo "  Search: $SEARCH"
echo "  Output: $OUTPUT"
echo "  Started: $(date)"
echo "============================================================"

run_one() {
    local dataset=$1
    local preset=$2
    local flags=$3
    local tag="${dataset}_${preset}"

    echo ""
    echo ">>> $tag (flags: ${flags:-none})"
    echo "    Started: $(date)"
    python -u tests/find_best_hybrid_antibody.py "$dataset" \
        --reps "$preset" --search "$SEARCH" --augment $flags \
        --output "${OUTPUT}/results_${tag}.json" 2>&1
    echo "    Finished: $(date)"
}

run_intra() {
    echo ""
    echo "========== INTRA-FAMILY DMS DATASETS =========="
    echo "Variable-position reps should shine here."
    echo ""

    # FLAb expression (4,275 single-site mutants, G6 SSM) — best shot at R² > 0.5
    run_one flab_expression_koenig dms ""
    run_one flab_expression_koenig dms_full ""

    # FLAb expression with delta reps (mutation_fast now includes varpos_physchem)
    run_one flab_expression_koenig mutation_fast "--delta"

    # FLAb binding (same family, different property)
    run_one flab_binding_koenig dms ""

    # FLAb thermostability (Hie et al.)
    run_one flab_thermo_hie dms ""

    # FLAb binding ROSACE (adalimumab DMS)
    run_one flab_binding_rosace dms ""

    # AbAgym intra-family DMS datasets
    run_one abagym_Ang2_2017_G6 dms ""
    run_one abagym_EGFR_2013_Cetuximab dms ""
    run_one abagym_lysozyme_2019_D441 dms ""
}

run_inter() {
    echo ""
    echo "========== INTER-FAMILY DATASETS =========="
    echo "Full-sequence LM embeddings appropriate here."
    echo "Using all_server preset (14 reps) for comprehensive baseline survey."
    echo ""

    # Combined ΔΔG (SKEMPI 2.0 + AB-Bind, deduplicated, 544 entries)
    run_one ddg_combined all_server ""

    # AbAgym pooled (5 experiments, 2,247 entries)
    run_one abagym_all all_server ""

    # Ab-CoV binding affinity (141 entries)
    run_one abcov_kd all_server ""

    # FLAb inter-family (Jain et al., 137 entries each)
    run_one flab_thermo_jain all_server ""
    run_one flab_aggregation_jain all_server ""
}

if [ "$MODE" = "intra" ]; then
    run_intra
elif [ "$MODE" = "inter" ]; then
    run_inter
elif [ "$MODE" = "single" ]; then
    DATASET=${2:?  "Usage: run_antibody_full_search.sh single <dataset> <preset> [flags]"}
    PRESET=${3:-dms}
    FLAGS="${@:4}"
    run_one "$DATASET" "$PRESET" "$FLAGS"
elif [ "$MODE" = "all" ]; then
    run_intra
    run_inter
else
    echo "Usage: sbatch run_antibody_full_search.sh [all|intra|inter|single] [search_speed]"
    exit 1
fi

echo ""
echo "============================================================"
echo "ANTIBODY FULL SEARCH COMPLETE"
echo "Results in: $OUTPUT"
echo "Finished: $(date)"
echo "============================================================"
