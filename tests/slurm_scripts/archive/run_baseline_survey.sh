#!/bin/bash
#SBATCH --job-name=kirby_survey
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# =============================================================================
# BASELINE SURVEY: Comprehensive state of all domains with ALL representations
# =============================================================================
#
# Uses maximal presets for each domain so every available rep is tested.
# Adaptive pruning and rep subset testing enabled to understand each rep's
# contribution without exhaustive runtime.
#
# Fine-tuned reps are NOT included — all reps are pretrained/frozen.
#
# Usage:
#   sbatch run_baseline_survey.sh              # everything, medium search
#   sbatch run_baseline_survey.sh molecular    # molecular only, medium
#   sbatch run_baseline_survey.sh molecular fast  # molecular, fast search
#   sbatch run_baseline_survey.sh antibody     # antibody only (intra + inter)
#   sbatch run_baseline_survey.sh ab_intra     # antibody intra-family only
#   sbatch run_baseline_survey.sh ab_inter     # antibody inter-family only
#   sbatch run_baseline_survey.sh tcga         # TCGA only
#   sbatch run_baseline_survey.sh dta          # drug-target only

DOMAIN=${1:-all}
SEARCH=${2:-medium}
OUTPUT="tests/results/baseline_survey"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT"

echo "============================================================"
echo "KIRBy Baseline Survey (COMPREHENSIVE)"
echo "  Domain:  $DOMAIN"
echo "  Search:  $SEARCH"
echo "  Flags:   --adaptive-pruning"
echo "  Output:  $OUTPUT"
echo "  Started: $(date)"
echo "============================================================"

# Track successes and failures
PASS=0
FAIL=0

run() {
    local script=$1
    local dataset=$2
    local preset=$3
    local tag=$4
    shift 4
    local extra="$@"

    echo ""
    echo "--- $tag ---"
    echo "  Script:  $script"
    echo "  Dataset: $dataset"
    echo "  Preset:  $preset"
    echo "  Extra:   ${extra:-none}"
    echo "  Started: $(date)"

    if python "$script" "$dataset" \
        --reps "$preset" --search "$SEARCH" \
        --adaptive-pruning \
        $extra \
        --output "${OUTPUT}/${tag}.json" 2>&1; then
        echo "  PASSED: $(date)"
        PASS=$((PASS + 1))
    else
        echo "  FAILED (exit $?): $(date)"
        FAIL=$((FAIL + 1))
    fi
}

# =============================================================================
# MOLECULAR: 4 datasets × 'all' preset (15 reps) = 4 runs
# =============================================================================
run_molecular() {
    echo ""
    echo "========== MOLECULAR (all preset, 15 reps) =========="

    for dataset in logd caco2 herg qm9; do
        run tests/find_best_hybrid_molecular.py \
            "$dataset" all "mol_${dataset}_all" --augment
    done
}

# =============================================================================
# ANTIBODY INTRA-FAMILY (DMS): 7 datasets × 'dms_full' preset (18 reps)
# Variable-position reps incl. CDR-filtered variants + all LM scores.
# =============================================================================
run_antibody_intra() {
    echo ""
    echo "========== ANTIBODY INTRA-FAMILY (dms_full preset, 18 reps) =========="

    for dataset in \
        flab_expression_koenig \
        flab_binding_koenig \
        abagym_Ang2_2017_G6 \
        abagym_EGFR_2013_Cetuximab \
        abagym_lysozyme_2019_D441 \
        abagym_VEGF_2017b_G6 \
        abagym_HER2_2021_trastuzumab; do

        run tests/find_best_hybrid_antibody.py \
            "$dataset" dms_full "ab_dms_${dataset}" --augment
    done
}

# =============================================================================
# ANTIBODY INTER-FAMILY: 5 datasets × 'all_server' preset (14 reps)
# Full-sequence LM embeddings + structural + hand-crafted.
# Dropped: sabdab_affinity (was SKEMPI-only, replaced by ddg_combined),
#          abbind/abbind_spm (strict subsets of ddg_combined).
# =============================================================================
run_antibody_inter() {
    echo ""
    echo "========== ANTIBODY INTER-FAMILY (all_server preset, 14 reps) =========="

    for dataset in \
        ddg_combined \
        abagym_all \
        abcov_kd \
        flab_thermo_jain \
        flab_aggregation_jain; do

        run tests/find_best_hybrid_antibody.py \
            "$dataset" all_server "ab_inter_${dataset}" --augment
    done
}

# =============================================================================
# TCGA: 3 datasets × 'all' preset (27 reps) = 3 runs
# All genomic + WSI reps. Needs 96G+ memory.
# =============================================================================
run_tcga() {
    echo ""
    echo "========== TCGA (all preset, 27 reps) =========="

    for dataset in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        run tests/find_best_hybrid_tcga.py \
            "$dataset" all "tcga_${dataset}_all"
    done
}

# =============================================================================
# DTA: 2 datasets × 'full' preset (10 reps) = 2 runs
# All drug + protein reps. Needs 96G+ memory.
# =============================================================================
run_dta() {
    echo ""
    echo "========== DRUG-TARGET AFFINITY (full preset, 10 reps) =========="

    for dataset in davis kiba; do
        run tests/find_best_hybrid_dta.py \
            "$dataset" full "dta_${dataset}_full"
    done
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$DOMAIN" in
    molecular)  run_molecular ;;
    antibody)   run_antibody_intra; run_antibody_inter ;;
    ab_intra)   run_antibody_intra ;;
    ab_inter)   run_antibody_inter ;;
    tcga)       run_tcga ;;
    dta)        run_dta ;;
    all)
        run_molecular
        run_antibody_intra
        run_antibody_inter
        run_tcga
        run_dta
        ;;
    *)
        echo "Usage: sbatch run_baseline_survey.sh [all|molecular|antibody|ab_intra|ab_inter|tcga|dta]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "BASELINE SURVEY COMPLETE"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Results: $OUTPUT"
echo "  Finished: $(date)"
echo "============================================================"
