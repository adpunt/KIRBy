#!/bin/bash
# ARC resubmission — CPU-only jobs.
# Run from ARC login node: bash tests/slurm_scripts/submit_missing_arc.sh [MODE]
#
# All jobs: long partition, NO GPU, 48h max.

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-47:59:00}

    J=$(sbatch \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (long, ${mem}, ${wall}, CPU)"
}

echo "============================================"
echo "ARC Missing Surveys (CPU-only jobs)"
echo "============================================"
echo ""

# ── Reaction: 3 datasets × full reps (no GPU needed) ──
run_reaction() {
    local OUT="tests/results/phase5_baselines/reaction"
    echo "--- Reaction (3 datasets, 64G, 48h) ---"
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "rxn_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            64G
    done
    echo ""
}

# ── Nucleotide NT Revised: 18 datasets (classical reps, no GPU) ──
run_nucleotide_nt() {
    local OUT="tests/results/nucleotide_survey"
    echo "--- Nucleotide NT Revised (18 datasets, 128G, 48h) ---"
    for ds in nt_promoter_all nt_promoter_tata nt_promoter_no_tata \
              nt_enhancers nt_enhancers_types \
              nt_splice_sites_all nt_splice_sites_acceptors nt_splice_sites_donors \
              nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 \
              nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 \
              nt_H3K9ac nt_H3K9me3 nt_H4K20me1; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --skip-existing --output ${OUT}/" \
            128G
    done
    echo ""
}

# ── Nucleotide Genomic Benchmarks: 4 datasets (classical reps, no GPU) ──
run_nucleotide_gb() {
    local OUT="tests/results/nucleotide_survey"
    echo "--- Nucleotide Genomic Benchmarks (4 datasets, 128G, 48h) ---"
    for ds in gb_human_nontata_promoters gb_human_enhancers_cohn \
              gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --skip-existing --output ${OUT}/" \
            128G
    done
    echo ""
}

# ── Nucleotide: all 22 datasets ──
run_nucleotide() {
    run_nucleotide_nt
    run_nucleotide_gb
}

# ── Geometry: 15 datasets (uses cached reps, no GPU) ──
run_geometry() {
    local OUT="tests/results/geometry_survey"
    local BASE_REPS="esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac"
    echo "--- Geometry (15 datasets, 64G, 48h) ---"
    for ds in flab_thermo_tresanco \
                abcov_kd \
              ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ; do
        submit "geo_${ds}" \
            "mkdir -p ${OUT} && python tests/experiments/structural_geometry_survey.py ${ds} --base-reps ${BASE_REPS} --models ridge,rf --output ${OUT}/" \
            64G
    done
    echo ""
}

# ── Forward Compat: ReactionT5 only (CPU, no GPU needed) ──
run_forward_compat() {
    local OUT="tests/results/forward_compat/exp7_reactiont5"
    echo "--- FC Exp 7: ReactionT5 (3 datasets, 64G, 48h) ---"
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "fc_rxnt5_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            64G
    done
    echo ""
}

case "$MODE" in
    reaction)       run_reaction ;;
    nucleotide)     run_nucleotide ;;
    nucleotide_nt)  run_nucleotide_nt ;;
    nucleotide_gb)  run_nucleotide_gb ;;
    geometry)       run_geometry ;;
    forward_compat) run_forward_compat ;;
    all)
        run_reaction
        run_nucleotide
        run_geometry
        run_forward_compat
        ;;
    *)
        echo "Usage: bash submit_missing_arc.sh [MODE]"
        echo "  Run on ARC login node. All jobs: long partition, CPU, 48h."
        echo ""
        echo "  reaction        3 CPU jobs (64G)"
        echo "  nucleotide      22 CPU jobs (128G)"
        echo "  nucleotide_nt   18 NT Revised only (128G)"
        echo "  nucleotide_gb   4 Genomic Benchmarks only (128G)"
        echo "  geometry        15 CPU jobs (64G)"
        echo "  forward_compat  3 CPU jobs (64G, ReactionT5 only)"
        echo "  all             43 CPU jobs"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
