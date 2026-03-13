#!/bin/bash
# Phase 5C: Forward Compatibility Experiments (CORE THESIS)
#
# NOTE: If you already ran submit_phase5_baselines.sh, the DTA baseline
# jobs here are redundant (they run the same dta_baseline_survey.py).
# You can skip them and just reference phase5_baselines/dta/ results.
# The antibody and molecular baseline jobs here are NOT redundant because
# they target specific reps only.
#
# Runs the 5 thesis experiments from ISSUES.md Phase 4C. Each experiment:
#   1. Takes a model released AFTER KIRBy was built
#   2. Extracts embeddings (zero code changes needed)
#   3. Runs baseline survey → individual rep score
#   4. Runs hybrid search → finds which existing reps augment the new model
#   5. Result: new model alone < new model + KIRBy augmentation
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_forward_compat.sh              # all experiments
#   bash tests/slurm_scripts/submit_forward_compat.sh exp1         # MINT only
#   bash tests/slurm_scripts/submit_forward_compat.sh exp2         # SaProt only
#   bash tests/slurm_scripts/submit_forward_compat.sh exp3         # MolE/MiniMol
#   bash tests/slurm_scripts/submit_forward_compat.sh exp4         # IgT5
#   bash tests/slurm_scripts/submit_forward_compat.sh exp5         # Boltz-2
#
# Output: tests/results/forward_compat/<experiment>/

set -euo pipefail

MODE=${1:-all}
BASE_OUTPUT="tests/results/forward_compat"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-23:59:00}

    J=$(sbatch \
        --job-name="fc_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}, GPU)"
}

echo "============================================"
echo "Phase 5C: Forward Compatibility Experiments"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo ""

# ---------------------------------------------------------------
# Experiment 1: MINT (Jan 2026, PPI-aware, 1280D, built on ESM-2)
#   Domains: DTA (protein_mint) + Antibody (mint)
#   Thesis claim: MINT alone < MINT + KIRBy hybrid
# ---------------------------------------------------------------
run_exp1() {
    local OUT="${BASE_OUTPUT}/exp1_mint"
    echo "--- Exp 1: MINT → DTA + Antibody ---"

    # DTA: Full baseline (includes protein_mint) + hybrid search on Davis
    # NOTE: dta_baseline_survey.py runs ALL reps, no --reps filter
    submit mint_davis_baseline \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis_mint_baseline.json" \
        128G 47:59:00
    submit mint_davis_hybrid \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py davis --reps full --search fast --n-folds 5 --output ${OUT}/davis_mint_hybrid.json" \
        128G 47:59:00

    # Antibody: MINT on key DMS + inter-family datasets
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "mint_ab_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps mint" \
            128G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# Experiment 2: SaProt (ICLR 2024, structure-aware, 1280D)
#   Domains: DTA (protein_saprot) + Antibody (saprot, saprot_cdr)
#   Thesis claim: SaProt alone < SaProt + KIRBy hybrid
# ---------------------------------------------------------------
run_exp2() {
    local OUT="${BASE_OUTPUT}/exp2_saprot"
    echo "--- Exp 2: SaProt → DTA + Antibody ---"

    # DTA: Full baseline (includes protein_saprot) + pick out saprot from results
    submit saprot_davis_baseline \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis_saprot_baseline.json" \
        128G 47:59:00

    # Antibody: SaProt + CDR on key datasets
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "saprot_ab_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps saprot,saprot_cdr" \
            128G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# Experiment 3: MolE / MiniMol / GP-MoLFormer (new molecular FMs)
#   Domain: Molecular
#   Thesis claim: new FM alone < new FM + KIRBy hybrid
# ---------------------------------------------------------------
run_exp3() {
    local OUT="${BASE_OUTPUT}/exp3_molecular_fms"
    echo "--- Exp 3: MolE / MiniMol / GP-MoLFormer → Molecular ---"

    # Baseline survey on key datasets with forward-compat reps
    for ds in logd caco2 herg adme_hlm; do
        submit "molfm_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/" \
            128G 23:59:00
    done

    # Hybrid search on same datasets (all reps includes Phase 4)
    for ds in logd caco2 herg adme_hlm; do
        submit "molfm_hybrid_${ds}" \
            "mkdir -p ${OUT} && python tests/find_best_hybrid_molecular.py ${ds} --reps all --search fast --n-folds 5 --output ${OUT}/${ds}_hybrid.json" \
            128G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# Experiment 4: IgT5 / AbRoBERTa (newer antibody LMs)
#   Domain: Antibody
#   Thesis claim: new AbLM alone < new AbLM + KIRBy hybrid
# ---------------------------------------------------------------
run_exp4() {
    local OUT="${BASE_OUTPUT}/exp4_ablm"
    echo "--- Exp 4: IgT5 / Ab-RoBERTa → Antibody ---"

    for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 \
              ddg_combined flab_thermo_tresanco abcov_kd; do
        submit "ablm_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps igt5,abroberta,abroberta_cdr" \
            128G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# Experiment 5: Boltz-2 (cross-domain structure embeddings, 384D)
#   Domain: DTA (protein_boltz2)
#   Thesis claim: Boltz-2 alone < Boltz-2 + KIRBy hybrid
# ---------------------------------------------------------------
run_exp5() {
    local OUT="${BASE_OUTPUT}/exp5_boltz2"
    echo "--- Exp 5: Boltz-2 → DTA ---"

    # DTA: Full baseline (includes protein_boltz2) + hybrid search
    submit boltz2_davis_baseline \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis_boltz2_baseline.json" \
        128G 47:59:00
    submit boltz2_davis_hybrid \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py davis --reps full --search fast --n-folds 5 --output ${OUT}/davis_boltz2_hybrid.json" \
        128G 47:59:00
    echo ""
}

# ---------------------------------------------------------------
# Experiment extras: ESM3 (multimodal, 1536D)
#   Domain: Antibody + DTA
# ---------------------------------------------------------------
run_esm3() {
    local OUT="${BASE_OUTPUT}/exp_esm3"
    echo "--- ESM3 → Antibody + DTA ---"

    # Antibody
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "esm3_ab_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps esm3,esm3_cdr" \
            128G 23:59:00
    done

    # DTA
    # DTA: Full baseline (includes protein_esm3)
    submit esm3_davis_baseline \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis_esm3_baseline.json" \
        128G 47:59:00
    echo ""
}

# ---------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------

case "$MODE" in
    exp1|mint)
        run_exp1
        ;;
    exp2|saprot)
        run_exp2
        ;;
    exp3|molecular|mole|minimol)
        run_exp3
        ;;
    exp4|igt5|ablm)
        run_exp4
        ;;
    exp5|boltz2)
        run_exp5
        ;;
    esm3)
        run_esm3
        ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        run_exp4
        run_exp5
        run_esm3
        echo "============================================"
        echo "Total: ~35 forward compatibility jobs"
        echo ""
        echo "Experiment 1 (MINT):         6 jobs (DTA + antibody)"
        echo "Experiment 2 (SaProt):       5 jobs (DTA + antibody)"
        echo "Experiment 3 (MolE/MiniMol): 8 jobs (molecular)"
        echo "Experiment 4 (IgT5/AbRoBERTa): 6 jobs (antibody)"
        echo "Experiment 5 (Boltz-2):      2 jobs (DTA)"
        echo "ESM3 (extra):                4 jobs (antibody + DTA)"
        echo "============================================"
        ;;
    *)
        echo "Phase 5C: Forward Compatibility Experiments"
        echo ""
        echo "Usage: bash submit_forward_compat.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all       All 5 experiments + ESM3 (~35 jobs)"
        echo "  exp1      MINT → DTA + antibody"
        echo "  exp2      SaProt → DTA + antibody"
        echo "  exp3      MolE/MiniMol/GP-MoLFormer → molecular"
        echo "  exp4      IgT5/Ab-RoBERTa → antibody"
        echo "  exp5      Boltz-2 → DTA"
        echo "  esm3      ESM3 → antibody + DTA"
        echo ""
        echo "Output: ${BASE_OUTPUT}/<experiment>/"
        echo ""
        echo "Each experiment:"
        echo "  1. Baseline: new model scored individually"
        echo "  2. Hybrid: KIRBy searches for augmentation partners"
        echo "  3. Result: proves new model + KIRBy > new model alone"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
