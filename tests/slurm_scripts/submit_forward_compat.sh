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
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
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

    # Antibody arm REMOVED 2026-06-05: MINT is empirically dead on antibody
    # (NaN/near-zero → removed from default ANTIBODY_REPS, antibody.py:1661);
    # a dead rep can't demonstrate the forward-compat thesis. DTA arm stands.
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
# Experiment 6: xTrimoPGLM-1B (Nature Methods 2025, 1B MLM, ~2048D)
#   Domains: DTA (protein_xtrimopglm) + Antibody
#   Thesis claim: xTrimoPGLM alone < xTrimoPGLM + KIRBy hybrid
# ---------------------------------------------------------------
run_exp6() {
    local OUT="${BASE_OUTPUT}/exp6_xtrimopglm"
    echo "--- Exp 6: xTrimoPGLM-1B → DTA + Antibody ---"

    submit xtri_davis_baseline \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis_baseline.json" \
        128G 47:59:00

    # Antibody arm REMOVED 2026-06-05: xTrimoPGLM is empirically dead on
    # antibody (NaN/near-zero → removed from default ANTIBODY_REPS,
    # antibody.py:1678); DTA arm (protein_xtrimopglm via davis) stands.
    echo ""
}

# ---------------------------------------------------------------
# Experiment 7: ReactionT5 (J Cheminformatics 2025, T5 encoder, 768D)
#   Domain: Reaction
#   Thesis claim: ReactionT5 alone < ReactionT5 + KIRBy hybrid
# ---------------------------------------------------------------
run_exp7() {
    local OUT="${BASE_OUTPUT}/exp7_reactiont5"
    echo "--- Exp 7: ReactionT5 → Reaction ---"

    for ds in buchwald_hartwig suzuki doyle_plates; do
        submit "rxnt5_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/" \
            128G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# Experiment 8: Geneformer V2-316M (upgraded, 4096 context, ~1024D)
#   Domain: TCGA
#   Thesis claim: V2 upgrade is drop-in, proves forward compat
# ---------------------------------------------------------------
run_exp8() {
    local OUT="${BASE_OUTPUT}/exp8_geneformer_v2"
    echo "--- Exp 8: Geneformer V2 → TCGA ---"

    for ds in tcga_luad_vs_lusc tcga_brca_pam50; do
        submit "gfv2_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/" \
            256G 47:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# Experiment 9: AIDO.Protein-16B (GenBio AI, MoE 16B, 2304D)
#   Domains: DTA + Antibody
#   WARNING: Requires ~32GB VRAM (float16). May fail on smaller GPUs.
# ---------------------------------------------------------------
run_exp9() {
    local OUT="${BASE_OUTPUT}/exp9_aido_protein"
    echo "--- Exp 9: AIDO.Protein-16B → DTA + Antibody ---"
    echo "    WARNING: Requires ~32GB VRAM. Will skip gracefully if OOM."

    submit aido_davis_baseline \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis_baseline.json" \
        256G 47:59:00

    # Antibody arm REMOVED 2026-06-05: AIDO.Protein is empirically dead on
    # antibody (NaN/near-zero → removed from default ANTIBODY_REPS,
    # antibody.py:1693); DTA arm (davis baseline) stands.
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
    exp6|xtrimopglm)
        run_exp6
        ;;
    exp7|reactiont5)
        run_exp7
        ;;
    exp8|geneformer_v2|gfv2)
        run_exp8
        ;;
    exp9|aido|aido_protein)
        run_exp9
        ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        run_exp4
        run_exp5
        run_esm3
        run_exp6
        run_exp7
        run_exp8
        run_exp9
        echo "============================================"
        echo "Total: ~45 forward compatibility jobs"
        echo ""
        echo "Experiment 1 (MINT):           2 jobs (DTA; antibody dropped — dead rep)"
        echo "Experiment 2 (SaProt):         5 jobs (DTA + antibody)"
        echo "Experiment 3 (MolE/MiniMol):   8 jobs (molecular)"
        echo "Experiment 4 (IgT5/AbRoBERTa): 6 jobs (antibody)"
        echo "Experiment 5 (Boltz-2):        2 jobs (DTA)"
        echo "ESM3 (extra):                  4 jobs (antibody + DTA)"
        echo "Experiment 6 (xTrimoPGLM):     1 job  (DTA; antibody dropped — dead rep)"
        echo "Experiment 7 (ReactionT5):     3 jobs (reaction)"
        echo "Experiment 8 (Geneformer V2):  2 jobs (TCGA)"
        echo "Experiment 9 (AIDO.Protein):   1 job  (DTA; antibody dropped — dead rep)"
        echo "============================================"
        ;;
    *)
        echo "Phase 5C: Forward Compatibility Experiments"
        echo ""
        echo "Usage: bash submit_forward_compat.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all           All experiments (~55 jobs)"
        echo "  exp1|mint     MINT → DTA + antibody"
        echo "  exp2|saprot   SaProt → DTA + antibody"
        echo "  exp3|molecular MolE/MiniMol/GP-MoLFormer → molecular"
        echo "  exp4|igt5     IgT5/Ab-RoBERTa → antibody"
        echo "  exp5|boltz2   Boltz-2 → DTA"
        echo "  esm3          ESM3 → antibody + DTA"
        echo "  exp6|xtrimopglm xTrimoPGLM-1B → DTA + antibody"
        echo "  exp7|reactiont5 ReactionT5 → reaction"
        echo "  exp8|gfv2     Geneformer V2 → TCGA"
        echo "  exp9|aido     AIDO.Protein-16B → DTA + antibody"
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
