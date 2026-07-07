#!/bin/bash
# Forward Compatibility Experiments (CORE THESIS) — canonical c13 driver
#
# ── DRIVER DECISION (2026-07-07) ──────────────────────────────────────────
# This script now drives the c13 module
#   (tests/experiments/complexity/c13_forward_compatibility.py)
# and NOT the old baseline-survey + separate-hybrid-search pair.
#
# Why: the old driver ran a WEAKER experiment. It scored the new rep alone
# (baseline survey) and, separately, ran a full hybrid search — but it never
# built the "best hybrid WITHOUT the new rep" control, so it could not isolate
# the new rep's contribution. The c13 module runs the full ablation per new
# rep r:
#   A: r + ridge            B: r + best model
#   C: best hybrid WITHOUT r        D: best hybrid WITH r
#   D_random: best hybrid WITH a random-Gaussian matrix shaped like r
# and reports the DEBIASED contribution D - D_random (D - C alone is
# selection-biased positive), plus D - B (does the hybrid improve r itself).
# The c13 module is "the" forward-compat result; present its numbers.
#
# ── ANTIBODY CAVEAT (baked in) ────────────────────────────────────────────
# MINT, xTrimoPGLM and AIDO.Protein are empirically DEAD on antibody
# (NaN / near-zero → removed from default ANTIBODY_REPS). They are NOT passed
# as antibody --new-reps here. Antibody forward-compat leans on SaProt / IgT5 /
# ESM3 only. Those three models keep their DTA arms (they are fine on DTA).
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_forward_compat.sh          # all domains
#   bash tests/slurm_scripts/submit_forward_compat.sh molecular
#   bash tests/slurm_scripts/submit_forward_compat.sh dta
#   bash tests/slurm_scripts/submit_forward_compat.sh antibody
#   bash tests/slurm_scripts/submit_forward_compat.sh reaction
#   bash tests/slurm_scripts/submit_forward_compat.sh tcga
#
# Output: tests/results/forward_compat/c13_<domain>/

set -euo pipefail

MODE=${1:-all}
BASE_OUTPUT="tests/results/forward_compat"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
C13="tests/experiments/complexity/c13_forward_compatibility.py"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# run_c13 <job-name> <dataset> <output-subdir> <new-reps-csv|auto> [extra-flags] [mem] [wall]
#   new-reps-csv = "auto" → omit --new-reps (auto-detect anything not in
#                  LEGACY_REPS). Otherwise pass the explicit --new-reps list
#                  (used for antibody to exclude the dead MINT/xTrimo/AIDO).
run_c13() {
    local name=$1
    local dataset=$2
    local outsub=$3
    local newreps=$4
    local extra=${5:-}
    local mem=${6:-128G}
    local wall=${7:-23:59:00}

    local OUT="${BASE_OUTPUT}/${outsub}"
    local newreps_flag=""
    if [[ "${newreps}" != "auto" ]]; then
        newreps_flag="--new-reps ${newreps}"
    fi

    local cmd="mkdir -p ${OUT} && python ${C13} --dataset ${dataset} --reps all --mode full ${newreps_flag} ${extra} --output ${OUT}/"

    J=$(sbatch \
        --job-name="fc13_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (ds=${dataset}, new=${newreps}, mem=${mem}, time=${wall}, GPU)"
}

echo "============================================"
echo "Forward Compatibility — canonical c13 driver"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo ""

# ── Molecular (home turf; MolE / MiniMol / GP-MoLFormer auto-detected) ─────
run_molecular() {
    echo "--- Molecular: c13 forward-compat (auto-detect new reps) ---"
    for ds in logd caco2 herg adme_hlm; do
        run_c13 "mol_${ds}" "${ds}" "c13_molecular" auto ""
    done
    echo ""
}

# ── DTA (defensible; MINT/SaProt/Boltz-2/ESM3/xTrimo/AIDO all fine here) ────
run_dta() {
    echo "--- DTA: c13 forward-compat (auto-detect new protein reps) ---"
    for ds in davis kiba bindingdb_kd; do
        run_c13 "dta_${ds}" "${ds}" "c13_dta" auto "" 128G 47:59:00
    done
    echo ""
}

# ── Antibody (SaProt / IgT5 / ESM3 ONLY — dead reps excluded) ──────────────
run_antibody() {
    echo "--- Antibody: c13 forward-compat (SaProt/IgT5/ESM3 only) ---"
    # Live new antibody reps. MINT / xTrimoPGLM / AIDO.Protein deliberately absent.
    local AB_NEW="saprot,saprot_cdr,igt5,esm3,esm3_cdr"
    for ds in abcov_kd flab_thermo_tresanco ginkgo_hic flab_expression_koenig \
              flab_binding_koenig ddg_combined; do
        run_c13 "ab_${ds}" "${ds}" "c13_antibody" "${AB_NEW}" "--augment"
    done
    echo ""
}

# ── Reaction (ReactionT5 auto-detected) ────────────────────────────────────
run_reaction() {
    echo "--- Reaction: c13 forward-compat (auto-detect new reps) ---"
    for ds in buchwald_hartwig bh_doyle_plates; do
        run_c13 "rxn_${ds}" "${ds}" "c13_reaction" auto ""
    done
    echo ""
}

# ── TCGA (Geneformer V2 auto-detected; off-topic for the Ab talk) ──────────
run_tcga() {
    echo "--- TCGA: c13 forward-compat (auto-detect new reps) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50; do
        run_c13 "tcga_${ds}" "${ds}" "c13_tcga" auto "" 256G 47:59:00
    done
    echo ""
}

case "$MODE" in
    molecular|mol)   run_molecular ;;
    dta)             run_dta ;;
    antibody|ab)     run_antibody ;;
    reaction|rxn)    run_reaction ;;
    tcga)            run_tcga ;;
    all)
        run_molecular
        run_dta
        run_antibody
        run_reaction
        run_tcga
        echo "============================================"
        echo "All forward-compat c13 jobs submitted."
        echo "  Molecular: 4  DTA: 3  Antibody: 6  Reaction: 2  TCGA: 2"
        echo ""
        echo "Present the DEBIASED effect D - D_random (not D - C)."
        echo "Antibody arm = SaProt/IgT5/ESM3 only (MINT/xTrimo/AIDO dead)."
        echo "============================================"
        ;;
    *)
        echo "Forward Compatibility — canonical c13 driver"
        echo ""
        echo "Usage: bash submit_forward_compat.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all          All domains"
        echo "  molecular    MolE/MiniMol/GP-MoLFormer → molecular"
        echo "  dta          MINT/SaProt/Boltz-2/ESM3/xTrimo/AIDO → DTA"
        echo "  antibody     SaProt/IgT5/ESM3 → antibody (dead reps excluded)"
        echo "  reaction     ReactionT5 → reaction"
        echo "  tcga         Geneformer V2 → TCGA"
        echo ""
        echo "Each job runs the c13 ablation (A/B/C/D/D_random) and reports"
        echo "the DEBIASED contribution D - D_random plus D - B."
        echo ""
        echo "Output: ${BASE_OUTPUT}/c13_<domain>/"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
