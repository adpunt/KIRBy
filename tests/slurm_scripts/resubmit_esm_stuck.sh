#!/bin/bash
# Cancel and resubmit all jobs that are stuck on ESM-C / model downloads.
# Run AFTER precache_models.sh completes.
#
# Jobs affected by ESM-C stalls:
#   - DTA baselines (protein_esmc in PROTEIN_REPS)
#   - Antibody LM (esmc, esmc_cdr in LM rep list)
#   - Antibody mutation (esmc_llr in mutation rep list)
#   - OPIG clf_lm, ginkgo_lm, ddg_lm (esmc in ALL_LM)
#   - OPIG ddg_mut (esmc_llr in ALL_MUTATION)
#   - Comparison: antibody region ablation (uses esmc)
#   - Comparison: DTA E2E, DTA fine-tune (load protein models)
#
# Rep caching means completed reps are saved as .npy and will be
# skipped on resubmission — no work is lost.
#
# Usage:
#   bash tests/slurm_scripts/resubmit_esm_stuck.sh

set -euo pipefail

echo "============================================"
echo "Step 1: Identify stuck jobs"
echo "============================================"
echo ""
echo "Current jobs with ESM-dependent reps:"
squeue -u $USER -o "%.10i %.30j %.8T %.10M %.6D %R" | grep -E "p5_(ab_lm|ab_mut|dta_|mol_)|opig_(clf_lm|ginkgo_lm|ddg_lm|ddg_mut)|cmp_(ab_region|dta_e2e|dta_ft)" || echo "(none found)"
echo ""

read -p "Cancel ALL of the above and resubmit? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "============================================"
echo "Step 2: Cancel stuck jobs"
echo "============================================"

# Cancel Phase 5 DTA jobs
for pat in p5_dta_davis p5_dta_kiba p5_dta_bindingdb p5_dta_universal; do
    jid=$(squeue -u $USER -o "%.10i %.30j" -h | grep "$pat" | awk '{print $1}' | tr -d ' ')
    if [ -n "$jid" ]; then
        echo "  Cancelling $pat ($jid)"
        scancel $jid
    fi
done

# Cancel Phase 5 antibody LM + mutation jobs (these have esmc/esmc_llr)
for pat in p5_ab_lm_dms p5_ab_lm_inter p5_ab_mut_dms p5_ab_mut_inter; do
    jid=$(squeue -u $USER -o "%.10i %.30j" -h | grep "$pat" | awk '{print $1}' | tr -d ' ')
    if [ -n "$jid" ]; then
        echo "  Cancelling $pat ($jid)"
        scancel $jid
    fi
done

# Cancel Phase 5 molecular (has LM reps that download models)
for pat in p5_mol_baselines; do
    jid=$(squeue -u $USER -o "%.10i %.30j" -h | grep "$pat" | awk '{print $1}' | tr -d ' ')
    if [ -n "$jid" ]; then
        echo "  Cancelling $pat ($jid)"
        scancel $jid
    fi
done

# Cancel OPIG LM + mutation jobs
for pat in opig_clf_lm opig_ginkgo_lm opig_ddg_lm opig_ddg_mut; do
    jid=$(squeue -u $USER -o "%.10i %.30j" -h | grep "$pat" | awk '{print $1}' | tr -d ' ')
    if [ -n "$jid" ]; then
        echo "  Cancelling $pat ($jid)"
        scancel $jid
    fi
done

# Cancel comparison jobs that use ESM-C
for pat in p5_cmp_ab_region p5_cmp_dta_e2e p5_cmp_dta_ft; do
    jid=$(squeue -u $USER -o "%.10i %.30j" -h | grep "$pat" | awk '{print $1}' | tr -d ' ')
    if [ -n "$jid" ]; then
        echo "  Cancelling $pat ($jid)"
        scancel $jid
    fi
done

echo ""
echo "Waiting 5s for cancellations to propagate..."
sleep 5

echo ""
echo "============================================"
echo "Step 3: Resubmit"
echo "============================================"

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-47:59:00}
    local gpu=${5:-gpu}

    local gpu_flag=""
    if [ "$gpu" = "gpu" ]; then
        gpu_flag="--gres=gpu:1"
    fi

    J=$(sbatch \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        ${gpu_flag} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu_flag:+, GPU})"
}

OUT_P5="tests/results/phase5_baselines"
OUT_OPIG="tests/results/opig"
OUT_CMP="tests/results/comparisons"

ALL_LM="esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr"
ALL_MUTATION="mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr"

echo ""
echo "--- Phase 5: DTA baselines (4 jobs) ---"
submit p5_dta_davis \
    "mkdir -p ${OUT_P5}/dta && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT_P5}/dta/davis.json"
submit p5_dta_kiba \
    "mkdir -p ${OUT_P5}/dta && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT_P5}/dta/kiba.json"
submit p5_dta_bindingdb \
    "mkdir -p ${OUT_P5}/dta && python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT_P5}/dta/bindingdb_kd.json"
submit p5_dta_universal \
    "mkdir -p ${OUT_P5}/dta_universal && python tests/run_baseline_survey.py dta --n-folds 5 --augment --output ${OUT_P5}/dta_universal/" \
    256G

echo ""
echo "--- Phase 5: Antibody LM + mutation (4 jobs) ---"
submit p5_ab_lm_dms \
    "mkdir -p ${OUT_P5}/antibody && python tests/antibody_baseline_survey.py all_dms --output ${OUT_P5}/antibody/ --n-folds 5 --augment --reps ${ALL_LM}"
submit p5_ab_lm_inter \
    "mkdir -p ${OUT_P5}/antibody && python tests/antibody_baseline_survey.py all_inter --output ${OUT_P5}/antibody/ --n-folds 5 --augment --reps ${ALL_LM}"
submit p5_ab_mut_dms \
    "mkdir -p ${OUT_P5}/antibody && python tests/antibody_baseline_survey.py all_dms --output ${OUT_P5}/antibody/ --n-folds 5 --augment --reps ${ALL_MUTATION}"
submit p5_ab_mut_inter \
    "mkdir -p ${OUT_P5}/antibody && python tests/antibody_baseline_survey.py all_inter --output ${OUT_P5}/antibody/ --n-folds 5 --augment --reps ${ALL_MUTATION}"

echo ""
echo "--- Phase 5: Molecular (1 job) ---"
submit p5_mol_baselines \
    "mkdir -p ${OUT_P5}/molecular && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --output ${OUT_P5}/molecular/"

echo ""
echo "--- OPIG: LM + mutation jobs (4 jobs) ---"
submit opig_clf_lm \
    "mkdir -p ${OUT_OPIG}/clf && python tests/antibody_baseline_survey.py all_clf --output ${OUT_OPIG}/clf/ --n-folds 5 --augment --reps ${ALL_LM}"
submit opig_ginkgo_lm \
    "mkdir -p ${OUT_OPIG}/ginkgo && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT_OPIG}/ginkgo/ --n-folds 5 --augment --reps ${ALL_LM}"
submit opig_ddg_lm \
    "mkdir -p ${OUT_OPIG}/ddg && python tests/antibody_baseline_survey.py all_ddg --output ${OUT_OPIG}/ddg/ --n-folds 5 --augment --reps ${ALL_LM}"
submit opig_ddg_mut \
    "mkdir -p ${OUT_OPIG}/ddg && python tests/antibody_baseline_survey.py all_ddg --output ${OUT_OPIG}/ddg/ --n-folds 5 --augment --reps ${ALL_MUTATION}"

echo ""
echo "--- Comparisons using ESM-C (up to 5 jobs) ---"
submit p5_cmp_ab_region \
    "mkdir -p ${OUT_CMP} && python tests/experiments/antibody_region_ablation.py --n-folds 5 --lms esm2,esmc,ablang2 --output ${OUT_CMP}"
submit p5_cmp_dta_e2e_davis \
    "mkdir -p ${OUT_CMP} && python tests/experiments/dta_e2e_comparison.py davis --output ${OUT_CMP}"
submit p5_cmp_dta_e2e_kiba \
    "mkdir -p ${OUT_CMP} && python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000 --output ${OUT_CMP}"
submit p5_cmp_dta_ft_davis \
    "mkdir -p ${OUT_CMP} && python tests/experiments/dta_finetune_comparison.py davis --n-folds 5 --output ${OUT_CMP}"
submit p5_cmp_dta_ft_kiba \
    "mkdir -p ${OUT_CMP} && python tests/experiments/dta_finetune_comparison.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT_CMP}"

echo ""
echo "============================================"
echo "Total: 18 resubmitted jobs"
echo "Rep caching means completed reps are skipped."
echo "============================================"
