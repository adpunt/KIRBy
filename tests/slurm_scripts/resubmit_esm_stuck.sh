#!/bin/bash
# Cancel and resubmit all jobs that are stuck on ESM-C / model downloads.
# Run AFTER precache_models.sh completes.
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
echo "Current jobs that may be stuck on model downloads:"
squeue -u $USER -o "%.10i %.30j %.8T %.10M %.6D %R" | grep -E "p5_(ab_|dta_|mol_)|opig_(clf|ginkgo|ddg)|cmp_(ab_region|dta_e2e|dta_ft)" || echo "(none found)"
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

for pat in p5_dta_davis p5_dta_kiba p5_dta_bindingdb p5_dta_universal \
           p5_ab_dms p5_ab_inter p5_ab_lm_dms p5_ab_lm_inter \
           p5_ab_mut_dms p5_ab_mut_inter p5_ab_fast_dms p5_ab_fast_inter \
           p5_mol_baselines \
           opig_clf opig_clf_lm opig_clf_fast opig_clf_mut opig_clf_var opig_clf_struct \
           opig_ginkgo opig_ginkgo_lm opig_ginkgo_fast opig_ginkgo_mut opig_ginkgo_var opig_ginkgo_struct \
           opig_ddg opig_ddg_lm opig_ddg_fast opig_ddg_mut opig_ddg_var opig_ddg_struct \
           p5_cmp_ab_region p5_cmp_dta_e2e_davis p5_cmp_dta_e2e_kiba \
           p5_cmp_dta_ft_davis p5_cmp_dta_ft_kiba; do
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
echo "Step 3: Resubmit using simplified scripts"
echo "============================================"

echo ""
echo "--- Phase 5: DTA + antibody + molecular ---"
bash tests/slurm_scripts/submit_phase5_htc.sh dta
bash tests/slurm_scripts/submit_phase5_htc.sh antibody
bash tests/slurm_scripts/submit_phase5_htc.sh molecular

echo ""
echo "--- OPIG: clf + ginkgo + ddg ---"
bash tests/slurm_scripts/submit_opig_surveys.sh clf
bash tests/slurm_scripts/submit_opig_surveys.sh ginkgo
bash tests/slurm_scripts/submit_opig_surveys.sh ddg

echo ""
echo "--- Comparisons using ESM-C ---"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"
OUT_CMP="tests/results/comparisons"

for job_args in \
    "p5_cmp_ab_region:mkdir -p ${OUT_CMP} && python tests/experiments/antibody_region_ablation.py --n-folds 5 --lms esm2,esmc,ablang2 --output ${OUT_CMP}" \
    "p5_cmp_dta_e2e_davis:mkdir -p ${OUT_CMP} && python tests/experiments/dta_e2e_comparison.py davis --output ${OUT_CMP}" \
    "p5_cmp_dta_e2e_kiba:mkdir -p ${OUT_CMP} && python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000 --output ${OUT_CMP}" \
    "p5_cmp_dta_ft_davis:mkdir -p ${OUT_CMP} && python tests/experiments/dta_finetune_comparison.py davis --n-folds 5 --output ${OUT_CMP}" \
    "p5_cmp_dta_ft_kiba:mkdir -p ${OUT_CMP} && python tests/experiments/dta_finetune_comparison.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT_CMP}"; do
    name="${job_args%%:*}"
    cmd="${job_args#*:}"
    J=$(sbatch \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem=256G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (256G, 48h, GPU)"
done

echo ""
echo "============================================"
echo "Resubmission complete."
echo "Rep caching means completed reps are skipped."
echo "============================================"
