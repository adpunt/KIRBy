#!/bin/bash
# One-shot clean resubmit (2026-06-28).
# Cancels ALL current jobs, then:
#   - PRECOMPUTE ONLY for the not-yet-precomputed datasets (no experiments)
#   - FULL experiment suite for the precompute-DONE datasets
# Hardcoded, verified dataset lists -> cannot balloon into a runaway submission.
# This is just an orchestrator around the existing submit_precompute.sh /
# submit_full_suite.sh — it does not reimplement them.
#
# Run:  cd /data/stat-ecr/scat9264/KIRBy && git pull && bash tests/slurm_scripts/resubmit_clean.sh
set -euo pipefail

cd /data/stat-ecr/scat9264/KIRBy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 9 datasets whose precompute is NOT finished -> precompute only.
NOTDONE="bindingdb_kd davis kiba qm9 gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm nt_H3K4me1 nt_splice_sites_donors bh_doyle_plates"

# 60 datasets whose precompute IS done -> full experiment suite.
DONE_DATASETS="abagym_Ang2 abagym_EGFR abagym_VEGF abcov_kd ddg_combined flab_binding_adams flab_binding_koenig flab_binding_phillips_h3 flab_expression_adams flab_expression_koenig flab_immunogenicity_marks flab_thermo_tresanco ginkgo_acsins ginkgo_aggregation ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_purity ginkgo_titer ginkgo_tm2 humanness_therapeutic tz_her2_binary tz_her2_multiclass adme_hlm adme_mdr1 adme_rlm caco2 herg herg_ki logd gb_human_enhancers_cohn gb_human_nontata_promoters nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me2 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_H4K20me1 nt_enhancers nt_enhancers_types nt_promoter_all nt_promoter_no_tata nt_promoter_tata nt_splice_sites_acceptors buchwald_hartwig tcga_brca_pam50 tcga_luad_vs_lusc"

N_NOTDONE=$(echo $NOTDONE | wc -w)
N_DONE=$(echo $DONE_DATASETS | wc -w)

# Blowup guard: refuse to run if the lists are not the expected size.
if [[ "$N_NOTDONE" -ne 9 || "$N_DONE" -ne 60 ]]; then
    echo "ABORT: list sizes wrong (notdone=$N_NOTDONE expected 9, done=$N_DONE expected 60). Nothing submitted." >&2
    exit 1
fi

echo "=================================================================="
echo "Clean resubmit: cancel all, then $N_NOTDONE precompute-only + $N_DONE experiment jobs."
echo "=================================================================="

echo ">>> Cancelling all current jobs for $USER ..."
scancel -u "$USER" || true
sleep 5
echo "    queue after cancel: $(squeue -u "$USER" -h | wc -l) jobs"

echo ">>> Submitting PRECOMPUTE-ONLY for $N_NOTDONE not-done datasets ..."
bash "$SCRIPT_DIR/submit_precompute.sh" --datasets "$NOTDONE"

echo ">>> Submitting FULL SUITE for $N_DONE done datasets ..."
bash "$SCRIPT_DIR/submit_full_suite.sh" $DONE_DATASETS

echo "=================================================================="
echo "Done. Current queue size: $(squeue -u "$USER" -h | wc -l) jobs"
echo "(expect ~70-90 total; if it's hundreds, something is wrong — scancel -u \$USER)"
echo "=================================================================="
