#!/bin/bash
# Submit the DONE datasets (precompute already built), broken up: one SLURM job
# per dataset per category — complexity, similarity — INDEPENDENT, NO
# dependencies. Precompute already exists, so the categories run in parallel with
# no rebuild race.
#
# noise is HELD — it is being reworked for large-n domains. Do NOT resubmit noise
# here until that rework lands. (If a done dataset already has a noise job running,
# leave it be; this script simply does not START new noise.)
#
# Companion to submit_notdone.sh (which builds precompute for the not-done
# datasets via complexity, similarity waiting). This script does NOT touch those.
#
# DRY RUN by default — prints the sbatch commands and submits NOTHING.
# To actually submit:   bash tests/slurm_scripts/submit_done.sh --go
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/submit_dataset.sh"
# Bill compute to stat-cadd for its much higher fair-share (priority ~3x); the
# cache/data stay on /data/stat-ecr (account != filesystem). Override: ACCOUNT=..
ACCOUNT="${ACCOUNT:-stat-cadd}"

# 42 datasets whose precompute IS done (distance-inclusive --done basis).
DATASETS=(
  abagym_Ang2 abagym_EGFR abagym_HER2 abagym_VEGF abagym_lysozyme
  abcov_kd adme_mdr1 buchwald_hartwig caco2 ddg_combined
  flab_aggregation_jain flab_aggregation_jain_hicrt
  flab_expression_adams flab_expression_koenig flab_immunogenicity_marks
  flab_pk_jain_fcrnrt flab_pk_jain_heprt
  flab_polyreact_jain_bvp flab_polyreact_jain_cic
  flab_thermo_jain flab_thermo_tresanco
  gb_human_enhancers_cohn
  ginkgo_acsins ginkgo_aggregation ginkgo_colloidal ginkgo_heparin ginkgo_hic
  ginkgo_pr_cho ginkgo_purity ginkgo_titer ginkgo_tm2
  herg herg_ki humanness_therapeutic
  nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_promoter_tata
  tcga_brca_pam50 tcga_luad_vs_lusc tz_her2_binary tz_her2_multiclass
)

# Blowup guard: refuse to run unless the list is exactly the expected size.
N=${#DATASETS[@]}
if [[ "$N" -ne 42 ]]; then
  echo "ABORT: expected 42 datasets, got $N. Nothing submitted." >&2
  exit 1
fi

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

echo "=================================================================="
echo "DONE datasets: $N   |   jobs: $((N*2))   |   no dependencies (parallel)   |   noise: HELD   |   account: $ACCOUNT"
echo "Mode: $([[ $GO == 1 ]] && echo '*** SUBMIT ***' || echo 'DRY RUN (nothing submitted)')"
echo "=================================================================="
echo

for ds in "${DATASETS[@]}"; do
  if [[ $GO == 1 ]]; then
    for cat in complexity similarity; do
      JID=$(sbatch --parsable --account="$ACCOUNT" "$WORKER" "$cat" "$ds")
      echo "[$ds] $cat -> job $JID"
    done
    echo
  else
    echo "# $ds"
    echo "sbatch --account=$ACCOUNT $WORKER complexity $ds"
    echo "sbatch --account=$ACCOUNT $WORKER similarity $ds"
    echo
  fi
done

if [[ $GO == 0 ]]; then
  echo "=================================================================="
  echo "DRY RUN — nothing submitted. Re-run with --go to submit all $((N*2)) jobs:"
  echo "    bash tests/slurm_scripts/submit_done.sh --go"
  echo "=================================================================="
fi
