#!/bin/bash
# PRIORITY SET (2026-07-07): the ONLY datasets in scope for the near-term
# presentation. Everything else is cancelled. Precompute is done for these, so
# categories run as INDEPENDENT PARALLEL jobs — no complexity-builds-first
# dependency. (A few similarity runs may cold-build `*_distance`; acknowledged,
# re-run later if they hit the wall.)
#
# Forward-compatibility is NOT a separate category — it's experiment c13, which
# is already in complexity's default set. Running complexity runs forward-compat.
#
# Federated runs ONLY for the 3 datasets with dedicated silo loaders
# (caco2/herg/logd); RUN_FEDERATED=0 skips it.
#
# Noise is HELD (rework in progress). When ready, fire it with the SAME script;
# it runs across all 33 like every other category:
#     CATS=noise RUN_FEDERATED=0 bash tests/slurm_scripts/submit_priority.sh --go
#
# Optimal-scheduling knobs (see where_to_submit.sh):
#   ACCOUNT  stat-ecr  (live fair-share ~2x stat-cadd, 2026-07-07)
#   TIME     1-00:00:00 (24h) — short wall BACKFILLS into idle slots, starts now;
#            you have no more time and can re-run later, so nothing is lost.
#   PARTITION long (medium is equivalent — same nodes; a 24h wall fits both)
#
# DRY RUN by default. Submit with --go.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/submit_dataset.sh"
ACCOUNT="${ACCOUNT:-stat-ecr}"
TIME="${TIME:-1-00:00:00}"
PARTITION="${PARTITION:-long}"
CATS="${CATS:-complexity similarity}"     # space-separated run_dataset categories
RUN_FEDERATED="${RUN_FEDERATED:-1}"       # 1 = also submit federated for the 3 supported
FEDERATED_DS=(caco2 herg logd)

# 33 priority datasets.
DATASETS=(
  # Antibody (15)
  abcov_kd abagym_Ang2 abagym_VEGF abagym_EGFR
  flab_expression_koenig humanness_therapeutic flab_thermo_tresanco
  ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho
  ginkgo_acsins ginkgo_colloidal ginkgo_purity ginkgo_aggregation
  # Molecular (8)
  caco2 herg herg_ki adme_mdr1 qm9 logd adme_hlm adme_rlm
  # Reaction (2)
  buchwald_hartwig bh_doyle_plates
  # TCGA (2)
  tcga_brca_pam50 tcga_luad_vs_lusc
  # Nucleotide (6)
  nt_H3K27me3 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_promoter_tata gb_human_enhancers_cohn
)

# Blowup guard: refuse to run unless the list is exactly the expected size.
N=${#DATASETS[@]}
if [[ "$N" -ne 33 ]]; then
  echo "ABORT: expected 33 datasets, got $N. Nothing submitted." >&2
  exit 1
fi

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

read -ra CAT_ARR <<< "$CATS"
NFED=0; [[ "$RUN_FEDERATED" == "1" ]] && NFED=${#FEDERATED_DS[@]}
TOTAL=$(( N * ${#CAT_ARR[@]} + NFED ))

echo "=================================================================="
echo "PRIORITY set: $N datasets | cats: [$CATS]${RUN_FEDERATED:+ +federated×$NFED} | jobs: $TOTAL"
echo "account: $ACCOUNT | wall: $TIME | partition: $PARTITION"
echo "Mode: $([[ $GO == 1 ]] && echo '*** SUBMIT ***' || echo 'DRY RUN (nothing submitted)')"
echo "=================================================================="
echo

submit() {  # cat dataset
  local cat=$1 ds=$2
  if [[ $GO == 1 ]]; then
    local jid
    jid=$(sbatch --parsable --account="$ACCOUNT" --time="$TIME" --partition="$PARTITION" "$WORKER" "$cat" "$ds")
    echo "[$ds] $cat -> job $jid"
  else
    echo "sbatch --account=$ACCOUNT --time=$TIME --partition=$PARTITION $WORKER $cat $ds"
  fi
}

for ds in "${DATASETS[@]}"; do
  for cat in "${CAT_ARR[@]}"; do submit "$cat" "$ds"; done
done

if [[ "$RUN_FEDERATED" == "1" ]]; then
  echo
  echo "--- federated (only caco2/herg/logd have silo loaders) ---"
  for ds in "${FEDERATED_DS[@]}"; do submit federated "$ds"; done
fi

if [[ $GO == 0 ]]; then
  echo
  echo "=================================================================="
  echo "DRY RUN — nothing submitted. Re-run with --go to submit all $TOTAL jobs."
  echo "Noise later:  CATS=noise RUN_FEDERATED=0 bash $0 --go"
  echo "=================================================================="
fi
