#!/bin/bash
# OPIG priority set (2026-07-07): exactly the 6 OPIG experiments being KEPT for
# the presentation. These are SELF-CONTAINED — each loads its own data and runs
# once (NOT parameterised by the 33-dataset priority set), so they submit as
# standalone --wrap jobs, not through submit_dataset.sh.
#
#   1 activity_cliff_qsar   Dablander 2023 — faithful; KEEP/present
#   2 ginkgo_developability Ginkgo GDPa1 — anchor to cv_top, state CV->test gap
#   3 graphinity_ddg        Hummer 2025 — leave-one-complex-out; "reproduce FoldX"
#   4 trastuzumab_enrichment Chinery 2024 — framing fix pending (code, not run)
#   5 blueprint_transfer    KIRBy original — novel contribution (logd->herg_ki)
#   6 cgr_comparison        KIRBy original — CGR-vs-SMILES; verify neural-CGR embeds
#
# Account stat-ecr (live fair-share). Wall default 24h (graphinity is heaviest;
# override TIME= per run if it needs longer). DRY RUN by default; --go submits.
set -euo pipefail

PROJECT_DIR="/data/stat-ecr/scat9264/KIRBy"
ACCOUNT="${ACCOUNT:-stat-ecr}"
TIME="${TIME:-1-00:00:00}"
PARTITION="${PARTITION:-long}"
PRUNING_MODE="${PRUNING_MODE:-aggressive_pruning}"
PM="--pruning-mode ${PRUNING_MODE}"
GO=0
[[ "${1:-}" == "--go" ]] && GO=1

# name | mem | python invocation
EXPERIMENTS=(
  "qsar|64G|python tests/experiments/opig_activity_cliff_qsar.py ${PM}"
  "ginkgo|128G|python tests/experiments/opig_ginkgo_developability.py ${PM}"
  "graphinity|128G|python tests/experiments/opig_graphinity_ddg.py ${PM} --max-foldx 5000"
  "trastuzumab|64G|python tests/experiments/opig_trastuzumab_enrichment.py ${PM}"
  "blueprint|64G|python tests/experiments/opig_blueprint_transfer.py ${PM} --source logd --target herg_ki --mode full"
  "cgr|64G|python tests/experiments/opig_cgr_comparison.py ${PM}"
)

echo "=================================================================="
echo "OPIG priority: ${#EXPERIMENTS[@]} experiments | account: $ACCOUNT | wall: $TIME | partition: $PARTITION"
echo "Mode: $([[ $GO == 1 ]] && echo '*** SUBMIT ***' || echo 'DRY RUN (nothing submitted)')"
echo "=================================================================="
echo

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name mem cmd <<< "$spec"
  wrap="cd ${PROJECT_DIR} && git pull --ff-only origin main 2>/dev/null && . ./setup.sh && export PYTHONWARNINGS=ignore && ${cmd}"
  if [[ $GO == 1 ]]; then
    J=$(sbatch --parsable --job-name="opig_${name}" \
        --account="$ACCOUNT" --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="$TIME" --partition="$PARTITION" --mem="$mem" \
        --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=END,FAIL \
        --wrap="$wrap")
    echo "  $J  opig_${name}  (${mem}, ${TIME}, ${PARTITION})"
  else
    echo "# opig_${name} (${mem})"
    echo "sbatch --job-name=opig_${name} --account=$ACCOUNT -c8 --time=$TIME --partition=$PARTITION --mem=$mem --wrap=\"...${cmd}\""
    echo
  fi
done

if [[ $GO == 0 ]]; then
  echo
  echo "DRY RUN — nothing submitted. Re-run with --go to submit all ${#EXPERIMENTS[@]}."
fi
