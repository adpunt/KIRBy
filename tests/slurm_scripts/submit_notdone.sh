#!/bin/bash
# Submit the NOT-DONE datasets, broken up: one SLURM job per dataset per category.
#
# complexity runs FIRST and builds the precompute; similarity then WAITS for it
# (--dependency=afterany) and reuses the warm cache. So precompute is built ONCE
# per dataset, not per-category — avoids the concurrent-rebuild waste. complexity
# is the builder because it precomputes the FULLEST condition set (general + top +
# distance + top_distance); similarity and noise build subsets of that, so a
# complexity-built cache is complete for all three.
#
# noise is HELD — it is being reworked for large-n domains (subsample sizing +
# whether noise even applies to a given domain). Do NOT resubmit noise here until
# that rework lands. Getting every dataset's precompute done, then complexity and
# similarity, is the priority.
#
# afterany (not afterok): similarity runs after complexity FINISHES whether it
# passes or fails — the precompute is already on disk by then, so a single crashed
# complexity *experiment* does not block similarity.
#
# DRY RUN by default — prints the sbatch commands and submits NOTHING.
# To actually submit:   bash tests/slurm_scripts/submit_notdone.sh --go
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/submit_dataset.sh"
DEP="afterany"
# Account + wall are LIVE decisions — run where_to_submit.sh (§2/§4) first.
# Fair-share flips: as of 2026-07-07 stat-ecr (0.0142) > stat-cadd (0.0072), so
# default is stat-ecr. Account != filesystem (data stays on /data/stat-ecr).
# Override: ACCOUNT=..
ACCOUNT="${ACCOUNT:-stat-ecr}"
# Requested wall. SHORTER walls backfill into idle slots and start sooner; a
# 7-day job only starts when a full 7-day window opens on a node. For a near-term
# deadline set TIME=1-00:00:00 — you lose nothing that wouldn't have finished
# anyway, and jobs start NOW. Default stays 7d for normal runs. Override: TIME=..
TIME="${TIME:-7-00:00:00}"

# 27 datasets whose precompute is NOT done (distance-inclusive --done basis),
# EXCLUDING the 5 OPIG-only datasets that don't run through the standard
# complexity/similarity/noise path (cdr_flexibility, chembl_qsar,
# graphinity_foldx, graphinity_rosetta, humanness_species).
DATASETS=(
  adme_hlm adme_rlm bh_doyle_plates bindingdb_kd davis
  flab_binding_adams flab_binding_koenig flab_binding_phillips_h3
  gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm gb_human_nontata_promoters
  kiba logd
  nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1 nt_H3K4me2 nt_H4K20me1
  nt_enhancers nt_enhancers_types nt_promoter_all nt_promoter_no_tata
  nt_splice_sites_acceptors nt_splice_sites_donors
  qm9
)

# Blowup guard: refuse to run unless the list is exactly the expected size.
N=${#DATASETS[@]}
if [[ "$N" -ne 27 ]]; then
  echo "ABORT: expected 27 datasets, got $N. Nothing submitted." >&2
  exit 1
fi

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

echo "=================================================================="
echo "Not-done datasets: $N   |   jobs: $((N*2))   |   builder: complexity   |   noise: HELD   |   dep: $DEP   |   account: $ACCOUNT   |   wall: $TIME"
echo "Mode: $([[ $GO == 1 ]] && echo '*** SUBMIT ***' || echo 'DRY RUN (nothing submitted)')"
echo "=================================================================="
echo

for ds in "${DATASETS[@]}"; do
  if [[ $GO == 1 ]]; then
    JID=$(sbatch --parsable --account="$ACCOUNT" --time="$TIME" "$WORKER" complexity "$ds")
    echo "[$ds] complexity -> job $JID  (builds precompute)"
    sbatch --account="$ACCOUNT" --time="$TIME" --dependency=${DEP}:"$JID" "$WORKER" similarity "$ds" >/dev/null \
      && echo "[$ds] similarity -> waits on $JID"
    echo
  else
    echo "# $ds"
    echo "JID=\$(sbatch --parsable --account=$ACCOUNT --time=$TIME $WORKER complexity $ds)"
    echo "sbatch --account=$ACCOUNT --time=$TIME --dependency=${DEP}:\$JID $WORKER similarity $ds"
    echo
  fi
done

if [[ $GO == 0 ]]; then
  echo "=================================================================="
  echo "DRY RUN — nothing submitted. Re-run with --go to submit all $((N*2)) jobs:"
  echo "    bash tests/slurm_scripts/submit_notdone.sh --go"
  echo "=================================================================="
fi
