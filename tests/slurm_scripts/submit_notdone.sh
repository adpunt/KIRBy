#!/bin/bash
# Submit the NOT-DONE datasets, broken up: one SLURM job per dataset per category.
#
# noise runs FIRST and builds the precompute; complexity + similarity then WAIT
# for it (--dependency=afterany) and reuse the warm cache. So precompute is built
# ONCE per dataset, not 3x — avoids the concurrent-rebuild waste.
#
# afterany (not afterok): complexity/similarity run after noise FINISHES whether
# noise passes or fails. This way a single crashed noise *experiment* does NOT
# block the other two categories — the precompute is already on disk by then.
# (Flip to afterok below if you'd rather have them only run on a clean noise exit.)
#
# DRY RUN by default — prints the sbatch commands and submits NOTHING.
# To actually submit:   bash tests/slurm_scripts/submit_notdone.sh --go
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/submit_dataset.sh"
DEP="afterany"

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
echo "Not-done datasets: $N   |   jobs: $((N*3))   |   builder: noise   |   dep: $DEP"
echo "Mode: $([[ $GO == 1 ]] && echo '*** SUBMIT ***' || echo 'DRY RUN (nothing submitted)')"
echo "=================================================================="
echo

for ds in "${DATASETS[@]}"; do
  if [[ $GO == 1 ]]; then
    JID=$(sbatch --parsable "$WORKER" noise "$ds")
    echo "[$ds] noise      -> job $JID  (builds precompute)"
    sbatch --dependency=${DEP}:"$JID" "$WORKER" complexity "$ds" >/dev/null \
      && echo "[$ds] complexity -> waits on $JID"
    sbatch --dependency=${DEP}:"$JID" "$WORKER" similarity "$ds" >/dev/null \
      && echo "[$ds] similarity -> waits on $JID"
    echo
  else
    echo "# $ds"
    echo "JID=\$(sbatch --parsable $WORKER noise $ds)"
    echo "sbatch --dependency=${DEP}:\$JID $WORKER complexity $ds"
    echo "sbatch --dependency=${DEP}:\$JID $WORKER similarity $ds"
    echo
  fi
done

if [[ $GO == 0 ]]; then
  echo "=================================================================="
  echo "DRY RUN — nothing submitted. Re-run with --go to submit all $((N*3)) jobs:"
  echo "    bash tests/slurm_scripts/submit_notdone.sh --go"
  echo "=================================================================="
fi
