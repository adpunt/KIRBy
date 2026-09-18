#!/bin/bash
# The two shared-vector experiments. Self-contained, like every opig_* script:
# each loads its own data and runs once, so they submit as standalone --wrap
# jobs rather than through submit_dataset.sh.
#
#   census   opig_shared_vectors_census.py
#            Counts, per dataset and per representation, how many molecules get
#            a feature vector identical to another molecule's, split by cause:
#            the dataset holding one molecule twice, a featuriser writing a row
#            of zeros, and the representation merging two different molecules.
#            Turns the count into the best score any model could reach. Fits no
#            model. One job, every molecular and antibody dataset.
#
#   remedy   opig_shared_vectors_remedy.py
#            Does a hybrid or an augmentation fix it. One job per dataset.
#            Reports the held-out score AND the molecules still sharing a row in
#            the matrix the engine built for that fold.
#
# ⛔ READ THIS BEFORE THE REMEDY JOBS. The remedy script REFUSES to start when a
# saved matrix no longer matches what its builder returns, because that is
# exactly how ChemBERTa's count read 0 instead of 130 on herg_ki. Run the
# preflight first and fix anything it names:
#
#   python tests/check_rep_pools.py --all --check-scaling
#
# Sizing below is NOT MEASURED. Read MaxRSS and Elapsed off the first task that
# finishes and reset them. The census reads arrays and counts, so its memory is
# the widest matrix a dataset holds, not the whole pool.
#
# THREE STAGES, because the sweep is too big for one job (see the arithmetic the
# dry run prints):
#
#   bash tests/slurm_scripts/submit_opig_shared_vectors.sh census
#   bash tests/slurm_scripts/submit_opig_shared_vectors.sh spread
#   DATASETS=herg_ki SPREAD=0.0140 bash tests/slurm_scripts/submit_opig_shared_vectors.sh sweep
#
# Stage 3 takes ONE dataset per submission: SPREAD is that dataset's own number
# in that dataset's own metric.
#
# SPREAD comes off the stage-1 log, the line reading
# "<target> spread across outer folds, worst model:". Without it every array task
# re-runs stage 1 to get the same number.
#
# DRY RUN by default; --go submits.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/stat-ecr/scat9264/KIRBy}"
ACCOUNT="${ACCOUNT:-stat-ecr}"
PARTITION="${PARTITION:-long}"
TIME="${TIME:-1-00:00:00}"
CENSUS_TIME="${CENSUS_TIME:-04:00:00}"
SPEED="${SPEED:-full}"
FOLDS="${FOLDS:-5}"
# Every molecular dataset with a pruned pool. Antibody sets have no SMILES, so
# the remedy script's stereochemistry column is empty on them; the census still
# covers them and runs in one job.
DATASETS="${DATASETS:-herg_ki caco2 logd qm9}"
TASKS="${TASKS:-64}"
CONCURRENT="${CONCURRENT:-16}"
SPREAD="${SPREAD:-}"
STAGE="${1:-census}"
GO=0
for _a in "$@"; do [ "$_a" = "--go" ] && GO=1; done

echo "=================================================================="
echo "opig shared vectors | account: $ACCOUNT | partition: $PARTITION"
echo "stage: ${STAGE} | remedy datasets: ${DATASETS}"
echo "Mode: $([[ $GO == 1 ]] && echo '*** SUBMIT ***' || echo 'DRY RUN (nothing submitted)')"
echo "=================================================================="
echo

submit_one() {
  local name="$1" mem="$2" wall="$3" cmd="$4"
  local wrap="cd ${PROJECT_DIR} && . ./setup.sh && export PYTHONWARNINGS=ignore && ${cmd}"
  if [[ $GO == 1 ]]; then
    J=$(sbatch --parsable --job-name="opig_${name}" \
        --account="$ACCOUNT" --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="$wall" --partition="$PARTITION" --mem="$mem" \
        --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=END,FAIL \
        --wrap="$wrap")
    echo "  $J  opig_${name}  (${mem}, ${wall})"
  else
    echo "# opig_${name} (${mem}, ${wall})"
    echo "  ${cmd}"
    echo
  fi
}

submit_array() {
  local name="$1" mem="$2" wall="$3" n="$4" cmd="$5"
  local wrap="cd ${PROJECT_DIR} && . ./setup.sh && export PYTHONWARNINGS=ignore && ${cmd}"
  if [[ $GO == 1 ]]; then
    J=$(sbatch --parsable --job-name="opig_${name}" \
        --account="$ACCOUNT" --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="$wall" --partition="$PARTITION" --mem="$mem" \
        --array=0-$((n-1))%"${CONCURRENT}" \
        --mail-user=adelaide.punt@stcatz.ox.ac.uk --mail-type=END,FAIL \
        --wrap="$wrap")
    echo "  $J  opig_${name}  (array 0-$((n-1)), ${mem}, ${wall})"
  else
    echo "# opig_${name}  array 0-$((n-1))%${CONCURRENT}  (${mem}, ${wall})"
    echo "  ${cmd}"
    echo
  fi
}

case "$STAGE" in

census)
  # Reads matrices and counts. Fits nothing. One job, every dataset.
  submit_one census 64G "$CENSUS_TIME" \
    "python tests/experiments/opig_shared_vectors_census.py --check-freshness 200"
  ;;

spread)
  # Stage 1. The hybrids with no donor. Cheap relative to the sweep, and it
  # prints the target's spread across outer folds -- the unit a margin is a
  # multiple of. Read that number off the log and pass it to stage 2, or every
  # array task re-runs this same work.
  for ds in ${DATASETS}; do
    submit_one "sv_spread_${ds}" 128G "$TIME" \
      "python tests/experiments/opig_shared_vectors_remedy.py --datasets ${ds} --compare C --speed ${SPEED} --n-outer-folds ${FOLDS}"
  done
  ;;

sweep)
  # Stage 2. The augmentation sweep, split over ${TASKS} array tasks. The whole
  # grid still runs -- --sweep-slice takes every Nth setting, so each task gets
  # a mix of strategies and a part-finished array is still readable. Nothing is
  # cut from the grid by doing this.
  [ -n "${SPREAD}" ] || { echo "ERROR: set SPREAD=<number> from the stage-1 log." >&2; exit 2; }
  # One dataset per submission. The spread is that dataset's own number in that
  # dataset's own metric -- herg_ki is scored in RMSE on pKi and qm9 in RMSE on
  # electronvolts, so one SPREAD across four datasets would give the margin
  # axis a different meaning on three of them.
  _n_ds=$(echo ${DATASETS} | wc -w)
  if [ "${_n_ds}" -ne 1 ]; then
    echo "ERROR: stage 'sweep' takes ONE dataset, because SPREAD is that" >&2
    echo "       dataset's own spread in its own metric. Got: ${DATASETS}" >&2
    echo "       Submit one at a time:" >&2
    echo "         DATASETS=herg_ki SPREAD=<its number> $0 sweep --go" >&2
    exit 2
  fi
  for ds in ${DATASETS}; do
    submit_array "sv_sweep_${ds}" 128G "$TIME" "${TASKS}" \
      "python tests/experiments/opig_shared_vectors_remedy.py --datasets ${ds} --compare B D --fold-spread ${SPREAD} --sweep-slice \${SLURM_ARRAY_TASK_ID}/${TASKS} --speed ${SPEED} --n-outer-folds ${FOLDS}"
  done
  ;;

*)
  echo "usage: $0 [census|spread|sweep] [--go]" >&2
  exit 2
  ;;
esac

if [[ $GO == 0 ]]; then
  echo "Dry run. Re-run with --go to submit."
  echo
  echo "⛔ THE WALL AND TASK COUNT BELOW ARE NOT MEASURED."
  echo "At --speed full on a three-representation pool the arithmetic is, from the"
  echo "default list lengths in find_best_hybrid.py:"
  echo "    2,040 configurations per outer fold"
  echo "           (5 allocation x 3 importance x 8 budgets x 17 model-evaluator"
  echo "            pairs. rep_counts is pinned to [None] at"
  echo "            opig_shared_vectors_remedy.py:424, so the count axis is 1, not 4."
  echo "            This line said 4,080 until 2026-09-18 and every figure below it"
  echo "            was therefore twice the truth.)"
  echo "   51,000 fits for one run with no donor  (x 5 inner x 5 outer)"
  echo "  306,000 fits for one run with a donor   (the k-ladder inside create_hybrid)"
  echo "  228 augmentation settings x 3 swept comparisons = about 218 million fits"
  echo "            per dataset. The sweep was 516 settings until 2026-09-18, when"
  echo "            288 of them were found to be duplicates: the margin is not read"
  echo "            when a fraction is set (hybrid.py:4111), so each fraction ran"
  echo "            its 4 margin multiples as 4 identical searches."
  echo "So TASKS=${TASKS} is a starting guess, not a size. Read Elapsed and MaxRSS off"
  echo "the first task that finishes and reset TIME, mem and TASKS from them."
  echo
  echo "How big is the grid before you commit the queue to it:"
  echo "  python tests/experiments/opig_shared_vectors_remedy.py --datasets herg_ki --count-only"
fi
