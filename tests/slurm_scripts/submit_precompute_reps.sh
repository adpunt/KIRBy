#!/bin/bash
# Submit the representation precompute on htc. Run on a login node:
#
#   bash tests/slurm_scripts/submit_precompute_reps.sh report   # submits NOTHING
#   bash tests/slurm_scripts/submit_precompute_reps.sh all      # the whole roster
#   bash tests/slurm_scripts/submit_precompute_reps.sh missing   # only the gaps
#   bash tests/slurm_scripts/submit_precompute_reps.sh extra     # off-roster names
#   bash tests/slurm_scripts/submit_precompute_reps.sh one logd  # one dataset
#   bash tests/slurm_scripts/submit_precompute_reps.sh one logd --reps pdv,chemberta
#                                                 # rebuild two named reps only
#   bash tests/slurm_scripts/submit_precompute_reps.sh dta       # kiba + bindingdb_kd
#
# Add --with-augmentation to any of them to build the augmentation
# representations as well as the pruned pool. Without them a dataset can only
# run the 'none' precompute condition.
#
# Mode 644, so `bash`, not `./`.
#
# WHAT EACH MODE DOES
#   report   `precompute_reps.py --all --dry-run`. Reads DATASET_REPS and the
#            filesystem; loads no dataset and fits nothing. Measured locally on
#            2026-09-07: 13 seconds to import, rdkit pulled in and torch NOT, so
#            it is safe on a login node. Run it first; it says what is left.
#   all      One array task per name in `precompute_reps.py --list`. That is 40
#            names as of 2026-09-07, read at submit time so the array tracks the
#            roster with no edit here.
#   missing  The same, restricted to the datasets whose pool is incomplete on
#            disk. This is the resubmit path. A blind rerun of `all` after a
#            partial pass asks for 40 GPUs to do nothing.
#   extra    The datasets in the antibody survey's registry that have NO entry
#            in DATASET_REPS and therefore no pruned pool. Thirteen of them on
#            2026-09-07, computed here rather than listed, so it follows the
#            registry. They get GEO_REPS below, not a pruned pool, because the
#            structural geometry survey asks for those nine names.
#   one      A single job on a named dataset. No array.
#
# WHAT THIS SCRIPT DOES NOT SET. Wall, memory, cores and GPU live in the #SBATCH
# header of run_precompute_reps.sh and nowhere else. This script adds only
# --clusters, --account and --array. Change a resource there.
#
# 47:59:00 and 64G in that header are NOT MEASURED — its own header says so.
# Read MaxRSS and Elapsed off the first task that finishes and reset them.
#
# WHICH CLUSTER. Set by the LOGIN NODE, not by --partition. From arc-login a
# --gres=gpu:1 request comes back "Invalid feature specification"; --clusters=htc
# redirects it. Override with CLUSTERS= if you are already on htc-login.
#
# ENVIRONMENT
#   ACCOUNT=...      skip the fair-share lookup and bill this account
#   CLUSTERS=...     default htc
#   CONCURRENT=...   the %N cap on the array, default 8
#   PRINT_ONLY=1     print the sbatch line and submit nothing
set -uo pipefail

MODE="${1:-report}"
# --with-augmentation anywhere on the line builds the augmentation
# representations too, not only the pruned pool.
WITH_AUG=""
# --reps a,b,c anywhere on the line replaces the pruned pool with those names.
# The rebuild case wants it: when a BUILDER changed, one descriptor set has to
# be rebuilt and the GPU-built embeddings beside it are still correct. Without
# this the 'one' mode ignored --reps and rebuilt nothing, because
# precompute_reps.py only builds what is absent from disk.
ONLY_REPS=""
_next_is_reps=0
for _a in "$@"; do
    [ "$_a" = "--with-augmentation" ] && WITH_AUG="--with-augmentation"
    if [ "${_next_is_reps}" = "1" ]; then ONLY_REPS="$_a"; _next_is_reps=0; fi
    [ "$_a" = "--reps" ] && _next_is_reps=1
done
REPS_ARG=()
[ -n "${ONLY_REPS}" ] && REPS_ARG=(--reps "${ONLY_REPS}")
CLUSTERS="${CLUSTERS:-htc}"
CONCURRENT="${CONCURRENT:-8}"
BATCH=tests/slurm_scripts/run_precompute_reps.sh

# The nine names structural_geometry_survey.py asks for when --base-reps is not
# given (its DEFAULT_BASE_REPS, line 66). aac and blosum it computes itself.
GEO_REPS="esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac"

[ -f "${BATCH}" ] || { echo "ERROR: run from the repo root; ${BATCH} not found." >&2; exit 2; }

if [ -z "${KIRBY_CACHE_DIR:-}" ]; then
    echo "!!! KIRBY_CACHE_DIR is not set. Run '. ./setup.sh' first, or every"
    echo "!!! path below resolves to a different cache than the jobs use."
    exit 2
fi

# ------------------------------------------------------------------ report
if [ "${MODE}" = "report" ]; then
    python tests/precompute_reps.py --all --dry-run
    echo
    echo "Nothing was submitted. To build what is missing:"
    echo "  bash tests/slurm_scripts/submit_precompute_reps.sh missing"
    exit 0
fi

# ----------------------------------------------------------------- account
# where_to_submit.sh --emit prints "ACCOUNT PARTITION", applying the fair-share
# rule. The partition it emits is ignored here: this job wants a GPU and a
# 48-hour wall, so the batch script's own --partition=long stands.
if [ -z "${ACCOUNT:-}" ]; then
    ACCOUNT=$(bash tests/slurm_scripts/where_to_submit.sh --emit 2>/dev/null | awk '{print $1}')
fi
if [ -z "${ACCOUNT}" ]; then
    echo "ERROR: could not resolve an account. Run" >&2
    echo "  bash tests/slurm_scripts/where_to_submit.sh" >&2
    echo "and pass the one with the higher FairShare as ACCOUNT=..." >&2
    exit 2
fi

echo "account=${ACCOUNT}  clusters=${CLUSTERS}  concurrency=%${CONCURRENT}"
echo
echo "--- can this shape be placed? (8 cores, 64G — the batch script's header) ---"
bash tests/slurm_scripts/where_to_submit.sh --fits 8 64G 2>/dev/null \
    || echo "  (--fits reported nothing; it reads THIS login node's partitions)"
echo

LIST_DIR="${KIRBY_CACHE_DIR}/precompute_lists"
STAMP=$(date +%Y%m%d_%H%M%S)

submit() {
    echo "--- submitting ---"
    echo "  sbatch --clusters=${CLUSTERS} --account=${ACCOUNT} $*"
    if [ "${PRINT_ONLY:-0}" = "1" ]; then
        echo "  PRINT_ONLY=1 — not submitted."
        return 0
    fi
    JOBID=$(sbatch --clusters="${CLUSTERS}" --account="${ACCOUNT}" "$@" | awk '{print $NF}')
    echo "  job ${JOBID}"
    echo
    echo "Then:"
    echo "  sacct -j ${JOBID} -o JobID,JobName,MaxRSS,Elapsed,State"
    echo "     (if that returns nothing, add -M ${CLUSTERS} — the job is on the other cluster)"
    echo "  bash tests/slurm_scripts/submit_precompute_reps.sh report"
}

case "${MODE}" in
    all)
        N=$(python tests/precompute_reps.py --list | wc -l | tr -d ' ')
        [ "${N}" -gt 0 ] || { echo "ERROR: the roster came back empty." >&2; exit 2; }
        echo "roster: ${N} datasets, one task each"
        submit --array=0-$((N-1))%"${CONCURRENT}" "${BATCH}"
        ;;

    missing)
        mkdir -p "${LIST_DIR}"
        F="${LIST_DIR}/missing_${STAMP}.txt"
        python tests/precompute_reps.py --list-missing > "${F}"
        N=$(grep -c . "${F}" | tr -d ' ')
        if [ "${N}" -eq 0 ]; then
            echo "Every dataset in the roster has its whole pool on disk."
            echo "Nothing to submit. The list file is ${F}."
            exit 0
        fi
        echo "${N} datasets are short. The array indexes this file, so the list"
        echo "is frozen at submit time and a task cannot drift onto another name:"
        echo "  ${F}"
        sed 's/^/    /' "${F}"
        echo
        submit --array=0-$((N-1))%"${CONCURRENT}" "${BATCH}" --list-file "${F}"
        ;;

    extra)
        mkdir -p "${LIST_DIR}"
        F="${LIST_DIR}/extra_${STAMP}.txt"
        python - > "${F}" <<'PYEOF'
import sys
sys.path.insert(0, 'tests')
sys.path.insert(0, 'tests/experiments')
from antibody_baseline_survey import DATASET_REGISTRY
from experiments.complexity.complexity_helpers import DATASET_REPS
for ds in sorted(DATASET_REGISTRY):
    if ds not in DATASET_REPS:
        print(ds)
PYEOF
        N=$(grep -c . "${F}" | tr -d ' ')
        [ "${N}" -gt 0 ] || { echo "ERROR: no off-roster datasets found." >&2; exit 2; }
        echo "${N} datasets in the antibody registry have no entry in DATASET_REPS,"
        echo "so they have no pruned pool. They are built against the geometry"
        echo "survey's nine names instead:"
        echo "  ${GEO_REPS}"
        sed 's/^/    /' "${F}"
        echo
        echo "This does not touch DATASET_REPS and does not pre-empt rep pruning"
        echo "for these datasets. It fills \$KIRBY_CACHE_DIR/rep_cache/<dataset>/,"
        echo "which structural_geometry_survey.py reads (_precompute_cache_path)."
        echo
        submit --array=0-$((N-1))%"${CONCURRENT}" "${BATCH}" --list-file "${F}" --reps "${GEO_REPS}"
        ;;

    one)
        DS="${2:-}"
        [ -n "${DS}" ] || { echo "usage: $0 one <dataset> [--with-augmentation]" >&2; exit 2; }
        echo "one dataset: ${DS}${WITH_AUG:+ (pool + augmentation)}${ONLY_REPS:+ (only: ${ONLY_REPS})}"
        submit "${BATCH}" "${DS}" ${WITH_AUG} ${REPS_ARG[@]+"${REPS_ARG[@]}"}
        ;;

    dta)
        # kiba and bindingdb_kd. Owner's decision 2026-09-14: build them.
        # Measured on ARC the same day, both have 0 of their 10 pool
        # representations and 0 of their 3 augmentation representations on
        # disk, so every experiment that reads either was going to build the
        # whole search live. Five of the ten each are protein language models
        # (esm2, esm2_midlayers, esm2_pocket, esmc, saprot / esm3, kmer), which
        # is why this goes to htc with a GPU and not to arc.
        #
        # davis is NOT here. It is 0 of 16 as well, but it is parked
        # (per-drug-mean degeneracy) and is already absent from the roster in
        # submit_all_datasets.sh.
        #
        # ONE JOB EACH, not an array: two tasks must never share a dataset --
        # the temporary file written before the atomic rename is <rep>.tmp.npy,
        # the same path in both (tests/experiments/_shared.py:213).
        for DS in kiba bindingdb_kd; do
            echo "--- ${DS} (pool + augmentation) ---"
            submit "${BATCH}" "${DS}" --with-augmentation
        done
        ;;

    *)
        echo "usage: bash $0 [report|all|missing|extra|dta|one <dataset>] [--with-augmentation]" >&2
        exit 2
        ;;
esac
