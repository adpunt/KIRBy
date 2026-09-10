#!/bin/bash
# Build the pruned representation pool for one dataset per array task.
#
# Writes $KIRBY_CACHE_DIR/rep_cache/<dataset>/<rep>.npy and nothing else. Fits
# no model, runs no cross-validation, chooses no recipe. It imports
# kirby.find_best_hybrid and kirby.hybrid — tests/experiments/_shared.py:39-40
# pulls both in at module level — but calls nothing from either.
#
# ROSTER. sorted(DATASET_REPS.keys()), read from the code at run time, so the
# array tracks the roster with no edit here. Today that is 40 names:
#   python tests/precompute_reps.py --list
# One dataset per task. Do NOT run two tasks on the same dataset: the temporary
# file written before the atomic rename is <rep>.tmp.npy, the same path in both
# processes (tests/experiments/_shared.py:213).
#
# WHICH CLUSTER. Set by the LOGIN NODE, not by --partition. From arc-login a
# --gres=gpu:1 request comes back "Invalid feature specification"; --clusters=htc
# redirects it. Submit from htc-login, or keep --clusters=htc below.
#
# SUBMIT. Use the wrapper, which resolves the account and the array size:
#   bash tests/slurm_scripts/submit_precompute_reps.sh report
#   bash tests/slurm_scripts/submit_precompute_reps.sh all
#
# By hand, if you want to see the whole line:
#   N=$(python tests/precompute_reps.py --list | wc -l)
#   sbatch --clusters=htc --account=<acct> --array=0-$((N-1))%8 \
#          tests/slurm_scripts/run_precompute_reps.sh
#
# One dataset, no array:
#   sbatch --clusters=htc --account=<acct> \
#          tests/slurm_scripts/run_precompute_reps.sh logd
#
# ARGUMENTS. A bare name is that dataset. --list-file names a file of dataset
# names, one per line, indexed by SLURM_ARRAY_TASK_ID, which is how the wrapper
# submits a subset without renumbering the roster. --reps takes a
# comma-separated list and replaces the pruned pool with it; it is the only way
# to build a dataset that has no entry in DATASET_REPS, which is the case for
# thirteen of the names in the antibody survey's registry.
#
# WALL AND MEMORY BELOW ARE NOT MEASURED. Nothing in this repo has timed a
# whole-pool rep build. Read MaxRSS and Elapsed off the first finished task
# (sacct -j <id> -o JobID,JobName,MaxRSS,Elapsed,State) and reset them.
#
#SBATCH --job-name=precompute_reps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --mem=64G
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

DS=""
LIST_FILE=""
REPS=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --list-file) LIST_FILE="$2"; shift 2 ;;
        --reps)      REPS="$2";      shift 2 ;;
        *)           DS="$1";        shift   ;;
    esac
done

if [ -z "${DS}" ]; then
    if [ -n "${LIST_FILE}" ] && [ ! -f "${LIST_FILE}" ]; then
        echo "ERROR: --list-file ${LIST_FILE} does not exist." >&2
        exit 2
    fi
    # Read with `while read`, not mapfile: mapfile needs bash 4 and the blank
    # lines it would keep are what make an array index and a `wc -l` count
    # disagree. Blank lines are dropped here and by the submit wrapper's count,
    # so the two agree on which task gets which name.
    DATASETS=()
    # `|| [ -n "$line" ]` keeps the last line of a file that ends without a
    # newline. Without it a hand-written list silently loses its last dataset
    # and the array is one wider than what it indexes.
    while IFS= read -r line || [ -n "${line}" ]; do
        [ -n "${line}" ] && DATASETS+=("${line}")
    done < <(if [ -n "${LIST_FILE}" ]; then cat "${LIST_FILE}"; \
             else python tests/precompute_reps.py --list; fi)

    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: no dataset argument and no SLURM_ARRAY_TASK_ID." >&2
        exit 2
    fi
    if [ "${#DATASETS[@]}" -eq 0 ]; then
        echo "ERROR: the dataset list is empty. Nothing to build." >&2
        exit 2
    fi
    if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#DATASETS[@]}" ]; then
        echo "ERROR: task ${SLURM_ARRAY_TASK_ID} but only ${#DATASETS[@]} datasets" \
             "in the list. The array is wider than what it indexes." >&2
        exit 2
    fi
    DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
fi

# Guarded expansion below: under `set -u` an empty array is an unbound variable
# in bash 4.2, which is what the compute nodes run.
REPS_ARG=()
[ -n "${REPS}" ] && REPS_ARG=(--reps "${REPS}")

echo "dataset=${DS}  reps=${REPS:-pruned pool}  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "no GPU visible to this task"

# Antibody CDR reps number residues through ANARCI, which shells out to hmmscan.
# With anarci importable and hmmscan absent the rep raises and is dropped; with
# anarci NOT importable get_cdr_residue_indices returns EVERY residue index
# (src/kirby/representations/antibody.py:140-143), so a *_cdr rep becomes
# whole-sequence pooled and is cached under the CDR name. Say which it is before
# building anything.
python tests/check_structure_prediction.py || \
    echo "WARNING: structure/numbering check failed — *_cdr reps for this dataset are suspect"

python -u tests/precompute_reps.py "${DS}" ${REPS_ARG[@]+"${REPS_ARG[@]}"}
STATUS=$?
echo "precompute_reps exit=${STATUS} (non-zero means at least one rep is still absent)"
exit ${STATUS}
