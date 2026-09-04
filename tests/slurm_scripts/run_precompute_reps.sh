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
# SUBMIT:
#   N=$(python tests/precompute_reps.py --list | wc -l)
#   sbatch --clusters=htc --account=<acct> --array=0-$((N-1))%8 \
#          tests/slurm_scripts/run_precompute_reps.sh
#
# One dataset, no array:
#   sbatch --clusters=htc --account=<acct> \
#          tests/slurm_scripts/run_precompute_reps.sh logd
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

if [ "$#" -ge 1 ]; then
    DS="$1"
else
    mapfile -t DATASETS < <(python tests/precompute_reps.py --list)
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: no dataset argument and no SLURM_ARRAY_TASK_ID." >&2
        exit 2
    fi
    DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
fi

echo "dataset=${DS}  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"
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

python -u tests/precompute_reps.py "${DS}"
STATUS=$?
echo "precompute_reps exit=${STATUS} (non-zero means at least one rep is still absent)"
exit ${STATUS}
