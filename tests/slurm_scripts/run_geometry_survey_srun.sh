#!/bin/bash
# Antibody geometry survey, run INSIDE an interactive srun. No sbatch.
#
#   bash tests/slurm_scripts/run_geometry_survey_srun.sh
#
# Safe to run again after an srun hits its wall clock. Each dataset writes its
# own JSON and --skip-existing reloads the finished ones, so a second run picks
# up where the first stopped. Predicted structures cache per sequence pair under
# $KIRBY_CACHE_DIR/structures, so nothing is predicted twice either.
#
# Start it under tmux or screen on the login node, then srun inside that — a
# dropped connection otherwise takes the allocation with it.

set -u
cd /data/stat-ecr/scat9264/KIRBy || exit 1

OUTPUT="${OUTPUT:-tests/results/geometry_survey}"

# SKIP_EXISTING=1 (the default) reloads any dataset already written and not
# marked FAILED, so a session cut off by its wall clock resumes. Set it to 0
# when the CODE changed and the existing results are stale — a result file
# looks finished whether or not the run that made it was correct.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_ARG=""
[ "$SKIP_EXISTING" = "1" ] && SKIP_ARG="--skip-existing"
BASE_REPS="${BASE_REPS:-esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac}"
MODELS="${MODELS:-rf,dnn}"

# Ordered by how many antibodies each one needs predicted, cheapest first, so an
# interrupted session keeps as many finished datasets as possible. The Jain
# panel and the Ginkgo panel are each one set of antibodies measured for several
# properties, so the first dataset of each pays for structures and the rest of
# that group reload them.
# Override with DATASETS="a b c". The eight with precomputed pools are:
#   DATASETS="abcov_kd ddg_combined flab_thermo_tresanco ginkgo_acsins \
#             ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer"
# The other twelve are absent from DATASET_REPS, so they have no pool and
# refuse in seconds. That is a blocked reason, not a compute one.
DATASETS="${DATASETS:-
abcov_kd
flab_immunogenicity_marks
flab_thermo_jain
flab_aggregation_jain
flab_aggregation_jain_hicrt
flab_pk_jain_fcrnrt
flab_pk_jain_heprt
flab_polyreact_jain_bvp
flab_polyreact_jain_cic
ginkgo_hic
ginkgo_tm2
ginkgo_titer
ginkgo_pr_cho
ginkgo_acsins
ginkgo_purity
ginkgo_aggregation
ginkgo_colloidal
ddg_combined
flab_thermo_tresanco
abagym_all
}"

# Which commit this is actually running. A stale checkout gave jobs 8771117
# and 8771118 a NameError that had been fixed the day before, and nothing in
# the log said so.
git log -1 --format='[srun] running commit %h %ad %s' --date=short

. ./setup.sh || exit 1
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

python tests/check_structure_prediction.py || {
    echo "Structure prediction is broken on this node — stopping before the survey."
    exit 1
}

mkdir -p "$OUTPUT"
n_done=0
n_failed=0

for ds in $DATASETS; do
    echo ""
    echo "############ $ds ############"
    python tests/experiments/structural_geometry_survey.py "$ds" \
        --base-reps "$BASE_REPS" \
        --models "$MODELS" \
        --output "$OUTPUT/" \
        ${SKIP_ARG}
    if [ $? -eq 0 ]; then
        n_done=$((n_done + 1))
    else
        n_failed=$((n_failed + 1))
        echo "  $ds did not finish. Its JSON is either absent or marked FAILED, "
        echo "  so re-running this script retries it."
    fi
done

echo ""
echo "==== $n_done datasets done, $n_failed still outstanding ===="
echo "Results: $OUTPUT/"
echo "Re-run this script to continue; finished datasets are reloaded, not recomputed."
