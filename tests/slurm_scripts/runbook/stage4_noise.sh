#!/bin/bash
# STAGE 4: every noise experiment on the full dataset roster, with the control
# twins on. Written 2026-09-25.
#
#   bash tests/slurm_scripts/runbook/stage4_noise.sh --dry-run
#   bash tests/slurm_scripts/runbook/stage4_noise.sh --max-concurrent 200
#   bash tests/slurm_scripts/runbook/stage4_noise.sh --datasets "herg_ki caco2"
#
# The same arrays as stage 1 (stage1_prelim_noise.py; one task per dataset,
# experiment and condition, each experiment its own process), with four
# differences:
#   - datasets: every one in DATASET_REPS except chembl_qsar and davis (the
#     exclusion submit_everything.sh and precompute_status.py make), or
#     --datasets, and then only those tests/tools/precompute_status.py calls
#     DONE with no pc_ job still queued on either cluster. Held-back datasets
#     are listed with their reason. The gate replaces a dependency because a
#     dependency cannot name a job on the other cluster (see stage3_full.sh).
#   - KIRBY_NOISE_CONTROLS=1: the random and shuffled twins run
#     (tests/experiments/noise/controls.py).
#   - output tests/results/noise, the real tree; logs logs/noise_full.
#   - job names ns4_*.
# --include-not-done switches the gate off. Every other argument goes to
# stage1_prelim_noise.py (--clusters, --account, --max-concurrent, --wall,
# --partition, --experiments, --similarity-settings, --exp-timeout, --dry-run).
#
# The noise cache (noise_cache.py) is keyed on the label hash, not on the
# output directory, so stage 4 reuses what stage 1 computed on the trial
# datasets, as long as KIRBY_LGBM_THREADS is the same in both (it is 1 in
# submit_dataset.sh unless overridden).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"

DATASETS=""
INCLUDE_NOT_DONE=false
PASS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)         DATASETS=$2; shift 2 ;;
        --include-not-done) INCLUDE_NOT_DONE=true; shift ;;
        -h|--help)          sed -n '2,30p' "$0"; exit 0 ;;
        *)                  PASS+=("$1"); shift ;;
    esac
done

cd "$REPO" || exit 2
if [ "$INCLUDE_NOT_DONE" = true ]; then
    echo "--include-not-done: the precompute gate is OFF."
    DS_ARGS=(--all-datasets)
    [ -n "$DATASETS" ] && DS_ARGS=(--datasets "$DATASETS")
else
    STATUS_OUT=$(python tests/tools/precompute_status.py $DATASETS 2>&1)
    echo "$STATUS_OUT" | sed 's/^/  | /'
    # Same parse as stage3_full.sh.
    GATE=$(printf '%s\n' "$STATUS_OUT" | awk '
        /^DONE /                 {ds=$2; st="done"; next}
        /^NOT DONE /             {ds=$3; st="not done"; next}
        /^⛔ NOT DONE /          {ds=$4; st="not done, nothing queued"; next}
        /^ +queued: / && ds!=""  {q=$0; sub(/^ +queued: /, "", q)
                                  if (st=="done" && q!="nothing queued") st="done, but pc_ still queued: " q
                                  print ds "\t" st; ds=""}')
    PASSED=$(printf '%s\n' "$GATE" | awk -F'\t' '$2=="done" {print $1}' | xargs)
    HELD=$(printf '%s\n' "$GATE" | awk -F'\t' '$2!="done" && $1!=""')
    echo
    echo "through the gate ($(echo "$PASSED" | wc -w | tr -d ' ')): ${PASSED:-none}"
    [ -n "$HELD" ] && { echo "held back:"; printf '%s\n' "$HELD" | awk -F'\t' '{printf "  %-28s %s\n", $1, $2}'; }
    [ -n "$PASSED" ] || { echo "Nothing to submit."; exit 0; }
    DS_ARGS=(--datasets "$PASSED")
fi

exec python "$HERE/stage1_prelim_noise.py" "${DS_ARGS[@]}" \
    --output tests/results/noise --log-dir logs/noise_full \
    --controls --job-prefix ns4 "${PASS[@]}"
