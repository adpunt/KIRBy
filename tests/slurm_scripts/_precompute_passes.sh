#!/bin/bash
# What a precompute PASS is, in one place. Sourced, never executed.
#
# WHY THIS FILE EXISTS. Three scripts need to agree on three things: which
# precompute flags a pass means, what suffix its job name carries, and how to
# read that suffix back. They are
#
#   submit_precompute_extra_passes.sh   launches the job
#   migrate_short_to_medium.sh          re-launches it on another partition
#   migrate_medium_to_long.sh           the same, one partition further
#
# Each held its own copy, and the copies drifted. Found 2026-09-16, with 55
# stage-1 jobs pending:
#
#   - the knn pass launches as `--with-knn --objectives similarity`, and both
#     migrate scripts rebuilt it as `--with-knn` alone. A migrated knn job
#     therefore ran a PREDICTIVE six-model search instead of the similarity-only
#     one the owner's 2026-09-12 rule asks for. Different objective, different
#     result-database identity, and s02 would still have found nothing.
#   - the dist pass was in neither migrate script's list at all, so a migrated
#     `__dist` job came back with no `--distance` and wrote to tier 'general'
#     instead of 'general_distance'. Wrong tier, wrong search, no message.
#
# 25 of those 55 jobs are knn or dist. Migrating a pending queue is the standard
# remedy for exactly the state they were in, so both defects were live.
#
# Change a pass HERE and all three scripts change together.

# The precompute flags each pass means. Must match what the submitter runs, or
# a migrated job is a different search from the one that was queued.
pass_flags() {
    case "$1" in
        main|"") echo "" ;;
        knn)     echo "--with-knn --objectives similarity" ;;
        ridge)   echo "--with-ridge" ;;
        sim)     echo "--objectives predictive,similarity" ;;
        dist)    echo "--distance" ;;
        *)       echo "__INVALID__" ;;
    esac
}

# The suffix a pass puts on a job name. The plain predictive pass has NONE:
# every name written before the suffix existed is a plain pass, and parse_name
# below returns PASS="" for it, so old queued names still read correctly.
pass_suffix() {
    case "$1" in
        main|"") echo "" ;;
        *)       echo "__$1" ;;
    esac
}

# Read a job name back into DS, COND, PASS and TUNE.
#
# ORDER MATTERS. The tuned marker is the LAST thing on the name and is stripped
# FIRST, so `pc_caco2_general_none__sim__tuned` leaves `...__sim` for the pass
# match. Strip them the other way round and the pass never matches.
#
# Requires CONDITIONS to be set by the caller (the conditions that script
# handles: the three general ones, or the two distance ones).
parse_name() {
    local name=${1#pc_}
    DS=""
    COND=""
    PASS=""
    TUNE=""
    case "$name" in
        *__tuned) TUNE="--tune"; name="${name%__tuned}" ;;
    esac
    case "$name" in
        *__knn)   PASS="knn";   name="${name%__knn}"   ;;
        *__ridge) PASS="ridge"; name="${name%__ridge}" ;;
        *__sim)   PASS="sim";   name="${name%__sim}"   ;;
        *__dist)  PASS="dist";  name="${name%__dist}"  ;;
    esac
    for c in "${CONDITIONS[@]}"; do
        if [[ "$name" == *"_${c}" ]]; then
            COND="$c"
            DS="${name%_${c}}"
            return 0
        fi
    done
    return 1
}

# The whole job name for one (dataset, condition, pass, tuned) job.
# One definition, so a launched name and a migrated name cannot diverge.
job_name() {
    local ds="$1" cond="$2" pass="$3" tuned="$4"
    printf 'pc_%s_%s%s%s' "$ds" "$cond" "$(pass_suffix "$pass")" \
           "${tuned:+__tuned}"
}
