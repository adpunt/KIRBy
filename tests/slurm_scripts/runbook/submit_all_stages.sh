#!/bin/bash
# Submit stages 1-4 in one go, each waiting for what it needs. Written
# SINCE 2026-10-06 jobs are submitted with tests/slurm_scripts/submit_workers.sh
# and tracked in the ledger (runbook/WORKERS.md). This file is kept because jobs
# submitted before then run a frozen copy of it; do not submit new ones with it.
# 2026-09-25 (owner: "submit everything in stages 2-4 with dependencies").
#
#   bash tests/slurm_scripts/runbook/submit_all_stages.sh --trial "herg_ki herg" --dry-run
#   bash tests/slurm_scripts/runbook/submit_all_stages.sh --trial "herg_ki herg"
#
# WHAT WAITS FOR WHAT (every dependency is afterany: a job starts when the jobs
# it waits for have ENDED, whether they passed or failed):
#   stage 1  prelim noise, trial datasets      waits for those datasets' pc_ jobs
#   stage 2  trials: complexity, similarity,   waits for those datasets' pc_ jobs
#            federated, opig
#   stage 3  full suite except noise           waits for its dataset's pc_ jobs
#                                              AND every stage-2 job
#   stage 4  full noise, controls on           waits for every dataset's pc_ jobs
#                                              AND every stage-1 job,
#                                              and is submitted HELD (see below)
#
# STAGE 4 IS HELD. You change the code after reading stage 1's results. A held
# job never starts until you release it, so stage 4 cannot start on the old
# code the moment stage 1 ends. When the code is ready, pushed and pulled:
#     scontrol -M htc release <ids>        (the ids are printed and saved below)
# --no-hold-stage4 submits it unheld.
#
# PRECOMPUTE. A dependency can only name a job on the same cluster. Stages 1,
# 2 and 4 go to --cluster (default htc) and wait for the pc_ jobs there. Stage 3
# (changed 2026-10-03) reads the pc_ jobs on BOTH arc and htc and sends each
# job to the cluster its datasets' pc_ jobs are on, with afterany on them. A
# job whose datasets have pc_ jobs on both clusters goes to --cluster, waits
# for the ones there, and is submitted HELD; the line printed for it names the
# other cluster's jobs, and you release it once they have ended. A dataset
# whose pc_ jobs have all ended gets no dependency and may start at once, on
# --cluster. A dataset whose precompute failed still runs: its experiments
# refuse a missing WHOLE-DATASET search of more than
# KIRBY_COLD_SEARCH_MAX_CONFIGS (default 500) configurations
# (KIRBY_NO_COLD_SEARCH=1) and the refusal lands in _errors/. Per-fold
# searches are never refused (since 2026-10-04): each experiment builds the
# ones it needs.
#
# STAGE 3 ORDER (2026-10-04, owner): complexity, then similarity, then
# federated, then opig; within each, datasets in the precompute's size classes SMALL, MEDIUM,
# LARGE (run_precompute_dataset.sh). Submission order only breaks ties between
# jobs that become eligible together. Inside each job the experiments run
# quickest first (submit_dataset.sh, resolve_experiments).
#
# STAGE 3 ACROSS-DATASET JOBS (2026-10-03). Three experiments read several
# datasets and are in no per-dataset job: s09, c03 and c14. Stage 3 sends one
# job for each, through runbook/cross_one.sh, last:
#   x_cross_s09  short, 2 h, 1 CPU. Waits (afterany) for every x_similarity_*
#                job of this --output-root: the ones this command submits and
#                the ones already queued under the same comment. Reads
#                DIR/similarity/, writes DIR/similarity/s09_cross_dataset_summary/.
#   x_cross_c03  long, 7 days, 256G. Reads the precompute, not other
#                experiments, so it waits for the pc_ jobs of all its datasets
#                (every --full dataset). Writes DIR/complexity/.
#   x_cross_c14  long, 29 days 23 h, 128G. Runs its own searches and waits for
#                nothing. Tasks: --c14-tasks, default the five datasets named in
#                c14's usage lines. Writes DIR/complexity/.
# A dependency names jobs on one cluster only; a job whose inputs are on both
# is submitted HELD, as in PRECOMPUTE above. --categories cross submits just
# these three; s09, c03 or c14 submits one.
#
# STAGE 3 OUTPUT (2026-10-03): --output-root DIR is required. Results go to
# DIR/<category>/ and DIR/opig/, each experiment's log to logs/<DIR's name>/,
# SLURM's own log to logs/<DIR's name>/slurm/. Use a NEW DIR for new code:
# a resubmitted job skips every experiment DIR already records as finished.
# Resubmitting the same command with the same DIR is how a timed-out run
# continues.
#
# OPTIONS
#   --trial "A B"            stage 1 and 2 datasets (required when --stages has 1 or 2)
#   --output-root DIR        stage 3 results root (required when --stages has 3)
#   --cluster C              default cluster, htc (stage 3: see PRECOMPUTE)
#   --categories "..."       stage 3 categories, default "complexity similarity
#                            federated opig cross" (that order is the submission
#                            order). cross = s09, c03 and c14; each can also be
#                            named alone.
#   --c14-tasks "A B"        c14's datasets (default: logd caco2 herg
#                            flab_thermo_tresanco tcga_luad_vs_lusc, the union of
#                            the two usage lines in c14_cross_domain_summary.py)
#   --fed-datasets "A B"     stage 2 federated datasets (default: caco2); --fed-trial is the same
#   --full "A B ..."         stage 3 and 4 datasets (default: every dataset in
#                            DATASET_REPS except chembl_qsar and davis)
#   --fed-full "A B"         stage 3 federated datasets (default: lhasa_herg caco2 logd,
#                            as submit_everything.sh)
#   --stages "1 2 3 4"       which stages to submit (default all four)
#   --similarity-settings S  KIRBY_SIMILARITY_SETTINGS for stages 1 and 4 (default unset = all)
#   --max-concurrent N       %N on each noise array (default 40)
#   --account A              default: best fair share on --cluster (where_to_submit.sh)
#   --no-hold-stage4
#   --hold-stage3 / --no-hold-stage3   default: held when stage 2 is not in
#                            --stages (the trials were run by srun, so there is
#                            no job for stage 3 to wait on; you release it)
#   --dry-run                print every sbatch line; submit nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SLURM_DIR="$REPO/tests/slurm_scripts"
CLUSTER=htc
EMAIL=adelaide.punt@stcatz.ox.ac.uk
BIG_RE='^(nt_|gb_|tcga_|kiba$|bindingdb_kd$|qm9$)'

TRIAL=""; FED_TRIAL="caco2"; FULL=""; FED_FULL="lhasa_herg caco2 logd"
OUTROOT=""; RUNTAG=""; S3CATS="complexity similarity federated opig cross"
C14_TASKS="logd caco2 herg flab_thermo_tresanco tcga_luad_vs_lusc"
STAGES="1 2 3 4"; SIM=""; MAXC=40; ACCOUNT=""; HOLD4=true; HOLD3=""; DRY=false; HOLD_NOW=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --trial)               TRIAL=$2; shift 2 ;;
        --output-root)         OUTROOT=${2%/}; shift 2 ;;
        --cluster)             CLUSTER=$2; shift 2 ;;
        --categories)          S3CATS=$2; shift 2 ;;
        --c14-tasks)           C14_TASKS=$2; shift 2 ;;
        --fed-trial|--fed-datasets) FED_TRIAL=$2; shift 2 ;;
        --full)                FULL=$2; shift 2 ;;
        --fed-full)            FED_FULL=$2; shift 2 ;;
        --stages)              STAGES=$2; shift 2 ;;
        --similarity-settings) SIM=$2; shift 2 ;;
        --max-concurrent)      MAXC=$2; shift 2 ;;
        --account)             ACCOUNT=$2; shift 2 ;;
        --no-hold-stage4)      HOLD4=false; shift ;;
        --hold-stage3)         HOLD3=true; shift ;;
        --no-hold-stage3)      HOLD3=false; shift ;;
        --dry-run)             DRY=true; shift ;;
        -h|--help)             sed -n '2,90p' "$0"; exit 0 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
if has_stage 1 || has_stage 2; then
    [ -n "$TRIAL" ] || { echo "--trial \"<regression> <classification>\" is required for stages 1 and 2"; exit 2; }
fi
if has_stage 3; then
    [ -n "$OUTROOT" ] || { echo "--output-root DIR is required for stage 3 (a new folder for new code, e.g. tests/results/run_$(date +%Y_%m_%d))"; exit 2; }
fi

cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh >/dev/null 2>&1

if [ -z "$ACCOUNT" ]; then
    PAIR=$(bash "$SLURM_DIR/where_to_submit.sh" --best-cluster-account "$CLUSTER" 2>/dev/null)
    ACCOUNT=${PAIR##* }
fi
[ -n "$ACCOUNT" ] || { echo "Could not read fair share on ${CLUSTER}. Pass --account stat-cadd (or stat-ecr)."; exit 1; }
if has_stage 3 || has_stage 4; then
    [ -n "$FULL" ] || FULL=$(python "$HERE/stage1_prelim_noise.py" --all-datasets --print-datasets 2>/dev/null | tail -1)
    [ -n "$FULL" ] || { echo "Could not list the full dataset roster. Pass --full."; exit 1; }
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RECORD="$REPO/logs/submit_all_stages_${STAMP}.txt"
mkdir -p "$REPO/logs"
echo "cluster ${CLUSTER}   account ${ACCOUNT}   stages ${STAGES}   $( $DRY && echo DRY RUN )"
echo "trial: ${TRIAL}   federated trial: ${FED_TRIAL}"
echo "full ($(echo $FULL | wc -w | tr -d ' ')): ${FULL}"
$DRY || { echo "# submit_all_stages.sh ${STAMP}  cluster ${CLUSTER} account ${ACCOUNT}"; } > "$RECORD"

# pc_ jobs still queued or running, per cluster: lines "<cluster> <id> <name>".
# Both clusters are read; stage 3 uses both, stages 1, 2 and 4 only $CLUSTER.
PCQ=$(for C in arc htc; do
          squeue -M "$C" -u "$USER" -h -o '%i %j' 2>/dev/null \
              | awk -v c="$C" '$1 ~ /^[0-9]+$/ && $2 ~ /^pc_/ {print c, $1, $2}'
      done)
# Every job of this user on both clusters: "<cluster> <id> <name> <comment>".
# Stage 3 stamps each job's comment with its output folder's name and does not
# submit a job whose name AND comment are already queued or running. So pasting
# the same command again while the first jobs still run cannot start two jobs
# on one dataset's files, and a held job left from an older submission (no
# comment, or another folder's) does not stop the new one.
QALL=$(for C in arc htc; do
           squeue -M "$C" -u "$USER" -h -o '%i %j %k' 2>/dev/null \
               | awk -v c="$C" '$1 ~ /^[0-9]+$/ {print c, $1, $2, $3}'
       done)
queued_as() { printf '%s\n' "$QALL" | awk -v n="$1" -v t="$RUNTAG" '$3==n && $4==t {printf "%s %s ", $1, $2}'; }
pc_on() {    # cluster dataset... -> colon list of those datasets' pc_ job ids there
    local cl=$1 out="" d; shift
    for d in "$@"; do
        # exact dataset: pc_<d>_<n>, pc_<d>_x<n> or pc_<d>_general_... (a
        # prefix match took herg_ki's jobs for herg, 2026-09-28)
        out="${out}$(printf '%s\n' "$PCQ" | awk -v c="$cl" -v d="$d" \
            '$1==c && $3 ~ ("^pc_" d "_(x?[0-9]+|general_.*)$") {printf "%s:", $2}')"
    done
    echo "${out%:}"
}
pc_ids() { pc_on "$CLUSTER" "$@"; }
pc_clusters() {   # dataset... -> the clusters holding any of their pc_ jobs
    local c
    for c in arc htc; do [ -n "$(pc_on "$c" "$@")" ] && printf '%s ' "$c"; done
}

# Stage 3 placement for one job reading datasets "$@". Sets SUB_CL (cluster),
# DEP (job ids to wait for there), HOLD_THIS and NOTE.
place() {
    local cls n
    cls=$(pc_clusters "$@"); n=$(echo $cls | wc -w | tr -d ' ')
    HOLD_THIS=false; NOTE=""
    if [ "$n" = "0" ]; then
        SUB_CL=$CLUSTER; DEP=""
    elif [ "$n" = "1" ]; then
        SUB_CL=${cls%% *}; DEP=$(pc_on "$SUB_CL" "$@")
    else
        SUB_CL=$CLUSTER; DEP=$(pc_on "$CLUSTER" "$@"); HOLD_THIS=true
        local o; for o in arc htc; do [ "$o" = "$CLUSTER" ] || NOTE="HELD: also waits for ${o} jobs $(pc_on "$o" "$@" | tr ':' ' '); release it once they have ended"; done
    fi
    if [ -n "$S2" ]; then
        if [ "$SUB_CL" = "$CLUSTER" ]; then DEP=$(join_deps "$DEP" "$S2")
        else HOLD_THIS=true; NOTE="${NOTE:+$NOTE; }HELD: stage 2 is on ${CLUSTER}, this job on ${SUB_CL}; release it after stage 2"; fi
    fi
}

# Datasets in the precompute's size classes: SMALL, then MEDIUM (everything
# not listed), then LARGE, each in the given order. Read from
# run_precompute_dataset.sh, which keeps the two lists.
order_by_size() {
    local f="$SLURM_DIR/run_precompute_dataset.sh" small large d s="" m="" l=""
    small=$(sed -n 's/^SMALL_DATASETS="\${KIRBY_SMALL_DATASETS:-\(.*\)}"$/\1/p' "$f")
    large=$(sed -n 's/^LARGE_DATASETS="\${KIRBY_LARGE_DATASETS:-\(.*\)}"$/\1/p' "$f")
    for d in "$@"; do
        if [[ " $small " == *" $d "* ]]; then s="$s $d"
        elif [[ " $large " == *" $d "* ]]; then l="$l $d"
        else m="$m $d"; fi
    done
    echo $s $m $l
}
join_deps() { local s; s=$(printf '%s:' "$@" | tr -s ':'); s=${s#:}; echo "${s%:}"; }

# sbatch one job; prints the id (or DRY). Args: name mem deps export-extra script args...
# SUB_PART, SUB_TIME and SUB_CPUS, when set, replace medium,long, 47:59:00 and 8.
# Cluster: SUB_CL when set (stage 3), else CLUSTER. HOLD_THIS=true holds this
# one job. SLURM_LOG_DIR, when set, is where SLURM writes the job's own log.
submit() {
    local name=$1 mem=$2 deps=$3 exports=$4; shift 4
    local cl=${SUB_CL:-$CLUSTER}
    local cmd=(sbatch --parsable --clusters="$cl" --account="$ACCOUNT"
               --partition="${SUB_PART:-medium,long}" --time="${SUB_TIME:-47:59:00}"
               --nodes=1 --ntasks=1
               --cpus-per-task="${SUB_CPUS:-8}" --mem="$mem" --job-name="$name"
               --mail-user="$EMAIL" --mail-type=FAIL
               --export="ALL,KIRBY_REPO=${REPO},KIRBY_NO_PULL=1,KIRBY_SKIP_INSTALLS=1,KIRBY_NO_COLD_SEARCH=1,PRUNING_MODE=calibrated${exports:+,$exports}")
    [ -n "$deps" ] && cmd+=(--dependency="afterany:${deps}")
    [ -n "${SLURM_LOG_DIR:-}" ] && cmd+=(--output="${SLURM_LOG_DIR}/%x_%j.out")
    [ -n "${RUNTAG:-}" ] && cmd+=(--comment="$RUNTAG")
    { $HOLD_NOW || ${HOLD_THIS:-false}; } && cmd+=(--hold)
    cmd+=("$@")
    if $DRY; then
        echo "    ${cmd[*]}" >&2; echo "DRY"
    else
        local jid; jid=$("${cmd[@]}" 2>/tmp/sas_err_$$ | cut -d';' -f1)
        if [ -z "$jid" ]; then echo "    FAILED ${name}: $(cat /tmp/sas_err_$$)" >&2; echo ""; return; fi
        echo "$name $jid $cl$( { $HOLD_NOW || ${HOLD_THIS:-false}; } && echo ' held')" >> "$RECORD"; echo "$jid"
    fi
}
mem_for() { [[ $1 =~ $BIG_RE ]] && echo 128G || echo 64G; }

S1=""; S2=""; S3=""; S4=""

# ── stage 1 ──
if [[ " $STAGES " == *" 1 "* ]]; then
    echo; echo "== stage 1: prelim noise on ${TRIAL}"
    A=(--datasets "$TRIAL" --clusters "$CLUSTER" --account "$ACCOUNT" --wait-for-precompute
       --max-concurrent "$MAXC" --job-prefix ns1)
    [ -n "$SIM" ] && A+=(--similarity-settings "$SIM")
    $DRY && A+=(--dry-run)
    OUT1=$(python "$HERE/stage1_prelim_noise.py" "${A[@]}" 2>&1); echo "$OUT1" | sed 's/^/  | /'
    S1=$(printf '%s\n' "$OUT1" | awk '/^JOBIDS /{print $2}')
    $DRY || echo "stage1 ${S1}" >> "$RECORD"
fi

# ── stage 2 ──
if [[ " $STAGES " == *" 2 "* ]]; then
    echo; echo "== stage 2: trials on ${TRIAL} (federated: ${FED_TRIAL})"
    for C in complexity similarity; do
        for D in $TRIAL; do
            J=$(submit "t2_${C}_${D}" "$(mem_for "$D")" "$(pc_ids "$D")" \
                "KIRBY_OUTPUT_DIR=tests/results/trial_${C},KIRBY_EXP_LOG_DIR=logs/trial,KIRBY_FORCE_RERUN=1" \
                "$SLURM_DIR/submit_dataset.sh" "$C" "$D")
            echo "  t2_${C}_${D}: ${J:-FAILED}"; S2="${S2}:${J}"
        done
    done
    for D in $FED_TRIAL; do
        J=$(submit "t2_federated_${D}" "$(mem_for "$D")" "$(pc_ids "$D")" \
            "KIRBY_OUTPUT_DIR=tests/results/trial_federated,KIRBY_EXP_LOG_DIR=logs/trial,KIRBY_FORCE_RERUN=1" \
            "$SLURM_DIR/submit_dataset.sh" federated "$D")
        echo "  t2_federated_${D}: ${J:-FAILED}"; S2="${S2}:${J}"
    done
    while read -r N OM DSL; do
        [ -n "$N" ] || continue
        J=$(submit "t2_opig_${N}" "$OM" "$(pc_ids $DSL)" \
            "KIRBY_OUTPUT_DIR=tests/results/trial_opig,KIRBY_EXP_LOG_DIR=logs/trial" \
            "$HERE/opig_one.sh" "$N")
        echo "  t2_opig_${N}: ${J:-FAILED}"; S2="${S2}:${J}"
    done < <(bash "$HERE/opig_one.sh" --list | sed 's/reads://')
    S2=$(join_deps $S2 | sed 's/DRY//g' | tr -s ':'); S2=${S2#:}; S2=${S2%:}
fi

# ── stage 3 ──
if has_stage 3; then
    if [ -z "$HOLD3" ]; then has_stage 2 && HOLD3=false || HOLD3=true; fi
    HOLD_NOW=$HOLD3
    RUNTAG=$(basename "$OUTROOT")
    LOG3="logs/${RUNTAG}"
    echo; echo "== stage 3: full suite except noise$( [ -n "$S2" ] && echo ', after stage 2')$( $HOLD3 && echo ', HELD until you release it')"
    echo "   results ${OUTROOT}/<category>/   experiment logs ${LOG3}/   SLURM logs ${LOG3}/slurm/"
    N_DONE=$(cat "$OUTROOT"/*/_job_summaries/*.tsv 2>/dev/null | awk -F'\t' '$7=="ok"' | wc -l | tr -d ' ')
    if [ -d "$OUTROOT" ]; then
        echo "   ${OUTROOT} already exists: ${N_DONE} finished experiment row(s) in it will be skipped."
        echo "   That is right for a resubmission of the same code. For new code, use a new --output-root."
    fi
    $DRY || mkdir -p "$REPO/${LOG3}/slurm"
    SLURM_LOG_DIR="$REPO/${LOG3}/slurm"
    FULL_ORDERED=$(order_by_size $FULL)
    FED_ORDERED=$(order_by_size $FED_FULL)
    echo "   dataset order (SMALL, MEDIUM, LARGE): ${FULL_ORDERED}"
    show() {   # name id
        echo "  $1: ${2:-FAILED} on ${SUB_CL}${DEP:+ after ${DEP//:/ }}${NOTE:+  <- ${NOTE}}"
    }
    SIMJOBS=""   # "<cluster>:<id>" of each x_similarity_* job this command submits
    for C in $S3CATS; do
        case " opig cross s09 c03 c14 " in *" $C "*) continue ;; esac
        case " similarity federated complexity " in *" $C "*) ;; *) echo "  unknown category $C (stage 3 knows similarity federated complexity opig cross s09 c03 c14)"; continue ;; esac
        if [ "$C" = federated ]; then LIST=$FED_ORDERED; else LIST=$FULL_ORDERED; fi
        for D in $LIST; do
            Q=$(queued_as "x_${C}_${D}")
            if [ -n "$Q" ]; then echo "  x_${C}_${D}: already queued or running (${Q% }); not submitted again"; continue; fi
            place "$D"
            J=$(submit "x_${C}_${D}" "$(mem_for "$D")" "$DEP" \
                "KIRBY_OUTPUT_DIR=${OUTROOT}/${C},KIRBY_EXP_LOG_DIR=${LOG3}" \
                "$SLURM_DIR/submit_dataset.sh" "$C" "$D")
            show "x_${C}_${D}" "$J"
            [ -n "$J" ] && S3="${S3}:${J}"
            [ "$C" = similarity ] && [ -n "$J" ] && SIMJOBS="${SIMJOBS} ${SUB_CL}:${J}"
        done
    done
    while [[ " $S3CATS " == *" opig "* ]] && read -r N OM DSL; do
        [ -n "$N" ] || continue
        Q=$(queued_as "opig_${N}")
        if [ -n "$Q" ]; then echo "  opig_${N}: already queued or running (${Q% }); not submitted again"; continue; fi
        place $DSL
        J=$(submit "opig_${N}" "$OM" "$DEP" \
            "KIRBY_OUTPUT_DIR=${OUTROOT}/opig,KIRBY_EXP_LOG_DIR=${LOG3}" \
            "$HERE/opig_one.sh" "$N")
        show "opig_${N}" "$J"
    done < <(bash "$HERE/opig_one.sh" --list | sed 's/reads://')

    # ── the three across-dataset experiments (2026-10-03) ──
    want_cross() { [[ " $S3CATS " == *" cross "* || " $S3CATS " == *" $1 "* ]]; }
    cross_submit() {   # name mem deps outdir partition time cpus datasets...
        local n=$1 m=$2 d=$3 o=$4; SUB_PART=$5; SUB_TIME=$6; SUB_CPUS=$7; shift 7
        J=$(submit "x_cross_${n}" "$m" "$d" \
            "KIRBY_OUTPUT_DIR=${OUTROOT}/${o},KIRBY_EXP_LOG_DIR=${LOG3}" \
            "$HERE/cross_one.sh" "$n" "$@")
        SUB_PART=""; SUB_TIME=""; SUB_CPUS=""
        show "x_cross_${n}" "$J"
    }
    if want_cross s09; then
        Q=$(queued_as "x_cross_s09")
        if [ -n "$Q" ]; then echo "  x_cross_s09: already queued or running (${Q% }); not submitted again"
        else
            # Every x_similarity_* job of this folder: submitted just now, or
            # already queued under the same comment (those were skipped above,
            # or --categories left similarity out).
            ALLSIM="$SIMJOBS"
            for D in $FULL_ORDERED; do
                ALLSIM="${ALLSIM} $(queued_as "x_similarity_${D}" | awk '{for (i=1; i<NF; i+=2) printf "%s:%s ", $i, $(i+1)}')"
            done
            SIMCL=$(for x in $ALLSIM; do echo "${x%%:*}"; done | sort -u | xargs)
            HOLD_THIS=false; NOTE=""
            case $(echo $SIMCL | wc -w | tr -d ' ') in
                0) SUB_CL=$CLUSTER ;;
                1) SUB_CL=$SIMCL ;;
                *) SUB_CL=$CLUSTER; HOLD_THIS=true
                   NOTE="HELD: also waits for x_similarity_* jobs on the other cluster: $(for x in $ALLSIM; do [ "${x%%:*}" = "$CLUSTER" ] || printf '%s ' "$x"; done | xargs); release it once they have ended" ;;
            esac
            DEP=$(for x in $ALLSIM; do [ "${x%%:*}" = "$SUB_CL" ] && [ "${x#*:}" != DRY ] && printf '%s:' "${x#*:}"; done)
            DEP=${DEP%:}
            $DRY && [ -n "$SIMJOBS" ] && NOTE="${NOTE:+$NOTE; }dry run: also waits for the x_similarity_* jobs above, whose numbers exist only on a real submit"
            cross_submit s09 16G "$DEP" similarity short 02:00:00 1 $FULL_ORDERED
        fi
    fi
    if want_cross c03; then
        Q=$(queued_as "x_cross_c03")
        if [ -n "$Q" ]; then echo "  x_cross_c03: already queued or running (${Q% }); not submitted again"
        else
            place $FULL_ORDERED
            cross_submit c03 256G "$DEP" complexity long 6-23:59:00 8 $FULL_ORDERED
        fi
    fi
    if want_cross c14; then
        Q=$(queued_as "x_cross_c14")
        if [ -n "$Q" ]; then echo "  x_cross_c14: already queued or running (${Q% }); not submitted again"
        else
            SUB_CL=$CLUSTER; DEP=""; HOLD_THIS=false; NOTE=""
            cross_submit c14 128G "" complexity long 29-23:00:00 8 $C14_TASKS
        fi
    fi
    SUB_CL=""; HOLD_THIS=false; SLURM_LOG_DIR=""; RUNTAG=""
fi

HOLD_NOW=false
if has_stage 3 && ! $DRY; then
    for CL in arc htc; do
        if ${HOLD3:-false}; then
            IDS3=$(awk -v c="$CL" '$1 ~ /^(x_|opig_)/ && $3==c {print $2}' "$RECORD" | paste -sd, -)
        else
            IDS3=$(awk -v c="$CL" '$1 ~ /^(x_|opig_)/ && $3==c && $4=="held" {print $2}' "$RECORD" | paste -sd, -)
        fi
        [ -n "$IDS3" ] || continue
        echo "  held stage 3 jobs on ${CL}; release with:  scontrol -M ${CL} release ${IDS3}"
        echo "release stage 3 on ${CL}: scontrol -M ${CL} release ${IDS3}" >> "$RECORD"
    done
fi

# ── stage 4 ──
if [[ " $STAGES " == *" 4 "* ]]; then
    echo; echo "== stage 4: full noise with controls, after stage 1$( $HOLD4 && echo ', HELD until you release it')"
    A=(--all-datasets --clusters "$CLUSTER" --account "$ACCOUNT" --wait-for-precompute
       --controls --output tests/results/noise --log-dir logs/noise_full
       --max-concurrent "$MAXC" --job-prefix ns4)
    [ -n "$FULL" ] && A=(--datasets "$FULL" "${A[@]:1}")
    S1D=$(echo "$S1" | sed 's/DRY//g')
    [ -n "$S1D" ] && A+=(--after "$S1D")
    [ -n "$SIM" ] && A+=(--similarity-settings "$SIM")
    $HOLD4 && A+=(--hold)
    $DRY && A+=(--dry-run)
    OUT4=$(python "$HERE/stage1_prelim_noise.py" "${A[@]}" 2>&1); echo "$OUT4" | sed 's/^/  | /'
    S4=$(printf '%s\n' "$OUT4" | awk '/^JOBIDS /{print $2}')
    $DRY || echo "stage4 ${S4}" >> "$RECORD"
    if $HOLD4 && [ -n "$S4" ] && ! $DRY; then
        echo "  release stage 4 when the code is ready:  scontrol -M htc release ${S4//:/,}"
        echo "release: scontrol -M htc release ${S4//:/,}" >> "$RECORD"
    fi
fi

echo
$DRY && echo "DRY RUN: nothing submitted." || echo "Job ids saved to ${RECORD}"
