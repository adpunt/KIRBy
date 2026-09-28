#!/bin/bash
# Submit stages 1-4 in one go, each waiting for what it needs. Written
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
# PRECOMPUTE. Every pc_ job is on htc now (precompute_status.py, 2026-09-25),
# and a dependency can only name a job on the same cluster, so everything here
# goes to htc. A dataset whose pc_ jobs have all ended gets no dependency and
# may start at once. A dataset whose precompute failed still runs: its
# experiments refuse the missing searches (KIRBY_NO_COLD_SEARCH=1) and the
# refusal lands in _errors/.
#
# OPTIONS
#   --trial "A B"            stage 1 and 2 datasets (required)
#   --fed-datasets "A B"     stage 2 federated datasets (default: caco2); --fed-trial is the same
#   --full "A B ..."         stage 3 and 4 datasets (default: every dataset in
#                            DATASET_REPS except chembl_qsar and davis)
#   --fed-full "A B"         stage 3 federated datasets (default: lhasa_herg caco2 logd,
#                            as submit_everything.sh)
#   --stages "1 2 3 4"       which stages to submit (default all four)
#   --similarity-settings S  KIRBY_SIMILARITY_SETTINGS for stages 1 and 4 (default unset = all)
#   --max-concurrent N       %N on each noise array (default 40)
#   --account A              default: best fair share on htc (where_to_submit.sh)
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
STAGES="1 2 3 4"; SIM=""; MAXC=40; ACCOUNT=""; HOLD4=true; HOLD3=""; DRY=false; HOLD_NOW=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --trial)               TRIAL=$2; shift 2 ;;
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
        -h|--help)             sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done
[ -n "$TRIAL" ] || { echo "--trial \"<regression> <classification>\" is required"; exit 2; }

cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh >/dev/null 2>&1

if [ -z "$ACCOUNT" ]; then
    PAIR=$(bash "$SLURM_DIR/where_to_submit.sh" --best-cluster-account htc 2>/dev/null)
    ACCOUNT=${PAIR##* }
fi
[ -n "$ACCOUNT" ] || { echo "Could not read fair share on htc. Pass --account stat-cadd (or stat-ecr)."; exit 1; }
[ -n "$FULL" ] || FULL=$(python "$HERE/stage1_prelim_noise.py" --all-datasets --print-datasets 2>/dev/null | tail -1)
[ -n "$FULL" ] || { echo "Could not list the full dataset roster. Pass --full."; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
RECORD="$REPO/logs/submit_all_stages_${STAMP}.txt"
mkdir -p "$REPO/logs"
echo "cluster ${CLUSTER}   account ${ACCOUNT}   stages ${STAGES}   $( $DRY && echo DRY RUN )"
echo "trial: ${TRIAL}   federated trial: ${FED_TRIAL}"
echo "full ($(echo $FULL | wc -w | tr -d ' ')): ${FULL}"
$DRY || { echo "# submit_all_stages.sh ${STAMP}  cluster ${CLUSTER} account ${ACCOUNT}"; } > "$RECORD"

# pc_ jobs still queued or running on htc, per dataset.
PCQ=$(squeue -M "$CLUSTER" -u "$USER" -h -o '%i %j' 2>/dev/null | awk '$1 ~ /^[0-9]+$/')
pc_ids() {   # dataset names -> colon list of their pc_ job ids
    local out="" d
    for d in "$@"; do
        # exact dataset: pc_<d>_<n> or pc_<d>_x<n> (a prefix match took
        # herg_ki's jobs for herg, 2026-09-28)
        out="${out}$(printf '%s\n' "$PCQ" | awk -v d="$d" '$2 ~ ("^pc_" d "_x?[0-9]+$") {printf "%s:", $1}')"
    done
    echo "${out%:}"
}
join_deps() { local s; s=$(printf '%s:' "$@" | tr -s ':'); s=${s#:}; echo "${s%:}"; }

# sbatch one job; prints the id (or DRY). Args: name mem deps export-extra script args...
submit() {
    local name=$1 mem=$2 deps=$3 exports=$4; shift 4
    local cmd=(sbatch --parsable --clusters="$CLUSTER" --account="$ACCOUNT"
               --partition=medium,long --time=47:59:00 --nodes=1 --ntasks=1
               --cpus-per-task=8 --mem="$mem" --job-name="$name"
               --mail-user="$EMAIL" --mail-type=FAIL
               --export="ALL,KIRBY_REPO=${REPO},KIRBY_NO_PULL=1,KIRBY_SKIP_INSTALLS=1,KIRBY_NO_COLD_SEARCH=1,PRUNING_MODE=calibrated${exports:+,$exports}")
    [ -n "$deps" ] && cmd+=(--dependency="afterany:${deps}")
    $HOLD_NOW && cmd+=(--hold)
    cmd+=("$@")
    if $DRY; then
        echo "    ${cmd[*]}" >&2; echo "DRY"
    else
        local jid; jid=$("${cmd[@]}" 2>/tmp/sas_err_$$ | cut -d';' -f1)
        if [ -z "$jid" ]; then echo "    FAILED ${name}: $(cat /tmp/sas_err_$$)" >&2; echo ""; return; fi
        echo "$name $jid" >> "$RECORD"; echo "$jid"
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
if [[ " $STAGES " == *" 3 "* ]]; then
    if [ -z "$HOLD3" ]; then [[ " $STAGES " == *" 2 "* ]] && HOLD3=false || HOLD3=true; fi
    HOLD_NOW=$HOLD3
    echo; echo "== stage 3: full suite except noise$( [ -n "$S2" ] && echo ', after stage 2')$( $HOLD3 && echo ', HELD until you release it')"
    for C in complexity similarity; do
        for D in $FULL; do
            J=$(submit "x_${C}_${D}" "$(mem_for "$D")" "$(join_deps "$(pc_ids "$D")" "$S2")" "" \
                "$SLURM_DIR/submit_dataset.sh" "$C" "$D")
            [ -n "$J" ] && S3="${S3}:${J}"
        done
        echo "  ${C}: $(echo $FULL | wc -w | tr -d ' ') jobs"
    done
    for D in $FED_FULL; do
        J=$(submit "x_federated_${D}" "$(mem_for "$D")" "$(join_deps "$(pc_ids "$D")" "$S2")" "" \
            "$SLURM_DIR/submit_dataset.sh" federated "$D")
        echo "  x_federated_${D}: ${J:-FAILED}"
    done
    while read -r N OM DSL; do
        [ -n "$N" ] || continue
        J=$(submit "opig_${N}" "$OM" "$(join_deps "$(pc_ids $DSL)" "$S2")" \
            "KIRBY_OUTPUT_DIR=tests/results/opig,KIRBY_EXP_LOG_DIR=logs/opig" \
            "$HERE/opig_one.sh" "$N")
        echo "  opig_${N}: ${J:-FAILED}"
    done < <(bash "$HERE/opig_one.sh" --list | sed 's/reads://')
fi

HOLD_NOW=false
if [[ " $STAGES " == *" 3 "* ]] && ${HOLD3:-false} && ! $DRY; then
    IDS3=$(awk '$1 ~ /^(x_|opig_)/ {print $2}' "$RECORD" | paste -sd, -)
    echo "  release stage 3 when the srun trials look right:  scontrol -M htc release ${IDS3}"
    echo "release stage 3: scontrol -M htc release ${IDS3}" >> "$RECORD"
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
