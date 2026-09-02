#!/bin/bash
# ONE SOURCE OF TRUTH for the "where/which partition/which account/why stuck"
# decision. Read-only, touches no jobs. Run before any submit or resubmit.
#
#   cd /data/stat-ecr/scat9264/KIRBy && git pull --ff-only origin main \
#     && bash tests/slurm_scripts/where_to_submit.sh
#
# Prints SUMMARIES, not 150-line job dumps. How to read the output:
#   - ACCOUNT   : bill to whichever account has the higher FairShare (§2).
#   - PARTITION : skip a partition with a huge pending backlog (§5). long/medium
#                 share the same nodes; short is separate + usually swamped.
#   - MEMORY    : §3 prints GB PER CORE. That ratio is the cluster's currency.
#                 A job asking more memory per core than a node has reserves
#                 cores it will never use, and the idle-CPU count stops meaning
#                 anything for it. Check a shape BEFORE submitting it:
#                     bash tests/slurm_scripts/where_to_submit.sh --fits 16 256G
#   - WHY STUCK : pending on (Priority) while §3 shows idle CPUs means one of
#                 two things. Either the REQUESTED WALL (§4) — a 2-day job
#                 cannot backfill into a short idle gap — or the MEMORY, if
#                 --fits reports that no node can hold the shape.
#
# 2026-09-02: f28 job 12923011 sat PENDING for a week at 16 cores / 256 GB /
# 48 h. §3 reported "69 free 32-cpu slots" throughout. Both node types are
# 8 GB per core (arc-c046 48c/384G, arc-c312 288c/2266G), so 256 GB is 33
# cores' worth of memory and the task needed a whole node to itself. The old §3
# had no memory term and could not say that. §6 also offered a devel srun while
# devel was printing "maint 0/96/0/96", which cannot accept a job.
set -uo pipefail
U=scat9264

# Node states that cannot take new work, whatever their idle-CPU count says.
# [$] is a node inside a RESERVATION: its idle CPUs are only usable by jobs in
# that reservation, so counting them as free is what produced the phantom
# capacity above. If you are in the reservation the counts here are
# conservative, never optimistic.
UNAVAILABLE='drain|drng|down|maint|resv|fail|unk|boot|pow|[$]'

# ---------------------------------------------------------------- helpers
# Memory per core, in MB, read from the partition's own nodes. Do not assume
# 8 GB/core -- read it, because a fat node changes it.
mem_per_core_mb() {
  sinfo -h -N -p "$1" -o "%c %m" 2>/dev/null \
    | awk 'NF==2 && $1+0>0 && $2+0>0 {c+=$1; m+=$2} END {printf "%d", (c>0 ? m/c : 0)}'
}

# "<fits now> <allocatable nodes> <big enough anywhere> <all nodes> <free-mem unknown>"
# for a (cores, MB) shape on one partition.
#   fits now          = idle cores AND free memory both sufficient on one node
#                       that is allocatable
#   big enough anywhere = the node's TOTAL memory is sufficient, counted over
#                       EVERY node including the drained ones, so "no node is
#                       large enough" is never confused with "every node is
#                       busy". devel reported 0/0 on 2026-09-02 because it was
#                       in maintenance, not because it was too small.
# %e is per-node free memory. Not every Slurm build fills it in; when it is
# missing the node's total memory is used instead, which is an upper bound, and
# the last number is set so the caller can say so out loud.
nodes_that_fit() {
  sinfo -h -N -p "$1" -o "%n %c %m %e %t %C" 2>/dev/null \
    | awk -v rc="$2" -v rm="$3" -v bad="$UNAVAILABLE" '
        {
          n_all++
          if ($3 + 0 >= rm) big_all++
          if ($5 ~ bad) next
          n_ok++
          split($6, a, "/"); idle_c = a[2] + 0
          free_m = ($4 ~ /^[0-9]+$/) ? $4 + 0 : -1
          if (free_m < 0) { unknown = 1; free_m = $3 + 0 }
          if (idle_c >= rc && free_m >= rm) now++
        }
        END { printf "%d %d %d %d %d", now+0, n_ok+0, big_all+0, n_all+0, unknown+0 }'
}

to_mb() {
  local v n
  v=$(printf '%s' "$1" | tr 'a-z' 'A-Z' | tr -d ' ')
  case "$v" in
    *TB|*T) n=${v%%T*}; echo $(( n * 1024 * 1024 )) ;;
    *GB|*G) n=${v%%G*}; echo $(( n * 1024 )) ;;
    *MB|*M) n=${v%%M*}; echo "$n" ;;
    *)      echo "$v" ;;
  esac
}

# ------------------------------------------------------- --fits <cores> <mem>
# Answer, before submitting, whether a shape can be placed at all. This is the
# sentence that was missing when 12923011 was written.
if [ "${1:-}" = "--fits" ]; then
  req_c="${2:-}"; req_m_in="${3:-}"
  if [ -z "$req_c" ] || [ -z "$req_m_in" ]; then
    echo "usage: $0 --fits <cores-per-task> <memory-per-task, e.g. 256G>" >&2
    exit 2
  fi
  req_m=$(to_mb "$req_m_in")
  case "$req_c$req_m" in *[!0-9]*) echo "--fits: cores and memory must be numbers (got '$req_c' '$req_m_in')" >&2; exit 2;; esac

  echo "REQUESTED   ${req_c} cores, $(( req_m / 1024 )) GB per task"
  echo
  printf "  %-12s %-9s %-14s %-11s %-11s %s\n" \
         PART GB/CORE "MEM-FOOTPRINT" "FITS-NOW" "BIG-ENOUGH" NOTE
  for P in short medium long interactive devel; do
    mpc=$(mem_per_core_mb "$P")
    read -r now n_ok big n_all unknown <<<"$(nodes_that_fit "$P" "$req_c" "$req_m")"
    if [ "${n_all:-0}" -eq 0 ] || [ "${mpc:-0}" -le 0 ]; then
      printf "  %-12s %-9s %s\n" "$P" "?" "no node info for this partition"
      continue
    fi
    # What the request really costs the scheduler: the cores you asked for, or
    # the cores its memory ties up, whichever is bigger.
    equiv=$(( (req_m + mpc - 1) / mpc ))
    foot=$(( equiv > req_c ? equiv : req_c ))
    if [ "$big" -eq 0 ]; then
      note="NO NODE IS LARGE ENOUGH — this shape can never run here"
    elif [ "$n_ok" -eq 0 ]; then
      note="every node draining, reserved or in maintenance right now"
    elif [ "$now" -eq 0 ]; then
      note="large enough nodes exist, none free now"
    else
      note=""
    fi
    [ "${unknown:-0}" -eq 1 ] && note="${note:+$note; }free memory not reported, node totals used"
    printf "  %-12s %-9s %-14s %-11s %-11s %s\n" \
           "$P" "$(( (mpc + 512) / 1024 ))" "${foot} cores" \
           "$now / $n_all" "$big / $n_all" "$note"
  done
  echo
  echo "MEM-FOOTPRINT is what the scheduler really sets aside: the cores you asked"
  echo "for, or the cores your memory ties up, whichever is bigger. When it exceeds"
  echo "the cores you asked for, the idle-CPU counts in section 3 do not apply to"
  echo "this job and a large FITS-NOW elsewhere is no comfort."
  exit 0
fi

# --emit: print ONLY "ACCOUNT PARTITION" for a submit script to consume, applying
# the documented rule — account = the allowed account with the higher FairShare
# (§2); partition = medium by default (long/medium share nodes; short is usually
# swamped — §3). Override the partition with EMIT_PARTITION=long. Everything below
# the guard is the normal human-read diagnostic and is unaffected.
# It does NOT consider memory. Run --fits by hand before trusting it for a
# memory-heavy job; changing what it emits would change every submit script that
# consumes it.
if [ "${1:-}" = "--emit" ]; then
  allowed=$(sacctmgr -n show assoc user="$U" format=account%40 2>/dev/null | awk 'NF{print $1}' | sort -u)
  acct=$(sshare -n -u "$U" -o Account,FairShare 2>/dev/null \
    | awk 'NF>=2 {print $2, $1}' \
    | while read -r fs a; do printf '%s\n' "$allowed" | grep -qx "$a" && echo "$fs $a"; done \
    | sort -rn | head -1 | awk '{print $2}')
  part="${EMIT_PARTITION:-medium}"
  [ -n "$acct" ] || { echo "where_to_submit --emit: could not determine account" >&2; exit 1; }
  echo "$acct $part"
  exit 0
fi

echo "########## 1. ALLOWED account / partition / qos ##########"
sacctmgr -n show assoc user=$U format=account%20,partition%12,qos%30

echo; echo "########## 2. FAIR-SHARE — bill to the higher one ##########"
sshare -u $U -o Account,User,RawShares,NormShares,EffectvUsage,FairShare \
  | awk -v u="$U" 'NR==1 || NR==2 || $2==u'

echo; echo "########## 3. PARTITIONS: wall limit, IDLE capacity, MEMORY PER CORE ##########"
sinfo -o "%.12P %.6a %.12l %.6t %C" | grep -v -i drain
echo "-- idle CPUs (backfill room) and GB PER CORE (the ratio a request must respect) --"
for P in short medium long; do
  C=$(sinfo -h -p $P -o "%C" 2>/dev/null | awk -F/ '{a+=$1;i+=$2;t+=$4} END{printf "%d %d",i,t}')
  I=${C% *}; T=${C#* }
  MPC=$(mem_per_core_mb "$P")
  printf "  %-8s idle=%-6s / total=%-6s  (~%s free 32-cpu slots)   %s GB/core\n" \
         "$P" "${I:-?}" "${T:-?}" "$(( ${I:-0} / 32 ))" "$(( (${MPC:-0} + 512) / 1024 ))"
done
echo "  ★ IDLE CPUs ARE NOT FREE CAPACITY FOR A MEMORY-HEAVY JOB. At 8 GB/core a"
echo "    16-core 256G task ties up 33 cores' worth of memory and needs a node to"
echo "    itself, however many CPUs sit idle. Before you submit:"
echo "        bash tests/slurm_scripts/where_to_submit.sh --fits <cores> <mem>"

echo; echo "########## 4. YOUR JOBS — summary (no per-job dump) ##########"
echo "-- by state --"
squeue -h -u $U -o "%T" | sort | uniq -c
echo "-- pending, by reason --"
squeue -h -u $U -t PENDING -o "%r" | sort | uniq -c | sort -rn
echo "-- pending, by REQUESTED WALL (7-day = can't backfill; short = starts sooner) --"
squeue -h -u $U -t PENDING -o "%l" | sort | uniq -c | sort -rn
echo "-- pending, by REQUESTED MEMORY (above the GB/core in §3 = waiting for whole nodes) --"
squeue -h -u $U -t PENDING -o "%m" | sort | uniq -c | sort -rn
echo "-- pending, by partition --"
squeue -h -u $U -t PENDING -o "%P" | sort | uniq -c | sort -rn
echo "-- RUNNING: time LEFT, soonest first (what can finish in your window) --"
squeue -h -u $U -t RUNNING -o "%L|%i|%P|%b" | sort | awk -F'|' \
  '{printf "  left=%-14s job=%-10s %-8s %s\n",$1,$2,$3,$4}' | head -30
RN=$(squeue -h -u $U -t RUNNING -o "%i" | wc -l | tr -d ' ')
[ "$RN" -gt 30 ] && echo "  ...($RN running total)"

echo; echo "########## 5. QUEUE CONTENTION: everyone's pending, by partition ##########"
squeue -h -t PENDING -o "%P" | sort | uniq -c | sort -rn

echo; echo "########## 6. INTERACTIVE SESSION: where + ready command ##########"
# Interactive partitions on ARC: 'interactive' (4h wall) for real work,
# 'devel' (10min wall) only for a quick compile or smoke. A partition whose
# nodes are ALL draining, reserved or in maintenance has ZERO allocatable CPUs
# however many it reports idle — devel printed "maint 0/96/0/96" on 2026-09-02
# and the old version of this script offered an srun to it anyway. Only a
# partition with allocatable CPUs gets a command printed.
sinfo -o "%.12P %.6a %.12l %.6t %C" | awk 'NR==1 || /interactive|devel/'
USABLE=""
for P in interactive devel; do
  LIVE=$(sinfo -h -N -p "$P" -o "%t %C" 2>/dev/null \
         | awk -v bad="$UNAVAILABLE" '$1 ~ bad {next} {split($2,a,"/"); i+=a[2]} END{print i+0}')
  ALL=$(sinfo -h -p "$P" -o "%C" 2>/dev/null | awk -F/ '{t+=$4} END{print t+0}')
  WALL=$(sinfo -h -p "$P" -o "%l" 2>/dev/null | head -1)
  MPC=$(mem_per_core_mb "$P")
  if [ "${LIVE:-0}" -gt 0 ]; then
    USABLE="$USABLE $P"
    printf "  %-12s USABLE      %s allocatable CPUs of %s, wall %s, %s GB/core\n" \
           "$P" "${LIVE}" "${ALL:-?}" "${WALL:-?}" "$(( (${MPC:-0} + 512) / 1024 ))"
  else
    printf "  %-12s NOT USABLE  every node draining, reserved or in maintenance — an srun here will not start\n" "$P"
  fi
done
if [ -n "$USABLE" ]; then
  echo "-- paste to get a shell (no --account: SLURM resolves the default association) --"
  for P in $USABLE; do
    if [ "$P" = "devel" ]; then
      echo "   srun -p devel -c 8 --mem=16G --time=10:00 --pty /bin/bash          # 10-min smoke only"
    else
      echo "   srun -p interactive --cpus-per-task=8 --mem=32G --time=4:00:00 --pty /bin/bash"
    fi
  done
  echo "   Keep --mem at or under the GB/core above times the cores, or the shell"
  echo "   queues for a whole node the same way a batch job would."
else
  echo "-- NO interactive partition can start a job right now. Do not paste an srun."
fi
