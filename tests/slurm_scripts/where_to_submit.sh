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
#   - WHY STUCK : if pending reason is (Priority) AND idle CPUs exist (§3), the
#                 blocker is your REQUESTED WALL (§4) — a 7-day job can't backfill
#                 into a short idle gap. Lower --time on fast datasets to start now.
set -uo pipefail
U=scat9264

# --emit: print ONLY "ACCOUNT PARTITION" for a submit script to consume, applying
# the documented rule — account = the allowed account with the higher FairShare
# (§2); partition = medium by default (long/medium share nodes; short is usually
# swamped — §3). Override the partition with EMIT_PARTITION=long. Everything below
# the guard is the normal human-read diagnostic and is unaffected.
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

echo; echo "########## 3. PARTITIONS: wall limit + IDLE capacity (backfill room) ##########"
sinfo -o "%.12P %.6a %.12l %.6t %C" | grep -v -i drain
echo "-- idle CPUs per partition (= backfill room; ~/32 = job-slots free NOW) --"
for P in short medium long; do
  C=$(sinfo -h -p $P -o "%C" 2>/dev/null | awk -F/ '{a+=$1;i+=$2;t+=$4} END{printf "%d %d",i,t}')
  I=${C% *}; T=${C#* }
  printf "  %-8s idle=%-6s / total=%-6s  (~%s free 32-cpu slots)\n" "$P" "${I:-?}" "${T:-?}" "$(( ${I:-0} / 32 ))"
done

echo; echo "########## 4. YOUR JOBS — summary (no per-job dump) ##########"
echo "-- by state --"
squeue -h -u $U -o "%T" | sort | uniq -c
echo "-- pending, by reason --"
squeue -h -u $U -t PENDING -o "%r" | sort | uniq -c | sort -rn
echo "-- pending, by REQUESTED WALL (7-day = can't backfill; short = starts sooner) --"
squeue -h -u $U -t PENDING -o "%l" | sort | uniq -c | sort -rn
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
# 'devel' (10min wall) only for quick compile/smoke. Pick the one with idle
# CPUs + enough wall. Print their state, then the srun to paste.
sinfo -o "%.12P %.6a %.12l %.6t %C" | awk 'NR==1 || /interactive|devel/'
for P in interactive devel; do
  C=$(sinfo -h -p $P -o "%C" 2>/dev/null | awk -F/ '{i+=$2;t+=$4} END{printf "%d %d",i,t}')
  I=${C% *}; T=${C#* }
  printf "  %-12s idle=%-5s / total=%-5s\n" "$P" "${I:-?}" "${T:-?}"
done
echo "-- paste to get a shell (swap ACCT for the higher-FairShare account from §2) --"
echo "   srun -p interactive --account=ACCT --cpus-per-task=8 --mem=32G --time=4:00:00 --pty /bin/bash"
echo "   # quick 10-min smoke instead:  srun -p devel --account=ACCT -c8 --mem=16G --time=10:00 --pty /bin/bash"
