#!/bin/bash
# Surgical recovery of jobs hung on the capped cold-build bug.
#
# A job is "stuck" if it is RUNNING but its slurm-<id>.out has been SILENT for
# more than THRESHOLD minutes (a live job writes with python -u; a capped-hung
# job stops). Fresh jobs (still writing — e.g. davis/kiba precompute) and
# anything below the threshold are left untouched. Resubmitted jobs pick up the
# fix automatically because submit_dataset.sh git-pulls at job start.
#
# Granularity is per (dataset, category):
#   - DONE datasets: cancel ONLY the stuck category job(s); fresh categories
#     of the same dataset keep running. Resubmit the stuck category (precompute
#     already exists; resume skips finished experiments).
#   - NOT-DONE datasets: cancel ALL that dataset's jobs and resubmit
#     noise->cx/sim with the dependency, so precompute is rebuilt once.
#
# DRY RUN by default: lists each stuck job + its LAST log line (so you can
# confirm the capped signature) and submits nothing.
#   bash tests/slurm_scripts/recover_stuck.sh
#   THRESHOLD=1000 bash tests/slurm_scripts/recover_stuck.sh   # widen the net
#   bash tests/slurm_scripts/recover_stuck.sh --go             # execute
set -uo pipefail

THRESHOLD=${THRESHOLD:-1440}          # minutes of log silence => stuck (24h default)
ACCT=${ACCOUNT:-stat-ecr}             # stat-ecr has the higher fair-share (2x); matches submit_done/notdone/priority
DIR=/data/stat-ecr/scat9264/KIRBy
WORKER=$DIR/tests/slurm_scripts/submit_dataset.sh
GO=0; [ "${1:-}" = "--go" ] && GO=1
now=$(date +%s)

NOTDONE=" adme_hlm adme_rlm bh_doyle_plates bindingdb_kd davis flab_binding_adams flab_binding_koenig flab_binding_phillips_h3 gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm gb_human_nontata_promoters kiba logd nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1 nt_H3K4me2 nt_H4K20me1 nt_enhancers nt_enhancers_types nt_promoter_all nt_promoter_no_tata nt_splice_sites_acceptors nt_splice_sites_donors qm9 "

cd "$DIR"

# Map every long-partition job -> "category dataset" (single scontrol pass).
declare -A JOB_CAT JOB_DS
while read -r jid; do
  [ -n "$jid" ] || continue
  line=$(scontrol show job "$jid" 2>/dev/null | grep -oE "submit_dataset.sh [a-z]+ [A-Za-z0-9_]+")
  JOB_CAT[$jid]=$(echo "$line" | awk '{print $2}')
  JOB_DS[$jid]=$(echo "$line" | awk '{print $3}')
done < <(squeue -u "$USER" -h -p long -o %i)

# Identify stuck (RUNNING + stale log) jobs.
declare -A STUCK_DS STUCK_JOB
while read -r jid st; do
  [ "$st" = "RUNNING" ] || continue
  log="slurm-${jid}.out"; [ -f "$log" ] || continue
  age=$(( (now - $(stat -c %Y "$log")) / 60 ))
  [ "$age" -gt "$THRESHOLD" ] || continue
  ds=${JOB_DS[$jid]:-}; cat=${JOB_CAT[$jid]:-}
  [ -n "$ds" ] && [ -n "$cat" ] || continue
  STUCK_DS[$ds]=1; STUCK_JOB[$jid]=1
  if [ $GO -eq 0 ]; then
    last=$(grep -vE '^[[:space:]]*$' "$log" | tail -1 | cut -c1-60)
    printf "  STUCK %6dm  %-11s %-27s job %-10s | %s\n" "$age" "$cat" "$ds" "$jid" "$last"
  fi
done < <(squeue -u "$USER" -h -p long -o "%i %T")

echo
_dslist="${!STUCK_DS[*]}"
echo "Stuck datasets (${#STUCK_DS[@]}): ${_dslist:-none}"
echo "Threshold: ${THRESHOLD}min silence   Account: $ACCT   Mode: $([ $GO -eq 1 ] && echo '*** SUBMIT ***' || echo 'DRY RUN')"
if [ $GO -eq 0 ]; then
  echo
  echo "Eyeball the 'last log line' above — a capped hang ends on a 'strategy=capped'"
  echo "skip line or a stalled search. Re-run with --go to cancel+resubmit these,"
  echo "or THRESHOLD=1000 bash $0  to widen the net."
  exit 0
fi

# NOTE: noise is HELD (not resubmitted) — the user is reworking the noise
# experiments and will resubmit noise afterwards. We only bring cx/sim back.
# complexity is the builder (heavier of the two: c05/c07); similarity waits on it.
for ds in "${!STUCK_DS[@]}"; do
  if [[ "$NOTDONE" == *" $ds "* ]]; then
    echo "[$ds] not-done -> cancel all its jobs + resubmit complexity(builder)->similarity  (noise HELD)"
    for jid in "${!JOB_DS[@]}"; do
      [ "${JOB_DS[$jid]}" = "$ds" ] && scancel "$jid"
    done
    JID=$(sbatch --parsable --account="$ACCT" "$WORKER" complexity "$ds")
    sbatch --account="$ACCT" --dependency=afterany:"$JID" "$WORKER" similarity "$ds" >/dev/null
    echo "    complexity=$JID builds precompute; similarity waits on it; noise not submitted"
  else
    for jid in "${!STUCK_JOB[@]}"; do
      [ "${JOB_DS[$jid]}" = "$ds" ] || continue
      cat=${JOB_CAT[$jid]}
      if [ "$cat" = "noise" ]; then
        echo "[$ds] done -> cancel noise (job $jid); NOT resubmitting (noise held)"
        scancel "$jid"
      else
        echo "[$ds] done -> cancel $cat (job $jid) + resubmit $cat (fresh categories untouched)"
        scancel "$jid"
        sbatch --account="$ACCT" "$WORKER" "$cat" "$ds" >/dev/null
      fi
    done
  fi
done
echo "Done. (noise held for your rework — resubmit noise separately when ready)"
