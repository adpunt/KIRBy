#!/bin/bash
# run_jobs.sh [N]: runs the jobs in jobs.tsv, N at a time (default 6), each capped at 3 hours.
# Safe to start again after the srun ends: finished jobs are skipped; a job cut off by the end
# of the allocation has no .done file, so it runs again. Noise jobs skip noise levels already on
# disk and every job reads hybrid searches already in the cache, so a rerun resumes rather than restarts.
#
# Several sruns can run this at once on the same list. Before starting a job, a runner reserves it
# by creating state/<job>.claim (mkdir, which only one runner can win). While the job runs, its
# claim is touched every minute. A claim untouched for 10 minutes belongs to an srun that ended,
# and the next runner to reach that job takes it over.
set -u
N=${1:-6}; CAP=${JOB_CAP_S:-10800}; STALE=600
cd /data/stat-ecr/scat9264/KIRBy_localtest
HOST=$(hostname -s)
say(){ echo "$(date '+%F %T') [$HOST] $*" | tee -a $RUN/run.log; }
claim(){ c=$RUN/state/$1.claim
  mkdir $c 2>/dev/null && { touch $c/beat; return 0; }
  age=$(( $(date +%s) - $(stat -c %Y $c/beat 2>/dev/null || echo 0) ))
  [ $age -lt $STALE ] && return 1
  # Stale: rename it away first, so only one runner can take it over.
  mv $c $c.stale.$HOST.$$ 2>/dev/null || return 1
  rm -rf $c.stale.$HOST.$$
  mkdir $c 2>/dev/null && { touch $c/beat; say "TAKEOVER $1 (claim untouched ${age}s)"; return 0; }
  return 1
}
run_one(){ name=$1; cmd=$2; log=$RUN/logs/$name.log; t0=$(date +%s); start=$(date '+%F %T')
  c=$RUN/state/$name.claim
  ( while true; do touch $c/beat 2>/dev/null; sleep 60; done ) & hb=$!
  # Appends: an attempt cut off by the end of an srun keeps its output above the next attempt's.
  echo "### $name attempt start $(date '+%F %T') on $HOST cap ${CAP}s" >> $log; echo "### $cmd" >> $log
  timeout -k 30 $CAP bash -c "$cmd" >> $log 2>&1; rc=$?
  kill $hb 2>/dev/null
  wall=$(( $(date +%s) - t0 )); nan=$(grep -ci '\bnan\b' $log)
  to=0; [ $wall -ge $CAP ] && to=1
  echo "### end rc=$rc wall=${wall}s timed_out=$to" >> $log
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$start" "$wall" "$rc" "$to" "$nan" "$HOST" >> $RUN/status.tsv
  touch $RUN/state/$name.done; rm -rf $c
  if [ $to = 1 ]; then say "TIMEOUT $name after ${wall}s"; elif [ $rc != 0 ]; then say "FAILED $name rc=$rc (log: $log)"; else say "OK $name ${wall}s"; fi
}
[ -f $RUN/status.tsv ] || printf "#name\tstart\twall_s\trc\ttimed_out\tnan_lines\thost\n" > $RUN/status.tsv
say "start, $N at a time"
while IFS=$'\t' read -r name cmd; do
  case "$name" in ''|'#'*) continue;; esac
  [ -f $RUN/state/$name.done ] && continue
  while [ $(jobs -rp | wc -l) -ge $N ]; do sleep 10; done
  [ -f $RUN/state/$name.done ] && continue
  claim "$name" || continue
  say "STARTED $name"; run_one "$name" "$cmd" < /dev/null &
done < $LT/jobs.tsv
wait
say "this runner is done; finished $(ls $RUN/state/*.done 2>/dev/null | wc -l) of $(grep -vc '^#' $LT/jobs.tsv) jobs, $(ls -d $RUN/state/*.claim 2>/dev/null | wc -l) running elsewhere"
