#!/bin/bash
# run_jobs.sh [N]: runs the jobs in jobs.tsv, N at a time (default 6), each capped at 3 hours.
# Safe to start again after the srun ends: finished jobs are skipped; a job cut off by the end
# of the allocation has no .done file, so it runs again. Noise jobs skip noise levels already on
# disk and every job reads hybrid searches already in the cache, so a rerun resumes rather than restarts.
set -u
N=${1:-6}; CAP=${JOB_CAP_S:-10800}
cd /data/stat-ecr/scat9264/KIRBy_localtest
say(){ echo "$(date '+%F %T') $*" | tee -a $RUN/run.log; }
run_one(){ name=$1; cmd=$2; log=$RUN/logs/$name.log; t0=$(date +%s); start=$(date '+%F %T')
  # Appends: an attempt cut off by the end of an srun keeps its output above the next attempt's.
  echo "### $name attempt start $(date '+%F %T') cap ${CAP}s" >> $log; echo "### $cmd" >> $log
  timeout -k 30 $CAP bash -c "$cmd" >> $log 2>&1; rc=$?
  wall=$(( $(date +%s) - t0 )); nan=$(grep -ci '\bnan\b' $log)
  to=0; [ $wall -ge $CAP ] && to=1
  echo "### end rc=$rc wall=${wall}s timed_out=$to" >> $log
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$start" "$wall" "$rc" "$to" "$nan" >> $RUN/status.tsv
  touch $RUN/state/$name.done
  if [ $to = 1 ]; then say "TIMEOUT $name after ${wall}s"; elif [ $rc != 0 ]; then say "FAILED $name rc=$rc (log: $log)"; else say "OK $name ${wall}s"; fi
}
[ -f $RUN/status.tsv ] || printf "#name\tstart\twall_s\trc\ttimed_out\tnan_lines\n" > $RUN/status.tsv
say "start, $N at a time"
while IFS=$'\t' read -r name cmd; do
  case "$name" in ''|'#'*) continue;; esac
  [ -f $RUN/state/$name.done ] && continue
  while [ $(jobs -rp | wc -l) -ge $N ]; do sleep 10; done
  say "STARTED $name"; run_one "$name" "$cmd" < /dev/null &
done < $LT/jobs.tsv
wait
say "all done: $(ls $RUN/state/*.done | wc -l) of $(grep -vc '^#' $LT/jobs.tsv) jobs"
