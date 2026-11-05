# Running both ledgers: copy and paste

`S` is the screening sweep (`screen/`). `P` is Papers 2 and 3 (`papers/`).
Cluster: **htc**. ARC's guide puts low-core jobs there, and these are 8 CPUs each.

## Every login
```
cd /data/stat-ecr/scat9264/KIRBy
git pull
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
```

## Queue more jobs (login node; change 10 to how many)
```
bash tests/slurm_scripts/submit_screen_workers.sh 10 -- --clusters=htc --exclude=htc-g011 -p medium,long
```
```
bash tests/slurm_scripts/submit_papers_workers.sh 10 -- --clusters=htc --exclude=htc-g011 -p medium,long
```
One dataset only:
```
bash tests/slurm_scripts/submit_screen_workers.sh 2 --dataset dronerf -- --clusters=htc --exclude=htc-g011 -p medium,long
```
```
bash tests/slurm_scripts/submit_papers_workers.sh 2 --dataset qm9 -- --clusters=htc --exclude=htc-g011 -p medium,long
```
If htc is full, the same jobs on arc:
```
bash tests/slurm_scripts/submit_screen_workers.sh 10 -- --clusters=arc -p medium
```
```
bash tests/slurm_scripts/submit_papers_workers.sh 10 -- --clusters=arc -p medium
```
Each job gets 47:59:00, 8 CPUs and 64G. Is there room for that shape on the cluster you are logged into:
```
bash tests/slurm_scripts/where_to_submit.sh --fits 8 64G
```

## Track queued jobs
```
squeue --clusters=all -u $USER -o "%.12i %.9P %.16j %.2t %.10M %.10l %.5C %.6m %R"
```
Counts per job name; R is running, PD is waiting (`ks` is S, `kp` is P):
```
squeue --clusters=all -u $USER -h -o "%j %t" | grep -E "^k[sp]" | sort | uniq -c
```
Estimated start times:
```
squeue --clusters=all -u $USER --start
```
One job's output:
```
tail -30 logs/screen_workers/ks_JOBID.out
```
```
tail -30 logs/papers_workers/kp_JOBID.out
```

## Progress
```
python tests/experiments/screen/ledger.py progress
```
```
python tests/experiments/papers/ledger.py progress
```
Failed units and why a step is held:
```
python tests/experiments/screen/ledger.py status | less
```
```
python tests/experiments/papers/ledger.py status | less
```

## Interactive session
Start it from the login node:
```
srun --clusters=htc --exclude=htc-g011 -p interactive --cpus-per-task=8 --mem=64G --time=04:00:00 --pty /bin/bash
```
Inside, for S:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
[ -f tests/results/SCREEN_ENV ] && . tests/results/SCREEN_ENV
python tests/experiments/screen/worker.py
```
Inside, for P:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/experiments/papers/worker.py
```
One dataset: add `--dataset NAME` to the worker line.

ARC's guide says the interactive partition is for pre- and post-processing and building
software. For long worker sessions, the same shell on `short` runs up to 12 hours but can wait in the queue:
```
srun --clusters=htc --exclude=htc-g011 -p short --cpus-per-task=8 --mem=64G --time=11:59:00 --pty /bin/bash
```

## P only: open the next steps
A step that is not open never starts. Queued P jobs end once nothing is left that may run.

Open the three searches:
```
python - <<'EOF'
import json, pathlib
p = pathlib.Path(open('tests/results/CURRENT_PAPERS').read().strip()) / 'RUN_SETTINGS.json'
s = json.loads(p.read_text())
s['open'].update(search_pruned=True, search_fast=True, search_full=True)
p.write_text(json.dumps(s, indent=1))
print(s['open'])
EOF
```
Open Paper 2, once you have chosen the search the experiments read. Replace CHOSEN with `pruned`, `fast` or `full`:
```
python - <<'EOF'
import json, pathlib
p = pathlib.Path(open('tests/results/CURRENT_PAPERS').read().strip()) / 'RUN_SETTINGS.json'
s = json.loads(p.read_text())
s['experiments_search'] = 'CHOSEN'
s['open'].update(paper2=True)
p.write_text(json.dumps(s, indent=1))
print(s['experiments_search'], s['open'])
EOF
```

## When a unit failed
Read its error, then rerun it once the code is fixed:
```
python tests/experiments/papers/ledger.py reset DATASET UNIT
```
```
python tests/experiments/screen/ledger.py reset DATASET UNIT
```
