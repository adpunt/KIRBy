# Running both ledgers: copy and paste

There are two ledgers, and each has its own jobs, progress table and run folder:

- **Screening sweep (`screen/`)**: scores each candidate dataset's views under its models. Its jobs are named `ks`.
- **Papers 2 and 3 (`papers/`)**: builds representations, runs the three searches, then runs the experiments. Its jobs are named `kp`.

Every job or interactive session does the same thing: it takes the next dataset with work,
runs its units one at a time and records each one in the ledger. Queued jobs and interactive
sessions can run at once without repeating work. If one dies, its unfinished unit runs again
in the next job or session that takes that dataset.

Use **htc**. ARC's guide puts low-core jobs there, and these jobs use 8 CPUs.

## Every login
Go to the repo, get the latest code and load the Python environment. `KIRBY_SKIP_INSTALLS=1`
stops `setup.sh` installing packages. Every command below assumes you did this.
```
cd /data/stat-ecr/scat9264/KIRBy
git pull
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
```

## Queue more jobs
Run these on the login node. The number after the script name is how many jobs to queue.
Each job gets 47:59:00, 8 CPUs and 64G.

`-p medium,long` lets the job run in either partition. `--exclude=htc-g011` keeps it off a
node that stopped and restarted jobs about every four hours.

Screening sweep, 10 jobs:
```
bash tests/slurm_scripts/submit_screen_workers.sh 10 -- --clusters=htc --exclude=htc-g011 -p medium,long
```
Papers 2 and 3, 10 jobs:
```
bash tests/slurm_scripts/submit_papers_workers.sh 10 -- --clusters=htc --exclude=htc-g011 -p medium,long
```
To keep jobs on one dataset, add `--dataset NAME`. Screening sweep, 2 jobs on dronerf:
```
bash tests/slurm_scripts/submit_screen_workers.sh 2 --dataset dronerf -- --clusters=htc --exclude=htc-g011 -p medium,long
```
Papers 2 and 3, 2 jobs on qm9:
```
bash tests/slurm_scripts/submit_papers_workers.sh 2 --dataset qm9 -- --clusters=htc --exclude=htc-g011 -p medium,long
```
If htc is full, send the same jobs to arc. Screening sweep:
```
bash tests/slurm_scripts/submit_screen_workers.sh 10 -- --clusters=arc -p medium
```
Papers 2 and 3:
```
bash tests/slurm_scripts/submit_papers_workers.sh 10 -- --clusters=arc -p medium
```
To check whether a job this size fits right now, on the cluster you are logged into:
```
bash tests/slurm_scripts/where_to_submit.sh --fits 8 64G
```

## Track queued jobs
All your jobs on both clusters. In the `ST` column, `R` is running and `PD` is waiting.
```
squeue --clusters=all -u $USER -o "%.12i %.9P %.16j %.2t %.10M %.10l %.5C %.6m %R"
```
How many jobs of each name are running and waiting:
```
squeue --clusters=all -u $USER -h -o "%j %t" | grep -E "^k[sp]" | sort | uniq -c
```
When Slurm estimates waiting jobs will start:
```
squeue --clusters=all -u $USER --start
```
One job's output. Replace JOBID with the number from `squeue`. A job on one dataset has that
dataset in its name, for example `kp_qm9_JOBID.out`. Screening sweep:
```
tail -30 logs/screen_workers/ks_JOBID.out
```
Papers 2 and 3:
```
tail -30 logs/papers_workers/kp_JOBID.out
```

## Progress
One row per dataset. Each step shows units finished out of units planned. `hours` adds up the
recorded time of that dataset's units. A unit from a dead job still counts as `running` until
another job picks it up.

Screening sweep:
```
python tests/experiments/screen/ledger.py progress
```
Papers 2 and 3:
```
python tests/experiments/papers/ledger.py progress
```
The longer version, with each failed unit's error and log. For Papers 2 and 3 it also says why
a step is held. Screening sweep:
```
python tests/experiments/screen/ledger.py status | less
```
Papers 2 and 3:
```
python tests/experiments/papers/ledger.py status | less
```

## Interactive session
This runs the same worker as a queued job, in a shell you can watch. Start the session from
the login node. It ends after 4 hours; whatever was unfinished runs again later.
```
srun --clusters=htc --exclude=htc-g011 -p interactive --cpus-per-task=8 --mem=64G --time=04:00:00 --pty /bin/bash
```
At the new prompt, for the screening sweep. The third line loads the file of interpreter paths, if the run has one.
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
[ -f tests/results/SCREEN_ENV ] && . tests/results/SCREEN_ENV
python tests/experiments/screen/worker.py
```
Or for Papers 2 and 3:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/experiments/papers/worker.py
```
To stay on one dataset, add `--dataset NAME` to the worker line.

ARC's guide says the interactive partition is for pre- and post-processing and building
software, so long worker sessions fall outside that. The same shell on `short` runs for up to
12 hours but can wait in the queue:
```
srun --clusters=htc --exclude=htc-g011 -p short --cpus-per-task=8 --mem=64G --time=11:59:00 --pty /bin/bash
```

## Papers 2 and 3 only: open the next steps
Representation builds always run. Every later step is closed until you open it, and a closed
step never starts. A queued job ends when nothing is left that it may run, so open a step before
queueing jobs for it. Opening a step does not need a restart: running jobs see it at their next claim.

Open the three searches. A dataset still waits until its representations have built without failure.
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
Open Paper 2 once you have chosen the search the experiments read. Replace CHOSEN with `pruned`,
`fast` or `full`. Each dataset's experiments then wait for that search's folds on that dataset.
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
A unit that fails runs once more by itself, then stays failed. Read its error in `status`. Once
the code is fixed and pulled, clear the unit's record so the next job runs it. Replace DATASET
and UNIT with the names `status` shows. Screening sweep:
```
python tests/experiments/screen/ledger.py reset DATASET UNIT
```
Papers 2 and 3:
```
python tests/experiments/papers/ledger.py reset DATASET UNIT
```
