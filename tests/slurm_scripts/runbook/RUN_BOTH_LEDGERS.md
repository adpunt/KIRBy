# Running both ledgers: copy and paste

There are two ledgers, and each has its own jobs, progress table and run folder:

- **Screening sweep (`screen/`)**: scores each candidate dataset's views under its models. Its jobs are named `ks`.
- **Papers 2 and 3 (`papers/`)**: builds representations, runs the three searches, then runs the experiments. Its jobs are named `kp`.

Every job or interactive session does the same thing: it takes the next dataset with work,
runs its units one at a time and records each one in the ledger. Queued jobs and interactive
sessions can run at once without repeating work. If one dies, its unfinished unit runs again
in the next job or session that takes that dataset.

## Every login
Go to the repo, get the latest code and load the Python environment. `KIRBY_SKIP_INSTALLS=1`
stops `setup.sh` installing packages. Every command below assumes you did this.
```
cd /data/stat-ecr/scat9264/KIRBy
git pull
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
```

## Queue more jobs
Every job serves both ledgers and names no dataset. It takes whatever work is runnable, and
ends by itself when nothing is left, so a job queued too many costs nothing. Jobs are named `kl`.

See when a job would start on arc and htc, in each partition. This submits nothing. It also
lists your jobs and any Papers steps still closed.
```
python3 tests/slurm_scripts/queue_ledgers.py check
```
Queue 20 jobs. They are split evenly between arc and htc. On each cluster they take the
earliest start. A wall under 47:59:00 is taken only if it starts more than 6 hours sooner,
because a job killed at its wall uses up one of the 4 attempts its unit gets.
```
python3 tests/slurm_scripts/queue_ledgers.py submit 20
```
How many jobs the remaining units need, per step. It reads the ledgers and changes nothing.
```
python3 tests/slurm_scripts/queue_ledgers.py estimate
```
Each job gets 8 CPUs and 64G. On arc, 64G is exactly what 8 CPUs come with, so it does not
slow the job's start. No dataset in either ledger is in the memory class that needs 128G.

## Old held jobs
Jobs named `x_*`, `opig_*` or `pc_*` now run the same two ledgers as a `kl` job, with their
queue position kept. They never write to the old experiments' run. Release 5:
```
python3 tests/slurm_scripts/queue_ledgers.py release 5
```
Or all of them:
```
python3 tests/slurm_scripts/queue_ledgers.py release all
```
This also lowers each job from 128G to 64G, and removes any wait on its twin job. Where Slurm
refuses a change, the line says FAILED and the job keeps its old setting.

## Memory actually used
Peak memory of finished jobs on both clusters. `MaxRSS` is on the `.batch` lines.
```
sacct --clusters=arc,htc -u $USER -S 2026-11-01 -o JobID%20,JobName%20,ReqMem,MaxRSS,Elapsed,State%15 | grep -E "batch|JobID"
```

## Track queued jobs
All your jobs on both clusters. In the `ST` column, `R` is running and `PD` is waiting.
```
squeue --clusters=all -u $USER -o "%.12i %.9P %.16j %.2t %.10M %.10l %.5C %.6m %R"
```
How many jobs of each name are running and waiting:
```
squeue --clusters=all -u $USER -h -o "%j %t" | grep -E "^(k[spl]|x_|opig_|pc_)" | sort | uniq -c
```
When Slurm estimates waiting jobs will start:
```
squeue --clusters=all -u $USER --start
```
One job's output. Replace JOBID with the number from `squeue`.
```
tail -30 logs/ledger_workers/kl_JOBID.out
```
For an old-named job, this prints where its output goes:
```
scontrol -M arc show job JOBID | grep StdOut
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

Opening Paper 2 does not start its meta-learner over models with their own hybrids. Its units run
only on datasets you name. Replace DATASET below. Add to `datasets` at any time; the next job picks
it up. Details are in `PAPERS.md`.
```
python - <<'EOF'
import json, pathlib
p = pathlib.Path(open('tests/results/CURRENT_PAPERS').read().strip()) / 'RUN_SETTINGS.json'
s = json.loads(p.read_text())
s['meta_learner'].update(datasets=['DATASET'])
p.write_text(json.dumps(s, indent=1))
print(s['meta_learner'])
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
