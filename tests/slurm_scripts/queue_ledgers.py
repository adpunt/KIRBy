#!/usr/bin/env python3
"""Queue jobs for both ledgers, and release the old held jobs into them.

    python3 tests/slurm_scripts/queue_ledgers.py check         submits and changes nothing
    python3 tests/slurm_scripts/queue_ledgers.py submit 20     20 jobs
    python3 tests/slurm_scripts/queue_ledgers.py release 5     5 held jobs (or: release all)

Add --dry-run to submit or release to print the commands instead. Standard library
only, so it runs on the login node before setup.sh. No job names a dataset, and every
job runs ledgers_worker.sh.

QUEUE TIME. `sbatch --test-only` asks each cluster when a job of this shape would start,
and submits nothing. It is asked for arc and htc, for medium,long at two walls and short
at its limit. On each cluster the job takes the earliest start. A wall under 47:59:00 is
taken only when it starts more than PREFER_LONG_H hours sooner, because a job killed at
its wall counts as an attempt on the unit it was running, and a unit gets 4 such attempts
(MAX_ATTEMPTS_STALE in orchestrate/ledger.py). The jobs are split evenly between arc and
htc: this is the scheduler's estimate, not a promise.

MEMORY. 8 CPUs and 64G. arc has 8 GB per core, so 64G sets aside no cores beyond the 8
asked for. 128G on 8 CPUs sets aside 16 cores' worth on arc, and waits longer for it. No
dataset in either ledger is in the memory class that needs 128G (plan.BIG_RE).

Never passes --account.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CPUS = 8
MEM = '64G'
MEM_MB = 64 * 1024
OLD_NAMES = ('x_', 'opig_', 'pc_')
# htc-g011 stops and restarts jobs about every four hours (WORKERS.md, 2026-11-02).
EXCLUDE = {'arc': [], 'htc': ['--exclude=htc-g011']}
CLUSTERS = ('arc', 'htc')
OPTIONS = (('medium,long', '47:59:00'), ('medium,long', '23:59:00'), ('short', '11:59:00'))
FULL_WALL = '47:59:00'
PREFER_LONG_H = 6
START_RE = re.compile(r'to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def shape(cluster, partition, wall):
    return [f'--clusters={cluster}', '-p', partition, f'--time={wall}',
            f'--cpus-per-task={CPUS}', f'--mem={MEM}'] + EXCLUDE[cluster]


def probe():
    """[{cluster, partition, wall, start (datetime or None), said}] for every option."""
    out = []
    for cluster in CLUSTERS:
        for partition, wall in OPTIONS:
            _, said = run(['sbatch', '--test-only'] + shape(cluster, partition, wall)
                          + ['--wrap=hostname'])
            m = START_RE.search(said)
            out.append({'cluster': cluster, 'partition': partition, 'wall': wall,
                        'start': datetime.fromisoformat(m.group(1)) if m else None, 'said': said})
    return out


def choose(options):
    """cluster -> the option its jobs take. A cluster that gave no estimate for its
    full wall and for every other wall is left out."""
    chosen = {}
    for cluster in CLUSTERS:
        known = [o for o in options if o['cluster'] == cluster and o['start']]
        if not known:
            continue
        best = min(known, key=lambda o: o['start'])
        full = next((o for o in known if o['wall'] == FULL_WALL), None)
        if full and (full['start'] - best['start']).total_seconds() <= PREFER_LONG_H * 3600:
            best = full
        chosen[cluster] = best
    return chosen


def split(n, clusters):
    """n jobs shared evenly; the first clusters take the remainder."""
    return {c: n // len(clusters) + (1 if i < n % len(clusters) else 0)
            for i, c in enumerate(clusters)}


def my_jobs():
    """Pending and running jobs on both clusters: {cluster, id, name, state, reason, mem_mb, dep}."""
    jobs = []
    for cluster in CLUSTERS:
        _, text = run(['squeue', f'--clusters={cluster}', '-u', os.environ.get('USER', ''),
                       '-h', '-o', '%i|%j|%t|%r|%m|%E'])
        for line in text.splitlines():
            parts = line.split('|')
            if len(parts) != 6:
                continue            # a "CLUSTER: arc" line
            jid, name, state, reason, mem, dep = parts
            jobs.append({'cluster': cluster, 'id': jid, 'name': name, 'state': state,
                         'reason': reason, 'mem_mb': mem_mb(mem), 'dep': dep})
    return jobs


def mem_mb(text):
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)B?', text.strip())
    if not m:
        return None
    return int(float(m.group(1)) * {'K': 1 / 1024, 'M': 1, '': 1, 'G': 1024, 'T': 1024 ** 2}[m.group(2)])


def closed_steps():
    """The Papers steps still closed, or None if the run's settings cannot be read."""
    try:
        run_root = (REPO / 'tests/results/CURRENT_PAPERS').read_text().strip()
        settings = json.loads((Path(run_root) / 'RUN_SETTINGS.json').read_text())
    except (OSError, ValueError):
        return None
    return [k for k, v in settings.get('open', {}).items() if not v]


def print_closed():
    closed = closed_steps()
    if closed is None:
        print('Papers steps open or closed: could not read RUN_SETTINGS.json')
    elif closed:
        print(f"Papers steps still closed, so no job runs them: {', '.join(closed)}")


def cmd_check(_a):
    options = probe()
    chosen = choose(options)
    print(f'One job: {CPUS} CPUs, {MEM}. Estimated start from sbatch --test-only; nothing submitted.\n')
    print(f"{'cluster':8s}{'partition':13s}{'wall':10s}{'estimated start':21s}")
    for o in options:
        mark = 'takes this' if chosen.get(o['cluster']) is o else ''
        start = o['start'].isoformat(sep=' ') if o['start'] else 'no estimate'
        print(f"{o['cluster']:8s}{o['partition']:13s}{o['wall']:10s}{start:21s}{mark}")
        if not o['start']:
            print(f"    slurm said: {o['said'][:200]}")
    jobs = my_jobs()
    counts = {}
    for j in jobs:
        kind = 'old held' if j['reason'] == 'JobHeldUser' else ('running' if j['state'] == 'R' else 'waiting')
        counts[(j['cluster'], kind)] = counts.get((j['cluster'], kind), 0) + 1
    print('\nYour jobs now, all names:')
    for c in CLUSTERS:
        print(f"  {c}: {counts.get((c, 'running'), 0)} running, {counts.get((c, 'waiting'), 0)} waiting, "
              f"{counts.get((c, 'old held'), 0)} held")
    print()
    print_closed()


def cmd_submit(a):
    chosen = choose(probe())
    if not chosen:
        # Without an estimate, the shape the guide has always used.
        print('No cluster gave an estimate: submitting medium,long at 47:59:00 on both.')
        chosen = {c: {'partition': 'medium,long', 'wall': FULL_WALL} for c in CLUSTERS}
    (REPO / 'logs/ledger_workers').mkdir(parents=True, exist_ok=True)
    stagger = 0
    for cluster, n in split(a.n, list(chosen)).items():
        o = chosen[cluster]
        print(f"{n} jobs on {cluster}: {o['partition']}, {o['wall']}, {CPUS} CPUs, {MEM}")
        for _ in range(n):
            cmd = (['sbatch', '--parsable'] + shape(cluster, o['partition'], o['wall'])
                   + ['--job-name=kl', '--output=logs/ledger_workers/kl_%j.out',
                      f'--export=ALL,KW_SLEEP={stagger}',
                      str(REPO / 'tests/slurm_scripts/ledgers_worker.sh')])
            stagger += 10
            if a.dry_run:
                print(' '.join(cmd))
                continue
            rc, said = run(cmd)
            print(f'submitted {said}' if rc == 0 else f"FAILED: {' '.join(cmd)}\n  {said}")
    print_closed()


def cmd_release(a):
    old = [j for j in my_jobs() if j['name'].startswith(OLD_NAMES) and j['state'] == 'PD']
    held = [j for j in old if j['reason'] == 'JobHeldUser']
    # Jobs released earlier get the same memory and dependency changes.
    earlier = [j for j in old if j['reason'] != 'JobHeldUser']
    take = held if a.n == 'all' else held[:int(a.n)]
    print(f'{len(held)} held jobs with old names; releasing {len(take)}.')
    for j in earlier + take:
        changes = []
        if j['mem_mb'] and j['mem_mb'] > MEM_MB:
            changes.append(f'MinMemoryNode={MEM_MB}')
        if j['dep'] not in ('(null)', ''):
            # A copy made to wait for its twin (rewire_held_jobs.py). The ledger stops
            # two jobs repeating a unit, so the wait only delays it.
            changes.append('Dependency=')
        for change in changes:
            step(a, ['scontrol', '-M', j['cluster'], 'update', f"JobId={j['id']}", change], j)
    for j in take:
        step(a, ['scontrol', '-M', j['cluster'], 'release', j['id']], j)
    print_closed()


def step(a, cmd, job):
    if a.dry_run:
        print(' '.join(cmd))
        return
    rc, said = run(cmd)
    print(f"{job['id']} {job['name']}: {cmd[4] if cmd[3] == 'update' else 'released'}"
          f"{'' if rc == 0 else ' FAILED: ' + said}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check').set_defaults(fn=cmd_check)
    p = sub.add_parser('submit')
    p.add_argument('n', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser('release')
    p.add_argument('n', help='how many, or all')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=cmd_release)
    a = ap.parse_args(argv)
    if a.cmd == 'release' and a.n != 'all' and not a.n.isdigit():
        sys.exit('release takes a number or all')
    a.fn(a)


if __name__ == '__main__':
    main()
