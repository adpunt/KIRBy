#!/usr/bin/env python3
"""Stop the searches still running on datasets cut from the live run.

    python tests/tools/stop_cut_searches.py            # stop them
    python tests/tools/stop_cut_searches.py --dry-run  # only list them

`ledger.py cut` marks a cut dataset's units skipped, but a job that started
before the cut runs older code and carries on with its search until it ends
or hits the wall. This finds those searches in the ledger (a skipped unit on a
cut row that still names a job and never recorded an exit) and stops only the
search process inside the job, with `srun --jobid=<job> --overlap pkill`. The
job itself keeps running and goes on to another dataset.

Only jobs on the cluster this is run from can be reached; the others are
listed, to run the same command from that cluster's login node.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def this_cluster():
    try:
        out = subprocess.run(['scontrol', 'show', 'config'], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return None
    for ln in out.splitlines():
        if ln.split('=')[0].strip() == 'ClusterName':
            return ln.split('=', 1)[1].strip()
    return None


def live_jobs(cluster):
    try:
        out = subprocess.run(['squeue', '-M', cluster, '-h', '-u', subprocess.os.environ.get('USER', ''),
                              '-t', 'R', '-o', '%i'], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return None
    return {ln.strip() for ln in out.splitlines() if ln.strip().isdigit()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--run', default=None, help='run root (default: tests/results/CURRENT_RUN)')
    a = ap.parse_args()
    run = Path(a.run or (REPO / 'tests/results/CURRENT_RUN').read_text().strip())
    rows = json.loads((run / 'ledger.json').read_text())['rows']
    todo = []
    for ds, row in rows.items():
        if not row.get('cut'):
            continue
        for st in row['stages']:
            for k, u in st['units'].items():
                if (u['status'] == 'skipped' and u.get('job') and u.get('rc') is None
                        and str(u.get('error', '')).startswith('cut:')):
                    cluster, jobid = u['job'].split(':')[:2]
                    todo.append((ds, k, cluster, jobid))
    if not todo:
        print('No search on a cut dataset is recorded as still running.')
        return
    here = this_cluster()
    live = live_jobs(here) if here else None
    other = []
    for ds, unit, cluster, jobid in todo:
        what = f'{ds} {unit} in job {cluster}:{jobid}'
        if cluster != here:
            other.append(what)
            continue
        if live is not None and jobid not in live:
            print(f'gone already: {what}')
            continue
        fold = unit.split('search_f')[-1]
        pattern = f'search_unit.py {ds} --fold {fold}' if unit.startswith('search_f') else ds
        cmd = ['timeout', '120', 'srun', f'--jobid={jobid}', '--overlap', 'pkill', '-f', pattern]
        if a.dry_run:
            print('would run: ' + ' '.join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        # pkill: 0 = stopped, 1 = no such process (already finished)
        msg = {0: 'stopped', 1: 'not running any more'}.get(r.returncode,
                                                            f'FAILED (exit {r.returncode}) {r.stderr.strip()[:200]}')
        print(f'{msg}: {what}')
    if other:
        print(f'\nOn the other cluster (run this same command from its login node):')
        for w in other:
            print('  ' + w)


if __name__ == '__main__':
    main()
