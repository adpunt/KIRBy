#!/usr/bin/env python3
"""One line per dataset: is its precompute finished, and if not, is a job queued.

Written 2026-09-25. The coverage table (tests/precompute_coverage.py) counts
rows, and rows cannot say whether a pass finished: a pruned pass finishes with
fewer rows than the full grid asks for, and the knn and ridge rows are counted
as stored but not as asked. So this reads the precompute's OWN verdict instead.

WHAT IT READS.
  1. squeue on arc and htc: every pc_<dataset>_* job queued or running.
  2. sacct on arc and htc: the most recent pc_<dataset>_* job that has ended,
     and its log, slurm-<jobid>.out in the repo root.
  3. In that log, per pass: the line precompute_hybrids.py prints at the end,
     "A/B searches written", and the pass's exit code. A search counts as
     written only when find_best_hybrid returned for it, which is after its
     last configuration.

VERDICT PER DATASET.
  DONE       the latest ended job reached the end of its script, and every pass
             in it wrote every search it asked for, exit 0.
  NOT DONE   anything else. The reason is printed: the pass the job was cut off
             in (it ran out of wall), or the first failure message.
  Then what is queued for it on either cluster. A dataset NOT DONE with nothing
  queued is marked ⛔ -- that is the one that needs a job.

READS ONLY. Submits nothing, cancels nothing, opens no database.

Usage (login node is fine, no setup.sh needed):
    python tests/tools/precompute_status.py
    python tests/tools/precompute_status.py --since 2026-09-24 caco2 herg_ki
"""
import argparse
import ast
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXCLUDED = {'chembl_qsar', 'davis'}   # same exclusion as submit_everything.sh


def all_datasets():
    src = (REPO / 'tests/experiments/complexity/complexity_helpers.py').read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == 'DATASET_REPS' for t in node.targets):
            return sorted({k.value for k in node.value.keys} - EXCLUDED)
    raise SystemExit('DATASET_REPS not found in complexity_helpers.py')


def pc_dataset(name, datasets):
    """Dataset a pc_ job name belongs to; the three namings in use."""
    for ds in sorted(datasets, key=len, reverse=True):
        if re.fullmatch(rf'pc_{re.escape(ds)}_(general_.*|\d+|x\d+)', name):
            return ds
    return None


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  WARNING: {" ".join(cmd[:3])} ... exited {r.returncode}: '
              f'{r.stderr.strip()[:200]}')
    return r.stdout


def queued(datasets, user):
    # One call per cluster: with -h, squeue -M a,b drops the CLUSTER: lines,
    # so a combined call cannot say which cluster a job is on.
    out = defaultdict(list)
    for cluster in ('arc', 'htc'):
        for line in run(['squeue', '-M', cluster, '-u', user, '-h',
                         '-o', '%i|%j|%T']).splitlines():
            parts = line.strip().split('|')
            if len(parts) != 3:
                continue
            ds = pc_dataset(parts[1], datasets)
            if ds:
                out[ds].append((cluster, parts[0], parts[2]))
    return out


def ended(datasets, user, since):
    """Latest ended pc_ job per dataset, as (end, jobid, state, cluster)."""
    best = {}
    for line in run(['sacct', '-M', 'arc,htc', '-u', user, '-S', since, '-X',
                     '-n', '-P', '-o', 'JobID,JobName,State,End,Cluster']
                    ).splitlines():
        parts = line.split('|')
        if len(parts) != 5:
            continue
        jid, name, state, end, cluster = parts
        if state.startswith(('PENDING', 'RUNNING', 'REQUEUED')) or end in ('', 'Unknown'):
            continue
        ds = pc_dataset(name, datasets)
        if ds and (ds not in best or end > best[ds][0]):
            best[ds] = (end, jid.split('.')[0], state.split()[0], cluster)
    return best


def read_log(path):
    """Passes in one link's log: (label, written, asked, exit) and whether the
    script reached its end line."""
    passes, cur, first_fail = [], None, None
    reached_end = False
    for line in path.read_text(errors='replace').splitlines():
        m = re.match(r'---- (\S+ / .+) ----$', line)
        if m:
            cur = {'label': m.group(1).split(' / ', 1)[1], 'written': None,
                   'asked': None, 'exit': None}
            passes.append(cur)
            continue
        m = re.search(r'(\d+)/(\d+) searches written', line)
        if m and cur is not None:
            cur['written'], cur['asked'] = int(m.group(1)), int(m.group(2))
            continue
        m = re.match(r'---- exit=(\d+) ----', line)
        if m and cur is not None:
            cur['exit'] = int(m.group(1))
            continue
        if first_fail is None and (': FAILED after' in line
                                   or 'Traceback' in line
                                   or 'aborted before any work' in line):
            first_fail = line.strip()
        if re.match(r'\s*end=', line):
            reached_end = True
    return passes, reached_end, first_fail


def verdict(passes, reached_end, first_fail):
    if not passes:
        return False, 'no pass started' + (f' -- {first_fail[:160]}' if first_fail else '')
    bad = [p for p in passes if p['exit'] is not None
           and (p['exit'] != 0 or p['written'] != p['asked'])]
    if bad:
        p = bad[0]
        return False, (f'pass "{p["label"]}" wrote {p["written"]}/{p["asked"]} '
                       f'searches, exit {p["exit"]}'
                       + (f' -- {first_fail[:160]}' if first_fail else ''))
    if not reached_end:
        return False, f'cut off (out of wall) in pass "{passes[-1]["label"]}"'
    return True, f'{len(passes)} passes, every search written'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*')
    ap.add_argument('--since', default='2026-09-24',
                    help='oldest job end to consider (the result_db migration '
                         'ran 2026-09-24; every link before it failed)')
    args = ap.parse_args()
    user = os.environ.get('USER', '')
    datasets = args.datasets or all_datasets()

    q = queued(datasets, user)
    last = ended(datasets, user, args.since)

    n_done = n_waiting = 0
    gaps = []
    print(f'{len(datasets)} datasets.  Logs read from {REPO}/slurm-<jobid>.out\n')
    for ds in datasets:
        jobs = q.get(ds, [])
        qtxt = (', '.join(f'{c}:{j}' + ('(R)' if s == 'RUNNING' else '')
                          for c, j, s in jobs) if jobs else 'nothing queued')
        if ds in last:
            end, jid, state, cluster = last[ds]
            log = REPO / f'slurm-{jid}.out'
            if log.exists():
                ok, why = verdict(*read_log(log))
                src = f'job {jid} ({cluster}, {state}, ended {end})'
            else:
                ok, why = False, f'log {log.name} not found'
                src = f'job {jid} ({cluster}, {state})'
        else:
            ok, why, src = False, f'no pc_ job has ended since {args.since}', ''
        if ok:
            n_done += 1
            mark = 'DONE    '
        elif jobs:
            n_waiting += 1
            mark = 'NOT DONE'
        else:
            gaps.append(ds)
            mark = '⛔ NOT DONE'
        print(f'{mark:10s} {ds}')
        print(f'           {why}' + (f'   [{src}]' if src else ''))
        print(f'           queued: {qtxt}')

    print(f'\n{n_done} DONE.  {n_waiting} not done with a job queued.  '
          f'{len(gaps)} not done with NOTHING queued'
          + (f': {" ".join(gaps)}' if gaps else '.'))


if __name__ == '__main__':
    main()
