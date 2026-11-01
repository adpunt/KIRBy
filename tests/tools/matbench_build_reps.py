#!/usr/bin/env python3
"""Build and cache the Matbench base representations, one (task, rep) at a time.

Each (task, rep) runs in its own subprocess with a time limit (default 30 min),
--parallel of them at once (default 6).
Rows are featurised in 5 chunks and progress is logged per chunk, so a run
that hits the limit records how far it got. sine_coulomb and soap_avg run in one
piece: their width depends on every compound (largest cell, element set).

Output: /Volumes/seagate/matbench/reps/<task>/<rep>.npy and <rep>_names.json,
plus one JSON line per (task, rep) in /Volumes/seagate/matbench/reps/build_log.jsonl.

Run with the Matbench environment:
    /Volumes/seagate/matbench/venv/bin/python tests/tools/matbench_build_reps.py
    ... --tasks matbench_jdft2d --reps magpie,rdf --timeout 1800
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

OUT = Path(os.environ.get('KIRBY_MATBENCH_REPS', '/Volumes/seagate/matbench/reps'))   # the server sets it
N_CHUNKS = 5
WHOLE_DATASET_REPS = {'sine_coulomb', 'soap_avg'}
# Representations not computed through matminer's worker pool.
THREADED_REPS = {'soap_avg', 'chgnet', 'mace_mp'}
# Featurised one compound at a time with the formula logged first, so a build
# that stalls records which compound it was on. Same featurizer, same settings.
ROW_PROGRESS_REPS = {'ion_property'}
DEFER_FILE = OUT / 'DEFER'


def worker(task, rep):
    # Lets a running driver be drained without killing the builds it already
    # started: with DEFER_FILE present, every newly launched build exits here.
    if DEFER_FILE.exists() and not os.environ.get('MATBENCH_IGNORE_DEFER'):
        print('DEFERRED: queued behind the one-at-a-time rebuild',
              file=sys.stderr, flush=True)
        sys.exit(3)
    import numpy as np
    from kirby.datasets.matbench import load_matbench
    from kirby.representations.materials import COMPOSITION_REPS, MATERIALS_REPS

    d = load_matbench(task)
    entries = d['compositions'] if rep in COMPOSITION_REPS else d['structures']
    fn = MATERIALS_REPS[rep]
    t0 = time.time()
    if rep in WHOLE_DATASET_REPS:
        X, names = fn(entries, return_names=True)
    elif rep in ROW_PROGRESS_REPS:
        parts = []
        for i, e in enumerate(entries):
            print(f'CURRENT {i} {d["compositions"][i].formula} '
                  f'{time.time() - t0:.0f}s', flush=True)
            Xi, names = fn([e], return_names=True)
            parts.append(Xi)
        X = np.vstack(parts)
    else:
        bounds = np.linspace(0, len(entries), N_CHUNKS + 1).astype(int)
        parts = []
        for k in range(N_CHUNKS):
            Xk, names = fn(entries[bounds[k]:bounds[k + 1]], return_names=True)
            parts.append(Xk)
            print(f'PROGRESS {bounds[k + 1]}/{len(entries)} '
                  f'{time.time() - t0:.0f}s', flush=True)
        X = np.vstack(parts)
    out = OUT / task
    out.mkdir(parents=True, exist_ok=True)
    # names first, the matrix last and by rename: a build killed part way
    # (wall time) never leaves a .npy that looks finished
    (out / f'{rep}_names.json').write_text(json.dumps(names))
    tmp = out / f'{rep}.tmp.{os.getpid()}.npy'
    np.save(tmp, X)
    os.replace(tmp, out / f'{rep}.npy')
    n_zero_rows = int((~X.any(axis=1)).sum())
    print('RESULT ' + json.dumps({
        'n_rows': int(X.shape[0]), 'n_features': int(X.shape[1]),
        'all_zero_rows': n_zero_rows, 'seconds': round(time.time() - t0, 1)}),
        flush=True)


def main():
    from kirby.datasets.matbench import MATBENCH_TASKS, COMPOSITION_ONLY_TASKS
    from kirby.representations.materials import COMPOSITION_REPS, STRUCTURE_REPS

    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(MATBENCH_TASKS))
    ap.add_argument('--reps', default=None)
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--parallel', type=int, default=6,
                    help='(task, rep) builds run at once')
    ap.add_argument('--worker', nargs=2, metavar=('TASK', 'REP'))
    ap.add_argument('--pairs', default=None,
                    help='task:rep,task:rep,... built in this order, cached or not '
                         'skipped as usual')
    ap.add_argument('--label', default=None,
                    help="written into each log record, e.g. 'alone'")
    args = ap.parse_args()
    if args.worker:
        worker(*args.worker)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / 'build_log.jsonl'
    jobs = []
    if args.pairs:
        for pr in args.pairs.split(','):
            task, rep = pr.split(':')
            if (OUT / task / f'{rep}.npy').exists():
                print(f'{task} {rep}: cached', flush=True)
            else:
                jobs.append((task, rep))
    for task in ([] if args.pairs else args.tasks.split(',')):
        reps = list(COMPOSITION_REPS)
        if task not in COMPOSITION_ONLY_TASKS:
            reps += list(STRUCTURE_REPS)
        if args.reps:
            reps = [r for r in reps if r in args.reps.split(',')]
        for rep in reps:
            if (OUT / task / f'{rep}.npy').exists():
                print(f'{task} {rep}: cached', flush=True)
            else:
                jobs.append((task, rep))

    lock = threading.Lock()

    def build_one(task, rep):
        t0 = time.time()
        rec = {'task': task, 'rep': rep}
        progress = None
        # Several builds share the 8 cores, so each numpy gets one thread
        # (two for the representations that do their own numerical work).
        # Without this, load average reached 45-54 on 2026-10-23.
        env = dict(os.environ, MATBENCH_IGNORE_DEFER='1')
        if args.label:
            rec['run'] = args.label
            rec['parallel'] = args.parallel
        # One build at a time (--parallel 1) leaves numpy's own thread count.
        n_threads = '2' if rep in THREADED_REPS else '1'
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                  'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            if args.parallel > 1:
                env[v] = n_threads
        try:
            p = subprocess.run(
                [sys.executable, __file__, '--worker', task, rep],
                capture_output=True, text=True, timeout=args.timeout,
                env=env)
            lines = p.stdout.splitlines()
            res = [l for l in lines if l.startswith('RESULT ')]
            if p.returncode == 0 and res:
                rec.update(json.loads(res[-1][7:]), status='ok')
            else:
                err = (p.stderr.strip().splitlines() or ['?'])[-1]
                rec.update(status='failed', error=err)
            progress = next((l for l in reversed(lines)
                             if l.startswith('PROGRESS ')), None)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
            progress = next((l for l in reversed(out.splitlines())
                             if l.startswith('PROGRESS ')), None)
            rec.update(status='timeout')
            cur = next((l for l in reversed(out.splitlines())
                        if l.startswith('CURRENT ')), None)
            if cur:
                rec['stuck_on'] = cur[8:]
        rec['wall_seconds'] = round(time.time() - t0, 1)
        if rec['status'] != 'ok':
            rec['last_progress'] = progress
        with lock:
            with open(log, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for fut in [ex.submit(build_one, t, r) for t, r in jobs]:
            fut.result()

if __name__ == '__main__':
    main()
