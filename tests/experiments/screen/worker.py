#!/usr/bin/env python3
"""What every screening job runs, queued or by hand in an srun: take the next
candidate dataset with work, run its units one at a time, record each one,
go on until nothing is left for this job or the job ends.

    python tests/experiments/screen/worker.py                     next dataset with work
    python tests/experiments/screen/worker.py --dataset dronerf   this dataset, even if another job holds it
    python tests/experiments/screen/worker.py --dataset dronerf --stage build
    python tests/experiments/screen/worker.py --dataset dronerf --unit spectrum__rf__10__grouped
    python tests/experiments/screen/worker.py --dry-run

The run is tests/results/CURRENT_SCREEN (written by ledger.py init), --run,
or KIRBY_SCREEN. The unit loop is the experiments' worker's own
(tests/experiments/worker.py work_row): a unit already ok, or running in a job
that is alive, never runs again; a failed unit runs once more; a unit whose
job died (wall time, OOM, an srun closed) runs again and resumes from what it
wrote. Nothing here holds state between units except the ledger file.
"""
import argparse
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments import worker as exp_worker                  # noqa: E402  tests/experiments/worker.py
from experiments.orchestrate import ledger as L_mod, runner   # noqa: E402
from experiments.screen import datasets as D                  # noqa: E402
from experiments.screen.ledger import ScreenLedger, resolve_run   # noqa: E402

# The runner's subprocess loop (heartbeat every 5 min, exit-code table, log
# per unit) is used as it is; only the command a unit runs is ours.
runner.command_for = D.command_for


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--stage', default=None, help='build, evaluate or summarise')
    ap.add_argument('--unit', default=None, help='only this unit; resets it when it failed')
    ap.add_argument('--max-units', type=int, default=None)
    ap.add_argument('--sleep', type=float, default=0)
    ap.add_argument('--mem-gb', type=float, default=None)
    ap.add_argument('--no-sync', action='store_true')
    ap.add_argument('--unit-timeout', type=float, default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)

    root = resolve_run(a.run)
    if root is None:
        sys.exit('no screen run: pass --run, set KIRBY_SCREEN, or run ledger.py init')
    # The interpreter paths, for a worker started by hand without sourcing the file
    # (a Matbench unit under the wrong Python fails with No module named 'matbench').
    env_file = REPO / 'tests' / 'results' / 'SCREEN_ENV'
    if env_file.exists():
        for ln in env_file.read_text().splitlines():
            if ln.startswith('export ') and '=' in ln:
                k, v = ln[len('export '):].split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    L = ScreenLedger(root)
    job = L_mod.this_job()
    host = socket.gethostname()
    mem_gb = a.mem_gb if a.mem_gb is not None else L_mod.this_mem_gb()
    print(f"screen worker {job} on {host}  run {root}  mem {mem_gb or 'unknown'} GB  "
          f"{'dataset ' + a.dataset if a.dataset else 'next dataset with work'}", flush=True)

    if a.dry_run:
        L.load()
        rows = L.state['rows']
        names = [a.dataset] if a.dataset else sorted(rows, key=lambda k: (rows[k]['priority'], k))
        for name in names:
            r = L.row(name)
            s, units = L.work_stage(r, mem_gb, a.stage, a.unit)
            if s is None:
                print(f'{name:12s} complete')
                continue
            print(f"{name:12s} stage {s['name']:9s} {len(units):4d} runnable"
                  f"{': ' + ', '.join(units[:4]) if units else ''}")
            for k in units[:3]:
                argv_, env, _ = D.command_for(name, r, s['name'], k, root)
                print(f"    {k}: {' '.join(argv_)}")
        return

    if a.sleep:
        print(f'sleeping {a.sleep:.0f}s before the first claim', flush=True)
        time.sleep(a.sleep)

    # work_row reads these attribute names (tests/experiments/worker.py).
    wa = argparse.Namespace(dataset=a.dataset, stage=a.stage, experiment=a.unit,
                            max_units=a.max_units, one_stage=False, unit_timeout=a.unit_timeout)
    n_rows = 0
    while True:
        found = {}
        if not a.no_sync:
            L.load()
            found = L.scan_disk()
        L_mod.refresh_live_jobs(force=True)
        with exp_worker.locked_patiently(L):
            if found:
                L.apply_scan(found)
            if a.unit and a.dataset:
                row = L.row(a.dataset)
                for s in row['stages']:
                    u = s['units'].get(a.unit)
                    if u and (u['status'] == 'failed' or (u['status'] == 'running' and not
                                                         L_mod.job_alive(u.get('job'), u.get('heartbeat')))):
                        L.reset_unit(a.dataset, a.unit, s['name'])
            row_name = L.claim(job, host, a.dataset, mem_gb)
        if row_name is None:
            print('no dataset with runnable work for this job; done', flush=True)
            break
        n_rows += 1
        print(f'=== took {row_name}', flush=True)
        exp_worker.work_row(L, row_name, job, host, mem_gb, wa)
        with exp_worker.locked_patiently(L):
            L.release(row_name, job)
        if a.dataset or (a.max_units and exp_worker._DONE[0] >= a.max_units):
            break
    print(f'screen worker {job} finished: {n_rows} dataset(s), {exp_worker._DONE[0]} unit(s)', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        os._exit(0)
