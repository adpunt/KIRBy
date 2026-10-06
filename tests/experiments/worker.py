#!/usr/bin/env python3
"""What every server job runs: take the next dataset from the ledger, run its
current stage one experiment at a time, record each one, close the stage,
move on. Also what an srun runs by hand.

    python tests/experiments/worker.py                      next dataset nobody holds
    python tests/experiments/worker.py --dataset qm9        this dataset, even if held
    python tests/experiments/worker.py --dataset herg_ki --experiment c07
    python tests/experiments/worker.py --dataset herg_ki --stage noise
    python tests/experiments/worker.py --dry-run            what would run, nothing runs

The run is tests/results/CURRENT_RUN (written by ledger.py init), or --run,
or KIRBY_RUN. The ledger is tests/experiments/orchestrate/ledger.py; the
document it keeps is <run>/LEDGER.md.

A unit already ok, or running in a job that is still alive, is never run
again. A failed unit is run once more, then left failed. A job given
--dataset joins a dataset another job holds; the per-unit marks keep the two
on different experiments. The memory this job has (SLURM_MEM_PER_NODE)
decides which datasets it may take: 64G jobs never take a 128G dataset.
"""
import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.orchestrate import ledger as L_mod, plan, runner   # noqa: E402


def remaining_wall_s():
    """Seconds left on this SLURM job, or None."""
    jid = os.environ.get('SLURM_JOB_ID')
    if not jid:
        return None
    try:
        r = subprocess.run(['squeue', '-h', '-j', jid, '-o', '%L'],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    txt = r.stdout.strip().splitlines()
    txt = [t for t in txt if t and not t.startswith('CLUSTER')]
    if r.returncode != 0 or not txt:
        return None
    return parse_slurm_time(txt[-1].strip())


def parse_slurm_time(s):
    """'1-03:33:52' | '11:19:19' | '19:19' | 'UNLIMITED' -> seconds or None."""
    if not s or s in ('UNLIMITED', 'NOT_SET', 'INVALID'):
        return None
    days = 0
    if '-' in s:
        d, s = s.split('-', 1)
        days = int(d)
    parts = [int(x) for x in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def unit_estimate(state, stage_name, unit_key):
    """The longest recorded time of this experiment in this stage on any
    row, or None."""
    exp = unit_key.split('__', 1)[0]
    best = None
    for r in state['rows'].values():
        s = L_mod.Ledger.stage_of(r, stage_name)
        if not s:
            continue
        for k, u in s['units'].items():
            if k.split('__', 1)[0] == exp and u.get('seconds'):
                best = max(best or 0, u['seconds'])
    return best


def pick_unit(state, stage, units, remaining):
    """The first unit in stage order that should fit the time left; when
    none is known to fit, the first (its checkpoints resume next time)."""
    if remaining is None:
        return units[0]
    for k in units:
        est = unit_estimate(state, stage['name'], k)
        if est is None or est * 1.2 < remaining:
            return k
    return units[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None)
    ap.add_argument('--dataset', default=None, help='row to work on: a dataset, opig:<name> or cross:<name>')
    ap.add_argument('--from-job-name', default=None,
                    help='a pre-ledger SLURM job name (x_<cat>_<ds>, opig_<name>, pc_<ds>_x<N>): work that row first')
    ap.add_argument('--then-any', action='store_true',
                    help='after the named row has no more work for this job, go on to any other row')
    ap.add_argument('--stage', default=None, help='only this stage of the row')
    ap.add_argument('--experiment', default=None, help='only this unit (e.g. c07, n16__gaussian, search_f2); resets a failed one')
    ap.add_argument('--one-stage', action='store_true', help='stop when the stage closes instead of going on to the next')
    ap.add_argument('--max-units', type=int, default=None)
    ap.add_argument('--max-rows', type=int, default=None, help='how many datasets to take in turn (default: until no work)')
    ap.add_argument('--sleep', type=float, default=0, help='seconds to wait before the first claim')
    ap.add_argument('--mem-gb', type=float, default=None, help='override the memory this job has')
    ap.add_argument('--no-sync', action='store_true', help='do not read result files on disk before claiming')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--unit-timeout', type=float, default=None, help='seconds per unit, then it is killed and marked failed')
    a = ap.parse_args(argv)

    root = L_mod.resolve_run(a.run)
    if root is None:
        sys.exit('no run: pass --run, set KIRBY_RUN, or run ledger.py init')
    L = L_mod.Ledger(root)
    job = L_mod.this_job()
    host = socket.gethostname()
    mem_gb = a.mem_gb if a.mem_gb is not None else L_mod.this_mem_gb()
    print(f"worker {job} on {host}  run {root}  mem {mem_gb or 'unknown'} GB"
          f"  {'targeted ' + a.dataset if a.dataset else 'next free dataset'}", flush=True)

    if a.from_job_name:
        L.load()
        row = plan.row_from_job_name(a.from_job_name, L.state['rows'])
        if row is None:
            print(f"job name {a.from_job_name!r} names no row of this run; taking the next free dataset", flush=True)
            a.then_any = True
        else:
            print(f"job name {a.from_job_name!r} -> row {row}", flush=True)
            a.dataset = row

    if a.dry_run:
        L.load()
        rows = L.state['rows']
        names = [a.dataset] if a.dataset else sorted(rows, key=lambda k: (rows[k]['priority'], k))
        for name in names:
            r = L.row(name)
            s, units = L.work_stage(r, mem_gb, a.stage, a.experiment)
            if s is None:
                print(f"{name:28s} complete")
                continue
            c = r['claim']
            held = f" held by {c['job']}" if c and L_mod.job_alive(c['job'], c.get('heartbeat')) else ''
            print(f"{name:28s} {r['mem']:5s} stage {s['name']:10s} {len(units):3d} runnable"
                  f"{': ' + ', '.join(units[:4]) if units else ''}{held}")
        return

    if a.sleep:
        print(f'sleeping {a.sleep:.0f}s before the first claim', flush=True)
        time.sleep(a.sleep)

    n_units = 0
    n_rows = 0
    while True:
        # The slow parts come first, without the lock: the result-file scan
        # (imports the suite modules) and the one squeue call for liveness.
        found = {}
        if not a.no_sync:
            L.load()
            found = L.scan_disk()
        L_mod.refresh_live_jobs(force=True)
        with locked_patiently(L):
            if found:
                L.apply_scan(found)
            if a.experiment and a.dataset:
                row = L.row(a.dataset)
                for s in row['stages']:
                    if (not a.stage or s['name'] == a.stage) and a.experiment in s['units']:
                        u = s['units'][a.experiment]
                        if u['status'] == 'failed' or (u['status'] == 'running' and not L_mod.job_alive(u.get('job'), u.get('heartbeat'))):
                            L.reset_unit(a.dataset, a.experiment, s['name'])
            row_name = L.claim(job, host, a.dataset, mem_gb)
        if row_name is None:
            print('no dataset with runnable work for this job; done', flush=True)
            break
        n_rows += 1
        print(f"=== took {row_name}", flush=True)
        work_row(L, row_name, job, host, mem_gb, a)
        n_units = _DONE[0]
        with locked_patiently(L):
            L.release(row_name, job)
        if a.dataset:
            if not a.then_any or a.experiment:
                break
            print(f"=== {a.dataset} has no more work for this job; going on to any dataset", flush=True)
            a.dataset = None
        if a.max_rows and n_rows >= a.max_rows:
            break
        if a.max_units and n_units >= a.max_units:
            break
    print(f'worker {job} finished: {n_rows} row(s)', flush=True)


_DONE = [0]      # units this process has finished


def locked_patiently(L):
    """L.locked(), but a lock that stays busy is waited for, not given up on:
    a worker must not die because thirty others started in the same minute."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        while True:
            try:
                with L.locked() as st:
                    yield st
                return
            except TimeoutError as exc:
                print(f'    {exc}; waiting', flush=True)
                time.sleep(30)
    return _cm()


def work_row(L, row_name, job, host, mem_gb, a):
    while True:
        remaining = remaining_wall_s()
        L_mod.refresh_live_jobs()
        with locked_patiently(L):
            row = L.row(row_name)
            stage, units = L.work_stage(row, mem_gb, a.stage, a.experiment)
            if stage is None:
                print(f"{row_name}: every stage closed", flush=True)
                return
            wait_for_running = False
            if not units:
                running = [k for k, u in stage['units'].items() if u['status'] == 'running'
                           and L_mod.job_alive(u.get('job'), u.get('heartbeat'))]
                waits = [f'{d}/{s}' for d, s in stage.get('after', []) if not L.stage_done(d, s)]
                why = (f"{len(running)} unit(s) running in other jobs ({', '.join(running[:3])})" if running
                       else f"waits for {', '.join(waits[:4])}" if waits
                       else 'nothing runnable (failed twice, or memory class too big for this job)')
                print(f"{row_name} {stage['name']}: {why}", flush=True)
                if running and a.dataset and not a.experiment:
                    # Sent to this dataset: what it waits for is being built
                    # by other jobs (search folds, c02 before c08). Wait for
                    # them rather than leave with work still to come (seen
                    # 2026-10-06: two herg_ki workers left after one fold).
                    wait_for_running = True
                else:
                    return
            else:
                unit = pick_unit(L.state, stage, units, remaining)
                log = str(runner.log_path(L.root, row_name, stage['name'], unit))
                if not L.start_unit(row_name, stage['name'], unit, job, host, log=log):
                    continue
        if wait_for_running:
            time.sleep(60)
            continue
        if remaining is not None:
            print(f"    time left on this job {remaining / 3600:.1f} h", flush=True)

        def beat():
            with locked_patiently(L):
                L.unit_heartbeat(row_name, stage['name'], unit, job)

        res = runner.run_unit(row_name, row, stage['name'], unit, L.root,
                              heartbeat=beat, timeout_s=a.unit_timeout)
        with locked_patiently(L):
            L.finish_unit(row_name, stage['name'], unit, res['status'],
                          seconds=res['seconds'], rc=res['rc'], error=res['error'], log=res['log'])
            closed = L.stage_of(L.row(row_name), stage['name'])['status']
        _DONE[0] += 1
        if a.experiment:
            return
        if a.max_units and _DONE[0] >= a.max_units:
            return
        if a.one_stage and closed in ('done', 'done_with_failures'):
            print(f"{row_name} {stage['name']}: {closed}", flush=True)
            return


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:          # e.g. --dry-run | head
        try:
            sys.stdout.close()
        except Exception:
            pass
        os._exit(0)
