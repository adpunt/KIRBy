#!/usr/bin/env python3
"""What every Papers 2 and 3 job runs, queued or by hand in an srun: take the
next dataset with work that is not held, run its units one at a time, record
each one, go on until nothing is left for this job or the job ends.

    python tests/experiments/papers/worker.py                     next dataset with work
    python tests/experiments/papers/worker.py --dataset herg_ki   this dataset, even if another job holds it
    python tests/experiments/papers/worker.py --dataset herg_ki --stage search_pruned
    python tests/experiments/papers/worker.py --dataset herg_ki --unit search_f2
    python tests/experiments/papers/worker.py --dry-run

The run is tests/results/CURRENT_PAPERS (written by ledger.py init), --run,
or KIRBY_PAPERS_RUN. The unit loop is the experiments' worker's own
(tests/experiments/worker.py work_row). Every unit runs with the environment
the run's RUN_SETTINGS.json gives (its own cache, the full search space, the
pruning mode), read when the unit starts.
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

from experiments import worker as exp_worker                  # noqa: E402
from experiments.orchestrate import ledger as L_mod, runner   # noqa: E402
from experiments.papers import plan as P                      # noqa: E402
from experiments.papers.ledger import (PapersLedger, resolve_run,   # noqa: E402
                                       read_settings, settings_env)


def command_for(row_name, row, stage_name, unit_key, run_root):
    """(argv, env, result_dir) for one unit."""
    run_root = Path(run_root)
    env = runner.common_env(run_root)
    env.pop('KIRBY_RUN', None)          # never the old experiments' ledger
    env['KIRBY_PAPERS_RUN'] = str(run_root)
    st = read_settings(run_root)
    search = None
    if stage_name in P.SEARCH_STEPS:
        search = stage_name[len('search_'):]
    elif stage_name in ('paper2', 'paper3', P.META_STEP):
        search = st.get('experiments_search')
        if not search:
            raise ValueError('experiments_search is not set in RUN_SETTINGS.json')
    env.update(settings_env(st, search))
    py = sys.executable
    ds = row['dataset']
    if stage_name == 'reps':
        # The general pool and the augmentation representations: the search
        # loads both (noise_helpers.load_noise_data). A dataset that borrows
        # another's pool (noise_helpers.REP_LIST_FROM: herg_ic50 uses
        # herg_ki's) is built from that list by name.
        argv = [py, '-u', str(TESTS / 'precompute_reps.py'), ds, '--include-augmentation']
        from experiments.noise.noise_helpers import REP_LIST_FROM
        if ds in REP_LIST_FROM:
            from experiments.complexity.complexity_helpers import DATASET_REPS
            argv = argv[:4] + ['--reps', ','.join(DATASET_REPS[REP_LIST_FROM[ds]]['general'])]
        return argv, env, None
    if stage_name in P.SEARCH_STEPS:
        m = P.SEARCH_UNIT_RE.match(unit_key)
        n = st.get('n_outer_folds') or 5
        return ([py, '-u', str(TESTS / 'experiments' / 'orchestrate' / 'search_unit.py'),
                 ds, '--fold', m.group(1), '--n-folds', str(n)], env, None)
    if stage_name in ('paper2', 'paper3'):
        exp = unit_key.split('__', 1)[0]
        cond = unit_key.split('__', 1)[1] if '__' in unit_key else None
        category = P.unit_category(unit_key)
        out = run_root / P.CATEGORY_DIR[category]
        argv = [py, '-u', str(TESTS / 'experiments' / 'run_dataset.py'),
                category, ds, '--experiments', exp, '--output', str(out)]
        if cond:
            argv += ['--noise-condition', cond]
        return argv, env, out
    if stage_name == P.META_STEP:
        m = st.get('meta_learner') or {}
        fold = P.META_UNIT_RE.match(unit_key).group(1)
        candidates = list(m['candidate_models']) + list(m['net_sizes'])
        out = run_root / P.META_DIR
        return ([py, '-u', str(HERE / 'meta_learner_own_hybrids.py'), ds, '--fold', fold,
                 '--n-folds', str(st.get('n_outer_folds') or 5),
                 '--candidates', ','.join(candidates),
                 '--inner-folds', str(m['inner_folds']),
                 '--min-rows', str(m['min_rows']),
                 '--output', str(out)], env, out)
    raise ValueError(f'no command for step {stage_name!r} yet')


# The runner's subprocess loop (heartbeat every 5 min, exit codes, a log per
# unit) is used as it is; only the command is ours.
runner.command_for = command_for


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--stage', default=None, help='reps, search_pruned, search_fast, search_full, paper2, paper3 or meta_learner')
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
        sys.exit('no papers run: pass --run, set KIRBY_PAPERS_RUN, or run ledger.py init')
    L = PapersLedger(root)
    job = L_mod.this_job()
    host = socket.gethostname()
    mem_gb = a.mem_gb if a.mem_gb is not None else L_mod.this_mem_gb()
    print(f"papers worker {job} on {host}  run {root}  mem {mem_gb or 'unknown'} GB  "
          f"{'dataset ' + a.dataset if a.dataset else 'next dataset with work'}", flush=True)

    if a.dry_run:
        L.load()
        rows = L.state['rows']
        names = [a.dataset] if a.dataset else sorted(rows, key=lambda k: (rows[k]['priority'], k))
        for name in names:
            r = L.row(name)
            s, units = L.work_stage(r, mem_gb, a.stage, a.unit)
            if s is None:
                print(f'{name:28s} complete')
                continue
            why = L.hold_reason(r, s)
            print(f"{name:28s} step {s['name']:7s} {len(units):3d} runnable"
                  f"{': ' + ', '.join(units[:4]) if units else ''}"
                  f"{'  (held: ' + why + ')' if why and not units else ''}")
            for k in units[:2]:
                argv_, _, _ = command_for(name, r, s['name'], k, root)
                print(f"    {k}: {' '.join(argv_)}")
        return

    if a.sleep:
        print(f'sleeping {a.sleep:.0f}s before the first claim', flush=True)
        time.sleep(a.sleep)

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
            print('no dataset with runnable work for this job (the rest is held or done); done', flush=True)
            break
        n_rows += 1
        print(f'=== took {row_name}', flush=True)
        exp_worker.work_row(L, row_name, job, host, mem_gb, wa)
        with exp_worker.locked_patiently(L):
            L.release(row_name, job)
        if a.dataset or (a.max_units and exp_worker._DONE[0] >= a.max_units):
            break
    print(f'papers worker {job} finished: {n_rows} dataset(s), {exp_worker._DONE[0]} unit(s)', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        os._exit(0)
