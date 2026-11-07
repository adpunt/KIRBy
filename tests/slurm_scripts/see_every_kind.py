#!/usr/bin/env python3
"""Watch one unit of every kind actually run, in an interactive session.

    python tests/slurm_scripts/see_every_kind.py screen
    python tests/slurm_scripts/see_every_kind.py searches --dataset humanness_therapeutic
    python tests/slurm_scripts/see_every_kind.py paper2 --dataset humanness_therapeutic --search fast

A kind is one view (and model) in one stage of one screening dataset, one of the three
searches, or one Paper 2 experiment. Every unit runs at its normal settings. --box is the
time box in minutes: a unit still running then counts as RUNNING, is stopped, and goes back
in the ledger unchanged, with no attempt used. A kind that has finished or passed its box is
remembered, so restarting the same command goes on with the rest.

screen and searches run real ledger units, and a unit that finishes is a real result.
paper2 is closed in RUN_SETTINGS.json and waits for the search you choose, so it runs each
experiment outside the ledger with results in <run>/see_paper2. The ledger records nothing.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / 'tests'
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
DONE_WORDS = ('ok', 'skipped')


def pointer(name):
    return Path((TESTS / 'results' / name).read_text().strip())


def rows_of(run):
    return json.loads((Path(run) / 'ledger.json').read_text())['rows']


def screen_kinds(rows):
    """[(kind, dataset, stage, unit, status)], one unit per kind that still has work.
    A unit already running is preferred, because another job is already showing that kind."""
    kinds = {}
    for name, row in rows.items():
        for stage in row['stages']:
            for key, u in stage['units'].items():
                if u['status'] in DONE_WORDS:
                    continue
                parts = key.split('__')
                kind = f"{name} {stage['name']} {' '.join(parts[:2]) if len(parts) > 1 else key}"
                have = kinds.get(kind)
                if have is None or (u['status'] == 'running' and have[4] != 'running'):
                    kinds[kind] = (kind, name, stage['name'], key, u['status'])
    # Builds before evaluates before summaries: each needs the one before.
    order = {'build': 0, 'evaluate': 1, 'summarise': 2}
    return sorted(kinds.values(), key=lambda k: (order.get(k[2], 3), k[1], k[0]))


def unit_of(run, dataset, stage, unit):
    for s in rows_of(run)[dataset]['stages']:
        if s['name'] == stage:
            return s['units'].get(unit) or {}
    return {}


def run_boxed(cmd, env, log, box_s):
    """(exit code or None if the time box ended it, seconds)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                             cwd=REPO, start_new_session=True)
        try:
            return p.wait(box_s), time.time() - t0
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(5)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            return None, time.time() - t0


def last_lines(log, n=3):
    try:
        return ' / '.join(x.strip() for x in Path(log).read_text(errors='replace').splitlines()[-n:])[:300]
    except OSError:
        return ''


def ledger_unit(a, run, worker, ledger, kind, dataset, stage, unit):
    """Run one ledger unit. Returns (word, detail)."""
    log = run / 'logs' / 'see_every_kind' / (kind.replace(' ', '__') + '.log')
    cmd = [sys.executable, '-u', str(worker), '--dataset', dataset, '--stage', stage, '--unit', unit, '--no-sync']
    rc, secs = run_boxed(cmd, dict(os.environ, KIRBY_LEDGER_MANAGED='1'), log, a.box * 60)
    u = unit_of(run, dataset, stage, unit)
    minutes = f'{secs / 60:.1f} min'
    if rc is None:
        cmd = [sys.executable, str(ledger), 'reset', dataset, unit]
        if worker.parent.name == 'papers':
            cmd += ['--stage', stage]      # search_f0 is in all three searches
        subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        return 'RUNNING', f'no error in {minutes}; stopped by the time box and put back'
    if u.get('status') == 'ok':
        return 'FINISHED', f'{minutes}'
    if u.get('status') == 'skipped':
        return 'SKIPPED', str(u.get('error'))[:200]
    if u.get('status') == 'failed':
        return 'FAILED', f"{str(u.get('error'))[:220]}  log: {u.get('log')}"
    return 'WAITING', f'did not start: {last_lines(log, 2)}'


def run_all(a, run, jobs, seen_name):
    """jobs: [(kind, fn)]. Prints one line per kind; remembers what passed."""
    seen_file = run / seen_name
    passed = set()
    if seen_file.exists():
        # A line that only repeated a login-node process's 'running' mark proved nothing.
        passed = {json.loads(x)['kind'] for x in seen_file.read_text().splitlines() if x.strip()
                  and json.loads(x)['word'] in ('RUNNING', 'FINISHED', 'SKIPPED')
                  and 'already running in job local:' not in json.loads(x)['detail']}
    todo = [(k, f) for k, f in jobs if k not in passed]
    print(f'{len(jobs)} kinds, {len(jobs) - len(todo)} already seen. {a.parallel} at a time, '
          f'time box {a.box} minutes each.', flush=True)
    counts = {}

    def one(item):
        kind, fn = item
        print(f'start    {kind}', flush=True)
        try:
            word, detail = fn()
        except Exception as e:      # one kind must not stop the others
            word, detail = 'FAILED', f'{type(e).__name__}: {e}'
        counts[word] = counts.get(word, 0) + 1
        print(f'{word:9s}{kind}  {detail}', flush=True)
        with open(seen_file, 'a') as f:
            f.write(json.dumps({'kind': kind, 'word': word, 'detail': detail,
                                'host': socket.gethostname(), 'time': time.ctime()}) + '\n')

    with ThreadPoolExecutor(a.parallel) as pool:
        list(pool.map(one, todo))
    print('\nSummary: ' + (', '.join(f'{v} {k}' for k, v in sorted(counts.items())) or 'nothing left to run'))
    print(f'Every line: {seen_file}')


def screen_unit_waits(run, dataset, stage, unit):
    """True when a unit this one needs has not finished. Naming a unit by hand skips that
    check in the worker, so without this an evaluate runs before its build and fails."""
    from experiments.screen.ledger import ScreenLedger
    L = ScreenLedger(run)
    L.load()
    row = L.row(dataset)
    return L.deps_state(row, L.stage_of(row, stage), unit) != 'ok'


def cmd_screen(a):
    run = pointer('CURRENT_SCREEN')
    worker = TESTS / 'experiments/screen/worker.py'
    ledger = TESTS / 'experiments/screen/ledger.py'
    jobs = []
    for kind, ds, stage, unit, status in screen_kinds(rows_of(run)):
        if status == 'running':
            u = unit_of(run, ds, stage, unit)
            job = str(u.get('job'))
            if job.startswith('local:'):      # no Slurm job: the ledger's mark is all there is
                jobs.append((kind, lambda job=job: ('WAITING', f'marked running by {job}, outside Slurm; not seen')))
            else:
                jobs.append((kind, lambda job=job: ('RUNNING', f'already running in job {job}')))
        else:
            def go(ds=ds, st=stage, un=unit, k=kind):
                if screen_unit_waits(run, ds, st, un):
                    return 'WAITING', f'{un} needs a unit that has not finished; not run'
                return ledger_unit(a, run, worker, ledger, k, ds, st, un)
            jobs.append((kind, go))
    run_all(a, run, jobs, 'seen_running.jsonl')


def cmd_searches(a):
    run = pointer('CURRENT_PAPERS')
    worker = TESTS / 'experiments/papers/worker.py'
    ledger = TESTS / 'experiments/papers/ledger.py'
    jobs = [(f'{a.dataset} {s} search_f0', lambda s=s, k=f'{a.dataset} {s} search_f0':
             ledger_unit(a, run, worker, ledger, k, a.dataset, s, 'search_f0'))
            for s in ('search_pruned', 'search_fast', 'search_full')]
    run_all(a, run, jobs, 'seen_running.jsonl')


def cmd_paper2(a):
    from experiments.papers import plan as P
    from experiments.papers.ledger import read_settings, settings_env
    from experiments.orchestrate import runner
    run = pointer('CURRENT_PAPERS')
    env = runner.common_env(run)
    env.pop('KIRBY_RUN', None)
    env['KIRBY_PAPERS_RUN'] = str(run)
    env.update(settings_env(read_settings(run), a.search))
    done = [k for k, u in next(s for s in rows_of(run)[a.dataset]['stages']
                               if s['name'] == f'search_{a.search}')['units'].items() if u['status'] == 'ok']
    print(f'{a.dataset}: {len(done)} of 5 folds of search_{a.search} finished. An experiment on a fold '
          f'with no search yet starts that search itself.', flush=True)
    out = run / 'see_paper2'
    jobs = []
    for unit, category in P.paper_units(2, a.dataset):
        if category not in ('complexity', 'similarity'):
            continue
        cmd = [sys.executable, '-u', str(TESTS / 'experiments/run_dataset.py'), category, a.dataset,
               '--experiments', unit, '--output', str(out / category)]
        kind = f'{a.dataset} paper2 {unit} {a.search}'

        def go(cmd=cmd, kind=kind):
            log = out / 'logs' / (kind.replace(' ', '__') + '.log')
            rc, secs = run_boxed(cmd, env, log, a.box * 60)
            if rc is None:
                return 'RUNNING', f'no error in {secs / 60:.1f} min; stopped by the time box'
            return ('FINISHED', f'{secs / 60:.1f} min') if rc == 0 else (
                'FAILED', f'exit {rc}: {last_lines(log)}  log: {log}')
        jobs.append((kind, go))
    run_all(a, run, jobs, 'seen_paper2.jsonl')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('screen', 'searches', 'paper2'))
    ap.add_argument('--dataset', default='humanness_therapeutic')
    ap.add_argument('--search', default='fast', choices=('pruned', 'fast', 'full'))
    ap.add_argument('--box', type=float, default=10, help='minutes per unit')
    ap.add_argument('--parallel', type=int, default=4)
    a = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        sys.exit('see_every_kind runs real units: start an srun first (this shell has no Slurm job)')
    {'screen': cmd_screen, 'searches': cmd_searches, 'paper2': cmd_paper2}[a.what](a)


if __name__ == '__main__':
    main()
