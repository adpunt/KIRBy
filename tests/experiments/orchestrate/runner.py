"""Run one unit as a subprocess and say how it ended.

The command for a dataset unit is the one submit_dataset.sh built
(run_dataset.py CATEGORY DATASET --experiments EXP [--noise-condition C]
--output <root>/<category>); a search unit runs search_unit.py; an OPIG row
runs runbook/opig_one.sh; a cross row runs runbook/cross_one.sh. Output goes
to <root>/logs/<row>__<stage>__<unit>.log. The exit-code table is
submit_dataset.sh's.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_TESTS = _Path(__file__).resolve().parent.parent.parent
for _p in (_TESTS.parent / 'src', _TESTS, _TESTS / 'experiments'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from experiments.orchestrate import plan  # noqa: E402

STATUS_BY_RC = {0: 'ok', 124: 'timeout', 137: 'killed_9_oom_or_wall',
                139: 'segfault', 135: 'bus_error'}
PROGRESS_EVERY_S = 300     # a line every few minutes keeps an srun terminal alive


def common_env(run_root):
    env = dict(os.environ)
    env.update({
        'KIRBY_RUN': str(run_root),
        'KIRBY_LEDGER_MANAGED': '1',       # the worker records; run_dataset.py's hook stays quiet
        'KIRBY_SKIP_INSTALLS': env.get('KIRBY_SKIP_INSTALLS', '1'),
        'KIRBY_NO_PULL': '1',
        'KIRBY_NO_COLD_SEARCH': env.get('KIRBY_NO_COLD_SEARCH', '1'),   # whole-dataset only; per-fold never refused
        'KIRBY_LGBM_THREADS': env.get('KIRBY_LGBM_THREADS', '1'),      # same value and reasons as submit_dataset.sh
        'KIRBY_START_DELAY': '0',           # the ledger spreads jobs; opig_one.sh's own delay is not needed
        'PYTHONUNBUFFERED': '1',
        'PYTHONWARNINGS': env.get('PYTHONWARNINGS', 'ignore'),
    })
    env.pop('KIRBY_NOISE_PROMOTE', None)
    return env


def command_for(row_name, row, stage_name, unit_key, run_root):
    """(argv, env, result_dir) for one unit."""
    run_root = Path(run_root)
    env = common_env(run_root)
    py = sys.executable
    if row['kind'] == 'dataset':
        ds = row['dataset']
        m = plan.SEARCH_UNIT_RE.match(unit_key)
        if m:
            argv = [py, '-u', str(plan.HERE / 'search_unit.py'), ds, '--fold', m.group(1)]
            return argv, env, None
        exp = unit_key.split('__', 1)[0]
        cond = unit_key.split('__', 1)[1] if '__' in unit_key else None
        category = 'noise' if stage_name == 'noise_full' else stage_name
        out = run_root / plan.CATEGORY_DIR[stage_name]
        argv = [py, '-u', str(plan.TESTS / 'experiments' / 'run_dataset.py'),
                category, ds, '--experiments', exp, '--output', str(out)]
        if cond:
            argv += ['--noise-condition', cond]
        if stage_name == 'noise_full':
            env['KIRBY_NOISE_PROMOTE'] = exp
        return argv, env, out
    if row['kind'] == 'opig':
        env['KIRBY_OUTPUT_DIR'] = str(run_root / 'opig')
        env['KIRBY_EXP_LOG_DIR'] = str(run_root / 'logs' / 'opig_inner')
        argv = ['bash', str(plan.REPO / 'tests/slurm_scripts/runbook/opig_one.sh'), row['dataset']]
        return argv, env, run_root / 'opig'
    if row['kind'] == 'cross':
        name = row['dataset']
        # s09 reads the similarity result files; c03 and c14 write beside the
        # complexity ones (cross_one.sh passes KIRBY_OUTPUT_DIR as both).
        sub = 'similarity' if name == 's09' else 'complexity'
        env['KIRBY_OUTPUT_DIR'] = str(run_root / sub)
        env['KIRBY_EXP_LOG_DIR'] = str(run_root / 'logs' / 'cross_inner')
        argv = ['bash', str(plan.REPO / 'tests/slurm_scripts/runbook/cross_one.sh'), name] + list(row['datasets'])
        return argv, env, run_root / sub
    raise ValueError(f'unknown row kind {row["kind"]!r}')


def log_path(run_root, row_name, stage_name, unit_key):
    safe = row_name.replace(':', '_')
    return Path(run_root) / 'logs' / f'{safe}__{stage_name}__{unit_key}.log'


def _tail(path, n=25):
    try:
        lines = Path(path).read_text(errors='replace').splitlines()
    except OSError:
        return ''
    return '\n'.join(lines[-n:])


def _error_text(result_dir, row, unit_key, started_at, log):
    """The error run_dataset.py wrote to _errors/ for this unit, else the
    last traceback-looking lines of the log."""
    if result_dir is not None and row['kind'] == 'dataset':
        ds = row['dataset']
        exp = unit_key.split('__', 1)[0]
        cond = unit_key.split('__', 1)[1] if '__' in unit_key else None
        f = Path(result_dir) / '_errors' / f"{ds}{'__' + cond if cond else ''}__{exp}.json"
        if f.exists() and f.stat().st_mtime >= started_at - 1:
            try:
                d = json.loads(f.read_text())
                recs = d.get('failed') or d.get('experiments') or d
                if isinstance(recs, dict):
                    for k, v in recs.items():
                        if isinstance(v, dict) and v.get('error'):
                            return f"{k}: {v['error']}"[:600]
                return json.dumps(d)[:600]
            except Exception:
                pass
    tail = _tail(log, 40)
    lines = [ln for ln in tail.splitlines() if 'Error' in ln or 'error' in ln or 'Traceback' in ln]
    return ('\n'.join(lines[-5:]) or tail.splitlines()[-1] if tail else '')[:600]


def run_unit(row_name, row, stage_name, unit_key, run_root, heartbeat=None,
             timeout_s=None):
    """Run the unit; returns dict(rc, status, seconds, error, log).

    heartbeat: callable run every PROGRESS_EVERY_S while the unit runs (the
    worker uses it to stamp the ledger and print a progress line).
    """
    argv, env, result_dir = command_for(row_name, row, stage_name, unit_key, run_root)
    log = log_path(run_root, row_name, stage_name, unit_key)
    log.parent.mkdir(parents=True, exist_ok=True)
    stub = os.environ.get('KW_RUNNER_CMD')        # tests: replace the real command
    if stub:
        argv = [stub] + argv
    t0 = time.time()
    print(f">>> {row_name} {stage_name} {unit_key}  start {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
          f"    {' '.join(argv)}\n    log {log}", flush=True)
    with open(log, 'a') as fh:
        fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        fh.flush()
        proc = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=str(plan.REPO))
        last = time.time()
        while True:
            try:
                rc = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
            if timeout_s and time.time() - t0 > timeout_s:
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                rc = 124
                break
            if time.time() - last >= PROGRESS_EVERY_S:
                last = time.time()
                el = (time.time() - t0) / 60
                print(f"    ... {row_name} {unit_key} {el:.0f} min: {_tail(log, 1)[:150]}", flush=True)
                if heartbeat:
                    try:
                        keep = heartbeat()
                    except Exception as exc:
                        print(f'    heartbeat failed: {exc}', flush=True)
                        keep = True
                    if keep is False:      # the row was cut while this ran
                        print(f'    {row_name} was cut; stopping {unit_key}', flush=True)
                        proc.terminate()
                        try:
                            proc.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        rc = 143
                        break
    secs = time.time() - t0
    if rc < 0:                     # killed by signal N -> 128+N, as a shell reports it
        rc = 128 - rc
    status = STATUS_BY_RC.get(rc, 'failed')
    error = None if rc == 0 else _error_text(result_dir, row, unit_key, t0, log)
    print(f"<<< {row_name} {stage_name} {unit_key}  exit={rc} {status} {secs:.0f}s", flush=True)
    return {'rc': rc, 'status': 'ok' if rc == 0 else 'failed', 'kind': status,
            'seconds': secs, 'error': error, 'log': str(log)}
