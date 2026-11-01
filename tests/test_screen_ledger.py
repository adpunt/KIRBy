"""The dataset screening ledger: plan, dependencies, file checks, extend,
skip-view, and several worker processes draining one run.

Run: python -m pytest tests/test_screen_ledger.py -q
No screen script runs: the drain test stubs every unit command
(KW_RUNNER_CMD=true, as tests/test_orchestrate_ledger.py does).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO / 'src', HERE, HERE / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.screen import datasets as D  # noqa: E402
from experiments.screen import ledger as SL   # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(SL, 'CURRENT_SCREEN_FILE', tmp_path / 'CURRENT_SCREEN')
    monkeypatch.setenv('KIRBY_DRONERF_DIR', str(tmp_path / 'dronerf'))
    monkeypatch.setenv('KIRBY_MATBENCH_REPS', str(tmp_path / 'reps'))
    monkeypatch.setenv('MATMINER_DATA', str(tmp_path / 'mbdata'))
    return tmp_path


def make(env, names=('dronerf', 'matbench'), models=None):
    root = env / 'run'
    rows = {n: D.build_row(n, root, (models or {}).get(n)) for n in names}
    L = SL.ScreenLedger(root)
    L.init(rows)
    return L


def unit(L, ds, key):
    for s in L.state['rows'][ds]['stages']:
        if key in s['units']:
            return s['units'][key]
    raise KeyError(key)


def test_plan_sizes(env):
    L = make(env)
    dr = {s['name']: len(s['units']) for s in L.state['rows']['dronerf']['stages']}
    # build: extract + 4 image views + 2 CNN views x 3 tasks x 2 splits x 10 folds
    assert dr == {'build': 1 + 4 + 120, 'evaluate': 12 * 5 * 3 * 2, 'summarise': 1}
    mb = {s['name']: len(s['units']) for s in L.state['rows']['matbench']['stages']}
    # 5 structure tasks x 13 views + expt_gap x 4 views = 69; + 6 targets units
    assert mb == {'build': 69 + 6, 'evaluate': 69, 'summarise': 1}


def test_evaluate_waits_for_its_view(env):
    L = make(env, ('dronerf',))
    with L.locked():
        row = L.row('dronerf')
        ev = L.stage_of(row, 'evaluate')
        runnable = set(L.runnable_units(row, ev))
        assert 'spectrum__rf__10__grouped' not in runnable      # extract not done
        unit(L, 'dronerf', 'extract').update(status='ok')
        runnable = set(L.runnable_units(row, ev))
        assert 'spectrum__rf__10__grouped' in runnable
        assert 'psd_resnet50__rf__10__grouped' not in runnable   # its image view not built
        assert 'alemadi_emb__rf__10__grouped' not in runnable    # needs all 10 fold builds
        for k in range(9):
            unit(L, 'dronerf', f'alemadi_emb__build__10__grouped__f{k}').update(status='ok')
        assert 'alemadi_emb__rf__10__grouped' not in set(L.runnable_units(row, ev))
        unit(L, 'dronerf', 'alemadi_emb__build__10__grouped__f9').update(status='ok')
        assert 'alemadi_emb__rf__10__grouped' in set(L.runnable_units(row, ev))


def test_jobs_take_evaluations_before_builds(env):
    L = make(env, ('dronerf',))
    with L.locked():
        unit(L, 'dronerf', 'extract').update(status='ok')
        stage, units = L.work_stage(L.row('dronerf'))
        assert stage['name'] == 'evaluate' and units


def test_exit_zero_without_file_is_failed(env):
    L = make(env, ('matbench',))
    k = 'magpie__rf__matbench_jdft2d'
    with L.locked():
        L.start_unit('matbench', 'evaluate', k, 'local:h:1', 'h')
        L.finish_unit('matbench', 'evaluate', k, 'ok', seconds=3, rc=0)
        assert unit(L, 'matbench', k)['status'] == 'failed'
        assert 'no result file' in unit(L, 'matbench', k)['error']
    f = Path(L.state['rows']['matbench']['paths']['results']) / 'b03_views' / 'matbench_jdft2d' / 'magpie__rf.json'
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({'result_status': 'ok'}))
    with L.locked():
        L.start_unit('matbench', 'evaluate', k, 'local:h:1', 'h')
        L.finish_unit('matbench', 'evaluate', k, 'ok', seconds=3, rc=0)
        assert unit(L, 'matbench', k)['status'] == 'ok'


def test_debug_result_does_not_count(env):
    L = make(env, ('dronerf',))
    f = Path(L.state['rows']['dronerf']['paths']['results']) / 'b06_views' / 'grouped' / '10' / 'spectrum__rf.json'
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({'result_status': 'ok', 'debug': True}))
    st, err = D.SPECS['dronerf'].check(L.state['rows']['dronerf'], 'spectrum__rf__10__grouped')
    assert st == 'failed' and 'debug' in err


def test_sync_marks_files_on_disk(env):
    L = make(env, ('matbench',))
    reps = env / 'reps' / 'matbench_phonons'
    reps.mkdir(parents=True)
    (reps / 'magpie.npy').write_bytes(b'x')
    L.load()
    found = L.scan_disk()
    with L.locked():
        assert L.apply_scan(found) == 1
        assert unit(L, 'matbench', 'magpie__build__matbench_phonons')['status'] == 'ok'


def test_skip_view_blocks_its_evaluations(env):
    L = make(env, ('matbench',))
    with L.locked():
        L.skip_experiments(['soap_avg'], 'too slow')
        assert unit(L, 'matbench', 'soap_avg__build__matbench_phonons')['status'] == 'skipped'
        assert unit(L, 'matbench', 'soap_avg__rf__matbench_phonons')['status'] == 'skipped'
        assert unit(L, 'matbench', 'magpie__rf__matbench_phonons')['status'] == 'pending'


def test_extend_adds_a_model_and_keeps_marks(env):
    L = make(env, ('dronerf',))
    with L.locked():
        unit(L, 'dronerf', 'spectrum__rf__2__published').update(status='ok')
        added = L.extend(['dronerf'], {'dronerf': ['svm']}, L.root)
        assert added == {'dronerf': '72 new unit(s)'}
        assert unit(L, 'dronerf', 'spectrum__rf__2__published')['status'] == 'ok'
        assert unit(L, 'dronerf', 'spectrum__svm__2__published')['status'] == 'pending'
        assert L.row('dronerf')['models'] == ['knn', 'rf', 'logreg', 'xgboost', 'dense', 'svm']


def test_document_renders(env):
    L = make(env)
    md = SL.render_md(L.state)
    assert '| dronerf |' in md and '| matbench |' in md and 'soap_avg' in md


def test_four_workers_drain_a_run(env):
    """Four worker processes, unit commands stubbed: every unit ends ok once,
    no unit runs twice."""
    L = make(env, ('matbench',))
    run = str(L.root)
    e = dict(os.environ, KW_RUNNER_CMD='true', KW_NO_FILE_CHECK='1', KIRBY_SCREEN=run,
             KIRBY_MATBENCH_REPS=str(env / 'reps'))
    cmd = [sys.executable, str(REPO / 'tests/experiments/screen/worker.py'), '--no-sync']
    procs = [subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for _ in range(4)]
    outs = [p.communicate(timeout=900)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    L.load()
    statuses = [u['status'] for s in L.state['rows']['matbench']['stages'] for u in s['units'].values()]
    assert statuses and all(st == 'ok' for st in statuses), set(statuses)
    attempts = [u['attempts'] for s in L.state['rows']['matbench']['stages'] for u in s['units'].values()]
    assert max(attempts) == 1
