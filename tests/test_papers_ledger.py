"""The Papers 2 and 3 ledger: plan, the gate in RUN_SETTINGS.json, the hold
on datasets whose models are not recorded, the run's own cache, the full
search space switch, and worker processes draining a run.

Run: python -m pytest tests/test_papers_ledger.py -q
No experiment runs: unit commands are stubbed (KW_RUNNER_CMD=true).
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

from experiments.papers import ledger as PL   # noqa: E402
from experiments.papers import plan as P      # noqa: E402

DS = ('herg_ki', 'logd')


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, 'CURRENT_PAPERS_FILE', tmp_path / 'CURRENT_PAPERS')
    shared = tmp_path / 'thesis_cache'
    for d in ('rep_cache', 'result_db', 'baselines', 'QM9', 'tcga', 'pdv_stale_20260918'):
        (shared / d).mkdir(parents=True)
    (shared / 'result_db' / 'herg_ki.db').write_text('thesis')
    return tmp_path


def make(env, names=DS):
    root = env / 'run'
    cache = root / 'cache'
    PL.prepare_cache(cache, env / 'thesis_cache')
    L = PL.PapersLedger(root)
    L.init(P.build_rows(names), PL.default_settings(cache, env / 'thesis_cache'))
    return L


def set_open(L, **steps):
    p = L.root / PL.SETTINGS_NAME
    s = json.loads(p.read_text())
    s['open'].update(steps)
    p.write_text(json.dumps(s))


def stage(L, ds, name):
    return L.stage_of(L.state['rows'][ds], name)


def test_plan_steps(env):
    L = make(env)
    r = L.state['rows']['herg_ki']
    assert [s['name'] for s in r['stages']] == list(P.STEPS)
    for st in P.SEARCH_STEPS:
        assert list(stage(L, 'herg_ki', st)['units']) == [f'search_f{i}' for i in range(5)]
    p2 = stage(L, 'herg_ki', 'paper2')['units']
    assert 'c02' in p2 and 's03' in p2 and 'c03' not in p2
    assert r['papers'] == [2, 3]


def test_paper2_only_dataset_has_no_paper3(env):
    L = make(env, ('humanness_therapeutic',))
    assert stage(L, 'humanness_therapeutic', 'paper3')['status'] == 'empty'


def test_cache_links_reps_and_leaves_searches_empty(env):
    make(env)
    cache = env / 'run' / 'cache'
    assert (cache / 'rep_cache').is_symlink() and (cache / 'QM9').is_symlink()
    assert not (cache / 'result_db').exists() and not (cache / 'baselines').exists()
    assert not (cache / 'tcga').exists() and not (cache / 'pdv_stale_20260918').exists()
    # the thesis cache is untouched
    assert (env / 'thesis_cache' / 'result_db' / 'herg_ki.db').read_text() == 'thesis'


def set_field(L, **kw):
    p = L.root / PL.SETTINGS_NAME
    s = json.loads(p.read_text())
    s.update(kw)
    p.write_text(json.dumps(s))


def close_step(L, ds, name):
    for u in stage(L, ds, name)['units'].values():
        u['status'] = 'ok'
    L.close_stage_if_done(ds, stage(L, ds, name))


def test_only_reps_runs_until_steps_are_opened(env):
    L = make(env)
    with L.locked():
        row = L.row('herg_ki')
        assert [s['name'] for s in L.open_stages(row)] == ['reps']
        close_step(L, 'herg_ki', 'reps')
        assert L.open_stages(row) == []
        # named by hand: still held
        assert L.runnable_units(row, stage(L, 'herg_ki', 'paper2'), only_unit='c02') == []
    set_open(L, search_pruned=True, search_fast=True, search_full=True)
    with L.locked():
        row = L.row('herg_ki')
        assert [s['name'] for s in L.open_stages(row)] == list(P.SEARCH_STEPS)
        for st in P.SEARCH_STEPS:
            close_step(L, 'herg_ki', st)
        assert L.open_stages(row) == []
    set_open(L, paper2=True)
    with L.locked():     # open, but no search chosen
        assert L.open_stages(L.row('herg_ki')) == []
        assert 'experiments_search' in L.hold_reason(L.row('herg_ki'), stage(L, 'herg_ki', 'paper2'))
    set_field(L, experiments_search='fast')
    with L.locked():
        assert [s['name'] for s in L.open_stages(L.row('herg_ki'))] == ['paper2']


def test_experiments_wait_for_the_chosen_search_only(env):
    L = make(env)
    set_open(L, paper2=True)
    set_field(L, experiments_search='pruned')
    with L.locked():
        close_step(L, 'herg_ki', 'reps')
        assert L.open_stages(L.row('herg_ki')) == []
        close_step(L, 'herg_ki', 'search_pruned')
        assert [s['name'] for s in L.open_stages(L.row('herg_ki'))] == ['paper2']


def test_each_search_has_its_own_cache_and_settings(env):
    L = make(env)
    from experiments.papers import worker as W
    seen = {}
    row = L.state['rows']['herg_ki']
    for st in P.SEARCH_STEPS:
        argv, e, _ = W.command_for('herg_ki', row, st, 'search_f3', L.root)
        assert argv[-3:] == ['3', '--n-folds', '5'] and 'KIRBY_RUN' not in e
        seen[st] = e
    assert len({e['KIRBY_CACHE_DIR'] for e in seen.values()}) == 3
    assert seen['search_pruned']['KIRBY_SEARCH_PRUNING'] == 'calibrated'
    assert seen['search_full']['KIRBY_SEARCH_PRUNING'] == 'none'
    assert seen['search_fast']['KIRBY_SEARCH_SPACE'] == 'fast'
    assert seen['search_full']['KIRBY_SEARCH_SPACE'] == 'full'


def test_replan_keeps_finished_units(env):
    L = make(env)
    with L.locked():
        L.state['rows']['herg_ki']['stages'].insert(1, {'name': 'check', 'status': 'empty', 'units': {}})
        close_step(L, 'herg_ki', 'reps')
        L.replan()
        names = [s['name'] for s in L.row('herg_ki')['stages']]
        assert names == list(P.STEPS)
        assert stage(L, 'herg_ki', 'reps')['units']['reps']['status'] == 'ok'


def test_failed_reps_hold_the_searches(env):
    L = make(env)
    set_open(L, search_pruned=True)
    with L.locked():
        u = stage(L, 'logd', 'reps')['units']['reps']
        u.update(status='failed', attempts=2)
        L.close_stage_if_done('logd', stage(L, 'logd', 'reps'))
        assert stage(L, 'logd', 'reps')['status'] == 'done_with_failures'
        assert L.open_stages(L.row('logd')) == []
        assert 'failed to build' in L.hold_reason(L.row('logd'), stage(L, 'logd', 'search_pruned'))


def test_future_work_holds_a_dataset(env):
    L = make(env)
    p = L.root / PL.SETTINGS_NAME
    s = json.loads(p.read_text())
    s['future_work'] = ['logd']
    p.write_text(json.dumps(s))
    with L.locked():
        assert L.open_stages(L.row('logd')) == []
        assert L.open_stages(L.row('herg_ki'))


def test_unrecorded_models_hold_every_step_but_reps(env, monkeypatch):
    L = make(env)
    set_open(L, search_pruned=True)
    monkeypatch.setattr(P, 'models_recorded', lambda ds: ds != 'logd')
    with L.locked():
        row = L.row('logd')
        close_step(L, 'logd', 'reps')
        assert L.open_stages(row) == []
        assert 'not recorded' in L.hold_reason(row, stage(L, 'logd', 'search_pruned'))


def test_ok_needs_a_result_file(env):
    L = make(env)
    with L.locked():
        L.finish_unit('herg_ki', 'paper2', 'c02', 'ok', seconds=1, rc=0)
        assert stage(L, 'herg_ki', 'paper2')['units']['c02']['status'] == 'failed'
    d = L.root / 'complexity' / 'c02_complexity_grid'
    d.mkdir(parents=True)
    (d / 'herg_ki.json').write_text(json.dumps({'result_status': 'ok'}))
    with L.locked():
        L.reset_unit('herg_ki', 'c02')
        L.finish_unit('herg_ki', 'paper2', 'c02', 'ok', seconds=1, rc=0)
        assert stage(L, 'herg_ki', 'paper2')['units']['c02']['status'] == 'ok'


def test_full_search_space(monkeypatch):
    from experiments import decided_grid as g
    monkeypatch.setenv('KIRBY_SEARCH_SPACE', 'full')
    gen = g.decided_search('herg_ki', 'general')
    dist = g.decided_search('herg', 'general_distance')
    assert gen['budgets'] == [10, 25, 50, 100, 250, 500, 1000, 2000]
    assert gen['importance_methods'] == ['deeplift', 'lightgbm_gain', 'treeshap']
    assert gen['rep_counts'] == [2, 3, 5, None]
    assert gen['models'] == g._table()['herg_ki']['general']['models']
    assert dist['models'][-1] == 'knn' and 'knn' in dist['internal_evaluators']
    monkeypatch.setenv('KIRBY_SEARCH_PRUNING', 'calibrated')
    assert g.search_pruning_mode() == 'calibrated'
    monkeypatch.setenv('KIRBY_SEARCH_SPACE', 'decided')
    assert g.search_pruning_mode() is None
    monkeypatch.setenv('KIRBY_SEARCH_SPACE', 'fast')
    f = g.decided_search('herg_ki', 'general')
    assert f['importance_methods'] == ['lightgbm_gain'] and f['rep_counts'] == [3, None]
    assert f['models'] == gen['models'] and f['speed'] == 'fast'
    monkeypatch.setenv('KIRBY_SEARCH_SPACE', 'decided')
    assert g.decided_search('herg_ki', 'general') == g._table()['herg_ki']['general']


def test_three_workers_drain_reps_and_searches(env):
    """Three worker processes, commands stubbed: every reps and search unit
    ends ok once; nothing behind the gate starts."""
    L = make(env)
    set_open(L, search_pruned=True, search_fast=True, search_full=True)
    e = dict(os.environ, KW_RUNNER_CMD='true', KIRBY_PAPERS_RUN=str(L.root))
    cmd = [sys.executable, str(REPO / 'tests/experiments/papers/worker.py'), '--no-sync']
    procs = [subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for _ in range(3)]
    outs = [p.communicate(timeout=900)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    L.load()
    for ds in DS:
        for name in ('reps',) + P.SEARCH_STEPS:
            us = stage(L, ds, name)['units'].values()
            assert all(u['status'] == 'ok' and u['attempts'] == 1 for u in us), (ds, name)
        assert all(u['status'] == 'pending' for u in stage(L, ds, 'paper2')['units'].values())
