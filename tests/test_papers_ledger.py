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

import numpy as np
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


def test_progress_table(env):
    L = make(env)
    with L.locked():
        stage(L, 'herg_ki', 'reps')['units']['reps'].update(status='ok', seconds=7200)
        stage(L, 'herg_ki', 'search_pruned')['units']['search_f0'].update(status='running')
        stage(L, 'herg_ki', 'search_fast')['units']['search_f1'].update(status='failed')
    lines = PL.render_progress(L.state).splitlines()
    assert lines[0].split() == ['dataset', 'reps', 'search_pruned', 'search_fast', 'search_full',
                                'paper2', 'paper3', 'running', 'failed', 'skipped', 'hours']
    row = next(x.split() for x in lines if x.startswith('herg_ki'))
    n2, n3 = (len(stage(L, 'herg_ki', s)['units']) for s in ('paper2', 'paper3'))
    assert row == ['herg_ki', '1/1', '0/5', '0/5', '0/5', f'0/{n2}', f'0/{n3}', '1', '1', '0', '2.0']


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


# ── The meta-learner, a Paper 2 experiment (2026-11-05) ────────────────────

def set_meta(L, **kw):
    p = L.root / PL.SETTINGS_NAME
    s = json.loads(p.read_text())
    s['meta_learner'].update(kw)
    p.write_text(json.dumps(s))


def _ready_for_paper2(L, ds='herg_ki'):
    """The paper2 step open, the search chosen and done, reps done."""
    p = L.root / PL.SETTINGS_NAME
    s = json.loads(p.read_text())
    s['open']['paper2'] = True
    s['experiments_search'] = 'full'
    p.write_text(json.dumps(s))
    with L.locked():
        stage(L, ds, 'reps')['status'] = 'done'
        for u in stage(L, ds, 'search_full')['units'].values():
            u['status'] = 'ok'


META_KEYS = [f'meta_f{i}' for i in range(5)]


def test_meta_learner_is_in_the_paper2_step_and_no_step_of_its_own(env):
    L = make(env)
    assert P.STEPS == ('reps', 'search_pruned', 'search_fast', 'search_full', 'paper2', 'paper3')
    units = list(stage(L, 'herg_ki', 'paper2')['units'])
    assert units[-5:] == META_KEYS
    assert not set(META_KEYS) & set(stage(L, 'herg_ki', 'paper3')['units'])
    assert P.unit_category('meta_f0') == 'meta_learner'


def test_meta_learner_settings_are_only_what_she_has_written(env):
    L = make(env)
    m = L.settings()['meta_learner']
    assert m['datasets'] == []
    assert m['inner_folds'] == 5 and m['min_rows'] == 500
    assert m['net_sizes'] == ['dnn_small', 'dnn_wide']


def test_opening_paper2_runs_the_others_and_holds_it_until_named(env, monkeypatch):
    monkeypatch.setattr(P, 'models_recorded', lambda ds: True)
    L = make(env)
    _ready_for_paper2(L)
    r = L.state['rows']['herg_ki']
    s = stage(L, 'herg_ki', 'paper2')
    L.settings()
    units = L.runnable_units(r, s)
    assert 'c02' in units and not set(META_KEYS) & set(units)
    assert 'not named' in L.unit_hold_reason(r, s, 'meta_f0')
    assert not set(META_KEYS) & set(L.runnable_units(r, s))
    set_meta(L, datasets=['herg_ki'])
    L.settings()
    assert L.unit_hold_reason(r, s, 'meta_f0') is None
    assert 'meta_f0' in L.runnable_units(r, s)
    other = L.state['rows']['logd']
    assert 'not named' in L.unit_hold_reason(other, stage(L, 'logd', 'paper2'), 'meta_f0')


def test_meta_learner_command_carries_every_setting(env):
    from experiments.papers import worker as W
    L = make(env)
    _ready_for_paper2(L)
    set_meta(L, net_sizes=['dnn_small'], datasets=['herg_ki'])
    argv, e, out = W.command_for('herg_ki', L.state['rows']['herg_ki'], 'paper2',
                                 'meta_f2', L.root)
    assert argv[1:3] == ['-u', str(HERE / 'experiments' / 'papers' / 'meta_learner_own_hybrids.py')]
    assert argv[argv.index('--fold') + 1] == '2'
    assert argv[argv.index('--inner-folds') + 1] == '5'
    assert argv[argv.index('--min-rows') + 1] == '500'
    assert argv[argv.index('--candidates') + 1].endswith('ngboost,dnn_small')
    assert e['KIRBY_SEARCH_SPACE'] == 'full'
    assert e['KIRBY_CACHE_DIR'].endswith('_full')


def test_other_paper2_units_keep_their_command(env):
    from experiments.papers import worker as W
    L = make(env)
    _ready_for_paper2(L)
    argv, _, _ = W.command_for('herg_ki', L.state['rows']['herg_ki'], 'paper2', 'c02', L.root)
    assert 'run_dataset.py' in argv[2] and argv[argv.index('--experiments') + 1] == 'c02'


def test_meta_learner_ok_needs_its_result_file(env):
    L = make(env)
    with L.locked():
        L.finish_unit('herg_ki', 'paper2', 'meta_f0', 'ok', seconds=1, rc=0)
        assert stage(L, 'herg_ki', 'paper2')['units']['meta_f0']['status'] == 'failed'
    f = L.root / P.META_DIR / 'herg_ki__meta_f0.json'
    f.write_text(json.dumps({'result_status': 'skipped', 'reason': 'below min_rows'}))
    with L.locked():
        L.reset_unit('herg_ki', 'meta_f0')
        L.finish_unit('herg_ki', 'paper2', 'meta_f0', 'ok', seconds=1, rc=0)
        assert stage(L, 'herg_ki', 'paper2')['units']['meta_f0']['status'] == 'skipped'


def test_extend_all_adds_the_units_to_an_older_run_and_keeps_settings(env):
    L = make(env)
    set_open(L, search_full=True)
    with L.locked():
        for k in META_KEYS:
            for ds in DS:
                del stage(L, ds, 'paper2')['units'][k]
        stage(L, 'herg_ki', 'reps')['units']['reps'].update(status='ok')
    PL.main(['--run', str(L.root), 'extend', '--all'])
    L2 = PL.PapersLedger(L.root)
    L2.load()
    assert set(META_KEYS) <= set(stage(L2, 'herg_ki', 'paper2')['units'])
    assert stage(L2, 'herg_ki', 'reps')['units']['reps']['status'] == 'ok'
    assert L2.settings()['open']['search_full'] is True


def _meta_data(task, seed=0, n=120):
    rng = np.random.default_rng(seed)
    if task == 'regression':
        y = rng.normal(size=n)
        X = np.column_stack([y + rng.normal(scale=s, size=n) for s in (0.2, 0.5, 2.0)])
        return X, y
    y = rng.integers(0, 2, size=n)
    p = np.clip(y[:, None] + rng.normal(scale=0.3, size=(n, 3)), 0.01, 0.99)
    X = np.empty((n, 6))
    X[:, 0::2], X[:, 1::2] = 1 - p, p
    return X, y


def test_every_meta_learner_runs_on_both_tasks_without_a_choice_being_made():
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from experiments.papers import meta_learner_own_hybrids as M
    splits = [(np.where(np.arange(100) % 4 != i)[0], np.where(np.arange(100) % 4 == i)[0])
              for i in range(4)]
    X, y = _meta_data('regression')
    r = M.score_meta_learners(X[:100], y[:100], X[100:], y[100:], splits, metric_fn=mean_squared_error)
    assert set(r) == set(M.META_LEARNERS)
    assert not [k for k, v in r.items() if v['error']], r
    assert all(v['train_rows_cv_score'] is not None for v in r.values())
    # The informative model dominates, so constrained weights beat equal ones.
    assert r['simplex_weights']['heldout_score'] < r['equal_weights']['heldout_score']
    X, y = _meta_data('classification')
    r = M.score_meta_learners(X[:100], y[:100], X[100:], y[100:], splits, metric_fn=roc_auc_score,
                              classes=np.array([0, 1]), needs_proba=True)
    assert not [k for k, v in r.items() if v['error']], r


def test_simplex_weights_are_nonnegative_and_sum_to_one():
    from experiments.papers import meta_learner_own_hybrids as M
    X, y = _meta_data('regression')
    w = M.SimplexWeights().fit(X, y).w
    assert (w >= 0).all() and abs(w.sum() - 1) < 1e-6 and w[0] > w[2]


def test_top_models_follows_the_metric_direction():
    from experiments.papers import meta_learner_own_hybrids as M
    s = {m: {'entry': {}, 'score': v, 'lower_is_better': True}
         for m, v in zip('abcdefg', (7, 6, 5, 4, 3, 2, 1))}
    s['z'] = {'entry': None, 'score': None}
    assert M.top_models(s) == ['g', 'f', 'e', 'd', 'c']
    for r in s.values():
        r['lower_is_better'] = False
    assert M.top_models(s) == ['a', 'b', 'c', 'd', 'e']


def test_a_candidates_search_never_shares_a_file_with_the_ledgers():
    from experiments.papers import meta_learner_own_hybrids as M
    assert M.candidate_suffix('xgboost') == 'meta_xgboost'
    assert M.candidate_suffix('xgboost', 'molmean') == 'molmean__meta_xgboost'


def test_every_fit_of_a_model_uses_one_fixed_feature_set(monkeypatch):
    """The recipe runs once. Inner fits and the final fit take columns of that one
    matrix, so the feature count cannot differ between them."""
    from experiments.papers import meta_learner_own_hybrids as M
    widths = []

    class Rec:
        def fit(self, X, y):
            widths.append(X.shape[1])
            return self

        def predict(self, X):
            widths.append(X.shape[1])
            return np.zeros(len(X))

    X = np.random.default_rng(0).normal(size=(50, 7))
    X_te = np.random.default_rng(1).normal(size=(10, 7))
    y = np.arange(50.0)
    for itr, iva in [(np.arange(40), np.arange(40, 50)), (np.arange(10, 50), np.arange(10))]:
        M.fit_predict_fixed(X[itr], y[itr], X[iva], Rec, None)
    M.fit_predict_fixed(X, y, X_te, Rec, None)
    assert set(widths) == {7}


def test_the_meta_learner_that_counts_ignores_the_test_score():
    from experiments.papers import meta_learner_own_hybrids as M
    meta = {'a': {'train_rows_cv_score': 0.5, 'heldout_score': 0.99},
            'b': {'train_rows_cv_score': 0.7, 'heldout_score': 0.10},
            'c': {'train_rows_cv_score': None, 'heldout_score': 0.99}}
    assert M.meta_learner_that_counts(meta, lower=False) == 'b'
    assert M.meta_learner_that_counts(meta, lower=True) == 'a'


def test_best_base_pair_is_over_every_candidate():
    from experiments.papers import meta_learner_own_hybrids as M
    s = {'rf': {'base_rep': 'ecfp4', 'base_score': 0.6},
         'svm': {'base_rep': 'pdv', 'base_score': 0.4},
         'gp': {'base_rep': None, 'base_score': None}}
    assert M.best_base_pair(s, lower=True)['model'] == 'svm'
    assert M.best_base_pair(s, lower=False)['representation'] == 'ecfp4'


def test_summary_joins_c11_on_the_same_fold(tmp_path):
    from experiments.papers import meta_learner_summary as S
    (tmp_path / 'meta_learner').mkdir()
    (tmp_path / 'meta_learner' / 'logd__meta_f1.json').write_text(json.dumps({
        'result_status': 'ok', 'dataset': 'logd', 'fold': 1, 'metric': 'rmse',
        'comparison': {'meta_learner_that_counts': {'name': 'ridge_a1', 'heldout_score': 0.4},
                       'best_model_hybrid': {'heldout_score': 0.5},
                       'plain_average_of_5': {'heldout_score': 0.45},
                       'best_model_base_pair': {'heldout_score': 0.6}}}))
    d = tmp_path / 'complexity' / 'c11_ensemble_vs_hybrid'
    d.mkdir(parents=True)
    (d / 'logd.json').write_text(json.dumps({'results': [
        {'model': 'svm', 'method': 'ensemble_s1_auto', 'fold_scores': [9, 0.41]},
        {'model': 'optimal', 'method': 'ensemble_s2_auto', 'fold_scores': [9, 0.42]},
        {'model': 'svm', 'method': 'hybrid_top1', 'fold_scores': [9, 9]}]}))
    r = S.table(tmp_path)[0]
    assert r['meta_learner'] == 'ridge_a1' and r['ensemble_svm'] == 0.41
    assert r['ensemble_best_model_per_representation'] == 0.42


def test_a_saved_prediction_is_tied_to_its_recipe():
    from experiments.papers import meta_learner_own_hybrids as M
    assert M.recipe_tag('rf', 'a') == M.recipe_tag('rf', 'a')
    assert M.recipe_tag('rf', 'a') != M.recipe_tag('rf', 'b')
