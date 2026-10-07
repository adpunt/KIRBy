"""n09: the per-fold dnn search runs on each pass's damaged training labels,
and the importance-weighted copy of hybrid 1 is a fifth feature space.

RESTORED 2026-10-07 (Adelaide). The search, the hybrid builds, the damage and
the network are stubbed; what is checked is which labels reach the search, how
each search is named, and which feature spaces are fitted. Fold 0 only, as
under the noise screen (config['only_folds']).
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('src', 'tests', 'tests/experiments'):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)


def test_n09_imports():
    from experiments.noise import n09_lnl_survey as n09
    assert n09.SEARCH_LABELS == 'damaged'
    assert n09.ALL_SPACES == ('base_1', 'base_2', 'hybrid',
                              'hybrid__general2', 'weighted_hybrid')
    assert n09.RESULT_LAYOUT == 'restored_2026-10-07'


class _Bp:
    reps = ('r1', 'r2')


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    from experiments.noise import n09_lnl_survey as n09
    from experiments.noise import noise_cache as nc
    from experiments.noise import n07_remediation as n07
    import experiments._shared as sh
    import experiments.decided_grid as dg

    n09._DNN_ENTRY_MEMO.clear()
    n09._NEIGHBOUR_MEMO.clear()
    rng = np.random.default_rng(0)
    n = 60
    data = {
        'labels': rng.normal(size=n),
        'groups': np.repeat(np.arange(12), 5),
        'reps': {'r1': rng.normal(size=(n, 4)), 'r2': rng.normal(size=(n, 3))},
        'task_type': 'regression',
        'dataset_name': 'n09_stub',
    }
    levels = [0.0, 0.5, 1.0]
    calls = []

    def fake_search(reps, labels, groups, *, train_idx, condition_suffix=None,
                    train_labels=None, **kw):
        calls.append({'y': np.asarray(labels)[np.asarray(train_idx)].copy(),
                      'suffix': condition_suffix,
                      'train_labels': train_labels,
                      'fold': kw.get('fold_i')})

        class _S:
            lower_is_better = True
        return {'none': _S()}

    def fake_top_base(search, model, n=2, **kw):
        return {'selected': [{'rep': 'r1', 'score': 0.1},
                             {'rep': 'r2', 'score': 0.2}]}

    def fake_top_hybrids(search, model, n=2, **kw):
        return {'selected': [
            {'label': f'h{i}', 'score_mean': 0.1 * i, 'blueprint': _Bp(),
             'internal_evaluator': 'x', 'rep_count': 2} for i in range(3)]}

    monkeypatch.setattr(sh, 'search_on_fold', fake_search)
    monkeypatch.setattr(sh, 'top_base_reps', fake_top_base)
    monkeypatch.setattr(sh, 'top_hybrids', fake_top_hybrids)
    monkeypatch.setattr(dg, 'decided_models', lambda ds, tier: ['dnn', 'rf'])
    monkeypatch.setattr(n09, '_distinct_dnn_hybrids',
                        lambda pooled, k, **kw: (pooled[:k], []))
    monkeypatch.setattr(n09, 'decided_levels', lambda d, ds, c: levels)
    monkeypatch.setattr(n09, 'noise_groups_for', lambda d: (None, 0, None))

    def fake_inject(data, ds, cond, level, seed, *, train_idx, groups):
        y = np.asarray(data['labels'])[np.asarray(train_idx)]
        if level == 0:
            return y.copy(), np.zeros(len(y), bool), {'level': 0}
        return (y + 10 * level + seed, np.ones(len(y), bool),
                {'level': level, 'seed': seed})
    monkeypatch.setattr(n09, 'inject_condition', fake_inject)

    # The hybrid build: raw matrices whose standardised copy is the hybrid
    # arm, with one importance score per selected feature, so the real
    # _weighted_copy runs.
    def raw(idx):
        idx = np.asarray(idx)
        return np.hstack([data['reps']['r1'][idx], data['reps']['r2'][idx]])

    def fake_cached_hybrid(ds, tr_reps, va_reps, y_train, bp, g, model, fn,
                           lower, *, train_idx, want_feature_info=False, **kw):
        n_tr = len(np.asarray(y_train))
        X = np.vstack([tr_reps['r1'].T, tr_reps['r2'].T]).T
        Xv = np.vstack([va_reps['r1'].T, va_reps['r2'].T]).T
        assert X.shape[0] == n_tr
        info = {'r1': {'selected_indices': [0, 1, 2, 3],
                       'importance_scores': np.array([1., 2., 3., 4.])},
                'r2': {'selected_indices': [0, 1, 2],
                       'importance_scores': np.array([1., 1., 4.])}}
        return (X, Xv, info) if want_feature_info else (X, Xv)
    monkeypatch.setattr(nc, 'cached_hybrid_for_fold', fake_cached_hybrid)

    def fake_inputs(data_, ds, model, entry, *, train_idx, val_idx, y_train,
                    factory=None, metric=None, spaces=()):
        tr_s, va_s = n07._scale_pair(raw(train_idx), raw(val_idx))
        out = {'base_1': (data['reps']['r1'][train_idx],
                          data['reps']['r1'][val_idx]),
               'base_2': (data['reps']['r2'][train_idx],
                          data['reps']['r2'][val_idx]),
               'hybrid': (tr_s, va_s),
               'hybrid__general2': (tr_s.copy(), va_s.copy())}
        return ({k: v for k, v in out.items() if k in spaces}, {},
                {k: k for k in spaces})
    monkeypatch.setattr(n09, 'fold_model_inputs', fake_inputs)

    fitted_widths = []

    def fake_baseline(X_tr, y_tr, X_va, y_va, **kw):
        fitted_widths.append(np.asarray(X_tr).shape[1])
        return np.full(len(X_va), float(np.mean(y_tr)))
    monkeypatch.setattr(n09, '_run_baseline', fake_baseline)

    config = {'n_splits': 3, 'only_folds': [0]}
    return dict(n09=n09, data=data, calls=calls, config=config,
                out=str(tmp_path), levels=levels)


def test_one_damaged_search_per_level_and_draw(stubbed):
    n09 = stubbed['n09']
    data = stubbed['data']
    res = n09._run_one_condition(
        data, stubbed['config'], stubbed['out'], 'n09_stub',
        noise_condition='gaussian', methods=[])
    calls = stubbed['calls']
    # Fold 0 only (the screen).
    assert {c['fold'] for c in calls} == {0}
    assert res['folds_run'] == [0]
    # One search for level 0 plus one per non-zero level and draw.
    n_nonzero = len([lv for lv in stubbed['levels'] if lv > 0])
    assert len(calls) == 1 + n_nonzero * len(n09.NOISE_DRAWS)
    # Every search got its own name, and the damaged labels went into the
    # label column itself (no train_labels=).
    assert len({c['suffix'] for c in calls}) == len(calls)
    assert all(c['train_labels'] is None for c in calls)
    folds, _ = n09.noise_outer_folds(data, stubbed['config'])
    tr = folds[0][1]
    y_clean = np.asarray(data['labels'])[tr]
    np.testing.assert_allclose(calls[0]['y'], y_clean)
    expected = [(lv, s) for lv in stubbed['levels'] if lv > 0
                for s in n09.NOISE_DRAWS]
    for c, (lv, s) in zip(calls[1:], expected):
        np.testing.assert_allclose(c['y'], y_clean + 10 * lv + s)
    assert res['search_labels'] == 'damaged'


def test_censoring_runs_one_draw(stubbed):
    n09 = stubbed['n09']
    n09._run_one_condition(stubbed['data'], stubbed['config'], stubbed['out'],
                           'n09_stub', noise_condition='censoring', methods=[])
    n_nonzero = len([lv for lv in stubbed['levels'] if lv > 0])
    assert len(stubbed['calls']) == 1 + n_nonzero


def test_five_feature_spaces_fitted(stubbed):
    n09 = stubbed['n09']
    res = n09._run_one_condition(
        stubbed['data'], stubbed['config'], stubbed['out'], 'n09_stub',
        noise_condition='gaussian', methods=[])
    assert res['configuration_spaces'] == list(n09.ALL_SPACES)
    for lv, agg in res['per_sigma'].items():
        for sp in n09.ALL_SPACES:
            assert agg[f'{sp}_no_lnl']['n_folds'] > 0, (lv, sp)
    for rec in res['configurations']:
        assert rec['unavailable'] == {}
    assert 'ranking_changed_weighted' in res['summary']
