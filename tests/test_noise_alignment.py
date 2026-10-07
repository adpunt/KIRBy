"""Shared noise-suite pieces added for chat 5 of HANDOFF_EXPERIMENT_ALIGNMENT.md
(2026-10-05): outer folds, fold spread, the decided noise grid, the per-dataset
neighbour screen, and the counting scheme reaching the neighbour cleanups.
Synthetic data only; no search runs."""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), '..')
for _p in ('src', 'tests', 'tests/experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _p))

from experiments.noise import noise_helpers as nh  # noqa: E402
from experiments.noise import n07_remediation as n07  # noqa: E402
from experiments._shared import get_domain_metric  # noqa: E402


def _data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 20).astype(np.float32)
    X2 = (rng.rand(n, 64) > 0.7).astype(np.float32)
    y = X1[:, 0] * 2 + rng.randn(n) * 0.3
    return {'reps': {'a': X1, 'b': X2}, 'labels': y,
            'groups': np.repeat(np.arange(n // 10), 10),
            'task_type': 'regression', 'smiles': None, 'augmentations': None}


def test_outer_folds_any_count(monkeypatch):
    d = _data()
    monkeypatch.delenv('KIRBY_NOISE_ONLY_FOLDS', raising=False)
    monkeypatch.delenv('KIRBY_NOISE_N_FOLDS', raising=False)
    folds, n = nh.noise_outer_folds(d, {'n_splits': 5})
    assert n == 5 and len(folds) == 5
    for _, tr, va in folds:   # groups never straddle
        assert not set(d['groups'][tr]) & set(d['groups'][va])
    monkeypatch.setenv('KIRBY_NOISE_N_FOLDS', '1')
    folds, n = nh.noise_outer_folds(d, {'n_splits': 5})
    assert n == 1 and len(folds) == 1
    monkeypatch.setenv('KIRBY_NOISE_N_FOLDS', '5')
    monkeypatch.setenv('KIRBY_NOISE_ONLY_FOLDS', '0')
    folds, _ = nh.noise_outer_folds(d, {'n_splits': 5})
    assert [i for i, _, _ in folds] == [0]


def test_fold_spread_one_fold():
    assert nh.fold_spread([0.3])['note'] == 'one fold, no spread'
    assert nh.fold_spread([0.3, 0.5])['sd'] is not None


def test_decided_grid():
    d = _data()
    assert nh.decided_conditions(d) == list(nh.DECIDED_REGRESSION_CONDITIONS)
    assert nh.decided_levels(d, 'x', 'gaussian') == (0.0, 0.5, 1.0, 1.5)
    assert nh.decided_levels(d, 'x', 'censoring') == (0.0, 0.10, 0.25, 0.40)
    big = dict(d, labels=np.zeros(6000), smiles=[str(i) for i in range(6000)])
    assert nh.decided_levels(big, 'x', 'gaussian') == (0.0, 1.0)
    _, tr, _ = nh.noise_outer_folds(d, {'n_splits': 5})[0][0]
    for c in nh.decided_conditions(d):
        y0, m0, _ = nh.inject_condition(d, 'x', c, 0.0, 0, train_idx=tr,
                                        groups=d['groups'])
        assert np.array_equal(y0, d['labels'][tr]) and not m0.any()
        top = nh.decided_levels(d, 'x', c)[-1]
        y1, _, p = nh.inject_condition(d, 'x', c, top, 0, train_idx=tr,
                                       groups=d['groups'])
        assert len(y1) == len(tr) and not np.array_equal(y1, y0), c
        assert p['level_unit']


def test_screen_and_scheme():
    d = _data()
    (_, tr, va), = nh.noise_outer_folds(d, {'n_splits': 5})[0][:1]
    metric = get_domain_metric(d)
    entry = {'base': ['a', 'b'], 'hybrids': [], 'recipes': {},
             'unavailable': {}}
    ins, un, names = nh.fold_model_inputs(
        d, 'synthetic', 'svm', entry, train_idx=tr, val_idx=va,
        y_train=d['labels'][tr], metric=metric)
    assert set(ins) == {'base_1', 'base_2'} and 'hybrid' in un
    assert names == {'base_1': 'a', 'base_2': 'b'}
    sets = [s for s in n07.sweep_settings('synthetic', len(d['labels']),
                                          has_smiles=False)
            if s.space in ('base_1', 'base_2')]
    assert all(s.space != 'model_input' for s in sets)
    sp_tr = {k: v[0] for k, v in ins.items()}
    sp_va = {k: v[1] for k, v in ins.items()}
    nb = n07.fold_neighbours('synthetic', sets, sp_tr, sp_va, d['labels'][tr])
    kept, table = n07.screen_settings(sets, nb, d['labels'][tr],
                                      metric=metric, is_clf=False)
    assert len(kept) == 3
    assert len(table) == len(sets) * 3 * 2   # settings x ks x schemes
    best = min(r['score'] for r in table)    # rmse: lower is better
    assert any(r['kept'] and r['score'] == best for r in table)
    s, k, sch = kept[0]
    scores = np.abs(np.random.RandomState(1).randn(len(tr)))
    a = n07.cleaned_training_set('knn_smooth', sp_tr['base_1'],
                                 d['labels'][tr], scores, 0.1,
                                 similarity_neighbours=nb[s.name],
                                 similarity_k=k)
    b = n07.cleaned_training_set('knn_smooth', sp_tr['base_1'],
                                 d['labels'][tr], scores, 0.1,
                                 similarity_neighbours=dict(nb[s.name],
                                                            scheme=sch),
                                 similarity_k=k)
    assert not np.allclose(a['y'], b['y'])


def _measurement_data():
    """12 measurement rows of 5 molecules, repeats interleaved, the way
    herg_ic50's loader returns them: every row of one molecule carries the
    same features, SMILES and scaffold group."""
    mol = np.array([3, 0, 3, 1, 0, 2, 3, 4, 1, 2, 0, 3])
    y = np.arange(12, dtype=np.float64) * 0.5 + 1.0
    feats = np.arange(5 * 4, dtype=np.float64).reshape(5, 4)
    return {'reps': {'a': feats[mol], 'b': (feats[mol] > 7).astype(np.float64)},
            'labels': y, 'groups': mol % 3, 'replicate_groups': mol,
            'task_type': 'regression',
            'smiles': [f'C{m}' for m in mol], 'augmentations': None}, mol


def test_molecule_mean_data_one_row_per_molecule():
    d, mol = _measurement_data()
    out = nh.molecule_mean_data(d, 'toy')
    # molecules in order of their first measurement's row: 3, 0, 1, 2, 4
    assert list(out['row_index']) == [0, 1, 3, 5, 7]
    order = [3, 0, 1, 2, 4]
    assert list(out['n_measurements']) == [int((mol == m).sum()) for m in order]
    for j, m in enumerate(order):
        assert out['labels'][j] == np.mean(d['labels'][mol == m])
        assert out['reps']['a'][j].tolist() == d['reps']['a'][mol == m][0].tolist()
        assert out['groups'][j] == m % 3 and out['smiles'][j] == f'C{m}'
    assert out['reps']['a'].dtype == np.float32
    assert out['dataset_name'] == 'toy' and out['augmentations'] is None
    # deterministic: a second call gives the same rows
    again = nh.molecule_mean_data(d, 'toy')
    assert np.array_equal(again['labels'], out['labels'])
    assert np.array_equal(again['row_index'], out['row_index'])
    # no repeat structure -> None
    d2 = {k: v for k, v in d.items() if k != 'replicate_groups'}
    assert nh.molecule_mean_data(d2, 'toy') is None


def test_four_experiments_share_the_molecule_mean_rows_and_name():
    """n17, n19, n26 and n29 build their molecule rows through
    molecule_mean_data and file their searches under one name."""
    from experiments.noise import n17_replicate_removal as n17
    from experiments.noise import n19_measurement_count as n19
    from experiments.noise import n26_noise_variance_from_repeats as n26
    from experiments.noise import n29_document_bias as n29
    shared = nh.MOLECULE_MEAN_SEARCH_SUFFIX
    assert n17.SEARCH_SUFFIX == shared
    assert n19.MOLECULE_MEAN_SEARCH_SUFFIX == shared
    assert n19.SEARCH_CONDITION_SUFFIX == shared
    assert n26.SEARCH_CONDITION_SUFFIX == shared
    assert n29.SEARCH_CONDITION_SUFFIX == shared
    d, mol = _measurement_data()
    ref = nh.molecule_mean_data(d, 'toy')
    ent = n17.build_entities(d['labels'], mol, seed=0)
    stats = n19.measurement_stats(d['labels'], mol)
    built = [n17.entity_data(d, ent),
             n19._molecule_rows_data(d, stats, stats['mean_all'], 'toy'),
             n26.molecule_rows_data(d, stats, 'toy'),
             n29._molecule_rows_data(d, stats, 'toy')]
    for b in built:
        assert np.array_equal(b['labels'], ref['labels'])
        assert np.array_equal(b['groups'], ref['groups'])
        for r in ref['reps']:
            assert np.array_equal(b['reps'][r], ref['reps'][r])
    # a different molecule order is refused, not silently used
    with pytest.raises(RuntimeError):
        nh.check_molecule_order(ref, ref['row_index'][::-1], who='t')
