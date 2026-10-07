"""c10 Part 3 (restored 2026-10-07): one search on damaged training labels per
(kind, non-zero level, draw) per fold, over c10's own label sets, compared
with the clean-label search. Synthetic data; the search is replaced by a stub
that records what it was asked for. The damaged labels are the real
inject_condition's."""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), '..')
for _p in ('src', 'tests', 'tests/experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _p))

from experiments.noise import c10_noise_robustness as c10  # noqa: E402
from experiments.noise import noise_helpers as nh  # noqa: E402

SCREEN_KINDS = ['gaussian', 'outlier_p05', 'grouped_shifted', 'censoring',
                'shift_only_p10']


def _data(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X1 = rng.randn(n, 6).astype(np.float32)
    X2 = rng.randn(n, 4).astype(np.float32)
    y = X1[:, 0] * 2 + rng.randn(n) * 0.3
    return {'reps': {'a': X1, 'b': X2}, 'labels': y,
            'groups': np.repeat(np.arange(n // 10), 10),
            'task_type': 'regression', 'smiles': None, 'augmentations': None}


def _hyb(alloc, budget):
    return {'label': f'{alloc}_{budget}', 'score_mean': 0.5,
            'allocation_method': alloc, 'importance_method': 'lightgbm_gain',
            'budget': budget, 'internal_evaluator': 'lightgbm',
            'step_size': None, 'rep_count': 2, 'aug_condition': 'none'}


CLEAN_ENTRY = {'base': ['a', 'b'], 'hybrids': [_hyb('fixed', 50)],
               'recipes': {}}


def _stub_search(calls, labels):
    def fake(reps, lab, groups, *, train_idx, train_labels=None,
             label_noise=None, condition_suffix=None, n_hybrids=None,
             **kw):
        assert label_noise is None
        tr = np.asarray(train_idx)
        calls.append({'suffix': condition_suffix, 'n_hybrids': n_hybrids,
                      'labels_clean': np.array_equal(lab, labels),
                      'trained_differs': not np.allclose(
                          np.asarray(train_labels)[tr], labels[tr])})
        # gaussian moves the recipe and the representations; the rest do not
        if 'gaussian' in condition_suffix:
            return {'m1': {'base': ['b', 'a'], 'hybrids': [_hyb('fixed', 100)]}}
        return {'m1': {'base': ['a', 'b'], 'hybrids': [_hyb('fixed', 50)]}}
    return fake


def _screen_cells(data, name='synthetic'):
    kinds = nh.decided_conditions(data)
    return [c10.CLEAN_CELL] + [
        (k, float(lv), int(d)) for k in kinds
        for lv in nh.decided_levels(data, name, k) if float(lv) != 0.0
        for d in nh.draws_for(k, c10.NOISE_DRAWS)]


def test_screen_kinds_and_count(monkeypatch):
    monkeypatch.delenv('KIRBY_NOISE_PROMOTE', raising=False)
    d = _data()
    cells = c10.recipe_movement_cells(_screen_cells(d))
    assert sorted({c[0] for c in cells}) == sorted(SCREEN_KINDS)
    assert {(c[0], c[1]) for c in cells} == {
        (k, 0.25 if k == 'censoring' else 1.0) for k in SCREEN_KINDS}
    # 3 draws on every kind but censoring (1): 4*3 + 1
    assert len(cells) == 13


def test_fold_one_search_per_cell_and_summary(monkeypatch):
    monkeypatch.delenv('KIRBY_NOISE_PROMOTE', raising=False)
    d = _data()
    calls = []
    monkeypatch.setattr(c10, 'fold_standard_selection',
                        _stub_search(calls, d['labels']))
    tr = np.arange(0, 90)
    cells = _screen_cells(d)
    rows, skipped = c10.recipe_movement_fold(
        d, 'synthetic', fold_i=0, tr=tr, n_folds=4,
        entries={'m1': CLEAN_ENTRY}, cells=cells,
        noise_groups=d['groups'], noise_seed=42)
    assert skipped == {}
    assert len(calls) == 13
    assert len({c['suffix'] for c in calls}) == 13
    assert all(c['suffix'].startswith('c10p3_') for c in calls)
    assert all(c['n_hybrids'] == 1 and c['labels_clean'] for c in calls)
    # every kind actually damaged the training column it searched on
    assert all(c['trained_differs'] for c in calls)
    assert len(rows) == 13 and all(r['compared'] for r in rows)

    mv = c10.recipe_movement_summary(rows)
    g = mv['m1']['gaussian']['1.0']
    assert g['n_compared'] == 3 and g['n_recipe_differs'] == 3
    assert g['n_base_differs'] == 3
    assert g['n_field_differs']['budget'] == 3
    assert g['n_field_differs']['allocation_method'] == 0
    c = mv['m1']['censoring']['0.25']
    assert c['n_compared'] == 1 and c['n_recipe_differs'] == 0
    assert mv['m1']['shift_only_p10']['1.0']['n_compared'] == 3
    assert mv['m1']['grouped_shifted']['1.0']['n_compared'] == 3


def test_run_wires_part3(monkeypatch, tmp_path):
    """run() end to end with the per-fold search and the fits stubbed: Part 3
    runs over Part 1's own label sets on every fold and the result carries
    the movement summary."""
    monkeypatch.delenv('KIRBY_NOISE_PROMOTE', raising=False)
    monkeypatch.delenv('KIRBY_NOISE_ONLY_FOLDS', raising=False)
    monkeypatch.delenv('KIRBY_NOISE_N_FOLDS', raising=False)
    d = _data()
    calls = []
    monkeypatch.setattr(c10, 'HAS_NOISEINJECT', True)
    monkeypatch.setattr(c10, 'fold_standard_selection',
                        _stub_search(calls, d['labels']))
    monkeypatch.setattr(c10, 'dataset_knn_group', lambda *a: 'full_knn')
    monkeypatch.setattr(c10, 'noise_groups_for',
                        lambda data: (data['groups'], 12, 10 / 120))
    monkeypatch.setattr(c10, 'fold_configurations',
                        lambda *a, **k: {'m1': CLEAN_ENTRY})
    monkeypatch.setattr(c10, 'get_model_factories', lambda *a, **k: {})
    res = c10.run(d, {'n_splits': 2}, str(tmp_path), 'synthetic')
    assert res['result_schema'] == c10.RESULT_SCHEMA
    assert res['folds_done_recipe_movement'] == [0, 1]
    assert len(calls) == 2 * 13
    assert len(res['recipe_movement_rows']) == 2 * 13
    assert res['recipe_movement']['m1']['gaussian']['1.0']['n_compared'] == 6
    assert sorted(res['recipe_movement']['m1']) == sorted(SCREEN_KINDS)
