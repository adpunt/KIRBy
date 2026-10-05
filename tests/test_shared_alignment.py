"""Shared code changed for HANDOFF_EXPERIMENT_ALIGNMENT.md chat 1 (2026-10-05):
the 0/1 scaling rule, the decided grid in every config, no start-up searches,
and the identical-features check."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('KIRBY_CACHE_DIR', '/tmp')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def test_feature_scaler_leaves_01_features():
    from sklearn.preprocessing import StandardScaler
    from kirby.scaling import FeatureScaler, binary_feature_mask
    r = np.random.RandomState(0)
    X = np.c_[r.randn(40, 3), (r.rand(40, 4) < 0.2)].astype(np.float32)
    sc = FeatureScaler().fit(X)
    assert binary_feature_mask(X).tolist() == [False] * 3 + [True] * 4
    Z = sc.transform(X)
    assert np.array_equal(Z[:, 3:], X[:, 3:])
    assert np.array_equal(Z[:, :3], StandardScaler().fit_transform(X)[:, :3])
    # decided on the training rows: a 2 in a validation row passes through
    V = X[:2].copy()
    V[0, 4] = 2.0
    assert sc.transform(V)[0, 4] == 2.0


def test_similarity_standardise_leaves_01_features():
    from kirby.similarity.preprocessing import standardise
    X = np.c_[np.arange(10.0), np.r_[np.ones(3), np.zeros(7)]]
    Z, mu, sd = standardise(X)
    assert np.array_equal(Z[:, 1], X[:, 1]) and mu[1] == 0 and sd[1] == 1
    Zv, _, _ = standardise(X[:2], mu, sd)
    assert np.array_equal(Zv, Z[:2])


def test_decided_config_replaces_five_models():
    from experiments._shared import decided_config
    from experiments.decided_grid import decided_search, UndecidedSearch
    cfg = decided_config({'models': ['rf', 'xgboost', 'svm', 'dnn', 'lightgbm'],
                          'budgets': [10, 25]}, 'herg_ki', verbose=False)
    g = decided_search('herg_ki')
    assert cfg['models'] == g['models'] and cfg['budgets'] == g['budgets']
    assert cfg['distance_models'] is not None
    with pytest.raises(UndecidedSearch):
        decided_config({}, 'not_a_dataset', verbose=False)


def test_own_search_kwargs_needs_the_dataset():
    from experiments._shared import own_search_kwargs
    with pytest.raises(ValueError):
        own_search_kwargs(100)
    kw = own_search_kwargs(100, dataset_name='herg_ki')
    assert kw['models'] == ['svm', 'dnn'] or set(kw['models']) == {'svm', 'dnn'}
    assert kw['adaptive_pruning_schedule'] is None
    assert kw['model_pruning_aggressiveness'] is None
    kw = own_search_kwargs(100, dataset_name='herg_ki', models=['rf', 'svm'])
    assert kw['models'] == ['svm']


def test_no_startup_searches_leaves_stores_empty():
    from experiments._shared import no_startup_searches
    d = {}
    no_startup_searches(d, 'aggressive_pruning', 'test')
    assert d['precomputed_searches'] == {} and d['precomputed_searches_distance'] == {}


def test_feature_signature_ignores_order_and_names_donors():
    from experiments._shared import hybrid_feature_signature
    a = {'ecfp4': {'selected_indices': np.array([3, 1])},
         'pdv': {'selected_indices': np.array([0])}, 'allocation': {'ecfp4': 2}}
    b = {'pdv': {'selected_indices': np.array([0])},
         'ecfp4': {'selected_indices': np.array([1, 3])}}
    c = dict(b, augmentation={'maccs': {'selected_indices': np.array([5])}})
    assert hybrid_feature_signature(a) == hybrid_feature_signature(b)
    assert hybrid_feature_signature(c) != hybrid_feature_signature(b)


def test_distinct_feature_hybrids_skips_a_duplicate(monkeypatch):
    from experiments import _shared as sh
    sigs = {'A': {'r': {'selected_indices': np.array([0, 1])}},
            'B': {'r': {'selected_indices': np.array([1, 0])}},   # same as A
            'C': {'r': {'selected_indices': np.array([2])}}}

    def fake_build(train_reps, val_reps, y, bp, **kw):
        return None, None, sigs[bp]
    monkeypatch.setattr(sh, 'build_fold_hybrid', fake_build)
    monkeypatch.setattr(sh, 'fold_hybrid_scoring_kwargs', lambda bp, **kw: {})
    pooled = [{'label': k, 'blueprint': k, 'score_mean': i, 'aug_condition': 'none'}
              for i, k in enumerate('ABC')]
    X = np.random.RandomState(0).randn(20, 3)
    chosen, skipped = sh._distinct_feature_hybrids(
        pooled, 2, reps={'r': X}, labels=np.random.RandomState(1).randn(20),
        groups=None, train_idx=np.arange(15), augmentations=None, model='svm',
        dataset_name='herg_ki', task_type='regression', metric='rmse',
        domain='molecular', lower_is_better=True)
    assert [c['label'] for c in chosen] == ['A', 'C']
    assert skipped[0]['label'] == 'B' and skipped[0]['same_features_as'] == 'A'
