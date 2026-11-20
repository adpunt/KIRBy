"""A training set that two noise-handling methods both produce is fitted once.

Adelaide, 2026-11-04: "just make sure the same thing doesn't get run multiple
times". In n44 (and every experiment that scores through
d_vs_dprime.score_d_dprime) two methods can hand back the same cleaned training
set, or hand back the damaged training set unchanged. Each method still gets
its own result under its own name. This test checks that the model fit behind
the second one is read from noise_cache, not run again: the fit key is the
content of the features, labels, weights and held-out molecules plus the
model, never the method's name.

Run: PYTHONPATH=src:tests:tests/experiments pytest tests/test_repeated_training_set_fitted_once.py
"""
import os

import numpy as np
import pytest


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('KIRBY_NOISE_CACHE', '1')
    from experiments.noise import noise_cache as nc
    nc._COUNTS.clear()
    return nc


def _misses(nc):
    return nc.report().get('fit_predict', {}).get('misses', 0)


def _data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 5))
    y = X @ rng.normal(size=5) + 0.1 * rng.normal(size=60)
    return X[:45], y[:45], X[45:], y[45:]


def _score(nc, cleaned, X_tr, y_tr, X_va, y_va):
    from experiments.noise import d_vs_dprime as dvd
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    return dvd.score_d_dprime('repeat_test', cleaned, X_tr, y_tr, X_va, y_va,
                              metric=('rmse', rmse, True),
                              task_type='regression', models=('rf',))


def test_same_cleaned_set_under_two_names_is_one_fit(fresh_cache):
    nc = fresh_cache
    X_tr, y_tr, X_va, y_va = _data()
    keep = np.arange(len(y_tr)) % 5 != 0
    first = {'X': X_tr[keep], 'y': y_tr[keep], 'sample_weight': None}
    # The same molecules removed by a second method: new arrays, same content.
    second = {'X': X_tr[keep].copy(), 'y': y_tr[keep].copy(), 'sample_weight': None}

    a = _score(nc, first, X_tr, y_tr, X_va, y_va)
    after_first = _misses(nc)
    b = _score(nc, second, X_tr, y_tr, X_va, y_va)
    assert after_first == 2            # D, then the cleaned set
    assert _misses(nc) == after_first  # nothing fitted for the second name
    assert a['rf']['score_Dprime'] == b['rf']['score_Dprime']


def test_unchanged_training_set_is_not_fitted_again(fresh_cache):
    nc = fresh_cache
    X_tr, y_tr, X_va, y_va = _data()
    _score(nc, {'X': X_tr, 'y': y_tr, 'sample_weight': None},
           X_tr, y_tr, X_va, y_va)
    # A method that changed nothing hands back D: its fit is D's fit.
    assert _misses(nc) == 1


def test_a_different_training_set_is_fitted(fresh_cache):
    nc = fresh_cache
    X_tr, y_tr, X_va, y_va = _data()
    keep = np.arange(len(y_tr)) % 5 != 0
    _score(nc, {'X': X_tr[keep], 'y': y_tr[keep], 'sample_weight': None},
           X_tr, y_tr, X_va, y_va)
    n = _misses(nc)
    _score(nc, {'X': X_tr, 'y': y_tr, 'sample_weight': np.where(keep, 1.0, 0.5)},
           X_tr, y_tr, X_va, y_va)
    assert _misses(nc) == n + 1        # a weighted set is a new fit
