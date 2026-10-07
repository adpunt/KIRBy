"""c22 grouped permutation importance (2026-10-07).

A representation whose features determine y gets a large drop; a noise
representation gets about 0; the numbers are identical on a second call.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('src', 'tests', os.path.join('tests', 'experiments')):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)


def _c22():
    import experiments.complexity.c22_explainability as c22
    return c22


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _setup():
    from sklearn.svm import SVR
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=(n, 4))
    noise = rng.normal(size=(n, 6))
    y = signal @ np.array([1.0, -1.0, 0.5, 0.8])
    X = np.hstack([signal, noise])
    fi = {
        'signal_rep': {'selected_indices': np.arange(4), 'n_features': 4},
        'noise_rep': {'selected_indices': np.arange(6), 'n_features': 6},
        'allocation': {'signal_rep': 4, 'noise_rep': 6},
    }
    model = SVR(kernel='linear', C=10.0).fit(X[:200], y[:200])
    return model, X[200:], y[200:], fi


def test_import():
    c22 = _c22()
    assert hasattr(c22, '_grouped_permutation_drop')
    assert c22.RESULT_FORMAT.endswith('2026_10_07')


def test_signal_vs_noise_and_deterministic():
    c22 = _c22()
    model, Xv, yv, fi = _setup()
    out = c22._grouped_permutation_drop(model, Xv, yv, fi, _rmse,
                                        lower_is_better=True, fold=0)
    d = out['perm_drop_by_rep']
    assert d['signal_rep'] > 1.0
    assert abs(d['noise_rep']) < 0.05
    assert out['perm_repeats'] == 5
    assert out['perm_n_predict_calls'] == 1 + 2 * 5
    assert out['perm_share_by_rep']['signal_rep'] == pytest.approx(
        d['signal_rep'] / sum(v for v in d.values() if v > 0))
    again = c22._grouped_permutation_drop(model, Xv, yv, fi, _rmse,
                                          lower_is_better=True, fold=0)
    assert again == out


def test_higher_is_better_orientation():
    from scipy.stats import spearmanr
    c22 = _c22()
    model, Xv, yv, fi = _setup()
    out = c22._grouped_permutation_drop(
        model, Xv, yv, fi, lambda y, p: float(spearmanr(y, p)[0]),
        lower_is_better=False, fold=1)
    assert out['perm_drop_by_rep']['signal_rep'] > 0.5
    assert abs(out['perm_drop_by_rep']['noise_rep']) < 0.05


def test_blocks_follow_feature_info_order():
    c22 = _c22()
    _, _, _, fi = _setup()
    assert c22.rep_feature_blocks(fi) == {'signal_rep': (0, 4),
                                          'noise_rep': (4, 10)}
