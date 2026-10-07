"""s02 importance-weighted distances (restored 2026-10-07, Adelaide).

A hybrid's weights come from ONE importance fit on the hybrid's own feature
matrix (training rows only), one weight per hybrid feature; the weights are
applied after the features are scaled on the training rows.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('src', 'tests', 'tests/experiments'):
    q = os.path.join(ROOT, p)
    if q not in sys.path:
        sys.path.insert(0, q)

s02 = pytest.importorskip('experiments.similarity.s02_hybrid_similarity')
from experiments.similarity.similarity_helpers import scale_fold  # noqa: E402

AXES = {'metrics': ['euclidean'], 'ks': [5], 'schemes': ['rank']}


def _synthetic(n=80, n_feat=7, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, n_feat)) * np.arange(1, n_feat + 1) + 3.0
    y = X[:, 0] - 0.5 * X[:, 3] + rng.normal(scale=0.1, size=n)
    tr, te = np.arange(0, 50), np.arange(50, n)
    return X, y, tr, te


def test_import():
    assert callable(s02.run)
    assert s02._RESULT_FORMAT.endswith('2026_10_07')


def test_hybrid_weights_one_per_feature_real_importance():
    X, y, tr, _ = _synthetic()
    got = s02._hybrid_weightings(X[tr], y[tr], ['lightgbm_gain'])
    w, info = got['lightgbm_gain']
    assert not isinstance(w, str), w
    assert w.shape == (X.shape[1],)
    assert info['n_features'] == X.shape[1]
    assert info['importance_fit_rows'] == len(tr)
    assert info['n_features_nonzero_importance'] == int(np.sum(w > 0))


def test_hybrid_importance_is_one_fit_on_hybrid_training_rows(monkeypatch):
    X, y, tr, te = _synthetic()
    calls = []

    def spy(X_train, y_train, method):
        calls.append((np.array(X_train), np.array(y_train), method))
        return np.linspace(0.1, 1.0, X_train.shape[1])

    monkeypatch.setattr(s02, '_importance_weights', spy)
    monkeypatch.setattr(
        s02, 'build_fold_entry',
        lambda data, entry, train_idx, val_idx, **kw: (
            X[train_idx], X[val_idx], {}))
    monkeypatch.setattr(s02, 'entry_record', lambda e: {})
    data = {'labels': y}
    out = s02._score_hybrid(
        data, {}, model=None, train_idx=tr, test_idx=te, factories={},
        metric_fn=None, dataset_lower=False, weight_methods=['lightgbm_gain'],
        axes=AXES, n_sample=500, is_clf=False)
    # One fit per method, on the whole hybrid matrix, training rows only.
    assert len(calls) == 1
    Xc, yc, m = calls[0]
    assert m == 'lightgbm_gain'
    np.testing.assert_array_equal(Xc, X[tr])
    np.testing.assert_array_equal(yc, y[tr])
    rows = {r['weighting']: r for r in out['scores']}
    assert set(rows) == {'unweighted', 'lightgbm_gain'}
    assert rows['lightgbm_gain']['n_features'] == X.shape[1]
    assert rows['lightgbm_gain'].get('sac_rho') is not None


def test_weights_applied_after_training_row_scaling(monkeypatch):
    X, y, tr, te = _synthetic()
    w = np.linspace(0.2, 2.0, X.shape[1])
    seen = []
    real = s02.compute_distance_matrix

    def spy(Xs, metric=None, feature_weights=None, standardize=True, **kw):
        seen.append((np.array(Xs), feature_weights, standardize))
        return real(Xs, metric=metric, feature_weights=feature_weights,
                    standardize=standardize, **kw)

    monkeypatch.setattr(s02, 'compute_distance_matrix', spy)
    rows = s02._score_space(
        X[tr], X[te], y[te], axes=AXES, n_sample=500, is_clf=False,
        weightings={'lightgbm_gain': (w, {})})
    assert [r['weighting'] for r in rows] == ['unweighted', 'lightgbm_gain']
    _, B = scale_fold(X[tr], X[te])
    assert len(seen) == 2
    for Xs, fw, std in seen:
        # held-out rows scaled with the TRAINING-row scaler, not re-standardised
        np.testing.assert_allclose(Xs, B)
        assert std is False
    assert seen[0][1] is None
    np.testing.assert_allclose(seen[1][1], w)
    # Euclidean: weight (mean 1, exponent 0.5) multiplies the scaled features.
    D = s02._held_out_distance(B, 'euclidean', w)
    Bw = B * np.sqrt(w / w.mean())
    ref = np.sqrt(((Bw[:, None, :] - Bw[None, :, :]) ** 2).sum(-1))
    np.testing.assert_allclose(D, ref, rtol=1e-6, atol=1e-8)


def test_comparisons_pair_weighted_hybrid_with_weighted_single():
    h = {'scores': [{'distance': 'euclidean', 'weighting': 'unweighted',
                     'sac_rho': 0.3},
                    {'distance': 'euclidean', 'weighting': 'lightgbm_gain',
                     'sac_rho': 0.5}]}
    s = {'scores': [{'distance': 'euclidean', 'weighting': 'unweighted',
                     'sac_rho': 0.2},
                    {'distance': 'euclidean', 'weighting': 'lightgbm_gain',
                     'sac_rho': 0.1}]}
    out = s02._comparisons({'r': s}, {'h': h}, [('c', 'h', 'r')], axes=AXES,
                           weight_methods=['lightgbm_gain'])
    rows = {r['hybrid_weighting']: r for r in out['c']['rows']}
    assert rows['unweighted']['delta'] == pytest.approx(0.1)
    assert rows['lightgbm_gain']['single_weighting'] == 'lightgbm_gain'
    assert rows['lightgbm_gain']['delta'] == pytest.approx(0.4)
