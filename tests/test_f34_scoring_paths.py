#!/usr/bin/env python3
"""f34's precomputed-matrix path scores what the raw-features path scores.

f34 computes one distance matrix per (representation, coefficient, fold) and
fits sklearn with metric='precomputed' for the 36 cells that share it, because
fitting on raw features recomputes those distances 36 times over. That is only
sound if the two give the same score.

They agree on clean data, and this file asserts it on all 24 (coefficient,
weighting scheme) pairs. They can disagree when molecules sit at exactly equal
distance and a tie straddles the k-th neighbour, which is why the two paths are
also measured against each other on the real datasets by f35 rather than only
asserted here.
"""
import sys
import os
import warnings

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'tests'))
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'experiments', 'framework'))

from sklearn.metrics import mean_squared_error  # noqa: E402

from kirby.similarity import distance_config, neighbour_weighting  # noqa: E402
import f34_neighbour_weighting_cross as f34  # noqa: E402

K_VALUES = (5, 10, 25)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _data(seed=0, n_train=120, n_val=40, n_features=12):
    """Continuous features drawn at random, so no two molecules tie exactly."""
    rng = np.random.RandomState(seed)
    return rng.rand(n_train, n_features), rng.rand(n_val, n_features), rng


@pytest.mark.parametrize('coefficient', distance_config.DISTANCE_METRICS)
@pytest.mark.parametrize('scheme', neighbour_weighting.WEIGHTING_SCHEMES)
def test_the_two_paths_score_the_same_on_data_with_no_ties(coefficient,
                                                           scheme):
    if distance_config.resolve(coefficient) in distance_config.OVERLAP_METRICS:
        pytest.skip(
            'no raw-features path exists: sklearn has no metric string for the '
            'continuous min/max Tanimoto or Dice, so these two coefficients '
            'reach a neighbour search only as a precomputed matrix. There is '
            'nothing to compare the precomputed path against.')
    X_tr, X_va, rng = _data()
    y_tr, y_va = rng.rand(len(X_tr)), rng.rand(len(X_va))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d_tr, d_va = f34.distance_matrices(X_tr, X_va, coefficient)
        for k in K_VALUES:
            pre, _ = f34.score_precomputed_cell(
                d_tr, y_tr, d_va, y_va, k=k, scheme=scheme,
                task_type='regression', metric_fn=rmse)
            raw, _ = f34.score_one_cell(
                X_tr, y_tr, X_va, y_va, k=k, scheme=scheme,
                coefficient=coefficient, task_type='regression',
                metric_fn=rmse)
        assert np.isclose(pre, raw, atol=1e-9), (
            f'{coefficient}/{scheme}/k={k}: precomputed {pre:.10f} against '
            f'raw features {raw:.10f}')


@pytest.mark.parametrize('coefficient', distance_config.DISTANCE_METRICS)
def test_the_distance_matrices_have_the_right_shape_and_are_finite(coefficient):
    X_tr, X_va, _rng = _data(seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d_tr, d_va = f34.distance_matrices(X_tr, X_va, coefficient)
    assert d_tr.shape == (len(X_tr), len(X_tr))
    assert d_va.shape == (len(X_va), len(X_tr))
    assert np.all(np.isfinite(d_tr)) and np.all(np.isfinite(d_va))


def test_the_fractional_minkowski_is_not_the_euclidean_number():
    """scipy returns p=2 for metric='minkowski' with no p, which is Euclidean.

    distance_config.scipy_metric carries p in its kwargs so that cannot
    happen. This asserts the two coefficients give different matrices.
    """
    X_tr, X_va, _rng = _data(seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frac, _ = f34.distance_matrices(X_tr, X_va, 'minkowski_p0.5')
        eucl, _ = f34.distance_matrices(X_tr, X_va, 'euclidean')
    assert not np.allclose(frac, eucl)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
