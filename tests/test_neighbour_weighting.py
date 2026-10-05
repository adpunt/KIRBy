#!/usr/bin/env python3
"""The six neighbour-weighting schemes: no NaN, no infinity, and p=0/p=1 pinned.

What one test checks is named in its own docstring. The point of the file is the
zero-distance case: two molecules with identical features are at distance zero,
one over zero is infinite, and a scheme that lets that through turns a whole
fold into NaN.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.similarity import neighbour_weighting as nw  # noqa: E402

# 3 molecules x 4 neighbours. Row 0 has one exact match, row 1 has two, row 2
# has none. Distances are in whatever unit the coefficient produced; only their
# relative size matters here.
DIST = np.array([
    [0.0, 0.5, 1.0, 2.0],
    [0.0, 0.0, 3.0, 4.0],
    [0.25, 0.5, 1.0, 2.0],
])


@pytest.mark.parametrize('scheme', nw.WEIGHTING_SCHEMES)
def test_no_nan_or_infinity_when_a_neighbour_is_an_exact_match(scheme):
    """Every scheme returns finite weights on distances containing a zero."""
    w = nw.weight_array(DIST, scheme)
    assert w.shape == DIST.shape
    assert np.all(np.isfinite(w)), f'{scheme} produced {w}'


@pytest.mark.parametrize('scheme', nw.WEIGHTING_SCHEMES)
def test_every_molecule_has_at_least_one_neighbour_with_weight(scheme):
    """No row sums to zero, which would make the prediction undefined."""
    w = nw.weight_array(DIST, scheme)
    assert np.all(w.sum(axis=1) > 0), f'{scheme} produced {w}'


@pytest.mark.parametrize('scheme', ['inverse_sqrt', 'inverse', 'inverse_square'])
def test_exact_matches_take_all_the_weight_in_the_power_family(scheme):
    """The three schemes that divide by the distance follow sklearn's rule.

    sklearn's own 'distance' gives the zero-distance neighbours a weight of 1
    and the rest of that row 0. The two written callables do the same.
    """
    w = nw.weight_array(DIST, scheme)
    assert np.allclose(w[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(w[1], [1.0, 1.0, 0.0, 0.0])


def test_gaussian_does_not_invoke_the_guard_when_the_median_is_positive():
    """exp(-(0/h)**2) is 1, so an exact match needs no rescue while h > 0.

    Row 0's median neighbour distance is 0.75, so its four weights are a smooth
    curve starting at 1, not the 1/0/0/0 mask the power family gives.
    """
    w = nw.weight_array(DIST, 'gaussian')
    assert np.isclose(w[0, 0], 1.0)
    assert w[0, 1] > 0.0
    assert w[0, 2] > 0.0
    assert w[0, 1] > w[0, 2] > w[0, 3]


def test_gaussian_falls_back_to_the_exact_match_mask_when_the_median_is_zero():
    """With at least half the neighbours identical, h is 0 and d/h is 0/0."""
    d = np.array([[0.0, 0.0, 0.0, 5.0]])
    w = nw.weight_array(d, 'gaussian')
    assert np.all(np.isfinite(w))
    assert np.allclose(w[0], [1.0, 1.0, 1.0, 0.0])


def test_rank_ignores_how_far_away_the_neighbours_are():
    """Two rows with the same neighbour ORDER get the same weights.

    Row 2's distances are 0.25, 0.5, 1.0, 2.0 and the row below is 10x those.
    Under rank the weights are identical; under every other scheme they are not.
    """
    near = np.array([[0.25, 0.5, 1.0, 2.0]])
    far = near * 10.0
    assert np.allclose(nw.weight_array(near, 'rank'),
                       nw.weight_array(far, 'rank'))
    assert np.allclose(nw.weight_array(near, 'rank')[0],
                       [1.0, 0.5, 1.0 / 3.0, 0.25])


def test_rank_does_not_depend_on_the_order_the_distances_arrive_in():
    """argsort, not the caller's ordering, decides the rank."""
    shuffled = np.array([[2.0, 0.25, 1.0, 0.5]])
    assert np.allclose(nw.weight_array(shuffled, 'rank')[0],
                       [0.25, 1.0, 1.0 / 3.0, 0.5])


def test_the_two_sklearn_builtins_travel_as_strings_not_callables():
    """p=0 and p=1 pin the family to what the suite already runs.

    Handing sklearn its own 'uniform' and 'distance' means the sweep's two
    endpoint arms are the code s04, s05, s10, noise/ and similarity_metrics/
    execute today, not a reimplementation that might differ.
    """
    assert nw.sklearn_weights('uniform') == 'uniform'
    assert nw.sklearn_weights('inverse') == 'distance'
    for scheme in ('inverse_sqrt', 'inverse_square', 'gaussian', 'rank'):
        assert callable(nw.sklearn_weights(scheme))


def test_weight_array_reproduces_what_sklearn_computes_for_its_own_distance():
    """The out-of-sklearn path and sklearn's 'distance' agree, zeros included."""
    from sklearn.neighbors._base import _get_weights
    mine = nw.weight_array(DIST, 'inverse')
    theirs = _get_weights(DIST.copy(), 'distance')
    assert np.allclose(mine, theirs)


def test_the_power_family_orders_the_schemes_by_how_steeply_they_prefer_close():
    """Larger p puts more of the weight on the nearest neighbour.

    Measured as the nearest neighbour's share of the row's total weight, on a
    row with no exact match.
    """
    row = DIST[2:3]

    def share(scheme):
        w = nw.weight_array(row, scheme)
        return float(w[0, 0] / w[0].sum())

    assert (share('uniform') < share('inverse_sqrt') < share('inverse')
            < share('inverse_square'))


def test_canonical_raises_on_a_spelling_that_is_not_a_scheme():
    """A silent fallback is the failure this module exists to end."""
    with pytest.raises(ValueError, match='unknown neighbour weighting scheme'):
        nw.canonical('nonsense')


def test_canonical_accepts_the_aliases_the_repo_already_writes():
    """'distance' and 'uniform' are what the existing files pass to sklearn."""
    assert nw.canonical('distance') == 'inverse'
    assert nw.canonical('uniform') == 'uniform'
    assert nw.canonical('  Inverse_Square ') == 'inverse_square'


@pytest.mark.parametrize('scheme', nw.WEIGHTING_SCHEMES)
def test_sklearn_accepts_every_scheme_in_a_fitted_knn(scheme):
    """Each scheme survives a real fit and predict, duplicated rows included.

    The first two training molecules carry identical features, so the query row
    that copies them puts a zero into the distance array sklearn hands the
    callable.
    """
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.RandomState(42)
    X = rng.rand(20, 4)
    X[1] = X[0]
    y = rng.rand(20)
    m = KNeighborsRegressor(n_neighbors=5,
                            **nw.sklearn_kwargs(scheme)).fit(X, y)
    pred = m.predict(np.vstack([X[0], rng.rand(3, 4)]))
    assert np.all(np.isfinite(pred))


def test_the_environment_variable_sets_the_scheme_for_the_process():
    """Same shape as distance_config's KIRBY_DISTANCE_METRIC."""
    before = nw.get_weighting_scheme()
    try:
        nw.set_weighting_scheme('gaussian')
        assert nw.resolve() == 'gaussian'
        assert nw.resolve('rank') == 'rank'
    finally:
        nw.set_weighting_scheme(before)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
