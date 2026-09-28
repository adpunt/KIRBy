"""One place that decides how much each neighbour's label counts.

WHY THIS EXISTS. Take a molecule, find its k nearest training molecules, average
their activities. Either every neighbour counts the same, or a close one counts
more than a far one. That choice is made at roughly twenty places across
similarity/, similarity_metrics/, noise/, complexity/ and the engine, and today
the three folders answer it differently: similarity_metrics/ counts close ones
more in 3 places, noise/ counts every neighbour the same in 3, similarity/ reads
an environment variable in s05 and s10 and lets s04 count every neighbour the
same. Nobody chose that; it is what each file happened to do.

This module is the companion of distance_config, which sits next to it and does
the same job for the other setting -- which coefficient decides what "near"
means. The two settings have never been crossed. F34 crosses them.

WHAT SKLEARN GIVES. KNeighborsRegressor and KNeighborsClassifier accept exactly
three things as `weights`: the string 'uniform', the string 'distance' (one over
the distance, with no choice of steepness), or a callable taking an array of
distances and returning an array of weights. There is no third built-in, so
anything between or beyond those two has to be a callable.

THE SIX SCHEMES. Four are one family, w = 1 / d ** p:

    name             p     what it is                     how it travels
    uniform          0     every neighbour equal          sklearn's 'uniform'
    inverse_sqrt     0.5   mild preference for close ones  callable
    inverse          1     one over the distance           sklearn's 'distance'
    inverse_square   2     steep preference                callable

p = 0 and p = 1 are handed to sklearn as its own two strings rather than
reimplemented, so the family is pinned at both ends to what the suite already
does and a sweep over the six subsumes today's behaviour rather than replacing
it.

Two more, because a power law is not the only shape:

    gaussian    w = exp(-(d/h) ** 2), h the median distance among that molecule's
                own k neighbours. Falls off smoothly rather than diverging.
    rank        w = 1 / rank. The order of the neighbours decides everything and
                how far away they are decides nothing.

ZERO DISTANCE. Two molecules with identical features are at distance zero and
one over zero is infinite. sklearn's own 'distance' handles that by giving the
exact matches all of the weight and the other neighbours none, and
`_finite_or_exact_match` below does the same for every callable here. It fires
only on a row whose weights are not all finite, so:

  * inverse_sqrt, inverse and inverse_square hit it whenever a neighbour is an
    exact match, which is what sklearn does;
  * gaussian hits it only when the median neighbour distance is itself zero,
    because exp(-(0/h) ** 2) is 1 and needs no rescue when h > 0;
  * rank never hits it, because it never divides by a distance. Routing it
    through the same helper keeps one code path without changing what it means.

Without that helper a fold containing one duplicated molecule silently produces
NaN and the fold is lost.

WHAT THIS CANNOT REACH. A KNN fitted as a MODEL inside the engine's
hyperparameter search tunes its own `weights` per fold through
_HP_SEARCH_SPACES['knn'] in src/kirby/find_best_hybrid.py. Forcing this module's
choice on it would overwrite the tuned one, the same way make_nearest_neighbors
in tests/experiments/_shared.py leaves those alone.
"""
import os
import threading

import numpy as np

# Canonical names, in increasing order of how much a close neighbour is
# preferred, then the two shapes outside the power-law family.
WEIGHTING_SCHEMES = ('uniform', 'inverse_sqrt', 'inverse', 'inverse_square',
                     'gaussian', 'rank')

# 'uniform' is what noise/ writes explicitly and what s04, s05 and s10 fall back
# to, so it is the default until F34 says otherwise.
DEFAULT_WEIGHTING_SCHEME = 'uniform'

ENV_VAR = 'KIRBY_NEIGHBOUR_WEIGHTING'

# The exponent p in w = 1 / d ** p, for the four schemes that are one family.
# uniform and inverse are here so the family is complete on paper even though
# both travel to sklearn as its own string.
FAMILY_EXPONENT = {
    'uniform': 0.0,
    'inverse_sqrt': 0.5,
    'inverse': 1.0,
    'inverse_square': 2.0,
}

# Spellings a caller might reasonably write, mapped to the canonical name.
_ALIASES = {
    'uniform': 'uniform',
    'equal': 'uniform',
    'p0': 'uniform',
    'inverse_sqrt': 'inverse_sqrt',
    'inverse_root': 'inverse_sqrt',
    'p0.5': 'inverse_sqrt',
    'inverse': 'inverse',
    'distance': 'inverse',
    'p1': 'inverse',
    'inverse_square': 'inverse_square',
    'inverse_squared': 'inverse_square',
    'p2': 'inverse_square',
    'gaussian': 'gaussian',
    'rbf': 'gaussian',
    'rank': 'rank',
    'reciprocal_rank': 'rank',
}

_lock = threading.Lock()
_current = None


def canonical(name):
    """Resolve a spelling to one of WEIGHTING_SCHEMES. Raises on anything else.

    Raising rather than falling back is the point, for the same reason
    distance_config.canonical raises: a silent fallback to counting every
    neighbour the same is exactly the failure this module exists to end.
    """
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"unknown neighbour weighting scheme {name!r}. The suite sweeps "
            f"exactly {list(WEIGHTING_SCHEMES)}. Add a new one here before "
            f"using it, and give it an entry in FAMILY_EXPONENT if it belongs "
            f"to the w = 1 / d ** p family.")
    return _ALIASES[key]


def get_weighting_scheme():
    """The scheme in force for this process."""
    global _current
    with _lock:
        if _current is None:
            _current = canonical(os.environ.get(ENV_VAR,
                                                DEFAULT_WEIGHTING_SCHEME))
        return _current


def set_weighting_scheme(name):
    """Set the scheme for this process and return it."""
    global _current
    resolved = canonical(name)
    with _lock:
        _current = resolved
    return resolved


def resolve(name=None):
    """`name` if given, otherwise whatever is in force."""
    return get_weighting_scheme() if name is None else canonical(name)


# ---------------------------------------------------------------------------
# The one zero-distance guard. Every callable below goes through it.
# ---------------------------------------------------------------------------
def _finite_or_exact_match(dist, w):
    """Replace any row of `w` that is not all finite with its exact-match mask.

    `dist` and `w` are both (n_queries, n_neighbours). A row of `w` holding an
    infinity or a NaN means the scheme divided by a zero distance somewhere in
    that row. sklearn's own 'distance' answers that by giving the zero-distance
    neighbours a weight of 1 and every other neighbour in that row a weight of
    0, so the exact matches decide the prediction on their own. This does the
    same, and leaves every row whose weights are already finite untouched.
    """
    dist = np.asarray(dist, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != dist.shape:
        raise ValueError(
            f'weight array shape {w.shape} does not match the distance array '
            f'shape {dist.shape}')
    bad_row = ~np.all(np.isfinite(w), axis=-1)
    if np.any(bad_row):
        w = w.copy()
        w[bad_row] = (dist[bad_row] == 0.0).astype(np.float64)
        # A row that is not finite and holds no zero distance has nothing to
        # fall back on. Counting its neighbours equally loses the scheme for
        # that one molecule but keeps the fold, where leaving a NaN would lose
        # every molecule in it.
        empty = ~np.any(w[bad_row] > 0, axis=-1)
        if np.any(empty):
            idx = np.flatnonzero(bad_row)[empty]
            w[idx] = 1.0
    return w


def _inverse_power(p):
    """w = 1 / d ** p, guarded. p = 0.5 and p = 2 are the two written here."""
    def weights(dist):
        d = np.asarray(dist, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / np.power(d, p)
        return _finite_or_exact_match(d, w)
    weights.__name__ = f'inverse_power_{p}'
    return weights


def _gaussian_weights(dist):
    """w = exp(-(d/h) ** 2), h the median distance among that row's neighbours.

    h is per molecule, not per dataset, so the width follows how crowded each
    molecule's own neighbourhood is. A molecule whose k neighbours are all far
    away is not punished for living in a sparse region.
    """
    d = np.asarray(dist, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.median(d, axis=-1, keepdims=True)
        w = np.exp(-np.square(d / h))
    return _finite_or_exact_match(d, w)


def _rank_weights(dist):
    """w = 1 / rank, counting the closest neighbour as rank 1.

    Ranks come from argsort rather than from the order sklearn hands over, so
    the scheme does not depend on whether the caller sorted. Ties take
    consecutive ranks; two neighbours at the same distance therefore get
    slightly different weights, which is the price of ignoring magnitude.
    """
    d = np.asarray(dist, dtype=np.float64)
    order = np.argsort(d, axis=-1, kind='stable')
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(np.arange(1, d.shape[-1] + 1), d.shape), axis=-1)
    w = 1.0 / ranks.astype(np.float64)
    return _finite_or_exact_match(d, w)


# canonical -> what sklearn's `weights=` argument is given.
_SKLEARN = {
    'uniform': 'uniform',
    'inverse_sqrt': _inverse_power(0.5),
    'inverse': 'distance',
    'inverse_square': _inverse_power(2.0),
    'gaussian': _gaussian_weights,
    'rank': _rank_weights,
}


def sklearn_weights(name=None):
    """What to pass as `weights=` to a sklearn KNeighbors estimator.

    Returns the string 'uniform' for uniform and the string 'distance' for
    inverse, because those are sklearn's own two and reimplementing them would
    make the sweep's p = 0 and p = 1 arms something other than what the suite
    runs today. The other four come back as callables.
    """
    return _SKLEARN[resolve(name)]


def sklearn_kwargs(name=None):
    """Ready to splat into a sklearn KNeighbors estimator's constructor."""
    return {'weights': sklearn_weights(name)}


def weight_array(dist, name=None):
    """The weights `name` gives these distances, as an array.

    For the two sklearn strings this computes what sklearn would compute, so a
    caller outside sklearn -- a hand-rolled prediction, a test -- gets the same
    numbers as a caller inside it. `dist` is (n_queries, n_neighbours).
    """
    resolved = resolve(name)
    w = _SKLEARN[resolved]
    if w == 'uniform':
        return np.ones_like(np.asarray(dist, dtype=np.float64))
    if w == 'distance':
        return _inverse_power(1.0)(dist)
    return w(dist)
