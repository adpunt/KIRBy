"""One place that decides which pairwise distance the experimental suite uses.

WHY THIS EXISTS. The distance between two records is computed at roughly sixty
places across complexity/, similarity/, noise/, opig_* and federated/. Every one
of them was Euclidean, and none of them said so on purpose -- Euclidean is what
scipy's cdist and sklearn's NearestNeighbors do when nobody passes a metric.
f18 measured the choice on herg_ki and logd and Euclidean came last in 13 of 14
representation blocks at predicting a record's activity from its five nearest
neighbours.

WHAT IT DOES. Holds one process-wide name. `--distance-metric` on any experiment
sets it, `KIRBY_DISTANCE_METRIC` in the environment sets it, and the three places
that actually compute a distance read it when their caller passed nothing:

    kirby.utils.noise_mitigation_methods.compute_distance_matrix
    kirby.utils.noise_estimation._knn_indices (via each estimator's `metric`)
    tests.experiments._shared.make_nearest_neighbors

The default is manhattan, not Euclidean. In f18 manhattan is the only one of the
four names below that is never last: mean normalised rank 0.64 / 0.57 / 0.55
across the three scores f18 measured, against Euclidean's 0.60 / 0.98 / 0.51.

THE FOUR NAMES are the same four the KNN hyperparameter grid tunes over
(`_HP_SEARCH_SPACES['knn']` in src/kirby/find_best_hybrid.py), so a KNN result
and a suite result can be read side by side.

WHAT THIS CANNOT REACH. Three kinds of distance in the suite are fixed by what
they are, not by a metric argument, and this module deliberately leaves them
alone:

  * k-means (s13, federated silo construction). Its objective is the sum of
    squared Euclidean distances to a centroid. There is no metric argument;
    another distance is a different algorithm.
  * RBF kernels (s10's GP and SVR arms, s11, c15's kernel analysis). exp(-gamma *
    squared Euclidean) by definition.
  * The representation-independent activity-cliff references: ECFP4 Tanimoto for
    molecules, sequence edit distance for antibodies, Butina Tanimoto for
    scaffold groups. Those are fixed on purpose so no representation is graded
    against cliffs derived from its own embedding.
"""
import os
import threading

# Canonical names. Same four as the KNN grid.
DISTANCE_METRICS = ('manhattan', 'euclidean', 'cosine', 'minkowski_p0.5')

DEFAULT_DISTANCE_METRIC = 'manhattan'

ENV_VAR = 'KIRBY_DISTANCE_METRIC'

# Spellings a caller might reasonably write, mapped to the canonical name.
_ALIASES = {
    'manhattan': 'manhattan',
    'cityblock': 'manhattan',
    'l1': 'manhattan',
    'euclidean': 'euclidean',
    'l2': 'euclidean',
    'cosine': 'cosine',
    'minkowski_p0.5': 'minkowski_p0.5',
    'minkowski_p05': 'minkowski_p0.5',
    'minkowski0.5': 'minkowski_p0.5',
    'fractional': 'minkowski_p0.5',
}

# canonical -> (scipy cdist name, extra scipy kwargs)
_SCIPY = {
    'manhattan': ('cityblock', {}),
    'euclidean': ('euclidean', {}),
    'cosine': ('cosine', {}),
    'minkowski_p0.5': ('minkowski', {'p': 0.5}),
}

# canonical -> (sklearn metric name, extra sklearn kwargs)
_SKLEARN = {
    'manhattan': ('manhattan', {}),
    'euclidean': ('euclidean', {}),
    'cosine': ('cosine', {}),
    'minkowski_p0.5': ('minkowski', {'p': 0.5}),
}

_lock = threading.Lock()
_current = None


def canonical(name):
    """Resolve a spelling to one of DISTANCE_METRICS. Raises on anything else.

    Raising rather than falling back is the point: a silent fallback to
    Euclidean is exactly the failure this module exists to end.
    """
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"unknown distance metric {name!r}. The suite sweeps exactly "
            f"{list(DISTANCE_METRICS)}. Add a new one here and to "
            f"COEFFICIENT_FAMILY in kirby.similarity.preprocessing before "
            f"using it, because the block divisor and the weight fold-in "
            f"exponent both depend on which family it belongs to.")
    return _ALIASES[key]


def get_distance_metric():
    """The name in force for this process."""
    global _current
    with _lock:
        if _current is None:
            _current = canonical(os.environ.get(ENV_VAR,
                                                DEFAULT_DISTANCE_METRIC))
        return _current


def set_distance_metric(name):
    """Set the name for this process and return it. Called by get_config()."""
    global _current
    resolved = canonical(name)
    with _lock:
        _current = resolved
    return resolved


def resolve(name=None):
    """`name` if given, otherwise whatever is in force."""
    return get_distance_metric() if name is None else canonical(name)


def scipy_metric(name=None):
    """(cdist metric string, kwargs) for scipy.spatial.distance.cdist/pdist.

    The kwargs carry p for the fractional case. scipy's cdist takes p=2 when
    nobody passes one, so metric='minkowski' alone returns the Euclidean number
    -- measured 3.319733 for both on the same input.
    """
    m, kw = _SCIPY[resolve(name)]
    return m, dict(kw)


def sklearn_metric(name=None):
    """(metric string, kwargs) for sklearn NearestNeighbors / KNeighbors*.

    sklearn ignores `p` unless metric='minkowski', which is why the fractional
    case has to travel as a pair rather than as one string.
    """
    m, kw = _SKLEARN[resolve(name)]
    return m, dict(kw)


def sklearn_kwargs(name=None):
    """Ready to splat into a sklearn neighbour estimator's constructor."""
    m, kw = sklearn_metric(name)
    return {'metric': m, **kw}


def base_and_p(name=None):
    """(name compute_distance_matrix understands, p or None).

    compute_distance_matrix speaks scipy's family names plus 'manhattan', and
    takes p separately, so 'minkowski_p0.5' has to be split before it is passed.
    """
    resolved = resolve(name)
    if resolved == 'minkowski_p0.5':
        return 'minkowski', 0.5
    return resolved, None


def block_divisor_for(width, name=None):
    """How much to divide one z-scored block by so its width stops being its weight.

    sqrt(width) under Euclidean, width under manhattan, width ** (1/p) under
    fractional Minkowski, 1.0 under cosine. Getting this wrong is silent: a
    2560-feature protein embedding stacked next to a 200-feature descriptor set
    supplies 2560 of the terms in a squared Euclidean distance and decides the
    neighbours on its own.
    """
    from kirby.similarity.preprocessing import (
        block_divisor, minkowski_block_divisor)
    base, p = base_and_p(name)
    if base == 'minkowski':
        return minkowski_block_divisor(width, p)
    return block_divisor(width, base)


def is_true_metric(name=None):
    """False for minkowski_p0.5, which breaks the triangle inequality.

    sklearn falls back to brute force on it and computes every pair: measured 7
    to 50 seconds per representation over 5 folds in the KNN grid, against 0.1
    to 2 seconds for the tree-backed metrics.
    """
    return resolve(name) != 'minkowski_p0.5'
