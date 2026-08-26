"""_nanrank -- non-finite-safe ranking and averaging for the read-across rho.

simeval.eval_knn_readacross returns NaN when the Spearman is undefined (constant
kNN predictions, or fewer than two finite predictions). NaN is not coerced to
0.0 there, so an undefined correlation stays distinguishable from a measured
zero correlation.

Two consequences the callers in this directory have to handle:

  * `max([nan, 1.0, 2.0])` is nan, and `sorted(v, key=lambda x: -x)` leaves nan
    wherever it started. A non-finite value therefore never loses a ranking.
  * `np.mean` over a list holding one NaN is NaN, so one undefined subset makes
    every aggregate NaN.

`rank_desc` and `nanmean` below are what those call sites use instead. Neither
converts a non-finite value to a number that could be read as a measurement:
rank_desc's -inf exists only inside the sort key, and nanmean returns NaN when
every input is non-finite.
"""
import warnings

import numpy as np

NEG_INF = float('-inf')


def is_defined(v):
    """True when v is a finite number -- i.e. a correlation that was computed.
    False for NaN and for +/-inf."""
    try:
        return bool(np.isfinite(v))
    except (TypeError, ValueError):
        return False


def rank_desc(v):
    """Sort key for higher-is-better values, placing non-finite values last.

    Use as `sorted(rows, key=lambda r: rank_desc(r['x']), reverse=True)` or
    `max(rows, key=lambda r: rank_desc(r['x']))`. reverse=True is stable, so
    the non-finite entries keep their original relative order at the end.

    The returned -inf is a sort position, never a stored quantity."""
    return float(v) if is_defined(v) else NEG_INF


def nanmean(values):
    """Mean over the finite entries of `values`. NaN when none are finite.

    np.nanmean of an all-NaN input warns and returns NaN; the warning is
    silenced here, the NaN return is kept."""
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).any():
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(np.nanmean(a[np.isfinite(a)]))


def n_defined(values):
    """How many of `values` are finite -- the sample size behind a nanmean."""
    return int(sum(1 for v in values if is_defined(v)))
