"""Importance method registry for ANNITIA (Phase D.1 + D.3).

Wraps `kirby.hybrid.compute_feature_importance` (8 methods) into a survival-
aware API. Survival data doesn't natively give feature scores, so we derive
a scalar binary target — "event happened by horizon τ" with τ = median
event time, restricted to non-censored-before-τ patients — and compute
feature importance with respect to that target. This produces a per-feature
scalar score that ranks features by their contribution to the binary
"survives past τ" decision.

Why this is reasonable as a survival proxy:
    - Most features useful for predicting survival are also useful for
      predicting "event by τ"; the rank-correlation is strong.
    - The binary derivation is fast, deterministic, and works with all 8
      importance methods (which expect either a regression or classification
      target).
    - For tree-based survival models (RSF, GBSA), this gives the same
      information that `feature_importances_` would, plus access to TreeSHAP
      / KernelSHAP / DeepLIFT etc.

Disk-cached via `_cache.cached_call`. Cache key = (data hash, target hash,
method, horizon, kwargs).

Two public APIs:
    compute_survival_importance(X, y_struct, method) → scores
        — single method, optionally cached. Used by D.2 weighting.
    compute_survival_importance_ensemble(X, y_struct, methods) → scores
        — D.3, rank-average of multiple methods.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from experiments.annitia._cache import cached_call, hash_data


SUPPORTED_METHODS = (
    'treeshap',
    'xgboost_gain',
    'lightgbm_gain',
    'permutation',
    'kernelshap',
    # NN methods — only relevant when DeepHit / NN models are in play; they
    # require torch / captum and are slow. Listed for completeness.
    'deeplift',
    'integrated_gradients',
    'gradientshap',
)


def _binary_target_at_median(time: np.ndarray, event: np.ndarray):
    """Derive a (mask, y) pair: y=1 if event by τ, 0 if event-free past τ,
    drop censored-before-τ patients."""
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    finite = np.isfinite(time)
    ev_times = time[finite & event]
    if ev_times.size == 0:
        # No events — fall back to median of all finite times.
        tau = float(np.median(time[finite])) if finite.any() else 0.0
    else:
        tau = float(np.median(ev_times))
    by = (time <= tau) & event & finite
    past = (time > tau) & finite
    keep = by | past
    y = np.zeros(time.size, dtype=int)
    y[by] = 1
    return keep, y, tau


def _call_kirby_importance(X: np.ndarray, y: np.ndarray, method: str,
                           **kwargs) -> np.ndarray:
    """Thin wrapper around kirby.hybrid.compute_feature_importance for one
    rep."""
    from kirby.hybrid import compute_feature_importance
    base_reps = {'feat': np.asarray(X, dtype=np.float64)}
    scores_dict = compute_feature_importance(
        base_reps, np.asarray(y), method=method, **kwargs)
    return np.asarray(scores_dict['feat'], dtype=np.float64)


def compute_survival_importance(X: np.ndarray,
                                y_struct: np.ndarray,
                                method: str = 'treeshap',
                                cache: bool = True,
                                cache_version: int = 1,
                                **kwargs) -> np.ndarray:
    """Per-feature importance vector (length = X.shape[1]).

    Args:
        X: feature matrix.
        y_struct: sksurv structured array.
        method: one of SUPPORTED_METHODS.
        cache: if True, results are pickled to the disk cache under a key
            built from (data hash, method, kwargs).
        cache_version: bump to invalidate prior entries.
        **kwargs: forwarded to compute_feature_importance.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f'unknown importance method {method!r}; '
            f'supported: {", ".join(SUPPORTED_METHODS)}')
    X = np.asarray(X, dtype=np.float64)
    keep, y, tau = _binary_target_at_median(y_struct['time'], y_struct['event'])
    if keep.sum() < 4 or y[keep].sum() in (0, keep.sum()):
        # Degenerate (no events / no controls / too few samples) — return
        # uniform importance so weighting/selection is a no-op rather than
        # blowing up.
        return np.ones(X.shape[1], dtype=np.float64)

    Xk = X[keep]
    yk = y[keep]

    def _fn():
        return _call_kirby_importance(Xk, yk, method=method, **kwargs)

    if not cache:
        return _fn()
    key = {
        'kind': 'survival_importance',
        'method': method,
        'tau': float(round(tau, 6)),
        'data_hash': hash_data(Xk, yk),
        'kwargs': {k: kwargs[k] for k in sorted(kwargs)},
    }
    return cached_call(key=key, fn=_fn, version=cache_version)


def compute_survival_importance_ensemble(X: np.ndarray,
                                         y_struct: np.ndarray,
                                         methods: Sequence[str] = ('treeshap',
                                                                    'permutation',
                                                                    'lightgbm_gain'),
                                         cache: bool = True,
                                         **method_kwargs) -> np.ndarray:
    """Rank-average importance across multiple methods (D.3).

    For each method m: compute scores_m, replace with within-vector rank
    (higher importance → higher rank), then average across methods.

    Failures (e.g. missing optional dep, rare crashes) are dropped — we
    average over the methods that returned a usable score.
    """
    rank_sum = None
    n_used = 0
    for m in methods:
        kw = method_kwargs.get(m, {})
        try:
            s = compute_survival_importance(
                X, y_struct, method=m, cache=cache, **kw)
        except Exception:
            continue
        if not np.all(np.isfinite(s)):
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(s.size)
        if rank_sum is None:
            rank_sum = ranks
        else:
            rank_sum += ranks
        n_used += 1
    if n_used == 0:
        return np.ones(X.shape[1], dtype=np.float64)
    return rank_sum / n_used


def importance_top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores (largest first). Used by D.2 selection."""
    k = min(k, scores.size)
    return np.argsort(scores)[::-1][:k]
