"""Landmark binary target framing (Phase C.2).

For each landmark horizon τ, recast survival as binary classification:

    label_i(τ) = 1   if patient i had the event by time τ          (event & time ≤ τ)
              = 0   if patient i was event-free past τ            (time > τ)
              = excluded   if censored before τ                    (drop or IPCW-weight)

Two ways to handle the "censored before τ" group:
    1. drop them entirely (`drop`) — simplest and unbiased under independent
       censoring.
    2. IPCW: keep them as soft negatives weighted by 1 / G(time_i),
       where G(t) is the KM estimator of the censoring distribution. Used by
       most landmark survival papers; preserves sample size.

We support both via `mode='drop' | 'ipcw'`.

Per-horizon, we train a classifier (LightGBM, CatBoost, MLP, TabPFN, ...)
on the binary target with optional IPCW sample weights, and predict the
event probability at τ. Risk = average rank across all horizons (aggregating
over time without coupling magnitudes from different horizons).

API mirrors SURVIVAL_MODELS:
    fit(X, y_struct)
    predict_risk(X) → ndarray (higher = worse)

Plugs into the classifiers from `survival.py` (B.4, B.6) and the standalone
classifier wrappers in `tabpfn.py` / `tabicl.py` (B.5, B.11). Pass any class
with .fit(X, y, sample_weight=None) and either .predict_proba(X)[:, 1] or
a .predict_risk(X) method.

Typical hepatic horizons for the ANNITIA cohort: [3, 5, 7, 10] years (the
challenge was scored on long-horizon hepatic and death survival; using
multiple horizons gives the model multiple lookouts and the rank-aggregator
makes it scale-free).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Type

import numpy as np

from experiments.annitia.pseudo_obs import _km_survival_at


def _ipcw_weights(time: np.ndarray, event: np.ndarray,
                  horizon: float) -> np.ndarray:
    """Inverse-probability-of-censoring weights at a horizon.

    G(t) = KM survival of the censoring time (event flag flipped). For a
    patient with event by τ, weight = 1 / G(time_i). For a patient
    event-free past τ, weight = 1 / G(τ). Both bounded below by ε.
    """
    g_full_at = _km_survival_at
    cens_event = ~event.astype(bool)
    # G(time_i) for event-by-τ patients
    n = time.size
    weights = np.ones(n, dtype=np.float64)
    eps = 1e-3
    by_horizon_mask = (time <= horizon) & event
    past_horizon_mask = time > horizon
    g_at_t = g_full_at(time[by_horizon_mask], time, cens_event)
    weights[by_horizon_mask] = 1.0 / np.maximum(g_at_t, eps)
    g_at_tau = g_full_at(np.asarray([horizon]), time, cens_event)[0]
    weights[past_horizon_mask] = 1.0 / max(g_at_tau, eps)
    # Censored before horizon → label undefined; we let caller drop.
    return weights


def _binary_targets_at(time: np.ndarray, event: np.ndarray,
                       horizon: float, mode: str = 'drop'):
    """Return (X_keep_mask, y, sample_weight, ...) for one landmark.

    mode='drop':
        keep = (event & time<=τ) | (time>τ)    — censored-before-τ dropped
        y    = 1 / 0
        w    = 1.0 (all)
    mode='ipcw':
        keep = (event & time<=τ) | (time>τ)    — same masking, but...
        y    = 1 / 0
        w    = 1/G(time_i) for cases, 1/G(τ) for controls (and the dropped
               are still dropped — IPCW reweights what's KEPT to be
               population-representative).
    """
    by_horizon = (time <= horizon) & event
    past_horizon = time > horizon
    keep = by_horizon | past_horizon
    y = np.zeros(time.size, dtype=int)
    y[by_horizon] = 1
    if mode == 'drop':
        w = np.ones(time.size, dtype=np.float64)
    elif mode == 'ipcw':
        w = _ipcw_weights(time, event, horizon)
    else:
        raise ValueError(f'mode must be drop|ipcw, got {mode}')
    return keep, y, w


def _predict_proba_compatible(model, X: np.ndarray) -> np.ndarray:
    """Get class-1 probability from a fitted classifier. Tolerant to wrappers
    that expose `predict_proba` (sklearn-style) OR `predict_risk` (our
    survival convention used by TabPFN landmark / TabICL landmark)."""
    X = np.asarray(X, dtype=np.float64)
    if hasattr(model, 'predict_proba'):
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, 'predict_risk'):
        return np.asarray(model.predict_risk(X), dtype=np.float64)
    if hasattr(model, 'predict'):
        return np.asarray(model.predict(X), dtype=np.float64)
    raise RuntimeError(
        f'{type(model).__name__} has no predict_proba / predict_risk / predict')


def _rank_average(prob_list: List[np.ndarray]) -> np.ndarray:
    """Mean of within-vector argsort ranks. Scale-free across horizons."""
    n = prob_list[0].size
    ranks_sum = np.zeros(n, dtype=np.float64)
    for p in prob_list:
        order = np.argsort(p)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(n)
        ranks_sum += r
    return ranks_sum / len(prob_list)


class LandmarkBinarySurvival:
    """Landmark binary classification framing.

    Args:
        classifier_cls: class with .fit(X, y, sample_weight=None) +
            .predict_proba(X) (or .predict_risk(X)). Defaults to
            ClassifierCatBoost.
        horizons: list of landmark times. Defaults to median+upper-median
            event times.
        horizon_quantiles: quantiles of event times if `horizons` not given.
        mode: 'drop' or 'ipcw'.
        classifier_kwargs: forwarded to classifier_cls.
        aggregator: 'rank' (default — mean rank across horizons) or 'mean'
            (mean of probabilities, scale-sensitive).
    """

    def __init__(self,
                 classifier_cls: Optional[Type] = None,
                 horizons: Optional[Sequence[float]] = None,
                 horizon_quantiles: Sequence[float] = (0.5, 0.75),
                 mode: str = 'drop',
                 classifier_kwargs: Optional[dict] = None,
                 aggregator: str = 'rank'):
        self.classifier_cls = classifier_cls
        self.horizons = list(horizons) if horizons is not None else None
        self.horizon_quantiles = list(horizon_quantiles)
        if mode not in ('drop', 'ipcw'):
            raise ValueError(f'mode must be drop|ipcw, got {mode}')
        self.mode = mode
        self.classifier_kwargs = dict(classifier_kwargs or {})
        if aggregator not in ('rank', 'mean'):
            raise ValueError(f'aggregator must be rank|mean, got {aggregator}')
        self.aggregator = aggregator
        self._models: List = []
        self._horizons_used: Optional[np.ndarray] = None

    def _resolve_classifier_cls(self):
        if self.classifier_cls is not None:
            return self.classifier_cls
        from experiments.annitia.survival import ClassifierCatBoost
        return ClassifierCatBoost

    def _resolve_horizons(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        if self.horizons is not None:
            return np.asarray(self.horizons, dtype=np.float64)
        ev_times = time[event]
        if ev_times.size == 0:
            ev_times = time
        return np.quantile(ev_times, self.horizon_quantiles).astype(np.float64)

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        keep_finite = np.isfinite(time)
        Xk = X[keep_finite]
        time_k = time[keep_finite]
        event_k = event[keep_finite]

        self._horizons_used = self._resolve_horizons(time_k, event_k)
        cls = self._resolve_classifier_cls()
        self._models = []
        for tau in self._horizons_used:
            keep, y, w = _binary_targets_at(time_k, event_k, float(tau), self.mode)
            if keep.sum() < 2 or y[keep].sum() == 0 or y[keep].sum() == keep.sum():
                # Degenerate fold for this horizon — skip it.
                self._models.append(None)
                continue
            mdl = cls(**self.classifier_kwargs)
            try:
                mdl.fit(Xk[keep], y[keep], sample_weight=w[keep])
            except TypeError:
                # classifier doesn't accept sample_weight
                mdl.fit(Xk[keep], y[keep])
            self._models.append(mdl)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError('LandmarkBinarySurvival not fitted')
        X = np.asarray(X, dtype=np.float64)
        probs = []
        for mdl in self._models:
            if mdl is None:
                continue
            probs.append(_predict_proba_compatible(mdl, X))
        if not probs:
            return np.zeros(X.shape[0], dtype=np.float64)
        if self.aggregator == 'rank':
            return _rank_average(probs)
        return np.mean(np.stack(probs, axis=0), axis=0)
