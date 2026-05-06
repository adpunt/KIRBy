"""Andersen pseudo-observations target framing (Phase C.1).

Idea: replace the (event, time) target with a continuous "pseudo-survival"
value at a chosen landmark τ, then train any plain regressor on it. The
pseudo-value for patient i at horizon τ is

    P_i(τ) = n * S_full(τ) - (n-1) * S_{-i}(τ)

where S_full is the Kaplan-Meier estimator on all patients and S_{-i} is the
KM estimator with patient i removed (jackknife). Pseudo-values are unbiased
under standard KM assumptions and act as a continuous regression target for
event-time prediction.

Why this is useful for ANNITIA:
    - Lets us use any tabular regressor (LightGBM, CatBoost, MLP) on a
      well-defined real-valued target, even with heavy censoring.
    - Higher pseudo-value = higher predicted survival probability at τ →
      LOWER risk; we negate at predict time to keep risk semantics.
    - Pseudo-values are computed once on the training-fold data and reused
      across model evaluations on that fold.

Notes on horizon choice:
    Default uses the median observed time among events (cause-specific).
    Caller can override via `horizon`. Multiple horizons can be averaged
    (set `horizons=[t1, t2, ...]`); per-patient targets are then the mean
    of the per-horizon pseudo-values.

API mirrors the SURVIVAL_MODELS contract:
    fit(X, y_struct) → self
    predict_risk(X) → np.ndarray (higher = worse)

Plugs into the regressors from `survival.py` (B.3-B.6): pass any class with
.fit(X, y) / .predict(X) (e.g. RegressorLightGBM, RegressorCatBoost,
MLPRegressorSurvival).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Type

import numpy as np


def _km_survival_at(t_eval: np.ndarray, time: np.ndarray,
                    event: np.ndarray) -> np.ndarray:
    """Plain Kaplan-Meier S(t) evaluated at the points in t_eval.

    Returns S(t) ∈ [0, 1], step-decreasing with each observed event. Right-
    continuous: S(t) = S(t-) until the next event time. Censored times do
    not change S; they only shrink the risk set going forward.

    Implementation: O((n + |t_eval|) log n).
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    if time.size == 0:
        return np.ones_like(t_eval, dtype=np.float64)
    # Sort by time
    order = np.argsort(time, kind='mergesort')
    t_sorted = time[order]
    e_sorted = event[order]
    n = t_sorted.size

    # At each time point we need the number at risk (n - i) and the number
    # of events at that exact time.
    surv = 1.0
    # Build step function as (t_event, S_after_event)
    step_t = []
    step_s = []
    i = 0
    while i < n:
        j = i
        # Group ties by time
        while j + 1 < n and t_sorted[j + 1] == t_sorted[i]:
            j += 1
        d = int(e_sorted[i:j + 1].sum())  # events at this time
        n_risk = n - i  # n - (number of patients with time < t_i)
        if d > 0 and n_risk > 0:
            surv *= (1.0 - d / n_risk)
        step_t.append(t_sorted[i])
        step_s.append(surv)
        i = j + 1
    step_t = np.asarray(step_t, dtype=np.float64)
    step_s = np.asarray(step_s, dtype=np.float64)

    # Evaluate. S(t) for t < first event = 1, S(t) for t >= last event time
    # = last computed surv. For t_eval values, find the largest step_t <= t.
    out = np.ones_like(t_eval, dtype=np.float64)
    idx = np.searchsorted(step_t, t_eval, side='right') - 1
    has_pred = idx >= 0
    out[has_pred] = step_s[idx[has_pred]]
    return out


def compute_pseudo_observations(time: np.ndarray, event: np.ndarray,
                                horizons: Sequence[float]) -> np.ndarray:
    """Vectorised jackknife KM pseudo-observations.

    For each (patient i, horizon τ_h):
        P_i(τ_h) = n * S_full(τ_h) - (n - 1) * S_{-i}(τ_h)

    Returns a length-n array. If multiple horizons given, returns the
    per-patient mean across horizons (a single regression target).

    For the leave-one-out KM we exploit the fact that removing one patient
    changes only the death/risk-set counts at times ≤ that patient's time:
    the rest of the curve is identical to S_full. Computing per-horizon S
    only requires recomputing the curve up to τ.

    Complexity: O(n * |horizons|), per-patient KM evaluation is O(n).
    For n ~ 1250 and 1 horizon, that's ~10⁶ ops — fine.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    horizons = np.asarray(horizons, dtype=np.float64).ravel()
    n = time.size
    if n == 0:
        return np.array([], dtype=np.float64)

    s_full = _km_survival_at(horizons, time, event)
    pseudo = np.zeros((n, horizons.size), dtype=np.float64)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        s_loo = _km_survival_at(horizons, time[keep], event[keep])
        pseudo[i] = n * s_full - (n - 1) * s_loo
    if horizons.size == 1:
        return pseudo[:, 0]
    return pseudo.mean(axis=1)


class PseudoObservationsSurvival:
    """Pseudo-observation framing wrapper.

    Args:
        regressor_cls: any class with .fit(X, y) / .predict(X). Defaults to
            RegressorLightGBM if available.
        horizons: list of evaluation times. If None, uses the median event time.
        horizon_quantiles: alternative — quantiles of event times to use as
            horizons. Ignored if `horizons` is set.
        regressor_kwargs: forwarded to regressor_cls(**regressor_kwargs).

    Risk semantics: the regressor is trained to predict pseudo-survival
    (higher = better). predict_risk returns the NEGATIVE of the prediction
    so that higher-output = higher-risk, consistent with the rest of
    SURVIVAL_MODELS.
    """

    def __init__(self,
                 regressor_cls: Optional[Type] = None,
                 horizons: Optional[Sequence[float]] = None,
                 horizon_quantiles: Sequence[float] = (0.5,),
                 regressor_kwargs: Optional[dict] = None):
        self.regressor_cls = regressor_cls
        self.horizons = list(horizons) if horizons is not None else None
        self.horizon_quantiles = list(horizon_quantiles)
        self.regressor_kwargs = dict(regressor_kwargs or {})
        self._regressor = None
        self._horizons_used: Optional[np.ndarray] = None

    def _resolve_regressor_cls(self):
        if self.regressor_cls is not None:
            return self.regressor_cls
        # Lazy import to avoid pulling lightgbm at module load.
        from experiments.annitia.survival import RegressorLightGBM
        return RegressorLightGBM

    def _resolve_horizons(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        if self.horizons is not None:
            return np.asarray(self.horizons, dtype=np.float64)
        ev_times = time[event]
        if ev_times.size == 0:
            # No events at all — fall back to global quantiles of time.
            ev_times = time
        return np.quantile(ev_times, self.horizon_quantiles).astype(np.float64)

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        keep = np.isfinite(time)
        Xk = X[keep]
        time_k = time[keep]
        event_k = event[keep]

        self._horizons_used = self._resolve_horizons(time_k, event_k)
        targets = compute_pseudo_observations(time_k, event_k, self._horizons_used)

        cls = self._resolve_regressor_cls()
        self._regressor = cls(**self.regressor_kwargs)
        self._regressor.fit(Xk, targets)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self._regressor is None:
            raise RuntimeError('PseudoObservationsSurvival not fitted')
        pred = np.asarray(self._regressor.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)
        return -pred  # invert: higher pseudo-survival → lower risk
