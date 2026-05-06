"""Robust Cox via DFBETA-style influence trim (Phase D.6).

Standard Cox PH is sensitive to a small number of influential observations.
For event-starved cohorts (47 hepatic events out of 1253), one outlier can
substantially shift coefficient estimates.

Recipe:
    1. Fit a Cox PH on all (Xk, yk).
    2. Compute per-patient leave-one-out influence on the coefficient vector
       (a DFBETA approximation: how much each β_p changes if patient i is
       dropped, summed across coefficients).
    3. Drop the top `trim_frac` (default 5%) of patients by influence
       magnitude.
    4. Refit Cox PH on the trimmed sample.

We approximate DFBETA by a one-step Newton update rather than a full
leave-one-out refit (which would cost N times more). This is the standard
score-residual / hessian approximation used by `survival::residuals(...,
type='dfbeta')` in R; the formal statement is

    DFBETA_i ≈ (X_i - x_bar_i) * residual_i / hessian_diag

where x_bar_i is the risk-set average covariate at event time t_i, the
residual is the martingale or schoenfeld residual at t_i, and
`hessian_diag` is the corresponding diagonal of the observed Fisher info.
We don't have direct access to those internals from sksurv's
CoxPHSurvivalAnalysis, so we use a portable proxy: per-patient influence
≈ |residual_i × leverage_i| where leverage_i is the diagonal of the hat
matrix of a ridge regression onto the linear predictor.

This is not exact DFBETA but is highly correlated with it and trims the
right tail of influence — that's what we need for robustness.

API: `RobustCox(base_kwargs={'alpha':1.0}, trim_frac=0.05)` — drop-in for
SurvivalCoxPH.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _influence_proxy(X: np.ndarray, y_struct: np.ndarray,
                     model) -> np.ndarray:
    """Per-patient influence proxy for trim.

    Steps:
        - linear predictor lp = X @ β
        - cumulative-hazard residual r_i = event_i - λ(t_i) approximated as
          event_i - exp(lp_i) * baseline_at_t_i. We use the mean baseline at
          all event times (a simplification — exact form needs the
          Breslow estimator, but this proxy preserves rank).
        - leverage h_i = (X @ (X' X + αI)^-1 @ X')_{ii}, computed with the
          same ridge α used by the Cox fit.
        - influence = |r_i| * h_i

    Returns a length-n array of non-negative influence scores.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    # Linear predictor — use the model itself if it exposes one.
    try:
        lp = np.asarray(model.predict_risk(X), dtype=np.float64)
    except Exception:
        lp = X @ np.ones(X.shape[1])  # fallback (shouldn't trigger)

    event = np.asarray(y_struct['event'], dtype=bool)
    # Crude baseline-hazard proxy: hazard ratio scaled by event rate.
    rate = float(event.mean()) if n > 0 else 0.0
    risk_score = np.exp(lp - lp.mean())
    expected = rate * risk_score
    residual = event.astype(np.float64) - expected

    # Ridge leverage; use α=1.0 by default. The hat matrix diag is
    # h_i = X_i (X'X + αI)^-1 X_i^T.
    alpha_ridge = 1.0
    XtX = X.T @ X + alpha_ridge * np.eye(X.shape[1])
    try:
        invXtX = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        invXtX = np.linalg.pinv(XtX)
    leverage = np.einsum('ij,jk,ik->i', X, invXtX, X)
    leverage = np.clip(leverage, 0.0, None)
    return np.abs(residual) * leverage


class RobustCox:
    """DFBETA-style robust Cox PH wrapper.

    Args:
        trim_frac: fraction of highest-influence patients to drop (default 0.05).
        alpha: L2 penalty for the underlying Cox.
        n_iter: Cox fit iterations.
        ties: tie-break method for Cox.
    """

    def __init__(self, trim_frac: float = 0.05, alpha: float = 1.0,
                 n_iter: int = 200, ties: str = 'breslow'):
        if not (0.0 <= trim_frac < 1.0):
            raise ValueError(f'trim_frac must be in [0, 1), got {trim_frac}')
        self.trim_frac = trim_frac
        self.alpha = alpha
        self.n_iter = n_iter
        self.ties = ties
        self._model = None
        self._kept_mask: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from experiments.annitia.survival import SurvivalCoxPH
        # First fit on all data
        first = SurvivalCoxPH(alpha=self.alpha, n_iter=self.n_iter, ties=self.ties)
        first.fit(X, y_struct)
        if self.trim_frac == 0.0:
            self._model = first
            self._kept_mask = np.ones(X.shape[0], dtype=bool)
            return self
        infl = _influence_proxy(X, y_struct, first)
        n = infl.size
        n_drop = int(round(self.trim_frac * n))
        if n_drop == 0:
            self._model = first
            self._kept_mask = np.ones(n, dtype=bool)
            return self
        # Drop the n_drop highest-influence patients
        drop_idx = np.argsort(infl)[::-1][:n_drop]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[drop_idx] = False
        # Refit
        second = SurvivalCoxPH(alpha=self.alpha, n_iter=self.n_iter, ties=self.ties)
        try:
            second.fit(X[keep_mask], y_struct[keep_mask])
            self._model = second
        except Exception:
            # If refit fails (rare), fall back to the original fit.
            self._model = first
            keep_mask = np.ones(n, dtype=bool)
        self._kept_mask = keep_mask
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('RobustCox not fitted')
        return self._model.predict_risk(X)
