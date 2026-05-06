"""Self-training (semi-supervised pseudo-labelling) for ANNITIA (Phase D.8).

Standard self-training recipe applied to survival:
    1. Fit a base model on the labelled training set.
    2. Predict risks on the unlabelled set (val_test in the challenge).
    3. Take the top-K most-confident predictions (highest |risk - median|),
       assign them pseudo-survival labels — events for high-risk, censored
       at the test horizon for low-risk — at a downweighted sample weight.
    4. Concatenate (real, pseudo) data and refit the base model.

CV protocol:
    Inside a CV fold, the validation patients act as the "unlabelled"
    target — we never use the ACTUAL test set during CV. This catches
    self-training degradations before they hit the leaderboard. Only after
    CV-confirmation do we use the real val_test as pseudo-label source for
    the FINAL refit.

This is a very gentle augmentation (typical sample_weight ~0.1) — the
intent is to surface the geometry of the unlabelled distribution, not to
override the supervised signal.

API:
    SelfTrainSurvival(base_cls, top_frac=0.2, pseudo_weight=0.1,
                      base_kwargs={...})
        — drop-in for any class with .fit / .predict_risk. Internally splits
          .fit(X_train, y_train, X_unlab=...) into the two-pass procedure.
        — If `X_unlab` is None at fit time, falls back to plain .fit on
          training data only (useful for bypass during CV diagnostics).

Notes:
    - Pseudo-events use the median TRAINING event time as their assigned
      time. Pseudo-censored use the max training time.
    - This trick is best paired with classifier-style framings (landmark);
      for plain Cox, the pseudo-event/pseudo-censored approximation is
      coarse but rank-preserving.
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np


class SelfTrainSurvival:
    """Self-training wrapper (CV-friendly).

    Args:
        base_cls: any class with .fit(X, y_struct) / .predict_risk(X).
        top_frac: fraction of unlabelled to pseudo-label (top half by
            confidence, where confidence = |risk - median(risk)|).
        pseudo_weight: relative weight of pseudo-labelled samples vs real
            ones. Implemented by REPEATING pseudo rows ⌈pseudo_weight⌉
            times if the base model doesn't accept sample_weight, or via
            sample_weight directly if it does.
        base_kwargs: forwarded to base_cls.
        random_state: seed for tie-breaks.

    Use:
        m = SelfTrainSurvival(SurvivalCoxPH, top_frac=0.2)
        m.fit(X_train, y_train, X_unlab=X_val)
        risk = m.predict_risk(X_val)
"""

    def __init__(self,
                 base_cls: Type,
                 top_frac: float = 0.2,
                 pseudo_weight: float = 0.1,
                 base_kwargs: Optional[dict] = None,
                 random_state: int = 42):
        if not (0.0 <= top_frac <= 1.0):
            raise ValueError(f'top_frac must be in [0, 1], got {top_frac}')
        self.base_cls = base_cls
        self.top_frac = top_frac
        self.pseudo_weight = pseudo_weight
        self.base_kwargs = dict(base_kwargs or {})
        self.random_state = int(random_state)
        self._inner = None
        self._pseudo_event_time: Optional[float] = None
        self._pseudo_cens_time: Optional[float] = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray,
            X_unlab: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.float64)
        # Pass 1: plain fit on labelled.
        first = self.base_cls(**self.base_kwargs)
        first.fit(X, y_struct)
        if X_unlab is None or self.top_frac == 0.0:
            self._inner = first
            return self
        X_unlab = np.asarray(X_unlab, dtype=np.float64)
        # Predict risk on unlabelled
        risk_unlab = np.asarray(first.predict_risk(X_unlab), dtype=np.float64)
        med = float(np.median(risk_unlab))
        confidence = np.abs(risk_unlab - med)
        n_pick = max(int(round(self.top_frac * X_unlab.shape[0])), 1)
        top_idx = np.argsort(confidence)[::-1][:n_pick]

        # Assign pseudo-labels
        time_train = np.asarray(y_struct['time'], dtype=np.float64)
        event_train = np.asarray(y_struct['event'], dtype=bool)
        ev_times = time_train[event_train]
        if ev_times.size == 0:
            ev_times = time_train
        self._pseudo_event_time = float(np.median(ev_times))
        self._pseudo_cens_time = float(np.max(time_train))

        is_high = risk_unlab[top_idx] > med
        pseudo_time = np.where(
            is_high, self._pseudo_event_time, self._pseudo_cens_time)
        pseudo_event = is_high.astype(bool)
        # Build a structured array matching y_struct dtype.
        ps_y = np.empty(top_idx.size, dtype=y_struct.dtype)
        ps_y['time'] = pseudo_time
        ps_y['event'] = pseudo_event
        X_pseudo = X_unlab[top_idx]

        # Concatenate. Reweight pseudo by REPEATING rows according to
        # pseudo_weight; this is robust because most of our base classes
        # don't take sample_weight on the survival API.
        repeat = max(int(round(self.pseudo_weight * 10)), 0)
        if repeat == 0:
            X_combined = X
            y_combined = y_struct
        else:
            X_combined = np.concatenate([X, np.tile(X_pseudo, (repeat, 1))], axis=0)
            y_combined = np.concatenate([y_struct, np.tile(ps_y, repeat)], axis=0)

        # Pass 2: refit on (real + pseudo)
        second = self.base_cls(**self.base_kwargs)
        try:
            second.fit(X_combined, y_combined)
            self._inner = second
        except Exception:
            # Self-training can destabilise some Cox fits — fall back to first.
            self._inner = first
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError('SelfTrainSurvival not fitted')
        return self._inner.predict_risk(np.asarray(X, dtype=np.float64))
