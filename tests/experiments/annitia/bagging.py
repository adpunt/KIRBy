"""Event-stratified bootstrap bagging (Phase D.4).

For event-starved survival data (hepatic: 47 events out of 1253), naive
bootstrap may yield draws with ~30 events while a stratified version always
includes all events plus a bootstrap of the censored. This is the standard
trick for low-event survival ensembles (e.g. RSF effectively does this
internally with `bootstrap=True`; we surface it as an explicit Layer-4
trick that wraps any model class).

Strategy:
    Bootstrap b ∈ 1..B:
        - resample the EVENT patients with replacement, size = n_events
        - resample the CENSORED patients with replacement, size = n_censored
        - fit the base model
        - record the predict_risk vector on the held-out split

Predict-time risk = MEAN of predict_risk vectors across the B bagged models.

Why mean rather than rank-average:
    Inside a single (framing, algorithm) the predictors are on the same
    scale, so straight averaging is fine. Rank-average is reserved for
    cross-model ensembles where scales differ (Layer 5 fusion).

API: `BaggedSurvivalModel(base_cls, B=200, ...)` — drop-in for any class
with `.fit(X, y_struct)` / `.predict_risk(X)`.
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np


class BaggedSurvivalModel:
    """Event-stratified bootstrap bagging wrapper.

    Args:
        base_cls: any class with .fit(X, y_struct) / .predict_risk(X).
        B: number of bootstrap rounds.
        random_state: seed.
        base_kwargs: kwargs forwarded to base_cls.
        verbose: print per-round progress.
        skip_failed: if a bootstrap fails to fit, skip it and continue.
            Default True (a few failed rounds shouldn't kill the run).

    Notes:
        - At predict-time, returns the mean across all rounds that fit
          successfully. If ALL rounds fail, raises RuntimeError.
        - Random state per-round is deterministic given the constructor seed.
    """

    def __init__(self,
                 base_cls: Type,
                 B: int = 200,
                 random_state: int = 42,
                 base_kwargs: Optional[dict] = None,
                 verbose: bool = False,
                 skip_failed: bool = True):
        self.base_cls = base_cls
        self.B = int(B)
        self.random_state = int(random_state)
        self.base_kwargs = dict(base_kwargs or {})
        self.verbose = verbose
        self.skip_failed = skip_failed
        self._models = []

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        keep = np.isfinite(y_struct['time'])
        Xk = X[keep]
        yk = y_struct[keep]
        event = np.asarray(yk['event'], dtype=bool)
        ev_idx = np.where(event)[0]
        ce_idx = np.where(~event)[0]
        if ev_idx.size == 0:
            raise ValueError('BaggedSurvivalModel: zero events in training set')
        rng = np.random.default_rng(self.random_state)
        self._models = []
        n_failed = 0
        for b in range(self.B):
            ev_boot = rng.choice(ev_idx, size=ev_idx.size, replace=True)
            ce_boot = (rng.choice(ce_idx, size=ce_idx.size, replace=True)
                       if ce_idx.size > 0 else np.array([], dtype=int))
            sample = np.concatenate([ev_boot, ce_boot])
            Xb = Xk[sample]
            yb = yk[sample]
            try:
                mdl = self.base_cls(**self.base_kwargs)
                mdl.fit(Xb, yb)
                self._models.append(mdl)
            except Exception as exc:
                n_failed += 1
                if not self.skip_failed:
                    raise
                if self.verbose:
                    print(f'  [bag {b}] FAIL {exc!r}')
            if self.verbose and ((b + 1) % max(self.B // 10, 1) == 0):
                print(f'  [bag {b + 1}/{self.B}] {len(self._models)} fit, '
                      f'{n_failed} failed')
        if not self._models:
            raise RuntimeError('All bootstrap rounds failed')
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError('BaggedSurvivalModel not fitted')
        X = np.asarray(X, dtype=np.float64)
        risks = []
        for mdl in self._models:
            try:
                risks.append(np.asarray(mdl.predict_risk(X), dtype=np.float64))
            except Exception:
                continue
        if not risks:
            raise RuntimeError('All bagged predict_risk calls failed')
        return np.mean(np.stack(risks, axis=0), axis=0)
