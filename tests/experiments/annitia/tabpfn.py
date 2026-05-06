"""TabPFN-v2 wrapper for the landmark-binary framing.

TabPFN-v2 is an in-context tabular classifier (ICL transformer): zero-shot
on small tabular tasks. License: research/non-commercial via Prior Labs.

Constraints:
    - Default cap: ≤100 features. We project / select to fit.
    - Numerical only — string columns must be encoded upstream.
    - Pre-impute NaN (TabPFN does NOT tolerate NaN).
    - n_train ≤ ~10k for v2 small.

For ANNITIA's landmark binary task (n≈1250) the constraints are not binding,
but the feature cap matters once `_fpca_view` + `_gp_smooth_view` are added
(they push the all-views concat well past 200).

Provides:
    - TabPFNClassifierWrapper.fit(X, y) / .predict_proba(X)
    - TabPFNSurvivalLandmark.fit(X, y_struct, horizon) / .predict_risk(X)
      — convenience wrapper for landmark-binary at a given horizon.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TabPFNClassifierWrapper:
    """Thin wrapper around `tabpfn.TabPFNClassifier`.

    Args:
        max_features: feature cap. If X has more, the wrapper applies a
            top-K variance selection (cheap and label-free) before fit.
            Set to None to disable (TabPFN will reject if >100).
        device: 'cpu' or 'cuda'. TabPFN-v2 runs on CPU; CUDA gives ~10x.
        ignore_pretraining_limits: pass-through to TabPFN. Set True if you
            want to use it on n>10k or d>100 with a degradation warning.
    """

    def __init__(self,
                 max_features: Optional[int] = 100,
                 device: str = 'cpu',
                 random_state: int = 42,
                 ignore_pretraining_limits: bool = False):
        self.max_features = max_features
        self.device = device
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self._model = None
        self._keep = None

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if self.max_features is None or X.shape[1] <= self.max_features:
            self._keep = np.arange(X.shape[1])
            return X
        # Top-K by variance (label-free, scale-invariant on standardised data).
        var = X.var(axis=0)
        self._keep = np.argsort(var)[::-1][:self.max_features]
        return X[:, self._keep]

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                'tabpfn not installed — `pip install tabpfn` '
                f'(missing on this host: {e})')
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self._select_features(X)
        kwargs = {'device': self.device, 'random_state': self.random_state}
        if self.ignore_pretraining_limits:
            kwargs['ignore_pretraining_limits'] = True
        try:
            self._model = TabPFNClassifier(**kwargs)
        except TypeError:
            # Older TabPFN: drop unsupported kwargs
            self._model = TabPFNClassifier(device=self.device)
        self._model.fit(Xs, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if self._keep is not None:
            X = X[:, self._keep]
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)


class TabPFNSurvivalLandmark:
    """Landmark-binary survival via TabPFN.

    At fit time:
        - keep only patients whose follow-up is unambiguous at `horizon`
          (event before horizon → 1; censored ≥ horizon → 0; censored
          before horizon is dropped — could be salvaged with IPCW later).
    At predict time: the predicted P(event by horizon) is the risk score.
    """

    def __init__(self, horizon: float, **kwargs):
        self.horizon = horizon
        self._cls = TabPFNClassifierWrapper(**kwargs)

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        time = y_struct['time']
        event = y_struct['event'].astype(bool)
        finite = np.isfinite(time)
        # Define landmark labels: 1 = event by horizon, 0 = censored beyond horizon
        positive = finite & event & (time <= self.horizon)
        negative = finite & ~event & (time >= self.horizon)
        keep = positive | negative
        if keep.sum() < 20 or positive.sum() < 5:
            raise ValueError(
                f'TabPFNSurvivalLandmark: too few labelled patients at '
                f'horizon={self.horizon}: keep={keep.sum()} pos={positive.sum()}')
        Xk = np.asarray(X)[keep]
        yk = positive[keep].astype(int)
        self._cls.fit(Xk, yk)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        proba = self._cls.predict_proba(X)
        return proba[:, 1].astype(np.float64)
