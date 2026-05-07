"""TabICL wrapper for the landmark-binary framing.

TabICL (Tabular In-Context Learner, Soklaski et al. 2024) is a transformer
that does few-shot tabular classification by attending across train+test
in a single pass. Open-weights; install via `pip install tabicl`.

Constraints:
    - Numerical features only (or pre-encoded).
    - n_train ≤ ~10k recommended.
    - No NaN.

Provides:
    - TabICLClassifierWrapper.fit / .predict_proba — same interface as
      TabPFNClassifierWrapper so they can be drop-in for landmark.
    - TabICLSurvivalLandmark — convenience landmark-binary wrapper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _import_upstream_tabicl_classifier():
    """Resolve the pip-installed `tabicl.TabICLClassifier`, bypassing the
    local `tests/experiments/annitia/tabicl.py` shadowing on sys.path."""
    import importlib
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    saved_path = sys.path[:]
    saved_mod = sys.modules.pop('tabicl', None)
    try:
        sys.path = [p for p in sys.path
                    if os.path.abspath(p) != here]
        mod = importlib.import_module('tabicl')
        if not hasattr(mod, 'TabICLClassifier'):
            raise ImportError(
                f'tabicl package found at {getattr(mod, "__file__", "?")} '
                f'has no TabICLClassifier — wrong package?')
        return mod.TabICLClassifier
    except ImportError as e:
        raise ImportError(
            f'tabicl not installed — `pip install tabicl` '
            f'(missing on this host: {e})')
    finally:
        sys.path = saved_path
        if saved_mod is not None:
            sys.modules['tabicl'] = saved_mod


class TabICLClassifierWrapper:
    """Thin wrapper around `tabicl.TabICLClassifier` (or the version-specific
    factory). Falls back to a clear ImportError if the package is missing."""

    def __init__(self,
                 max_features: Optional[int] = None,
                 device: str = 'cpu',
                 random_state: int = 42):
        self.max_features = max_features
        self.device = device
        self.random_state = random_state
        self._model = None
        self._keep = None

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if self.max_features is None or X.shape[1] <= self.max_features:
            self._keep = np.arange(X.shape[1])
            return X
        var = X.var(axis=0)
        self._keep = np.argsort(var)[::-1][:self.max_features]
        return X[:, self._keep]

    def fit(self, X: np.ndarray, y: np.ndarray):
        TabICLClassifier = _import_upstream_tabicl_classifier()
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self._select_features(X)
        # TabICLClassifier kwargs vary slightly across versions.
        try:
            self._model = TabICLClassifier(device=self.device,
                                           random_state=self.random_state)
        except TypeError:
            self._model = TabICLClassifier(device=self.device)
        self._model.fit(Xs, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if self._keep is not None:
            X = X[:, self._keep]
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)


class TabICLSurvivalLandmark:
    """Landmark-binary survival via TabICL. Same conventions as TabPFN's."""

    def __init__(self, horizon: float, **kwargs):
        self.horizon = horizon
        self._cls = TabICLClassifierWrapper(**kwargs)

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        time = y_struct['time']
        event = y_struct['event'].astype(bool)
        finite = np.isfinite(time)
        positive = finite & event & (time <= self.horizon)
        negative = finite & ~event & (time >= self.horizon)
        keep = positive | negative
        if keep.sum() < 20 or positive.sum() < 5:
            raise ValueError(
                f'TabICLSurvivalLandmark: too few labelled patients at '
                f'horizon={self.horizon}: keep={keep.sum()} pos={positive.sum()}')
        Xk = np.asarray(X)[keep]
        yk = positive[keep].astype(int)
        self._cls.fit(Xk, yk)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        proba = self._cls.predict_proba(X)
        return proba[:, 1].astype(np.float64)
