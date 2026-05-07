"""Generic discretised-classifier survival wrapper.

Wraps any classifier with a sklearn-style fit(X, y) / predict_proba(X)
interface (returning P(class=1)) into a survival predictor by:
    1. Time-discretising into K bins via SurvivalDiscretiser.
    2. Fitting one classifier per bin on the rows whose label at that bin
       is not NaN (i.e. patients still at risk at the bin's start AND for
       whom the bin's outcome is known).
    3. At predict time, getting per-bin event probabilities + collapsing
       to a single risk score via 1 - prod_k (1 - p_k).

Designed for tabular foundation models (MITRA, TabPFN-2.5, TabICL) where
the per-bin classifier call is in-context (no real training cost), but
also works for LightGBM / CatBoost / any sklearn classifier.

Usage in experiment_branches._run():
    DiscretisedSurvival(classifier_factory=lambda: MitraClassifier(),
                        K=10, strategy='equal_event')
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ._survival_discretiser import SurvivalDiscretiser


class DiscretisedSurvival:
    """Per-bin classifier reduction for survival.

    Parameters
    ----------
    classifier_factory : Callable[[], object]
        Returns a fresh classifier instance per bin. The classifier must
        implement fit(X, y) where y in {0, 1}, and either
        predict_proba(X) -> (n, 2) or predict(X) -> (n,) probabilities.
    K : int
        Number of time bins.
    strategy : str
        SurvivalDiscretiser strategy.
    min_per_bin : int
        Minimum count of usable rows in a bin before we skip it (the bin
        contributes 0 to risk if skipped).
    """

    def __init__(self, classifier_factory: Callable[[], object],
                 K: int = 10, strategy: str = 'equal_event',
                 min_per_bin: int = 8):
        self.classifier_factory = classifier_factory
        self.K = int(K)
        self.strategy = strategy
        self.min_per_bin = int(min_per_bin)
        self._disc: SurvivalDiscretiser | None = None
        self._models: list = []
        self._active_bins: list[int] = []
        self._d_in: int | None = None

    def fit(self, X, y_struct):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        self._disc = SurvivalDiscretiser(K=self.K, strategy=self.strategy).fit(time, event)
        Y = self._disc.labels(time, event)
        self._models = []
        self._active_bins = []
        self._d_in = X.shape[1]
        for k in range(self._disc.n_bins):
            mask = ~np.isnan(Y[:, k])
            yk = Y[mask, k].astype(np.int64)
            if mask.sum() < self.min_per_bin or yk.sum() == 0 or yk.sum() == mask.sum():
                self._models.append(None)  # degenerate bin
                continue
            try:
                clf = self.classifier_factory()
                clf.fit(X[mask], yk)
                self._models.append(clf)
                self._active_bins.append(k)
            except Exception as e:  # noqa: BLE001
                self._models.append(None)
                # Don't propagate per-bin failures — degrade gracefully.
                _ = e
        if not self._active_bins:
            raise RuntimeError('No bins had enough data to fit a classifier')
        return self

    def _predict_p(self, clf, X) -> np.ndarray:
        if hasattr(clf, 'predict_proba'):
            P = np.asarray(clf.predict_proba(X), dtype=np.float64)
            if P.ndim == 2 and P.shape[1] >= 2:
                return P[:, 1]
            return P.ravel()
        return np.asarray(clf.predict(X), dtype=np.float64).ravel()

    def predict_per_bin(self, X) -> np.ndarray:
        if self._disc is None:
            raise RuntimeError('not fit')
        X = np.asarray(X, dtype=np.float64)
        n, K = X.shape[0], self._disc.n_bins
        P = np.zeros((n, K), dtype=np.float64)
        for k, clf in enumerate(self._models):
            if clf is None:
                continue
            P[:, k] = np.clip(self._predict_p(clf, X), 0.0, 1.0)
        return P

    def predict_risk(self, X) -> np.ndarray:
        P = self.predict_per_bin(X)
        return self._disc.survival_to_risk(P)
