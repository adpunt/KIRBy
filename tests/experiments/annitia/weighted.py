"""Reliability-weighted survival models.

Survival adaptation of `tests/experiments/noise/n02_sample_weighting.py`'s
importance-weighted k-NN reliability scores. For each training sample we
compute the fraction of its k nearest neighbours (in importance-weighted
feature space) that share its event status. Reliable samples (clean labels,
homogeneous neighbourhood) get weight ~1; unreliable samples (mixed
neighbourhood, possibly noisy event labels) get weight floored at 0.1.

Importance scores are computed once per fit by a quick GBSA fit; this matches
n02's pattern of using feature importance from a fast tree model. Weights are
passed through to RSF / GBSA via sample_weight (Cox/Coxnet in sksurv 0.27 do
not accept sample_weight).
"""

from typing import Optional

import numpy as np


def _compute_reliability(X: np.ndarray, y_struct: np.ndarray,
                         importance: Optional[np.ndarray] = None,
                         k: int = 10) -> np.ndarray:
    """Per-sample reliability = fraction of k nearest neighbours with the
    same event status, in importance-weighted feature space.

    Returns array (n_samples,) clipped to [0.1, 1.0].
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    if importance is not None and importance.shape[0] == Xs.shape[1]:
        w = np.sqrt(np.maximum(importance, 0))
        w_sum = w.sum()
        if w_sum > 1e-10:
            w = w / w_sum
            Xs = Xs * w

    n = Xs.shape[0]
    k_actual = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1)
    nn.fit(Xs)
    _, idx = nn.kneighbors(Xs)
    neighbor = idx[:, 1:]  # drop self

    event = y_struct['event'].astype(int)
    same = (event[neighbor] == event[:, None]).astype(float)
    rel = same.mean(axis=1)
    return np.clip(rel, 0.1, 1.0)


def _quick_importance(X: np.ndarray, y_struct: np.ndarray,
                       random_state: int = 42) -> np.ndarray:
    """Fit a fast GBSA and return feature_importances_."""
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    est = GradientBoostingSurvivalAnalysis(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        subsample=0.8, random_state=random_state)
    est.fit(X, y_struct)
    return np.asarray(est.feature_importances_, dtype=np.float64)


class SurvivalRSFWeighted:
    """RSF with reliability-weighted samples."""

    def __init__(self, n_estimators: int = 300,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 20,
                 max_features: str = 'sqrt',
                 k_neighbors: int = 10,
                 n_jobs: int = -1,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sksurv.ensemble import RandomSurvivalForest
        X = np.asarray(X, dtype=np.float64)
        importance = _quick_importance(X, y_struct, self.random_state)
        weights = _compute_reliability(X, y_struct,
                                       importance=importance, k=self.k_neighbors)
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(X, y_struct, sample_weight=weights)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


def _inv_freq_weights(y_struct: np.ndarray) -> np.ndarray:
    """Inverse-frequency event weighting: events get higher weight (rare class).

    weight_i = 1 / p_class. With ~47/1106 = 4.2% event rate, events get
    weight ~23.5 and censored get weight ~1.04. Normalised so weights mean 1.
    """
    event = y_struct['event'].astype(int)
    n = len(event)
    n_event = int(event.sum())
    n_censored = n - n_event
    if n_event == 0 or n_censored == 0:
        return np.ones(n)
    w_event = 0.5 * n / n_event
    w_censored = 0.5 * n / n_censored
    w = np.where(event == 1, w_event, w_censored)
    return w / w.mean()


class SurvivalRSFInvFreq:
    """RSF with inverse-frequency event weighting (no kNN reliability)."""

    def __init__(self, n_estimators: int = 300,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 20,
                 max_features: str = 'sqrt',
                 n_jobs: int = -1,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sksurv.ensemble import RandomSurvivalForest
        weights = _inv_freq_weights(y_struct)
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(np.asarray(X, dtype=np.float64), y_struct,
                        sample_weight=weights)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


class SurvivalGBSAInvFreq:
    """GBSA with inverse-frequency event weighting."""

    def __init__(self, n_estimators: int = 200,
                 learning_rate: float = 0.05,
                 max_depth: int = 3,
                 min_samples_leaf: int = 10,
                 subsample: float = 0.8,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        weights = _inv_freq_weights(y_struct)
        self._model = GradientBoostingSurvivalAnalysis(
            loss='coxph',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._model.fit(np.asarray(X, dtype=np.float64), y_struct,
                        sample_weight=weights)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


class SurvivalGBSAWeighted:
    """GBSA with reliability-weighted samples."""

    def __init__(self, n_estimators: int = 200,
                 learning_rate: float = 0.05,
                 max_depth: int = 3,
                 min_samples_leaf: int = 10,
                 subsample: float = 0.8,
                 k_neighbors: int = 10,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        X = np.asarray(X, dtype=np.float64)
        importance = _quick_importance(X, y_struct, self.random_state)
        weights = _compute_reliability(X, y_struct,
                                       importance=importance, k=self.k_neighbors)
        self._model = GradientBoostingSurvivalAnalysis(
            loss='coxph',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._model.fit(X, y_struct, sample_weight=weights)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)
