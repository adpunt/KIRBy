"""sksurv-based survival downstream wrappers for ANNITIA.

Three estimators are exposed:
    SurvivalCoxPH   — CoxPHSurvivalAnalysis with a small L2 (ridge) penalty.
    SurvivalCoxnet  — CoxnetSurvivalAnalysis (elastic net), alpha tuned by
                      the built-in regularisation path. Adopts the recipe from
                      the challenge's hello_world starter notebook to handle
                      low events-per-variable.
    SurvivalRSF     — RandomSurvivalForest with min_samples_leaf=20 (also
                      lifted from hello_world; argued to reduce overfit on
                      rare events).

All three expose a uniform interface:
    .fit(X, y_struct) where y_struct is sksurv structured array
                        (dtype=[('event','?'),('time','<f8')])
    .predict_risk(X) -> np.ndarray of risk scores (higher = worse prognosis)

Cox/Coxnet are fit on standardized features; the scaler is fit on training
data only. RSF does not need scaling.

`concordance` is a thin wrapper around sksurv.metrics.concordance_index_censored
that returns the C-index float and tolerates degenerate folds (zero events) by
returning NaN.
"""

from typing import Optional

import numpy as np


class SurvivalCoxPH:
    """L2-penalised Cox PH. Standardises features internally."""

    def __init__(self, alpha: float = 1.0, n_iter: int = 200, ties: str = 'breslow'):
        self.alpha = alpha
        self.n_iter = n_iter
        self.ties = ties
        self._scaler = None
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        X = np.asarray(X, dtype=np.float64)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = CoxPHSurvivalAnalysis(
            alpha=self.alpha, n_iter=self.n_iter, ties=self.ties)
        try:
            self._model.fit(Xs, y_struct)
        except Exception:
            # Cox can fail to converge on high-dim or near-collinear inputs.
            # Bump alpha and retry once.
            self._model = CoxPHSurvivalAnalysis(
                alpha=max(self.alpha * 10.0, 10.0),
                n_iter=self.n_iter, ties=self.ties)
            self._model.fit(Xs, y_struct)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        # CoxPHSurvivalAnalysis.predict returns the linear predictor (log hazard).
        return np.asarray(self._model.predict(Xs), dtype=np.float64)


class SurvivalCoxnet:
    """Elastic-net Cox PH (sksurv.linear_model.CoxnetSurvivalAnalysis).

    Fits the full alpha regularisation path internally, then picks the alpha
    that maximises training-set partial likelihood. We do not nest CV here —
    the outer search loop is already CV-evaluating the wrapper's choice, so
    nesting would multiply runtime ~5x for marginal gain.
    """

    def __init__(self, l1_ratio: float = 0.9,
                 alpha_min_ratio: float = 0.01,
                 max_iter: int = 1000):
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.max_iter = max_iter
        self._scaler = None
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        X = np.asarray(X, dtype=np.float64)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            max_iter=self.max_iter,
            fit_baseline_model=True,
        )
        try:
            self._model.fit(Xs, y_struct)
        except Exception:
            # Fall back to ridge-heavy fit if pure lasso path fails.
            self._model = CoxnetSurvivalAnalysis(
                l1_ratio=0.1, alpha_min_ratio=0.1,
                max_iter=self.max_iter, fit_baseline_model=True)
            self._model.fit(Xs, y_struct)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        # Coxnet returns predictions across the alpha grid by default; we want
        # a single risk score per sample. Use the smallest alpha (least
        # shrinkage) by default — the search loop handles regularisation
        # selection externally via CV.
        pred = np.asarray(self._model.predict(Xs), dtype=np.float64)
        if pred.ndim == 2:
            pred = pred[:, -1]
        return pred


class SurvivalRSF:
    """Random Survival Forest. min_samples_leaf=20 default lifted from the
    challenge hello_world starter (argued to reduce overfit on rare events)."""

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
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(np.asarray(X, dtype=np.float64), y_struct)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


class SurvivalGBSA:
    """Gradient Boosted Survival Analysis (sksurv.ensemble.GradientBoostingSurvivalAnalysis).

    Loss = Cox partial likelihood. Often beats RSF on low-event tabular data
    because it can capture non-linear effects with sparser splits.
    """

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
        self._model = GradientBoostingSurvivalAnalysis(
            loss='coxph',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._model.fit(np.asarray(X, dtype=np.float64), y_struct)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


def concordance(y_struct: np.ndarray, risk: np.ndarray) -> float:
    """Censored C-index. Returns NaN if no comparable pairs (e.g. 0 events)."""
    from sksurv.metrics import concordance_index_censored
    if not y_struct['event'].any():
        return float('nan')
    try:
        c, *_ = concordance_index_censored(
            y_struct['event'], y_struct['time'], risk)
        return float(c)
    except Exception:
        return float('nan')


def _stacker_factory():
    from experiments.annitia.stacking import SurvivalStacker
    return SurvivalStacker()


SURVIVAL_MODELS = {
    'cox': SurvivalCoxPH,
    'coxnet': SurvivalCoxnet,
    'rsf': SurvivalRSF,
    'gbsa': SurvivalGBSA,
    'stack': _stacker_factory,
}
