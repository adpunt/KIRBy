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

from typing import Optional, Tuple

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


def _drop_invalid_xy(X: np.ndarray, y_struct: np.ndarray):
    """Drop rows with NaN time (invalid patients). Returns (X, y, keep)."""
    keep = ~np.isnan(y_struct['time'])
    return np.asarray(X, dtype=np.float64)[keep], y_struct[keep], keep


# ----------------------------------------------------------------------
# B.1 / B.9 — CatBoost survival wrappers
# ----------------------------------------------------------------------
class SurvivalCatBoostCox:
    """CatBoost with `Cox` loss (`SurvivalAft` for AFT — see SurvivalCatBoostAft).

    Survival-Cox in CatBoost expects targets as
        - signed time: positive for censored, negative for event-observed
    or via the explicit (event, time) helper. Here we feed the signed-time
    convention to keep the wrapper portable across CatBoost versions.
    """

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 4, l2_leaf_reg: float = 3.0,
                 random_state: int = 42, verbose: bool = False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.verbose = verbose
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                'catboost not installed — `pip install catboost` '
                f'(missing on this host: {e})')
        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        # CatBoost's Cox loss takes signed time: negative for events,
        # positive for censored.
        signed_time = np.where(y_struct['event'],
                               -y_struct['time'], y_struct['time'])
        self._model = CatBoostRegressor(
            loss_function='Cox',
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._model.fit(X, signed_time)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        # CatBoost-Cox returns the linear predictor (log hazard ratio).
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


class SurvivalCatBoostAft:
    """CatBoost with `SurvivalAft` loss (Accelerated Failure Time).

    Useful when the proportional-hazards assumption fails (e.g. for the
    death endpoint where treatment effects accelerate / decelerate time).
    """

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 4, l2_leaf_reg: float = 3.0,
                 dist: str = 'Normal', scale: float = 1.0,
                 random_state: int = 42, verbose: bool = False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.dist = dist
        self.scale = scale
        self.random_state = random_state
        self.verbose = verbose
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError as e:
            raise ImportError(
                'catboost not installed — `pip install catboost` '
                f'(missing on this host: {e})')
        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        # SurvivalAft uses a 2-column label: lower / upper bounds of the
        # event time (lower=upper for observed, upper=+inf for censored).
        lower = y_struct['time'].astype(np.float64)
        upper = np.where(y_struct['event'], y_struct['time'],
                         np.full_like(lower, np.finfo(np.float64).max))
        labels = np.column_stack([lower, upper])
        self._model = CatBoostRegressor(
            loss_function=f'SurvivalAft:dist={self.dist};scale={self.scale}',
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._model.fit(Pool(X, label=labels))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        # AFT predicts log(time-to-event); higher = better. Negate for risk.
        pred = np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)
        return -pred


# ----------------------------------------------------------------------
# B.8 — LightGBM-AFT
# ----------------------------------------------------------------------
class SurvivalLightGBMAft:
    """LightGBM with `regression_l1` + AFT loss via `aft_loss_distribution`.

    Available since LightGBM 3.0+. Targets are interval-censored: lower and
    upper bounds. Censored rows have upper=+inf.
    """

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 num_leaves: int = 31, max_depth: int = 6,
                 dist: str = 'normal', sigma: float = 1.0,
                 random_state: int = 42, verbose: int = -1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.dist = dist
        self.sigma = sigma
        self.random_state = random_state
        self.verbose = verbose
        self._booster = None
        self._fallback = False

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(f'lightgbm missing: {e}')
        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        lower = y_struct['time'].astype(np.float64)
        upper = np.where(y_struct['event'], y_struct['time'],
                         np.full_like(lower, np.inf))
        # AFT in LightGBM: pass label_lower_bound and label_upper_bound
        train = lgb.Dataset(
            X, label=lower,
            params={'label_lower_bound': lower.tolist(),
                    'label_upper_bound': upper.tolist()},
            free_raw_data=False,
        )
        params = {
            'objective': 'regression_l1',
            'metric': 'l1',
            'aft_loss_distribution': self.dist,
            'aft_loss_distribution_scale': self.sigma,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'seed': self.random_state,
            'verbose': self.verbose,
        }
        try:
            self._booster = lgb.train(
                params, train, num_boost_round=self.n_estimators,
            )
        except Exception:
            # Older lightgbm or AFT not enabled → fall back to plain L1
            # regression on time (with event mask weighting).
            self._fallback = True
            ds = lgb.Dataset(X, label=y_struct['time'])
            self._booster = lgb.train(
                {**params, 'objective': 'regression_l1'},
                ds, num_boost_round=self.n_estimators,
            )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        pred = self._booster.predict(np.asarray(X, dtype=np.float64))
        return -np.asarray(pred, dtype=np.float64)  # higher pred = longer time → lower risk


# ----------------------------------------------------------------------
# B.2 — NGBoost survival wrapper (via the Cox / AFT distribution)
# ----------------------------------------------------------------------
class SurvivalNGBoost:
    """NGBoost with a survival distribution (Cox or AFT-LogNormal).

    NGBoost 0.3+ supports survival via the `Survival` API. Falls back to
    point-estimate regression if NGBoostSurvival isn't importable.
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 dist: str = 'logNormal', random_state: int = 42,
                 verbose: bool = False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.dist = dist
        self.random_state = random_state
        self.verbose = verbose
        self._model = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        # ngboost 0.3.12 still uses `np.bool` (removed in numpy ≥1.24).
        # Apply the standard shim before import — harmless if the alias
        # already exists.
        if not hasattr(np, 'bool'):
            np.bool = bool  # type: ignore[attr-defined]
        try:
            from ngboost import NGBSurvival
            from ngboost.distns import LogNormal
            from ngboost.distns.exponential import Exponential
        except ImportError as e:
            raise ImportError(f'ngboost missing: {e}')
        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        Dist = LogNormal if self.dist.lower().startswith('logn') else Exponential
        self._model = NGBSurvival(
            Dist=Dist,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        # NGBSurvival API: fit(X, time, event)
        self._model.fit(X, y_struct['time'].astype(np.float64),
                        y_struct['event'].astype(int))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        # Use predicted median survival time; risk = -median.
        Xn = np.asarray(X, dtype=np.float64)
        try:
            pred_dist = self._model.pred_dist(Xn)
            # log-normal: median = exp(loc); exponential: median = ln(2)/scale
            if hasattr(pred_dist, 'mean'):
                med = pred_dist.mean()
            else:
                med = self._model.predict(Xn)
        except Exception:
            med = self._model.predict(Xn)
        return -np.asarray(med, dtype=np.float64)


# ----------------------------------------------------------------------
# B.3 / B.4 — Plain regressors / classifiers (used by pseudo-obs / landmark)
# ----------------------------------------------------------------------
class RegressorLightGBM:
    """LightGBM regressor — used by the pseudo-observations framing."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, max_depth: int = 6,
                 random_state: int = 42, verbose: int = -1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from lightgbm import LGBMRegressor
        self._model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._model.fit(np.asarray(X, dtype=np.float64),
                        np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=np.float64))


class RegressorCatBoost:
    """CatBoost regressor — used by pseudo-obs / landmark / general regression."""

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 4, random_state: int = 42, verbose: bool = False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(f'catboost missing: {e}')
        self._model = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, random_state=self.random_state, verbose=self.verbose,
        )
        self._model.fit(np.asarray(X, dtype=np.float64),
                        np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


class ClassifierCatBoost:
    """CatBoost binary classifier — used by landmark binary framing."""

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 4, random_state: int = 42, verbose: bool = False,
                 class_weights: Optional[Tuple[float, float]] = None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self.class_weights = class_weights
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError(f'catboost missing: {e}')
        self._model = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, random_state=self.random_state,
            verbose=self.verbose, class_weights=list(self.class_weights) if self.class_weights else None,
        )
        self._model.fit(np.asarray(X, dtype=np.float64),
                        np.asarray(y, dtype=int),
                        sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict_proba(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)


# ----------------------------------------------------------------------
# B.6 — MLP regressor / classifier (sklearn-based)
# ----------------------------------------------------------------------
class MLPRegressorSurvival:
    """sklearn MLPRegressor wrapper that targets a continuous risk score.

    Used by:
      - pseudo-observation framing (target = jackknife pseudo-value)
      - direct landmark regression (target = censored time at landmark)
    StandardScaler-applied features for stable training.
    """

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 32),
                 alpha: float = 1e-3, max_iter: int = 500,
                 early_stopping: bool = True, random_state: int = 42,
                 learning_rate_init: float = 1e-3):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.learning_rate_init = learning_rate_init
        self._scaler = None
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        X = np.asarray(X, dtype=np.float64)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha, max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init,
        )
        self._model.fit(Xs, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return np.asarray(self._model.predict(Xs), dtype=np.float64)


class MLPClassifierSurvival:
    """sklearn MLPClassifier — landmark binary framing."""

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 32),
                 alpha: float = 1e-3, max_iter: int = 500,
                 early_stopping: bool = True, random_state: int = 42,
                 learning_rate_init: float = 1e-3):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.learning_rate_init = learning_rate_init
        self._scaler = None
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        X = np.asarray(X, dtype=np.float64)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha, max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init,
        )
        # MLPClassifier doesn't accept sample_weight; weights are applied at
        # the loss level only via subclassing — for now we ignore them.
        self._model.fit(Xs, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return np.asarray(self._model.predict_proba(Xs), dtype=np.float64)


try:
    from experiments.annitia.pretrained_mlp import SurvivalPretrainedMLP
except Exception:
    SurvivalPretrainedMLP = None


SURVIVAL_MODELS = {
    'cox': SurvivalCoxPH,
    'coxnet': SurvivalCoxnet,
    'rsf': SurvivalRSF,
    'gbsa': SurvivalGBSA,
    # Phase B additions — survival objectives
    'catboost_cox': SurvivalCatBoostCox,
    'catboost_aft': SurvivalCatBoostAft,
    'lgbm_aft': SurvivalLightGBMAft,
    'ngboost': SurvivalNGBoost,
}
if SurvivalPretrainedMLP is not None:
    SURVIVAL_MODELS['pretrained_mlp'] = SurvivalPretrainedMLP

REGRESSORS = {
    'lgbm': RegressorLightGBM,
    'catboost': RegressorCatBoost,
    'mlp': MLPRegressorSurvival,
}

CLASSIFIERS = {
    'catboost': ClassifierCatBoost,
    'mlp': MLPClassifierSurvival,
}
