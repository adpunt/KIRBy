"""
Domain-agnostic hybrid representation search.

Accepts pre-computed numpy feature matrices and finds the best hybrid
configuration across models, budgets, allocation methods, and importance methods.

Works for any domain (molecular, antibody, fMRI, gene expression, etc.) — the
caller is responsible for generating representations and group labels.

Supports fine-tuned representations via `rep_factories`: per-fold callables that
produce train/val embeddings. These are pre-computed once and merged with frozen
base_reps during evaluation.

Usage:
    from kirby.find_best_hybrid import find_best_hybrid

    result = find_best_hybrid(
        base_reps={'ecfp4': X_ecfp, 'maccs': X_maccs, 'pdv': X_pdv},
        labels=y,
        groups=scaffold_groups,
        speed='fast',
    )
    print(result.best_overall)

    # With fine-tuned reps:
    from kirby.finetuning import make_antibody_lora_factory
    result = find_best_hybrid(
        base_reps={'kmer': X_kmer, 'aac': X_aac},
        labels=y,
        rep_factories={'lora_esm2': make_antibody_lora_factory(seqs, ...)},
    )
"""

import hashlib
import pickle
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Blueprint:
    """Minimal transferable recipe for recreating a hybrid."""
    reps: List[str]
    allocation_method: str
    importance_method: str
    budget: int
    internal_evaluator: str = 'ridge'
    step_size: int = 5
    speed: str = ''
    n_folds: int = 0
    seed: int = 42
    metric: str = 'r2'
    task_type: str = 'regression'
    augmentation_strategy: str = 'uncapped'

    def to_dict(self) -> dict:
        return {
            'reps': self.reps,
            'allocation_method': self.allocation_method,
            'importance_method': self.importance_method,
            'internal_evaluator': self.internal_evaluator,
            'budget': self.budget,
            'step_size': self.step_size,
            'speed': self.speed,
            'n_folds': self.n_folds,
            'seed': self.seed,
            'metric': self.metric,
            'task_type': self.task_type,
            'augmentation_strategy': self.augmentation_strategy,
        }


@dataclass
class FittedResult:
    """Concrete fitted result with feature indices and optional weights."""
    blueprint: Blueprint
    selected_indices: Dict[str, np.ndarray]
    feature_weights: Optional[Dict[str, np.ndarray]] = None
    n_features: int = 0
    augmentation_info: Optional[dict] = None


@dataclass
class SearchResult:
    """Full output of find_best_hybrid()."""
    best_overall: dict
    best_by_model: Dict[str, dict]
    best_by_dimension: Dict[int, dict]
    performance_matrix: Dict[str, Dict[str, float]]
    baselines: Dict[str, Dict[str, float]]
    all_results: List[dict]
    metric_name: str = 'r2'
    lower_is_better: bool = False
    fitted_result: Optional[FittedResult] = None
    finetuned_embeddings: Optional[Dict[str, np.ndarray]] = None


# =============================================================================
# METRIC ABSTRACTION
# =============================================================================

def _concordance_index(y_true, y_pred):
    """Concordance index (CI) — standard DTA metric. O(n²), fine for CV folds."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            concordant += (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) > 0
            discordant += (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) < 0
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


BUILTIN_METRICS = {
    'r2':          (r2_score, False),
    'rmse':        (lambda y, p: np.sqrt(mean_squared_error(y, p)), True),
    'mae':         (mean_absolute_error, True),
    'accuracy':    (accuracy_score, False),
    'auc':         (roc_auc_score, False),
    'spearman':    (lambda y, p: float(spearmanr(y, p)[0]), False),
    'pearson':     (lambda y, p: float(pearsonr(y, p)[0]), False),
    'ci':          (_concordance_index, False),
    'f1_weighted': (lambda y, p: f1_score(y, p, average='weighted'), False),
    'mcc':         (matthews_corrcoef, False),
}


def _resolve_metric(metric) -> Tuple[Callable, bool]:
    """Return (scorer_fn, lower_is_better).

    Accepts:
    - str: lookup in BUILTIN_METRICS
    - callable: assume metric(y_true, y_pred), higher is better
    - tuple (callable, lower_is_better): explicit specification
    """
    if isinstance(metric, str):
        if metric not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Available: {list(BUILTIN_METRICS.keys())}"
            )
        return BUILTIN_METRICS[metric]

    if isinstance(metric, tuple) and len(metric) == 2:
        return metric[0], metric[1]

    if callable(metric):
        return metric, False

    raise TypeError(f"metric must be str, callable, or (callable, bool). Got {type(metric)}")


def _score_to_comparable(raw_score: float, lower_is_better: bool) -> float:
    """Convert a raw score so that higher is always better (for comparisons)."""
    return -raw_score if lower_is_better else raw_score


def _should_prune(best_comparable: float, global_best_comparable: float,
                  threshold: float) -> bool:
    """Check if a search dimension should be pruned.

    Uses gap-based comparison that works for both positive comparables
    (R², accuracy) and negative comparables (negated RMSE/MAE).
    Prunes if the gap from global best exceeds (1-threshold) * |global_best|.
    """
    if abs(global_best_comparable) < 1e-10:
        return False  # global best near zero — pruning meaningless
    gap = global_best_comparable - best_comparable
    return gap > (1 - threshold) * abs(global_best_comparable)


def _screen_reps(
    base_reps: Dict[str, np.ndarray],
    baselines: Dict[str, Dict[str, float]],
    lower_is_better: bool,
    threshold: float,
    *,
    fold_reps=None,
    verbose: bool = True,
) -> tuple:
    """Screen out weak reps before hybrid search.

    For each rep, takes the best score across all models from baselines.
    Converts to comparable (higher=better) and applies a gap-based cutoff:
    reps more than ``threshold`` fraction away from the best are dropped.

    Returns (filtered_base_reps, filtered_fold_reps).
    """
    # Best single-rep score across models
    best_per_rep = {}
    for rep_name, scores in baselines.items():
        model_scores = [v for k, v in scores.items()
                        if k not in ('dimension', '_secondary')
                        and isinstance(v, (int, float)) and np.isfinite(v)]
        if model_scores:
            if lower_is_better:
                best_per_rep[rep_name] = min(model_scores)
            else:
                best_per_rep[rep_name] = max(model_scores)

    if not best_per_rep:
        return base_reps, fold_reps

    comparable = {k: _score_to_comparable(v, lower_is_better)
                  for k, v in best_per_rep.items()}
    best_comparable = max(comparable.values())

    # Gap-based cutoff: drop reps more than `threshold` away from best
    cutoff = best_comparable - threshold * abs(best_comparable)

    surviving = set()
    for rep_name, comp in comparable.items():
        if comp >= cutoff:
            surviving.add(rep_name)

    # Always keep at least 2 reps
    if len(surviving) < 2:
        top2 = sorted(comparable, key=comparable.get, reverse=True)[:2]
        surviving = set(top2)

    screened = set(base_reps.keys()) - surviving
    if verbose and screened:
        print(f"\nRep screening (threshold={threshold}): "
              f"kept {len(surviving)}/{len(base_reps)} reps, "
              f"screened out: {sorted(screened)}")

    filtered_reps = {k: v for k, v in base_reps.items() if k in surviving}

    filtered_fold_reps = fold_reps
    if fold_reps is not None:
        filtered_fold_reps = {}
        for fold_idx, fold_data in fold_reps.items():
            filtered_fold_reps[fold_idx] = {
                k: v for k, v in fold_data.items() if k in surviving
            }

    return filtered_reps, filtered_fold_reps


# =============================================================================
# MODEL RESOLUTION
# =============================================================================

def _get_xgb_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, colsample_bytree=0.5,
                            random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                     random_state=42, n_jobs=-1)


def _get_xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, colsample_bytree=0.5,
                             random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                      random_state=42, n_jobs=-1)


def _get_lgbm_regressor():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                     random_state=42, n_jobs=-1)


def _get_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                      random_state=42, n_jobs=-1)


def _get_catboost_regressor():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=500, learning_rate=0.1, depth=6, verbose=0,
            random_seed=42)
    except ImportError:
        return _get_lgbm_regressor()


def _get_catboost_classifier():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=6, verbose=0,
            random_seed=42)
    except ImportError:
        return _get_lgbm_classifier()


def _get_gp_regressor():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    return GaussianProcessRegressor(
        kernel=RBF() + WhiteKernel(), n_restarts_optimizer=3, random_state=42)


def _get_gp_classifier():
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    return GaussianProcessClassifier(kernel=RBF(), random_state=42)


class TanimotoKernel(__import__('sklearn.gaussian_process.kernels',
                                fromlist=['Kernel']).Kernel):
    """Tanimoto (Jaccard) kernel for binary fingerprint vectors.

    k(x, x') = <x, x'> / (||x||^2 + ||x'||^2 - <x, x'>)

    Returns 1 for identical non-zero vectors, 0 for zero vectors.
    No learnable hyperparameters.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=np.float64)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=np.float64)

        XY = X @ Y.T
        X_sq = np.sum(X ** 2, axis=1)
        Y_sq = np.sum(Y ** 2, axis=1)

        # denom[i,j] = ||x_i||^2 + ||y_j||^2 - <x_i, y_j>
        denom = X_sq[:, None] + Y_sq[None, :] - XY
        K = np.where(denom > 0, XY / denom, 0.0)

        if eval_gradient:
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        return K

    def diag(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.sum(X ** 2, axis=1)
        return np.where(norms > 0, 1.0, 0.0)

    def is_stationary(self):
        return False

    @property
    def hyperparameter_names(self):
        return []

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, value):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def clone_with_theta(self, theta):
        return TanimotoKernel()


def _get_dnn_regressor():
    from kirby.models.dnn import DNNRegressor
    return DNNRegressor()


def _get_dnn_classifier():
    from kirby.models.dnn import DNNClassifier
    return DNNClassifier()


def _get_cnn_regressor():
    from kirby.models.cnn import CNNRegressor
    return CNNRegressor()


def _get_cnn_classifier():
    from kirby.models.cnn import CNNClassifier
    return CNNClassifier()


DEFAULT_MODELS = {
    'rf': {
        'regressor': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestRegressor']).RandomForestRegressor(
            n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1),
        'classifier': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
            n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1),
        'tier': 'fast',
    },
    'xgboost': {
        'regressor': _get_xgb_regressor,
        'classifier': _get_xgb_classifier,
        'tier': 'fast',
    },
    'ridge': {
        'regressor': lambda: __import__('sklearn.linear_model', fromlist=['RidgeCV']).RidgeCV(
            alphas=np.logspace(-4, 4, 50)),
        'classifier': lambda: __import__('sklearn.linear_model', fromlist=['LogisticRegressionCV']).LogisticRegressionCV(
            Cs=10, random_state=42, max_iter=1000, n_jobs=-1),
        'tier': 'full',
    },
    'knn': {
        'regressor': lambda: __import__('sklearn.neighbors', fromlist=['KNeighborsRegressor']).KNeighborsRegressor(
            n_neighbors=5, n_jobs=-1),
        'classifier': lambda: __import__('sklearn.neighbors', fromlist=['KNeighborsClassifier']).KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1),
        'tier': 'full',
        '_adaptive_k': True,
    },
    'lightgbm': {
        'regressor': _get_lgbm_regressor,
        'classifier': _get_lgbm_classifier,
        'tier': 'full',
    },
    'mlp': {
        'regressor': lambda: __import__('sklearn.neural_network', fromlist=['MLPRegressor']).MLPRegressor(
            hidden_layer_sizes=(256,), random_state=42, max_iter=500,
            early_stopping=True, validation_fraction=0.1),
        'classifier': lambda: __import__('sklearn.neural_network', fromlist=['MLPClassifier']).MLPClassifier(
            hidden_layer_sizes=(256,), random_state=42, max_iter=500,
            early_stopping=True, validation_fraction=0.1),
        'tier': 'full',
    },
    'svm': {
        'regressor': lambda: __import__('sklearn.svm', fromlist=['SVR']).SVR(),
        'classifier': lambda: __import__('sklearn.svm', fromlist=['SVC']).SVC(),
        'tier': 'full',
        '_max_samples': 10000,
    },
    'gp': {
        'regressor': _get_gp_regressor,
        'classifier': _get_gp_classifier,
        'tier': 'full',
        '_max_samples': 5000,
    },
    'catboost': {
        'regressor': _get_catboost_regressor,
        'classifier': _get_catboost_classifier,
        'tier': 'full',
    },
    'dnn': {
        'regressor': _get_dnn_regressor,
        'classifier': _get_dnn_classifier,
        'tier': 'full',
    },
    'cnn': {
        'regressor': _get_cnn_regressor,
        'classifier': _get_cnn_classifier,
        'tier': 'full',
    },
}

# Tier ordering for filtering
_TIER_ORDER = {'fast': 0, 'full': 1}


def _resolve_models(models, task_type: str, speed: str,
                     n_samples: Optional[int] = None) -> Dict[str, Any]:
    """Resolve model spec to dict of {name: sklearn_model_instance}.

    models can be:
    - None: use defaults filtered by speed tier
    - List[str]: lookup by name
    - Dict[str, model]: custom models (passthrough)
    - Single model instance: wrap as {'custom': model}

    Models with _max_samples are auto-skipped when n_samples exceeds the limit.
    """
    key = 'classifier' if task_type == 'classification' else 'regressor'
    max_tier = _TIER_ORDER.get(speed, 1)

    def _check_max_samples(name, spec):
        max_s = spec.get('_max_samples')
        if max_s is not None and n_samples is not None and n_samples > max_s:
            print(f"  Skipping {name}: n_samples={n_samples} > _max_samples={max_s}")
            return False
        return True

    if models is None:
        resolved = {}
        for name, spec in DEFAULT_MODELS.items():
            if _TIER_ORDER.get(spec['tier'], 0) <= max_tier:
                if _check_max_samples(name, spec):
                    resolved[name] = spec[key]()
        return resolved

    if isinstance(models, list):
        resolved = {}
        for name in models:
            if name not in DEFAULT_MODELS:
                raise ValueError(
                    f"Unknown model '{name}'. "
                    f"Available: {list(DEFAULT_MODELS.keys())}"
                )
            if _check_max_samples(name, DEFAULT_MODELS[name]):
                resolved[name] = DEFAULT_MODELS[name][key]()
        return resolved

    if isinstance(models, dict):
        return {name: deepcopy(m) for name, m in models.items()}

    # Single model instance
    if hasattr(models, 'fit') and hasattr(models, 'predict'):
        return {'custom': deepcopy(models)}

    raise TypeError(
        f"models must be None, list of names, dict of models, or a single model. "
        f"Got {type(models)}"
    )


def get_model(name: str, task_type: str = 'regression',
              n_samples: Optional[int] = None) -> Any:
    """Return a fresh model instance from DEFAULT_MODELS.

    Returns None (with warning) if n_samples exceeds the model's _max_samples.
    Useful for survey scripts to avoid duplicating model definitions.

    Args:
        name: Model name (e.g. 'ridge', 'gp', 'dnn').
        task_type: 'regression' or 'classification'.
        n_samples: Dataset size — models with _max_samples auto-skip if exceeded.

    Returns:
        Model instance, or None if skipped due to size guard.
    """
    if name not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(DEFAULT_MODELS.keys())}")

    spec = DEFAULT_MODELS[name]
    max_s = spec.get('_max_samples')
    if max_s is not None and n_samples is not None and n_samples > max_s:
        print(f"  Skipping {name}: n_samples={n_samples} > _max_samples={max_s}")
        return None

    key = 'classifier' if task_type == 'classification' else 'regressor'
    return spec[key]()


def _detect_task_type(labels: np.ndarray) -> str:
    """Auto-detect regression vs classification from labels."""
    unique = np.unique(labels)
    if len(unique) <= 20 and np.allclose(labels, labels.astype(int)):
        return 'classification'
    return 'regression'


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

_HP_SEARCH_SPACES = {
    'rf': {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
    },
    'mlp': {
        'hidden_layer_sizes': [(128,), (256,), (256, 128), (512,)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
    },
    'gp': {
        'kernel': [
            __import__('sklearn.gaussian_process.kernels', fromlist=['RBF']).RBF(),
            __import__('sklearn.gaussian_process.kernels', fromlist=['RBF', 'WhiteKernel']).RBF()
            + __import__('sklearn.gaussian_process.kernels', fromlist=['WhiteKernel']).WhiteKernel(),
            __import__('sklearn.gaussian_process.kernels', fromlist=['Matern']).Matern(nu=1.5),
            __import__('sklearn.gaussian_process.kernels', fromlist=['Matern']).Matern(nu=2.5),
            TanimotoKernel(),
        ],
        'alpha': [1e-10, 1e-5, 1e-2, 0.1],
    },
    'catboost': {
        'iterations': [200, 500, 1000],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 10],
    },
    # ridge: skip — uses RidgeCV/LogisticRegressionCV internally
    # dnn: skip — too slow for RandomizedSearchCV
    # cnn: skip — too slow for RandomizedSearchCV
}


def _count_grid_size(param_grid: dict) -> int:
    """Count total combinations in a parameter grid."""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def tune_models(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    task_type: Optional[str] = None,
    speed: str = 'fast',
    n_iter: int = 20,
    n_folds: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tune model hyperparameters via RandomizedSearchCV on concatenated reps.

    Runs RandomizedSearchCV for each model in the speed tier (except ridge)
    using the concatenation of all base_reps as features. Returns a dict of
    tuned model instances ready to pass to find_best_hybrid(models=...).

    Args:
        base_reps: Dict of {name: ndarray} base representations.
        labels: Target values.
        groups: Group labels for GroupKFold (optional).
        task_type: 'regression' or 'classification' (auto-detected if None).
        speed: Speed tier — determines which models to tune.
        n_iter: Max RandomizedSearchCV iterations per model.
        n_folds: Number of CV folds for tuning.
        seed: Random seed.

    Returns:
        Dict[str, model] — tuned model instances keyed by model name.
    """
    from sklearn.model_selection import (
        RandomizedSearchCV, GroupKFold, KFold, StratifiedKFold,
    )

    if task_type is None:
        task_type = _detect_task_type(labels)

    # Concatenate all base reps into a single feature matrix
    # NOTE: scaling is done inside Pipeline per CV fold (no pre-scaling leakage)
    X = np.hstack([base_reps[name] for name in sorted(base_reps.keys())])
    y = labels

    # Build CV splitter
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique_groups)
        cv = GroupKFold(n_splits=actual_folds)
        cv_args = {'groups': groups}
    elif task_type == 'classification':
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_args = {}
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_args = {}

    # Scoring metric
    if task_type == 'classification':
        scoring = 'accuracy'
    else:
        scoring = 'r2'

    # Resolve which models to tune (pass n_samples for size guards)
    default_models = _resolve_models(None, task_type, speed,
                                      n_samples=len(y))

    tuned = {}
    for model_name, model in default_models.items():
        if model_name == 'ridge':
            # RidgeCV/LogisticRegressionCV tunes internally
            tuned[model_name] = model
            print(f"  {model_name}: skip (uses internal CV)")
            continue
        if model_name in ('dnn', 'cnn'):
            # DNN/CNN too slow for RandomizedSearchCV — uses internal early stopping
            tuned[model_name] = model
            print(f"  {model_name}: skip (uses internal early stopping)")
            continue

        param_grid = _HP_SEARCH_SPACES.get(model_name)
        if param_grid is not None:
            # Filter out HP params that don't exist on the actual model
            # (e.g., catboost fallback to LightGBM won't have 'iterations')
            valid_params = model.get_params().keys() if hasattr(model, 'get_params') else set()
            filtered_grid = {k: v for k, v in param_grid.items() if k in valid_params}
            if not filtered_grid:
                param_grid = None  # all params invalid → skip tuning
            elif len(filtered_grid) < len(param_grid):
                param_grid = filtered_grid
        if param_grid is None:
            tuned[model_name] = model
            print(f"  {model_name}: skip (no search space defined)")
            continue

        # Cap n_iter at total grid size to avoid sampling duplicates
        grid_size = _count_grid_size(param_grid)
        actual_iter = min(n_iter, grid_size)

        try:
            # Wrap model in Pipeline so StandardScaler fits per CV fold
            from sklearn.pipeline import Pipeline as _Pipeline
            pipe = _Pipeline([('scaler', StandardScaler()), ('model', deepcopy(model))])
            pipe_grid = {f'model__{k}': v for k, v in param_grid.items()}

            search = RandomizedSearchCV(
                pipe, pipe_grid,
                n_iter=actual_iter,
                cv=cv,
                scoring=scoring,
                random_state=seed,
                n_jobs=1,
                error_score=np.nan,
            )
            search.fit(X, y, **cv_args)

            # Extract the tuned model from the Pipeline (caller's _evaluate_single
            # does its own per-fold scaling, so we return the raw model)
            best_model = search.best_estimator_.named_steps['model']

            # Silence verbosity on tuned estimators
            # LightGBM: verbose=-1 (silent), RF/MLP: verbose=0 (silent)
            if 'LGBM' in type(best_model).__name__:
                best_model.set_params(verbose=-1)
            elif hasattr(best_model, 'verbose'):
                best_model.set_params(verbose=0)
            if hasattr(best_model, 'verbosity'):
                best_model.set_params(verbosity=0)

            tuned[model_name] = best_model
            # Report best params without 'model__' prefix for readability
            clean_params = {k.replace('model__', ''): v
                           for k, v in search.best_params_.items()}
            print(f"  {model_name}: {search.best_score_:.4f} "
                  f"({actual_iter} iters, best: {clean_params})")

        except Exception as e:
            print(f"  {model_name}: tuning FAILED ({e}), using defaults")
            tuned[model_name] = model

    return tuned


# =============================================================================
# SPEED TIER CONFIGURATION
# =============================================================================

SPEED_TIERS = {
    # F12/F13 results (2026-03-29). Two tiers: fast (screening) and full (publication).
    # Budgets NOT in tiers — user must specify.
    # Downstream models NOT in tiers — experiment-controlled.
    'fast': {
        'importance_methods': ['treeshap', 'lightgbm_gain'],
        'allocation_methods': [
            'entropy_1.0', 'top_3', 'greedy_fill_s25',
        ],
        'internal_evaluators': ['lightgbm', 'mlp', 'xgboost'],
        'n_folds': 3,
    },
    'full': {
        'importance_methods': ['treeshap', 'deeplift', 'lightgbm_gain'],
        'allocation_methods': [
            # All 31 F12 survivors — PW family
            'performance_weighted',
            'softmax_T0.5', 'softmax_T2', 'softmax_T10',
            'power_2', 'power_3',
            'thresh_0.3', 'thresh_0.5', 'thresh_0.7',
            'top_2', 'top_3', 'top_5',
            'rank_linear',
            'rank_exp_d0.1', 'rank_exp_d0.3', 'rank_exp_d0.5',
            'score_gap_0.2', 'score_gap_0.3', 'score_gap_0.5', 'score_gap_0.7',
            'entropy_0.5', 'entropy_1.0', 'entropy_2.0',
            # Greedy family
            'greedy_forced_s5', 'greedy_forced_s10', 'greedy_forced_s25',
            'greedy_fill_s5', 'greedy_fill_s10', 'greedy_fill_s25',
            'dom_greedy_f0.7', 'dom_greedy_f0.9',
        ],
        'internal_evaluators': ['ridge', 'rf', 'xgboost', 'lightgbm', 'mlp', 'svm', 'gp'],
        'n_folds': 5,
    },
}


# =============================================================================
# INTERNAL EVALUATION FUNCTIONS
# =============================================================================

def _make_cv_splitter(groups: Optional[np.ndarray], n_folds: int, seed: int, cv_splitter=None):
    """Return (splitter, use_groups) depending on whether groups are provided."""
    if cv_splitter is not None:
        return cv_splitter, False
    if n_folds == 1:
        # Single 80/20 split (intermediary mode)
        if groups is not None:
            return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), True
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), False
    if groups is not None:
        n_unique = len(np.unique(groups))
        if n_unique >= n_folds:
            return GroupKFold(n_splits=n_folds), True
        # Not enough groups for requested n_folds — reduce folds to match groups
        if n_unique >= 2:
            import warnings
            warnings.warn(
                f"Only {n_unique} unique groups for {n_folds}-fold CV. "
                f"Reducing to {n_unique}-fold GroupKFold to preserve group structure.",
                stacklevel=2,
            )
            return GroupKFold(n_splits=n_unique), True
        # Only 1 group — groups are meaningless, warn and fall back
        import warnings
        warnings.warn(
            f"Only 1 unique group — GroupKFold impossible. "
            f"Falling back to KFold({n_folds}).",
            stacklevel=2,
        )
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed), False


def _precompute_fold_reps(
    rep_factories: Dict[str, Callable],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
    cv_splitter=None,
    verbose: bool = True,
) -> Tuple[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Pre-compute fine-tuned representations for each fold.

    Calls each factory once per fold. This happens BEFORE the search grid,
    so fine-tuning cost is n_folds × n_factories (NOT per search config).

    Args:
        rep_factories: {rep_name: factory_callable}
        labels: Target values.
        groups: Group labels for GroupKFold.
        n_folds: Number of CV folds.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        (fold_reps, precomputed_folds)
        fold_reps: {fold_idx: {rep_name: (train_embs, val_embs)}}
        precomputed_folds: [(train_idx, val_idx), ...] — materialized splits
    """
    if not rep_factories:
        return {}, []

    if verbose:
        print(f"\nPre-computing fine-tuned representations "
              f"({len(rep_factories)} factories × {n_folds} folds)...")

    # Materialize fold splits (ensures identical splits downstream)
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed, cv_splitter=cv_splitter)
    # We need a dummy X for splitting
    dummy_X = np.zeros((len(labels), 1))
    split_args = (dummy_X, labels, groups) if use_groups else (dummy_X, labels)
    precomputed_folds = list(splitter.split(*split_args))

    fold_reps = {}
    for fold_idx, (train_idx, val_idx) in enumerate(precomputed_folds):
        fold_reps[fold_idx] = {}
        train_labels = labels[train_idx]

        for rep_name, factory in rep_factories.items():
            start = time.time()
            if verbose:
                print(f"  Fold {fold_idx+1}/{n_folds}, {rep_name}...", end=' ', flush=True)

            try:
                train_embs, val_embs = factory(train_idx, val_idx, train_labels)
                fold_reps[fold_idx][rep_name] = (train_embs, val_embs)
                elapsed = time.time() - start
                if verbose:
                    print(f"done ({train_embs.shape[1]}D, {elapsed:.1f}s)")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")

    return fold_reps, precomputed_folds


def _extract_full_embeddings(
    rep_factories: Dict[str, Callable],
    labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Fine-tune each factory on ALL data and return full-dataset embeddings.

    These full-data embeddings can be reused as standalone base_reps in
    downstream experiments (e.g., passed directly to find_best_hybrid).

    Each factory is called with train_idx=val_idx=all_idx so it trains on
    the full dataset and extracts embeddings for every sample.
    """
    if not rep_factories:
        return {}

    if verbose:
        print(f"\nExtracting full-dataset embeddings "
              f"({len(rep_factories)} factories)...")

    all_idx = np.arange(len(labels))
    full_embeddings = {}

    for rep_name, factory in rep_factories.items():
        start = time.time()
        if verbose:
            print(f"  {rep_name}...", end=" ", flush=True)
        try:
            # Train on all data, extract for all data
            train_embs, _ = factory(all_idx, all_idx, labels)
            full_embeddings[rep_name] = train_embs
            elapsed = time.time() - start
            if verbose:
                print(f"done ({train_embs.shape[1]}D, {elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")

    return full_embeddings


def _evaluate_single(
    X: np.ndarray,
    y: np.ndarray,
    model,
    groups: Optional[np.ndarray],
    n_folds: int,
    metric_fn: Callable,
    seed: int = 42,
    cv_splitter=None,
    secondary_metric_fns: Optional[Dict[str, Callable]] = None,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Tuple[float, float]]]]:
    """Evaluate one (X, model) with CV.

    Returns (mean_score, std_score) if no secondary metrics.
    Returns (mean_score, std_score, {name: (mean, std)}) if secondary metrics provided.
    """
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed, cv_splitter=cv_splitter)
    split_args = (X, y, groups) if use_groups else (X, y)
    scores = []
    sec_scores = {name: [] for name in (secondary_metric_fns or {})}

    # Only scale continuous features; leave binary (ECFP, MACCS, etc.) raw
    is_binary = (X.dtype == np.uint8 or
                 (len(np.unique(X)) <= 2 and
                  set(np.unique(X).tolist()).issubset({0, 0.0, 1, 1.0})))

    for train_idx, val_idx in splitter.split(*split_args):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if is_binary:
            X_train_s = X_train.astype(np.float32)
            X_val_s = X_val.astype(np.float32)
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=1e10, neginf=-1e10)
            X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=1e10, neginf=-1e10)

        m = deepcopy(model)
        # Adaptive k for KNN: scale to sqrt(N_train), capped at 5
        if hasattr(m, 'n_neighbors') and hasattr(m, 'set_params'):
            k = min(5, max(1, int(np.sqrt(len(train_idx)))))
            m.set_params(n_neighbors=k)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_val_s)
        scores.append(metric_fn(y_val, y_pred))

        if secondary_metric_fns:
            for sec_name, sec_fn in secondary_metric_fns.items():
                try:
                    sec_scores[sec_name].append(sec_fn(y_val, y_pred))
                except Exception:
                    pass

    mean, std = float(np.mean(scores)), float(np.std(scores))
    if secondary_metric_fns is None:
        return mean, std

    sec_results = {}
    for sec_name, vals in sec_scores.items():
        if vals:
            sec_results[sec_name] = (float(np.mean(vals)), float(np.std(vals)))
    return mean, std, sec_results


def _evaluate_hybrid_config(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    allocation_method: str,
    budget: int,
    importance_method: str,
    model,
    model_name: str,
    internal_evaluator: str = 'ridge',
    groups: Optional[np.ndarray] = None,
    n_folds: int = 5,
    metric_fn: Callable = None,
    seed: int = 42,
    step_size: int = 5,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    candidate_ranking: str = 'rf_importance',
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
    importance_cache: Optional[Dict] = None,
    alloc_cache: Optional[Dict] = None,
) -> dict:
    """Evaluate a single hybrid configuration. Returns result dict."""
    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    start_time = time.time()

    # Use precomputed folds if available (ensures identical splits with fine-tuned reps)
    if precomputed_folds is not None:
        fold_splits = precomputed_folds
    else:
        splitter, use_groups = _make_cv_splitter(groups, n_folds, seed, cv_splitter=cv_splitter)
        sample_X = next(iter(base_reps.values()))
        split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
        fold_splits = list(splitter.split(*split_args))

    fold_scores = []
    fold_n_features = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Merge fine-tuned reps into train/val reps for this fold
        if fold_reps and fold_idx in fold_reps:
            for rep_name, (ft_train, ft_val) in fold_reps[fold_idx].items():
                train_reps[rep_name] = ft_train
                val_reps[rep_name] = ft_val

        # Split augmentations per fold
        train_augs = None
        val_augs = None
        if augmentations is not None:
            train_augs = {name: X[train_idx] for name, X in augmentations.items()}
            val_augs = {name: X[val_idx] for name, X in augmentations.items()}

        # Look up precomputed importance for this (fold, importance_method)
        fold_importance = None
        if importance_cache is not None:
            fold_importance = importance_cache.get((fold_idx, importance_method))

        # Look up precomputed allocation scores for this (fold, evaluator)
        fold_alloc_scores = None
        if alloc_cache is not None:
            fold_alloc_scores = alloc_cache.get((fold_idx, internal_evaluator))

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                step_size=step_size,
                importance_method=importance_method,
                internal_evaluator=internal_evaluator,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
                augmentation_threshold=augmentation_threshold,
                augmentation_budget=augmentation_budget,
                candidate_ranking=candidate_ranking,
                model=model,
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                precomputed_importance=fold_importance,
                precomputed_alloc_scores=fold_alloc_scores,
            )
            X_val = apply_feature_selection(val_reps, feature_info)

            # Apply augmentation selection to val set
            aug_info = feature_info.get('augmentation_info')
            if aug_info and val_augs is not None:
                aug_val_features = apply_augmentation_selection(val_augs, aug_info)
                if aug_val_features is not None:
                    X_val = np.hstack([X_val, aug_val_features])
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=1e10, neginf=-1e10)

        m = deepcopy(model)
        # Adaptive k for KNN: scale to sqrt(N_train), capped at 5
        if hasattr(m, 'n_neighbors') and hasattr(m, 'set_params'):
            k = min(5, max(1, int(np.sqrt(len(train_idx)))))
            m.set_params(n_neighbors=k)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_val_s)
        fold_scores.append(metric_fn(y_val, y_pred))
        fold_n_features.append(X_train.shape[1])

    if not fold_scores:
        raise ValueError("All folds failed")

    elapsed = time.time() - start_time
    # Include both frozen and fine-tuned rep names
    rep_names = list(base_reps.keys())
    if fold_reps:
        # Add factory rep names from first fold
        first_fold = next(iter(fold_reps.values()), {})
        for name in first_fold:
            if name not in rep_names:
                rep_names.append(name)

    blueprint = Blueprint(
        reps=rep_names,
        allocation_method=allocation_method,
        importance_method=importance_method,
        internal_evaluator=internal_evaluator,
        budget=budget,
        step_size=step_size,
        augmentation_strategy=augmentation_strategy,
    )

    return {
        'blueprint': blueprint,
        'model': model_name,
        'score_mean': float(np.mean(fold_scores)),
        'score_std': float(np.std(fold_scores)),
        'fold_scores': [float(s) for s in fold_scores],
        'n_features': int(np.mean(fold_n_features)),
        'time_seconds': elapsed,
        'allocation_method': allocation_method,
        'importance_method': importance_method,
        'internal_evaluator': internal_evaluator,
        'budget': budget,
        'step_size': step_size,
    }


def _evaluate_baselines(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    models: Dict[str, Any],
    groups: Optional[np.ndarray],
    n_folds: int,
    metric_fn: Callable,
    seed: int = 42,
    verbose: bool = True,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    secondary_metric_fns: Optional[Dict[str, Callable]] = None,
    cv_splitter=None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate full-dimension baselines for each rep x model.

    Factory reps (from fold_reps) are evaluated per-fold since they have
    different train/val embeddings per fold.

    If secondary_metric_fns is provided, stores secondary metrics under
    '_secondary' key per rep: {f'{model}_{metric}': {'mean': ..., 'std': ...}}.
    """
    if verbose:
        print("\nEvaluating full-dimension baselines...")

    baselines = {}

    # Frozen reps: standard global-array evaluation
    for rep_name, X in base_reps.items():
        baselines[rep_name] = {'dimension': X.shape[1]}
        sec_all = {}
        for model_name, model in models.items():
            result = _evaluate_single(
                X, labels, model, groups, n_folds, metric_fn, seed,
                cv_splitter=cv_splitter,
                secondary_metric_fns=secondary_metric_fns,
            )
            if secondary_metric_fns is not None:
                mean_score, _, sec_dict = result
                for sec_name, (sec_mean, sec_std) in sec_dict.items():
                    sec_all[f'{model_name}_{sec_name}'] = {
                        'mean': round(sec_mean, 4), 'std': round(sec_std, 4),
                    }
            else:
                mean_score, _ = result
            baselines[rep_name][model_name] = round(mean_score, 4)

        if sec_all:
            baselines[rep_name]['_secondary'] = sec_all

        if verbose:
            scores_str = ", ".join(
                f"{m}={baselines[rep_name][m]:.3f}" for m in models
            )
            print(f"  {rep_name} (dim={X.shape[1]}): {scores_str}")

    # Factory reps: per-fold evaluation
    if fold_reps and precomputed_folds:
        factory_rep_names = set()
        for fold_data in fold_reps.values():
            factory_rep_names.update(fold_data.keys())

        for rep_name in sorted(factory_rep_names):
            # Get dimension from first fold
            first_fold = next(iter(fold_reps.values()))
            if rep_name not in first_fold:
                continue
            dim = first_fold[rep_name][0].shape[1]
            baselines[rep_name] = {'dimension': dim}
            sec_all = {}

            for model_name, model in models.items():
                fold_scores = []
                fold_sec_scores = {name: [] for name in (secondary_metric_fns or {})}
                for fold_idx, (train_idx, val_idx) in enumerate(precomputed_folds):
                    if fold_idx not in fold_reps or rep_name not in fold_reps[fold_idx]:
                        continue
                    ft_train, ft_val = fold_reps[fold_idx][rep_name]
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(ft_train)
                    X_val_s = scaler.transform(ft_val)

                    m = deepcopy(model)
                    m.fit(X_train_s, y_train)
                    y_pred = m.predict(X_val_s)
                    fold_scores.append(metric_fn(y_val, y_pred))

                    if secondary_metric_fns:
                        for sec_name, sec_fn in secondary_metric_fns.items():
                            try:
                                fold_sec_scores[sec_name].append(sec_fn(y_val, y_pred))
                            except Exception:
                                pass

                if fold_scores:
                    baselines[rep_name][model_name] = round(float(np.mean(fold_scores)), 4)

                if secondary_metric_fns:
                    for sec_name, vals in fold_sec_scores.items():
                        if vals:
                            sec_all[f'{model_name}_{sec_name}'] = {
                                'mean': round(float(np.mean(vals)), 4),
                                'std': round(float(np.std(vals)), 4),
                            }

            if sec_all:
                baselines[rep_name]['_secondary'] = sec_all

            if verbose:
                scores_str = ", ".join(
                    f"{m}={baselines[rep_name].get(m, 'N/A')}" for m in models
                )
                print(f"  {rep_name} (dim={dim}, fine-tuned): {scores_str}")

    return baselines


# =============================================================================
# SEARCH LOOP
# =============================================================================

def _run_search(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    speed: str,
    models: Dict[str, Any],
    metric_fn: Callable,
    budgets: List[int],
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    internal_evaluators: Optional[List[str]] = None,
    n_folds: Optional[int] = None,
    test_rep_subsets: bool = False,
    seed: int = 42,
    verbose: bool = True,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    candidate_ranking: str = 'rf_importance',
    adaptive_pruning: bool = True,
    pruning_threshold: float = 0.5,
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
) -> List[dict]:
    """Main search loop over all configurations."""
    tier = SPEED_TIERS[speed]
    search_importance = importance_methods if importance_methods is not None else tier['importance_methods']
    search_allocation = allocation_methods if allocation_methods is not None else tier['allocation_methods']
    search_evaluators = internal_evaluators if internal_evaluators is not None else tier['internal_evaluators']
    search_budgets = budgets
    search_folds = n_folds if n_folds is not None else tier['n_folds']

    # Resolve 'match' evaluators: for each model, the evaluator IS that model
    use_matched = 'match' in search_evaluators
    explicit_evaluators = [e for e in search_evaluators if e != 'match']

    # Build list of rep sets to test
    rep_sets = [base_reps]
    rep_set_names = ['all']

    if test_rep_subsets and len(base_reps) >= 3:
        for leave_out in base_reps:
            subset = {k: v for k, v in base_reps.items() if k != leave_out}
            rep_sets.append(subset)
            rep_set_names.append(f'no_{leave_out}')

    # Build effective evaluator list per model
    def _evaluators_for_model(model_name):
        """Return list of evaluator names to test for a given downstream model."""
        evs = list(explicit_evaluators)
        if use_matched:
            # Map model name to evaluator name
            match_map = {
                'rf': 'rf', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost',
                'mlp': 'mlp', 'svm': 'svm', 'gp': 'gp', 'ridge': 'ridge',
            }
            matched_ev = match_map.get(model_name)
            if matched_ev and matched_ev not in evs:
                evs.append(matched_ev)
        return evs if evs else ['ridge']

    # Compute effective config count
    total = 0
    for alloc in search_allocation:
        n_imp = 1 if not _is_importance_sensitive(alloc) else len(search_importance)
        for model_name in models:
            n_ev = len(_evaluators_for_model(model_name))
            total += len(rep_sets) * len(search_budgets) * n_imp * n_ev

    if verbose:
        print(f"\nSearching {total} unique configurations ({speed} mode)...")

    # ── Pre-compute importance scores for deduplication ──
    # Importance for a given (rep, fold_train_data, method) is identical across
    # all (alloc, budget, evaluator, model) configs. Compute once, reuse everywhere.
    from kirby.hybrid import compute_feature_importance

    if precomputed_folds is not None:
        _fold_splits_for_cache = precomputed_folds
    else:
        _splitter, _use_groups = _make_cv_splitter(groups, search_folds, seed, cv_splitter=cv_splitter)
        _sample_X = next(iter(base_reps.values()))
        _split_args = (_sample_X, labels, groups) if _use_groups else (_sample_X, labels)
        _fold_splits_for_cache = list(_splitter.split(*_split_args))

    importance_cache = {}  # (fold_idx, imp_method) -> {rep_name: scores_array}
    n_imp_total = len(_fold_splits_for_cache) * len(search_importance)
    if verbose:
        print(f"  Pre-computing importance: {len(search_importance)} methods × "
              f"{len(_fold_splits_for_cache)} folds × {len(base_reps)} reps...")
    _imp_t0 = time.time()
    for fold_idx, (train_idx, _) in enumerate(_fold_splits_for_cache):
        fold_train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        fold_train_labels = labels[train_idx]
        # Merge fine-tuned reps for this fold
        if fold_reps and fold_idx in fold_reps:
            for rep_name, (ft_train, _) in fold_reps[fold_idx].items():
                fold_train_reps[rep_name] = ft_train
        for imp_method in search_importance:
            try:
                importance_cache[(fold_idx, imp_method)] = compute_feature_importance(
                    fold_train_reps, fold_train_labels, method=imp_method)
            except Exception as e:
                if verbose:
                    print(f"    importance {imp_method} fold {fold_idx}: FAILED ({e})")
    if verbose:
        print(f"  Importance pre-computed in {time.time() - _imp_t0:.1f}s "
              f"({len(importance_cache)}/{n_imp_total} successful)")

    # ── Pre-compute allocation baseline scores for deduplication ──
    # allocate_performance_weighted trains the internal evaluator on each rep
    # to score them. For a given (fold, evaluator), these scores are identical
    # across all (budget, importance, model) configs. Compute once, reuse.
    from kirby.hybrid import _evaluate_allocation

    alloc_cache = {}  # (fold_idx, evaluator) -> {rep_name: score}
    _alloc_t0 = time.time()
    for fold_idx, (train_idx, _) in enumerate(_fold_splits_for_cache):
        fold_train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        fold_train_labels = labels[train_idx]
        if fold_reps and fold_idx in fold_reps:
            for rep_name, (ft_train, _) in fold_reps[fold_idx].items():
                fold_train_reps[rep_name] = ft_train

        # Replicate create_hybrid's internal 80/20 split (same RandomState(42))
        n_fold = len(fold_train_labels)
        n_val_internal = max(1, int(n_fold * 0.2))
        _indices = np.random.RandomState(42).permutation(n_fold)
        _tr_idx = _indices[n_val_internal:]
        _va_idx = _indices[:n_val_internal]
        _train_labels = fold_train_labels[_tr_idx]
        _val_labels = fold_train_labels[_va_idx]

        for ev in search_evaluators:
            if ev == 'match':
                continue
            scores = {}
            for rep_name, X_full in fold_train_reps.items():
                X_tr = np.nan_to_num(np.clip(X_full[_tr_idx], -1e10, 1e10),
                                     nan=0.0, posinf=1e10, neginf=-1e10)
                X_va = np.nan_to_num(np.clip(X_full[_va_idx], -1e10, 1e10),
                                     nan=0.0, posinf=1e10, neginf=-1e10)
                try:
                    scores[rep_name] = max(
                        _evaluate_allocation(X_tr, X_va, _train_labels,
                                             _val_labels, internal_evaluator=ev),
                        0.0)
                except Exception:
                    scores[rep_name] = 0.0
            alloc_cache[(fold_idx, ev)] = scores

    if verbose:
        print(f"  Allocation scores pre-computed in {time.time() - _alloc_t0:.1f}s "
              f"({len(alloc_cache)} (fold, evaluator) combos)")

    results = []
    idx = 0

    # Adaptive pruning state — all tracking uses comparable scores (higher=better)
    pruned_models = set()
    model_best_scores = {}  # {model_name: best_comparable}
    pruned_importance = set()
    importance_best_scores = {}  # {imp_method: best_comparable}
    pruned_alloc = set()
    alloc_best_scores = {}  # {alloc_method: best_comparable}
    global_best_comparable = -np.inf
    first_model_sweep_done = False
    first_importance_sweep_done = False
    first_alloc_sweep_done = False

    for rep_set, rep_set_name in zip(rep_sets, rep_set_names):
        for alloc in search_allocation:
            # Importance-insensitive allocs (greedy, dominant_greedy) produce
            # identical results regardless of importance_method — run once.
            imp_sensitive = _is_importance_sensitive(alloc)
            effective_importance = search_importance if imp_sensitive else [search_importance[0]]

            # Skip pruned allocation methods
            if adaptive_pruning and alloc in pruned_alloc:
                skipped_alloc = 0
                for mn in models:
                    skipped_alloc += len(search_budgets) * len(effective_importance) * len(_evaluators_for_model(mn))
                if verbose:
                    print(f"  [skipping {skipped_alloc} configs] alloc={alloc}: PRUNED")
                idx += skipped_alloc
                continue

            for bgt in search_budgets:
                for imp in effective_importance:
                    # Skip pruned importance methods (only relevant for sensitive allocs)
                    if imp_sensitive and adaptive_pruning and imp in pruned_importance:
                        skipped_imp = sum(len(_evaluators_for_model(mn)) for mn in models)
                        if verbose:
                            print(
                                f"  [skipping {skipped_imp} configs] {alloc}/{imp}/b={bgt}: "
                                f"PRUNED (importance)"
                            )
                        idx += skipped_imp
                        continue

                    for model_name, model in models.items():
                        model_evaluators = _evaluators_for_model(model_name)

                        for eval_name in model_evaluators:
                            idx += 1

                            # Skip pruned models
                            if adaptive_pruning and model_name in pruned_models:
                                if verbose:
                                    print(
                                        f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}/ev={eval_name}: "
                                        f"PRUNED"
                                    )
                                continue

                            try:
                                result = _evaluate_hybrid_config(
                                    base_reps=rep_set,
                                    labels=labels,
                                    allocation_method=alloc,
                                    budget=bgt,
                                    importance_method=imp,
                                    model=model,
                                    model_name=model_name,
                                    internal_evaluator=eval_name,
                                    groups=groups,
                                    n_folds=search_folds,
                                    metric_fn=metric_fn,
                                    seed=seed,
                                    augmentations=augmentations,
                                    augmentation_strategy=augmentation_strategy,
                                    augmentation_threshold=augmentation_threshold,
                                    augmentation_budget=augmentation_budget,
                                    candidate_ranking=candidate_ranking,
                                    lower_is_better=lower_is_better,
                                    fold_reps=fold_reps,
                                    precomputed_folds=precomputed_folds,
                                    cv_splitter=cv_splitter,
                                    importance_cache=importance_cache,
                                    alloc_cache=alloc_cache,
                                )
                                if rep_set_name != 'all':
                                    result['rep_subset'] = rep_set_name
                                results.append(result)

                                if verbose:
                                    print(
                                        f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}/ev={eval_name}: "
                                        f"score={result['score_mean']:.4f}+/-{result['score_std']:.3f}, "
                                        f"dim={result['n_features']}"
                                    )

                                # Track comparable scores for pruning
                                if adaptive_pruning:
                                    comp = _score_to_comparable(result['score_mean'], lower_is_better)
                                    if model_name not in model_best_scores or comp > model_best_scores[model_name]:
                                        model_best_scores[model_name] = comp
                                    if imp not in importance_best_scores or comp > importance_best_scores[imp]:
                                        importance_best_scores[imp] = comp
                                    if alloc not in alloc_best_scores or comp > alloc_best_scores[alloc]:
                                        alloc_best_scores[alloc] = comp
                                    if comp > global_best_comparable:
                                        global_best_comparable = comp

                            except Exception as e:
                                if verbose:
                                    print(
                                        f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}/ev={eval_name}: "
                                        f"FAILED - {e}"
                                    )

                    # After each (alloc, bgt, imp) model sweep: prune models
                    if adaptive_pruning and first_model_sweep_done and global_best_comparable > -np.inf:
                        for mn in list(models.keys()):
                            if mn in pruned_models:
                                continue
                            best = model_best_scores.get(mn, -np.inf)
                            if _should_prune(best, global_best_comparable, pruning_threshold):
                                pruned_models.add(mn)
                                if verbose:
                                    print(f"  >> PRUNED model '{mn}' "
                                          f"(best_comparable={best:.4f}, "
                                          f"global_best={global_best_comparable:.4f})")

                    if not first_model_sweep_done and idx >= len(models):
                        first_model_sweep_done = True

                # After each (alloc, bgt) full imp×model sweep: prune importance methods
                if adaptive_pruning and first_importance_sweep_done and global_best_comparable > -np.inf:
                    for im in list(search_importance):
                        if im in pruned_importance:
                            continue
                        best = importance_best_scores.get(im, -np.inf)
                        if _should_prune(best, global_best_comparable, pruning_threshold):
                            pruned_importance.add(im)
                            if verbose:
                                print(f"  >> PRUNED importance '{im}' "
                                      f"(best_comparable={best:.4f}, "
                                      f"global_best={global_best_comparable:.4f})")

                if not first_importance_sweep_done and len(importance_best_scores) >= len(search_importance):
                    first_importance_sweep_done = True

            # After each alloc's full bgt×imp×model sweep: prune allocation methods
            if adaptive_pruning and first_alloc_sweep_done and global_best_comparable > -np.inf:
                for am in list(search_allocation):
                    if am in pruned_alloc:
                        continue
                    best = alloc_best_scores.get(am, -np.inf)
                    if _should_prune(best, global_best_comparable, pruning_threshold):
                        pruned_alloc.add(am)
                        if verbose:
                            print(f"  >> PRUNED allocation '{am}' "
                                  f"(best_comparable={best:.4f}, "
                                  f"global_best={global_best_comparable:.4f})")

            if not first_alloc_sweep_done and len(alloc_best_scores) >= len(search_allocation):
                first_alloc_sweep_done = True

    return results


# =============================================================================
# TWO-PHASE REFINEMENT
# =============================================================================

# Allocation methods where step_size has an effect (greedy-based methods)
_STEP_SIZE_SENSITIVE = {'greedy', 'dominant_greedy'}

# Allocation methods that ignore importance_method (use internal RF feature_importances_)
_IMPORTANCE_INSENSITIVE = {'greedy', 'dominant_greedy'}


def _is_importance_sensitive(alloc_name: str) -> bool:
    """Check if an allocation method actually uses importance_method.

    Greedy and dominant_greedy use internal RF feature_importances_ regardless
    of the importance_method parameter, so iterating over importance methods
    for these allocations produces duplicate results.
    """
    from kirby.hybrid import NAMED_ALLOCATIONS
    base = alloc_name
    if alloc_name in NAMED_ALLOCATIONS:
        base = NAMED_ALLOCATIONS[alloc_name][0]
    return base not in _IMPORTANCE_INSENSITIVE


def _run_refinement_phase(
    phase1_results: List[dict],
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    models: Dict[str, Any],
    metric_fn: Callable,
    n_folds: int,
    seed: int = 42,
    top_n: int = 5,
    step_sizes: Optional[List[int]] = None,
    verbose: bool = True,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    candidate_ranking: str = 'rf_importance',
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
) -> List[dict]:
    """Phase 2: re-evaluate top configs from Phase 1 with alternative step_sizes.

    Filters to greedy-based allocations only (step_size irrelevant for
    performance_weighted/fixed). Deduplicates by (model, alloc, budget, importance)
    before picking top N.

    Returns list of result dicts (same shape as _run_search output).
    """
    from kirby.hybrid import NAMED_ALLOCATIONS

    if step_sizes is None:
        step_sizes = [1, 2, 5, 10, 25]

    # Resolve named allocations to find which are step-size sensitive
    def _is_step_sensitive(alloc_name):
        base = alloc_name
        if alloc_name in NAMED_ALLOCATIONS:
            base = NAMED_ALLOCATIONS[alloc_name][0]
        return base in _STEP_SIZE_SENSITIVE

    # Filter to step-size sensitive configs
    sensitive = [r for r in phase1_results if _is_step_sensitive(r['allocation_method'])]
    if not sensitive:
        if verbose:
            print("  Refinement: no step-size-sensitive allocations to refine.")
        return []

    # Sort by score descending, deduplicate by (model, alloc, budget, importance).
    # Normalize importance for insensitive allocations to avoid duplicate entries.
    sensitive.sort(key=lambda r: r['score_mean'], reverse=True)
    seen = set()
    unique = []
    for r in sensitive:
        dedup_imp = r['importance_method'] if _is_importance_sensitive(r['allocation_method']) else '_any_'
        key = (r['model'], r['allocation_method'], r['budget'], dedup_imp)
        if key not in seen:
            seen.add(key)
            unique.append(r)
            if len(unique) >= top_n:
                break

    # Phase 1 default step_size
    phase1_step = 10

    # Evaluate each top config with alternative step_sizes
    refinement_results = []
    total = len(unique) * len([s for s in step_sizes if s != phase1_step])

    if verbose:
        print(f"\nRefinement phase: {len(unique)} configs x "
              f"{len(step_sizes) - (1 if phase1_step in step_sizes else 0)} step_sizes "
              f"= {total} evaluations")

    idx = 0
    for r in unique:
        model_name = r['model']
        model = models[model_name]
        for ss in step_sizes:
            if ss == r.get('step_size', phase1_step):
                continue  # Already evaluated in phase 1
            idx += 1
            try:
                result = _evaluate_hybrid_config(
                    base_reps=base_reps,
                    labels=labels,
                    allocation_method=r['allocation_method'],
                    budget=r['budget'],
                    importance_method=r['importance_method'],
                    model=model,
                    model_name=model_name,
                    internal_evaluator=r.get('internal_evaluator', 'ridge'),
                    groups=groups,
                    n_folds=n_folds,
                    metric_fn=metric_fn,
                    seed=seed,
                    step_size=ss,
                    augmentations=augmentations,
                    augmentation_strategy=augmentation_strategy,
                    augmentation_threshold=augmentation_threshold,
                    augmentation_budget=augmentation_budget,
                    candidate_ranking=candidate_ranking,
                    fold_reps=fold_reps,
                    precomputed_folds=precomputed_folds,
                    cv_splitter=cv_splitter,
                )
                result['refinement_phase'] = True
                refinement_results.append(result)

                if verbose:
                    print(
                        f"  [refine {idx}/{total}] {model_name}/"
                        f"{r['allocation_method']}/b={r['budget']}/step={ss}: "
                        f"score={result['score_mean']:.4f}+/-{result['score_std']:.3f}"
                    )
            except Exception as e:
                if verbose:
                    print(
                        f"  [refine {idx}/{total}] {model_name}/"
                        f"{r['allocation_method']}/b={r['budget']}/step={ss}: "
                        f"FAILED - {e}"
                    )

    return refinement_results


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def _aggregate_results(
    results: List[dict],
    baselines: Dict[str, Dict[str, float]],
    lower_is_better: bool,
    dimension_tiers: List[int],
) -> dict:
    """Aggregate results into best_overall, best_by_model, best_by_dimension, performance_matrix."""
    if not results:
        return {
            'best_overall': None,
            'best_by_model': {},
            'best_by_dimension': {},
            'performance_matrix': {},
        }

    def _key(r):
        return _score_to_comparable(r['score_mean'], lower_is_better)

    best_overall = max(results, key=_key)

    # Best by model
    model_names = sorted(set(r['model'] for r in results))
    best_by_model = {}
    for mname in model_names:
        model_results = [r for r in results if r['model'] == mname]
        if model_results:
            best_by_model[mname] = max(model_results, key=_key)

    # Best by dimension tier
    best_by_dimension = {}
    for tier in dimension_tiers:
        eligible = [r for r in results if r['n_features'] <= tier]
        if eligible:
            best_by_dimension[tier] = max(eligible, key=_key)

    # Performance matrix: model x dimension
    performance_matrix = {}
    for mname in model_names:
        performance_matrix[mname] = {}
        for tier in dimension_tiers:
            eligible = [
                r for r in results
                if r['model'] == mname and r['n_features'] <= tier
            ]
            if eligible:
                best = max(eligible, key=_key)
                performance_matrix[mname][str(tier)] = round(best['score_mean'], 4)

    return {
        'best_overall': best_overall,
        'best_by_model': best_by_model,
        'best_by_dimension': best_by_dimension,
        'performance_matrix': performance_matrix,
    }


# =============================================================================
# FITTED RESULT
# =============================================================================

def _compute_fitted_result(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    blueprint: Blueprint,
    weight_method: Optional[str] = 'lightgbm_gain',
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    candidate_ranking: str = 'rf_importance',
    finetuned_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> FittedResult:
    """Fit on full data and return concrete feature indices + optional weights."""
    from kirby.hybrid import create_hybrid, compute_feature_importance

    # Use only the reps in the blueprint (frozen + factory)
    reps_to_use = {k: v for k, v in base_reps.items() if k in blueprint.reps}
    if finetuned_embeddings:
        for k, v in finetuned_embeddings.items():
            if k in blueprint.reps:
                reps_to_use[k] = v

    _, feature_info = create_hybrid(
        reps_to_use, labels,
        allocation_method=blueprint.allocation_method,
        budget=blueprint.budget,
        step_size=blueprint.step_size,
        importance_method=blueprint.importance_method,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        candidate_ranking=candidate_ranking,
    )

    selected_indices = {}
    total_features = 0
    for rep_name, info in feature_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        selected_indices[rep_name] = info['selected_indices']
        total_features += len(info['selected_indices'])

    feature_weights = None
    if weight_method is not None:
        try:
            importance = compute_feature_importance(
                reps_to_use, labels, method=weight_method
            )
            feature_weights = {}
            for rep_name, indices in selected_indices.items():
                if rep_name in importance:
                    feature_weights[rep_name] = importance[rep_name][indices]
        except Exception:
            pass

    # Preserve augmentation info if present
    aug_info = feature_info.get('augmentation_info')

    return FittedResult(
        blueprint=blueprint,
        selected_indices=selected_indices,
        feature_weights=feature_weights,
        n_features=total_features,
        augmentation_info=aug_info,
    )


# =============================================================================
# CACHE
# =============================================================================

def _hybrid_cache_key(base_reps, labels, groups, speed, budgets, allocation_methods,
                      importance_methods, internal_evaluators, n_folds, models,
                      seed, metric, task_type,
                      return_fitted=False, weight_method=None,
                      two_phase=False, refinement_top_n=5,
                      refinement_step_sizes=None,
                      rep_factory_names=None,
                      augmentations=None,
                      augmentation_strategy='uncapped') -> str:
    """SHA256 hash of all search-affecting inputs."""
    h = hashlib.sha256()
    for name in sorted(base_reps.keys()):
        arr = base_reps[name]
        h.update(f"{name}:{arr.shape}:{arr.ravel()[:5].tobytes()}".encode())
    h.update(labels.tobytes())
    if groups is not None:
        h.update(groups.tobytes())
    if augmentations is not None:
        for aname in sorted(augmentations.keys()):
            h.update(f"aug:{aname}:{augmentations[aname].shape}".encode())
        h.update(f"aug_strategy:{augmentation_strategy}".encode())
    h.update(f"{speed}|{budgets}|{allocation_methods}|{importance_methods}|{internal_evaluators}|{n_folds}|{seed}|{metric}|{task_type}|{return_fitted}|{weight_method}|{two_phase}|{refinement_top_n}|{refinement_step_sizes}".encode())
    if models:
        model_names = sorted(models.keys()) if isinstance(models, dict) else sorted(models)
        h.update("|".join(model_names).encode())
    if rep_factory_names:
        h.update("|".join(sorted(rep_factory_names)).encode())
    return h.hexdigest()[:16]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def find_best_hybrid(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    budgets: Optional[List[int]] = None,
    models: Optional[Any] = None,
    metric: Union[str, Callable, Tuple[Callable, bool]] = 'r2',
    task_type: str = 'auto',
    speed: str = 'fast',
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    internal_evaluators: Optional[List[str]] = None,
    n_folds: Optional[int] = None,
    test_rep_subsets: bool = False,
    return_fitted: bool = False,
    weight_method: Optional[str] = None,
    dimension_tiers: Optional[List[int]] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    candidate_ranking: str = 'rf_importance',
    adaptive_pruning: bool = True,
    pruning_threshold: float = 0.5,
    two_phase: bool = False,
    refinement_top_n: int = 5,
    refinement_step_sizes: Optional[List[int]] = None,
    rep_factories: Optional[Dict[str, Callable]] = None,
    secondary_metrics: Optional[List[str]] = None,
    cv_splitter=None,
    screen_threshold: Optional[float] = None,
    seed: int = 42,
    verbose: bool = True,
    cache: bool = False,
    cache_dir: Optional[str] = None,
) -> SearchResult:
    """
    Find the best hybrid representation configuration.

    Domain-agnostic: works with any pre-computed feature matrices.

    Args:
        base_reps: Dict of {rep_name: np.ndarray} feature matrices (all same n_samples).
        labels: Target values (n_samples,).
        groups: Group labels for GroupKFold (n_samples,). If None, uses random KFold.
        models: Model specification. None=defaults for speed tier.
            List[str] for named models. Dict[str, model] for custom models.
        metric: Evaluation metric. String ('r2','rmse','mae','accuracy','auc',
            'spearman','pearson','ci','f1_weighted')
            or callable(y_true, y_pred)->float, or (callable, lower_is_better).
        task_type: 'regression', 'classification', or 'auto' (infer from labels).
        speed: Search comprehensiveness: 'fast' or 'full'.
        budgets: List of feature budgets to search over (required).
        allocation_methods: Override allocation methods for the speed tier.
        importance_methods: Override importance methods for the speed tier.
        internal_evaluators: Override internal evaluator list for the speed tier.
            Fast tier default is ['match'] (matched to downstream model).
            Medium/long tiers default to multiple evaluators.
        n_folds: Override number of CV folds for the speed tier.
        test_rep_subsets: Also test leave-one-rep-out subsets (when >=3 reps).
        return_fitted: If True, fit best config on full data and return FittedResult.
        weight_method: Importance method for feature weights (only if return_fitted=True).
        dimension_tiers: Dimension breakpoints for best_by_dimension.
            Defaults to [50, 100, 200, 500].
        augmentations: Dict of {aug_name: np.ndarray} supplementary feature arrays.
            These are small feature sets (e.g. developability, humanness) that augment
            the hybrid after main feature selection. Each is independently evaluated
            and only kept if it improves validation performance.
        augmentation_strategy: How to handle augmentations. 'uncapped' (default)
            adds features beyond budget if they help. 'capped' fills then replaces within
            budget. 'greedy_ablation' tests each rep-level. 'all' includes all. 'none' ignores.
        augmentation_threshold: Minimum improvement to keep an augmentation (default: any).
        augmentation_budget: Max features per augmentation to select (default: 20).
        rep_factories: Dict of {rep_name: factory_callable} for per-fold fine-tuned
            representations. Each factory is called with (train_idx, val_idx, train_labels)
            and returns (train_embeddings, val_embeddings). Fine-tuning happens once per
            fold (NOT per search config). The resulting embeddings are merged with frozen
            base_reps during evaluation. Requires peft + torch for actual fine-tuning
            (but the interface itself is dependency-free).
        secondary_metrics: List of metric names (e.g. ['spearman', 'rmse']) to compute
            alongside the primary metric. Only computed for baselines (not during search
            to avoid overhead). Results stored in baselines[rep]['_secondary'].
        screen_threshold: Optional pre-search rep screening threshold (0.0–1.0).
            After baselines, drops reps whose best single-rep score is more than
            this fraction away from the best rep. E.g. 0.2 means "drop reps more
            than 20% worse than the best". None (default) disables screening.
        seed: Random seed.
        verbose: Print progress.
        cache: If True, cache results to disk and load from cache on subsequent
            calls with identical inputs. Default False.
        cache_dir: Directory for cache files. Defaults to ~/.cache/kirby/hybrid_search.

    Returns:
        SearchResult with best configs, baselines, and optional fitted result.
    """
    # --- Check cache ---
    cache_path = None
    if cache:
        factory_names = sorted(rep_factories.keys()) if rep_factories else None
        key = _hybrid_cache_key(
            base_reps, labels, groups, speed, budgets, allocation_methods,
            importance_methods, internal_evaluators, n_folds, models, seed,
            metric, task_type,
            return_fitted=return_fitted, weight_method=weight_method,
            two_phase=two_phase, refinement_top_n=refinement_top_n,
            refinement_step_sizes=refinement_step_sizes,
            rep_factory_names=factory_names,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
        )
        cache_path = Path(cache_dir or Path.home() / '.cache' / 'kirby' / 'hybrid_search') / f'{key}.pkl'
        if cache_path.exists():
            try:
                result = pickle.loads(cache_path.read_bytes())
                if verbose:
                    print(f"  Loaded cached result from {cache_path}")
                return result
            except Exception:
                pass  # Corrupted cache, recompute

    # --- Validate inputs ---
    if budgets is None:
        import warnings
        warnings.warn(
            "budgets not specified — using default [50, 150, 300]. "
            "Pass budgets= explicitly for reproducible results.",
            stacklevel=2,
        )
        budgets = [50, 150, 300]

    labels = np.asarray(labels).ravel()

    if not base_reps:
        raise ValueError("base_reps cannot be empty")
    total_reps = len(base_reps) + (len(rep_factories) if rep_factories else 0)
    if total_reps < 2:
        raise ValueError("Need at least 2 representations for a hybrid")

    n_samples_list = [X.shape[0] for X in base_reps.values()]
    if len(set(n_samples_list)) > 1:
        raise ValueError(f"All representations must have same n_samples. Got: {n_samples_list}")
    n_samples = n_samples_list[0]

    if len(labels) != n_samples:
        raise ValueError(f"labels length ({len(labels)}) != n_samples ({n_samples})")

    if groups is not None:
        groups = np.asarray(groups).ravel()
        if len(groups) != n_samples:
            raise ValueError(f"groups length ({len(groups)}) != n_samples ({n_samples})")

    if speed not in SPEED_TIERS:
        raise ValueError(f"speed must be one of {list(SPEED_TIERS.keys())}. Got '{speed}'")

    if dimension_tiers is None:
        dimension_tiers = [50, 100, 200, 500]

    # --- Validate augmentations ---
    if augmentations is not None:
        for aug_name, X_aug in augmentations.items():
            if X_aug.shape[0] != n_samples:
                raise ValueError(
                    f"Augmentation '{aug_name}' has {X_aug.shape[0]} samples, "
                    f"expected {n_samples}"
                )
        if verbose:
            aug_dims = {k: v.shape[1] for k, v in augmentations.items()}
            print(f"Augmentations: {aug_dims} (strategy={augmentation_strategy})")

    # --- Infer n_folds from cv_splitter ---
    if cv_splitter is not None and n_folds is None:
        n_folds = cv_splitter.get_n_splits()

    # --- Auto-detect task type ---
    if task_type == 'auto':
        task_type = _detect_task_type(labels)
        if verbose:
            print(f"Auto-detected task type: {task_type}")

    # --- Resolve metric ---
    metric_fn, lower_is_better = _resolve_metric(metric)

    # --- Resolve secondary metrics ---
    resolved_secondary = None
    if secondary_metrics:
        resolved_secondary = {name: _resolve_metric(name)[0] for name in secondary_metrics}
        if verbose:
            print(f"Secondary metrics: {secondary_metrics}")

    # --- Resolve models ---
    resolved_models = _resolve_models(models, task_type, speed,
                                       n_samples=n_samples)
    if verbose:
        print(f"Models: {list(resolved_models.keys())}")
        print(f"Representations: {list(base_reps.keys())} "
              f"(dims: {[X.shape[1] for X in base_reps.values()]})")
        if rep_factories:
            print(f"Fine-tuning factories: {list(rep_factories.keys())}")

    # --- Pre-compute fine-tuned reps ---
    fold_reps = {}
    precomputed_folds = []
    if rep_factories:
        actual_folds_for_ft = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
        fold_reps, precomputed_folds = _precompute_fold_reps(
            rep_factories, labels, groups, actual_folds_for_ft, seed,
            cv_splitter=cv_splitter, verbose=verbose,
        )
    else:
        fold_reps = None
        precomputed_folds = None

    # --- Evaluate baselines ---
    baseline_folds = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
    baselines = _evaluate_baselines(
        base_reps, labels, resolved_models, groups,
        baseline_folds, metric_fn, seed, verbose,
        fold_reps=fold_reps,
        precomputed_folds=precomputed_folds,
        secondary_metric_fns=resolved_secondary,
        cv_splitter=cv_splitter,
    )

    # --- Pre-search rep screening ---
    if screen_threshold is not None:
        base_reps, fold_reps = _screen_reps(
            base_reps, baselines, lower_is_better, screen_threshold,
            fold_reps=fold_reps, verbose=verbose,
        )

    # --- Run search ---
    all_results = _run_search(
        base_reps, labels, groups, speed, resolved_models,
        metric_fn,
        budgets=budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        internal_evaluators=internal_evaluators,
        n_folds=n_folds,
        test_rep_subsets=test_rep_subsets,
        seed=seed,
        verbose=verbose,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        candidate_ranking=candidate_ranking,
        adaptive_pruning=adaptive_pruning,
        pruning_threshold=pruning_threshold,
        lower_is_better=lower_is_better,
        fold_reps=fold_reps,
        precomputed_folds=precomputed_folds,
        cv_splitter=cv_splitter,
    )

    # --- Two-phase refinement ---
    if two_phase and all_results:
        actual_folds_for_refine = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
        refinement_results = _run_refinement_phase(
            phase1_results=all_results,
            base_reps=base_reps,
            labels=labels,
            groups=groups,
            models=resolved_models,
            metric_fn=metric_fn,
            n_folds=actual_folds_for_refine,
            seed=seed,
            top_n=refinement_top_n,
            step_sizes=refinement_step_sizes,
            verbose=verbose,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
            augmentation_threshold=augmentation_threshold,
            augmentation_budget=augmentation_budget,
            candidate_ranking=candidate_ranking,
            fold_reps=fold_reps,
            precomputed_folds=precomputed_folds,
            cv_splitter=cv_splitter,
        )
        all_results.extend(refinement_results)

    # --- Aggregate ---
    agg = _aggregate_results(all_results, baselines, lower_is_better, dimension_tiers)

    # --- Extract full-data factory embeddings ---
    finetuned_embeddings = None
    if rep_factories:
        finetuned_embeddings = _extract_full_embeddings(
            rep_factories, labels, verbose
        )

    # --- Fitted result ---
    fitted = None
    if return_fitted and agg['best_overall'] is not None:
        if verbose:
            print("\nFitting best config on full data...")
        fitted = _compute_fitted_result(
            base_reps, labels, agg['best_overall']['blueprint'], weight_method,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
            augmentation_threshold=augmentation_threshold,
            augmentation_budget=augmentation_budget,
            candidate_ranking=candidate_ranking,
            finetuned_embeddings=finetuned_embeddings,
        )

    # --- Resolve metric name for display ---
    metric_name = metric if isinstance(metric, str) else 'custom'
    actual_folds = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']

    # --- Populate Blueprint reproducibility fields ---
    for r in all_results:
        bp = r['blueprint']
        bp.speed = speed
        bp.n_folds = actual_folds
        bp.seed = seed
        bp.metric = metric_name
        bp.task_type = task_type

    search_result = SearchResult(
        best_overall=agg['best_overall'],
        best_by_model=agg['best_by_model'],
        best_by_dimension=agg['best_by_dimension'],
        performance_matrix=agg['performance_matrix'],
        baselines=baselines,
        all_results=all_results,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        fitted_result=fitted,
        finetuned_embeddings=finetuned_embeddings,
    )

    # --- Save to cache ---
    if cache and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pickle.dumps(search_result))
            if verbose:
                print(f"  Cached result to {cache_path}")
        except Exception:
            pass  # Cache write failure is non-fatal

    return search_result


# =============================================================================
# TODO EXPERIMENTS
# =============================================================================

def find_best_hybrid_with_aggregation(**kwargs):
    """TODO: Test aggregating importance from non-correlated cluster reps.
    Hypothesis: borda/avg of 6 cluster reps may improve feature selection.
    """
    raise NotImplementedError("Aggregation experiment — see benchmark_importance_aggregation.py")


def find_best_hybrid_low_dim(**kwargs):
    """TODO: Optimize for dim 25-50 for downstream similarity/clustering.
    Hypothesis: different methods work best at very low budgets.
    """
    raise NotImplementedError("Low-dim experiment")
