"""
Domain-agnostic hybrid representation search.

Accepts pre-computed numpy feature matrices and finds the best hybrid
configuration across models, budgets, allocation methods, and importance methods.

Works for any domain (molecular, antibody, fMRI, gene expression, etc.) — the
caller is responsible for generating representations and group labels.

Usage:
    from kirby.find_best_hybrid import find_best_hybrid

    result = find_best_hybrid(
        base_reps={'ecfp4': X_ecfp, 'maccs': X_maccs, 'pdv': X_pdv},
        labels=y,
        groups=scaffold_groups,
        speed='fast',
    )
    print(result.best_overall)
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
from sklearn.metrics import (
    accuracy_score,
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
    step_size: int = 10
    speed: str = ''
    n_folds: int = 0
    seed: int = 42
    metric: str = 'r2'
    task_type: str = 'regression'

    def to_dict(self) -> dict:
        return {
            'reps': self.reps,
            'allocation_method': self.allocation_method,
            'importance_method': self.importance_method,
            'budget': self.budget,
            'step_size': self.step_size,
            'speed': self.speed,
            'n_folds': self.n_folds,
            'seed': self.seed,
            'metric': self.metric,
            'task_type': self.task_type,
        }


@dataclass
class FittedResult:
    """Concrete fitted result with feature indices and optional weights."""
    blueprint: Blueprint
    selected_indices: Dict[str, np.ndarray]
    feature_weights: Optional[Dict[str, np.ndarray]] = None
    n_features: int = 0


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


# =============================================================================
# METRIC ABSTRACTION
# =============================================================================

BUILTIN_METRICS = {
    'r2':       (r2_score, False),
    'rmse':     (lambda y, p: np.sqrt(mean_squared_error(y, p)), True),
    'mae':      (mean_absolute_error, True),
    'accuracy': (accuracy_score, False),
    'auc':      (roc_auc_score, False),
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


# =============================================================================
# MODEL RESOLUTION
# =============================================================================

def _get_xgb_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def _get_xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def _get_lgbm_regressor():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def _get_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


DEFAULT_MODELS = {
    'rf': {
        'regressor': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestRegressor']).RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1),
        'classifier': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
        'tier': 'fast',
    },
    'xgboost': {
        'regressor': _get_xgb_regressor,
        'classifier': _get_xgb_classifier,
        'tier': 'fast',
    },
    'ridge': {
        'regressor': lambda: __import__('sklearn.linear_model', fromlist=['Ridge']).Ridge(alpha=1.0),
        'classifier': lambda: __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(
            C=1.0, random_state=42, max_iter=1000, n_jobs=-1),
        'tier': 'medium',
    },
    'knn': {
        'regressor': lambda: __import__('sklearn.neighbors', fromlist=['KNeighborsRegressor']).KNeighborsRegressor(
            n_neighbors=5, n_jobs=-1),
        'classifier': lambda: __import__('sklearn.neighbors', fromlist=['KNeighborsClassifier']).KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1),
        'tier': 'medium',
    },
    'lightgbm': {
        'regressor': _get_lgbm_regressor,
        'classifier': _get_lgbm_classifier,
        'tier': 'long',
    },
    'mlp': {
        'regressor': lambda: __import__('sklearn.neural_network', fromlist=['MLPRegressor']).MLPRegressor(
            hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'classifier': lambda: __import__('sklearn.neural_network', fromlist=['MLPClassifier']).MLPClassifier(
            hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'tier': 'long',
    },
    'svm': {
        'regressor': lambda: __import__('sklearn.svm', fromlist=['SVR']).SVR(),
        'classifier': lambda: __import__('sklearn.svm', fromlist=['SVC']).SVC(),
        'tier': 'long',
    },
}

# Tier ordering for filtering
_TIER_ORDER = {'fast': 0, 'medium': 1, 'long': 2}


def _resolve_models(models, task_type: str, speed: str) -> Dict[str, Any]:
    """Resolve model spec to dict of {name: sklearn_model_instance}.

    models can be:
    - None: use defaults filtered by speed tier
    - List[str]: lookup by name
    - Dict[str, model]: custom models (passthrough)
    - Single model instance: wrap as {'custom': model}
    """
    key = 'classifier' if task_type == 'classification' else 'regressor'
    max_tier = _TIER_ORDER.get(speed, 2)

    if models is None:
        resolved = {}
        for name, spec in DEFAULT_MODELS.items():
            if _TIER_ORDER.get(spec['tier'], 0) <= max_tier:
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


def _detect_task_type(labels: np.ndarray) -> str:
    """Auto-detect regression vs classification from labels."""
    unique = np.unique(labels)
    if len(unique) <= 20 and np.allclose(labels, labels.astype(int)):
        return 'classification'
    return 'regression'


# =============================================================================
# SPEED TIER CONFIGURATION
# =============================================================================

SPEED_TIERS = {
    'fast': {
        'importance_methods': ['treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift'],
        'allocation_methods': ['greedy'],
        'budgets': [100],
        'n_folds': 3,
    },
    'medium': {
        'importance_methods': [
            'treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift',
            'permutation', 'kernelshap',
        ],
        'allocation_methods': ['greedy', 'performance_weighted', 'softmax_T5'],
        'budgets': [50, 100, 200],
        'n_folds': 5,
    },
    'long': {
        'importance_methods': [
            'treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift',
            'permutation', 'kernelshap',
            'integrated_gradients', 'gradientshap',
        ],
        'allocation_methods': [
            'greedy', 'performance_weighted', 'softmax_T5', 'dominant_greedy',
            'two_stage', 'stability_selection',
        ],
        'budgets': [50, 100, 200, 500],
        'n_folds': 5,
    },
}


# =============================================================================
# INTERNAL EVALUATION FUNCTIONS
# =============================================================================

def _make_cv_splitter(groups: Optional[np.ndarray], n_folds: int, seed: int):
    """Return (splitter, use_groups) depending on whether groups are provided."""
    if n_folds == 1:
        # Single 80/20 split (intermediary mode)
        if groups is not None:
            return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), True
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), False
    if groups is not None:
        n_unique = len(np.unique(groups))
        if n_unique >= n_folds:
            return GroupKFold(n_splits=n_folds), True
        # Not enough groups for GroupKFold — fall back to regular KFold
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed), False


def _evaluate_single(
    X: np.ndarray,
    y: np.ndarray,
    model,
    groups: Optional[np.ndarray],
    n_folds: int,
    metric_fn: Callable,
    seed: int = 42,
) -> Tuple[float, float]:
    """Evaluate one (X, model) with CV. Returns (mean_score, std_score) in raw metric units."""
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed)
    split_args = (X, y, groups) if use_groups else (X, y)
    scores = []

    for train_idx, val_idx in splitter.split(*split_args):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        m = deepcopy(model)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_val_s)
        scores.append(metric_fn(y_val, y_pred))

    return float(np.mean(scores)), float(np.std(scores))


def _evaluate_hybrid_config(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    allocation_method: str,
    budget: int,
    importance_method: str,
    model,
    model_name: str,
    groups: Optional[np.ndarray],
    n_folds: int,
    metric_fn: Callable,
    seed: int = 42,
    step_size: int = 10,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'greedy_ablation',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
) -> dict:
    """Evaluate a single hybrid configuration. Returns result dict."""
    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    start_time = time.time()
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed)
    sample_X = next(iter(base_reps.values()))
    split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)

    fold_scores = []
    fold_n_features = []

    for train_idx, val_idx in splitter.split(*split_args):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Split augmentations per fold
        train_augs = None
        val_augs = None
        if augmentations is not None:
            train_augs = {name: X[train_idx] for name, X in augmentations.items()}
            val_augs = {name: X[val_idx] for name, X in augmentations.items()}

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                step_size=step_size,
                importance_method=importance_method,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
                augmentation_threshold=augmentation_threshold,
                augmentation_budget=augmentation_budget,
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

        m = deepcopy(model)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_val_s)
        fold_scores.append(metric_fn(y_val, y_pred))
        fold_n_features.append(X_train.shape[1])

    if not fold_scores:
        raise ValueError("All folds failed")

    elapsed = time.time() - start_time
    rep_names = list(base_reps.keys())

    blueprint = Blueprint(
        reps=rep_names,
        allocation_method=allocation_method,
        importance_method=importance_method,
        budget=budget,
        step_size=step_size,
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
) -> Dict[str, Dict[str, float]]:
    """Evaluate full-dimension baselines for each rep x model."""
    if verbose:
        print("\nEvaluating full-dimension baselines...")

    baselines = {}

    for rep_name, X in base_reps.items():
        baselines[rep_name] = {'dimension': X.shape[1]}
        for model_name, model in models.items():
            mean_score, _ = _evaluate_single(
                X, labels, model, groups, n_folds, metric_fn, seed
            )
            baselines[rep_name][model_name] = round(mean_score, 4)

        if verbose:
            scores_str = ", ".join(
                f"{m}={baselines[rep_name][m]:.3f}" for m in models
            )
            print(f"  {rep_name} (dim={X.shape[1]}): {scores_str}")

    # All concatenated
    X_concat = np.hstack([base_reps[k] for k in sorted(base_reps.keys())])
    baselines['all_concat'] = {'dimension': X_concat.shape[1]}
    for model_name, model in models.items():
        mean_score, _ = _evaluate_single(
            X_concat, labels, model, groups, n_folds, metric_fn, seed
        )
        baselines['all_concat'][model_name] = round(mean_score, 4)

    if verbose:
        scores_str = ", ".join(
            f"{m}={baselines['all_concat'][m]:.3f}" for m in models
        )
        print(f"  all_concat (dim={X_concat.shape[1]}): {scores_str}")

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
    budgets: Optional[List[int]] = None,
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    n_folds: Optional[int] = None,
    test_rep_subsets: bool = False,
    seed: int = 42,
    verbose: bool = True,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'greedy_ablation',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    adaptive_pruning: bool = False,
    pruning_threshold: float = 0.5,
) -> List[dict]:
    """Main search loop over all configurations."""
    tier = SPEED_TIERS[speed]
    search_importance = importance_methods if importance_methods is not None else tier['importance_methods']
    search_allocation = allocation_methods if allocation_methods is not None else tier['allocation_methods']
    search_budgets = budgets if budgets is not None else tier['budgets']
    search_folds = n_folds if n_folds is not None else tier['n_folds']

    # Build list of rep sets to test
    rep_sets = [base_reps]
    rep_set_names = ['all']

    if test_rep_subsets and len(base_reps) >= 3:
        for leave_out in base_reps:
            subset = {k: v for k, v in base_reps.items() if k != leave_out}
            rep_sets.append(subset)
            rep_set_names.append(f'no_{leave_out}')

    total = (
        len(rep_sets) * len(search_allocation) * len(search_budgets)
        * len(search_importance) * len(models)
    )

    if verbose:
        print(f"\nSearching {total} configurations ({speed} mode)...")

    results = []
    idx = 0

    # Adaptive pruning state
    pruned_models = set()
    model_best_scores = {}  # {model_name: best_score_so_far}
    global_best_score = -np.inf
    first_sweep_done = False

    for rep_set, rep_set_name in zip(rep_sets, rep_set_names):
        for alloc in search_allocation:
            for bgt in search_budgets:
                for imp in search_importance:
                    for model_name, model in models.items():
                        idx += 1

                        # Skip pruned models
                        if adaptive_pruning and model_name in pruned_models:
                            if verbose:
                                print(
                                    f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}: "
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
                                groups=groups,
                                n_folds=search_folds,
                                metric_fn=metric_fn,
                                seed=seed,
                                augmentations=augmentations,
                                augmentation_strategy=augmentation_strategy,
                                augmentation_threshold=augmentation_threshold,
                                augmentation_budget=augmentation_budget,
                            )
                            if rep_set_name != 'all':
                                result['rep_subset'] = rep_set_name
                            results.append(result)

                            if verbose:
                                print(
                                    f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}: "
                                    f"score={result['score_mean']:.4f}+/-{result['score_std']:.3f}, "
                                    f"dim={result['n_features']}"
                                )

                            # Track scores for pruning
                            if adaptive_pruning:
                                score = result['score_mean']
                                if model_name not in model_best_scores or score > model_best_scores[model_name]:
                                    model_best_scores[model_name] = score
                                if score > global_best_score:
                                    global_best_score = score

                        except Exception as e:
                            if verbose:
                                print(
                                    f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}: "
                                    f"FAILED - {e}"
                                )

                    # After each (alloc, budget, importance) sweep: prune underperformers
                    if adaptive_pruning and first_sweep_done and global_best_score > 0:
                        for mn in list(models.keys()):
                            if mn in pruned_models:
                                continue
                            best = model_best_scores.get(mn, -np.inf)
                            if best < pruning_threshold * global_best_score:
                                pruned_models.add(mn)
                                if verbose:
                                    print(f"  >> Pruning {mn} (best={best:.4f} < "
                                          f"{pruning_threshold}*{global_best_score:.4f})")

                # Mark first sweep complete after all models have had at least one chance
                if not first_sweep_done and idx >= len(models):
                    first_sweep_done = True

    return results


# =============================================================================
# TWO-PHASE REFINEMENT
# =============================================================================

# Allocation methods where step_size has an effect (greedy-based methods)
_STEP_SIZE_SENSITIVE = {'greedy', 'dominant_greedy', 'greedy_feature'}


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
    augmentation_strategy: str = 'greedy_ablation',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
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

    # Sort by score descending, deduplicate by (model, alloc, budget, importance)
    sensitive.sort(key=lambda r: r['score_mean'], reverse=True)
    seen = set()
    unique = []
    for r in sensitive:
        key = (r['model'], r['allocation_method'], r['budget'], r['importance_method'])
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
                    groups=groups,
                    n_folds=n_folds,
                    metric_fn=metric_fn,
                    seed=seed,
                    step_size=ss,
                    augmentations=augmentations,
                    augmentation_strategy=augmentation_strategy,
                    augmentation_threshold=augmentation_threshold,
                    augmentation_budget=augmentation_budget,
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
    augmentation_strategy: str = 'greedy_ablation',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
) -> FittedResult:
    """Fit on full data and return concrete feature indices + optional weights."""
    from kirby.hybrid import create_hybrid, compute_feature_importance

    # Use only the reps in the blueprint
    reps_to_use = {k: v for k, v in base_reps.items() if k in blueprint.reps}

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

    return FittedResult(
        blueprint=blueprint,
        selected_indices=selected_indices,
        feature_weights=feature_weights,
        n_features=total_features,
    )


# =============================================================================
# CACHE
# =============================================================================

def _hybrid_cache_key(base_reps, labels, groups, speed, budgets, allocation_methods,
                      importance_methods, n_folds, models, seed, metric, task_type,
                      return_fitted=False, weight_method=None,
                      two_phase=False, refinement_top_n=5,
                      refinement_step_sizes=None) -> str:
    """SHA256 hash of all search-affecting inputs."""
    h = hashlib.sha256()
    for name in sorted(base_reps.keys()):
        arr = base_reps[name]
        h.update(f"{name}:{arr.shape}:{arr.ravel()[:5].tobytes()}".encode())
    h.update(labels.tobytes())
    if groups is not None:
        h.update(groups.tobytes())
    h.update(f"{speed}|{budgets}|{allocation_methods}|{importance_methods}|{n_folds}|{seed}|{metric}|{task_type}|{return_fitted}|{weight_method}|{two_phase}|{refinement_top_n}|{refinement_step_sizes}".encode())
    if models:
        model_names = sorted(models.keys()) if isinstance(models, dict) else sorted(models)
        h.update("|".join(model_names).encode())
    return h.hexdigest()[:16]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def find_best_hybrid(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    models: Optional[Any] = None,
    metric: Union[str, Callable, Tuple[Callable, bool]] = 'r2',
    task_type: str = 'auto',
    speed: str = 'fast',
    budgets: Optional[List[int]] = None,
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    n_folds: Optional[int] = None,
    test_rep_subsets: bool = False,
    return_fitted: bool = False,
    weight_method: Optional[str] = None,
    dimension_tiers: Optional[List[int]] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'greedy_ablation',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    adaptive_pruning: bool = False,
    pruning_threshold: float = 0.5,
    two_phase: bool = False,
    refinement_top_n: int = 5,
    refinement_step_sizes: Optional[List[int]] = None,
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
        metric: Evaluation metric. String ('r2','rmse','mae','accuracy','auc')
            or callable(y_true, y_pred)->float, or (callable, lower_is_better).
        task_type: 'regression', 'classification', or 'auto' (infer from labels).
        speed: Search comprehensiveness: 'fast', 'medium', 'long'.
        budgets: Override default budgets for the speed tier.
        allocation_methods: Override allocation methods for the speed tier.
        importance_methods: Override importance methods for the speed tier.
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
        augmentation_strategy: How to handle augmentations. 'greedy_ablation' (default)
            tests each and keeps only those that help. 'all' includes all. 'none' ignores.
        augmentation_threshold: Minimum improvement to keep an augmentation (default: any).
        augmentation_budget: Max features per augmentation to select (default: 20).
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
        key = _hybrid_cache_key(
            base_reps, labels, groups, speed, budgets, allocation_methods,
            importance_methods, n_folds, models, seed, metric, task_type,
            return_fitted=return_fitted, weight_method=weight_method,
            two_phase=two_phase, refinement_top_n=refinement_top_n,
            refinement_step_sizes=refinement_step_sizes,
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
    labels = np.asarray(labels).ravel()

    if not base_reps:
        raise ValueError("base_reps cannot be empty")
    if len(base_reps) < 2:
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

    # --- Auto-detect task type ---
    if task_type == 'auto':
        task_type = _detect_task_type(labels)
        if verbose:
            print(f"Auto-detected task type: {task_type}")

    # --- Resolve metric ---
    metric_fn, lower_is_better = _resolve_metric(metric)

    # --- Resolve models ---
    resolved_models = _resolve_models(models, task_type, speed)
    if verbose:
        print(f"Models: {list(resolved_models.keys())}")
        print(f"Representations: {list(base_reps.keys())} "
              f"(dims: {[X.shape[1] for X in base_reps.values()]})")

    # --- Evaluate baselines ---
    baseline_folds = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
    baselines = _evaluate_baselines(
        base_reps, labels, resolved_models, groups,
        baseline_folds, metric_fn, seed, verbose
    )

    # --- Run search ---
    all_results = _run_search(
        base_reps, labels, groups, speed, resolved_models,
        metric_fn,
        budgets=budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        n_folds=n_folds,
        test_rep_subsets=test_rep_subsets,
        seed=seed,
        verbose=verbose,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        adaptive_pruning=adaptive_pruning,
        pruning_threshold=pruning_threshold,
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
        )
        all_results.extend(refinement_results)

    # --- Aggregate ---
    agg = _aggregate_results(all_results, baselines, lower_is_better, dimension_tiers)

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
