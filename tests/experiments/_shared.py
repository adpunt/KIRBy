"""Shared utilities for complexity and similarity experiment scripts.

Extracted from complexity_theory_experiments.py and similarity_experiments.py.
Provides mode configs, CV helpers, model factories, metrics, data loading, and IO.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import tempfile
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma
from scipy.stats import spearmanr, f as f_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.find_best_hybrid import find_best_hybrid, SearchResult
from kirby.registry import DATASET_REGISTRY, lookup as registry_lookup

warnings.filterwarnings('ignore')


# =============================================================================
# FOLD-AWARE HYBRID CACHING
# =============================================================================

import hashlib
import pickle

def _hybrid_fold_cache_dir() -> Path:
    """Return cache directory for per-fold hybrid feature_info."""
    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby')))
    d = cache_root / 'hybrid_fold_cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hybrid_fold_cache_key(train_idx, rep_shapes, allocation_method,
                           importance_method, budget, internal_evaluator,
                           augmentation_names, augmentation_strategy,
                           step_size=5, patience=20, greedy_tolerance=0.0):
    """Hash fold + hybrid config into a cache key."""
    h = hashlib.sha256()
    h.update(train_idx.tobytes())
    for name, shape in sorted(rep_shapes.items()):
        h.update(f"{name}:{shape}".encode())
    h.update(f"{allocation_method}|{importance_method}|{budget}|{internal_evaluator}|{step_size}|{patience}|{greedy_tolerance}".encode())
    h.update(f"aug_strategy:{augmentation_strategy}".encode())
    if augmentation_names:
        h.update(f"aug:{'|'.join(sorted(augmentation_names))}".encode())
    return h.hexdigest()[:24]


def create_hybrid_cached(train_reps, y_train, train_idx,
                         allocation_method='top_3',
                         importance_method='lightgbm_gain',
                         budget=100,
                         internal_evaluator='svm',
                         augmentations=None,
                         augmentation_strategy='uncapped',
                         **kwargs):
    """Fold-aware cached create_hybrid.

    Caches feature_info (selected indices) keyed by fold train indices +
    hybrid config. The actual X_hybrid is reconstructed from feature_info
    since it's just index selection on the rep arrays.

    Args:
        train_reps: {rep_name: train_features} for this fold.
        y_train: Training labels for this fold.
        train_idx: np.ndarray of training sample indices (used for cache key).
        Other args: same as create_hybrid.

    Returns:
        (X_hybrid, feature_info) — same as create_hybrid.
    """
    rep_shapes = {name: X.shape for name, X in train_reps.items()}
    aug_names = sorted(augmentations.keys()) if augmentations else None

    key = _hybrid_fold_cache_key(
        train_idx, rep_shapes, allocation_method, importance_method,
        budget, internal_evaluator, aug_names, augmentation_strategy,
        step_size=kwargs.get('step_size', 5),
        patience=kwargs.get('patience', 20),
        greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
    )
    cache_dir = _hybrid_fold_cache_dir()
    cache_path = cache_dir / f'{key}.pkl'

    if cache_path.exists():
        try:
            feature_info = pickle.loads(cache_path.read_bytes())
            X_hybrid = apply_feature_selection(train_reps, feature_info)
            # Reconstruct augmentation features if present
            aug_info = feature_info.get('augmentation_info')
            if aug_info and augmentations:
                from kirby.hybrid import apply_augmentation_selection
                X_aug = apply_augmentation_selection(augmentations, aug_info)
                if X_aug is not None:
                    X_hybrid = np.hstack([X_hybrid, X_aug])
            return X_hybrid, feature_info
        except Exception:
            pass  # corrupted, recompute

    X_hybrid, feature_info = create_hybrid(
        train_reps, y_train,
        allocation_method=allocation_method,
        importance_method=importance_method,
        budget=budget,
        internal_evaluator=internal_evaluator,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        **kwargs,
    )

    try:
        fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix='.tmp')
        fd_closed = False
        try:
            os.write(fd, pickle.dumps(feature_info))
            os.close(fd)
            fd_closed = True
            os.replace(tmp, str(cache_path))
        except Exception:
            if not fd_closed:
                os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        pass  # non-fatal

    return X_hybrid, feature_info


# =============================================================================
# REPRESENTATION CACHING
# =============================================================================

def _rep_cache_dir() -> Path:
    """Return cache directory for pre-computed representations."""
    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby')))
    d = cache_root / 'rep_cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_rep(dataset_name: str, rep_name: str, compute_fn) -> np.ndarray:
    """Load rep from cache if available, otherwise compute and save."""
    cache_dir = _rep_cache_dir() / dataset_name
    cache_file = cache_dir / f'{rep_name}.npy'

    if cache_file.exists():
        arr = np.load(cache_file)
        print(f"    {rep_name}: cached ({arr.shape[1]}D)")
        return arr

    arr = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_file.with_suffix('.tmp.npy')
    try:
        np.save(str(tmp_path), arr)
        os.replace(str(tmp_path), str(cache_file))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    print(f"    {rep_name}: computed + cached ({arr.shape[1]}D)")
    return arr


def _cache_rep_dict(dataset_name: str, compute_all_fn,
                    filter_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Cache a function that returns {rep_name: array} dict.

    Loads individual reps from cache. If any are missing, calls compute_all_fn
    once and caches everything it returns.
    """
    cache_dir = _rep_cache_dir() / dataset_name
    reps = {}
    missing = []

    # First pass: load what's cached
    if filter_names:
        wanted = filter_names
    else:
        # If no filter, check what's cached and what compute_all_fn would produce
        # We don't know the keys without calling it, so just call it
        wanted = None

    if wanted is not None:
        for rn in wanted:
            cache_file = cache_dir / f'{rn}.npy'
            if cache_file.exists():
                reps[rn] = np.load(cache_file)
                print(f"    {rn}: cached ({reps[rn].shape[1]}D)")
            else:
                missing.append(rn)

        if not missing:
            return reps

    # Compute all, cache each, return wanted
    print(f"    Computing representations...")
    all_reps = compute_all_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for k, v in all_reps.items():
        cache_file = cache_dir / f'{k}.npy'
        if not cache_file.exists():
            tmp_path = cache_file.with_suffix('.tmp.npy')
            try:
                np.save(str(tmp_path), v)
                os.replace(str(tmp_path), str(cache_file))
            except Exception:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise
        print(f"    {k}: computed + cached ({v.shape[1]}D)")

    if filter_names:
        return {k: v for k, v in all_reps.items() if k in filter_names}
    return all_reps


# =============================================================================
# MODE CONFIGURATIONS — derived from SPEED_TIERS in find_best_hybrid.py
# =============================================================================
# No standalone config dicts here. Experiments use SPEED_TIERS directly.
# The only experiment-level configs are: n_splits, models, budgets.
# These are set per-experiment or via get_config(args).

# Canonical budget lists — used by get_config() and all 4 helpers' run_all().
# Change here to change everywhere.
SCREENING_BUDGETS = [50, 500, 1000, 2000]
FULL_BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000]
DISTANCE_BUDGETS = [10, 25, 50, 100, 250, 500]
DISTANCE_EVALUATORS = ['knn', 'ridge', 'rf', 'mlp', 'lightgbm', 'svm']

# Core 6 downstream models (lit-validated across molecular, antibody, DTA, TCGA, reaction).
# RF + XGBoost (tree ensembles), Ridge (linear probe), SVM (kernel), MLP (neural), LightGBM (boosting).
DOWNSTREAM_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'mlp', 'lightgbm']


# Pruning modes — single source of truth for adaptive-pruning configuration.
# Used by both tests/precompute_hybrids.py (precompute) and the experiment
# helpers (run_all in complexity/similarity/noise).
#
# Each mode bundles:
#   - schedule: global multi-stage schedule applied to every model.
#   - model_overrides: per-model schedule overrides (or None).
#   - min_samples_for_overrides: dataset n_samples threshold below which
#     model_overrides are dropped. Mid-flight pruning on small datasets
#     risks dropping the actual winner because fold-0 noise dominates;
#     the gate prevents this.
#
# Calibration source: f16_multistage_pruning sweep on f15 data (2026-04-26).
# For caco2, herg_ki, logd, buchwald_hartwig (n>=2000), td=0.5 at ckpt=[50]
# preserves best_by_model[m] within 2% for SVM/MLP while saving ~48% of
# those models' fold-evals. For flab_pk_jain_fcrnrt (n=132) and abagym_EGFR,
# MLP/Ridge fail at td>=0.3 — hence the n_samples gate at 1000.
PRUNING_MODES = {
    'default_pruning': {
        'description': (
            'Existing single-stage strategy pruning (adaptive_pruning=0.15) '
            'only. No mid-flight pruning on any model. Bit-identical to '
            'pre-Phase-2 behaviour and to f15-validated default.'
        ),
        'schedule': {
            'checkpoints': [],
            'tail_drop_per_checkpoint': 0.0,
            'final_strategy_threshold': 0.15,
            'min_configs_per_model': 1,
        },
        'model_overrides': None,
        'min_samples_for_overrides': 0,
    },
    'aggressive_pruning': {
        'description': (
            'F16-validated aggressive schedule applied globally to all '
            'models with no n_samples gate: checkpoints=[25, 50, 100], '
            'tail_drop=0.5. F16 calibration (caco2 n=2158, 2026-04-25): '
            '58.0% eval reduction, top-3 overlap=1.0, worst per-model '
            'relative loss=0%. Replaces prior conservative setting '
            '(checkpoints=[50], gated at n>=1000 for gp/svm/mlp only) '
            'because that gate produced zero pruning on antibody/small '
            'datasets where it was most needed.'
        ),
        'schedule': {
            'checkpoints': [25, 50, 100],
            'tail_drop_per_checkpoint': 0.5,
            'final_strategy_threshold': 0.15,
            'min_configs_per_model': 1,
        },
        'model_overrides': None,
        'min_samples_for_overrides': 0,
    },
}


def resolve_pruning_mode(mode_name, override_checkpoints=None,
                         override_tail_drop=None, override_min_samples=None):
    """Return (schedule, model_overrides_template, min_samples_gate).

    `model_overrides_template` is the unconditional per-model dict; the
    n_samples gate is applied by the caller once data is loaded.
    """
    if mode_name not in PRUNING_MODES:
        raise ValueError(f"Unknown pruning mode '{mode_name}'. "
                         f"Choose from {list(PRUNING_MODES.keys())}")
    mode = PRUNING_MODES[mode_name]
    schedule = dict(mode['schedule'])
    model_overrides = (
        {k: dict(v) for k, v in mode['model_overrides'].items()}
        if mode['model_overrides'] else None
    )
    min_samples = int(mode['min_samples_for_overrides'])
    if model_overrides is not None:
        if override_checkpoints is not None:
            for m in model_overrides:
                model_overrides[m]['checkpoints'] = list(override_checkpoints)
        if override_tail_drop is not None:
            for m in model_overrides:
                model_overrides[m]['tail_drop_per_checkpoint'] = float(
                    override_tail_drop)
    if override_min_samples is not None:
        min_samples = int(override_min_samples)
    return schedule, model_overrides, min_samples


def apply_pruning_gate(model_overrides, n_samples, min_samples_gate):
    """Return model_overrides if n_samples >= gate else None.

    Centralized gate logic so precompute and experiment runners agree.
    """
    if model_overrides is None:
        return None
    if n_samples < min_samples_gate:
        return None
    return model_overrides


def resolve_pruning_kwargs(pruning_mode, n_samples, verbose=True,
                           dataset_label=None):
    """Convenience helper for scripts that call find_best_hybrid directly.

    Resolves the pruning_mode, applies the n_samples gate, and returns the
    kwargs dict ready to splat:

        kwargs = resolve_pruning_kwargs('aggressive_pruning', len(labels))
        result = find_best_hybrid(..., **kwargs)

    Returns a dict with two keys:
        adaptive_pruning_schedule: schedule dict
        model_pruning_aggressiveness: per-model overrides (or None after gate)

    When verbose=True, prints a single line indicating which mode is active
    and whether the gate fired.
    """
    schedule, ovr_template, min_n = resolve_pruning_mode(pruning_mode)
    overrides = apply_pruning_gate(ovr_template, n_samples, min_n)
    if verbose:
        prefix = f"  [{dataset_label}] " if dataset_label else "  "
        if overrides is not None:
            print(f"{prefix}pruning_mode={pruning_mode}, "
                  f"overrides active for {sorted(overrides.keys())}")
        elif ovr_template is not None:
            print(f"{prefix}pruning_mode={pruning_mode}, "
                  f"overrides gated off (n={n_samples} < {min_n})")
        else:
            print(f"{prefix}pruning_mode={pruning_mode} (no per-model overrides)")
    return {
        'adaptive_pruning_schedule': schedule,
        'model_pruning_aggressiveness': overrides,
    }

# Downstream models for noise experiments. Mirrors DOWNSTREAM_MODELS — no
# tier='extra' additions, so iteration cache-hits the precompute model set.
#
# 2026-05-22: ngboost removed. It is tier='extra' and not in precompute's
# default model list, so including it here caused ngboost to be trained from
# scratch on every (rep × LNL method × noise type) combo during noise
# experiments — burning hours per dataset across the 23-dataset done set.
# n08 (estimation survey), n11 (estimation×remediation grid) and n13
# (alt distributions) still use ngboost as an *estimator* (literature
# methods, not through find_best_hybrid); n13 already gates via
# get_noise_estimator import (line 47-48). s11 uses ngboost only as a BO
# surrogate when explicitly requested. None of these were affected by this
# change — only the implicit downstream iteration was killed.
DOWNSTREAM_MODELS_NOISE = list(DOWNSTREAM_MODELS)

# Generic internal evaluators for hybrid construction.
# 'match' resolves to the downstream model type; the rest test robustness
# across inductive biases: tree-ensemble (rf), kernel (svm), boosted (lightgbm), neural (mlp).
INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp']


def extract_top3_base_reps(search, model='rf'):
    """Extract top 3 base reps ranked by a single model from a SearchResult.

    Args:
        search: SearchResult from find_best_hybrid (must have .baselines).
        model: Model name to rank by (default 'rf').

    Returns:
        List of (rep_name, score) tuples, sorted best-first, length <= 3.
    """
    baselines = getattr(search, 'baselines', {})
    lower = getattr(search, 'lower_is_better', False)
    scores = []
    for rep_name, model_scores in baselines.items():
        if rep_name.startswith('_'):
            continue  # skip _secondary etc.
        s = model_scores.get(model)
        if s is not None and not np.isnan(s):
            scores.append((rep_name, float(s)))
    scores.sort(key=lambda x: x[1], reverse=not lower)
    return scores[:3]


# =============================================================================
# REP TIER CLASSIFICATION
# =============================================================================

# Per-rep tier overrides (exceptions to pure dimensionality classification)
REP_TIER_OVERRIDES = {
    'aac': 2,  # 40D but bumped to tier 2 per user decision
}


def classify_rep_tier(dim: int, rep_name: str = None) -> int:
    """Classify a representation into a tier based on dimensionality.

    Tier 1 (<50D):   augmentation-only, NEVER standalone baselines
    Tier 2 (50-200D): both standalone baselines and augmentation candidates
    Tier 3 (>200D):  base reps, standalone baselines only

    Per-rep exceptions are handled via REP_TIER_OVERRIDES.
    Matches both exact names and domain-stripped suffixes (e.g. protein_aac → aac).
    """
    if rep_name:
        if rep_name in REP_TIER_OVERRIDES:
            return REP_TIER_OVERRIDES[rep_name]
        # Strip domain prefix (drug_, protein_) for DTA reps
        for prefix in ('drug_', 'protein_'):
            if rep_name.startswith(prefix):
                suffix = rep_name[len(prefix):]
                if suffix in REP_TIER_OVERRIDES:
                    return REP_TIER_OVERRIDES[suffix]
                break
    if dim < 50:
        return 1
    elif dim <= 200:
        return 2
    else:
        return 3


# =============================================================================
# CV SPLITTING
# =============================================================================

def cv_split(groups, n_splits, X, labels, seed=42, mode='intermediary'):
    """Yield (train_idx, val_idx) using appropriate CV strategy.

    - n_splits=1 (any mode): single 80/20 GroupShuffleSplit/ShuffleSplit
    - n_splits>1: GroupKFold (or KFold if not enough groups)
    """
    if n_splits == 1:
        if groups is not None and len(np.unique(groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            yield from gss.split(X, labels, groups=groups)
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            yield from ss.split(X, labels)
    else:
        if groups is not None and len(np.unique(groups)) >= n_splits:
            # Use StratifiedGroupKFold for classification (preserves class balance
            # within site-grouped folds). Falls back to GroupKFold if unavailable.
            if labels is not None and np.issubdtype(np.asarray(labels).dtype, np.integer):
                try:
                    from sklearn.model_selection import StratifiedGroupKFold
                    yield from StratifiedGroupKFold(
                        n_splits=n_splits, shuffle=True, random_state=seed
                    ).split(X, labels, groups=groups)
                    return
                except ImportError:
                    pass
            yield from GroupKFold(n_splits=n_splits).split(X, labels, groups=groups)
        else:
            yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, labels)


def _cv_split(groups, n_folds, X, labels, seed=42):
    """Legacy wrapper — delegates to cv_split."""
    mode = 'intermediary' if n_folds == 1 else 'full'
    yield from cv_split(groups, n_folds, X, labels, seed=seed, mode=mode)


def make_cv_splitter(n_folds, groups=None, seed=42):
    """Return (splitter, use_groups) for direct sklearn CV loops.

    Handles n_folds=1 (intermediary) via GroupShuffleSplit/ShuffleSplit.
    """
    if n_folds == 1:
        if groups is not None and len(np.unique(groups)) >= 2:
            return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), True
        from sklearn.model_selection import ShuffleSplit
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), False
    if groups is not None and len(np.unique(groups)) >= n_folds:
        return GroupKFold(n_splits=n_folds), True
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed), False


# =============================================================================
# DOMAIN-AWARE METRIC DISPATCH
# =============================================================================

def _heuristic_metric_name(task_type: str, domain: Optional[str],
                           labels: Optional[np.ndarray]) -> str:
    """Fallback metric dispatcher for datasets not in DATASET_REGISTRY.

    Matches the historical behaviour of get_domain_metric, _resolve_search_metric,
    and precompute_hybrids._get_metric. Prefer the registry (see
    _resolve_dataset_metric); this exists only for datasets that haven't been
    registered yet.
    """
    if task_type == 'classification':
        n_classes = len(np.unique(labels)) if labels is not None else 2
        return 'f1_weighted' if n_classes > 2 else 'auc'
    if domain == 'dta':
        return 'ci'
    if domain == 'antibody':
        return 'spearman'
    if domain == 'nucleotide':
        return 'mcc'
    # molecular, reaction, tcga-regression (none currently), other
    return 'rmse'


def _resolve_dataset_metric(dataset_name: Optional[str] = None,
                            labels: Optional[np.ndarray] = None,
                            task_type: Optional[str] = None,
                            domain: Optional[str] = None,
                            metric: str = 'auto',
                            ) -> Tuple[str, str, bool]:
    """Canonical (metric, task_type, lower_is_better) resolver.

    Resolution order:
    1. If `metric` is an explicit string (not 'auto'), honour it and look up
       direction from BUILTIN_METRICS.
    2. If `dataset_name` is registered in DATASET_REGISTRY, use that entry —
       this is the preferred path and the one experiments should hit.
    3. Fallback: derive metric from task_type + domain via the heuristic
       (historical behaviour).

    Every return path is consistent with BUILTIN_METRICS for direction.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    # 1. Explicit metric wins. Still need task_type.
    if metric != 'auto':
        if task_type not in ('classification', 'regression') and dataset_name:
            info = registry_lookup(dataset_name)
            if info is not None:
                task_type = info.task_type
        if task_type not in ('classification', 'regression'):
            raise ValueError(
                f"_resolve_dataset_metric: explicit metric={metric!r} requires "
                f"task_type. Got task_type={task_type!r}, "
                f"dataset_name={dataset_name!r}."
            )
        if metric not in BUILTIN_METRICS:
            raise ValueError(
                f"_resolve_dataset_metric: metric={metric!r} is not in "
                f"BUILTIN_METRICS. Available: {sorted(BUILTIN_METRICS.keys())}"
            )
        _, lower_is_better = BUILTIN_METRICS[metric]
        return metric, task_type, lower_is_better

    # 2. Registry lookup by dataset_name.
    if dataset_name:
        info = registry_lookup(dataset_name)
        if info is not None:
            # If caller also passed task_type, cross-check for safety.
            if task_type is not None and task_type != info.task_type:
                raise ValueError(
                    f"_resolve_dataset_metric: caller passed "
                    f"task_type={task_type!r} for '{dataset_name}' but "
                    f"registry says {info.task_type!r}. Mismatch likely "
                    f"indicates a loader bug — fix one side."
                )
            return info.metric, info.task_type, info.lower_is_better

    # 3. Heuristic fallback.
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"_resolve_dataset_metric: task_type={task_type!r} invalid and "
            f"dataset_name={dataset_name!r} is not in DATASET_REGISTRY. "
            f"Register the dataset or pass task_type explicitly."
        )
    chosen = _heuristic_metric_name(task_type, domain, labels)
    _, lower_is_better = BUILTIN_METRICS[chosen]
    return chosen, task_type, lower_is_better


def get_domain_metric(data: dict) -> Tuple[str, Any, bool]:
    """Return (metric_name, metric_fn, lower_is_better) for a loaded dataset.

    Prefers DATASET_REGISTRY when data['dataset_name'] is set (load_data sets
    it as of 2026-04-17). Falls back to the task_type + domain heuristic for
    datasets not yet registered.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    dataset_name = data.get('dataset_name')
    task_type = data.get('task_type')
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"get_domain_metric: data['task_type']={task_type!r}. "
            f"Must be 'classification' or 'regression'. Check the loader."
        )
    domain = data.get('domain', 'molecular')
    labels = data.get('labels')

    metric, _, lower_is_better = _resolve_dataset_metric(
        dataset_name=dataset_name, labels=labels, task_type=task_type,
        domain=domain, metric='auto')
    fn, _ = BUILTIN_METRICS[metric]
    return metric, fn, lower_is_better


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate_cv(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    allocation_method: str = 'top_3',
    importance_method: str = 'lightgbm_gain',
    budget: int = 100,
    seed: int = 42,
    mode: str = 'full',
    metric_fn=None,
) -> Tuple[float, float, int]:
    """Proper per-fold CV: create_hybrid on train, apply_feature_selection on val.

    Returns (mean_score, std_score, n_features).
    """
    if metric_fn is None:
        metric_fn = r2_score

    sample_X = next(iter(base_reps.values()))
    scores = []
    n_feats = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed, mode):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        scores.append(metric_fn(y_val, model.predict(X_val_s)))
        n_feats.append(X_train.shape[1])

    if not scores:
        return float('nan'), float('nan'), 0
    return float(np.mean(scores)), float(np.std(scores)), int(np.mean(n_feats))


def evaluate_cv_with_train(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    allocation_method: str = 'top_3',
    importance_method: str = 'lightgbm_gain',
    budget: int = 100,
    seed: int = 42,
    mode: str = 'full',
    metric_fn=None,
    internal_evaluator: str = 'ridge',
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
) -> Tuple[float, float, float, float, int]:
    """Like evaluate_cv but also returns train score.

    Returns (val_mean, val_std, train_mean, train_std, n_features).
    """
    from kirby.hybrid import apply_augmentation_selection

    if metric_fn is None:
        metric_fn = r2_score

    sample_X = next(iter(base_reps.values()))
    val_scores = []
    train_scores = []
    n_feats = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed, mode):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
                internal_evaluator=internal_evaluator,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
            aug_info = feature_info.get('augmentation_info')
            if aug_info and augmentations is not None:
                val_augs = {k: v[val_idx] for k, v in augmentations.items()}
                X_val_aug = apply_augmentation_selection(val_augs, aug_info)
                if X_val_aug is not None:
                    X_val = np.hstack([X_val, X_val_aug])
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        val_scores.append(metric_fn(y_val, model.predict(X_val_s)))
        train_scores.append(metric_fn(y_train, model.predict(X_train_s)))
        n_feats.append(X_train.shape[1])

    if not val_scores:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    return (
        float(np.mean(val_scores)), float(np.std(val_scores)),
        float(np.mean(train_scores)), float(np.std(train_scores)),
        int(np.mean(n_feats)),
    )


def cap_dimensions(X_train, X_val=None, max_dim=None):
    """PCA-reduce if features exceed max_dim. Returns (X_train, X_val) or (X_train,).

    If max_dim is None, defaults to n_train (X_train.shape[0]).
    Only applies PCA when X_train.shape[1] > max_dim.
    """
    if max_dim is None:
        max_dim = X_train.shape[0]
    if X_train.shape[1] <= max_dim:
        return (X_train, X_val) if X_val is not None else (X_train,)
    n_components = max(1, min(max_dim, X_train.shape[0] - 1))
    pca = PCA(n_components=n_components)
    X_train_r = pca.fit_transform(X_train)
    if X_val is not None:
        return X_train_r, pca.transform(X_val)
    return (X_train_r,)


def evaluate_single_rep_cv(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    mode: str = 'full',
    metric_fn=None,
) -> Tuple[float, float]:
    """Simple CV for a single representation matrix. Returns (mean_score, std_score)."""
    if metric_fn is None:
        metric_fn = r2_score

    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels, mode=mode):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        # PCA-reduce if features exceed training samples
        X_train, X_val = cap_dimensions(X_train, X_val)

        model = model_factory()
        model.fit(X_train, labels[train_idx])
        scores.append(metric_fn(labels[val_idx], model.predict(X_val)))

    return float(np.mean(scores)), float(np.std(scores))


def evaluate_weighted_knn(
    X: np.ndarray,
    y: np.ndarray,
    importance_scores: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 3,
    k_values: List[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate importance scores as feature weights for k-NN regression.

    Compares unweighted vs importance-weighted k-NN.
    Returns {k: {'unweighted_r2', 'weighted_r2', 'delta'}}.
    """
    from sklearn.neighbors import KNeighborsRegressor as _KNR

    if k_values is None:
        k_values = [5, 10, 20]

    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)

    results = {}
    for k in k_values:
        unweighted_scores = []
        weighted_scores = []

        for train_idx, val_idx in cv_split(groups, n_folds, X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            knn = _KNR(n_neighbors=min(k, len(y_train) - 1))
            knn.fit(X_train_s, y_train)
            unweighted_scores.append(r2_score(y_val, knn.predict(X_val_s)))

            X_train_w = X_train_s * weights
            X_val_w = X_val_s * weights
            knn_w = _KNR(n_neighbors=min(k, len(y_train) - 1))
            knn_w.fit(X_train_w, y_train)
            weighted_scores.append(r2_score(y_val, knn_w.predict(X_val_w)))

        uw = float(np.mean(unweighted_scores))
        w = float(np.mean(weighted_scores))
        results[k] = {'unweighted_r2': uw, 'weighted_r2': w, 'delta': w - uw}

    return results


def evaluate_similarity_retrieval(
    X: np.ndarray,
    y: np.ndarray,
    importance_scores: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 3,
    k_neighbors: int = 10,
) -> Dict[str, float]:
    """Evaluate importance scores as weights for similarity-based retrieval.

    For each test molecule, find k nearest neighbors in training set.
    Measure how well neighbor target values predict the test target.
    Returns {'unweighted_mae', 'weighted_mae', 'delta_mae',
             'unweighted_r2', 'weighted_r2', 'delta_r2'}.
    """
    from sklearn.metrics import mean_absolute_error

    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)

    all_unweighted_preds = []
    all_weighted_preds = []
    all_true = []

    for train_idx, val_idx in cv_split(groups, n_folds, X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        nn_uw = NearestNeighbors(n_neighbors=min(k_neighbors, len(y_train) - 1))
        nn_uw.fit(X_train_s)
        dists_uw, idx_uw = nn_uw.kneighbors(X_val_s)
        w_uw = 1.0 / (dists_uw + 1e-10)
        preds_uw = np.average(y_train[idx_uw], axis=1, weights=w_uw)

        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights
        nn_w = NearestNeighbors(n_neighbors=min(k_neighbors, len(y_train) - 1))
        nn_w.fit(X_train_w)
        dists_w, idx_w = nn_w.kneighbors(X_val_w)
        w_w = 1.0 / (dists_w + 1e-10)
        preds_w = np.average(y_train[idx_w], axis=1, weights=w_w)

        all_unweighted_preds.extend(preds_uw)
        all_weighted_preds.extend(preds_w)
        all_true.extend(y_val)

    all_true = np.array(all_true)
    all_unweighted_preds = np.array(all_unweighted_preds)
    all_weighted_preds = np.array(all_weighted_preds)

    uw_mae = mean_absolute_error(all_true, all_unweighted_preds)
    w_mae = mean_absolute_error(all_true, all_weighted_preds)
    uw_r2 = r2_score(all_true, all_unweighted_preds)
    w_r2 = r2_score(all_true, all_weighted_preds)

    return {
        'unweighted_mae': float(uw_mae),
        'weighted_mae': float(w_mae),
        'delta_mae': float(uw_mae - w_mae),
        'unweighted_r2': float(uw_r2),
        'weighted_r2': float(w_r2),
        'delta_r2': float(w_r2 - uw_r2),
    }


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_model_factories(model_names: List[str],
                        task_type: str = 'regression',
                        n_samples: Optional[int] = None) -> Dict[str, callable]:
    """Return {name: factory_fn} for requested models.

    Delegates to find_best_hybrid.DEFAULT_MODELS for authoritative model
    definitions (all 11 models, with size guards and clf/reg dispatch).

    Args:
        model_names: List of model names (e.g. ['rf', 'ridge', 'gp']).
        task_type: 'regression' or 'classification'.
        n_samples: Dataset size — models with _max_samples auto-skip.

    Returns:
        {name: callable} where each callable returns a fresh model instance.
    """
    from kirby.find_best_hybrid import get_model as _get_model, DEFAULT_MODELS

    factories = {}
    for name in model_names:
        if name not in DEFAULT_MODELS:
            print(f"  WARNING: model '{name}' not in DEFAULT_MODELS, skipping")
            continue
        spec = DEFAULT_MODELS[name]
        max_s = spec.get('_max_samples')
        if max_s is not None and n_samples is not None and n_samples > max_s:
            print(f"  Skipping {name}: n_samples={n_samples} > _max_samples={max_s}")
            continue

        def _make_factory(_name, _task_type):
            return lambda: _get_model(_name, _task_type)

        factories[name] = _make_factory(name, task_type)
    return factories


def get_model_family_variants() -> Dict[str, List[Tuple[str, callable, float]]]:
    """Return model families with complexity gradients.

    Each family: list of (variant_name, factory_fn, capacity_index).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    families = {
        'RF': [
            ('depth=3', lambda: RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1), 1),
            ('depth=5', lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1), 2),
            ('depth=10', lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), 3),
            ('depth=None', lambda: RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1), 4),
        ],
        'Ridge': [
            ('alpha=100', lambda: Ridge(alpha=100.0), 1),
            ('alpha=10', lambda: Ridge(alpha=10.0), 2),
            ('alpha=1', lambda: Ridge(alpha=1.0), 3),
            ('alpha=0.01', lambda: Ridge(alpha=0.01), 4),
        ],
        'kNN': [
            ('k=20', lambda: KNeighborsRegressor(n_neighbors=20, n_jobs=-1), 1),
            ('k=10', lambda: KNeighborsRegressor(n_neighbors=10, n_jobs=-1), 2),
            ('k=5', lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1), 3),
            ('k=1', lambda: KNeighborsRegressor(n_neighbors=1, n_jobs=-1), 4),
        ],
        'MLP': [
            ('16', lambda: MLPRegressor(hidden_layer_sizes=(16,), random_state=42, max_iter=500), 1),
            ('64', lambda: MLPRegressor(hidden_layer_sizes=(64,), random_state=42, max_iter=500), 2),
            ('128,64', lambda: MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500), 3),
            ('256,128,64', lambda: MLPRegressor(hidden_layer_sizes=(256, 128, 64), random_state=42, max_iter=500), 4),
        ],
    }

    try:
        from xgboost import XGBRegressor
        families['XGBoost'] = [
            ('depth=2', lambda: XGBRegressor(n_estimators=100, max_depth=2, random_state=42, n_jobs=-1, verbosity=0), 1),
            ('depth=4', lambda: XGBRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0), 2),
            ('depth=6', lambda: XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0), 3),
            ('depth=None', lambda: XGBRegressor(n_estimators=100, max_depth=0, random_state=42, n_jobs=-1, verbosity=0, tree_method='hist'), 4),
        ]
    except ImportError:
        pass

    # SVM: C controls capacity (low C = high regularisation = low capacity)
    from sklearn.svm import SVR
    families['SVM'] = [
        ('C=0.01', lambda: SVR(kernel='rbf', C=0.01), 1),
        ('C=0.1', lambda: SVR(kernel='rbf', C=0.1), 2),
        ('C=1.0', lambda: SVR(kernel='rbf', C=1.0), 3),
        ('C=10.0', lambda: SVR(kernel='rbf', C=10.0), 4),
    ]

    # GP: length_scale controls capacity (large = smooth/low capacity, small = wiggly/high)
    # Size guard: GP is O(n³), only viable for n ≤ 2000
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF as _RBF, WhiteKernel as _WK
        families['GP'] = [
            ('ls=10', lambda: GaussianProcessRegressor(
                kernel=_RBF(length_scale=10.0) + _WK(), n_restarts_optimizer=2, random_state=42), 1),
            ('ls=1', lambda: GaussianProcessRegressor(
                kernel=_RBF(length_scale=1.0) + _WK(), n_restarts_optimizer=2, random_state=42), 2),
            ('ls=0.1', lambda: GaussianProcessRegressor(
                kernel=_RBF(length_scale=0.1) + _WK(), n_restarts_optimizer=2, random_state=42), 3),
            ('ls=0.01', lambda: GaussianProcessRegressor(
                kernel=_RBF(length_scale=0.01) + _WK(), n_restarts_optimizer=2, random_state=42), 4),
        ]
    except ImportError:
        pass

    return families


# =============================================================================
# METRICS
# =============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Dict[str, Any]:
    """Compute intrinsic dimensionality using PCA and MLE methods."""
    results = {'nominal_dim': X.shape[1]}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        max_comp = min(500, n - 1)
        svd = TruncatedSVD(n_components=max_comp, random_state=42)
        svd.fit(X_scaled)
        cumvar = np.cumsum(svd.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            idx = np.searchsorted(cumvar, threshold)
            results[key] = int(min(idx + 1, max_comp)) if idx < len(cumvar) else max_comp
    else:
        n_components = min(n, d)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            intrinsic_dim = np.searchsorted(cumvar, threshold) + 1
            results[key] = int(min(intrinsic_dim, d))

    X_for_mle = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_mle = TruncatedSVD(n_components=min(200, n - 1), random_state=42).fit_transform(X_scaled)
    try:
        results['mle'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        results['mle'] = None

    if results.get('pca_95'):
        results['compression_ratio'] = X.shape[1] / results['pca_95']

    return results


def estimate_intrinsic_dim_mle(X: np.ndarray, k: int = 10) -> float:
    """MLE estimator of intrinsic dimensionality (Levina-Bickel 2004)."""
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]

    estimates = []
    for i in range(n_samples):
        dists = distances[i]
        if dists[-1] > 0:
            log_ratios = np.log(dists[-1] / dists[:-1])
            if np.sum(log_ratios) > 0:
                estimates.append((k - 1) / np.sum(log_ratios))

    return np.mean(estimates) if estimates else np.nan


def estimate_mutual_information_knn(X: np.ndarray, y: np.ndarray, k: int = 5,
                                    max_dim: int = 50) -> float:
    """Estimate I(X; Y) using KSG estimator (Kraskov et al. 2004).

    Args:
        max_dim: Maximum dimensionality before PCA reduction. Higher values
            preserve more nonlinear structure but need more samples for stable
            estimates. Capped at n_samples // 5 for the n >> d requirement.
    """
    n_samples, d = X.shape
    k = min(k, n_samples - 1)

    X_scaled = StandardScaler().fit_transform(X)

    # KSG needs n >> d; reduce dimensions when d is large relative to n.
    max_components = min(max_dim, n_samples // 5)
    if d > max_components >= 1:
        from sklearn.decomposition import PCA as _PCA
        X_scaled = _PCA(n_components=max_components, random_state=42).fit_transform(X_scaled)
    elif max_components < 1:
        return 0.0  # too few samples for any MI estimate
    y_scaled = (y - y.mean()) / (y.std() + 1e-10)

    XY = np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])

    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
    nn_joint.fit(XY)
    distances_joint, _ = nn_joint.kneighbors(XY)
    eps = distances_joint[:, -1]

    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(X_scaled)

    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(y_scaled.reshape(-1, 1))

    n_x = np.zeros(n_samples)
    n_y = np.zeros(n_samples)

    for i in range(n_samples):
        n_x[i] = len(nn_x.radius_neighbors([X_scaled[i]], radius=eps[i],
                      return_distance=False)[0]) - 1
        n_y[i] = len(nn_y.radius_neighbors([y_scaled[i:i + 1]], radius=eps[i],
                      return_distance=False)[0]) - 1

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n_samples)
    return max(0, mi)


def fit_learning_curve(n_samples: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Fit asymptotic model: R² = R²_max - a/n^b."""
    from scipy.optimize import curve_fit

    def model(n, r2_max, a, b):
        return r2_max - a / np.power(n, b)

    try:
        p0 = [np.max(scores), 1.0, 0.5]
        bounds = ([0, 0, 0.1], [1, 100, 2])
        popt, _ = curve_fit(model, n_samples, scores, p0=p0, bounds=bounds, maxfev=5000)
        r2_max, a, b = popt

        predicted = model(n_samples, *popt)
        residuals = scores - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {'r2_max': float(r2_max), 'a': float(a), 'b': float(b), 'fit_r2': float(fit_r2)}
    except Exception as e:
        return {'error': str(e), 'r2_max': float(np.max(scores))}


def compute_effective_rank(X: np.ndarray) -> float:
    """Effective rank: exp(entropy of normalised singular values)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(500, n - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X_scaled)
        s = svd.singular_values_
    else:
        try:
            s = np.linalg.svd(X_scaled, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            # gesdd LAPACK driver fails on some near-singular reps (observed
            # on abagym_Ang2). gesvd is slower but more robust.
            from scipy.linalg import svd as scipy_svd
            s = scipy_svd(X_scaled, full_matrices=False, compute_uv=False,
                          lapack_driver='gesvd')

    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_dataset_complexity(reps: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, float]:
    """Compute dataset complexity metrics (Lorena et al. 2018 inspired)."""
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    n, d = X_concat.shape
    y = labels

    metrics = {'n_samples': n, 'n_features': d}

    erank = compute_effective_rank(X_concat)
    metrics['erank'] = erank
    metrics['n_over_d'] = n / d
    metrics['n_over_erank'] = n / erank

    X_scaled = StandardScaler().fit_transform(X_concat)
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(500, n - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X_scaled)
        s = svd.singular_values_
    else:
        _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
    eigvals = s ** 2 / n
    eigvals = eigvals[eigvals > 1e-10]
    metrics['participation_ratio'] = float((eigvals.sum()) ** 2 / (eigvals ** 2).sum())

    max_features = min(d, 1000)
    if d > max_features:
        rng = np.random.RandomState(42)
        feat_idx = rng.choice(d, max_features, replace=False)
        X_sub = X_concat[:, feat_idx]
    else:
        X_sub = X_concat

    abs_corrs = []
    for j in range(X_sub.shape[1]):
        col = X_sub[:, j]
        if col.std() < 1e-10:
            abs_corrs.append(0.0)
            continue
        rho, _ = spearmanr(col, y)
        abs_corrs.append(abs(rho) if not np.isnan(rho) else 0.0)
    abs_corrs = np.array(abs_corrs)
    metrics['C1'] = float(np.max(abs_corrs))
    metrics['C2'] = float(np.mean(abs_corrs))

    X_for_model = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_model = TruncatedSVD(n_components=min(500, n - 1), random_state=42).fit_transform(X_scaled)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_for_model, y)
    y_pred_linear = ridge.predict(X_for_model)
    y_range = y.max() - y.min()
    metrics['L1'] = float(np.mean(np.abs(y - y_pred_linear)) / y_range) if y_range > 0 else 0.0

    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    nn.fit(X_for_model)
    _, indices = nn.kneighbors(X_for_model)
    y_nn = y[indices[:, 1]]
    metrics['S3'] = float(np.mean(np.abs(y - y_nn)) / y_range) if y_range > 0 else 0.0

    metrics['label_std'] = float(np.std(y))
    counts, _ = np.histogram(y, bins=20)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    metrics['target_entropy'] = float(-np.sum(probs * np.log(probs)))

    X_for_mle = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_mle = TruncatedSVD(n_components=min(200, n - 1), random_state=42).fit_transform(X_scaled)
    try:
        metrics['mle_id'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        metrics['mle_id'] = None

    return metrics


# =============================================================================
# DATA LOADING (domain-aware)
# =============================================================================

_ANTIBODY_PREFIXES = ('flab_', 'abcov_', 'sabdab_', 'abagym_', 'abbind', 'ddg_',
                      'ginkgo_', 'tz_her2', 'humanness_', 'graphinity_')
_TCGA_PREFIXES = ('tcga_',)
_NUCLEOTIDE_PREFIXES = ('nt_', 'gb_')
_REACTION_DATASETS = ('buchwald_hartwig', 'suzuki', 'bh_doyle_plates')
_DTA_DATASETS = ('davis', 'kiba', 'bindingdb_kd')
_DTA_PREFIXES = ('pdbbind',)


# =============================================================================
# FRAMEWORK EXPERIMENT TASK & REP CONFIGURATION
# =============================================================================

FRAMEWORK_TASKS = {
    # Molecular (RMSE, scaffold-split)
    'logd': [
        'mhg_gnn', 'grover', 'mordred', 'mol2vec', 'ecfp4', 'molformer',
        'pdv', 'smited', 'avalon', 'chemberta', 'graph_kernel', 'maccs',
    ],
    'herg_ki': [
        'mhg_gnn', 'graph_kernel', 'avalon', 'grover', 'ecfp4', 'mordred',
        'pdv', 'molformer', 'smited', 'chemberta', 'maccs',
    ],
    # Antibody (Spearman, identity-grouped)
    'flab_thermo_tresanco': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'mutation_features', 'ablang2', 'antiberty',
        'aac', 'structural',
    ],
    'flab_pk_jain_fcrnrt': [
        'ablang2', 'igt5', 'esm2_early', 'prot_t5', 'saprot',
        'mutation_features', 'antiberty', 'esmc', 'esm2', 'aac',
        'abroberta', 'onehot', 'structural',
    ],
    # TCGA (AUROC / F1w, TSS-grouped)
    'tcga_luad_vs_lusc': [
        'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
        'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
        'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
        'rppa', 'immune_xcell', 'mut_driver', 'stemness_mrnasi',
    ],
    'tcga_brca_pam50': [
        'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
        'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
        'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
        'rppa', 'immune_xcell', 'mut_driver', 'stemness_mrnasi',
    ],
    # Small antibody (Spearman, identity-grouped) — same family as flab_pk_jain_fcrnrt
    'flab_pk_jain_heprt': [
        'ablang2', 'igt5', 'esm2_early', 'prot_t5', 'saprot',
        'mutation_features', 'antiberty', 'esmc', 'esm2', 'aac',
        'abroberta', 'onehot', 'structural',
    ],
    # AbAgym DMS (Spearman, position_modulo split)
    'abagym_EGFR': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'ablang2', 'antiberty', 'aac', 'structural',
    ],
    # Trastuzumab binding classification (AUROC/F1w, edit_distance grouped)
    'tz_her2_binary': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'ablang2', 'antiberty', 'aac', 'structural',
    ],
    # Ginkgo developability (Spearman, pre-computed 5-fold splits, n=242)
    'ginkgo_hic': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'ablang2', 'antiberty', 'aac', 'structural',
    ],
    # Reaction (R², leave-additive-out GroupKFold)
    'buchwald_hartwig': [
        'reactant_ecfp4', 'drfp_2048', 'onehot_components', 'rxnfp',
        'product_ecfp4', 'reactant_maccs', 'product_maccs', 'reactant_pdv',
        'product_pdv', 'diff_ecfp4', 'diff_maccs', 'component_descriptors',
        'ligand_descriptors', 'reactiont5', 'concat_ecfp4', 'concat_maccs',
        'concat_pdv',
    ],
}


def _is_antibody_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _ANTIBODY_PREFIXES)


def _is_tcga_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _TCGA_PREFIXES)


def _is_nucleotide_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _NUCLEOTIDE_PREFIXES)


def _is_reaction_dataset(name: str) -> bool:
    return name.lower() in _REACTION_DATASETS


def _is_dta_dataset(name: str) -> bool:
    n = name.lower()
    return n in _DTA_DATASETS or any(n.startswith(p) for p in _DTA_PREFIXES)


def _build_rep_factories(dataset_name, data):
    """Construct FT factory callables from DATASET_REPS ft_reps config.

    Returns dict mapping rep_name -> factory callable, or empty dict.
    Each factory has signature: (train_idx, val_idx, train_labels) -> (train_embs, val_embs).
    """
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
    except ImportError:
        return {}

    ds_config = DATASET_REPS.get(dataset_name, {})
    ft_config = ds_config.get('ft_reps', {})
    if not ft_config:
        return {}

    factories = {}
    for rep_name, spec in ft_config.items():
        factory_type = spec['factory']
        base_model = spec['base_model']
        try:
            if factory_type == 'molecular_lora':
                from kirby.finetuning import make_molecular_lora_factory
                factories[rep_name] = make_molecular_lora_factory(
                    smiles_list=data['smiles'], dataset_name=dataset_name,
                    base_model=base_model)
            elif factory_type == 'molecular_contrastive':
                from kirby.finetuning import make_molecular_contrastive_factory
                factories[rep_name] = make_molecular_contrastive_factory(
                    smiles_list=data['smiles'], dataset_name=dataset_name,
                    base_model=base_model)
            elif factory_type == 'molecular_gnn':
                from kirby.finetuning import make_molecular_gnn_factory
                factories[rep_name] = make_molecular_gnn_factory(
                    smiles_list=data['smiles'], dataset_name=dataset_name,
                    base_model=base_model)
            elif factory_type == 'antibody_lora':
                from kirby.finetuning import make_antibody_lora_factory
                factories[rep_name] = make_antibody_lora_factory(
                    heavy_seqs=data['heavy_seqs'], light_seqs=data['light_seqs'],
                    dataset_name=dataset_name, base_model=base_model)
            else:
                print(f"  WARNING: unknown FT factory type '{factory_type}'")
        except Exception as e:
            print(f"  WARNING: FT factory {rep_name} failed: {e}")

    if factories:
        print(f"  [ft] {len(factories)} FT factories: {', '.join(factories)}")

    return factories


def _require_task_type(data: dict, source: str) -> str:
    """Assert data['task_type'] is set to a valid value. Raise if not.

    Every dataset loader MUST set 'task_type' to 'classification' or
    'regression'. Silent defaulting to 'regression' has caused classification
    datasets to be scored with regression metrics in the past — this guard
    turns that into a loud failure.
    """
    tt = data.get('task_type')
    if tt not in ('classification', 'regression'):
        raise ValueError(
            f"Loader for '{source}' returned task_type={tt!r}. "
            f"Every loader MUST set 'task_type' to 'classification' or "
            f"'regression' — silent defaults are not allowed."
        )
    return tt


def _finalize_load_data(data: dict, dataset_name: str) -> dict:
    """Post-process a load_data result: inject dataset_name, cross-check registry.

    - Adds `data['dataset_name']` for downstream consumers.
    - If the dataset is in DATASET_REGISTRY, verifies that the loader's
      task_type matches the registry entry, and backfills `data['domain']`
      if the loader didn't set it. Mismatches raise loudly — a silent drift
      between loader and registry is exactly the class of bug this registry
      exists to prevent.
    """
    data['dataset_name'] = dataset_name
    info = registry_lookup(dataset_name)
    if info is not None:
        loader_task = data.get('task_type')
        if loader_task != info.task_type:
            raise ValueError(
                f"Registry/loader mismatch for '{dataset_name}': "
                f"loader returned task_type={loader_task!r}, but "
                f"DATASET_REGISTRY says {info.task_type!r}. "
                f"Fix one side — they must agree."
            )
        if 'domain' not in data or data['domain'] is None:
            data['domain'] = info.domain
        elif data['domain'] != info.domain:
            # Loader set a domain that disagrees — warn once, registry wins.
            import warnings
            warnings.warn(
                f"Registry/loader domain mismatch for '{dataset_name}': "
                f"loader set domain={data['domain']!r}, registry says "
                f"{info.domain!r}. Using registry value."
            )
            data['domain'] = info.domain
    return data


def load_data(dataset_name: str, rep_preset: str = 'fast',
              delta: bool = False, augment: bool = False,
              rep_names: Optional[List[str]] = None,
              finetune: bool = False) -> dict:
    """Public entry point — loads data and stamps it with registry metadata.

    Thin wrapper around `_load_data_impl` that:
    - Injects `data['dataset_name']`.
    - Cross-checks `data['task_type']` against DATASET_REGISTRY.
    - Backfills `data['domain']` from registry when missing.
    """
    data = _load_data_impl(
        dataset_name, rep_preset=rep_preset, delta=delta, augment=augment,
        rep_names=rep_names, finetune=finetune)
    return _finalize_load_data(data, dataset_name)


def _load_data_impl(dataset_name: str, rep_preset: str = 'fast',
                    delta: bool = False, augment: bool = False,
                    rep_names: Optional[List[str]] = None,
                    finetune: bool = False) -> dict:
    """Load dataset, generate representations, assign groups.

    Domain-aware: detects TCGA, antibody, or molecular dataset names and
    dispatches accordingly.
    Returns dict with keys: labels, reps, groups, n_groups, task_type,
    augmentations, plus domain-specific keys.

    When rep_names is None, auto-filters to the 'general' reps from
    DATASET_REPS (if the dataset has an entry).  Pass rep_names explicitly
    to override.

    **Do not call directly** — use `load_data()` so registry cross-checks
    and `dataset_name` injection happen.
    """
    # Auto-filter to DATASET_REPS general reps when no explicit list given
    if rep_names is None:
        try:
            from experiments.complexity.complexity_helpers import DATASET_REPS
            ds_reps = DATASET_REPS.get(dataset_name)
            if ds_reps is not None and ds_reps.get('general'):
                rep_names = ds_reps['general']
                print(f"  [auto] {dataset_name}: filtering to "
                      f"{len(rep_names)} general reps from DATASET_REPS")
        except ImportError:
            pass

    if _is_tcga_dataset(dataset_name):
        from find_best_hybrid_tcga import (
            load_dataset as load_tcga_dataset,
            get_representations as get_tcga_representations,
            assign_tss_groups,
        )

        expression, labels, task_type, metadata = load_tcga_dataset(dataset_name)
        sample_ids = metadata['sample_ids']
        gene_names = metadata['gene_names']
        wsi_dir = metadata.get('wsi_dir')
        cohort = metadata.get('cohort', '')

        reps = _cache_rep_dict(
            dataset_name,
            lambda: get_tcga_representations(
                expression, gene_names, sample_ids, wsi_dir,
                rep_preset, cohort=cohort,
            ),
            filter_names=rep_names,
        )

        tss_groups, n_groups = assign_tss_groups(sample_ids)

        print(f"  {dataset_name}: {len(sample_ids)} samples, {n_groups} TSS sites, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'expression': expression,
            'labels': labels,
            'reps': reps,
            'groups': tss_groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': task_type,
            'domain': 'tcga',
            'augmentations': None,
            'metadata': metadata,
            'rep_factories': {},
        }

    elif _is_antibody_dataset(dataset_name):
        from antibody_baseline_survey import load_dataset as load_ab_dataset
        from kirby.representations.antibody import ANTIBODY_REPS, ANTIBODY_FAST_REPS
        from kirby.representations.mutation import MUTATION_REPS

        data = load_ab_dataset(dataset_name)
        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = data['labels']
        groups = data['groups']
        n_groups = data['n_groups']
        mutations = data.get('mutations')

        # Generate reps based on preset or custom list
        if rep_preset in ('fast', 'skip_lm'):
            _rep_names_from_preset = sorted(ANTIBODY_FAST_REPS)
        else:
            _rep_names_from_preset = sorted(
                set(ANTIBODY_REPS.keys()) | set(MUTATION_REPS.keys())
            )

        _wanted_reps = rep_names if rep_names else _rep_names_from_preset
        reps = {}

        # Pass 1: standard antibody reps (heavy_seqs, light_seqs)
        for rn in _wanted_reps:
            if rn in ANTIBODY_REPS:
                try:
                    fn = ANTIBODY_REPS[rn]
                    reps[rn] = _cache_rep(dataset_name, rn,
                                          lambda _fn=fn: _fn(heavy_seqs, light_seqs))
                except Exception as e:
                    print(f"  WARNING: {rn} failed: {e}")
            elif rn not in MUTATION_REPS:
                print(f"  WARNING: antibody rep '{rn}' not found, skipping")

        # Pass 2: mutation-aware reps (require mutations metadata)
        _has_mutations = mutations and any(len(m) > 0 for m in mutations)
        if _has_mutations:
            for rn in _wanted_reps:
                if rn in MUTATION_REPS and rn not in reps:
                    try:
                        fn = MUTATION_REPS[rn]
                        reps[rn] = _cache_rep(
                            dataset_name, rn,
                            lambda _fn=fn: _fn(heavy_seqs, light_seqs, mutations))
                    except Exception as e:
                        print(f"  WARNING: {rn} failed: {e}")

        print(f"  {dataset_name}: {len(heavy_seqs)} antibodies, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        result = {
            'heavy_seqs': heavy_seqs,
            'light_seqs': light_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_groups,
            'n_groups': n_groups,
            'task_type': _require_task_type(data, dataset_name),
            'domain': 'antibody',
            'augmentations': None,
        }
        result['rep_factories'] = _build_rep_factories(dataset_name, result) if finetune else {}
        return result
    elif _is_nucleotide_dataset(dataset_name):
        from find_best_hybrid_nucleotide import (
            load_dataset as load_nuc_dataset,
            get_representations as get_nuc_representations,
        )

        data = load_nuc_dataset(dataset_name)
        sequences = data['sequences']
        labels = data['labels']
        groups = data['groups']
        n_groups = data['n_groups']
        task_type = data['task_type']

        # Nucleotide rep generation: when explicit rep_names are given
        # (precompute / experiments), pass them as a comma-joined list so
        # foundation-model reps (mistral_dna, hyenadna, nt_v2_*) and
        # classical augmentation reps both get generated. Without this,
        # the default 'fast' preset skips foundation models entirely and
        # the precompute never sees the reps DATASET_REPS asks for.
        if rep_names:
            _nuc_preset = ','.join(rep_names)
        else:
            _nuc_preset = rep_preset
        reps = _cache_rep_dict(
            dataset_name,
            lambda: get_nuc_representations(sequences, _nuc_preset),
            filter_names=rep_names,
        )

        print(f"  {dataset_name}: {len(sequences)} sequences, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'sequences': sequences,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': task_type,
            'domain': 'nucleotide',
            'augmentations': None,
            'rep_factories': {},
        }

    elif _is_reaction_dataset(dataset_name):
        from kirby.datasets.reaction_yields import (
            load_buchwald_hartwig, load_suzuki, load_bh_doyle_plates,
        )
        from kirby.representations.reaction import (
            create_drfp, create_concat_ecfp4, create_concat_maccs,
            create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
            create_product_maccs, create_reactant_maccs,
            create_product_pdv, create_reactant_pdv,
            create_rxnfp, create_onehot_components,
            create_diff_ecfp4, create_diff_maccs,
            create_component_descriptors, create_ligand_descriptors,
            create_reactiont5,
            create_cgr_morgan,
            create_chytorch_graphormer,
            create_cgr_smiles_lm,
        )

        loaders = {
            'buchwald_hartwig': load_buchwald_hartwig,
            'suzuki': load_suzuki,
            'bh_doyle_plates': load_bh_doyle_plates,
        }
        data = loaders[dataset_name]()
        rxn_smiles = data['rxn_smiles']
        labels = np.array(data['labels'], dtype=np.float32)
        groups = np.array(data['groups'])
        components = data.get('components')

        # Registry of all reaction rep generators
        _rxn_generators = {
            'drfp_2048': lambda: create_drfp(rxn_smiles),
            'concat_ecfp4': lambda: create_concat_ecfp4(components),
            'concat_maccs': lambda: create_concat_maccs(components),
            'concat_pdv': lambda: create_concat_pdv(components),
            'product_ecfp4': lambda: create_product_ecfp4(rxn_smiles),
            'reactant_ecfp4': lambda: create_reactant_ecfp4(rxn_smiles),
            'product_maccs': lambda: create_product_maccs(rxn_smiles),
            'reactant_maccs': lambda: create_reactant_maccs(rxn_smiles),
            'product_pdv': lambda: create_product_pdv(rxn_smiles),
            'reactant_pdv': lambda: create_reactant_pdv(rxn_smiles),
            'rxnfp': lambda: create_rxnfp(rxn_smiles),
            'onehot_components': lambda: create_onehot_components(components),
            'diff_ecfp4': lambda: create_diff_ecfp4(rxn_smiles),
            'diff_maccs': lambda: create_diff_maccs(rxn_smiles),
            'component_descriptors': lambda: create_component_descriptors(components),
            'ligand_descriptors': lambda: create_ligand_descriptors(components),
            'reactiont5': lambda: create_reactiont5(rxn_smiles),
            'cgr_morgan': lambda: create_cgr_morgan(rxn_smiles),
            'chytorch_graphormer': lambda: create_chytorch_graphormer(rxn_smiles),
            'cgr_smiles_lm': lambda: create_cgr_smiles_lm(rxn_smiles),
        }

        # Determine which reps to generate
        if rep_names:
            wanted = rep_names
        elif rep_preset == 'all':
            wanted = list(_rxn_generators.keys())
        else:
            # fast preset: the original 8
            wanted = ['drfp_2048', 'concat_ecfp4', 'concat_maccs', 'concat_pdv',
                       'product_ecfp4', 'reactant_ecfp4', 'rxnfp', 'onehot_components']

        reps = {}
        for name in wanted:
            if name not in _rxn_generators:
                print(f"  WARNING: unknown reaction rep '{name}', skipping")
                continue
            try:
                gen = _rxn_generators[name]
                reps[name] = _cache_rep(dataset_name, name,
                                        lambda _g=gen: _g())
            except Exception as e:
                print(f"  WARNING: {name} failed: {e}")

        n_groups = len(np.unique(groups))
        print(f"  {dataset_name}: {len(labels)} reactions, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': 'regression',
            'domain': 'reaction',
            'augmentations': None,
            'rep_factories': {},
        }

    elif _is_dta_dataset(dataset_name):
        # Lazy import to avoid pulling DTA deps for non-DTA runs.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from find_best_hybrid_dta import (
            load_dataset as load_dta_dataset,
            get_representations as get_dta_representations,
        )
        from kirby.datasets.dta import (
            assign_cold_target_groups, assign_cold_drug_groups,
        )

        data = load_dta_dataset(dataset_name)
        drug_smiles = data['drug_smiles']
        protein_seqs = data['protein_seqs']
        labels = np.asarray(data['labels'], dtype=np.float64)
        metadata = data['metadata']

        # Subsample to 30K pairs to match the DATASET_REPS comment
        # ("n=30,000") and the dta_baseline_survey convention. Cold-protein
        # split is the default — same as dta_baseline_survey.
        DTA_MAX_PAIRS = 30_000
        if len(labels) > DTA_MAX_PAIRS:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(labels), size=DTA_MAX_PAIRS, replace=False)
            idx.sort()
            drug_smiles = [drug_smiles[i] for i in idx]
            protein_seqs = [protein_seqs[i] for i in idx]
            labels = labels[idx]
            metadata['drug_ids'] = [metadata['drug_ids'][i] for i in idx]
            metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]
            metadata['n_drugs'] = len(set(metadata['drug_ids']))
            metadata['n_proteins'] = len(set(metadata['protein_ids']))
            print(f"  Subsampled {dataset_name} to {DTA_MAX_PAIRS} pairs")

        groups, n_groups = assign_cold_target_groups(metadata['protein_ids'])
        print(f"  {dataset_name}: {len(labels)} pairs, "
              f"{metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins, "
              f"{n_groups} cold-target groups")

        # DTA reps: drug + protein. get_representations handles the dispatch.
        if rep_names:
            preset_or_list = ','.join(rep_names)
        else:
            preset_or_list = rep_preset
        reps = get_dta_representations(
            drug_smiles, protein_seqs, preset_or_list,
            cache=True, protein_ids=metadata.get('protein_ids'))

        if rep_names:
            reps = {k: v for k, v in reps.items() if k in rep_names}

        print(f"  {dataset_name}: {len(labels)} samples, {n_groups} groups, "
              f"{len(reps)} reps "
              f"({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'drug_smiles': drug_smiles,
            'protein_seqs': protein_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': data.get('task_type', 'regression'),
            'domain': 'dta',
            'augmentations': None,
            'rep_factories': {},
            'metadata': metadata,
        }

    else:
        from find_best_hybrid_molecular import (
            load_dataset as load_molecular_dataset,
            get_representations as get_molecular_representations,
            assign_scaffold_groups,
        )

        smiles, labels, task_type = load_molecular_dataset(dataset_name)
        groups, n_scaffolds = assign_scaffold_groups(smiles)

        reps = _cache_rep_dict(dataset_name,
                               lambda: get_molecular_representations(smiles, rep_preset),
                               filter_names=rep_names)

        print(f"  {dataset_name}: {len(smiles)} samples, {n_scaffolds} scaffolds, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        result = {
            'smiles': smiles,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_scaffolds,
            'task_type': task_type,
            'domain': 'molecular',
            'augmentations': None,
        }
        result['rep_factories'] = _build_rep_factories(dataset_name, result) if finetune else {}
        return result


# =============================================================================
# SIMILARITY-SPECIFIC HELPERS
# =============================================================================

def get_hybrid_features_and_weights(
    reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    search_speed: str = 'full',
    weight_method: str = 'lightgbm_gain',
    budget: Optional[int] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    metric: str = 'rmse',
) -> Tuple[np.ndarray, np.ndarray, SearchResult]:
    """Run find_best_hybrid and return hybrid features + flat importance weights."""
    kwargs = dict(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed=search_speed,
        return_fitted=True,
        weight_method=weight_method,
        verbose=False,
        cache=True,
        metric=metric,
    )
    if budget is not None:
        kwargs['budgets'] = [budget]
    if augmentations is not None:
        kwargs['augmentations'] = augmentations

    search = find_best_hybrid(**kwargs)
    # Use RF as default downstream model for weight extraction
    fitted = (search.fitted_results or {}).get('rf')

    if fitted is None:
        raise RuntimeError("find_best_hybrid returned no fitted result for 'rf'")

    X_hybrid = apply_feature_selection(reps, _fitted_to_feature_info(fitted))
    # Append augmentation features if present (must match flatten_weights layout)
    if fitted.augmentation_info and augmentations is not None:
        from kirby.hybrid import apply_augmentation_selection
        X_aug = apply_augmentation_selection(augmentations, fitted.augmentation_info)
        if X_aug is not None:
            X_hybrid = np.hstack([X_hybrid, X_aug])
    weights_flat = flatten_weights(fitted)

    return X_hybrid, weights_flat, search


def _fitted_to_feature_info(fitted) -> dict:
    """Convert FittedResult to feature_info dict for apply_feature_selection."""
    feature_info = {}
    for rep_name, indices in fitted.selected_indices.items():
        feature_info[rep_name] = {'selected_indices': indices}
    return feature_info


def flatten_weights(fitted) -> np.ndarray:
    """Flatten per-rep feature_weights into a single array matching X_hybrid columns.

    Includes augmentation feature weights when augmentation_info is present.
    The returned array has one weight per column of the full hybrid matrix
    (base features + augmentation features).
    """
    parts = []
    # Base rep weights
    for rep_name in fitted.selected_indices:
        if fitted.feature_weights and rep_name in fitted.feature_weights:
            parts.append(fitted.feature_weights[rep_name])
        else:
            parts.append(np.ones(len(fitted.selected_indices[rep_name])))
    # Augmentation weights (appended after base features, matching X_hybrid layout)
    # NOTE: iteration order must match apply_augmentation_selection (insertion order)
    if fitted.augmentation_info:
        for aug_name in fitted.augmentation_info:
            aug = fitted.augmentation_info[aug_name]
            if 'importance_scores' in aug:
                parts.append(np.asarray(aug['importance_scores']))
            else:
                n_feat = aug.get('n_features', len(aug.get('selected_indices', [])))
                parts.append(np.ones(n_feat))
    return np.concatenate(parts) if parts else np.array([])


def flatten_fold_weights(feat_info) -> np.ndarray:
    """Flatten per-rep importance scores from per-fold feature_info.

    Unlike flatten_weights (which uses FittedResult from global search),
    this uses the feature_info dict returned by create_hybrid / build_fold_hybrid.
    Weights match the columns of the per-fold hybrid matrix exactly.
    """
    parts = []
    for rep_name, info in feat_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        if 'importance_scores' in info:
            parts.append(np.asarray(info['importance_scores']))
        else:
            parts.append(np.ones(info.get('n_features', len(info['selected_indices']))))
    # Augmentation weights
    aug_info = feat_info.get('augmentation_info')
    if aug_info:
        for aug_name, aug in aug_info.items():
            if 'importance_scores' in aug:
                parts.append(np.asarray(aug['importance_scores']))
            else:
                n_feat = aug.get('n_features', len(aug.get('selected_indices', [])))
                parts.append(np.ones(n_feat))
    return np.concatenate(parts) if parts else np.array([])


def build_fold_hybrid(train_reps, val_reps, y_train, blueprint,
                      augmentations_train=None, augmentations_val=None):
    """Per-fold hybrid construction using a search blueprint.

    Properly passes internal_evaluator and augmentations from the blueprint,
    unlike bare create_hybrid calls that only pass allocation/budget/importance.

    Args:
        train_reps: {rep_name: X_train} for this fold.
        val_reps: {rep_name: X_val} for this fold.
        y_train: Training labels for this fold.
        blueprint: Blueprint from search result (e.g. best_by_model[model]['blueprint']).
        augmentations_train: {aug_name: X_train} sliced augmentation reps, or None.
        augmentations_val: {aug_name: X_val} sliced augmentation reps, or None.

    Returns:
        (X_tr_h, X_va_h, feat_info)
    """
    from kirby.hybrid import apply_augmentation_selection

    X_tr_h, feat_info = create_hybrid(
        train_reps, y_train,
        allocation_method=blueprint.allocation_method,
        budget=blueprint.budget,
        importance_method=blueprint.importance_method,
        internal_evaluator=blueprint.internal_evaluator,
        augmentations=augmentations_train,
        augmentation_strategy=blueprint.augmentation_strategy,
    )
    X_va_h = apply_feature_selection(val_reps, feat_info)

    # Apply augmentation selection to val if the hybrid included augmentations
    aug_info = feat_info.get('augmentation_info')
    if aug_info and augmentations_val:
        X_va_aug = apply_augmentation_selection(augmentations_val, aug_info)
        if X_va_aug is not None:
            X_va_h = np.hstack([X_va_h, X_va_aug])

    return X_tr_h, X_va_h, feat_info


# =============================================================================
# HYBRID SEARCH PRECOMPUTE
# =============================================================================

def precompute_hybrid_searches(reps, labels, groups, config,
                               augmentations=None, verbose=True,
                               task_type='auto', metric='auto',
                               domain=None,
                               dataset_name=None, tier='general',
                               adaptive_pruning_schedule=None,
                               model_pruning_aggressiveness=None):
    """Find the best hybrid across all budgets for each augmentation condition.

    ONE find_best_hybrid call per aug_condition with ALL budgets. Baselines
    and importance scores are computed once and reused across budgets.
    best_by_model from each search gives per-model best configs across all budgets.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        config: Dict with keys: search_speed, budgets.
        augmentations: {aug_name: array} or None.
        verbose: Print progress.
        dataset_name: REQUIRED. Used as part of the cache filename
            (e.g. '{dataset_name}__general__uncapped__full.pkl').
        tier: Rep tier label for the cache filename. 'general' (default) or
            'top' (experiments using the top_tier subset).

    Returns:
        Dict mapping aug_cond -> SearchResult. Keys: 'none', and 'uncapped'
        if augmentations are available. Each SearchResult contains results
        across ALL budgets.
    """
    from kirby.find_best_hybrid import find_best_hybrid

    if dataset_name is None:
        raise ValueError(
            "precompute_hybrid_searches requires dataset_name= "
            "(used as part of the cache filename)"
        )

    # Always use speed='full' + FULL_BUDGETS + models=None for precompute.
    # This matches the SLURM precompute cache key regardless of the
    # experiment's own --mode setting. When SLURM cache exists on disk,
    # these calls hit it instantly (no compute).
    budgets = FULL_BUDGETS
    speed = 'full'
    models = None

    # Resolve metric using shared logic (matches get_precomputed_search fallback)
    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    # Precompute aug conditions. Top tier emits {none, uncapped, capped};
    # general tier emits {none, uncapped} only (capped dropped 2026-05-22
    # because the iterative growth loop is ~9× more expensive on the
    # large general-rep pool and no experiment now requires it — all
    # capped consumers were redirected to top/capped).
    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions.append('uncapped')
        if tier == 'top':
            aug_conditions.append('capped')
    if os.environ.get('KIRBY_PRECOMPUTE_REVERSE_ORDER') == '1':
        aug_conditions = list(reversed(aug_conditions))
        if verbose:
            print(f"  KIRBY_PRECOMPUTE_REVERSE_ORDER=1 → "
                  f"iterating aug_conditions in reverse: {aug_conditions}")

    results = {}
    t_start = time.time()

    for idx, aug_cond in enumerate(aug_conditions, 1):
        augs = augmentations if aug_cond != 'none' else None
        # Pass aug_strat verbatim so 'none'/'uncapped'/'capped' each get
        # their own cache filename. Mapping 'none'→'uncapped' collided
        # the no-aug results with real uncapped files on the same path.
        aug_strat = aug_cond

        if verbose:
            print(f"  [{idx}/{len(aug_conditions)}] aug={aug_cond} "
                  f"budgets={budgets} ...", end='', flush=True)

        t0 = time.time()
        try:
            search = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                budgets=budgets,
                models=models,
                augmentations=augs,
                augmentation_strategy=aug_strat,
                speed=speed,
                task_type=task_type,
                metric=metric,
                cache=True,
                verbose=verbose,
                dataset_name=dataset_name,
                tier=tier,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            results[aug_cond] = search
            elapsed = time.time() - t0

            if verbose:
                if search.best_by_model:
                    # Show summary from best across all models
                    all_bests = list(search.best_by_model.values())
                    _cmp = min if search.lower_is_better else max
                    top = _cmp(all_bests, key=lambda r: r.get('score_mean', 0))
                    score = top.get('score_mean', 0)
                    print(f" done {elapsed:.1f}s "
                          f"({len(all_bests)} models, "
                          f"top={top.get('model', '?')} score={score:.4f})")
                else:
                    print(f" done {elapsed:.1f}s (no results)")

        except Exception as e:
            if verbose:
                print(f" FAILED: {e}")
            results[aug_cond] = None

    if verbose:
        total_time = time.time() - t_start
        n_ok = sum(1 for v in results.values() if v is not None)
        print(f"  Precompute complete: {n_ok}/{len(aug_conditions)} "
              f"searches in {total_time:.0f}s")

    return results


def _resolve_search_metric(labels, task_type, metric='auto', domain=None,
                           dataset_name=None):
    """Resolve (metric_name, task_type) for cache alignment.

    Delegates to _resolve_dataset_metric — prefers DATASET_REGISTRY when
    dataset_name is given, falls back to heuristic. Kept as a thin wrapper
    for existing callers that don't yet pass dataset_name.
    """
    metric_name, resolved_task, _ = _resolve_dataset_metric(
        dataset_name=dataset_name, labels=labels, task_type=task_type,
        domain=domain, metric=metric)
    return metric_name, resolved_task


def get_precomputed_search(reps, labels, groups,
                           aug_condition='uncapped', augmentations=None,
                           speed='full', return_fitted=False,
                           weight_method=None, precomputed=None,
                           metric='auto', task_type=None, domain=None,
                           dataset_name=None, tier='general',
                           **kwargs):
    """Retrieve a hybrid search result for a given aug_condition.

    If precomputed dict is passed (from data['precomputed_searches']),
    returns the cached SearchResult directly — instant, no computation.
    Otherwise calls find_best_hybrid with all budgets and cache=True.

    The returned SearchResult contains results across ALL budgets.
    Use search.best_by_model[model] or search.best_by_model_budget[model][budget]
    for per-model best configs.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        aug_condition: 'none', 'uncapped', or 'capped'.
        augmentations: {aug_name: array} or None.
        speed: Search speed tier ('fast' or 'full').
        return_fitted: If True, also fit best config on full data.
        weight_method: Importance method for weights (needed if return_fitted).
        precomputed: Dict from data['precomputed_searches'] (optional).
        metric: Metric name ('auto' resolves from domain/task_type).
            Must match what was used during precomputation for cache alignment.
        task_type: REQUIRED. 'classification' or 'regression'. Pass
            `data['task_type']` from load_data — no auto-detect.
        domain: Domain hint for metric resolution ('molecular', 'antibody', etc.).
        dataset_name: REQUIRED when precomputed is None (falls back to
            find_best_hybrid with cache=True). Used as part of the cache
            filename.
        tier: Rep tier label for the cache filename ('general' or 'top').
        **kwargs: Additional args passed to find_best_hybrid (NOT `budget`;
            per-budget selection is done downstream via best_by_model_budget).

    Returns:
        SearchResult from find_best_hybrid.
    """
    # Fast path: return from in-memory precomputed results
    if precomputed is not None:
        result = precomputed.get(aug_condition)
        if result is not None:
            return result

    # On-demand: call find_best_hybrid with all budgets and cache=True
    from kirby.find_best_hybrid import find_best_hybrid

    if dataset_name is None:
        raise ValueError(
            "get_precomputed_search requires dataset_name= when "
            "precomputed is None (cache filenames are dataset-keyed)"
        )

    # Resolve metric and task_type so cache keys match precomputation
    resolved_metric, resolved_task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    augs = augmentations if aug_condition != 'none' else None
    aug_strat = aug_condition

    # Allow kwargs to override defaults (e.g., budgets, internal_evaluators)
    call_kwargs = dict(
        base_reps=reps,
        labels=labels,
        groups=groups,
        budgets=FULL_BUDGETS,
        augmentations=augs,
        augmentation_strategy=aug_strat,
        speed=speed,
        metric=resolved_metric,
        task_type=resolved_task_type,
        return_fitted=return_fitted,
        weight_method=weight_method,
        cache=True,
        verbose=False,
        dataset_name=dataset_name,
        tier=tier,
    )
    call_kwargs.update(kwargs)

    return find_best_hybrid(**call_kwargs)


def get_best_precomputed_search(reps, labels, groups, augmentations=None,
                                *, speed='full', precomputed=None,
                                metric='auto', task_type=None, domain=None,
                                dataset_name=None, tier='general',
                                return_fitted=False, weight_method=None,
                                **kwargs):
    """Return the better of {'none', 'uncapped'} SearchResults.

    For experiments that want the best downstream performance without
    constraining feature count. Loads both precomputed SearchResults
    (from cache or the precomputed dict), compares their top-model scores,
    and returns the winner. When augmentations are unavailable, returns
    just the 'none' result (uncapped is identical with no augs).

    Does NOT consider 'capped' — capped is for dimension-controlled
    experiments and rarely wins unconstrained. Callers who explicitly
    want capped should use get_precomputed_search(aug_condition='capped').

    Args:
        Same signature as get_precomputed_search, minus aug_condition.
        All other args are passed through to both underlying calls.

    Returns:
        (SearchResult, str) — the winning SearchResult and the condition
        name ('none' or 'uncapped') it came from. Caller can log which
        was chosen for reproducibility.
    """
    has_augs = bool(augmentations)

    s_none = get_precomputed_search(
        reps, labels, groups,
        aug_condition='none', augmentations=None,
        speed=speed, return_fitted=return_fitted, weight_method=weight_method,
        precomputed=precomputed,
        metric=metric, task_type=task_type, domain=domain,
        dataset_name=dataset_name, tier=tier, **kwargs)

    if not has_augs:
        return s_none, 'none'

    s_uncapped = get_precomputed_search(
        reps, labels, groups,
        aug_condition='uncapped', augmentations=augmentations,
        speed=speed, return_fitted=return_fitted, weight_method=weight_method,
        precomputed=precomputed,
        metric=metric, task_type=task_type, domain=domain,
        dataset_name=dataset_name, tier=tier, **kwargs)

    def _top_score(search):
        """Top-across-models best score from a SearchResult. Returns
        +inf / -inf on empty so the other side wins."""
        if search is None or not search.best_by_model:
            return float('inf') if getattr(search, 'lower_is_better', False) else float('-inf')
        scores = [r.get('score_mean') for r in search.best_by_model.values()
                  if r.get('score_mean') is not None
                  and np.isfinite(r.get('score_mean'))]
        if not scores:
            return float('inf') if search.lower_is_better else float('-inf')
        return min(scores) if search.lower_is_better else max(scores)

    none_score = _top_score(s_none)
    unc_score = _top_score(s_uncapped)
    lower_is_better = getattr(s_uncapped, 'lower_is_better',
                              getattr(s_none, 'lower_is_better', False))

    # Tie-break: prefer 'none' because it's smaller and simpler.
    if lower_is_better:
        return (s_none, 'none') if none_score <= unc_score else (s_uncapped, 'uncapped')
    return (s_none, 'none') if none_score >= unc_score else (s_uncapped, 'uncapped')


def get_fitted_hybrid(reps, labels, groups, *, downstream_model,
                      budget=None, aug_condition='uncapped',
                      augmentations=None, speed='full', precomputed=None,
                      internal_evaluators=None, budgets=None,
                      metric='auto', task_type=None, domain=None,
                      dataset_name=None, tier='general',
                      adaptive_pruning_schedule=None,
                      model_pruning_aggressiveness=None,
                      **extra_kwargs):
    """Get X_hybrid and feature_info from a cached search (return_fitted=True).

    For full-dataset experiments that need the hybrid matrix without per-fold splitting.
    Each downstream model gets its own best hybrid — downstream_model is required.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        downstream_model: Model name (e.g., 'rf', 'knn', 'gp'). Required.
            Selects from best_by_model_budget[downstream_model].
        budget: Specific budget to select (None = best across budgets for this model).
        aug_condition: 'none', 'uncapped', or 'capped'.
        augmentations: {aug_name: array} or None.
        speed: Search speed tier ('fast' or 'full').
        precomputed: Dict from data['precomputed_searches'] (optional).
        internal_evaluators: Override evaluator list (e.g., DISTANCE_EVALUATORS).
            When set, bypasses precomputed dict and runs on-demand (disk-cached).
        budgets: Override budget list (e.g., DISTANCE_BUDGETS).
            When set, bypasses precomputed dict and runs on-demand (disk-cached).
        metric: Metric name ('auto' resolves from domain/task_type).
        task_type: REQUIRED. 'classification' or 'regression'. Pass
            `data['task_type']` from load_data — no auto-detect.
        domain: Domain hint for metric resolution.

    Returns:
        (X_hybrid, feature_info, SearchResult)
    """
    # If custom evaluators, budgets, or pruning schedule requested, bypass
    # the in-memory precomputed dict (it was computed with different params).
    # Disk cache handles dedup based on full meta.
    kwargs = dict(extra_kwargs)
    use_precomputed = precomputed
    if internal_evaluators is not None:
        kwargs['internal_evaluators'] = internal_evaluators
        use_precomputed = None  # force on-demand search
    if budgets is not None:
        kwargs['budgets'] = budgets
        use_precomputed = None
    if adaptive_pruning_schedule is not None:
        kwargs['adaptive_pruning_schedule'] = adaptive_pruning_schedule
        use_precomputed = None  # different schedule = different cache key
    if model_pruning_aggressiveness is not None:
        kwargs['model_pruning_aggressiveness'] = model_pruning_aggressiveness
        use_precomputed = None

    search = get_precomputed_search(
        reps, labels, groups,
        aug_condition=aug_condition, augmentations=augmentations,
        speed=speed, return_fitted=True, precomputed=use_precomputed,
        metric=metric, task_type=task_type, domain=domain,
        dataset_name=dataset_name, tier=tier,
        **kwargs)

    best_config = get_best_for_downstream(search, downstream_model, budget=budget)

    if best_config is not None and 'blueprint' in best_config:
        # Build hybrid from the selected blueprint on full data
        from kirby.hybrid import create_hybrid
        bp = best_config['blueprint']
        augs = augmentations if augmentations is not None else None
        X_hybrid, fi = create_hybrid(
            reps, labels,
            allocation_method=bp.allocation_method,
            budget=bp.budget,
            step_size=getattr(bp, 'step_size', 5),
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            augmentations=augs,
            augmentation_strategy=getattr(bp, 'augmentation_strategy', 'uncapped'),
        )
        # NOTE: create_hybrid already includes augmentation features in X_hybrid
        # when augmentations are provided — do NOT append them again.
        return X_hybrid, fi, search

    # Fallback: use per-model fitted_results from search
    fitted = search.fitted_results.get(downstream_model)
    if fitted is None:
        available = list(search.fitted_results.keys()) if search.fitted_results else []
        raise RuntimeError(
            f"No fitted result for model={downstream_model}, aug={aug_condition}. "
            f"Available models: {available}")
    fi = {}
    for rep_name, indices in fitted.selected_indices.items():
        fi[rep_name] = {
            'selected_indices': indices,
            'n_selected': len(indices),
        }
        if fitted.feature_weights and rep_name in fitted.feature_weights:
            fi[rep_name]['feature_weights'] = fitted.feature_weights[rep_name]
    X_hybrid = apply_feature_selection(reps, fi)

    # Include augmentation features if present
    if fitted.augmentation_info and augmentations is not None:
        from kirby.hybrid import apply_augmentation_selection
        fi['augmentation_info'] = fitted.augmentation_info
        X_aug = apply_augmentation_selection(augmentations, fitted.augmentation_info)
        if X_aug is not None:
            X_hybrid = np.hstack([X_hybrid, X_aug])

    return X_hybrid, fi, search


def get_best_for_downstream(search, downstream_model, budget=None):
    """Select the best config from a SearchResult for a specific downstream model.

    Lookup chain:
    1. best_by_model_budget[model][budget] (exact match)
    2. best across all budgets for this model in best_by_model_budget
    3. best_by_budget[budget] (model-agnostic, budget-specific)
    4. best_by_model[model] (model-specific, budget-agnostic)
    5. None — caller must handle.

    Args:
        search: SearchResult from find_best_hybrid.
        downstream_model: Model name (e.g., 'knn', 'gp', 'svm').
        budget: Specific budget to select (None = best across budgets for this model).

    Returns:
        best_config dict with 'blueprint', 'score_mean', etc. or None.
    """
    # Try model-specific budget lookup
    if search.best_by_model_budget and downstream_model in search.best_by_model_budget:
        model_budgets = search.best_by_model_budget[downstream_model]
        if budget is not None and budget in model_budgets:
            return model_budgets[budget]
        # Best across all budgets for this model
        if model_budgets:
            lower = getattr(search, 'lower_is_better', False)
            return (min if lower else max)(
                model_budgets.values(),
                key=lambda r: r.get('score_mean', 0))

    # Fallback: budget-specific (model-agnostic)
    if budget is not None and search.best_by_budget and budget in search.best_by_budget:
        return search.best_by_budget[budget]

    # Fallback: model-specific (budget-agnostic)
    if search.best_by_model and downstream_model in search.best_by_model:
        return search.best_by_model[downstream_model]

    # Cross-model fallback: use overall best blueprint from any model.
    # The blueprint (allocation/importance/budget) is largely model-agnostic —
    # create_hybrid builds the same hybrid regardless of downstream model.
    if search.best_by_model:
        lower = getattr(search, 'lower_is_better', False)
        all_bests = list(search.best_by_model.values())
        return (min if lower else max)(
            all_bests, key=lambda r: r.get('score_mean', 0))

    return None


# =============================================================================
# OPTUNA HP TUNING
# =============================================================================

def tune_xgboost_optuna(X_train, y_train, n_trials=50, n_cv_folds=3,
                        task_type='regression', seed=42, verbose=False):
    """Tune XGBoost hyperparameters with Optuna. Returns best model instance.

    Requires: pip install optuna xgboost
    """
    import optuna
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.model_selection import cross_val_score

    is_clf = task_type == 'classification'
    scoring = 'accuracy' if is_clf else 'r2'
    ModelClass = XGBClassifier if is_clf else XGBRegressor

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': seed,
            'n_jobs': -1,
            'verbosity': 0,
        }
        model = ModelClass(**params)
        scores = cross_val_score(model, X_train, y_train, cv=n_cv_folds,
                                 scoring=scoring)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = ModelClass(**study.best_params, random_state=seed, n_jobs=-1, verbosity=0)
    best.fit(X_train, y_train)
    return best


# =============================================================================
# IO HELPERS
# =============================================================================

def convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, 'to_dict'):
        return convert_numpy(obj.to_dict())
    elif hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return convert_numpy(asdict(obj))
    elif isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj


def save_experiment_results(results: dict, output_dir: str, experiment_id: str,
                            dataset: str = None):
    """Save results JSON to output_dir/experiment_id/{dataset}.json.

    If results contains a 'meta' key, the output uses the structured schema.
    Otherwise, wraps results with a timestamp for traceability.
    """
    import datetime

    out = Path(output_dir) / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}.json" if dataset else "results.json"
    filepath = out / filename

    output = convert_numpy(results)
    if 'meta' not in output:
        output = {'timestamp': datetime.datetime.now().isoformat(), **output}
    else:
        output.setdefault('timestamp', datetime.datetime.now().isoformat())
        output.setdefault('experiment', experiment_id)

    # Atomic write: write to temp file then rename, so parallel jobs
    # can't corrupt each other's output by truncating mid-write.
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=str(out), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(output, f, indent=2)
        os.replace(tmp_path, str(filepath))  # atomic on same filesystem
    except BaseException:
        os.unlink(tmp_path)
        raise
    print(f"  [saved {filepath}]")
    return filepath


def load_experiment_results(output_dir: str, experiment_id: str,
                            dataset: str = None) -> Optional[dict]:
    """Load results JSON from output_dir/experiment_id/{dataset}.json."""
    out = Path(output_dir) / experiment_id
    filename = f"{dataset}.json" if dataset else "results.json"
    filepath = out / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def make_structured_output(experiment_id: str, dataset: str, data: dict,
                           config: dict, results: Any,
                           primary_metric: str = None,
                           lower_is_better: bool = None) -> dict:
    """Build the common structured JSON schema for framework experiments.

    Returns a dict ready for save_experiment_results().
    """
    import datetime
    reps = data.get('reps', {})
    return {
        'experiment': experiment_id,
        'dataset': dataset,
        'domain': data.get('domain', 'molecular'),
        'timestamp': datetime.datetime.now().isoformat(),
        'primary_metric': primary_metric,
        'lower_is_better': lower_is_better,
        'config': convert_numpy(config),
        'results': convert_numpy(results),
        'meta': {
            'n_samples': len(data.get('labels', [])),
            'n_reps': len(reps),
            'rep_dims': {k: v.shape[1] for k, v in reps.items()},
            'task_type': data['task_type'],
        },
    }


def run_across_tasks(run_fn, tasks: List[dict], output_dir: str,
                     experiment_id: str):
    """Run an experiment function across multiple (dataset, rep_preset) tasks.

    Args:
        run_fn: callable(data, config, output_dir, dataset_name) -> results
        tasks: list of dicts with keys: dataset, reps, mode (optional).
        output_dir: base output directory.
        experiment_id: experiment ID for structured output.

    Returns:
        list of (dataset_name, results) tuples.
    """
    all_results = []
    for task in tasks:
        dataset = task['dataset']
        rep_preset = task.get('reps', 'all')
        mode = task.get('mode', 'full')

        print(f"\n{'='*70}")
        print(f"  TASK: {dataset} (reps={rep_preset}, mode={mode})")
        print(f"{'='*70}")

        # Use curated rep list from FRAMEWORK_TASKS if available
        custom_reps = task.get('rep_names') or FRAMEWORK_TASKS.get(dataset)
        data = load_data(dataset, rep_preset, rep_names=custom_reps)
        # Build config from mode
        # Build config from SPEED_TIERS
        from kirby.find_best_hybrid import SPEED_TIERS
        speed = 'full' if mode == 'full' else 'fast'
        tier = SPEED_TIERS[speed]
        config = {
            'n_splits': tier['n_folds'],
            'search_speed': speed,
            'allocations': tier['allocation_methods'],
            'importance_methods': tier['importance_methods'],
            'internal_evaluators': tier['internal_evaluators'],
            'models': ['rf', 'xgboost', 'ridge', 'knn', 'lightgbm', 'mlp',
                        'svm', 'gp', 'catboost', 'dnn', 'cnn'],
        }
        results = run_fn(data, config, output_dir, dataset)
        save_experiment_results(results, output_dir, experiment_id, dataset)
        all_results.append((dataset, results))

    return all_results


def create_experiment_parser(description: str, has_delta: bool = False,
                             has_augment: bool = False) -> argparse.ArgumentParser:
    """Create a standard argument parser for experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mode', choices=['intermediary', 'full'], default='intermediary',
                        help='intermediary (1 split, fast) or full (5-fold CV)')
    parser.add_argument('--dataset', type=str, default='logd',
                        help='Dataset name (logd, caco2, herg, qm9, flab_*, etc.)')
    parser.add_argument('--reps', type=str, default='fast',
                        help='Representation preset (fast, fast+hf, all) or comma-separated list')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    parser.add_argument('--framework', action='store_true',
                        help='Use curated FRAMEWORK_TASKS rep lists (overrides --reps)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model list (overrides mode default)')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list (overrides script default)')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Max folds to evaluate (uses same splitter, stops early)')
    if has_delta:
        parser.add_argument('--delta', action='store_true',
                            help='Generate delta representations for antibody datasets')
    if has_augment:
        parser.add_argument('--augment', action='store_true',
                            help='Include augmented features for antibody datasets')
    return parser


def parse_rep_args(args) -> Tuple[str, Optional[List[str]]]:
    """Parse --reps and --framework args into (rep_preset, rep_names).

    Returns:
        (rep_preset, rep_names) where rep_names is None if using a preset,
        or a list of rep names if using --framework or comma-separated --reps.
    """
    if getattr(args, 'framework', False):
        dataset = args.dataset
        if dataset in FRAMEWORK_TASKS:
            return 'all', FRAMEWORK_TASKS[dataset]
        else:
            print(f"  WARNING: {dataset} not in FRAMEWORK_TASKS, falling back to --reps")

    reps_arg = args.reps
    if ',' in reps_arg:
        return 'all', [r.strip() for r in reps_arg.split(',')]
    return reps_arg, None


def get_config(args) -> dict:
    """Return config dict based on args.mode (fast or full).

    Pulls allocations/importance/evaluators from SPEED_TIERS.
    Models and budgets are experiment-controlled (set via CLI or experiment code).
    """
    from kirby.find_best_hybrid import SPEED_TIERS

    # Always use speed='full' for hybrid search cache alignment with
    # SLURM precompute. --mode only affects the experiment's own CV
    # evaluation (n_splits), not the hybrid search speed.
    speed = 'full'
    tier = SPEED_TIERS[speed]
    default_budgets = FULL_BUDGETS

    # n_splits for experiment CV: 1 for intermediary (quick check), 5 for full
    exp_folds = tier['n_folds'] if args.mode == 'full' else 1

    config = {
        'n_splits': exp_folds,
        'search_speed': speed,
        'allocations': tier['allocation_methods'],
        'importance_methods': tier['importance_methods'],
        'internal_evaluators': tier['internal_evaluators'],
        'budgets': default_budgets,
        # Models for experiment use (baselines, evaluation, etc.)
        # NOTE: precompute_hybrid_searches ignores this — uses models=None
        # so the cache key matches the SLURM precompute.
        'models': ['rf', 'xgboost', 'ridge', 'knn', 'lightgbm', 'mlp',
                    'svm', 'gp', 'catboost', 'dnn', 'cnn'],
    }

    # CLI overrides
    if getattr(args, 'models', None):
        config['models'] = [m.strip() for m in args.models.split(',')]
    if getattr(args, 'budgets', None):
        config['budgets'] = [int(b.strip()) for b in args.budgets.split(',')]
    if getattr(args, 'max_folds', None):
        config['max_folds'] = args.max_folds
    return config
