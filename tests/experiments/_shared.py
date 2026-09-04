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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.find_best_hybrid import find_best_hybrid, SearchResult
from kirby.registry import DATASET_REGISTRY, lookup as registry_lookup

from experiments import dataset_policy  # noqa: F401  (registers dataset skip policy + registry)

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
                           step_size=5, patience=20, greedy_tolerance=0.0,
                           augmentation_margin=0.0,
                           augmentation_fraction=None,
                           dataset_name=None, labels=None, groups=None):
    """Hash fold + hybrid config into a cache key.

    dataset_name, labels and groups are part of the key: row indices and
    representation shapes alone can repeat across two datasets of the same
    size, and the same rows under different labels or a different grouping
    select different features.
    """
    h = hashlib.sha256()
    h.update(train_idx.tobytes())
    h.update(f"ds:{dataset_name}".encode())
    if labels is not None:
        h.update(np.ascontiguousarray(labels).tobytes())
    if groups is not None:
        h.update(np.ascontiguousarray(
            np.asarray(groups).astype('U')).tobytes())
    for name, shape in sorted(rep_shapes.items()):
        h.update(f"{name}:{shape}".encode())
    h.update(f"{allocation_method}|{importance_method}|{budget}|{internal_evaluator}|{step_size}|{patience}|{greedy_tolerance}".encode())
    h.update(f"aug_strategy:{augmentation_strategy}".encode())
    h.update(f"aug_margin:{float(augmentation_margin)}".encode())
    h.update(f"aug_fraction:{augmentation_fraction}".encode())
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
                         groups=None, dataset_name=None,
                         **kwargs):
    """Fold-aware cached create_hybrid.

    Caches feature_info (selected indices) keyed by fold train indices +
    hybrid config. The actual X_hybrid is reconstructed from feature_info
    since it's just index selection on the rep arrays.

    Args:
        train_reps: {rep_name: train_features} for this fold.
        y_train: Training labels for this fold.
        train_idx: np.ndarray of training sample indices (used for cache key).
        groups: this fold's training groups. Without them create_hybrid cuts
            its internal split at random, so each representation's share of
            the budget is decided on rows that share scaffolds with their own
            training rows.
        dataset_name: part of the cache key, so two datasets of the same size
            cannot serve each other's features.
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
        dataset_name=dataset_name, labels=y_train, groups=groups,
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
        train_reps, y_train, groups=groups,
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


def _save_rep_npy(cache_dir: Path, rep_name: str, arr: np.ndarray) -> None:
    """Atomically write one rep array into the per-dataset rep cache."""
    cache_file = cache_dir / f'{rep_name}.npy'
    if cache_file.exists():
        return
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


def _cache_rep_dict(dataset_name: str, compute_all_fn,
                    filter_names: Optional[List[str]] = None,
                    compute_subset_fn=None) -> Dict[str, np.ndarray]:
    """Cache a function that returns {rep_name: array} dict.

    Loads individual reps from cache. When some are missing:
      - if compute_subset_fn is given, only the MISSING reps are generated
        (compute_subset_fn(missing_names) -> {name: array});
      - otherwise compute_all_fn() regenerates the whole preset.

    The subset path matters because compute_all_fn re-enters every rep's
    generator on a single miss. Callers that can narrow their preset to a
    name list should pass compute_subset_fn.
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

        if compute_subset_fn is not None:
            print(f"    Computing {len(missing)} missing rep(s): {missing}")
            new_reps = compute_subset_fn(missing)
            cache_dir.mkdir(parents=True, exist_ok=True)
            for k, v in new_reps.items():
                _save_rep_npy(cache_dir, k, v)
                print(f"    {k}: computed + cached ({v.shape[1]}D)")
            # A rep that failed to generate is simply absent — same
            # drop-and-warn contract as the compute_all_fn path.
            reps.update({k: v for k, v in new_reps.items() if k in wanted})
            return reps

    # Compute all, cache each, return wanted
    print(f"    Computing representations...")
    all_reps = compute_all_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for k, v in all_reps.items():
        _save_rep_npy(cache_dir, k, v)
        print(f"    {k}: computed + cached ({v.shape[1]}D)")

    if filter_names:
        return {k: v for k, v in all_reps.items() if k in filter_names}
    return all_reps


# =============================================================================
# MODE CONFIGURATIONS — derived from the DEFAULT_* constants in find_best_hybrid.py
# =============================================================================
# No standalone config dicts here. Experiments use the DEFAULT_* constants directly.
# The only experiment-level configs are: n_splits, models, budgets.
# These are set per-experiment or via get_config(args).

# Canonical budget lists — used by get_config() and all 4 helpers' run_all().
# Change here to change everywhere.
FULL_BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000]
DISTANCE_BUDGETS = [10, 25, 50, 100, 250, 500]
# Internal evaluators for the DISTANCE-pool hybrid search (s04/s05/s06/s07/
# s10/s13/c03). Ridge REMOVED everywhere (2026-07-16, hybrid rebuild Phase 0.1)
# — the earlier "keep ridge as a distance-pool evaluator" carve-out (2026-06-08)
# is superseded. Removing it changes what the distance-tier search computes, so
# distance-experiment search caches for any re-scored dataset must be deleted
# before re-running (reps/evaluator set are NOT in the cache key).
DISTANCE_EVALUATORS = ['knn', 'rf', 'mlp', 'lightgbm', 'svm']

# The 250-row gate on `capped` was REMOVED 2026-08-27 by the owner.
#
# It existed because the OLD capped refitted a model and ran a
# permutation-importance pass for EVERY candidate feature, which is what
# 'has only ever completed for n<=242' measured. capped was rebuilt on
# 2026-08-25: one ranking fit, one base-ranking fit and five scoring fits
# per fold. The gate outlived its reason and was excluding capped from
# every molecular dataset -- logd 5,028 rows, herg_ki 1,412, buchwald
# 3,955 -- and capped is the only method where an extra feature has to be
# worth more than a base one.
#
# capped is now emitted for every dataset. If a precompute job overruns
# because of it, shorten that job -- do not put the gate back.

# Large-dataset experiment gating (2026-06-28). The uncached experiments
# (c05 learning-curves, c07 partitions, c13 forward-compat) re-search from
# scratch and blow the 7-day wall on big datasets. Above LARGE_DATASET_N the
# gate kicks in: c05 caps its learning curve at LARGE_SUBSAMPLE_K rows instead
# of going to 100% of n, c07 is skipped, c13 subsamples to LARGE_SUBSAMPLE_K
# (c17 already subsamples to 2000 internally). Chosen so only the genuine
# wall-blockers (nt_*/gb_*/DTA, all >5k) are gated; everything <=adme-size runs
# full. qm9 is loaded at 10000 (user decision 2026-09-02, reverting the 5000 that
# rode in on 6064089, an allocator commit) so it now trips this gate: c05, c07 and
# c13 take the gated path for qm9. That is the intended trade -- the qm9 rep pool
# was selected by rep pruning at ~10k, and its scaffold count nearly doubles
# (2,015 -> 3,457) against 5000.
LARGE_DATASET_N = 5000
LARGE_SUBSAMPLE_K = 2000

# NOTE: the large-n feasibility gate (limit 'uncapped'/distance cost on big
# datasets) was prototyped then REVERTED on 2026-06-09 — it was a methodology
# decision that needs a proper discussion with the user first (see
# REPRODUCIBILITY_LEDGER.md §K/§P/§R). No gate is active right now.

# Core downstream models (lit-validated across molecular, antibody, DTA, TCGA, reaction).
# RF + XGBoost (tree ensembles), SVM (kernel), DNN (neural), LightGBM (boosting).
# Ridge REMOVED 2026-07-16 (hybrid rebuild Phase 0.1) — never a downstream model again.
# MLP REPLACED BY DNN 2026-09-04 (owner). DNN settings live in kirby/models/dnn.py:
# (256,128,64), dropout 0.2, lr 1e-3, 200 epochs, batch 256, patience 20, Adam +
# ReduceLROnPlateau, 80/20 internal holdout, targets z-normalised.
#
# This list MUST stay equal to the models=None default set in find_best_hybrid.py
# (tiers 'fast' + 'full'). Every precomputed search runs models=None, so a name
# here that is not in that set gets no blueprint: get_best_search_for_downstream
# returns (None, None) and the experiment writes NO rows for it, silently. gp and
# knn are tier 'extra' and are therefore NOT in this list; name them directly in
# an experiment that wants them.
DOWNSTREAM_MODELS = ['rf', 'xgboost', 'svm', 'dnn', 'lightgbm']


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
# HONEST EVALUATION: NESTED CROSS-VALIDATION
# =============================================================================
# The honest leave-one-fold-out (LOFO) re-score that used to live here has been
# REMOVED (2026-07-13). It was a pure-arithmetic patch over a whole-dataset
# search: the hybrid recipe was chosen and scored on the SAME folds, and LOFO
# re-picked the winner per held-out fold to claw back some of that optimism.
#
# It was never equivalent to holding data out, because the per-fold scores it
# re-selected over came from models trained on data that INCLUDED the held-out
# fold, and the candidate pool had already been pruned using every fold.
#
# The replacement is real nested cross-validation: kirby.find_best_hybrid.
# nested_cv_evaluate. The outer fold is held out, the search only ever sees
# outer-train, and BOTH arms (hybrid and each individual base rep) are scored on
# the untouched outer fold. Experiments call run_nested_cv() below.
#
# Do NOT reintroduce a re-score over a whole-dataset search.


def outer_rep_builders_from_factories(rep_factories, n_samples, names=None):
    """Turn per-fold representation factories into nested-CV outer builders.

    A factory is called factory(train_idx, val_idx, train_labels) and returns
    (train_matrix, val_matrix), with everything it fits — column selection,
    z-score statistics — taken from train_idx alone. nested_cv_evaluate wants
    the other shape: builder(train_idx, y_train) returning the FULL-length
    matrix, which it then slices for the fold itself. Asking the factory to
    featurise every row as its validation half gives exactly that, and every
    row outside train_idx is featurised from training rows only.

    Use this rather than passing rep_factories straight through to
    nested_cv_evaluate. rep_factories reaches the INNER search only, so a
    representation supplied that way is invisible on the outer test fold: the
    hybrid silently loses it there, and the baseline arm never scores it at all.

    Args:
        rep_factories: {name: factory}.
        n_samples: rows in the dataset.
        names: restrict to these names (default: all of rep_factories).

    Returns: {name: builder}.
    """
    all_rows = np.arange(int(n_samples))
    wanted = list(names) if names is not None else sorted(rep_factories)

    def _make(fn):
        def _build(train_idx, y_train):
            _, full = fn(np.asarray(train_idx), all_rows, y_train)
            return np.asarray(full)
        return _build

    return {n: _make(rep_factories[n]) for n in wanted if n in rep_factories}


def run_nested_cv(reps, labels, groups, config, *,
                  dataset_name,
                  augmentations=None,
                  task_type='auto',
                  metric='auto',
                  domain=None,
                  tier='general',
                  budgets=None,
                  internal_evaluators=None,
                  n_outer_folds=None,
                  seed=42,
                  verbose=True,
                  label_noise=None,
                  augmentation_strategy=None,
                  **search_kwargs):
    """Honest hybrid-vs-baseline comparison for one dataset.

    THE standard entry point for experiments. Wraps nested_cv_evaluate with this
    repo's metric/task resolution and config conventions.

    The search runs on outer-train only, per outer fold; the hybrid AND every
    individual base representation are then scored on the untouched outer fold.
    Nothing here is selection-leaked, so the numbers can be reported as-is.

    Pass the FULL reps/labels/groups. Do NOT pre-slice — this function does the
    outer splitting.

    Returns:
        NestedCVResult. Call .to_dict() for the JSON block to persist.
    """
    from kirby.find_best_hybrid import nested_cv_evaluate

    if dataset_name is None:
        raise ValueError("run_nested_cv requires dataset_name= (cache key)")

    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    if budgets is None:
        budgets = FULL_BUDGETS
    if n_outer_folds is None:
        n_outer_folds = config.get('n_splits', 5)

    # Match precompute's cache key exactly (models=None), so the
    # SLURM precompute cache is hit rather than recomputed on the spot.
    kwargs = dict(
        budgets=budgets,
        models=None,
        task_type=task_type,
        metric=metric,
        augmentations=augmentations,
        cache=True,
        tier=tier,
    )
    if augmentation_strategy is not None:
        kwargs['augmentation_strategy'] = augmentation_strategy
    if internal_evaluators is not None:
        kwargs['internal_evaluators'] = internal_evaluators
    if label_noise is not None:
        kwargs['label_noise'] = label_noise
    kwargs.update(search_kwargs)

    return nested_cv_evaluate(
        base_reps=reps,
        labels=labels,
        groups=groups,
        n_outer_folds=n_outer_folds,
        seed=seed,
        verbose=verbose,
        dataset_name=dataset_name,
        **kwargs,
    )


def precompute_nested_cv_searches(reps, labels, groups, config,
                                  augmentations=None, verbose=True,
                                  task_type='auto', metric='auto',
                                  domain=None,
                                  dataset_name=None, tier='general',
                                  adaptive_pruning_schedule=None,
                                  model_pruning_aggressiveness=None,
                                  internal_evaluators=None,
                                  budgets=None,
                                  n_outer_folds=None,
                                  seed=42,
                                  label_noise=None,
                                  objective=None):
    """Warm the per-outer-fold inner-search caches that run_nested_cv reads.

    objective: search objective forwarded to find_best_hybrid —
    'predictive' (the default when None) or 'similarity'. 'similarity_sac'
    was CUT on 2026-08-26 and now raises.
    Without it a geometry study could only ever warm predictive inner searches,
    which is why nothing in the similarity work has run through nested CV.

    Nested CV runs one search per OUTER FOLD, each on a different training
    subset. Precompute must therefore warm n_outer_folds searches per
    (dataset, aug_condition), not one.

    The outer splits and the per-fold cache namespace both come from the shared
    helpers in kirby.find_best_hybrid, so precompute and the experiment cannot
    drift apart — if they did, precompute would warm entries nothing ever reads
    and every experiment would recompute the whole search from cold.

    Returns:
        {aug_cond: {fold_idx: SearchResult}}
    """
    from kirby.find_best_hybrid import (find_best_hybrid,
                                        nested_cv_cache_namespace,
                                        nested_cv_outer_folds)

    if dataset_name is None:
        raise ValueError(
            "precompute_nested_cv_searches requires dataset_name= "
            "(used as part of the cache filename)")

    if budgets is None:
        budgets = FULL_BUDGETS
    models = None

    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    if n_outer_folds is None:
        n_outer_folds = config.get('n_splits', 5)

    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions.append('uncapped')
        if tier == 'general':
            _n = len(labels)
            aug_conditions.append('capped')

    outer_folds = nested_cv_outer_folds(
        reps, labels, groups, n_outer_folds, seed, task_type)
    n_outer = len(outer_folds)

    results = PrecomputeResults()
    t_start = time.time()

    for idx, aug_cond in enumerate(aug_conditions, 1):
        augs = augmentations if aug_cond != 'none' else None
        aug_strat = aug_cond
        results[aug_cond] = {}

        for fold_i, (tr, _te) in enumerate(outer_folds):
            if verbose:
                print(f"  [{idx}/{len(aug_conditions)}] aug={aug_cond} "
                      f"outer-fold {fold_i + 1}/{n_outer} "
                      f"(train={len(tr)}) ...", end='', flush=True)
            t0 = time.time()
            try:
                _fbh_kwargs = dict(
                    base_reps={k: v[tr] for k, v in reps.items()},
                    labels=labels[tr],
                    groups=(groups[tr] if groups is not None else None),
                    budgets=budgets,
                    models=models,
                    augmentations=({k: v[tr] for k, v in augs.items()}
                                   if augs else None),
                    augmentation_strategy=aug_strat,
                    task_type=task_type,
                    metric=metric,
                    cache=True,
                    verbose=verbose,
                    dataset_name=dataset_name,
                    tier=tier,
                    cache_namespace=nested_cv_cache_namespace(
                        fold_i, n_outer, seed),
                    adaptive_pruning_schedule=adaptive_pruning_schedule,
                    model_pruning_aggressiveness=model_pruning_aggressiveness,
                )
                if internal_evaluators is not None:
                    _fbh_kwargs['internal_evaluators'] = internal_evaluators
                if label_noise is not None:
                    _fbh_kwargs['label_noise'] = label_noise
                if objective is not None:
                    _fbh_kwargs['objective'] = objective
                search = find_best_hybrid(**_fbh_kwargs)
                results[aug_cond][fold_i] = search
                if verbose:
                    print(f" done ({time.time() - t0:.1f}s)")
            except Exception as e:
                # Fold value stays None; the exception goes into
                # results.failures (fold index in the record) so the caller
                # can read which folds failed instead of scraping stdout.
                results[aug_cond][fold_i] = None
                rec = results.record_failure(
                    e, tier=tier, dataset=dataset_name, aug_condition=aug_cond)
                rec['fold'] = fold_i
                print(f" FAILED ({type(e).__name__}: {e})")

    if verbose:
        print(f"  precompute (nested CV) total: {time.time() - t_start:.1f}s")
        if results.failures:
            _failed = ', '.join(f"{f['aug_condition']}/fold{f['fold']}"
                                f"={f['error_type']}" for f in results.failures)
            print(f"  Precompute failures recorded ({len(results.failures)}): "
                  f"{_failed}")

    return results


# =============================================================================
# PER-FOLD RECIPE SEARCH
# The whole-dataset pool (get_best_search_for_downstream / get_fitted_pool_hybrid)
# ranks recipes by a score measured over every row, so an experiment that then
# reports a held-out score has chosen its allocation method, importance method,
# internal evaluator and budget using the rows it is about to be scored on.
# These two functions run the same search on ONE fold's TRAINING rows and hand
# back the same (search, metadata) pair, so the recipe never sees the held-out
# fold. Split and cache namespace both come from kirby.find_best_hybrid, so a
# fold search here reads exactly what precompute_nested_cv_searches warms.
# =============================================================================

# In-process memo so one experiment asking for the same fold's search once per
# budget, per model and per condition pays for it once. Keyed by everything that
# changes which search is run; the disk cache still backs it across processes.
_FOLD_SEARCH_MEMO: Dict[tuple, Any] = {}


def search_on_fold(reps, labels, groups, *, train_idx, fold_i, n_outer_folds,
                   dataset_name, augmentations=None, task_type='auto',
                   metric='auto', domain=None, tier='general',
                   budgets=None, internal_evaluators=None, seed=42,
                   verbose=False, label_noise=None, objective=None,
                   adaptive_pruning_schedule=None,
                   model_pruning_aggressiveness=None,
                   cache_namespace=None):
    """Search hybrid recipes on ONE outer fold's TRAINING rows.

    Same per-augmentation-condition loop as precompute_hybrid_searches, with
    two differences: every array is sliced to `train_idx` first, and the search
    is filed under this fold's nested-CV cache namespace so it cannot collide
    with the whole-dataset search or with another fold's.

    Args:
        train_idx: row numbers of this fold's training rows.
        fold_i, n_outer_folds, seed: name this fold's cache namespace. They must
            match the split the caller scores on — build the folds with
            kirby.find_best_hybrid.nested_cv_outer_folds so they do.
        tier: 'general' for the main lineage, 'general_distance' for the
            distance lineage (pass DISTANCE_EVALUATORS and DISTANCE_BUDGETS
            alongside it).
        cache_namespace: overrides the nested-CV namespace. Pass one when the
            training subset is NOT a nested-CV outer fold — a Bayesian search's
            seed rows, say — so the search cannot be served from, or overwrite,
            a cache entry belonging to a different subset of the same size.

    Returns:
        PrecomputeResults, {aug_condition: SearchResult or None}, with one
        `.failures` record per condition that threw.
    """
    from kirby.find_best_hybrid import (find_best_hybrid,
                                        nested_cv_cache_namespace)

    if dataset_name is None:
        raise ValueError("search_on_fold requires dataset_name= (cache key)")

    train_idx = np.asarray(train_idx)
    labels = np.asarray(labels)
    if budgets is None:
        budgets = FULL_BUDGETS

    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    # Same conditions the whole-dataset precompute emits: 'capped' is on the
    # main lineage only, and the distance lineage has never carried it.
    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions.append('uncapped')
        if tier == 'general':
            aug_conditions.append('capped')

    namespace = (cache_namespace if cache_namespace is not None
                 else nested_cv_cache_namespace(fold_i, n_outer_folds, seed))
    results = PrecomputeResults()

    for aug_cond in aug_conditions:
        augs = augmentations if aug_cond != 'none' else None
        try:
            _kw = dict(
                base_reps={k: v[train_idx] for k, v in reps.items()},
                labels=labels[train_idx],
                groups=(np.asarray(groups)[train_idx]
                        if groups is not None else None),
                budgets=budgets,
                models=None,
                augmentations=({k: v[train_idx] for k, v in augs.items()}
                               if augs else None),
                augmentation_strategy=aug_cond,
                task_type=task_type,
                metric=metric,
                cache=True,
                verbose=verbose,
                dataset_name=dataset_name,
                tier=tier,
                cache_namespace=namespace,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            if internal_evaluators is not None:
                _kw['internal_evaluators'] = internal_evaluators
            if label_noise is not None:
                _kw['label_noise'] = label_noise
            if objective is not None:
                _kw['objective'] = objective
            results[aug_cond] = find_best_hybrid(**_kw)
        except Exception as e:
            results[aug_cond] = None
            rec = results.record_failure(e, tier=tier, dataset=dataset_name,
                                         aug_condition=aug_cond)
            rec['fold'] = int(fold_i)
    return results


def get_fold_best_search(reps, labels, groups, *, train_idx, fold_i,
                         n_outer_folds, downstream_model, budget=None,
                         dataset_name, augmentations=None, distance=False,
                         task_type='auto', metric='auto', domain=None,
                         seed=42, verbose=False, objective=None,
                         label_noise=None, budgets=None,
                         internal_evaluators=None,
                         adaptive_pruning_schedule=None,
                         model_pruning_aggressiveness=None,
                         cache_namespace=None):
    """The recipe for one fold, chosen on that fold's TRAINING rows only.

    Drop-in replacement for get_best_search_for_downstream inside a fold loop:
    it returns the same (search, metadata) pair, with the same metadata keys —
    'tier', 'aug_condition', 'score_mean', 'model', 'budget', 'best_config' —
    so a caller keeps reading meta['best_config']['blueprint'] and
    meta['aug_condition'] unchanged.

    Returns (None, None) when no augmentation condition produced a usable
    config for (downstream_model, budget).
    """
    if distance:
        tier = 'general_distance'
        if budgets is None:
            budgets = DISTANCE_BUDGETS
        if internal_evaluators is None:
            internal_evaluators = DISTANCE_EVALUATORS
    else:
        tier = 'general'

    memo_key = (
        dataset_name, tier, cache_namespace,
        int(fold_i), int(n_outer_folds), int(seed),
        tuple(budgets) if budgets is not None else None,
        tuple(internal_evaluators) if internal_evaluators is not None else None,
        objective, str(label_noise) if label_noise is not None else None,
        tuple(sorted(reps.keys())),
        tuple(sorted(augmentations.keys())) if augmentations else None,
        int(len(train_idx)),
    )
    searches = _FOLD_SEARCH_MEMO.get(memo_key)
    if searches is None:
        searches = search_on_fold(
            reps, labels, groups,
            train_idx=train_idx, fold_i=fold_i, n_outer_folds=n_outer_folds,
            dataset_name=dataset_name, augmentations=augmentations,
            task_type=task_type, metric=metric, domain=domain, tier=tier,
            budgets=budgets, internal_evaluators=internal_evaluators,
            seed=seed, verbose=verbose, label_noise=label_noise,
            objective=objective,
            adaptive_pruning_schedule=adaptive_pruning_schedule,
            model_pruning_aggressiveness=model_pruning_aggressiveness,
            cache_namespace=cache_namespace,
        )
        _FOLD_SEARCH_MEMO[memo_key] = searches

    candidates = []
    for aug_cond, search in searches.items():
        if search is None:
            continue
        best = get_best_for_downstream(search, downstream_model, budget=budget)
        if not best or best.get('score_mean') is None:
            continue
        if not np.isfinite(best['score_mean']):
            continue
        candidates.append((aug_cond, search, best))
    if not candidates:
        return None, None

    # Same direction guard and tie-break order as the whole-dataset pool.
    ref_lib = candidates[0][1].lower_is_better
    candidates = [c for c in candidates if c[1].lower_is_better == ref_lib]
    if not candidates:
        return None, None
    aug_pref = {'none': 0, 'uncapped': 1, 'capped': 2}
    candidates.sort(key=lambda item: (
        item[2]['score_mean'] if ref_lib else -item[2]['score_mean'],
        aug_pref.get(item[0], 99)))
    aug_cond, search, best = candidates[0]

    return search, {
        'tier': tier,
        'aug_condition': aug_cond,
        'score_mean': best['score_mean'],
        'model': downstream_model,
        'budget': budget,
        'best_config': best,
        'fold': int(fold_i),
        'n_outer_folds': int(n_outer_folds),
        # The counterpart of get_best_search_for_downstream's 'whole_dataset'.
        # Every part of the recipe here — allocation method, budget, importance
        # method, internal evaluator, augmentation condition — was scored on
        # this fold's training rows only, so a caller that also rebuilds the
        # features per fold can report the held-out score as held out.
        'selection_scope': 'fold_train_rows',
        'selection_note': ('recipe chosen by its mean over inner folds of this '
                           "outer fold's training rows only"),
    }


# =============================================================================
# HONEST (leave-one-fold-out) RE-SCORING
# Restored 2026-08-24: dropped from _shared as collateral in e921e22 (a
# 'remove ridge' commit); recovered verbatim from d471d37, the last commit
# that had it (paired per-fold deltas + hard metric-direction guards).
# ~30 experiments import honest_rescore / build_honest_rescore_input here.
# =============================================================================


def _hr_is_bad(x) -> bool:
    """True if x is None or NaN (or non-numeric). Used to skip bad fold scores."""
    if x is None:
        return True
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return True
    return xf != xf  # NaN != NaN


def _hr_lofo_score(items, lower_is_better, exclude_folds=()):
    """Leave-one-fold-out honest score for one arm (pure arithmetic).

    Args:
        items: list of (name, fold_scores_list). fold_scores may be ragged,
            contain NaN/None, or be empty.
        lower_is_better: metric direction.
        exclude_folds: fold indices that contribute NO held-out term. They are
            still read when averaging the OTHER folds to pick each held-out
            fold's winner — the exclusion is only of the score reported on
            them. Used for fold 0 of a search that filtered its own candidate
            list on fold 0's scores.

    Returns:
        (honest_score, winners_per_fold, n_folds_used, single_fold_flag,
         per_fold)

        honest_score: mean over held-out folds of the LOFO-selected item's
            held-out score; NaN if the arm is empty or nothing scorable.
        winners_per_fold: name selected for each held-out fold (skips folds
            where no item has usable other-fold data, so may be shorter than F).
        n_folds_used: number of held-out folds that contributed a finite score.
        single_fold_flag: True when only one fold is present, so no held-out
            selection is possible and the plain best single-fold score is
            returned instead (the caller should treat it as optimistic).
        per_fold: length-F list, fold-INDEX-aligned, of this arm's held-out
            score on each fold (None where the arm produced none). Aligned so
            the caller can pair hybrid against baseline fold-by-fold; the
            compacted `winners_per_fold` above cannot be used for that.
    """
    # Keep only items with at least one fold score present.
    clean = [(n, [s for s in fs]) for n, fs in items
             if fs is not None and len(fs) > 0]
    if not clean:
        return float('nan'), [], 0, False, []
    excluded = {int(f) for f in (exclude_folds or ())}

    F = max(len(fs) for _, fs in clean)
    # Strictly-better comparison so ties keep the first-seen item (stable).
    if lower_is_better:
        strictly_better = lambda a, b: a < b
    else:
        strictly_better = lambda a, b: a > b

    if F == 1:
        # Only one fold: no held-out fold available. Fall back to the plain
        # best single-fold score and flag it as not-honest.
        if 0 in excluded:
            # The one fold there is, is the fold the candidate list was
            # filtered on. Nothing honest can be read off it.
            return float('nan'), [], 0, True, [None]
        best_name, best_val = None, None
        for n, fs in clean:
            v = fs[0]
            if _hr_is_bad(v):
                continue
            v = float(v)
            if best_val is None or strictly_better(v, best_val):
                best_name, best_val = n, v
        if best_val is None:
            return float('nan'), [], 0, True, [None]
        return best_val, [best_name], 1, True, [best_val]

    held_out = []
    winners = []
    per_fold = [None] * F
    for f in range(F):
        if f in excluded:
            continue
        best_name, best_mean, best_heldout = None, None, None
        for n, fs in clean:
            others = [float(fs[k]) for k in range(len(fs))
                      if k != f and k < len(fs) and not _hr_is_bad(fs[k])]
            if not others:
                continue
            m = sum(others) / len(others)
            if best_mean is None or strictly_better(m, best_mean):
                best_name, best_mean = n, m
                # This item's held-out score on fold f (may be missing/NaN).
                best_heldout = (float(fs[f])
                                if f < len(fs) and not _hr_is_bad(fs[f])
                                else None)
        if best_name is None:
            continue  # no item had usable other-fold data for this fold
        winners.append(best_name)
        if best_heldout is not None:
            held_out.append(best_heldout)
            per_fold[f] = best_heldout

    if not held_out:
        return float('nan'), winners, 0, False, per_fold
    return (sum(held_out) / len(held_out), winners, len(held_out), False,
            per_fold)


def hr_exclude_folds(block):
    """Fold indices a stored honest_rescore block says to drop from both arms.

    A search that filters its own candidate list on fold 0 — every default
    schedule with more than three allocation methods does, and the cluster
    submitters run a schedule that also drops half of each allocation/importance
    pair at three checkpoints — leaves fold 0's hybrid score picked from the
    recipes fold 0 kept. The single representations pass through no such filter.
    build_honest_rescore_input writes [0] into the block in that case.

    A block written before this key existed carries neither 'exclude_folds' nor
    'fold0_pruned'. That reads as [0], not []: the searches those blocks came
    from used the default five allocation methods against a strategy_top_n of 3,
    which prunes. Pass exclude_folds=[] explicitly to override.
    """
    if not isinstance(block, dict):
        return [0]
    if 'exclude_folds' in block:
        return [int(f) for f in (block.get('exclude_folds') or [])]
    if block.get('fold0_pruned') is False:
        return []
    return [0]


def hr_direction(block):
    """Resolve lower_is_better from a stored honest_rescore block — no silent default.

    The historic sign bug came from readers doing bool(block.get('lower_is_better')),
    which turns a MISSING key into False = "higher is better" and silently inverts
    every RMSE/MAE conclusion. Direction is never guessed here:

      - if the block carries 'metric_name' and it is a BUILTIN_METRIC, that table
        is authoritative (and we raise if the stored flag contradicts it);
      - else the stored flag is used, but only if actually present;
      - else raise. A block with no direction is unscoreable, not higher-is-better.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    if not isinstance(block, dict):
        raise TypeError(f"honest_rescore block must be a dict, got {type(block).__name__}")

    metric = block.get('metric_name')
    stored = block.get('lower_is_better')

    if metric in BUILTIN_METRICS:
        canonical = BUILTIN_METRICS[metric][1]
        if stored is not None and bool(stored) != canonical:
            raise ValueError(
                f"metric direction contradiction: metric_name={metric!r} is "
                f"lower_is_better={canonical} per BUILTIN_METRICS, but the block "
                f"stores lower_is_better={bool(stored)}. Refusing to guess — one "
                f"of the two is wrong and the delta sign depends on it.")
        return canonical

    if stored is None:
        raise ValueError(
            "honest_rescore block has neither 'metric_name' nor 'lower_is_better'. "
            "Refusing to default to higher-is-better: that silently inverts the "
            "delta sign on rmse/mae. Persist the direction.")
    return bool(stored)


def honest_rescore(candidates_by_model, baselines_by_model, lower_is_better,
                   metric_name=None, exclude_folds=None,
                   recipe_selection=None):
    """Leave-one-fold-out honest re-score of hybrid vs best-single-rep.

    Pure arithmetic on per-fold scores already computed — NO model fitting, NO
    I/O. Applies the identical held-out-fold tax to BOTH arms.

    Args:
        candidates_by_model: {model: [ {"name": str, "fold_scores": [..]}, .. ]}
            The hybrid CANDIDATE recipes for each model (all recipes searched,
            NOT just the same-fold winner — the winner is re-selected per
            held-out fold). Only include models actually evaluated.
        baselines_by_model: {model: {rep_name: [fold_scores]}}. The baseline
            single reps for each model.
        lower_is_better: metric direction (True for rmse/mae/etc.).
        exclude_folds: fold indices that contribute no term to either arm.
            Pass [0] for a search that pruned its candidate list on fold 0:
            the hybrid arm's fold-0 score is then measured over the recipes
            fold 0 helped keep, while the single representations were never
            filtered, so the two arms would not be paying the same tax.
            build_honest_rescore_input writes this into the persisted block as
            'exclude_folds'; read it back with hr_exclude_folds(block) and pass
            it here. Both arms drop the same folds, so the comparison stays
            paired.
        recipe_selection: what rows chose the hybrid recipe, copied onto every
            model entry as 'recipe_selection'. Pass 'whole_dataset' (the
            'selection_scope' get_best_search_for_downstream returns) whenever
            the candidates rebuild a recipe the search picked over every row:
            the per-fold rebuild fixes the feature weights, not the choice of
            allocation method, budget, importance method or evaluator, so the
            delta is not fully held out and a reader must be told. None when
            the candidates were not chosen on any of these rows.

    Returns:
        {model: {
            'hybrid_honest':          float (NaN if no candidates),
            'best_single_honest':     float (NaN if no baselines),
            'delta':                  float, difference of the two means above,
                                      direction-corrected so + means the hybrid
                                      is better (NaN if either side NaN),

            # --- the paired per-fold view: report these ---
            'hybrid_fold_scores':     [held-out score per fold, fold-aligned,
                                      None where absent],
            'baseline_fold_scores':   [same, for the best single rep],
            'per_fold_delta':         [hybrid - baseline on each fold,
                                      direction-corrected, None where either
                                      arm is missing that fold],
            'delta_mean_of_folds':    float, mean of per_fold_delta over folds
                                      where BOTH arms scored. This is the
                                      paired number to report; it equals
                                      'delta' only when both arms cover the
                                      same folds.
            'n_paired_folds':         int (folds where both arms scored),

            'hybrid_winners_per_fold':   [recipe name per held-out fold],
            'baseline_winners_per_fold': [rep name per held-out fold],
            'n_folds_used':           int (held-out folds that contributed),
            'excluded_folds':         [fold indices dropped from both arms],
            'recipe_selection':       str|None, what rows chose the recipe,
            'single_fold':            bool (True => n_folds==1, scores are the
                                      plain best, NOT honest — no held-out tax),
        }}
        One entry per model appearing in EITHER input dict. Empty candidate or
        baseline sets yield NaN for that arm (and NaN delta), never an error.
    """
    # Direction guard. When the caller names the metric, BUILTIN_METRICS is
    # authoritative and a contradicting flag is a hard error — the sign of every
    # delta below depends on getting this right, and it has been wrong before.
    if metric_name is not None:
        from kirby.find_best_hybrid import BUILTIN_METRICS
        if metric_name in BUILTIN_METRICS:
            canonical = BUILTIN_METRICS[metric_name][1]
            if bool(lower_is_better) != canonical:
                raise ValueError(
                    f"metric direction contradiction: metric_name={metric_name!r} "
                    f"is lower_is_better={canonical} per BUILTIN_METRICS, but "
                    f"lower_is_better={bool(lower_is_better)} was passed.")
            lower_is_better = canonical

    # Both arms drop the same folds. Dropping them from one side only would
    # compare a four-fold mean against a five-fold mean.
    _excluded = sorted({int(f) for f in (exclude_folds or ())})

    models = set(candidates_by_model or {}) | set(baselines_by_model or {})
    out = {}
    for model in sorted(models):
        cand_list = (candidates_by_model or {}).get(model) or []
        cand_items = [
            (c.get('name', f'cand{i}'), c.get('fold_scores'))
            for i, c in enumerate(cand_list)
        ]
        base_items = list(((baselines_by_model or {}).get(model) or {}).items())

        h_score, h_win, h_nf, h_single, h_folds = _hr_lofo_score(
            cand_items, lower_is_better, exclude_folds=_excluded)
        b_score, b_win, b_nf, b_single, b_folds = _hr_lofo_score(
            base_items, lower_is_better, exclude_folds=_excluded)

        if _hr_is_bad(h_score) or _hr_is_bad(b_score):
            delta = float('nan')
        elif lower_is_better:
            delta = b_score - h_score  # lower hybrid == positive delta
        else:
            delta = h_score - b_score  # higher hybrid == positive delta

        # PAIRED per-fold comparison: on each held-out fold, the hybrid picked
        # on the other folds vs the single rep picked on the other folds. Both
        # arms paid the identical held-out tax, so the difference is meaningful
        # fold by fold. `delta` above is a difference of two MEANS, which only
        # equals the mean of these differences when both arms scored the same
        # fold set — it silently diverges when either arm drops a fold.
        # Report `delta_mean_of_folds`; it is the paired quantity.
        n_f = max(len(h_folds), len(b_folds))
        per_fold_delta = [None] * n_f
        paired = []
        for f in range(n_f):
            hv = h_folds[f] if f < len(h_folds) else None
            bv = b_folds[f] if f < len(b_folds) else None
            if hv is None or bv is None:
                continue
            d = (bv - hv) if lower_is_better else (hv - bv)
            per_fold_delta[f] = d
            paired.append(d)

        out[model] = {
            'hybrid_honest': h_score,
            'best_single_honest': b_score,
            'delta': delta,
            # Fold-index-aligned; None = that arm produced no held-out score.
            'hybrid_fold_scores': h_folds,
            'baseline_fold_scores': b_folds,
            'per_fold_delta': per_fold_delta,
            'delta_mean_of_folds': (sum(paired) / len(paired) if paired
                                    else float('nan')),
            'n_paired_folds': len(paired),
            'hybrid_winners_per_fold': h_win,
            'baseline_winners_per_fold': b_win,
            'n_folds_used': max(h_nf, b_nf),
            'single_fold': bool(h_single or b_single),
            'excluded_folds': list(_excluded),
            'recipe_selection': recipe_selection,
        }
    return out


def build_honest_rescore_input(search, *, skip_secondary=True):
    """Build the standard honest_rescore JSON block from a SearchResult.

    Groups hybrid candidates from search.all_results by model (each carries its
    own per-fold 'fold_scores'), and reads per-fold baseline scores from
    search.baseline_fold_scores (additive field; empty on old cache files).
    Only models actually present are included — the model set is never assumed.

    Returns a dict with the exact keys honest_rescore() expects, ready to
    json.dump under the top-level "honest_rescore" key:

        {"metric_name": str,
         "lower_is_better": bool,
         "fold0_pruned": bool | None,
         "exclude_folds": [int, ...],
         "candidates_by_model": {model: [{"name", "fold_scores"}, ...]},
         "baselines_by_model":  {model: {rep: [fold_scores]}}}

    'exclude_folds' is [0] whenever the search filtered its candidate list using
    fold 0's scores, and [] when it provably did not. Fold 0's hybrid score is
    otherwise read off recipes fold 0 helped keep, while the single
    representations were never filtered — the two arms would not be paying the
    same held-out tax. A SearchResult unpickled from a file written before
    fold0_pruned existed reports None, and None is treated as pruned: the
    default schedule with five allocation methods and strategy_top_n=3 prunes.

    metric_name is persisted alongside the flag so a reader can cross-check the
    direction against BUILTIN_METRICS (see hr_direction) instead of trusting a
    bare bool that may be missing from an old file — a missing flag reads as
    False = higher-is-better and silently inverts every rmse/mae delta.
    """
    metric = getattr(search, 'metric_name', None)
    # Direction is READ from the search, never defaulted. If the search carries a
    # known metric, BUILTIN_METRICS wins and a contradiction is a hard error.
    lower = getattr(search, 'lower_is_better', None)
    from kirby.find_best_hybrid import BUILTIN_METRICS
    if metric in BUILTIN_METRICS:
        canonical = BUILTIN_METRICS[metric][1]
        if lower is not None and bool(lower) != canonical:
            raise ValueError(
                f"SearchResult direction contradiction: metric_name={metric!r} is "
                f"lower_is_better={canonical} per BUILTIN_METRICS, but the search "
                f"carries lower_is_better={bool(lower)}.")
        lower = canonical
    elif lower is None:
        raise ValueError(
            "SearchResult carries neither a known metric_name nor lower_is_better; "
            "refusing to default to higher-is-better (inverts rmse/mae deltas).")
    lower = bool(lower)

    candidates_by_model: Dict[str, list] = {}
    for r in (getattr(search, 'all_results', None) or []):
        model = r.get('model')
        folds = r.get('fold_scores')
        if model is None or not folds:
            continue
        # Stable recipe id from the swept config (winner label only).
        name = "{}|{}|b{}|{}|s{}".format(
            r.get('allocation_method'), r.get('importance_method'),
            r.get('budget'), r.get('internal_evaluator'), r.get('step_size'))
        candidates_by_model.setdefault(model, []).append(
            {'name': name, 'fold_scores': list(folds)})

    # search.baseline_fold_scores is {rep: {model: [folds]}} — invert to
    # {model: {rep: [folds]}}.
    baselines_by_model: Dict[str, dict] = {}
    bfs = getattr(search, 'baseline_fold_scores', None) or {}
    for rep_name, per_model in bfs.items():
        if rep_name.startswith('_'):
            continue  # skip reserved keys, mirrors extract_top3_base_reps
        for model, folds in (per_model or {}).items():
            if not folds:
                continue
            baselines_by_model.setdefault(model, {})[rep_name] = list(folds)

    # None when the SearchResult predates the field; hr_exclude_folds reads
    # None as pruned, because the default schedule prunes.
    _f0 = getattr(search, 'fold0_pruned', None)
    return {
        'metric_name': metric,
        'lower_is_better': lower,
        'fold0_pruned': (None if _f0 is None else bool(_f0)),
        'exclude_folds': ([] if _f0 is False else [0]),
        'candidates_by_model': candidates_by_model,
        'baselines_by_model': baselines_by_model,
    }


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

def _should_stratify(task_type, labels):
    """Decide whether a k-fold split should be class-stratified.

    Two-tier rule (ISS-84):
      1. If `task_type` is given, it is authoritative — 'classification'
         stratifies, 'regression' never does. This is the correct signal:
         the loader / registry already knows the task, and the label *dtype*
         is a poor proxy (a regression target can be integer-typed, e.g.
         counts/ordinals, and a classification target can be float-typed,
         e.g. 0.0/1.0). Passing task_type removes both failure modes.
      2. If `task_type` is None (the many existing positional callers that
         predate this argument), fall back to the historical integer-dtype
         heuristic so their behaviour is unchanged.
    """
    if task_type == 'classification':
        return True
    if task_type == 'regression':
        return False
    # Heuristic fallback (task_type not supplied by caller).
    return labels is not None and np.issubdtype(
        np.asarray(labels).dtype, np.integer)


def cv_split(groups, n_splits, X, labels, seed=42, mode=None, *,
             task_type=None):
    """Yield (train_idx, val_idx) using a group-aware CV strategy.

    - n_splits=1: single 80/20 GroupShuffleSplit (scaffold/site-grouped).
    - n_splits>1: StratifiedGroupKFold for classification (class-balanced
      *and* group-respecting), GroupKFold otherwise.

    `groups` is REQUIRED and must have at least `max(2, n_splits)` distinct
    values. There is NO random-split fallback: per project rule
    (feedback_no_random_splits.md) random KFold/ShuffleSplit is forbidden for
    any production result, so an absent or under-resolved `groups` is a loud
    error rather than a silent degradation to random splitting. If a caller
    genuinely has no grouping it must be fixed upstream (assign scaffold/site
    groups in the loader) — not papered over here.

    `task_type` (keyword-only, ISS-84) selects the k-fold strategy
    authoritatively: 'classification' -> StratifiedGroupKFold (classes stay
    balanced across folds while groups are preserved — required for TCGA
    classification e.g. tcga_brca_pam50 / tcga_luad_vs_lusc and any other
    grouped classification task), 'regression' -> plain GroupKFold. When it is
    left None (existing positional callers), the strategy falls back to the
    historical integer-label heuristic, so no existing caller changes
    behaviour. Pass task_type explicitly for classification datasets whose
    labels are not integer-typed, which the heuristic would otherwise miss.
    StratifiedGroupKFold itself remains group-respecting (no sample's group
    spans train+test) — stratification only rebalances *which* groups go to
    each fold.

    The historical `mode` argument is accepted only for backward-compatibility
    with existing callers and is ignored: the split strategy is fully
    determined by `n_splits` (it always was — `mode` never altered behaviour).
    Passing it now emits a DeprecationWarning so callers can drop it.
    """
    if mode is not None:
        warnings.warn(
            "cv_split(mode=...) is vestigial and ignored; the split strategy "
            "is determined entirely by n_splits. Drop the `mode` argument.",
            DeprecationWarning, stacklevel=2,
        )

    n_groups_seen = 0 if groups is None else len(np.unique(groups))
    required = max(2, n_splits)
    if groups is None or n_groups_seen < required:
        raise ValueError(
            f"cv_split: group-aware splitting requires groups with >= "
            f"{required} distinct values for n_splits={n_splits}, but got "
            f"{'groups=None' if groups is None else f'{n_groups_seen} distinct group(s)'}. "
            "Random splitting is forbidden (feedback_no_random_splits.md) — "
            "fix the caller to supply scaffold/site/identity groups instead of "
            "falling back to a random split."
        )

    if n_splits == 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        yield from gss.split(X, labels, groups=groups)
        return

    # Use StratifiedGroupKFold for classification (preserves class balance
    # within grouped folds). Falls back to GroupKFold if unavailable.
    if _should_stratify(task_type, labels):
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            yield from StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(X, labels, groups=groups)
            return
        except ImportError:
            pass
    yield from GroupKFold(n_splits=n_splits).split(X, labels, groups=groups)


def _cv_split(groups, n_folds, X, labels, seed=42, *, task_type=None):
    """Legacy wrapper — delegates to cv_split (group-aware, no random fallback).

    Forwards the keyword-only `task_type` (ISS-84) so callers using the
    `_cv_split` alias can also request classification stratification.
    """
    yield from cv_split(groups, n_folds, X, labels, seed=seed,
                         task_type=task_type)


def make_cv_splitter(n_folds, groups=None, seed=42, *, task_type=None):
    """Return (splitter, use_groups) for direct sklearn CV loops.

    `groups` is REQUIRED and must have >= max(2, n_folds) distinct values.
    There is NO random-split fallback (feedback_no_random_splits.md): an
    absent or under-resolved `groups` raises rather than silently degrading
    to a random KFold/ShuffleSplit. `use_groups` is always True on success;
    callers must pass `groups` into `splitter.split(X, y, groups=groups)`.

    `task_type='classification'` (keyword-only, ISS-84) returns a
    StratifiedGroupKFold so classes stay balanced across grouped folds; this
    is compatible with the existing `splitter.split(X, y, groups=groups)`
    call convention (StratifiedGroupKFold also consumes `y`). Anything else
    (including the default None used by all current callers, which are
    regression splits) returns plain GroupKFold — unchanged behaviour.
    """
    n_groups_seen = 0 if groups is None else len(np.unique(groups))
    required = max(2, n_folds)
    if groups is None or n_groups_seen < required:
        raise ValueError(
            f"make_cv_splitter: group-aware splitting requires groups with "
            f">= {required} distinct values for n_folds={n_folds}, but got "
            f"{'groups=None' if groups is None else f'{n_groups_seen} distinct group(s)'}. "
            "Random splitting is forbidden (feedback_no_random_splits.md) — "
            "fix the caller to supply scaffold/site/identity groups."
        )
    if n_folds == 1:
        return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), True
    if task_type == 'classification':
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            return StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=seed), True
        except ImportError:
            pass
    return GroupKFold(n_splits=n_folds), True


# =============================================================================
# MOLECULAR GROUPING ALTERNATIVES (Butina clustering — ISS-85)
# =============================================================================

def butina_clusters(smiles_list, cutoff: float = 0.6,
                    radius: int = 2, n_bits: int = 2048):
    """Group molecules by RDKit Butina (Taylor-Butina) Tanimoto clustering.

    A lab-standard, group-aware alternative/complement to Murcko-scaffold
    splitting for the molecular domain (ISS-85). Two molecules with Tanimoto
    similarity >= `cutoff` (on Morgan/ECFP fingerprints) tend to land in the
    same cluster, so using the returned cluster ids as CV `groups` keeps
    near-duplicate / analog molecules together in a fold — tighter than a
    pure scaffold split when many molecules share a scaffold but differ in
    substituents, and it catches analog pairs that have *different* Murcko
    scaffolds.

    Butina is an exclusion-sphere clustering: it repeatedly picks the molecule
    with the most neighbours within the distance threshold (1 - cutoff) as a
    new cluster centroid and removes its neighbourhood. Singletons (molecules
    with no neighbour within the threshold) each become their own one-member
    cluster — handled naturally (every input row gets a cluster id, so the
    returned array is exactly len(smiles_list); no row is dropped).

    Args:
        smiles_list: list of SMILES strings. Invalid SMILES are tolerated:
            each invalid entry is given its own singleton cluster id (it has
            no fingerprint, so it cannot share a cluster — this keeps the
            output length aligned and avoids silently merging parse failures).
        cutoff: Tanimoto similarity threshold in [0, 1]. Higher cutoff =>
            stricter => more, smaller clusters (Butina distance = 1 - cutoff).
            0.6 is the common ECFP default (Butina 1999; RDKit cookbook).
        radius: Morgan fingerprint radius (2 == ECFP4).
        n_bits: fingerprint length.

    Returns:
        (groups, n_clusters): groups is an int np.ndarray of length
        len(smiles_list) with a cluster id per molecule (0..n_clusters-1,
        contiguous, assignment order = Butina cluster order with invalid /
        singleton molecules appended); n_clusters is the number of distinct
        clusters. Suitable to pass straight into cv_split / GroupKFold as
        `groups`.

    Wiring (how to select this instead of the default scaffold split):
        This is provided as an OPTION and deliberately does NOT change the
        default split of any existing molecular dataset (the default remains
        Murcko scaffold via assign_scaffold_groups in
        tests/find_best_hybrid_molecular.py — changing a dataset's split
        silently would change every published result for it). To use Butina
        for a molecular run, an experiment/loader that has the SMILES list can
        override the groups after load_data, e.g.:

            data = load_data('herg_ki', rep_names=[...])
            from experiments._shared import butina_clusters
            data['groups'], data['n_groups'] = butina_clusters(
                data['smiles'], cutoff=0.6)
            data['n_scaffolds'] = data['n_groups']

        (find_best_hybrid_molecular.assign_scaffold_groups is the function to
        mirror if/when a CLI flag for split-type is added there — that file is
        owned elsewhere, so the flag wiring is left as a documented follow-up.)
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    smiles_list = list(smiles_list)
    n = len(smiles_list)

    # Build fingerprints; track which rows are valid (parseable).
    fps = []
    valid_rows = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits))
        valid_rows.append(i)

    groups = np.full(n, -1, dtype=int)

    if len(fps) >= 2:
        # Condensed lower-triangle distance matrix (1 - Tanimoto), as RDKit
        # Butina.ClusterData expects.
        dists = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)
        clusters = Butina.ClusterData(
            dists, len(fps), 1.0 - cutoff, isDistData=True)
        # clusters is a tuple of tuples of *local* indices into `fps`.
        for cid, member_local_idxs in enumerate(clusters):
            for local_i in member_local_idxs:
                groups[valid_rows[local_i]] = cid
    elif len(fps) == 1:
        groups[valid_rows[0]] = 0

    # Any still-unassigned row (invalid SMILES, or the degenerate <2-fp case)
    # gets its own singleton cluster so the output stays length-n and no two
    # unparseable molecules are silently merged.
    next_id = int(groups.max()) + 1 if (groups >= 0).any() else 0
    for i in range(n):
        if groups[i] < 0:
            groups[i] = next_id
            next_id += 1

    return groups, int(len(np.unique(groups)))


def sequence_identity_clusters(seqs, identity_threshold: float = 0.95,
                               min_groups: int = 3):
    """Group protein/antibody sequences by near-identity (ISS-29/30).

    Exact-identity grouping (the current antibody `_identity_groups`) keeps
    *byte-identical* sequences together but puts two sequences that differ by
    a single residue into DIFFERENT groups, so analog / near-duplicate pairs
    leak across a GroupKFold train/test boundary. Measured on
    flab_thermo_tresanco (672 VHH rows, 556 unique): 899 row-level near-dup
    pairs (heavy chain Levenshtein distance <= 2), of which 784 (87%) span a
    train/test split under 5-fold exact-identity grouping. This helper closes
    that gap: any two sequences with normalised identity >= `identity_threshold`
    are placed in the same cluster (single-linkage), so near-duplicates stay
    on the same side of every fold.

    Normalised identity = 1 - levenshtein(a, b) / max(len(a), len(b)). Greedy
    single-linkage with length-band bucketing keeps it tractable for the
    antibody dataset sizes here (hundreds to a few thousand sequences); it is
    O(n^2) within a length band in the worst case but bands are small.

    Args:
        seqs: list of sequence strings (e.g. heavy chains; for paired data,
            concatenate heavy+light before calling so both chains count).
        identity_threshold: in (0, 1]. 1.0 reproduces exact-identity grouping
            (no near-dup merging). 0.95 merges sequences <=5% different — a
            sensible default that collapses single/double point-mutant analogs
            without over-merging genuinely distinct antibodies. Lower it to be
            stricter about leakage (bigger clusters), raise it to keep clusters
            tighter.
        min_groups: if the resulting cluster count is below this (degenerate,
            e.g. everything merged on a tiny near-clonal panel), fall back to a
            deterministic hash bucketing so GroupKFold still has enough groups —
            mirrors the existing `_identity_groups` min_groups guard.

    Returns:
        (groups, n_clusters): int np.ndarray of length len(seqs) with a cluster
        id per sequence, contiguous 0..n_clusters-1, plus the cluster count.
        Drop-in for use as CV `groups`.

    Wiring (ISS-29/30 fix, APPLIED — this helper is on the live path):
        tests/antibody_baseline_survey.py::_identity_groups is the single
        function that produces antibody CV groups for the `identity` strategy
        (FLAb panels, abcov, humanness) and it delegates to this helper. Paired
        datasets are handled there by concatenating heavy+light before calling;
        unpaired datasets pass heavy-only (empty light), which reduces to
        heavy with no behaviour change. tests/experiments/opig_humanness.py also
        routes its grouping through _identity_groups, so every antibody
        identity-strategy path uses this near-duplicate clustering. Strategies
        that fall back to identity when their metadata is missing
        (pre_split / edit_distance / experiment / position_modulo, see
        _assign_groups) reach it too.
    """
    seqs = list(seqs)
    n = len(seqs)
    if n == 0:
        return np.zeros(0, dtype=int), 0

    # MANDATORY Levenshtein -- no difflib fallback. difflib.SequenceMatcher
    # computes 2*matching_blocks/(len_a+len_b), NOT edit distance, so it yields
    # a DIFFERENT near-duplicate clustering -> different groups_hash/CV folds
    # than a machine that has Levenshtein. The ARC server was MISSING the
    # package and silently used difflib, so its antibody folds (and leakage
    # control) diverged from local and the cache self-drifted. Fail loud rather
    # than silently change the science. Pinned in setup.py / requirements.
    from Levenshtein import distance as _lev

    def _norm_identity(a, b):
        m = max(len(a), len(b))
        if m == 0:
            return 1.0
        return 1.0 - _lev(a, b) / m

    # Bucket by length band: two sequences can only reach `identity_threshold`
    # if their lengths are within (1 - threshold) * max_len of each other, so
    # we only need to compare within a band. Use exact length as the key and
    # also probe neighbouring lengths within the allowed gap.
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, s in enumerate(seqs):
        by_len[len(s)].append(i)
    lengths = sorted(by_len.keys())

    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # First merge exact duplicates cheaply (within each length band).
    for L in lengths:
        idxs = by_len[L]
        seen = {}
        for i in idxs:
            s = seqs[i]
            if s in seen:
                _union(seen[s], i)
            else:
                seen[s] = i

    # Then near-duplicate single-linkage across compatible length bands.
    for li, L in enumerate(lengths):
        idxs_L = by_len[L]
        # candidate partner lengths: those within the max allowed length gap
        for L2 in lengths[li:]:
            if L2 == L:
                partners = idxs_L
                same_band = True
            else:
                if (L2 - L) / max(L2, 1) > (1.0 - identity_threshold):
                    break  # lengths sorted: further ones only get more distant
                partners = by_len[L2]
                same_band = False
            for ai, i in enumerate(idxs_L):
                start = ai + 1 if same_band else 0
                for j in partners[start:]:
                    if _find(i) == _find(j):
                        continue
                    if _norm_identity(seqs[i], seqs[j]) >= identity_threshold:
                        _union(i, j)

    # Relabel roots to contiguous ids. Order clusters CANONICALLY by their
    # lexicographically-smallest member sequence (unique per cluster), so the
    # integer labels depend only on the PARTITION CONTENT -- not on input row
    # order. This makes groups_hash reproducible across runs/machines and
    # invariant to row permutation. It changes only label VALUES, never which
    # samples co-cluster, so the CV folds are identical (science-neutral).
    members = defaultdict(list)
    for i in range(n):
        members[_find(i)].append(i)
    ordered_roots = sorted(members, key=lambda r: min(seqs[j] for j in members[r]))
    root_to_id = {r: new_id for new_id, r in enumerate(ordered_roots)}
    groups = np.array([root_to_id[_find(i)] for i in range(n)], dtype=int)
    n_clusters = len(root_to_id)

    # Degenerate-collapse guard (mirror _identity_groups' min_groups behaviour).
    if n_clusters < min_groups:
        import hashlib
        n_buckets = max(min_groups, n // 5)
        groups = np.array([
            int(hashlib.sha256(s.encode()).hexdigest(), 16) % n_buckets
            for s in seqs
        ])
        n_clusters = len(np.unique(groups))

    return groups, int(n_clusters)


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

    # `mode` is vestigial in cv_split (split strategy is set by n_folds); do
    # not forward it, to avoid the cv_split deprecation warning per fold.
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
                # Outer CV is group-aware (cv_split above); the internal
                # train/val split must be too, else it falls back to a random
                # 20% carve and ranks features under scaffold leakage.
                groups=groups[train_idx] if groups is not None else None,
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
    internal_evaluator: str = 'match',
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
) -> Tuple[float, float, float, float, int]:
    """Like evaluate_cv but also returns train score.

    Returns (val_mean, val_std, train_mean, train_std, n_features).

    `internal_evaluator` defaults to 'match' (the canonical default used
    across the hybrid search). It was previously 'ridge'; Ridge is forbidden
    as an internal hybrid-search evaluator (feedback_no_ridge.md), so the
    default no longer silently selects it. Callers that want a specific
    evaluator (e.g. c09 forwarding a blueprint's evaluator) still pass it
    explicitly.
    """
    from kirby.hybrid import apply_augmentation_selection

    if metric_fn is None:
        metric_fn = r2_score

    sample_X = next(iter(base_reps.values()))
    val_scores = []
    train_scores = []
    n_feats = []

    # `mode` is vestigial in cv_split (split strategy is set by n_folds); do
    # not forward it, to avoid the cv_split deprecation warning per fold.
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed):
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
                # Same as evaluate_cv: keep create_hybrid's internal split
                # group-aware instead of the random 20% fallback.
                groups=groups[train_idx] if groups is not None else None,
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
    # `mode` is vestigial in cv_split; do not forward it.
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels):
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


def get_model_family_variants(
    task_type: str = 'regression',
) -> Dict[str, List[Tuple[str, callable, float]]]:
    """Return model families with complexity gradients, per task_type.

    Each family: list of (variant_name, factory_fn, capacity_index), where a
    larger capacity_index means a higher-capacity / less-regularised variant.

    `task_type` selects regressor vs classifier estimators so callers (c06,
    etc.) study a capacity gradient with the right model class — previously
    this hardcoded regressors, so on classification data the variants were
    trained on integer labels and scored with classification metrics, which
    is silently wrong (audit CXA-6).

    The linear family uses Lasso (alpha sweep) for regression and
    LogisticRegression (C sweep) for classification — both give a capacity
    gradient over a downstream linear model. Ridge was replaced by Lasso here
    (2026-07-18) so ridge is gone everywhere; LinearRegression could not be used
    because it has no regularisation knob and so cannot produce the capacity
    gradient this study needs.
    """
    is_clf = (task_type == 'classification')

    if is_clf:
        from sklearn.ensemble import RandomForestClassifier as _RF
        from sklearn.linear_model import LogisticRegression as _Logit
        from sklearn.neighbors import KNeighborsClassifier as _KNN
        from sklearn.neural_network import MLPClassifier as _MLP
    else:
        from sklearn.ensemble import RandomForestRegressor as _RF
        from sklearn.linear_model import Lasso
        from sklearn.neighbors import KNeighborsRegressor as _KNN
        from sklearn.neural_network import MLPRegressor as _MLP

    families = {
        'RF': [
            ('depth=3', lambda: _RF(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1), 1),
            ('depth=5', lambda: _RF(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1), 2),
            ('depth=10', lambda: _RF(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), 3),
            ('depth=None', lambda: _RF(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1), 4),
        ],
        'kNN': [
            ('k=20', lambda: _KNN(n_neighbors=20, n_jobs=-1), 1),
            ('k=10', lambda: _KNN(n_neighbors=10, n_jobs=-1), 2),
            ('k=5', lambda: _KNN(n_neighbors=5, n_jobs=-1), 3),
            ('k=1', lambda: _KNN(n_neighbors=1, n_jobs=-1), 4),
        ],
        'MLP': [
            ('16', lambda: _MLP(hidden_layer_sizes=(16,), random_state=42, max_iter=500), 1),
            ('64', lambda: _MLP(hidden_layer_sizes=(64,), random_state=42, max_iter=500), 2),
            ('128,64', lambda: _MLP(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500), 3),
            ('256,128,64', lambda: _MLP(hidden_layer_sizes=(256, 128, 64), random_state=42, max_iter=500), 4),
        ],
    }

    # Linear family: capacity rises with weaker regularisation. For regression
    # this is Lasso (small alpha = high capacity); for classification it is
    # LogisticRegression (large C = high capacity). No Ridge anywhere.
    if is_clf:
        families['Logit'] = [
            ('C=0.01', lambda: _Logit(C=0.01, max_iter=1000), 1),
            ('C=0.1', lambda: _Logit(C=0.1, max_iter=1000), 2),
            ('C=1.0', lambda: _Logit(C=1.0, max_iter=1000), 3),
            ('C=100', lambda: _Logit(C=100.0, max_iter=1000), 4),
        ]
    else:
        families['Lasso'] = [
            ('alpha=100', lambda: Lasso(alpha=100.0, max_iter=5000), 1),
            ('alpha=10', lambda: Lasso(alpha=10.0, max_iter=5000), 2),
            ('alpha=1', lambda: Lasso(alpha=1.0, max_iter=5000), 3),
            ('alpha=0.01', lambda: Lasso(alpha=0.01, max_iter=5000), 4),
        ]

    try:
        if is_clf:
            from xgboost import XGBClassifier as _XGB
        else:
            from xgboost import XGBRegressor as _XGB
        families['XGBoost'] = [
            ('depth=2', lambda: _XGB(n_estimators=100, max_depth=2, random_state=42, n_jobs=-1, verbosity=0), 1),
            ('depth=4', lambda: _XGB(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0), 2),
            ('depth=6', lambda: _XGB(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0), 3),
            ('depth=None', lambda: _XGB(n_estimators=100, max_depth=0, random_state=42, n_jobs=-1, verbosity=0, tree_method='hist'), 4),
        ]
    except ImportError:
        pass

    # SVM: C controls capacity (low C = high regularisation = low capacity)
    if is_clf:
        from sklearn.svm import SVC as _SVC
        families['SVM'] = [
            ('C=0.01', lambda: _SVC(kernel='rbf', C=0.01, probability=False), 1),
            ('C=0.1', lambda: _SVC(kernel='rbf', C=0.1, probability=False), 2),
            ('C=1.0', lambda: _SVC(kernel='rbf', C=1.0, probability=False), 3),
            ('C=10.0', lambda: _SVC(kernel='rbf', C=10.0, probability=False), 4),
        ]
    else:
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
        from sklearn.gaussian_process.kernels import RBF as _RBF, WhiteKernel as _WK
        if is_clf:
            from sklearn.gaussian_process import GaussianProcessClassifier as _GPC
            families['GP'] = [
                ('ls=10', lambda: _GPC(kernel=_RBF(length_scale=10.0), random_state=42), 1),
                ('ls=1', lambda: _GPC(kernel=_RBF(length_scale=1.0), random_state=42), 2),
                ('ls=0.1', lambda: _GPC(kernel=_RBF(length_scale=0.1), random_state=42), 3),
                ('ls=0.01', lambda: _GPC(kernel=_RBF(length_scale=0.01), random_state=42), 4),
            ]
        else:
            from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
            families['GP'] = [
                ('ls=10', lambda: _GPR(
                    kernel=_RBF(length_scale=10.0) + _WK(), n_restarts_optimizer=2, random_state=42), 1),
                ('ls=1', lambda: _GPR(
                    kernel=_RBF(length_scale=1.0) + _WK(), n_restarts_optimizer=2, random_state=42), 2),
                ('ls=0.1', lambda: _GPR(
                    kernel=_RBF(length_scale=0.1) + _WK(), n_restarts_optimizer=2, random_state=42), 3),
                ('ls=0.01', lambda: _GPR(
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
    """Estimate I(X; Y) with a robust kNN estimator.

    History: this previously ran a hand-rolled joint KSG estimator (Kraskov
    et al. 2004) on PCA-reduced features. With ``max_dim`` up to 50 it operated
    far outside the KSG validity regime (joint dimension >> the ~6 dims KSG
    tolerates for n in the low hundreds), so the estimate was systematically
    negative and clamped to 0. The result was that *every* MI value across
    c01/c09/c17/s03 came out 0 — the metric was dead. See
    COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.

    This replacement uses scikit-learn's per-feature Kraskov estimator
    (``mutual_info_regression`` / ``mutual_info_classif``), summed over a set of
    PCA-decorrelated components. It (a) does not collapse to 0, (b) auto-detects
    a discrete target and uses the classification estimator (the old code scaled
    class labels as if continuous, which was wrong for humanness/tcga/tz), and
    (c) keeps the n >> d guard so larger feature blocks can still carry more
    total information. Returns total estimated information in nats (>= 0).

    Args:
        k: neighbours for the Kraskov estimator (sklearn ``n_neighbors``).
        max_dim: cap on PCA components, further capped at n_samples // 5.
    """
    from sklearn.feature_selection import (
        mutual_info_regression, mutual_info_classif)

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    n_samples, d = X.shape
    if n_samples < 10:
        return 0.0  # too few samples for any stable MI estimate
    k = max(1, min(k, n_samples - 1))

    X_scaled = StandardScaler().fit_transform(X)

    # Keep the n >> d guard, but PCA-decorrelate so per-feature MI can be summed
    # without double-counting linearly-shared information.
    max_components = max(1, min(max_dim, n_samples // 5))
    if d > max_components:
        from sklearn.decomposition import PCA as _PCA
        X_scaled = _PCA(n_components=max_components,
                        random_state=42).fit_transform(X_scaled)

    # Detect a discrete target: few unique values and (near-)integer encoded.
    y_unique = np.unique(y)
    is_discrete = (y_unique.size <= max(20, int(0.05 * n_samples))
                   and np.allclose(y, np.round(y)))

    def _mi(target):
        if is_discrete:
            return float(np.sum(mutual_info_classif(
                X_scaled, target.astype(int), n_neighbors=k, random_state=42)))
        return float(np.sum(mutual_info_regression(
            X_scaled, target, n_neighbors=k, random_state=42)))

    mi_raw = _mi(y)

    # Permutation bias correction. The summed per-feature kNN estimator has a
    # positive finite-sample bias that grows with the number of components, so
    # it does NOT cancel when comparing feature blocks of different sizes
    # (e.g. s03 full-vs-selected) and can swamp weak signal at small n. Subtract
    # the mean MI under label permutation (the null) to de-bias. Deterministic
    # seeds keep the estimate reproducible across runs.
    n_perm = 3
    null = np.mean([_mi(np.random.default_rng(1000 + s).permutation(y))
                    for s in range(n_perm)])

    return float(max(0.0, mi_raw - null))


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
    # L1 linearity descriptor: how far a plain linear fit is from the target.
    # LinearRegression (OLS) replaces Ridge here (2026-07-18) so ridge is gone
    # everywhere — this is a measurement of linear predictability, not a model
    # in the study.
    from sklearn.linear_model import LinearRegression
    linear = LinearRegression()
    linear.fit(X_for_model, y)
    y_pred_linear = linear.predict(X_for_model)
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


# ── Large-n / domain experiment-scope policy (2026-06-09; user-approved) ──
# The similarity (s*) and neighbor-based noise (n*) experiments rest on the
# premise that "points close together in representation space have similar
# labels". Two rules decide where they are out of scope. The similarity/noise
# drivers (run_all) call should_skip_experiment() before each experiment and,
# on a skip, write the returned {'skipped':...} dict as the result marker so the
# wave-audit counts it as INTENTIONALLY-SKIPPED, not failed/missing. Full
# rationale + the noise neighbor-vs-model classification: REPRODUCIBILITY_LEDGER.md §E.
#
# Threshold matches s01's existing S01_MAX_DENSE_N so the family uses one cutoff;
# 15000 keeps qm9 (<=10k) and herg (7772) in, catches DTA (~30k) + graphinity (50k).
LARGE_N_SKIP_THRESHOLD = 15000

# Rule 1 (science): on NUCLEOTIDE datasets the similarity premise is unreliable
# -> skip ALL similarity + the neighbor/similarity-based noise experiments.
# (s09 is the cross-dataset aggregator, not a per-dataset experiment, so omitted.)
# KEPT on nucleotide (model/label-based, premise-independent): n06 (bindingdb-only),
# n07, n09, n13.
NUCLEOTIDE_SKIP_EXPERIMENTS = frozenset({
    's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's10', 's11', 's12', 's13',
    'n01', 'n02', 'n03', 'n04', 'n05', 'n08', 'n10', 'n11', 'n12', 'n14',
})

# Rule 2 (memory): these build an UNCAPPED n x n distance matrix -> OOM at large
# n. Skip on ANY dataset with n > LARGE_N_SKIP_THRESHOLD (covers non-nucleotide
# big sets — DTA, graphinity — which Rule 1 does not touch).
LARGE_N_OOM_SKIP_EXPERIMENTS = frozenset({'s02', 's07', 's08', 'n01', 'n05', 'n10', 'n14'})


def should_skip_experiment(exp_id, data):
    """Return a {'skipped':...} marker dict if (exp_id, data) is out of scope per
    the large-n/domain policy above, else None.

    exp_id is the short id ('s02' / 'n01'); data is the loaded dataset dict
    (must carry 'domain' and 'labels'). The marker is JSON-saved by the caller so
    monitoring sees a present, intentionally-skipped result.
    """
    exp = str(exp_id).lower()
    domain = data.get('domain', 'molecular')
    try:
        n = int(len(data.get('labels', [])))
    except Exception:
        n = 0

    if domain == 'nucleotide' and exp in NUCLEOTIDE_SKIP_EXPERIMENTS:
        return {
            'skipped': True,
            'reason': (f'nucleotide domain: similarity premise unreliable for '
                       f'sequence embeddings (exp={exp})'),
            'policy': 'nucleotide_premise',
            'domain': domain,
            'n_samples': n,
        }

    if exp in LARGE_N_OOM_SKIP_EXPERIMENTS and n > LARGE_N_SKIP_THRESHOLD:
        return {
            'skipped': True,
            'reason': (f'large-n OOM guard: n={n} > {LARGE_N_SKIP_THRESHOLD} '
                       f'builds an uncapped n x n matrix (exp={exp})'),
            'policy': 'large_n_oom',
            'domain': domain,
            'n_samples': n,
        }

    return None


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


def _smoke_subsample(data: dict) -> dict:
    """Test-only row cap, gated behind the KIRBY_SMOKE_N env var (default off).

    When KIRBY_SMOKE_N is set to a positive integer N and the dataset has more
    than N rows, deterministically keep N evenly-spaced rows (preserving class /
    scaffold diversity across the original order — NOT a random split). This
    exists purely so the experiment suite can be smoke-tested locally on tiny
    subsets; it is a no-op in any normal/server run where the env var is unset.
    """
    import os as _os
    raw = _os.environ.get('KIRBY_SMOKE_N', '').strip()
    if not raw:
        return data
    try:
        n_keep = int(raw)
    except ValueError:
        return data
    if n_keep <= 0:
        return data
    labels = data.get('labels')
    if labels is None:
        return data
    n = len(labels)
    if n <= n_keep:
        return data
    idx = np.linspace(0, n - 1, n_keep).round().astype(int)
    idx = np.unique(idx)

    def _take(v):
        if isinstance(v, np.ndarray) and len(v) == n:
            return v[idx]
        if isinstance(v, list) and len(v) == n:
            return [v[i] for i in idx]
        return v

    # Per-row top-level keys
    for key in ('labels', 'groups', 'heavy_seqs', 'light_seqs', 'sequences',
                'seqs', 'smiles', 'sample_ids', 'sample_weights'):
        if key in data and data[key] is not None:
            data[key] = _take(data[key])
    # Rep dicts (and augmentation dict)
    for key in ('reps', 'augmentations'):
        v = data.get(key)
        if isinstance(v, dict):
            data[key] = {r: (arr[idx] if isinstance(arr, np.ndarray)
                             and len(arr) == n else arr)
                         for r, arr in v.items()}
    # Recompute group counts if present
    if 'groups' in data and isinstance(data['groups'], np.ndarray):
        n_uniq = int(len(np.unique(data['groups'])))
        if 'n_groups' in data:
            data['n_groups'] = n_uniq
        if 'n_scaffolds' in data:
            data['n_scaffolds'] = n_uniq
    print(f"  [SMOKE] KIRBY_SMOKE_N={n_keep}: subsampled {n} -> {len(idx)} rows")
    return data


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

    # Row-count guard: every representation must have exactly one row per label.
    # A stale on-disk rep cache with the wrong length (e.g. from an older loader
    # version) would otherwise be silently reused and misalign features from
    # their labels — the exact silent-corruption class that has cost real time
    # here. Raise loudly (non-destructive) naming the offending rep(s) and the
    # purge command, rather than guessing or recomputing automatically. Checked
    # BEFORE _smoke_subsample so it sees the full, pre-subsample arrays.
    _reps = data.get('reps')
    _labels = data.get('labels')
    if _reps and _labels is not None:
        _n_labels = len(_labels)
        _bad = {rn: int(arr.shape[0]) for rn, arr in _reps.items()
                if hasattr(arr, 'shape') and arr.ndim >= 1
                and arr.shape[0] != _n_labels}
        if _bad:
            raise ValueError(
                f"Row-count mismatch in '{dataset_name}': {_n_labels} labels but "
                f"representation row counts {_bad}. This is almost always a stale "
                f"rep cache from an older loader version being silently reused. "
                f"Purge it and re-run: "
                f"python tests/purge_rep_cache.py {dataset_name} --delete"
            )

    return _smoke_subsample(data)


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
            compute_subset_fn=lambda names: get_tcga_representations(
                expression, gene_names, sample_ids, wsi_dir,
                ','.join(names), cohort=cohort,
            ),
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
        _failed_reps = []  # (rep, error) for REQUESTED reps that raised on load

        # Pass 1: standard antibody reps (heavy_seqs, light_seqs)
        for rn in _wanted_reps:
            if rn in ANTIBODY_REPS:
                try:
                    fn = ANTIBODY_REPS[rn]
                    reps[rn] = _cache_rep(dataset_name, rn,
                                          lambda _fn=fn: _fn(heavy_seqs, light_seqs))
                except Exception as e:
                    print(f"  WARNING: {rn} failed: {e}")
                    _failed_reps.append((rn, str(e)))
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
                        _failed_reps.append((rn, str(e)))

        # A2: a requested rep that failed to load is dropped — WARN, do not halt
        # (user preference 2026-06-09). NOTE: a dropped rep changes rep_names in
        # the cache key and can trigger a cache rebuild for this dataset; pin
        # network/HF reps offline to avoid intermittent drops.
        if _failed_reps:
            print(f"  WARNING: {dataset_name}: DROPPED {len(_failed_reps)} "
                  f"requested rep(s) that failed to load: "
                  f"{[r for r, _ in _failed_reps]} — rep_names changed, may "
                  f"rebuild this dataset's cache. First error: {_failed_reps[0][1]}")

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
            # What the folds hold out, in words, carried from the survey
            # loader. On the Jain panel, tresanco and marks the 95%-identity
            # clustering leaves one antibody per group, so the folds hold out
            # antibodies rather than homology families — a reader of the result
            # file cannot tell that from the strategy name alone.
            'group_strategy': data.get('group_strategy'),
            'group_scheme': data.get('group_scheme'),
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
            compute_subset_fn=lambda names: get_nuc_representations(
                sequences, ','.join(names)),
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
            create_reaction_rep_variant,
        )
        from kirby.representations.reaction import _MOL_REP_VARIANTS

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

        def _variant_generator(rep_name):
            """Fallback for reactant_/product_/diff_<molrep> reps not in the
            explicit registry. Returns a zero-arg generator or None. The
            molecular-rep variants (reactant_avalon, diff_mhg_gnn, …) are
            applicable to any reaction dataset with rxn_smiles; the ecfp4/maccs/
            pdv variants already have dedicated entries above and never reach
            here. Matrices for these are typically precomputed + cached."""
            for _role in ('reactant', 'product', 'diff'):
                _prefix = _role + '_'
                if rep_name.startswith(_prefix):
                    _molrep = rep_name[len(_prefix):]
                    if _molrep in _MOL_REP_VARIANTS:
                        return lambda: create_reaction_rep_variant(
                            rxn_smiles, _molrep, _role)
            return None

        reps = {}
        _failed_reps = []  # (rep, error) for REQUESTED reps that raised on load
        for name in wanted:
            gen = _rxn_generators.get(name) or _variant_generator(name)
            if gen is None:
                print(f"  WARNING: unknown reaction rep '{name}', skipping")
                continue
            try:
                reps[name] = _cache_rep(dataset_name, name,
                                        lambda _g=gen: _g())
            except Exception as e:
                print(f"  WARNING: {name} failed: {e}")
                _failed_reps.append((name, str(e)))

        # A2: dropped reps -> WARN, do not halt (user preference). A dropped rep
        # changes rep_names and may rebuild this dataset's cache.
        if _failed_reps:
            print(f"  WARNING: {dataset_name}: DROPPED {len(_failed_reps)} "
                  f"requested reaction rep(s): {[r for r, _ in _failed_reps]} — "
                  f"rep_names changed, may rebuild. First error: {_failed_reps[0][1]}")

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
            # The reaction SMILES and the parsed components the reps were built
            # from. Carried through so an experiment that needs a structural
            # reference can compute one from the chemistry instead of from the
            # representations it is grading — n01's cliff reference did the
            # latter, which made the cliffs a function of whichever rep
            # contributed the most features.
            'rxn_smiles': rxn_smiles,
            'components': components,
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

    elif dataset_name == 'cdr_flexibility':
        # ITsFlexible CDR-flexibility (OPIG). Has its own raw loader (not the
        # antibody survey loader). SPLIT = option A (user 2026-07-19): cluster on
        # the CDR3 LOOP sequences at 80% identity, NOT the whole Fv — antibody
        # frameworks are ~98% conserved, so whole-Fv clustering collapses to a
        # near-random singleton split (see opig_cdr_flexibility.py:229-246).
        from kirby.datasets.itsflexible import load_itsflexible
        from kirby.representations.antibody import ANTIBODY_REPS, ANTIBODY_FAST_REPS
        from kirby.representations.mutation import MUTATION_REPS

        data = load_itsflexible(mode='cdr', cdr_types=['CDRH3', 'CDRL3'])
        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = np.asarray(data['labels'])
        cdr_seqs = data['cdr_seqs']

        # Option A: group by CDR3-loop sequence identity (80%). The min_groups=3
        # hash-fallback guarantees a valid GroupKFold even if too few loops merge.
        groups, n_groups = sequence_identity_clusters(
            cdr_seqs, identity_threshold=0.80, min_groups=3)

        if rep_preset in ('fast', 'skip_lm'):
            _rep_names_from_preset = sorted(ANTIBODY_FAST_REPS)
        else:
            _rep_names_from_preset = sorted(ANTIBODY_REPS.keys())
        _wanted_reps = rep_names if rep_names else _rep_names_from_preset

        reps = {}
        _failed_reps = []
        for rn in _wanted_reps:
            if rn in ANTIBODY_REPS:
                try:
                    fn = ANTIBODY_REPS[rn]
                    reps[rn] = _cache_rep(dataset_name, rn,
                                          lambda _fn=fn: _fn(heavy_seqs, light_seqs))
                except Exception as e:
                    print(f"  WARNING: {rn} failed: {e}")
                    _failed_reps.append((rn, str(e)))
            elif rn not in MUTATION_REPS:
                print(f"  WARNING: antibody rep '{rn}' not found, skipping")
        if _failed_reps:
            print(f"  WARNING: {dataset_name}: DROPPED {len(_failed_reps)} "
                  f"requested rep(s): {[r for r, _ in _failed_reps]} — rep_names "
                  f"changed, may rebuild this dataset's cache.")

        # Split-quality guard: a near-all-singleton split is a random split in
        # disguise. Report the structure so the validity number can be trusted
        # (or rejected). NOT a hard gate — the user judges the number.
        _sizes = np.unique(groups, return_counts=True)[1]
        print(f"  {dataset_name}: {len(heavy_seqs)} CDRs, {n_groups} identity "
              f"groups (largest={int(_sizes.max())}, "
              f"singletons={int((_sizes == 1).sum())}/{n_groups}), {len(reps)} reps")

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

    else:
        from find_best_hybrid_molecular import (
            load_dataset as load_molecular_dataset,
            get_representations as get_molecular_representations,
            assign_scaffold_groups,
        )

        from find_best_hybrid_molecular import (
            FACTORY_REPS as _MOL_FACTORY_REPS,
            get_rep_factories as _get_mol_rep_factories,
        )

        smiles, labels, task_type = load_molecular_dataset(dataset_name)
        groups, n_scaffolds = assign_scaffold_groups(smiles)

        # The eleven sim_* similarity recipes, PCA and the pdv/sns reps are
        # never returned by get_representations: each one fits on the training
        # rows of a fold, so it has no whole-dataset matrix to cache. Asking for
        # one used to leave it silently absent — a run over the recipes came
        # back holding only the general representations, under the recipes'
        # name. Build them as per-fold factories instead and hand them to the
        # caller separately.
        _requested = list(rep_names) if rep_names else None
        if _requested is not None:
            _factory_names = [r for r in _requested if r in _MOL_FACTORY_REPS]
            _matrix_names = [r for r in _requested if r not in _MOL_FACTORY_REPS]
        else:
            _factory_names, _matrix_names = [], None

        reps = {}
        if _matrix_names is None or _matrix_names:
            reps = _cache_rep_dict(
                dataset_name,
                lambda: get_molecular_representations(smiles, rep_preset),
                filter_names=_matrix_names,
                compute_subset_fn=lambda names: get_molecular_representations(
                    smiles, ','.join(names)),
            )

        fold_factories = {}
        if _factory_names:
            fold_factories = _get_mol_rep_factories(
                smiles, ','.join(_factory_names), dataset_name=dataset_name)
            _missing = [r for r in _factory_names if r not in fold_factories]
            if _missing:
                # get_rep_factories prints and swallows a factory that raised.
                # Refuse rather than hand back a short list under the name the
                # caller asked for.
                raise RuntimeError(
                    f"{dataset_name}: {len(_missing)} per-fold representation(s) "
                    f"{_missing} were requested but could not be built. They "
                    f"have no whole-dataset matrix, so a run without them is "
                    f"not the run that was asked for.")

        print(f"  {dataset_name}: {len(smiles)} samples, {n_scaffolds} scaffolds, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")
        if fold_factories:
            print(f"    + {len(fold_factories)} per-fold representation(s): "
                  f"{', '.join(sorted(fold_factories))}")

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
        _clash = sorted(set(fold_factories) & set(result['rep_factories']))
        if _clash:
            raise RuntimeError(
                f"{dataset_name}: {_clash} is both a per-fold representation and "
                f"a fine-tuning factory; one would overwrite the other.")
        result['rep_factories'].update(fold_factories)
        # Named separately from the fine-tuning factories so a caller running
        # nested CV can turn exactly these into outer_rep_builders.
        result['fold_fitted_reps'] = sorted(fold_factories)
        return result


# =============================================================================
# SIMILARITY-SPECIFIC HELPERS
# =============================================================================

def get_hybrid_features_and_weights(
    reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    weight_method: str = 'lightgbm_gain',
    budget: Optional[int] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    metric: str = 'rmse',
    dataset_name: Optional[str] = None,
    tier: str = 'general',
    task_type: Optional[str] = None,
    augmentation_strategy: Optional[str] = None,
    objective: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, SearchResult]:
    """Run find_best_hybrid and return hybrid features + flat importance weights.

    `dataset_name`/`tier` are forwarded so `find_best_hybrid(cache=True)` can
    locate its cache file. Required (the cache layer raises ValueError without
    them); pass `dataset_name=None` only if the caller can tolerate raising.
    `augmentation_strategy` should be set explicitly when `augmentations` is
    None to avoid the cache-key collision documented in
    `project_cache_collision_fix.md`.
    """
    kwargs = dict(
        base_reps=reps,
        labels=labels,
        groups=groups,
        return_fitted=True,
        weight_method=weight_method,
        verbose=False,
        cache=True,
        metric=metric,
        dataset_name=dataset_name,
        tier=tier,
    )
    if task_type is not None:
        kwargs['task_type'] = task_type
    if objective is not None:
        kwargs['objective'] = objective
    if budget is not None:
        kwargs['budgets'] = [budget]
    if augmentations is not None:
        kwargs['augmentations'] = augmentations
    if augmentation_strategy is not None:
        kwargs['augmentation_strategy'] = augmentation_strategy
    elif augmentations is None:
        # Explicit 'none' avoids colliding with real uncapped-augmentation
        # cache files when augmentations are absent.
        kwargs['augmentation_strategy'] = 'none'

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


# Preference order for the single "plain base representation" control arm that
# the noise experiments compare their treatments against. One per domain:
# molecular, DTA/protein-descriptor, sequence.
NOISE_BASE_REP_PREFERENCE = ('ecfp4', 'pdv', 'blosum')


def pick_noise_base_rep(reps, dataset_name=None, verbose=True):
    """Choose the plain base representation a noise experiment uses as control.

    Was five copies of::

        for candidate in ['ecfp4', 'pdv', 'blosum']:
            if candidate in reps: base = candidate; break
        else: base = sorted(reps.keys())[0]

    Two problems with the fallback. It is alphabetical, so WHICH rep becomes the
    control depends on nothing but spelling; and it can select a structure-tuned
    similarity recipe (`sim_*`), which is a TREATMENT built by the study — as
    the control arm, silently. A recipe named 'sim_all' sorts ahead of 'smited',
    so simply registering the recipes for a dataset could have swapped the
    control out from under every noise result on it.

    This keeps the documented preference order, refuses to fall back onto a
    structure-tuned recipe, and says out loud which rep it picked and why.

    Raises ValueError when nothing is eligible, rather than picking something
    arbitrary: a noise experiment with no plain base representation has no
    control arm, and should stop instead of reporting against a treatment.
    """
    for candidate in NOISE_BASE_REP_PREFERENCE:
        if candidate in reps:
            return candidate

    # Structure-tuned similarity recipes are study treatments, not controls.
    eligible = sorted(n for n in reps if not str(n).startswith('sim_'))
    if not eligible:
        raise ValueError(
            f"pick_noise_base_rep: no eligible base representation for "
            f"{dataset_name or '<dataset>'}. Preference order "
            f"{NOISE_BASE_REP_PREFERENCE} is absent and every remaining rep "
            f"({sorted(reps)}) is a structure-tuned sim_* recipe, which is a "
            f"treatment in this study and must not become the control arm. "
            f"Add a plain base representation to this dataset's rep list.")

    chosen = eligible[0]
    if verbose:
        print(f"  [base-rep] {dataset_name or ''}: none of "
              f"{NOISE_BASE_REP_PREFERENCE} present; falling back to "
              f"{chosen!r} (first alphabetically among "
              f"{len(eligible)} non-sim_* reps)")
    return chosen


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


# =============================================================================
# PLANTED DECOY DONORS
#
# Two donors that a working admission rule must take NOTHING from. Both were
# written for scratchpad/aug_decoy_check.py and are moved here so c21 and the
# framework augmentation experiments share one implementation instead of three.
#
# They are offered alongside the real donors in the same pool, so they cost no
# extra conditions. What they buy is a count per fold: how many pure-noise
# features and how many near-copies of base features survived selection. On
# logd and herg_ki the shipping default took up to 5 of each per fold.
#
# The matched-shape RANDOM donor is a different thing and is not here: it is a
# separate arm (real donors replaced by random ones of the same shape), which
# isolates the selection from the extra model capacity. c21 builds that in
# _build_conditions.
# =============================================================================

NOISE_DONOR = '__gaussian__'
ECHO_DONOR = '__echo__'


def make_noise_donor(n_samples, n_features, rng):
    """A donor of pure Gaussian noise, `n_features` wide.

    Width is normally matched to the total width of the real donors, so the
    noise competes on equal terms in the pooled ranking rather than being
    drowned out. Anything taken from this donor is a defect.
    """
    return rng.standard_normal((n_samples, n_features))


def make_echo_donor(base_X, n_cols, jitter, rng):
    """Features copied out of the base representation and roughed up slightly.

    Each copied feature gets Gaussian noise added at `jitter` times that
    feature's own standard deviation, so a duplicate-detector cannot find it by
    an exact match. This is the only decoy that catches a rule rewarding
    features the base already carries; no amount of pure noise reveals that.

    Features with no spread are skipped: a constant feature is not a duplicate
    of anything, it is dead, and letting one in makes the decoy easier than it
    should be.

    Returns None when the base has no varying feature at all.
    """
    sd = base_X.std(axis=0)
    live = np.flatnonzero(sd > 0)
    if live.size == 0:
        return None
    take = rng.choice(live, size=min(n_cols, live.size), replace=False)
    echo = base_X[:, take].astype(float, copy=True)
    echo += rng.standard_normal(echo.shape) * (sd[take] * jitter)
    return echo


def plant_decoys(donors, base_X, rng, echo_cols=64, echo_jitter=0.05):
    """Add both decoy donors to a {name: features} pool. Returns a new dict.

    The noise donor is matched in width to the real donors combined. Build the
    decoys full length, once per fold with a fresh seed, so the same features
    can be sliced for the training rows, the admission holdout and the outer
    fold.
    """
    out = dict(donors)
    n_real = sum(v.shape[1] for v in donors.values())
    n_samples = base_X.shape[0]
    if n_real > 0:
        out[NOISE_DONOR] = make_noise_donor(n_samples, n_real, rng)
    echo = make_echo_donor(base_X, echo_cols, echo_jitter, rng)
    if echo is not None:
        out[ECHO_DONOR] = echo
    return out


def decoy_counts(augmentation_info):
    """How many features this fold kept from each decoy donor.

    Returns {'noise': int, 'echo': int}. Both should be 0.
    """
    info = augmentation_info or {}

    def _n(name):
        d = info.get(name) or {}
        return int(len(np.asarray(d.get('selected_indices', ()))))

    return {'noise': _n(NOISE_DONOR), 'echo': _n(ECHO_DONOR)}


def build_fold_hybrid(train_reps, val_reps, y_train, blueprint,
                      augmentations_train=None, augmentations_val=None,
                      budget_override=None, aug_per_source_budget=None,
                      *, groups=None, model=None, metric_fn=None,
                      lower_is_better=None, objective=None, is_clf=None,
                      candidate_ranking=None, augmentation_threshold=None,
                      augmentation_budget=None, aug_max_candidates=None,
                      precomputed_importance=None, precomputed_alloc_scores=None):
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
        budget_override: If set, apply the blueprint's recipe at this budget
            instead of `blueprint.budget`. Used by c15's fine-grained sweep
            (double-descent detection at non-canonical budgets like 75, 150,
            300, 750, etc.) where a single fixed recipe is applied across
            many budgets to isolate budget as the only variable.
        aug_per_source_budget: int or {aug_name: N}, forwarded to the uncapped
            augmentation path. Defaults to the blueprint's own value, so a
            rebuild reproduces the configuration the search chose. Pass it
            explicitly only to override. getattr guards blueprints unpickled
            from caches written before the field existed.
        groups: Group labels ALREADY SLICED to this fold's training rows
            (i.e. `groups[train_idx]`), one entry per row of y_train. Forwarded
            to create_hybrid, which uses it for the internal admission holdout:
            with groups it takes the first fold of a GroupKFold, without it it
            falls back to a RandomState(42) 20% RANDOM split. On a
            scaffold-split dataset that fallback judges keep/drop decisions on
            a holdout sharing scaffolds with its own training rows, so leaving
            this None reintroduces the leak the outer split exists to prevent.
        model: The real downstream estimator INSTANCE (not a factory) that this
            fold's hybrid will be fed to. Forwarded to create_hybrid, which
            hands it to _evaluate_allocation: with a model, accept/reject is
            scored by fitting that model and calling metric_fn; without one it
            falls back to the internal evaluator's r2_score / accuracy_score.
            Also resolves internal_evaluator='match'. Requires metric_fn.
        metric_fn: The dataset's real metric, called as metric_fn(y_true,
            y_pred). Only used by create_hybrid when `model` is also given.
        lower_is_better: Direction for metric_fn. create_hybrid negates the
            score when True (hybrid.py:1374), and it applies that ONLY on the
            model branch — so the direction must match metric_fn, not the
            internal evaluator. None (default) takes the direction from the
            blueprint, which is the metric the search itself used; pass it
            explicitly if metric_fn is not the blueprint's metric, because a
            wrong direction silently inverts every accept decision.
        candidate_ranking: 'residual_correlation' ranks each candidate feature
            by how well it explains what the base representation could NOT;
            'rf_importance' ranks it by how well it predicts the property on
            its own. None (default) takes the blueprint's value. Before
            2026-08-26 the blueprint carried no such field, so this always fell
            back to 'residual_correlation' even for a search that ran the other
            rule — and the two are separate rows of the stored result, so the
            row said one thing and the rebuild did another.
        augmentation_threshold: how much a whole donor block must improve the
            score for greedy_ablation to keep it. None takes the blueprint's.
        augmentation_budget: features taken from EACH donor by the 'all' and
            'greedy_ablation' paths. Not aug_per_source_budget, which caps
            donors before pooling for 'capped'/'uncapped'. None takes the
            blueprint's.
        aug_max_candidates: length of the ranked candidate list a block size is
            chosen from. None takes the blueprint's, which is 50. At 50 an
            augmentation_fraction of 0.05, 0.10 or 0.25 of a 1000 budget all
            clamp to the same 50 features, so the size setting means nothing
            until this moves. Swept by its own experiment; it is NOT part of a
            stored result's key, so two values would collide on one row if a
            SEARCH ever varied it.

    Returns:
        (X_tr_h, X_va_h, feat_info)
    """
    from kirby.hybrid import apply_augmentation_selection

    if aug_per_source_budget is None:
        aug_per_source_budget = getattr(blueprint, 'aug_per_source_budget', None)

    # Direction defaults to the blueprint's own metric direction. Only meaningful
    # alongside metric_fn — create_hybrid applies it on the model branch only.
    if lower_is_better is None:
        lower_is_better = bool(getattr(blueprint, 'lower_is_better', False))

    # Objective defaults to the blueprint's own, so a rebuild reproduces the
    # quantity the search selected under. Blueprints pickled before the
    # objective field existed report 'predictive', which is what they were.
    if objective is None:
        objective = getattr(blueprint, 'objective', None)

    # Under a geometry objective create_hybrid fits no downstream model —
    # model/metric_fn are ignored and, worse, metric_fn IS the raising guard.
    # Refuse the contradiction rather than let a caller believe it passed one.
    from kirby.find_best_hybrid import (
        OBJECTIVE_PREDICTIVE as _OBJ_PRED, _is_geometry_objective,
    )
    if objective is not None and _is_geometry_objective(objective):
        if model is not None or metric_fn is not None:
            raise ValueError(
                f"build_fold_hybrid: objective={objective!r} scores held-out "
                f"geometry and fits no downstream model, but model/metric_fn "
                f"were passed. Under a geometry objective the dataset metric "
                f"is a raising guard — pass neither.")
        if is_clf is None:
            is_clf = bool(getattr(blueprint, 'is_clf', False))
    elif is_clf is None:
        is_clf = False

    # A model with no metric_fn is worse than no model at all: create_hybrid's
    # model branch calls metric_fn(y_val, y_pred) and its except clause swallows
    # TypeError (hybrid.py:1369), so metric_fn=None would score EVERY allocation
    # as -inf silently instead of raising.
    if model is not None and metric_fn is None:
        raise ValueError(
            "build_fold_hybrid: model was given without metric_fn. "
            "create_hybrid scores the model branch with metric_fn(y_val, y_pred) "
            "and silently treats the resulting TypeError as an infeasible "
            "allocation, so pass both or neither.")

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_train):
            raise ValueError(
                f"build_fold_hybrid: groups has {len(groups)} entries but "
                f"y_train has {len(y_train)}. Pass this fold's TRAIN slice "
                f"(groups[train_idx]), not the whole-dataset groups array.")

    _budget = budget_override if budget_override is not None else blueprint.budget
    X_tr_h, feat_info = create_hybrid(
        train_reps, y_train,
        allocation_method=blueprint.allocation_method,
        budget=_budget,
        importance_method=blueprint.importance_method,
        internal_evaluator=blueprint.internal_evaluator,
        augmentations=augmentations_train,
        augmentation_strategy=blueprint.augmentation_strategy,
        aug_per_source_budget=aug_per_source_budget,
        # Part of the recipe. Omitting it accepts blocks of extras the search
        # would have rejected. getattr guards blueprints pickled before the
        # field existed.
        augmentation_margin=float(
            getattr(blueprint, 'augmentation_margin', 0.0) or 0.0),
        augmentation_fraction=getattr(blueprint, 'augmentation_fraction', None),
        # Added 2026-08-26. Three of these four are columns of the stored
        # result's KEY, so the search files two values as two separate rows —
        # yet none of them was on the recipe, so every rebuild here took
        # create_hybrid's defaults ('residual_correlation', None, 20) whatever
        # the search ran under. A run under 'rf_importance' was rebuilt under
        # the other ranking rule and nothing raised. The fourth,
        # aug_max_candidates, is the length of the ranked candidate list; it is
        # NOT a key column and is swept by its own experiment.
        candidate_ranking=(candidate_ranking if candidate_ranking is not None
                           else getattr(blueprint, 'candidate_ranking',
                                        'residual_correlation')),
        augmentation_threshold=(
            augmentation_threshold if augmentation_threshold is not None
            else getattr(blueprint, 'augmentation_threshold', None)),
        augmentation_budget=int(
            augmentation_budget if augmentation_budget is not None
            else getattr(blueprint, 'augmentation_budget', 20)),
        aug_max_candidates=int(
            aug_max_candidates if aug_max_candidates is not None
            else getattr(blueprint, 'aug_max_candidates', 50)),
        groups=groups,
        # The per-representation scores the SEARCH chose. The engine's own
        # rebuild (find_best_hybrid.apply_blueprint) forwards these; this one
        # did not, so every rebuild here re-scored the representations from the
        # fold's own data instead of reusing what the search decided.
        #
        # That is not only slower. A top_k or performance_weighted allocation
        # turns these scores into WHICH representations get budget, so a rebuild
        # that recomputes them can select a different set than the search did --
        # the same shape of defect as the recipe fields fixed on 2026-08-26.
        # getattr guards recipes pickled before the field existed.
        # Per-feature importance over the base representations. It depends on
        # the fold and the importance method and on NOTHING else -- not the
        # allocation, not the downstream model, not the augmentation settings.
        # create_hybrid recomputes it on every call, so a caller sweeping those
        # settings pays for it every time. Measured 2026-08-27 on
        # flab_thermo_tresanco: building the base hybrid for ONE fold with
        # augmentation OFF took 261.9s, and the grid rebuilds it for every
        # configuration.
        **({'precomputed_importance': precomputed_importance}
           if precomputed_importance is not None else {}),
        # Per-representation baseline scores. These depend on the fold and the
        # downstream model, not on the allocation method -- the allocation only
        # turns them into a budget split.
        #
        # Only an explicit argument is used. The recipe's own rep_scores were
        # measured over the WHOLE dataset by the search that produced it, so
        # falling back to them lets this fold's validation rows decide how much
        # budget each representation gets. A caller that has fold-local scores
        # passes them; otherwise create_hybrid re-scores on this fold's
        # training rows, which costs a rebuild per fold and is the honest one.
        **({'precomputed_alloc_scores': precomputed_alloc_scores}
           if precomputed_alloc_scores is not None else {}),
        model=model,
        metric_fn=metric_fn,
        lower_is_better=lower_is_better,
        **({'objective': objective, 'is_clf': is_clf}
           if objective is not None else {}),
    )
    X_va_h = apply_feature_selection(val_reps, feat_info)

    # Apply augmentation selection to val if the hybrid included augmentations.
    #
    # A top_k allocation auto-demotes every base rep it gave zero budget into
    # the augmentation pool, INSIDE create_hybrid. Those reps then appear in
    # augmentation_info but not in the caller's val-side dict, and
    # apply_augmentation_selection raises. top_3 is the default allocation and
    # a pool is typically ~10 reps, so on a searched blueprint this fired on
    # most folds; the raise then surfaced as a failed fold wherever the caller
    # wrapped this in try/except. The demoted column indices are recorded by
    # create_hybrid so the same columns can be taken from val_reps here.
    aug_info = feat_info.get('augmentation_info')
    if aug_info:
        val_pool = dict(augmentations_val) if augmentations_val else {}
        demoted = feat_info.get('demoted_reps') or {}
        for _name, _cols in demoted.items():
            if _name not in val_pool and _name in val_reps:
                val_pool[_name] = val_reps[_name][:, _cols]
        if val_pool:
            X_va_aug = apply_augmentation_selection(val_pool, aug_info)
            if X_va_aug is not None:
                X_va_h = np.hstack([X_va_h, X_va_aug])

    return X_tr_h, X_va_h, feat_info


# =============================================================================
# HYBRID SEARCH PRECOMPUTE
# =============================================================================

class PrecomputeResults(dict):
    """{aug_cond: SearchResult|None} plus a readable record of what failed.

    A plain dict in every respect the callers use (keyed lookup, `.get`,
    membership, len), so existing consumers are unaffected. The extra
    `failures` list holds one dict per caught exception:
      {'tier', 'dataset', 'aug_condition', 'error_type', 'error', 'traceback'}
    Before this existed a caught precompute exception left only `None` under
    the aug_condition key and a printed line, so a caller could not tell a
    failure from a condition that was never requested.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failures = []

    def record_failure(self, exc, *, tier=None, dataset=None,
                       aug_condition=None):
        """Append one failure record and return it."""
        import traceback as _tb
        rec = {
            'tier': tier,
            'dataset': dataset,
            'aug_condition': aug_condition,
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': _tb.format_exc(),
        }
        self.failures.append(rec)
        return rec


def precompute_failures(obj):
    """Failure records carried by a precompute dict; [] for a plain dict.

    Lets a caller read the failures without an isinstance check, and without
    breaking on precompute dicts built before PrecomputeResults existed.
    """
    return list(getattr(obj, 'failures', []) or [])


def precompute_hybrid_searches(reps, labels, groups, config,
                               augmentations=None, verbose=True,
                               task_type='auto', metric='auto',
                               domain=None,
                               dataset_name=None, tier='general',
                               adaptive_pruning_schedule=None,
                               model_pruning_aggressiveness=None,
                               internal_evaluators=None,
                               budgets=None,
                               label_noise=None,
                               objective=None):
    """Find the best hybrid across all budgets for each augmentation condition.

    objective: search objective forwarded to find_best_hybrid —
    'predictive' (the default when None) or 'similarity'. 'similarity_sac'
    was CUT on 2026-08-26 and now raises.
    find_best_hybrid namespaces its cache tier by objective, so a geometry
    precompute lives beside the predictive one and never overwrites it.

    label_noise: a kirby.noise_spec.NoiseSpec. When given, the search TRAINS on a
    deterministic noisy copy of the labels and SCORES on the clean ones, so the
    hybrid gets no more access to the true labels than a base representation does.
    find_best_hybrid derives its own cache tier from the spec, so the noisy lineage
    lives beside the clean one and never overwrites it.

    ONE find_best_hybrid call per aug_condition with ALL budgets. Baselines
    and importance scores are computed once and reused across budgets.
    best_by_model from each search gives per-model best configs across all budgets.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        config: Dict with keys: budgets.
        augmentations: {aug_name: array} or None.
        verbose: Print progress.
        dataset_name: REQUIRED. Used as part of the cache filename
            (e.g. '{dataset_name}__general__uncapped__full.pkl').
        tier: Cache-condition label for the filename. 'general' (default);
            for distance lineage pass 'general_distance' (must be combined
            with internal_evaluators=DISTANCE_EVALUATORS and
            budgets=DISTANCE_BUDGETS to populate the correct cache key).
        internal_evaluators: Override evaluator list. Pass DISTANCE_EVALUATORS
            for distance lineage. None = find_best_hybrid uses its defaults.
        budgets: Override budget list. Pass DISTANCE_BUDGETS for distance lineage.
            None = use FULL_BUDGETS.

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

    # Always use FULL_BUDGETS + models=None for precompute
    # unless overridden (distance lineage uses DISTANCE_BUDGETS+EVALUATORS).
    # This matches the SLURM precompute cache key regardless of the
    # experiment's own --mode setting. When SLURM cache exists on disk,
    # these calls hit it instantly (no compute).
    if budgets is None:
        budgets = FULL_BUDGETS
    models = None

    # Resolve metric using shared logic (matches get_precomputed_search fallback)
    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    # Precompute aug conditions. The pool is one 10-rep set (tier 'general');
    # the 'top' tier is retired (2026-08-24). It emits {none, uncapped, capped}.
    # capped used to live on 'top' only because 'general' was a big expensive
    # pool — now 'general' IS that 10-rep set, so capped folds back onto it.
    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions.append('uncapped')
        # capped runs on the standard (non-distance) lineage only. Distance
        # precompute doesn't emit capped (no consumer needs it). Above
        aug_conditions.append('capped')   # gate removed 2026-08-27

    results = PrecomputeResults()
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
            _fbh_kwargs = dict(
                base_reps=reps,
                labels=labels,
                groups=groups,
                budgets=budgets,
                models=models,
                augmentations=augs,
                augmentation_strategy=aug_strat,
                task_type=task_type,
                metric=metric,
                cache=True,
                verbose=verbose,
                dataset_name=dataset_name,
                tier=tier,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            if internal_evaluators is not None:
                _fbh_kwargs['internal_evaluators'] = internal_evaluators
            if label_noise is not None:
                _fbh_kwargs['label_noise'] = label_noise
            if objective is not None:
                _fbh_kwargs['objective'] = objective
            search = find_best_hybrid(**_fbh_kwargs)
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
            # Value stays None (what every consumer already tests for); the
            # exception itself goes into results.failures so the caller can
            # read it instead of scraping stdout.
            results[aug_cond] = None
            results.record_failure(e, tier=tier, dataset=dataset_name,
                                   aug_condition=aug_cond)

    if verbose:
        total_time = time.time() - t_start
        n_ok = sum(1 for v in results.values() if v is not None)
        print(f"  Precompute complete: {n_ok}/{len(aug_conditions)} "
              f"searches in {total_time:.0f}s")
        if results.failures:
            _failed = ', '.join(f"{f['aug_condition']}={f['error_type']}"
                                for f in results.failures)
            print(f"  Precompute failures recorded ({len(results.failures)}): "
                  f"{_failed}")

    return results


def populate_precompute_dicts(data, dataset_name, *, domain=None,
                              task_type=None, metric='auto',
                              top_reps=None, verbose=True,
                              adaptive_pruning_schedule=None,
                              model_pruning_aggressiveness=None,
                              include_distance=True):
    """Populate the precompute dicts on `data` for the pool helper.

    Replaces the boilerplate that each category helper (complexity, similarity,
    noise, federated) was duplicating. After this call, data has keys:
      data['precomputed_searches']            (the 10-rep pool)
      data['precomputed_searches_distance']   (general_distance tier)
      data['precompute_errors']               {tier: [failure record, ...]}

    Both precompute dicts are PrecomputeResults, so `.failures` (or
    precompute_failures(d)) lists every caught exception. The same records are
    mirrored onto data['precompute_errors'] keyed by tier, including the case
    where the whole general_distance pass raised.

    The pool helper looks for exactly these keys, so categories that call
    this get cache-hit performance on distance-pool consumers (s04/s05/s06/
    s07/s10/s13/c03) without paying disk-load per pool query.

    `top_reps` is accepted for backward compatibility and ignored: the 'top'
    tier is retired (2026-08-24) — there is one 10-rep pool per dataset.

    Args:
        data: dict with 'reps', 'labels', 'groups' (modified in place).
        dataset_name: cache file key.
        include_distance: set False if no distance-pool experiment will run
            (saves the distance precompute pass).
    """
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    augs = data.get('augmentations')
    errors = data.setdefault('precompute_errors', {})

    # Main-pool precompute over the one 10-rep pool (tier 'general').
    if len(reps) >= 2:
        if verbose:
            print(f"  Precomputing main-pool (general, {len(reps)} reps)...")
        data['precomputed_searches'] = precompute_hybrid_searches(
            reps=reps, labels=labels, groups=groups,
            config={'budgets': FULL_BUDGETS},
            augmentations=augs, verbose=verbose,
            task_type=task_type, metric=metric, domain=domain,
            dataset_name=dataset_name, tier='general',
            adaptive_pruning_schedule=adaptive_pruning_schedule,
            model_pruning_aggressiveness=model_pruning_aggressiveness,
        )
        errors['general'] = precompute_failures(data['precomputed_searches'])
    else:
        data['precomputed_searches'] = PrecomputeResults()

    # Distance-tier precompute. Populates the dict the pool helper consumes
    # when distance=True. Uses DISTANCE_EVALUATORS + DISTANCE_BUDGETS.
    if not include_distance:
        data['precomputed_searches_distance'] = PrecomputeResults()
        return data

    if len(reps) >= 2:
        if verbose:
            print(f"  Precomputing distance-pool (general_distance)...")
        try:
            data['precomputed_searches_distance'] = precompute_hybrid_searches(
                reps=reps, labels=labels, groups=groups,
                config={'budgets': DISTANCE_BUDGETS},
                augmentations=augs, verbose=verbose,
                task_type=task_type, metric=metric, domain=domain,
                dataset_name=dataset_name, tier='general_distance',
                internal_evaluators=DISTANCE_EVALUATORS,
                budgets=DISTANCE_BUDGETS,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            errors['general_distance'] = precompute_failures(
                data['precomputed_searches_distance'])
        except Exception as e:
            if verbose:
                print(f"  general_distance precompute failed: {e}")
            # The whole pass raised, so there is no per-aug_condition value to
            # key on: record one failure with aug_condition None on the empty
            # dict, and mirror it onto data['precompute_errors'].
            _empty = PrecomputeResults()
            _empty.record_failure(e, tier='general_distance',
                                  dataset=dataset_name, aug_condition=None)
            data['precomputed_searches_distance'] = _empty
            errors['general_distance'] = precompute_failures(_empty)
    else:
        data['precomputed_searches_distance'] = PrecomputeResults()

    return data


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
                           weight_method=None, precomputed=None,
                           metric='auto', task_type=None, domain=None,
                           dataset_name=None, tier='general',
                           return_fitted=False,
                           objective=None,
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
        tier: Cache-condition label for the store ('general').
        return_fitted: If True, also fit the best config on full data.
        objective: Search objective — 'predictive' (default when None),
            'similarity' ('similarity_sac' was CUT). Forwarded to find_best_hybrid.
            A non-predictive objective is REFUSED against the in-memory
            `precomputed` store: that store is keyed by aug_condition alone and
            is filled by a predictive precompute, so serving from it would have
            answered a geometry question with a predictive search — silently,
            with no error and a plausible-looking number.
        **kwargs: Additional args passed to find_best_hybrid (NOT `budget`;
            per-budget selection is done downstream via best_by_model_budget).

    Returns:
        SearchResult from find_best_hybrid.
    """
    from kirby.find_best_hybrid import (
        OBJECTIVE_PREDICTIVE as _OBJ_PRED, _VALID_OBJECTIVES as _OBJS,
    )
    if objective is not None and objective not in _OBJS:
        raise ValueError(
            f"get_precomputed_search: objective={objective!r}. "
            f"Must be one of {_OBJS}."
        )
    _geo_request = objective is not None and objective != _OBJ_PRED
    # Guard against kwargs leakage: aug_condition drives augmentation_strategy
    # per pool iteration. If a caller forwards augmentation_strategy via
    # **kwargs, it would silently override the per-iteration value below.
    if 'augmentation_strategy' in kwargs:
        import warnings as _warnings
        _warnings.warn(
            "augmentation_strategy passed via kwargs is ignored; "
            "get_precomputed_search sets it per aug_condition iteration",
            stacklevel=2,
        )
        kwargs = {k: v for k, v in kwargs.items() if k != 'augmentation_strategy'}

    # Fast path: return from in-memory precomputed results.
    # The store's keys are aug_conditions only — nothing in them records which
    # objective the search optimised. Handing a geometry request whatever sits
    # under its aug_condition is how a similarity experiment ended up reporting
    # a predictive search's answer. Refuse instead: either the caller has a
    # geometry precompute of its own, or it should fall through to the
    # objective-namespaced on-disk cache below.
    if precomputed is not None:
        if _geo_request:
            raise ValueError(
                f"get_precomputed_search: objective={objective!r} cannot be "
                f"served from the in-memory `precomputed` store — that store "
                f"is keyed by aug_condition alone and holds predictive "
                f"searches, so it would answer a geometry question with a "
                f"predictive result. Pass precomputed=None so the call falls "
                f"through to the objective-namespaced on-disk cache."
            )
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
        metric=resolved_metric,
        task_type=resolved_task_type,
        return_fitted=return_fitted,
        weight_method=weight_method,
        cache=True,
        verbose=False,
        dataset_name=dataset_name,
        tier=tier,
    )
    if objective is not None:
        call_kwargs['objective'] = objective
    call_kwargs.update(kwargs)

    return find_best_hybrid(**call_kwargs)


def get_best_precomputed_search(reps, labels, groups, augmentations=None,
                                *, precomputed=None,
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
        precomputed=precomputed,
        metric=metric, task_type=task_type, domain=domain,
        dataset_name=dataset_name, tier=tier,
        # Both were captured by name and then dropped, so a caller asking for
        # a fitted result got a SearchResult whose fitted_results was empty.
        return_fitted=return_fitted, weight_method=weight_method, **kwargs)

    if not has_augs:
        return s_none, 'none'

    s_uncapped = get_precomputed_search(
        reps, labels, groups,
        aug_condition='uncapped', augmentations=augmentations,
        precomputed=precomputed,
        metric=metric, task_type=task_type, domain=domain,
        dataset_name=dataset_name, tier=tier,
        return_fitted=return_fitted, weight_method=weight_method, **kwargs)

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


def get_best_search_for_downstream(
    reps, labels, groups, *, downstream_model, budget,
    augmentations=None, 
    metric='auto', task_type=None, domain=None,
    dataset_name=None, distance=False,
    data=None, return_fitted=False, weight_method=None,
    label_noise=None,
    objective=None,
    **kwargs,
):
    """Pick the best precomputed SearchResult from the (tier x aug_condition) pool.

    objective: search objective — 'predictive' (default when None),
    'similarity' ('similarity_sac' was CUT). Under a geometry objective the in-memory
    `data['precomputed_searches']` dicts are bypassed entirely (they hold
    predictive searches and carry no objective marker), so the pool resolves
    from the objective-namespaced on-disk cache instead.

    label_noise: a kirby.noise_spec.NoiseSpec. When given, resolves the NOISY
    lineage — a blueprint searched on noisy training labels and scored on clean
    ones — instead of the clean precompute. Use this in any experiment that injects
    label noise, so the hybrid does not enter the comparison having been built with
    access to the very labels the experiment then corrupts.

    Selection is per-(downstream_model, budget). For each candidate SearchResult
    in the pool, looks up the best config for (downstream_model, budget) via
    get_best_for_downstream and compares by score_mean (respecting
    search.lower_is_better). Returns the winning SearchResult and metadata about
    which (tier, aug_condition) won.

    Main pool (distance=False):
      tier='general'  x aug=['none', 'uncapped']
      tier='top'      x aug=['none', 'uncapped', 'capped']
      (general/capped intentionally absent -- dropped 2026-05-22)

    Distance pool (distance=True):
      tier='general_distance' x aug=['none', 'uncapped']
      tier='top_distance'     x aug=['none', 'uncapped']
      (DISTANCE_EVALUATORS + DISTANCE_BUDGETS lineage; no capped)

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        downstream_model: Model name to optimise for (e.g. 'rf', 'svm', 'knn').
        budget: Specific budget the caller cares about (used for selection).
            Pass None to compare across all budgets for the model.
        augmentations: {aug_name: array} or None. Determines which aug
            conditions are eligible; with no augs, only 'none' is considered.
        speed: 'fast' or 'full'.
        metric: Metric name ('auto' = resolve from domain/task_type).
        task_type: 'classification' or 'regression' (required by underlying
            get_precomputed_search).
        domain: Domain hint for metric resolution.
        dataset_name: REQUIRED. Cache files are dataset-keyed.
        distance: If True, iterate the distance-tier cache lineage instead
            of the main pool. Auto-passes internal_evaluators=DISTANCE_EVALUATORS
            and budgets=DISTANCE_BUDGETS to the underlying loaders.
        data: Optional. If provided, prefers the data['precomputed_searches']
            dict (in-memory hit). Falls back to disk-cached find_best_hybrid
            when missing.
        return_fitted: If True, request fitted=True from underlying searches.
        weight_method: Importance method for fitted results.
        **kwargs: Extra kwargs forwarded to get_precomputed_search.

    Returns:
        (search, metadata) where metadata is either:
            {
                'tier': str,
                'aug_condition': str,
                'score_mean': float,
                'model': str,
                'budget': int or None,
                'best_config': dict,  # from get_best_for_downstream
            }
        or (None, None) if no candidate had a valid result for the requested
        (downstream_model, budget).
    """
    if dataset_name is None:
        raise ValueError(
            "get_best_search_for_downstream requires dataset_name= "
            "(cache files are dataset-keyed)"
        )

    # Guard against kwargs leakage (same as get_precomputed_search): the pool
    # below iterates aug conditions and we must not let a caller-supplied
    # augmentation_strategy override the per-iteration value via **extra.
    if 'augmentation_strategy' in kwargs:
        import warnings as _warnings
        _warnings.warn(
            "augmentation_strategy passed via kwargs is ignored; "
            "get_best_search_for_downstream sets it per pool iteration",
            stacklevel=2,
        )
        kwargs = {k: v for k, v in kwargs.items() if k != 'augmentation_strategy'}

    has_augs = bool(augmentations)

    # One 10-rep pool (tier 'general'); the 'top' tier is retired (2026-08-24).
    if distance:
        pool = [
            ('general_distance', 'none', None),
        ]
        if has_augs:
            pool.append(('general_distance', 'uncapped', augmentations))
        # Distance lineage uses its own evaluators/budgets.
        extra = dict(kwargs)
        extra.setdefault('internal_evaluators', DISTANCE_EVALUATORS)
        extra.setdefault('budgets', DISTANCE_BUDGETS)
    else:
        pool = [
            ('general', 'none', None),
        ]
        if has_augs:
            pool.append(('general', 'uncapped', augmentations))
            # This has to match what the precompute EMITS, or the lookup
            # misses and the experiment cold-builds the search live. The
            # precompute's 250-row gate was removed on 2026-08-27, so capped is
            # emitted for every dataset and this asks for it unconditionally.
            # The two must be changed together.
            pool.append(('general', 'capped', augmentations))
        extra = dict(kwargs)

    # Resolve in-memory precomputed dicts if data was passed.
    # Distance lineage now ALSO uses the in-memory precomputed_searches_distance
    # dict when populated by category helpers.
    # A geometry objective must never read the in-memory dicts: they are filled
    # by a predictive precompute and record no objective, so a hit there answers
    # a geometry question with a predictive search. Force the disk path, whose
    # cache filenames ARE objective-namespaced.
    from kirby.find_best_hybrid import OBJECTIVE_PREDICTIVE as _OBJ_PRED
    _geo_request = objective is not None and objective != _OBJ_PRED
    # Same rule for a search asked to pick its winner on the noisy labels: the
    # in-memory dicts hold searches that picked theirs on the clean ones and
    # record nothing to tell them apart, so a hit there answers the question
    # with the search the caller is trying to avoid.
    _noisy_selection = bool(kwargs.get('score_on_noisy_labels'))

    main_dict = None
    if data is not None and not _geo_request and not _noisy_selection:
        if label_noise is not None:
            # Noisy lineage: one {aug_cond: SearchResult} dict per noise spec, keyed
            # by spec_id. Never falls back to the clean dicts — a clean blueprint
            # served under a noisy request is exactly the silent bug this parameter
            # exists to kill.
            _sid = label_noise.spec_id
            main_dict = (data.get('precomputed_searches_noisy') or {}).get(_sid)
        elif distance:
            main_dict = data.get('precomputed_searches_distance')
        else:
            main_dict = data.get('precomputed_searches')

    _verbose_pool = os.environ.get('KIRBY_POOL_VERBOSE') == '1'

    candidates = []  # list of (tier, aug_cond, search, best_config)
    for tier, aug_cond, augs in pool:
        # Only the 'general'/'general_distance' pool remains; it uses main_dict.
        if tier in ('general', 'general_distance'):
            precomputed = main_dict
        else:
            precomputed = None
        try:
            search = get_precomputed_search(
                reps, labels, groups,
                aug_condition=aug_cond, augmentations=augs,
                weight_method=weight_method, precomputed=precomputed,
                metric=metric, task_type=task_type, domain=domain,
                dataset_name=dataset_name, tier=tier,
                # Captured by name and then dropped, so get_fitted_pool_hybrid's
                # return_fitted=True never reached the search and every
                # SearchResult came back with fitted_results empty.
                return_fitted=return_fitted,
                # Forwarded into find_best_hybrid on the disk-fallback path, which
                # derives the noisy cache tier from it.
                **({'label_noise': label_noise} if label_noise is not None else {}),
                **({'objective': objective} if objective is not None else {}),
                **extra,
            )
        except Exception as e:
            # Always surface a failed pool condition. A swallowed error here
            # (e.g. a removed speed-tier key) silently drops the hybrid, which
            # previously hid real bugs (c15/c16/opig smoke 'fast' KeyError).
            # Verbose mode adds the full traceback.
            print(f"[pool] WARNING: {tier}/{aug_cond} search failed: "
                  f"{type(e).__name__}: {e}")
            if _verbose_pool:
                import traceback
                traceback.print_exc()
            continue
        # Memoize: if we have an in-memory dict and just computed/loaded a result,
        # write it back so subsequent pool calls in this process hit memory
        # instead of paying disk-load (or worse, retriggering find_best_hybrid).
        if precomputed is not None and search is not None and aug_cond not in precomputed:
            precomputed[aug_cond] = search
        if search is None or not search.best_by_model:
            continue
        # Reject cross-model fallback. get_best_for_downstream's level-5
        # fallback returns the "best blueprint from any model" with that
        # model's score — comparing those scores across pool candidates
        # would let the pool pick whichever (tier, aug) has the highest
        # score in ANY model, not the best for downstream_model. Require
        # downstream_model is actually represented in this search.
        if (downstream_model not in (search.best_by_model_budget or {})
                and downstream_model not in (search.best_by_model or {})):
            if _verbose_pool:
                print(f"[pool] {tier}/{aug_cond}: model={downstream_model} absent, skipping")
            continue
        best = get_best_for_downstream(search, downstream_model, budget=budget)
        if not best:
            continue
        score = best.get('score_mean')
        if score is None or not np.isfinite(score):
            continue
        candidates.append((tier, aug_cond, search, best))

    if not candidates:
        # No pool member produced a usable hybrid for this (model, budget).
        # The caller will fall back to a non-hybrid path — make that loud
        # rather than returning None silently (this is how a hybrid can
        # silently vanish from an experiment's results).
        print(f"[pool] WARNING: no usable hybrid for model={downstream_model} "
              f"budget={budget} dataset={dataset_name} — all {len(pool)} pool "
              f"conditions failed or were empty; caller will use a fallback.")
        return None, None

    # All pool members for one (dataset, metric) should agree on lower_is_better.
    # If they don't, the same dataset was searched under different metrics —
    # that's a bug elsewhere, but to avoid silent score-direction confusion
    # we drop any candidate that disagrees with the first.
    ref_lib = candidates[0][2].lower_is_better
    candidates = [c for c in candidates if c[2].lower_is_better == ref_lib]
    if not candidates:
        return None, None
    lower_is_better = ref_lib

    # Sort: best score first (respecting lower_is_better), then by tie-break
    # preference: none > uncapped > capped.
    aug_pref = {'none': 0, 'uncapped': 1, 'capped': 2}
    tier_pref = {'general': 0, 'general_distance': 0}

    def _key(item):
        tier, aug_cond, search, best = item
        s = best['score_mean']
        primary = s if lower_is_better else -s
        return (primary, aug_pref.get(aug_cond, 99), tier_pref.get(tier, 99))

    candidates.sort(key=_key)
    tier, aug_cond, search, best = candidates[0]

    metadata = {
        'tier': tier,
        'aug_condition': aug_cond,
        'score_mean': best['score_mean'],
        'model': downstream_model,
        'budget': budget,
        'best_config': best,
        # What rows chose this recipe. 'score_mean' is the mean over the
        # SEARCH's own inner folds, run over every row of the dataset, so the
        # allocation method, budget, importance method, internal evaluator and
        # augmentation condition were all picked with the held-out rows of any
        # later split among them. An experiment that rebuilds this recipe per
        # fold fixes the feature WEIGHTS, not the choice of recipe. Carry this
        # into whatever the experiment reports so the number is never read as a
        # held-out one; nested_cv_evaluate is the path that does not need it.
        'selection_scope': 'whole_dataset',
        'selection_note': ('recipe chosen by its mean over the search\'s inner '
                           'folds on all rows; only the features are refit'),
    }
    return search, metadata


def get_fitted_pool_hybrid(
    reps, labels, groups, *, downstream_model, budget,
    augmentations=None, data=None,
    metric='auto', task_type=None, domain=None,
    dataset_name=None, distance=False, objective=None, **kwargs,
):
    """Pool-best variant of get_fitted_hybrid.

    Picks the best precomputed SearchResult from the pool for
    (downstream_model, budget) via get_best_search_for_downstream, then
    builds X_hybrid on full data using the winning blueprint.

    Returns:
        (X_hybrid, feature_info, search, metadata) where metadata is the
        dict returned by get_best_search_for_downstream (tier/aug_condition
        of the winner). Raises RuntimeError if no candidate produced a
        usable result.
    """
    from kirby.hybrid import create_hybrid

    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model=downstream_model, budget=budget,
        augmentations=augmentations,
        dataset_name=dataset_name, distance=distance,
        data=data, return_fitted=True, objective=objective, **kwargs,
    )
    if not meta:
        raise RuntimeError(
            f"get_fitted_pool_hybrid: no pool candidate produced a result for "
            f"model={downstream_model} budget={budget} dataset={dataset_name}"
        )

    bp = meta['best_config']['blueprint']
    aug_strat = meta['aug_condition']
    augs = augmentations if aug_strat != 'none' else None

    X_hybrid, fi = create_hybrid(
        reps, labels,
        allocation_method=bp.allocation_method,
        budget=bp.budget,
        step_size=getattr(bp, 'step_size', 5),
        importance_method=bp.importance_method,
        internal_evaluator=bp.internal_evaluator,
        augmentations=augs,
        # Trust meta['aug_condition'] (the actual condition that loaded this
        # SearchResult), not blueprint's stored augmentation_strategy. The
        # blueprint can be stale due to the 2026-04-21 cache-collision bug.
        augmentation_strategy=aug_strat,
        # The search that chose this blueprint ran group-aware; the full-data
        # rebuild must use the same grouping for create_hybrid's internal
        # split, else it falls back to a random 20% carve.
        groups=groups,
        # Rebuild under the SAME objective the search selected under, so the
        # internal allocation/admission tests score the same quantity. Without
        # this the full-data rebuild of a geometry-selected blueprint silently
        # reverted to predictive scoring.
        **({'objective': objective} if objective is not None else {}),
    )
    return X_hybrid, fi, search, meta


def get_fitted_hybrid(reps, labels, groups, *, downstream_model,
                      budget=None, aug_condition='uncapped',
                      augmentations=None, precomputed=None,
                      internal_evaluators=None, budgets=None,
                      metric='auto', task_type=None, domain=None,
                      dataset_name=None, tier='general',
                      adaptive_pruning_schedule=None,
                      model_pruning_aggressiveness=None,
                      objective=None,
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
    if objective is not None:
        kwargs['objective'] = objective
        use_precomputed = None  # geometry/predictive both resolve from disk here

    # NOTE: `use_precomputed` is computed above but deliberately NOT passed —
    # this call has never forwarded it, so get_precomputed_search always takes
    # its precomputed=None default and resolves from the on-disk cache. Left as
    # found: wiring it up now would change which search every existing caller
    # gets. Flagged, not fixed.
    search = get_precomputed_search(
        reps, labels, groups,
        aug_condition=aug_condition, augmentations=augmentations,
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
            # Same as get_fitted_pool_hybrid: keep the full-data rebuild on the
            # same grouping the search used for its internal split.
            groups=groups,
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
    # Filter entries to those with valid (finite, non-None) score_mean.
    # Previously the key= used `r.get('score_mean', 0)` which silently
    # treated missing scores as 0 — for lower-is-better metrics (RMSE etc.)
    # 0 is always the min, so a broken/missing entry would win. That's the
    # 8-week silent-wrong-result bug.
    def _valid(entries):
        return [r for r in entries
                if r.get('score_mean') is not None
                and np.isfinite(r['score_mean'])]

    # Try model-specific budget lookup
    if search.best_by_model_budget and downstream_model in search.best_by_model_budget:
        model_budgets = search.best_by_model_budget[downstream_model]
        if budget is not None and budget in model_budgets:
            entry = model_budgets[budget]
            if entry.get('score_mean') is not None and np.isfinite(entry['score_mean']):
                return entry
            # Fall through to "best across budgets" if the requested one is broken.
        # Best across all budgets for this model
        valid = _valid(model_budgets.values())
        if valid:
            lower = getattr(search, 'lower_is_better', False)
            return (min if lower else max)(valid, key=lambda r: r['score_mean'])

    # Fallback: budget-specific (model-agnostic)
    if budget is not None and search.best_by_budget and budget in search.best_by_budget:
        entry = search.best_by_budget[budget]
        if entry.get('score_mean') is not None and np.isfinite(entry['score_mean']):
            return entry

    # Fallback: model-specific (budget-agnostic)
    if search.best_by_model and downstream_model in search.best_by_model:
        entry = search.best_by_model[downstream_model]
        if entry.get('score_mean') is not None and np.isfinite(entry['score_mean']):
            return entry

    # Cross-model fallback: use overall best blueprint from any model.
    # The blueprint (allocation/importance/budget) is largely model-agnostic —
    # create_hybrid builds the same hybrid regardless of downstream model.
    if search.best_by_model:
        valid = _valid(search.best_by_model.values())
        if valid:
            lower = getattr(search, 'lower_is_better', False)
            return (min if lower else max)(valid, key=lambda r: r['score_mean'])

    return None


# =============================================================================
# OPTUNA HP TUNING
# =============================================================================

def tune_xgboost_optuna(X_train, y_train, groups=None, n_trials=50,
                        n_cv_folds=3, task_type='regression', seed=42,
                        verbose=False):
    """Tune XGBoost hyperparameters with Optuna. Returns best model instance.

    `groups` are this training set's scaffold/identity groups. Each trial is
    scored on a group-aware split, so a hyperparameter cannot be chosen because
    it memorised a scaffold that sits on both sides of the trial's own split.
    Required whenever the caller has groups; passing None raises rather than
    scoring trials on a random split.

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
    trial_splitter, _ = make_cv_splitter(
        n_cv_folds, groups=groups, seed=seed,
        task_type='classification' if is_clf else None)

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
        scores = cross_val_score(model, X_train, y_train, groups=groups,
                                 cv=trial_splitter, scoring=scoring)
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


# Top-level keys that describe the run rather than carry a computed quantity.
# classify_result_payload ignores them when deciding whether a saved result is
# empty; everything else in the dict counts as a payload section.
RESULT_META_KEYS = frozenset({
    'timestamp', 'experiment', 'experiment_id', 'dataset', 'domain',
    'config', 'meta', 'primary_metric', 'metric_name', 'metric',
    'lower_is_better', 'task_type', 'n_samples', 'n_reps', 'seed', 'mode',
    'notes', 'skipped', 'reason', 'policy',
    'result_status', 'has_computed_results',
})


# --- Status records vs measurements -------------------------------------------
# Exact key names that carry a status note or a status count rather than a
# measured quantity. A value stored under one of these never makes a payload
# non-empty, at any nesting depth.
MEASUREMENT_STATUS_KEYS = frozenset({
    'error', 'errors', 'error_type', 'error_source', 'error_source_note',
    'reason', 'reasons', 'traceback', 'message', 'status',
    'note', 'notes', 'warning', 'warnings', 'legend', 'arm_legend',
    'computed', 'ok', 'success', 'succeeded',
    'result_status', 'has_computed_results',
    'skipped', 'failed', 'failures',
})

# Suffixes that mark a key as a status count. '_error'/'_errors' are NOT here:
# 'cal_error', 'mean_error', 'prediction_error', 'cv_error' and 'decile_errors'
# are measurements in this repository. The substring 'skipped' is handled
# separately because status counts spell it mid-name ('cliff_n_skipped_clf',
# 'n_skipped_cells').
MEASUREMENT_STATUS_SUFFIXES = ('_skipped', '_failed', '_failure', '_failures',
                               '_reason', '_reasons')

# Keys whose False value declares the enclosing dict a status record.
STATUS_FLAG_KEYS = ('computed', 'ok', 'success', 'succeeded',
                    'has_computed_results')


def _is_status_key(key) -> bool:
    """True when `key` names a status note or a status count, not a measurement.

    Matches the 25 exact names in MEASUREMENT_STATUS_KEYS, the 6 suffixes in
    MEASUREMENT_STATUS_SUFFIXES, and any name containing 'skipped'.
    """
    if not isinstance(key, str):
        return False
    k = key.lower()
    if k in MEASUREMENT_STATUS_KEYS:
        return True
    if 'skipped' in k:
        return True
    return k.endswith(MEASUREMENT_STATUS_SUFFIXES)


def _is_status_record(item) -> bool:
    """True when `item` is a dict that declares itself a non-result.

    Two triggers: a status flag set False (computed=False, ok=False,
    success=False, succeeded=False, has_computed_results=False), or an 'error'
    or 'reason' key with no measurement anywhere else in the dict.
    """
    if not isinstance(item, dict):
        return False
    for flag in STATUS_FLAG_KEYS:
        if flag in item and item[flag] is False:
            return True
    if 'error' in item or 'reason' in item:
        rest = {k: v for k, v in item.items()
                if k not in ('error', 'reason')}
        return not _contains_measurement(rest)
    return False


def _is_measurement(item) -> bool:
    """True when `item` is a finite number that is not a bool.

    bool is excluded (True/False are flags, and bool subclasses int). NaN and
    +/-inf are excluded: they record that a quantity was not obtained.
    """
    if isinstance(item, (bool, np.bool_)):
        return False
    if isinstance(item, int):        # a Python int is always finite
        return True
    if isinstance(item, (float, np.floating, np.integer)):
        try:
            return bool(np.isfinite(item))
        except (TypeError, ValueError):
            return False
    return False


def _contains_measurement(value) -> bool:
    """True when a finite number sits somewhere in `value` under a non-status key.

    Iterative walk (no recursion limit), returning on the first measurement
    found. Skipped on the way down: values under a status key (_is_status_key),
    dicts that declare themselves status records (_is_status_record), strings,
    bools, None. So a payload built only from error strings, refusal reasons,
    skip counts and status flags contains no measurement.
    """
    stack = [value]
    seen = set()
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, (dict, list, tuple, set)):
            marker = id(item)
            if marker in seen:      # guard against a self-referencing payload
                continue
            seen.add(marker)
        if isinstance(item, dict):
            if _is_status_record(item):
                continue
            for k, v in item.items():
                if _is_status_key(k):
                    continue
                stack.append(v)
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif isinstance(item, str):
            continue
        elif _is_measurement(item):
            return True
        elif isinstance(item, np.ndarray):
            # A string or object array casts badly; walk its elements instead.
            try:
                if item.size and np.isfinite(item.astype(float)).any():
                    return True
            except (TypeError, ValueError):
                stack.extend(item.ravel().tolist())
    return False


def _payload_is_empty(value) -> bool:
    """True when `value` records no numeric measurement.

    A finite number under a non-status key is the only thing that makes a
    payload non-empty. Strings, bools, None and empty containers never do, so a
    payload holding only failure records, refusal records or status records is
    empty. See _contains_measurement for the walk.
    """
    return not _contains_measurement(value)


def classify_result_payload(results) -> str:
    """Return 'skipped', 'empty' or 'ok' for a result dict about to be saved.

    'skipped' — carries the {'skipped': True} marker should_skip_experiment
                writes, i.e. the experiment was never meant to run here.
    'empty'   — no finite number outside RESULT_META_KEYS and outside status
                keys, so the file records no measurement. All-failed runs,
                refusal payloads and status-only payloads land here.
    'ok'      — at least one finite numeric measurement is present.
    """
    if not isinstance(results, dict):
        return 'empty' if _payload_is_empty(results) else 'ok'
    if results.get('skipped'):
        return 'skipped'
    payload = {k: v for k, v in results.items() if k not in RESULT_META_KEYS}
    return 'empty' if _payload_is_empty(payload) else 'ok'


def experiment_result_is_empty(loaded, classify_unmarked: bool = False) -> bool:
    """True when a loaded result JSON records no computed quantity.

    Reads the 'result_status' / 'has_computed_results' keys save_experiment_
    results stamps. Files written before those keys existed carry neither; they
    return False unless classify_unmarked=True, which re-derives the status from
    the file's own sections via classify_result_payload.

    A resume check that tests only for the file existing treats an empty result
    as finished; this is the test that separates the two.
    """
    if not isinstance(loaded, dict):
        return False
    status = loaded.get('result_status')
    if status is not None:
        return status == 'empty'
    if 'has_computed_results' in loaded:
        return loaded['has_computed_results'] is False
    if classify_unmarked:
        return classify_result_payload(loaded) == 'empty'
    return False


def input_fingerprint(data, config=None):
    """What a result file was computed FROM, small enough to store in it.

    A resumed run decides whether to recompute an experiment. Asking only
    whether the output file exists cannot see that the fold grouping changed,
    that a representation was added or dropped, or that the labels moved — the
    stale file is kept and the run reports a number produced under a different
    dataset. This is what a resume compares.

    Contents: row count, split count, a content hash of the labels and of the
    group vector, and the sorted representation names with their widths. The
    hashes are the same canonical-bytes hashes find_best_hybrid's cache uses, so
    a string group vector hashes stably across processes.

    Returns a dict, or {} when `data` carries no labels (nothing to compare).
    """
    from kirby.find_best_hybrid import _hash_bytes, _canonical_bytes

    if not isinstance(data, dict):
        return {}
    labels = data.get('labels')
    if labels is None:
        return {}
    labels = np.asarray(labels)
    groups = data.get('groups')
    reps = data.get('reps') or {}
    fp = {
        'n_samples': int(len(labels)),
        'labels_hash': int(_hash_bytes(_canonical_bytes(labels))),
        'groups_hash': (int(_hash_bytes(_canonical_bytes(groups)))
                        if groups is not None else None),
        'n_groups': (int(len(np.unique(np.asarray(groups))))
                     if groups is not None else None),
        'reps': sorted(
            [str(k), int(np.shape(v)[1])] for k, v in reps.items()),
        # A representation with no whole-dataset matrix, listed by name only.
        'fold_fitted_reps': sorted(data.get('fold_fitted_reps') or []),
    }
    if config:
        fp['n_splits'] = int(config.get('n_splits') or 0)
    return fp


def fingerprint_mismatch(stored, current):
    """Name the first field of an input_fingerprint that moved, or None.

    A stored fingerprint of None or {} means the file predates the stamp: it
    cannot be compared, so it is treated as stale and the reason says so.
    """
    if not stored:
        return 'the file records nothing about the data it was computed from'
    if not current:
        return None  # nothing to compare against; leave the file alone
    for key in ('n_samples', 'labels_hash', 'groups_hash', 'n_groups',
                'n_splits'):
        if key in current and stored.get(key) != current.get(key):
            return f'{key} changed: {stored.get(key)!r} -> {current.get(key)!r}'
    _old = {tuple(r) for r in (stored.get('reps') or [])}
    _new = {tuple(r) for r in (current.get('reps') or [])}
    if _old != _new:
        _gone = sorted(n for n, _ in _old - _new)
        _added = sorted(n for n, _ in _new - _old)
        return (f'representations changed: dropped {_gone or "none"}, '
                f'added {_added or "none"}')
    if (sorted(stored.get('fold_fitted_reps') or [])
            != sorted(current.get('fold_fitted_reps') or [])):
        return 'the per-fold representation list changed'
    return None


def save_experiment_results(results: dict, output_dir: str, experiment_id: str,
                            dataset: str = None,
                            metric_name: str = None,
                            lower_is_better: bool = None,
                            fingerprint: dict = None):
    """Save results JSON to output_dir/experiment_id/{dataset}.json.

    If results contains a 'meta' key, the output uses the structured schema.
    Otherwise, wraps results with a timestamp for traceability.

    Always stamps the metric DIRECTION (`metric_name` + `lower_is_better`) at the
    top level so analysis/aggregation never has to re-guess it — the root cause of
    a recurring sign bug where lower-is-better (RMSE/MAE) molecular results were
    read as regressions. Callers may pass the authoritative values (e.g. from
    get_domain_metric(data)); otherwise they are derived from the registered
    `dataset` name. Values already present in `results` are never overwritten.

    Also stamps `result_status` ('ok' | 'empty' | 'skipped') and the boolean
    `has_computed_results`, so a reader can tell a file whose sections are all
    empty from one that finished with numbers in it. Existing keys and values
    are untouched: this only adds two keys.
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

    # --- Persist metric direction (see docstring) ---
    if (metric_name is None or lower_is_better is None) and dataset:
        try:
            info = registry_lookup(dataset)
        except Exception:
            info = None
        if info is not None:
            if metric_name is None:
                metric_name = info.metric
            if lower_is_better is None:
                lower_is_better = info.lower_is_better
    # Fill in only when missing/None; never clobber an explicit value (incl. a
    # legitimate False) already set by make_structured_output or the caller.
    if metric_name is not None and output.get('metric_name') is None:
        output['metric_name'] = metric_name
    if lower_is_better is not None and output.get('lower_is_better') is None:
        output['lower_is_better'] = lower_is_better

    # --- Stamp whether this file records any computed quantity ---
    # The resume checks in complexity/similarity/noise treat the file existing
    # as proof the experiment finished, so a run that returned only empty
    # sections was skipped on every resubmission. 'result_status' is 'ok',
    # 'empty' or 'skipped'; experiment_result_is_empty() reads it back.
    # setdefault: a caller that set either key keeps its own value.
    if isinstance(output, dict):
        _status = classify_result_payload(output)
        output.setdefault('result_status', _status)
        output.setdefault('has_computed_results', _status == 'ok')

    # --- Stamp what this file was computed from ---
    # A resumed run compares this against the data it just loaded. Without it a
    # file written before a fold-grouping, label or representation change is
    # kept and never recomputed. See input_fingerprint / fingerprint_mismatch.
    if isinstance(output, dict) and fingerprint:
        output.setdefault('input_fingerprint', convert_numpy(fingerprint))

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
    _mark = ''
    if isinstance(output, dict) and output.get('result_status') == 'empty':
        _mark = ' (result_status=empty: no computed quantity in the payload)'
    print(f"  [saved {filepath}]{_mark}")
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
        # Build config from the DEFAULT_* constants
        from kirby.find_best_hybrid import (
            DEFAULT_N_FOLDS, DEFAULT_ALLOCATION_METHODS,
            DEFAULT_IMPORTANCE_METHODS, DEFAULT_INTERNAL_EVALUATORS)
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': DEFAULT_INTERNAL_EVALUATORS,
            'models': ['rf', 'xgboost', 'knn', 'lightgbm', 'mlp',
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

    Pulls allocations/importance/evaluators from the DEFAULT_* constants.
    Models and budgets are experiment-controlled (set via CLI or experiment code).
    """
    from kirby.find_best_hybrid import (
        DEFAULT_N_FOLDS, DEFAULT_ALLOCATION_METHODS,
        DEFAULT_IMPORTANCE_METHODS, DEFAULT_INTERNAL_EVALUATORS)

    # Use FULL_BUDGETS for hybrid search cache alignment with
    # SLURM precompute. --mode only affects the experiment's own CV
    # evaluation (n_splits), not the hybrid search speed.
    default_budgets = FULL_BUDGETS

    # n_splits for experiment CV: 1 for intermediary (quick check), 5 for full
    exp_folds = DEFAULT_N_FOLDS if args.mode == 'full' else 1

    config = {
        'n_splits': exp_folds,
        'allocations': DEFAULT_ALLOCATION_METHODS,
        'importance_methods': DEFAULT_IMPORTANCE_METHODS,
        'internal_evaluators': DEFAULT_INTERNAL_EVALUATORS,
        'budgets': default_budgets,
        # Models for experiment use (baselines, evaluation, etc.)
        # NOTE: precompute_hybrid_searches ignores this — uses models=None
        # so the cache key matches the SLURM precompute.
        'models': ['rf', 'xgboost', 'knn', 'lightgbm', 'mlp',
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
