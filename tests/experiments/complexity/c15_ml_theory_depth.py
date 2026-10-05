#!/usr/bin/env python3
"""C15: ML Theory Depth — concentration of measure, kernel degradation,
double descent, and approximation theory analysis.

Paper: Complexity
Conclusion: Compact hybrid reps avoid high-D pathologies: distance concentration,
    kernel degeneration, hubness, and interpolation-threshold instability.
Dependencies: c01 (rep characterization for comparison), c02 (grid results optional).

Four sub-analyses:
  15a: Distance concentration — Beyer et al. (1999) ratio + pairwise distance
       distribution comparison across budgets. The theoretical WHY behind everything.
  15b: Kernel matrix analysis — condition number, eigenspectrum, and kernel value
       distribution for RBF/Tanimoto at different dimensionalities. Explains why
       GP/SVM benefit disproportionately from compact reps.
  15c: Double descent probe — sweep budget through n ≈ p regime. Test whether
       hybrids skip the interpolation threshold entirely.
  15d: Approximation theory — Rademacher complexity estimation, generalization
       gap vs compression ratio. Empirical PAC-style bounds.

Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4; selection rule (B) of
2.3, "keep the comparison, on the dataset's two models and per-fold searches"):
  - Models are the dataset's two decided models (decided_models(ds, 'general'))
    built from find_best_hybrid.DEFAULT_MODELS, the same estimators the search
    scores. The random forest and the Ridge(alpha=1.0) this file fitted are
    gone; ridge is removed everywhere.
  - Every hybrid for an outer fold comes from a search on that fold's training
    rows. 15a, 15b and 15d take each model's 2 best hybrids at each decided
    budget from the fold's general search (fold_standard_selection, budget=).
    15c runs its own search per budget on each fold's training rows
    (own_search_kwargs with the real dataset name; the budget goes in
    condition_suffix). No whole-dataset search is read or built.
  - Hybrids are rebuilt on the fold's training rows with the fold search's own
    per-representation scores (dirty-run bug 1) and scored on the held-out rows.
  - No distance hybrid: c15 is not one of the experiments handoff 2.3 item 1
    gives distance hybrids to.
  - Outer folds come from config['n_splits'] through
    kirby.find_best_hybrid.nested_cv_outer_folds; one fold is one held-out
    split, and every spread then says "one fold, no spread".
  - Features are scaled with kirby.scaling.FeatureScaler (0/1 features left as
    they are) and representations stay float32.

References:
  - Beyer et al. (ICDT 1999) "When Is Nearest Neighbor Meaningful?"
  - Aggarwal et al. (ICDT 2001) "On the Surprising Behavior of Distance Metrics"
  - Radovanovic et al. (JMLR 2010) "Hubs in Space"
  - Bengio et al. (2005) "The Curse of Dimensionality for Local Kernel Machines"

Usage:
    python c15_ml_theory_depth.py --dataset logd --mode full
    python c15_ml_theory_depth.py --dataset logd --mode full --sub 15a 15b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import pdist, squareform

from kirby.similarity.distance_config import scipy_metric as scipy_distance_metric
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from experiments._shared import make_nearest_neighbors  # honours --distance-metric

# Turns a fitted estimator into the vector the metric wants -- labels for an
# accuracy-style metric, probabilities for a ranking one.
#
# At module scope since 2026-09-25: imported inside one function it was not
# visible to the module-level scorers, every fold raised NameError, and every
# 15c and 15d score came out nan.
from kirby.find_best_hybrid import (
    _predict_for_metric, nested_cv_outer_folds, nested_cv_cache_namespace,
)
from kirby.scaling import FeatureScaler

from experiments._shared import (
    own_search_kwargs,
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    get_domain_metric,
    get_model_factories,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection, search_on_fold, top_hybrids,
    _distinct_feature_hybrids,
)
from experiments.decided_grid import decided_models, decided_search

# 2026-10-05: the per-fold layout. Stamped on every result block; a file
# without it was written by the whole-dataset version of this script.
RESULT_VERSION = 'c15_per_fold_2026-10-05'
SEED = 42
ONE_FOLD_NOTE = 'one fold, no spread'
# Recipe fields copied into every hybrid row of the result file.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')


# =========================================================================
# Computed once instead of once per call. Not a cut.
# =========================================================================

# Per-fold feature importance.
#
# Importance depends on the fold's training rows and on the importance method,
# and on nothing else -- not the budget, not the downstream model, not the
# allocation method. c15 rebuilds many recipes on the same fold, and every one
# of them recomputed importance from scratch.
#
# Measured 2026-09-24, recorded in tests/experiments/_local_grid.py: importance
# is 62% of a create_hybrid call, 93s of 150s on flab_thermo_tresanco.
#
# The rows below are the ones create_hybrid itself uses -- the training part of
# internal_holdout_split over the fold's training rows (hybrid.py:5988, 6125).
# f31's verify_defaults() checks this exact recipe against letting create_hybrid
# compute it and reports an identical allocation and identical selected
# features, so the numbers this file prints do not change.
_IMPORTANCE_MEMO = {}


def _fold_importance(train_idx, train_reps, y_train, groups_train, method,
                     augs_train, space='reps'):
    """Importance for one fold, computed once and kept.

    `train_idx` is the fold's row numbers and, with `space`, is what the entry
    is filed under, so two folds can never read each other's vector.

    Returns None when augmentations are present: importance is then computed
    over a different set of matrices and this shortcut has not been checked
    against that path. A None return puts the caller back on create_hybrid's
    own computation.
    """
    if augs_train:
        return None
    from kirby.hybrid import compute_feature_importance, internal_holdout_split
    key = (space, np.asarray(train_idx, dtype=np.int64).tobytes(), method)
    if key in _IMPORTANCE_MEMO:
        return _IMPORTANCE_MEMO[key]
    inner_tr, _inner_va = internal_holdout_split(
        len(y_train), y_train, groups_train)
    imp = compute_feature_importance(
        {k: v[inner_tr] for k, v in train_reps.items()},
        y_train[inner_tr], method=method)
    _IMPORTANCE_MEMO[key] = imp
    return imp


# 2026-10-05: per-fold memos. One run() handles one dataset; run() clears them.
#   _SELECTION_MEMO  (fold, budget)               -> fold_standard_selection
#   _HYBRID_MEMO     (fold, source, model, label, aug) -> built hybrid
#   _PCA_MEMO        (fold, n_components)          -> PCA of the concatenation
_SELECTION_MEMO = {}
_HYBRID_MEMO = {}
_PCA_MEMO = {}


def _env_floats(var, default, local_default=None):
    """Comma-separated floats from the environment, else a default.

    `local_default` is used when KIRBY_LOCAL_GRID is set, which is the flag
    tests/experiments/_local_grid.py already uses to mean "this is a laptop
    smoke run, the conditions are cut for compute and nothing measured under
    them is a finding". The server never sets it and gets `default`.
    """
    raw = os.environ.get(var)
    if raw:
        return [float(p.strip()) for p in raw.split(',') if p.strip()]
    if local_default is not None and os.environ.get('KIRBY_LOCAL_GRID'):
        return list(local_default)
    return list(default)


def _env_int(var, default, local_default=None):
    """One integer from the environment, else a default. See _env_floats."""
    raw = os.environ.get(var)
    if raw:
        return int(raw)
    if local_default is not None and os.environ.get('KIRBY_LOCAL_GRID'):
        return int(local_default)
    return int(default)


def _stage(label, t0):
    """Print how long a sub-analysis took, so a capped run says where it died."""
    import time as _t
    print(f"  [c15 timing] {label}: {_t.time() - t0:.0f}s", flush=True)


# =========================================================================
# Per-fold context (2026-10-05)
# =========================================================================

def _make_context(data, config, dataset_name):
    """Everything the four sub-analyses share for one dataset.

    reps: the run's pool, handed whole to every hybrid builder and search
        (run_all loads DATASET_REPS[ds]['general']; handoff 2.4 step 9).
    general: the names of DATASET_REPS[ds]['general'] in the pool. Every
        per-representation loop, and the concatenation behind the PCA, random
        and all-features arms, uses these only.
    folds: nested_cv_outer_folds at config['n_splits'], the same splits the
        per-fold searches are filed under (ocv{n}f{i}s42).
    models: the dataset's decided models that have a factory at the largest
        fold's training-row count (svm stops above 10,000 rows; the drop is
        printed).
    """
    from experiments.complexity.complexity_helpers import DATASET_REPS
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    if groups is None:
        raise ValueError('c15: the dataset has no groups; a group-aware outer '
                         'split is impossible and there is no random fallback')
    groups = np.asarray(groups)
    if dataset_name not in DATASET_REPS:
        raise ValueError(f'c15: {dataset_name!r} has no DATASET_REPS entry, so '
                         f'its representations are undecided')
    general_all = list(DATASET_REPS[dataset_name].get('general') or [])
    general = [r for r in sorted(reps) if r in general_all]
    outside = sorted(r for r in reps if r not in general_all)
    if outside:
        print(f"  WARNING: representations outside DATASET_REPS['general'] in "
              f"this run's pool, left out of every per-representation loop "
              f"and concatenation: {outside}")
    if not general:
        raise ValueError(f'c15: none of DATASET_REPS[{dataset_name!r}]'
                         f"['general'] is in the loaded pool {sorted(reps)}")
    augs = data.get('augmentations') or None
    # c15 compares nothing with and without added features (2.3 item 4). The
    # per-fold searches add an augmentation condition only where Step 8 keeps
    # one for the general search (decided_aug_conditions).
    _aug_kept = [c for c in decided_search(dataset_name, 'general')
                 ['aug_conditions'] if c != 'none']
    if augs and not _aug_kept:
        print("  Augmentation: not searched. DECISIONS.md Step 8 keeps no "
              "augmentation on this dataset's general search.")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_outer = int(config['n_splits'])
    folds = nested_cv_outer_folds(reps, labels, groups, n_outer, SEED,
                                  'regression')
    n_outer = len(folds)

    wanted = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    # 2026-10-05: the row cap (svm 10,000, gp 1,000) is checked against the
    # training rows, the count the per-fold search checks (n_samples=len(tr)).
    # Every model here is fitted on a fold's training rows only. A model is
    # kept when it fits on the largest fold's training rows; one the search
    # would fit on some folds but not all is printed and recorded.
    n_train = [len(tr) for tr, _va in folds]
    factories = get_model_factories(wanted, task_type='regression',
                                    n_samples=max(n_train))
    models = [m for m in wanted if m in factories]
    dropped = [m for m in wanted if m not in factories]
    if dropped:
        _some = get_model_factories(dropped, task_type='regression',
                                    n_samples=min(n_train))
        if _some:
            print(f"  WARNING: {sorted(_some)} fit on some folds' training "
                  f"rows ({min(n_train)} to {max(n_train)} rows) but not all; "
                  f"c15 leaves them out on every fold")
        print(f"  decided models without a factory at {max(n_train)} training "
              f"rows, not fitted: {dropped}")

    X_concat = np.hstack([np.asarray(reps[k]) for k in general])
    print(f"  c15: models {models}; {n_outer} outer fold(s); metric "
          f"{metric_name} ({'lower' if lower_is_better else 'higher'} is "
          f"better); {len(general)} representations, "
          f"{X_concat.shape[1]} features concatenated")
    return {
        'ds': dataset_name, 'reps': reps, 'labels': labels, 'groups': groups,
        'augs': augs, 'general': general, 'X_concat': X_concat,
        'metric_name': metric_name, 'metric_fn': metric_fn,
        'lower_is_better': bool(lower_is_better),
        'folds': folds, 'n_outer': n_outer,
        'models': models, 'models_without_factory': dropped,
        'factories': factories,
    }


def _common_fields(ctx):
    """What every sub-analysis block states about how it was computed."""
    return {
        'result_version': RESULT_VERSION,
        'selection_scope': 'fold_train_rows',
        'metric': ctx['metric_name'],
        'lower_is_better': ctx['lower_is_better'],
        'models': list(ctx['models']),
        'models_without_factory': list(ctx['models_without_factory']),
        'n_outer_folds': ctx['n_outer'],
        'representations': list(ctx['general']),
    }


def _fold_meta(ctx, k):
    tr, va = ctx['folds'][k]
    return {'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
            'cache_namespace': nested_cv_cache_namespace(k, ctx['n_outer'],
                                                         SEED)}


def _spread(values):
    """Mean and spread over folds of finite values; one fold says so."""
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    if not vals:
        return {'mean': None, 'std': None, 'n_folds': 0}
    out = {'mean': float(np.mean(vals)),
           'std': float(np.std(vals)) if len(vals) > 1 else None,
           'n_folds': len(vals)}
    if len(vals) == 1:
        out['spread_note'] = ONE_FOLD_NOTE
    return out


def _fmt(x, spec='.4f'):
    return 'nan' if x is None or not np.isfinite(float(x)) else format(x, spec)


def _hybrid_row(entry, slot, model):
    """The recipe of one chosen hybrid, as stored in the result file."""
    return {'slot': slot, 'model': model,
            **{f: entry.get(f) for f in HYBRID_FIELDS},
            'aug_condition': entry.get('aug_condition', 'none'),
            # Mean over the search's inner folds, inside this outer fold's
            # training rows.
            'training_rows_score': entry.get('score_mean')}


def _fold_selection(ctx, k, budget):
    """Each model's 2 best hybrids at `budget` (features), outer fold k.

    From the fold's general search on the decided grid (fold_standard_selection,
    memoised by fold_searches, so one search per fold whatever the budget).
    A second hybrid whose selected features equal the first's is skipped for
    the next. Returns {model: {'hybrids': [...], 'skipped': [...]} or None}.
    """
    key = (k, int(budget))
    if key not in _SELECTION_MEMO:
        tr, _va = ctx['folds'][k]
        try:
            sel = fold_standard_selection(
                ctx['reps'], ctx['labels'], ctx['groups'], train_idx=tr,
                fold_i=k, n_outer_folds=ctx['n_outer'],
                dataset_name=ctx['ds'], models=list(ctx['models']),
                augmentations=ctx['augs'], task_type='regression',
                metric=ctx['metric_name'], seed=SEED, budget=int(budget))
        except Exception as e:
            print(f"    fold {k} budget {budget}: the per-fold search raised "
                  f"{type(e).__name__}: {e}")
            sel = {m: None for m in ctx['models']}
        out = {}
        for m in ctx['models']:
            s = sel.get(m)
            if s is None:
                out[m] = None
                continue
            out[m] = {'hybrids': list(s.get('hybrids') or []),
                      'skipped': [x.get('label') for x in
                                  (s.get('hybrids_skipped_identical_features')
                                   or [])]}
        _SELECTION_MEMO[key] = out
    return _SELECTION_MEMO[key]


def _built_hybrid(ctx, k, source, model, entry):
    """One chosen hybrid rebuilt on fold k's training rows.

    Returns {'X_tr', 'X_va', 'n_features', 'features_by_rep', 'error'}. The
    rebuild passes the fold search's own per-representation scores
    (blueprint.rep_scores of a per-fold search; dirty-run bug 1) and the
    decided model this hybrid was chosen for.
    """
    cond = entry.get('aug_condition') or 'none'
    key = (k, source, model, entry.get('label'), cond)
    if key in _HYBRID_MEMO:
        return _HYBRID_MEMO[key]
    tr, va = ctx['folds'][k]
    reps, y, g = ctx['reps'], ctx['labels'], ctx['groups']
    tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
    va_reps = {r: np.asarray(X)[va] for r, X in reps.items()}
    augs = ctx['augs'] if cond != 'none' else None
    tr_augs = ({a: np.asarray(X)[tr] for a, X in augs.items()}
               if augs else None)
    va_augs = ({a: np.asarray(X)[va] for a, X in augs.items()}
               if augs else None)
    bp = entry['blueprint']
    out = {'X_tr': None, 'X_va': None, 'n_features': None,
           'features_by_rep': None, 'error': None}
    try:
        X_tr, X_va, fi = build_fold_hybrid(
            tr_reps, va_reps, y[tr], bp,
            augmentations_train=tr_augs, augmentations_val=va_augs,
            groups=g[tr],
            precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
            precomputed_importance=_fold_importance(
                tr, tr_reps, y[tr], g[tr], bp.importance_method, tr_augs),
            **fold_hybrid_scoring_kwargs(
                bp, model=ctx['factories'][model](),
                metric_fn=ctx['metric_fn'],
                lower_is_better=ctx['lower_is_better']))
        out.update(X_tr=X_tr, X_va=X_va, n_features=int(X_tr.shape[1]),
                   features_by_rep={
                       r: int(len(info['selected_indices']))
                       for r, info in fi.items()
                       if isinstance(info, dict)
                       and 'selected_indices' in info})
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
        print(f"    fold {k} {model} {entry.get('label')}: rebuild raised "
              f"{out['error']}")
    _HYBRID_MEMO[key] = out
    return out


def _fold_pca(ctx, k, n_components):
    """PCA of the concatenation, fitted on fold k's training rows.

    PCA cannot give more components than training rows - 1 or features, so the
    count actually used is returned with the matrices.
    """
    tr, va = ctx['folds'][k]
    X = ctx['X_concat']
    n_comp = int(min(n_components, len(tr) - 1, X.shape[1]))
    key = (k, n_comp)
    if key not in _PCA_MEMO:
        sc = FeatureScaler().fit(X[tr])
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr = pca.fit_transform(sc.transform(X[tr]))
        X_va = pca.transform(sc.transform(X[va]))
        _PCA_MEMO[key] = (X_tr, X_va, n_comp)
    return _PCA_MEMO[key]


def _heldout_score(ctx, model, X_tr, X_va, k):
    """Fit `model` on fold k's training rows, score on its held-out rows."""
    tr, va = ctx['folds'][k]
    y = ctx['labels']
    sc = FeatureScaler().fit(X_tr)
    m = ctx['factories'][model]()
    m.fit(sc.transform(X_tr), y[tr])
    pred = _predict_for_metric(m, sc.transform(X_va), ctx['metric_fn'])
    return float(ctx['metric_fn'](y[va], pred))


def _safe_score(ctx, model, X_tr, X_va, k, label):
    try:
        return _heldout_score(ctx, model, X_tr, X_va, k), None
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
        print(f"    fold {k} {model} {label}: fit raised {err}")
        return float('nan'), err


def _compute_concentration_metrics(X, k=5, max_subsample=2000):
    """Compute distance concentration metrics for a feature matrix.

    Returns dict with:
    - far_near_ratio_per_molecule_median / _mean: each molecule's farthest
      distance / its nearest non-zero distance, median and mean across
      molecules (→ 1 means concentrated). Replaced the all-pairs ratio
      2026-10-05 (owner delegated).
    - relative_contrast: (max_dist - min_dist) / mean_dist
    - distance_cv: std(dists) / mean(dists)
    - distance_iqr_ratio: IQR / median
    - hubness_skewness: skewness of k-occurrence counts (Radovanovic 2010)
    - hubness_robin_hood: fraction of total occurrences held by top 1% of hubs
    """
    n = X.shape[0]

    # Pairwise distances (subsample if large)
    if n > max_subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
        idx = np.arange(n)

    _m, _kw = scipy_distance_metric()
    dists = pdist(X_sub, metric=_m, **_kw)

    if len(dists) == 0:
        return {
            'far_near_ratio_per_molecule_median': float('nan'),
            'far_near_ratio_per_molecule_mean': float('nan'),
            'far_near_ratio_n_molecules': 0,
            'relative_contrast': float('nan'),
            'distance_cv': float('nan'), 'distance_iqr_ratio': float('nan'),
            'hubness_skewness': float('nan'), 'hubness_robin_hood': float('nan'),
        }

    min_d = np.min(dists[dists > 0]) if np.any(dists > 0) else 1e-10
    max_d = np.max(dists)
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    median_d = np.median(dists)
    q25, q75 = np.percentile(dists, [25, 75])

    # 2026-10-05, owner delegated, 2026-10-05: the all-pairs ratio stored as
    # 'beyer_ratio' ((largest - smallest non-zero distance over ALL pairs) /
    # smallest) is replaced by Beyer's per-molecule ratio: for each molecule,
    # its farthest distance divided by its nearest non-zero distance, from the
    # same distance matrix. Its median and mean across molecules are stored
    # under new key names; 'beyer_ratio' is no longer written, so no reader
    # gets a different quantity under the old name. A molecule whose
    # distances to every other molecule are all zero has no ratio and is
    # left out of the median and mean.
    _D = squareform(dists)
    np.fill_diagonal(_D, np.nan)
    _far = np.nanmax(_D, axis=1)
    _near = np.min(np.where(_D > 0, _D, np.inf), axis=1)  # nan > 0 is False
    with np.errstate(divide='ignore', invalid='ignore'):
        _per_mol = _far / _near
    _per_mol = _per_mol[np.isfinite(_per_mol) & (_per_mol > 0)]
    far_near_median = (float(np.median(_per_mol)) if _per_mol.size
                       else float('nan'))
    far_near_mean = (float(np.mean(_per_mol)) if _per_mol.size
                     else float('nan'))
    contrast = (max_d - min_d) / mean_d if mean_d > 0 else 0
    cv = std_d / mean_d if mean_d > 0 else 0
    iqr_ratio = (q75 - q25) / median_d if median_d > 0 else 0

    # Hubness: how many times each point appears as a k-NN of others
    k_use = min(k, len(X_sub) - 1)
    nn = make_nearest_neighbors(n_neighbors=k_use + 1, n_jobs=-1)
    nn.fit(X_sub)
    _, indices = nn.kneighbors(X_sub)
    indices = indices[:, 1:]  # exclude self

    # k-occurrence count: N_k(i) = how many times point i appears in others' k-NN
    n_occ = np.zeros(len(X_sub))
    for i in range(len(X_sub)):
        for j in range(k_use):
            n_occ[indices[i, j]] += 1

    hub_skewness = float(skew(n_occ))

    # Robin Hood index: fraction of total occurrences from top 1%
    n_top = max(1, int(0.01 * len(X_sub)))
    sorted_occ = np.sort(n_occ)[::-1]
    robin_hood = float(sorted_occ[:n_top].sum() / n_occ.sum()) if n_occ.sum() > 0 else 0

    return {
        'far_near_ratio_per_molecule_median': far_near_median,
        'far_near_ratio_per_molecule_mean': far_near_mean,
        'far_near_ratio_n_molecules': int(_per_mol.size),
        'relative_contrast': float(contrast),
        'distance_cv': float(cv),
        'distance_iqr_ratio': float(iqr_ratio),
        'hubness_skewness': hub_skewness,
        'hubness_robin_hood': robin_hood,
    }


# =========================================================================
# 15a: Distance Concentration (Beyer et al. 1999)
# =========================================================================

def _run_distance_concentration(ctx, budgets):
    """15a: Measure distance concentration at each dimensionality.

    Per-molecule Beyer ratio: each molecule's farthest / nearest non-zero
    distance → 1 as d → ∞ (median and mean across molecules; was an
    all-pairs ratio until 2026-10-05).
    Also measures:
    - Relative contrast (Aggarwal et al. 2001)
    - Pairwise distance CV (coefficient of variation)
    - Hubness: skewness of k-occurrence distribution (Radovanovic 2010)

    2026-10-05, per fold: for each outer fold, every space is measured on that
    fold's held-out rows -- each representation, the PCA of the concatenation
    at each budget (fitted on the training rows), and each decided model's 2
    hybrids at each budget (chosen and built on the training rows). So the
    hybrid and the spaces beside it are read on the same rows, none of which
    chose the hybrid's features. 'per_rep' and 'pca' over every row are kept
    as before; neither uses the labels.
    """
    print("\n  15a: Distance Concentration Analysis")
    print("  " + "-" * 60)

    reps = ctx['reps']
    n_samples = len(ctx['labels'])
    X_concat = ctx['X_concat']

    results = {**_common_fields(ctx), 'budgets': list(budgets),
               'per_rep': {}, 'pca': {}, 'per_fold': []}

    # --- Per-rep analysis over every row (no labels involved) ---
    for rep_name in ctx['general']:
        X = np.asarray(reps[rep_name])
        X_s = FeatureScaler().fit_transform(X)
        metrics = _compute_concentration_metrics(X_s, k=5)
        results['per_rep'][rep_name] = {'dim': X.shape[1], **metrics}
        print(f"    {rep_name:<20} dim={X.shape[1]:>5}  "
              f"far/near_med={metrics['far_near_ratio_per_molecule_median']:.4f}  "
              f"cv={metrics['distance_cv']:.4f}  "
              f"hubness={metrics['hubness_skewness']:.3f}")

    # --- PCA over every row (unsupervised) ---
    X_concat_s = FeatureScaler().fit_transform(X_concat)
    for budget in sorted(budgets):
        n_comp = min(budget, X_concat.shape[1], n_samples - 1)
        try:
            X_pca = PCA(n_components=n_comp,
                        random_state=SEED).fit_transform(X_concat_s)
            results['pca'][str(budget)] = {
                'dim': n_comp, **_compute_concentration_metrics(X_pca, k=5)}
        except Exception as e:
            print(f"      pca budget={budget}: FAILED - {e}")

    # --- Per fold, on the held-out rows ---
    for k, (tr, va) in enumerate(ctx['folds']):
        fold = {**_fold_meta(ctx, k), 'reps': {}, 'budgets': {}}
        for rep_name in ctx['general']:
            X = np.asarray(reps[rep_name])
            try:
                X_va_s = FeatureScaler().fit(X[tr]).transform(X[va])
                fold['reps'][rep_name] = {
                    'dim': X.shape[1],
                    **_compute_concentration_metrics(X_va_s, k=5)}
            except Exception as e:
                fold['reps'][rep_name] = {'error': f'{type(e).__name__}: {e}'}
        for budget in sorted(budgets):
            b_rec = {'pca': None, 'models': {}}
            try:
                _Xtr_p, X_va_p, n_comp = _fold_pca(ctx, k, budget)
                b_rec['pca'] = {'dim': n_comp,
                                **_compute_concentration_metrics(X_va_p, k=5)}
            except Exception as e:
                b_rec['pca'] = {'error': f'{type(e).__name__}: {e}'}
            sel = _fold_selection(ctx, k, budget)
            for m in ctx['models']:
                s = sel.get(m)
                if s is None:
                    b_rec['models'][m] = {'error': 'the per-fold search failed'}
                    continue
                rows = []
                for i, e in enumerate(s['hybrids'], 1):
                    row = _hybrid_row(e, f'hybrid{i}', m)
                    h = _built_hybrid(ctx, k, 'general', m, e)
                    row['n_features'] = h['n_features']
                    row['features_by_rep'] = h['features_by_rep']
                    if h['error']:
                        row['error'] = h['error']
                    else:
                        try:
                            X_va_h = FeatureScaler().fit(h['X_tr']).transform(
                                h['X_va'])
                            row.update(_compute_concentration_metrics(
                                X_va_h, k=5))
                        except Exception as ex:
                            row['error'] = f'{type(ex).__name__}: {ex}'
                    rows.append(row)
                    if 'error' not in row:
                        print(f"    fold {k} budget={budget:>4} {m:<9} "
                              f"hybrid{i} dim={row['n_features']:>4}  "
                              f"far/near_med="
                              f"{row['far_near_ratio_per_molecule_median']:.4f}  "
                              f"cv={row['distance_cv']:.4f}  "
                              f"hubness={row['hubness_skewness']:.3f}")
                b_rec['models'][m] = {
                    'hybrids': rows,
                    'hybrids_skipped_identical_features': s['skipped']}
            fold['budgets'][str(budget)] = b_rec
        results['per_fold'].append(fold)

    # --- Summary over folds (held-out rows) ---
    summ = {'reps': {}, 'pca': {}, 'hybrids': {}}
    keys = ('far_near_ratio_per_molecule_median', 'distance_cv',
            'hubness_skewness')
    for rep_name in ctx['general']:
        summ['reps'][rep_name] = {
            kk: _spread([f['reps'].get(rep_name, {}).get(kk)
                         for f in results['per_fold']]) for kk in keys}
    for budget in sorted(budgets):
        b = str(budget)
        summ['pca'][b] = {kk: _spread([
            (f['budgets'][b]['pca'] or {}).get(kk)
            for f in results['per_fold']]) for kk in keys}
        for m in ctx['models']:
            for i in (1, 2):
                vals = {kk: [] for kk in keys + ('n_features',)}
                for f in results['per_fold']:
                    rows = f['budgets'][b]['models'].get(m, {}).get('hybrids')
                    if rows and len(rows) >= i and 'error' not in rows[i - 1]:
                        for kk in vals:
                            vals[kk].append(rows[i - 1].get(kk))
                summ['hybrids'][f'{b}__{m}__hybrid{i}'] = {
                    kk: _spread(v) for kk, v in vals.items()}
    results['summary_heldout'] = summ

    print(f"\n    Summary (held-out rows, mean over folds):")
    print(f"    {'Space':<30} {'Dim':>6} {'FarNear':>8} {'CV':>8} {'Hubness':>8}")
    for rep_name in ctx['general']:
        s = summ['reps'][rep_name]
        print(f"    {rep_name:<30} {results['per_rep'][rep_name]['dim']:>6} "
              f"{_fmt(s[keys[0]]['mean']):>8} {_fmt(s[keys[1]]['mean']):>8} "
              f"{_fmt(s[keys[2]]['mean'], '.3f'):>8}")
    for name, s in summ['hybrids'].items():
        dim = s['n_features']['mean']
        print(f"    {name:<30} {_fmt(dim, '.0f'):>6} "
              f"{_fmt(s[keys[0]]['mean']):>8} {_fmt(s[keys[1]]['mean']):>8} "
              f"{_fmt(s[keys[2]]['mean'], '.3f'):>8}")
    return results


def _compute_kernel_metrics(X, y, gamma=None):
    """Compute RBF kernel matrix metrics.

    Uses the default sklearn gamma = 1 / (n_features * var(X)).
    """
    n, d = X.shape

    # Auto gamma (sklearn default for RBF)
    if gamma is None:
        var = np.var(X)
        gamma = 1.0 / (d * var) if var > 0 else 1.0

    # Compute RBF kernel matrix
    # Squared Euclidean, and NOT --distance-metric: exp(-gamma * ||x-y||^2) is
    # what an RBF kernel is. Another distance here is a different kernel, not
    # the same kernel measured differently.
    sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-gamma * sq_dists)

    # Off-diagonal entries only
    mask = ~np.eye(n, dtype=bool)
    k_off = K[mask]

    k_mean = float(np.mean(k_off))
    k_std = float(np.std(k_off))
    k_min = float(np.min(k_off))
    k_max = float(np.max(k_off))
    k_cv = k_std / k_mean if k_mean > 0 else 0

    # Condition number
    eigvals = np.linalg.eigvalsh(K)
    eigvals_pos = eigvals[eigvals > 1e-12]
    if len(eigvals_pos) > 0:
        cond = eigvals_pos[-1] / eigvals_pos[0]
        log_cond = float(np.log10(cond)) if cond > 0 else float('inf')
    else:
        log_cond = float('inf')

    # Effective rank of kernel matrix
    eigvals_norm = eigvals_pos / eigvals_pos.sum()
    k_erank = float(np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm + 1e-20))))

    # Participation ratio
    part_ratio = float((eigvals_pos.sum()) ** 2 / (eigvals_pos ** 2).sum())

    # Kernel-target alignment (Cristianini et al. 2001)
    # alignment = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    y_centered = y - y.mean()
    yy = np.outer(y_centered, y_centered)
    alignment = np.sum(K * yy) / (np.linalg.norm(K, 'fro') * np.linalg.norm(yy, 'fro') + 1e-10)

    return {
        'kernel_mean': k_mean,
        'kernel_std': k_std,
        'kernel_min': k_min,
        'kernel_max': k_max,
        'kernel_cv': float(k_cv),
        'log_condition': log_cond,
        'kernel_effective_rank': k_erank,
        'kernel_participation_ratio': part_ratio,
        'kernel_target_alignment': float(alignment),
        'gamma': float(gamma),
    }


# =========================================================================
# 15b: Kernel Matrix Analysis
# =========================================================================

# Kernel matrices are n x n; at most this many rows are used in one.
KERNEL_MAX_ROWS = 1000

# 2026-10-05, owner delegated, 2026-10-05: these six values are MULTIPLIERS of
# each space's own default length scale, not absolute gammas. gamma =
# multiplier / (n_features * mean feature variance on the training rows), the
# sklearn 'scale' default times the multiplier. A fixed absolute grid made the
# kernel the identity matrix on wide spaces (results audit C15c). Same six
# settings per space, so the same compute. The absolute gamma used is stored
# beside each multiplier.
GAMMA_MULTIPLIERS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _heldout_kernel(X_train, X_test, y_test, multiplier=1.0):
    """Kernel metrics on held-out rows, with everything fitted on training rows.

    The scaler and the RBF length scale both come from the training rows, so
    nothing about the held-out rows shapes the kernel they are read in.
    Returns the metrics dict with 'gamma' the absolute gamma used.
    """
    sc = FeatureScaler().fit(X_train)
    Xtr_s, Xte_s = sc.transform(X_train), sc.transform(X_test)
    d = Xtr_s.shape[1]
    var = float(np.var(Xtr_s, axis=0).mean())
    gamma = multiplier / (d * var) if d > 0 and var > 0 else float(multiplier)
    return _compute_kernel_metrics(Xte_s, y_test, gamma=gamma)


def _run_kernel_analysis(ctx, budgets):
    """15b: Kernel matrix properties at different dimensionalities.

    For RBF kernel: k(x,x') = exp(-γ||x-x'||²).
    In high-D, all ||x-x'||² converge → all kernel values converge → matrix
    becomes near-singular and the GP/SVM loses discriminative power.

    Measures:
    - Kernel value distribution: mean, std, min, max of off-diagonal entries
    - Condition number of K (log10)
    - Eigenspectrum: effective rank, participation ratio
    - Kernel-target alignment: normalized Frobenius inner product <K, yy^T>

    2026-10-05, per fold: each outer fold's held-out rows (at most
    KERNEL_MAX_ROWS of them, drawn with seed 42) are where every kernel is
    read -- each representation and each decided model's 2 hybrids at each
    budget. Scaler and length scale come from the fold's training rows, and
    the hybrids are chosen and built on them. This replaces the one 80/20
    split the held-out alignment used before. 'per_rep' over at most
    KERNEL_MAX_ROWS of every row is kept as before (no labels choose a
    representation's features).

    The length-scale sweep reads the highest-dimensional representation and
    each model's 2 hybrids at the SMALLEST decided budget. It read the
    budget-50 hybrids before 2026-10-05; 50 features is no longer a decided
    budget on any dataset.
    """
    print("\n  15b: Kernel Matrix Analysis")
    print("  " + "-" * 60)

    reps = ctx['reps']
    labels = ctx['labels']
    n_samples = len(labels)

    results = {**_common_fields(ctx), 'budgets': list(budgets),
               'kernel_max_rows': KERNEL_MAX_ROWS,
               'gamma_multipliers': list(GAMMA_MULTIPLIERS),
               'gamma_rule': ('gamma = multiplier / (n_features * mean '
                              'feature variance on the training rows)'),
               'per_rep': {}, 'per_fold': []}

    # --- Per-rep kernel analysis over (a subsample of) every row ---
    if n_samples > KERNEL_MAX_ROWS:
        idx = np.sort(np.random.RandomState(SEED).choice(
            n_samples, KERNEL_MAX_ROWS, replace=False))
    else:
        idx = np.arange(n_samples)
    for rep_name in ctx['general']:
        X = np.asarray(reps[rep_name])[idx]
        try:
            metrics = _compute_kernel_metrics(FeatureScaler().fit_transform(X),
                                              labels[idx])
            results['per_rep'][rep_name] = {'dim': X.shape[1], **metrics}
            print(f"    {rep_name:<20} dim={X.shape[1]:>5}  "
                  f"cond={metrics['log_condition']:>7.1f}  "
                  f"k_cv={metrics['kernel_cv']:.4f}  "
                  f"align_all_rows={metrics['kernel_target_alignment']:.4f}")
        except Exception as e:
            print(f"    {rep_name}: FAILED - {e}")

    highest_rep = max(ctx['general'], key=lambda r: np.asarray(reps[r]).shape[1])
    sweep_budget = min(budgets) if budgets else None
    results['gamma_sweep_spaces'] = {
        'representation': highest_rep, 'hybrid_budget': sweep_budget}

    # --- Per fold, on the held-out rows ---
    for k, (tr, va) in enumerate(ctx['folds']):
        if len(va) > KERNEL_MAX_ROWS:
            sub = np.sort(np.random.RandomState(SEED).choice(
                len(va), KERNEL_MAX_ROWS, replace=False))
        else:
            sub = np.arange(len(va))
        y_te = labels[va][sub]
        fold = {**_fold_meta(ctx, k), 'n_kernel_rows': int(len(sub)),
                'reps': {}, 'budgets': {}, 'gamma_sweep': {}}

        def _sweep(name, X_tr, X_te):
            row = {'dim': int(X_tr.shape[1]), 'gamma_multipliers': {}}
            for g in GAMMA_MULTIPLIERS:
                try:
                    m_ = _heldout_kernel(X_tr, X_te, y_te, multiplier=g)
                    row['gamma_multipliers'][g] = {
                        'gamma_absolute': m_['gamma'],
                        'log_condition': m_['log_condition'],
                        'kernel_cv': m_['kernel_cv'],
                        'kernel_target_alignment':
                            m_['kernel_target_alignment']}
                except Exception as e:
                    row['gamma_multipliers'][g] = {
                        'error': f'{type(e).__name__}: {e}'}
            fold['gamma_sweep'][name] = row

        for rep_name in ctx['general']:
            X = np.asarray(reps[rep_name])
            try:
                fold['reps'][rep_name] = {
                    'dim': X.shape[1],
                    **_heldout_kernel(X[tr], X[va][sub], y_te)}
            except Exception as e:
                fold['reps'][rep_name] = {'error': f'{type(e).__name__}: {e}'}
        X_hi = np.asarray(reps[highest_rep])
        _sweep(highest_rep, X_hi[tr], X_hi[va][sub])

        for budget in sorted(budgets):
            b_rec = {'models': {}}
            sel = _fold_selection(ctx, k, budget)
            for m in ctx['models']:
                s = sel.get(m)
                if s is None:
                    b_rec['models'][m] = {'error': 'the per-fold search failed'}
                    continue
                rows = []
                for i, e in enumerate(s['hybrids'], 1):
                    row = _hybrid_row(e, f'hybrid{i}', m)
                    h = _built_hybrid(ctx, k, 'general', m, e)
                    row['n_features'] = h['n_features']
                    if h['error']:
                        row['error'] = h['error']
                    else:
                        try:
                            row.update(_heldout_kernel(
                                h['X_tr'], h['X_va'][sub], y_te))
                        except Exception as ex:
                            row['error'] = f'{type(ex).__name__}: {ex}'
                        if budget == sweep_budget:
                            _sweep(f'{m}__hybrid{i}', h['X_tr'],
                                   h['X_va'][sub])
                    rows.append(row)
                    if 'error' not in row:
                        print(f"    fold {k} budget={budget:>4} {m:<9} "
                              f"hybrid{i} dim={row['n_features']:>4}  "
                              f"cond={row['log_condition']:>7.1f}  "
                              f"k_cv={row['kernel_cv']:.4f}  "
                              f"align_heldout="
                              f"{row['kernel_target_alignment']:.4f}")
                b_rec['models'][m] = {
                    'hybrids': rows,
                    'hybrids_skipped_identical_features': s['skipped']}
            fold['budgets'][str(budget)] = b_rec
        results['per_fold'].append(fold)

    # --- Summary over folds (held-out rows) ---
    keys = ('log_condition', 'kernel_cv', 'kernel_target_alignment',
            'kernel_effective_rank')
    summ = {'reps': {}, 'hybrids': {}, 'gamma_sweep': {}}
    for rep_name in ctx['general']:
        summ['reps'][rep_name] = {kk: _spread(
            [f['reps'].get(rep_name, {}).get(kk) for f in results['per_fold']])
            for kk in keys}
    for budget in sorted(budgets):
        b = str(budget)
        for m in ctx['models']:
            for i in (1, 2):
                vals = {kk: [] for kk in keys + ('n_features',)}
                for f in results['per_fold']:
                    rows = f['budgets'][b]['models'].get(m, {}).get('hybrids')
                    if rows and len(rows) >= i and 'error' not in rows[i - 1]:
                        for kk in vals:
                            vals[kk].append(rows[i - 1].get(kk))
                summ['hybrids'][f'{b}__{m}__hybrid{i}'] = {
                    kk: _spread(v) for kk, v in vals.items()}
    names = sorted({n for f in results['per_fold'] for n in f['gamma_sweep']})
    for n in names:
        summ['gamma_sweep'][n] = {g: _spread([
            f['gamma_sweep'].get(n, {}).get('gamma_multipliers', {})
            .get(g, {}).get('log_condition') for f in results['per_fold']])
            for g in GAMMA_MULTIPLIERS}
    results['summary_heldout'] = summ

    print(f"\n    Length-scale sensitivity (held-out logCond, mean over folds):")
    print(f"    {'Space':<24} | " +
          "  ".join(f"x{g:<8}" for g in GAMMA_MULTIPLIERS))
    for n in names:
        print(f"    {n:<24} | " + "  ".join(
            f"{_fmt(summ['gamma_sweep'][n][g]['mean'], '.1f'):>9}"
            for g in GAMMA_MULTIPLIERS))

    print(f"\n    Summary (held-out rows, mean over folds):")
    print(f"    {'Space':<30} {'Dim':>6} {'logCond':>8} {'K_cv':>8} "
          f"{'Align':>8} {'EigRank':>8}")
    for rep_name in ctx['general']:
        s = summ['reps'][rep_name]
        print(f"    {rep_name:<30} "
              f"{np.asarray(reps[rep_name]).shape[1]:>6} "
              f"{_fmt(s['log_condition']['mean'], '.1f'):>8} "
              f"{_fmt(s['kernel_cv']['mean']):>8} "
              f"{_fmt(s['kernel_target_alignment']['mean']):>8} "
              f"{_fmt(s['kernel_effective_rank']['mean'], '.1f'):>8}")
    for name, s in summ['hybrids'].items():
        print(f"    {name:<30} {_fmt(s['n_features']['mean'], '.0f'):>6} "
              f"{_fmt(s['log_condition']['mean'], '.1f'):>8} "
              f"{_fmt(s['kernel_cv']['mean']):>8} "
              f"{_fmt(s['kernel_target_alignment']['mean']):>8} "
              f"{_fmt(s['kernel_effective_rank']['mean'], '.1f'):>8}")
    return results



# =========================================================================
# 15c: Double Descent Probe
# =========================================================================

def _c15c_fold_hybrids(ctx, k, budget):
    """Each model's 2 best hybrids at `budget` from fold k's own 15c search.

    15c's budgets sit around the row count and are not decided budgets, so the
    fold's general search holds none of them. Each (fold, budget) gets its own
    search on the fold's training rows: the decided grid on every other axis
    (own_search_kwargs with the real dataset name), budgets=[budget], pruning
    off, filed under the fold's namespace with condition_suffix
    'c15c_b<budget>' so it never shares a stored result with the general
    search. A second hybrid whose selected features equal the first's is
    skipped for the next, as in fold_standard_selection.

    Returns ({model: {'hybrids', 'skipped'} or None}, [failure records]).
    """
    tr, _va = ctx['folds'][k]
    kw = own_search_kwargs(len(tr), models=list(ctx['models']),
                           dataset_name=ctx['ds'], tier='general')
    kw['budgets'] = [int(budget)]
    failures = []
    try:
        searches = search_on_fold(
            ctx['reps'], ctx['labels'], ctx['groups'], train_idx=tr,
            fold_i=k, n_outer_folds=ctx['n_outer'], dataset_name=ctx['ds'],
            augmentations=ctx['augs'], task_type='regression',
            metric=ctx['metric_name'], seed=SEED,
            condition_suffix=f'c15c_b{int(budget)}', fixed_grid=kw)
    except Exception as e:
        failures.append({'fold': int(k), 'budget': int(budget),
                         'error': f'{type(e).__name__}: {e}'})
        print(f"    fold {k} 15c search at budget {budget} raised "
              f"{failures[-1]['error']}")
        return {m: None for m in ctx['models']}, failures
    for f in (getattr(searches, 'failures', None) or []):
        failures.append({'fold': int(k), 'budget': int(budget),
                         'aug_condition': f.get('aug_condition'),
                         'error': f.get('error') or f.get('error_type')})
    base = searches.get('none')
    if base is None:
        print(f"    fold {k} 15c search at budget {budget} failed: "
              f"{[f['error'] for f in failures]}")
        return {m: None for m in ctx['models']}, failures
    out = {}
    for m in ctx['models']:
        pooled = []
        for cond, srch in searches.items():
            if srch is None:
                continue
            th = top_hybrids(srch, m, n=2 + 4, budget=int(budget),
                             require_full=False,
                             context=f"c15c {ctx['ds']} fold {k} {m} "
                                     f"b={budget} {cond}")
            pooled += [dict(e, aug_condition=cond) for e in th['selected']]
        lower = base.lower_is_better
        pooled.sort(key=lambda e: (e['score_mean'] if lower
                                   else -e['score_mean'], e['label']))
        if len(pooled) > 1:
            chosen, skipped = _distinct_feature_hybrids(
                pooled, 2, reps=ctx['reps'], labels=ctx['labels'],
                groups=ctx['groups'], train_idx=tr, augmentations=ctx['augs'],
                model=m, dataset_name=ctx['ds'], task_type='regression',
                metric=ctx['metric_name'], domain=None, lower_is_better=lower)
        else:
            chosen, skipped = pooled[:2], []
        out[m] = {'hybrids': chosen,
                  'skipped': [x.get('label') for x in skipped]}
    return out, failures


def _run_double_descent(ctx, config):
    """15c: Probe for double descent near the interpolation threshold.

    Classical double descent: test error peaks when p ≈ n (number of features
    ≈ number of samples), then decreases again in the overparameterized regime.

    For KIRBy hybrids, p = budget. If budget << n always, the hybrid never
    enters the interpolation regime → no double descent → stable performance.

    Sweep budget through fine granularity around n_samples to detect any
    instability. Compare hybrid vs PCA vs random feature subsets.

    2026-10-05, per fold: for each outer fold and each budget, each decided
    model's 2 best hybrids come from a search on that fold's training rows at
    that budget (_c15c_fold_hybrids). Each is rebuilt on the training rows and
    scored on the held-out rows with that model. PCA of the concatenation and
    a random subset of its features at the same budget are scored with the
    same model. The random forest, the Ridge arms and the distance hybrid are
    gone (ridge removed everywhere; no distance hybrid in c15).
    """
    print("\n  15c: Double Descent Probe")
    print("  " + "-" * 60)

    n_samples = len(ctx['labels'])
    X_concat = ctx['X_concat']
    total_dim = X_concat.shape[1]

    # 2026-09-12, owner's decision: 15c's budget list is 0.25, 0.8, 0.9, 1.0,
    # 1.1, 1.2 and 4.0 times the row count (every row of the dataset).
    # KIRBY_C15_BUDGET_FACTORS overrides this list. Under KIRBY_LOCAL_GRID
    # (laptop smoke runs only) it is 0.25, 0.9, 1.1, 4.0.
    #
    # The one ceiling kept is the number of features there are to select
    # from: a budget above total_dim asks for more features than exist. The
    # 3 x row count and the 2048 caps were compute caps and are gone.
    BUDGET_FACTORS = _env_floats('KIRBY_C15_BUDGET_FACTORS',
                                 [0.25, 0.8, 0.9, 1.0, 1.1, 1.2, 4.0],
                                 local_default=[0.25, 0.9, 1.1, 4.0])
    all_budgets = sorted(set(int(n_samples * f) for f in BUDGET_FACTORS))
    all_budgets = [b for b in all_budgets if 5 <= b <= total_dim]

    print(f"    n_samples={n_samples}, total_dim={total_dim}")
    print(f"    Budgets: {all_budgets}")

    results = {**_common_fields(ctx),
               'n_samples': n_samples, 'total_dim': total_dim,
               'budget_factors': list(BUDGET_FACTORS),
               'budgets': all_budgets,
               'search_failures': [], 'per_fold': []}

    # The random subset is drawn once per budget, on the feature numbers only
    # (no labels), so every fold and every model reads the same features.
    random_idx = {}
    for b in all_budgets:
        rng = np.random.RandomState(SEED)
        random_idx[b] = rng.choice(total_dim, min(b, total_dim),
                                   replace=False)

    for k, (tr, va) in enumerate(ctx['folds']):
        fold = {**_fold_meta(ctx, k), 'budgets': {}}
        for budget in all_budgets:
            sel, fails = _c15c_fold_hybrids(ctx, k, budget)
            results['search_failures'] += fails
            b_rec = {'p_over_n_train': float(budget) / len(tr), 'models': {}}
            X_tr_p, X_va_p, n_comp = _fold_pca(ctx, k, budget)
            X_tr_r = X_concat[tr][:, random_idx[budget]]
            X_va_r = X_concat[va][:, random_idx[budget]]
            for m in ctx['models']:
                rec = {'hybrids': [], 'pca': None, 'random': None}
                s = sel.get(m)
                if s is None:
                    rec['error'] = 'the per-fold 15c search failed'
                else:
                    rec['hybrids_skipped_identical_features'] = s['skipped']
                    if len(s['hybrids']) < 2:
                        rec['shortfall'] = (f"asked for 2 hybrids, the search "
                                            f"gave {len(s['hybrids'])}")
                    for i, e in enumerate(s['hybrids'], 1):
                        row = _hybrid_row(e, f'hybrid{i}', m)
                        h = _built_hybrid(ctx, k, f'c15c_b{budget}', m, e)
                        row['n_features'] = h['n_features']
                        row['features_by_rep'] = h['features_by_rep']
                        if h['error']:
                            row['error'] = h['error']
                            row['heldout_score'] = float('nan')
                        else:
                            row['heldout_score'], err = _safe_score(
                                ctx, m, h['X_tr'], h['X_va'], k, f'hybrid{i}')
                            if err:
                                row['error'] = err
                        rec['hybrids'].append(row)
                sc, err = _safe_score(ctx, m, X_tr_p, X_va_p, k, 'pca')
                rec['pca'] = {'n_components': n_comp, 'heldout_score': sc,
                              **({'error': err} if err else {})}
                sc, err = _safe_score(ctx, m, X_tr_r, X_va_r, k, 'random')
                rec['random'] = {'n_features': int(len(random_idx[budget])),
                                 'heldout_score': sc,
                                 **({'error': err} if err else {})}
                b_rec['models'][m] = rec
                hy = '  '.join(
                    f"hybrid{i}={_fmt(r.get('heldout_score'))} "
                    f"[{r.get('n_features')} feat]"
                    for i, r in enumerate(rec['hybrids'], 1))
                print(f"    fold {k} budget={budget:>5} {m:<9} {hy}  "
                      f"pca={_fmt(rec['pca']['heldout_score'])} "
                      f"[{n_comp} comp]  "
                      f"random={_fmt(rec['random']['heldout_score'])} "
                      f"[{len(random_idx[budget])} feat]", flush=True)
            fold['budgets'][str(budget)] = b_rec
        results['per_fold'].append(fold)

    # --- Summary over folds, in the dataset's metric ---
    summ = {}
    for budget in all_budgets:
        b = str(budget)
        summ[b] = {}
        for m in ctx['models']:
            arms = {}
            for i in (1, 2):
                sc, w = [], []
                for f in results['per_fold']:
                    rows = f['budgets'][b]['models'][m]['hybrids']
                    if len(rows) >= i:
                        sc.append(rows[i - 1].get('heldout_score'))
                        w.append(rows[i - 1].get('n_features'))
                arms[f'hybrid{i}'] = {'heldout_score': _spread(sc),
                                      'n_features': _spread(w)}
            for arm, wkey in (('pca', 'n_components'), ('random', 'n_features')):
                arms[arm] = {
                    'heldout_score': _spread(
                        [f['budgets'][b]['models'][m][arm]['heldout_score']
                         for f in results['per_fold']]),
                    'n_features': _spread(
                        [f['budgets'][b]['models'][m][arm][wkey]
                         for f in results['per_fold']])}
            summ[b][m] = arms
    results['summary'] = summ

    print(f"\n    Summary ({ctx['metric_name']}, "
          f"{'lower' if ctx['lower_is_better'] else 'higher'} is better; "
          f"mean over folds; features in brackets):")
    for budget in all_budgets:
        b = str(budget)
        for m in ctx['models']:
            a = summ[b][m]
            print(f"    budget={budget:>5} (p/n={budget / n_samples:.2f}) "
                  f"{m:<9} " + "  ".join(
                      f"{arm}={_fmt(a[arm]['heldout_score']['mean'])} "
                      f"[{_fmt(a[arm]['n_features']['mean'], '.0f')}]"
                      for arm in ('hybrid1', 'hybrid2', 'pca', 'random')))
    return results


# =========================================================================
# 15d: Approximation Theory (Rademacher complexity)
# =========================================================================

def _memorisation(ctx, model, X_tr, k, n_trials):
    """How much of a permuted label the model fits on fold k's training rows.

    Each trial permutes the labels of every row (seed 42, a fresh generator
    per call, so every arm sees the same permutations), fits `model` on the
    fold's training rows against the permuted labels, and scores R² on those
    same rows, floored at 0. Returns the mean over trials, or nan.
    """
    tr, _va = ctx['folds'][k]
    y = ctx['labels']
    rng = np.random.RandomState(SEED)
    Xs = FeatureScaler().fit_transform(X_tr)
    scores = []
    for _t in range(n_trials):
        y_r = rng.permutation(y)[tr]
        try:
            m = ctx['factories'][model]()
            m.fit(Xs, y_r)
            scores.append(max(0.0, float(r2_score(y_r, m.predict(Xs)))))
        except Exception as e:
            print(f"    fold {k} {model}: permuted-label fit raised "
                  f"{type(e).__name__}: {e}")
    return float(np.mean(scores)) if scores else float('nan')


def _run_approximation_theory(ctx, budgets):
    """15d: Empirical Rademacher complexity and generalization bounds.

    Rademacher complexity measures how well a hypothesis class can fit random
    labels. High Rademacher = model can memorize noise = poor generalization.

    For a linear model on d-dimensional data: R_n ∝ sqrt(d/n).
    Compressing from D to d should reduce Rademacher by sqrt(d/D).

    We estimate empirically: fit model on random labels, measure R².
    If hybrid compression reduces this R² → tighter generalization bound.

    2026-10-05, per fold and on the dataset's two decided models: the model
    that fits the permuted labels, and the model scored on the real labels, is
    each decided model in turn. It was Ridge(alpha=1.0) for both, which is
    removed everywhere. Per outer fold and decided budget, each model's 2
    hybrids (from the fold's general search) and the PCA of the concatenation
    at the same budget; once per fold, all features concatenated. The
    all-features row is now the raw concatenation; before 2026-10-05 its real
    score was a PCA of it at training rows - 1 components (dirty run, c15).
    """
    print("\n  15d: Approximation Theory (Rademacher Complexity)")
    print("  " + "-" * 60)

    n_samples = len(ctx['labels'])
    X_concat = ctx['X_concat']

    # KIRBY_C15_RADEMACHER_TRIALS overrides this. Unset it is 5, the shipped
    # value; under KIRBY_LOCAL_GRID it is 2. A trial is one label permutation;
    # the features are identical across trials.
    n_random_trials = _env_int('KIRBY_C15_RADEMACHER_TRIALS', 5,
                               local_default=2)

    results = {**_common_fields(ctx), 'n_samples': n_samples,
               'budgets': list(budgets), 'n_random_trials': n_random_trials,
               'memorisation_score': ('R2 on the training rows of a fit to '
                                      'permuted labels, floored at 0, mean '
                                      'over trials'),
               'per_fold': []}

    for k, (tr, va) in enumerate(ctx['folds']):
        fold = {**_fold_meta(ctx, k), 'full': {}, 'budgets': {}}
        X_tr_f, X_va_f = X_concat[tr], X_concat[va]
        for m in ctx['models']:
            sc, err = _safe_score(ctx, m, X_tr_f, X_va_f, k, 'all features')
            fold['full'][m] = {
                'n_features': int(X_concat.shape[1]), 'heldout_score': sc,
                'memorisation': _memorisation(ctx, m, X_tr_f, k,
                                              n_random_trials),
                **({'error': err} if err else {})}
            print(f"    fold {k} {m:<9} all {X_concat.shape[1]} features: "
                  f"memorisation={_fmt(fold['full'][m]['memorisation'])}  "
                  f"real_{ctx['metric_name']}={_fmt(sc)}", flush=True)
        for budget in sorted(budgets):
            b_rec = {'models': {}}
            X_tr_p, X_va_p, n_comp = _fold_pca(ctx, k, budget)
            sel = _fold_selection(ctx, k, budget)
            for m in ctx['models']:
                rec = {'hybrids': [], 'pca': None}
                s = sel.get(m)
                if s is None:
                    rec['error'] = 'the per-fold search failed'
                else:
                    rec['hybrids_skipped_identical_features'] = s['skipped']
                    for i, e in enumerate(s['hybrids'], 1):
                        row = _hybrid_row(e, f'hybrid{i}', m)
                        h = _built_hybrid(ctx, k, 'general', m, e)
                        row['n_features'] = h['n_features']
                        if h['error']:
                            row['error'] = h['error']
                            row['heldout_score'] = float('nan')
                            row['memorisation'] = float('nan')
                        else:
                            row['heldout_score'], err = _safe_score(
                                ctx, m, h['X_tr'], h['X_va'], k, f'hybrid{i}')
                            if err:
                                row['error'] = err
                            row['memorisation'] = _memorisation(
                                ctx, m, h['X_tr'], k, n_random_trials)
                        rec['hybrids'].append(row)
                sc, err = _safe_score(ctx, m, X_tr_p, X_va_p, k, 'pca')
                rec['pca'] = {'n_components': n_comp, 'heldout_score': sc,
                              'memorisation': _memorisation(
                                  ctx, m, X_tr_p, k, n_random_trials),
                              **({'error': err} if err else {})}
                b_rec['models'][m] = rec
                hy = '  '.join(
                    f"hybrid{i}: mem={_fmt(r.get('memorisation'))} "
                    f"real={_fmt(r.get('heldout_score'))}"
                    for i, r in enumerate(rec['hybrids'], 1))
                print(f"    fold {k} budget={budget:>5} {m:<9} {hy}  pca: "
                      f"mem={_fmt(rec['pca']['memorisation'])} "
                      f"real={_fmt(sc)}", flush=True)
            fold['budgets'][str(budget)] = b_rec
        results['per_fold'].append(fold)

    # --- Summary over folds ---
    summ = {'full': {}, 'budgets': {}}
    for m in ctx['models']:
        summ['full'][m] = {kk: _spread([f['full'][m][kk]
                                        for f in results['per_fold']])
                           for kk in ('heldout_score', 'memorisation')}
    for budget in sorted(budgets):
        b = str(budget)
        summ['budgets'][b] = {}
        for m in ctx['models']:
            arms = {}
            for i in (1, 2):
                vals = {'heldout_score': [], 'memorisation': [],
                        'n_features': []}
                for f in results['per_fold']:
                    rows = f['budgets'][b]['models'][m]['hybrids']
                    if len(rows) >= i:
                        for kk in vals:
                            vals[kk].append(rows[i - 1].get(kk))
                arms[f'hybrid{i}'] = {kk: _spread(v) for kk, v in vals.items()}
            arms['pca'] = {kk: _spread(
                [f['budgets'][b]['models'][m]['pca'][kk]
                 for f in results['per_fold']])
                for kk in ('heldout_score', 'memorisation', 'n_components')}
            summ['budgets'][b][m] = arms
    results['summary'] = summ

    # Theoretical prediction: Rademacher ∝ sqrt(d/n)
    print(f"\n    Theoretical sqrt(d/n) scaling:")
    for budget in sorted(budgets):
        theoretical = np.sqrt(min(budget, X_concat.shape[1]) / n_samples)
        print(f"      budget={budget:>5}: sqrt(d/n) = {theoretical:.4f}")
    return results


# =========================================================================
# Main
# =========================================================================

def run(data, config, output_dir, dataset_name, sub_analyses=None, **kwargs):
    """C15: ML Theory Depth — 4 sub-analyses, per outer fold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C15: ML THEORY DEPTH")
    print("=" * 70)

    task_type = data.get('task_type')
    if task_type != 'regression':
        # 15c and 15d score regressors and R² on permuted labels. Mirroring
        # c04: gate the whole experiment to regression.
        print(f"  Skipping: c15 is regression-only (task_type={task_type}).")
        return {'skipped': True, 'reason': 'regression_only',
                'task_type': task_type, 'result_version': RESULT_VERSION}

    import time as _time

    for _memo in (_SELECTION_MEMO, _HYBRID_MEMO, _PCA_MEMO, _IMPORTANCE_MEMO):
        _memo.clear()
    ctx = _make_context(data, config, dataset_name)
    # The decided budgets (decided_config sets config['budgets'] from
    # DECISIONS.md); 15c sweeps its own budgets around the row count.
    budgets = sorted(int(b) for b in config['budgets'])
    all_subs = sub_analyses or ['15a', '15b', '15c', '15d']
    results = {'result_version': RESULT_VERSION,
               'selection_scope': 'fold_train_rows',
               'metric': ctx['metric_name'],
               'lower_is_better': ctx['lower_is_better'],
               'models': list(ctx['models']),
               'n_outer_folds': ctx['n_outer'],
               'budgets': budgets}

    if '15a' in all_subs:
        _t0 = _time.time()
        results['distance_concentration'] = _run_distance_concentration(
            ctx, budgets)
        _stage('15a', _t0)

    if '15b' in all_subs:
        _t0 = _time.time()
        results['kernel_analysis'] = _run_kernel_analysis(ctx, budgets)
        _stage('15b', _t0)

    if '15c' in all_subs:
        _t0 = _time.time()
        results['double_descent'] = _run_double_descent(ctx, config)
        _stage('15c', _t0)

    if '15d' in all_subs:
        _t0 = _time.time()
        results['approximation_theory'] = _run_approximation_theory(
            ctx, budgets)
        _stage('15d', _t0)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C15: ML Theory Depth',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--sub', type=str, nargs='+', default=None,
                        choices=['15a', '15b', '15c', '15d'],
                        help='Sub-analyses to run (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    # 2026-10-05: the decided grid for this dataset, whatever --mode set.
    from experiments._shared import decided_config
    config = decided_config(dict(config), args.dataset)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    # 2026-10-05: representations are DATASET_REPS[ds]['general'] only
    # (handoff 2.4 step 9), as complexity_helpers.run_all loads them; --reps
    # is not read.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _rep_names = list(DATASET_REPS[args.dataset]['general'])
    print(f"Reps: {_rep_names} (DATASET_REPS['general']; --reps ignored)")

    data = load_data(args.dataset, rep_preset='all', rep_names=_rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  sub_analyses=args.sub)
    save_experiment_results(results, args.output, 'c15_ml_theory_depth',
                            args.dataset)
