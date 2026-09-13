#!/usr/bin/env python3
"""N02: Importance-weighted sample weighting for noise robustness.

Paper: Noise
Hypothesis: Feature importance scores define an importance-weighted distance
    space where per-sample reliability can be measured. Samples whose labels
    are consistent with their importance-weighted neighbors are "reliable";
    inconsistent samples are "unreliable" (noisy). Training with reliability-
    based sample weights should improve predictions.

Key question: Do importance-weighted sample weights help MORE with base reps
    or hybrids?
    - If base reps benefit more  -> hybrids already handle noise implicitly
    - If both benefit equally    -> effects are complementary

5 conditions:
    1. base_unweighted    — individual rep, no sample weights (baseline)
    2. base_weighted      — individual rep, importance-weighted sample weights
    3. hybrid_unweighted  — hybrid rep, no sample weights
    4. hybrid_weighted    — hybrid rep, importance-weighted sample weights
    5. hybrid_cliff_aware — hybrid rep, cliff-aware weights
                            (ActivityCliffAwareEstimator)

Conditions 3, 4 and 5 share one hybrid build per model per fold, and every
condition is dropped from a fold on which any of its fits failed, so the
means they are compared through cover the same folds.

2026-09-12, owner's decision: the scorer that defines the importance-weighted
distance is swept over lightgbm_gain and deeplift. It scales the axes of the
distance the neighbour search runs in; it selects no features and it does not
touch the hybrid's recipe. Each scorer's own weighted score is under
results['per_rep'][rep][model]['by_importance'][method] and
results['hybrid'][model]['by_importance'][method]; the flat weighted_mean and
delta keys hold the first scorer in the sweep.

2026-09-12, owner's decision: the subset of the precomputed pool this file may
pick its hybrid from includes similarity hybrids — searches run under
objective='similarity', whose features were chosen to make the distance
geometry predict each molecule's activity from its five nearest neighbours —
not only prediction-built ones. n02 does not pass distance=True and that is
unchanged; the pool subset is a separate thing. Every hybrid condition records
which objective its recipe came from, and results['pool_selection'] says which
numbers in the result file are in the dataset's own metric and which are read
straight out of the search.

Usage:
    python n02_sample_weighting.py --dataset logd --reps fast --mode intermediary
    python n02_sample_weighting.py --dataset caco2 --reps all --mode full
    python n02_sample_weighting.py --dataset flab_thermo_tresanco --reps fast --mode full
    python n02_sample_weighting.py --dataset logd --k-neighbors 15
    python n02_sample_weighting.py --dataset logd --importance-methods lightgbm_gain,deeplift
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

# Torch before anything that loads LightGBM. hybrid.py:3410 records the
# hazard: torch and LightGBM each load an OpenMP runtime and whichever loads
# second segfaults the process, and KIRBY_LGBM_THREADS=1 does not prevent it.
# n02 now scores with deeplift, which is a torch method, so this file reaches
# both. Measured here 2026-09-12 on a 160-row synthetic set: lightgbm_gain
# then deeplift exited 139 with no traceback; deeplift then lightgbm_gain ran
# to the end and gave the same per-scorer numbers. Importing torch up here
# lets the sweep keep lightgbm_gain first, which is what the flat
# weighted_mean and delta keys report. Guarded: a machine without torch runs
# every scorer that is not deeplift.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    compute_feature_importance,
    get_precomputed_search, get_best_search_for_downstream, build_fold_hybrid,
    extract_top3_base_reps,
)

from kirby.utils.noise_estimation import ActivityCliffAwareEstimator


# Models that support sample_weight in .fit().
#
# svm joined 2026-09-12, owner's decision. The comment above used to be the whole
# reason it was absent, and it does not cover svm: scikit-learn's SVR and SVC both
# accept sample_weight. Four of the roster's five now run here.
#
# dnn stays out, and now for a measured reason rather than an assumed one:
# DNNRegressor.fit() raises TypeError: unexpected keyword argument 'sample_weight'.
# Seen in a smoke run 2026-09-12. If the network ever gains a per-molecule weight,
# add it here and this file needs no other change.
WEIGHTED_MODELS = ['rf', 'xgboost', 'lightgbm', 'svm']

# Scorers swept to define the importance-weighted distance. treeshap is
# deliberately absent: the owner excluded it, and it cannot run locally at all.
DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain', 'deeplift']


def compute_cliff_aware_weights(X, y, k=10, strength=1.0, is_clf=False):
    """Use ActivityCliffAwareEstimator to compute sample weights.

    Unlike compute_reliability_scores, this distinguishes cliffs from noise:
    cliff compounds get higher weights (they're not noise), while genuinely
    noisy samples get lower weights.

    is_clf must be threaded through: the estimator's default (is_clf=False)
    runs the REGRESSION noise-scoring path (neighbor-mean deviation / local
    variance) on raw class labels, which is meaningless for classification.
    With is_clf=True it uses the class-disagreement x neighbor-purity path.
    """
    # metric=None reads --distance-metric. It was 'cosine' here, 'cosine' in
    # n07 and 'euclidean' in n12, for the same estimator class.
    estimator = ActivityCliffAwareEstimator(k=k, metric=None, is_clf=is_clf)
    estimator.fit(X, y)
    return estimator.get_weights(strength=strength)


def compute_reliability_scores(X, y, importance_scores, k=10, is_clf=False):
    """Compute per-sample reliability as neighbor-agreement in importance-weighted space.

    For regression: reliability_i = 1 - |y_i - mean(y_neighbors)| / range(y)
    For classification: reliability_i = fraction of k-NN with same class label.

    Clamped to [0.1, 1.0] so no sample is fully zero-weighted.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        y: Labels (n_samples,).
        importance_scores: Per-feature importance (n_features,).
        k: Number of neighbors.
        is_clf: If True, use class-agreement instead of label proximity.

    Returns:
        reliability: Array (n_samples,) in [0.1, 1.0].
    """
    # Weight features by sqrt(importance)
    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)
    X_weighted = X * weights

    # Find k nearest neighbors in importance-weighted space
    k_actual = min(k, len(y) - 1)
    nn = make_nearest_neighbors(n_neighbors=k_actual + 1)  # +1 because query includes self
    nn.fit(X_weighted)
    _, indices = nn.kneighbors(X_weighted)
    # Exclude self (first neighbor)
    neighbor_idx = indices[:, 1:]

    if is_clf:
        # Classification: fraction of neighbors with same class
        same_class = (y[neighbor_idx] == y[:, None]).astype(float)
        reliability = np.mean(same_class, axis=1)
    else:
        # Regression: neighbor mean agreement
        neighbor_mean = np.mean(y[neighbor_idx], axis=1)
        y_range = y.max() - y.min()
        if y_range < 1e-10:
            return np.ones(len(y))
        deviation = np.abs(y - neighbor_mean) / y_range
        reliability = 1.0 - deviation

    # Clamp to [0.1, 1.0]
    reliability = np.clip(reliability, 0.1, 1.0)

    return reliability


def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance-weighted sample weighting across 5 conditions
    (base_unweighted, base_weighted, hybrid_unweighted, hybrid_weighted,
    hybrid_cliff_aware)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N02: IMPORTANCE-WEIGHTED SAMPLE WEIGHTING")
    print("=" * 70)

    k_neighbors = kwargs.get('k_neighbors', 10)
    # 2026-09-12, owner's decision: sweep the scorer that defines the weighted
    # distance over lightgbm_gain and deeplift, not treeshap. The scorer only
    # scales the axes of the distance the k-nearest-neighbour search runs in —
    # it selects no features and it does not touch the hybrid's recipe.
    # lightgbm_gain stays first, so an unflagged run's flat weighted_mean /
    # delta keys hold the number they held before the sweep existed.
    importance_methods = kwargs.get('importance_methods') or DEFAULT_IMPORTANCE_METHODS
    importance_methods = list(importance_methods)
    primary_importance = importance_methods[0]

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    domain = data.get('domain', 'molecular')

    # AUROC ('auc') expects continuous scores; passing hard 0/1 class labels
    # produces a degenerate score equivalent to balanced accuracy under
    # thresholding 0.5, not the true ranking-based AUROC. Use predict_proba
    # for binary AUROC; for everything else fall back to predict (class
    # labels for F1/accuracy/MCC, raw floats for regression metrics).
    def _predict_for_metric(model, X):
        if metric_name == 'auc' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return model.predict(X)

    # Only use models that support sample_weight
    model_factories = get_model_factories(
        WEIGHTED_MODELS, task_type=task_type, n_samples=n_samples)

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    # (Dead `augmentation_strategy = 'uncapped'` removed — was never read.
    # Per-model aug_condition is fetched from the pool below.)
    sample_X = next(iter(reps.values()))

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Models: {list(model_factories.keys())}")
    print(f"  Reps: {len(reps)} ({', '.join(sorted(reps.keys()))})")
    print(f"  k-neighbors: {k_neighbors}, folds: {n_folds}")
    print(f"  Importance scorers swept: {', '.join(importance_methods)} "
          f"(flat keys = {primary_importance})\n", flush=True)

    results = {
        'per_rep': {},      # conditions 1-2: per individual rep
        'hybrid': {},       # conditions 3-4: hybrid
        'summary': {},      # conditions 3-5 share one loop; 5 lands in
                            # results['hybrid_cliff_aware'] below

    }

    # =========================================================================
    # CONDITIONS 1-2: Individual reps (base_unweighted, base_weighted)
    # =========================================================================
    print("  --- Conditions 1-2: Individual reps ---")

    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        print(f"\n  Rep: {rep_name} (dim={X_rep.shape[1]})")

        rep_results = {}
        for model_name, model_factory in model_factories.items():
            # One unweighted list, and one weighted list per importance
            # scorer. A fold only enters any of them if every fit on it
            # succeeded, so all the means below cover the same folds and a
            # delta is a difference over one fold set.
            unweighted_scores = []
            weighted_scores = {m: [] for m in importance_methods}
            reliability_stats = {m: [] for m in importance_methods}

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                labels, mode=mode):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_rep[train_idx])
                X_val = scaler.transform(X_rep[val_idx])

                # Condition 1: unweighted
                try:
                    model_uw = model_factory()
                    model_uw.fit(X_train, y_train)
                    fold_uw = metric_fn(
                        y_val, _predict_for_metric(model_uw, X_val))
                except Exception:
                    continue

                # Condition 2: importance-weighted sample weights, once per
                # scorer in the sweep.
                fold_w = {}
                fold_stats = {}
                for imp_method in importance_methods:
                    try:
                        # Compute feature importance on the SAME (scaled)
                        # matrix the reliability scorer will consume.
                        # Previously importance was computed on raw
                        # X_rep[train_idx] while reliability ran on scaled
                        # X_train — fine for tree-based importance
                        # (scale-invariant) but breaks the moment `method`
                        # switches to a scale-sensitive one (permutation,
                        # deeplift on continuous features).
                        imp = compute_feature_importance(
                            {rep_name: X_train}, y_train,
                            method=imp_method)
                        imp_scores = imp[rep_name]

                        # Compute per-sample reliability in importance-weighted space
                        sample_weights = compute_reliability_scores(
                            X_train, y_train, imp_scores, k=k_neighbors,
                            is_clf=is_clf)

                        fold_stats[imp_method] = {
                            'mean': float(np.mean(sample_weights)),
                            'std': float(np.std(sample_weights)),
                            'min': float(np.min(sample_weights)),
                            'pct_below_05': float(
                                np.mean(sample_weights < 0.5) * 100),
                        }

                        model_w = model_factory()
                        model_w.fit(X_train, y_train,
                                    sample_weight=sample_weights)
                        fold_w[imp_method] = metric_fn(
                            y_val, _predict_for_metric(model_w, X_val))
                    except Exception:
                        break

                # Any failure drops the fold from every list, weighted and
                # unweighted alike, so no mean is taken over a fold the
                # others do not have.
                if len(fold_w) != len(importance_methods):
                    continue

                unweighted_scores.append(fold_uw)
                for imp_method in importance_methods:
                    weighted_scores[imp_method].append(fold_w[imp_method])
                    reliability_stats[imp_method].append(
                        fold_stats[imp_method])

            if not unweighted_scores:
                continue

            uw_mean = float(np.mean(unweighted_scores))
            uw_std = float(np.std(unweighted_scores))

            by_importance = {}
            for imp_method in importance_methods:
                m_scores = weighted_scores[imp_method]
                m_mean = float(np.mean(m_scores))
                by_importance[imp_method] = {
                    'weighted_mean': m_mean,
                    'weighted_std': float(np.std(m_scores)),
                    'delta': m_mean - uw_mean,
                    'reliability_stats': reliability_stats[imp_method],
                }

            # The flat keys carry the first scorer in the sweep, so a reader
            # written before the sweep still finds the same field with the
            # same meaning. Every scorer's own numbers sit in by_importance.
            primary = by_importance[primary_importance]
            w_mean = primary['weighted_mean']
            delta = primary['delta']

            rep_results[model_name] = {
                'unweighted_mean': uw_mean,
                'unweighted_std': uw_std,
                'weighted_mean': w_mean,
                'weighted_std': primary['weighted_std'],
                'delta': delta,
                'reliability_stats': primary['reliability_stats'],
                'n_folds_scored': len(unweighted_scores),
                'importance_methods': list(importance_methods),
                'primary_importance_method': primary_importance,
                'by_importance': by_importance,
            }

            sign = '+' if delta > 0 else ''
            print(f"    {model_name}: unweighted={uw_mean:.4f}, "
                  f"weighted={w_mean:.4f} (delta={sign}{delta:.4f}) "
                  f"[{primary_importance}]")
            for imp_method in importance_methods[1:]:
                _b = by_importance[imp_method]
                print(f"      {'':<{len(model_name)}}  weighted="
                      f"{_b['weighted_mean']:.4f} "
                      f"(delta={_b['delta']:+.4f}) [{imp_method}]")

        results['per_rep'][rep_name] = rep_results

    # =========================================================================
    # CONDITIONS 3-5: Get hybrid blueprint ONCE (shared across all fold loops)
    # =========================================================================
    # 2026-09-12, owner's decision: n02's pool subset includes similarity
    # hybrids. The pool may now pick a hybrid whose features were chosen under
    # objective='similarity' — chosen to make the distance geometry predict
    # each molecule's activity from its five nearest neighbours — not only one
    # chosen to predict the label with a downstream model. This is NOT
    # distance=True; n02 does not pass distance=True and that is unchanged.
    # Every result below records which objective the winning hybrid came from.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset='predictive_and_similarity',
    )
    # 2026-09-12: which family this top-level lookup landed in, and in what
    # units its stored numbers are. Under 'similarity' the search's stored
    # score is a Spearman correlation between each held-out molecule's activity
    # and what its five nearest neighbours predict; under 'predictive' it is
    # the dataset's own metric. The two are not comparable and the pool does
    # not rank them against each other.
    pool_family = meta.get('objective') if meta else None
    pool_metric_name = meta.get('metric_name') if meta else None
    pool_family_choice = meta.get('objective_choice') if meta else None
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            # 2026-09-12: the label on this line used to say "by RF"
            # unconditionally. Under objective='similarity' find_best_hybrid
            # stores the SAME model-independent geometry number under every
            # model name, so "by RF" would name a model that never ran. Say
            # which family and which metric these numbers are in.
            _lbl = (f"by RF, {pool_metric_name}" if pool_family == 'predictive'
                    else f"family={pool_family}, {pool_metric_name}")
            print(f"  Top-3 base reps ({_lbl}): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping hybrid conditions")
        return {'error': 'no_blueprint', 'metric': metric_name}

    # Matched evaluator: fetch a per-model blueprint instead of reusing the
    # RF one for xgboost/lightgbm/ridge. Per project_model_matching the
    # downstream model used to evaluate a hybrid must match the model the
    # search optimised against. Each call is cached, so the extra fetches
    # cost ~nothing when the pool already has the entry. Also align each
    # per-model bp's aug_strategy with its own meta['aug_condition'] to
    # avoid the 2026-04-21 cache-collision pattern.
    bp_by_model = {}
    aug_by_model = {}
    # 2026-09-12, owner's decision: n02's pool subset includes similarity
    # hybrids, so each model's hybrid can come from either family. Record the
    # family per model — two models in the same run can land in different
    # families, and a reader cannot tell them apart from the score alone.
    family_by_model = {}
    for _mname in model_factories:
        try:
            _search, _meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset='predictive_and_similarity',
            )
            if _meta is not None:
                _bp = _meta['best_config']['blueprint']
                _aug_cond = _meta.get('aug_condition', 'uncapped')
                if getattr(_bp, 'augmentation_strategy', None) != _aug_cond:
                    try:
                        _bp.augmentation_strategy = _aug_cond
                    except Exception:
                        pass
                bp_by_model[_mname] = _bp
                aug_by_model[_mname] = _aug_cond
                # Every eligible family's own best sits beside the winner,
                # because the pool refuses to rank the two against each other
                # when their metric names differ.
                family_by_model[_mname] = {
                    'objective': _meta.get('objective'),
                    'objective_choice': _meta.get('objective_choice'),
                    'pool_subset': _meta.get('pool_subset'),
                    'search_metric_name': _meta.get('metric_name'),
                    'search_lower_is_better': _meta.get('lower_is_better'),
                    'search_score_mean': _meta.get('score_mean'),
                    'tier': _meta.get('tier'),
                    'aug_condition': _aug_cond,
                    'by_objective': {
                        _o: {'tier': _d.get('tier'),
                             'aug_condition': _d.get('aug_condition'),
                             'score_mean': _d.get('score_mean'),
                             'metric_name': _d.get('metric_name'),
                             'lower_is_better': _d.get('lower_is_better')}
                        for _o, _d in (_meta.get('by_objective') or {}).items()
                    },
                    'blueprint_source': 'own_model',
                }
                print(f"    {_mname}: hybrid from family "
                      f"{_meta.get('objective')!r} "
                      f"({_meta.get('objective_choice')}), search metric "
                      f"{_meta.get('metric_name')!r}")
            else:
                # Fall back to the RF blueprint so the model still gets a
                # hybrid (better than dropping it from the comparison).
                bp_by_model[_mname] = bp
                aug_by_model[_mname] = meta.get('aug_condition', 'uncapped')
                # 2026-09-12: the fallback inherits RF's family, not its own.
                family_by_model[_mname] = {
                    'objective': pool_family,
                    'objective_choice': pool_family_choice,
                    'pool_subset': meta.get('pool_subset'),
                    'search_metric_name': pool_metric_name,
                    'search_lower_is_better': meta.get('lower_is_better'),
                    'search_score_mean': meta.get('score_mean'),
                    'tier': meta.get('tier'),
                    'aug_condition': meta.get('aug_condition', 'uncapped'),
                    'by_objective': {
                        _o: {'tier': _d.get('tier'),
                             'aug_condition': _d.get('aug_condition'),
                             'score_mean': _d.get('score_mean'),
                             'metric_name': _d.get('metric_name'),
                             'lower_is_better': _d.get('lower_is_better')}
                        for _o, _d in (meta.get('by_objective') or {}).items()
                    },
                    'blueprint_source': 'rf_fallback_no_pool_entry',
                }
        except Exception as _exc:
            print(f"  WARNING: blueprint fetch failed for {_mname} "
                  f"({_exc}); using RF blueprint as fallback.")
            bp_by_model[_mname] = bp
            aug_by_model[_mname] = meta.get('aug_condition', 'uncapped')
            family_by_model[_mname] = {
                'objective': pool_family,
                'objective_choice': pool_family_choice,
                'pool_subset': meta.get('pool_subset'),
                'search_metric_name': pool_metric_name,
                'search_lower_is_better': meta.get('lower_is_better'),
                'search_score_mean': meta.get('score_mean'),
                'tier': meta.get('tier'),
                'aug_condition': meta.get('aug_condition', 'uncapped'),
                'by_objective': {},
                'blueprint_source': f'rf_fallback_fetch_failed: {_exc}',
            }

    # =========================================================================
    # CONDITIONS 3-5: Hybrid (hybrid_unweighted, hybrid_weighted,
    #                         hybrid_cliff_aware)
    # =========================================================================
    # 2026-09-12, owner's decision: condition 5 runs inside this loop instead
    # of its own. It used the same blueprint, the same augmentations and the
    # same fold indices, so its separate loop rebuilt the hybrid a second time
    # and threw the first copy away — 3 models x 5 folds = 15 rebuilds per
    # dataset. All three conditions now score the matrix this fold built once.
    print(f"\n  --- Conditions 3-5: Hybrid ---")

    hybrid_results = {}
    cliff_aware_results = {}
    for model_name, model_factory in model_factories.items():
        model_bp = bp_by_model[model_name]
        model_aug_cond = aug_by_model[model_name]
        model_augs = augmentations if model_aug_cond != 'none' else None
        # As in the base-rep block: a fold enters every list or none of them,
        # so unweighted, weighted and cliff-aware means cover one fold set.
        unweighted_scores = []
        weighted_scores = {m: [] for m in importance_methods}
        reliability_stats = {m: [] for m in importance_methods}
        ca_scores = []
        n_features_list = []

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Create hybrid per fold from THIS MODEL'S blueprint
            try:
                train_augs = {k: v[train_idx] for k, v in model_augs.items()} if model_augs else None
                val_augs = {k: v[val_idx] for k, v in model_augs.items()} if model_augs else None
                X_train_h, X_val_h, feature_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, model_bp,
                    augmentations_train=train_augs, augmentations_val=val_augs,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=model_factory(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
            except Exception as e:
                print(f"    {model_name}: hybrid creation failed - {e}")
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_h)
            X_val_s = scaler.transform(X_val_h)

            # Condition 3: hybrid unweighted
            try:
                model_uw = model_factory()
                model_uw.fit(X_train_s, y_train)
                fold_uw = metric_fn(
                    y_val, _predict_for_metric(model_uw, X_val_s))
            except Exception:
                continue

            # Condition 4: hybrid + importance-weighted sample weights, once
            # per scorer in the sweep.
            fold_w = {}
            fold_stats = {}
            for imp_method in importance_methods:
                try:
                    # Compute importance on the SAME (scaled) hybrid matrix
                    # the reliability scorer consumes — same alignment fix as
                    # the base-rep block above. The raw X_train_h was
                    # previously passed to compute_feature_importance while
                    # X_train_s (the scaled version) was passed to
                    # compute_reliability_scores.
                    imp = compute_feature_importance(
                        {'hybrid': X_train_s}, y_train, method=imp_method)
                    imp_scores = imp['hybrid']

                    sample_weights = compute_reliability_scores(
                        X_train_s, y_train, imp_scores, k=k_neighbors,
                        is_clf=is_clf)

                    fold_stats[imp_method] = {
                        'mean': float(np.mean(sample_weights)),
                        'std': float(np.std(sample_weights)),
                        'min': float(np.min(sample_weights)),
                        'pct_below_05': float(
                            np.mean(sample_weights < 0.5) * 100),
                    }

                    model_w = model_factory()
                    model_w.fit(X_train_s, y_train,
                                sample_weight=sample_weights)
                    fold_w[imp_method] = metric_fn(
                        y_val, _predict_for_metric(model_w, X_val_s))
                except Exception:
                    break

            if len(fold_w) != len(importance_methods):
                continue

            # Condition 5: same hybrid matrix, weights from
            # ActivityCliffAwareEstimator instead of neighbour agreement.
            try:
                ca_weights = compute_cliff_aware_weights(
                    X_train_s, y_train, k=k_neighbors, is_clf=is_clf)
                model_ca = model_factory()
                model_ca.fit(X_train_s, y_train, sample_weight=ca_weights)
                fold_ca = metric_fn(
                    y_val, _predict_for_metric(model_ca, X_val_s))
            except Exception:
                continue

            unweighted_scores.append(fold_uw)
            for imp_method in importance_methods:
                weighted_scores[imp_method].append(fold_w[imp_method])
                reliability_stats[imp_method].append(fold_stats[imp_method])
            ca_scores.append(fold_ca)
            n_features_list.append(X_train_h.shape[1])

        if not unweighted_scores:
            continue

        uw_mean = float(np.mean(unweighted_scores))
        uw_std = float(np.std(unweighted_scores))

        by_importance = {}
        for imp_method in importance_methods:
            m_scores = weighted_scores[imp_method]
            m_mean = float(np.mean(m_scores))
            by_importance[imp_method] = {
                'weighted_mean': m_mean,
                'weighted_std': float(np.std(m_scores)),
                'delta': m_mean - uw_mean,
                'reliability_stats': reliability_stats[imp_method],
            }

        primary = by_importance[primary_importance]
        w_mean = primary['weighted_mean']
        w_std = primary['weighted_std']
        delta = primary['delta']
        n_features = int(np.mean(n_features_list)) if n_features_list else 0

        hybrid_results[model_name] = {
            'unweighted_mean': uw_mean,
            'unweighted_std': uw_std,
            'weighted_mean': w_mean,
            'weighted_std': w_std,
            'delta': delta,
            'n_features': n_features,
            'reliability_stats': primary['reliability_stats'],
            'n_folds_scored': len(unweighted_scores),
            'importance_methods': list(importance_methods),
            'primary_importance_method': primary_importance,
            'by_importance': by_importance,
            # 2026-09-12, owner's decision (pool subset includes similarity
            # hybrids): which family chose this hybrid's recipe. The two
            # scores above are measured by n02 itself with metric_fn, so they
            # are in the dataset's own metric whichever family chose the
            # recipe; this key says what chose it.
            'hybrid_objective': family_by_model.get(model_name, {}).get(
                'objective'),
            'hybrid_objective_choice': family_by_model.get(model_name, {}).get(
                'objective_choice'),
            'pool_subset': family_by_model.get(model_name, {}).get(
                'pool_subset'),
        }

        # Condition 5, assembled from the lists the merged loop produced. Its
        # folds are the folds behind uw_mean, so this subtraction is over one
        # fold set.
        ca_mean = float(np.mean(ca_scores))
        cliff_aware_results[model_name] = {
            'score_mean': ca_mean,
            'score_std': float(np.std(ca_scores)),
            'delta_vs_unweighted': ca_mean - uw_mean,
            'n_folds_scored': len(ca_scores),
            # 2026-09-12: same hybrid recipe as conditions 3-4, so the
            # same family chose it. Recorded here too so this block reads
            # on its own.
            'hybrid_objective': family_by_model.get(model_name, {}).get(
                'objective'),
        }

        sign = '+' if delta > 0 else ''
        _fam = family_by_model.get(model_name, {}).get('objective')
        print(f"    {model_name}: unweighted={uw_mean:.4f}, "
              f"weighted={w_mean:.4f} (delta={sign}{delta:.4f}, "
              f"dim={n_features}, family={_fam}) [{primary_importance}]")
        for imp_method in importance_methods[1:]:
            _b = by_importance[imp_method]
            print(f"      {'':<{len(model_name)}}  weighted="
                  f"{_b['weighted_mean']:.4f} "
                  f"(delta={_b['delta']:+.4f}) [{imp_method}]")
        print(f"      {'':<{len(model_name)}}  cliff_aware={ca_mean:.4f} "
              f"(delta={cliff_aware_results[model_name]['delta_vs_unweighted']:+.4f})")

    results['hybrid'] = hybrid_results
    results['hybrid_cliff_aware'] = cliff_aware_results

    # =========================================================================
    # SUMMARY: Compare weighting benefit for base reps vs hybrid
    # =========================================================================
    print(f"\n  --- Summary ---")

    hybrid_deltas = []
    for model_name, model_res in results['hybrid'].items():
        d = model_res.get('delta', float('nan'))
        if not np.isnan(d):
            hybrid_deltas.append(d)
    n_hybrid_improved = int(sum(1 for d in hybrid_deltas if (d < 0 if lower_is_better else d > 0)))
    print(f"  Hybrid weighting: {n_hybrid_improved}/{len(hybrid_deltas)} models improved")

    # Per-model comparison: top-3 base reps vs hybrid
    print(f"\n  PER-MODEL COMPARISON (top-3 base reps vs hybrid):")
    results['per_model_top3'] = {}
    for model_name in model_factories:
        # Collect (rep_name, unweighted_score) for this model
        rep_scores = []
        for rep_name, rep_results in results['per_rep'].items():
            if model_name in rep_results:
                uw = rep_results[model_name].get('unweighted_mean', float('nan'))
                w = rep_results[model_name].get('weighted_mean', float('nan'))
                d = rep_results[model_name].get('delta', float('nan'))
                if not np.isnan(uw):
                    rep_scores.append((rep_name, uw, w, d))

        # Sort by unweighted score (best first)
        rep_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
        top3 = rep_scores[:3]

        model_hybrid_uw = float('nan')
        model_hybrid_w = float('nan')
        model_hybrid_delta = float('nan')
        if model_name in results['hybrid']:
            model_hybrid_uw = results['hybrid'][model_name].get('unweighted_mean', float('nan'))
            model_hybrid_w = results['hybrid'][model_name].get('weighted_mean', float('nan'))
            model_hybrid_delta = results['hybrid'][model_name].get('delta', float('nan'))

        results['per_model_top3'][model_name] = {
            'top3': [(n, uw, w, d) for n, uw, w, d in top3],
            'hybrid_unweighted': model_hybrid_uw,
            'hybrid_weighted': model_hybrid_w,
            'hybrid_delta': model_hybrid_delta,
        }

        print(f"\n  {model_name}:")
        for rn, uw, w, d in top3:
            print(f"    {rn:<20} uw={uw:.4f}, w={w:.4f} (delta={d:+.4f})")
        print(f"    {'HYBRID':<20} uw={model_hybrid_uw:.4f}, w={model_hybrid_w:.4f} "
              f"(delta={model_hybrid_delta:+.4f})")

    # `top3` inside the loop above held whichever model iterated last (ridge,
    # since model_factories is insertion-ordered ['rf','xgboost','lightgbm',
    # 'ridge']). Save the canonical RF-baseline ranking from the search pool
    # — it's what was already printed at the top of the run and is consistent
    # with the rest of the suite. Per-model rankings remain in
    # results['per_model_top3'].
    search_top3 = []
    if search is not None:
        try:
            search_top3 = [(n, float(s)) for n, s in extract_top3_base_reps(search)]
        except Exception:
            search_top3 = []
    results['top3_base_reps'] = search_top3

    # =========================================================================
    # 2026-09-12, owner's decision: n02's pool subset includes similarity
    # hybrids. Record which objective each hybrid was built under, and which
    # numbers in this file change units when that objective is 'similarity'.
    # =========================================================================
    # What is and is not in the dataset's own metric, read off this file:
    #
    #   Conditions 1-2 (per_rep) and conditions 3-5 (hybrid,
    #   hybrid_cliff_aware) are scored by n02 itself with metric_fn, on the
    #   held-out rows of its own folds. Every one of those numbers is in the
    #   dataset's own metric whichever family chose the hybrid's recipe, so
    #   'delta' (weighted minus unweighted) has the same feature space and the
    #   same metric top and bottom, and so does per_model_top3 (top-3 base
    #   representations beside the hybrid).
    #
    #   'top3_base_reps' is the one field that does not: it is read straight
    #   out of the stored search. Under objective='predictive' those are RF
    #   scores in the dataset's own metric. Under objective='similarity'
    #   find_best_hybrid stores a model-independent number under every model
    #   name — the Spearman correlation between each held-out molecule's
    #   activity and what its five nearest neighbours predict — so the same
    #   field then holds a different quantity, ranked in a different space.
    #   'top3_base_reps_metric' says which, and when the winner is not the
    #   predictive family the predictive family's own top-3 is stored beside
    #   it so both numbers are on the page rather than one silently replacing
    #   the other.
    results['pool_selection'] = {
        'pool_subset': (meta.get('pool_subset') if meta else None),
        'top_level_objective': pool_family,
        'top_level_objective_choice': pool_family_choice,
        'top_level_search_metric': pool_metric_name,
        'by_model': family_by_model,
        'dataset_metric': metric_name,
        'scores_in_dataset_metric': [
            'per_rep', 'hybrid', 'hybrid_cliff_aware', 'per_model_top3'],
        'scores_read_from_the_search': ['top3_base_reps'],
    }
    results['top3_base_reps_metric'] = pool_metric_name
    results['top3_base_reps_objective'] = pool_family

    # The predictive family's top-3, in the dataset's own metric, whenever the
    # widened subset handed back a different family. A second pool lookup, not
    # a second search: 'predictive_only' is the subset every other experiment
    # already reads.
    results['top3_base_reps_predictive'] = None
    if pool_family is not None and pool_family != 'predictive':
        try:
            _p_search, _p_meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset='predictive_only',
            )
            if _p_search is not None:
                results['top3_base_reps_predictive'] = {
                    'metric_name': (_p_meta or {}).get('metric_name'),
                    'objective': (_p_meta or {}).get('objective'),
                    'top3': [(n, float(s))
                             for n, s in extract_top3_base_reps(_p_search)],
                }
        except Exception as _exc:
            results['top3_base_reps_predictive'] = {'error': str(_exc)}
        print(f"  NOTE: top3_base_reps came from family {pool_family!r}, "
              f"metric {pool_metric_name!r} — not the dataset's metric "
              f"({metric_name}). The predictive family's own top-3 is in "
              f"results['top3_base_reps_predictive'].")

    return make_structured_output(
        experiment_id='n02',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(model_factories.keys()),
            'k_neighbors': k_neighbors,
            # The scorers swept to define the weighted distance, in order.
            # The first one is what the flat weighted_mean / weighted_std /
            # delta / reliability_stats keys hold.
            'importance_methods': list(importance_methods),
            'primary_importance_method': primary_importance,
            'conditions': [
                'base_unweighted', 'base_weighted',
                'hybrid_unweighted', 'hybrid_weighted',
                'hybrid_cliff_aware',
            ],
            # 2026-09-12, owner's decision: n02's pool subset includes
            # similarity hybrids. Recorded in the config so a reader can tell
            # a run under the widened subset from an older one.
            'pool_subset': 'predictive_and_similarity',
            'hybrid_objective_by_model': {
                _m: _d.get('objective') for _m, _d in family_by_model.items()
            },
        },
        results=results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser(
        'N02: Importance-Weighted Sample Weighting',
        has_delta=True, has_augment=True)
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='Number of neighbors for reliability scoring (default: 10)')
    parser.add_argument('--importance-methods', type=str,
                        default=','.join(DEFAULT_IMPORTANCE_METHODS),
                        help='comma-separated scorers defining the weighted '
                             'distance; the first is what the flat '
                             'weighted_mean/delta keys hold '
                             f'(default: {",".join(DEFAULT_IMPORTANCE_METHODS)})')
    args = parser.parse_args()
    config = get_config(args)

    importance_methods = [m.strip() for m in args.importance_methods.split(',')
                          if m.strip()]

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {WEIGHTED_MODELS}")
    print(f"k-neighbors: {args.k_neighbors}")
    print(f"Importance scorers: {importance_methods}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_neighbors=args.k_neighbors,
                  importance_methods=importance_methods)
    save_experiment_results(results, args.output, 'n02_sample_weighting',
                            args.dataset)
