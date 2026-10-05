#!/usr/bin/env python3
"""C19: Importance Weights as Training Signal.

Paper: Complexity
Core question: How should hybrid importance scores enter the training process?
    Tests 5 weighting strategies across models on the HYBRID feature space.

    NOTE (design): an earlier version also evaluated a raw full-concat space to
    ask whether weighting helps more on raw vs hybrid features. The raw-concat
    arm was deliberately removed per design directive — with this many reps the
    full concat is not a meaningful comparison (see the input_spaces block in
    run()). So this experiment answers the narrower question "which weighting
    strategy, if any, helps on the hybrid" — NOT the raw-vs-hybrid contrast.
    The raw-vs-hybrid framing is retained here only as historical context.

Conditions:
  0. unweighted (baseline)
  1. feature_weighted: X * sqrt(importance) — rescale features
  2. sample_weighted: reliability scores as sample_weight in fit
  3. combined: feature_weighted + sample_weighted
  4. importance_pruned: zero out features with importance < median

Usage:
    python c19_importance_weighted_training.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    compute_feature_importance,
)
from experiments._shared import (
    get_rule_hybrids, build_fold_hybrid, cap_dimensions,
    fold_hybrid_scoring_kwargs,
)
# Scores a binary-AUROC dataset on class-1 probabilities instead of hard class
# labels; falls through to model.predict for every other metric.
from kirby.find_best_hybrid import _predict_for_metric


# =========================================================================
# IMPORTANCE WEIGHT TRANSFORMS
# =========================================================================

def _compute_importance_weights(X, y, method='lightgbm_gain'):
    """Compute normalised sqrt-importance weights for features.

    Normalised to MEAN ≈ 1 (not sum=1) so that multiplying StandardScaler
    columns by w differentially rescales high-importance features instead of
    uniformly crushing the whole feature space toward zero.  The sum=1
    normalisation was the same no-op bug we already fixed in n12.
    """
    imp_dict = compute_feature_importance({'_features': X}, y, method=method)
    imp = imp_dict['_features']
    w = np.sqrt(np.maximum(imp, 0))
    w = w / (w.sum() + 1e-10)
    w = w * len(w)  # mean-1 normalisation
    return w


def _compute_reliability_scores(X, y, importance_weights, k=10):
    """Per-sample reliability based on neighbor label agreement in importance-weighted space.

    reliability_i = 1 - |y_i - mean(y_neighbors)| / range(y), clamped to [0.1, 1.0].
    """
    X_weighted = X * importance_weights
    k_actual = min(k, len(y) - 1)
    nn = make_nearest_neighbors(n_neighbors=k_actual + 1)
    nn.fit(X_weighted)
    _, indices = nn.kneighbors(X_weighted)

    y_range = float(np.max(y) - np.min(y))
    if y_range < 1e-10:
        return np.ones(len(y))

    reliability = np.zeros(len(y))
    for i in range(len(y)):
        nbr_idx = indices[i, 1:]  # exclude self
        nbr_mean = np.mean(y[nbr_idx])
        reliability[i] = 1.0 - abs(y[i] - nbr_mean) / y_range

    return np.clip(reliability, 0.1, 1.0)


# =========================================================================
# MODELS THAT SUPPORT SAMPLE WEIGHT
# =========================================================================

WEIGHTED_MODELS = {'rf', 'lightgbm', 'xgboost'}

# 2026-10-05, owner delegated, 2026-10-05: stored beside every tree model's
# result (all three models above are tree models). No model is added.
_TREE_FEATURE_WEIGHT_NOTE = (
    'tree model: multiplying a feature by a positive weight does not change '
    'the order of its values, so it cannot change any split; feature_weighted '
    'differs from unweighted only where a feature with importance 0 gets '
    'weight 0 and is in effect dropped. The combined condition differs from '
    'sample_weighted the same way.')


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C19: Importance-weighted training."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C19: IMPORTANCE-WEIGHTED TRAINING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    n_samples = len(labels)

    model_names = [m for m in config['models'] if m in WEIGHTED_MODELS]
    if not model_names:
        model_names = ['rf']
    models = get_model_factories(model_names, task_type=task_type,
                                 n_samples=n_samples)

    budget = kwargs.get('budget', 100)

    conditions = ['unweighted', 'feature_weighted', 'sample_weighted',
                  'combined', 'importance_pruned']
    # `_compute_reliability_scores` uses `y_range = max - min` and
    # `abs(y_i - mean(nbr_y))` — nominal-on-ordinal arithmetic. Meaningless
    # for multi-class clf (k>=3). Drop the conditions that depend on it.
    is_multiclass = (task_type == 'classification'
                     and len(np.unique(labels)) > 2)
    if is_multiclass:
        dropped = {'sample_weighted', 'combined'}
        conditions = [c for c in conditions if c not in dropped]
        print(f"  Multi-class clf detected: dropping {sorted(dropped)} "
              f"(reliability score not well-defined on nominal labels)")

    # Single input space: hybrid (concat baseline removed per design
    # directive — too many reps for full-concat to be a meaningful
    # comparator).
    #
    # Only the RECIPE is fetched here — allocation method, importance method,
    # internal evaluator, budget — and it comes from a search over every row,
    # which is a separate finding. Its FEATURES are selected inside the fold
    # loop below, from each fold's training labels, the way c18 does it.
    # Building the feature matrix once over every row put each fold's own
    # validation labels into the features it was then scored on, while the
    # scores were reported as held out.
    input_spaces = {}

    augmentations = data.get('augmentations')
    # ONE RECIPE PER DOWNSTREAM MODEL, NOT ONE FOR ALL THREE.
    #
    # ⛔ FIXED 2026-09-18. It asked the pool for downstream_model='rf' once and
    # put that single recipe in input_spaces['hybrid'], and the model loop below
    # then built the hybrid space from it for rf, lightgbm AND xgboost. So two
    # of the three columns in every c19 result were measured on features chosen
    # for a random forest. The stored search scores each downstream model
    # separately and has a recipe for each, so the rf answer was never the one
    # to use for the other two.
    #
    # The blueprint is per model now. `input_spaces['hybrid']` holds a dict from
    # model name to blueprint, and the loop below picks the entry for the column
    # it is building. A model the pool has no recipe for is dropped from the
    # hybrid space rather than given another model's, and is named in the result
    # under 'hybrid_recipe_missing'.
    #
    # This moves the lightgbm and xgboost columns of every c19 result already on
    # disk. Those numbers were measured with rf's features; the new ones are not.
    # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
    # model, per family. Each one is its own input space below, built and
    # scored under every weighting condition. The space named 'hybrid' is the
    # general rank-1 recipe, as before; the others are 'hybrid__general2' and
    # 'hybrid__distance1' (plus '__<family>' for a non-predictive family).
    hybrid_bp_by_model = {}      # general rank 1, kept under its old name
    hybrid_recipe_missing = []
    rule_missing = []
    rule_hybrids = {}            # model -> [slot record, ...]
    for _m in models:
        try:
            _rule = get_rule_hybrids(
                reps, labels, groups, downstream_model=_m, budget=budget,
                augmentations=augmentations, data=data,
                dataset_name=dataset_name,
                task_type=task_type, metric=metric_name,
            )
        except Exception as e:
            hybrid_recipe_missing.append(_m)
            rule_missing.append(f'{_m}: the pool read raised {e}')
            print(f"  Hybrid recipe lookup failed for {_m}: {e}")
            continue
        rule_missing += [f'{_m}: {w}' for w in _rule['missing']]
        rule_hybrids[_m] = []
        for _fam, _part in _rule['by_objective'].items():
            for _e in _part['hybrids']:
                _space = 'hybrid' + ('' if _e['slot'] == 'general1'
                                     else f"__{_e['slot']}")
                if _fam != 'predictive':
                    _space += f'__{_fam}'
                input_spaces.setdefault(_space, {})[_m] = _e['blueprint']
                rule_hybrids[_m].append({
                    'space': _space, 'slot': _e['slot'],
                    'lineage': _e['lineage'], 'tier': _e.get('tier'),
                    'objective': _fam,
                    'aug_condition': _e.get('aug_condition'),
                    'recipe': _e.get('recipe'),
                    'budget': _e.get('budget'),
                    'budget_relaxed': _e.get('budget_relaxed'),
                    'score_in_search': _e.get('score_other_folds'),
                })
        _g1 = input_spaces.get('hybrid', {}).get(_m)
        if _g1 is not None:
            hybrid_bp_by_model[_m] = _g1
        else:
            hybrid_recipe_missing.append(_m)
            print(f"  Hybrid recipe lookup returned nothing for {_m} "
                  f"at this budget")
    if hybrid_recipe_missing:
        print(f"  no general rank-1 hybrid column for: "
              f"{sorted(hybrid_recipe_missing)}")

    print(f"  Input spaces: {list(input_spaces.keys())}")
    print(f"  Models: {list(models.keys())}")
    print(f"  Conditions: {conditions}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        # The budget ASKED FOR at the recipe lookup. The recipe that came back
        # carries its own, and that is the one the fold matrices below are
        # built at, so both are stored.
        'budget': budget,
        # The recipe budget PER DOWNSTREAM MODEL. It was one number, read off
        # the single rf recipe, back when one recipe served all three columns.
        # Each model has its own recipe now and they need not agree, so a
        # reader cannot be handed one number for all three.
        'recipe_budget_by_model': {
            _m: (int(_bp.budget) if getattr(_bp, 'budget', None) is not None
                 else None)
            for _m, _bp in hybrid_bp_by_model.items()
        },
        # Downstream models the pool had no recipe for at this budget. They
        # have no hybrid column; their other spaces still ran.
        'hybrid_recipe_missing': sorted(hybrid_recipe_missing),
        # Owner's set (2026-10-02): every hybrid scored, per model, with the
        # input space it is filed under, and every slot that was not filled.
        'rule_hybrids': rule_hybrids,
        'rule_missing': rule_missing,
        'conditions': conditions,
        # Which rows chose what. Every condition in a model's column shares one
        # feature set per fold, so the deltas between conditions — what c19
        # asks — are unaffected by the recipe having come from every row.
        # Since 2026-09-18 that feature set is also built from the recipe the
        # search chose for THAT model, not from rf's.
        'features_selected_on': "each fold's training rows",
        # Which model decided keep/drop inside each fold's build. It is the
        # same model the column is then scored with, one build per model.
        'features_selected_with': 'the model the column is scored under',
        'recipe_selected_on': 'all rows',
        'per_space': {},
    }

    # No recipe came back, so nothing below runs: per_space stays {}, every
    # count in condition_wins stays 0, and max() over an all-zero dict returns
    # 'unweighted' — a result file that reads as "unweighted won" when nothing
    # was scored at all. Say so instead.
    if not input_spaces:
        results['error'] = 'no_hybrid_recipe'
        results['summary'] = {
            'condition_wins': {c: 0 for c in conditions},
            'most_winning_condition': None,
        }
        print("  No hybrid recipe for this dataset — nothing was scored.")
        return results

    # One set of folds, shared by every condition and every model, so the
    # weighting conditions are compared on identical rows AND on identical
    # features: the fold's hybrid is built once per model and reused across
    # every condition in that model's column.
    _sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, n_folds, _sample_X, labels, mode=mode))

    for space_name, blueprint_or_map in input_spaces.items():
        print(f"\n  === {space_name} ===")
        space_results = {}

        for model_name, model_factory in models.items():
            # The hybrid space carries one recipe PER downstream model; every
            # other space is one blueprint shared by all of them.
            if isinstance(blueprint_or_map, dict):
                if model_name not in blueprint_or_map:
                    continue
                blueprint = blueprint_or_map[model_name]
            else:
                blueprint = blueprint_or_map
            # Build this space's features per fold, from the fold's training
            # rows, with the model this column is scored under. create_hybrid's
            # keep/drop test fits the model it is handed, so a single matrix
            # shared across the whole space chose every column's features with
            # whichever model came first in the list.
            fold_matrices = []
            for train_idx, val_idx in splits:
                tr_reps = {k: v[train_idx] for k, v in reps.items()}
                va_reps = {k: v[val_idx] for k, v in reps.items()}
                tr_a = ({k: v[train_idx] for k, v in augmentations.items()}
                        if augmentations else None)
                va_a = ({k: v[val_idx] for k, v in augmentations.items()}
                        if augmentations else None)
                try:
                    X_tr_f, X_va_f, _fi = build_fold_hybrid(
                        tr_reps, va_reps, labels[train_idx], blueprint,
                        augmentations_train=tr_a, augmentations_val=va_a,
                        groups=(groups[train_idx] if groups is not None else None),
                        # A recipe chosen under a geometry objective fits no
                        # model; this hands back no model for it.
                        **fold_hybrid_scoring_kwargs(
                            blueprint, model=model_factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                    fold_matrices.append((X_tr_f, X_va_f))
                except Exception as e:
                    print(f"  {space_name}/{model_name}: "
                          f"fold hybrid build failed - {e}")
                    fold_matrices.append(None)

            _dims = [m[0].shape[1] for m in fold_matrices if m is not None]
            print(f"\n    {model_name} (dim per fold={_dims}):")
            model_results = {}
            # 2026-10-05: per-fold scores, one entry per fold in split order
            # (nan where the fold's hybrid did not build), and the features
            # each condition was fitted on per fold (None where not built).
            # importance_pruned keeps fewer features than the hybrid built.
            model_fold_scores = {}
            model_fold_widths = {}
            # Folds where the model's fit rejected sample_weight (2026-10-05).
            # Those folds used to be refitted WITHOUT weights and scored under
            # the weighted condition's name; they are now left NaN and listed.
            model_weight_rejected = {}

            for cond in conditions:
                fold_scores = []
                _pos_scores = [float('nan')] * len(splits)
                _pos_widths = [None] * len(splits)

                for _fold_i, (train_idx, val_idx) in enumerate(splits):
                    if fold_matrices[_fold_i] is None:
                        continue
                    X_tr, X_va = fold_matrices[_fold_i]
                    y_train = labels[train_idx]
                    y_val = labels[val_idx]

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    X_tr_s, X_va_s = cap_dimensions(X_tr_s, X_va_s)

                    # Compute importance on training data
                    # Weight features with the importance method the search
                    # picked for this hybrid, not a second one. Ranking the
                    # hybrid's own features by lightgbm_gain when the search
                    # selected them with deeplift measures two methods at once.
                    imp_weights = _compute_importance_weights(
                        X_tr_s, y_train,
                        method=getattr(blueprint, 'importance_method',
                                       'lightgbm_gain'))

                    # Apply condition transforms
                    fit_kwargs = {}

                    if cond == 'feature_weighted':
                        X_tr_s = X_tr_s * imp_weights
                        X_va_s = X_va_s * imp_weights

                    elif cond == 'sample_weighted':
                        reliability = _compute_reliability_scores(
                            X_tr_s, y_train, imp_weights, k=10)
                        fit_kwargs['sample_weight'] = reliability

                    elif cond == 'combined':
                        # Compute reliability on the unweighted (standardized)
                        # space FIRST — otherwise we'd pass already-weighted X
                        # to _compute_reliability_scores which would multiply
                        # by imp_weights a second time (kNN in X*w^2 space).
                        reliability = _compute_reliability_scores(
                            X_tr_s, y_train, imp_weights, k=10)
                        fit_kwargs['sample_weight'] = reliability
                        X_tr_s = X_tr_s * imp_weights
                        X_va_s = X_va_s * imp_weights

                    elif cond == 'importance_pruned':
                        # Strictly above the median, not at or above it. When
                        # more than half the hybrid features score zero
                        # importance the median is zero, '>=' passes every
                        # feature, and the delta reads as "deleting made no
                        # difference" when nothing had been deleted.
                        mask = imp_weights > np.median(imp_weights)
                        if mask.sum() < 2:
                            # Every feature tied at the median. Delete the
                            # bottom half by rank instead, so the condition
                            # still deletes something.
                            order = np.argsort(imp_weights)
                            mask = np.zeros(len(imp_weights), dtype=bool)
                            mask[order[len(order) // 2:]] = True
                        X_tr_s = X_tr_s[:, mask]
                        X_va_s = X_va_s[:, mask]

                    _pos_widths[_fold_i] = int(X_tr_s.shape[1])
                    _n_before = len(fold_scores)
                    try:
                        model = model_factory()
                        model.fit(X_tr_s, y_train, **fit_kwargs)
                        # _predict_for_metric, not .predict: an AUROC dataset
                        # scored on hard class labels is a degraded AUROC.
                        fold_scores.append(metric_fn(
                            y_val, _predict_for_metric(model, X_va_s, metric_fn)))
                    except TypeError:
                        # Model doesn't accept sample_weight kwarg.  Only fall
                        # back for this specific case — broader except would
                        # silently mask real bugs (numerical instability, etc.).
                        if 'sample_weight' in fit_kwargs:
                            model_weight_rejected.setdefault(cond, []).append(
                                int(_fold_i))
                            print(f"      {cond} fold {_fold_i}: the model's "
                                  f"fit takes no sample_weight; left NaN")
                        else:
                            raise
                    if len(fold_scores) > _n_before:
                        _pos_scores[_fold_i] = float(fold_scores[-1])

                score = float(np.mean(fold_scores)) if fold_scores else float('nan')
                model_results[cond] = score
                model_fold_scores[cond] = _pos_scores
                model_fold_widths[cond] = _pos_widths
                print(f"      {cond}: {score:.4f}")

            # Compute deltas vs unweighted
            baseline = model_results.get('unweighted', float('nan'))
            deltas = {}
            deltas_oriented = {}
            for cond in conditions:
                if cond == 'unweighted':
                    continue
                s = model_results.get(cond, float('nan'))
                if not np.isnan(s) and not np.isnan(baseline):
                    # Raw difference in the dataset's own metric, kept because
                    # `scores` beside it is raw too, plus a sign-normalised
                    # sibling so a positive number always means the weighting
                    # helped. Same pairing c21 and n09 already use. Added
                    # 2026-09-10 (metric-direction audit).
                    deltas[cond] = float(s - baseline)
                    deltas_oriented[cond] = float(
                        (baseline - s) if lower_is_better else (s - baseline))

            _sentinel = -1e10 if not lower_is_better else 1e10
            _best_fn = min if lower_is_better else max
            space_results[model_name] = {
                'scores': model_results,
                'fold_scores': model_fold_scores,
                'width_per_fold': model_fold_widths,
                'sample_weight_rejected_folds': model_weight_rejected,
                # Features the hybrid itself built per fold, before any
                # condition (None = build failed).
                'width_built_per_fold': [
                    (int(m[0].shape[1]) if m is not None else None)
                    for m in fold_matrices],
                'deltas': deltas,
                # Positive always means the weighting helped, on rmse as well as
                # on spearman/auc. Added 2026-09-10 (metric-direction audit).
                'deltas_oriented': deltas_oriented,
                'best_condition': _best_fn(model_results, key=lambda k: model_results[k]
                                           if not np.isnan(model_results[k]) else _sentinel),
                # 2026-10-05, owner delegated, 2026-10-05: see
                # _TREE_FEATURE_WEIGHT_NOTE.
                'feature_weighting_note': (_TREE_FEATURE_WEIGHT_NOTE
                                           if model_name in WEIGHTED_MODELS
                                           else None),
            }

        results['per_space'][space_name] = convert_numpy(space_results)

    # Summary: which condition wins most often?
    condition_wins = {c: 0 for c in conditions}
    for space_name, space_res in results['per_space'].items():
        for model_name, model_res in space_res.items():
            best = model_res.get('best_condition', 'unweighted')
            condition_wins[best] = condition_wins.get(best, 0) + 1

    results['summary'] = {
        'condition_wins': condition_wins,
        # CXA-16/OTHER-14: renamed from 'best_overall' — that name collides with
        # the removed internal-evaluator concept and was easy to misread. This is
        # simply the weighting condition that won the most (model x space) cells.
        # None when no cell was scored — an all-zero count dict otherwise makes
        # max() return whichever condition is listed first.
        'most_winning_condition': (max(condition_wins, key=condition_wins.get)
                                   if any(condition_wins.values()) else None),
    }
    print(f"\n  Condition wins: {condition_wins}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C19: Importance-Weighted Training')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'c19_importance_weighted_training', args.dataset)
