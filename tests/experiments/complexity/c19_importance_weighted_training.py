#!/usr/bin/env python3
"""C19: Importance Weights as Training Signal.

Paper: Complexity
Core question: How should hybrid importance scores enter the training process?
    Tests 5 weighting strategies across models on the HYBRID feature space.

    NOTE (design): an earlier version also evaluated a raw full-concat space to
    ask whether weighting helps more on raw vs hybrid features. The raw-concat
    arm was deliberately removed per design directive — with this many reps the
    full concat is not a meaningful comparison. So this experiment answers the
    narrower question "which weighting strategy, if any, helps on the hybrid" —
    NOT the raw-vs-hybrid contrast.

    2026-10-05 (handoff 2.4, selection rule decided 2026-10-05: standard 2+2):
    per outer fold, per decided model (DECISIONS.md Step 1), the 2 best single
    representations and the 2 best hybrids, all ranked by ONE search on that
    fold's training rows (fold_standard_selection, decided grid, pruning off).
    Each of the four is scored under every weighting condition below. Nothing
    is read from a search over every row. No distance hybrid (handoff 2.3
    item 1: c19 measures nothing with neighbours as its answer).

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
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
    compute_feature_importance,
)
from experiments._shared import (
    build_fold_hybrid, cap_dimensions, fold_hybrid_scoring_kwargs,
    fold_standard_selection,
)
from experiments.decided_grid import decided_models, decided_search
# 2026-10-05 (handoff 2.3 item 17): 0/1 features are never scaled.
from kirby.scaling import FeatureScaler
# Scores a binary-AUROC dataset on class-1 probabilities instead of hard class
# labels; falls through to model.predict for every other metric.
from kirby.find_best_hybrid import _predict_for_metric, nested_cv_outer_folds


# 2026-10-05: stored in every result file. complexity run_all resumes a
# finished file on its input fingerprint alone, which does not see that the
# selection changed; this names the layout and the selection rule.
C19_RESULT_IDENTITY = 'c19_per_fold_standard_selection_2026-10-06'
ONE_FOLD_NOTE = 'one fold, no spread'
SEED = 42

# The budget c19 asks for, in features. Kept at 100 by the owner (tracker C19b,
# 2026-10-05). Used only where 100 is one of the dataset's decided budgets;
# see _selection_budget.
DEFAULT_BUDGET = 100

# Slot names. The rank-1 hybrid keeps its old name 'hybrid', so a reader of
# per_space['hybrid'] still finds the best hybrid.
HYBRID_SLOTS = ('hybrid', 'hybrid__general2')
BASE_SLOTS = ('base1', 'base2')
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')


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
    # 2026-10-05 (handoff 2.3 item 18): the weights take the features' dtype,
    # so multiplying a float32 matrix by them does not turn it into float64.
    return w.astype(np.asarray(X).dtype, copy=False)


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
        # 2026-10-05 (dirty-run bug 7): drop the molecule itself by its row
        # number, not by position. With two identical rows at distance 0 the
        # first neighbour returned can be the twin, and `indices[i, 1:]` then
        # dropped the twin and kept the molecule as its own neighbour.
        row = indices[i]
        nbr_idx = row[row != i][:k_actual]
        nbr_mean = np.mean(y[nbr_idx])
        reliability[i] = 1.0 - abs(y[i] - nbr_mean) / y_range

    return np.clip(reliability, 0.1, 1.0)


# =========================================================================
# MODELS
# =========================================================================

# Tree models. Until 2026-10-05 these were the only models c19 fitted
# (WEIGHTED_MODELS); it now fits the dataset's two decided models, whichever
# they are, and this set only decides where the note below is stored.
TREE_MODELS = {'rf', 'lightgbm', 'xgboost'}

# 2026-10-05, owner delegated, 2026-10-05: stored beside every tree model's
# result. No model is added.
_TREE_FEATURE_WEIGHT_NOTE = (
    'tree model: multiplying a feature by a positive weight does not change '
    'the order of its values, so it cannot change any split; feature_weighted '
    'differs from unweighted only where a feature with importance 0 gets '
    'weight 0 and is in effect dropped. The combined condition differs from '
    'sample_weighted the same way.')


def _selection_budget(dataset_name, budget_asked):
    """(budget passed to fold_standard_selection, note).

    2026-10-05. c19 asks for 100 features (tracker C19b). The per-fold search
    scores only the dataset's decided budgets (DECISIONS.md Step 2), and 100 is
    one of them on 16 of the 43 decided datasets. Where it is, the hybrids are
    the best at 100 features. Where it is not, they are the best over every
    decided budget, as the old whole-dataset lookup did when 100 was missing
    ('budget_relaxed'), and the note says so.
    DECIDED 2026-10-06 (Adelaide delegated): where 100 is not a decided budget,
    c19 takes the best hybrid over the dataset's decided budgets (as above).
    """
    decided = sorted(int(b) for b in
                     decided_search(dataset_name, 'general')['budgets'])
    if budget_asked is None:
        return None, decided, 'no budget asked: best over every decided budget'
    if int(budget_asked) in decided:
        return int(budget_asked), decided, None
    return None, decided, (
        f'{budget_asked} features is not one of the decided budgets '
        f'{decided}; the hybrids are the best over every decided budget')


# =========================================================================
# ONE SLOT, ONE FOLD: EVERY CONDITION
# =========================================================================

# Models whose fit takes no per-molecule weights. The sample_weighted and
# combined conditions are recorded as not run on them, with this reason.
MODELS_WITHOUT_SAMPLE_WEIGHT = {
    'gp': ('gp: the Gaussian process fit takes no per-molecule weights; '
           'not run'),
}


def _score_conditions(X_tr, X_va, y_tr, y_va, factory, conditions,
                      importance_method, metric_fn, model_name=None):
    """Fit and score one feature matrix under every weighting condition.

    The matrix, the scaling, the importance weights and the reliability
    scores are computed once and shared by every condition, so the
    conditions differ only in how the weights enter the fit.

    Returns ({cond: record}, slot_meta).
    """
    # 2026-10-05 (handoff 2.3 item 17): FeatureScaler, so 0/1 features pass
    # through unscaled; continuous features are standardised.
    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    n_in = int(X_tr_s.shape[1])
    X_tr_s, X_va_s = cap_dimensions(X_tr_s, X_va_s)
    meta = {'n_features_in': n_in,
            'pca_capped_to': (int(X_tr_s.shape[1])
                              if X_tr_s.shape[1] != n_in else None),
            'importance_method': importance_method}

    # Importance on this fold's training rows only. Computed once here; it
    # was recomputed inside every condition, with the same inputs.
    imp_weights, imp_error = None, None
    try:
        imp_weights = _compute_importance_weights(X_tr_s, y_tr,
                                                  method=importance_method)
    except Exception as e:
        imp_error = f'importance ({importance_method}): {type(e).__name__}: {e}'
        print(f"      {imp_error}")
    reliability = None
    if imp_weights is not None and any(
            c in conditions for c in ('sample_weighted', 'combined')):
        # Reliability on the unweighted (standardized) space: the function
        # multiplies by imp_weights itself (kNN in X*w space, not X*w^2).
        reliability = _compute_reliability_scores(X_tr_s, y_tr, imp_weights,
                                                  k=10)

    out = {}
    for cond in conditions:
        rec = {'heldout': None}
        if (cond in ('sample_weighted', 'combined')
                and model_name in MODELS_WITHOUT_SAMPLE_WEIGHT):
            rec['not_run'] = MODELS_WITHOUT_SAMPLE_WEIGHT[model_name]
            rec['sample_weight_rejected'] = True
            out[cond] = rec
            continue
        if cond != 'unweighted' and imp_weights is None:
            rec['error'] = imp_error
            out[cond] = rec
            continue
        Xt, Xv = X_tr_s, X_va_s
        fit_kwargs = {}
        if cond == 'feature_weighted':
            Xt = Xt * imp_weights
            Xv = Xv * imp_weights
        elif cond == 'sample_weighted':
            fit_kwargs['sample_weight'] = reliability
        elif cond == 'combined':
            fit_kwargs['sample_weight'] = reliability
            Xt = Xt * imp_weights
            Xv = Xv * imp_weights
        elif cond == 'importance_pruned':
            # Strictly above the median, not at or above it. When more than
            # half the hybrid features score zero importance the median is
            # zero, '>=' passes every feature, and the delta reads as
            # "deleting made no difference" when nothing had been deleted.
            mask = imp_weights > np.median(imp_weights)
            if mask.sum() < 2:
                # Every feature tied at the median. Delete the bottom half by
                # rank instead, so the condition still deletes something.
                order = np.argsort(imp_weights)
                mask = np.zeros(len(imp_weights), dtype=bool)
                mask[order[len(order) // 2:]] = True
            Xt = Xt[:, mask]
            Xv = Xv[:, mask]
        rec['n_features_fitted'] = int(Xt.shape[1])
        try:
            model = factory()
            model.fit(Xt, y_tr, **fit_kwargs)
            # _predict_for_metric, not .predict: an AUROC dataset scored on
            # hard class labels is a degraded AUROC.
            rec['heldout'] = float(metric_fn(
                y_va, _predict_for_metric(model, Xv, metric_fn)))
        except TypeError as e:
            if 'sample_weight' in fit_kwargs:
                # The model's fit takes no sample_weight (the Gaussian
                # process). Left empty and named; never refitted unweighted
                # under the weighted name (tracker X5).
                rec['sample_weight_rejected'] = True
                print(f"      {cond}: the model's fit takes no "
                      f"sample_weight; left empty")
            else:
                rec['error'] = f'{type(e).__name__}: {e}'
                print(f"      {cond}: fit failed - {rec['error']}")
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
            print(f"      {cond}: fit failed - {rec['error']}")
        out[cond] = rec
    return out, meta


def _base_importance_method(sel_m, dataset_name):
    """The importance method that weights a single representation's features.

    2026-10-05. A single representation has no recipe, so no importance method
    of its own. It takes the method of the same model's rank-1 hybrid on the
    same fold, so both rows of a model's column are weighted the same way, and
    the method is one the dataset's search kept. With no hybrid, the first of
    the dataset's decided importance methods. Hers to confirm (open question).
    """
    hyb = sel_m.get('hybrids') or []
    if hyb and hyb[0].get('importance_method'):
        return hyb[0]['importance_method'], 'rank-1 hybrid of this model and fold'
    decided = decided_search(dataset_name, 'general')['importance_methods']
    return decided[0], 'first decided importance method (no hybrid this fold)'


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C19: Importance-weighted training."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C19: IMPORTANCE-WEIGHTED TRAINING")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    groups_arr = np.asarray(groups) if groups is not None else None
    # The outer fold count comes from the config only (handoff 2.4 step 14).
    n_folds = int(config['n_splits'])
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    domain = data.get('domain')
    augmentations = data.get('augmentations')

    # Models: the dataset's two decided models (DECISIONS.md Step 1), never a
    # fixed list. Until 2026-10-05 c19 fitted rf, lightgbm and xgboost on
    # every dataset.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_without_factory = {
        m: 'no model factory on this dataset (size guard or unknown name)'
        for m in models if m not in factories}
    for m, why in models_without_factory.items():
        print(f"  {m}: {why}; not fitted")

    # Representations: the pruned list (and the augmentation list where Step 8
    # keeps it). data['reps'] is handed to the search whole; a representation
    # outside these lists is named and never scored as a single slot.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _ds_reps = DATASET_REPS.get(dataset_name) or {}
    allowed_reps = set(_ds_reps.get('general') or reps) | set(
        _ds_reps.get('augmentation') or [])
    reps_outside = sorted(r for r in reps if r not in allowed_reps)
    if reps_outside:
        print(f"  representations outside DATASET_REPS[{dataset_name!r}]: "
              f"{reps_outside}; not scored as single representations")

    budget_asked = kwargs.get('budget', DEFAULT_BUDGET)
    sel_budget, decided_budgets, budget_note = _selection_budget(
        dataset_name, budget_asked)
    if budget_note:
        print(f"  budget: {budget_note}")

    conditions = ['unweighted', 'feature_weighted', 'sample_weighted',
                  'combined', 'importance_pruned']
    # `_compute_reliability_scores` uses `y_range = max - min` and
    # `abs(y_i - mean(nbr_y))` — nominal-on-ordinal arithmetic. Meaningless
    # for multi-class clf (k>=3). Drop the conditions that depend on it.
    conditions_dropped = []
    is_multiclass = (task_type == 'classification'
                     and len(np.unique(labels)) > 2)
    if is_multiclass:
        dropped = {'sample_weighted', 'combined'}
        conditions = [c for c in conditions if c not in dropped]
        conditions_dropped = sorted(dropped)
        print(f"  Multi-class clf detected: dropping {sorted(dropped)} "
              f"(reliability score not well-defined on nominal labels)")

    folds = nested_cv_outer_folds(reps, labels, groups_arr, n_folds, SEED,
                                  task_type)
    print(f"  {len(folds)} outer fold(s) (n_splits={n_folds}); models "
          f"{models}; {len(reps)} representations")
    print(f"  Conditions: {conditions}")

    search_kw = dict(dataset_name=dataset_name, augmentations=augmentations,
                     task_type=task_type, metric=metric_name, domain=domain,
                     seed=SEED)

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        tr = np.asarray(tr)
        va = np.asarray(va)
        fold_rec = {'fold': int(k), 'n_train': int(len(tr)),
                    'n_val': int(len(va)),
                    'selection_scope': 'fold_train_rows', 'models': {}}
        per_fold.append(fold_rec)
        print(f"\n  === fold {k} ({len(tr)} training rows, {len(va)} "
              f"held out) ===")
        try:
            sel = fold_standard_selection(
                reps, labels, groups_arr, train_idx=tr, fold_i=k,
                n_outer_folds=n_folds, models=models, budget=sel_budget,
                **search_kw)
        except Exception as e:
            fold_rec['error'] = f'{type(e).__name__}: {e}'
            print(f"  fold {k}: per-fold search failed - {fold_rec['error']}")
            continue

        tr_reps = {r: X[tr] for r, X in reps.items()}
        va_reps = {r: X[va] for r, X in reps.items()}
        y_tr, y_va = labels[tr], labels[va]
        g_tr = groups_arr[tr] if groups_arr is not None else None

        for m in models:
            s = sel.get(m)
            if s is None:
                fold_rec['models'][m] = None
                fold_rec.setdefault('model_errors', {})[m] = (
                    "the search on this fold's training rows failed")
                continue
            factory = factories.get(m)
            if factory is None:
                fold_rec['models'][m] = None
                fold_rec.setdefault('model_errors', {})[m] = (
                    models_without_factory[m])
                continue
            slots = {}

            # ── the 2 best hybrids ──
            for slot, e in zip(HYBRID_SLOTS, s['hybrids']):
                bp = e['blueprint']
                aug_cond = e.get('aug_condition') or 'none'
                row = {'kind': 'hybrid',
                       **{f: e.get(f) for f in HYBRID_FIELDS},
                       'aug_condition': aug_cond,
                       # Mean over the search's inner folds, inside this outer
                       # fold's training rows.
                       'train_row_score': e.get('score_mean')}
                tr_a = va_a = None
                if aug_cond != 'none' and augmentations:
                    tr_a = {a: X[tr] for a, X in augmentations.items()}
                    va_a = {a: X[va] for a, X in augmentations.items()}
                try:
                    X_tr_h, X_va_h, fi = build_fold_hybrid(
                        tr_reps, va_reps, y_tr, bp,
                        augmentations_train=tr_a, augmentations_val=va_a,
                        groups=g_tr,
                        # Dirty-run bug 1: the representation scores of THIS
                        # fold's search, measured on its training rows.
                        precomputed_alloc_scores=getattr(bp, 'rep_scores',
                                                         None),
                        **fold_hybrid_scoring_kwargs(
                            bp, model=factory(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
                    print(f"    {m} {slot}: build failed - {row['error']}")
                    slots[slot] = row
                    continue
                # The width built, not the budget asked for.
                row['n_features_built'] = int(np.asarray(X_tr_h).shape[1])
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                print(f"    {m} {slot}: {row['n_features_built']} features "
                      f"({row['allocation_method']}, "
                      f"{row['importance_method']}, budget {row['budget']}, "
                      f"eval {row['internal_evaluator']}, "
                      f"rep_count {row['rep_count']})")
                # Weight with the importance method the search picked for this
                # hybrid, not a second one.
                row['conditions'], row['scaling'] = _score_conditions(
                    X_tr_h, X_va_h, y_tr, y_va, factory, conditions,
                    getattr(bp, 'importance_method', None)
                    or e.get('importance_method'), metric_fn, model_name=m)
                for cond in conditions:
                    h = row['conditions'][cond].get('heldout')
                    if h is not None:
                        print(f"      {cond}: {h:.4f}")
                slots[slot] = row
            if len(s['hybrids']) < len(HYBRID_SLOTS):
                fold_rec.setdefault('hybrid_shortfall', {})[m] = (
                    f"the search on this fold's training rows gave "
                    f"{len(s['hybrids'])} hybrid(s) with distinct features"
                    + (f" at {sel_budget} features" if sel_budget else ''))

            # ── the 2 best single representations ──
            base_imp, base_imp_rule = _base_importance_method(s, dataset_name)
            base_sel = [e for e in s['base_detail']['selected']
                        if e['rep'] in allowed_reps]
            for slot, e in zip(BASE_SLOTS, base_sel):
                rep = e['rep']
                row = {'kind': 'base', 'rep': rep,
                       'train_row_score': e.get('score'),
                       'n_features_built': int(reps[rep].shape[1]),
                       'importance_method_rule': base_imp_rule}
                print(f"    {m} {slot}: {rep} ({row['n_features_built']} "
                      f"features)")
                row['conditions'], row['scaling'] = _score_conditions(
                    tr_reps[rep], va_reps[rep], y_tr, y_va, factory,
                    conditions, base_imp, metric_fn, model_name=m)
                for cond in conditions:
                    h = row['conditions'][cond].get('heldout')
                    if h is not None:
                        print(f"      {cond}: {h:.4f}")
                slots[slot] = row
            if not s['base_detail'].get('complete', True) or len(base_sel) < 2:
                fold_rec.setdefault('base_shortfall', {})[m] = (
                    s['base_detail'].get('shortfall_reason')
                    or f'{len(base_sel)} single representation(s) inside '
                       f'the pruned list')

            fold_rec['models'][m] = {
                'slots': slots,
                'hybrids_skipped_identical_features': [
                    h.get('label') for h in
                    s.get('hybrids_skipped_identical_features') or []],
            }

    per_space = _summarise(per_fold, models, conditions, lower_is_better,
                           len(folds))

    results = {
        'result_identity': C19_RESULT_IDENTITY,
        'selection_scope': 'fold_train_rows',
        'selection_rule': ('per outer fold, per decided model: 2 best single '
                           'representations + 2 best hybrids, ranked on the '
                           "fold's training rows (fold_standard_selection)"),
        'metric': metric_name,
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'task_type': task_type,
        'models': models,
        'models_without_factory': models_without_factory,
        'n_outer_folds': n_folds,
        'n_outer_folds_built': len(folds),
        'representations': sorted(reps),
        'representations_outside_pruned_list': reps_outside,
        # The budget ASKED FOR, and the one the hybrids were selected at
        # (None = best over every decided budget). Each hybrid row stores the
        # budget its recipe carries and the width it built.
        'budget': budget_asked,
        'budget_selected_at': sel_budget,
        'decided_budgets': decided_budgets,
        'budget_note': budget_note,
        'conditions': conditions,
        'conditions_dropped': conditions_dropped,
        'features_selected_on': "each fold's training rows",
        'features_selected_with': 'the model the column is scored under',
        'recipe_selected_on': "each fold's training rows",
        'per_fold': per_fold,
        # Per slot, per model, per condition: the mean held-out score over
        # the folds it was scored on. Slot 'hybrid' is the rank-1 hybrid of
        # each fold, which need not be the same recipe on every fold.
        'per_space': per_space,
    }

    # Summary: which condition wins most often?
    condition_wins = {c: 0 for c in conditions}
    for space_res in per_space.values():
        for model_res in space_res.values():
            best = model_res.get('best_condition')
            if best is not None:
                condition_wins[best] = condition_wins.get(best, 0) + 1
    results['summary'] = {
        'condition_wins': condition_wins,
        # CXA-16/OTHER-14: renamed from 'best_overall'. The weighting
        # condition that won the most (model x slot) cells. None when no cell
        # was scored.
        'most_winning_condition': (max(condition_wins, key=condition_wins.get)
                                   if any(condition_wins.values()) else None),
    }
    if not any(condition_wins.values()):
        results['error'] = 'nothing_scored'
        print("  Nothing was scored on any fold.")
    print(f"\n  Condition wins: {condition_wins}")

    return convert_numpy(results)


def _summarise(per_fold, models, conditions, lower_is_better, n_built):
    """{slot: {model: summary}} over the outer folds."""
    out = {}
    for slot in HYBRID_SLOTS + BASE_SLOTS:
        space = {}
        for m in models:
            fold_scores = {c: [] for c in conditions}
            fold_widths = {c: [] for c in conditions}
            rejected = {}
            built = []
            for fr in per_fold:
                row = (((fr.get('models') or {}).get(m) or {})
                       .get('slots') or {}).get(slot)
                built.append(row.get('n_features_built')
                             if row and 'conditions' in row else None)
                for c in conditions:
                    rc = ((row or {}).get('conditions') or {}).get(c) or {}
                    h = rc.get('heldout')
                    fold_scores[c].append(float(h) if h is not None
                                          else float('nan'))
                    fold_widths[c].append(rc.get('n_features_fitted'))
                    if rc.get('sample_weight_rejected'):
                        rejected.setdefault(c, []).append(int(fr['fold']))
            if all(b is None for b in built):
                continue
            scores, n_scored = {}, {}
            for c in conditions:
                v = [x for x in fold_scores[c] if not np.isnan(x)]
                scores[c] = float(np.mean(v)) if v else float('nan')
                n_scored[c] = len(v)
            baseline = scores.get('unweighted', float('nan'))
            deltas, deltas_oriented = {}, {}
            for c in conditions:
                if c == 'unweighted':
                    continue
                s = scores[c]
                if not np.isnan(s) and not np.isnan(baseline):
                    # Raw difference in the dataset's own metric, plus a
                    # sign-normalised sibling so a positive number always
                    # means the weighting helped (2026-09-10).
                    deltas[c] = float(s - baseline)
                    deltas_oriented[c] = float(
                        (baseline - s) if lower_is_better else (s - baseline))
            finite = {c: v for c, v in scores.items() if not np.isnan(v)}
            best = ((min if lower_is_better else max)(finite, key=finite.get)
                    if finite else None)
            space[m] = {
                'scores': scores,
                'fold_scores': fold_scores,
                'n_folds_scored': n_scored,
                'fold_note': ONE_FOLD_NOTE if n_built == 1 else None,
                'width_per_fold': fold_widths,
                'width_built_per_fold': built,
                'sample_weight_rejected_folds': rejected,
                'deltas': deltas,
                'deltas_oriented': deltas_oriented,
                'best_condition': best,
                # 2026-10-05, owner delegated: see _TREE_FEATURE_WEIGHT_NOTE.
                'feature_weighting_note': (_TREE_FEATURE_WEIGHT_NOTE
                                           if m in TREE_MODELS else None),
            }
        if space:
            out[slot] = space
    return out


if __name__ == '__main__':
    from experiments.complexity.complexity_helpers import DATASET_REPS
    parser = create_experiment_parser(
        'C19: Importance-Weighted Training')
    args = parser.parse_args()
    config = get_config(args)
    # 2026-10-05 (handoff 2.4 step 9): the dataset's pruned representations
    # (DATASET_REPS[ds]['general']), as run_all loads them, not a --reps preset.
    _general = (DATASET_REPS.get(args.dataset) or {}).get('general')
    if _general:
        data = load_data(args.dataset, rep_preset='all', rep_names=_general)
    else:
        data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'c19_importance_weighted_training', args.dataset)
