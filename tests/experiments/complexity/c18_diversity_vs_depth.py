#!/usr/bin/env python3
"""C18: Degrees of Freedom — Diversity vs Depth.

Paper: Complexity
Core question: Given a fixed total budget, is it better to draw features from many
    reps (diversity) or concentrate on few reps (depth)? Tests whether the hybrid's
    allocation across rep families adds value beyond selecting the best single rep.

Three levels at each decided budget, per decided model, per outer fold
(2026-10-30, Adelaide). Every level ranks features by the same importance
method and is fitted and scored the same way; only how many representations
the features may come from changes.

  Level 1, no diversity: the budget filled from the best single
    representation (depth_1).
  Level 2, pooled: depth_<n> for n in 1, 2, 3, 5 and the whole pool. The
    budget is filled from the best n representations, their features ranked
    against each other. This is the global-pool allocation method by hand.
  Level 3, the searched hybrid: natural and natural__general2. Its number of
    representations is whatever the search picked, recorded per fold as
    `n_reps_used` -- the representations that actually received features --
    so it reads on the same axis as level 2.

Conditions at fixed budget, per decided model, per outer fold:
  - depth_1: all features from top-1 rep
  - depth_2: all features split between top-2 reps
  - depth_3: all features from top-3 reps
  - depth_5, depth_<pool size>: the same, at 5 and at every representation
  - uniform_all: equal allocation across the representations the natural
    arm used, to the natural arm's width (decided 2026-10-06; the key name
    is kept for readers)
  - uniform_every_rep: equal allocation across every general representation
    of the dataset (DATASET_REPS general list), filled to the budget; one fit
    per budget per model per fold, no search, does not depend on the natural
    arm (the pre-2026-10-06 uniform_all, back 2026-10-07)
  - natural: let the hybrid allocate freely (the best hybrid at this budget
    from the search on this fold's training rows)
  - natural__general2: the second-best hybrid at this budget from the same
    search (skipped for the next one when its selected features equal the
    first's)

Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). Before, every hybrid
came from the whole-dataset search and every model was a RandomForest. Now:
  - models are the dataset's two decided models (DECISIONS.md Step 1); every
    arm is fitted with each of them;
  - each outer fold (kirby.find_best_hybrid.nested_cv_outer_folds, count from
    config['n_splits']) runs one search on its training rows on the decided
    grid, pruning off (_shared.fold_standard_selection), and the natural arms
    are its 2 best hybrids per model at each decided budget;
  - each model's 2 best base representations on that fold are fitted and
    scored on the held-out rows, beside the arms;
  - the distance hybrid arm (natural__distance1) is gone: c18 measures nothing
    with neighbours or distances (handoff 2.3 item 1);
  - features are scaled with kirby.scaling.FeatureScaler (0/1 features left
    as they are);
  - each finished fold is written to a partial file and read back on resume.

Usage:
    python c18_diversity_vs_depth.py --dataset logd --mode intermediary
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# Ranking metrics need graded scores: an AUROC read off hard 0/1 labels
# collapses to the mean of sensitivity and specificity (hybrid.py:2661).
from kirby.find_best_hybrid import (
    _predict_for_metric, nested_cv_outer_folds, nested_cv_cache_namespace)
# 0/1 features are never scaled, continuous ones are standardised (Adelaide,
# 2026-10-05; handoff 2.3 item 17). Replaces this file's StandardScaler.
from kirby.scaling import FeatureScaler
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
    compute_feature_importance,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection,
    top_base_reps, hybrid_deltas,
)

EXPERIMENT_DIR = 'c18_diversity_vs_depth'
SEED = 42
# The result identity (2026-10-05). Per-fold searches on the decided grid and
# the dataset's two models replaced the whole-dataset search and rf, so a
# partial file written under any other format is moved aside, never read back.
# 2026-10-06: the uniform arm now matches the natural arm's representations
# and width, and the all-rows reference ranking is gone, so the identity moved.
# 2026-10-07: uniform_every_rep added beside uniform_all, so the identity moved.
RESULT_FORMAT = 'per_fold_pooled_n_reps_range_2026_10_30'

# The numbers of representations the pooled level runs at (2026-10-30,
# Adelaide). 1 is the best base representation on its own; the dataset's own
# representation count tops the range. A value above the pool is dropped, and
# a duplicate of the pool size with it.
POOLED_N_REPS_ASKED = (1, 2, 3, 5)


def pooled_n_reps(n_reps):
    """The asked numbers that fit this pool, plus the whole pool."""
    vals = [n for n in POOLED_N_REPS_ASKED if n <= int(n_reps)]
    if int(n_reps) not in vals:
        vals.append(int(n_reps))
    return vals


# Fields of a hybrid configuration written for every natural arm (2026-10-05).
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'n_features')

# Arms the whole-dataset version ran that the decided rules do not allow.
# Recorded in the result file with the reason; none is run (2026-10-05).
ARMS_NOT_RUN = {
    'natural__distance1': (
        'c18 measures nothing with neighbours or distances, so it uses the 2 '
        'general hybrids only (handoff 2.3 item 1); no distance hybrid'),
    'natural__<slot>__<family>': (
        'the extra similarity-family arms under KIRBY_POOL_SUBSET read the '
        'whole-dataset search (rule R2); not run'),
}


# =========================================================================
# PER-FOLD HELPERS
# =========================================================================

def _importance_for_reps(rep_arrays, labels, method, cache=None):
    """Per-representation importance, re-used inside one fold instead of refitted.

    ⛔ NOTHING IS APPROXIMATED HERE AND NO ARM'S FEATURES CHANGE. This returns
    the same arrays compute_feature_importance returns; it just stops asking for
    the same one repeatedly.

    Why it is safe. compute_feature_importance (hybrid.py:4629) loops over the
    dict it is handed and fits one model per representation on that
    representation's own features -- every one of its seven methods does, and
    the lightgbm_gain branch this run uses scales, variance-pre-screens and fits
    inside that per-representation loop (hybrid.py:5104-5145). So the array it
    returns for a representation does not depend on which other representations
    were in the dict, and two calls with the same representation, the same rows
    and the same method return the same numbers.

    What it saves, measured on flab_thermo_tresanco (5 representations, 1000
    features each after KIRBY_LOCAL_MAX_FEATURES_PER_REP, 672 rows, 3 folds,
    budgets 100 and 250). c18 asked for importance 16 times per fold per budget:
    5 in the fold ranking, 1 + 2 + 3 in the three depth arms, and 5 one
    representation at a time in the uniform arm -- and then asked for the whole
    fold again at the next budget, because only how many features get taken
    depends on the budget. That is 96 LightGBM fits where 15 distinct ones
    exist.

    Measured 2026-09-25 on this laptop with nine to fifteen other experiments
    running beside it, so every wall time below is a share of one core, not a
    clean cost. Without this, the run had not finished the first of its two
    budgets after 20 minutes and was killed. With it the same run finished in
    36.6 minutes, and the two budgets took 1145s and 1053s -- the second of
    which does no importance work at all, because the first already did it.
    """
    if cache is None:
        return compute_feature_importance(rep_arrays, labels, method=method)
    out, missing = {}, {}
    for name, X in rep_arrays.items():
        if (method, name) in cache:
            out[name] = cache[(method, name)]
        else:
            missing[name] = X
    if missing:
        for name, scores in compute_feature_importance(
                missing, labels, method=method).items():
            cache[(method, name)] = scores
            out[name] = scores
    return out


def _rank_reps_by_importance(reps, labels, method='lightgbm_gain', cache=None):
    """Rank reps by aggregate importance. Returns sorted list of (name, score).

    `method` is the search's own importance method, passed in by the caller. The
    four hand-built arms and the search's hybrid have to rank features the same
    way, or a win for the hybrid could be the allocation or could be the
    importance method and the two cannot be separated.
    """
    imp_dict = _importance_for_reps(reps, labels, method, cache)
    rep_scores = {name: float(np.sum(scores)) for name, scores in imp_dict.items()}
    return sorted(rep_scores.items(), key=lambda x: x[1], reverse=True)


def _select_top_features(train_reps, val_reps, y_train, rep_subset, budget,
                         method='lightgbm_gain', cache=None):
    """Per-fold: select top features by train-only importance, apply to both.

    `method` is the search's own importance method — see _rank_reps_by_importance.
    """
    sub_tr = {k: train_reps[k] for k in rep_subset if k in train_reps}
    sub_va = {k: val_reps[k] for k in rep_subset if k in val_reps}
    if not sub_tr:
        # Three values, not two. The caller unpacks three
        # (X_tr, X_va, _wn = _select_top_features(...)), so the two-value return
        # that stood here raised ValueError instead of skipping the arm.
        return None, None, None

    rep_names = sorted(sub_tr.keys())
    imp_dict = _importance_for_reps(sub_tr, y_train, method, cache)
    imp_concat = np.concatenate([imp_dict[n] for n in rep_names])

    X_tr = np.hstack([sub_tr[n] for n in rep_names])
    X_va = np.hstack([sub_va[n] for n in rep_names])

    # The pooled conditions fill the budget; the searched hybrid does not.
    # Taking the top n_select by argsort keeps features whatever they scored,
    # and lightgbm_gain writes exactly 0.0 for every feature no tree split on,
    # so past the number that scored above zero this is padded from a block of
    # exact ties broken by feature position. The searched hybrid goes through
    # build_fold_hybrid -> create_hybrid, which stops at the last positively
    # scored feature: on flab at budget 1,000 that is roughly 1,000 features
    # against roughly 400.
    #
    # DECIDED 2026-10-30 (Adelaide): leave it. Stopping early is part of the
    # search, and a hybrid that used 400 of a budget of 1,000 is still at a
    # budget of 1,000. Both are labelled by their budget, not by the features
    # they used. The realised number of features is returned either way, so
    # the difference is visible.
    n_select = min(budget, X_tr.shape[1])
    top_idx = np.argsort(imp_concat)[::-1][:n_select]
    _n_positive = int(np.sum(np.asarray(imp_concat) > 0))
    _width_note = {'built': int(len(top_idx)),
                   'scored_above_zero': _n_positive,
                   'would_be_if_cut': int(min(len(top_idx), _n_positive))}
    return X_tr[:, top_idx], X_va[:, top_idx], _width_note


def _uniform_allocation(train_reps, val_reps, y_train, rep_subset, budget,
                        method='lightgbm_gain', cache=None):
    """Per-fold: equal features per rep, top by train-only importance.

    `method` is the search's own importance method — see _rank_reps_by_importance.
    """
    sub_names = sorted([k for k in rep_subset if k in train_reps])
    if not sub_names:
        return None, None

    # The equal share has to add up to the SAME number of features the depth
    # arms get, or a loss for it is a loss on width and not on spreading the
    # budget. budget // n_reps threw the remainder away: at 10 representations
    # and a budget of 25 this arm ran on 20 features against the depth arms'
    # 25, and at a budget of 10 on 10 against 10 only by accident. The
    # remainder now goes one feature at a time to the representations in the
    # order they are named, and a representation narrower than its share hands
    # what it cannot fill back to the others.
    quota = {n: 0 for n in sub_names}
    width = {n: int(train_reps[n].shape[1]) for n in sub_names}
    remaining = int(min(budget, sum(width.values())))
    while remaining > 0:
        open_names = [n for n in sub_names if quota[n] < width[n]]
        if not open_names:
            break
        share, extra = divmod(remaining, len(open_names))
        if share == 0:
            for n in open_names[:remaining]:
                quota[n] += 1
            break
        for i, n in enumerate(open_names):
            take = min(share + (1 if i < extra else 0), width[n] - quota[n])
            quota[n] += take
        remaining = int(min(budget, sum(width.values()))) - sum(quota.values())

    tr_parts, va_parts = [], []
    for name in sub_names:
        n_sel = quota[name]
        if n_sel <= 0:
            continue
        imp_dict = _importance_for_reps(
            {name: train_reps[name]}, y_train, method, cache)
        imp = imp_dict[name]
        top_idx = np.argsort(imp)[::-1][:n_sel]
        tr_parts.append(train_reps[name][:, top_idx])
        va_parts.append(val_reps[name][:, top_idx])

    if not tr_parts:
        return None, None
    return np.hstack(tr_parts), np.hstack(va_parts)


# =========================================================================
# RUN
# =========================================================================

def _natural_arm_name(slot_i):
    """Arm name for the i-th hybrid (1-based) of one model's per-fold pair.

    Hybrid 1 keeps the old name 'natural', so a reader of
    per_budget[budget]['natural'] reads the arm it always read. Hybrid 2 keeps
    the old 'natural__general2'.
    """
    return 'natural' if slot_i == 1 else f'natural__general{slot_i}'


def _hybrid_row(e):
    """The recipe fields of one fold_standard_selection hybrid (no blueprint)."""
    return {f: e.get(f) for f in HYBRID_FIELDS}


def _score_prediction(model, X_val, y_val, metric_fn, task_type):
    """Score model predictions appropriately for the metric.

    Classification + binary: AUROC needs class-1 probabilities, not labels.
    Classification + multi-class: F1 needs class labels, not probabilities.
    Regression: anything that has .predict() works.
    """
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 2:
                return metric_fn(y_val, probs[:, 1])
        except Exception:
            pass
    return metric_fn(y_val, _predict_for_metric(model, X_val, metric_fn))


def _fit_score(X_tr, X_va, y_tr, y_va, factory, metric_fn, task_type):
    """Fit one model on a fold's training rows, score it on its held-out rows.

    2026-10-05: FeatureScaler fitted on the training rows (0/1 features left
    as they are; this file used StandardScaler on every feature), and each
    held-out feature capped at the range it takes on the scaled training rows,
    as c02, c05, c07 and c08 do since the owner's delegated default of
    2026-10-05 (82ef2a1). With rf only the cap changed nothing; svm and dnn
    are fitted here now. No float64 conversion (handoff 2.3 item 18).
    """
    X_tr = np.nan_to_num(np.asarray(X_tr), nan=0.0)
    X_va = np.nan_to_num(np.asarray(X_va), nan=0.0)
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_va = np.clip(X_va, X_tr.min(axis=0), X_tr.max(axis=0))
    m = factory()
    m.fit(X_tr, y_tr)
    return float(_score_prediction(m, X_va, y_va, metric_fn, task_type))


def _over_folds(values):
    """Mean and raw spread over outer folds. No interval, no test."""
    vals = [float(v) for v in values
            if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': None, 'min': None, 'max': None, 'n_folds': 0}
    out = {'mean': float(np.mean(vals)), 'n_folds': len(vals)}
    if len(vals) == 1:
        out.update(min=None, max=None, spread='one fold, no spread')
    else:
        out.update(min=float(np.min(vals)), max=float(np.max(vals)))
    return out


def _plain(obj):
    """Blueprints and searches dropped, numpy values to Python ones. Dict keys
    are kept as they are, so the integer budget keys stay integers in the
    per-fold partial file."""
    if isinstance(obj, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _plain(v)
                for k, v in obj.items() if k not in ('blueprint', 'search')}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return convert_numpy(obj)


def _fallback_importance_method(decided_methods):
    """The importance method the hand-built arms use when the fold's search
    gave a model no hybrid at a budget: lightgbm_gain, the old fallback, when
    the dataset's decided grid keeps it, else the first decided method."""
    if not decided_methods or 'lightgbm_gain' in decided_methods:
        return 'lightgbm_gain'
    return decided_methods[0]


def _run_fold(k, tr, va, *, reps, labels, groups, dataset_name, n_outer,
              models, factories, budgets, general_names, metric_name,
              metric_fn, lower_is_better, task_type, decided_imp_methods):
    """Every arm c18 measures on one outer fold, per model and budget, as
    plain data."""
    ns = nested_cv_cache_namespace(k, n_outer, SEED)
    y = np.asarray(labels)
    y_tr, y_va = y[tr], y[va]
    g_tr = None if groups is None else np.asarray(groups)[tr]
    # The hybrid builder gets every representation the search saw (handoff
    # 2.4 step 9: data['reps'] is never filtered before a hybrid builder). The
    # hand-built arms loop over representations, so they get the dataset's
    # general list only.
    tr_all = {r: np.asarray(X)[tr] for r, X in reps.items()}
    va_all = {r: np.asarray(X)[va] for r, X in reps.items()}
    tr_gen = {r: tr_all[r] for r in general_names}
    va_gen = {r: va_all[r] for r in general_names}
    n_reps = len(general_names)

    rec = {'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'cache_namespace': ns, 'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'search_failed': [], 'models': {}}

    # Importance arrays for this fold's training rows, shared by every model
    # and budget (see _importance_for_reps for why re-using them changes no
    # number).
    imp_cache = {}
    fold_t0 = time.time()
    natural_secs = 0.0

    for b in budgets:
        # One search on this fold's training rows (memoised by fold_searches,
        # so every budget reads the same search); the 2 best hybrids per model
        # at this budget, a second hybrid with the first's selected features
        # skipped for the next.
        sel = fold_standard_selection(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            models=list(models), task_type=task_type, metric=metric_name,
            seed=SEED, budget=b)
        for m in models:
            s = sel.get(m)
            mrec = rec['models'].setdefault(m, {'base': None, 'per_budget': {}})
            if s is None:
                rec['search_failed'].append({'model': m, 'budget': b})
                print(f"    fold {k} {m:8s} budget {b}: the search on this "
                      f"fold's training rows failed; no arm scored")
                continue
            factory = factories.get(m)
            search = s['search']

            # --- The model's 2 best base representations (budget-free) -----
            if mrec['base'] is None:
                base_rows = []
                for i, e in enumerate(s['base_detail']['selected'], 1):
                    row = {'slot': f'base{i}', 'rep': e['rep'],
                           'n_features': int(np.asarray(
                               reps[e['rep']]).shape[1]),
                           'train_score': e.get('score'),
                           'heldout_score': None}
                    if factory is None:
                        row['error'] = f'no model factory for {m}'
                    else:
                        try:
                            row['heldout_score'] = _fit_score(
                                tr_all[e['rep']], va_all[e['rep']], y_tr, y_va,
                                factory, metric_fn, task_type)
                        except Exception as ex:
                            row['error'] = f'{type(ex).__name__}: {ex}'
                    base_rows.append(row)
                mrec['base'] = base_rows
                mrec['base_shortfall'] = s['base_detail'].get(
                    'shortfall_reason')

            hybrids = s['hybrids']
            brec = {
                'hybrids_skipped_identical_features': [
                    x.get('label') for x in
                    (s.get('hybrids_skipped_identical_features') or [])],
                'hybrid_shortfall': (
                    None if len(hybrids) >= 2 else
                    f'the search on this fold gave {m} {len(hybrids)} '
                    f'hybrid(s) with distinct features at budget {b}'),
                'arms': {},
            }

            # The importance method the hand-built arms rank features with:
            # hybrid 1's, so hand-built arms and 'natural' differ in how the
            # budget is split and not in how features are ranked.
            # 'fallback' = this model has no hybrid at this budget on this fold.
            if hybrids and hybrids[0].get('importance_method'):
                imp_method = hybrids[0].get('importance_method')
                imp_source = 'search'
            else:
                imp_method = _fallback_importance_method(decided_imp_methods)
                imp_source = 'fallback'
                print(f"    ⛔ fold {k} {m} budget {b}: importance method for "
                      f"the hand-built arms {imp_method} — FALLBACK, the "
                      f"search gave no hybrid here")
            brec['importance_method'] = imp_method
            brec['importance_method_source'] = imp_source

            # Top 2 base representations no wider than the budget, from the
            # same search (training rows), so a reader can compare a base
            # number with a hybrid number filtered the same way.
            try:
                brec['top2_base_reps_within_budget'] = _plain(top_base_reps(
                    search, m, 2, max_features=b, require_full=False,
                    context=f'c18 {dataset_name} fold {k} {m} budget={b} '
                            f'(representations no wider than the budget)'))
            except Exception as ex:
                brec['top2_base_reps_within_budget'] = {
                    'error': f'{type(ex).__name__}: {ex}'}
            # The two reference numbers for hybrid 1 on this fold's training
            # rows: against the mean of the representations that fed it, and
            # against the best one.
            if hybrids:
                try:
                    brec['hybrid1_train_row_deltas'] = _plain(hybrid_deltas(
                        search, m, hybrid=hybrids[0], budget=b,
                        context=f'c18 {dataset_name} fold {k} {m} budget={b}'))
                except Exception as ex:
                    brec['hybrid1_train_row_deltas'] = {
                        'error': f'{type(ex).__name__}: {ex}'}

            if factory is None:
                brec['error'] = f'no model factory for {m} at this size'
                mrec['per_budget'][b] = brec
                continue

            arms = brec['arms']
            # Rank representations on this fold's training rows only.
            fold_ranked = _rank_reps_by_importance(
                tr_gen, y_tr, method=imp_method, cache=imp_cache)

            # The budget filled from the best n representations, their
            # features ranked against each other (2026-10-30, Adelaide: the
            # same n values the searched hybrid can be read at, so the two sit
            # on one axis). n = 1 is the best base representation cut to the
            # budget.
            for n_top in pooled_n_reps(n_reps):
                cond = f'depth_{n_top}'
                top_names = [name for name, _ in fold_ranked[:n_top]]
                X_tr, X_va, wn = _select_top_features(
                    tr_gen, va_gen, y_tr, top_names, b,
                    method=imp_method, cache=imp_cache)
                arm = {'reps_used': top_names, 'n_reps_used': len(top_names),
                       'train_score': None,
                       'heldout_score': None, 'importance_method': imp_method}
                if wn:
                    arm.update(width_built=wn['built'],
                               width_scored_above_zero=wn['scored_above_zero'])
                if X_tr is not None:
                    try:
                        arm['heldout_score'] = _fit_score(
                            X_tr, X_va, y_tr, y_va, factory, metric_fn,
                            task_type)
                    except Exception as ex:
                        arm['error'] = f'{type(ex).__name__}: {ex}'
                arms[cond] = arm

            # Natural hybrids: the fold search's 2 best at this budget, each
            # rebuilt on this fold's training rows. Base representations only
            # (the search ran the 'none' augmentation condition), as the depth
            # and uniform arms are.
            for i, e in enumerate(hybrids, 1):
                name = _natural_arm_name(i)
                bp = e['blueprint']
                arm = {'slot': f'hybrid{i}', **_hybrid_row(e),
                       'aug_condition': e.get('aug_condition') or 'none',
                       # Mean over the search's inner folds, inside this
                       # outer fold's training rows.
                       'train_score': e.get('score_mean'),
                       'heldout_score': None,
                       'recipe_budget': e.get('budget'),
                       # Owner delegated, 2026-10-05 (tracker X7): built at
                       # this loop's budget. The recipe is ranked at this
                       # budget now, so the two agree.
                       'built_at_budget': b}
                try:
                    nt0 = time.time()
                    X_tr_h, X_va_h, fi = build_fold_hybrid(
                        tr_all, va_all, y_tr, bp,
                        augmentations_train=None, augmentations_val=None,
                        groups=g_tr, budget_override=b,
                        # Dirty-run bug 1: the representation scores come from
                        # this fold's own search (the blueprint was built on
                        # this fold's training rows), never a whole-dataset one.
                        precomputed_alloc_scores=getattr(bp, 'rep_scores',
                                                         None),
                        **fold_hybrid_scoring_kwargs(
                            bp, model=factory(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                    natural_secs += time.time() - nt0
                    arm['width_built'] = int(np.asarray(X_tr_h).shape[1])
                    arm['width_scored_above_zero'] = None
                    arm['features_by_rep'] = {
                        r: int(len(info['selected_indices']))
                        for r, info in fi.items()
                        if isinstance(info, dict)
                        and 'selected_indices' in info}
                    arm['rep_count_applied'] = fi.get('rep_count_applied')
                    # How many representations actually gave this hybrid
                    # features, so it reads on the same axis as the pooled
                    # level (2026-10-30, Adelaide). The recipe's rep_count can
                    # be above this when a kept representation ends up with no
                    # feature.
                    arm['n_reps_used'] = int(sum(
                        1 for c in arm['features_by_rep'].values() if c > 0))
                    arm['heldout_score'] = _fit_score(
                        X_tr_h, X_va_h, y_tr, y_va, factory, metric_fn,
                        task_type)
                except Exception as ex:
                    arm['error'] = f'{type(ex).__name__}: {ex}'
                    print(f"    fold {k} {m:8s} budget {b}: {name} FAILED: "
                          f"{arm['error']}")
                arms[name] = arm

            # DECIDED 2026-10-06 (Adelaide delegated): the uniform arm uses the
            # same representations and the same number of features as the
            # natural arm (hybrid 1 at this budget on this fold), so the two
            # differ only in how the budget is split across representations.
            # Before, it spread the budget over every general representation
            # and filled the budget, while the natural arm used the
            # representations its recipe kept and stopped at its last feature
            # scored above zero (on buchwald_hartwig: 5 of 6 representations,
            # 233 of 250 features). The key stays 'uniform_all' so readers of
            # per_budget[b]['uniform_all'] keep working.
            # 2026-10-07 (Adelaide): the equal split over every general
            # representation is back as `uniform_every_rep`, beside the matched
            # `uniform_all`.
            nat = arms.get('natural')
            arm = {'reps_used': None, 'train_score': None,
                   'heldout_score': None, 'importance_method': imp_method,
                   'matched_to': 'natural'}
            nat_reps = sorted(
                r for r, c in ((nat or {}).get('features_by_rep') or {}).items()
                if c and c > 0 and r in tr_all)
            nat_width = (nat or {}).get('width_built')
            if nat is None:
                arm['not_run'] = (f'the search on this fold gave {m} no hybrid '
                                  f'at budget {b}, so there is no natural arm '
                                  f'to match')
            elif nat.get('error') or not nat_reps or not nat_width:
                arm['not_run'] = ('the natural arm was not built on this fold '
                                  '(see its error), so there is nothing to '
                                  'match')
            else:
                arm['reps_used'] = nat_reps
                arm['n_reps_used'] = len(nat_reps)
                arm['width_target'] = int(nat_width)
                X_tr, X_va = _uniform_allocation(
                    {r: tr_all[r] for r in nat_reps},
                    {r: va_all[r] for r in nat_reps},
                    y_tr, nat_reps, int(nat_width),
                    method=imp_method, cache=imp_cache)
                if X_tr is not None:
                    # scored_above_zero is not computed for this set (None).
                    arm.update(width_built=int(X_tr.shape[1]),
                               width_scored_above_zero=None)
                    try:
                        arm['heldout_score'] = _fit_score(
                            X_tr, X_va, y_tr, y_va, factory, metric_fn,
                            task_type)
                    except Exception as ex:
                        arm['error'] = f'{type(ex).__name__}: {ex}'
            arms['uniform_all'] = arm

            # uniform_every_rep (Adelaide, 2026-10-07): the arm uniform_all was
            # before f2d2c98. The budget is split equally over every general
            # representation of the dataset and filled to the budget. It does
            # not read the natural arm, so it runs when that arm failed.
            X_tr, X_va = _uniform_allocation(
                tr_gen, va_gen, y_tr, list(general_names), b,
                method=imp_method, cache=imp_cache)
            arm = {'reps_used': list(general_names),
                   'n_reps_used': len(general_names), 'train_score': None,
                   'heldout_score': None, 'importance_method': imp_method,
                   'width_target': int(b)}
            if X_tr is not None:
                # scored_above_zero is not computed for this set (None).
                arm.update(width_built=int(X_tr.shape[1]),
                           width_scored_above_zero=None)
                try:
                    arm['heldout_score'] = _fit_score(
                        X_tr, X_va, y_tr, y_va, factory, metric_fn, task_type)
                except Exception as ex:
                    arm['error'] = f'{type(ex).__name__}: {ex}'
            else:
                arm['not_run'] = 'no general representation on this fold'
            arms['uniform_every_rep'] = arm

            for name, arm in arms.items():
                sc = arm.get('heldout_score')
                print(f"    fold {k} {m:8s} budget {b:5d} {name:18s} "
                      f"width={arm.get('width_built', '--')!s:>5} "
                      f"{metric_name}={'--' if sc is None else round(sc, 4)}"
                      f"{'  ' + arm['error'] if arm.get('error') else ''}")
            mrec['per_budget'][b] = brec

    rec['seconds'] = float(time.time() - fold_t0)
    rec['seconds_rebuilding_natural_hybrids'] = float(natural_secs)
    print(f"    [fold {k}: {rec['seconds']:.0f}s, of which "
          f"{natural_secs:.0f}s rebuilding the natural hybrids]")
    return _plain(rec)


def _summarise(per_fold, models, budgets):
    """Per model, per budget, per arm: held-out scores per fold, their mean
    and raw spread, and the widths built. Shape of the old per_budget block,
    one per model."""
    out = {}
    for m in models:
        per_budget = {}
        for b in budgets:
            arms = {}
            for rec in per_fold:
                brec = (((rec.get('models') or {}).get(m) or {})
                        .get('per_budget') or {}).get(b) or {}
                for name, arm in (brec.get('arms') or {}).items():
                    a = arms.setdefault(name, {
                        'fold_scores': [], 'folds': [],
                        'width_built_per_fold': [],
                        'width_scored_above_zero_per_fold': [],
                        'n_reps_used_per_fold': [],
                        'importance_method_per_fold': [],
                        'budget': b})
                    sc = arm.get('heldout_score')
                    if sc is None:
                        continue
                    a['fold_scores'].append(float(sc))
                    a['folds'].append(int(rec['fold']))
                    a['width_built_per_fold'].append(arm.get('width_built'))
                    a['width_scored_above_zero_per_fold'].append(
                        arm.get('width_scored_above_zero'))
                    a['n_reps_used_per_fold'].append(arm.get('n_reps_used'))
                    a['importance_method_per_fold'].append(
                        arm.get('importance_method'))
                    if name.startswith('natural'):
                        a.setdefault('recipe_per_fold', []).append(
                            {f: arm.get(f) for f in HYBRID_FIELDS
                             + ('aug_condition', 'train_score',
                                'rep_count_applied')})
            for name, a in arms.items():
                spread = _over_folds(a['fold_scores'])
                a['score'] = spread['mean']
                a['spread_over_folds'] = spread
            per_budget[b] = arms
        base = {}
        for slot in ('base1', 'base2'):
            vals, picked = [], []
            for rec in per_fold:
                rows = (((rec.get('models') or {}).get(m) or {})
                        .get('base') or [])
                row = next((r for r in rows if r.get('slot') == slot), None)
                if row is None:
                    continue
                picked.append({'fold': rec['fold'], 'rep': row['rep'],
                               'train_score': row.get('train_score'),
                               'heldout_score': row.get('heldout_score')})
                vals.append(row.get('heldout_score'))
            base[slot] = {'per_fold': picked, 'over_folds': _over_folds(vals)}
        out[m] = {'per_budget': per_budget, 'base': base}
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """C18: Diversity vs Depth, per outer fold, on the dataset's two models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C18: DIVERSITY VS DEPTH (per fold)")
    print("=" * 70)

    from experiments.decided_grid import decided_models, decided_search
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    task_type = data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # Models: the dataset's two decided models (DECISIONS.md Step 1). knn is
    # never a general model here.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_without_factory = [m for m in models if m not in factories]
    decided = decided_search(dataset_name, 'general')
    decided_imp_methods = list(decided.get('importance_methods') or [])

    # Representations: the hand-built arms loop over the dataset's general
    # list only (handoff 2.4 step 9). data['reps'] goes to the search and the
    # hybrid builder unfiltered.
    general_names = [r for r in ((DATASET_REPS.get(dataset_name) or {})
                                 .get('general') or sorted(reps))
                     if r in reps]
    extra = sorted(set(reps) - set(general_names))
    if extra:
        print(f"  representations outside DATASET_REPS['{dataset_name}']"
              f"['general'] left out of the hand-built arms: {extra}")
    n_reps = len(general_names)

    # Budgets: the decided general budgets (features). A narrower
    # config['budgets'] (--budgets, or the local smoke shrink) is kept where
    # its members are decided. A budget above the general representations'
    # total width collapses to the full width and adds nothing to the
    # diversity-vs-depth contrast, so it is left out and named.
    budgets = list(decided['budgets'])
    asked = config.get('budgets')
    if asked:
        kept = [b for b in budgets if b in set(asked)]
        if kept:
            budgets = kept
    total_dim = sum(int(np.asarray(reps[r]).shape[1]) for r in general_names)
    too_wide = sorted(b for b in budgets if b > total_dim)
    budgets = sorted({b for b in budgets if b <= total_dim} or {min(budgets)})
    if too_wide:
        print(f"  budgets above the {total_dim} features of the general "
              f"representations left out: {too_wide}")

    n_requested = int(config['n_splits'])
    folds = nested_cv_outer_folds(reps, labels, groups, n_requested, SEED,
                                  task_type)
    n_outer = len(folds)
    print(f"  {n_reps} reps, models {models}, budgets {budgets} features, "
          f"{n_outer} outer fold(s) (asked {n_requested})")
    for arm, why in ARMS_NOT_RUN.items():
        print(f"  {arm}: {why}")

    # One partial file per dataset; each finished fold is stored at once and
    # read back on the next attempt (dirty-run bug 12).
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, config,
                     result_format=RESULT_FORMAT))

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        if ck.done(fold=k):
            rec = ck.get(fold=k)
            print(f"  fold {k}: finished in an earlier attempt, read back")
        else:
            print(f"\n  --- outer fold {k + 1}/{n_outer}: {len(tr)} training "
                  f"rows, {len(va)} held out ---")
            rec = _run_fold(
                k, tr, va, reps=reps, labels=labels, groups=groups,
                dataset_name=dataset_name, n_outer=n_outer, models=models,
                factories=factories, budgets=budgets,
                general_names=general_names, metric_name=metric_name,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                task_type=task_type, decided_imp_methods=decided_imp_methods)
            # A fold whose search failed for a model is in the result file but
            # not stored, so the next attempt searches it again.
            if not rec.get('search_failed'):
                ck.put(rec, fold=k)
        per_fold.append(rec)

    results = {
        'result_format': RESULT_FORMAT,
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'models': models,
        'models_without_factory': models_without_factory,
        'budgets': budgets,
        'budgets_left_out_above_pool_width': too_wide,
        'n_reps': n_reps,
        'representations': list(general_names),
        'n_outer_folds': n_outer,
        'n_outer_folds_asked': n_requested,
        'outer_split': {'splitter': 'nested_cv_outer_folds', 'seed': SEED},
        'arms_not_run': dict(ARMS_NOT_RUN),
        # Per model: per_budget[budget][arm] = {'score' (mean held-out score
        # over folds), 'fold_scores', 'spread_over_folds', widths built per
        # fold, the importance method per fold, and for a natural arm the
        # recipe per fold}; base = the model's 2 best base representations per
        # fold, held out. The arms at one budget are depth_1/2/3, uniform_all,
        # uniform_every_rep, natural and natural__general2; the base representations are kept
        # out of per_budget because they are not cut to the budget.
        'per_model': _summarise(per_fold, models, budgets),
        # Everything per fold, as computed (models, base representations,
        # hybrids with their recipes, training-row and held-out scores).
        'per_fold': per_fold,
        'search_failed': [dict(f, fold=r['fold']) for r in per_fold
                          for f in (r.get('search_failed') or [])],
    }

    for m in models:
        print(f"\n  {m}: mean held-out {metric_name} over {n_outer} fold(s)"
              f"{' (one fold, no spread)' if n_outer == 1 else ''}")
        for b in budgets:
            arms = results['per_model'][m]['per_budget'].get(b) or {}
            line = ', '.join(
                f"{name}={a['score']:.4f}" for name, a in arms.items()
                if a.get('score') is not None)
            print(f"    budget {b}: {line or '(nothing scored)'}")
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C18: Diversity vs Depth')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c18_diversity_vs_depth',
                            args.dataset)
