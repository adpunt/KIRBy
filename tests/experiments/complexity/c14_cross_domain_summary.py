#!/usr/bin/env python3
"""C14: Cross-Domain Summary — hybrid against best single representation.

Paper: Complexity
Question: on each dataset, does the best hybrid beat the best single
    representation on held-out rows? Reported per dataset and per model,
    side by side across domains (molecular, antibody, TCGA).
Dependencies: None. c14 runs its own per-fold searches; it reads no other
    experiment's result file.

Aligned 2026-10-05 (handoff HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4):
For each dataset named in --tasks:
  1. Load DATASET_REPS[ds]['general'] only (rep preset ignored). No
     augmentation representations: the decided grid runs the general search
     without augmentation on every dataset (Step 8), and c14 runs no distance
     search (2.3 item 1: distance hybrids only where neighbours are measured).
  2. Models: decided_models(ds, 'general'), the dataset's two (Step 1).
  3. Outer folds from nested_cv_outer_folds with config['n_splits'] (works for
     1: one held-out split, no spread).
  4. Per outer fold: every representation x model fitted on the fold's
     training rows and scored on its held-out rows (the per-representation
     table); then fold_standard_selection on the fold's training rows: per
     model the 2 best base representations and the 2 best hybrids (a second
     hybrid with the same selected features is skipped). Each hybrid is
     rebuilt on the fold's training rows with that fold search's own
     representation scores and scored on the held-out rows.
  5. Per model: per-fold delta of hybrid1 (and hybrid2) against base1, both
     chosen on the training rows, positive = hybrid better; their mean over
     the dataset's folds.

Across datasets: a table, one row per dataset and model, plus a count of rows
where the hybrid is ahead. No average across datasets and no significance
test (owner's rules; removed 2026-10-05).

Each finished fold and each finished dataset is written to
<output>/c14_cross_domain_summary/results.partial.jsonl at once and read back
on the next attempt.

Usage:
    python c14_cross_domain_summary.py --tasks logd,caco2,herg --mode full
    python c14_cross_domain_summary.py --tasks logd,flab_thermo_tresanco,tcga_luad_vs_lusc --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import datetime
from pathlib import Path

import numpy as np

from experiments._shared import (
    load_data, get_domain_metric, get_model_factories,
    save_experiment_results, convert_numpy,
    get_config, decided_config, should_skip_experiment,
    top_hybrids,
    fold_standard_selection, fold_searches,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
)
from experiments.decided_grid import decided_models
from kirby.find_best_hybrid import (_predict_for_metric,
                                    nested_cv_outer_folds,
                                    nested_cv_cache_namespace)
from kirby.scaling import FeatureScaler
from experiments.complexity.complexity_helpers import DATASET_REPS

EXPERIMENT_ID = 'c14_cross_domain_summary'
SEED = 42

# Written into the checkpoint identity and into every result file. Bumped
# 2026-10-05: the whole-dataset-model nested CV version's files are never read
# back as this one's.
SELECTION = 'fold_standard_selection_2026_10_05'

# Recipe fields copied off each selected hybrid into the result file.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')

ONE_FOLD_NOTE = 'one fold, no spread'


# =========================================================================
# Kept 2026-10-05 only because other scripts import them: c33
# (rule_general2_outer, rule_slot_vs_single), opig_scaffold_hybrid
# (_score_blueprint_outer, rule_slot_vs_single), opig_shared_vectors_remedy
# (rule_general2_outer, rule_slot_vs_single). c14 itself no longer calls any
# of the four below; its own selection is fold_standard_selection.
#
# Owner's set (2026-10-02, superseded for c14 by the 2026-10-03 rule):
#   general1  the nested CV's own winner per outer fold
#   general2  the second-ranked configuration of the same inner search,
#             rebuilt on the outer training rows and scored on the outer test
#             fold
# =========================================================================

def _bp_identity(bp):
    """The fields that make two blueprints the same recipe."""
    return tuple(getattr(bp, a, None) for a in (
        'allocation_method', 'importance_method', 'budget',
        'internal_evaluator', 'step_size', 'augmentation_strategy'))


def _score_blueprint_outer(bp, model_name, tr, te, reps, labels, groups, *,
                           augmentations, task_type, metric_fn, lower_is_better,
                           seed, tuned_model=None):
    """One blueprint rebuilt on outer-train rows, scored on the outer test fold.

    The same steps nested_cv_evaluate's own hybrid arm takes
    (find_best_hybrid.py, the 'HYBRID ARM' block): apply_blueprint on the
    training rows with the real model and metric, the same feature selection
    applied to the test rows, augmentation features appended including any
    representation create_hybrid demoted, then _outer_fit_score.
    """
    from kirby.find_best_hybrid import (
        apply_blueprint, _outer_fit_score, get_model, Blueprint, _resolve_tune)
    from kirby.hybrid import apply_feature_selection, apply_augmentation_selection

    tr = np.asarray(tr)
    te = np.asarray(te)
    y_tr, y_te = labels[tr], labels[te]
    train_reps = {k: v[tr] for k, v in reps.items()}
    test_reps = {k: v[te] for k, v in reps.items()}
    train_groups = groups[tr] if groups is not None else None
    train_augs = ({k: v[tr] for k, v in augmentations.items()}
                  if augmentations else None)
    test_augs = ({k: v[te] for k, v in augmentations.items()}
                 if augmentations else None)
    _model = get_model(model_name, task_type, n_samples=len(y_tr))
    X_tr_h, finfo = apply_blueprint(
        train_reps, bp, labels=y_tr, augmentations=train_augs,
        groups=train_groups, model=_model,
        metric_fn=(metric_fn if _model is not None else None),
        lower_is_better=lower_is_better, row_index=tr,
        is_clf=(task_type == 'classification'))
    bp_obj = bp if isinstance(bp, Blueprint) else Blueprint.from_dict(bp)
    X_te_h = apply_feature_selection(
        {k: test_reps[k] for k in bp_obj.reps if k in test_reps}, finfo)
    aug_info = finfo.get('augmentation_info')
    if aug_info:
        _te_augs = dict(test_augs) if test_augs else {}
        for _dn, _di in (finfo.get('demoted_reps') or {}).items():
            if _dn in test_reps:
                _te_augs[_dn] = test_reps[_dn][:, _di]
        if _te_augs:
            aug_te = apply_augmentation_selection(_te_augs, aug_info)
            if aug_te is not None:
                X_te_h = np.hstack([X_te_h, aug_te])
    score, _ = _outer_fit_score(
        X_tr_h, y_tr, X_te_h, y_te, model_name, task_type, metric_fn,
        seed=seed, train_groups=train_groups, test_idx=te,
        test_groups=(groups[te] if groups is not None else None),
        tuned_model=tuned_model, tune_per_fold=_resolve_tune(None))
    return float(score), int(X_tr_h.shape[1])


def rule_general2_outer(ncv, reps, labels, groups, *, models, metric_name,
                        task_type, n_outer_folds, seed=42, augmentations=None,
                        context=''):
    """general2 per model: each outer fold's second-ranked inner configuration,
    scored on that outer fold. Returns {model: {...}}.

    'Second-ranked' is top_hybrids over the inner search's own CV scores on
    the outer training rows, skipping any configuration identical to the
    nested CV's winner for that fold, so general1 and general2 are always two
    different recipes.
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds, _resolve_metric
    metric_fn, lower = _resolve_metric(metric_name)
    folds = nested_cv_outer_folds(reps, labels, groups, n_outer_folds, seed,
                                  task_type)
    inner = list(getattr(ncv, 'inner_searches', None) or [])
    out = {}
    for m in models:
        rec = {'slot': 'general2', 'lineage': 'general', 'tier': 'general',
               'objective': 'predictive',
               'fold_scores': [None] * len(folds),
               'n_features_per_fold': [None] * len(folds),
               'recipes_per_fold': [None] * len(folds), 'missing': []}
        for i, (tr, te) in enumerate(folds):
            s = inner[i] if i < len(inner) else None
            if s is None:
                rec['missing'].append(f'outer fold {i}: the inner search failed')
                continue
            win = (getattr(s, 'best_by_model', None) or {}).get(m)
            if not win:
                rec['missing'].append(
                    f'outer fold {i}: the inner search scored no recipe for {m}')
                continue
            try:
                sel = top_hybrids(s, m, 3, require_full=False,
                                  context=f'{context} fold {i}')
            except Exception as e:
                rec['missing'].append(
                    f'outer fold {i}: ranking raised {type(e).__name__}: {e}')
                continue
            win_id = _bp_identity(win.get('blueprint'))
            pick = next((e for e in sel['selected']
                         if e.get('blueprint') is not None
                         and _bp_identity(e['blueprint']) != win_id), None)
            if pick is None:
                rec['missing'].append(
                    f'outer fold {i}: the inner search ranked '
                    f'{len(sel["selected"])} configuration(s) for {m} and none '
                    f'differs from the winner')
                continue
            bp = pick['blueprint']
            rec['recipes_per_fold'][i] = {
                'label': pick.get('label'), 'budget': pick.get('budget'),
                'inner_score_mean': pick.get('score_mean'),
                'allocation_method': getattr(bp, 'allocation_method', None),
                'importance_method': getattr(bp, 'importance_method', None),
                'internal_evaluator': getattr(bp, 'internal_evaluator', None),
                'augmentation_strategy': getattr(bp, 'augmentation_strategy', None)}
            try:
                sc, width = _score_blueprint_outer(
                    bp, m, tr, te, reps, labels, groups,
                    augmentations=augmentations, task_type=task_type,
                    metric_fn=metric_fn, lower_is_better=lower, seed=seed,
                    tuned_model=(getattr(s, 'tuned_models', None) or {}).get(m))
                rec['fold_scores'][i] = sc if np.isfinite(sc) else None
                rec['n_features_per_fold'][i] = width
            except Exception as e:
                rec['missing'].append(
                    f'outer fold {i}: rebuild/score raised '
                    f'{type(e).__name__}: {e}')
        out[m] = rec
    return out


def rule_slot_vs_single(fold_scores, single_fold_scores, lower_is_better):
    """Mean, spread and the paired per-fold delta against the best single
    representation (positive = the hybrid is better)."""
    fs = [None if v is None or not np.isfinite(v) else float(v)
          for v in (fold_scores or [])]
    vals = [v for v in fs if v is not None]
    per_fold = []
    for f in range(max(len(fs), len(single_fold_scores or []))):
        h = fs[f] if f < len(fs) else None
        b = (single_fold_scores[f]
             if single_fold_scores and f < len(single_fold_scores) else None)
        if h is None or b is None or not np.isfinite(b):
            per_fold.append(None)
            continue
        per_fold.append(float((b - h) if lower_is_better else (h - b)))
    paired = [d for d in per_fold if d is not None]
    return {
        'hybrid_fold_scores': fs,
        'hybrid_mean': float(np.mean(vals)) if vals else None,
        'hybrid_std': float(np.std(vals)) if vals else None,
        'per_fold_delta': per_fold,
        'delta_mean_of_folds': float(np.mean(paired)) if paired else None,
        'n_paired_folds': len(paired),
    }



# =========================================================================
# Per-fold fitting (2026-10-05)
# =========================================================================

def _fit_score(X_train, X_val, y_train, y_val, factory, metric_fn):
    """Fit one model on a fold's training rows, score it on its held-out rows.

    FeatureScaler fitted on the training rows only: each 0/1 feature passes
    through unscaled, every other feature is standardised (handoff 2.3 item
    17; replaced a StandardScaler over every feature, 2026-10-05). The
    matrices are not converted to float64 (item 18). Raises on failure; the
    caller records the error.
    """
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(np.asarray(X_train))
    X_va = scaler.transform(np.asarray(X_val))
    model = factory()
    if model is None:
        raise ValueError('the model factory returned no model (size limit)')
    model.fit(X_tr, y_train)
    return float(metric_fn(y_val, _predict_for_metric(model, X_va, metric_fn)))


def _spread(vals):
    """Mean and spread over folds; one fold says so instead of a spread."""
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': None, 'std': None, 'n_folds': 0}
    out = {'mean': float(np.mean(vals)),
           'std': float(np.std(vals)) if len(vals) > 1 else None,
           'n_folds': len(vals)}
    if len(vals) == 1:
        out['spread_note'] = ONE_FOLD_NOTE
    return out


def _delta(h, b, lower_is_better):
    """Hybrid minus single representation, positive = the hybrid is better."""
    if h is None or b is None or not np.isfinite(h) or not np.isfinite(b):
        return None
    return float((b - h) if lower_is_better else (h - b))


def _search_failures(reps, labels, groups, tr, k, n_splits, task_name,
                     task_type, metric_name, domain):
    """The recorded failures of the search fold_standard_selection just ran.

    Reads the in-process memo with the same arguments fold_standard_selection
    passed, so nothing is searched again.
    """
    try:
        srch = fold_searches(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_splits, dataset_name=task_name,
            task_type=task_type, metric=metric_name, domain=domain, seed=SEED)
        return [{kk: (str(vv) if not isinstance(vv, (int, float, str,
                                                     type(None))) else vv)
                 for kk, vv in f.items()}
                for f in (getattr(srch, 'failures', None) or [])]
    except Exception as e:
        return [{'error': f'could not read the failure record: '
                          f'{type(e).__name__}: {e}'}]


def _evaluate_fold(k, tr, va, *, reps, labels, groups, factories, task_name,
                   task_type, domain, n_splits, metric_name, metric_fn,
                   lower_is_better):
    """One outer fold: the per-representation table, the standard selection,
    and the held-out score of every selected configuration."""
    tr = np.asarray(tr)
    va = np.asarray(va)
    y_tr, y_va = labels[tr], labels[va]
    groups_tr = groups[tr] if groups is not None else None
    tr_reps = {r: X[tr] for r, X in reps.items()}
    va_reps = {r: X[va] for r, X in reps.items()}

    rec = {'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'cache_namespace': nested_cv_cache_namespace(k, n_splits, SEED),
           'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'rep_scores': {}, 'fit_errors': [],
           'search_failures': [], 'shortfall': [], 'models': {}}
    print(f"    fold {k}: {len(tr)} training rows, {len(va)} held-out rows; "
          f"{len(reps)} representations x {list(factories)}, then the "
          f"per-fold search ({rec['cache_namespace']})")

    # Every representation x every decided model on this fold. A description
    # of the pool, not a selection: nothing below is picked from these
    # held-out numbers. The base slots read their held-out score from here.
    for rep in sorted(reps):
        rec['rep_scores'][rep] = {}
        for m, fac in factories.items():
            try:
                s = _fit_score(tr_reps[rep], va_reps[rep], y_tr, y_va, fac,
                               metric_fn)
            except Exception as e:
                s = None
                rec['fit_errors'].append({'rep': rep, 'model': m,
                                          'error': f'{type(e).__name__}: {e}'})
                print(f"    fold {k} {m:8s} {rep:25s} fit failed: "
                      f"{type(e).__name__}: {e}")
            rec['rep_scores'][rep][m] = s

    # The standard selection, on this fold's training rows only.
    try:
        sel = fold_standard_selection(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_splits, dataset_name=task_name,
            models=list(factories), task_type=task_type, metric=metric_name,
            domain=domain, seed=SEED)
    except Exception as e:
        sel = {m: None for m in factories}
        rec['search_failures'].append({
            'dataset': task_name, 'fold': int(k), 'model': None,
            'errors': [{'error': f'{type(e).__name__}: {e}'}]})
        print(f"    fold {k}: selection raised {type(e).__name__}: {e}")

    failed = sorted(m for m, s in sel.items() if s is None)
    if failed and not rec['search_failures']:
        errs = _search_failures(reps, labels, groups, tr, k, n_splits,
                                task_name, task_type, metric_name, domain)
        for m in failed:
            rec['search_failures'].append({
                'dataset': task_name, 'fold': int(k), 'model': m,
                'errors': errs})
            print(f"    fold {k} {m:8s} SEARCH FAILED, slots skipped: "
                  f"{[e.get('error') or e.get('error_type') for e in errs]}")

    for m, s in sel.items():
        if s is None:
            continue
        factory = factories[m]
        bd = s.get('base_detail') or {}
        if len(bd.get('selected') or []) < 2 or len(s['hybrids']) < 2:
            rec['shortfall'].append({
                'dataset': task_name, 'fold': int(k), 'model': m,
                'n_base': len(bd.get('selected') or []),
                'n_hybrids': len(s['hybrids']),
                'reason': bd.get('shortfall_reason')})

        base_rows = []
        for i, e in enumerate(bd.get('selected') or [], 1):
            rep = e['rep']
            base_rows.append({
                'slot': f'base{i}', 'rep': rep,
                # Mean over the search's inner folds, inside this outer
                # fold's training rows.
                'training_rows_score': e.get('score'),
                'heldout': rec['rep_scores'].get(rep, {}).get(m)})

        hyb_rows = []
        for i, e in enumerate(s['hybrids'], 1):
            bp = e['blueprint']
            row = {'slot': f'hybrid{i}',
                   **{f: e.get(f) for f in HYBRID_FIELDS},
                   'aug_condition': e.get('aug_condition'),
                   'training_rows_score': e.get('score_mean'),
                   'heldout': None}
            try:
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp, groups=groups_tr,
                    # This fold's search measured these on this fold's
                    # training rows (dirty-run bug 1).
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factory(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                row['n_features_built'] = int(np.asarray(X_tr_h).shape[1])
                row['heldout'] = _fit_score(X_tr_h, X_va_h, y_tr, y_va,
                                            factory, metric_fn)
            except Exception as ex:
                row['error'] = f'{type(ex).__name__}: {ex}'
                print(f"    fold {k} {m:8s} hybrid{i} failed: {row['error']}")
            hyb_rows.append(row)

        b1 = base_rows[0]['heldout'] if base_rows else None
        deltas = {f'{h["slot"]}_vs_base1': _delta(h['heldout'], b1,
                                                  lower_is_better)
                  for h in hyb_rows}
        rec['models'][m] = {
            'base': base_rows, 'hybrids': hyb_rows,
            'hybrids_skipped_identical_features': [
                x.get('label') for x in
                (s.get('hybrids_skipped_identical_features') or [])],
            'delta': deltas}
        _fmt = lambda v: 'n/a' if v is None else f'{v:.4f}'
        print(f"    fold {k} {m:8s} "
              + '  '.join(f"{r['slot']}={_fmt(r['heldout'])}"
                          for r in base_rows + hyb_rows))
    return rec


# =========================================================================
# Per-task evaluation
# =========================================================================

def _model_summary(per_fold, m, lower_is_better):
    """One model's slots over the folds of one dataset."""
    slots = {}
    deltas = {'hybrid1_vs_base1': [], 'hybrid2_vs_base1': []}
    for rec in per_fold:
        mr = (rec.get('models') or {}).get(m)
        for key in deltas:
            deltas[key].append(None if mr is None
                               else (mr.get('delta') or {}).get(key))
        if mr is None:
            continue
        for row in mr['base'] + mr['hybrids']:
            d = slots.setdefault(row['slot'], {'per_fold': []})
            d['per_fold'].append({
                'fold': rec['fold'],
                'name': row.get('rep') or row.get('label'),
                'training_rows_score': row.get('training_rows_score'),
                'heldout': row.get('heldout')})
    for d in slots.values():
        d['heldout'] = _spread([p['heldout'] for p in d['per_fold']])
    out = {'slots': slots}
    for key, vals in deltas.items():
        paired = [v for v in vals if v is not None]
        out[key] = {
            'per_fold': vals,
            'delta_mean_of_folds': (float(np.mean(paired)) if paired
                                    else None),
            'n_paired_folds': len(paired),
        }
        if len(paired) == 1:
            out[key]['spread_note'] = ONE_FOLD_NOTE
    d1 = out['hybrid1_vs_base1']['delta_mean_of_folds']
    b1 = (slots.get('base1') or {}).get('heldout', {}).get('mean')
    out['hybrid_wins'] = bool(d1 is not None and d1 > 0)
    # Floor on the denominator so a base score near 0 (MCC ~ 0.02 on hard
    # genomic tasks) does not turn a small gain into hundreds of percent.
    out['pct_improvement'] = (float(d1 / max(abs(b1), 0.05) * 100)
                              if d1 is not None and b1 is not None else None)
    return out


def _evaluate_task(task_name, config, ck=None):
    """The standard selection per outer fold for one dataset.

    Returns a dict with task, domain, metric_name, lower_is_better, models,
    representations, per_fold (one record per outer fold) and by_model
    (per-slot held-out mean and spread over folds, the per-fold deltas of
    each hybrid against base1, their mean over folds). Returns a dict with
    'skipped' when should_skip_experiment says so.
    """
    print(f"\n  --- Task: {task_name} ---")

    rep_names = list((DATASET_REPS.get(task_name) or {}).get('general') or [])
    if not rep_names:
        raise ValueError(f"No DATASET_REPS['{task_name}']['general'] list.")
    # Local compute cut (fewer representations, never fewer molecules), the
    # same call complexity_helpers.run_all makes. Absent on the cluster.
    try:
        from experiments._local_grid import cut_pool
        rep_names = cut_pool(task_name, rep_names)
    except ImportError:
        pass

    config = decided_config(dict(config), task_name)
    n_splits = int(config['n_splits'])

    data = load_data(task_name, rep_preset='all', rep_names=rep_names)
    reps = data['reps']
    for rname, X in list(reps.items()):
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            print(f"    WARNING: {rname} has {nan_count} NaN values, imputing 0")
            reps[rname] = np.nan_to_num(X, nan=0.0)
    labels = np.asarray(data['labels'])
    groups = (np.asarray(data['groups']) if data.get('groups') is not None
              else None)
    domain = data.get('domain', 'molecular')
    task_type = data['task_type']
    n_samples = len(labels)

    skip = should_skip_experiment('c14', data)
    if skip:
        print(f"    skipped: {skip.get('reason')}")
        return dict(skip, task=task_name, domain=domain)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    models = [m for m in decided_models(task_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=n_samples)
    models_dropped = [{'model': m, 'reason': 'size limit in DEFAULT_MODELS '
                                             f'(n_samples={n_samples})'}
                      for m in models if m not in factories]

    print(f"    Domain: {domain}, Metric: {metric_name} "
          f"({'lower better' if lower_is_better else 'higher better'})")
    print(f"    {n_samples} samples, {len(reps)} representations, "
          f"{n_splits} outer fold(s), models {list(factories)}")
    if DATASET_REPS.get(task_name, {}).get('augmentation'):
        print("    Augmentation representations not loaded: the decided "
              "general search runs without augmentation on every dataset "
              "(Step 8), and c14 runs no distance search.")

    folds = nested_cv_outer_folds(reps, labels, groups, n_splits, SEED,
                                  task_type)
    per_fold = []
    for k, (tr, va) in enumerate(folds):
        ck_key = dict(task=task_name, fold=int(k))
        if ck is not None and ck.done(**ck_key):
            per_fold.append(ck.get(**ck_key))
            print(f"    fold {k}: finished in an earlier attempt, read back")
            continue
        rec = _evaluate_fold(
            k, tr, va, reps=reps, labels=labels, groups=groups,
            factories=factories, task_name=task_name, task_type=task_type,
            domain=domain, n_splits=n_splits, metric_name=metric_name,
            metric_fn=metric_fn, lower_is_better=lower_is_better)
        per_fold.append(rec)
        # A fold whose search failed is in the result file but not stored, so
        # the next attempt searches it again.
        if ck is not None and not rec['search_failures']:
            ck.put(convert_numpy(rec), **ck_key)

    by_model = {m: _model_summary(per_fold, m, lower_is_better)
                for m in factories}
    for m, s in by_model.items():
        d = s['hybrid1_vs_base1']
        b = (s['slots'].get('base1') or {}).get('heldout', {})
        h = (s['slots'].get('hybrid1') or {}).get('heldout', {})
        _f = lambda v: 'n/a' if v is None else f'{v:.4f}'
        print(f"    {m}: base1 {metric_name} {_f(b.get('mean'))}, hybrid1 "
              f"{_f(h.get('mean'))}, hybrid1 - base1 per fold "
              f"{[None if v is None else round(v, 4) for v in d['per_fold']]}")

    return {
        'task': task_name,
        'domain': domain,
        'task_type': task_type,
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'selection': SELECTION,
        'selection_scope': 'fold_train_rows',
        'n_samples': n_samples,
        'n_folds_configured': n_splits,
        'n_folds': len(folds),
        'representations': sorted(reps),
        'models': list(factories),
        'models_dropped': models_dropped,
        'per_fold': per_fold,
        'by_model': by_model,
        'search_failures': [f for r in per_fold
                            for f in r.get('search_failures', [])],
        'delta_basis': ('per outer fold: held-out score of hybrid1 (or '
                        'hybrid2) minus held-out score of base1, both picked '
                        "on that fold's training rows; positive = hybrid "
                        'better; delta_mean_of_folds = mean over this '
                        "dataset's folds"),
    }


# =========================================================================
# Aggregation
# =========================================================================

def _aggregate(task_results):
    """One row per dataset and model, and a count of rows where hybrid1 is
    ahead of base1.

    2026-10-05: the mean, median and spread of the delta and of the percent
    improvement across datasets, the per-domain means, the per-domain mean
    absolute delta ('complexity_matching_strength') and the Wilcoxon tests
    (across datasets and per domain) were removed. The owner's rules: no
    average across datasets, no significance tests.
    """
    print("\n" + "=" * 70)
    print("  ACROSS DATASETS (one row per dataset and model; no averages)")
    print("=" * 70)

    rows = []
    for r in task_results:
        for m, s in (r.get('by_model') or {}).items():
            b = (s['slots'].get('base1') or {}).get('heldout', {})
            h1 = (s['slots'].get('hybrid1') or {}).get('heldout', {})
            h2 = (s['slots'].get('hybrid2') or {}).get('heldout', {})
            rows.append({
                'task': r['task'], 'domain': r['domain'], 'model': m,
                'metric_name': r['metric_name'],
                'lower_is_better': r['lower_is_better'],
                'n_folds': r['n_folds'],
                'base1_heldout_mean': b.get('mean'),
                'hybrid1_heldout_mean': h1.get('mean'),
                'hybrid2_heldout_mean': h2.get('mean'),
                'hybrid1_vs_base1': s['hybrid1_vs_base1']['delta_mean_of_folds'],
                'hybrid2_vs_base1': s['hybrid2_vs_base1']['delta_mean_of_folds'],
                'n_paired_folds': s['hybrid1_vs_base1']['n_paired_folds'],
                'pct_improvement': s['pct_improvement'],
                'hybrid_wins': s['hybrid_wins'],
            })

    _f = lambda v, sgn=False: ('n/a' if v is None
                               else (f'{v:+.4f}' if sgn else f'{v:.4f}'))
    print(f"  {'Dataset':<28} {'Model':<9} {'Metric':<9} {'Folds':>5} "
          f"{'Base1':>9} {'Hybrid1':>9} {'H1-B1':>9} {'H2-B1':>9}")
    for x in rows:
        print(f"  {x['task']:<28} {x['model']:<9} {x['metric_name']:<9} "
              f"{x['n_paired_folds']:>5} {_f(x['base1_heldout_mean']):>9} "
              f"{_f(x['hybrid1_heldout_mean']):>9} "
              f"{_f(x['hybrid1_vs_base1'], True):>9} "
              f"{_f(x['hybrid2_vs_base1'], True):>9}")
    print("  Base1 and Hybrid1 are held-out means over the dataset's outer "
          "folds, in the dataset's own metric. H1-B1 is the fold-by-fold "
          "difference averaged over that dataset's folds; positive = hybrid "
          "better.")

    scored = [x for x in rows if x['hybrid1_vs_base1'] is not None]
    wins = sum(1 for x in scored if x['hybrid_wins'])
    per_domain = {}
    for x in scored:
        d = per_domain.setdefault(x['domain'], {
            'n_rows': 0, 'rows_hybrid1_ahead': 0, 'tasks': []})
        d['n_rows'] += 1
        d['rows_hybrid1_ahead'] += int(x['hybrid_wins'])
        if x['task'] not in d['tasks']:
            d['tasks'].append(x['task'])
    print(f"\n  hybrid1 ahead of base1 in {wins} of {len(scored)} "
          f"(dataset, model) rows")
    for dom, d in sorted(per_domain.items()):
        print(f"    {dom}: {d['rows_hybrid1_ahead']} of {d['n_rows']} rows "
              f"({len(d['tasks'])} datasets)")

    return {
        'rows': rows,
        'row': 'one dataset x one decided model',
        'counts': {'n_rows': len(scored), 'rows_hybrid1_ahead': wins},
        'per_domain_counts': per_domain,
    }


# =========================================================================
# Main entry
# =========================================================================

def run(tasks, config, output_dir, ck=None):
    """Run c14 over every task and lay the results side by side.

    Args:
        tasks: list of dataset names.
        config: from get_config; only 'n_splits' is read here, the search
            grid of each dataset comes from decided_config.
        output_dir: base output directory.
        ck: checkpoint (experiments._checkpoint) or None.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C14: CROSS-DOMAIN SUMMARY")
    print(f"  Tasks: {tasks}")
    print(f"  Outer folds: {config['n_splits']}")
    print("=" * 70)

    task_results, skipped_tasks, failed_tasks = [], [], []
    for task_name in tasks:
        try:
            if ck is not None and ck.done(task=task_name):
                result = ck.get(task=task_name)
                print(f"\n  {task_name}: finished in an earlier attempt, "
                      f"read back")
            else:
                result = _evaluate_task(task_name, config, ck=ck)
                if (result is not None and not result.get('skipped')
                        and not result.get('search_failures')
                        and ck is not None):
                    ck.put(convert_numpy(result), task=task_name)
            if result is None:
                failed_tasks.append({'task': task_name,
                                     'reason': 'no result for this task'})
            elif result.get('skipped'):
                skipped_tasks.append(result)
            else:
                task_results.append(result)
                if result.get('search_failures'):
                    failed_tasks.append({
                        'task': task_name,
                        'reason': (f"{len(result['search_failures'])} "
                                   f"per-fold search failure(s); folds "
                                   f"retried on the next attempt")})
        except Exception as e:
            failed_tasks.append({'task': task_name,
                                 'reason': f'{type(e).__name__}: {e}'})
            print(f"\n  FAILED: {task_name} — {e}")
            import traceback
            traceback.print_exc()

    base = {
        'experiment': EXPERIMENT_ID,
        'timestamp': datetime.datetime.now().isoformat(),
        'selection': SELECTION,
        'selection_scope': 'fold_train_rows',
        'config': {'tasks': tasks, 'n_splits': config['n_splits']},
        'failed_tasks': failed_tasks,
        'skipped_tasks': convert_numpy(skipped_tasks),
    }
    if not task_results:
        # Carry an error key so a saved file from this run cannot be read as
        # a completed run.
        print("\nNo tasks completed successfully.")
        return dict(base, error=f'none of the {len(tasks)} requested task(s) '
                                f'produced a result',
                    per_task=[], aggregate={})

    aggregate = _aggregate(task_results)
    return dict(base, per_task=convert_numpy(task_results),
                aggregate=convert_numpy(aggregate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C14: Cross-domain summary — hybrid vs best single rep')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of dataset names '
                             '(e.g., "logd,caco2,herg,flab_thermo_tresanco,'
                             'tcga_luad_vs_lusc")')
    parser.add_argument('--reps', type=str, default='all',
                        help="Ignored since 2026-10-05: every dataset uses "
                             "DATASET_REPS[ds]['general']. Kept so old "
                             "command lines still parse.")
    parser.add_argument('--mode', choices=['intermediary', 'full'],
                        default='intermediary',
                        help='intermediary (1 split) or full (5-fold CV)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore finished folds and datasets stored by '
                             'an earlier attempt')
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    if not tasks:
        print("ERROR: --tasks must specify at least one dataset name")
        sys.exit(1)
    if args.reps != 'all':
        print(f"  --reps {args.reps} ignored: every dataset uses its "
              f"DATASET_REPS 'general' list (2026-10-05).")
    config = get_config(args)

    from experiments._checkpoint import script_checkpoint
    _out_json = Path(args.output) / EXPERIMENT_ID / 'results.json'
    _out_json.parent.mkdir(parents=True, exist_ok=True)
    # 'selection' in the identity: a partial file from the nested-CV version
    # (or any earlier one) is never read back as this one's.
    _ck = script_checkpoint(EXPERIMENT_ID, str(_out_json), args,
                            exclude=('tasks', 'output', 'reps'),
                            selection=SELECTION)

    results = run(tasks, config, args.output, ck=_ck)
    save_experiment_results(results, args.output, EXPERIMENT_ID)
