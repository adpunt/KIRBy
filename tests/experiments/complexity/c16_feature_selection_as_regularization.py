#!/usr/bin/env python3
"""C16: Feature Selection as Regularization — the hybrid's budget subsumes model-level regularization.

Paper: Complexity
Core thesis: Selecting features via the hybrid is a form of inductive bias equivalent
    to explicit regularization (max_depth, C). At the optimal hybrid budget,
    model regularization becomes irrelevant because feature selection already controls
    the bias-variance tradeoff.

Aligned 2026-10-05 (handoff HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4, chat 3):
  - Families are the dataset's decided models (DECISIONS.md Step 1) that have a
    regularisation sweep in this file: 'rf' (RF family, max_depth) and 'svm'
    (SVM family, C). Each family's estimator is the search's own model
    (find_best_hybrid.get_model) with one parameter changed, so the forest is
    the search's forest (max_features='sqrt', default depth None) and svm
    regression is the search's label-standardising SVR. A decided model with no
    sweep here (dnn, lightgbm, xgboost, gp) is named in
    results['decided_models_without_family'] and not run.
  - Ridge and LogisticRegression are gone (ridge is removed everywhere), and
    with them the old "equivalent L2 alpha" analysis.
  - Per outer fold (kirby.find_best_hybrid.nested_cv_outer_folds,
    config['n_splits'] folds): one search on that fold's training rows on the
    decided grid, pruning off (_shared.fold_standard_selection). At each
    heatmap budget, each family's model gets its 2 best hybrids at that budget
    (a second hybrid with the same selected features is skipped). Arm '<family>'
    is the first of the two, '<family>__hybrid2' the second. Nothing is read
    from a whole-dataset search.
  - The same per-fold search's 2 best base representations for each family's
    model are fitted under every regularisation setting at their own full
    width (2026-10-05; c16 is not one of the handoff 2.5 exceptions, so it
    gets the standard 2 base + 2 hybrids). They have no budget, so they sit
    in results['base_representations'], not in the heatmap.
  - Heatmap budgets: the dataset's decided budgets up to the pool width, plus
    capped_full = min(pool width, 2 x rows, 2048) features. capped_full is not
    a searched budget, so its hybrids are the fold's 2 best over every budget,
    rebuilt at capped_full (recorded as recipe_budget_relaxed).

Two analyses:
  1. Heatmap: held-out score over (regularization setting x hybrid budget) per family
  2. Sensitivity: std(score across regularization) at each budget
  plus the per-family comparison of the best small-budget hybrid at default
  regularisation against the capped_full hybrid at its best setting.

Usage:
    python c16_feature_selection_as_regularization.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection,
    fold_reselection,
)
from experiments.decided_grid import decided_models
from kirby.find_best_hybrid import (_predict_for_metric, get_model,
                                    nested_cv_outer_folds)
from kirby.scaling import FeatureScaler


# Seed of the outer folds and of the per-fold searches' cache namespace. The
# same 42 every other per-fold experiment uses, so this file reads the same
# stored per-fold searches.
SEED = 42

# The identity of this file's per-fold records. A partial file written under
# any other identity -- every version before the 2026-10-05 alignment, which
# read whole-dataset recipes -- is moved aside, never read back.
C16_RESULT_IDENTITY = 'c16_per_fold_standard_selection_base_reps_2026-10-05'


# =========================================================================
# WHERE THE WALL CLOCK GOES
# =========================================================================
# Every line c16 prints about its own progress carries seconds since run()
# started. Added 2026-09-25 after a local run sat for 25 minutes between the
# experiment banner and its first heatmap row with nothing printed, so there
# was no way to tell a slow recipe lookup from a slow cell.
_C16_T0 = [None]

# =========================================================================
# ONE QUANTITY EVERY CELL RECOMPUTES AND THAT DOES NOT DEPEND ON THE CELL
# =========================================================================
# Per-feature importance over the representations. create_hybrid's own comment
# (_shared.py, where precomputed_importance is forwarded) says it depends on
# the fold and the importance method and on NOTHING else -- not the
# allocation, not the downstream model, not the budget. Every (budget,
# regularisation setting) cell of a fold can read one vector per importance
# method. f31's verification block (f31_internal_evaluator_tuning.py:548) is
# the existing check that precomputed_importance gives the identical
# allocation.
#
# Default: ON for a local run (KIRBY_LOCAL_GRID set by
# /Volumes/seagate/kirby_localrun/env.sh), OFF everywhere else.
#
# 2026-10-05: the second reuse this block used to describe (each regularisation
# setting's own per-representation scores for the budget split) is gone. Every
# rebuild now passes the per-fold search's own representation scores
# (dirty-run bug 1), so there is nothing left to cache there.
_C16_REUSE_FOLD_WORK = os.environ.get(
    'KIRBY_C16_REUSE_FOLD_WORK',
    '1' if os.environ.get('KIRBY_LOCAL_GRID') else '0') == '1'


def _c16_t():
    if _C16_T0[0] is None:
        _C16_T0[0] = time.time()
    return time.time() - _C16_T0[0]


# =========================================================================
# MODEL FAMILIES WITH REGULARIZATION AXES
# =========================================================================

# Family -> the decided-model name it is (DECISIONS.md Step 1). A family runs
# on a dataset only when its model is one of that dataset's two.
_FAMILY_MODEL = {
    'RF': 'rf',
    'SVM': 'svm',
}

# The regularisation setting each family's "default regularisation" column is
# read at, stated as a VALUE and looked up in that family's sweep list. It used
# to be read as variants[len(variants) // 2], a position.
#
# 2026-10-05, dirty-run bug "rf depth 10": RF's default was max_depth=10 while
# the search's rf has no depth limit (find_best_hybrid DEFAULT_MODELS['rf']).
# The default column is now the search's own setting, None. SVM's C=1.0 is
# already the search's svm default.
_FAMILY_DEFAULT_PARAM = {
    'SVM': 1.0,       # C, inverse regularisation
    'RF': None,       # max_depth; None = no limit, the search's rf
}


def _default_param_value(fam_name, variants):
    """This family's default regularisation setting, found by value.

    Raises when the value is absent from the sweep, so editing a sweep list
    cannot silently move the column onto a different setting.
    """
    want = _FAMILY_DEFAULT_PARAM.get(fam_name, 'missing')
    values = [pval for _, pval, _ in variants]
    if fam_name not in _FAMILY_DEFAULT_PARAM:
        raise KeyError(
            f"c16: no default regularisation setting recorded for family "
            f"{fam_name!r}; add one to _FAMILY_DEFAULT_PARAM.")
    for pval in values:
        if pval == want:
            return pval
    raise ValueError(
        f"c16: {fam_name}'s default {variants[0][0]}={want} is not in its "
        f"sweep {values}; the default-regularisation column would silently "
        f"move to another setting.")


def _search_model_factory(model_name, task_type, param, value):
    """The search's own estimator for `model_name`, with one parameter set.

    2026-10-05 (dirty-run bug "rf differs from the search's"): the families used
    to be hand-built -- RandomForest without max_features='sqrt', and a plain
    SVR on unscaled labels. They are now find_best_hybrid.get_model's
    estimators, the ones the per-fold search scored, with only the swept
    parameter changed.
    """
    def _make():
        m = get_model(model_name, task_type)
        m.set_params(**{param: value})
        return m
    return _make


def _get_regularization_families(task_type='regression'):
    """Return {family: [(param_name, param_value, factory_fn), ...]}.

    Same sweep values as before 2026-10-05, including the delegated grid-edge
    extension (results audit C16d: SVM C=100 above the old highest setting,
    because the tuned C sat at that edge). Classification gets the search's
    RandomForestClassifier and SVC(probability=True).
    """
    return {
        'RF': [('max_depth', d,
                _search_model_factory('rf', task_type, 'max_depth', d))
               for d in (3, 5, 10, None)],
        'SVM': [('C', c, _search_model_factory('svm', task_type, 'C', c))
                for c in (0.01, 0.1, 1.0, 10.0, 100.0)],
    }


# =========================================================================
# CORE EVALUATION
# =========================================================================

def _score_prediction(model, X_val, y_val, metric_fn, task_type):
    """Score predictions with metric-appropriate format (see c18 for rationale)."""
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 2:
                return metric_fn(y_val, probs[:, 1])
        except Exception:
            pass
    return metric_fn(y_val, _predict_for_metric(model, X_val, metric_fn))


def _fit_cell(X_train, X_val, y_train, y_val, factory, metric_fn, task_type):
    """Fit one regularisation setting on a built hybrid; (held-out, training) score.

    Scaled with FeatureScaler (2026-10-05, handoff 2.3 item 17): 0/1 features
    pass through, every other feature is standardised on the training rows.
    The matrices are not converted to float64 (item 18).
    """
    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    model = factory()
    model.fit(X_tr_s, y_train)
    return (float(_score_prediction(model, X_va_s, y_val, metric_fn, task_type)),
            float(_score_prediction(model, X_tr_s, y_train, metric_fn,
                                    task_type)))


def _hybrid_fields(e):
    """The recipe fields of one fold_standard_selection hybrid entry."""
    return {f: e.get(f) for f in ('label', 'allocation_method',
                                  'importance_method', 'budget',
                                  'internal_evaluator', 'rep_count',
                                  'aug_condition')}


def _run_fold(k, train_idx, val_idx, *, reps, labels, groups, augmentations,
              n_folds, dataset_name, families, budgets, searched_budgets,
              total_dim, metric_name, metric_fn, lower_is_better, task_type):
    """Everything c16 computes on one outer fold, as one plain record.

    Selection, per heatmap budget: fold_standard_selection on this fold's
    training rows (decided grid, pruning off) gives each family's model its 2
    best hybrids at that budget, with distinct selected features. At a budget
    the search did not run (capped_full) the 2 best over every budget are
    taken and rebuilt at that budget; 'recipe_budget_relaxed' says so.

    Each hybrid is rebuilt once per regularisation setting on this fold's
    training rows with build_fold_hybrid, with the per-fold search's own
    representation scores (dirty-run bug 1), and the setting's estimator is
    fitted and scored on the held-out rows.
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    tr_reps = {r: X[train_idx] for r, X in reps.items()}
    va_reps = {r: X[val_idx] for r, X in reps.items()}
    y_tr, y_va = labels[train_idx], labels[val_idx]
    g_tr = groups[train_idx] if groups is not None else None
    tr_augs = ({a: X[train_idx] for a, X in augmentations.items()}
               if augmentations else None)
    va_augs = ({a: X[val_idx] for a, X in augmentations.items()}
               if augmentations else None)
    fam_models = sorted({_FAMILY_MODEL[f] for f in families})

    rec = {
        'fold': int(k),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        # {str(budget): {model: {'base': [...], 'hybrids': [...], ...}}}
        'selection': {},
        # {arm: {str(budget): {str(pval): {'val', 'train', 'width'}}}}
        'cells': {},
        # {family: {'base<i>': {'rep': name, 'settings':
        #   {str(pval): {'val', 'train', 'width'}}}}} -- 2026-10-05
        'base_cells': {},
        'errors': [],
    }

    importance_by_imeth = {} if _C16_REUSE_FOLD_WORK else None

    for budget in budgets:
        _t_sel = time.time()
        sel_budget = budget if budget in searched_budgets else None
        try:
            sel = fold_standard_selection(
                reps, labels, groups, train_idx=train_idx, fold_i=k,
                n_outer_folds=n_folds, dataset_name=dataset_name,
                models=fam_models, augmentations=augmentations,
                task_type=task_type, metric=metric_name, seed=SEED,
                budget=sel_budget)
        except Exception as e:
            rec['errors'].append({'budget': int(budget), 'stage': 'selection',
                                  'error': f'{type(e).__name__}: {e}'})
            print(f"    fold {k} budget={budget}: selection failed: {e}")
            continue
        sel_rec = {}
        for m in fam_models:
            s = sel.get(m)
            if s is None:
                sel_rec[m] = None
                rec['errors'].append({
                    'budget': int(budget), 'model': m, 'stage': 'search',
                    'error': 'the per-fold search failed for this model'})
                continue
            bd = s.get('base_detail') or {}
            sel_rec[m] = {
                # Fitted once per fold under every regularisation setting
                # (rec['base_cells']); the base ranking does not depend on the
                # budget.
                'base': list(s.get('base') or []),
                'base_training_rows_score': [
                    e.get('score') for e in (bd.get('selected') or [])],
                'hybrids': [dict(_hybrid_fields(e),
                                 training_rows_score=e.get('score_mean'))
                            for e in s['hybrids']],
                'hybrids_skipped_identical_features': [
                    x.get('label') for x in
                    (s.get('hybrids_skipped_identical_features') or [])],
                'recipe_budget_relaxed': sel_budget is None,
            }
        rec['selection'][str(budget)] = sel_rec
        print(f"    [c16 {_c16_t():.0f}s] fold {k} budget={budget}: "
              f"selection in {time.time() - _t_sel:.0f}s", flush=True)

        for fam_name, variants in families.items():
            m = _FAMILY_MODEL[fam_name]
            s = sel.get(m)
            if s is None:
                continue
            # The 2 base representations, once per fold (2026-10-05): each
            # fitted at its own full width under every regularisation setting,
            # on this fold's training rows, scored on the held-out rows.
            if fam_name not in rec['base_cells']:
                rec['base_cells'][fam_name] = _fit_base_reps(
                    k, fam_name, variants, list(s.get('base') or [])[:2],
                    tr_reps, va_reps, y_tr, y_va, metric_fn, task_type)
            for slot_i, e in enumerate(s['hybrids'][:2], 1):
                arm = _arm_name(fam_name, slot_i)
                row = rec['cells'].setdefault(arm, {}).setdefault(
                    str(budget), {})
                # A hybrid this wide is the concatenation of every
                # representation; concat-all baselines are not allowed
                # (2026-05-12), so the cell is left empty.
                if budget >= total_dim:
                    continue
                bp = e['blueprint']
                aug_on = (e.get('aug_condition') or 'none') != 'none'
                _pre_imp = None
                if importance_by_imeth is not None:
                    _imeth = getattr(bp, 'importance_method', None)
                    if _imeth not in importance_by_imeth:
                        # Computed on the rows create_hybrid itself ranks
                        # features on: the training side of its internal
                        # holdout carve of this fold's training rows.
                        from kirby.hybrid import (compute_feature_importance,
                                                  internal_holdout_split)
                        _inner_tr, _ = internal_holdout_split(
                            len(y_tr), y_tr, g_tr)
                        importance_by_imeth[_imeth] = compute_feature_importance(
                            {r: X[_inner_tr] for r, X in tr_reps.items()},
                            y_tr[_inner_tr], method=_imeth)
                    _pre_imp = importance_by_imeth[_imeth]
                for pname, pval, factory in variants:
                    _t_cell = time.time()
                    try:
                        # budget_override: the row is built at the budget it is
                        # labelled with. Equal to the recipe's own budget except
                        # at capped_full.
                        X_tr_h, X_va_h, _fi = build_fold_hybrid(
                            tr_reps, va_reps, y_tr, bp,
                            augmentations_train=tr_augs if aug_on else None,
                            augmentations_val=va_augs if aug_on else None,
                            groups=g_tr,
                            budget_override=budget,
                            precomputed_importance=_pre_imp,
                            # This fold's search measured these on this fold's
                            # training rows (dirty-run bug 1, 2026-10-05).
                            precomputed_alloc_scores=getattr(bp, 'rep_scores',
                                                             None),
                            **fold_hybrid_scoring_kwargs(
                                bp, model=factory(), metric_fn=metric_fn,
                                lower_is_better=lower_is_better))
                        val_s, train_s = _fit_cell(
                            X_tr_h, X_va_h, y_tr, y_va, factory, metric_fn,
                            task_type)
                        row[str(pval)] = {
                            'val': val_s, 'train': train_s,
                            # 2026-10-05: features actually built. Below
                            # `budget` when the engine fills less.
                            'width': int(np.asarray(X_tr_h).shape[1])}
                    except Exception as ex:
                        # Recorded, never silent (audit: a failed build used
                        # to be skipped with no record of why).
                        row[str(pval)] = {'val': None, 'train': None,
                                          'width': None,
                                          'error': f'{type(ex).__name__}: {ex}'}
                        print(f"      fold {k} {arm} budget={budget} "
                              f"{pname}={pval}: failed: {ex}")
                    print(f"    [c16 {_c16_t():.0f}s] fold {k} cell {arm} "
                          f"budget={budget} {pname}={pval}: "
                          f"{time.time() - _t_cell:.0f}s", flush=True)
    return rec


def _fit_base_reps(k, fam_name, variants, base_names, tr_reps, va_reps,
                   y_tr, y_va, metric_fn, task_type):
    """Fit one family's 2 base representations under every setting.

    Added 2026-10-05: the standard selection's 2 base representations (handoff
    2.4 step 5). Each is the representation's full feature set, scaled by
    FeatureScaler inside _fit_cell, no budget. A failed fit is recorded with
    its error.
    """
    out = {}
    for bi, rep_name in enumerate(base_names, 1):
        slot = {'rep': rep_name, 'settings': {}}
        out[f'base{bi}'] = slot
        if rep_name not in tr_reps:
            slot['error'] = (f'{rep_name!r} is not among the loaded '
                             f'representations')
            continue
        for pname, pval, factory in variants:
            _t_cell = time.time()
            try:
                val_s, train_s = _fit_cell(
                    tr_reps[rep_name], va_reps[rep_name], y_tr, y_va,
                    factory, metric_fn, task_type)
                slot['settings'][str(pval)] = {
                    'val': val_s, 'train': train_s,
                    'width': int(np.asarray(tr_reps[rep_name]).shape[1])}
            except Exception as ex:
                slot['settings'][str(pval)] = {
                    'val': None, 'train': None, 'width': None,
                    'error': f'{type(ex).__name__}: {ex}'}
                print(f"      fold {k} {fam_name} base{bi} ({rep_name}) "
                      f"{pname}={pval}: failed: {ex}")
            print(f"    [c16 {_c16_t():.0f}s] fold {k} base {fam_name} "
                  f"base{bi}={rep_name} {pname}={pval}: "
                  f"{time.time() - _t_cell:.0f}s", flush=True)
    return out


def _fold_failed(rec):
    """True when any unit of this fold failed: a selection, a search, a base
    representation fit or a hybrid cell. Such a fold is not stored, so the
    next attempt computes it again (2026-10-05; Checkpoint.put only filters a
    top-level 'error' key, which this record does not have)."""
    if rec.get('errors'):
        return True
    for row in (rec.get('cells') or {}).values():
        for by_p in row.values():
            if any(c and c.get('error') for c in by_p.values()):
                return True
    for slots in (rec.get('base_cells') or {}).values():
        for slot in slots.values():
            if slot.get('error') or any(
                    c and c.get('error')
                    for c in (slot.get('settings') or {}).values()):
                return True
    return False


def _arm_name(fam, slot_i):
    """Arm key: the family name for its first hybrid (the key every reader of
    results['families'] already uses), '<family>__hybrid2' for the second."""
    return fam if slot_i == 1 else f'{fam}__hybrid{slot_i}'


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C16: Feature selection as regularization."""
    _C16_T0[0] = time.time()
    print("\n" + "=" * 70)
    print("EXPERIMENT C16: FEATURE SELECTION AS REGULARIZATION")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = (np.asarray(data['groups']) if data.get('groups') is not None
              else None)
    # The one fold setting (handoff 2.4 step 14). 1 = a single held-out split.
    n_folds = int(config['n_splits'])

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    augmentations = data.get('augmentations') or None
    total_dim = sum(v.shape[1] for v in reps.values())
    n_samples = len(labels)
    # 2026-10-06 (Adelaide): no all-features representation anywhere; this
    # column stays the capped hybrid below. A short-lived all-features version
    # was reverted.
    # capped_full = the budget of the "full-dim baseline" column. It is NOT all
    # features: it is a hybrid built at min(total_dim, 2 x rows, 2048)
    # features, and a cell at or above total_dim is skipped (no concat-all
    # baselines), so it is always a selected subset (label corrected
    # 2026-10-05). Bounded by 2n to keep SVM tractable, plus a 2048 hard cap.
    capped_full = min(total_dim, 2 * n_samples, 2048)

    # The budgets the per-fold search runs: the dataset's decided ones
    # (DECISIONS.md Step 2, set on config by decided_config). Until 2026-10-05
    # the heatmap used all eight FULL_BUDGETS read off a whole-dataset search.
    searched_budgets = [int(b) for b in config['budgets']]
    budgets = sorted(set([b for b in searched_budgets if b <= total_dim]
                         + [capped_full]))

    decided = list(decided_models(dataset_name, 'general'))
    all_families = _get_regularization_families(task_type=task_type)
    families = {f: v for f, v in all_families.items()
                if _FAMILY_MODEL[f] in decided}
    without_family = [m for m in decided
                      if m not in _FAMILY_MODEL.values()]
    families_not_run = {
        f: (f"'{_FAMILY_MODEL[f]}' is not one of {dataset_name}'s decided "
            f"models {decided}")
        for f in all_families if f not in families}
    for f, why in families_not_run.items():
        print(f"  Family {f} not run: {why}")
    for m in without_family:
        print(f"  Decided model {m} not run: c16 has no regularisation sweep "
              f"for it")

    results = {
        'result_identity': C16_RESULT_IDENTITY,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'selection_scope': 'fold_train_rows',
        'n_outer_folds': n_folds,
        'decided_models': decided,
        'families_run': {f: _FAMILY_MODEL[f] for f in families},
        'families_not_run': families_not_run,
        'decided_models_without_family': without_family,
        'budgets': budgets,
        'searched_budgets': searched_budgets,
        'total_dim': total_dim,
        'capped_full': capped_full,
        'capped_full_recipe_budget_relaxed': capped_full not in searched_budgets,
        'families': {},
    }

    if not families:
        results['skipped'] = True
        results['reason'] = (
            f"none of {dataset_name}'s decided models {decided} has a "
            f"regularisation sweep in c16 (RF = rf, SVM = svm)")
        print(f"  SKIP c16: {results['reason']}")
        return results

    print(f"  Families: {', '.join(f'{f} ({_FAMILY_MODEL[f]})' for f in families)}; "
          f"budgets {budgets} features; {n_folds} outer fold(s)")

    # ── Per outer fold, stored as each finishes (2026-10-05) ──
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    ck = Checkpoint(
        experiment_sidecar(output_dir, 'c16_feature_selection_as_regularization',
                           dataset_name),
        run_identity('c16_feature_selection_as_regularization', dataset_name,
                     data, config, result_identity=C16_RESULT_IDENTITY,
                     families=sorted(families), budgets=budgets))

    folds = nested_cv_outer_folds(reps, labels, groups, n_folds, SEED,
                                  task_type)
    per_fold = []
    for k, (train_idx, val_idx) in enumerate(folds):
        if ck.done(fold=int(k)):
            per_fold.append(ck.get(fold=int(k)))
            print(f"  fold {k}: finished in an earlier attempt, read back")
            continue
        print(f"\n  --- fold {k + 1}/{len(folds)} "
              f"({len(train_idx)} training rows, {len(val_idx)} held out) ---")
        rec = _run_fold(
            k, train_idx, val_idx, reps=reps, labels=labels, groups=groups,
            augmentations=augmentations, n_folds=n_folds,
            dataset_name=dataset_name, families=families, budgets=budgets,
            searched_budgets=searched_budgets, total_dim=total_dim,
            metric_name=metric_name, metric_fn=metric_fn,
            lower_is_better=lower_is_better, task_type=task_type)
        per_fold.append(rec)
        # A fold with any failed unit (selection, search, base fit or hybrid
        # cell) is not stored, so the next attempt computes it again.
        if _fold_failed(rec):
            print(f"  fold {k}: not stored, a unit failed; the next attempt "
                  f"computes it again")
        else:
            ck.put(convert_numpy(rec), fold=int(k))
    results['per_fold'] = convert_numpy(per_fold)

    # Honest leave-one-fold-out re-score inputs (per arm = per "model").
    #   hybrid arm    = the simple-hybrid budget sweep at DEFAULT regularization
    #                   (best_simple_hybrid = best over budgets < capped_full);
    #   baseline arm  = the tuned regularization sweep at capped_full
    #                   (best_tuned_full = best over settings).
    reselection_candidates = {}
    reselected_baselines = {}

    arms = [(_arm_name(f, i), f, i) for f in families for i in (1, 2)]
    for arm_name, fam_name, slot_i in arms:
        variants = families[fam_name]
        param_name = variants[0][0]
        # Positionally correct per fold: fold_scores[k] is fold k, None where
        # the fold has no score (fold_reselection treats None as missing).
        heatmap = {}
        for budget in budgets:
            heatmap[budget] = {}
            for _, pval, _ in variants:
                cells = [((r.get('cells') or {}).get(arm_name) or {})
                         .get(str(budget), {}).get(str(pval))
                         for r in per_fold]
                vals = [c.get('val') if c else None for c in cells]
                trains = [c.get('train') if c else None for c in cells]
                widths = [c.get('width') if c else None for c in cells]
                vv = [v for v in vals if v is not None]
                tt = [t for t in trains if t is not None]
                val_score = float(np.mean(vv)) if vv else float('nan')
                train_score = float(np.mean(tt)) if tt else float('nan')
                heatmap[budget][str(pval)] = {
                    'val': val_score,
                    'train': train_score,
                    # Positive gap means overfitting under BOTH metric
                    # directions. Fixed 2026-09-10 (metric-direction audit).
                    'gap': ((val_score - train_score) if lower_is_better
                            else (train_score - val_score)),
                    'val_std_over_folds': (float(np.std(vv)) if len(vv) > 1
                                           else None),
                    'spread_note': ('one fold, no spread' if len(vv) == 1
                                    else None),
                    'n_folds_scored': len(vv),
                    'fold_scores': vals,
                    'width_built_per_fold': widths,
                }
        print(f"\n  --- {arm_name} ({len(variants)} regularization levels "
              f"x {len(budgets)} budgets) ---")
        for budget in budgets:
            print(f"    budget={budget}: " + ', '.join(
                f'{pval}={heatmap[budget][str(pval)]["val"]:.4f}'
                for _, pval, _ in variants))

        # --- Analysis 1: Regularization sensitivity per budget ---
        sensitivity = {}
        for budget in budgets:
            vals = [heatmap[budget][str(pval)]['val'] for _, pval, _ in variants]
            vals = [v for v in vals if not np.isnan(v)]
            sensitivity[budget] = float(np.std(vals)) if len(vals) >= 2 else float('nan')

        # --- Analysis 2: Best-tuned capped_full vs simple-hybrid ---
        full_dim_budget = capped_full
        full_vals = [(pval, heatmap[full_dim_budget][str(pval)]['val'])
                     for _, pval, _ in variants]
        _sentinel = -1e10 if not lower_is_better else 1e10
        _best_fn = min if lower_is_better else max
        best_tuned_full = _best_fn(full_vals, key=lambda x: x[1] if not np.isnan(x[1]) else _sentinel)

        default_pval = _default_param_value(fam_name, variants)
        hybrid_scores = [(b, heatmap[b][str(default_pval)]['val'])
                         for b in budgets if b < full_dim_budget]
        best_simple_hybrid = _best_fn(hybrid_scores, key=lambda x: x[1] if not np.isnan(x[1]) else _sentinel) if hybrid_scores else (None, float('nan'))

        # ⛔ SAY WHY IT IS None. hybrid_wins=None was the same value for a
        # genuine tie and for a comparison that could not be made at all.
        hybrid_wins_reason = None
        if not any(np.isnan(x) for x in [best_simple_hybrid[1], best_tuned_full[1]]):
            if lower_is_better:
                hybrid_wins = best_simple_hybrid[1] < best_tuned_full[1]
            else:
                hybrid_wins = best_simple_hybrid[1] > best_tuned_full[1]
        else:
            hybrid_wins = None
            if np.isnan(best_tuned_full[1]) and full_dim_budget >= total_dim:
                hybrid_wins_reason = (
                    f'no full-dimension cell: capped_full ({full_dim_budget}) '
                    f'is at or above the pool width ({total_dim}), and cells '
                    f'that wide are skipped because they are the '
                    f'concatenation of every representation')
            elif np.isnan(best_tuned_full[1]) and np.isnan(best_simple_hybrid[1]):
                hybrid_wins_reason = 'neither arm produced a finite score'
            elif np.isnan(best_tuned_full[1]):
                hybrid_wins_reason = (
                    'the full-dimension arm produced no finite score at '
                    f'budget {full_dim_budget}')
            else:
                hybrid_wins_reason = (
                    'the simple-hybrid arm produced no finite score at any '
                    'budget below the full-dimension one')

        comparison = {
            'hybrid_wins_reason': hybrid_wins_reason,
            'best_tuned_full_dim': {
                # 2026-10-05: despite the key name, this is NOT the model on
                # all features. It is a hybrid built at capped_full =
                # min(total_dim, 2 x rows, 2048) features, tuned over this
                # family's regularisation settings. Its recipe is the fold's
                # best over every searched budget, rebuilt at capped_full.
                'description': ('hybrid at min(total features, 2 x rows, '
                                '2048) features, best regularisation '
                                'setting; not all features'),
                'width_built_per_fold': (
                    heatmap[full_dim_budget].get(str(best_tuned_full[0]), {})
                    .get('width_built_per_fold')),
                'param_value': best_tuned_full[0],
                'score': best_tuned_full[1],
                'budget': int(full_dim_budget),
                'is_capped': bool(full_dim_budget < total_dim),
                'max_budget_in_grid': int(max(budgets)),
                'capped_by': ('total_dim' if capped_full == total_dim
                              else '2*n_samples' if capped_full == 2 * n_samples
                              else '2048 hard cap'),
                'recipe_budget_relaxed': capped_full not in searched_budgets,
            },
            'simple_hybrid': {
                'budget': best_simple_hybrid[0],
                'default_param': default_pval,
                'score': best_simple_hybrid[1],
                'width_built_per_fold': (
                    heatmap.get(best_simple_hybrid[0], {})
                    .get(str(default_pval), {}).get('width_built_per_fold')
                    if best_simple_hybrid[0] is not None else None),
            },
            'hybrid_wins': hybrid_wins,
        }
        _cells = [c['val'] for row in heatmap.values() for c in row.values()]
        _finite = [v for v in _cells if not np.isnan(float(v))]
        comparison['family_blank'] = (len(_finite) == 0)
        if comparison['family_blank']:
            _errs = [f"fold {r['fold']}: {e.get('error')}"
                     for r in per_fold for e in (r.get('errors') or [])
                     if e.get('model') in (None, _FAMILY_MODEL[fam_name])]
            comparison['family_blank_reason'] = (
                '; '.join(_errs[:4])
                or (f'no fold gave {_FAMILY_MODEL[fam_name]} a hybrid number '
                    f'{slot_i} with distinct features, or no cell finished'))

        results['families'][arm_name] = {
            'param_name': param_name,
            'family': fam_name,
            'hybrid_slot': slot_i,
            # The decided model whose per-fold search chose this arm's hybrids.
            'recipe_downstream_model': _FAMILY_MODEL[fam_name],
            'heatmap': convert_numpy(heatmap),
            'sensitivity_by_budget': convert_numpy(sensitivity),
            'comparison': convert_numpy(comparison),
        }

        # --- Per-fold reselection inputs for this arm ---
        reselection_candidates[arm_name] = [
            {'name': f'b{b}',
             'fold_scores': list(heatmap[b][str(default_pval)]['fold_scores'])}
            for b in budgets if b < full_dim_budget]
        reselected_baselines[arm_name] = {
            f'{param_name}={pval}':
                list(heatmap[full_dim_budget][str(pval)]['fold_scores'])
            for _, pval, _ in variants}

        print(f"    Sensitivity: {', '.join(f'b={b}:{s:.4f}' for b, s in sensitivity.items())}")
        if comparison.get('hybrid_wins') is not None:
            winner = 'simple_hybrid' if comparison['hybrid_wins'] else 'tuned_full_dim'
            print(f"    Winner on the folds that picked it: {winner}")

    # ── The 2 base representations per family (2026-10-05) ──
    # Per family, per slot, per regularisation setting: held-out score per
    # fold (fold-aligned, None where missing) and the representation each
    # fold's search picked for that slot (it can differ between folds).
    base_summary = {}
    for fam_name, variants in families.items():
        fam_out = {}
        for bi in (1, 2):
            slot_key = f'base{bi}'
            slots = [((r.get('base_cells') or {}).get(fam_name) or {})
                     .get(slot_key) for r in per_fold]
            by_setting = {}
            for _, pval, _ in variants:
                cells = [(sl.get('settings') or {}).get(str(pval))
                         if sl else None for sl in slots]
                vals = [c.get('val') if c else None for c in cells]
                trains = [c.get('train') if c else None for c in cells]
                vv = [v for v in vals if v is not None]
                tt = [t for t in trains if t is not None]
                by_setting[str(pval)] = {
                    'val': float(np.mean(vv)) if vv else float('nan'),
                    'train': float(np.mean(tt)) if tt else float('nan'),
                    'val_std_over_folds': (float(np.std(vv)) if len(vv) > 1
                                           else None),
                    'spread_note': ('one fold, no spread' if len(vv) == 1
                                    else None),
                    'n_folds_scored': len(vv),
                    'fold_scores': vals,
                    'width_built_per_fold': [c.get('width') if c else None
                                             for c in cells],
                }
            fam_out[slot_key] = {
                'rep_per_fold': [sl.get('rep') if sl else None
                                 for sl in slots],
                'by_setting': by_setting,
            }
            print(f"  {fam_name} {slot_key} "
                  f"({[sl.get('rep') if sl else None for sl in slots]}): "
                  + ', '.join(f"{pv}={d['val']:.4f}"
                              for pv, d in by_setting.items()))
        base_summary[fam_name] = fam_out
    results['base_representations'] = convert_numpy(base_summary)

    # Standard fold_reselection block (exact CONTRACT key names). Keys are the
    # ARM names (each arm is one "model" for the re-score).
    results['fold_reselection'] = {
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        'exclude_folds': [],
        'candidates_by_model': reselection_candidates,
        'baselines_by_model': reselected_baselines,
    }

    # Each arm's winning budget and winning regularization are re-picked on the
    # other folds and read on the held-out one. It costs no fits. With one
    # outer fold there is nothing to re-pick on; fold_reselection marks that
    # single_fold.
    reselection_result = {}
    try:
        # 2026-10-05: every hybrid was chosen by a search on the fold's own
        # training rows, so the recipe is held out from that fold's scores.
        reselection_result = fold_reselection(
            reselection_candidates, reselected_baselines, lower_is_better,
            metric_name=metric_name, recipe_selection='fold_train_rows')
    except Exception as e:
        print(f"  leave-one-fold-out re-score unavailable: {e}")
    results['fold_reselection_result'] = reselection_result

    if reselection_result:
        print("\n  Leave-one-fold-out re-score (per arm):")
        print(f"    tuned_full = hybrid at {capped_full} features "
              f"(min of {total_dim} total, 2 x rows, 2048), best "
              f"regularisation setting -- not all features")
        print(f"    {'Arm':<14} {'hybrid':>10} {'tuned_full':>11} "
              f"{'delta':>10}  winner")
        for arm in sorted(reselection_result):
            r = reselection_result[arm]
            d = r.get('delta_mean_of_folds', float('nan'))
            win = ('simple_hybrid' if d > 0 else 'tuned_full_dim') \
                if not np.isnan(d) else 'n/a'
            note = ' ONE FOLD (no held-out re-pick)' if r.get('single_fold') else ''
            print(f"    {arm:<14} {r['hybrid_reselected']:>10.4f} "
                  f"{r['best_single_reselected']:>11.4f} {d:>+10.4f}  {win}{note}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C16: Feature Selection as Regularization')
    args = parser.parse_args()
    # get_config applies the dataset's decided grid (it is given --dataset),
    # so config['budgets'] is the decided list.
    config = get_config(args)
    print(f"Dataset: {args.dataset}")
    print(f"Mode:    {args.mode}")

    # Representations: the dataset's pruned list only (handoff 2.4 step 9).
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _rep_names = DATASET_REPS[args.dataset]['general']
    print(f"Reps:    {_rep_names}")
    data = load_data(args.dataset, rep_preset='all', rep_names=_rep_names)
    data['augmentations'] = None
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c16_feature_selection_as_regularization',
                            args.dataset)
