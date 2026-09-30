#!/usr/bin/env python3
"""C16: Feature Selection as Regularization — the hybrid's budget subsumes model-level regularization.

Paper: Complexity
Core thesis: Selecting features via the hybrid is a form of inductive bias equivalent
    to explicit regularization (L2 penalty, max_depth, C). At the optimal hybrid budget,
    model regularization becomes irrelevant because feature selection already controls
    the bias-variance tradeoff.

Three analyses:
  1. Heatmap: R² over (regularization_strength × hybrid_budget) per model family
  2. Sensitivity: std(R² across regularization) at each budget — should → 0 at optimal budget
  3. Equivalence: for Ridge, compute what L2 alpha is equivalent to hybrid budget=k

Usage:
    python c16_feature_selection_as_regularization.py --dataset logd --reps all --mode intermediary
    python c16_feature_selection_as_regularization.py --dataset flab_thermo_tresanco --mode intermediary

Run the ridge-scored precompute once per dataset FIRST (2026-09-12 decision —
each family asks the pool for its own downstream model, and 'ridge' is only in
a search that named it):
    python tests/precompute_hybrids.py <dataset> --with-ridge
That writes an extra search file and invalidates nothing. Skip it and the first
lookup here builds the same search inside this experiment instead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from experiments._shared import (
    _local_aug_conditions,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    build_fold_hybrid,
    get_best_search_for_downstream,
    fold_reselection,
    # 2026-09-12, owner's decision: c16 asks the pool for 'ridge' for the Ridge
    # family, 'svm' for SVR and 'rf' for RandomForest. RIDGE_MODELS is the named
    # model list that puts 'ridge' into the search — the same shape as
    # DISTANCE_MODELS for knn — so models=None, which every precompute uses,
    # still scores no ridge. Ridge stays banned as a downstream evaluator
    # everywhere else; c16 is one of the approved exceptions.
    RIDGE_MODELS,
)
from kirby.find_best_hybrid import _predict_for_metric, _resolve_models, DEFAULT_MODELS


# =========================================================================
# WHERE THE WALL CLOCK GOES
# =========================================================================
# Every line c16 prints about its own progress carries seconds since run()
# started. Added 2026-09-25 after a local run sat for 25 minutes between the
# experiment banner and its first heatmap row with nothing printed, so there
# was no way to tell a slow recipe lookup from a slow cell.
_C16_T0 = [None]

# =========================================================================
# TWO QUANTITIES EVERY CELL RECOMPUTES AND THAT DO NOT DEPEND ON THE CELL
# =========================================================================
# Measured 2026-09-25 on flab_thermo_tresanco (672 antibodies, 5
# representations capped at 1000 features each, 3 folds). One cell is one
# (regularisation setting, budget) pair and it calls build_fold_hybrid once per
# fold, so the heatmap is 14 settings x 3 budgets x 3 folds = 126 builds. Two
# things inside each of those builds are paid for again every time:
#
#  1. per-feature importance over the five representations. create_hybrid's own
#     comment (_shared.py, where precomputed_importance is forwarded) says it
#     depends on the fold and the importance method and on NOTHING else -- not
#     the allocation, not the downstream model, not the budget. 126 builds
#     recompute what is 3 distinct vectors, one per fold.
#  2. the per-representation baseline scores the allocation turns into a budget
#     split. Those depend on the fold and on the estimator, and NOT on the
#     budget or the allocation method -- hybrid.py:6656 says recomputing them
#     costs ~240s and hands them back on feature_info['allocation_baseline_
#     scores'] so a caller sweeping budgets can pass them straight back in.
#     Each (regularisation setting, fold) pays for them 3 times, once per
#     budget, where 1 would do.
#
# Neither is a condition of this experiment and neither changes a number: they
# are the same vectors, computed once instead of 126 and 3 times. f31's
# verification block (f31_internal_evaluator_tuning.py:548) is the existing
# check that precomputed_importance gives the identical allocation.
#
# Default: ON for a local run (KIRBY_LOCAL_GRID set by
# /Volumes/seagate/kirby_localrun/env.sh), OFF everywhere else, so a cluster
# run recomputes exactly as it did before and is bit-identical.
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

# The regularisation setting each family's "default regularisation" column is
# read at, stated as a VALUE and looked up in that family's sweep list. It used
# to be read as variants[len(variants) // 2], a position. LogReg's six C values
# run 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, so position 3 is C=10.0 while the
# stated default is C=1.0 — every classification run read its default-
# regularisation column, its l2_equivalence hybrid_gap and its fold-reselection
# hybrid arm at ten times the intended penalty. Ridge's six alphas happen to
# put 1.0 at the same position, which is why the expression looked right.
_FAMILY_DEFAULT_PARAM = {
    'Ridge': 1.0,     # alpha
    'LogReg': 1.0,    # C, inverse regularisation
    'SVM': 1.0,       # C, inverse regularisation
    'RF': 10,         # max_depth
}


def _default_param_value(fam_name, variants):
    """This family's default regularisation setting, found by value.

    Raises when the value is absent from the sweep, so editing a sweep list
    cannot silently move the column onto a different setting.
    """
    want = _FAMILY_DEFAULT_PARAM.get(fam_name)
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


def _get_regularization_families(task_type='regression'):
    """Return {family: [(param_name, param_value, factory_fn), ...]}.

    Regression families use Ridge/RFR/SVR. Classification families use
    LogisticRegression/RFClassifier/SVC (the latter with probability=True so
    AUROC can use class-1 probabilities downstream). LogReg's C parameter is
    inverse regularization: smaller C → stronger L2 penalty (opposite
    convention to Ridge's alpha). The sweep values are chosen so that
    _best_fn picks the most-regularized model that still fits well.
    """
    if task_type == 'classification':
        return {
            'LogReg': [
                # C is inverse reg: small C = strong L2 penalty.
                ('C', 0.01, lambda: LogisticRegression(C=0.01, max_iter=2000)),
                ('C', 0.1, lambda: LogisticRegression(C=0.1, max_iter=2000)),
                ('C', 1.0, lambda: LogisticRegression(C=1.0, max_iter=2000)),
                ('C', 10.0, lambda: LogisticRegression(C=10.0, max_iter=2000)),
                ('C', 100.0, lambda: LogisticRegression(C=100.0, max_iter=2000)),
                ('C', 1000.0, lambda: LogisticRegression(C=1000.0, max_iter=2000)),
            ],
            'RF': [
                ('max_depth', 3, lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)),
                ('max_depth', 5, lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
                ('max_depth', 10, lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('max_depth', None, lambda: RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
            ],
            'SVM': [
                ('C', 0.01, lambda: SVC(kernel='rbf', C=0.01, probability=True)),
                ('C', 0.1, lambda: SVC(kernel='rbf', C=0.1, probability=True)),
                ('C', 1.0, lambda: SVC(kernel='rbf', C=1.0, probability=True)),
                ('C', 10.0, lambda: SVC(kernel='rbf', C=10.0, probability=True)),
            ],
        }
    return {
        'Ridge': [
            ('alpha', 0.001, lambda: Ridge(alpha=0.001)),
            ('alpha', 0.01, lambda: Ridge(alpha=0.01)),
            ('alpha', 0.1, lambda: Ridge(alpha=0.1)),
            ('alpha', 1.0, lambda: Ridge(alpha=1.0)),
            ('alpha', 10.0, lambda: Ridge(alpha=10.0)),
            ('alpha', 100.0, lambda: Ridge(alpha=100.0)),
        ],
        'RF': [
            ('max_depth', 3, lambda: RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)),
            ('max_depth', 5, lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
            ('max_depth', 10, lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('max_depth', None, lambda: RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
        ],
        'SVM': [
            ('C', 0.01, lambda: SVR(kernel='rbf', C=0.01)),
            ('C', 0.1, lambda: SVR(kernel='rbf', C=0.1)),
            ('C', 1.0, lambda: SVR(kernel='rbf', C=1.0)),
            ('C', 10.0, lambda: SVR(kernel='rbf', C=10.0)),
        ],
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


def _evaluate_cell(reps, labels, groups, model_factory, budget, n_folds, mode,
                   metric_fn, augmentations=None, blueprint=None,
                   task_type='regression',
                   importance_by_fold=None, alloc_scores_by_fold=None):
    """Evaluate one (model, budget) cell. Returns (score, train_score, fold_scores).

    importance_by_fold: {(fold, importance_method): {rep: scores}} shared by the
        WHOLE heatmap and filled here on first use. Per-feature importance over
        the base representations depends on the fold and the importance method
        and on nothing else, so every cell can read the same vector.
    alloc_scores_by_fold: {(fold, internal_evaluator): {rep: score}} shared by
        the budgets of ONE (family, regularisation setting) and filled from
        feature_info['allocation_baseline_scores'] on first use. Those scores
        depend on the fold and the estimator, so they may not be shared across
        regularisation settings -- alpha=0.001 and alpha=100 are different
        estimators and score the representations differently.
    Both None (the cluster default) recomputes everything inside create_hybrid,
    which is what this file did before 2026-09-25.

    fold_scores is the list of per-fold VALIDATION scores (same values that are
    averaged into `score`). It is threaded out so the honest leave-one-fold-out
    re-score can re-pick the winning budget / regularization per held-out fold
    (see the fold_reselection block built in run()). Empty list when no fold ran.
    """
    sample_X = next(iter(reps.values()))
    # ⛔ POSITIONALLY CORRECT, NOT COMPACTED. The per-fold list this returns is
    # read back by fold_reselection -> _hr_lofo_score, which treats
    # fold_scores[k] as FOLD k (_shared.py:1596). Appending only the folds that
    # survived meant a cell that lost fold 2 returned three numbers, and one
    # candidate's fold 2 was then compared against another candidate's fold 3 --
    # the leave-one-fold-out delta built from mismatched folds. A slot is now
    # written for every fold and a failed one holds None, which _hr_is_bad
    # already treats as bad (_shared.py:1515).
    val_by_fold = [None] * int(n_folds)
    train_by_fold = [None] * int(n_folds)
    # (row budget, the blueprint's own budget) for every cell where the pool
    # returned a recipe filed under a different budget than the one asked for.
    blueprint_budget_mismatch = set()

    for _fold_i, (train_idx, val_idx) in enumerate(
            cv_split(groups, n_folds, sample_X, labels, mode=mode,
                     task_type=task_type)):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        augs_full = augmentations or {}
        train_augs = {k: v[train_idx] for k, v in augs_full.items()} if augs_full else None
        val_augs = {k: v[val_idx] for k, v in augs_full.items()} if augs_full else None

        # Hybrid blueprint is the ONLY path for getting X_train/X_val here.
        # The previous else-branch concatenated every rep at full dim and
        # trained on it — that's a concat-train baseline, removed 2026-05-12
        # per user directive (no concat-all baselines). Cells where the
        # blueprint is missing OR budget >= total_dim are now skipped.
        if blueprint is None:
            continue
        if budget is None or budget >= sum(v.shape[1] for v in reps.values()):
            continue
        # Per-feature importance for this fold, computed on the rows
        # create_hybrid itself would use: it carves an internal holdout out of
        # the fold's training rows with internal_holdout_split and ranks
        # features on the training side of that carve. Computing it on the
        # whole fold would rank features on rows create_hybrid holds out.
        _pre_imp = None
        if importance_by_fold is not None and blueprint is not None:
            _imeth = getattr(blueprint, 'importance_method', None)
            _ikey = (_fold_i, _imeth)
            if _ikey not in importance_by_fold:
                from kirby.hybrid import (compute_feature_importance,
                                          internal_holdout_split)
                _inner_tr, _ = internal_holdout_split(
                    len(y_train), y_train,
                    (groups[train_idx] if groups is not None else None))
                importance_by_fold[_ikey] = compute_feature_importance(
                    {k: v[_inner_tr] for k, v in train_reps.items()},
                    y_train[_inner_tr], method=_imeth)
            _pre_imp = importance_by_fold[_ikey]

        # The per-representation baseline scores this fold's estimator gives.
        # Keyed by the internal evaluator as well as the fold, because a
        # blueprint naming a different evaluator scores the representations
        # with a different model.
        _akey = (_fold_i, getattr(blueprint, 'internal_evaluator', None))
        _pre_alloc = (alloc_scores_by_fold.get(_akey)
                      if alloc_scores_by_fold is not None else None)

        try:
            # ⛔ budget_override, OR THE ROW IS NOT THE BUDGET IT IS LABELLED.
            # get_best_for_downstream returns the exact-budget entry when there
            # is one and otherwise the best entry across ALL budgets for that
            # model (_shared.py:7105), and further fallbacks after that. Without
            # an override build_fold_hybrid builds at blueprint.budget
            # (_shared.py:5127), so a heatmap row labelled 250 could be built at
            # 1000 -- and this file's x-axis IS the budget. Same defect as c15's
            # sweep, which is the case build_fold_hybrid's docstring names as
            # the reason budget_override exists.
            X_train, X_val, fi = build_fold_hybrid(
                train_reps, val_reps, y_train, blueprint,
                augmentations_train=train_augs, augmentations_val=val_augs,
                groups=(groups[train_idx] if groups is not None else None),
                budget_override=budget,
                model=model_factory(),
                metric_fn=metric_fn,
                precomputed_importance=_pre_imp,
                precomputed_alloc_scores=_pre_alloc)
            if (alloc_scores_by_fold is not None and _pre_alloc is None):
                _sc = (fi or {}).get('allocation_baseline_scores')
                if _sc:
                    alloc_scores_by_fold[_akey] = dict(_sc)
            _bp_budget = getattr(blueprint, 'budget', None)
            if _bp_budget is not None and int(_bp_budget) != int(budget):
                blueprint_budget_mismatch.add((int(budget), int(_bp_budget)))
        except Exception:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_tr_s, y_train)
        if _fold_i < len(val_by_fold):
            val_by_fold[_fold_i] = _score_prediction(
                model, X_va_s, y_val, metric_fn, task_type)
            train_by_fold[_fold_i] = _score_prediction(
                model, X_tr_s, y_train, metric_fn, task_type)

    val_scores = [v for v in val_by_fold if v is not None]
    train_scores = [v for v in train_by_fold if v is not None]
    if blueprint_budget_mismatch:
        print(f"      budget {budget}: the pool's recipe was filed under a "
              f"different budget {sorted(blueprint_budget_mismatch)}; built at "
              f"{budget} via budget_override")
    if not val_scores:
        return float('nan'), float('nan'), [None] * int(n_folds)
    return (float(np.mean(val_scores)), float(np.mean(train_scores)),
            [None if v is None else float(v) for v in val_by_fold])


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
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    augmentations = data.get('augmentations')
    total_dim = sum(v.shape[1] for v in reps.values())
    n_samples = len(labels)
    # capped_full = "full-dim baseline" budget. Bounded by 2n to keep MLP/SVM
    # tractable, plus a 2048 hard cap on top of that. Searched via fast
    # find_best_hybrid (not the pool helper — precompute never searched at
    # this non-canonical value, so a pool lookup would silently return None).
    capped_full = min(total_dim, 2 * n_samples, 2048)
    # Canonical pool budgets: FULL_BUDGETS filtered by total_dim.
    # The `b <= n_samples` half of this filter is GONE since 2026-09-10:
    # find_best_hybrid's allow_budget_exceeds_n now defaults to True, so
    # precompute searches and stores every budget asked for, including ones
    # above the row count. Keeping the clause made this file ask for a shorter
    # list than the pool holds — on the small sets, three budgets out of eight —
    # which is the divergence that makes a blueprint lookup return None and
    # silently drop a column.
    from experiments._shared import FULL_BUDGETS
    pool_budgets = sorted(set(
        b for b in FULL_BUDGETS
        if b <= total_dim
    ))
    # capped_full is treated separately (fast search), so add it to the
    # full budget list but track it as a non-pool entry.
    budgets = sorted(set(pool_budgets + [capped_full]))

    families = _get_regularization_families(task_type=task_type)

    # Apply canonical model-size gate via _resolve_models (list form triggers
    # _max_samples and _DOMAIN_MODEL_SKIPS / _DATASET_STRATEGY_SKIPS lookups).
    # 'LogReg' has no entry in DEFAULT_MODELS; route it through 'ridge' for
    # _max_samples and skip gating purposes (same family for gating intent).
    #
    # 2026-09-12: this map is also the pool ask now. The owner's decision was
    # "c16 asks the pool for 'ridge' for the Ridge family, 'svm' for SVR and
    # 'rf' for RandomForest" — these are those names, and the blueprint block
    # below reads them.
    _family_to_canonical = {'Ridge': 'ridge', 'LogReg': 'ridge',
                            'RF': 'rf', 'SVM': 'svm'}

    # The model list c16 names into the search cache key. RIDGE_MODELS is
    # DOWNSTREAM_MODELS + ridge, and on the cluster DOWNSTREAM_MODELS already
    # holds svm, so this line is a NO-OP there. Locally KIRBY_LOCAL_MODELS
    # rebinds DOWNSTREAM_MODELS to rf and lightgbm, which took svm out of the
    # list while `_family_to_canonical` still asks the pool for 'svm' for the
    # SVM family -- so every budget of that family came back
    # "no usable hybrid for model=svm" and the family was written with
    # hybrid_wins=None, which reads like a family that was measured and tied.
    # The owner's 2026-09-12 decision says c16 asks the pool for 'svm' for SVR;
    # this makes the named list carry what that decision asks for.
    C16_MODELS = sorted(set(RIDGE_MODELS) | set(_family_to_canonical.values()))
    # ⛔ THE TWO ENDS OF THE BUDGET AXIS WERE CHOSEN UNDER DIFFERENT
    # AUGMENTATION CONDITIONS. Fixed 2026-09-25.
    #
    # This file names an augmentation condition in two places: here, and again
    # for the fast search at capped_full below. Both said 'uncapped' whenever
    # the dataset carries augmentation representations. The POOL side does not:
    # _pool_conditions runs its list through _local_aug_conditions, so with
    # KIRBY_LOCAL_AUG_CONDITIONS=none the recipes at budgets 100 and 250 come
    # from a search with no extra features while the recipe at the
    # full-dimension budget came from a search that pooled the 48 mutation
    # features on top of the five representations. The heatmap's x-axis is the
    # budget, and its last column was built under a different condition from
    # the rest of the row.
    #
    # It is also what made the run unfinishable. _local_aug_conditions' own
    # docstring: an 'uncapped' or 'capped' configuration-fold costs about eight
    # model fits where 'none' costs one. Measured 2026-09-25 on
    # flab_thermo_tresanco: one family's capped_full=1344 search took 28
    # minutes under 'uncapped', and there are three of them, so 84 minutes went
    # by before the first heatmap cell.
    #
    # Naming it once, through the same filter the pool uses, makes the two ends
    # agree. KIRBY_LOCAL_AUG_CONDITIONS is unset on the cluster, where
    # _local_aug_conditions returns its input untouched, so a cluster run still
    # says 'uncapped' in both places and is unchanged.
    _c16_strategy = (_local_aug_conditions(['uncapped'])[0]
                     if augmentations else 'none')
    # What the fast search at capped_full is allowed to pool. None under
    # 'none', mirroring get_precomputed_search, which passes the augmentations
    # only when its condition is not 'none'.
    _c16_augs = augmentations if _c16_strategy != 'none' else None
    for fam_name in list(families.keys()):
        canonical = _family_to_canonical.get(fam_name)
        if canonical is None or canonical not in DEFAULT_MODELS:
            continue  # no canonical entry → library has no opinion, keep family
        resolved = _resolve_models(
            [canonical], task_type=task_type, 
            n_samples=n_samples, dataset_name=dataset_name,
            strategy=_c16_strategy)
        if not resolved:
            print(f"  Skipping {fam_name}: gated by _resolve_models "
                  f"(canonical={canonical}, n_samples={n_samples}, "
                  f"strategy={_c16_strategy})")
            del families[fam_name]

    # Pre-compute search blueprints PER FAMILY per budget.
    #
    # DECIDED 2026-09-12 (EXPERIMENTS_FINALIZED, c16): "c16 asks the pool for
    # 'ridge' for the Ridge family, 'svm' for SVR and 'rf' for RandomForest."
    # Until this change every family's recipe came from one lookup at
    # downstream_model='rf', so the allocation method, the importance method and
    # the internal evaluator that the Ridge and the SVR cells ran on had all
    # been chosen for a RandomForest. Each family now asks for its own model
    # name, which is what _family_to_canonical already maps ('LogReg' asks for
    # 'ridge' too — it is this file's linear family on a classification set).
    #
    # 'ridge' is named through RIDGE_MODELS, the way knn was named through
    # DISTANCE_MODELS for the distance work, NOT by changing the default tier:
    # 'ridge' is not auto-selected, so models=None still scores no ridge.
    # The model list is part of the search cache key, so the six-model search is
    # an EXTRA file and nothing already precomputed is invalidated. It must be
    # passed on both sides. Write side, once per dataset, before c16 runs:
    #     python tests/precompute_hybrids.py <dataset> --with-ridge
    # Without it the first lookup below builds that six-model search itself
    # inside the experiment instead of reading it off disk.
    #
    # - pool_budgets: the precompute has results, use the pool helper (free).
    # - capped_full: the precompute never searched this value, so run one fast
    #   find_best_hybrid PER FAMILY, with that family's own model name as the
    #   downstream model, cached under a dedicated tier so it cannot collide
    #   with the general precompute. It was one rf search shared by all three
    #   families; the decision is one recipe per family, so it is three now.
    #
    # Each blueprint's allocation method, importance method and internal
    # evaluator come from a search ranked over every row of the dataset,
    # including the rows each fold below validates on. How much budget each
    # representation gets is NOT decided that way any more: build_fold_hybrid
    # no longer falls back to the blueprint's whole-dataset representation
    # scores, so every fold re-scores the representations on its own training
    # rows. results['recipe_selected_on'] below carries what is left.
    blueprints = {fam: {} for fam in families}          # {family: {budget: bp}}
    pool_lookup_errors = {fam: {} for fam in families}  # {family: {budget: str}}
    for fam_name in families:
        pool_model = _family_to_canonical.get(fam_name)
        if pool_model is None:
            pool_lookup_errors[fam_name]['all'] = (
                f'no pool model name for family {fam_name}')
            continue
        for budget in pool_budgets:
            _t_look = time.time()
            try:
                search, meta = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=pool_model, budget=budget,
                    augmentations=augmentations,
                    dataset_name=dataset_name, task_type=task_type,
                    data=data,
                    # Names 'ridge' into the search cache key. Required on the
                    # read side or the key misses the --with-ridge file.
                    models=C16_MODELS,
                )
            except Exception as e:
                # Was one try/except around the whole budget loop, so a single
                # failing budget dropped every later one with no message.
                pool_lookup_errors[fam_name][budget] = f'{type(e).__name__}: {e}'
                continue
            if meta:
                blueprints[fam_name][budget] = meta['best_config']['blueprint']
            print(f"  [c16 {_c16_t():.0f}s] recipe lookup {fam_name}/"
                  f"{pool_model} budget={budget}: "
                  f"{'found' if meta else 'none'} in {time.time() - _t_look:.0f}s",
                  flush=True)

    # Fast search for the capped_full slot (not in FULL_BUDGETS), one per family.
    # capped_full_error is now keyed by family — it was a single string while one
    # rf search served all three. It is surfaced in results so a NaN full-dim
    # baseline (blueprint never built -> _evaluate_cell skips all folds) is
    # diagnosable without the slurm log. That was the cause of full-dim NaN on
    # VEGF/Ang2.
    capped_full_error = {}
    for fam_name in families:
        pool_model = _family_to_canonical.get(fam_name)
        capped_full_error[fam_name] = None
        if pool_model is None or capped_full in blueprints[fam_name]:
            continue
        _t_cf = time.time()
        try:
            from kirby.find_best_hybrid import find_best_hybrid as _fbh16
            cf_search = _fbh16(
                base_reps=reps, labels=labels, groups=groups,
                # The model NAME, not a hand-built estimator. The previous
                # rf here was RandomForest(max_features default, max_depth=None),
                # which is not the 'rf' the pool means; naming it keeps this
                # budget's recipe on the same estimator as the other budgets'.
                models=[pool_model],
                budgets=[capped_full],
                task_type=task_type, metric=metric_name,
                cache=True,
                # Model name in the cache name: three searches now, one per
                # family, and they must not overwrite each other.
                dataset_name=f"{dataset_name}__c16_capped{capped_full}__{pool_model}",
                tier='c16_fast',
                augmentations=_c16_augs,
                augmentation_strategy=_c16_strategy,
                verbose=False,
            )
            cf_best = (cf_search.best_by_model_budget.get(pool_model, {}).get(capped_full)
                       or cf_search.best_by_model.get(pool_model))
            if cf_best:
                blueprints[fam_name][capped_full] = cf_best['blueprint']
            else:
                # Search ran without raising but produced no usable blueprint.
                # Record it so the resulting NaN full-dim baseline (all cells at
                # the top budget get skipped in _evaluate_cell) is diagnosable
                # instead of a silent blank.
                capped_full_error[fam_name] = (
                    f'fast search at capped_full={capped_full} returned no '
                    f'blueprint (no eligible {pool_model} result)')
                print(f"  c16 fast search at capped_full={capped_full} "
                      f"[{fam_name}/{pool_model}]: no usable blueprint returned")
        except Exception as e:
            capped_full_error[fam_name] = f'{type(e).__name__}: {e}'
            print(f"  c16 fast search at capped_full={capped_full} "
                  f"[{fam_name}/{pool_model}] failed: {e}")
        print(f"  [c16 {_c16_t():.0f}s] capped_full={capped_full} search "
              f"{fam_name}/{pool_model} took {time.time() - _t_cf:.0f}s",
              flush=True)

    # Say out loud which budgets each family got a recipe for. A family with an
    # empty list here produces a blank column, and on a binary-classification
    # set scored by auc the Ridge/LogReg family is the one that can go blank:
    # the name 'ridge' resolves to RidgeClassifier, which has no predict_proba,
    # so the search cannot score it.
    for fam_name in families:
        got = sorted(blueprints[fam_name])
        print(f"  Recipes for {fam_name} (asked the pool for "
              f"'{_family_to_canonical.get(fam_name)}'): "
              f"{len(got)} of {len(budgets)} budgets — {got}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'budgets': budgets,
        'total_dim': total_dim,
        'capped_full': capped_full,
        # 2026-09-12: these three are keyed by FAMILY now. Each family asks the
        # pool for its own downstream model, so each can succeed or fail on its
        # own. They were a bool and a string while one rf search served all
        # three families.
        'capped_full_blueprint_built': {fam: capped_full in blueprints[fam]
                                        for fam in families},
        'capped_full_error': capped_full_error,
        'pool_lookup_errors': {fam: convert_numpy(errs)
                               for fam, errs in pool_lookup_errors.items()},
        # Which downstream model chose each family's recipe, and the model list
        # that had to be named to get 'ridge' scored at all.
        'recipe_downstream_model': {fam: _family_to_canonical.get(fam)
                                    for fam in families},
        'recipe_model_list': list(C16_MODELS),
        'budgets_without_recipe': {fam: [b for b in budgets
                                         if b not in blueprints[fam]]
                                   for fam in families},
        # Which rows chose what. The budget split is per fold; the rest of the
        # recipe is not, so the heatmap, the sensitivity curve and the
        # L2-equivalence numbers are all read under a recipe picked with every
        # row's label.
        'recipe_selected_on': 'all rows',
        'budget_split_selected_on': "each fold's training rows",
        'families': {},
    }

    # Honest leave-one-fold-out re-score inputs (per-family = per-"model").
    # Two arms, mirroring the two selection axes this experiment argmaxes over:
    #   hybrid arm    = the simple-hybrid budget sweep at DEFAULT regularization
    #                   (best_simple_hybrid = max over budgets < full);
    #   baseline arm  = the tuned regularization sweep at the full-dim budget
    #                   (best_tuned_full = max over reg).
    # Persisting per-fold scores for BOTH lets a reader call fold_reselection()
    # and get a selection-optimism-corrected simple-hybrid vs tuned-full delta,
    # re-picking the winning budget / reg per held-out fold instead of on the
    # same folds it is scored on. Per-family structure is preserved (family name
    # is the model key). See experiments._shared.fold_reselection.
    reselection_candidates = {}  # {family: [{"name": <budget id>, "fold_scores": [...]}, ...]}
    reselected_baselines = {}   # {family: {<reg id>: [fold_scores], ...}}

    # Shared by every cell of every family: one importance vector per (fold,
    # importance method). Empty and unused when _C16_REUSE_FOLD_WORK is off.
    _importance_by_fold = {} if _C16_REUSE_FOLD_WORK else None

    for fam_name, variants in families.items():
        print(f"\n  --- {fam_name} ({len(variants)} regularization levels × {len(budgets)} budgets) ---")
        param_name = variants[0][0]
        # 2026-09-12: this family's own recipes, chosen for its own downstream
        # model, instead of the one rf recipe every family used to share.
        fam_blueprints = blueprints.get(fam_name, {})

        heatmap = {}  # {budget: {param_value: {'val': score, 'train': score}}}
        # One entry per regularisation setting of this family, reused across
        # this family's budgets and across nothing else. The scores belong to
        # the estimator, and the estimator is what the setting changes.
        _alloc_by_setting = {}
        for budget in budgets:
            heatmap[budget] = {}
            for pname, pval, factory in variants:
                _t_cell = time.time()
                if _C16_REUSE_FOLD_WORK:
                    _alloc_cache = _alloc_by_setting.setdefault(str(pval), {})
                else:
                    _alloc_cache = None
                val_score, train_score, fold_scores = _evaluate_cell(
                    reps, labels, groups, factory, budget, n_folds, mode, metric_fn,
                    augmentations=augmentations,
                    blueprint=fam_blueprints.get(budget),
                    task_type=task_type,
                    importance_by_fold=_importance_by_fold,
                    alloc_scores_by_fold=_alloc_cache)
                heatmap[budget][str(pval)] = {
                    'val': val_score,
                    'train': train_score,
                    # Positive gap means overfitting under BOTH metric directions:
                    # on rmse the train error is the smaller number. Matches
                    # c09:623 and c33. Fixed 2026-09-10 (metric-direction audit).
                    'gap': ((val_score - train_score) if lower_is_better
                            else (train_score - val_score)),
                    # Per-fold validation scores kept for the per-fold reselection
                    # (leave-one-fold-out budget / reg selection). Additive.
                    'fold_scores': list(fold_scores),
                }
                print(f"    [c16 {_c16_t():.0f}s] cell {fam_name} "
                      f"budget={budget} {pname}={pval}: "
                      f"{time.time() - _t_cell:.0f}s", flush=True)
            # Print row
            scores_str = ', '.join(f'{pval}={heatmap[budget][str(pval)]["val"]:.4f}'
                                   for _, pval, _ in variants
                                   if str(pval) in heatmap[budget])
            print(f"    budget={budget}: {scores_str}")

        # --- Analysis 1: Regularization sensitivity per budget ---
        sensitivity = {}
        for budget in budgets:
            vals = [heatmap[budget][str(pval)]['val']
                    for _, pval, _ in variants if str(pval) in heatmap[budget]]
            vals = [v for v in vals if not np.isnan(v)]
            sensitivity[budget] = float(np.std(vals)) if len(vals) >= 2 else float('nan')

        # --- Analysis 2: Best-tuned-full-dim vs simple-hybrid ---
        # ⛔ capped_full, NOT max(budgets). The comment below used to assert
        # the two were equal. They are not: capped_full is
        # min(total_dim, 2 * n_samples, 2048) while pool_budgets keeps every
        # FULL_BUDGETS value <= total_dim, so on a wide pool with few molecules
        # the list still holds 2000 while capped_full is far smaller. On
        # flab_thermo_tresanco -- 672 molecules, 16,736 pool features --
        # capped_full is 1,344 and max(budgets) is 2,000. So the "full-dim
        # baseline" ran at an interior budget, and the four per-family fast
        # searches built for capped_full were never the column it read.
        full_dim_budget = capped_full
        full_vals = [(pval, heatmap[full_dim_budget].get(str(pval), {}).get('val', float('nan')))
                     for _, pval, _ in variants]
        _sentinel = -1e10 if not lower_is_better else 1e10
        _best_fn = min if lower_is_better else max
        best_tuned_full = _best_fn(full_vals, key=lambda x: x[1] if not np.isnan(x[1]) else _sentinel)

        # Simple hybrid: default regularization, best non-full budget
        default_pval = _default_param_value(fam_name, variants)
        hybrid_scores = [(b, heatmap[b].get(str(default_pval), {}).get('val', float('nan')))
                         for b in budgets if b < full_dim_budget]
        best_simple_hybrid = _best_fn(hybrid_scores, key=lambda x: x[1] if not np.isnan(x[1]) else _sentinel) if hybrid_scores else (None, float('nan'))

        # ⛔ SAY WHY IT IS None. hybrid_wins=None was the same value for a
        # genuine tie and for a comparison that could not be made at all, which
        # is the failure the note further down was written against, reached by
        # a different route. The commonest route: _evaluate_cell SKIPS any cell
        # whose budget is at or above the pool's total width -- a hybrid that
        # wide is the concatenation of everything, and concat-all baselines are
        # not allowed -- so when capped_full EQUALS the pool width the whole
        # full-dimension column is nan and no family can be compared.
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
                'param_value': best_tuned_full[0],
                'score': best_tuned_full[1],
                # This baseline is evaluated at capped_full =
                # min(total_dim, 2n, 2048), which is NOT necessarily the true
                # full dimensionality. Record the actual budget and a flag so
                # the "full_dim" label cannot be misread when total_dim is
                # larger than the cap. max_budget_in_grid is recorded too,
                # because the two used to be confused for each other.
                'budget': int(full_dim_budget),
                'is_capped': bool(full_dim_budget < total_dim),
                'max_budget_in_grid': int(max(budgets)),
                'capped_by': ('total_dim' if capped_full == total_dim
                              else '2*n_samples' if capped_full == 2 * n_samples
                              else '2048 hard cap'),
            },
            'simple_hybrid': {
                'budget': best_simple_hybrid[0],
                'default_param': default_pval,
                'score': best_simple_hybrid[1],
            },
            'hybrid_wins': hybrid_wins,
        }

        # ⛔ A FAMILY THAT PRODUCED NO NUMBER WAS WRITTEN IN THE SAME SHAPE AS
        # ONE THAT DID. With no blueprint, _evaluate_cell skips every fold, the
        # heatmap fills with NaN, and `comparison` is still written with
        # hybrid_wins=None -- which on its own reads like a family that WAS
        # measured and tied. The SVM family came back blank on every budget,
        # because the pool had no hybrid for model=svm.
        # Each heatmap cell is a dict carrying 'val', not a bare float.
        _cells = [c.get('val') if isinstance(c, dict) else c
                  for row in (heatmap or {}).values()
                  for c in (row or {}).values()]
        _finite = [v for v in _cells
                   if v is not None and not np.isnan(float(v))]
        comparison['family_blank'] = (len(_finite) == 0)
        if comparison['family_blank']:
            _errs = pool_lookup_errors.get(fam_name) or {}
            comparison['family_blank_reason'] = (
                '; '.join(f'{k}: {v}' for k, v in sorted(_errs.items())[:4])
                or f'no finite cell in {len(_cells)} (budget, setting) cells')

        fam_result = {
            'param_name': param_name,
            # 2026-09-12: the downstream model this family's recipe was chosen
            # for. 'ridge' for Ridge and LogReg, 'svm' for SVR/SVC, 'rf' for
            # RandomForest. Every family read 'rf' before this.
            'recipe_downstream_model': _family_to_canonical.get(fam_name),
            'budgets_with_recipe': sorted(fam_blueprints),
            'heatmap': convert_numpy(heatmap),
            'sensitivity_by_budget': convert_numpy(sensitivity),
            'comparison': convert_numpy(comparison),
        }

        # --- Analysis 3: L2 equivalence (linear family — Ridge or LogReg) ---
        if fam_name in ('Ridge', 'LogReg'):
            # For Ridge, the default regularization pval is 1.0 (alpha=1.0).
            # For LogReg, the default is 1.0 (C=1.0 = moderate regularization).
            # Looked up by value, not by position — see _default_param_value.
            default_pval = _default_param_value(fam_name, variants)
            l2_equivalence = {}
            for budget in budgets:
                if budget >= full_dim_budget:
                    continue
                hybrid_gap = heatmap[budget].get(str(default_pval), {}).get('gap', float('nan'))
                if np.isnan(hybrid_gap):
                    continue
                # Find the regularization pval at full_dim that gives the same gap
                full_gaps = [(pval, heatmap[full_dim_budget].get(str(pval), {}).get('gap', float('nan')))
                             for _, pval, _ in variants]
                full_gaps = [(p, g) for p, g in full_gaps if not np.isnan(g)]
                if len(full_gaps) >= 2:
                    # Simple interpolation: find pval where gap ≈ hybrid_gap
                    pvals = np.array([p for p, _ in full_gaps])
                    gaps = np.array([g for _, g in full_gaps])
                    # Find nearest gap
                    idx = np.argmin(np.abs(gaps - hybrid_gap))
                    l2_equivalence[budget] = {
                        'hybrid_gap': hybrid_gap,
                        'equivalent_param': float(pvals[idx]),
                        'param_gap': float(gaps[idx]),
                    }
                    print(f"    L2 equiv: budget={budget} ≈ {param_name}={pvals[idx]:.3f} "
                          f"(gap diff={abs(hybrid_gap - gaps[idx]):.4f})")

            fam_result['l2_equivalence'] = convert_numpy(l2_equivalence)

        # --- Per-fold reselection inputs for this family (per-fold, both arms) ---
        # Hybrid arm: simple-hybrid budgets (b < full) at default regularization.
        _hybrid_default = _default_param_value(fam_name, variants)
        _cand_list = []
        for b in budgets:
            if b >= full_dim_budget:
                continue
            _fs = heatmap[b].get(str(_hybrid_default), {}).get('fold_scores', [])
            _cand_list.append({'name': f'b{b}', 'fold_scores': list(_fs)})
        reselection_candidates[fam_name] = _cand_list
        # Baseline arm: tuned regularization settings at the full-dim budget.
        _base_map = {}
        for _, pval, _f in variants:
            _fs = heatmap[full_dim_budget].get(str(pval), {}).get('fold_scores', [])
            _base_map[f'{param_name}={pval}'] = list(_fs)
        reselected_baselines[fam_name] = _base_map

        results['families'][fam_name] = fam_result
        print(f"    Sensitivity: {', '.join(f'b={b}:{s:.4f}' for b, s in sensitivity.items())}")
        if comparison.get('hybrid_wins') is not None:
            winner = 'simple_hybrid' if comparison['hybrid_wins'] else 'tuned_full_dim'
            print(f"    Winner on the folds that picked it: {winner}")

    # Standard fold_reselection block (exact CONTRACT key names). Keys of
    # candidates_by_model / baselines_by_model are the FAMILY names (each family
    # is one "model" for the re-score). A reader loads this block and calls
    # experiments._shared.fold_reselection(candidates_by_model, baselines_by_model,
    # lower_is_better) to get the selection-optimism-corrected delta per family.
    results['fold_reselection'] = {
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': reselection_candidates,
        'baselines_by_model': reselected_baselines,
    }

    # Compute it here rather than leaving it to whoever reads the file. Each
    # family's winning budget and winning regularization are re-picked on the
    # other folds and read on the held-out one, so both sides pay the same tax
    # for being a maximum. It costs no fits — the per-fold scores already
    # exist. hybrid_wins in each family's comparison block is the same-fold
    # verdict and is kept unchanged beside it.
    reselection_result = {}
    try:
        # ⛔ recipe_selection='whole_dataset'. Every candidate here rebuilds a
        # recipe the pool search picked over EVERY row of the dataset --
        # results['recipe_selected_on'] says exactly that, a few hundred lines
        # up. fold_reselection's docstring says to pass 'whole_dataset' in that
        # case, because the per-fold rebuild fixes the feature weights but not
        # the choice of allocation method, budget, importance method or
        # evaluator, so the delta is not fully held out. Left unset, the block
        # was written with recipe_selection=None, which the same docstring
        # defines as the OPPOSITE: that nothing on these rows chose the recipe.
        reselection_result = fold_reselection(reselection_candidates, reselected_baselines,
                                       lower_is_better, metric_name=metric_name,
                                       recipe_selection='whole_dataset')
    except Exception as e:
        print(f"  honest delta unavailable: {e}")
    results['fold_reselection_result'] = reselection_result

    if reselection_result:
        print("\n  Leave-one-fold-out re-score (per family):")
        print(f"    {'Family':<10} {'hybrid':>10} {'tuned_full':>11} "
              f"{'delta':>10}  winner")
        for fam in sorted(reselection_result):
            r = reselection_result[fam]
            d = r.get('delta_mean_of_folds', float('nan'))
            win = ('simple_hybrid' if d > 0 else 'tuned_full_dim') \
                if not np.isnan(d) else 'n/a'
            note = ' SINGLE-FOLD (no held-out tax)' if r.get('single_fold') else ''
            print(f"    {fam:<10} {r['hybrid_reselected']:>10.4f} "
                  f"{r['best_single_reselected']:>11.4f} {d:>+10.4f}  {win}{note}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C16: Feature Selection as Regularization')
    args = parser.parse_args()
    config = get_config(args)
    print(f"Dataset: {args.dataset}")
    print(f"Mode:    {args.mode}")
    print(f"Reps:    {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c16_feature_selection_as_regularization',
                            args.dataset)
