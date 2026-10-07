#!/usr/bin/env python3
"""C06: Within-Family Complexity Gradient — vary capacity within model families.

Paper: Complexity
Conclusion: Within-family capacity gradient confirms complexity matching.
Dependencies: None (standalone).

For each variant, runs find_best_hybrid() with that variant as the sole
downstream model. The library searches importance methods, allocation methods,
and budgets to find the variant-optimal hybrid recipe. Optimal budget per
variant is the budget where val score peaks. Spearman correlation between
capacity index and optimal budget tests whether capacity and budget are coupled.

Train scores / bias-variance gap are not collected here (find_best_hybrid does
not return train scores). Moved to a dedicated experiment if/when needed.

2026-09-12, decided and built: each cell of families[<family>]['heatmap'] now
carries the per-fold scores behind its mean, and the internal evaluator under
its resolved name — an SVM downstream with the matched evaluator reads
'SVM/SVM'. c06 passes that name in its own evaluator list.

2026-10-05, aligned with HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4 (Adelaide's
grouping (B) for c06: one hybrid per budget per capacity setting, no
representations, on the dataset's decided models, per-fold searches):

- Outer folds from nested_cv_outer_folds at config['n_splits'] (1 = one
  held-out split). Every search runs on ONE outer fold's training rows
  (search_on_fold, the fold's nested-CV cache namespace). No search over all
  rows, and the precompute importance scores are no longer read.
- Families: only those whose model is one of the dataset's two decided models
  (decided_grid.decided_models(ds, 'general')). kNN and Lasso are never
  decided models; lightgbm is decided on some datasets and has no capacity
  family. Each family left out is listed with its reason in the result.
- Grid: the dataset's decided budgets, importance methods, allocation methods,
  internal evaluators ('match' resolved to the family's own name) and
  representation counts. Pruning off. Budgets are no longer widened past the
  decided list.
- Augmentation: only the decided conditions of the general search, which is
  'none' on every dataset, so no augmentation features enter any search here.
- Per fold, per capacity setting, per budget: the best configuration by its
  score on the fold's training rows (mean over the search's inner folds), then
  rebuilt on the training rows with that search's own representation scores and
  scored on the held-out rows. The optimal budget is chosen on the training-row
  score, as before; the held-out score at every budget is stored beside it.
- The SVM family fits SVRScaledTarget (labels standardised inside each fit,
  as the search's own svm has since fb1df54), not a plain SVR.
- Results are written per fold to a checkpoint file carrying
  C06_RESULT_IDENTITY, so a partial file from the old whole-dataset design is
  never read back.
"""
import sys
import os
import time as _time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_family_variants, get_domain_metric,
    search_on_fold, build_fold_hybrid, fold_hybrid_scoring_kwargs,
    first_outer_folds,
    input_fingerprint, fingerprint_mismatch,
)
from experiments.decided_grid import decided_search
from experiments._checkpoint import force_rerun_requested
from kirby.find_best_hybrid import (
    _resolve_models, _impute_nan, KNN_TUNED_K_ATTR,
    nested_cv_outer_folds, nested_cv_cache_namespace)
from kirby.scaling import FeatureScaler

# 2026-10-05: names this file's result layout. It is written into the per-fold
# checkpoint and the result, and into every search's condition_suffix, so
# nothing written by the whole-dataset design (one search per capacity setting
# over every row, a plain SVR) is read back as a fold of this one.
C06_RESULT_IDENTITY = 'c06_per_fold_v2_2026_10_05'
_SEED = 42


# Map c06 family labels to canonical names in DEFAULT_MODELS. List-form
# _resolve_models is the only way to trigger _max_samples + _DOMAIN_MODEL_SKIPS
# + _DATASET_STRATEGY_SKIPS; the dict form is a passthrough.
# ⛔ 'Ridge' WAS A DEAD KEY. get_model_family_variants has returned Lasso, not
# Ridge, since 2026-07-18. The stale key did two things: `--families Ridge`
# matched nothing, so the filter below produced {}, the family loop never ran,
# and c06 SAVED `{'families': {}}` as a result with no error. And Lasso, having
# no entry, fell through the `canonical is None` branch below and so never went
# through _resolve_models -- the only thing that applies _max_samples,
# _DOMAIN_MODEL_SKIPS and _DATASET_STRATEGY_SKIPS.
#
# Lasso maps to None deliberately: DEFAULT_MODELS has no 'lasso' (it holds
# ridge and linear, which are different estimators), so there is no canonical
# name to gate it by. That is now stated rather than silent -- the loop prints
# it.
_FAMILY_TO_CANONICAL = {
    'RF': 'rf', 'kNN': 'knn', 'DNN': 'dnn',
    'XGBoost': 'xgboost', 'SVM': 'svm', 'GP': 'gp',
    'Lasso': None,
}

# DECIDED 2026-09-12: c06 passes the family's evaluator name in its own list.
# What changed and why it had to change here: find_best_hybrid resolves the
# word 'match' through a table of plain model names — rf, lightgbm, xgboost,
# mlp, svm, gp, knn, catboost, dnn, cnn, bnn, ngboost
# (find_best_hybrid.py:5622-5628). c06 names its downstream model
# 'SVM:C=1.0' so that lookup returned nothing and 'match' added no evaluator
# to any variant's search. Stripping the family prefix inside
# find_best_hybrid would change every caller that passes a prefixed model
# name, so the decision was to resolve the name in c06 alone. This touches
# c06 and nothing else.
#
# The linear family has no entry. The internal evaluator is fitted by
# hybrid.py:_evaluate_allocation, whose branches are rf, xgboost, lightgbm,
# mlp, svm, gp, knn, catboost, dnn, cnn, ngboost, bnn — no linear model — so
# there is no name to pass for Lasso. Its searches run on the rest of the
# list and a line below says so.
_FAMILY_TO_EVALUATOR = {
    'RF': 'rf', 'kNN': 'knn', 'DNN': 'dnn',
    'XGBoost': 'xgboost', 'SVM': 'svm', 'GP': 'gp',
}

# The fast speed tier c06 set for itself on 2026-09-12 (three allocation
# methods, lightgbm_gain only, representation counts 3 and all, on every
# dataset) is gone, 2026-10-05: every axis is now the dataset's decided grid
# (DECISIONS.md through decided_grid.decided_search), rule R3.


def _family_evaluators(fam_name, base):
    """Internal evaluator names searched for one family, 'match' resolved here.

    `base` is the dataset's decided internal-evaluator list (2026-10-05; it
    was the fast tier's list). 'match' is replaced by the family's plain model
    name, because c06 names its downstream model 'SVM:C=1.0' and the engine's
    'match' lookup finds nothing for that name.
    """
    matched = _FAMILY_TO_EVALUATOR.get(fam_name)
    names = []
    for ev in base:
        name = matched if ev == 'match' else ev
        if name is None:
            continue  # no plain model name for this family; nothing to pass
        if name not in names:
            names.append(name)
    if not names:
        # A decided list of ['match'] alone (abagym_VEGF) for a family with no
        # plain model name. Lasso is the only such family and it is never a
        # decided model, so this does not fire in a decided run.
        names = ['rf']
        print(f"    {fam_name}: the evaluator list resolved to nothing "
              f"(decided={list(base)}, no matched name), falling back to ['rf']")
    return names


def _apply_model_njobs(est, fam_name, var_name):
    """Set the estimator's worker-process count, when it has one.

    MEASURED 2026-09-25, cProfile over one c06 search on flab_thermo_tresanco
    (672 antibodies, 5 representations at 1000 features each, 2 budgets, 3
    allocation methods, 1 importance method, 1 internal evaluator, 3 folds =
    18 downstream fits). The search took 175.4s. 157.9s of that was the 18
    RandomForestRegressor.fit calls, at 8.8s each. The trees are not the cost:

      102.1s  time.sleep inside joblib's _retrieve — the parent process
              polling for worker processes
       53.5s  sklearn's _make_estimator / clone / get_params /
              inspect.signature — 1800 estimator clones, 100 trees x 18 fits

    get_model_family_variants builds RF, kNN and XGBoost with n_jobs=-1, so one
    fit asks for one worker per core. The local env file already pins every
    other thread pool to one (OMP_NUM_THREADS=1, KIRBY_LGBM_THREADS=1) because
    this laptop runs several experiments at once; n_jobs=-1 was the one place
    left oversubscribing it, and the workers then spent their time waiting on
    each other.

    This changes no condition and no row: same estimator, same
    hyperparameters, same molecules, same folds, identical predictions. It
    changes only how many processes one fit spawns.

    KIRBY_C06_MODEL_NJOBS sets the value. Unset, the estimator is left exactly
    as get_model_family_variants built it EXCEPT under the local debug grid,
    where it defaults to 1. The server is unchanged.
    """
    _default = '1' if os.environ.get('KIRBY_LOCAL_GRID') else ''
    _njobs = os.environ.get('KIRBY_C06_MODEL_NJOBS', _default).strip()
    if not _njobs:
        return est
    try:
        if 'n_jobs' in est.get_params():
            est.set_params(n_jobs=int(_njobs))
    except (AttributeError, ValueError, TypeError) as _e:
        print(f"      {fam_name}/{var_name}: n_jobs not set ({_e})")
    return est


def _hold_variant_setting(est, fam_name, var_name):
    """Make the search fit the setting this variant is named after.

    Added 2026-10-05. Two families did not keep the setting they vary:

    kNN: find_best_hybrid._set_knn_k resets n_neighbors to min(5, sqrt of the
    training rows) before every fit unless the estimator carries the attribute
    named by KNN_TUNED_K_ATTR, so k=20, k=10, k=5 and k=1 were all fitted at
    k=5. Stamping it keeps the variant's own k (clamped to the training rows
    minus 1 by _set_knn_k).

    GP: the four variants differed only in the RBF length scale they started
    from, and sklearn re-fits the length scale by maximising the marginal
    likelihood, so all four could end at the same length scale. Setting
    length_scale_bounds='fixed' on the RBF keeps it at the variant's value. The
    WhiteKernel noise level is still fitted.

    SVM, added 2026-10-05 (dirty-run bug 4): _shared.get_model_family_variants
    now builds the regression SVM family as kirby.hybrid.SVRScaledTarget, which
    standardises the labels on each fit's training rows, as the search's own
    svm has since fb1df54. The check below only converts a plain sklearn SVR,
    so with the current _shared it changes nothing; it stays so that c06 never
    fits a plain SVR on unscaled labels.

    The variant definitions themselves live in _shared.get_model_family_variants;
    this is applied to the estimator c06 hands the search and to the one it
    fits for the held-out score.
    """
    if fam_name == 'SVM':
        from sklearn.svm import SVR as _SVR
        from kirby.hybrid import SVRScaledTarget
        if isinstance(est, _SVR) and not isinstance(est, SVRScaledTarget):
            est = SVRScaledTarget(**est.get_params())
    if fam_name == 'kNN' and hasattr(est, 'n_neighbors'):
        setattr(est, KNN_TUNED_K_ATTR, True)
    if fam_name == 'GP' and hasattr(est, 'kernel') and est.kernel is not None:
        from sklearn.gaussian_process.kernels import RBF as _RBF

        def _fix(k):
            if isinstance(k, _RBF):
                return _RBF(length_scale=k.length_scale,
                            length_scale_bounds='fixed')
            if hasattr(k, 'k1') and hasattr(k, 'k2'):
                # Sum / Product of two kernels, e.g. RBF + WhiteKernel.
                return type(k)(_fix(k.k1), _fix(k.k2))
            return k
        est.set_params(kernel=_fix(est.kernel))
    return est


def _no_score_reason(search, model_key):
    """Why a variant has no scored budget, from the search's own rows.

    Added 2026-10-05. find_best_hybrid leaves a model out of
    best_by_model_budget when none of its configurations has a finite score,
    and c06 saved budget None with nothing saying why (the Lasso alpha=100,
    10 and 1 variants on the server run). This counts the configurations the
    search scored for this model and how many have a finite score. A Lasso
    whose coefficients are all zero predicts one constant, and a rank
    correlation of a constant is undefined; that is the likely cause for the
    large alphas but is NOT checked here.
    """
    rows = [r for r in (getattr(search, 'all_results', None) or [])
            if r.get('model') == model_key]
    finite = [r for r in rows if r.get('score_mean') is not None
              and np.isfinite(r['score_mean'])]
    if not rows:
        reason = 'the search returned no configuration for this variant'
    else:
        reason = (f'{len(rows)} configurations scored, {len(finite)} with a '
                  f'finite score')
    return {'reason': reason, 'n_configurations': len(rows),
            'n_configurations_finite': len(finite)}


def _evaluator_label(fam_name, evaluator):
    """DECIDED 2026-09-12: file the matched evaluator under its resolved name.

    An SVM downstream with the matched internal evaluator reads 'SVM/SVM',
    never 'match'. An SVM downstream scored by the rf evaluator reads
    'SVM/rf'.
    """
    if evaluator is None:
        return None
    matched = _FAMILY_TO_EVALUATOR.get(fam_name)
    return f"{fam_name}/{fam_name if evaluator == matched else evaluator}"


def _family_plan(all_families, decided_models, task_type, n_samples,
                 dataset_name):
    """Which capacity families run on this dataset, and why each other does not.

    2026-10-05, Adelaide's grouping (B) for c06: the comparison stays, on the
    dataset's two decided models. A family runs when its model is one of
    them; _resolve_models then applies the size and dataset gates as before
    (GP above 1,000 rows, for one). Returns (kept {family: variants},
    not_run {family: reason}, decided models with no family [names]).
    """
    kept, not_run = {}, {}
    for fam_name, variants in all_families.items():
        if fam_name not in _FAMILY_TO_CANONICAL:
            raise ValueError(
                f"family {fam_name!r} has no entry in _FAMILY_TO_CANONICAL. "
                f"Add one, or map it to None if no DEFAULT_MODELS name "
                f"matches it. Falling through silently is how Ridge went "
                f"stale for two months.")
        canonical = _FAMILY_TO_CANONICAL[fam_name]
        if canonical is None:
            not_run[fam_name] = (
                f'{fam_name} is not a downstream model in DECISIONS.md Step 1 '
                f'for any dataset (decided here: {decided_models})')
            continue
        if canonical == 'knn':
            not_run[fam_name] = (
                'knn is a downstream model only on the distance search '
                '(DECISIONS.md Step 9); c06 runs no distance search')
            continue
        if canonical not in decided_models:
            not_run[fam_name] = (
                f'{canonical} is not one of the decided models '
                f'{decided_models}')
            continue
        # strategy='none': the general search's decided augmentation
        # condition on every dataset (see run below).
        resolved = _resolve_models(
            [canonical], task_type=task_type, n_samples=n_samples,
            dataset_name=dataset_name, strategy='none')
        if not resolved:
            not_run[fam_name] = (
                f'gated by _resolve_models (canonical={canonical}, '
                f'n_samples={n_samples})')
            continue
        kept[fam_name] = variants
    with_family = {c for c in _FAMILY_TO_CANONICAL.values() if c}
    no_family = [m for m in decided_models if m not in with_family]
    return kept, not_run, no_family


def _fit_heldout(est, X_tr, y_tr, X_va, y_va, metric_fn):
    """Fit one capacity setting on the training rows, score the held-out rows.

    Features scaled with FeatureScaler fitted on the training rows (0/1
    features left as they are, handoff 2.3 item 17) and missing values
    imputed from the training rows, the same two steps the search applies
    before its own downstream fit (find_best_hybrid._evaluate_single).
    """
    sc = FeatureScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_tr_s, X_va_s, _ = _impute_nan(X_tr_s, X_va_s)
    est.fit(X_tr_s, y_tr)
    return float(metric_fn(y_va, est.predict(X_va_s)))


def _checkpoint_path(output_dir, dataset_name):
    return os.path.join(output_dir, 'c06_within_family_gradient',
                        f'{dataset_name}__per_fold_checkpoint.partial.jsonl')
# 2026-10-05: the name ends in .partial.jsonl, not .json, so readers that list
# every *.json in this directory as a dataset result never pick it up (the
# same convention as experiments/_checkpoint.py). The file holds one JSON
# object.


def _load_checkpoint(path, fingerprint, grid_record):
    """Folds already finished by an earlier run of THIS design, or {}.

    A file is used only when its identity, its input fingerprint (rows,
    labels, groups, representations, outer fold count, metric) and its search
    grid all equal this run's. Anything else is ignored and overwritten.
    """
    if not os.path.exists(path):
        return {}
    if force_rerun_requested():
        # 2026-10-05: --force-rerun or KIRBY_FORCE_RERUN=1 recomputes every
        # fold; the file is overwritten after each fold of this run.
        print(f"  checkpoint {path} not read: force rerun requested")
        return {}
    try:
        with open(path) as f:
            ck = json.load(f)
    except (ValueError, OSError) as e:
        print(f"  checkpoint {path} unreadable ({e}); starting from fold 0")
        return {}
    if ck.get('identity') != C06_RESULT_IDENTITY:
        print(f"  checkpoint {path} is from another c06 design "
              f"({ck.get('identity')!r}); not read")
        return {}
    why = fingerprint_mismatch(ck.get('input_fingerprint'), fingerprint)
    if why:
        print(f"  checkpoint {path} not read: {why}")
        return {}
    if ck.get('grid') != grid_record:
        print(f"  checkpoint {path} not read: the search grid changed")
        return {}
    done = {int(r['fold']): r for r in ck.get('per_fold') or []}
    if done:
        print(f"  checkpoint: folds {sorted(done)} already done, reused")
    return done


def _write_checkpoint(path, fingerprint, grid_record, per_fold):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(convert_numpy({'identity': C06_RESULT_IDENTITY,
                                 'input_fingerprint': fingerprint,
                                 'grid': grid_record,
                                 'per_fold': per_fold}), f)
    os.replace(tmp, path)


def _capacity_rho(fam_variants):
    """Spearman rho between capacity index and optimal budget (features).

    rho only; no p-value (owner's rule: no significance tests). spearmanr is
    NaN unless both vary, so a family whose settings all peak at one budget
    is recorded as 'no_budget_variance' rather than a bare NaN.
    """
    capacities, opt_budgets = [], []
    for info in fam_variants.values():
        ob = info.get('optimal_budget') or {}
        if ob.get('budget') is not None:
            capacities.append(info['capacity_index'])
            opt_budgets.append(ob['budget'])
    n_distinct = len(set(opt_budgets))
    if len(capacities) >= 3 and n_distinct >= 2:
        return {'rho': float(spearmanr(capacities, opt_budgets)[0]),
                'n_distinct_budgets': n_distinct,
                'n_variants': len(capacities), 'status': 'measured'}
    return {'rho': float('nan'), 'n_distinct_budgets': n_distinct,
            'n_variants': len(capacities),
            'status': ('insufficient_variants' if len(capacities) < 3
                       else 'no_budget_variance')}


def _run_variant_on_fold(*, reps, labels, groups, tr, va, k, n_outer,
                         dataset_name, metric_name, metric_fn,
                         lower_is_better, fam_name, var_name, var_factory,
                         capacity_idx, grid, fam_evaluators, aug_for_search):
    """One capacity setting on one outer fold: search, per-budget rebuild.

    The search runs on the fold's training rows only, with this capacity
    setting as its sole downstream model, on the decided grid with 'match'
    resolved to the family's name. For each budget the best configuration by
    its training-row score (mean over the search's inner folds) is rebuilt on
    the training rows -- with the representation scores this same search
    measured (dirty-run bug 1) -- and scored on the held-out rows by a fresh
    fit of the same capacity setting.
    """
    model_key = f"{fam_name}:{var_name}"

    def _make():
        return _hold_variant_setting(
            _apply_model_njobs(var_factory(), fam_name, var_name),
            fam_name, var_name)

    rec = {'capacity_index': capacity_idx, 'model_key': model_key,
           'model_class': type(_make()).__name__, 'per_budget': [],
           'optimal_budget': None}
    fixed = {
        'models': {model_key: _make()},
        'budgets': list(grid['budgets']),
        'importance_methods': list(grid['importance_methods']),
        'allocation_methods': list(grid['allocation_methods']),
        'internal_evaluators': list(fam_evaluators),
        'rep_counts': grid['rep_counts'],
        'speed': grid['speed'],
        # Not 'general': the downstream model is a capacity setting, not one
        # of the decided model names, so these searches are filed apart from
        # the general per-fold search. search_kind() of this name is
        # 'general', so the general search's augmentation rule applies.
        'tier': 'c06_variant',
    }
    _t = _time.time()
    try:
        searches = search_on_fold(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            augmentations=aug_for_search, task_type='regression',
            metric=metric_name, seed=_SEED, verbose=False,
            condition_suffix=(f"{C06_RESULT_IDENTITY}_{fam_name}_"
                              f"{var_name}"),
            fixed_grid=fixed)
    except SystemExit as e:
        # SystemExit is NOT an Exception. A library that calls sys.exit()
        # inside a search -- AntiFold's loader does -- would otherwise end
        # the process with no line saying which setting it died on.
        print(f"      fold {k} {model_key}: a library called "
              f"sys.exit({e.code}) instead of raising. Recorded; the "
              f"remaining settings still run.")
        rec['error'] = f'SystemExit({e.code})'
        rec['error_kind'] = 'SystemExit'
        return rec
    fails = getattr(searches, 'failures', None) or []
    if fails:
        rec['search_failures'] = [
            {kk: f.get(kk) for kk in ('aug_condition', 'error_type', 'error')}
            for f in fails]
    rec['search_seconds'] = round(_time.time() - _t, 1)

    # Best configuration per budget over the augmentation conditions the
    # search ran, by training-row score; ties go to 'none'.
    aug_pref = {'none': 0, 'uncapped': 1, 'capped': 2}
    best_per_budget = {}
    any_search = None
    for cond, srch in searches.items():
        if srch is None:
            continue
        any_search = srch
        if srch.lower_is_better != lower_is_better:
            raise RuntimeError(
                f'{dataset_name} fold {k} {model_key}: the search ranks '
                f'{metric_name} with lower_is_better={srch.lower_is_better}, '
                f'get_domain_metric says {lower_is_better}')
        for b, best in (srch.best_by_model_budget.get(model_key) or {}).items():
            if best is None or best.get('score_mean') is None:
                continue
            if not np.isfinite(best['score_mean']):
                continue
            cand = (best['score_mean'] if lower_is_better
                    else -best['score_mean'], aug_pref.get(cond, 99))
            cur = best_per_budget.get(b)
            if cur is None or cand < cur[0]:
                best_per_budget[b] = (cand, cond, best)
    if not best_per_budget:
        if any_search is None:
            rec['error'] = ('the search on this fold failed: '
                            f'{[f.get("error") for f in fails]}')
        else:
            why = _no_score_reason(any_search, model_key)
            rec['error'] = why['reason']
            rec['n_configurations'] = why['n_configurations']
            rec['n_configurations_finite'] = why['n_configurations_finite']
        print(f"      fold {k} {model_key}: no budget scored: {rec['error']}")
        return rec

    y_tr, y_va = labels[tr], labels[va]
    tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
    va_reps = {r: np.asarray(X)[va] for r, X in reps.items()}
    aug_tr = aug_va = None
    if aug_for_search:
        aug_tr = {r: np.asarray(X)[tr] for r, X in aug_for_search.items()}
        aug_va = {r: np.asarray(X)[va] for r, X in aug_for_search.items()}

    for b in sorted(best_per_budget):
        _, cond, best = best_per_budget[b]
        bp = best['blueprint']
        evaluator = best.get('internal_evaluator',
                             getattr(bp, 'internal_evaluator', None))
        fs = best.get('fold_scores')
        cell = {
            'budget': int(b),
            'aug_condition': cond,
            'allocation': bp.allocation_method,
            'importance': bp.importance_method,
            'evaluator': evaluator,
            'evaluator_label': _evaluator_label(fam_name, evaluator),
            'rep_count': getattr(bp, 'rep_count', None),
            # The search's score on this fold's TRAINING rows: the mean over
            # its inner folds, and those inner-fold scores.
            'train_score': float(best['score_mean']),
            'train_inner_fold_scores': ([float(s) for s in fs]
                                        if fs else None),
            'n_inner_folds_scored': best.get('n_folds_scored'),
            'n_inner_folds_asked': best.get('n_folds_asked'),
            'n_features_search': best.get('n_features'),
        }
        try:
            X_tr_h, X_va_h, fi = build_fold_hybrid(
                tr_reps, va_reps, y_tr, bp,
                augmentations_train=aug_tr if cond != 'none' else None,
                augmentations_val=aug_va if cond != 'none' else None,
                groups=np.asarray(groups)[tr] if groups is not None else None,
                # This fold's search measured these on this fold's training
                # rows; never the whole dataset's (dirty-run bug 1).
                precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                **fold_hybrid_scoring_kwargs(
                    bp, model=_make(), metric_fn=metric_fn,
                    lower_is_better=lower_is_better))
            cell['n_features'] = int(np.asarray(X_tr_h).shape[1])
            cell['features_by_rep'] = {
                r: int(len(info['selected_indices']))
                for r, info in fi.items()
                if isinstance(info, dict) and 'selected_indices' in info}
            cell['held_out_score'] = _fit_heldout(
                _make(), X_tr_h, y_tr, X_va_h, y_va, metric_fn)
        except Exception as e:
            cell['held_out_score'] = None
            cell['error'] = f'{type(e).__name__}: {e}'
        rec['per_budget'].append(cell)
        _ho = cell.get('held_out_score')
        print(f"      fold {k} {model_key} b={b}: {metric_name} train "
              f"{cell['train_score']:.4f}, held-out "
              f"{'n/a' if _ho is None else f'{_ho:.4f}'} "
              f"[{cell['importance']}/{cell['allocation']}/"
              f"{cell['evaluator_label']}, rep_count {cell['rep_count']}]"
              + (f" build failed: {cell['error']}" if 'error' in cell else ''))

    # The optimal budget is picked on the training-row score, the analogue of
    # the old design's pick on the search's mean (which then included the
    # held-out rows). The held-out score at that budget is reported with it.
    cmp_fn = min if lower_is_better else max
    opt = cmp_fn(rec['per_budget'], key=lambda c: c['train_score'])
    rec['optimal_budget'] = {
        'budget': opt['budget'], 'chosen_on': 'train_score',
        'train_score': opt['train_score'],
        'held_out_score': opt.get('held_out_score'),
        'allocation': opt['allocation'], 'importance': opt['importance'],
        'evaluator': opt['evaluator'],
        'evaluator_label': opt['evaluator_label'],
        'rep_count': opt['rep_count'], 'aug_condition': opt['aug_condition'],
        'n_features': opt.get('n_features'),
    }
    print(f"      fold {k} {model_key} >> optimal b={opt['budget']} "
          f"(train {opt['train_score']:.4f})")
    return rec


def run(data, config, output_dir, dataset_name, families=None):
    """Exp 6: vary complexity within each model family, per outer fold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: WITHIN-FAMILY MODEL COMPLEXITY GRADIENT")
    print("=" * 70)

    task_type = data.get('task_type')
    if task_type != 'regression':
        # Reason text corrected 2026-10-05. get_model_family_variants has had
        # classifier variants since the CXA-6 fix, but this file still runs
        # regression only: it calls get_model_family_variants() with no
        # task_type, hands 'regression' to _resolve_models and every search,
        # and has no entry for the classification-only 'Logit' family in
        # _FAMILY_TO_CANONICAL. Kept regression-only, 2026-10-05 (tracker
        # C06f, delegated: wiring classification is new compute).
        print(f"  Skipping: c06 is not wired for classification "
              f"(task_type={task_type}).")
        return {'skipped': True,
                'reason': ('c06 passes task_type=regression to every search '
                           'and has no entry for the Logit family; the '
                           'classifier variants exist but are not wired here'),
                'task_type': task_type}

    # Metric name and direction from the dataset's registry, never assumed.
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = np.asarray(data['labels'], dtype=np.float64)
    groups = data['groups']
    n_samples = len(labels)
    n_outer = int(config['n_splits'])

    # The decided general-search grid (R1, R3). Budgets are config['budgets']
    # (decided_config already set them) narrowed to the decided list.
    grid = decided_search(dataset_name, 'general')
    decided_models = list(grid['models'])
    budgets = [b for b in config['budgets'] if b in grid['budgets']]
    if not budgets:
        raise ValueError(
            f"{dataset_name}: config budgets {config['budgets']} hold none of "
            f"the decided budgets {grid['budgets']}")
    grid = dict(grid, budgets=sorted(budgets))
    # c06 used to widen a grid of fewer than 4 budgets with data-relative
    # values (5%..50% of the rows). Those are outside the decided range, so
    # the widening is removed, 2026-10-05. On a dataset with 3 decided budgets
    # each setting's optimal budget is one of those 3.
    print(f"  Decided grid: models {decided_models}; budgets {grid['budgets']} "
          f"features; importance {grid['importance_methods']}; allocation "
          f"{grid['allocation_methods']}; evaluators "
          f"{grid['internal_evaluators']}; representation counts "
          f"{grid['rep_counts']}; pruning off")

    # Representations: data['reps'] is DATASET_REPS[ds]['general'] under
    # run_all. Not filtered here (rule 2.4 step 9); names outside the general
    # pool, possible only on a direct --reps run, are printed and recorded.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _general = list((DATASET_REPS.get(dataset_name) or {}).get('general') or [])
    _outside = sorted(r for r in reps if r not in _general)
    if _outside:
        print(f"  WARNING: representations outside DATASET_REPS['{dataset_name}']"
              f"['general']: {_outside}")

    # Augmentation (handoff 2.3 item 4): only the decided conditions of the
    # general search. That is ['none'] on every dataset; ginkgo_colloidal and
    # ginkgo_hic keep 'uncapped' on the distance search only, and c06 runs no
    # distance search. c06 used to search 'uncapped' alone wherever the
    # dataset had augmentation representations.
    aug_conds = list(grid['aug_conditions'])
    augs = data.get('augmentations')
    if augs and any(c != 'none' for c in aug_conds):
        aug_for_search = augs
        aug_note = (f"augmentation conditions {aug_conds} (decided for the "
                    f"general search)")
    else:
        aug_for_search = None
        aug_note = ("no augmentation: DECISIONS.md Step 8 keeps it only on the "
                    "distance search of ginkgo_colloidal and ginkgo_hic, and "
                    "c06 runs no distance search"
                    if augs else
                    "no augmentation representations on this dataset")
    print(f"  Augmentation: {aug_note}")

    all_families = get_model_family_variants()

    # KIRBY_C06_FAMILIES: comma-separated family names, same effect as
    # --families. Unset, every family runs. An explicit --families on the
    # command line wins over it.
    _fams_env = os.environ.get('KIRBY_C06_FAMILIES', '').strip()
    if _fams_env and not families:
        families = [f.strip() for f in _fams_env.split(',') if f.strip()]
        print(f"  KIRBY_C06_FAMILIES={_fams_env}: running {families} only")

    if families:
        _unknown = [f for f in families if f not in all_families]
        if _unknown:
            raise ValueError(
                f"--families named {_unknown}, which "
                f"get_model_family_variants does not return. It returns "
                f"{sorted(all_families)}. c06 used to filter these out and "
                f"save an empty result with no error.")
        all_families = {k: v for k, v in all_families.items() if k in families}

    run_families, not_run, no_family = _family_plan(
        all_families, decided_models, 'regression', n_samples, dataset_name)
    for fam, why in not_run.items():
        print(f"  Not run: {fam}: {why}")
    for m in no_family:
        print(f"  Decided model {m} has no capacity family in "
              f"get_model_family_variants; nothing runs for it")

    # KIRBY_C06_VARIANTS_PER_FAMILY: how many capacity settings of each family
    # are searched. Unset (the shipped value) every family keeps all four.
    # Set to k, each family keeps k settings evenly spaced along its capacity
    # index INCLUDING both ends. The rho below needs at least 3 points, so 3
    # is the floor. A local CONDITION cut: every setting still trains on every
    # training row.
    _vpf = os.environ.get('KIRBY_C06_VARIANTS_PER_FAMILY', '').strip()
    if _vpf:
        _k = int(_vpf)
        if _k < 3:
            raise ValueError(
                f"KIRBY_C06_VARIANTS_PER_FAMILY={_k}: spearmanr over capacity "
                f"and optimal budget needs at least 3 variants, and c06 records "
                f"'insufficient_variants' below that. Use 3 or more.")
        for _fam in list(run_families):
            _vs = run_families[_fam]
            if len(_vs) > _k:
                _idx = sorted({round(i * (len(_vs) - 1) / (_k - 1))
                               for i in range(_k)})
                _kept = [_vs[i] for i in _idx]
                print(f"  {_fam}: {len(_vs)} capacity settings cut to {_k} "
                      f"({[v[0] for v in _kept]}); "
                      f"dropped {[v[0] for v in _vs if v not in _kept]}")
                run_families[_fam] = _kept

    fam_evaluators = {f: _family_evaluators(f, grid['internal_evaluators'])
                      for f in run_families}
    grid_record = convert_numpy({
        'budgets': grid['budgets'],
        'importance_methods': grid['importance_methods'],
        'allocation_methods': grid['allocation_methods'],
        'rep_counts': grid['rep_counts'],
        'aug_conditions': aug_conds if aug_for_search else ['none'],
        'families': {f: [v[0] for v in vs] for f, vs in run_families.items()},
        'internal_evaluators': fam_evaluators,
    })

    results = {
        'experiment_identity': C06_RESULT_IDENTITY,
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'per outer fold, per capacity setting, per budget: the best '
            "configuration by its score on that fold's training rows (mean "
            "over the search's inner folds); rebuilt on the training rows and "
            'scored on the held-out rows. The optimal budget is chosen on the '
            'training-row score.'),
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_outer_folds': n_outer,
        'decided_models': decided_models,
        'families_run': sorted(run_families),
        'families_not_run': not_run,
        'decided_models_without_family': no_family,
        'augmentation': aug_note,
        'representations': sorted(reps),
        'representations_outside_general': _outside,
        'grid': grid_record,
        'per_fold': [],
        'families': {},
    }
    if not run_families:
        print("  No capacity family matches a decided model; nothing to run.")
        results['skipped'] = True
        results['reason'] = ('no capacity family matches the decided models '
                             f'{decided_models}')
        return results

    folds = nested_cv_outer_folds(reps, labels, groups, n_outer, _SEED,
                                  'regression')
    # n_outer stays the layout's count (search names, namespaces); the folds
    # run may be the first few of it (OUTER_FOLDS_RUN, 2026-10-07).
    folds = first_outer_folds(folds, 'c06', dataset_name)
    results['n_outer_folds_run'] = len(folds)
    fingerprint = input_fingerprint(data, config)
    ck_path = _checkpoint_path(output_dir, dataset_name)
    done = _load_checkpoint(ck_path, fingerprint, grid_record)

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        if k in done:
            per_fold.append(done[k])
            continue
        print(f"\n  Outer fold {k + 1}/{n_outer}: {len(tr)} training rows, "
              f"{len(va)} held-out rows")
        fold_rec = {'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
                    'cache_namespace': nested_cv_cache_namespace(
                        k, n_outer, _SEED),
                    'selection_scope': 'fold_train_rows', 'families': {}}
        for fam_name, variants in run_families.items():
            print(f"\n    Family: {fam_name} ({len(variants)} settings, "
                  f"internal evaluators={fam_evaluators[fam_name]})")
            if fam_name not in _FAMILY_TO_EVALUATOR:
                print(f"    {fam_name} has no plain model name among the "
                      f"internal evaluators hybrid.py can fit, so no matched "
                      f"evaluator is searched for it.")
            fam_variants = {}
            for var_name, var_factory, capacity_idx in variants:
                fam_variants[var_name] = _run_variant_on_fold(
                    reps=reps, labels=labels, groups=groups, tr=tr, va=va,
                    k=k, n_outer=n_outer, dataset_name=dataset_name,
                    metric_name=metric_name, metric_fn=metric_fn,
                    lower_is_better=lower_is_better, fam_name=fam_name,
                    var_name=var_name, var_factory=var_factory,
                    capacity_idx=capacity_idx, grid=grid,
                    fam_evaluators=fam_evaluators[fam_name],
                    aug_for_search=aug_for_search)
            corr = _capacity_rho(fam_variants)
            fold_rec['families'][fam_name] = {
                'variants': fam_variants,
                'capacity_budget_correlation': corr}
            print(f"    >> fold {k} {fam_name}: capacity vs optimal budget "
                  + (f"rho={corr['rho']:.3f}" if corr['status'] == 'measured'
                     else f"UNDEFINED ({corr['status']})")
                  + f" over {corr['n_variants']} settings")
        per_fold.append(convert_numpy(fold_rec))
        # Written after every fold (dirty-run bug 12), so a run stopped part
        # way keeps the folds it finished.
        _write_checkpoint(ck_path, fingerprint, grid_record, per_fold)

    results['per_fold'] = per_fold

    # Per family, across folds: the per-fold values side by side, and their
    # range. No mean-of-folds test, no p-value, no interval.
    for fam_name, variants in run_families.items():
        fam_sum = {'heatmap': {}, 'optimal_budget': {},
                   'internal_evaluators': fam_evaluators[fam_name]}
        rhos = []
        for fr in per_fold:
            fd = fr['families'].get(fam_name) or {}
            c = fd.get('capacity_budget_correlation') or {}
            rhos.append({'fold': fr['fold'], 'rho': c.get('rho'),
                         'status': c.get('status')})
            for var_name, vr in (fd.get('variants') or {}).items():
                hm = fam_sum['heatmap'].setdefault(var_name, {})
                for cell in vr.get('per_budget') or []:
                    d = hm.setdefault(cell['budget'], {
                        'train_score_by_fold': [],
                        'held_out_score_by_fold': []})
                    d['train_score_by_fold'].append(
                        {'fold': fr['fold'], 'score': cell['train_score']})
                    d['held_out_score_by_fold'].append(
                        {'fold': fr['fold'],
                         'score': cell.get('held_out_score')})
                ob = vr.get('optimal_budget') or {}
                fam_sum['optimal_budget'].setdefault(var_name, {
                    'capacity_index': vr.get('capacity_index'),
                    'by_fold': []})['by_fold'].append(
                    {'fold': fr['fold'], 'budget': ob.get('budget'),
                     'train_score': ob.get('train_score'),
                     'held_out_score': ob.get('held_out_score'),
                     'error': vr.get('error')})
        measured = [r['rho'] for r in rhos
                    if r['status'] == 'measured' and r['rho'] is not None]
        fam_sum['capacity_budget_correlation'] = {
            'by_fold': rhos,
            'n_folds_measured': len(measured),
            'spread': ('one fold, no spread' if n_outer == 1 else
                       ({'min': float(min(measured)),
                         'max': float(max(measured))}
                        if len(measured) >= 2 else
                        f'{len(measured)} fold(s) measured, no spread')),
        }
        results['families'][fam_name] = fam_sum

    # Summary print: per family and setting, the optimal budget on each fold.
    print("\n  SUMMARY: optimal budget (features) by setting, per outer fold")
    for fam_name, fam_sum in results['families'].items():
        for var_name, info in fam_sum['optimal_budget'].items():
            per = ', '.join(f"f{o['fold']}={o['budget']}"
                            for o in info['by_fold'])
            print(f"    {fam_name}/{var_name}: {per}")
        c = fam_sum['capacity_budget_correlation']
        per_rho = ', '.join(
            f"f{r['fold']}=" + (f"{r['rho']:.3f}" if r['status'] == 'measured'
                                else str(r['status']))
            for r in c['by_fold'])
        print(f"    {fam_name} rho per fold: {per_rho}; spread: {c['spread']}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C06: Within-Family Complexity Gradient',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--families', type=str, nargs='+', default=None,
                        help='Model families to test (RF, Lasso, kNN, DNN, '
                             'XGBoost, SVM, GP). Ridge was replaced by Lasso '
                             'on 2026-07-18.')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, families=args.families)
    save_experiment_results(results, args.output, 'c06_within_family_gradient', args.dataset)
