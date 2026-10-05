#!/usr/bin/env python3
"""C24: Blueprint Transfer Test — Cross-Fold Generalization of Feature Selection.

Paper: Complexity / Discussion (feature selection overfitting defense)
Conclusion: Blueprints discovered on one fold generalize to other folds,
    demonstrating that feature selection is not overfit to training data.

Dependencies: None (standalone). Every search is a per-fold search on the
    dataset's decided grid (2026-10-05); no stored whole-dataset search is read.

⛔ WHAT THE CODE ACTUALLY MEASURES, WHICH IS NOT WHAT THE THREE LINES BELOW
    USED TO SAY. Step 2 does NOT apply fold k's Step-1 blueprint to fold j.
    The recipe used on the off-diagonal comes from a FRESH search on the rows
    left after dropping BOTH fold k and fold j, so that the selection never
    sees the target fold's rows.

    That is a deliberate choice and the leak-free one: fold k's own Step-1
    recipe was searched on every fold except k, which INCLUDES fold j's rows,
    so applying it to fold j would score a selection on rows it had seen. But
    it means the diagonal and the off-diagonal differ in HOW MANY ROWS THE
    SEARCH SAW -- one fold's worth -- not in whose recipe was applied. Whether
    that is the experiment you want is a science decision and has not been
    made; the code, the printed reading and delta_meaning now describe the same
    thing, which they did not before.

In one line (2026-10-05, owner delegated): each off-diagonal cell is a fresh
search with one more fold left out. Nothing is transferred from the source fold.

Aligned 2026-10-05 (handoff 2.4; selection by design, 2.5 group B):
    - Models: the dataset's two decided models (decided_models(ds, 'general')).
      The whole comparison is run once per model, so every number in one
      model's transfer table comes from that model's recipes scored with that
      model. It used to keep, per fold, the single best of the five engine
      models, so different target folds could be scored with different models.
    - Searches: the diagonal's search is the standard per-fold search on that
      fold's training rows (fold_searches; shared with every other experiment
      that searches the same fold). The off-diagonal's search runs on the rows
      outside both folds, under its own cache namespace. Both are on the
      decided grid, pruning off, augmentation as decided_aug_conditions keeps.
    - Every rebuild passes the per-representation scores its own search
      measured on the same rows (dirty-run bug 1).
    - Outer folds: config['n_splits'], through nested_cv_outer_folds. With one
      fold only the diagonal exists; with two, removing both folds leaves no
      rows. The transfer comparison needs at least 3 outer folds and says so.

Design (as run):
    For each CV fold k, for each decided model:
        1. Discover the optimal blueprint on fold k's training data.
        2. For each other fold j: search a recipe on the rows outside BOTH k
           and j, build the hybrid and fit the model on those rows, apply it
           to fold j's held-out rows.
        3. Measure performance vs fold j's own-fold blueprint.

    If feature selection is NOT overfit:
        - Own-fold and cross-fold blueprints should perform similarly.
        - Small delta (own - cross) = generalizable selection.

    If feature selection IS overfit:
        - Cross-fold blueprints will degrade significantly.
        - Large delta = selection is training-data-specific.

    Also reports Nogueira stability index (phi) across fold blueprints, per
    model; its standard error is not computed (removed 2026-10-05, owner
    delegated).

Usage:
    python c24_blueprint_transfer.py --dataset logd --mode full
    python c24_blueprint_transfer.py --dataset caco2 --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    build_fold_hybrid, fold_hybrid_scoring_kwargs, fold_searches,
    get_config, create_experiment_parser,
    convert_numpy,
    get_model_factories,
    get_domain_metric,
)
from experiments.decided_grid import decided_models, decided_search
from kirby.find_best_hybrid import (
    _predict_for_metric, _fit_model,
    nested_cv_outer_folds, nested_cv_cache_namespace,
)
from kirby.scaling import FeatureScaler

# The outer split's seed. The same 42 fold_searches defaults to, so the
# diagonal's per-fold search is the one every other experiment's fold search
# is filed under (ocv{n}f{i}s42).
SEED = 42

# Stored in every result file (2026-10-05). The per-fold, decided-model
# rewrite changes what each number in the file is, so a file without this
# value was written by the old code and is not comparable.
RESULT_IDENTITY = 'c24_per_fold_decided_models_2026_10_05'

# =============================================================================
# LOCAL CONDITION CUTS
# =============================================================================
# 2026-10-05: the local budget cut (KIRBY_C24_BUDGETS, and the "largest
# budget only" default under KIRBY_LOCAL_GRID) is removed. Every c24 search
# now takes the dataset's decided budgets, like every other search (R3); a
# per-experiment budget list is no longer this file's to set.
#
# ⛔ NOTHING BELOW CUTS A ROW OR A CONDITION. What is saved is how many times
# the same feature-importance vectors are recomputed.
import time as _time

# Reuse each row-set's feature-importance vectors across c24's own hybrid
# builds. Importance depends on the row set and the importance method and on
# nothing else -- not the allocation, not the model -- and build_fold_hybrid
# recomputes it on every call, at 93 s of every 150 s measured on
# flab_thermo_tresanco on 2026-09-24. Since 2026-10-05 c24 builds once per
# decided model on every row set, so the two models' builds on one row set
# share one computation when this is on.
#
# This changes no number: it hands build_fold_hybrid the same vectors it would
# have computed, over the same rows create_hybrid ranks features on (the
# training side of internal_holdout_split). Same pattern and same reasoning as
# c16's KIRBY_C16_REUSE_FOLD_WORK; f31_internal_evaluator_tuning.py:548 is the
# existing check that precomputed_importance gives the identical allocation.
#
# Default: ON for a local run, OFF everywhere else, so a cluster run recomputes
# exactly as it did before.
_C24_REUSE_FOLD_WORK = os.environ.get(
    'KIRBY_C24_REUSE_FOLD_WORK',
    '1' if os.environ.get('KIRBY_LOCAL_GRID') else '0') == '1'


def _c24_fold_importance(cache, row_key, reps_subset, y, groups_subset,
                         blueprint):
    """Importance vectors for one row set, computed once and kept.

    Returns None when reuse is off or the recipe names no importance method,
    which leaves build_fold_hybrid computing its own exactly as before.
    """
    if cache is None or blueprint is None:
        return None
    method = getattr(blueprint, 'importance_method', None)
    if not method:
        return None
    key = (row_key, method)
    if key not in cache:
        # ⛔ THE ROWS MATTER. create_hybrid carves an internal holdout out of
        # the rows it is given and ranks features on the TRAINING side of that
        # carve. Computing importance over the whole row set would rank
        # features on rows create_hybrid holds out, which is a different
        # quantity, not a faster one.
        from kirby.hybrid import (compute_feature_importance,
                                  internal_holdout_split)
        inner_tr, _ = internal_holdout_split(len(y), y, groups_subset)
        cache[key] = compute_feature_importance(
            {r: v[inner_tr] for r, v in reps_subset.items()},
            y[inner_tr], method=method)
    return cache[key]


def nogueira_stability(feature_sets, total_features):
    """Compute the Nogueira et al. (2018) stability index.

    Args:
        feature_sets: list of sets, each containing (rep_name, feature_idx) tuples
            selected on each fold/resample.
        total_features: int, total number of features available.

    Returns:
        float in [-1, 1]. 1 = identical selection every time. nan with fewer
        than 2 feature sets. The standard error is no longer computed
        (2026-10-05, owner delegated: no intervals at 3-5 folds).

    Reference:
        Nogueira, Sechidis & Brown (2018). "On the Stability of Feature
        Selection Algorithms." JMLR 18(174), 1-54.
    """
    M = len(feature_sets)
    if M < 2 or total_features == 0:
        return float('nan')

    # Convert sets to binary indicator vectors
    # Build a unified feature index
    all_features = set()
    for fs in feature_sets:
        all_features.update(fs)
    all_features = sorted(all_features)
    feat_to_idx = {f: i for i, f in enumerate(all_features)}

    # ⛔ d IS THE POOL WIDTH, NOT THE UNION OF WHAT WAS SELECTED. It used to be
    # len(all_features), which made the denominator depend on the selections
    # being measured and made the perfectly stable case UNDEFINED: five folds
    # picking the identical 200 features give a union of 200, so k_bar == d,
    # the k_bar == d guard below fires and phi comes back nan instead of 1.
    # total_features was passed in for exactly this and was only ever used in
    # the `== 0` check above.
    d = int(total_features)
    n_union = len(all_features)
    if d < n_union:
        # Can only happen if the caller's count is wrong; trust the data.
        d = n_union
    if d == 0:
        return float('nan')

    # Binary matrix Z over the UNION, then padded: features no fold ever
    # selected have p_f = 0, contribute 0 to the numerator, and matter only
    # through d in the denominator -- which is the point of using the pool
    # width there.
    Z = np.zeros((M, n_union), dtype=float)
    for i, fs in enumerate(feature_sets):
        for f in fs:
            if f in feat_to_idx:
                Z[i, feat_to_idx[f]] = 1.0

    # Average number of selected features
    k_bar = Z.sum(axis=1).mean()

    if k_bar == 0 or k_bar == d:
        return float('nan')

    # Per-feature selection frequency, over the union. The d - n_union features
    # nobody picked are all-zero rows of p_f and add nothing to the sum.
    p_f = Z.mean(axis=0)  # (n_union,)

    # Nogueira, Sechidis & Brown (2018), Eq. 4:
    #   phi = 1 - [ (1/d) * sum_f s_f^2 ] / [ (k_bar/d) * (1 - k_bar/d) ]
    # with s_f^2 = (M/(M-1)) * p_f * (1 - p_f). The 1/d in the numerator and
    # the 1/d inside the denominator CANCEL, leaving
    #   phi = 1 - [ sum_f (M/(M-1)) p_f (1-p_f) ] / [ k_bar * (1 - k_bar/d) ]
    #
    # ⛔ THE DENOMINATOR CARRIED AN EXTRA FACTOR OF d. With d in the thousands
    # of features that pushed phi to just under 1 whatever the selections were,
    # and the threshold below then printed HIGH stability every time. Checked
    # numerically over 2,000 features and 5 folds picking 200 each:
    #   five folds pick        old formula   this formula
    #   the identical 200         1.000000       1.000000
    #   200 sharing half          0.999722       0.444444
    #   200 entirely disjoint     0.999444      -0.111111
    # The comment here stated the same wrong formula, so comment and code
    # agreed with each other and both disagreed with the paper.
    numerator = (M / (M - 1)) * np.sum(p_f * (1 - p_f))
    denominator = k_bar * (1 - k_bar / d)

    if denominator == 0:
        return float('nan')

    return float(1 - numerator / denominator)


def _recipe_fields(bp, aug_cond):
    """What one hybrid recipe is, as stored beside its scores (2026-10-05)."""
    return {
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        # In features.
        'budget': getattr(bp, 'budget', None),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
        # None = the whole pool of representations.
        'rep_count': getattr(bp, 'rep_count', None),
        'aug_condition': aug_cond,
    }


def _best_entry_for_model(searches, model, lower_is_better):
    """(aug_condition, best entry) for `model` over the conditions searched.

    Reads only `model`'s own entry. get_best_for_downstream falls back to
    another model's recipe when the asked model has none; here a model with no
    entry gives (None, None) and the caller records it, so a recipe is never
    scored under a model it was not chosen for (2026-10-05).
    """
    cands = []
    for cond, search in (searches or {}).items():
        if search is None:
            continue
        entry = (getattr(search, 'best_by_model', None) or {}).get(model)
        if not entry or entry.get('score_mean') is None:
            continue
        if not np.isfinite(entry['score_mean']):
            continue
        cands.append((cond, entry))
    if not cands:
        return None, None
    # Same order get_fold_best_search uses: score first, then 'none' before an
    # augmentation condition on a tie.
    aug_pref = {'none': 0, 'uncapped': 1, 'capped': 2}
    cands.sort(key=lambda c: (c[1]['score_mean'] if lower_is_better
                              else -c[1]['score_mean'],
                              aug_pref.get(c[0], 99)))
    return cands[0]


def _search_failures(searches, where):
    """The caught failures of one per-fold search, without tracebacks."""
    out = []
    for rec in getattr(searches, 'failures', None) or []:
        out.append({**where,
                    'tier': rec.get('tier'),
                    'aug_condition': rec.get('aug_condition'),
                    'error': f"{rec.get('error_type')}: {rec.get('error')}"})
    return out


def _selected_features(fi):
    """{(representation, feature index)} a build selected, for phi."""
    selected = set()
    for rep_name, rep_info in (fi or {}).items():
        if isinstance(rep_info, dict) and 'selected_indices' in rep_info:
            for idx in rep_info['selected_indices']:
                selected.add((rep_name, int(idx)))
    return selected


def _features_by_rep(fi):
    """Features the build took from each representation."""
    return {r: int(len(info['selected_indices']))
            for r, info in (fi or {}).items()
            if isinstance(info, dict) and 'selected_indices' in info}


def run(data, config, output_dir, dataset_name):
    """C24: Blueprint transfer test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 24: BLUEPRINT TRANSFER TEST")
    print("=" * 70)
    # Whatever cuts are in force, printed beside the run that used them, so no
    # result exists without them.
    print(f"  local condition cuts: reuse_fold_importance={_C24_REUSE_FOLD_WORK}")
    _t0 = _time.time()

    def _el():
        return _time.time() - _t0

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    # Use None (not {}) so downstream `if augs:` checks correctly skip the
    # aug-handling path when no augmentations exist for this dataset.
    augs = data.get('augmentations') or None
    task_type = data['task_type']

    # The dataset's two models (DECISIONS.md Step 1). knn is not one of them
    # here: c24 runs no distance search.
    models = list(decided_models(dataset_name, 'general'))
    print(f"  models: {models} (decided for {dataset_name})")

    total_features = sum(v.shape[1] for v in reps.values())

    # cv_split RAISES (no random-split fallback, feedback_no_random_splits) when
    # the dataset has fewer distinct groups than the requested fold count. If
    # <2 distinct groups exist, a leak-free grouped CV is impossible -> return
    # a clean skip (no 'error' key) rather than crash.
    n_distinct_groups = 0 if groups is None else len(np.unique(groups))
    if n_distinct_groups < 2:
        print(f"  Skipping: blueprint transfer needs >=2 distinct groups for a "
              f"leak-free grouped CV, but got {n_distinct_groups}.")
        return {'skipped': True, 'reason': 'insufficient_groups',
                'n_distinct_groups': n_distinct_groups,
                'result_identity': RESULT_IDENTITY}

    # The outer fold count is the run's setting and nothing else (handoff 2.4
    # step 14). Fewer groups than folds reduces the count to the group count,
    # as before; nested_cv_outer_folds would do the same with only a warning,
    # and the per-fold cache namespace has to carry the count actually used.
    n_folds_requested = int(config['n_splits'])
    n_outer = n_folds_requested
    if n_outer > 1 and n_distinct_groups < n_outer:
        print(f"  Only {n_distinct_groups} distinct groups for {n_outer}-fold "
              f"CV; reducing to {n_distinct_groups} folds (group-aware, no "
              f"random fallback).")
        n_outer = n_distinct_groups
    folds = nested_cv_outer_folds(reps, labels, groups, n_outer, SEED,
                                  task_type)
    n_outer = len(folds)

    # --- Step 1: per fold, per model: own-fold recipe and held-out score ---
    print(f"\n  Step 1: Discover blueprints on each of {n_outer} folds")

    per_fold = []          # one record per outer fold (handoff 2.4 step 11)
    search_failures = []   # every caught search failure, with fold and rows
    fold_errors = []       # (fold, model, stage, error)
    # model -> fold -> {...} for the transfer analysis below
    own = {m: {} for m in models}
    # (row-set key, importance method) -> importance vectors. Empty and unused
    # when _C24_REUSE_FOLD_WORK is off.
    _imp_cache = {} if _C24_REUSE_FOLD_WORK else None

    def _build_and_score(fit_idx, apply_idx, bp, aug_cond, model_name,
                         imp_key):
        """Build bp's hybrid on fit_idx rows, fit model_name there, and
        predict apply_idx rows. Returns (predictions, width, fi)."""
        fit_reps = {k: v[fit_idx] for k, v in reps.items()}
        app_reps = {k: v[apply_idx] for k, v in reps.items()}
        use_augs = augs if (augs and aug_cond != 'none') else None
        y_fit = labels[fit_idx]
        g_fit = groups[fit_idx]
        factories = get_model_factories([model_name], task_type=task_type,
                                        n_samples=len(fit_idx))
        if model_name not in factories:
            raise ValueError(f"model {model_name} skipped at "
                             f"{len(fit_idx)} training rows")
        factory = factories[model_name]
        X_fit, X_app, fi = build_fold_hybrid(
            fit_reps, app_reps, y_fit, bp,
            augmentations_train=({k: v[fit_idx] for k, v in use_augs.items()}
                                 if use_augs else None),
            augmentations_val=({k: v[apply_idx] for k, v in use_augs.items()}
                               if use_augs else None),
            groups=g_fit,
            # The search that chose bp measured these on these same rows;
            # reused so the budget split is the one it settled on (dirty-run
            # bug 1, 2026-10-05).
            precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
            precomputed_importance=_c24_fold_importance(
                _imp_cache, imp_key, fit_reps, y_fit, g_fit, bp),
            **fold_hybrid_scoring_kwargs(
                bp, model=factory(), metric_fn=metric_fn,
                lower_is_better=lower_is_better))
        # 0/1 features pass through unscaled, every other feature is
        # standardised on the fit rows (2.3 item 17, 2026-10-05). It was a
        # whole-matrix StandardScaler.
        scaler = FeatureScaler()
        X_fit = scaler.fit_transform(X_fit)
        X_app = scaler.transform(X_app)
        model = factory()
        # The search's own fit: neural models get the groups for their
        # early-stopping carve.
        _fit_model(model, X_fit, y_fit, groups=g_fit)
        preds = _predict_for_metric(model, X_app, metric_fn)
        return preds, int(X_fit.shape[1]), fi

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        ns = nested_cv_cache_namespace(fold_i, n_outer, SEED)
        fold_rec = {
            'fold': int(fold_i),
            'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
            'cache_namespace': ns,
            'selection_scope': 'fold_train_rows',
            'metric': metric_name,
            'lower_is_better': bool(lower_is_better),
            # c24 scores no single representation: its comparison is one
            # recipe per fold against a recipe chosen on fewer folds
            # (approved by design, handoff 2.5).
            'base_representations': None,
            'models': {},
        }
        _ts = _time.time()
        searches = fold_searches(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            augmentations=augs, task_type=task_type, metric=metric_name,
            seed=SEED)
        fails = _search_failures(searches, {'fold': int(fold_i),
                                            'rows': 'fold_train_rows'})
        search_failures.extend(fails)
        fold_rec['search_failures'] = fails
        print(f"    [t={_el():.0f}s] fold {fold_i} search done in "
              f"{_time.time() - _ts:.0f}s")

        for m in models:
            aug_cond, entry = _best_entry_for_model(searches, m,
                                                    lower_is_better)
            if entry is None:
                print(f"    Fold {fold_i} {m}: no scored recipe, skipped")
                fold_errors.append({'fold': int(fold_i), 'model': m,
                                    'stage': 'search',
                                    'exception': 'no scored recipe for model'})
                fold_rec['models'][m] = {'error': 'no scored recipe'}
                continue
            bp = entry['blueprint']
            rec = {'hybrid': _recipe_fields(bp, aug_cond),
                   # Mean over the search's inner folds, inside this outer
                   # fold's training rows.
                   'training_rows_score': float(entry['score_mean'])}
            try:
                preds, width, fi = _build_and_score(
                    train_idx, val_idx, bp, aug_cond, m, ('own', fold_i))
                score = float(metric_fn(labels[val_idx], preds))
            except Exception as e:
                print(f"    Fold {fold_i} {m}: own-fold eval failed: {e}")
                fold_errors.append({'fold': int(fold_i), 'model': m,
                                    'stage': 'own_fold_eval',
                                    'exception': f'{type(e).__name__}: {e}'})
                rec['error'] = f'{type(e).__name__}: {e}'
                fold_rec['models'][m] = rec
                continue
            rec['heldout_score'] = score
            rec['width_built'] = width
            rec['features_by_rep'] = _features_by_rep(fi)
            fold_rec['models'][m] = rec
            own[m][fold_i] = {'blueprint': bp, 'aug_condition': aug_cond,
                              'score': score, 'width': width,
                              'features': _selected_features(fi)}
            print(f"    Fold {fold_i} {m}: budget={bp.budget}, "
                  f"alloc={bp.allocation_method}, imp={bp.importance_method}, "
                  f"eval={bp.internal_evaluator}, "
                  f"rep_count={getattr(bp, 'rep_count', None)}, "
                  f"{width} features; own-fold {metric_name}: {score:.4f}")
        per_fold.append(fold_rec)

    # --- Step 2: off-diagonal cells ---
    # One search per unordered pair of folds: the rows outside both are the
    # same for A->B and B->A. The hybrid is built and the model fitted on
    # those rows once per model, and applied to both folds' held-out rows, so
    # the two cells of a pair share one recipe, one feature set and one fitted
    # model and differ only in the rows scored.
    print(f"\n  Step 2: per pair of folds, search on the rows outside both, "
          f"score on each fold's held-out rows")
    transfer = {m: {} for m in models}       # (src, tgt) -> score
    provenance = {m: {} for m in models}     # 'src->tgt' -> {...}
    transfer_note = None
    if n_outer < 3:
        transfer_note = (
            f'{n_outer} outer fold(s): the transfer comparison needs at '
            f'least 3, because each off-diagonal cell searches the rows '
            f'outside two folds and two folds leave none'
            + ('; one fold, no spread' if n_outer == 1 else ''))
        print(f"    {transfer_note}")
    else:
        for a in range(n_outer):
            for b in range(a + 1, n_outer):
                if not any(a in own[m] and b in own[m] for m in models):
                    continue
                pair_mask = np.ones(len(labels), dtype=bool)
                pair_mask[folds[a][1]] = False
                pair_mask[folds[b][1]] = False
                pair_idx = np.where(pair_mask)[0]
                where = {'fold_pair': [int(a), int(b)],
                         'rows': 'outside both folds'}
                if len(np.unique(groups[pair_idx])) < 2:
                    msg = (f"only {len(np.unique(groups[pair_idx]))} distinct "
                           f"group(s) left after dropping folds {a} and {b}")
                    print(f"    pair {a},{b}: {msg}, skipped")
                    fold_errors.append({**where, 'stage': 'pair_search',
                                        'exception': msg})
                    continue
                # Its own namespace: these rows are not an outer fold's
                # training rows, so the search must not be filed under one.
                pair_ns = f'c24_ocv{n_outer}drop{a}_{b}s{SEED}'
                _tp = _time.time()
                pair_searches = fold_searches(
                    reps, labels, groups, train_idx=pair_idx, fold_i=a,
                    n_outer_folds=n_outer, dataset_name=dataset_name,
                    augmentations=augs, task_type=task_type,
                    metric=metric_name, seed=SEED, cache_namespace=pair_ns)
                search_failures.extend(_search_failures(pair_searches, where))
                print(f"    [t={_el():.0f}s] pair search dropping folds "
                      f"[{a}, {b}] done in {_time.time() - _tp:.0f}s")
                for m in models:
                    aug_cond, entry = _best_entry_for_model(
                        pair_searches, m, lower_is_better)
                    if entry is None:
                        fold_errors.append({**where, 'model': m,
                                            'stage': 'pair_recipe_search',
                                            'exception': 'no scored recipe '
                                                         'for model'})
                        print(f"    pair {a},{b} {m}: no scored recipe, "
                              f"cells skipped")
                        continue
                    bp = entry['blueprint']
                    tgts = [t for t in (a, b) if t in own[m]]
                    if not tgts:
                        continue
                    apply_idx = np.concatenate([folds[t][1] for t in tgts])
                    try:
                        preds, width, fi = _build_and_score(
                            pair_idx, apply_idx, bp, aug_cond, m,
                            ('pair', (a, b)))
                    except Exception as exc:
                        print(f"    pair {a},{b} {m}: build failed: "
                              f"{type(exc).__name__}: {exc}")
                        fold_errors.append({**where, 'model': m,
                                            'stage': 'pair_build',
                                            'exception':
                                                f'{type(exc).__name__}: {exc}'})
                        continue
                    preds = np.asarray(preds)
                    start = 0
                    for t in tgts:
                        n_t = len(folds[t][1])
                        p_t = preds[start:start + n_t]
                        start += n_t
                        src = b if t == a else a
                        score = float(metric_fn(labels[folds[t][1]], p_t))
                        transfer[m][(src, t)] = score
                        provenance[m][f'{src}->{t}'] = {
                            'rows_selection_saw': int(len(pair_idx)),
                            'rows_model_fitted_on': int(len(pair_idx)),
                            'folds_excluded_from_selection': [int(a), int(b)],
                            'cache_namespace': pair_ns,
                            'selection_scope': 'fold_train_rows',
                            'selection_rows': ('every fold except the target '
                                               'and the source'),
                            'hybrid': _recipe_fields(bp, aug_cond),
                            'training_rows_score': float(entry['score_mean']),
                            'heldout_score': score,
                            'width_built': width,
                            'features_by_rep': _features_by_rep(fi),
                        }
                        print(f"    [t={_el():.0f}s] {m} cell {src}->{t} "
                              f"{metric_name}={score:.4f}")

    # --- Step 3: Compute the deltas, per model ---
    print(f"\n  Step 3: Transfer analysis")
    print("    Diagonal: recipe and features chosen on every fold but the "
          "target's, scored on the target. Off-diagonal: chosen on every fold "
          "but the target's and one more, with the model fitted on those same "
          "rows, then scored on the target's rows. Nothing that produced a "
          "cell has seen the rows that cell is scored on. What differs "
          "between the two is which rows the selection and the model saw — "
          "one fold's worth.")

    by_model = {}
    for m in models:
        tm = {(k, k): v['score'] for k, v in own[m].items()}
        tm.update(transfer[m])
        per_target = {}
        n_nonfinite_targets = 0  # folds dropped because a score was NaN/inf
        for tgt in sorted(own[m]):
            own_score = tm.get((tgt, tgt))
            # Drop non-finite (NaN/inf) fold scores before averaging. A
            # Spearman on near-constant predictions returns NaN; left in, it
            # turns the whole aggregate into NaN.
            cross_scores = [tm[(src, tgt)] for src in sorted(own[m])
                            if src != tgt and (src, tgt) in tm
                            and np.isfinite(tm[(src, tgt)])]
            if own_score is None or not np.isfinite(own_score):
                n_nonfinite_targets += 1
                continue
            if cross_scores:
                mean_cross = float(np.mean(cross_scores))
                if lower_is_better:
                    delta = mean_cross - own_score  # positive = cross is worse
                else:
                    delta = own_score - mean_cross  # positive = own is better
                per_target[tgt] = {'own_score': own_score,
                                   'mean_cross_score': mean_cross,
                                   'cross_scores': cross_scores,
                                   'delta': delta}
                print(f"    {m} fold {tgt}: own={own_score:.4f}, "
                      f"mean_cross={mean_cross:.4f}, delta={delta:+.4f}")

        deltas = [v['delta'] for v in per_target.values()]
        if deltas:
            print(f"    {m}: mean absolute delta {np.mean(np.abs(deltas)):.4f}, "
                  f"max {np.max(np.abs(deltas)):.4f} over {len(deltas)} "
                  f"target fold(s)")
        else:
            print(f"    {m}: no off-diagonal cell, no delta")

        # --- Step 4: Nogueira stability index, per model ---
        folds_with_sets = sorted(own[m])
        feature_set_list = [own[m][k]['features'] for k in folds_with_sets]
        phi = nogueira_stability(feature_set_list, total_features)
        if len(feature_set_list) < 2:
            print(f"    {m}: Nogueira phi needs 2 folds' selections; "
                  f"{len(feature_set_list)} available")
        else:
            print(f"    {m}: Nogueira phi {phi:.4f}; features selected per "
                  f"fold {[len(fs) for fs in feature_set_list]} (of "
                  f"{total_features} features in the pool)")

        by_model[m] = {
            'transfer_matrix': {f"{s},{t}": v for (s, t), v in tm.items()},
            'cell_recipe_provenance': provenance[m],
            'per_target': convert_numpy(per_target),
            'aggregate': {
                'mean_abs_delta': (float(np.mean(np.abs(deltas)))
                                   if deltas else None),
                'max_abs_delta': (float(np.max(np.abs(deltas)))
                                  if deltas else None),
                'mean_delta': float(np.mean(deltas)) if deltas else None,
                'n_target_folds': len(deltas),
                'n_nonfinite_targets': n_nonfinite_targets,
            },
            'stability': {
                'nogueira_phi': phi,
                'n_feature_sets': len(feature_set_list),
                # phi's inputs, one per fold: the folds in order, the number
                # of features each fold's hybrid selected, and the selected
                # features themselves as [representation, feature index].
                'folds': folds_with_sets,
                'n_features_selected_per_fold': [len(fs) for fs in
                                                 feature_set_list],
                'feature_sets_per_fold': {
                    int(k): sorted([r, int(i)] for r, i in
                                   own[m][k]['features'])
                    for k in folds_with_sets},
                'mean_features_selected': (
                    float(np.mean([len(fs) for fs in feature_set_list]))
                    if feature_set_list else None),
            },
            'fold_blueprints': {
                int(k): {**_recipe_fields(v['blueprint'], v['aug_condition']),
                         'width_built': v['width']}
                for k, v in own[m].items()},
        }

    if not any(own[m] for m in models):
        print("  Skipping: no fold produced a scored recipe for any model")
        return {'skipped': True, 'reason': 'insufficient_folds',
                'n_folds_succeeded': 0,
                'fold_diagnostics': fold_errors,
                'search_failures': search_failures,
                'result_identity': RESULT_IDENTITY}

    # --- Compile results ---
    results = {
        'result_identity': RESULT_IDENTITY,
        'n_folds': int(n_outer),
        'n_folds_requested': n_folds_requested,
        'models': models,
        'total_features': total_features,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'selection_scope': 'fold_train_rows',
        # The grid every c24 search ran on, stored beside its numbers.
        'decided_grid': decided_search(dataset_name, 'general'),
        'outer_split': {'splitter': 'nested_cv_outer_folds', 'seed': SEED,
                        'n_folds': int(n_outer)},
        'per_fold': convert_numpy(per_fold),
        'by_model': convert_numpy(by_model),
        'transfer_not_computed': transfer_note,
        'diagonal_selection_saw': 'every fold except the target',
        # 2026-10-05, owner delegated, 2026-10-05: what one off-diagonal cell
        # is, in one line.
        'off_diagonal_cell_description': (
            'a fresh search with one more fold left out: the recipe and '
            'features are searched on every fold except the target and the '
            'source, the model is fitted on those rows, then scored on the '
            'target fold; nothing is transferred from the source fold'),
        'condition_cuts': {
            'reuse_fold_importance': _C24_REUSE_FOLD_WORK,
            'n_outer_folds_requested': n_folds_requested,
            'n_outer_folds_used': int(n_outer),
        },
        'features_transferred': True,
        'delta_meaning': (
            'how much the held-out score moves when the recipe and the '
            'selected features are searched, and the model fitted, on the '
            'rows outside BOTH folds rather than on the target fold own '
            'training rows -- one fold fewer. It is NOT a measurement of '
            'transferring one fold blueprint to another: the source fold '
            'Step-1 blueprint is never applied off the diagonal, because it '
            'was searched on rows that include the target fold'),
        'fold_diagnostics': fold_errors,
        'search_failures': search_failures,
    }

    # Summary table, one per model
    for m in models:
        ks = sorted(own[m])
        if not ks:
            continue
        tm = {(k, k): own[m][k]['score'] for k in ks}
        tm.update(transfer[m])
        print(f"\n  BLUEPRINT TRANSFER SUMMARY, {m} ({metric_name}, score per "
              f"source-selection x target fold):")
        print(f"  {'Source→Target':<16} " + " ".join(f"T{t:<4}" for t in ks))
        print("  " + "-" * (16 + 6 * len(ks)))
        for src in ks:
            row = f"  S{src:<15} "
            for tgt in ks:
                score = tm.get((src, tgt))
                if score is not None:
                    marker = " *" if src == tgt else "  "
                    row += f"{score:.3f}{marker}"
                else:
                    row += "  -    "
            print(row)
        print("  (* = own-fold blueprint)")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C24: Blueprint Transfer Test',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)
    # Through the suite's runner (2026-10-05), so a standalone run gets the
    # same representations (DATASET_REPS[ds]['general'], plus the
    # augmentation list), the same decided config and the same result file as
    # the cluster. --reps, --delta and --augment are not used.
    print("  c24 standalone: representations from DATASET_REPS through "
          "complexity_helpers.run_all; --reps/--delta/--augment ignored")
    from experiments.complexity.complexity_helpers import run_all
    run_all(args.dataset, args.output, experiments=['c24'], config=config,
            force_rerun=True)
