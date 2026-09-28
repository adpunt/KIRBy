#!/usr/bin/env python3
"""C24: Blueprint Transfer Test — Cross-Fold Generalization of Feature Selection.

Paper: Complexity / Discussion (feature selection overfitting defense)
Conclusion: Blueprints discovered on one fold generalize to other folds,
    demonstrating that feature selection is not overfit to training data.

Dependencies: None (standalone). Uses find_best_hybrid or precomputed search.

⛔ WHAT THE CODE ACTUALLY MEASURES, WHICH IS NOT WHAT THE THREE LINES BELOW
    USED TO SAY. Step 2 does NOT apply fold k's Step-1 blueprint to fold j.
    fold_blueprints[k] is read only to check the key exists. The recipe used on
    the off-diagonal comes from a FRESH find_best_hybrid run on the rows left
    after dropping BOTH fold k and fold j, so that the selection never sees the
    target fold's rows.

    That is a deliberate choice and the leak-free one: fold k's own Step-1
    recipe was searched on every fold except k, which INCLUDES fold j's rows,
    so applying it to fold j would score a selection on rows it had seen. But
    it means the diagonal and the off-diagonal differ in HOW MANY ROWS THE
    SEARCH SAW -- one fold's worth -- not in whose recipe was applied. Whether
    that is the experiment you want is a science decision and has not been
    made; the code, the printed reading and delta_meaning now describe the same
    thing, which they did not before.

Design (as run):
    For each CV fold k:
        1. Discover the optimal blueprint on fold k's training data.
        2. For each other fold j: search a recipe and select features on the
           rows outside BOTH k and j. Fit the downstream model on fold j's own
           training rows (as the diagonal does) with those features, and score
           it on fold j's held-out rows.
        3. Measure performance vs fold j's own-fold blueprint.

    If feature selection is NOT overfit:
        - Own-fold and cross-fold blueprints should perform similarly.
        - Small delta (own - cross) = generalizable selection.

    If feature selection IS overfit:
        - Cross-fold blueprints will degrade significantly.
        - Large delta = selection is training-data-specific.

    Also reports Nogueira stability index across fold blueprints.

Usage:
    python c24_blueprint_transfer.py --dataset logd --mode full
    python c24_blueprint_transfer.py --dataset caco2 --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    build_fold_hybrid,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, _cv_split,
    get_domain_metric,
    get_precomputed_search, get_best_for_downstream,
    INTERNAL_EVALUATORS, FULL_BUDGETS,
)
from kirby.find_best_hybrid import find_best_hybrid, _predict_for_metric

# =============================================================================
# LOCAL CONDITION CUTS
# =============================================================================
# c24 runs one find_best_hybrid per outer fold plus one per unordered PAIR of
# outer folds. At the 3 outer folds env.sh asks for that is 3 + 3 = 6 searches;
# at the shipped 5 it is 5 + 10 = 15.
#
# MEASURED on flab_thermo_tresanco (672 molecules, 5 representations capped to
# 1000 features each), 2026-09-25, with the laptop carrying eight other
# experiments: the run reached "Step 1" 129 s in and was still inside fold 0's
# FIRST search 670 s later, having printed nothing. Six searches at that rate
# is over an hour before a single hybrid is built or scored. The run never got
# past fold 0 in 798 s.
#
# Each search scores 2 budgets x 2 models x 3 allocation methods = 12
# configurations over 3 inner folds = 36 fits. The budgets are the only one of
# those three axes c24 passes itself; the other two come from env.sh and are
# not this file's to touch.
#
# ⛔ NOTHING BELOW CUTS A ROW. Every fold still sees every molecule, and the
# outer fold count is untouched. What is cut is how many configurations each
# search scores and how many times the same feature-importance vectors are
# recomputed.
#
# Unset, and with KIRBY_LOCAL_GRID unset, every knob below is the value the
# cluster has always run, so a server run is unchanged.
import time as _time


def _c24_local_grid():
    """True when the laptop grid in env.sh is active (KIRBY_LOCAL_GRID)."""
    return bool(os.environ.get('KIRBY_LOCAL_GRID'))


def _c24_env_ints(name, shipped, local=None):
    """One condition list: explicit environment variable, else the local
    default when the laptop grid is on, else the shipped value."""
    raw = os.environ.get(name)
    if raw:
        picked = tuple(int(x.strip()) for x in raw.split(',') if x.strip())
        if picked:
            return picked, f'{name}={raw}'
    if local is not None and _c24_local_grid():
        return tuple(local), 'KIRBY_LOCAL_GRID'
    return tuple(shipped), 'shipped'


# The budgets every c24 search chooses its recipe from. Shipped: whatever
# FULL_BUDGETS holds, which env.sh has already cut to 100 and 250 for a local
# run. The local default keeps only the LARGEST of those, which halves all six
# searches, the dominant cost measured above.
#
# WHAT THAT COSTS. The budget stops being a searched axis: every fold's recipe
# is then chosen at one budget, so the `budget` field of every entry in
# fold_blueprints is the same number and says nothing. c24's reported
# quantities -- the transfer matrix, the per-fold deltas and the Nogueira
# index -- are then measured at that one budget rather than at the better of
# two. Nothing in this file compares across budgets, so no reported comparison
# is lost; what is lost is the chance that a fold's best recipe sat at the
# smaller budget.
#
# ⛔ A COMPUTE CUT FOR A LAPTOP, NOT A FINDING. Nothing run under it says
# anything about which budget wins.
C24_BUDGETS, C24_BUDGETS_SOURCE = _c24_env_ints(
    'KIRBY_C24_BUDGETS', FULL_BUDGETS,
    local=(max(FULL_BUDGETS),) if FULL_BUDGETS else None)

# Reuse each row-set's feature-importance vectors across c24's own hybrid
# builds. c24 calls build_fold_hybrid 9 times at 3 outer folds: 3 own-fold
# builds on 3 distinct training row sets, and 6 off-diagonal builds on only 3
# distinct row sets, because dropping folds {A,B} leaves the same rows whether
# the cell is A->B or B->A. Importance depends on the row set and the
# importance method and on nothing else -- not the allocation, not the model --
# and build_fold_hybrid recomputes it on every call, at 93 s of every 150 s
# measured on this dataset on 2026-09-24. So the 6 off-diagonal builds pay for
# 6 importance computations where 3 would do.
#
# This changes no number: it hands build_fold_hybrid the same vectors it would
# have computed, over the same rows create_hybrid ranks features on (the
# training side of internal_holdout_split). Same pattern and same reasoning as
# c16's KIRBY_C16_REUSE_FOLD_WORK; f31_internal_evaluator_tuning.py:548 is the
# existing check that precomputed_importance gives the identical allocation.
#
# Default: ON for a local run, OFF everywhere else, so a cluster run recomputes
# exactly as it did before and is bit-identical.
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
        float in [-1, 1]. 1 = identical selection every time.
        Also returns standard error for confidence interval.

    Reference:
        Nogueira, Sechidis & Brown (2018). "On the Stability of Feature
        Selection Algorithms." JMLR 18(174), 1-54.
    """
    M = len(feature_sets)
    if M < 2 or total_features == 0:
        return float('nan'), float('nan')

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
        return float('nan'), float('nan')

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
        return float('nan'), float('nan')

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
        return float('nan'), float('nan')

    phi = 1 - numerator / denominator

    # Standard error: the paper's asymptotic variance (Nogueira et al. 2018,
    # Theorem 9), var = (4 / M^2) * sum_i (phi_i - mean(phi_i))^2, with one
    # term phi_i per selection. Written as in the authors' reference code
    # (getVarianceofStability), with every mean over the d features of the
    # pool. Features no selection picked have z = 0 and p = 0 and add nothing
    # to the sums, so summing over the union and dividing by d is exact.
    #
    # ⛔ Until 2026-09-28 this was sqrt(4/(M * denom^2) * sum (p(1-p))^2),
    # labelled "Eq. 7, simplified". It is not the paper's estimator and does
    # not reduce to it: it has no per-selection term at all.
    if M > 2:
        k_i = Z.sum(axis=1)                       # features picked per fold
        denom = (k_bar / d) * (1 - k_bar / d)
        phi_i = (1.0 / denom) * (
            (Z @ p_f) / d
            - k_i * k_bar / d**2
            + (phi / 2) * (2 * k_i * k_bar / d**2 - k_i / d - k_bar / d + 1))
        se = float(np.sqrt((4.0 / M**2) * np.sum((phi_i - phi_i.mean())**2)))
    else:
        se = float('nan')

    return float(phi), float(se)


def run(data, config, output_dir, dataset_name):
    """C24: Blueprint transfer test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 24: BLUEPRINT TRANSFER TEST")
    print("=" * 70)
    # Whatever cuts are in force, printed beside the run that used them, so no
    # result exists without them.
    print(f"  local condition cuts: search budgets={list(C24_BUDGETS)} "
          f"({C24_BUDGETS_SOURCE}); reuse_fold_importance={_C24_REUSE_FOLD_WORK}")
    _t0 = _time.time()

    def _el():
        return _time.time() - _t0

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    cmp = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    # Use None (not {}) so downstream `if augs:` checks correctly skip the
    # aug-handling path when no augmentations exist for this dataset.
    augs = data.get('augmentations') or None
    n_folds = config['n_splits']

    sample_X = next(iter(reps.values()))
    total_features = sum(v.shape[1] for v in reps.values())

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    task_type = data['task_type']

    # cv_split RAISES (no random-split fallback, feedback_no_random_splits) when
    # the dataset has fewer distinct groups than the requested fold count. Small
    # antibody-regression datasets (e.g. abagym_Ang2, intra-family single-DMS,
    # and other few-group sets) trip this and propagate an uncaught ValueError
    # out of run() -> the driver serializes it as {'error': ...}, capping the
    # production wave at 19/20. Reduce the outer fold count to the number of
    # distinct groups (min 2), mirroring _make_cv_splitter's reduce-folds rule
    # and c05's fulldata-bp guard. If <2 distinct groups exist, a leak-free
    # grouped CV is impossible -> return a clean skip (no 'error' key) rather
    # than crash.
    n_distinct_groups = 0 if groups is None else len(np.unique(groups))
    if n_distinct_groups < 2:
        print(f"  Skipping: blueprint transfer needs >=2 distinct groups for a "
              f"leak-free grouped CV, but got {n_distinct_groups}.")
        return {'skipped': True, 'reason': 'insufficient_groups',
                'n_distinct_groups': n_distinct_groups}
    if n_distinct_groups < n_folds:
        print(f"  Only {n_distinct_groups} distinct groups for {n_folds}-fold CV; "
              f"reducing to {n_distinct_groups} folds (group-aware, no random fallback).")
        n_folds = n_distinct_groups

    # --- Step 1: Collect per-fold blueprints and own-fold scores ---
    print(f"\n  Step 1: Discover blueprints on each of {n_folds} folds")

    folds = list(_cv_split(groups, n_folds, sample_X, labels))
    fold_blueprints = {}  # fold_idx -> blueprint
    fold_best_models = {}  # fold_idx -> best downstream model name for that fold
    fold_feature_sets = {}  # fold_idx -> set of (rep, idx) tuples
    fold_feature_infos = {}  # fold_idx -> feature_info from that fold's build
    own_fold_scores = {}  # fold_idx -> score on own test data
    # fold_idx -> the search's own inner-CV score for the recipe it picked, on
    # that fold's training rows. Recorded so the gap between what the search
    # promised and what the held-out fold gave can be read off the results.
    fold_search_scores = {}
    # (row-set key, importance method) -> importance vectors. Empty and unused
    # when _C24_REUSE_FOLD_WORK is off.
    _imp_cache = {} if _C24_REUSE_FOLD_WORK else None
    fold_errors = []  # (fold_idx, stage, error) — surfaced in the result so
    # an 'insufficient_folds' failure is self-diagnosing without the slurm log.

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
        val_augs = {k: v[val_idx] for k, v in augs.items()} if augs else None

        # Discover blueprint via find_best_hybrid
        try:
            _ts = _time.time()
            search = find_best_hybrid(
                train_reps, y_train, groups=groups[train_idx],
                budgets=list(C24_BUDGETS),
                augmentations=train_augs,
                augmentation_strategy='uncapped' if train_augs else 'none',
                internal_evaluators=INTERNAL_EVALUATORS,
                metric=metric_name,
                task_type=task_type,
                verbose=False,
                cache=True,
                # The real dataset name, so the per-dataset budget cap and
                # the per-dataset model skips match. It used to be
                # f"{dataset_name}_fold{fold_i}", and both are looked up BY
                # NAME, so c24 searched budgets its dataset is capped away from
                # and searched models the policy removes. The fold namespacing
                # moves to condition_suffix, which rides on `tier`. Each fold
                # trains on a different subset of molecules anyway, so their
                # rows already carried different label hashes and could not
                # have collided.
                dataset_name=dataset_name,
                condition_suffix=f"c24_fold{fold_i}",
                tier='general',
            )
            print(f"    [t={_el():.0f}s] fold {fold_i} search done in "
                  f"{_time.time() - _ts:.0f}s")
            # Pick best across models
            best = None
            best_model = None
            for m_name, m_best in search.best_by_model.items():
                if best is None or cmp(m_best['score_mean'], best['score_mean']):
                    best = m_best
                    best_model = m_name
        except Exception as e:
            print(f"    Fold {fold_i}: search failed ({e}), skipping")
            fold_errors.append({'fold': fold_i, 'stage': 'search',
                                'exception': f'{type(e).__name__}: {e}'})
            continue

        if best is None:
            print(f"    Fold {fold_i}: no valid config found, skipping")
            fold_errors.append({'fold': fold_i, 'stage': 'search',
                                'exception': 'search returned no model'})
            continue

        bp = best['blueprint']
        fold_blueprints[fold_i] = bp
        fold_best_models[fold_i] = best_model
        fold_search_scores[fold_i] = float(best['score_mean'])
        print(f"    Fold {fold_i}: {best_model}, budget={bp.budget}, "
              f"alloc={bp.allocation_method}, imp={bp.importance_method}")

        # Create hybrid with this blueprint and evaluate on own test data.
        # build_fold_hybrid, not a bare create_hybrid: it forwards the fold's
        # groups (so the internal admission holdout is not a random 20% carve
        # of scaffold-sharing rows), the internal evaluator, the augmentation
        # strategy and budgets, the candidate ranking, and the estimator with
        # the dataset's own metric — a bare call scored accept/reject with
        # r2_score and silently took create_hybrid's defaults for the rest.
        try:
            model_factories = get_model_factories(
                [best_model], task_type=task_type,
                n_samples=len(y_train))
            X_train, X_val, fi = build_fold_hybrid(
                train_reps, val_reps, y_train, bp,
                augmentations_train=train_augs,
                augmentations_val=val_augs,
                groups=groups[train_idx],
                model=model_factories[best_model](),
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                precomputed_importance=_c24_fold_importance(
                    _imp_cache, ('own', fold_i), train_reps, y_train,
                    groups[train_idx], bp))
            fold_feature_infos[fold_i] = fi

            # Collect selected features for stability analysis
            selected = set()
            for rep_name, rep_info in fi.items():
                if isinstance(rep_info, dict) and 'selected_indices' in rep_info:
                    for idx in rep_info['selected_indices']:
                        selected.add((rep_name, int(idx)))
            fold_feature_sets[fold_i] = selected

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_va = scaler.transform(X_val)

            model = model_factories[best_model]()
            model.fit(X_tr, y_train)
            own_fold_scores[fold_i] = float(metric_fn(
                y_val, _predict_for_metric(model, X_va, metric_fn)))
            print(f"      Own-fold {metric_name}: {own_fold_scores[fold_i]:.4f}")
        except Exception as e:
            print(f"      Own-fold eval failed: {e}")
            fold_errors.append({'fold': fold_i, 'stage': 'own_fold_eval',
                                'exception': f'{type(e).__name__}: {e}'})
            continue

    if len(fold_blueprints) < 2:
        # Fewer than 2 folds produced a scored blueprint (e.g. every outer
        # training fold collapsed to <2 groups inside find_best_hybrid, or no
        # config scored on this dataset). The transfer test genuinely cannot
        # run -> return a clean skip (no top-level 'error' key, which the
        # audit/wave counts as a hard failure). Diagnostics are kept under
        # 'fold_diagnostics' (inner key 'exception', not 'error') so the cause
        # is self-evident without the slurm log.
        print("  Skipping: need at least 2 successful folds for transfer test")
        return {'skipped': True, 'reason': 'insufficient_folds',
                'n_folds_succeeded': len(fold_blueprints),
                'fold_diagnostics': fold_errors}

    # --- Step 2: Cross-fold transfer ---
    print(f"\n  Step 2: Apply each source selection to every other fold's test rows")

    transfer_matrix = {}  # (source_fold, target_fold) -> score

    # Seed the diagonal with Step 1's own-fold scores so the diagonal is
    # consistent with the explicit own-fold eval (avoids stochastic divergence
    # from re-running the build + fit a second time on the same data).
    for fold_i, sc in own_fold_scores.items():
        transfer_matrix[(fold_i, fold_i)] = sc

    # Which rows each cell's recipe was searched on, so the size difference
    # between the diagonal and the off-diagonal is on the record.
    cell_recipe_provenance = {}
    # One pair search per UNORDERED pair of folds. A->B and B->A drop the same
    # two folds, so they are the same rows, labels, groups and budgets, and the
    # same search. It used to run twice in memory whenever the disk cache did
    # not hit (it did not in a synthetic check on 2026-09-28).
    _pair_search_memo = {}

    for src_fold in list(fold_blueprints):
        for tgt_fold, (train_idx, val_idx) in enumerate(folds):
            if tgt_fold not in fold_blueprints:
                continue  # skip folds that failed
            if src_fold == tgt_fold:
                continue  # diagonal already populated from own-fold eval

            # Two changes here, together.
            #
            # First, WHICH ROWS. The source fold's recipe and features from
            # Step 1 were chosen on every fold except the source — which
            # INCLUDES this target fold's validation rows. Scored on them, the
            # cell measured a selection made with the answer in hand. The
            # recipe AND the features are chosen here on rows that leave out
            # both folds, so nothing that produced this cell has seen the rows
            # it is scored on. One confound is left and recorded per cell
            # below: the off-diagonal sees one fold less data than the
            # diagonal.
            #
            # Second, WHAT CROSSES. The features are carried over, not
            # re-selected on the target fold. Rebuilding on the target's own
            # rows let only the recipe cross, so two folds landing on the same
            # recipe gave a delta of exactly 0 and read as stable.
            _pair_mask = np.ones(len(labels), dtype=bool)
            _pair_mask[folds[src_fold][1]] = False
            _pair_mask[val_idx] = False
            _pair_idx = np.where(_pair_mask)[0]
            y_val_tgt = labels[val_idx]
            src_bp = None
            _pair_exc = 'no blueprint from the pair-excluded search'
            try:
                if len(np.unique(groups[_pair_idx])) < 2:
                    raise ValueError(
                        f"only {len(np.unique(groups[_pair_idx]))} distinct "
                        f"group(s) left after dropping folds "
                        f"{src_fold} and {tgt_fold}")
                _pair_reps = {k: v[_pair_idx] for k, v in reps.items()}
                _pair_augs = ({k: v[_pair_idx] for k, v in augs.items()}
                              if augs else None)
                _tp = _time.time()
                _memo_key = tuple(sorted((src_fold, tgt_fold)))
                _pair_search = _pair_search_memo.get(_memo_key)
                if _pair_search is None:
                    _pair_search = find_best_hybrid(
                        _pair_reps, labels[_pair_idx], groups=groups[_pair_idx],
                        budgets=list(C24_BUDGETS),
                        augmentations=_pair_augs,
                        augmentation_strategy='uncapped' if _pair_augs else 'none',
                        internal_evaluators=INTERNAL_EVALUATORS,
                        metric=metric_name,
                        task_type=task_type,
                        verbose=False,
                        cache=True,
                        dataset_name=dataset_name,
                        # SORTED, so the two ordered pairs that drop the SAME two
                        # folds share one cache key. _pair_idx is the rows left
                        # after dropping src_fold AND tgt_fold, which is identical
                        # for (A,B) and (B,A): same rows, same labels, same groups,
                        # same budgets, one search. condition_suffix rides on `tier`
                        # and so enters the key, so c24_drop1_3 and c24_drop3_1 were
                        # two keys for one search and each was computed cold. Ten of
                        # the twenty pair searches were duplicates -- flagged in the
                        # 2026-09-18 compute audit and never acted on. Sorting takes
                        # c24 from 25 searches per dataset to 15 with no axis cut
                        # and no reported number changed.
                        condition_suffix=(
                            "c24_drop%d_%d" % tuple(sorted((src_fold, tgt_fold)))),
                        tier='general',
                    )
                    _pair_search_memo[_memo_key] = _pair_search
                print(f"    [t={_el():.0f}s] pair search dropping folds "
                      f"{sorted((src_fold, tgt_fold))} done in "
                      f"{_time.time() - _tp:.0f}s")
                # ⛔ TAKE THE RECIPE FOR THE MODEL THAT WILL SCORE IT. This
                # used to be the best entry across EVERY model, with the
                # winning model's name thrown away, while the estimator fitted
                # below is fold_best_models[tgt_fold] -- the target fold's own
                # Step 1 winner. On the diagonal the recipe and the estimator
                # come from one entry and always agree; off the diagonal they
                # did not have to. So the diagonal and the off-diagonal
                # differed in two ways at once, not one: which rows the search
                # saw, AND whether the recipe was chosen under the estimator it
                # is scored with.
                _tgt_model = fold_best_models.get(tgt_fold)
                _pair_by_model = _pair_search.best_by_model
                _pair_best = _pair_by_model.get(_tgt_model)
                _recipe_model = _tgt_model
                if _pair_best is None:
                    # That model produced nothing in this pair search. Fall
                    # back to the best across models, and RECORD that the two
                    # came apart rather than letting it pass unseen.
                    for _m, _mb in _pair_by_model.items():
                        if _pair_best is None or cmp(_mb['score_mean'],
                                                     _pair_best['score_mean']):
                            _pair_best, _recipe_model = _mb, _m
                if _pair_best is not None:
                    src_bp = _pair_best['blueprint']
            except Exception as exc:
                print(f"    recipe search excluding folds {src_fold},{tgt_fold} "
                      f"failed: {type(exc).__name__}: {exc}")
                _pair_exc = f'{type(exc).__name__}: {exc}'
            if src_bp is None:
                fold_errors.append({
                    'fold': tgt_fold, 'stage': 'pair_recipe_search',
                    'source_fold': src_fold,
                    'exception': _pair_exc})
                continue
            cell_recipe_provenance[f'{src_fold}->{tgt_fold}'] = {
                'rows_selection_saw': int(len(_pair_idx)),
                # The downstream model is fitted on the target's own training
                # rows, the same rows as the diagonal's model.
                'rows_model_fitted_on': int(len(train_idx)),
                'folds_excluded_from_selection': [int(src_fold), int(tgt_fold)],
                'features_transferred': True,
                # Which model the recipe was chosen for, and which one scores
                # it. Equal on every cell unless the pair search had no entry
                # for the target's model.
                'recipe_chosen_for_model': _recipe_model,
                'scored_with_model': fold_best_models.get(tgt_fold),
                'recipe_model_matches_scoring_model': (
                    _recipe_model == fold_best_models.get(tgt_fold)),
            }

            try:
                # Use the TARGET fold's best model so the feature-selection
                # contrast (own-selection vs cross-selection) is not confounded
                # by model-type variation across folds.
                best_model_tgt = fold_best_models.get(tgt_fold)
                if best_model_tgt is not None:
                    factories = get_model_factories(
                        [best_model_tgt], task_type=task_type,
                        n_samples=len(train_idx))
                    model = factories[best_model_tgt]()
                elif task_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

                # ⛔ ONLY THE SELECTION LOSES A FOLD. The downstream model is
                # fitted on the TARGET's own training rows (every fold but the
                # target), the same rows the diagonal's model is fitted on.
                # Until 2026-09-28 it was fitted on the pair-excluded rows, so
                # the downstream model ALSO lost a fold. At 3 outer folds that
                # halved its training rows (two folds' worth on the diagonal,
                # one fold's worth here), and the delta then measured mostly
                # training-set size, not the selection that delta_meaning and
                # the printed reading say it measures.
                #
                # How: build_fold_hybrid chooses the features on the
                # pair-excluded rows only (create_hybrid never sees val_reps),
                # then applies that same column choice to whatever val_reps it
                # is handed. It is handed the target's training rows followed by
                # the target's held-out rows, and the result is split back.
                # Nothing that chose a feature has seen the target's rows.
                y_pair = labels[_pair_idx]
                _pair_key = ('pair', tuple(sorted((src_fold, tgt_fold))))
                _apply_idx = np.concatenate([train_idx, val_idx])
                _apply_reps = {k: v[_apply_idx] for k, v in reps.items()}
                _apply_augs = ({k: v[_apply_idx] for k, v in augs.items()}
                               if augs else None)
                _, X_applied, fi = build_fold_hybrid(
                    _pair_reps, _apply_reps, y_pair, src_bp,
                    augmentations_train=_pair_augs,
                    augmentations_val=_apply_augs,
                    groups=groups[_pair_idx],
                    model=model,
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    precomputed_importance=_c24_fold_importance(
                        _imp_cache, _pair_key, _pair_reps, y_pair,
                        groups[_pair_idx], src_bp))
                X_train = X_applied[:len(train_idx)]
                X_val = X_applied[len(train_idx):]
                y_tgt_train = labels[train_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_va = scaler.transform(X_val)

                # A fresh instance: build_fold_hybrid may have fitted `model`
                # for its admission checks on the pair-excluded rows.
                if best_model_tgt is not None:
                    model = factories[best_model_tgt]()
                model.fit(X_tr, y_tgt_train)
                score = float(metric_fn(
                    y_val_tgt, _predict_for_metric(model, X_va, metric_fn)))
                transfer_matrix[(src_fold, tgt_fold)] = score
                print(f"    [t={_el():.0f}s] cell {src_fold}->{tgt_fold} "
                      f"{metric_name}={score:.4f}")
            except Exception as exc:
                # N6: surface the failed cross-fold transfer instead of
                # silently shrinking the transfer matrix. Without this a thrown
                # transfer is indistinguishable from "not attempted", and the
                # aggregate quietly degrades (per_target shrinks, aggregate->None)
                # with no diagnostic. The diagonal is seeded in Step 1, so this
                # degrades rather than errors.
                print(f"    transfer src={src_fold}->tgt={tgt_fold} failed: "
                      f"{type(exc).__name__}: {exc}")
                # Printed only, until 2026-09-28: the comment above said the
                # failure was surfaced, but nothing reached the result file, so
                # a missing cell read the same as one never attempted.
                fold_errors.append({
                    'fold': tgt_fold, 'stage': 'transfer_cell',
                    'source_fold': src_fold,
                    'exception': f'{type(exc).__name__}: {exc}'})
                continue

    # --- Step 3: Compute the deltas ---
    print(f"\n  Step 3: Transfer analysis")
    print("    Diagonal: recipe and features chosen on every fold but the "
          "target's, scored on the target. Off-diagonal: chosen on every fold "
          "but the target's and one more, then those same features applied to "
          "the target's rows. On both, the downstream model is fitted on every "
          "fold but the target's. Nothing that produced a cell has seen the "
          "rows that cell is scored on. What differs between the two is which "
          "rows the selection saw — one fold's worth.")

    # For each target fold: compare own blueprint score vs cross-fold blueprint scores
    per_target = {}
    n_nonfinite_targets = 0  # folds dropped because a score was NaN/inf
    n_nonfinite_cross = 0  # off-diagonal cells dropped because NaN/inf
    for tgt_fold in fold_blueprints:
        own_score = transfer_matrix.get((tgt_fold, tgt_fold))
        n_nonfinite_cross += sum(
            1 for src in fold_blueprints
            if src != tgt_fold and (src, tgt_fold) in transfer_matrix
            and not np.isfinite(transfer_matrix[(src, tgt_fold)]))
        # Drop non-finite (NaN/inf) fold scores before averaging. A Spearman
        # (or other metric) on near-constant predictions returns NaN; if that
        # NaN own-fold score is left in, this fold's delta is NaN and np.mean
        # turns the ENTIRE aggregate (mean_delta, max_abs_delta, mean_abs_delta)
        # into NaN — a run whose other folds transferred fine gets reported as a
        # total blow-up. Skip the poisoned fold instead and count it.
        cross_scores = [transfer_matrix[(src, tgt_fold)]
                        for src in fold_blueprints
                        if src != tgt_fold and (src, tgt_fold) in transfer_matrix
                        and np.isfinite(transfer_matrix[(src, tgt_fold)])]
        if own_score is None or not np.isfinite(own_score):
            n_nonfinite_targets += 1
            continue
        if own_score is not None and cross_scores:
            mean_cross = float(np.mean(cross_scores))
            if lower_is_better:
                delta = mean_cross - own_score  # positive = cross is worse (higher error)
            else:
                delta = own_score - mean_cross  # positive = own is better
            per_target[tgt_fold] = {
                'own_score': own_score,
                'mean_cross_score': mean_cross,
                'cross_scores': cross_scores,
                'delta': delta,
            }
            print(f"    Fold {tgt_fold}: own={own_score:.4f}, "
                  f"mean_cross={mean_cross:.4f}, delta={delta:+.4f}")

    # Aggregate
    deltas = [v['delta'] for v in per_target.values()]
    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    max_delta = float(np.max(np.abs(deltas))) if deltas else float('nan')

    print(f"\n  Transfer summary:")
    print(f"    Mean |delta|: {np.mean(np.abs(deltas)):.4f}" if deltas else "    No data")
    print(f"    Max  |delta|: {max_delta:.4f}" if deltas else "")
    if deltas:
        print(f"    Interpretation: {'STABLE (small delta)' if np.mean(np.abs(deltas)) < 0.02 else 'SOME DRIFT' if np.mean(np.abs(deltas)) < 0.05 else 'UNSTABLE (large delta)'}")
        print("    Reads as: how much the held-out score moves when the "
              "recipe AND the selected features come from one fold's worth "
              "less data. Two folds landing on the same recipe no longer "
              "give a delta of exactly zero — the features cross now.")

    # --- Step 4: Nogueira stability index ---
    print(f"\n  Step 4: Nogueira stability index")

    feature_set_list = [fold_feature_sets[k] for k in sorted(fold_feature_sets.keys())]
    phi, se = nogueira_stability(feature_set_list, total_features)
    print(f"    Nogueira phi: {phi:.4f} (SE={se:.4f})")
    if not np.isnan(phi):
        ci_lo = phi - 1.96 * se
        ci_hi = phi + 1.96 * se
        print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    Interpretation: {'HIGH stability' if phi > 0.7 else 'MODERATE stability' if phi > 0.4 else 'LOW stability'}")

    # --- Compile results ---
    results = {
        'n_folds': len(fold_blueprints),
        'total_features': total_features,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'transfer_matrix': {f"{s},{t}": v for (s, t), v in transfer_matrix.items()},
        # Which rows each off-diagonal cell's selection saw. The diagonal saw
        # every fold but its own; each off-diagonal cell saw every fold but
        # its own and the source's, so the off-diagonal sees one fold less
        # data. No cell has seen the rows it is scored on.
        'cell_recipe_provenance': cell_recipe_provenance,
        'diagonal_selection_saw': 'every fold except the target',
        # ⛔ What the search was allowed to choose from, stored beside the
        # numbers it produced. 'shipped' means no cut.
        'condition_cuts': {
            'search_budgets': list(C24_BUDGETS),
            'search_budgets_source': C24_BUDGETS_SOURCE,
            'reuse_fold_importance': _C24_REUSE_FOLD_WORK,
            # n_folds is cut to the number of distinct groups above, so it was
            # the count USED stored under the name "requested".
            'n_outer_folds_requested': int(config['n_splits']),
            'n_outer_folds_used': int(n_folds),
        },
        'features_transferred': True,
        'delta_meaning': (
            'how much the held-out score moves when the recipe and the '
            'selected features are searched on the rows outside BOTH folds '
            'rather than on the target fold own training rows -- one fold '
            'fewer for the search to work with. The downstream model is '
            'fitted on the target fold own training rows in both cases. '
            'It is NOT a measurement of '
            'transferring one fold blueprint to another: the source fold '
            'Step-1 blueprint is never applied off the diagonal, because it '
            'was searched on rows that include the target fold'),
        'per_target': convert_numpy(per_target),
        # The search's inner-CV score for the recipe it chose, per fold, in the
        # dataset's own metric. Beside own-fold held-out scores (the diagonal
        # of transfer_matrix) it shows how optimistic the search's choice was.
        # The inner folds train on fewer rows than the outer fit, so the gap
        # mixes that optimism with a sample-size difference.
        'fold_search_inner_score': convert_numpy(fold_search_scores),
        # Every search, own-fold build, pair search and off-diagonal cell that
        # failed, with its exception. Written only on the skip path until
        # 2026-09-28, so a run that lost cells stored nothing saying which.
        'fold_diagnostics': convert_numpy(fold_errors),
        'aggregate': {
            'mean_abs_delta': float(np.mean(np.abs(deltas))) if deltas else None,
            'max_abs_delta': max_delta if deltas else None,
            'mean_delta': mean_delta,
            'n_target_folds': len(deltas),
            'n_nonfinite_targets': n_nonfinite_targets,
            'n_nonfinite_cross_cells': n_nonfinite_cross,
        },
        'stability': {
            'nogueira_phi': phi,
            'nogueira_se': se,
            'ci_95': [phi - 1.96 * se, phi + 1.96 * se] if not np.isnan(se) else None,
            'n_feature_sets': len(feature_set_list),
            'mean_features_selected': float(np.mean([len(fs) for fs in feature_set_list])) if feature_set_list else None,
        },
        'fold_blueprints': {
            k: {
                'budget': bp.budget,
                'allocation': bp.allocation_method,
                'importance': bp.importance_method,
                'internal_evaluator': bp.internal_evaluator,
            } for k, bp in fold_blueprints.items()
        },
    }

    # Summary table
    print(f"\n  BLUEPRINT TRANSFER SUMMARY (score per source-selection x target fold):")
    print(f"  {'Source→Target':<16} " + " ".join(f"T{t:<4}" for t in sorted(fold_blueprints.keys())))
    print("  " + "-" * (16 + 6 * len(fold_blueprints)))
    for src in sorted(fold_blueprints.keys()):
        row = f"  S{src:<15} "
        for tgt in sorted(fold_blueprints.keys()):
            score = transfer_matrix.get((src, tgt))
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
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c24_blueprint_transfer', args.dataset)
