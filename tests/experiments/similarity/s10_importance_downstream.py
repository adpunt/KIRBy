#!/usr/bin/env python3
"""S10: Importance Scores as Feature Weights for Distance-Dependent Downstream Tasks.

Paper: Similarity
Question: does importance-weighted distance help GP, SVR, and q-RASAR-style
    similarity-augmented features (beyond the kNN case in S05)?
Dependencies: S05 (extends kNN), C04 (importance weighting).

Goes beyond kNN (S05) to test importance weighting on GP, SVR, and
similarity-augmented features (q-RASAR style). Four downstream tasks:
  1. Weighted kNN regression (extends S05 with per-rep + hybrid)
  2. GP with importance-weighted RBF kernel
  3. SVR with importance-weighted RBF kernel
  4. Similarity-augmented features (q-RASAR style)

Delta convention (2026-06-03): every stored `delta*` field is direction-
oriented by `_orient_deltas` so that POSITIVE always means improvement,
regardless of whether the dataset metric is higher-better (Spearman/AUC/F1) or
lower-better (RMSE/MAE). Before this fix, buchwald's RMSE delta "+13.077" (a
near-doubling of error) was misread as a gain. `results['deltas_oriented']`
flags that the convention is applied. Empirically weighting is neutral-to-
negative downstream -- a real result, not a wiring bug (the weights ARE applied).

How many features the hybrid arms are built with (2026-09-11). One recipe is
loaded from the distance pool, and every hybrid arm here is built at the number
of features that recipe was searched at, read off the recipe itself. A number of
features is part of a recipe, alongside its allocation method and its importance
method, so applying the recipe at some other width would be applying a different
recipe. The width this file asks the pool for is stored beside the applied one in
results['budget']: when the pool holds no recipe at the requested width it
substitutes one searched at another width, and then the two numbers differ.
Nothing here sweeps the number of features -- one width per run.

Which searches the recipe may come from (2026-09-12, owner's decision:
"s10's pool subset includes similarity hybrids"). Until now the one recipe
this file loads could only come from a search run under the predictive
objective. It may now also come from a search run under objective='similarity',
whose features were chosen to make the distance geometry predict each
molecule's activity from its five nearest neighbours. Both families are asked
for; which one the loaded recipe came from is stored in
results['hybrid_family'], and every eligible family's own best is stored in
results['pool_by_objective']. This is NOT the distance=True flag, which this
file already passed and which is left exactly as it was.

2026-09-19: both families are BUILT now, not only recorded. The per-fold pool
read returns a flat 'hybrids' list in the subset's PREFERENCE order and this
file took hybrids[0], which is the predictive family's entry on every dataset
that has one — so the geometry family was read and never used. It now takes the
top of each family's own list (by_objective[family]['hybrids']) per fold, and
all four tasks run once per space. The four task keys — weighted_knn,
weighted_gp, weighted_svr and qrasar — hold the predictive space and are
unchanged; every space is under results['by_hybrid_space'][space], with its own
recipes in results['hybrid_space_recipes'] and its per-fold read failures in
results['hybrid_space_errors'].

Top 2 of each arm, and the two deltas (2026-09-18, owner's decision: "I'll add
the top 2 rows everywhere as an extra, change nothing that's already reported").
Additive: four new TOP-LEVEL keys on `results`, no existing key changed, and the
nine widths each task builds an arm at are untouched.
- `top2_base_reps[model]`: the 2 best base representations for that downstream
  model, off the whole-dataset pool search. Every representation the search
  scored is still ranked; 2 of them are reported.
- `top2_hybrids[model]`: the 2 best hybrid configurations for that downstream
  model. Each carries the number of features it was searched at under 'budget'.
- `top2_pairs_by_model`: the same 2-and-2 grid over the six downstream models in
  one block, with 'n_distinct_units' -- how many separate things would have to
  be BUILT once one representation picked by several models counts once.
- `hybrid_deltas_by_model[model]`: the rank-1 hybrid against TWO reference
  numbers, 'delta_vs_source_mean' (the mean of the representations the budget
  actually gave features to, read off Blueprint.allocation_per_fold) and
  'delta_vs_best_rep' (the best single base representation for that model).
  Positive means the hybrid won, whichever way the metric runs.
Top level, not inside a task block, because _orient_deltas negates every field
named 'delta*' inside the four task blocks and these two are already oriented.
All four read rows the pool search already stored: 0 new model fits.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    cv_split, get_domain_metric, get_model_factories, convert_numpy,
    compute_feature_importance, build_fold_hybrid,
    fold_hybrid_scoring_kwargs,
    get_precomputed_search, get_best_for_downstream, extract_top3_base_reps,
    get_best_search_for_downstream, get_fold_pool_recipes,
    # One sentence saying what a search family chose its features for. Added
    # 2026-09-19 with the second hybrid space.
    describe_pool_family,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS, FULL_BUDGETS,
    flatten_fold_weights,
    fold_reselection, build_fold_reselection_input,
)
from kirby.find_best_hybrid import nested_cv_outer_folds

from kirby.hybrid import apply_feature_selection


# =============================================================================
# CONSTANTS
# =============================================================================

_GP_MAX_SAMPLES = 5000
_SVR_MAX_SAMPLES = 10000

# The seed the folds are built with. The stored pool searches cross-validated
# over the same partition (5 folds, seed 42, group-aware), so this file's fold i
# and the pool's fold i cover the same rows.
_FOLD_SEARCH_SEED = 42

# 2026-09-12 decision: s10's pool subset includes the hybrids built under
# objective='similarity' — features chosen to make the distance geometry
# predict each molecule's activity from its five nearest neighbours — as well
# as the ones built under the predictive objective.
_POOL_SUBSET = 'predictive_and_similarity'

# Which result-file space each family of stored search is built into. The
# predictive one is named 'hybrid' and its numbers fill the four unsuffixed
# task keys, so nothing reported before 2026-09-19 moves.
_SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# One recipe per held-out fold: the pool's rank-1 hybrid for that fold.
_N_POOL_HYBRIDS = 1

# How many features every hybrid arm is built with. Owner's decision: s10
# sweeps the canonical eight rather than one number. None is the ninth entry
# and means 'whatever number of features this fold's own recipe was searched
# at' — the arm the file computed before the sweep existed.
_SWEEP_BUDGETS = [None] + list(FULL_BUDGETS)


def _budget_key(budget):
    """Result key for one swept number of features.

    'recipe' is the arm built at the fold's own recipe width; every other key
    is the feature count itself, as a string so it survives JSON.
    """
    return 'recipe' if budget is None else str(int(budget))

# =============================================================================
# LOCAL HELPERS
# =============================================================================

def _importance_weights(imp_scores):
    """Convert raw importance scores to sqrt-normalised feature weights."""
    w = np.sqrt(np.maximum(imp_scores, 0))
    w = w / (w.sum() + 1e-10)
    return w


def _paired_delta(treatment_scores, reference_scores):
    """Mean per-fold (treatment - reference) over folds BOTH arms scored.

    Each arm appends float('nan') from its own try/except, so the two NaN
    positions need not line up: a difference of the two nanmeans can subtract a
    4-fold mean from a 5-fold mean. This restricts the subtraction to folds
    present in both arms. Returns (delta, n_paired_folds); delta is NaN and the
    count 0 when no fold carries a score in both arms.
    """
    t = np.asarray(treatment_scores, dtype=float).ravel()
    r = np.asarray(reference_scores, dtype=float).ravel()
    n = min(t.shape[0], r.shape[0])
    both = ~np.isnan(t[:n]) & ~np.isnan(r[:n])
    if not both.any():
        return float('nan'), 0
    return float(np.mean(t[:n][both] - r[:n][both])), int(both.sum())


# Written into every task's result dict so a reader can tell the two stored
# differences apart without consulting this file.
_DELTA_BASIS = (
    "delta* = nanmean(treatment) - nanmean(reference), each arm averaged over "
    "its own non-NaN folds (fold sets may differ). delta*_paired = mean of the "
    "per-fold difference over the n_paired_folds* folds scored by both arms."
)


# Written beside every task's hybrid-build failure list so a reader knows what
# the entries mean and what their absence means.
_HYBRID_BUILD_ERROR_BASIS = (
    "hybrid_build_errors = one entry per fold whose build_fold_hybrid call "
    "raised, carrying the fold index and the exception type and message. That "
    "fold contributes no score to either arm of this task's hybrid block. An "
    "empty list means every fold built."
)


def _drop_blueprint(obj):
    """Strip the pickled Blueprint out of the top-2 blocks added below.

    It does survive the JSON write -- convert_numpy routes it through
    Blueprint.to_dict (find_best_hybrid.py:1438) -- but it carries one list of
    feature indices per fold per donor representation, and every recipe field on
    it (allocation method, importance method, number of features, internal
    evaluator, step size, representation count) is already a key of the entry it
    hangs off. Which representations actually fed a hybrid is recorded by
    hybrid_deltas under 'source_reps'.
    """
    if isinstance(obj, dict):
        return {k: _drop_blueprint(v) for k, v in obj.items()
                if k != 'blueprint'}
    if isinstance(obj, list):
        return [_drop_blueprint(v) for v in obj]
    return obj


def _record_build_error(store, exc, **where):
    """Append one build_fold_hybrid failure to `store`.

    Records the exception type and message plus whatever fold/k coordinates the
    caller passes. Before 2026-08-26 these handlers ran a bare `continue`, so a
    fold that never built left no trace in the stored result.
    """
    entry = dict(where)
    entry['error'] = f'{type(exc).__name__}: {exc}'
    store.append(entry)
    return store


def _orient_deltas(obj, lower_is_better):
    """Recursively flip the sign of every 'delta*' field so positive always
    means improvement, regardless of metric direction.

    Each stored delta is (treatment - reference). For higher-is-better metrics
    (Spearman/AUC/F1) positive already means improvement -> no-op. For
    lower-is-better metrics (RMSE/MAE, e.g. buchwald) a positive raw delta means
    the treatment got WORSE, so it is negated. This fixes the trap where
    buchwald GP "delta +13.077" (an RMSE increase) read as a gain.
    """
    if not lower_is_better:
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _orient_deltas(v, lower_is_better)
            elif k.startswith('delta') and isinstance(v, (int, float)) and not isinstance(v, bool):
                obj[k] = -v
    elif isinstance(obj, list):
        for item in obj:
            _orient_deltas(item, lower_is_better)
    return obj


def _compute_importance_for_reps(train_reps, y_train, method='lightgbm_gain'):
    """Compute importance and return flat weight vector matching sorted rep concat."""
    imp_dict = compute_feature_importance(train_reps, y_train, method=method)
    imp_concat = np.concatenate([imp_dict[k] for k in sorted(imp_dict.keys())])
    return _importance_weights(imp_concat)


def _compute_importance_for_hybrid(feat_info):
    """Compute importance weights for hybrid-selected features.

    Uses flatten_fold_weights so the returned vector matches the hybrid
    matrix shape exactly (including augmentation columns appended by
    build_fold_hybrid).
    """
    raw = flatten_fold_weights(feat_info)
    return _importance_weights(raw)


def _compute_sim_descriptors(X_train, y_train, X_query, k=10, is_self=False):
    """Compute q-RASAR-style similarity descriptors for each query sample.

    For each query sample, finds k nearest neighbors in X_train and computes:
      - mean(y_neighbors): average target value of neighbors
      - std(y_neighbors): std of target values of neighbors
      - min_distance: distance to nearest neighbor
      - mean_distance: mean distance to k neighbors

    Args:
        is_self: Pass True when X_query is X_train (training-fold descriptors)
            so the self-match (distance=0) is excluded from neighbors.

    Returns (n_query, 4) array.
    """
    k_nn = min(k + (1 if is_self else 0), len(y_train) - 1)
    if k_nn < 1:
        return np.zeros((X_query.shape[0], 4))

    nn = make_nearest_neighbors(n_neighbors=k_nn)
    nn.fit(X_train)
    dists, indices = nn.kneighbors(X_query)

    if is_self:
        # Drop the first column (self-match with distance=0)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

    y_neighbors = y_train[indices]  # (n_query, k)
    mean_y = np.mean(y_neighbors, axis=1)
    # Reviewed 2026-09-18: this std is a q-RASAR engineered feature fed to the
    # downstream model (see column_stack below), not a reported confidence
    # number -- the small-n uncertainty-display question doesn't apply to it.
    # No change requested.
    std_y = np.std(y_neighbors, axis=1)
    min_dist = dists[:, 0]
    mean_dist = np.mean(dists, axis=1)

    return np.column_stack([mean_y, std_y, min_dist, mean_dist])


def _reselected_best_k_within_arm(knn_task, lower_is_better):
    """Held-out best-k re-score of the weighted-vs-unweighted kNN comparison.

    The stored kNN result selects its best k on the SAME folds it scores on
    (s10:210 and _print_summary) -> selection-optimistic. Using the per-fold
    scores now persisted by _run_weighted_knn, this re-picks the winning k on
    LEFT-OUT folds via the shared fold_reselection arithmetic: the WEIGHTED arm's
    k-variants are the 'candidates' and the UNWEIGHTED arm's are the 'baselines',
    so fold_reselection's `delta` is the honest weighted-minus-unweighted gap
    (direction-corrected, + == weighted better). Keyed per rep, plus 'hybrid'.

    This is a LOCAL adapter, not a redesign: the shared helper's hybrid-vs-
    baseline shape does not map to a within-arm best-k selection, so we reshape
    the persisted fold scores into it rather than re-implement the LOFO math.
    """
    def _rescore(section):
        cands, base = {}, {}
        for name, kdict in section.items():
            if not isinstance(kdict, dict):
                continue
            cands[name] = [
                {'name': kk, 'fold_scores': v.get('weighted_folds', [])}
                for kk, v in kdict.items() if isinstance(v, dict)
            ]
            base[name] = {
                kk: v.get('unweighted_folds', [])
                for kk, v in kdict.items() if isinstance(v, dict)
            }
        return fold_reselection(cands, base, lower_is_better)

    out = {}
    per_rep = knn_task.get('per_rep') or {}
    if per_rep:
        out['per_rep'] = _rescore(per_rep)
    hybrid = knn_task.get('hybrid') or {}
    if isinstance(hybrid, dict) and hybrid:
        out['hybrid'] = _rescore({'hybrid': hybrid}).get('hybrid', {})
    return out


# =============================================================================
# TASK 1: WEIGHTED kNN REGRESSION
# =============================================================================

def _run_weighted_knn(reps, labels, groups, config, fold_blueprints,
                      metric_fn, metric_name, lower_is_better=False,
                      augmentations=None, task_type='regression',
                      fold_splits=None, fold_importance=None,
                      hybrid_budgets=None):
    """Weighted kNN: per-rep and hybrid, weighted vs unweighted.

    fold_splits: list of (train_idx, val_idx) tuples, shared across all tasks.
    fold_importance: dict {fold_i: {rep_name: importance_array}}, precomputed.
    fold_blueprints: {fold_i: recipe} read from the pool, one recipe per
        held-out fold, chosen on the other folds' stored scores.
    hybrid_budgets: the numbers of features to build the hybrid at, one arm
        each. None inside the list means the fold's own recipe width.
    """
    is_clf = task_type == 'classification'
    print(f"\n  --- Task 1: Weighted kNN {'Classification' if is_clf else 'Regression'} ---")

    k_values = [5, 10, 20]
    results = {'per_rep': {}, 'hybrid': {}}

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        rep_results = {}

        for k in k_values:
            uw_scores, w_scores = [], []

            for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])
                k_eff = min(k, len(y_train) - 1)

                # Unweighted
                KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                model = KNN(n_neighbors=k_eff)
                model.fit(X_tr, y_train)
                uw_scores.append(metric_fn(y_val, model.predict(X_va)))

                # Weighted (use precomputed importance)
                try:
                    imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                    if imp_arr is None:
                        raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                    w = _importance_weights(imp_arr)
                    model_w = KNN(n_neighbors=k_eff)
                    model_w.fit(X_tr * w, y_train)
                    w_scores.append(metric_fn(y_val, model_w.predict(X_va * w)))
                except Exception:
                    w_scores.append(float('nan'))

            uw_mean = float(np.nanmean(uw_scores))
            w_mean = float(np.nanmean(w_scores))
            paired, n_paired = _paired_delta(w_scores, uw_scores)
            rep_results[f'k{k}'] = {
                'unweighted': uw_mean,
                'weighted': w_mean,
                'delta': w_mean - uw_mean,
                'delta_paired': paired,
                'n_paired_folds': n_paired,
                # Per-fold scores persisted so a reader can re-pick the best k on
                # held-out folds (honest) instead of on the same folds (s10:210).
                'unweighted_folds': [float(s) for s in uw_scores],
                'weighted_folds': [float(s) for s in w_scores],
            }

        results['per_rep'][rep_name] = rep_results
        sign = -1 if lower_is_better else 1
        best_k_key = max(rep_results, key=lambda kk: sign * rep_results[kk]['weighted'])
        best = rep_results[best_k_key]
        print(f"    {rep_name}: best {best_k_key} weighted={best['weighted']:.4f} "
              f"(delta={best['delta']:+.4f}, delta_paired={best['delta_paired']:+.4f} "
              f"over {best['n_paired_folds']} folds)")

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}
        for budget in budgets:
            bkey = _budget_key(budget)
            by_k = {}
            for k in k_values:
                uw_scores, w_scores = [], []

                for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
                    blueprint = fold_blueprints.get(fold_i)
                    if blueprint is None:
                        continue
                    train_reps = {n: r[train_idx] for n, r in reps.items()}
                    val_reps = {n: r[val_idx] for n, r in reps.items()}
                    y_train, y_val = labels[train_idx], labels[val_idx]
                    aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                    aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

                    k_eff = min(k, len(y_train) - 1)
                    _KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                    try:
                        X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                            train_reps, val_reps, y_train, blueprint,
                            augmentations_train=aug_tr, augmentations_val=aug_va,
                            # The number of features this arm is built with,
                            # one of the eight swept in run(). None leaves the
                            # number this fold's own recipe was searched at.
                            budget_override=budget,
                            groups=(groups[train_idx] if groups is not None else None),
                            # This arm's downstream estimator IS a kNN scored with
                            # metric_fn — pass both so create_hybrid's internal
                            # accept/reject test judges allocations by the model that
                            # actually consumes them, instead of falling back to its
                            # internal r2/accuracy proxy.
                            **fold_hybrid_scoring_kwargs(
                                blueprint, model=_KNN(n_neighbors=k_eff),
                                metric_fn=metric_fn,
                                lower_is_better=lower_is_better))
                    except Exception as e:
                        _record_build_error(hybrid_build_errors, e, k=k,
                                            fold=fold_i, budget=budget)
                        continue

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr_h)
                    X_va = scaler.transform(X_va_h)

                    # Unweighted
                    KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                    model = KNN(n_neighbors=k_eff)
                    model.fit(X_tr, y_train)
                    uw_scores.append(metric_fn(y_val, model.predict(X_va)))

                    # Weighted
                    try:
                        w = _compute_importance_for_hybrid(feat_info)
                        assert w.shape[0] == X_tr.shape[1], (
                            f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                        model_w = KNN(n_neighbors=k_eff)
                        model_w.fit(X_tr * w, y_train)
                        w_scores.append(metric_fn(y_val, model_w.predict(X_va * w)))
                    except Exception:
                        w_scores.append(float('nan'))

                if uw_scores:
                    uw_mean = float(np.nanmean(uw_scores))
                    w_mean = float(np.nanmean(w_scores))
                    paired, n_paired = _paired_delta(w_scores, uw_scores)
                    by_k[f'k{k}'] = {
                        'unweighted': uw_mean,
                        'weighted': w_mean,
                        'delta': w_mean - uw_mean,
                        'delta_paired': paired,
                        'n_paired_folds': n_paired,
                        'unweighted_folds': [float(s) for s in uw_scores],
                        'weighted_folds': [float(s) for s in w_scores],
                    }
                    print(f"    hybrid [{bkey} features] k={k}: "
                          f"uw={uw_mean:.4f}, w={w_mean:.4f} "
                          f"(delta={w_mean - uw_mean:+.4f}, delta_paired={paired:+.4f} "
                          f"over {n_paired} folds)")

            results['hybrid_by_budget'][bkey] = by_k

        # 'hybrid' stays the arm built at each fold's own recipe width, so the
        # summary and the within-arm per-fold reselection below read the same block
        # they always did. The other eight widths sit beside it.
        results['hybrid'] = results['hybrid_by_budget'].get('recipe', {})
        results['hybrid_budgets'] = [_budget_key(b) for b in budgets]
        results['hybrid_build_errors'] = hybrid_build_errors
        results['n_hybrid_build_failures'] = len(hybrid_build_errors)
        results['hybrid_build_error_basis'] = _HYBRID_BUILD_ERROR_BASIS
        if hybrid_build_errors:
            print(f"    hybrid build failed on {len(hybrid_build_errors)} of "
                  f"{len(budgets) * len(k_values) * len(fold_splits)} "
                  f"(features, k, fold) triples; first: "
                  f"{hybrid_build_errors[0]['error']}")

    results['delta_basis'] = _DELTA_BASIS
    return results


# =============================================================================
# TASK 2: GP WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _run_weighted_gp(reps, labels, groups, config, fold_blueprints,
                     metric_fn, metric_name, n_samples,
                     lower_is_better=False,
                     augmentations=None,
                     fold_splits=None, fold_importance=None,
                     hybrid_budgets=None):
    """GP with importance-weighted RBF kernel.

    lower_is_better: direction of metric_fn, forwarded to build_fold_hybrid
        alongside model=/metric_fn= so create_hybrid's accept/reject test reads
        the score the right way round. The hybrid branch already passed this
        name to build_fold_hybrid without the parameter existing here, which
        raised NameError inside the try and skipped every fold.
    fold_blueprints: {fold_i: recipe} read from the pool, one recipe per
        held-out fold, chosen on the other folds' stored scores.
    hybrid_budgets: the numbers of features to build the hybrid at, one arm
        each. None inside the list means the fold's own recipe width.
    """
    print("\n  --- Task 2: GP with Importance-Weighted RBF ---")

    if n_samples > _GP_MAX_SAMPLES:
        print(f"    SKIPPED: n_samples={n_samples} > {_GP_MAX_SAMPLES}")
        return {'skipped': True, 'reason': f'n_samples={n_samples} > {_GP_MAX_SAMPLES}'}

    results = {'per_rep': {}, 'hybrid': {}}

    def _make_gp():
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
        return GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            alpha=1e-6, random_state=42,
        )

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        std_scores, w_scores = [], []

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Standard GP
            try:
                gp = _make_gp()
                gp.fit(X_tr, y_train)
                std_scores.append(metric_fn(y_val, gp.predict(X_va)))
            except Exception:
                std_scores.append(float('nan'))

            # Importance-weighted GP (pre-transform: X * sqrt(w))
            try:
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
                gp_w = _make_gp()
                gp_w.fit(X_tr * w, y_train)
                w_scores.append(metric_fn(y_val, gp_w.predict(X_va * w)))
            except Exception:
                w_scores.append(float('nan'))

        std_mean = float(np.nanmean(std_scores))
        w_mean = float(np.nanmean(w_scores))
        paired, n_paired = _paired_delta(w_scores, std_scores)
        results['per_rep'][rep_name] = {
            'standard': std_mean,
            'weighted': w_mean,
            'delta': w_mean - std_mean,
            'delta_paired': paired,
            'n_paired_folds': n_paired,
            'standard_folds': [float(s) for s in std_scores],
            'weighted_folds': [float(s) for s in w_scores],
        }
        print(f"    {rep_name}: std={std_mean:.4f}, weighted={w_mean:.4f} "
              f"(delta={w_mean - std_mean:+.4f}, delta_paired={paired:+.4f} "
              f"over {n_paired} folds)")

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}

        for budget in budgets:
            bkey = _budget_key(budget)
            std_scores, w_scores = [], []

            for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
                blueprint = fold_blueprints.get(fold_i)
                if blueprint is None:
                    continue
                train_reps = {n: r[train_idx] for n, r in reps.items()}
                val_reps = {n: r[val_idx] for n, r in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

                try:
                    X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                        train_reps, val_reps, y_train, blueprint,
                        augmentations_train=aug_tr, augmentations_val=aug_va,
                        # The number of features this arm is built with, one of
                        # the eight swept in run(). None leaves the number this
                        # fold's own recipe was searched at.
                        budget_override=budget,
                        groups=(groups[train_idx] if groups is not None else None),
                        **fold_hybrid_scoring_kwargs(
                            blueprint, model=_make_gp(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as e:
                    _record_build_error(hybrid_build_errors, e, fold=fold_i,
                                        budget=budget)
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_h)
                X_va = scaler.transform(X_va_h)

                # Standard GP
                try:
                    gp = _make_gp()
                    gp.fit(X_tr, y_train)
                    std_scores.append(metric_fn(y_val, gp.predict(X_va)))
                except Exception:
                    std_scores.append(float('nan'))

                # Weighted GP
                try:
                    w = _compute_importance_for_hybrid(feat_info)
                    assert w.shape[0] == X_tr.shape[1], (
                        f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                    gp_w = _make_gp()
                    gp_w.fit(X_tr * w, y_train)
                    w_scores.append(metric_fn(y_val, gp_w.predict(X_va * w)))
                except Exception:
                    w_scores.append(float('nan'))

            if std_scores:
                std_mean = float(np.nanmean(std_scores))
                w_mean = float(np.nanmean(w_scores))
                paired, n_paired = _paired_delta(w_scores, std_scores)
                results['hybrid_by_budget'][bkey] = {
                    'standard': std_mean,
                    'weighted': w_mean,
                    'delta': w_mean - std_mean,
                    'delta_paired': paired,
                    'n_paired_folds': n_paired,
                    'standard_folds': [float(s) for s in std_scores],
                    'weighted_folds': [float(s) for s in w_scores],
                }
                print(f"    hybrid [{bkey} features]: std={std_mean:.4f}, "
                      f"weighted={w_mean:.4f} "
                      f"(delta={w_mean - std_mean:+.4f}, delta_paired={paired:+.4f} "
                      f"over {n_paired} folds)")

        # 'hybrid' stays the arm built at each fold's own recipe width.
        results['hybrid'] = results['hybrid_by_budget'].get('recipe', {})
        results['hybrid_budgets'] = [_budget_key(b) for b in budgets]
        results['hybrid_build_errors'] = hybrid_build_errors
        results['n_hybrid_build_failures'] = len(hybrid_build_errors)
        results['hybrid_build_error_basis'] = _HYBRID_BUILD_ERROR_BASIS
        if hybrid_build_errors:
            print(f"    hybrid build failed on {len(hybrid_build_errors)} of "
                  f"{len(budgets) * len(fold_splits)} (features, fold) pairs; "
                  f"first: {hybrid_build_errors[0]['error']}")

    results['delta_basis'] = _DELTA_BASIS
    return results


# =============================================================================
# TASK 3: SVR WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _run_weighted_svr(reps, labels, groups, config, fold_blueprints,
                      metric_fn, metric_name, n_samples,
                      lower_is_better=False,
                      augmentations=None,
                      fold_splits=None, fold_importance=None,
                      hybrid_budgets=None):
    """SVR with importance-weighted RBF kernel.

    lower_is_better: direction of metric_fn, forwarded to build_fold_hybrid
        alongside model=/metric_fn= so create_hybrid's accept/reject test reads
        the score the right way round. The hybrid branch already passed this
        name to build_fold_hybrid without the parameter existing here, which
        raised NameError inside the try and skipped every fold.
    fold_blueprints: {fold_i: recipe} read from the pool, one recipe per
        held-out fold, chosen on the other folds' stored scores.
    hybrid_budgets: the numbers of features to build the hybrid at, one arm
        each. None inside the list means the fold's own recipe width.
    """
    print("\n  --- Task 3: SVR with Importance-Weighted RBF ---")

    if n_samples > _SVR_MAX_SAMPLES:
        print(f"    SKIPPED: n_samples={n_samples} > {_SVR_MAX_SAMPLES}")
        return {'skipped': True, 'reason': f'n_samples={n_samples} > {_SVR_MAX_SAMPLES}'}

    results = {'per_rep': {}, 'hybrid': {}}

    def _make_svr():
        return SVR(kernel='rbf', C=1.0, gamma='scale')

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        std_scores, w_scores = [], []

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Standard SVR
            try:
                svr = _make_svr()
                svr.fit(X_tr, y_train)
                std_scores.append(metric_fn(y_val, svr.predict(X_va)))
            except Exception:
                std_scores.append(float('nan'))

            # Importance-weighted SVR
            try:
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
                svr_w = _make_svr()
                svr_w.fit(X_tr * w, y_train)
                w_scores.append(metric_fn(y_val, svr_w.predict(X_va * w)))
            except Exception:
                w_scores.append(float('nan'))

        std_mean = float(np.nanmean(std_scores))
        w_mean = float(np.nanmean(w_scores))
        paired, n_paired = _paired_delta(w_scores, std_scores)
        results['per_rep'][rep_name] = {
            'standard': std_mean,
            'weighted': w_mean,
            'delta': w_mean - std_mean,
            'delta_paired': paired,
            'n_paired_folds': n_paired,
            'standard_folds': [float(s) for s in std_scores],
            'weighted_folds': [float(s) for s in w_scores],
        }
        print(f"    {rep_name}: std={std_mean:.4f}, weighted={w_mean:.4f} "
              f"(delta={w_mean - std_mean:+.4f}, delta_paired={paired:+.4f} "
              f"over {n_paired} folds)")

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}

        for budget in budgets:
            bkey = _budget_key(budget)
            std_scores, w_scores = [], []

            for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
                blueprint = fold_blueprints.get(fold_i)
                if blueprint is None:
                    continue
                train_reps = {n: r[train_idx] for n, r in reps.items()}
                val_reps = {n: r[val_idx] for n, r in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

                try:
                    X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                        train_reps, val_reps, y_train, blueprint,
                        augmentations_train=aug_tr, augmentations_val=aug_va,
                        # The number of features this arm is built with, one of
                        # the eight swept in run(). None leaves the number this
                        # fold's own recipe was searched at.
                        budget_override=budget,
                        groups=(groups[train_idx] if groups is not None else None),
                        **fold_hybrid_scoring_kwargs(
                            blueprint, model=_make_svr(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as e:
                    _record_build_error(hybrid_build_errors, e, fold=fold_i,
                                        budget=budget)
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_h)
                X_va = scaler.transform(X_va_h)

                # Standard SVR
                try:
                    svr = _make_svr()
                    svr.fit(X_tr, y_train)
                    std_scores.append(metric_fn(y_val, svr.predict(X_va)))
                except Exception:
                    std_scores.append(float('nan'))

                # Weighted SVR
                try:
                    w = _compute_importance_for_hybrid(feat_info)
                    assert w.shape[0] == X_tr.shape[1], (
                        f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                    svr_w = _make_svr()
                    svr_w.fit(X_tr * w, y_train)
                    w_scores.append(metric_fn(y_val, svr_w.predict(X_va * w)))
                except Exception:
                    w_scores.append(float('nan'))

            if std_scores:
                std_mean = float(np.nanmean(std_scores))
                w_mean = float(np.nanmean(w_scores))
                paired, n_paired = _paired_delta(w_scores, std_scores)
                results['hybrid_by_budget'][bkey] = {
                    'standard': std_mean,
                    'weighted': w_mean,
                    'delta': w_mean - std_mean,
                    'delta_paired': paired,
                    'n_paired_folds': n_paired,
                    'standard_folds': [float(s) for s in std_scores],
                    'weighted_folds': [float(s) for s in w_scores],
                }
                print(f"    hybrid [{bkey} features]: std={std_mean:.4f}, "
                      f"weighted={w_mean:.4f} "
                      f"(delta={w_mean - std_mean:+.4f}, delta_paired={paired:+.4f} "
                      f"over {n_paired} folds)")

        # 'hybrid' stays the arm built at each fold's own recipe width.
        results['hybrid'] = results['hybrid_by_budget'].get('recipe', {})
        results['hybrid_budgets'] = [_budget_key(b) for b in budgets]
        results['hybrid_build_errors'] = hybrid_build_errors
        results['n_hybrid_build_failures'] = len(hybrid_build_errors)
        results['hybrid_build_error_basis'] = _HYBRID_BUILD_ERROR_BASIS
        if hybrid_build_errors:
            print(f"    hybrid build failed on {len(hybrid_build_errors)} of "
                  f"{len(budgets) * len(fold_splits)} (features, fold) pairs; "
                  f"first: {hybrid_build_errors[0]['error']}")

    results['delta_basis'] = _DELTA_BASIS
    return results


# =============================================================================
# TASK 4: SIMILARITY-AUGMENTED FEATURES (q-RASAR STYLE)
# =============================================================================

def _run_qrasar(reps, labels, groups, config, fold_blueprints,
                metric_fn, metric_name, task_type='regression',
                lower_is_better=False,
                augmentations=None,
                fold_splits=None, fold_importance=None,
                hybrid_budgets=None):
    """q-RASAR-style similarity-augmented features.

    lower_is_better: direction of metric_fn, forwarded to build_fold_hybrid
    alongside model=/metric_fn= so create_hybrid's accept/reject test reads the
    score the right way round. The hybrid branch already passed this name to
    build_fold_hybrid without the parameter existing here, which raised
    NameError inside the try and skipped every fold.

    fold_blueprints: {fold_i: recipe} read from the pool, one recipe per
    held-out fold, chosen on the other folds' stored scores.

    hybrid_budgets: the numbers of features to build the hybrid at, one arm
    each. None inside the list means the fold's own recipe width.

    For each sample, find k nearest neighbors and compute similarity
    descriptors: mean(y_neighbors), std(y_neighbors), min_distance,
    mean_distance. Append as 4 extra features.

    Compare:
      - base features only (RF baseline)
      - base + sim descriptors from unweighted distance
      - base + sim descriptors from importance-weighted distance
    For per-rep and hybrid.
    """
    print("\n  --- Task 4: Similarity-Augmented Features (q-RASAR style) ---")

    k_neighbors = 10
    results = {'per_rep': {}, 'hybrid': {}}

    _rf_fac = get_model_factories(['rf'], task_type=task_type,
                                    n_samples=len(labels))['rf']
    def _make_rf():
        return _rf_fac()

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        base_scores, uw_sim_scores, w_sim_scores = [], [], []

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Base features only
            try:
                rf = _make_rf()
                rf.fit(X_tr, y_train)
                base_scores.append(metric_fn(y_val, rf.predict(X_va)))
            except Exception:
                base_scores.append(float('nan'))

            # Base + unweighted sim descriptors
            try:
                sim_tr = _compute_sim_descriptors(X_tr, y_train, X_tr,
                                                  k=k_neighbors, is_self=True)
                sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                  k=k_neighbors, is_self=False)
                X_tr_aug = np.hstack([X_tr, sim_tr])
                X_va_aug = np.hstack([X_va, sim_va])
                rf_aug = _make_rf()
                rf_aug.fit(X_tr_aug, y_train)
                uw_sim_scores.append(metric_fn(y_val, rf_aug.predict(X_va_aug)))
            except Exception:
                uw_sim_scores.append(float('nan'))

            # Base + importance-weighted sim descriptors
            try:
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                    k=k_neighbors, is_self=True)
                sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                    k=k_neighbors, is_self=False)
                X_tr_aug_w = np.hstack([X_tr_w, sim_tr_w])
                X_va_aug_w = np.hstack([X_va_w, sim_va_w])
                rf_aug_w = _make_rf()
                rf_aug_w.fit(X_tr_aug_w, y_train)
                w_sim_scores.append(metric_fn(y_val, rf_aug_w.predict(X_va_aug_w)))
            except Exception:
                w_sim_scores.append(float('nan'))

        base_mean = float(np.nanmean(base_scores))
        uw_mean = float(np.nanmean(uw_sim_scores))
        w_mean = float(np.nanmean(w_sim_scores))
        d_uw_p, n_uw_p = _paired_delta(uw_sim_scores, base_scores)
        d_w_p, n_w_p = _paired_delta(w_sim_scores, base_scores)
        d_wvu_p, n_wvu_p = _paired_delta(w_sim_scores, uw_sim_scores)
        results['per_rep'][rep_name] = {
            'base_only': base_mean,
            'base_plus_sim_unweighted': uw_mean,
            'base_plus_sim_weighted': w_mean,
            'delta_uw': uw_mean - base_mean,
            'delta_w': w_mean - base_mean,
            'delta_w_vs_uw': w_mean - uw_mean,
            'delta_uw_paired': d_uw_p,
            'delta_w_paired': d_w_p,
            'delta_w_vs_uw_paired': d_wvu_p,
            'n_paired_folds_uw': n_uw_p,
            'n_paired_folds_w': n_w_p,
            'n_paired_folds_w_vs_uw': n_wvu_p,
            'base_only_folds': [float(s) for s in base_scores],
            'base_plus_sim_unweighted_folds': [float(s) for s in uw_sim_scores],
            'base_plus_sim_weighted_folds': [float(s) for s in w_sim_scores],
        }
        print(f"    {rep_name}: base={base_mean:.4f}, "
              f"+sim_uw={uw_mean:.4f} (d={uw_mean - base_mean:+.4f}, "
              f"d_paired={d_uw_p:+.4f} over {n_uw_p} folds), "
              f"+sim_w={w_mean:.4f} (d={w_mean - base_mean:+.4f}, "
              f"d_paired={d_w_p:+.4f} over {n_w_p} folds)")

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}

        for budget in budgets:
            bkey = _budget_key(budget)
            base_scores, uw_sim_scores, w_sim_scores = [], [], []

            for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
                blueprint = fold_blueprints.get(fold_i)
                if blueprint is None:
                    continue
                train_reps = {n: r[train_idx] for n, r in reps.items()}
                val_reps = {n: r[val_idx] for n, r in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

                try:
                    X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                        train_reps, val_reps, y_train, blueprint,
                        augmentations_train=aug_tr, augmentations_val=aug_va,
                        # The number of features this arm is built with, one of
                        # the eight swept in run(). None leaves the number this
                        # fold's own recipe was searched at.
                        budget_override=budget,
                        groups=(groups[train_idx] if groups is not None else None),
                        **fold_hybrid_scoring_kwargs(
                            blueprint, model=_make_rf(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as e:
                    _record_build_error(hybrid_build_errors, e, fold=fold_i,
                                        budget=budget)
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_h)
                X_va = scaler.transform(X_va_h)

                # Base only
                try:
                    rf = _make_rf()
                    rf.fit(X_tr, y_train)
                    base_scores.append(metric_fn(y_val, rf.predict(X_va)))
                except Exception:
                    base_scores.append(float('nan'))

                # Unweighted sim descriptors
                try:
                    sim_tr = _compute_sim_descriptors(X_tr, y_train, X_tr,
                                                      k=k_neighbors, is_self=True)
                    sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                      k=k_neighbors, is_self=False)
                    X_tr_aug = np.hstack([X_tr, sim_tr])
                    X_va_aug = np.hstack([X_va, sim_va])
                    rf_aug = _make_rf()
                    rf_aug.fit(X_tr_aug, y_train)
                    uw_sim_scores.append(metric_fn(y_val, rf_aug.predict(X_va_aug)))
                except Exception:
                    uw_sim_scores.append(float('nan'))

                # Weighted sim descriptors
                try:
                    w = _compute_importance_for_hybrid(feat_info)
                    assert w.shape[0] == X_tr.shape[1], (
                        f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                    X_tr_w = X_tr * w
                    X_va_w = X_va * w
                    sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                        k=k_neighbors, is_self=True)
                    sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                        k=k_neighbors, is_self=False)
                    X_tr_aug_w = np.hstack([X_tr_w, sim_tr_w])
                    X_va_aug_w = np.hstack([X_va_w, sim_va_w])
                    rf_aug_w = _make_rf()
                    rf_aug_w.fit(X_tr_aug_w, y_train)
                    w_sim_scores.append(metric_fn(y_val, rf_aug_w.predict(X_va_aug_w)))
                except Exception:
                    w_sim_scores.append(float('nan'))

            if base_scores:
                base_mean = float(np.nanmean(base_scores))
                uw_mean = float(np.nanmean(uw_sim_scores))
                w_mean = float(np.nanmean(w_sim_scores))
                d_uw_p, n_uw_p = _paired_delta(uw_sim_scores, base_scores)
                d_w_p, n_w_p = _paired_delta(w_sim_scores, base_scores)
                d_wvu_p, n_wvu_p = _paired_delta(w_sim_scores, uw_sim_scores)
                results['hybrid_by_budget'][bkey] = {
                    'base_only': base_mean,
                    'base_plus_sim_unweighted': uw_mean,
                    'base_plus_sim_weighted': w_mean,
                    'delta_uw': uw_mean - base_mean,
                    'delta_w': w_mean - base_mean,
                    'delta_w_vs_uw': w_mean - uw_mean,
                    'delta_uw_paired': d_uw_p,
                    'delta_w_paired': d_w_p,
                    'delta_w_vs_uw_paired': d_wvu_p,
                    'n_paired_folds_uw': n_uw_p,
                    'n_paired_folds_w': n_w_p,
                    'n_paired_folds_w_vs_uw': n_wvu_p,
                    'base_only_folds': [float(s) for s in base_scores],
                    'base_plus_sim_unweighted_folds': [float(s) for s in uw_sim_scores],
                    'base_plus_sim_weighted_folds': [float(s) for s in w_sim_scores],
                }
                print(f"    hybrid [{bkey} features]: base={base_mean:.4f}, "
                      f"+sim_uw={uw_mean:.4f} (d={uw_mean - base_mean:+.4f}, "
                      f"d_paired={d_uw_p:+.4f} over {n_uw_p} folds), "
                      f"+sim_w={w_mean:.4f} (d={w_mean - base_mean:+.4f}, "
                      f"d_paired={d_w_p:+.4f} over {n_w_p} folds)")

        # 'hybrid' stays the arm built at each fold's own recipe width.
        results['hybrid'] = results['hybrid_by_budget'].get('recipe', {})
        results['hybrid_budgets'] = [_budget_key(b) for b in budgets]
        results['hybrid_build_errors'] = hybrid_build_errors
        results['n_hybrid_build_failures'] = len(hybrid_build_errors)
        results['hybrid_build_error_basis'] = _HYBRID_BUILD_ERROR_BASIS
        if hybrid_build_errors:
            print(f"    hybrid build failed on {len(hybrid_build_errors)} of "
                  f"{len(budgets) * len(fold_splits)} (features, fold) pairs; "
                  f"first: {hybrid_build_errors[0]['error']}")

    results['delta_basis'] = _DELTA_BASIS
    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance scores as feature weights for distance-dependent
    downstream tasks: kNN, GP, SVR, and q-RASAR similarity-augmented features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S10: IMPORTANCE WEIGHTS FOR DOWNSTREAM TASKS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_samples = len(labels)
    task_type = data['task_type']

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")
    print(f"  Samples: {n_samples}, Reps: {len(reps)}")

    # Get hybrid blueprint via find_best_hybrid
    hybrid_blueprint = None
    domain = data.get('domain', 'molecular')
    # The number of features this file ASKS the pool for. It only steers which
    # stored recipe comes back: when the pool holds no recipe searched at this
    # many features, get_best_for_downstream hands back the best recipe it has
    # at any number of features instead. 100 is fixed here rather than read off
    # config['budgets'], whose first entry under similarity_helpers.run_all is
    # FULL_BUDGETS[0] = 10 — that asked the pool for a 10-feature recipe on
    # every dispatched run while the command line asked for 100.
    requested_budget = kwargs.get('budget', 100)

    # ── Group-aware folds, built before the pool is read ─────────────────────
    # nested_cv_outer_folds is the partition the stored pool searches
    # cross-validated over (5 folds, seed 42, group-aware), so this file's fold
    # i and the pool's fold i cover the same rows. cv_split built the same
    # partition; the pool read below is what requires the two to line up.
    n_folds = config['n_splits']
    if n_folds > 1:
        fold_splits = nested_cv_outer_folds(reps, labels, groups, n_folds,
                                            _FOLD_SEARCH_SEED, task_type)
    else:
        # --mode intermediary: one 80/20 split, which is not a partition the
        # pool holds fold scores for. The pool read below still runs and its
        # warnings are stored.
        _ref_X = reps[sorted(reps.keys())[0]]
        fold_splits = list(cv_split(groups, n_folds, _ref_X, labels))

    # The whole-dataset pool read. It no longer decides what the hybrid arms
    # are built from — the per-fold read below does that — and supplies three
    # things: the SearchResult whose per-fold scores become the stored
    # fold_reselection block, the top-3 base representations, and which search
    # family the pool's whole-dataset winner came from. Its outcome is recorded
    # rather than swallowed.
    whole_dataset_status = 'ok'
    search = None  # set on a successful pooled search; carries per-fold scores
    pool_meta = None  # metadata of the search the recipe came from
    try:
        # Pass metric/task_type/domain + data so the pooled search resolves the
        # SAME cache lineage used at precompute time and can hit the in-memory
        # precomputed_searches_distance dict (instead of a cold recompute /
        # mismatched metric).
        #
        # 2026-09-12, owner's decision: "s10's pool subset includes similarity
        # hybrids". The searches this file may take its recipe from now include
        # the ones run under objective='similarity' — features chosen to make
        # the distance geometry predict each molecule's activity from its five
        # nearest neighbours — not only the ones run under the predictive
        # objective. distance=True below is a separate thing and is unchanged.
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='knn', budget=requested_budget,
            augmentations=data.get('augmentations'),
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, data=data,
            distance=True,
            pool_subset='predictive_and_similarity',
        )
        if meta:
            hybrid_blueprint = meta['best_config']['blueprint']
            pool_meta = meta
        if hybrid_blueprint:
            print(f"  Hybrid blueprint: alloc={hybrid_blueprint.allocation_method}, "
                  f"budget={hybrid_blueprint.budget}, "
                  f"imp={hybrid_blueprint.importance_method}, "
                  f"ev={hybrid_blueprint.internal_evaluator}")
            # Which of the two families the recipe came from, printed so a log
            # reader can tell them apart without opening the result file.
            print(f"  Recipe came from the {meta['objective']} search "
                  f"(scored in {meta['metric_name']}), chosen by "
                  f"{meta['objective_choice']} out of pool subset "
                  f"{meta['pool_subset']}.")
        else:
            whole_dataset_status = 'no_blueprint'
            print("  The whole-dataset pool read returned no recipe; the "
                  "per-fold read below is what the hybrid arms are built from.")
        top3 = extract_top3_base_reps(search)
        if top3:
            # These scores come off the same search the recipe came from, so
            # their metric is that search's metric — named here because, since
            # 2026-09-12, that search can be a similarity one.
            _fam = meta['objective'] if meta else 'unknown'
            _fam_metric = meta['metric_name'] if meta else 'unknown'
            print(f"  Top-3 base reps (from the {_fam} search, "
                  f"{_fam_metric}): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        whole_dataset_status = f'error: {type(e).__name__}: {e}'
        print(f"  Whole-dataset pool read failed: {e}")
        print("  The stored fold_reselection block and the top-3 base "
              "representations are absent; the per-fold read below still runs.")
        top3 = None

    # ── One recipe per held-out fold, read from the pool ─────────────────────
    # The recipe every hybrid arm is built from used to be picked once over
    # every row. It is now picked for each held-out fold from the OTHER folds'
    # stored scores, the way s12 and s13 pick theirs: the pool ranks its stored
    # configurations on folds other than i and hands back the top one, so fold
    # i's own score never chose the recipe fold i is scored under.
    #
    # What is held out changed, and the result says so. The pool's searches ran
    # over every row, so this is a recipe chosen without fold i's SCORES, not by
    # a search that never saw fold i's rows.
    fold_blueprints = {}
    fold_recipes = {}
    # {space: {fold: recipe}}, one space per family of stored search. The two
    # unsuffixed dicts above hold the family the pool returned, which is the
    # predictive one, so nothing they feed moves. Added 2026-09-19.
    fold_blueprints_by_space = {}
    fold_recipes_by_space = {}
    pool_family_errors = {}
    pool_fold_errors = {}
    pool_fold_warnings = {}
    pool_fold0_pruned = None
    pool_n_folds_seen = set()
    print(f"  Reading the pool per fold (subset={_POOL_SUBSET}, model=knn, "
          f"{requested_budget} features asked for, top {_N_POOL_HYBRIDS} "
          f"per fold)...")
    for fold_i in range(len(fold_splits)):
        try:
            pool = get_fold_pool_recipes(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(fold_splits),
                downstream_model='knn', budget=requested_budget,
                dataset_name=dataset_name,
                augmentations=data.get('augmentations'),
                distance=True, data=data,
                metric=metric_name, task_type=task_type, domain=domain,
                pool_subset=_POOL_SUBSET,
                n_hybrids=_N_POOL_HYBRIDS)
        except Exception as e:
            pool_fold_errors[f'fold{fold_i}'] = f'{type(e).__name__}: {e}'
            print(f"  [warn] pool read failed (fold {fold_i}): "
                  f"{type(e).__name__}: {e}")
            continue
        _p = pool.get('fold0_pruned')
        _p = True if _p is None else bool(_p)
        pool_fold0_pruned = (_p if pool_fold0_pruned is None
                             else (pool_fold0_pruned or _p))
        if pool.get('n_folds_in_pool') is not None:
            pool_n_folds_seen.add(int(pool['n_folds_in_pool']))
        _warn = list(pool.get('warnings') or [])
        if _warn:
            pool_fold_warnings[f'fold{fold_i}'] = _warn
            for w in _warn:
                print(f"  [pool fold {fold_i}] {w}")
        hybrids = pool.get('hybrids') or []
        if not hybrids:
            pool_fold_errors[f'fold{fold_i}'] = (
                f'the pool held no kNN hybrid configuration under subset '
                f'{_POOL_SUBSET}')
            continue

        # ⛔ ONE RECIPE PER FAMILY, NOT THE FIRST OF THE LIST. Added 2026-09-19.
        #
        # `hybrids` is ordered by the subset's PREFERENCE order, so hybrids[0]
        # is the predictive family's on every dataset that has one -- which is
        # every dataset. From 2026-09-12 to 2026-09-19 this file asked for both
        # families and built that one, and the other appeared only in
        # 'pool_by_objective'. get_fold_pool_recipes already returns each
        # family's own ranked list under by_objective[family]['hybrids'];
        # this reads the top of each.
        #
        # It matters here because three of the four tasks turn on a distance
        # between molecules -- the weighted k-nearest-neighbours fit, and the
        # Gaussian-process and support-vector arms whose RBF kernel is a
        # distance -- and the similarity search chooses features so that
        # exactly that distance predicts activity from the five nearest
        # neighbours. The fourth, q-RASAR, adds similarity-derived FEATURES
        # rather than reweighting a distance.
        for _fam, _fam_block in (pool.get('by_objective') or {}).items():
            _fam_hybrids = (_fam_block or {}).get('hybrids') or []
            _space = _SPACE_FOR_FAMILY.get(_fam, f'hybrid_{_fam}')
            if not _fam_hybrids:
                (pool_family_errors.setdefault(_space, {}))[f'fold{fold_i}'] = (
                    f'the pool held no kNN hybrid configuration under '
                    f'objective={_fam} on the other folds')
                continue
            _ftop = _fam_hybrids[0]
            fold_blueprints_by_space.setdefault(_space, {})[fold_i] = \
                _ftop['blueprint']
            (fold_recipes_by_space.setdefault(_space, {}))[fold_i] = {
                'objective': _ftop['objective'],
                'tier': _ftop['tier'],
                'aug_condition': _ftop['aug_condition'],
                'allocation_method': _ftop['allocation_method'],
                'importance_method': _ftop['importance_method'],
                'internal_evaluator': _ftop['internal_evaluator'],
                'budget': _ftop['budget'],
                'metric_name': _ftop['metric_name'],
                'lower_is_better': _ftop['lower_is_better'],
                'score_other_folds': _ftop['score_other_folds'],
                'n_other_folds': _ftop['n_other_folds'],
            }

        top = hybrids[0]
        fold_blueprints[fold_i] = top['blueprint']
        fold_recipes[fold_i] = {
            'objective': top['objective'],
            'tier': top['tier'],
            'aug_condition': top['aug_condition'],
            'allocation_method': top['allocation_method'],
            'importance_method': top['importance_method'],
            'internal_evaluator': top['internal_evaluator'],
            'budget': top['budget'],
            'metric_name': top['metric_name'],
            'lower_is_better': top['lower_is_better'],
            'score_other_folds': top['score_other_folds'],
            'n_other_folds': top['n_other_folds'],
        }
        print(f"  Fold {fold_i} recipe: alloc={top['allocation_method']}, "
              f"imp={top['importance_method']}, "
              f"ev={top['internal_evaluator']}, "
              f"budget={top['budget']} features, {top['objective']} family "
              f"({top['metric_name']}={top['score_other_folds']:.4f} on "
              f"{top['n_other_folds']} other folds)")
    # 'hybrid_status' is the per-fold read's outcome, because that is what
    # decides whether the hybrid branches in Tasks 1-4 run at all.
    if fold_blueprints:
        hybrid_status = 'ok'
    else:
        hybrid_status = 'no_blueprint'
        print("  WARNING: the pool gave no recipe on any fold -- all hybrid "
              "branches (Tasks 1-4) will be SKIPPED (per-rep baselines only).")

    # ── How many features every hybrid arm below is built with ───────────────
    # Owner's decision: s10 sweeps the canonical eight — 10, 25, 50, 100, 250,
    # 500, 1,000 and 2,000 features — not one number. It reads a finished
    # search, so the sweep runs no search; each width costs one build and one
    # model fit per fold. A ninth arm, keyed 'recipe', is built at whatever
    # number of features that fold's own recipe was searched at, which is the
    # arm the file computed before the sweep.
    sweep_budgets = list(_SWEEP_BUDGETS)
    # Kept for the record: the width the whole-dataset recipe was searched at,
    # against the width this file asked the pool for.
    applied_budget = (getattr(hybrid_blueprint, 'budget', None)
                      if hybrid_blueprint is not None else None)
    print(f"  Hybrid arms are built at {len(sweep_budgets)} widths: "
          f"{', '.join(_budget_key(b) for b in sweep_budgets)}")

    start_total = time.time()
    results = {}
    # Flag the hybrid outcome so downstream readers/aggregators (s09) can tell a
    # baseline-only run apart from a complete hybrid run.
    results['hybrid_status'] = hybrid_status
    results['whole_dataset_pool_status'] = whole_dataset_status
    results['hybrid_evaluated'] = bool(fold_blueprints)
    results['budget'] = {
        # features asked of the pool
        'requested': requested_budget,
        # features on the whole-dataset recipe that came back — the width the
        # recipe was searched at, kept so a reader can see which of the swept
        # widths is the recipe's own
        'applied': applied_budget,
        'matched': (applied_budget is not None
                    and applied_budget == requested_budget),
        'source': ('the recipe loaded from the distance pool'
                   if hybrid_blueprint is not None else None),
        # every width each task builds an arm at, in result-key form
        'swept': [_budget_key(b) for b in sweep_budgets],
        'swept_basis': (
            "hybrid_by_budget holds one arm per entry. 'recipe' is built at "
            "the number of features that fold's own recipe was searched at; "
            "every other key is that many features, taken from FULL_BUDGETS."),
        # the width each fold's own recipe was searched at
        'per_fold_recipe_budget': {str(f): r['budget']
                                   for f, r in fold_recipes.items()},
    }

    # ── The recipe each fold's hybrid arms are built from (2026-09-12) ───────
    # One per held-out fold, chosen on the other folds' stored scores. Stored so
    # the result file says which recipe produced each fold's number instead of
    # implying a single whole-run recipe.
    results['fold_recipes'] = fold_recipes
    results['recipe_selection'] = ('other_folds_of_pool_search'
                                   if fold_recipes else None)
    results['recipe_selection_note'] = (
        "Each fold's recipe was ranked on the pool search's OTHER folds, so "
        "fold i's own score did not choose it. The pool's searches ran over "
        "every row, so fold i's ROWS were seen when the pool was built.")
    results['pool_fold_read'] = {
        'pool_subset': _POOL_SUBSET,
        'n_hybrids_per_fold': _N_POOL_HYBRIDS,
        'downstream_model': 'knn',
        'budget': requested_budget,
        'n_folds': len(fold_splits),
        'n_folds_in_pool': sorted(pool_n_folds_seen),
        'fold0_pruned': pool_fold0_pruned,
        'warnings': pool_fold_warnings,
        'errors': pool_fold_errors,
    }

    # ── Which search the one recipe came from (2026-09-12) ───────────────────
    # Owner's decision: "s10's pool subset includes similarity hybrids". The
    # recipe can now have been built either under the predictive objective or
    # under objective='similarity', and the two are scored in different units:
    # a predictive search stores the dataset's own metric, a similarity search
    # stores the correlation between each held-out molecule's activity and what
    # its five nearest neighbours predict. So the family is recorded here, and
    # every eligible family's own best is recorded beside it, rather than one
    # number being reported as if both were on the same scale.
    results['pool_subset'] = 'predictive_and_similarity'
    results['hybrid_family'] = (pool_meta['objective']
                                if pool_meta is not None else None)
    results['hybrid_family_choice'] = (pool_meta['objective_choice']
                                       if pool_meta is not None else None)
    results['hybrid_family_metric'] = (pool_meta['metric_name']
                                       if pool_meta is not None else None)
    results['hybrid_family_lower_is_better'] = (
        pool_meta['lower_is_better'] if pool_meta is not None else None)
    results['pool_by_objective'] = (
        {o: {'tier': d['tier'], 'aug_condition': d['aug_condition'],
             'score_mean': d['score_mean'], 'metric_name': d['metric_name'],
             'lower_is_better': d['lower_is_better']}
         for o, d in pool_meta['by_objective'].items()}
        if pool_meta is not None else None)
    results['pool_selection_scope'] = (pool_meta['selection_scope']
                                       if pool_meta is not None else None)
    results['pool_selection_note'] = (pool_meta['selection_note']
                                      if pool_meta is not None else None)

    if top3:
        results['top3_base_reps'] = [(n, s) for n, s in top3]
        # Same search, so the same metric as the recipe's family (2026-09-12).
        results['top3_base_reps_family'] = (pool_meta['objective']
                                            if pool_meta is not None else None)
        results['top3_base_reps_metric'] = (pool_meta['metric_name']
                                            if pool_meta is not None else None)

    augmentations = data.get('augmentations')
    is_clf = task_type == 'classification'

    # -------------------------------------------------------------------------
    # Shared precompute: per-(fold, rep) importance ONCE so all four downstream
    # tasks reuse the same numbers. Previously each task recomputed importance
    # per (fold, rep), 4x duplicated work. fold_splits was built above, before
    # the pool read, and is the same list for every task.
    # -------------------------------------------------------------------------
    t_imp = time.time()
    fold_importance = {}
    for fold_i, (train_idx, _val_idx) in enumerate(fold_splits):
        train_reps = {n: r[train_idx] for n, r in reps.items()}
        try:
            imp_dict = compute_feature_importance(
                train_reps, labels[train_idx], method='lightgbm_gain')
            fold_importance[fold_i] = imp_dict
        except Exception as e:
            print(f"  WARN: precompute importance failed at fold={fold_i}: {e}")
            fold_importance[fold_i] = {}
    print(f"  [Precomputed fold importance for {len(fold_splits)} folds x "
          f"{len(reps)} reps in {time.time() - t_imp:.1f}s]")

    # The four tasks, once per hybrid space. Added 2026-09-19: the pool holds
    # two families of stored search and this file built one of them. The four
    # top-level keys keep the numbers they had -- they are the PREDICTIVE
    # space, which is the family the pool returned on every run before today --
    # and every space, that one included, is under 'by_hybrid_space'.
    #
    # The per-representation baselines inside each task do not depend on the
    # recipe, so they are recomputed identically in each space; the hybrid arms
    # are what differ.
    _spaces_to_run = [sp for sp in (_SPACE_FOR_FAMILY.get('predictive'),
                                    _SPACE_FOR_FAMILY.get('similarity'))
                      if fold_blueprints_by_space.get(sp)]
    results['hybrid_spaces_scored'] = list(_spaces_to_run)
    results['hybrid_space_recipes'] = convert_numpy(fold_recipes_by_space)
    results['hybrid_space_errors'] = pool_family_errors
    results['hybrid_spaces_note'] = (
        'One hybrid space per family of stored search. A predictive search '
        "chose its features to predict the label; a similarity search chose "
        "them so that each molecule's activity is predicted by its five "
        'nearest neighbours in the distance those features define, and fits no '
        'downstream model. Their stored search scores are in different units '
        'and nothing here ranks one against the other. Every number under '
        "'by_hybrid_space' is measured by this file on the same folds in the "
        "dataset's own metric, so those ARE comparable. The four top-level "
        'task keys hold the predictive space alone and are unchanged.')
    results['by_hybrid_space'] = {}
    _primary_space = _SPACE_FOR_FAMILY['predictive']

    for _space in _spaces_to_run:
        _fold_bps = fold_blueprints_by_space[_space]
        _space_out = {}
        print(f"\n  ===== hybrid space {_space} =====")

        # Task 1: Weighted kNN
        t0 = time.time()
        _space_out['weighted_knn'] = _run_weighted_knn(
            reps, labels, groups, config, _fold_bps,
            metric_fn, metric_name, lower_is_better,
            augmentations=augmentations, task_type=task_type,
            fold_splits=fold_splits, fold_importance=fold_importance,
            hybrid_budgets=sweep_budgets)
        print(f"  [Task 1 elapsed: {time.time() - t0:.1f}s]")

        # Task 2: GP with importance-weighted RBF (regression only)
        if is_clf:
            print("\n  --- Task 2: GP --- SKIP (regression-only)")
            _space_out['weighted_gp'] = {'skipped': True,
                                         'reason': 'classification'}
        else:
            t0 = time.time()
            _space_out['weighted_gp'] = _run_weighted_gp(
                reps, labels, groups, config, _fold_bps,
                metric_fn, metric_name, n_samples,
                lower_is_better=lower_is_better,
                augmentations=augmentations,
                fold_splits=fold_splits, fold_importance=fold_importance,
                hybrid_budgets=sweep_budgets)
            print(f"  [Task 2 elapsed: {time.time() - t0:.1f}s]")

        # Task 3: SVR with importance-weighted RBF (regression only)
        if is_clf:
            print("  --- Task 3: SVR --- SKIP (regression-only)")
            _space_out['weighted_svr'] = {'skipped': True,
                                          'reason': 'classification'}
        else:
            t0 = time.time()
            _space_out['weighted_svr'] = _run_weighted_svr(
                reps, labels, groups, config, _fold_bps,
                metric_fn, metric_name, n_samples,
                lower_is_better=lower_is_better,
                augmentations=augmentations,
                fold_splits=fold_splits, fold_importance=fold_importance,
                hybrid_budgets=sweep_budgets)
            print(f"  [Task 3 elapsed: {time.time() - t0:.1f}s]")

        # Task 4: q-RASAR similarity-augmented features
        t0 = time.time()
        _space_out['qrasar'] = _run_qrasar(
            reps, labels, groups, config, _fold_bps,
            metric_fn, metric_name, task_type=task_type,
            lower_is_better=lower_is_better,
            augmentations=augmentations,
            fold_splits=fold_splits, fold_importance=fold_importance,
            hybrid_budgets=sweep_budgets)
        print(f"  [Task 4 elapsed: {time.time() - t0:.1f}s]")

        _space_out['built_under_objective'] = (
            'predictive' if _space == _primary_space else 'similarity')
        _space_out['features_chosen_by'] = describe_pool_family(
            _space_out['built_under_objective'])
        results['by_hybrid_space'][_space] = _space_out
        if _space == _primary_space:
            for _k in ('weighted_knn', 'weighted_gp', 'weighted_svr', 'qrasar'):
                results[_k] = _space_out[_k]

    # Every task key exists on every run, whatever the pool held, so a reader
    # never has to tell an absent key from a skipped one.
    for _k in ('weighted_knn', 'weighted_gp', 'weighted_svr', 'qrasar'):
        results.setdefault(_k, {
            'skipped': True,
            'reason': (f'the pool gave no recipe under objective=predictive on '
                       f'any fold, so the {_primary_space} space was not '
                       f'built. Spaces scored: '
                       f'{_spaces_to_run or "none"}.')})

    # -------------------------------------------------------------------------
    # HONEST (selection-corrected) RE-SCORES. Both blocks are kept OUT of the
    # _orient_deltas loop below: their 'delta' fields are ALREADY direction-
    # corrected by fold_reselection, so re-orienting would double-negate them.
    # -------------------------------------------------------------------------
    # (1) Standard contract block: per-fold scores for BOTH arms of the pooled
    #     hybrid search (all hybrid candidate recipes + baseline single reps),
    #     straight from the SearchResult, so a reader can call fold_reselection()
    #     without re-running the search. Empty / old-cache searches yield empty
    #     dicts (the helper tolerates them). Only present when a search was got.
    if search is not None:
        try:
            results['fold_reselection'] = build_fold_reselection_input(search)
            # 2026-09-12, owner's decision: "s10's pool subset includes
            # similarity hybrids". These fold scores are the winning search's
            # own, so they are in that search's metric — the dataset's metric
            # under the predictive objective, and the correlation between each
            # held-out molecule's activity and what its five nearest
            # neighbours predict under objective='similarity'. Named here so
            # they are not read against the per-fold scores this file computes.
            results['fold_reselection_family'] = (
                pool_meta['objective'] if pool_meta is not None else None)
            results['fold_reselection_metric'] = (
                pool_meta['metric_name'] if pool_meta is not None else None)
        except Exception as e:
            results['fold_reselection_error'] = f'{type(e).__name__}: {e}'

    # ── The top TWO of each arm, and the two reference numbers ───────────────
    # Everything above reports one thing per arm: 'top3_base_reps' is one
    # ranking by one downstream model, and each fold's recipe is the pool's
    # rank-1 for that fold. Nothing said what came second or by how much. All
    # four calls below read rows the whole-dataset pool search already stored,
    # so they cost 0 new model fits and touch no fold.
    #
    # They sit at the TOP LEVEL of `results`, not inside one of the four task
    # blocks, and that placement is load-bearing: _orient_deltas below negates
    # every field whose name starts with 'delta' inside 'weighted_knn',
    # 'weighted_gp', 'weighted_svr' and 'qrasar' when the dataset's metric is
    # lower-is-better. hybrid_deltas already returns both of its numbers with
    # positive meaning the hybrid won, so either of them inside one of those
    # blocks would be negated a second time.
    #
    # DISTANCE_MODELS is the six downstream models this lineage scores (rf,
    # xgboost, svm, dnn, lightgbm, knn); every one is asked for.
    # require_full=False so a downstream model this search never scored comes
    # back as a short list carrying its own shortfall_reason, instead of raising
    # and losing the four tasks' results after every fit has been paid for.
    if search is not None:
        _top2_base, _top2_hyb, _hyb_deltas = {}, {}, {}
        for _m in DISTANCE_MODELS:
            try:
                _top2_base[_m] = top_base_reps(search, _m, 2,
                                               require_full=False)
            except Exception as _e:
                print(f"  top-2 base representations for {_m} unavailable: {_e}")
            _th = None
            try:
                # No budget= filter. This file asks the pool for a recipe at
                # `requested_budget` features and the pool substitutes one
                # searched at another number when it holds none at that number,
                # so filtering the stored rows to the requested number can leave
                # nothing and the two deltas below would then be absent. Every
                # returned entry carries the number of features it was searched
                # at under 'budget'.
                _th = top_hybrids(search, _m, 2, require_full=False)
                _top2_hyb[_m] = _drop_blueprint(_th)
            except Exception as _e:
                print(f"  top-2 hybrid configurations for {_m} unavailable: {_e}")
            # The rank-1 hybrid is handed to hybrid_deltas explicitly. Left to
            # pick its own it reads _picked['selected'][0]['row']
            # (_shared.py:7432), and a top_hybrids entry has no 'row' key, so
            # that path raises KeyError('row'). The entry does carry
            # 'score_mean' and 'blueprint', which is all hybrid_deltas reads off
            # the hybrid it is given.
            if _th and _th['selected']:
                try:
                    _hyb_deltas[_m] = hybrid_deltas(
                        search, _m, hybrid=_th['selected'][0])
                except Exception as _e:
                    print(f"  hybrid deltas for {_m} unavailable: {_e}")
        _top2_pairs = {}
        try:
            # The same 2-and-2 grid in one block: per downstream model the 2
            # base representations and the 2 hybrids, plus 'n_distinct_units' --
            # how many separate things would have to be BUILT once one
            # representation picked by several downstream models counts once.
            _top2_pairs = _drop_blueprint(
                select_model_rep_pairs(search, models=list(DISTANCE_MODELS),
                                       n_base=2, n_hybrids=2,
                                       require_full=False))
        except Exception as _e:
            print(f"  top-2 model-and-representation grid unavailable: {_e}")

        # Added 2026-09-18, additive. Four new top-level keys; no key set above
        # changes value -- 'top3_base_reps', 'fold_recipes', 'budget',
        # 'fold_reselection' and the four task blocks are what they were.
        results['top2_base_reps'] = _top2_base
        results['top2_hybrids'] = _top2_hyb
        results['top2_pairs_by_model'] = _top2_pairs
        results['hybrid_deltas_by_model'] = _hyb_deltas

        # Printed because the two deltas answer different questions and a reader
        # given one number cannot tell which was asked. The first is the hybrid
        # against the mean of the representations the budget actually gave
        # features to, read off Blueprint.allocation_per_fold, so a
        # representation given 0 features is not in that mean. The second is the
        # hybrid against the single best base representation for the same
        # downstream model. A hybrid can win the first and lose the second.
        #
        # The unit is the winning SEARCH's metric, not the dataset's: since
        # 2026-09-12 the recipe can come from a similarity-built search, whose
        # stored scores are a Spearman correlation named 'readacross'.
        if _hyb_deltas:
            print(f"\n  TOP HYBRID vs TWO REFERENCE NUMBERS "
                  f"({getattr(search, 'metric_name', None) or metric_name}, "
                  f"+ means the hybrid won):")
            for _m in DISTANCE_MODELS:
                _hd = _hyb_deltas.get(_m)
                if not _hd:
                    continue
                _dsm = _hd.get('delta_vs_source_mean')
                _dbr = _hd.get('delta_vs_best_rep')
                _left = (f"vs the mean of the {_hd['n_source_reps']} "
                         f"representations that fed it "
                         f"({_hd['source_mean']:.4f}): {_dsm:+.4f}"
                         if _dsm is not None else
                         "vs the mean of its source representations: --")
                _right = (f"vs the best single representation "
                          f"{_hd['best_rep']} ({_hd['best_rep_score']:.4f}): "
                          f"{_dbr:+.4f}"
                          if _dbr is not None else
                          "vs the best single representation: --")
                print(f"    {_m}: {_left}; {_right}")

    # (2) Within-file weighted-vs-unweighted kNN: the stored best k is picked on
    #     the SAME folds it scores on (s10:210 + _print_summary). Re-pick the
    #     winning k per held-out fold via the shared honest arithmetic on the
    #     per-fold scores just persisted (weighted arm = candidates, unweighted
    #     arm = baselines; delta + == weighted better).
    if isinstance(results.get('weighted_knn'), dict):
        results['reselected_within_arm'] = {
            'weighted_knn': _reselected_best_k_within_arm(
                results['weighted_knn'], lower_is_better)
        }
    # The same arithmetic in every other hybrid space. Pure arithmetic on the
    # per-fold scores already stored: 0 new model fits. Added 2026-09-19 with
    # the second space; the key above is unchanged and is still the predictive
    # space's.
    for _sp, _blk in (results.get('by_hybrid_space') or {}).items():
        if isinstance(_blk.get('weighted_knn'), dict):
            _blk['reselected_within_arm'] = {
                'weighted_knn': _reselected_best_k_within_arm(
                    _blk['weighted_knn'], lower_is_better)
            }

    # Orient every stored delta so positive == improvement regardless of metric
    # direction (RMSE etc.). MUST run before _print_summary, which now treats
    # the stored deltas as already direction-corrected.
    #
    # ⛔ ONCE PER OBJECT. `results['weighted_knn']` and
    # `results['by_hybrid_space']['hybrid']['weighted_knn']` are the SAME dict,
    # and _orient_deltas negates in place -- orienting it twice would put every
    # rmse delta back the way it came. Tracked by identity, so each task block
    # is oriented exactly once whichever key reaches it first, and a space that
    # this file scored but never filed at the top level is oriented too.
    _TASK_KEYS = ('weighted_knn', 'weighted_gp', 'weighted_svr', 'qrasar')
    _oriented = set()
    for _blk in [results] + list((results.get('by_hybrid_space') or {}).values()):
        for _task_key in _TASK_KEYS:
            _t = _blk.get(_task_key)
            if isinstance(_t, dict) and id(_t) not in _oriented:
                _orient_deltas(_t, lower_is_better)
                _oriented.add(id(_t))
    results['deltas_oriented'] = True  # positive delta == improvement everywhere

    # Summary
    print(f"\n  Total elapsed: {time.time() - start_total:.1f}s")
    _print_summary(results, metric_name, lower_is_better)

    output = make_structured_output(
        's10_importance_downstream', dataset_name, data, config, results,
        primary_metric=metric_name, lower_is_better=lower_is_better,
    )
    return output


def _print_summary(results, metric_name, lower_is_better):
    """Print a concise cross-task summary."""
    print(f"\n  {'='*60}")
    print(f"  SUMMARY (metric={metric_name}, lower_is_better={lower_is_better})")
    print(f"  {'='*60}")

    # Deltas are already direction-oriented (positive == improvement) by
    # _orient_deltas before this is called, so select/print them as-is.
    sign = 1

    for task_name, task_key, delta_key in [
        ('Weighted kNN', 'weighted_knn', 'delta'),
        ('Weighted GP', 'weighted_gp', 'delta'),
        ('Weighted SVR', 'weighted_svr', 'delta'),
        ('q-RASAR (w vs base)', 'qrasar', 'delta_w'),
    ]:
        task_res = results.get(task_key, {})
        if task_res.get('skipped'):
            print(f"    {task_name}: SKIPPED ({task_res.get('reason', '')})")
            continue

        per_rep = task_res.get('per_rep', {})
        hybrid = task_res.get('hybrid', {})
        # mean_delta averages the unpaired differences (each arm over its own
        # non-NaN folds); mean_delta_paired averages the per-fold differences
        # taken only over folds both arms scored.
        paired_key = f'{delta_key}_paired'

        if per_rep:
            # For kNN, per_rep values are dicts of k-values; for others, flat
            if task_key == 'weighted_knn':
                deltas, paired = [], []
                for rep_name, k_results in per_rep.items():
                    for kk, vals in k_results.items():
                        if not isinstance(vals, dict):
                            continue
                        deltas.append((rep_name, kk, vals.get(delta_key, 0)))
                        if paired_key in vals:
                            paired.append(vals[paired_key])
                best = max(deltas, key=lambda x: sign * x[2])
                mean_delta = float(np.mean([d[2] for d in deltas]))
                print(f"    {task_name}: mean_delta={mean_delta:+.4f}, "
                      f"best={best[0]}({best[1]}) delta={best[2]:+.4f}", end='')
            else:
                deltas = [(rn, v.get(delta_key, 0)) for rn, v in per_rep.items()]
                paired = [v[paired_key] for v in per_rep.values()
                          if isinstance(v, dict) and paired_key in v]
                best = max(deltas, key=lambda x: sign * x[1])
                mean_delta = float(np.mean([d[1] for d in deltas]))
                print(f"    {task_name}: mean_delta={mean_delta:+.4f}, "
                      f"best={best[0]} delta={best[1]:+.4f}", end='')
            if paired and not np.all(np.isnan(paired)):
                print(f", mean_delta_paired={float(np.nanmean(paired)):+.4f}", end='')

        # Hybrid
        if hybrid and not isinstance(hybrid, dict):
            pass
        elif hybrid and delta_key in hybrid:
            print(f", hybrid_delta={hybrid[delta_key]:+.4f}", end='')
            if paired_key in hybrid:
                print(f", hybrid_delta_paired={hybrid[paired_key]:+.4f}", end='')
        elif isinstance(hybrid, dict):
            # kNN hybrid: iterate k-values
            h_deltas = [v.get(delta_key, 0) for v in hybrid.values()
                        if isinstance(v, dict) and delta_key in v]
            h_paired = [v[paired_key] for v in hybrid.values()
                        if isinstance(v, dict) and paired_key in v]
            if h_deltas:
                print(f", hybrid_delta={np.mean(h_deltas):+.4f}", end='')
            if h_paired and not np.all(np.isnan(h_paired)):
                print(f", hybrid_delta_paired={float(np.nanmean(h_paired)):+.4f}",
                      end='')

        print()


if __name__ == '__main__':
    parser = create_experiment_parser(
        'S10: Importance Weights for Distance-Dependent Downstream Tasks',
        has_delta=True, has_augment=True)
    parser.add_argument('--budget', type=int, default=100,
                        help='Budget for hybrid feature selection')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  budget=args.budget)
    save_experiment_results(results, args.output,
                            's10_importance_downstream', args.dataset)
