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
results['hybrid_space_errors']. Both spaces are built and scored on every run
where the pool holds both families; a family the pool has nothing for on this
dataset is named in results['hybrid_space_errors'] instead.

ONE RECIPE PER TASK (2026-09-28). All four tasks were built from one recipe, the
one the pool ranked on the stored rows of 'knn'. Task 1 fits a
k-nearest-neighbours model, so that was its own. Task 2 fits a Gaussian process,
Task 3 a support vector regressor and Task 4 a random forest on four
neighbour-derived features, and all three were scored on features chosen for a
k-nearest-neighbours fit while their rows were written under their own names.
Each task now gets the recipe ranked on the rows of the model it fits, inside
each family, on the other folds as before. A Gaussian process is not one of the
six models any search in this lineage scores (rf, xgboost, svm, dnn, lightgbm,
knn), so it gets the best configuration over every model the search did score
and results['task_recipe_chosen_by'] names the model whose score put it there.
results['task_downstream_model'] names the model per task,
results['task_recipes'] holds one recipe per (space, task, fold) and
results['task_recipe_source'] says own_rows or across_models for each. Task 1's
recipe is the same entry as before, so its numbers do not move. This costs no
extra model fit: each task already built its own hybrid matrix per (space,
width, fold) and only which recipe it builds changes.

The weighted GP arm was a constant predictor (2026-09-20). The weight vector
this file built was rescaled to SUM to 1. normalise_weights in
src/kirby/similarity/preprocessing.py rescales so the weights AVERAGE 1, which
is what f17 measured on 2026-08-25 and what every other weighted distance in
the repository uses; its docstring says sum-to-1 "divides every distance by the
feature count" and that a kernel width does not survive it. So on a 2048-feature
representation the GP's RBF, whose length scale starts at 1.0 in the units of
the features, saw every pairwise distance divided by 2048 and fitted a kernel
matrix that is constant to within rounding. It predicted the training mean, and the arithmetic
says so exactly. A predictor that returns the training fold's mean label and
nothing else scores 0.4371 rmse on caco2 over these five folds, 0.6881 on
adme_mdr1 and 0.9160 on herg_ki. The weighted GP reported 0.4371 on six of
caco2's seven representations, 0.6881 on six of adme_mdr1's and 0.9160 on
herg_ki's two 2048-feature ones. Those are the same numbers to four decimals,
not near ones. pdv, at 200 features, is the one that partly escaped on each
dataset.

Two consequences. The printed per-task delta is RAW -- the print statements run
before _orient_deltas -- so on these three rmse datasets a POSITIVE printed
delta means the weighted arm had the HIGHER error, which is worse. And Tasks 1
and 3 are unaffected by the rescale: a kNN only ranks distances and an SVR at
gamma='scale' divides by the data's own variance, so both are exactly invariant
to a uniform shrink. Task 4's four neighbour-derived features feed a random
forest, whose splits are invariant to a monotone rescale of any one feature, so
it is unaffected too. Task 2 is the only one that moved.

KIRBY_S10_WEIGHT_CONVENTION now picks the rescale and defaults to 'mean1'; pass
'sum1' to reproduce the old arm. KIRBY_IMPORTANCE_EXPONENT picks the power the
importance is raised to (0.5 unset, 0 = unweighted control),
KIRBY_S10_K_VALUES the neighbour counts Task 1 scores, KIRBY_S10_KNN_VOTE
sklearn's uniform-or-distance vote, KIRBY_S10_QRASAR_K how many neighbours
Task 4's four features are built from, and KIRBY_S10_GP_RESTARTS how many extra
starts the GP's marginal-likelihood optimiser gets, now 0 rather than 2 --
both arms got all three starts, so they cancelled out of the difference this
task reports while costing three times the Cholesky work.

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

import copy
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel, Hyperparameter, RBF, WhiteKernel, ConstantKernel)
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
    # Ranks every model's stored hybrid configurations together and returns the
    # best. Added 2026-09-28 for the Gaussian process, which no stored search
    # names, so it has no configuration of its own to rank.
    top_hybrids_across_models,
    resolve_comparison_models,
    POOL_SUBSETS,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS, FULL_BUDGETS,
    flatten_fold_weights,
    fold_reselection, build_fold_reselection_input,
)
from kirby.find_best_hybrid import nested_cv_outer_folds

from kirby.hybrid import apply_feature_selection
from kirby.hybrid import (RANKING_METRICS as _RANKING_METRICS,
                          _graded_scores_for_clf)


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

# The families the subset above admits, in the order POOL_SUBSETS names them.
_POOL_FAMILIES = POOL_SUBSETS[_POOL_SUBSET]

# The downstream model each of the four tasks FITS, and so the model whose
# stored rows that task's recipe is ranked on.
#
# Until 2026-09-28 all four tasks were built from one recipe, ranked on the
# stored rows of 'knn'. Task 1 does fit a k-nearest-neighbours model. Task 2 fits
# a Gaussian process, Task 3 a support vector regressor and Task 4 a random
# forest on four neighbour-derived features, and those three were scored on
# features chosen for a k-nearest-neighbours fit while the rows were written
# under their own names.
#
# ⛔ 'gp' IS NOT A NAME ANY STORED SEARCH HOLDS. DISTANCE_MODELS is rf, xgboost,
# svm, dnn, lightgbm and knn, and a search under a geometry objective scores the
# same six. So the Gaussian process has no configuration of its own at all and
# takes the best configuration over every model the search scored
# (top_hybrids_across_models). results['task_recipe_source'] says which of the
# two every task got, per hybrid space.
#
# 'svm' in a stored search is this repository's support vector regressor and
# Task 3 fits sklearn's SVR at gamma='scale' directly, so the recipe is ranked on
# the rows of the same kind of model but not of the identical settings. Task 1's
# k-nearest-neighbours model and Task 4's random forest are the same model the
# search scored under those names.
_MODEL_FOR_TASK = {
    'weighted_knn': 'knn',
    'weighted_gp': 'gp',
    'weighted_svr': 'svm',
    'qrasar': 'rf',
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

def _importance_weights(imp_scores, exponent=None, convention=None):
    """Importance scores turned into the multiplier for standardised features.

    Two steps. The importance is raised to a power, and the result is rescaled.

    The power is `exponent`. None reads KIRBY_IMPORTANCE_EXPONENT through the
    shared importance_feature_weights, and an unset environment gives 0.5, which
    is the square root this function always took. 0 makes every feature weigh
    the same, which is the unweighted control.

    The rescale is `convention`, read from KIRBY_S10_WEIGHT_CONVENTION when
    None. Until 2026-09-20 it was sum-to-1 and could not be changed. Under
    sum-to-1 a 2048-feature representation has every distance divided by 2048,
    and normalise_weights' own docstring says a kernel width does not survive
    that. Task 1 fits a kNN and Task 3 an SVR at gamma='scale', both of which
    only rank distances or rescale by the data's own variance, so neither
    notices. Task 2 fits a GP whose RBF length scale starts at 1.0 in the units
    of the features, and it collapsed to a constant predictor. The default is
    now 'mean1', which is this repository's measured choice (f17, 2026-08-25)
    and keeps a z-scored feature near its own scale. Pass 'sum1' to reproduce
    the old arm.
    """
    from kirby.similarity.preprocessing import normalise_weights
    from experiments._shared import importance_feature_weights
    w = importance_feature_weights(imp_scores, exponent=exponent)
    conv = convention if convention is not None else _weight_convention()
    if conv == 'sum1':
        return w / (w.sum() + 1e-10)
    return normalise_weights(w, conv)


def _weight_convention():
    """How the weight vector is rescaled before it multiplies the features.

    'mean1' divides by the mean weight, 'sum1' by the sum. Reads
    KIRBY_S10_WEIGHT_CONVENTION; unset gives 'mean1'.
    """
    return os.environ.get('KIRBY_S10_WEIGHT_CONVENTION', 'mean1').strip() or 'mean1'


def _knn_vote_weights():
    """sklearn's `weights` argument for every KNeighbors model in this file.

    'uniform' averages the k neighbours' labels, which is what this file did.
    'distance' averages them weighted by one over the distance. Reads
    KIRBY_S10_KNN_VOTE; unset gives 'uniform'.
    """
    return os.environ.get('KIRBY_S10_KNN_VOTE', 'uniform').strip() or 'uniform'


def _k_values():
    """The neighbour counts Task 1 scores, one arm each.

    Reads KIRBY_S10_K_VALUES as a comma-separated list; unset gives 5,10,20,
    which is what this file had written out.
    """
    raw = os.environ.get('KIRBY_S10_K_VALUES', '').strip()
    if not raw:
        return [5, 10, 20]
    return [int(v) for v in raw.split(',') if v.strip()]


def _qrasar_ks():
    """How many nearest training rows the four q-RASAR features are built from.

    A list now, one arm each, where it was one number. Reads
    KIRBY_S10_QRASAR_K as a comma-separated list; unset gives 5, 10 and 20.
    The FIRST entry is the one the unsuffixed 'per_rep' and 'hybrid_by_budget'
    blocks report, and 10 leads the default list because 10 is the single
    number this file used before the sweep, so those two blocks hold what they
    always held.

    The four features are mean and standard deviation of the neighbours'
    labels, the distance to the nearest of them and the mean distance to all of
    them, so the count changes what all four measure. Sweeping it costs one
    neighbour search and one forest fit per extra count per fold, and no extra
    hybrid build: the feature matrix the neighbours are found in does not
    depend on the count.
    """
    raw = os.environ.get('KIRBY_S10_QRASAR_K', '').strip()
    if not raw:
        return [10, 5, 20]
    return [int(v) for v in raw.split(',') if v.strip()]


def _qrasar_block(base_scores, uw_scores, w_scores):
    """One q-RASAR arm's three means and three differences.

    base = the forest on the features alone, uw = the same forest with the
    four neighbour features appended, w = the same four built in the
    importance-weighted distance instead.
    """
    base_mean = float(np.nanmean(base_scores))
    uw_mean = float(np.nanmean(uw_scores))
    w_mean = float(np.nanmean(w_scores))
    d_uw_p, n_uw_p = _paired_delta(uw_scores, base_scores)
    d_w_p, n_w_p = _paired_delta(w_scores, base_scores)
    d_wvu_p, n_wvu_p = _paired_delta(w_scores, uw_scores)
    return {
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
        'base_plus_sim_unweighted_folds': [float(s) for s in uw_scores],
        'base_plus_sim_weighted_folds': [float(s) for s in w_scores],
    }


def _predict_for_metric(model, X, metric_name, is_clf):
    """What metric_fn is handed for this fitted model.

    On a classification dataset scored by AUROC (humanness_therapeutic), the
    hard classes from predict() turn AUROC into (sensitivity + specificity) / 2
    and throw away the ranking it measures. The pool search scores the same
    models from predict_proba's positive column, so this does too, through the
    engine's own helper. Every other case gets predict(), as before.
    """
    y_hard = model.predict(X)
    if is_clf and metric_name in _RANKING_METRICS:
        return _graded_scores_for_clf(model, X, y_hard)
    return y_hard


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


def _per_rep_arms_to_score(reps, cache, results, task_name):
    """Which representations this task still has to score in this hybrid space.

    Every task's per-representation arms read `reps`, `fold_splits` and
    `fold_importance` and never touch a recipe, so the second hybrid space
    recomputed numbers identical to the first's. The Gaussian process is where
    that cost: 7 representations x 5 folds x 2 arms, fitted twice for one set
    of numbers.

    `cache` carries the first space's block. What goes into this space's result
    is a COPY of it, not the same object: _orient_deltas negates in place once
    per task block, and one dict reached from two blocks would be negated twice
    and put every rmse delta back the way it came.

    Returns the representation names still to score — all of them the first
    time, none of them afterwards.
    """
    if cache is not None and task_name in cache:
        for key, value in cache[task_name].items():
            results[key] = copy.deepcopy(value)
        results['per_rep_reused_from_first_space'] = True
        print(f"    per-representation arms: the {len(results['per_rep'])} "
              f"scored in the first hybrid space are reused, because they do "
              f"not read a recipe")
        return []
    results['per_rep_reused_from_first_space'] = False
    return sorted(reps.keys())


def _store_per_rep(results, cache, task_name, extra_keys=()):
    """Keep this task's per-representation blocks for the next hybrid space."""
    if cache is not None and task_name not in cache:
        cache[task_name] = {
            k: copy.deepcopy(results.get(k) or {})
            for k in ('per_rep',) + tuple(extra_keys)}


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
                      hybrid_budgets=None, per_rep_cache=None):
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

    k_values = _k_values()
    vote = _knn_vote_weights()
    results = {'per_rep': {}, 'hybrid': {}}

    # Per-rep evaluation
    for rep_name in _per_rep_arms_to_score(reps, per_rep_cache, results,
                                           'weighted_knn'):
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
                model = KNN(n_neighbors=k_eff, weights=vote)
                model.fit(X_tr, y_train)
                uw_scores.append(metric_fn(y_val, _predict_for_metric(model, X_va, metric_name, is_clf)))

                # Weighted (use precomputed importance)
                try:
                    imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                    if imp_arr is None:
                        raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                    w = _importance_weights(imp_arr)
                    model_w = KNN(n_neighbors=k_eff, weights=vote)
                    model_w.fit(X_tr * w, y_train)
                    w_scores.append(metric_fn(y_val, _predict_for_metric(model_w, X_va * w, metric_name, is_clf)))
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

    _store_per_rep(results, per_rep_cache, 'weighted_knn')

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}
        n_builds = 0
        for budget in budgets:
            bkey = _budget_key(budget)
            uw_by_k = {k: [] for k in k_values}
            w_by_k = {k: [] for k in k_values}

            # The fold is outside and the neighbour count inside, which is the
            # other way round from how this read before. A recipe searched
            # under a geometry objective fits no downstream model, so
            # fold_hybrid_scoring_kwargs hands the build nothing that depends
            # on the neighbour count and the three counts were building one
            # feature matrix three times. A predictive recipe still builds once
            # per neighbour count: the accept/reject test inside the build fits
            # the kNN that consumes the allocation, and that kNN carries k.
            for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
                blueprint = fold_blueprints.get(fold_i)
                if blueprint is None:
                    continue
                train_reps = {n: r[train_idx] for n, r in reps.items()}
                val_reps = {n: r[val_idx] for n, r in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None
                _KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                _build_reads_k = bool(fold_hybrid_scoring_kwargs(
                    blueprint, model=_KNN(n_neighbors=1), metric_fn=metric_fn,
                    lower_is_better=lower_is_better))
                built = None

                for k in k_values:
                    k_eff = min(k, len(y_train) - 1)
                    if built is None or _build_reads_k:
                        try:
                            built = build_fold_hybrid(
                                train_reps, val_reps, y_train, blueprint,
                                augmentations_train=aug_tr,
                                augmentations_val=aug_va,
                                # The number of features this arm is built
                                # with, one of the nine swept in run(). None
                                # leaves the number this fold's own recipe was
                                # searched at.
                                budget_override=budget,
                                groups=(groups[train_idx]
                                        if groups is not None else None),
                                # This arm's downstream estimator IS a kNN
                                # scored with metric_fn — pass both so
                                # create_hybrid's internal accept/reject test
                                # judges allocations by the model that actually
                                # consumes them, instead of falling back to its
                                # internal r2/accuracy proxy.
                                **fold_hybrid_scoring_kwargs(
                                    blueprint,
                                    model=_KNN(n_neighbors=k_eff, weights=vote),
                                    metric_fn=metric_fn,
                                    lower_is_better=lower_is_better))
                            n_builds += 1
                        except Exception as e:
                            _record_build_error(hybrid_build_errors, e, k=k,
                                                fold=fold_i, budget=budget)
                            built = None
                            continue
                    X_tr_h, X_va_h, feat_info = built

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr_h)
                    X_va = scaler.transform(X_va_h)

                    # Unweighted
                    KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                    model = KNN(n_neighbors=k_eff, weights=vote)
                    model.fit(X_tr, y_train)
                    uw_by_k[k].append(metric_fn(y_val, _predict_for_metric(model, X_va, metric_name, is_clf)))

                    # Weighted
                    try:
                        w = _compute_importance_for_hybrid(feat_info)
                        assert w.shape[0] == X_tr.shape[1], (
                            f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                        model_w = KNN(n_neighbors=k_eff, weights=vote)
                        model_w.fit(X_tr * w, y_train)
                        w_by_k[k].append(
                            metric_fn(y_val, _predict_for_metric(model_w, X_va * w, metric_name, is_clf)))
                    except Exception:
                        w_by_k[k].append(float('nan'))

                # Free this fold's feature matrices before the next one.
                built = None

            by_k = {}
            for k in k_values:
                uw_scores, w_scores = uw_by_k[k], w_by_k[k]
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
        results['n_hybrid_builds'] = n_builds
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

_GP_LS_BOUNDS = (1e-5, 1e5)


def _sqdist(X, Y):
    """Squared euclidean distance between every row of X and every row of Y."""
    d = (np.einsum('ij,ij->i', X, X)[:, None]
         + np.einsum('ij,ij->i', Y, Y)[None, :] - 2.0 * (X @ Y.T))
    np.maximum(d, 0, out=d)
    return d


class _CachedRBF(Kernel):
    """The same RBF, over squared distances computed once instead of per step.

    sklearn's RBF rebuilds every pairwise distance from all of the features at
    each step of the marginal-likelihood optimiser. Only the length scale
    changes between steps, so the distances do not: this computes them once per
    training matrix and then evaluates exp(-d / 2 l^2) on them. Same kernel,
    same gradient. On 2,048 features that is the difference between a step
    costing 1,412 x 1,412 x 2,048 multiplications and one costing
    1,412 x 1,412.

    The cached entry holds the matrix it was computed from as well as the
    distances, so the array cannot be freed and another array take its id.

    _gp_kernel_equivalence() below fits both kernels on the same small matrix
    and reports the largest difference in their held-out predictions; run()
    stores that number in the result file. KIRBY_S10_GP_KERNEL=sklearn puts
    sklearn's own RBF back.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=_GP_LS_BOUNDS):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self._cache = {}

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter('length_scale', 'numeric',
                              self.length_scale_bounds)

    def _d2(self, X):
        key = id(X)
        hit = self._cache.get(key)
        if hit is None or hit[0] is not X:
            hit = (X, _sqdist(X, X))
            self._cache[key] = hit
        return hit[1]

    def __call__(self, X, Y=None, eval_gradient=False):
        ls = float(np.squeeze(self.length_scale))
        d2 = self._d2(X) if Y is None else _sqdist(X, Y)
        K = np.exp(-0.5 * d2 / (ls * ls))
        if not eval_gradient:
            return K
        if Y is not None:
            raise ValueError('gradient is only defined when Y is None')
        return K, (K * (d2 / (ls * ls)))[:, :, np.newaxis]

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


def _gp_kernel_in_use():
    """'cached' (the default) or 'sklearn'. Reads KIRBY_S10_GP_KERNEL."""
    return os.environ.get('KIRBY_S10_GP_KERNEL', 'cached').strip() or 'cached'


def _gp_kernel_equivalence():
    """Largest difference in prediction between the two kernels, 20 held-out rows.

    Fits both on 60 rows of 8 gaussian features and predicts 20 more. A number
    at the size of floating-point rounding is what says the cached kernel
    changed nothing; anything larger means it did.
    """
    try:
        r = np.random.default_rng(0)
        X, y = r.normal(size=(60, 8)), r.normal(size=60)
        X_test = r.normal(size=(20, 8))
        preds = []
        for K in (RBF(1.0), _CachedRBF(1.0)):
            g = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * K + WhiteKernel(1e-3),
                n_restarts_optimizer=0, alpha=1e-6, random_state=42)
            g.fit(X, y)
            preds.append(g.predict(X_test))
        return float(np.abs(preds[0] - preds[1]).max())
    except Exception as e:
        return f'{type(e).__name__}: {e}'


_GP_FIT_BASIS = (
    "standard_fits / weighted_fits = one entry per fold: the fitted RBF "
    "length scale, in the units of the z-scored and weighted features; "
    "at_bound, meaning it finished at an end of (1e-5, 1e5) so the "
    "marginal-likelihood optimiser ran out of room rather than found an "
    "optimum; and pred_sd, the spread of that arm's own predictions over the "
    "held-out molecules, which is 0 for a constant predictor. At the bottom "
    "bound the kernel is the identity and every held-out molecule gets the "
    "prior mean; at the top it is constant and every held-out molecule gets "
    "the training mean. A fold at a bound fitted nothing and its score is not "
    "a measurement of weighting.")


def _gp_normalize_y():
    """Whether the GP centres and scales the labels before fitting.

    False, which is what this file has always done, leaves the prior mean at 0.
    A fit that finishes at the bottom of the length-scale bounds then predicts 0
    for every held-out molecule. True centres the labels, so the same fit
    predicts the training mean instead. Reads KIRBY_S10_GP_NORMALIZE_Y; unset
    gives False.
    """
    raw = os.environ.get('KIRBY_S10_GP_NORMALIZE_Y', '').strip().lower()
    return raw in ('1', 'true', 'yes', 'on')


def _gp_lengthscale_starts():
    """Where the RBF's length scale starts, one arm each.

    'fixed' is 1.0 with bounds 1e-5 to 1e5, which is what this file always
    fitted and what every number reported from it was measured under. 'median'
    starts at the median euclidean distance between two training molecules in
    the features the arm is fitted on, with bounds that value divided and
    multiplied by 100 — the same lever s11 carries as KIRBY_S11_LENGTHSCALE and
    the same convention, so the two files can be read side by side.

    Why it is swept and not replaced: the arms that collapse are part of what
    the comparison is about. A fit that finished at a bound is recorded by
    _gp_fit_report either way.

    Each arm takes the median of ITS OWN training features. The weighted arm
    multiplies every feature by its importance, so its distances are not the
    unweighted arm's and one median would mis-start one of them. The start is
    read off the data the GP sees, not chosen.

    Reads KIRBY_S10_GP_LENGTHSCALE as a comma-separated list; unset gives
    fixed then median, and the FIRST entry is the one the unsuffixed blocks
    report.
    """
    raw = os.environ.get('KIRBY_S10_GP_LENGTHSCALE', '').strip()
    if not raw:
        return ['fixed', 'median']
    return [v.strip() for v in raw.split(',') if v.strip()]


def _median_pair_distance(X):
    """Median euclidean distance between two rows of X, or None.

    Zero distances are dropped: two identical molecules carry no scale.
    """
    try:
        A = np.asarray(X, dtype=np.float64)
        if A.shape[0] < 3:
            return None
        d2 = _sqdist(A, A)
        iu = np.triu_indices(d2.shape[0], k=1)
        d = np.sqrt(d2[iu])
        d = d[d > 0]
        return float(np.median(d)) if d.size else None
    except Exception:
        return None


def _gp_fit_report(gp, preds, start=None, start_value=None):
    """What the GP settled on, so a degenerate fit is recorded, not just scored.

    Measured 2026-09-20 on herg_ki with the weights rescaled to sum to 1: pdv
    fold 0 finished at the 1e-5 bound and scored 5.6330 rmse against a 0.9154
    pKi label spread, while the same GP on unweighted features settled at a
    length scale of 21.28 and scored 0.5963.

    'at_bound' is read against the bounds the fitted kernel actually carries,
    which are 1e-5 to 1e5 under the 'fixed' start and the median distance
    divided and multiplied by 100 under 'median'.
    """
    try:
        ls = float(np.squeeze(gp.kernel_.k1.k2.length_scale))
    except Exception:
        return None
    try:
        lo, hi = (float(v) for v in
                  np.ravel(gp.kernel_.k1.k2.length_scale_bounds)[:2])
    except Exception:
        lo, hi = _GP_LS_BOUNDS
    return {'length_scale': ls,
            'length_scale_bounds': [lo, hi],
            'lengthscale_start': start,
            'lengthscale_start_value': start_value,
            'at_bound': bool(ls <= lo * 1.01 or ls >= hi * 0.99),
            'pred_sd': float(np.std(preds))}


def _gp_block(std_scores, w_scores, std_fits, w_fits):
    """One GP arm's two means, its difference and its fit records."""
    std_mean = float(np.nanmean(std_scores))
    w_mean = float(np.nanmean(w_scores))
    paired, n_paired = _paired_delta(w_scores, std_scores)
    return {
        'standard': std_mean,
        'weighted': w_mean,
        'delta': w_mean - std_mean,
        'delta_paired': paired,
        'n_paired_folds': n_paired,
        'standard_folds': [float(s) for s in std_scores],
        'weighted_folds': [float(s) for s in w_scores],
        'standard_fits': std_fits,
        'weighted_fits': w_fits,
        'n_standard_at_bound': sum(1 for f in std_fits if f and f['at_bound']),
        'n_weighted_at_bound': sum(1 for f in w_fits if f and f['at_bound']),
        'fit_basis': _GP_FIT_BASIS,
    }


def _run_weighted_gp(reps, labels, groups, config, fold_blueprints,
                     metric_fn, metric_name, n_samples,
                     lower_is_better=False,
                     augmentations=None,
                     fold_splits=None, fold_importance=None,
                     hybrid_budgets=None, per_rep_cache=None):
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

    ls_starts = _gp_lengthscale_starts()
    # The start the unsuffixed blocks report. Every start is under
    # 'per_rep_by_lengthscale' and 'hybrid_by_budget_by_lengthscale'.
    ls_reported = ls_starts[0]
    results = {'per_rep': {}, 'hybrid': {},
               'per_rep_by_lengthscale': {},
               'lengthscale_starts': list(ls_starts),
               'lengthscale_reported': ls_reported,
               'lengthscale_basis': (
                   "One arm per starting length scale. 'fixed' starts the RBF "
                   'at 1.0 with bounds 1e-5 to 1e5, which is what every number '
                   "this file reported before was measured under. 'median' "
                   'starts at the median euclidean distance between two '
                   "training molecules in that arm's own features, with bounds "
                   "that value divided and multiplied by 100. 'per_rep' and "
                   "'hybrid_by_budget' hold the start named in "
                   "'lengthscale_reported'.")}

    def _make_gp(X=None, start='fixed'):
        ls, bounds = 1.0, _GP_LS_BOUNDS
        if start == 'median' and X is not None:
            _m = _median_pair_distance(X)
            if _m:
                ls, bounds = _m, (_m / 100.0, _m * 100.0)
        _rbf = (RBF(length_scale=ls, length_scale_bounds=bounds)
                if _gp_kernel_in_use() == 'sklearn'
                else _CachedRBF(length_scale=ls, length_scale_bounds=bounds))
        kernel = ConstantKernel(1.0) * _rbf + WhiteKernel(1e-3)
        # n_restarts_optimizer counts EXTRA starts of the marginal-likelihood
        # optimiser, so 2 ran the whole Cholesky sequence three times per fit.
        # Both arms of this comparison got all three, so they cancel out of the
        # weighted-minus-unweighted difference this task reports, and on
        # herg_ki at 1130 training rows a fold they were most of the ~7 hours
        # six feature spaces took. KIRBY_S10_GP_RESTARTS puts them back.
        # normalize_y decides what the GP predicts when the kernel carries no
        # information. Left at False, which is what this file has always
        # fitted, the prior mean is 0: a fit that finishes at the BOTTOM of the
        # length-scale bounds has an identity kernel and predicts 0 for every
        # held-out molecule, which on herg_ki is an rmse of about 5.6 against a
        # pKi label spread of 0.9154 — measured, in
        # /Volumes/seagate/kirby_simrun_logs/gp_arms.log. s11 passes
        # normalize_y=True at three places, so the same collapse there predicts
        # the training mean and scores about 0.92. Same collapse, two different
        # numbers, in one suite. The default here is unchanged so no number
        # moves; KIRBY_S10_GP_NORMALIZE_Y=1 fits the other one.
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=int(os.environ.get('KIRBY_S10_GP_RESTARTS', '').strip() or '0'),
            alpha=1e-6, random_state=42,
            normalize_y=_gp_normalize_y(),
        )

    # Per-rep evaluation
    for rep_name in _per_rep_arms_to_score(reps, per_rep_cache, results,
                                           'weighted_gp'):
        X_rep = reps[rep_name]
        std_by_ls = {st: [] for st in ls_starts}
        w_by_ls = {st: [] for st in ls_starts}
        std_fits_by_ls = {st: [] for st in ls_starts}
        w_fits_by_ls = {st: [] for st in ls_starts}

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # The importance-weighted features, built once per fold and fitted
            # once per starting length scale.
            try:
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
                X_tr_w, X_va_w = X_tr * w, X_va * w
            except Exception:
                X_tr_w = X_va_w = None

            for st in ls_starts:
                _sv = (_median_pair_distance(X_tr) if st == 'median' else 1.0)
                # Standard GP
                try:
                    gp = _make_gp(X_tr, st)
                    gp.fit(X_tr, y_train)
                    _p = gp.predict(X_va)
                    std_by_ls[st].append(metric_fn(y_val, _p))
                    std_fits_by_ls[st].append(_gp_fit_report(gp, _p, st, _sv))
                except Exception:
                    std_by_ls[st].append(float('nan'))
                    std_fits_by_ls[st].append(None)

                # Importance-weighted GP
                try:
                    if X_tr_w is None:
                        raise RuntimeError(
                            f"no importance-weighted features for "
                            f"fold={fold_i} rep={rep_name}")
                    _svw = (_median_pair_distance(X_tr_w)
                            if st == 'median' else 1.0)
                    gp_w = _make_gp(X_tr_w, st)
                    gp_w.fit(X_tr_w, y_train)
                    _p = gp_w.predict(X_va_w)
                    w_by_ls[st].append(metric_fn(y_val, _p))
                    w_fits_by_ls[st].append(_gp_fit_report(gp_w, _p, st, _svw))
                except Exception:
                    w_by_ls[st].append(float('nan'))
                    w_fits_by_ls[st].append(None)

        for st in ls_starts:
            results['per_rep_by_lengthscale'].setdefault(st, {})[rep_name] = \
                _gp_block(std_by_ls[st], w_by_ls[st],
                          std_fits_by_ls[st], w_fits_by_ls[st])
        # A COPY: _orient_deltas negates in place and must not reach one dict
        # twice.
        _rb = copy.deepcopy(
            results['per_rep_by_lengthscale'][ls_reported][rep_name])
        results['per_rep'][rep_name] = _rb
        for st in ls_starts:
            _blk = results['per_rep_by_lengthscale'][st][rep_name]
            _nb = _blk['n_weighted_at_bound']
            if _nb:
                print(f"    {rep_name} [{st} start]: the weighted GP's length "
                      f"scale finished at an end of its bounds on {_nb} of "
                      f"{len(_blk['weighted_fits'])} folds. Those folds fitted "
                      f"nothing, and their score is not a measurement of "
                      f"weighting.")
            print(f"    {rep_name} [{st} start]: std={_blk['standard']:.4f}, "
                  f"weighted={_blk['weighted']:.4f} "
                  f"(delta={_blk['delta']:+.4f}, "
                  f"delta_paired={_blk['delta_paired']:+.4f} "
                  f"over {_blk['n_paired_folds']} folds)")

    # Re-point the headline to the start that collapsed LESS.
    #
    # `ls_reported` was ls_starts[0], which is 'fixed'. Measured 2026-09-23 on
    # the first complete run: under 'fixed' three of adme_mdr1's seven
    # representations became CONSTANT PREDICTORS — prediction spreads of
    # 1.11e-149, 4.93e-161 and 1.64e-83 across held-out folds — and all three
    # reported 0.7939, which is what a constant predictor scores there. Every
    # consumer reading `per_rep` without also reading `per_rep_by_lengthscale`
    # was therefore carrying collapsed arms.
    #
    # A fold whose weighted prediction spread is below 1e-6 predicted one number
    # for every held-out molecule. Whichever start produces fewer such folds
    # wins the headline; ties keep ls_starts[0], so nothing moves when both
    # starts are healthy. KIRBY_S10_GP_REPORT forces a start by name.
    def _n_constant(start):
        n = 0
        for _r, _b in (results['per_rep_by_lengthscale'].get(start) or {}).items():
            for _f in (_b.get('weighted_fits') or []):
                if _f and _f.get('pred_sd') is not None and _f['pred_sd'] < 1e-6:
                    n += 1
        return n
    _forced = os.environ.get('KIRBY_S10_GP_REPORT', '').strip()
    _counts = {st: _n_constant(st) for st in ls_starts}
    if _forced in ls_starts:
        _best, _why = _forced, 'KIRBY_S10_GP_REPORT'
    else:
        _best = min(ls_starts, key=lambda st: (_counts[st], ls_starts.index(st)))
        _why = 'fewest constant-predictor folds'
    results['lengthscale_reported_basis'] = {
        'chosen': _best, 'chosen_by': _why,
        'constant_predictor_folds': _counts,
        'note': ("A fold whose weighted GP predicted one number for every "
                 "held-out molecule, counted per start. per_rep and the "
                 "unsuffixed hybrid blocks report `chosen`; every start is "
                 "under per_rep_by_lengthscale either way.")}
    if _best != ls_reported:
        print(f"    headline start moved {ls_reported} -> {_best} "
              f"(constant-predictor folds {_counts})")
        ls_reported = _best
        results['lengthscale_reported'] = _best
        for _rep_name in list(results['per_rep']):
            _src = (results['per_rep_by_lengthscale'].get(_best) or {}).get(_rep_name)
            if _src is not None:
                results['per_rep'][_rep_name] = copy.deepcopy(_src)

    _store_per_rep(results, per_rep_cache, 'weighted_gp',
                   extra_keys=('per_rep_by_lengthscale',
                               'lengthscale_reported_basis'))

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}

        results['hybrid_by_budget_by_lengthscale'] = {}
        for budget in budgets:
            bkey = _budget_key(budget)
            std_by_ls = {st: [] for st in ls_starts}
            w_by_ls = {st: [] for st in ls_starts}
            std_fits_by_ls = {st: [] for st in ls_starts}
            w_fits_by_ls = {st: [] for st in ls_starts}

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
                        # the nine swept in run(). None leaves the number this
                        # fold's own recipe was searched at. The starting
                        # length scale below does not reach the build: the GP
                        # handed to the accept/reject test is always the
                        # 1.0 one, so one build serves every start.
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

                try:
                    w = _compute_importance_for_hybrid(feat_info)
                    assert w.shape[0] == X_tr.shape[1], (
                        f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                    X_tr_w, X_va_w = X_tr * w, X_va * w
                except Exception:
                    X_tr_w = X_va_w = None

                for st in ls_starts:
                    _sv = (_median_pair_distance(X_tr) if st == 'median' else 1.0)
                    # Standard GP
                    try:
                        gp = _make_gp(X_tr, st)
                        gp.fit(X_tr, y_train)
                        _p = gp.predict(X_va)
                        std_by_ls[st].append(metric_fn(y_val, _p))
                        std_fits_by_ls[st].append(
                            _gp_fit_report(gp, _p, st, _sv))
                    except Exception:
                        std_by_ls[st].append(float('nan'))
                        std_fits_by_ls[st].append(None)

                    # Weighted GP
                    try:
                        if X_tr_w is None:
                            raise RuntimeError(
                                f"no importance-weighted hybrid features for "
                                f"fold={fold_i} budget={bkey}")
                        _svw = (_median_pair_distance(X_tr_w)
                                if st == 'median' else 1.0)
                        gp_w = _make_gp(X_tr_w, st)
                        gp_w.fit(X_tr_w, y_train)
                        _p = gp_w.predict(X_va_w)
                        w_by_ls[st].append(metric_fn(y_val, _p))
                        w_fits_by_ls[st].append(
                            _gp_fit_report(gp_w, _p, st, _svw))
                    except Exception:
                        w_by_ls[st].append(float('nan'))
                        w_fits_by_ls[st].append(None)

            if std_by_ls[ls_reported]:
                for st in ls_starts:
                    results['hybrid_by_budget_by_lengthscale'].setdefault(
                        st, {})[bkey] = _gp_block(
                            std_by_ls[st], w_by_ls[st],
                            std_fits_by_ls[st], w_fits_by_ls[st])
                # A COPY, so _orient_deltas does not reach one dict twice.
                results['hybrid_by_budget'][bkey] = copy.deepcopy(
                    results['hybrid_by_budget_by_lengthscale'][
                        ls_reported][bkey])
                for st in ls_starts:
                    _b = results['hybrid_by_budget_by_lengthscale'][st][bkey]
                    print(f"    hybrid [{bkey} features, {st} start]: "
                          f"std={_b['standard']:.4f}, "
                          f"weighted={_b['weighted']:.4f} "
                          f"(delta={_b['delta']:+.4f}, "
                          f"delta_paired={_b['delta_paired']:+.4f} "
                          f"over {_b['n_paired_folds']} folds)")

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
                      hybrid_budgets=None, per_rep_cache=None):
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
    for rep_name in _per_rep_arms_to_score(reps, per_rep_cache, results,
                                           'weighted_svr'):
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

    _store_per_rep(results, per_rep_cache, 'weighted_svr')

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
                hybrid_budgets=None, per_rep_cache=None):
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
    is_clf = task_type == 'classification'

    k_list = _qrasar_ks()
    # The count the unsuffixed blocks report. Every count is under
    # 'per_rep_by_neighbours' and 'hybrid_by_budget_by_neighbours'.
    k_neighbors = k_list[0]
    results = {'per_rep': {}, 'hybrid': {},
               'per_rep_by_neighbours': {}, 'qrasar_neighbour_counts': list(k_list),
               'qrasar_neighbours_reported': k_neighbors,
               'qrasar_neighbours_basis': (
                   "One arm per neighbour count. 'per_rep' and "
                   "'hybrid_by_budget' hold the count in "
                   "'qrasar_neighbours_reported'; the same numbers and every "
                   "other count are under 'per_rep_by_neighbours' and "
                   "'hybrid_by_budget_by_neighbours', keyed by the count as a "
                   "string.")}

    _rf_fac = get_model_factories(['rf'], task_type=task_type,
                                    n_samples=len(labels))['rf']
    def _make_rf():
        return _rf_fac()

    # Per-rep evaluation
    for rep_name in _per_rep_arms_to_score(reps, per_rep_cache, results,
                                           'qrasar'):
        X_rep = reps[rep_name]
        base_scores = []
        uw_by_k = {k: [] for k in k_list}
        w_by_k = {k: [] for k in k_list}

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Base features only. It reads no neighbour, so it is fitted once
            # for every count.
            try:
                rf = _make_rf()
                rf.fit(X_tr, y_train)
                base_scores.append(metric_fn(y_val, _predict_for_metric(rf, X_va, metric_name, is_clf)))
            except Exception:
                base_scores.append(float('nan'))

            # The weighted feature matrix, built once per fold and reused by
            # every neighbour count.
            try:
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
                X_tr_w, X_va_w = X_tr * w, X_va * w
            except Exception:
                X_tr_w = X_va_w = None

            for k_n in k_list:
                # Base + unweighted sim descriptors
                try:
                    sim_tr = _compute_sim_descriptors(X_tr, y_train, X_tr,
                                                      k=k_n, is_self=True)
                    sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                      k=k_n, is_self=False)
                    X_tr_aug = np.hstack([X_tr, sim_tr])
                    X_va_aug = np.hstack([X_va, sim_va])
                    rf_aug = _make_rf()
                    rf_aug.fit(X_tr_aug, y_train)
                    uw_by_k[k_n].append(metric_fn(y_val, _predict_for_metric(rf_aug, X_va_aug, metric_name, is_clf)))
                except Exception:
                    uw_by_k[k_n].append(float('nan'))

                # Base + importance-weighted sim descriptors
                try:
                    if X_tr_w is None:
                        raise RuntimeError(
                            f"no importance-weighted features for fold={fold_i} "
                            f"rep={rep_name}")
                    sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                        k=k_n, is_self=True)
                    sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                        k=k_n, is_self=False)
                    X_tr_aug_w = np.hstack([X_tr_w, sim_tr_w])
                    X_va_aug_w = np.hstack([X_va_w, sim_va_w])
                    rf_aug_w = _make_rf()
                    rf_aug_w.fit(X_tr_aug_w, y_train)
                    w_by_k[k_n].append(metric_fn(y_val, _predict_for_metric(rf_aug_w, X_va_aug_w, metric_name, is_clf)))
                except Exception:
                    w_by_k[k_n].append(float('nan'))

        for k_n in k_list:
            results['per_rep_by_neighbours'].setdefault(str(k_n), {})[rep_name] = \
                _qrasar_block(base_scores, uw_by_k[k_n], w_by_k[k_n])
        # A COPY, not the same dict: _orient_deltas negates in place and would
        # otherwise reach these numbers twice.
        _rep_block = copy.deepcopy(
            results['per_rep_by_neighbours'][str(k_neighbors)][rep_name])
        results['per_rep'][rep_name] = _rep_block
        print(f"    {rep_name}: base={_rep_block['base_only']:.4f}, "
              f"+sim_uw={_rep_block['base_plus_sim_unweighted']:.4f} "
              f"(d={_rep_block['delta_uw']:+.4f}, "
              f"d_paired={_rep_block['delta_uw_paired']:+.4f} over "
              f"{_rep_block['n_paired_folds_uw']} folds), "
              f"+sim_w={_rep_block['base_plus_sim_weighted']:.4f} "
              f"(d={_rep_block['delta_w']:+.4f}, "
              f"d_paired={_rep_block['delta_w_paired']:+.4f} over "
              f"{_rep_block['n_paired_folds_w']} folds) "
              f"at {k_neighbors} neighbours")

    _store_per_rep(results, per_rep_cache, 'qrasar',
                   extra_keys=('per_rep_by_neighbours',))

    # Hybrid evaluation — one arm per swept number of features.
    if fold_blueprints:
        hybrid_build_errors = []
        budgets = list(hybrid_budgets or [None])
        results['hybrid_by_budget'] = {}

        results['hybrid_by_budget_by_neighbours'] = {}
        for budget in budgets:
            bkey = _budget_key(budget)
            base_scores = []
            uw_by_k = {k: [] for k in k_list}
            w_by_k = {k: [] for k in k_list}

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
                        # the nine swept in run(). None leaves the number this
                        # fold's own recipe was searched at. The neighbour
                        # count below does not reach the build.
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
                    base_scores.append(metric_fn(y_val, _predict_for_metric(rf, X_va, metric_name, is_clf)))
                except Exception:
                    base_scores.append(float('nan'))

                # The importance-weighted hybrid matrix, built once per fold.
                try:
                    w = _compute_importance_for_hybrid(feat_info)
                    assert w.shape[0] == X_tr.shape[1], (
                        f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                    X_tr_w, X_va_w = X_tr * w, X_va * w
                except Exception:
                    X_tr_w = X_va_w = None

                for k_n in k_list:
                    # Unweighted sim descriptors
                    try:
                        sim_tr = _compute_sim_descriptors(X_tr, y_train, X_tr,
                                                          k=k_n, is_self=True)
                        sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                          k=k_n, is_self=False)
                        X_tr_aug = np.hstack([X_tr, sim_tr])
                        X_va_aug = np.hstack([X_va, sim_va])
                        rf_aug = _make_rf()
                        rf_aug.fit(X_tr_aug, y_train)
                        uw_by_k[k_n].append(metric_fn(y_val, _predict_for_metric(rf_aug, X_va_aug, metric_name, is_clf)))
                    except Exception:
                        uw_by_k[k_n].append(float('nan'))

                    # Weighted sim descriptors
                    try:
                        if X_tr_w is None:
                            raise RuntimeError(
                                f"no importance-weighted hybrid features for "
                                f"fold={fold_i} budget={bkey}")
                        sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                            k=k_n, is_self=True)
                        sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                            k=k_n, is_self=False)
                        X_tr_aug_w = np.hstack([X_tr_w, sim_tr_w])
                        X_va_aug_w = np.hstack([X_va_w, sim_va_w])
                        rf_aug_w = _make_rf()
                        rf_aug_w.fit(X_tr_aug_w, y_train)
                        w_by_k[k_n].append(metric_fn(y_val, _predict_for_metric(rf_aug_w, X_va_aug_w, metric_name, is_clf)))
                    except Exception:
                        w_by_k[k_n].append(float('nan'))

            if base_scores:
                for k_n in k_list:
                    results['hybrid_by_budget_by_neighbours'].setdefault(
                        str(k_n), {})[bkey] = _qrasar_block(
                            base_scores, uw_by_k[k_n], w_by_k[k_n])
                # A COPY, so _orient_deltas does not reach one dict twice.
                _blk = copy.deepcopy(
                    results['hybrid_by_budget_by_neighbours'][
                        str(k_neighbors)][bkey])
                results['hybrid_by_budget'][bkey] = _blk
                print(f"    hybrid [{bkey} features] at {k_neighbors} "
                      f"neighbours: base={_blk['base_only']:.4f}, "
                      f"+sim_uw={_blk['base_plus_sim_unweighted']:.4f} "
                      f"(d={_blk['delta_uw']:+.4f}, "
                      f"d_paired={_blk['delta_uw_paired']:+.4f} over "
                      f"{_blk['n_paired_folds_uw']} folds), "
                      f"+sim_w={_blk['base_plus_sim_weighted']:.4f} "
                      f"(d={_blk['delta_w']:+.4f}, "
                      f"d_paired={_blk['delta_w_paired']:+.4f} over "
                      f"{_blk['n_paired_folds_w']} folds)")

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
    # {space: {fold: the pool's rank-1 entry}} for the k-nearest-neighbours read
    # below, kept so the per-task resolution further down reuses it instead of
    # reading the pool for 'knn' a second time.
    fold_entries_by_space = {}
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
            fold_entries_by_space.setdefault(_space, {})[fold_i] = _ftop
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

    # ── One recipe per (hybrid space, task, held-out fold) ───────────────────
    # Every one of the four tasks was built from the recipe above, which the pool
    # ranked on the stored rows of 'knn'. Task 1 fits a k-nearest-neighbours
    # model; Task 2 fits a Gaussian process, Task 3 a support vector regressor
    # and Task 4 a random forest on four neighbour-derived features. Those three
    # were scored on features chosen for a k-nearest-neighbours fit, under their
    # own names.
    #
    # Each task now gets the recipe ranked on ITS OWN model's stored rows, inside
    # each family, on the other folds exactly as the loop above ranks the
    # k-nearest-neighbours one. Task 1's recipe is the very entry that loop
    # already chose, so every number Task 1 reports is unchanged.
    #
    # This costs no model fit and no extra hybrid build: each task already called
    # build_fold_hybrid once per (space, width, fold) of its own, so only WHICH
    # recipe it builds changes. What it costs is one more read of stored rows per
    # (model, fold), which fits nothing.
    _task_models = sorted(set(_MODEL_FOR_TASK.values()))
    # {model: {space: {fold: the pool's rank-1 entry}}}
    _by_model_space = {'knn': fold_entries_by_space}
    task_recipe_errors = {}
    for _model in _task_models:
        if _model in _by_model_space:
            continue
        for fold_i in range(len(fold_splits)):
            try:
                _pm = get_fold_pool_recipes(
                    reps, labels, groups,
                    fold_i=fold_i, n_outer_folds=len(fold_splits),
                    downstream_model=_model, budget=requested_budget,
                    dataset_name=dataset_name,
                    augmentations=data.get('augmentations'),
                    distance=True, data=data,
                    metric=metric_name, task_type=task_type, domain=domain,
                    pool_subset=_POOL_SUBSET,
                    n_hybrids=_N_POOL_HYBRIDS)
            except Exception as e:
                (task_recipe_errors.setdefault(f'model:{_model}', {})
                 )[f'fold{fold_i}'] = f'{type(e).__name__}: {e}'
                continue
            for _fam, _blk in (_pm.get('by_objective') or {}).items():
                _fh = (_blk or {}).get('hybrids') or []
                if not _fh:
                    continue
                _sp = _SPACE_FOR_FAMILY.get(_fam, f'hybrid_{_fam}')
                (_by_model_space.setdefault(_model, {})
                 .setdefault(_sp, {}))[fold_i] = _fh[0]

    # One stored search per family, read once, for a model the search never
    # scored. top_hybrids_across_models needs the SearchResult itself, which
    # get_fold_pool_recipes does not hand back.
    #
    # ⛔ WHICH POOL MEMBER THIS IS. get_best_search_for_downstream picks one
    # (tier, augmentation condition) member per model, and the model named here is
    # 'knn', the same as the whole-dataset read above. So the configurations
    # ranked across models below come from the member knn's score preferred. The
    # member is recorded per family in
    # results['cross_model_family_member'].
    family_search = {}
    family_member = {}
    for _fam in _POOL_FAMILIES:
        try:
            _sf, _mf = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='knn', budget=requested_budget,
                augmentations=data.get('augmentations'),
                metric=metric_name, task_type=task_type, domain=domain,
                dataset_name=dataset_name, data=data,
                distance=True, objective=_fam)
        except Exception as e:
            family_member[_fam] = {'absent_reason': f'{type(e).__name__}: {e}'}
            continue
        if _sf is None or _mf is None:
            family_member[_fam] = {
                'absent_reason': 'the pool holds no search of this family'}
            continue
        family_search[_fam] = _sf
        family_member[_fam] = {
            'tier': _mf['tier'],
            'aug_condition': _mf['aug_condition'],
            'metric_name': _mf['metric_name'],
            'member_chosen_for_model': 'knn',
            'models_the_search_scored': sorted(
                (getattr(_sf, 'best_by_model', None) or {}).keys()),
        }

    # {space: {task: {fold: blueprint}}} and the recipe beside it.
    fold_bps_by_space_task = {}
    fold_recipes_by_space_task = {}
    # {space: {task: 'own_rows' | 'across_models' | 'unavailable'}}
    task_recipe_source = {}
    # {space: {task: {fold: the model whose score chose that fold's recipe}}}
    task_recipe_chosen_by = {}
    for _task_name, _model in sorted(_MODEL_FOR_TASK.items()):
        for _fam in _POOL_FAMILIES:
            _sp = _SPACE_FOR_FAMILY.get(_fam, f'hybrid_{_fam}')
            _own = ((_by_model_space.get(_model) or {}).get(_sp)) or {}
            if _own:
                for _f, _e in _own.items():
                    (fold_bps_by_space_task.setdefault(_sp, {})
                     .setdefault(_task_name, {}))[_f] = _e['blueprint']
                    (fold_recipes_by_space_task.setdefault(_sp, {})
                     .setdefault(_task_name, {}))[_f] = {
                        'objective': _e['objective'],
                        'tier': _e['tier'],
                        'aug_condition': _e['aug_condition'],
                        'allocation_method': _e['allocation_method'],
                        'importance_method': _e['importance_method'],
                        'internal_evaluator': _e['internal_evaluator'],
                        'budget': _e['budget'],
                        'metric_name': _e['metric_name'],
                        'lower_is_better': _e['lower_is_better'],
                        'score_other_folds': _e['score_other_folds'],
                        'n_other_folds': _e['n_other_folds'],
                        'ranked_on_rows_of_model': _model,
                    }
                task_recipe_source.setdefault(_sp, {})[_task_name] = 'own_rows'
                print(f"  {_sp}/{_task_name}: recipe ranked on {_model}'s own "
                      f"stored rows, per fold, on the other folds")
                continue
            # No stored row for this model in this family. Rank every model's
            # rows together instead, still leaving the held-out fold out.
            _sf = family_search.get(_fam)
            if _sf is None:
                task_recipe_source.setdefault(_sp, {})[_task_name] = 'unavailable'
                (task_recipe_errors.setdefault(_sp, {}))[_task_name] = (
                    f'no stored row for {_model} in the {_fam} family, and that '
                    f'family\'s search could not be read either, so nothing '
                    f'could be ranked over the other models: '
                    f'{family_member.get(_fam, {}).get("absent_reason")}')
                print(f"  {_sp}/{_task_name}: no hybrid space here -- "
                      f"{task_recipe_errors[_sp][_task_name]}")
                continue
            _n_filled = 0
            for fold_i in range(len(fold_splits)):
                try:
                    _xm = top_hybrids_across_models(
                        _sf, n=_N_POOL_HYBRIDS, rule='other_folds',
                        held_out_fold=fold_i)
                except Exception as e:
                    (task_recipe_errors.setdefault(_sp, {})
                     )[f'{_task_name}/fold{fold_i}'] = \
                        f'{type(e).__name__}: {e}'
                    continue
                _cand = next((_e for _e in (_xm.get('selected') or [])
                              if _e.get('blueprint') is not None), None)
                if _cand is None:
                    (task_recipe_errors.setdefault(_sp, {})
                     )[f'{_task_name}/fold{fold_i}'] = (
                        _xm.get('shortfall_reason')
                        or 'no configuration over any model carries a '
                           'blueprint on the other folds')
                    continue
                _bpx = copy.deepcopy(_cand['blueprint'])
                # Trust the condition the search was loaded under over the
                # blueprint's own field, the same way get_fold_pool_recipes does.
                _bpx.augmentation_strategy = family_member[_fam].get(
                    'aug_condition')
                (fold_bps_by_space_task.setdefault(_sp, {})
                 .setdefault(_task_name, {}))[fold_i] = _bpx
                (fold_recipes_by_space_task.setdefault(_sp, {})
                 .setdefault(_task_name, {}))[fold_i] = {
                    'objective': _fam,
                    'tier': family_member[_fam].get('tier'),
                    'aug_condition': family_member[_fam].get('aug_condition'),
                    'allocation_method': _cand.get('allocation_method'),
                    'importance_method': _cand.get('importance_method'),
                    'internal_evaluator': _cand.get('internal_evaluator'),
                    'budget': _cand.get('budget'),
                    'metric_name': getattr(_sf, 'metric_name', None),
                    'lower_is_better': bool(getattr(_sf, 'lower_is_better',
                                                    False)),
                    'score_other_folds': _cand.get('score_mean'),
                    'n_other_folds': _cand.get('n_folds_used'),
                    'ranked_on_rows_of_model': 'every model the search scored',
                    'recipe_chosen_by': _cand.get('recipe_chosen_by'),
                    'models_ranked_over': list(
                        _xm.get('models_ranked_over') or []),
                    'score_was_produced_by': (
                        f"{_cand.get('recipe_chosen_by')}, not by {_model}"),
                }
                (task_recipe_chosen_by.setdefault(_sp, {})
                 .setdefault(_task_name, {}))[fold_i] = \
                    _cand.get('recipe_chosen_by')
                _n_filled += 1
            if _n_filled:
                task_recipe_source.setdefault(_sp, {})[_task_name] = \
                    'across_models'
                _who = sorted(set(
                    (task_recipe_chosen_by.get(_sp, {}).get(_task_name) or {})
                    .values()))
                print(f"  {_sp}/{_task_name}: no stored row for {_model}, so "
                      f"the recipe is the best configuration over every model "
                      f"the search scored, on {_n_filled} of "
                      f"{len(fold_splits)} folds; the scores that chose them "
                      f"were produced by {', '.join(_who)}")
            else:
                task_recipe_source.setdefault(_sp, {})[_task_name] = \
                    'unavailable'

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
                      if (fold_blueprints_by_space.get(sp)
                          or fold_bps_by_space_task.get(sp))]
    results['hybrid_spaces_scored'] = list(_spaces_to_run)
    results['hybrid_space_recipes'] = convert_numpy(fold_recipes_by_space)
    results['hybrid_space_errors'] = pool_family_errors
    # ── The recipe each TASK is built from, per hybrid space ─────────────────
    # 'hybrid_space_recipes' above is the k-nearest-neighbours recipe, which is
    # what all four tasks were built from until 2026-09-28. These four keys say
    # what each task is built from now, one recipe per (space, task, fold).
    results['task_downstream_model'] = dict(_MODEL_FOR_TASK)
    results['task_recipes'] = convert_numpy(fold_recipes_by_space_task)
    results['task_recipe_source'] = task_recipe_source
    results['task_recipe_chosen_by'] = task_recipe_chosen_by
    results['task_recipe_errors'] = task_recipe_errors
    results['cross_model_family_member'] = convert_numpy(family_member)
    results['task_recipe_note'] = (
        "'task_downstream_model' names the model each task fits. Its recipe is "
        "ranked on that model's own stored rows where the search holds any, "
        "which 'task_recipe_source' reports as own_rows. A Gaussian process is "
        'not a model any search in this lineage scored, so its recipe is the '
        'best configuration over every model the search did score, reported as '
        "across_models, and 'task_recipe_chosen_by' names the model whose score "
        'put that configuration on top for each fold. That score was produced by '
        "that model and not by the one it is handed to. Task 1's recipe is the "
        'same entry as in hybrid_space_recipes, so its numbers are unchanged.')
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

    # One block per task, filled by the first hybrid space and copied into
    # every later one. The per-representation arms of all four tasks read no
    # recipe, so the second space was fitting the same models again for the
    # same numbers; the Gaussian process alone is 7 representations x 5 folds
    # x 2 arms per space. The hybrid arms are what differ between spaces and
    # they are not cached.
    _per_rep_cache = {}

    # Each (hybrid space, task) block is written to
    # s10_importance_downstream/<dataset>.partial.jsonl when it returns and
    # read back on the next attempt (2026-09-25; _checkpoint.py). The stored
    # unit also carries what that task added to _per_rep_cache, so a later
    # space still reuses the first space's per-representation arms and still
    # records per_rep_reused_from_first_space=True, as an uninterrupted run
    # does. It is stored before _orient_deltas runs further down, so the
    # orientation is applied exactly once either way. Every model here is
    # seeded and the folds and fold importances are recomputed identically, so
    # a block read back holds what a fresh computation gives. A block in which
    # any fold's hybrid failed to build (n_hybrid_build_failures > 0 anywhere
    # in it) is not stored, so it is retried.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's10_importance_downstream',
                           dataset_name),
        run_identity('s10_importance_downstream', dataset_name, data, config,
                     kwargs=kwargs, sweep_budgets=list(sweep_budgets),
                     spaces=list(_spaces_to_run),
                     recipes=results['hybrid_space_recipes'],
                     # Each task's own recipe is part of what a stored block
                     # holds, so a block written before 2026-09-28 -- when all
                     # four tasks were built from the k-nearest-neighbours one --
                     # must not be read back into this run.
                     task_recipes=results['task_recipes']))

    def _had_build_failure(obj):
        if isinstance(obj, dict):
            if (obj.get('n_hybrid_build_failures') or 0) > 0:
                return True
            return any(_had_build_failure(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(_had_build_failure(v) for v in obj)
        return False

    def _task(space, task, compute):
        _key = dict(space=space, task=task)
        if _ck.done(**_key):
            _saved = _ck.get(**_key)
            _per_rep_cache.update(_saved['cache'])
            return _saved['out']
        _before = set(_per_rep_cache)
        out = compute()
        if not _had_build_failure(out):
            _ck.put({'out': out,
                     'cache': {k: v for k, v in _per_rep_cache.items()
                               if k not in _before}}, **_key)
        return out

    for _space in _spaces_to_run:
        # One recipe per task now, each ranked on the rows of the model that task
        # fits. `_bps_for(task)` is empty when neither that model's own rows nor
        # the ranking over every model produced one for this space, and the task's
        # hybrid space is then absent rather than built from another model's recipe;
        # results['task_recipe_errors'] says why.
        _by_task = fold_bps_by_space_task.get(_space) or {}

        def _bps_for(task, _by_task=_by_task):
            return _by_task.get(task) or {}

        _space_out = {}
        print(f"\n  ===== hybrid space {_space} =====")

        # Task 1: Weighted kNN
        t0 = time.time()
        _space_out['weighted_knn'] = _task(_space, 'weighted_knn', lambda: _run_weighted_knn(
            reps, labels, groups, config, _bps_for('weighted_knn'),
            metric_fn, metric_name, lower_is_better,
            augmentations=augmentations, task_type=task_type,
            fold_splits=fold_splits, fold_importance=fold_importance,
            hybrid_budgets=sweep_budgets, per_rep_cache=_per_rep_cache))
        print(f"  [Task 1 elapsed: {time.time() - t0:.1f}s]")

        # Task 2: GP with importance-weighted RBF (regression only)
        if is_clf:
            print("\n  --- Task 2: GP --- SKIP (regression-only)")
            _space_out['weighted_gp'] = {'skipped': True,
                                         'reason': 'classification'}
        else:
            t0 = time.time()
            _space_out['weighted_gp'] = _task(_space, 'weighted_gp', lambda: _run_weighted_gp(
                reps, labels, groups, config, _bps_for('weighted_gp'),
                metric_fn, metric_name, n_samples,
                lower_is_better=lower_is_better,
                augmentations=augmentations,
                fold_splits=fold_splits, fold_importance=fold_importance,
                hybrid_budgets=sweep_budgets, per_rep_cache=_per_rep_cache))
            print(f"  [Task 2 elapsed: {time.time() - t0:.1f}s]")

        # Task 3: SVR with importance-weighted RBF (regression only)
        if is_clf:
            print("  --- Task 3: SVR --- SKIP (regression-only)")
            _space_out['weighted_svr'] = {'skipped': True,
                                          'reason': 'classification'}
        else:
            t0 = time.time()
            _space_out['weighted_svr'] = _task(_space, 'weighted_svr', lambda: _run_weighted_svr(
                reps, labels, groups, config, _bps_for('weighted_svr'),
                metric_fn, metric_name, n_samples,
                lower_is_better=lower_is_better,
                augmentations=augmentations,
                fold_splits=fold_splits, fold_importance=fold_importance,
                hybrid_budgets=sweep_budgets, per_rep_cache=_per_rep_cache))
            print(f"  [Task 3 elapsed: {time.time() - t0:.1f}s]")

        # Task 4: q-RASAR similarity-augmented features
        t0 = time.time()
        _space_out['qrasar'] = _task(_space, 'qrasar', lambda: _run_qrasar(
            reps, labels, groups, config, _bps_for('qrasar'),
            metric_fn, metric_name, task_type=task_type,
            lower_is_better=lower_is_better,
            augmentations=augmentations,
            fold_splits=fold_splits, fold_importance=fold_importance,
            hybrid_budgets=sweep_budgets, per_rep_cache=_per_rep_cache))
        print(f"  [Task 4 elapsed: {time.time() - t0:.1f}s]")

        _space_out['built_under_objective'] = (
            'predictive' if _space == _primary_space else 'similarity')
        _space_out['features_chosen_by'] = describe_pool_family(
            _space_out['built_under_objective'])
        # Repeated inside the space so a reader of one block can see which model's
        # rows chose each task's recipe without opening the top level.
        _space_out['task_recipe_source'] = task_recipe_source.get(_space, {})
        _space_out['task_recipe_chosen_by'] = task_recipe_chosen_by.get(
            _space, {})
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
        # WHICH MODEL THIS ONE SEARCH WAS FETCHED FOR, recorded 2026-09-28.
        # get_best_search_for_downstream above was called with
        # downstream_model='knn', and all six downstream models' rows are read
        # out of that one SearchResult. That is defensible for these rankings --
        # search.baselines and search.all_results carry a score under every
        # model the search scored -- but the saved result did not say which model
        # the search was fetched for, so a reader could not tell. Written under
        # the same key name n18, n21, n22 and n25 use.
        results['top2_pairs_search_of_model'] = 'knn'
        # WHICH DOWNSTREAM MODELS THE ROWS BELOW COVER, resolved 2026-09-28.
        # DISTANCE_MODELS is a fixed list of six names. The search on disk scored
        # whatever it scored: svm drops itself above 10,000 molecules, and a
        # dataset's skip policy can drop a name. The two disagree whenever that
        # happened, and the loops below then asked for a model with no stored
        # rows.
        #
        # The six names stay what this file ASKS for, so the set of models it
        # reports does not change: DISTANCE_MODELS goes in as `requested`, not
        # None. A name the search never scored is dropped and named, never
        # swapped for another model's rows.
        _cmp_models, _cmp_dropped = resolve_comparison_models(
            search, DISTANCE_MODELS)
        results['top2_models_compared'] = list(_cmp_models)
        # {downstream model: why it is not in the rows below}. These reasons were
        # printed to the terminal and saved nowhere until now.
        results['top2_models_dropped'] = {_n: _w for _n, _w in _cmp_dropped}
        for _n, _w in _cmp_dropped:
            print(f"  downstream model {_n} not compared: {_w}")
        _top2_base, _top2_hyb, _hyb_deltas = {}, {}, {}
        for _m in _cmp_models:
            try:
                _top2_base[_m] = top_base_reps(search, _m, 2,
                                               require_full=False)
            except Exception as _e:
                print(f"  top-2 base representations for {_m} unavailable: {_e}")
                results['top2_models_dropped'][_m] = (
                    'no top-2 base representations: %s: %s'
                    % (type(_e).__name__, _e))
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
                results['top2_models_dropped'][_m] = (
                    'no top-2 hybrid configurations: %s: %s'
                    % (type(_e).__name__, _e))
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
                    results['top2_models_dropped'][_m] = (
                        'no hybrid deltas: %s: %s' % (type(_e).__name__, _e))
        _top2_pairs = {}
        try:
            # The same 2-and-2 grid in one block: per downstream model the 2
            # base representations and the 2 hybrids, plus 'n_distinct_units' --
            # how many separate things would have to be BUILT once one
            # representation picked by several downstream models counts once.
            _top2_pairs = _drop_blueprint(
                select_model_rep_pairs(search, models=list(_cmp_models),
                                       n_base=2, n_hybrids=2,
                                       require_full=False))
        except Exception as _e:
            print(f"  top-2 model-and-representation grid unavailable: {_e}")
            results['top2_pairs_by_model_absent_reason'] = \
                f"{type(_e).__name__}: {_e}"

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
            for _m in _cmp_models:
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

    # The four settings the arms in this file are built with, stored so a
    # result file says which arm produced its numbers without anyone reading
    # the environment the run had.
    from kirby.similarity.preprocessing import importance_exponent_override
    _exp = importance_exponent_override()
    results['weighting_settings'] = {
        'importance_exponent': 0.5 if _exp is None else _exp,
        'weight_convention': _weight_convention(),
        'knn_vote': _knn_vote_weights(),
        'k_values': _k_values(),
        'qrasar_neighbour_counts': _qrasar_ks(),
        'gp_restarts': int(os.environ.get('KIRBY_S10_GP_RESTARTS', '').strip() or '0'),
        'gp_kernel': _gp_kernel_in_use(),
        'gp_lengthscale_starts': _gp_lengthscale_starts(),
        'gp_normalize_y': _gp_normalize_y(),
        'gp_kernel_largest_prediction_difference': _gp_kernel_equivalence(),
        'basis': (
            'Every feature is z-scored, then multiplied by its importance '
            'raised to importance_exponent and rescaled by weight_convention. '
            "mean1 divides by the mean weight and keeps a z-scored feature at "
            'its own scale; sum1 divides by the sum, which divides every '
            'distance by the number of features. knn_vote is sklearn\'s '
            "weights argument: uniform averages the k neighbours' labels, "
            'distance averages them weighted by one over the distance. '
            'qrasar_neighbour_counts are the neighbour counts the four '
            'q-RASAR features are built from, one arm each; the first is the '
            "one that task's unsuffixed blocks report."),
    }

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
