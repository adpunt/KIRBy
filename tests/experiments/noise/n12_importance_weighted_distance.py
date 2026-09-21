#!/usr/bin/env python3
"""N12: Feature Importance-Weighted Distance for Noise Estimation.

Paper: Noise
SCOPE: MOLECULAR datasets only, SYNTHETIC noise only (additive Gaussian for
regression / label flips for classification). This is a NARROW ablation, not a
general claim: does scaling features by sqrt(importance) before computing kNN
distances improve neighbor-based noise estimation, and is it redundant with
hybridization?

CAVEAT — READ BEFORE INTERPRETING RESULTS: the measured differences between
conditions are small and may fall entirely within run-to-run variance (fold
resampling + a single noise seed per fold). Where the summary aggregates across
folds it also reports the spread (std); treat any condition ordering as
suggestive only unless the gap clearly exceeds that spread. Do not read the
boolean "helps"/"redundant" flags as significance tests.

5 conditions × 3 estimators × 3 k values × 2 importance methods × 2 sigmas, per
hybrid space. With both search families in the pool it is 8 conditions: the two
base ones, plus three in each hybrid space.

2026-09-12, user decision: the hybrid conditions may draw their recipe from the
searches run under objective='similarity' as well as the prediction-built ones —
`pool_subset='predictive_and_similarity'`. The result file records which family
the winning recipe came from under 'hybrid_family', and every eligible family's
own best under 'hybrid_pool_by_objective'.

2026-09-19: both families are now BUILT, not only recorded. POOL_SUBSETS names
the families in PREFERENCE order and the pool returns the first that has an
entry; a predictive recipe always exists, so between those two dates this file
asked for both families, built the predictive one and wrote the geometry one
into the result file as a score nothing had measured in. That matters here more
than almost anywhere, because this file's whole question is which DISTANCE
finds a molecule's neighbours, and the similarity search chooses features so
that exactly that distance predicts activity from the five nearest neighbours.
The conditions named 'hybrid_*' are unchanged and still the predictive recipe;
the ones named 'hybrid_similarity_*' are the new ones.

Usage:
    python n12_importance_weighted_distance.py --dataset logd --mode intermediary
    python n12_importance_weighted_distance.py --dataset caco2 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.dataset_policy import (
    noise_groups_for, noise_levels as dataset_noise_levels,
)
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores,
)
from experiments._shared import (
    pick_noise_base_rep,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_precomputed_search, get_best_search_for_downstream, build_fold_hybrid,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See SPACE_FOR_FAMILY below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from kirby.hybrid import compute_feature_importance
from kirby.similarity.distance_config import (
    get_distance_metric as _get_distance_metric,
)
from kirby.find_best_hybrid import _is_geometry_objective
from kirby.utils.noise_estimation import (
    NeighborResidualEstimator,
    LocalVarianceRatioEstimator,
    weights_from_scores,
)


# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
SIGMAS = None

# The noise condition. Gaussian is what this experiment has always injected; the
# other ten are reachable by passing noise_condition=..., one run each.
NOISE_CONDITION = 'gaussian'
K_VALUES = [5, 10, 20]

# ⛔ THE REPAIR WAS TOO WEAK TO EXPRESS ANY DIFFERENCE THE DETECTOR MAKES.
# strength 0.1 under the 'exponential' shape is a row weight of exp(-0.1) at the
# most suspect end against 1 at the cleanest -- about 1.11 to 1, i.e. within a
# tenth of not weighting at all. This experiment exists to ask whether better
# DETECTION leads to better REPAIR, and it was reading that through a repair
# step that barely moved the fit, so two detectors that rank very differently
# would still produce nearly the same recovery number.
#
# 3.0 measured, not guessed: n18 swept the same knob over 2:1, 4:1 and 10:1 on
# the same molecules and folds and the best setting was 3.0 (a 4-to-1 ratio) on
# BOTH datasets and both metric directions -- 5 of 5 folds at +0.0069 rmse on
# herg_ki and 5 of 5 at +0.0447 spearman on flab_thermo_tresanco. 'inverse' is
# the shape n18 measured it under, so the number means the same thing here.
REPAIR_STRENGTH = 3.0
REPAIR_WEIGHT_METHOD = 'inverse'

# THE SMALLEST CLEAN-TO-NOISY GAP A RECOVERY PERCENTAGE MAY BE DIVIDED BY,
# as a share of the clean score. Same value and same meaning as
# n06_bindingdb_ground_truth, n07_remediation.GAP_MIN_FRAC,
# n11_estimation_remediation_grid.GAP_MIN_FRAC and
# n21_neighbour_distance.GAP_MIN_FRAC. n12 was the one file in this directory
# without it; see the recovery block in run() for what that cost.
GAP_MIN_FRAC = 0.02
IMPORTANCE_METHODS = ['lightgbm_gain', 'treeshap']

# activity_cliff_aware CUT 2026-09-12 (user decision, measured): it ranks the
# molecules the same way as neighbor_residual at Spearman 0.989 on herg_ki in
# ecfp4, because it IS the neighbour residual multiplied by how much the
# neighbours agree. n11 and n06 dropped it in the same change.
ESTIMATOR_FACTORIES = {
    'neighbor_residual': NeighborResidualEstimator,
    'local_variance_ratio': LocalVarianceRatioEstimator,
}

# Which searches the hybrid recipes may come from, and which result-file space
# each family is built into.
#
# ⛔ ONE SPACE PER FAMILY, NOT ONE RECIPE. Added 2026-09-19.
#
# POOL_SUBSETS names the two families in PREFERENCE order and the pool returns
# the first that has an entry, because their stored scores are in different
# units -- a predictive search's score is the dataset's own metric, a
# similarity search's is the correlation between each held-out molecule's
# activity and what its five nearest neighbours predict. A predictive recipe
# always exists, so from 2026-09-12 to 2026-09-19 this file asked for both
# families, built the predictive one and wrote the other into the result file
# as a score it never measured in.
#
# That matters here more than almost anywhere. This file's entire question is
# which DISTANCE finds a molecule's neighbours. The similarity search chooses
# features so that exactly that distance predicts activity from the five
# nearest neighbours; it is the space built for the question this file asks.
#
# 'hybrid' is still the predictive recipe, so every number reported before
# 2026-09-19 still means what it meant.
POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# The three conditions each hybrid space carries, named after that space. With
# both families in the pool there are eight conditions, not five.
HYBRID_CONDITION_SUFFIXES = ('_unweighted', '_imp_base', '_imp_own')

CONDITION_NAMES = [
    'base_unweighted',
    'base_imp_weighted',
] + [f'{sp}{suf}'
     for sp in ('hybrid', 'hybrid_similarity')
     for suf in HYBRID_CONDITION_SUFFIXES]


def zero_importance_twin(condition_name):
    """Which unweighted condition a weighted one collapses onto.

    When every importance score is zero _weight_features hands the unweighted
    matrix back, so the two spaces are the same geometry and their estimator
    scores are the same numbers. Recorded per (importance method, condition) in
    results['zero_importance_spaces'].

    A dict until 2026-09-19, when a second hybrid space arrived and each of its
    conditions needed the same mapping to ITS OWN unweighted twin -- pointing
    them at the predictive space's twin would have named a duplicate of the
    wrong matrix.
    """
    if condition_name == 'base_imp_weighted':
        return 'base_unweighted'
    for suf in ('_imp_base', '_imp_own'):
        if condition_name.endswith(suf):
            return condition_name[:-len(suf)] + '_unweighted'
    return None


# Kept for callers that read it. Every entry is what zero_importance_twin says.
ZERO_IMPORTANCE_TWIN = {c: zero_importance_twin(c) for c in CONDITION_NAMES
                        if zero_importance_twin(c) is not None}


# The private `_inject_noise` this module used to carry has been deleted. It added
# Gaussian noise at `sigma_frac * (max(y) - min(y))` -- a fraction of the label RANGE,
# which is roughly 4-6x the label SD, so a level of 0.3 here delivered far more noise
# than a level of 0.3 anywhere else in the repo. Injection now goes through
# `noise_helpers.inject_train_labels`, which is the same definition the hybrid search,
# c10 and the Rust half of the study all use, and which records what was delivered.


def _importance_is_all_zero(importance):
    """True when there is no importance to weight by.

    _weight_features hands back X_standardized untouched in that case, so the
    weighted space IS its unweighted twin: the same geometry, the same
    neighbours, the same estimator scores. Nothing in the result file said it
    had happened, so a duplicate read as a measurement of the weighting. The
    caller tests this and records it under 'zero_importance_spaces'.
    """
    return float(np.sqrt(np.maximum(importance, 0)).sum()) < 1e-10


# THE POWER THE IMPORTANCE IS RAISED TO, AND THE DISTANCE IT HAS TO MATCH.
#
# Weighting multiplies a z-scored feature by its importance raised to some
# power e. Under euclidean the per-feature term of the squared distance is
# (w ** e * delta) ** 2, which is w ** (2e) * delta ** 2, so e = 0.5 makes the
# weight a feature carries in the distance equal its own importance. Under
# manhattan the term is w ** e * |delta|, so e = 1.0 does. Under Minkowski with
# exponent p the matched value is 1 / p, which is what
# kirby.similarity.preprocessing.weight_exponent already returns. Cosine has no
# matched value: rescaling a feature does not enter that formula this way.
#
# This file has raised the importance to 0.5 since it was written, and the
# estimators have found their neighbours under manhattan ever since
# kirby.similarity.distance_config made manhattan the default. The two do not
# match.
#
# n21 measured the pair on flab_thermo_tresanco, 672 antibodies, 336 of them in
# each training fold, over 2 folds and 1 corruption draw. The reading is
# detection average precision against the molecules whose labels actually
# moved. Under manhattan, lightgbm_gain read 0.2098 at exponent 0.5 and 0.2388
# at exponent 1.0, and treeshap read 0.2185 and 0.2210. Under euclidean,
# lightgbm_gain read 0.2346 at 0.5 and 0.2175 at 1.0, and treeshap read 0.2030
# and 0.2103. The matched exponent won 3 of those 4 scorer-and-distance pairs,
# on 4 paired readings behind each. That is a direction, not a settled result.
#
# ⛔ NOT YET MEASURED IN THIS FILE. IMPORTANCE_EXPONENT stays at 0.5, which is
# exactly the np.sqrt this replaced, so every number n12 has ever produced is
# unchanged. A sweep sets the module attribute to compare the two arms; the
# default moves only once n12's own numbers say it should.
IMPORTANCE_EXPONENT = 0.5
EXPONENT_MATCHED_TO_METRIC = {'euclidean': 0.5, 'manhattan': 1.0}


def _weight_features(X_standardized, importance):
    """Raise the importance to IMPORTANCE_EXPONENT and multiply z-scored features by it.

    CRITICAL: caller must pass a standardized X. The previous version of this
    helper applied the weight first and the caller standardized after, which
    is a mathematical no-op (per-column StandardScaler exactly cancels a
    per-column scalar multiply) — the entire importance-weighting ablation
    silently produced identical Euclidean geometry across weighted and
    unweighted conditions.

    Returns X * w_normalized, where w_normalized is the importance raised to
    IMPORTANCE_EXPONENT and then scaled so its entries are positive and sum to
    the number of features (mean ≈ 1) — chosen so the weighting expands or
    contracts a feature relative to the unit-variance z-scored input rather
    than crushing every one of them toward zero.

    ⛔ THE POWER IS TAKEN BEFORE THE NORMALISATION, NOT AFTER.
    kirby.similarity.preprocessing.fold_weights does it the other way round: it
    normalises the importance to mean 1 and then raises that to the exponent.
    The two orders give different numbers at any exponent other than 1, and
    which one is right is a question about what "the importance of a feature"
    means, not a bug to fix in passing. This order is the one every n12 result
    on disk was produced under, so it is the one kept here.
    """
    if _importance_is_all_zero(importance):
        # All-zero importance → no signal to weight by; return X_standardized
        # unchanged so downstream distances are well-defined.
        return X_standardized
    w = np.power(np.maximum(importance, 0), IMPORTANCE_EXPONENT)
    # Normalize to mean ≈ 1 (not sum ≈ 1) so unit-variance columns are
    # rescaled around their original magnitude rather than crushed to ~1/d.
    w = w * (len(w) / w.sum())
    return X_standardized * w


def _build_fold_shared(train_reps, y_train, augmentations, bp, base_rep_name,
                       groups=None, val_reps=None, augmentations_val=None,
                       model=None, metric_fn=None, lower_is_better=None):
    """Build the pieces that don't depend on imp_method, ONCE per fold.

    val_reps / augmentations_val: this fold's held-out rows. build_fold_hybrid
        returns their features too; only the training geometry is used here, so
        they are discarded. When val_reps is None the training rows stand in
        for them and the discarded half is meaningless.
    model / metric_fn / lower_is_better: the random forest every condition is
        finally scored with, the dataset's metric, and its direction. They are
        forwarded so the accept-or-reject step inside the build fits that
        forest and calls that metric. Without them create_hybrid falls back to
        its internal evaluator scoring r2, which is a different learner
        judging a different quantity. Under a recipe from the similarity
        family no downstream model is fitted at all and the three are dropped,
        because build_fold_hybrid refuses the pair.

    Returns dict with:
      'X_base_std': standardized base rep
      'X_hyb_std' or None: standardized hybrid (or None if hybrid build failed)
      'imp_hybrid_via_blueprint' or None: importance of hybrid features under
            the blueprint's own importance method (used for hybrid_imp_base)
      'hybrid_objective': the objective the recipe was searched under, which
            is also the one the rebuild split its budget by
    """
    shared = {}
    # The importance method the search that produced this recipe ranked its
    # features by. Condition 4 of this space weights by it and condition 5
    # weights by whichever method the sweep is on, so when the two names agree
    # the two conditions are one condition. See _build_conditions.
    shared['blueprint_importance_method'] = getattr(bp, 'importance_method', None)
    X_base = train_reps[base_rep_name]
    sc_base = StandardScaler()
    shared['X_base_std'] = sc_base.fit_transform(X_base)

    # Subset to the rep pool the blueprint actually scored. Passing ALL
    # train_reps lets the allocation rank in reps the pool never considered,
    # so the per-fold hybrid would have a different rep mix than the
    # published blueprint. (Audit P1 finding.)
    bp_reps = getattr(bp, 'reps', None)
    if bp_reps:
        train_reps_for_bp = {r: train_reps[r]
                             for r in bp_reps if r in train_reps}
        if not train_reps_for_bp:
            # bp.reps not in this fold's rep pool — fall back rather than
            # silently produce an empty hybrid.
            train_reps_for_bp = train_reps
    else:
        train_reps_for_bp = train_reps

    # The same subset for the held-out rows. build_fold_hybrid applies the
    # selection to them; nothing here reads the result.
    if val_reps is None:
        val_reps_for_bp = train_reps_for_bp
    else:
        val_reps_for_bp = {r: val_reps[r] for r in train_reps_for_bp
                           if r in val_reps}
        if not val_reps_for_bp:
            val_reps_for_bp = train_reps_for_bp

    # The objective the recipe was searched under. The bare create_hybrid call
    # this replaced passed none, so it took the OBJECTIVE_PREDICTIVE default:
    # a recipe from the similarity family had its budget re-split by predicted
    # model performance instead of by the neighbour correlation it was chosen
    # under. build_fold_hybrid reads it off the blueprint.
    _objective = getattr(bp, 'objective', None)
    shared['hybrid_objective'] = _objective or 'predictive'
    _geometry = _objective is not None and _is_geometry_objective(_objective)

    X_hybrid = None
    try:
        # build_fold_hybrid, not a bare create_hybrid: it forwards the recipe
        # fields that call omitted — candidate_ranking, augmentation_margin,
        # augmentation_fraction, augmentation_budget, aug_max_candidates,
        # augmentation_threshold, aug_per_source_budget — every one of which
        # the search varied and three of which are columns of the stored
        # result's key, so the rebuild ran a different recipe than the row.
        X_hybrid, _X_val_hybrid, fi = build_fold_hybrid(
            train_reps_for_bp, val_reps_for_bp, y_train, bp,
            augmentations_train=augmentations,
            augmentations_val=augmentations_val,
            groups=groups,
            **({} if _geometry else {
                'model': model,
                'metric_fn': metric_fn,
                'lower_is_better': lower_is_better,
            }),
        )
    except Exception as e:
        print(f"      hybrid creation failed: {e}")

    if X_hybrid is not None:
        sc_hyb = StandardScaler()
        shared['X_hyb_std'] = sc_hyb.fit_transform(X_hybrid)
        shared['X_hybrid_raw'] = X_hybrid
        try:
            shared['imp_hybrid_via_blueprint'] = compute_feature_importance(
                {'hybrid': X_hybrid}, y_train, method=bp.importance_method
            )['hybrid']
        except Exception as e:
            print(f"      imp_hybrid_via_blueprint failed: {e}")
            shared['imp_hybrid_via_blueprint'] = None
    else:
        shared['X_hyb_std'] = None
        shared['X_hybrid_raw'] = None
        shared['imp_hybrid_via_blueprint'] = None
    return shared


def _build_conditions(train_reps, y_train, augmentations, bp,
                       imp_method, base_rep_name, shared=None,
                       shared_by_space=None):
    """Build all 5 conditions for one importance method.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.
        imp_method: Importance method name for weighting ablation.
        base_rep_name: Name of base representation.
        shared: Optional output of _build_fold_shared. If provided we reuse
            the cached hybrid + via_blueprint importance instead of
            recomputing them once per imp_method — the same hybrid was
            previously rebuilt 3× per (sigma, fold) on the inner imp_method
            loop, and the via_blueprint importance was recomputed at the
            same rate even though it's imp_method-independent.
        shared_by_space: {space name: _build_fold_shared output}, one entry per
            family of stored search the pool held. Added 2026-09-19. Takes the
            place of `shared`, which is the same thing for the one space
            'hybrid' and is still accepted.

    Returns three dicts.
      {condition_name: X_standardized}   the spaces to score.
      {condition_name: True}             every condition whose importance
            scores were all zero. Such a condition is the unweighted matrix
            itself, so its estimator scores are a duplicate of its unweighted
            twin rather than a measurement of the weighting. Conditions that
            weighted normally are absent.
      {condition_name: {...}}            every condition that is a second copy
            of another condition in this same call, with the one it copies and
            why. Added 2026-09-21: a hybrid space's _imp_own condition is its
            _imp_base condition whenever the recipe's own importance method is
            the method the sweep is on. Absent when it did not happen.
    """
    conditions = {}
    zero_importance = {}
    duplicate = {}
    X_base = train_reps[base_rep_name]

    # {space name: the _build_fold_shared output for that space's recipe}.
    # One entry until 2026-09-19, when the pool's second search family got its
    # own space. `shared=` alone still works and means the one space 'hybrid'.
    if shared_by_space is None:
        if shared is None:
            shared = _build_fold_shared(train_reps, y_train, augmentations, bp,
                                        base_rep_name)
        shared_by_space = {'hybrid': shared}

    if not shared_by_space:
        # No family produced a hybrid on this fold. The two base conditions
        # still have to be built, so the base matrix is scaled here rather than
        # read off a hybrid build that never happened.
        _sc = StandardScaler()
        X_base_std = _sc.fit_transform(X_base)
    else:
        X_base_std = next(iter(shared_by_space.values()))['X_base_std']

    # Condition 1: base unweighted
    conditions['base_unweighted'] = X_base_std

    # Compute base rep importance (imp_method-dependent)
    try:
        imp_base = compute_feature_importance(
            {base_rep_name: X_base}, y_train, method=imp_method
        )[base_rep_name]
    except Exception as e:
        print(f"      {imp_method} importance failed on base: {e}")
        imp_base = None

    # Condition 2: base importance-weighted (standardize-first, then weight).
    if imp_base is not None:
        conditions['base_imp_weighted'] = _weight_features(X_base_std, imp_base)
        if _importance_is_all_zero(imp_base):
            zero_importance['base_imp_weighted'] = True

    # Conditions 3 to 5, once per hybrid space. Each space's three conditions
    # are named after it, so 'hybrid_imp_own' and 'hybrid_similarity_imp_own'
    # are two rows and never one.
    for _space, _sh in shared_by_space.items():
        X_hyb_std = _sh['X_hyb_std']
        X_hybrid_raw = _sh['X_hybrid_raw']
        imp_hybrid_via_blueprint = _sh['imp_hybrid_via_blueprint']
        if X_hyb_std is None:
            continue

        # Condition 3: hybrid unweighted
        conditions[f'{_space}_unweighted'] = X_hyb_std

        # Condition 4: hybrid weighted by blueprint's importance method.
        # imp_method-INDEPENDENT — pulled from `shared` so we don't recompute
        # it on every inner-loop iteration.
        if imp_base is not None and imp_hybrid_via_blueprint is not None:
            conditions[f'{_space}_imp_base'] = _weight_features(
                X_hyb_std, imp_hybrid_via_blueprint)
            if _importance_is_all_zero(imp_hybrid_via_blueprint):
                zero_importance[f'{_space}_imp_base'] = True

        # Condition 5: hybrid weighted by the sweep's imp_method (the only
        # imp_method-dependent hybrid weighting).
        #
        # ⛔ WHEN THE TWO NAMES AGREE, CONDITIONS 4 AND 5 ARE ONE CONDITION.
        # Condition 4 weights by the recipe's own importance method; condition
        # 5 weights by the method the sweep is on. Both score the same hybrid
        # matrix against the same labels, so when those names are the same
        # string the two calls are the same call and the two spaces are the
        # same matrix. Measured on flab_thermo_tresanco under outlier_p10 at
        # level 0.5: both recipes ranked features by lightgbm_gain, and on the
        # lightgbm_gain half of the sweep hybrid_imp_base and hybrid_imp_own
        # agreed to 1e-12 in 6 of 6 readings, as did hybrid_similarity_imp_base
        # and hybrid_similarity_imp_own. Nothing in that result file said so, so
        # 2 of the 16 (importance method, condition) cells were a duplicate that
        # read as a measurement -- and each cost one importance computation on
        # the hybrid matrix.
        #
        # The array is reused rather than recomputed, so the condition is still
        # present under its own name and its numbers are unchanged to the bit.
        # Which cells are duplicates is recorded in 'duplicate_conditions'.
        _bp_imp = _sh.get('blueprint_importance_method')
        if (_bp_imp is not None and _bp_imp == imp_method
                and imp_hybrid_via_blueprint is not None):
            duplicate[f'{_space}_imp_own'] = {
                'duplicate_of': f'{_space}_imp_base',
                'reason': (f"the recipe for {_space} ranked its features by "
                           f"{_bp_imp!r}, which is the importance method this "
                           f"sweep is on, so both conditions weight the same "
                           f"hybrid matrix by the same scores"),
                'importance_method': imp_method,
            }
            conditions[f'{_space}_imp_own'] = conditions.get(
                f'{_space}_imp_base',
                _weight_features(X_hyb_std, imp_hybrid_via_blueprint))
            if _importance_is_all_zero(imp_hybrid_via_blueprint):
                zero_importance[f'{_space}_imp_own'] = True
            continue
        try:
            imp_hybrid_own = compute_feature_importance(
                {'hybrid': X_hybrid_raw}, y_train, method=imp_method
            )['hybrid']
            conditions[f'{_space}_imp_own'] = _weight_features(
                X_hyb_std, imp_hybrid_own)
            if _importance_is_all_zero(imp_hybrid_own):
                zero_importance[f'{_space}_imp_own'] = True
        except Exception as e:
            print(f"      {_space}_imp_own failed: {e}")

    return conditions, zero_importance, duplicate


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes.

    Same quantity measured several times, so the mean is the reading. The two
    counts are averaged and written beside the fractions, so no ratio appears
    without what it counts.
    """
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted', 'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None
                and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def _run_condition(X_space, y_noisy, X_eval_train, X_eval_val, y_val,
                    model_factories, metric_fn, is_clf=False,
                    corrupted_mask=None, noise_condition=None):
    """Run 3 estimators × 3 k values on one condition.

    Fixed remediation: exponential weighting, alpha=0.1.
    Passes is_clf to each estimator so classification uses class-disagreement
    instead of the regression value-residual that's meaningless for class
    labels.

    Returns {f'{model}|{est_name}_k{k}': denoised_score}. The model is in the key
    because this experiment fits the whole roster as of 2026-09-12, and recovery is
    a fraction of that model's own clean-to-noisy gap. The bar is a separator no
    model name and no estimator name contains. Plus `_detection` -- how well
    each (estimator, k) found the molecules whose label actually moved, before any
    repair. Owner's decision 2026-09-12. Absent under the six conditions where every
    training molecule moves, and it costs no extra model fit.
    """
    results = {}
    # Record per-(estimator, k) failures so a degenerate rerun isn't blind:
    # previously every error was swallowed into `results[key] = None`, the
    # aggregation dropped all-None cells, and per_importance_method came back
    # as empty {} with no trace of WHY. Now the error string is preserved.
    errors = {}
    detection = {}

    for est_name, est_cls in ESTIMATOR_FACTORIES.items():
        for k in K_VALUES:
            est_key = f'{est_name}_k{k}'
            try:
                est = est_cls(k=k, metric=None, is_clf=is_clf)
                scores = est.estimate(X_space, y_noisy)
                if corrupted_mask is not None:
                    _det = detection_scores(corrupted_mask, scores, noise_condition)
                    if _det is not None:
                        detection[est_key] = _det
                weights = weights_from_scores(
                    scores, strength=REPAIR_STRENGTH,
                    method=REPAIR_WEIGHT_METHOD)
            except Exception as e:
                # The suspicion score itself failed, so no model can be fitted on
                # its weights. Record it once per model rather than once, so the
                # missing cells are traceable per model.
                for mn in model_factories:
                    results[f'{mn}|{est_key}'] = None
                    errors[f'{mn}|{est_key}'] = f'{type(e).__name__}: {e}'
                continue
            for mn, mf in model_factories.items():
                key = f'{mn}|{est_key}'
                try:
                    model = mf()
                    model.fit(X_eval_train, y_noisy, sample_weight=weights)
                    y_pred = model.predict(X_eval_val)
                    results[key] = float(metric_fn(y_val, y_pred))
                except Exception as e:
                    results[key] = None
                    errors[key] = f'{type(e).__name__}: {e}'

    if errors:
        results['_errors'] = errors
    if detection:
        results['_detection'] = detection

    return results


def run(data, config, output_dir, dataset_name, **kwargs):
    """N12: Feature importance-weighted distance for noise estimation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N12: IMPORTANCE-WEIGHTED DISTANCE FOR NOISE ESTIMATION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # The whole roster, not random forest alone. Owner's decision 2026-09-12.
    # The distances and the suspicion scores do not involve a model at all; only
    # the score after repair does, so the added cost is the downstream fits.
    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg -- the only kwarg it ever
    # forwards is noise_condition (noise_helpers.py:1053). This file read
    # kwargs alone, so the list the runner was asked for was silently dropped
    # and every run fell back to the module default. Eight files in this
    # directory had it; the same files read config['n_splits'] correctly two
    # lines away. Fixed 2026-09-20. An explicit kwarg still wins, which is what
    # a smoke or a direct caller uses.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            "n12: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    noise_condition = kwargs.get('noise_condition', NOISE_CONDITION)
    # Three corruptions per fold, averaged — matching n07, n11 and n13.
    # User decision 2026-09-12. Overridable for a smoke.
    NOISE_SEEDS = kwargs.get('noise_seeds', [0, 1, 2])
    sigmas = kwargs.get('sigmas') or kwargs.get('levels') or SIGMAS
    if sigmas is None:
        # detect_grid=True: on censoring this asks for the four censored fractions
        # this experiment runs (0.0, 0.10, 0.25, 0.40), not n13's full six. Owner's
        # decision 2026-09-12. It changes nothing for the other ten conditions.
        sigmas = list(dataset_noise_levels(dataset_name, strategy=noise_condition,
                                           detect_grid=True, is_clf=is_clf))
    noise_groups, _n_ng, _lg = noise_groups_for(data)
    noise_provenance = {}
    sample_X = next(iter(reps.values()))

    # Control arm = one plain base representation. Shared helper: it will
    # NOT fall back onto a structure-tuned sim_* recipe, which is a study
    # treatment and must never become the thing treatments are measured
    # against. See _shared.pick_noise_base_rep.
    base_rep_name = pick_noise_base_rep(reps, dataset_name)

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Base rep: {base_rep_name}")
    print(f"  K values: {K_VALUES}")
    print(f"  Importance methods: {IMPORTANCE_METHODS}")
    print(f"  Sigmas: {sigmas}, Folds: {n_folds}")

    # Blueprint selected from the CLEAN-LABEL pool, per-fold build under noise
    # — best-case hybrid, NOT a fair baseline. The structural choice of
    # blueprint (which reps / allocation / budget) comes ONCE from the
    # clean-label pool helper; per fold below the hybrid is REBUILT under the
    # noisy labels using that blueprint. This carries an oracle leak: a fully
    # leak-free design would re-select the blueprint per fold under noise. We
    # keep clean-label blueprint selection because the ablation's question is
    # "does importance-weighting the distance help given a good hybrid", not
    # "can we discover the hybrid under noise" — but the hybrid conditions must
    # not be read as a realistic, leak-free comparison. (Audit P1#5.)
    #
    # 2026-09-12, user decision: the subset of the precomputed pool this file may
    # take its hybrid from now INCLUDES the searches run under
    # objective='similarity' — searches whose features were picked so that the
    # distance geometry predicts each molecule's activity from its five nearest
    # neighbours — and not only the prediction-built ones. Her words: n12 exists
    # to ask which distance geometry makes neighbour-based noise estimation work,
    # and every condition it compares is a geometry.
    #
    # What changed: pool_subset='predictive_and_similarity' on this one call.
    # This is NOT distance=True — that flag picks a different evaluator and
    # budget lineage and its searches are still prediction-built. n12 does not
    # pass it and it is untouched.
    #
    # The two families' stored scores are in different units: a prediction-built
    # search stores the dataset's own metric, a similarity-built one stores a
    # Spearman correlation between each held-out molecule's activity and what its
    # five nearest neighbours predict. _shared does not rank one against the
    # other; it returns the first family the subset names and hands back every
    # family's own best. Which family the winning recipe came from is recorded in
    # the result file below.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset=POOL_SUBSET,
    )
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    print(f"  Hybrid recipe built under objective: {meta['objective']} "
          f"(chosen by {meta['objective_choice']}, "
          f"pool subset {meta['pool_subset']})")

    # One recipe per family the pool holds, and a written reason for each family
    # that produced none. `bp` stays the recipe the pool RETURNED, which is what
    # every 'hybrid_*' key below already described.
    hybrid_recipes, hybrid_unavailable = pool_recipes_by_family(
        meta, SPACE_FOR_FAMILY,
        eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
    for r in hybrid_recipes:
        print(f"  hybrid space {r['space']}: family {r['objective']!r}, "
              f"budget {r['budget']} features, allocation "
              f"{r['allocation_method']}, tier {r['tier']}/"
              f"{r['aug_condition']}, search score "
              f"{r['search_score_mean']} in {r['search_metric_name']!r}")
    for _sp, _why in hybrid_unavailable.items():
        print(f"  no {_sp} space: {_why}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'base_rep': base_rep_name,
        'k_values': K_VALUES,
        'importance_methods': IMPORTANCE_METHODS,
        # The power the importance was raised to, the distance the neighbours
        # were found under, and the power that distance would match. Without
        # these three a result file cannot be told from one run at the other
        # exponent. See IMPORTANCE_EXPONENT above.
        'importance_exponent': float(IMPORTANCE_EXPONENT),
        'distance_metric': _get_distance_metric(),
        'exponent_matched_to_distance_metric': EXPONENT_MATCHED_TO_METRIC.get(
            _get_distance_metric()),
        'sigmas': sigmas,
        'noise_condition': noise_condition,
        'noise_provenance': noise_provenance,
        # 2026-09-12: which family the winning hybrid recipe came from, so a
        # reader can tell a prediction-built recipe from a similarity-built one
        # without re-running anything. 'hybrid_pool_by_objective' holds each
        # eligible family's own best, because the two scores are not on one
        # ruler and the pool did not compare them.
        'hybrid_family': meta['objective'],
        'hybrid_family_choice': meta['objective_choice'],
        'hybrid_pool_subset': meta['pool_subset'],
        'hybrid_metric': meta['metric_name'],
        'hybrid_tier': meta['tier'],
        'hybrid_aug_condition': meta['aug_condition'],
        'hybrid_pool_by_objective': convert_numpy({
            o: {'tier': d['tier'], 'aug_condition': d['aug_condition'],
                'score_mean': d['score_mean'], 'metric_name': d['metric_name'],
                'lower_is_better': d['lower_is_better']}
            for o, d in meta.get('by_objective', {}).items()
        }),
        # Which of the hybrid spaces below holds the recipe the pool returned,
        # what each of the others was built from, and why a family that was
        # asked for produced no space. Added 2026-09-19, with the spaces.
        'hybrid_space_of_pool_winner': SPACE_FOR_FAMILY.get(
            meta['objective'], f"hybrid_{meta['objective']}"),
        'hybrid_spaces': convert_numpy({
            r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'is_the_recipe_the_pool_returned': r['is_pool_winner'],
                'tier': r['tier'], 'aug_condition': r['aug_condition'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
                'conditions': [f"{r['space']}{suf}"
                               for suf in HYBRID_CONDITION_SUFFIXES],
            } for r in hybrid_recipes}),
        'hybrid_spaces_unavailable': hybrid_unavailable,
        'hybrid_spaces_note': (
            'One hybrid space per family of stored search, each carrying the '
            'same three conditions. A predictive search chose its features to '
            "predict the label and its search_score_mean is in the dataset's "
            "own metric. A similarity search chose its features so that each "
            "molecule's activity is predicted by its five nearest neighbours "
            'in the distance those features define, fits no downstream model, '
            'and its search_score_mean is that correlation. The two search '
            'scores are in different units and nothing here ranks one against '
            "the other. Every condition below is scored by this file's own "
            'estimators on the same folds under the same injected noise.'),
        # Every weighted space whose importance scores were all zero, so that
        # the space is its unweighted twin and its numbers are duplicates. One
        # entry per (importance method, condition), absent when it never
        # happened, and 'n_builds_all_zero' counts how many of that condition's
        # builds went that way. A condition is built once per (noise level,
        # fold, importance method) — 60 over 6 levels, 5 folds and 2 importance
        # methods — and the noise levels it happened at are listed beside it.
        'zero_importance_spaces': {},
        # Every condition that is a second copy of another condition rather than
        # a measurement of its own. One entry per (importance method,
        # condition), absent when it did not happen.
        #
        # ⛔ TWO OF THE SIXTEEN CELLS WERE A DUPLICATE AND NOTHING SAID SO.
        # A hybrid space carries an _imp_base condition, weighted by the
        # recipe's own importance method, and an _imp_own condition, weighted by
        # the method the sweep is on. When those two names are the same string
        # the two conditions weight the same hybrid matrix by the same scores.
        # Measured on flab_thermo_tresanco under outlier_p10 at level 0.5: both
        # recipes ranked by lightgbm_gain, so on the lightgbm_gain half of the
        # sweep hybrid_imp_base and hybrid_imp_own agreed to 1e-12 in 6 of 6
        # readings, and so did the two hybrid_similarity ones. On the treeshap
        # half they agreed in 0 of 6, which is what a real second measurement
        # looks like. Recorded here since 2026-09-21.
        'duplicate_conditions': {},
        # Hybrid feature matrices built: one round per (noise level, fold),
        # shared by both importance methods. 30 over 6 levels and 5 folds. A
        # round builds one matrix per hybrid space, so with both search
        # families in the pool it is 60 matrices from those 30 rounds — the
        # per-space counts are in 'n_hybrid_builds_by_space'.
        'n_hybrid_builds': 0,
        'n_hybrid_builds_by_space': {},
        'per_sigma': {},
    }
    zero_imp_acc = results['zero_importance_spaces']
    dup_cond_acc = results['duplicate_conditions']

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        # One list per model name for both references. Recovery is a fraction of
        # THIS model's clean-to-noisy gap, so the two references cannot be shared.
        clean_scores = {m: [] for m in downstream_models}
        noisy_scores = {m: [] for m in downstream_models}
        # {imp_method: {condition: {est_k: [scores]}}}
        grid_acc = {}

        # THREE NOISE DRAWS PER FOLD, averaged. User decision 2026-09-12.
        #
        # A run corrupts the labels one particular way. Another draw shifts the
        # answer, and with a single draw there is no telling how much of a
        # number is the noise and how much is that draw being kind or harsh.
        # n07, n11 and n13 already average three; this file used to average one.
        #
        # It matters more since the level came out of the seed on 2026-09-12:
        # one draw now serves the whole level sweep, so an unlucky draw tilts
        # every level the same way instead of only one of them. Measured on
        # ridge with 400 training molecules and 10 draws per level, the spread
        # across draws at one level was 1.0x to 2.0x the step to the next level.
        #
        # Written as a (fold, draw) pass list rather than a nested loop so the
        # body below is untouched. Every score list gains 3 entries per fold
        # instead of 1, so the means at the end average over both axes.
        # At level 0 a regression draw delivers exactly zero, so three draws are
        # three identical runs — collapse to one. NOT on a classification set:
        # calibrate_flip_probability returns a floor rather than zero at a target
        # of 0.0, so the 0.0 cell there really does flip a few labels and a
        # different number of them each draw.
        _seeds = NOISE_SEEDS if (sigma > 0 or is_clf) else NOISE_SEEDS[:1]
        _splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))
        _passes = [(_fi + 1, _tr, _va, _ns)
                   for _fi, (_tr, _va) in enumerate(_splits)
                   for _ns in _seeds]
        for fold_idx, train_idx, val_idx, noise_seed in _passes:
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Fixed eval features
            X_eval_train = reps[base_rep_name][train_idx]
            X_eval_val = reps[base_rep_name][val_idx]
            scaler_eval = StandardScaler()
            X_eval_train_s = scaler_eval.fit_transform(X_eval_train)
            X_eval_val_s = scaler_eval.transform(X_eval_val)

            # Clean baseline
            # The clean fit has no noise in it, so it is the same on all three
            # draws of a fold. Score it on the first draw only, or clean_scores
            # carries three copies of every fold.
            if noise_seed == _seeds[0]:
                for _mn in downstream_models:
                    _mdl = model_factories[_mn]()
                    _mdl.fit(X_eval_train_s, y_train_clean)
                    clean_scores[_mn].append(
                        float(metric_fn(y_val, _mdl.predict(X_eval_val_s))))

            # Inject noise (training only)
            _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                   noise_seed=noise_seed, is_clf=is_clf)
            y_noisy, _cmask, _prov = inject_train_labels(
                labels, _spec, train_idx=train_idx, is_clf=is_clf,
                groups=noise_groups, dataset_name=dataset_name)
            noise_provenance.setdefault(_spec.spec_id, _prov)

            # Noisy baseline
            for _mn in downstream_models:
                _mdl = model_factories[_mn]()
                _mdl.fit(X_eval_train_s, y_noisy)
                noisy_scores[_mn].append(
                    float(metric_fn(y_val, _mdl.predict(X_eval_val_s))))

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
            val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

            # Build the imp-method-independent pieces (hybrid + via_blueprint
            # importance) ONCE per (sigma, fold); the inner imp_method loop
            # then only recomputes the truly imp-method-dependent pieces.
            # The random forest and the dataset's metric go in so the build's
            # own accept-or-reject step is scored by the estimator and the
            # quantity every condition below is finally judged on.
            # One build per family the pool holds. build_fold_hybrid reads the
            # objective off each recipe, so a geometry recipe splits its budget
            # by the neighbour correlation it was chosen under and a predictive
            # one by the model's held-out score. Nothing else differs between
            # the two calls.
            fold_shared_by_space = {}
            for _r in hybrid_recipes:
                fold_shared_by_space[_r['space']] = _build_fold_shared(
                    train_reps_fold, y_noisy, train_augs, _r['blueprint'],
                    base_rep_name,
                    groups=(groups[train_idx] if groups is not None else None),
                    val_reps=val_reps_fold, augmentations_val=val_augs,
                    # The hybrid build's own accept-or-reject step keeps the
                    # first model in the roster. It decides which features go
                    # in, not which model is finally scored, and making it
                    # depend on the model would give every model a different
                    # feature space to be compared in.
                    model=model_factories[downstream_models[0]](),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
                results['n_hybrid_builds_by_space'][_r['space']] = (
                    results['n_hybrid_builds_by_space'].get(_r['space'], 0) + 1)
            # Unchanged in meaning: one per (noise level, fold), counted once
            # however many spaces that round built. Every space's own count is
            # in 'n_hybrid_builds_by_space'.
            results['n_hybrid_builds'] += 1
            # The objective each space's rebuild split its budget by. A single
            # string until 2026-09-19, when a second space arrived; the string
            # is kept under its own key so nothing that read it breaks.
            results['hybrid_objective_rebuilt_under_by_space'] = {
                _sp: _sh['hybrid_objective']
                for _sp, _sh in fold_shared_by_space.items()}
            if 'hybrid' in fold_shared_by_space:
                results['hybrid_objective_rebuilt_under'] = (
                    fold_shared_by_space['hybrid']['hybrid_objective'])

            for imp_method in IMPORTANCE_METHODS:
                if imp_method not in grid_acc:
                    grid_acc[imp_method] = {}

                conditions, zero_importance, duplicate_conds = _build_conditions(
                    train_reps_fold, y_noisy, train_augs, bp,
                    imp_method, base_rep_name,
                    shared_by_space=fold_shared_by_space,
                )
                for _cond_name, _rec in duplicate_conds.items():
                    _d = (dup_cond_acc.setdefault(imp_method, {})
                          .setdefault(_cond_name, dict(_rec, n_builds=0,
                                                       sigmas=[])))
                    _d['n_builds'] += 1
                    if sigma not in _d['sigmas']:
                        _d['sigmas'].append(sigma)
                for _cond_name in zero_importance:
                    _zi = (zero_imp_acc.setdefault(imp_method, {})
                           .setdefault(_cond_name,
                                       {'n_builds_all_zero': 0,
                                        'duplicate_of': zero_importance_twin(_cond_name),
                                        'sigmas': []}))
                    _zi['n_builds_all_zero'] += 1
                    if sigma not in _zi['sigmas']:
                        _zi['sigmas'].append(sigma)

                for cond_name, X_space in conditions.items():
                    if cond_name not in grid_acc[imp_method]:
                        grid_acc[imp_method][cond_name] = {}

                    cond_results = _run_condition(
                        X_space, y_noisy, X_eval_train_s, X_eval_val_s,
                        y_val, model_factories, metric_fn, is_clf=is_clf,
                        corrupted_mask=_cmask, noise_condition=noise_condition,
                    )

                    for ek, score in cond_results.items():
                        if ek == '_detection':
                            # One reading per (fold, draw, estimator, k). Kept as
                            # a list per estimator and averaged below, so a later
                            # draw does not overwrite an earlier one.
                            _dstore = grid_acc[imp_method][cond_name].setdefault(
                                '_detection_per_pass', {})
                            for _ek2, _row in score.items():
                                _dstore.setdefault(_ek2, []).append(_row)
                            continue
                        if ek == '_errors':
                            # Carry the per-(estimator,k) error dict through so
                            # a degenerate output records WHY, instead of an
                            # untraceable empty {}. Merge across folds/sigmas.
                            merged = grid_acc[imp_method][cond_name].setdefault(
                                '_errors', {})
                            merged.update(score)
                            continue
                        if ek not in grid_acc[imp_method][cond_name]:
                            grid_acc[imp_method][cond_name][ek] = []
                        if score is not None:
                            grid_acc[imp_method][cond_name][ek].append(score)

        # Aggregate
        clean_by_model = {m: float(np.mean(v)) for m, v in clean_scores.items() if v}
        noisy_by_model = {m: float(np.mean(v)) for m, v in noisy_scores.items() if v}
        _models_scored = [m for m in downstream_models
                          if m in clean_by_model and m in noisy_by_model]
        if not _models_scored:
            raise RuntimeError(
                f"n12: no model produced both a clean and a noisy reference score "
                f"at level {sigma}. Asked for {downstream_models}.")
        gap_by_model = {}
        for _m in _models_scored:
            _c, _nz = clean_by_model[_m], noisy_by_model[_m]
            gap_by_model[_m] = (_nz - _c) if lower_is_better else (_c - _nz)
        # The first model in the roster fills the three top-level fields, and
        # 'reference_model' says which model that is.
        _ref_model = _models_scored[0]
        clean_mean = clean_by_model[_ref_model]
        noisy_mean = noisy_by_model[_ref_model]
        # Sign-normalised so a positive gap always means the injected noise cost
        # the model something. Unguarded clean-minus-noisy is NEGATIVE on rmse
        # exactly when the noise did damage. n07 was fixed this way on 2026-09-10
        # (metric-direction audit); this file was not, so the same field name
        # meant opposite things on an rmse dataset. Fixed 2026-09-12.
        # The recovery RATIO below is unaffected either way: its numerator and
        # denominator are both differences in the same metric, so the sign cancels.
        gap = gap_by_model[_ref_model]

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
            'reference_model': _ref_model,
            'models_scored': list(_models_scored),
            'clean_score_by_model': clean_by_model,
            'noisy_score_by_model': noisy_by_model,
            'gap_by_model': {k: float(v) for k, v in gap_by_model.items()},
            'per_importance_method': {},
        }

        for imp_method in IMPORTANCE_METHODS:
            imp_result = {}
            for cond_name in CONDITION_NAMES:
                cond_agg = {}
                cond_errors = None
                for ek, score_list in grid_acc.get(imp_method, {}).get(cond_name, {}).items():
                    if ek == '_detection_per_pass':
                        cond_agg['detection'] = {
                            _ek2: _mean_detection(_rows)
                            for _ek2, _rows in score_list.items()}
                        continue
                    if ek == '_errors':
                        # Diagnostic marker: pass the error dict straight
                        # through rather than averaging it as a score list.
                        cond_errors = score_list
                        continue
                    if score_list:
                        denoised = float(np.mean(score_list))
                        _mn = ek.split('|', 1)[0]
                        _c = clean_by_model.get(_mn)
                        _nz = noisy_by_model.get(_mn)
                        # ⛔ THE GAP THIS DIVIDES BY HAS A FLOOR.
                        #
                        # Recovery is how far the repaired fit moved from the
                        # noisy fit, as a share of how far the clean fit is
                        # from the noisy fit. When that second distance is a
                        # few thousandths, a small swing in the repaired score
                        # comes out as hundreds of percent. Measured on
                        # flab_thermo_tresanco under outlier_p10 at level 0.5:
                        # the random forest's clean-to-noisy gap was 0.008551
                        # spearman against a clean score of 0.458964 -- 1.86% of
                        # it. Of the 192 recovery readings in that result file,
                        # 104 fell outside 0 to 100 percent: 68 below 0 and 36
                        # above 100. The random forest's 96 readings ran from
                        # -615.6 to +578.0 percent. lightgbm's gap on the same
                        # folds was 0.048543 against a clean score of 0.426777,
                        # which is 11.37% of it, and its 96 readings ran from
                        # -33.9 to +149.3 percent. This floor refuses all 96 of
                        # the random forest's and keeps all 96 of lightgbm's,
                        # which removes 79 of those 104 and leaves 25.
                        #
                        # n06, n07, n11 and n21 all refuse the figure when the
                        # gap is under GAP_MIN_FRAC of the clean score. n12 was
                        # the one file in this directory that did not: its only
                        # guard was abs(gap) > 1e-10. Same floor, same wording,
                        # since 2026-09-21. A refused cell keeps its denoised
                        # score and says in 'recovery_note' why no percentage
                        # was written.
                        _floor = (max(1e-10, GAP_MIN_FRAC * abs(_c))
                                  if (_c is not None and np.isfinite(_c))
                                  else 1e-10)
                        _gap_too_small = not (abs(gap_by_model[_mn]) > _floor)
                        if (_c is not None and _nz is not None
                                and np.isfinite(_c) and np.isfinite(_nz)
                                and not _gap_too_small):
                            recovery = float((denoised - _nz) / (_c - _nz) * 100)
                            recovery_note = None
                        else:
                            recovery = float('nan')
                            recovery_note = (
                                f'not computed: the clean-to-noisy gap '
                                f'({gap_by_model[_mn]:.6g}) is under '
                                f'{GAP_MIN_FRAC:.0%} of the clean score '
                                f'({_c:.6g})'
                                if (_gap_too_small and _c is not None
                                    and np.isfinite(_c))
                                else 'not computed: a score was not finite')
                        cond_agg[ek] = {
                            'denoised': denoised,
                            'recovery': recovery,
                            'recovery_note': recovery_note,
                            # Named, not only encoded in the key.
                            'model': _mn,
                            'estimator': ek.split('|', 1)[-1],
                        }
                if cond_errors and not cond_agg:
                    # Every estimator failed for this condition: keep the
                    # errors so a future rerun isn't blind to the cause.
                    cond_agg['_errors'] = cond_errors
                if cond_agg:
                    imp_result[cond_name] = cond_agg
            sigma_result['per_importance_method'][imp_method] = imp_result

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_result)

        # Print compact summary
        for imp_method in IMPORTANCE_METHODS:
            print(f"    {imp_method}:")
            imp_data = sigma_result['per_importance_method'].get(imp_method, {})
            for cond_name in CONDITION_NAMES:
                cond_data = imp_data.get(cond_name, {})
                if cond_data:
                    recoveries = [v['recovery'] for v in cond_data.values()
                                  if not np.isnan(v.get('recovery', float('nan')))]
                    avg_r = float(np.mean(recoveries)) if recoveries else float('nan')
                    n_pos = sum(1 for r in recoveries if r > 0)
                    print(f"      {cond_name:25s}: avg_recovery={avg_r:+.1f}%, "
                          f"positive={n_pos}/{len(recoveries)}")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # Summary
    results['summary'] = _compute_summary(results)

    s = results['summary']
    print(f"\n  === SUMMARY ===")
    print("  CAVEAT: differences may fall within run-to-run variance — read the")
    print("  boolean flags against per_condition_recovery_std/range, not as")
    print("  significance tests. Hybrid conditions use a clean-label blueprint")
    print("  rebuilt per-fold under noise (best-case hybrid, not a fair baseline).")
    for sk, sv in s.items():
        if isinstance(sv, dict) and 'weighting_helps_base' in sv:
            print(f"  {sk}:")
            print(f"    Weighting helps base: {sv['weighting_helps_base']}")
            print(f"    Weighting helps hybrid: {sv['weighting_helps_hybrid']}")
            print(f"    Hybrid already weighted: {sv['hybrid_already_weighted']}")
            print(f"    Best k: {sv.get('best_k')}")
            print(f"    Best importance method: {sv.get('best_importance_method')}")
            cond_std = sv.get('per_condition_recovery_std', {})
            if cond_std:
                print(f"    Per-condition recovery std: "
                      f"{ {c: round(v, 1) for c, v in cond_std.items() if not np.isnan(v)} }")

    return results


def _compute_summary(results):
    """Compute ablation answers."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        # Collect per-condition average recovery across importance methods
        cond_recoveries = {}  # {condition: [all recoveries]}

        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            for cond_name, cond_data in imp_data.items():
                if cond_name not in cond_recoveries:
                    cond_recoveries[cond_name] = []
                for ek, rv in cond_data.items():
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        cond_recoveries[cond_name].append(r)

        cond_avg = {c: float(np.mean(rs)) if rs else float('nan')
                    for c, rs in cond_recoveries.items()}
        # Spread of each condition's recovery across (fold × est × k × imp).
        # Reported alongside the mean so a small mean gap can be read against
        # its own noise floor — many condition differences here are expected to
        # sit inside this spread (see module-docstring caveat).
        cond_std = {c: float(np.std(rs)) if len(rs) > 1 else float('nan')
                    for c, rs in cond_recoveries.items()}
        cond_range = {c: [float(np.min(rs)), float(np.max(rs))] if rs else [float('nan'), float('nan')]
                      for c, rs in cond_recoveries.items()}

        # Ablation answers
        base_uw = cond_avg.get('base_unweighted', float('nan'))
        base_w = cond_avg.get('base_imp_weighted', float('nan'))

        def _answers_for(hybrid_space):
            """The three ablation answers, read in one hybrid space."""
            uw = cond_avg.get(f'{hybrid_space}_unweighted', float('nan'))
            w_base = cond_avg.get(f'{hybrid_space}_imp_base', float('nan'))
            w_own = cond_avg.get(f'{hybrid_space}_imp_own', float('nan'))
            return {
                'weighting_helps_base': bool(
                    not np.isnan(base_w) and not np.isnan(base_uw)
                    and base_w > base_uw),
                'weighting_helps_hybrid': bool(
                    not np.isnan(w_own) and not np.isnan(uw)
                    and (w_own > uw or w_base > uw)),
                # Redundant if <space>_unweighted is within 2 points of
                # <space>_imp_own, both in percent of the damage recovered.
                'hybrid_already_weighted': bool(
                    not np.isnan(uw) and not np.isnan(w_own)
                    and abs(uw - w_own) < 2.0),
            }

        # Which hybrid spaces this run produced conditions for. 2026-09-19:
        # there can be more than one, because the pool holds more than one
        # family of stored search.
        hybrid_spaces_seen = sorted({
            c[:-len('_unweighted')] for c in cond_avg
            if c.endswith('_unweighted') and c != 'base_unweighted'})
        answers_by_space = {hs: _answers_for(hs) for hs in hybrid_spaces_seen}

        # The three top-level flags stay the PREDICTIVE space's, unchanged.
        _pred = answers_by_space.get('hybrid', {
            'weighting_helps_base': bool(
                not np.isnan(base_w) and not np.isnan(base_uw)
                and base_w > base_uw),
            'weighting_helps_hybrid': False,
            'hybrid_already_weighted': False})
        weighting_helps_base = _pred['weighting_helps_base']
        weighting_helps_hybrid = _pred['weighting_helps_hybrid']
        hybrid_already_weighted = _pred['hybrid_already_weighted']

        # Best k. Read over the five conditions this file had before
        # 2026-09-19, so the number means what it meant; 'per_k_avg_recovery'
        # and 'best_k' would otherwise have moved because a second hybrid
        # space added three more conditions to the same average.
        _ORIGINAL_CONDITIONS = ('base_unweighted', 'base_imp_weighted',
                                'hybrid_unweighted', 'hybrid_imp_base',
                                'hybrid_imp_own')
        k_recoveries = {}
        k_recoveries_all = {}
        for imp_data in sigma_data.get('per_importance_method', {}).values():
            for _cond_name, cond_data in imp_data.items():
                for ek, rv in cond_data.items():
                    # Parse k from key like 'neighbor_residual_k10'
                    if '_k' not in ek:
                        continue
                    k_val = int(ek.split('_k')[-1])
                    r = rv.get('recovery', float('nan'))
                    if np.isnan(r):
                        continue
                    k_recoveries_all.setdefault(k_val, []).append(r)
                    if _cond_name in _ORIGINAL_CONDITIONS:
                        k_recoveries.setdefault(k_val, []).append(r)

        best_k = None
        best_k_avg = float('-inf')
        for k_val, rs in k_recoveries.items():
            avg = float(np.mean(rs))
            if avg > best_k_avg:
                best_k_avg = avg
                best_k = k_val

        # Best importance method. Same reading as best_k: over the five
        # conditions this file had before 2026-09-19, with every condition's
        # version of the number beside it.
        method_recoveries = {}
        method_recoveries_all = {}
        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            all_r, every_r = [], []
            for _cond_name, cond_data in imp_data.items():
                for rv in cond_data.values():
                    r = rv.get('recovery', float('nan'))
                    if np.isnan(r):
                        continue
                    every_r.append(r)
                    if _cond_name in _ORIGINAL_CONDITIONS:
                        all_r.append(r)
            if all_r:
                method_recoveries[imp_method] = float(np.mean(all_r))
            if every_r:
                method_recoveries_all[imp_method] = float(np.mean(every_r))

        best_imp = max(method_recoveries, key=method_recoveries.get) if method_recoveries else None

        summary[f'sigma_{sigma_key}'] = convert_numpy({
            'weighting_helps_base': weighting_helps_base,
            'weighting_helps_hybrid': weighting_helps_hybrid,
            'hybrid_already_weighted': hybrid_already_weighted,
            'best_k': best_k,
            'best_importance_method': best_imp,
            'per_condition_avg_recovery': cond_avg,
            'per_condition_recovery_std': cond_std,
            'per_condition_recovery_range': cond_range,
            'per_k_avg_recovery': {str(k): float(np.mean(rs))
                                    for k, rs in k_recoveries.items() if rs},
            'per_method_avg_recovery': method_recoveries,
            # Added 2026-09-19 with the second hybrid space. The three flags
            # above and the two 'best_*' answers read the five conditions this
            # file had before that date, so they mean what they meant; these
            # read every condition this run produced.
            'answers_by_hybrid_space': answers_by_space,
            'hybrid_spaces_scored': hybrid_spaces_seen,
            'per_k_avg_recovery_every_condition': {
                str(k): float(np.mean(rs))
                for k, rs in k_recoveries_all.items() if rs},
            'per_method_avg_recovery_every_condition': method_recoveries_all,
            'caveat': ('Condition differences may fall within run-to-run '
                       'variance; compare each mean against '
                       'per_condition_recovery_std/range before claiming an '
                       'effect. Hybrid conditions use a clean-label blueprint '
                       'rebuilt per-fold under noise (best-case hybrid, not a '
                       'fair/leak-free baseline).'),
        })

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser('N12: Importance-Weighted Distance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n12_importance_weighted_distance',
                            args.dataset)
