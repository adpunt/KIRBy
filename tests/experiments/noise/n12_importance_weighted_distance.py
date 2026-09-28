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

The base pair of conditions per rank, plus three per hybrid space, × 2
estimators (neighbor_residual, local_variance_ratio) × 3 k values × 2
importance methods × the dataset's noise levels. With both search families in the pool it is 8 conditions: the two
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

THE SINGLE REPRESENTATIONS, owner's decision 2026-09-25. Each downstream
model's top 2 single representations, ranked by that model's own score in the
stored search (_shared.noise_base_reps_by_model), do two jobs:

  * DETECTION. The two base conditions are run once per rank: base_1_unweighted
    and base_1_imp_weighted find neighbours in each model's rank-1
    representation, base_2_* in its rank-2 one. A representation shared by
    several models at the same rank is built once and its rows are fitted only
    for those models.
  * EVALUATION. Every repaired fit of a model is done twice, once on each of
    its two single representations -- including the rows where the neighbours
    were found in a hybrid space. Each (model, rank) pair is an evaluation unit
    keyed '<model>@base_<k>', with its own clean score, noisy score and gap, so
    the recovery denominator is one per model per rank. Every '*_by_model'
    field and the model half of every estimator key is keyed by that unit.

Which representations those are is written to 'base_reps_by_model'.

Usage:
    python n12_importance_weighted_distance.py --dataset logd --mode intermediary
    python n12_importance_weighted_distance.py --dataset caco2 --mode full
"""
import copy
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
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores, sweep_conditions, conditions_for_task)
from experiments._shared import (
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
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
# The similarity-sweep kit and the disk caches, added 2026-09-25 (plan
# sections 1-3). See n07_remediation's SIMILARITY-SWEEP KIT block.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, RowSink, sweep_context, similarity_sweep,
    detector_scores_with, cleaned_from_weights, similarity_record,
)
from experiments.noise import noise_cache as _nc
# DIRTY-LOCAL 2026-09-28: the representations the stored search built each
# outer fold's hybrid on (n02, commit 736ec93; n37 uses it the same way).
from experiments.noise.n02_sample_weighting import search_fold_reps


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

# The two base conditions, once per rank of the single-representation
# comparison: base_1_* in each model's rank-1 representation, base_2_* in its
# rank-2 one.
BASE_CONDITION_NAMES = [f'base_{_k}{suf}'
                        for _k in range(1, NOISE_BASE_REP_COUNT + 1)
                        for suf in ('_unweighted', '_imp_weighted')]

CONDITION_NAMES = BASE_CONDITION_NAMES + [
    f'{sp}{suf}'
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
    if condition_name.startswith('base_') and \
            condition_name.endswith('_imp_weighted'):
        return condition_name[:-len('_imp_weighted')] + '_unweighted'
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


def _is_imp_method_free(condition_name):
    """True when a condition's matrix does not depend on the sweep's
    importance method: every '_unweighted' condition, and each hybrid space's
    '_imp_base' condition, which is weighted by the recipe's own importance
    method. base_<k>_imp_weighted and <space>_imp_own do depend on it."""
    return (condition_name.endswith('_unweighted')
            or condition_name.endswith('_imp_base'))


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
    Xw = X_standardized * w

    # A FEATURE WITH ZERO IMPORTANCE IS A COLUMN OF EXACT ZEROS, AND IT WAS
    # STILL BEING CARRIED THROUGH EVERY NEIGHBOUR SEARCH.
    #
    # At any exponent above zero a feature the scorer gave no importance gets
    # weight zero, so that feature is zero in every row. Measured on
    # flab_thermo_tresanco: lightgbm_gain scored 181 of onehot's 5600 features
    # above zero on the first fold, so 5419 of the 5600 features the estimators
    # computed distances over were zero everywhere -- 97 percent of the width,
    # carrying nothing.
    #
    # Dropping them is exact, not an approximation, for every metric
    # kirby.similarity.distance_config sweeps. manhattan, euclidean and
    # minkowski_p0.5 all sum a per-feature term that is zero when both values
    # are zero; cosine's dot product and both its norms gain zero from such a
    # feature. There is no per-molecule centring anywhere in that list, which
    # is the one thing that would make a zero feature change an answer.
    #
    # ⛔ At exponent 0.0 numpy gives 0 ** 0 == 1, so no feature is dropped and
    # the space is the unweighted one. That is the control and it still holds.
    live = w > 0
    n_live = int(live.sum())
    if 0 < n_live < len(w):
        return Xw[:, live]
    return Xw


def _build_fold_shared(train_reps, y_train, augmentations, bp, base_rep_name,
                       groups=None, val_reps=None, augmentations_val=None,
                       model=None, metric_fn=None, lower_is_better=None,
                       fold_idx=None, n_folds=None):
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

    base_rep_name: a representation to scale into 'X_base_std', or None.
        run() passes None: the base conditions are built per rank in
        _build_conditions from each model's own top 2.

    Returns dict with:
      'X_base_std': standardized base_rep_name, or None
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
    if base_rep_name is not None:
        sc_base = StandardScaler()
        shared['X_base_std'] = sc_base.fit_transform(train_reps[base_rep_name])
    else:
        shared['X_base_std'] = None

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

    # DIRTY-LOCAL 2026-09-28. A recipe with rep_count below the pool size keeps,
    # in the search, the representations THIS fold's training rows rank highest.
    # Handed the whole pool and no per-fold scores, build_fold_hybrid ranks by
    # Blueprint.rep_scores instead (_shared.py:5270-5300), a score over every
    # molecule, this fold's held-out ones included. On herg_ki that built a
    # different hybrid from the one the search scored (n02: rf 0.6152 RMSE
    # against the search's 0.6058 on the same folds). So the fold's own list
    # from the recipe's allocation_per_fold is passed; with no such record the
    # old behaviour stands and the source says so.
    shared['hybrid_fold_reps'] = {'reps': None, 'source': 'fold not given'}
    if fold_idx is not None and n_folds is not None:
        _keep, _keep_src = search_fold_reps(bp, fold_idx, n_folds,
                                            list(train_reps_for_bp))
        if _keep is not None:
            train_reps_for_bp = {r: train_reps_for_bp[r]
                                 for r in train_reps_for_bp if r in _keep}
        shared['hybrid_fold_reps'] = {
            'reps': sorted(_keep) if _keep is not None else None,
            'source': _keep_src}

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
                       imp_method, base_groups, shared=None,
                       shared_by_space=None):
    """Build every condition for one importance method.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.
        imp_method: Importance method name for weighting ablation.
        base_groups: {rank: {representation: [models]}} -- each downstream
            model's top-2 single representations, grouped so a
            representation several models rank k-th is built once.
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

    Returns four dicts.
      {condition_name: X_standardized}   the spaces to score. A base
            condition has one entry per representation at that rank, named
            '<condition>|<representation>' ('base_1_unweighted|ecfp4').
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
      {condition_name: [models]}         for each base entry, the models whose
            rows are fitted on it. A condition absent here is open to all.
    """
    conditions = {}
    zero_importance = {}
    duplicate = {}
    models_for = {}

    # {space name: the _build_fold_shared output for that space's recipe}.
    # One entry until 2026-09-19, when the pool's second search family got its
    # own space. `shared=` alone still works and means the one space 'hybrid'.
    if shared_by_space is None:
        if shared is None:
            shared = _build_fold_shared(train_reps, y_train, augmentations, bp,
                                        None)
        shared_by_space = {'hybrid': shared}

    # Conditions 1 and 2, once per (rank, representation). The base matrix is
    # scaled on this fold's training rows here.
    _n_imp_base_ok = 0
    for _k, _groups in sorted((base_groups or {}).items()):
        for _rn, _ms in _groups.items():
            X_base = train_reps[_rn]
            X_base_std = StandardScaler().fit_transform(X_base)
            # Condition 1: base unweighted
            _c1 = f'base_{_k}_unweighted|{_rn}'
            conditions[_c1] = X_base_std
            models_for[_c1] = list(_ms)

            # This representation's importance (imp_method-dependent)
            try:
                imp_base = compute_feature_importance(
                    {_rn: X_base}, y_train, method=imp_method
                )[_rn]
            except Exception as e:
                print(f"      {imp_method} importance failed on {_rn}: {e}")
                imp_base = None

            # Condition 2: base importance-weighted (standardize-first, then
            # weight).
            if imp_base is not None:
                _n_imp_base_ok += 1
                _c2 = f'base_{_k}_imp_weighted|{_rn}'
                conditions[_c2] = _weight_features(X_base_std, imp_base)
                models_for[_c2] = list(_ms)
                if _importance_is_all_zero(imp_base):
                    zero_importance[f'base_{_k}_imp_weighted'] = True
    # Condition 4 below used to require the one base representation's
    # importance to have succeeded. With several, it requires one of them to
    # have, or none to have been asked for.
    imp_base = (True if (_n_imp_base_ok or not base_groups) else None)

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
            # dict.get evaluates its default even when the key is there, so
            # the weighting is looked up first and only built if condition 4
            # was not added -- which happens when the base representation's
            # own importance failed and condition 4 was skipped.
            if f'{_space}_imp_base' in conditions:
                conditions[f'{_space}_imp_own'] = conditions[
                    f'{_space}_imp_base']
            else:
                conditions[f'{_space}_imp_own'] = _weight_features(
                    X_hyb_std, imp_hybrid_via_blueprint)
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

    return conditions, zero_importance, duplicate, models_for


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes.

    Same quantity measured several times, so the mean is the reading. The two
    counts are averaged and written beside the fractions, so no ratio appears
    without what it counts.
    """
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_auroc',
            'detection_n_corrupted', 'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None
                and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def _neighbours_without_self(X, k):
    """Each molecule's k nearest OTHER molecules, the molecule removed by index.

    The estimators' own search (noise_estimation._knn_indices without
    exclude_groups) asks for k + 1 neighbours and drops column 0, assuming
    the molecule is its own first neighbour. When two molecules have the same
    features they tie at distance 0 and come back in either order, so about
    half of them keep themselves as a neighbour. Their own noisy label then
    enters their neighbour mean, and they look less suspect than they are.

    That hits the weighted conditions harder than the unweighted ones,
    because a weighted space drops every feature with zero importance and
    more molecules then share a vector. Measured on herg_ki ECFP4, one
    3-fold training split of 943 molecules, clean labels: 32 molecules share
    a vector in all 2048 bits, and 90 share one in the 171 bits lightgbm_gain
    scores above zero.

    Same query as before (k + 1 neighbours, the same distance), so every
    molecule whose own index was in column 0 gets exactly the list it had.
    A molecule whose own index is elsewhere loses itself instead of its first
    neighbour; one whose own index was not returned at all (more than k + 1
    exact ties) loses the farthest. Same rule as simopt's bc394a6 and c19's
    cbcd07e.
    """
    from sklearn.neighbors import NearestNeighbors
    from kirby.similarity.distance_config import (
        resolve as _resolve_metric, sklearn_kwargs as _sk_kwargs)
    X = np.asarray(X)
    n = len(X)
    k = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1,
                          **_sk_kwargs(_resolve_metric(None)))
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    is_self = idx == np.arange(n)[:, None]
    absent = ~is_self.any(axis=1)
    is_self[absent, -1] = True
    return idx[~is_self].reshape(n, k)


def _run_condition(X_space, y_noisy, X_eval_train, X_eval_val, y_val,
                    model_factories, metric_fn, is_clf=False,
                    corrupted_mask=None, noise_condition=None,
                    eval_by_model=None, predict_mode='predict'):
    """Run the 2 estimators of ESTIMATOR_FACTORIES × 3 k values on one condition.

    eval_by_model: {model key: (X_train, X_val)} -- the features each entry of
    model_factories is fitted and scored on. run() passes one pair per
    evaluation unit ('<model>@base_<k>'). None fits every entry on
    X_eval_train/X_eval_val.

    Fixed remediation: REPAIR_WEIGHT_METHOD at REPAIR_STRENGTH ('inverse' at
    3.0, a 4-to-1 weight ratio from least to most suspect).
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

    # One neighbour list per k, shared by both estimators, with each molecule
    # itself removed BY INDEX. See _neighbours_without_self.
    _lists = {}
    for est_name, est_cls in ESTIMATOR_FACTORIES.items():
        for k in K_VALUES:
            est_key = f'{est_name}_k{k}'
            try:
                if k not in _lists:
                    _lists[k] = _neighbours_without_self(X_space, k)
                est = est_cls(k=k, metric=None, is_clf=is_clf,
                              precomputed_neighbors=_lists[k])
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
                    _xtr, _xva = (eval_by_model[mn] if eval_by_model is not None
                                  else (X_eval_train, X_eval_val))
                    model.fit(_xtr, y_noisy, sample_weight=weights)
                    # Positive-class probability under auc, as the clean and
                    # noisy references below are scored (predict_mode). Hard
                    # labels under auc until 2026-09-28.
                    y_pred = _nc._predict(model, _xva, predict_mode)
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
    """n12_importance_weighted_distance across every condition this dataset can be given.

    The condition is an axis this experiment sweeps, not something a caller
    runs it once per. The sweep, the per-condition checkpointing, the resume
    and the stride slice are noise_helpers.sweep_conditions.

    Its headline is a downstream score, which is defined under every condition,
    so it sweeps the whole list rather than only the ones where a corrupted
    SUBSET exists. It also records detection, and detection_scores already
    returns None where that is undefined.
    """
    return sweep_conditions(
        _run_one_condition, 'n12_importance_weighted_distance',
        data, config, output_dir, dataset_name,
        default_conditions=conditions_for_task(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
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
    # 'proba' under auc, 'predict' otherwise. Every fit this file scores goes
    # through it: the clean and noisy references and every repaired fit.
    _predict_mode = _nc.predict_mode_for(metric_name)

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

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
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

    # ── Each downstream model's top-2 single representations ──
    # Ranked by that model's own score in the stored search fetched above (the
    # rf one: its baselines carry a score under every downstream model it
    # scored). A `base_reps` kwarg ({model: [reps]}) overrides the ranking, for
    # a smoke or a direct caller. A model with none has no evaluation features
    # and is left out, with the reason in 'base_reps_missing'.
    _base_override = kwargs.get('base_reps')
    if _base_override is not None:
        base_by_model = {m: list(_base_override[m])[:NOISE_BASE_REP_COUNT]
                         for m in downstream_models if _base_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: search for m in downstream_models}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    eval_models = [m for m in downstream_models if m in base_by_model]
    # Evaluation units: one per (model, rank). '<model>@base_<k>'.
    eval_units = {f'{m}@base_{k}': (m, k, rn)
                  for m in eval_models
                  for k, rn in enumerate(base_by_model[m], start=1)}
    unit_factories = {u: model_factories[m]
                      for u, (m, _k, _rn) in eval_units.items()}
    base_groups = {_k: base_rank_groups(base_by_model, _k, eval_models)
                   for _k in range(1, NOISE_BASE_REP_COUNT + 1)}
    base_groups = {_k: _g for _k, _g in base_groups.items() if _g}
    for _u, (_m, _k, _rn) in eval_units.items():
        print(f"  evaluation unit {_u}: {_m} fitted on {_rn}")
    for _m, _why in base_missing.items():
        print(f"  {_m}: no single representation, left out -- {_why}")
    if not eval_units:
        print("  no downstream model has a single representation to be "
              "fitted on, so nothing can be scored")
        return {'error': 'no_single_representation', 'metric': metric_name,
                'base_reps_by_model': base_by_model,
                'base_reps_missing': base_missing}

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        # Each downstream model's top 2 single representations, from its own
        # score in the stored search. See the module docstring.
        'base_reps_by_model': base_by_model,
        'base_reps_missing': base_missing,
        'base_reps_search_model': 'rf',
        'evaluation_units': {u: {'model': m, 'rank': k, 'representation': rn}
                             for u, (m, k, rn) in eval_units.items()},
        'evaluation_unit_key': (
            "'<model>@base_<k>': that model fitted on its rank-k single "
            "representation. Every *_by_model field and the model half of "
            "every estimator key is keyed by it."),
        'base_conditions_by_rank': {
            f'base_{_k}': {rn: ms for rn, ms in _g.items()}
            for _k, _g in base_groups.items()},
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
        # {importance method: {condition: how many times its scoring was
        # COPIED from its twin rather than run}}. A duplicate condition is the
        # same matrix object as the one it copies, so every estimator and every
        # model fitted on it returns the twin's number. Scoring it again cost 6
        # neighbour searches and 6 fits per model in the roster, per level and
        # per fold. Copying is bit-identical and is what the numbers already
        # were. Added 2026-09-21.
        'duplicate_conditions_copied': {},
        # {importance method: {condition: {'n_scorings_skipped': n}}}. The
        # conditions whose matrix does not depend on the importance method
        # (every '_unweighted' one and each hybrid space's '_imp_base') are
        # scored once per (level, fold, draw) and copied into the second
        # importance method's half. Bit-identical: same matrix, same labels,
        # deterministic estimators, seeded models. Added 2026-09-28.
        'imp_method_independent_copied': {},
        # {importance method: {condition: [widths seen]}}. How many features
        # the distance was computed over in each condition, which is not the
        # same number for every condition and was never recorded. A weighted
        # space carries only the features the scorer gave importance above
        # zero; the rest are zero in every molecule and are dropped, which changes
        # no neighbour under any of the four distances the suite sweeps. A list
        # rather than one number because the scorer is refitted per fold and
        # can keep a different count each time.
        'features_in_space': {},
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

    # ── The similarity axis, added 2026-09-25 (plan sections 1-2) ──────────
    # n12 already asks which distance finds a molecule's neighbours, through
    # its own conditions above, which are unchanged: they weight the features
    # by n12's own power-then-normalise rule (_weight_features) and every key
    # they fill is what it was. The sweep below is the suite-wide axis on top:
    # each estimator of ESTIMATOR_FACTORIES reads its neighbours from every
    # similarity setting (similarity_space, weights through
    # kirby.similarity.preprocessing.fold_weights) at every k in SWEPT_KS, and
    # repairs with the same weighting as the conditions above
    # (REPAIR_WEIGHT_METHOD at REPAIR_STRENGTH). The model input is each
    # evaluation unit's own single representation, as above. One D-versus-D'
    # row per (estimator, setting, k, model): RF, SVM and the unit's model.
    #
    # TODO(similarity-sweep 2026-09-25): the plan (section 1) says n12's own
    # weighting copy is to be replaced by fold_weights. That changes every
    # number the conditions above report (fold_weights normalises before the
    # power, _weight_features after -- see its docstring), so it is not done
    # here; the swept settings already go through fold_weights. Owner to say
    # whether the legacy conditions move.
    sim_settings = sweep_settings()
    sink = RowSink(output_dir, 'n12_importance_weighted_distance',
                   dataset_name, kwargs.get('noise_condition'))
    # TODO(similarity-sweep 2026-09-25): the hybrid and hybrid_similarity
    # similarity spaces use the one recipe this file reads, rf's
    # (get_best_search_for_downstream(downstream_model='rf') above), for every
    # evaluation unit's model. The plan says each model's own prediction-built
    # hybrid; reading one recipe per model is a pool read per model. Owner to
    # say whether rf's recipe stands in for every model here.
    sim_recipes = {r['space']: (r['blueprint'], r['aug_condition'])
                   for r in hybrid_recipes}
    sim_detection = {}
    sweep_ctx = sweep_context(
        experiment='n12_importance_weighted_distance',
        dataset_name=dataset_name, sink=sink, settings=sim_settings,
        metric=(metric_name, metric_fn, lower_is_better), task_type=task_type,
        condition=noise_condition, level=None, data=data,
        factories=model_factories)
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())}; rows -> {sink.path}")
    _dup_copied = results['duplicate_conditions_copied']
    features_in_space = results['features_in_space']

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        # One list per evaluation unit for both references. Recovery is a
        # fraction of THIS unit's clean-to-noisy gap -- one model on one of its
        # two single representations -- so the references cannot be shared.
        clean_scores = {u: [] for u in eval_units}
        noisy_scores = {u: [] for u in eval_units}
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

            # The evaluation features, one scaled pair per single
            # representation any unit is fitted on, shared by those units.
            # Cached 2026-09-25 (noise_cache.cached_scaled): the same
            # StandardScaler arithmetic, stored once per (representation
            # content, fold) across every level, draw and condition.
            _eval_by_rep = {}
            for _u, (_m, _k, _rn) in eval_units.items():
                if _rn not in _eval_by_rep:
                    _eval_by_rep[_rn] = _nc.cached_scaled(
                        dataset_name, reps[_rn], train_idx, val_idx)
            eval_by_unit = {_u: _eval_by_rep[_rn]
                            for _u, (_m, _k, _rn) in eval_units.items()}

            # Clean baseline
            # The clean fit has no noise in it, so it is the same on all three
            # draws of a fold. Score it on the first draw only, or clean_scores
            # carries three copies of every fold.
            # Fitted through noise_cache.cached_fit_predict since 2026-09-25:
            # the same fit and the same model.predict, stored once per
            # (features, labels, model) and read back by the D-versus-D' rows.
            if noise_seed == _seeds[0]:
                for _u in eval_units:
                    _xtr, _xva = eval_by_unit[_u]
                    clean_scores[_u].append(float(metric_fn(
                        y_val, _nc.cached_fit_predict(
                            dataset_name, eval_units[_u][0],
                            unit_factories[_u], _xtr, y_train_clean, _xva,
                            predict_mode=_predict_mode))))

            # Inject noise (training only)
            _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                   noise_seed=noise_seed, is_clf=is_clf)
            y_noisy, _cmask, _prov = inject_train_labels(
                labels, _spec, train_idx=train_idx, is_clf=is_clf,
                groups=noise_groups, dataset_name=dataset_name)
            noise_provenance.setdefault(_spec.spec_id, _prov)

            # Noisy baseline, through the same cache.
            for _u in eval_units:
                _xtr, _xva = eval_by_unit[_u]
                noisy_scores[_u].append(float(metric_fn(
                    y_val, _nc.cached_fit_predict(
                        dataset_name, eval_units[_u][0], unit_factories[_u],
                        _xtr, y_noisy, _xva, predict_mode=_predict_mode))))

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
                    None,
                    groups=(groups[train_idx] if groups is not None else None),
                    val_reps=val_reps_fold, augmentations_val=val_augs,
                    # The hybrid build's own accept-or-reject step keeps the
                    # first model in the roster. It decides which features go
                    # in, not which model is finally scored, and making it
                    # depend on the model would give every model a different
                    # feature space to be compared in.
                    model=model_factories[downstream_models[0]](),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    fold_idx=fold_idx, n_folds=n_folds)
                _hfr = fold_shared_by_space[_r['space']]['hybrid_fold_reps']
                results.setdefault('hybrid_fold_reps', {}).setdefault(
                    _r['space'], {})[str(fold_idx)] = _hfr
                print(f"    fold {fold_idx} {_r['space']}: built on "
                      f"{_hfr['reps'] if _hfr['reps'] is not None else 'the pool'}"
                      f" ({_hfr['source']})")
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

            # Scores of the conditions whose matrix does not depend on the
            # importance method, keyed by entry name, for this (level, fold,
            # draw). Every '_unweighted' condition and every hybrid space's
            # '_imp_base' condition (weighted by the RECIPE's importance
            # method) is the same matrix under both halves of the sweep, and
            # the estimators and the seeded models return the same numbers on
            # it. Until 2026-09-28 each was scored again on the second half.
            _imp_free_scored = {}
            for imp_method in IMPORTANCE_METHODS:
                if imp_method not in grid_acc:
                    grid_acc[imp_method] = {}

                (conditions, zero_importance, duplicate_conds,
                 models_for) = _build_conditions(
                    train_reps_fold, y_noisy, train_augs, bp,
                    imp_method, base_groups,
                    shared_by_space=fold_shared_by_space,
                )
                # How many features each condition's distance was actually
                # computed over. It is not the same for every condition and it
                # was never written down: a weighted space drops the features
                # the scorer gave no importance, because they are zero in every
                # molecule and add nothing to any of the four distances the
                # suite sweeps. On flab_thermo_tresanco lightgbm_gain scored 181 of
                # onehot's 5600 features above zero and treeshap 1086, so the
                # two scorers were compared over spaces of very different width
                # and the result file said only 'base_imp_weighted' for both.
                for _cond_name, _X in conditions.items():
                    # A base entry '<condition>|<representation>' is recorded
                    # under its condition.
                    _fw = (features_in_space.setdefault(imp_method, {})
                           .setdefault(_cond_name.split('|', 1)[0], []))
                    _n = int(np.asarray(_X).shape[1])
                    if _n not in _fw:
                        _fw.append(_n)
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

                # What each condition returned on THIS (level, fold,
                # importance method), so a condition that is a second copy of
                # another one is not scored twice. Cleared every build.
                _scored_here = {}
                for _entry_name, X_space in conditions.items():
                    # A base entry '<condition>|<representation>' is scored
                    # for the models that rank that representation there, and
                    # its rows join the condition's. The models of two entries
                    # at one rank never overlap, so their keys never collide.
                    cond_name = _entry_name.split('|', 1)[0]
                    _open = models_for.get(_entry_name)
                    _cond_units = {u: f for u, f in unit_factories.items()
                                   if _open is None
                                   or eval_units[u][0] in _open}
                    if cond_name not in grid_acc[imp_method]:
                        grid_acc[imp_method][cond_name] = {}

                    # ⛔ A DUPLICATE CONDITION IS COPIED, NOT SCORED AGAIN.
                    # _build_conditions hands back the SAME matrix object for
                    # both, so every estimator and every model fitted on it
                    # would return the number the twin already returned.
                    # Scoring it a second time costs 6 neighbour searches and
                    # 6 fits per model in the roster, per level and per fold,
                    # and buys a column of numbers that are already in the
                    # result file under another name.
                    _twin = (duplicate_conds.get(cond_name) or {}).get(
                        'duplicate_of')
                    if _twin is not None and _twin in _scored_here:
                        cond_results = copy.deepcopy(_scored_here[_twin])
                        _dup_copied.setdefault(imp_method, {}).setdefault(
                            cond_name, {'copied_from': _twin,
                                        'n_scorings_skipped': 0,
                                        'sigmas': []})
                        _dc = _dup_copied[imp_method][cond_name]
                        _dc['n_scorings_skipped'] += 1
                        if sigma not in _dc['sigmas']:
                            _dc['sigmas'].append(sigma)
                    elif (_is_imp_method_free(cond_name)
                          and _entry_name in _imp_free_scored):
                        cond_results = copy.deepcopy(
                            _imp_free_scored[_entry_name])
                        _ic = (results['imp_method_independent_copied']
                               .setdefault(imp_method, {})
                               .setdefault(cond_name,
                                           {'n_scorings_skipped': 0}))
                        _ic['n_scorings_skipped'] += 1
                    else:
                        cond_results = _run_condition(
                            X_space, y_noisy, None, None,
                            y_val, _cond_units, metric_fn, is_clf=is_clf,
                            corrupted_mask=_cmask,
                            noise_condition=noise_condition,
                            eval_by_model=eval_by_unit,
                            predict_mode=_predict_mode,
                        )
                        if _is_imp_method_free(cond_name):
                            _imp_free_scored[_entry_name] = copy.deepcopy(
                                cond_results)
                    _scored_here[cond_name] = cond_results

                    for ek, score in cond_results.items():
                        if ek == '_detection':
                            # One reading per (fold, draw, estimator, k). Kept as
                            # a list per estimator and averaged below, so a later
                            # draw does not overwrite an earlier one.
                            _dstore = grid_acc[imp_method][cond_name].setdefault(
                                '_detection_per_pass', {})
                            for _ek2, _row in score.items():
                                # Base entries: one detection reading per
                                # representation at this rank, kept apart.
                                _dk = (_ek2 if _entry_name == cond_name
                                       else f"{_ek2}|{_entry_name.split('|', 1)[1]}")
                                _dstore.setdefault(_dk, []).append(_row)
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

            # ── The similarity sweep for this (level, fold, draw). ──
            # One pass per evaluation unit: the model input is that unit's
            # single representation, the similarity spaces are that unit's
            # model's (its top 2, the two hybrids of the pool's recipes).
            # Neighbour lists without group exclusion, as ESTIMATOR_FACTORIES
            # are built above (no exclude_groups). MODEL_INPUT is the unit's
            # own model input (noise_cache.cached_scaled), which leaves an
            # all-binary representation unscaled. The base_<k>_unweighted
            # conditions z-score every representation (_build_conditions), so
            # on ECFP4 the two neighbour lists are NOT the same list.
            _g_tr = (groups[train_idx] if groups is not None else None)
            for _u, (_m, _k_rank, _rn) in eval_units.items():
                _xtr, _xva = eval_by_unit[_u]

                def _dprimes(setting, lists, k, _xtr=_xtr):
                    for est_name in ESTIMATOR_FACTORIES:
                        sc = detector_scores_with(
                            dataset_name, est_name, _xtr, y_noisy, lists, k,
                            is_clf=is_clf)
                        if sc is None:
                            yield est_name, None, None
                            continue
                        if _cmask is not None:
                            _det = detection_scores(_cmask, sc, noise_condition)
                            if _det is not None:
                                (sim_detection.setdefault(str(sigma), {})
                                 .setdefault(est_name, {})
                                 .setdefault(setting.name, {})
                                 .setdefault(str(k), []).append(_det))
                        try:
                            w = weights_from_scores(
                                sc, strength=REPAIR_STRENGTH,
                                method=REPAIR_WEIGHT_METHOD)
                        except Exception as exc:
                            print(f"      {est_name} weights failed: {exc}")
                            yield est_name, None, None
                            continue
                        yield (est_name,
                               f'{REPAIR_WEIGHT_METHOD}_{REPAIR_STRENGTH}',
                               cleaned_from_weights(_xtr, y_noisy, w))
                similarity_sweep(
                    sweep_ctx, model_name=_m, model_input=f'{_u}:{_rn}',
                    X_tr=_xtr,
                    X_va=_xva, y_tr_noisy=y_noisy, y_tr_clean=y_train_clean,
                    y_val=y_val, train_idx=train_idx, val_idx=val_idx,
                    groups_tr=_g_tr, fold=fold_idx - 1, draw=noise_seed,  # rows count folds from 0
                    recipes=sim_recipes, base_reps=base_by_model.get(_m, []),
                    make_dprimes=_dprimes, level=sigma)

        # Aggregate
        clean_by_model = {m: float(np.mean(v)) for m, v in clean_scores.items() if v}
        noisy_by_model = {m: float(np.mean(v)) for m, v in noisy_scores.items() if v}
        _models_scored = [u for u in eval_units
                          if u in clean_by_model and u in noisy_by_model]
        if not _models_scored:
            raise RuntimeError(
                f"n12: no evaluation unit produced both a clean and a noisy "
                f"reference score at level {sigma}. Units: {list(eval_units)}.")
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
                        # gap_by_model is sign-normalised: positive means the
                        # noise cost this unit something. When the noise HELPED
                        # (negative gap), dividing by it turns the ratio inside
                        # out -- a repair that moved the score back toward the
                        # clean fit, i.e. made it worse, read as positive
                        # recovery. n07 refuses that case (its _hurt test);
                        # n12 only tested the size of the gap. Fixed
                        # 2026-09-28.
                        _hurt = bool(np.isfinite(gap_by_model[_mn])
                                     and gap_by_model[_mn] > 0)
                        if (_c is not None and _nz is not None
                                and np.isfinite(_c) and np.isfinite(_nz)
                                and not _gap_too_small and _hurt):
                            recovery = float((denoised - _nz) / (_c - _nz) * 100)
                            recovery_note = None
                        else:
                            recovery = float('nan')
                            if not (_c is not None and _nz is not None
                                    and np.isfinite(_c) and np.isfinite(_nz)):
                                recovery_note = 'not computed: a score was not finite'
                            elif not _hurt:
                                recovery_note = (
                                    f'not computed: the noise did not degrade '
                                    f'this unit (clean {_c:.6g}, noisy '
                                    f'{_nz:.6g}, lower_is_better='
                                    f'{lower_is_better})')
                            else:
                                recovery_note = (
                                    f'not computed: the clean-to-noisy gap '
                                    f'({gap_by_model[_mn]:.6g}) is under '
                                    f'{GAP_MIN_FRAC:.0%} of the clean score '
                                    f'({_c:.6g})')
                        cond_agg[ek] = {
                            'denoised': denoised,
                            'recovery': recovery,
                            'recovery_note': recovery_note,
                            # Named, not only encoded in the key. 'model'
                            # is the evaluation unit ('rf@base_1'); the two
                            # fields after it split it.
                            'model': _mn,
                            'downstream_model': eval_units[_mn][0],
                            'fitted_on': f'base_{eval_units[_mn][1]}',
                            'fitted_on_representation': eval_units[_mn][2],
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

    # The similarity axis, 2026-09-25. Every key above is n12's own
    # conditions, unchanged. The swept numbers are the D-versus-D' rows in
    # results['similarity']['rows_file'] and, for detection,
    # {level: {estimator: {setting: {k: mean detection reading over (fold,
    # draw)}}}}.
    results['similarity'] = similarity_record(
        sim_settings, sink, spaces_unavailable=sweep_ctx['unavailable'],
        repair=f'{REPAIR_WEIGHT_METHOD} at strength {REPAIR_STRENGTH}')
    results['detection_by_similarity'] = {
        _lv: {_e: {_sn: {_k: _mean_detection(_rows)
                         for _k, _rows in _byk.items()}
                   for _sn, _byk in _bys.items()}
              for _e, _bys in _bye.items()}
        for _lv, _bye in sim_detection.items()}

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

        # Ablation answers. The base question is asked once per rank of the
        # single-representation comparison.
        _base_ranks = [f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1)
                       if f'base_{_k}_unweighted' in cond_avg]

        def _weighting_helps_base():
            out = {}
            for _b in _base_ranks:
                _uw = cond_avg.get(f'{_b}_unweighted', float('nan'))
                _w = cond_avg.get(f'{_b}_imp_weighted', float('nan'))
                out[_b] = bool(not np.isnan(_w) and not np.isnan(_uw)
                               and _w > _uw)
            return out

        def _answers_for(hybrid_space):
            """The three ablation answers, read in one hybrid space."""
            uw = cond_avg.get(f'{hybrid_space}_unweighted', float('nan'))
            w_base = cond_avg.get(f'{hybrid_space}_imp_base', float('nan'))
            w_own = cond_avg.get(f'{hybrid_space}_imp_own', float('nan'))
            return {
                'weighting_helps_base': _weighting_helps_base(),
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
            if c.endswith('_unweighted') and not c.startswith('base_')})
        answers_by_space = {hs: _answers_for(hs) for hs in hybrid_spaces_seen}

        # The three top-level flags stay the PREDICTIVE space's, unchanged.
        _pred = answers_by_space.get('hybrid', {
            'weighting_helps_base': _weighting_helps_base(),
            'weighting_helps_hybrid': False,
            'hybrid_already_weighted': False})
        weighting_helps_base = _pred['weighting_helps_base']
        weighting_helps_hybrid = _pred['weighting_helps_hybrid']
        hybrid_already_weighted = _pred['hybrid_already_weighted']

        # Best k. Read over the five conditions this file had before
        # 2026-09-19, so the number means what it meant; 'per_k_avg_recovery'
        # and 'best_k' would otherwise have moved because a second hybrid
        # space added three more conditions to the same average.
        # The one base pair is now one pair per rank.
        _ORIGINAL_CONDITIONS = tuple(BASE_CONDITION_NAMES) + (
            'hybrid_unweighted', 'hybrid_imp_base', 'hybrid_imp_own')
        k_recoveries = {}
        k_recoveries_all = {}
        for imp_data in sigma_data.get('per_importance_method', {}).values():
            for _cond_name, cond_data in imp_data.items():
                for ek, rv in cond_data.items():
                    # Parse k from key like 'rf@base_1|neighbor_residual_k10'
                    if '_k' not in ek or ek == 'detection':
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
