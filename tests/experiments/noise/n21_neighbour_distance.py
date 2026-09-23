#!/usr/bin/env python3
"""N21: which distance are the nearest neighbours found in?

Paper: Noise.

THE QUESTION. Several detectors in this suite score a molecule by comparing its
label to the labels of its nearest neighbours. Which distance those neighbours
are found in has never been swept. n12 sweeps one weighting against no
weighting; nothing has put a forest proximity or the hybrid's own feature space
beside them.

FIVE DISTANCES.

  plain                the base representation as it is, z-scored.
  importance-weighted  each feature multiplied by its importance raised to an
                       exponent, then z-scored first. n12's space.
  forest proximity     how often two molecules land in the same leaf of a
                       fitted forest; the distance is one minus that share.
  the hybrid's own     the feature space the prediction-objective recipe
                       built, z-scored.
  the similarity one   the feature space the similarity-objective recipe built,
                       z-scored. Its features were chosen so that the distance
                       predicts each molecule's activity from its five nearest
                       neighbours.

WHAT IS SCORED. `noise_helpers.detection_scores` against the molecules whose
label actually moved. Average precision is the headline: it needs no threshold,
so it does not inherit the defect at NOISE_AUDIT_2026_09_10.md 6.9, where a
filter's cut-off is two standard deviations of the CORRUPTED label column and
so grows with the amount of noise injected. Beside it, the share of the
clean-to-corrupted gap that repairing on each distance puts back -- n12's
recovery figure, computed the same way.

WHICH NOISE CONDITIONS THIS CAN RUN UNDER. Five, and no others:
`outlier_p01`, `outlier_p05`, `outlier_p10`, `grouped_wider`, `censoring`.
Those are `noise_helpers.DETECTION_IS_DEFINED`. Under the other six every
training molecule's label moves, so the list of molecules that moved is the
list of all of them and average precision is 1.0 by construction. n21 raises
rather than run there, because average precision is the headline and under
those six it does not exist. The default is `outlier_p10`, which corrupts about
a tenth of the training molecules.

ONE DEFECT IN n12 THIS SETTLES, AND IT IS NOT FIXED HERE. n12 multiplies each
feature by the square root of its importance and then compares molecules under
a Manhattan distance (audit section 4, n12 defect 1). Under a Euclidean
distance feature f contributes w_f * (x_if - x_jf)^2 to the squared distance
when the multiplier is sqrt(w_f), so its weight in the distance is its
importance. Under a Manhattan distance the same multiplier contributes
sqrt(w_f) * |x_if - x_jf|, so its weight is the square root of its importance
instead. Two ways of scaling, two distances, four cells, and n21 runs all four.
The exponent matched to the metric is 0.5 under euclidean and 1.0 under
manhattan; every cell records `convention_matches_metric` so the matched pair
and the mismatched pair can be read apart. n12 itself is untouched: this
measures which convention is right, and n12 changes after.

WHAT A LEVEL OF 0.0 WOULD GIVE, AND WHY IT IS NOT IN THE DEFAULT GRID. Under
every condition above, level 0.0 corrupts no molecule -- measured here, 0 of
240 under outlier_p10. Average precision is then undefined and
`detection_scores` returns None, and the clean fit and the corrupted fit are
the same number so the recovery figure divides by zero. Both readings are
undefined, which is why it is dropped; `--levels` puts it back. This is n12
defect 2, which n21 does not repeat.

THE HYBRID RECIPE IS CHOSEN ON CLEAN LABELS. It comes from the precomputed pool
and is rebuilt per fold under the corrupted labels through
`noise_helpers.hybrid_for_fold`, which is what n17, n18, n19 and n20 rebuild
with. That is n12's design and carries n12's caveat: the hybrid distance is a
best case, not a leak-free comparison. Recorded in the result file as
`hybrid.blueprint_is_clean_label`.

HOW THE NEIGHBOURS ARE FOUND. One function, for all five distances. Three of
them are feature spaces and go through sklearn's NearestNeighbors under the
named metric. Forest proximity is not a feature space -- it is a pairwise
distance and nothing else -- so it goes in as a square distance matrix. The
scoring arithmetic that turns neighbours into a suspicion score is the same
code for all four, transcribed from `NeighborResidualEstimator` and
`LocalVarianceRatioEstimator` (`src/kirby/utils/noise_estimation.py:236-370`),
because those classes take a feature space and cannot be handed a distance
matrix. `_smoke_n21.py` asserts the transcription reproduces both classes to
the last bit on a feature space, under both metrics and both task types, so the
two copies cannot drift apart unnoticed.

`--distance-metric` does not reach this experiment. Every neighbour call here
names its metric, because the metric is a swept axis.

Usage:
    python n21_neighbour_distance.py --dataset logd --mode full
    python n21_neighbour_distance.py --dataset herg_ki --noise-condition censoring
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
    DETECTION_IS_DEFINED, hybrid_for_fold,
)
from experiments.noise.n07_remediation import _predict_for_metric
from experiments._shared import (
    pick_noise_base_rep, load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split, get_domain_metric,
    get_model_factories, DOWNSTREAM_MODELS_NOISE, build_fold_hybrid,
    get_best_search_for_downstream, make_nearest_neighbors,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
)
from kirby.hybrid import compute_feature_importance
from kirby.utils.noise_estimation import weights_from_scores


# The two neighbour-based detectors n12 runs, so the two experiments can be read
# side by side. activity_cliff_aware was cut on 2026-09-12: it ranks the
# molecules the same way as neighbour_residual at Spearman 0.989 on herg_ki in
# ecfp4.
DETECTORS = ('neighbor_residual', 'local_variance_ratio')

# n12's three, unchanged.
K_VALUES = [5, 10, 20]

# n12's two.
IMPORTANCE_METHODS = ['lightgbm_gain', 'treeshap']

# The two metrics the weighting conventions are run against. Both are in
# distance_config's four; the other two are left out on purpose. cosine is
# scale-free along each ray, so multiplying a feature by a positive number
# changes the angle between two molecules but the convention argument above is
# about which power of the importance becomes the weight in a sum of per-feature
# terms, and cosine has no such sum. minkowski p=0.5 is defined on 2 of the 14
# representation blocks f18 measured. Pass --metrics to run either anyway.
DISTANCE_METRICS_SWEPT = ('euclidean', 'manhattan')

# The two scalings. 0.5 is what n12 applies. 1.0 is the one that gives a feature
# a weight equal to its importance under a Manhattan distance.
IMPORTANCE_EXPONENTS = (0.5, 1.0)

# Which exponent gives a feature a weight equal to its importance, per metric.
# Not a preference -- it is what the algebra says, and the sweep runs both
# against both so the claim is measured rather than assumed.
EXPONENT_MATCHED_TO_METRIC = {'euclidean': 0.5, 'manhattan': 1.0}

# The condition. outlier_p10 corrupts about a tenth of the training molecules,
# so a corrupted subset exists and average precision means something. The other
# four in DETECTION_IS_DEFINED are reachable with --noise-condition.
NOISE_CONDITION = 'outlier_p10'

# Draws of the corruption per fold, averaged. Matches n07, n09, n11, n12, n13,
# n15, n17, n19 and n20.
NOISE_SEEDS = (0, 1, 2)

# The forest whose leaves the proximity distance is read off. Fitted on the
# fold's TRAINING molecules only, on the z-scored base representation, against
# the CORRUPTED training labels -- a practitioner holds no others.
#
# min_samples_leaf is 5 rather than sklearn's 1 because a fully grown forest
# puts most training molecules alone in a leaf, and then two molecules share a
# leaf mainly when the bootstrap left one of them out. The proximity would be
# measuring the bootstrap.
#
# 200 trees, MEASURED 2026-09-21 and it replaces 100. Two molecules' distance
# is one minus the share of trees they land in the same leaf of, so the number
# of trees is the number of votes that share is taken over: at 100 the distance
# can only take 101 distinct values. `_loop_n21_forest.py` swept 100, 200 and
# 300 trees against 1, 5, 20 and 50 molecules a leaf on flab_thermo_tresanco in
# onehot, under outlier_p10 at level 0.5, 3 of its 6 passes in, one pass being
# one (fold, corruption draw) with 35 to 44 of the 448 training molecules moved.
# Against 100 trees at this leaf size, 200 trees gained +0.0082 detection
# average precision on 3 of the 3 passes, per pass +0.0049, +0.0091 and +0.0107,
# spread over passes 0.0030; 300 trees gained +0.0081 on 3 of 3, which is the
# same number, so the extra 100 trees buy nothing and are not taken. Precision
# at the number of molecules that really moved agrees in direction, +0.0112 on
# 2 of the 3 passes.
#
# WHAT IT COSTS: the forest fit is most of the cost of a proximity, and 200
# trees is twice the fit of 100. Measured on fold 0 of that dataset at 5
# molecules a leaf, 448 training molecules over 5600 features: 126 seconds for
# 300 trees, so about 84 for 200 against 42 for 100.
#
# THE LEAF SIZE DID NOT MOVE, and the reason is that the two readings disagree
# about it. One molecule a leaf leads on average precision, +0.0102 over 5 on 2
# of the 3 passes, and LOSES on precision at the moved count, -0.0160 on 1 of
# 3. 50 molecules a leaf is worse on both. 5 stays until the last three passes
# land.
FOREST_TREES = 200
FOREST_MIN_SAMPLES_LEAF = 5
FOREST_SEED = 42

# Rows of the leaf matrix compared at a time when the proximity distance is
# built. The intermediate is chunk x n x trees booleans, so 125 rows of a
# 4000-molecule fold at the 200 trees above is 100 MB. It was 250 rows while
# the forest had 100 trees, which was the same 100 MB; halving it keeps the
# intermediate the size it was and changes no number, because the chunk only
# says how many rows of the same matrix are filled at once.
_PROXIMITY_CHUNK = 125

# The repair. n12's setting, so the recovery figures are comparable.
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
WEIGHT_STRENGTH = 3.0
WEIGHT_METHOD = 'inverse'

# The smallest clean-to-corrupted gap that a recovery percentage may be divided
# by, as a fraction of the clean score. Same floor and same value as n06, n07
# (n07_remediation.GAP_MIN_FRAC) and n11.
GAP_MIN_FRAC = 0.02


# ---------------------------------------------------------------------------
# The five distances
# ---------------------------------------------------------------------------

def importance_is_all_zero(importance):
    """True when there is no importance to weight by.

    The weighted space is then the plain space -- the same geometry, the same
    neighbours, the same suspicion scores -- so its numbers are a duplicate of
    the plain row rather than a measurement of the weighting. The caller counts
    how many builds went that way and writes it beside the space.
    """
    imp = np.maximum(np.asarray(importance, dtype=float), 0)
    return float(imp.sum()) < 1e-10


def weight_features(X_standardized, importance, exponent):
    """Multiply already-z-scored features by importance ** exponent.

    THE CALLER MUST PASS A STANDARDIZED X. Weighting first and standardizing
    after is a mathematical no-op -- a per-column StandardScaler exactly cancels
    a per-column scalar multiply -- and that is how the whole weighting ablation
    silently produced identical geometry in an earlier version of n12.

    `exponent` 0.5 reproduces n12's `_weight_features` exactly, including the
    normalisation. The weights are scaled to average 1 across features, so a
    z-scored column is stretched or squeezed around its own unit variance
    instead of every column being crushed toward 1/width. That normalisation is
    a single positive multiplier applied to the whole matrix, so it moves no
    molecule's neighbours; it is kept only so this space IS n12's space at
    exponent 0.5 and the smoke can assert it.

    An all-zero importance returns X_standardized untouched.
    """
    imp = np.maximum(np.asarray(importance, dtype=float), 0)
    if importance_is_all_zero(imp):
        return X_standardized
    w = imp ** float(exponent)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return X_standardized
    w = w * (len(w) / total)
    return X_standardized * w


def forest_leaf_ids(X_train, y_train, is_clf, n_trees=FOREST_TREES,
                    min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
                    seed=FOREST_SEED):
    """One leaf id per tree per training molecule.

    `rf.apply(X)` is the call. It appears nowhere in src/kirby/utils/, so this
    is the first time a forest proximity is computed in this repository.

    The forest sees the fold's training molecules and their corrupted labels,
    and nothing else. Fitting it on the clean labels would hand the distance the
    answer the experiment is asking it to find.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = RandomForestClassifier if is_clf else RandomForestRegressor
    # 2026-09-20: was n_jobs=-1 while every other estimator in the run honours
    # the single-thread policy (KIRBY_RF_JOBS, KIRBY_LGBM_THREADS,
    # OMP_NUM_THREADS). Same shape as the defect commit 9cb3f43 fixed.
    rf = cls(n_estimators=int(n_trees), min_samples_leaf=int(min_samples_leaf),
             random_state=int(seed),
             n_jobs=int(os.environ.get('KIRBY_RF_JOBS', '1') or 1))
    rf.fit(np.asarray(X_train, dtype=np.float64), y_train)
    return np.asarray(rf.apply(X_train), dtype=np.int32)


def forest_proximity_distance(leaf_ids, chunk=_PROXIMITY_CHUNK):
    """One minus the share of trees in which two molecules land in one leaf.

    `leaf_ids` is one row per molecule, one column per tree. The proximity of
    two molecules is the number of columns where their ids match, divided by the
    number of trees. The distance is one minus that, so it runs from 0 for two
    molecules that are always together to 1 for two that never are.

    Returned as a square matrix, molecules by molecules, because a proximity is
    a pairwise quantity and there is no feature space it comes from.
    """
    leaves = np.asarray(leaf_ids)
    n, n_trees = leaves.shape
    D = np.empty((n, n), dtype=np.float32)
    step = max(1, int(chunk))
    for start in range(0, n, step):
        stop = min(n, start + step)
        same = (leaves[start:stop, None, :] == leaves[None, :, :])
        D[start:stop] = 1.0 - (same.sum(axis=2) / float(n_trees))
    return D


# ---------------------------------------------------------------------------
# Neighbours, and the suspicion score built from them
# ---------------------------------------------------------------------------

def neighbours_in_space(X, k, metric):
    """The k nearest OTHER molecules to each molecule, in a feature space.

    Line for line what `noise_estimation._knn_indices` does when no
    `exclude_groups` is passed: ask sklearn for k+1 neighbours, then drop the
    first column, which is the molecule itself at distance zero. n12 passes no
    exclude_groups either, so the two experiments find neighbours the same way.
    """
    X = np.asarray(X)
    n = len(X)
    n_query = int(min(k + 1, n))
    nn = make_nearest_neighbors(n_neighbors=n_query, metric=metric)
    nn.fit(X)
    _d, idx = nn.kneighbors(X)
    return idx[:, 1:k + 1]


def neighbours_in_distance_matrix(D, k):
    """The k nearest OTHER molecules to each molecule, from a square distance.

    The diagonal is forced below every off-diagonal entry before sorting, so the
    molecule itself is always the first column and dropping that column always
    drops the molecule itself. Without that, a second molecule at distance zero
    -- which a forest proximity produces whenever two molecules share every leaf
    -- could take the first column and the molecule would stay in its own
    neighbour list.

    Only the nearest k+1 of each row are sorted, not the whole row. A full
    argsort of a square matrix over 4000 training molecules allocates 130 MB of
    indices per call and this is called once per k value.
    """
    D = np.asarray(D)
    n = D.shape[0]
    diagonal = np.diag(D).copy()
    np.fill_diagonal(D, -np.inf)
    try:
        k1 = int(min(k + 1, n))
        rows = np.arange(n)[:, None]
        if k1 >= n:
            candidates = np.argsort(D, axis=1, kind='stable')
        else:
            part = np.argpartition(D, k1 - 1, axis=1)[:, :k1]
            order = np.argsort(D[rows, part], axis=1, kind='stable')
            candidates = part[rows, order]
        return candidates[:, 1:k + 1]
    finally:
        # The caller's matrix is handed back as it arrived, diagonal and all.
        np.fill_diagonal(D, diagonal)


def suspicion_from_neighbours(neighbour_indices, y, detector, is_clf):
    """How wrong each molecule's label looks, given its neighbours.

    TRANSCRIBED, NOT CALLED. `NeighborResidualEstimator.estimate` and
    `LocalVarianceRatioEstimator.estimate`
    (src/kirby/utils/noise_estimation.py:236-370) each find their own neighbours
    from a feature space, and a forest proximity is not a feature space. So the
    arithmetic below is a copy of theirs that takes the neighbours already
    found. `_smoke_n21.py` asserts it reproduces both classes exactly on a
    feature space, under euclidean and manhattan and under both task types.

    neighbor_residual      regression: how far the molecule's own label sits
                           from the mean of its neighbours' labels, divided by
                           the spread of the whole label column.
                           classification: the share of its neighbours carrying
                           a different class.
    local_variance_ratio   regression: the same deviation, divided by the spread
                           of the neighbours' own labels instead -- so a
                           molecule scores high when it disagrees with
                           neighbours who agree with each other.
                           classification: the share of neighbours carrying a
                           different class, divided by one minus the share of
                           neighbours carrying their own dominant class.
    """
    y = np.asarray(y)
    neighbour_labels = y[neighbour_indices]
    k_eff = neighbour_labels.shape[1]

    if detector == 'neighbor_residual':
        if is_clf:
            return (neighbour_labels != y[:, None]).mean(axis=1)
        neighbour_mean = np.mean(neighbour_labels, axis=1)
        residuals = np.abs(y - neighbour_mean)
        return residuals / (np.std(y) + 1e-10)

    if detector == 'local_variance_ratio':
        if is_clf:
            from scipy.stats import mode as _scipy_mode
            disagreement = (neighbour_labels != y[:, None]).mean(axis=1)
            dominant = _scipy_mode(neighbour_labels, axis=1,
                                   keepdims=False).mode
            purity = (neighbour_labels == dominant[:, None]).mean(axis=1)
            heterogeneity = 1.0 - purity
            floor = max(1.0 / k_eff, 1e-3)
            return disagreement / np.maximum(heterogeneity, floor)
        neighbour_mean = np.mean(neighbour_labels, axis=1)
        neighbour_std = np.std(neighbour_labels, axis=1)
        deviation = np.abs(y - neighbour_mean)
        std_floor = max(0.01 * np.std(y), 1e-3)
        return deviation / np.maximum(neighbour_std, std_floor)

    raise ValueError(f"unknown detector {detector!r}; the two are {DETECTORS}")


# ---------------------------------------------------------------------------
# Naming the spaces
# ---------------------------------------------------------------------------

def space_specs(importance_methods, metrics, exponents,
                include_hybrid=True):
    """Every (space, metric) cell this run scores, and what each one is.

    One entry per key. `family` says which of the five distances it belongs to.
    `distance_metric` is None for forest proximity, which is a distance and not
    a space a metric is applied to.
    """
    specs = {}
    for m in metrics:
        specs[f'plain|{m}'] = {
            'family': 'plain',
            'distance_metric': m,
            'importance_method': None,
            'importance_exponent': None,
            'convention_matches_metric': None,
            'what_it_is': (
                f'the base representation, z-scored, under a {m} distance'),
        }
    for imp in importance_methods:
        for e in exponents:
            for m in metrics:
                specs[f'imp_pow{e:g}|{imp}|{m}'] = {
                    'family': 'importance_weighted',
                    'distance_metric': m,
                    'importance_method': imp,
                    'importance_exponent': float(e),
                    'convention_matches_metric': bool(
                        EXPONENT_MATCHED_TO_METRIC.get(m) == float(e)),
                    'what_it_is': (
                        f'the base representation, z-scored, each feature then '
                        f'multiplied by its {imp} importance raised to {e:g}, '
                        f'under a {m} distance'),
                }
    specs['forest_proximity'] = {
        'family': 'forest_proximity',
        'distance_metric': None,
        'importance_method': None,
        'importance_exponent': None,
        'convention_matches_metric': None,
        'what_it_is': (
            'one minus the share of trees of a forest fitted on this fold in '
            'which two molecules land in the same leaf'),
    }
    if include_hybrid:
        # ⛔ TWO HYBRID DISTANCES, NOT ONE. This file's whole subject is which
        # distance the neighbours are found in, and until 2026-09-19 its fourth
        # distance was only ever the PREDICTION-built space. The pool is asked
        # with pool_subset='predictive_and_similarity', but POOL_SUBSETS names
        # the families in PREFERENCE order and the pool returns the first that
        # has an entry, so the geometry family was recorded and never built.
        #
        # The geometry search picks features so that the distance they define
        # predicts each molecule's activity from its five nearest neighbours --
        # which is this experiment's question, asked at search time. Leaving it
        # out meant the one candidate space built for the job was the one
        # candidate space never measured.
        for _fam, _what in (
                ('hybrid', 'the feature space the PREDICTIVE recipe built'),
                ('hybrid_similarity',
                 'the feature space the SIMILARITY recipe built -- its '
                 'features were chosen so that the distance they define '
                 'predicts activity from the five nearest neighbours')):
            for m in metrics:
                specs[f'{_fam}|{m}'] = {
                    'family': _fam,
                    'distance_metric': m,
                    'importance_method': None,
                    'importance_exponent': None,
                    'convention_matches_metric': None,
                    'what_it_is': f'{_what}, z-scored, under a {m} distance',
                }
    return specs


def plain_reference_for(spec, metrics):
    """The plain space a space is compared against.

    A space carrying a metric is compared against the plain space under that
    same metric, so the only thing differing is which features the distance is
    computed over. Forest proximity carries no metric, so it is compared against
    the plain space under every metric in the sweep and both numbers are
    written.
    """
    m = spec.get('distance_metric')
    if m is None:
        return [f'plain|{x}' for x in metrics]
    return [f'plain|{m}']


# ---------------------------------------------------------------------------
# One fold-and-draw pass
# ---------------------------------------------------------------------------

def _build_hybrid_space(train_reps, val_reps, y_noisy, bp, groups_train,
                        train_augs, val_augs, model, metric_fn,
                        lower_is_better):
    """This fold's hybrid features, rebuilt under the corrupted labels.

    `noise_helpers.hybrid_for_fold` does the work: it narrows the pool to the
    representations the recipe actually scored, and it drops the model, the
    metric and its direction under a recipe from the similarity family, where
    build_fold_hybrid refuses the pair. n17, n18, n19 and n20 all call it, and a
    private copy here would have been the fifth.

    Returns (X_hybrid, None) or (None, reason). `hybrid_for_fold` prints and
    returns (None, None) on a failure rather than raising, so the reason
    recorded here is that it returned nothing.
    """
    X_tr_h, _X_va_h = hybrid_for_fold(
        train_reps, val_reps, y_noisy, bp, groups_train,
        model, metric_fn, lower_is_better,
        augmentations_train=train_augs, augmentations_val=val_augs)
    if X_tr_h is None:
        return None, 'hybrid_for_fold could not build this fold'
    return X_tr_h, None


def _spaces_for_pass(X_base_raw, X_base_std, y_noisy, is_clf, specs,
                     importance_methods, forest_kw, hybrid_X, errors,
                     pass_key, hybrid_sim_X=None):
    """Build every space this pass scores, once.

    Returns ({key: ('features', X, metric) or ('distance', D, None)},
             {key: True for every weighted space whose importance was all zero}).
    An importance method that fails leaves its four cells absent and writes the
    error; the rest of the pass still runs.
    """
    built = {}
    zero_importance = {}

    for key, spec in specs.items():
        if spec['family'] == 'plain':
            built[key] = ('features', X_base_std, spec['distance_metric'])

    imp_by_method = {}
    for imp_method in importance_methods:
        try:
            imp_by_method[imp_method] = compute_feature_importance(
                {'base': X_base_raw}, y_noisy, method=imp_method)['base']
        except Exception as exc:
            errors[f'{pass_key}|importance|{imp_method}'] = \
                f'{type(exc).__name__}: {exc}'

    for key, spec in specs.items():
        if spec['family'] != 'importance_weighted':
            continue
        imp = imp_by_method.get(spec['importance_method'])
        if imp is None:
            continue
        built[key] = ('features',
                      weight_features(X_base_std, imp,
                                      spec['importance_exponent']),
                      spec['distance_metric'])
        if importance_is_all_zero(imp):
            zero_importance[key] = True

    if 'forest_proximity' in specs:
        try:
            leaves = forest_leaf_ids(X_base_std, y_noisy, is_clf, **forest_kw)
            built['forest_proximity'] = (
                'distance', forest_proximity_distance(leaves), None)
        except Exception as exc:
            errors[f'{pass_key}|forest_proximity'] = \
                f'{type(exc).__name__}: {exc}'

    for _fam, _X in (('hybrid', hybrid_X),
                     ('hybrid_similarity', hybrid_sim_X)):
        if _X is None:
            continue
        sc = StandardScaler()
        _X_std = sc.fit_transform(_X)
        for key, spec in specs.items():
            if spec['family'] == _fam:
                built[key] = ('features', _X_std, spec['distance_metric'])

    return built, zero_importance


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, **kwargs):
    """N21: which distance the neighbours are found in."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N21: WHICH DISTANCE ARE THE NEIGHBOURS FOUND IN")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']

    condition = kwargs.get('noise_condition') or NOISE_CONDITION
    if condition not in DETECTION_IS_DEFINED:
        raise ValueError(
            f"n21: average precision is the headline here, and under "
            f"{condition!r} every training molecule's label moves, so the list "
            f"of molecules that moved is the list of all of them and average "
            f"precision is 1.0 by construction. The five conditions where it "
            f"is defined are {list(DETECTION_IS_DEFINED)}.")

    k_values = list(kwargs.get('k_values') or K_VALUES)
    importance_methods = list(kwargs.get('importance_methods')
                              or IMPORTANCE_METHODS)
    metrics = list(kwargs.get('metrics') or DISTANCE_METRICS_SWEPT)
    exponents = [float(e) for e in (kwargs.get('importance_exponents')
                                    or IMPORTANCE_EXPONENTS)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    seeds = list(kwargs.get('noise_seeds') or NOISE_SEEDS)
    forest_kw = {
        'n_trees': int(kwargs.get('forest_trees', FOREST_TREES)),
        'min_samples_leaf': int(kwargs.get('forest_min_samples_leaf',
                                           FOREST_MIN_SAMPLES_LEAF)),
        'seed': int(kwargs.get('forest_seed', FOREST_SEED)),
    }

    # The level grid, minus the degenerate endpoint. See the module docstring.
    levels_dropped = {}
    if kwargs.get('levels'):
        levels = [float(x) for x in kwargs['levels']]
    else:
        levels = [float(x) for x in
                  dataset_noise_levels(dataset_name, strategy=condition,
                                       detect_grid=True, is_clf=is_clf)]
        kept = [x for x in levels if x > 0.0]
        for x in levels:
            if x <= 0.0:
                levels_dropped[f'{x:g}'] = (
                    f'no molecule is corrupted under {condition} at level '
                    f'{x:g}, so average precision is undefined and the '
                    f'clean-to-corrupted gap is zero. Pass --levels to run it.')
        levels = kept

    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg -- the only kwarg it ever
    # forwards is noise_condition (noise_helpers.py:1053). This file read
    # kwargs alone, so the list the runner was asked for was silently dropped
    # and every run fell back to the module default. Eight files in this
    # directory had it; the same files read config['n_splits'] correctly two
    # lines away. Fixed 2026-09-20. An explicit kwarg still wins, which is what
    # a smoke or a direct caller uses.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS_NOISE)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n21: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS_NOISE)}, got "
            f"{sorted(model_factories)}.")

    base_rep_name = pick_noise_base_rep(reps, dataset_name)
    noise_groups, _n_groups, _largest = noise_groups_for(data)

    # The hybrid recipe, once, from the precomputed pool. n12's call, including
    # the subset that admits searches run under objective='similarity'.
    include_hybrid = bool(kwargs.get('include_hybrid', True))
    bp = None
    bp_sim = None
    # The SearchResult the recipe came out of, kept so the top-2 block at the
    # end of run() can read the rows it already holds. Before this it was
    # dropped at the end of the try below, and the only way to report a second
    # base representation or a second hybrid recipe was a second pool lookup,
    # which on a cold cache is a full search. Held, it costs 0 model fits.
    # Stays None under --no-hybrid, where no pool lookup happens at all.
    pool_search = None
    hybrid_meta = {'present': False, 'blueprint_is_clean_label': True}
    if not include_hybrid:
        hybrid_meta['absent_because'] = 'include_hybrid=False was passed'
    else:
        try:
            search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=downstream_models[0], budget=None,
                augmentations=augmentations, dataset_name=dataset_name,
                task_type=task_type, data=data,
                pool_subset='predictive_and_similarity')
            pool_search = search
            bp = meta['best_config']['blueprint'] if meta else None
            # The geometry family's own winner, taken by NAME. The pool returns
            # one family by preference and names the rest under by_objective,
            # so without this the fourth distance is only ever the
            # prediction-built space -- see the two-hybrid note in _specs.
            _by_obj = (meta.get('by_objective') or {}) if meta else {}
            _sim = _by_obj.get('similarity') or {}
            bp_sim = _sim.get('best_config', {}).get('blueprint')
            hybrid_meta['similarity_present'] = bp_sim is not None
            if bp_sim is None:
                hybrid_meta['similarity_absent_because'] = (
                    'the pool held no search under objective=similarity for '
                    'this dataset'
                    if 'similarity' not in _by_obj else
                    'the similarity family was in the pool but its winning '
                    'configuration carried no blueprint')
            else:
                hybrid_meta['similarity_score_mean'] = _sim.get('score_mean')
                hybrid_meta['similarity_metric_name'] = _sim.get('metric_name')
                hybrid_meta['similarity_note'] = (
                    "Its features were chosen so that the distance they "
                    "define predicts each molecule's activity from its five "
                    "nearest neighbours. score_mean is that correlation, not "
                    "the dataset's own metric.")
            if bp is None:
                hybrid_meta['absent_because'] = (
                    'the precomputed pool returned no recipe for this dataset')
            else:
                hybrid_meta.update({
                    'present': True,
                    'family': meta['objective'],
                    'family_choice': meta['objective_choice'],
                    'pool_subset': meta['pool_subset'],
                    'pool_metric': meta['metric_name'],
                    'tier': meta['tier'],
                    'aug_condition': meta['aug_condition'],
                })
        except Exception as exc:
            hybrid_meta['absent_because'] = f'{type(exc).__name__}: {exc}'
    if include_hybrid and not hybrid_meta['present']:
        print(f"  no hybrid distance: {hybrid_meta['absent_because']}")
    hybrid_meta['blueprint_note'] = (
        'The recipe was chosen from a search over the clean labels of every '
        'molecule, including each fold\'s held-out ones, and only rebuilt under '
        'the corrupted labels. n12 does the same and carries the same caveat: '
        'the hybrid distance is a best case, not a leak-free comparison.')

    specs = space_specs(importance_methods, metrics, exponents,
                        include_hybrid=hybrid_meta['present'])

    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, n_folds, sample_X, labels))
    n_passes = len(levels) * len(splits) * len(seeds)
    n_fits = (n_passes * len(specs) * len(detectors) * len(k_values)
              * len(downstream_models))

    print(f"  Dataset: {dataset_name}, {len(labels)} molecules, "
          f"{len(splits)} folds")
    print(f"  Metric: {metric_name} (lower is better: {lower_is_better})")
    print(f"  Base representation: {base_rep_name}")
    print(f"  Noise: {condition}, levels {levels}, draws {seeds}")
    if levels_dropped:
        for lvl, why in levels_dropped.items():
            print(f"  level {lvl} dropped: {why}")
    print(f"  Distances: {len(specs)} spaces -- "
          f"{sorted(set(s['family'] for s in specs.values()))}")
    print(f"  Detectors {detectors}, k values {k_values}")
    print(f"  Models scored: {downstream_models}")
    print(f"  {n_fits} model fits for the repair figure "
          f"({n_passes} passes x {len(specs)} spaces x {len(detectors)} "
          f"detectors x {len(k_values)} k values x "
          f"{len(downstream_models)} models). The detection figure costs no "
          f"fit at all.")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'base_rep': base_rep_name,
        'noise_condition': condition,
        'levels': levels,
        'levels_dropped': levels_dropped,
        'noise_seeds': seeds,
        'n_folds': len(splits),
        'k_values': k_values,
        'detectors': detectors,
        'importance_methods': importance_methods,
        'importance_exponents': exponents,
        'distance_metrics_swept': metrics,
        'exponent_matched_to_metric': dict(EXPONENT_MATCHED_TO_METRIC),
        'models': downstream_models,
        'headline': (
            'detection_average_precision -- the area under the precision-recall '
            'curve when each training molecule is ranked by how wrong its label '
            'looks and graded against whether its label really moved. It needs '
            'no cut-off, so it does not inherit the defect at '
            'NOISE_AUDIT_2026_09_10.md 6.9.'),
        'second_reading': (
            'recovery -- the share of the clean-to-corrupted gap in the '
            'dataset\'s own metric that repairing on this distance puts back. '
            'The repair is the suspicion score turned into a training weight '
            f'({WEIGHT_METHOD}, strength {WEIGHT_STRENGTH}), which is n12\'s.'),
        'spaces': specs,
        'forest': dict(forest_kw, fitted_on=(
            'the fold\'s training molecules only, on the base representation '
            'after z-scoring, against the corrupted training labels'),
            distance='one minus the share of trees in which two molecules '
                     'land in the same leaf'),
        'hybrid': hybrid_meta,
        'weight_strength': WEIGHT_STRENGTH,
        'weight_method': WEIGHT_METHOD,
        'evaluation_features': (
            'every space is scored after repair on the SAME features -- the '
            'plain base representation, z-scored. Only the weights differ, so '
            'the difference between two spaces is the distance and nothing '
            'else.'),
        'exclude_groups': (
            'not used, matching n12, so a molecule may take a repeat '
            'measurement of itself as a neighbour on any dataset that holds '
            'repeats'),
        'per_level': {},
    }

    t0 = time.time()
    for level in levels:
        lvl_key = f'{level:g}'
        print(f"\n  === level {lvl_key} ===")
        t_level = time.time()

        clean = {m: {} for m in downstream_models}
        noisy = {m: {} for m in downstream_models}
        # {space: {f'{detector}_k{k}|{pass_key}': average precision}}
        det_ap = {s: {} for s in specs}
        det_pk = {s: {} for s in specs}
        # {space: {f'{model}|{detector}_k{k}|{pass_key}': score}}
        rep_score = {s: {} for s in specs}
        zero_imp_builds = {}
        counts = {'n_corrupted': [], 'n_training': []}
        errors = {}
        provenance = {}

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            y_val = labels[val_idx]
            X_base_raw = reps[base_rep_name][train_idx]
            X_eval_raw = reps[base_rep_name][val_idx]
            scaler = StandardScaler()
            X_base_std = scaler.fit_transform(
                np.asarray(X_base_raw, dtype=np.float64))
            X_eval_std = scaler.transform(
                np.asarray(X_eval_raw, dtype=np.float64))

            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            train_augs = ({k: v[train_idx] for k, v in augmentations.items()}
                          if augmentations else None)
            val_augs = ({k: v[val_idx] for k, v in augmentations.items()}
                        if augmentations else None)
            g_train = groups[train_idx] if groups is not None else None

            # The clean reference has no noise in it, so it is the same on all
            # three draws of a fold. Fitted once.
            fold_key0 = f'f{fold_i}'
            for mn in downstream_models:
                try:
                    mdl = model_factories[mn]()
                    mdl.fit(X_base_std, labels[train_idx])
                    clean[mn][fold_key0] = float(metric_fn(
                        y_val, _predict_for_metric(mdl, X_eval_std,
                                                   metric_name)))
                except Exception as exc:
                    errors[f'{fold_key0}|clean|{mn}'] = \
                        f'{type(exc).__name__}: {exc}'

            for seed in seeds:
                pass_key = f'f{fold_i}s{seed}'
                spec_noise = noise_spec_for(dataset_name, condition, level,
                                            noise_seed=seed, is_clf=is_clf)
                y_noisy, cmask, prov = inject_train_labels(
                    labels, spec_noise, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=dataset_name)
                provenance.setdefault(spec_noise.spec_id, prov)
                counts['n_corrupted'].append(int(np.sum(cmask)))
                counts['n_training'].append(int(cmask.size))

                for mn in downstream_models:
                    try:
                        mdl = model_factories[mn]()
                        mdl.fit(X_base_std, y_noisy)
                        noisy[mn][pass_key] = float(metric_fn(
                            y_val, _predict_for_metric(mdl, X_eval_std,
                                                       metric_name)))
                    except Exception as exc:
                        errors[f'{pass_key}|noisy|{mn}'] = \
                            f'{type(exc).__name__}: {exc}'

                # Both recipes, rebuilt on this fold under the corrupted
                # labels. hybrid_for_fold reads the objective off the
                # blueprint and drops the model and metric under a geometry
                # one, so the same call serves both.
                hybrid_X = hybrid_sim_X = None
                for _fam, _bp in (('hybrid', bp),
                                  ('hybrid_similarity', bp_sim)):
                    if _bp is None:
                        continue
                    _X, why = _build_hybrid_space(
                        train_reps, val_reps, y_noisy, _bp, g_train,
                        train_augs, val_augs,
                        model_factories[downstream_models[0]](),
                        metric_fn, lower_is_better)
                    if _X is None:
                        errors[f'{pass_key}|{_fam}'] = why
                    elif _fam == 'hybrid':
                        hybrid_X = _X
                    else:
                        hybrid_sim_X = _X

                built, zero_imp = _spaces_for_pass(
                    X_base_raw, X_base_std, y_noisy, is_clf, specs,
                    importance_methods, forest_kw, hybrid_X, errors, pass_key,
                    hybrid_sim_X=hybrid_sim_X)
                for key in zero_imp:
                    zero_imp_builds[key] = zero_imp_builds.get(key, 0) + 1

                for space_key, (kind, payload, space_metric) in built.items():
                    for k in k_values:
                        k_eff = int(min(k, len(y_noisy) - 1))
                        if k_eff < 1:
                            errors[f'{pass_key}|{space_key}|k{k}'] = \
                                'fewer than two training molecules'
                            continue
                        try:
                            if kind == 'features':
                                nbrs = neighbours_in_space(
                                    payload, k_eff, space_metric)
                            else:
                                nbrs = neighbours_in_distance_matrix(
                                    payload, k_eff)
                        except Exception as exc:
                            errors[f'{pass_key}|{space_key}|k{k}|neighbours'] \
                                = f'{type(exc).__name__}: {exc}'
                            continue

                        for detector in detectors:
                            est_key = f'{detector}_k{k}'
                            try:
                                scores = suspicion_from_neighbours(
                                    nbrs, y_noisy, detector, is_clf)
                            except Exception as exc:
                                errors[f'{pass_key}|{space_key}|{est_key}'] = \
                                    f'{type(exc).__name__}: {exc}'
                                continue

                            det = detection_scores(cmask, scores, condition)
                            if det is not None:
                                det_ap[space_key][f'{est_key}|{pass_key}'] = \
                                    det['detection_average_precision']
                                det_pk[space_key][f'{est_key}|{pass_key}'] = \
                                    det['detection_precision_at_k']

                            try:
                                weights = weights_from_scores(
                                    scores, strength=WEIGHT_STRENGTH,
                                    method=WEIGHT_METHOD)
                            except Exception as exc:
                                errors[f'{pass_key}|{space_key}|{est_key}|weights'] \
                                    = f'{type(exc).__name__}: {exc}'
                                continue
                            for mn in downstream_models:
                                cell = f'{mn}|{est_key}|{pass_key}'
                                try:
                                    mdl = model_factories[mn]()
                                    mdl.fit(X_base_std, y_noisy,
                                            sample_weight=weights)
                                    rep_score[space_key][cell] = float(
                                        metric_fn(y_val, _predict_for_metric(
                                            mdl, X_eval_std, metric_name)))
                                except Exception as exc:
                                    errors[f'{space_key}|{cell}'] = \
                                        f'{type(exc).__name__}: {exc}'

            print(f"    level {lvl_key} fold {fold_i} done, "
                  f"{len(errors)} failures so far [{time.time() - t0:.0f}s]")

        results['per_level'][lvl_key] = convert_numpy(_level_block(
            specs, downstream_models, detectors, k_values, seeds,
            clean, noisy, det_ap, det_pk, rep_score, zero_imp_builds,
            counts, errors, provenance, lower_is_better))
        print(f"    [{time.time() - t_level:.0f}s]")

    # ── the top 2, added beside what is already reported ───────────────────
    # Added 2026-09-18, owner's instruction: the top 2 rows everywhere as an
    # extra, changing nothing that is already reported. Nothing above moves.
    # The run still scores every space in `specs` -- 13 under the defaults --
    # over every level, fold, draw, detector and k value, and 'per_level' and
    # 'summary' hold the numbers they held before this block existed.
    #
    # WHAT THESE ARE A RANKING OF, AND WHAT THEY ARE NOT. They read the pool
    # search the hybrid recipe came from. n21 does not build its distances from
    # that search: `base_rep` is picked by name at pick_noise_base_rep above,
    # and the 8 importance-weighted spaces, the forest proximity and every
    # repaired fit are all computed on that one representation. So these keys
    # say what the pool search scored each representation at, per downstream
    # model, and they do not change which representation n21 measured.
    #
    # What was absent without them: the file named one base representation and
    # one hybrid recipe and reported no score for any other, so a reader could
    # not tell a comparator ahead by a clear margin from one ahead in the
    # fourth decimal.
    #
    # 0 new model fits. All four helpers read rows the search already holds in
    # search.baselines and search.all_results.
    top2_base_reps, top2_hybrids = {}, {}
    hybrid_deltas_by_model, top2_pairs_by_model = {}, []
    if pool_search is not None:
        for _m in downstream_models:
            _ctx = f'n21 {dataset_name}/{_m}'
            try:
                # require_full=False: this search was fetched for
                # downstream_models[0] only, so a model it never scored comes
                # back as a short list instead of raising at the last line of a
                # run that has already spent every fit it was going to spend.
                _b = top_base_reps(pool_search, _m, 2, require_full=False,
                                   context=_ctx)
                top2_base_reps[_m] = [(e['rep'], e['score'])
                                      for e in _b['selected']]
                _h = top_hybrids(pool_search, _m, 2, require_full=False,
                                 context=_ctx)
                # Scalars only. Each selected row also carries a Blueprint
                # object, and this dict is written to JSON by
                # save_experiment_results. The recipe is already named by
                # 'label', and which representations fed it is reported by
                # hybrid_deltas under 'source_reps'.
                top2_hybrids[_m] = [
                    {'label': e['label'], 'score_mean': e['score_mean'],
                     'budget': e['budget'], 'n_features': e['n_features'],
                     'rep_count': e['rep_count'],
                     'n_folds_scored': e['n_folds_scored']}
                    for e in _h['selected']]
                if _h['selected']:
                    # hybrid= is passed rather than left None: the None branch
                    # of hybrid_deltas reads ['row'] off a top_hybrids entry,
                    # and top_hybrids builds no 'row' key, so it raises
                    # KeyError. The rank-1 entry carries 'score_mean' and
                    # 'blueprint', which is what hybrid_deltas reads.
                    hybrid_deltas_by_model[_m] = hybrid_deltas(
                        pool_search, _m, hybrid=_h['selected'][0], context=_ctx)
            except Exception as _e:
                print(f"    {_m}: top-2 / hybrid deltas unavailable "
                      f"({type(_e).__name__}: {_e})")
        try:
            _sel = select_model_rep_pairs(
                pool_search, models=list(downstream_models), n_base=2,
                n_hybrids=2, require_full=False,
                context=f'n21 {dataset_name}')
            top2_pairs_by_model = _sel['pairs']
        except Exception as _e:
            print(f"    top-2 pair grid unavailable "
                  f"({type(_e).__name__}: {_e})")

    # New keys, keyed by downstream model name. A downstream model this search
    # never scored gets an empty list here and no entry in
    # 'hybrid_deltas_by_model'. Read by nothing else in the repository:
    # noise_ceiling_analyse._read_n21 reads 'per_level' alone.
    results['top2_base_reps'] = top2_base_reps
    results['top2_hybrids'] = top2_hybrids
    # One flat row per (downstream model, slot): 5 downstream models x
    # (2 base representations + 2 hybrids) = 20 rows when every slot is filled.
    results['top2_pairs_by_model'] = convert_numpy(top2_pairs_by_model)
    # Two deltas per downstream model, both POSITIVE when the hybrid is better:
    # 'delta_vs_source_mean' against the mean of the base representations the
    # budget gave a non-zero number of features, 'delta_vs_best_rep' against the
    # single best base representation for that model.
    results['hybrid_deltas_by_model'] = convert_numpy(hybrid_deltas_by_model)
    results['top2_read_from'] = (
        'the pool search that chose the hybrid recipe, fetched once for '
        f'downstream model {downstream_models[0]}. The distances n21 measured '
        f'are built on {base_rep_name}, which was picked by name and not by '
        'this ranking.'
        if pool_search is not None else
        'no pool search was held: ' + str(hybrid_meta.get('absent_because')))

    results['summary'] = _summarise(results, specs, metrics, detectors,
                                    k_values, importance_methods, exponents,
                                    lower_is_better)
    _print_summary(results)
    save_experiment_results(results, output_dir, 'n21_neighbour_distance',
                            dataset_name)
    return results


def _mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float('nan')


def _level_block(specs, models, detectors, k_values, seeds, clean, noisy,
                 det_ap, det_pk, rep_score, zero_imp_builds, counts, errors,
                 provenance, lower_is_better):
    """One level's readings, per space.

    `clean_score_by_model` is one number per model, averaged over folds; the
    clean fit carries no noise so it is fitted once per fold, not once per draw.
    `noisy_score_by_model` is averaged over (fold, draw). The gap between them
    is what a repair has to put back.
    """
    clean_by_model = {m: _mean(clean[m].values()) for m in models}
    noisy_by_model = {m: _mean(noisy[m].values()) for m in models}
    gap_by_model = {}
    for m in models:
        c, nz = clean_by_model[m], noisy_by_model[m]
        if np.isfinite(c) and np.isfinite(nz):
            gap_by_model[m] = float((nz - c) if lower_is_better else (c - nz))
        else:
            gap_by_model[m] = float('nan')

    block = {
        'molecules_corrupted_mean': _mean(counts['n_corrupted']),
        'molecules_in_training_fold_mean': _mean(counts['n_training']),
        'clean_score_by_model': clean_by_model,
        'noisy_score_by_model': noisy_by_model,
        'gap_by_model': gap_by_model,
        'gap_note': ('clean minus corrupted, signed so a positive gap always '
                     'means the injected noise cost the model something'),
        'per_space': {},
    }

    for space_key in specs:
        ap = det_ap.get(space_key, {})
        pk = det_pk.get(space_key, {})
        rs = rep_score.get(space_key, {})

        per_detector = {}
        for detector in detectors:
            for k in k_values:
                est_key = f'{detector}_k{k}'
                rows = [v for cell, v in ap.items()
                        if cell.split('|', 1)[0] == est_key]
                pkrows = [v for cell, v in pk.items()
                          if cell.split('|', 1)[0] == est_key]
                if not rows:
                    continue
                per_detector[est_key] = {
                    'average_precision_mean': _mean(rows),
                    'average_precision_spread': (
                        float(np.std(rows)) if len(rows) > 1 else float('nan')),
                    'precision_at_k_mean': _mean(pkrows),
                    'n_passes': len(rows),
                }

        repair = {}
        for detector in detectors:
            for k in k_values:
                est_key = f'{detector}_k{k}'
                for m in models:
                    prefix = f'{m}|{est_key}|'
                    vals = [v for cell, v in rs.items()
                            if cell.startswith(prefix)]
                    if not vals:
                        continue
                    denoised = _mean(vals)
                    c, nz = clean_by_model[m], noisy_by_model[m]
                    # 2026-09-20: the denominator is the clean-to-corrupted gap.
                    # A gap of a few thousandths turns a small swing in the
                    # repaired score into a recovery of hundreds of percent, so
                    # require the gap to be at least GAP_MIN_FRAC of the clean
                    # score. Same floor as n06, n07 and n11.
                    _floor = (max(1e-10, GAP_MIN_FRAC * abs(c))
                              if np.isfinite(c) else 1e-10)
                    gap_too_small = not (abs(gap_by_model[m]) > _floor)
                    if (np.isfinite(c) and np.isfinite(nz)
                            and not gap_too_small):
                        recovery = float((denoised - nz) / (c - nz) * 100.0)
                        recovery_note = None
                    else:
                        recovery = float('nan')
                        recovery_note = (
                            f'not computed: the clean-to-corrupted gap '
                            f'({gap_by_model[m]:.6g}) is under '
                            f'{GAP_MIN_FRAC:.0%} of the clean score ({c:.6g})'
                            if gap_too_small and np.isfinite(c)
                            else 'not computed: a score was not finite')
                    repair[f'{m}|{est_key}'] = {
                        'model': m,
                        'detector': est_key,
                        'recovery_note': recovery_note,
                        'denoised_score': denoised,
                        'corrupted_score': nz,
                        'clean_score': c,
                        'recovery_percent_of_gap': recovery,
                        'n_passes': len(vals),
                    }

        row = {
            'detection_per_detector': per_detector,
            'detection_average_precision_mean': _mean(
                [v['average_precision_mean'] for v in per_detector.values()]),
            'repair': repair,
            'recovery_percent_of_gap_mean': _mean(
                [v['recovery_percent_of_gap'] for v in repair.values()]),
            'detection_average_precision_per_pass': dict(ap),
            'repair_score_per_pass': dict(rs),
        }
        if space_key in zero_imp_builds:
            row['builds_with_all_zero_importance'] = zero_imp_builds[space_key]
            row['builds_with_all_zero_importance_note'] = (
                'the importance was zero on every feature in this many builds, '
                'so the weighted space was the plain space and these numbers '
                'are a duplicate of the plain row')
        block['per_space'][space_key] = row

    if errors:
        block['_errors'] = errors
    if provenance:
        block['noise_provenance'] = provenance
    return block


def _paired(a, b, higher_is_better):
    """Per-cell gains of `a` over `b`, over the cells both scored.

    A cell is one (fold, draw, detector, k) -- or one (fold, draw, detector, k,
    model) for the repair scores. So every gain compares two distances on the
    same molecules, the same corruption and the same detector settings.
    """
    shared = sorted(set(a) & set(b))
    g = np.array([(a[c] - b[c]) if higher_is_better else (b[c] - a[c])
                  for c in shared], dtype=float)
    return g[np.isfinite(g)]


def _compare(a, b, higher_is_better, what):
    g = _paired(a, b, higher_is_better)
    if g.size == 0:
        return {'cells_paired': 0,
                'absent_because': 'no cell was scored on both sides'}
    return {
        'what_one_cell_is': what,
        'cells_paired': int(g.size),
        'cells_won': int((g > 0).sum()),
        'cells_lost': int((g < 0).sum()),
        'cells_tied': int((g == 0).sum()),
        'mean_gain': float(g.mean()),
        'spread_over_cell_gains': float(g.std()),
        'gain_exceeds_its_own_spread': bool(g.mean() > g.std()),
    }


def _summarise(results, specs, metrics, detectors, k_values,
               importance_methods, exponents, lower_is_better):
    """Two paired comparisons per level, plus the convention question.

    Everything is paired. Two distances are compared on the same fold, the same
    corruption draw, the same detector and the same k, so nothing but the
    distance differs between the two sides of a gain.
    """
    out = {
        'headline': ('detection average precision, each distance against the '
                     'plain one, paired'),
        'what_a_cell_is': (
            'detection: one (fold, draw, detector, k). repair: one (fold, '
            'draw, detector, k, model).'),
        'per_level': {},
    }
    ap_cell = 'one (fold, draw, detector, k)'
    rep_cell = 'one (fold, draw, detector, k, model)'

    for lvl, block in results['per_level'].items():
        ap = {s: r['detection_average_precision_per_pass']
              for s, r in block['per_space'].items()}
        rp = {s: r['repair_score_per_pass']
              for s, r in block['per_space'].items()}

        beats_plain = {}
        for space_key, spec in specs.items():
            if spec['family'] == 'plain':
                continue
            for ref in plain_reference_for(spec, metrics):
                if ref not in ap:
                    continue
                beats_plain[f'{space_key} vs {ref}'] = {
                    'detection_average_precision': _compare(
                        ap.get(space_key, {}), ap.get(ref, {}), True, ap_cell),
                    'repair_score': _compare(
                        rp.get(space_key, {}), rp.get(ref, {}),
                        not lower_is_better, rep_cell),
                }

        # The convention question: at one metric and one importance method,
        # which exponent finds the corrupted molecules better.
        convention = {}
        for imp in importance_methods:
            for m in metrics:
                keys = [f'imp_pow{e:g}|{imp}|{m}' for e in exponents]
                if not all(kk in ap for kk in keys):
                    continue
                matched = EXPONENT_MATCHED_TO_METRIC.get(m)
                best, best_ap = None, float('-inf')
                for e, kk in zip(exponents, keys):
                    v = block['per_space'][kk][
                        'detection_average_precision_mean']
                    if np.isfinite(v) and v > best_ap:
                        best_ap, best = v, float(e)
                row = {
                    'exponent_matched_to_this_metric': matched,
                    'exponent_with_the_higher_average_precision': best,
                    'average_precision_by_exponent': {
                        f'{e:g}': block['per_space'][kk][
                            'detection_average_precision_mean']
                        for e, kk in zip(exponents, keys)},
                }
                if len(exponents) == 2:
                    a_key = f'imp_pow{exponents[0]:g}|{imp}|{m}'
                    b_key = f'imp_pow{exponents[1]:g}|{imp}|{m}'
                    row[f'{exponents[0]:g} against {exponents[1]:g}'] = {
                        'detection_average_precision': _compare(
                            ap[a_key], ap[b_key], True, ap_cell),
                        'repair_score': _compare(
                            rp[a_key], rp[b_key], not lower_is_better,
                            rep_cell),
                    }
                if matched is not None and best is not None:
                    row['the_matched_exponent_won'] = bool(best == matched)
                convention[f'{imp}|{m}'] = row

        ranked = sorted(
            ((s, block['per_space'][s]['detection_average_precision_mean'])
             for s in specs if s in block['per_space']),
            key=lambda t: (-t[1] if np.isfinite(t[1]) else np.inf))

        out['per_level'][lvl] = {
            'average_precision_by_space': {s: v for s, v in ranked},
            'recovery_percent_of_gap_by_space': {
                s: block['per_space'][s]['recovery_percent_of_gap_mean']
                for s in specs if s in block['per_space']},
            'beats_plain': beats_plain,
            'weighting_convention': convention,
        }
    return out


def _print_summary(results):
    s = results['summary']
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("  One row is one distance. Average precision ranks every training")
    print("  molecule by how wrong its label looks and asks how far up that")
    print("  ranking the molecules whose label really moved sit. Recovery is")
    print(f"  the share of the clean-to-corrupted gap in {results['metric']}")
    print("  that repairing on this distance puts back.")
    for lvl, row in s['per_level'].items():
        block = results['per_level'][lvl]
        print(f"\n  level {lvl}: "
              f"{block['molecules_corrupted_mean']:.0f} molecules corrupted of "
              f"{block['molecules_in_training_fold_mean']:.0f} in the training "
              f"fold, on average")
        for space, ap in row['average_precision_by_space'].items():
            rec = row['recovery_percent_of_gap_by_space'].get(space,
                                                             float('nan'))
            print(f"    {space:44s} ap {ap:.4f}   recovery {rec:+7.1f}%")
        print("    against the plain distance, paired per (fold, draw, "
              "detector, k):")
        for name, cmp_row in row['beats_plain'].items():
            d = cmp_row['detection_average_precision']
            if not d.get('cells_paired'):
                print(f"      {name}: {d.get('absent_because')}")
                continue
            mark = '  >spread' if d['gain_exceeds_its_own_spread'] else ''
            print(f"      {name:58s} won {d['cells_won']} of "
                  f"{d['cells_paired']}, mean {d['mean_gain']:+.4f} "
                  f"(spread {d['spread_over_cell_gains']:.4f}){mark}")
        if row['weighting_convention']:
            print("    the scaling convention, at each metric:")
            for name, c in row['weighting_convention'].items():
                print(f"      {name:34s} matched exponent "
                      f"{c['exponent_matched_to_this_metric']}, higher average "
                      f"precision at {c['exponent_with_the_higher_average_precision']}"
                      + ("  MATCHED WON" if c.get('the_matched_exponent_won')
                         else ""))
    print("\n  A mean gain no larger than the spread of the per-cell gains "
          "behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N21: which distance are the nearest neighbours found in')
    parser.add_argument('--noise-condition', type=str, default=NOISE_CONDITION,
                        choices=list(DETECTION_IS_DEFINED),
                        help='The five conditions under which a corrupted '
                             'subset exists and average precision is defined.')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels. Default is the shared '
                             'grid minus 0.0, where nothing is corrupted.')
    parser.add_argument('--metrics', type=str, default=None,
                        help='Comma-separated distance metrics to sweep. '
                             'Default euclidean,manhattan.')
    parser.add_argument('--k-values', type=str, default=None,
                        help='Comma-separated neighbour counts. Default 5,10,20.')
    parser.add_argument('--importance-methods', type=str, default=None,
                        help='Comma-separated. Default lightgbm_gain,treeshap.')
    parser.add_argument('--forest-trees', type=int, default=FOREST_TREES)
    parser.add_argument('--forest-min-samples-leaf', type=int,
                        default=FOREST_MIN_SAMPLES_LEAF)
    parser.add_argument('--no-hybrid', action='store_true',
                        help='Skip the hybrid distance. The other three still '
                             'run and the result file says why it is absent.')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {'noise_condition': args.noise_condition,
          'include_hybrid': not args.no_hybrid,
          'forest_trees': args.forest_trees,
          'forest_min_samples_leaf': args.forest_min_samples_leaf}
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.metrics:
        kw['metrics'] = [x.strip() for x in args.metrics.split(',')]
    if args.k_values:
        kw['k_values'] = [int(x) for x in args.k_values.split(',')]
    if args.importance_methods:
        kw['importance_methods'] = [x.strip() for x
                                    in args.importance_methods.split(',')]
    if args.models:
        kw['models'] = [x.strip() for x in args.models.split(',')]
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
