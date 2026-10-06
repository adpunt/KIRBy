#!/usr/bin/env python3
"""N21: which distance are the nearest neighbours found in?

Paper: Noise.

THE QUESTION. Several detectors in this suite score a molecule by comparing its
label to the labels of its nearest neighbours. Which distance those neighbours
are found in has never been swept. n12 sweeps one weighting against no
weighting; nothing has put a forest proximity or the hybrid's own feature space
beside them.

2026-10-05, chat 5 of HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.0 (aligned
with section 2.4, Adelaide's rules R1-R6):
  - Folds come from noise_helpers.noise_outer_folds (config['n_splits'] or
    KIRBY_NOISE_N_FOLDS; the splits the per-fold searches are filed under).
    Any count, including 1. Summaries over folds use fold_spread.
  - Per outer fold, per each of the dataset's two decided models
    (DECISIONS.md Step 1, decided_grid.decided_models): the 2 best base
    representations and the 2 best general hybrids, all ranked by searches
    on that fold's training rows (noise_helpers.fold_configurations). Each
    model is fitted on those 4 configurations ('base_1', 'base_2', 'hybrid',
    'hybrid__general2'), so 2 models x 4 = 8 fitted configurations (R5).
    Nothing is read from a whole-dataset search any more:
    get_best_search_for_downstream, pool_recipe_per_downstream_model,
    pool_recipes_by_family, noise_base_reps_by_model, the five-model roster,
    rf's ranking of the single representations, and the top-2 record block
    read from the whole-dataset searches are gone.
  - The distance-search hybrid ('hybrid__distance1') and the similarity-
    search hybrid ('hybrid_similarity') are neighbour spaces only, beside the
    two general hybrids; no model is fitted on them (handoff 2.3 item 1).
  - The damage is the decided grid (noise_helpers.conditions_for_task with
    detection as the headline: outlier_p05, censoring, shift_only_p10,
    sparse_dprime_up_p10 on regression; the flip kinds on classification),
    at the decided levels (noise_helpers.decided_levels), through
    noise_helpers.inject_condition. 3 draws per non-zero level; level 0 (the
    labels as delivered) once. Censoring runs one draw at every level
    (noise_helpers.draws_for; it reads no seed, so its draws are identical).
  - Every neighbour average in this file weights the neighbours by a
    counting scheme ('rank' or 'inverse_square', KNN_SEARCH_SPACE_RECORD
    axis 5), and both are run. A molecule's scaffold-mates stay in its
    neighbour list (axis 6), and the molecule itself is dropped by index
    (dirty-run bug 7).
  - The shared similarity axis is n07's kit (sweep_settings for this
    dataset's group, fold_similarity_spaces, fold_neighbours): detection is
    graded on every (setting, k, scheme) with no model fit, and only the
    screen's top 3 (n07.screen_settings) are refitted, with the
    configuration's own decided model only.
  - Each (fold, level, draw) is written to a checkpoint file as it finishes
    (dirty-run bug 12).

THE DISTANCES, per fitted configuration's representation or hybrid.

  plain                the single representation as it is, scaled
                       (kirby.scaling.FeatureScaler: 0/1 features unscaled,
                       the rest z-scored).
  importance-weighted  each feature multiplied by its importance raised to an
                       exponent, after scaling. n12's space.
  forest proximity     how often two molecules land in the same leaf of a
                       fitted forest; the distance is one minus that share.
  the hybrids' own     the feature spaces of the model's two general hybrids,
                       its distance-search hybrid and the similarity-search
                       hybrid, each rebuilt on the fold's training rows from
                       the damaged labels.

The plain, importance-weighted and forest distances are built on each base
representation a decided model holds on this fold, once per representation
(two models holding one representation share it), and their weights repair
every configuration fitted on that representation. A hybrid space's weights
repair all four of its model's configurations.
DECIDED 2026-10-06 (Adelaide delegated): a representation's own distances
repair only the configurations fitted on that representation; the hybrid
spaces repair all four of their model's configurations (as n12).

WHICH (k, COUNTING SCHEME) EACH OF THIS FILE'S OWN DISTANCES REFITS AT.
DECIDED 2026-10-06 (Adelaide delegated): detection is graded at all 6 (k,
scheme) combinations (k 5, 10, 20 x rank, inverse_square) and both
detectors. Only the 3 best of the 6 are refitted, ranked per distance by an
inner hold-out screen (_screen_own_distance): a fifth of the training
molecules, as whole scaffold groups (the same hash and share as
n07.screen_settings), is predicted from its neighbours among the other four
fifths in that distance, against the damaged training labels, in the
dataset's own metric. Nothing held out by the experiment is read. That is
about 648 refits per (fold, level, draw) where it was 1,296. Two distances can
keep different (k, scheme), so a paired repair comparison between them pairs
only the cells both refitted.

WHAT IS SCORED. `noise_helpers.detection_scores` against the molecules whose
label actually moved. Average precision is the headline: it needs no threshold,
so it does not inherit the defect at NOISE_AUDIT_2026_09_10.md 6.9, where a
filter's cut-off is two standard deviations of the CORRUPTED label column and
so grows with the amount of noise injected. Beside it, the share of the
clean-to-corrupted gap that repairing on each distance puts back -- n12's
recovery figure, computed the same way -- and the plain difference between the
damaged and the clean held-out score per configuration (robustness on
fast_knn datasets is reported this way by decision; this file never computed
auc_norm).

WHICH NOISE CONDITIONS THIS CAN RUN UNDER. Only those where a SUBSET of the
training molecules is damaged, so the list of molecules that moved is not the
list of all of them. Under gaussian and grouped_shifted every training label
moves and average precision is 1.0 by construction, so they are not run.

ONE DEFECT IN n12 THIS SETTLES. n12 multiplies each feature by the square root
of its importance and then compares molecules under a Manhattan distance
(audit section 4, n12 defect 1). Under a Euclidean distance feature f
contributes w_f * (x_if - x_jf)^2 to the squared distance when the multiplier
is sqrt(w_f), so its weight in the distance is its importance. Under a
Manhattan distance the same multiplier contributes sqrt(w_f) * |x_if - x_jf|,
so its weight is the square root of its importance instead. Two ways of
scaling, two distances, four cells, and n21 runs all four. The exponent
matched to the metric is 0.5 under euclidean and 1.0 under manhattan; every
cell records `convention_matches_metric` so the matched pair and the
mismatched pair can be read apart.

LEVEL 0.0 corrupts no molecule. Average precision is not written there
(`detection_scores` returns None) and the recovery share is NaN. What level 0.0
does give is the repaired fit against the fit on the labels as recorded.

THE HYBRID RECIPES ARE CHOSEN ON CLEAN TRAINING LABELS. The per-fold searches
see the fold's training rows with clean labels (noise_helpers.
fold_configurations; which labels they should see is OPEN in noise_helpers),
and each hybrid's features are re-chosen from the damaged labels at every
draw. No held-out molecule enters any search.

HOW THE NEIGHBOURS ARE FOUND. One function, for every distance. The feature
spaces go through sklearn's NearestNeighbors under the named metric. Forest
proximity is not a feature space -- it is a pairwise distance and nothing
else -- so it goes in as a square distance matrix. The scoring arithmetic that
turns neighbours into a suspicion score is the same code for all of them,
following `NeighborResidualEstimator` and `LocalVarianceRatioEstimator`
(`src/kirby/utils/noise_estimation.py:236-370`) with every neighbour mean,
spread and share weighted by the counting scheme.

Usage:
    python n21_neighbour_distance.py --dataset logd --mode full
    python n21_neighbour_distance.py --dataset herg_ki --noise-condition censoring
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import inspect
import time
import numpy as np

from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined, sweep_conditions,
    conditions_for_task,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments._shared import (
    load_data, get_config, create_experiment_parser, convert_numpy,
    get_domain_metric, get_model_factories, make_nearest_neighbors,
)
from kirby.hybrid import compute_feature_importance
from kirby.utils.noise_estimation import (
    weights_from_scores, forest_leaf_ids, forest_proximity_distance,
    FOREST_TREES, FOREST_MIN_SAMPLES_LEAF, FOREST_SEED)
# The similarity-sweep kit, the checkpoint and the row file (n07's SIMILARITY-
# SWEEP KIT block). Replaces _sim_axis_n15_n22 here since 2026-10-05.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, sweep_schemes, screen_settings, screen_top,
    usable_ks, fold_similarity_spaces, fold_neighbours, RowSink,
    sweep_context, cleaned_from_weights, dprime_rows, row_identity,
    similarity_record, FoldCheckpoint, SCREEN_SCORED_ON, SCREEN_INNER_SHARE,
    _screen_prediction,
)
from experiments.noise import noise_cache as _nc

# The name the rows file, the checkpoint and the result file are filed under.
EXP_FULL_NAME = 'n21_neighbour_distance'

# What this file writes, since chat 5 (2026-10-05). In the condition files'
# fingerprint (run() below), the checkpoint's and the rows file's name, so a
# file written by the whole-dataset design is recomputed, never read back.
# 2026-10-06: own distances refitted at the screen's top 3 (k, scheme) only;
# decided importance methods per dataset; one censoring draw; screen inner=.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'


# The two neighbour-based detectors n12 runs, so the two experiments can be read
# side by side. activity_cliff_aware was cut on 2026-09-12: it ranks the
# molecules the same way as neighbour_residual at Spearman 0.989 on herg_ki in
# ecfp4.
DETECTORS = ('neighbor_residual', 'local_variance_ratio')

# Neighbour counts: the three the record keeps (KNN_SEARCH_SPACE_RECORD axis
# 4), the same three n12 runs.
K_VALUES = list(sweep_ks())

# The importance methods this file's own weighted distances scale by.
# DECIDED 2026-10-06 (Adelaide delegated): the importance-weighted distances
# are kept, because which weighting convention is right is this experiment's
# question, although KNN_SEARCH_SPACE_RECORD axis 3 cut weighting from the
# shared neighbour settings (the similarity axis below follows that cut).
# DECIDED 2026-10-06 (Adelaide delegated): DECISIONS.md Step 3's cut applies
# here. Each dataset runs its decided importance methods only
# (importance_methods_for, decided_grid 'general' search), which can include
# deeplift. IMPORTANCE_METHODS is used only for a dataset with no decided row.
IMPORTANCE_METHODS = ['lightgbm_gain', 'treeshap']


def importance_methods_for(dataset_name):
    """The dataset's decided importance methods (DECISIONS.md Step 3)."""
    from experiments.decided_grid import decided_search, UndecidedSearch
    try:
        return list(decided_search(dataset_name, 'general')
                    ['importance_methods'])
    except UndecidedSearch:
        return list(IMPORTANCE_METHODS)


# How many of the 6 (k, scheme) combinations each own distance refits at.
# DECIDED 2026-10-06 (Adelaide delegated): the inner hold-out screen's top 3.
OWN_REFIT_TOP = 3

# The two metrics the weighting conventions are run against.
# DECIDED 2026-10-06 (Adelaide delegated): this file's own distances keep
# euclidean and manhattan, although KNN_SEARCH_SPACE_RECORD axis 2 cut
# euclidean from the noise suite's neighbour settings (the shared similarity
# axis below follows that cut). The question here is which exponent matches
# which of these two; cosine has no per-feature sum for it to apply to.
DISTANCE_METRICS_SWEPT = ('euclidean', 'manhattan')

# The two scalings. 0.5 is what n12 applies. 1.0 is the one that gives a feature
# a weight equal to its importance under a Manhattan distance.
IMPORTANCE_EXPONENTS = (0.5, 1.0)

# Which exponent gives a feature a weight equal to its importance, per metric.
# Not a preference -- it is what the algebra says, and the sweep runs both
# against both so the claim is measured rather than assumed.
EXPONENT_MATCHED_TO_METRIC = {'euclidean': 0.5, 'manhattan': 1.0}

# The condition run when a caller names one and nothing else. The sweep runs
# every decided kind where a subset is damaged (run() below). outlier_p05 is
# the decided member of the outlier family (KNN_SEARCH_SPACE_RECORD, "The
# noise grid", 2026-10-05); outlier_p10 was the default before and is cut.
NOISE_CONDITION = 'outlier_p05'

# Draws of the corruption per non-zero level, averaged within a fold. Level 0
# is run once (the decided noise grid, 2026-10-05). Censoring runs one draw at
# every level (noise_helpers.draws_for, decided 2026-10-06).
NOISE_SEEDS = (0, 1, 2)

# The forest whose leaves the proximity distance is read off. Fitted on the
# fold's TRAINING molecules only, on the scaled single representation, against
# the CORRUPTED training labels -- a practitioner holds no others.
#
# min_samples_leaf is 5 rather than sklearn's 1 because a fully grown forest
# puts most training molecules alone in a leaf, and then two molecules share a
# leaf mainly when the bootstrap left one of them out. The proximity would be
# measuring the bootstrap.
#
# 100 trees, and 200 was tried and put back on 2026-09-23. Two molecules'
# distance is one minus the share of trees they land in the same leaf of, so
# the number of trees is the number of votes that share is taken over: at 100
# the distance can only take 101 distinct values. `_loop_n21_forest.py` swept
# 100, 200 and 300 trees against 1, 5, 20 and 50 molecules a leaf on
# flab_thermo_tresanco in onehot, under outlier_p10 at level 0.5. It is now
# finished: all 6 passes, one pass being one (fold, corruption draw), with 34
# to 46 of the 448 training molecules given a moved label.
#
# At 3 passes 200 trees looked won: +0.0082 detection average precision on 3 of
# the 3 passes. At all 6 passes, paired on the same 36 cells of (fold,
# corruption draw, detector, neighbour count), 200 trees gains 0.0025 average
# precision and wins 20 of the 36 cells, against a spread over those 36 cell
# gains of 0.0151 -- six times the gain. The arm means are 0.2366 average
# precision at 100 trees and 0.2391 at 200. On precision at the number of
# molecules that really moved, 0.2637 against 0.2683, winning 14 of the 36
# cells. 300 trees gains 0.0020 on the same 36 cells. So the tree count is
# inside its own spread and 100 stands.
#
# WHAT IT WOULD HAVE COST: the forest fit is most of the cost of a proximity,
# and 200 trees is twice the fit of 100. That pair, about 84 seconds for 200
# against 42 for 100 on fold 0 of that dataset at 5 molecules a leaf, 448
# training molecules over 5600 features, is a figure from an earlier session
# and was NOT re-measured; the sweep's own timings are recorded per (pass, leaf
# size) across all three tree counts together, so they do not separate them.
#
# THE LEAF SIZE STAYS AT 5, settled by the same 6 passes. 50 molecules a leaf
# loses 0.0214 average precision in 26 of the 36 cells. 1 molecule a leaf gains
# 0.0072 and 20 molecules a leaf gains 0.0033, each inside its own spread over
# the 36 cell gains of 0.0446 and 0.0422, and both LOSE on precision at the
# moved count, -0.0191 and -0.0080.
#
# WHAT THE FOREST IS WORTH, on the same 36 cells: the proximity at 100 trees
# and 5 molecules a leaf beats the plain representation under euclidean by
# 0.0522 average precision in 31 of the 36 cells, larger than its own spread of
# 0.0429 over those gains.
FOREST_TREES = 100
FOREST_MIN_SAMPLES_LEAF = 5
FOREST_SEED = 42

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
# (n07_remediation.GAP_MIN_FRAC), n11 and n12.
GAP_MIN_FRAC = 0.02

# The configurations each decided model is fitted on (noise_helpers.
# CONFIGURATION_SPACES), and the hybrid spaces this file finds neighbours in.
BASE_SPACES = ('base_1', 'base_2')
HYBRID_FIT_SPACES = ('hybrid', 'hybrid__general2')
NEIGHBOUR_HYBRID_SPACES = ('hybrid', 'hybrid__general2', 'hybrid__distance1',
                           'hybrid_similarity')

# Keyword arguments this file took before 2026-10-05 and no longer reads: the
# models, the single representations, the levels, the draws and the neighbour
# counts are decided, and the hybrids are fitted configurations now, so they
# cannot be switched off. A caller passing one is told it is ignored.
_RETIRED_KWARGS = ('models', 'base_reps', 'levels', 'k_values', 'noise_seeds',
                   'include_hybrid')


# ---------------------------------------------------------------------------
# The distances
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
    """Multiply already-scaled features by importance ** exponent.

    THE CALLER MUST PASS A SCALED X. Weighting first and standardizing
    after is a mathematical no-op -- a per-column StandardScaler exactly cancels
    a per-column scalar multiply -- and that is how the whole weighting ablation
    silently produced identical geometry in an earlier version of n12.

    `exponent` 0.5 reproduces n12's `_weight_features` exactly, including the
    normalisation. The weights are scaled to average 1 across features, so a
    scaled column is stretched or squeezed around its own unit variance
    instead of every column being crushed toward 1/width. That normalisation is
    a single positive multiplier applied to the whole matrix, so it moves no
    molecule's neighbours; it is kept only so this space IS n12's space at
    exponent 0.5 and the smoke can assert it.

    The weights take the matrix's own precision, so a float32 representation
    stays float32 (handoff 2.3 item 18, 2026-10-05), as in n12.

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
    X_standardized = np.asarray(X_standardized)
    Xw = X_standardized * w.astype(X_standardized.dtype, copy=False)

    # DROP THE FEATURES THE SCORER GAVE NOTHING, exactly as n12 does since
    # commit ca249de. At any exponent above zero a feature with no importance
    # gets weight zero, so its column is zero in every row -- on
    # flab_thermo_tresanco lightgbm_gain scored 181 of onehot's 5600 features
    # above zero, leaving 5419 columns of zeros carrying nothing through every
    # neighbour search.
    #
    # This is EXACT, not an approximation, for every metric
    # kirby.similarity.distance_config sweeps: manhattan, euclidean and
    # minkowski sum a per-feature term that is zero when both values are zero,
    # and cosine's dot product and both norms gain zero from such a feature.
    # No metric in that list centres per molecule, which is the one thing that
    # would make a zero column change an answer.
    #
    # ⛔ At exponent 0.0, numpy gives 0 ** 0 == 1, so nothing is dropped and
    # the space is the unweighted control. That still holds.
    live = w > 0
    n_live = int(live.sum())
    if 0 < n_live < len(w):
        return Xw[:, live]
    return Xw


# forest_leaf_ids and forest_proximity_distance MOVED to
# src/kirby/utils/noise_estimation.py on 2026-09-23, unchanged, so that the
# label-smoothing cleanups can use the same proximity this experiment measures
# as a distance. Imported at the top of this file.


# ---------------------------------------------------------------------------
# Neighbours, and the suspicion score built from them
# ---------------------------------------------------------------------------

def neighbours_in_space(X, k, metric, return_distance=False):
    """The k nearest OTHER molecules to each molecule, in a feature space.

    Asks sklearn for k+1 neighbours and drops the molecule itself BY INDEX
    (dirty-run bug 7, handoff 2.7, 2026-10-05). Dropping the first column, as
    noise_estimation._knn_indices does and this function did until then,
    assumes the first neighbour returned is the molecule itself; when another
    training molecule is identical in this space either can come first, and
    then the molecule's own label sits in the average it is compared with
    while one real neighbour is dropped. Same fix as n02 (736ec93) and n12.
    A molecule whose own index did not come back (more than k other molecules
    tied at distance 0) keeps its first k neighbours.

    No scaffold group is excluded (KNN_SEARCH_SPACE_RECORD axis 6).

    Returns idx (n, k_eff), or (idx, dist) with return_distance=True.
    """
    X = np.asarray(X)
    n = len(X)
    k_eff = int(min(k, n - 1))
    if k_eff < 1:
        raise ValueError('fewer than two training molecules')
    nn = make_nearest_neighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    own = idx == np.arange(n)[:, None]
    drop = own.copy()
    drop[~own.any(axis=1), -1] = True
    idx = idx[~drop].reshape(n, k_eff)
    dist = dist[~drop].reshape(n, k_eff)
    return (idx, dist) if return_distance else idx


def _cached_neighbours_in_space(dataset, X, k, metric):
    """(idx, dist) of neighbours_in_space, read from noise_cache when the same
    matrix, k and metric were queried before. The process-wide distance
    setting is in every key (noise_cache.process_settings). The kind name
    changed on 2026-10-05 (drop by index, distances returned), so an entry
    written by the old first-column drop is never served."""
    out = _nc.cached(
        dataset, 'n21_neighbours_v2',
        {'X': _nc.array_hash(np.asarray(X)), 'k': int(k),
         'metric': None if metric is None else str(metric)},
        lambda: neighbours_in_space(X, k, metric, return_distance=True))
    return np.asarray(out[0]), np.asarray(out[1])


def neighbours_in_distance_matrix(D, k, return_distance=False):
    """The k nearest OTHER molecules to each molecule, from a square distance.

    The diagonal is forced below every off-diagonal entry before sorting, so the
    molecule itself is always the first column and dropping that column always
    drops the molecule itself. Without that, a second molecule at distance zero
    -- which a forest proximity produces whenever two molecules share every leaf
    -- could take the first column and the molecule would stay in its own
    neighbour list.

    Only the nearest k+1 of each row are sorted, not the whole row. A full
    argsort of a square matrix over 4000 training molecules allocates 130 MB of
    indices per call.
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
        idx = candidates[:, 1:k + 1]
    finally:
        # The caller's matrix is handed back as it arrived, diagonal and all.
        np.fill_diagonal(D, diagonal)
    if not return_distance:
        return idx
    return idx, np.take_along_axis(D, idx, axis=1)


def suspicion_from_neighbours(neighbour_indices, y, detector, is_clf,
                              dist=None, scheme=None):
    """How wrong each molecule's label looks, given its neighbours.

    The formulas of `NeighborResidualEstimator.estimate` and
    `LocalVarianceRatioEstimator.estimate`
    (src/kirby/utils/noise_estimation.py:236-370), taking the neighbours
    already found, because a forest proximity is not a feature space.

    scheme: how much each neighbour counts ('rank' or 'inverse_square',
    similarity_space.neighbour_weights, KNN_SEARCH_SPACE_RECORD axis 5).
    Every neighbour mean, spread, share and vote is weighted by it; `dist` is
    the neighbours' distances, needed by 'inverse_square'. Every measurement in
    this file passes a scheme (2026-10-05). scheme=None keeps the equal-weight
    form the two classes compute, which `_smoke_n21.py` asserts this function
    reproduces to the last bit.

    neighbor_residual      regression: |y - neighbour mean| / sd(y).
                           classification: the share of its neighbours carrying
                           a different class.
    local_variance_ratio   regression: |y - neighbour mean| / max(neighbour
                           sd, max(0.01 sd(y), 1e-3)).
                           classification: that share / max(1 - the share of
                           the heaviest class, max(1/k, 1e-3)).
    """
    y = np.asarray(y)
    idx = np.asarray(neighbour_indices)
    nb = y[idx]
    k_eff = nb.shape[1]
    if scheme is None:
        w = np.ones(nb.shape, dtype=float)
    else:
        from experiments.noise import similarity_space as ss
        if dist is None:
            raise ValueError('a counting scheme needs the neighbour distances')
        w = ss.neighbour_weights(np.asarray(dist), scheme)
    W = w.sum(axis=1)

    if detector not in DETECTORS:
        raise ValueError(f"unknown detector {detector!r}; the two are "
                         f"{DETECTORS}")
    if is_clf:
        disagree = (w * (nb != y[:, None])).sum(axis=1) / W
        if detector == 'neighbor_residual':
            return disagree
        classes = np.unique(nb)
        votes = np.stack([(w * (nb == c)).sum(axis=1) for c in classes],
                         axis=1)
        purity = votes.max(axis=1) / W
        floor = max(1.0 / k_eff, 1e-3)
        return disagree / np.maximum(1.0 - purity, floor)
    yf = y.astype(np.float64)
    nbf = nb.astype(np.float64)
    mean = (w * nbf).sum(axis=1) / W
    dev = np.abs(yf - mean)
    if detector == 'neighbor_residual':
        return dev / (np.std(yf) + 1e-10)
    sd = np.sqrt((w * (nbf - mean[:, None]) ** 2).sum(axis=1) / W)
    floor = max(0.01 * np.std(yf), 1e-3)
    return dev / np.maximum(sd, floor)


def inner_holdout_mask(groups_tr, n):
    """The held-back fifth of the training molecules for the own-distance
    screen: whole scaffold groups (each molecule its own group when there are
    none), chosen by the same fixed hash of the group label and the same share
    (n07.SCREEN_INNER_SHARE) as n07.screen_settings. None when fewer than 5
    molecules would be held back or fewer than 25 left to search."""
    import zlib
    keys = (np.asarray(groups_tr) if groups_tr is not None
            else np.arange(n)).astype(str)
    held = np.array([(zlib.crc32(k.encode()) % 1000)
                     < SCREEN_INNER_SHARE * 1000 for k in keys])
    if held.sum() < 5 or (~held).sum() < 25:
        return None
    return held


def _inner_forest_distance(X_std, y, held, is_clf, forest_kw):
    """Forest proximity of the held-back molecules to the rest, from a forest
    fitted on the rest only (the held-back labels never shape the leaves), with
    the same settings as this file's forest. Returns (n_held, n_rest)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = RandomForestClassifier if is_clf else RandomForestRegressor
    rf = cls(n_estimators=int(forest_kw['n_trees']),
             min_samples_leaf=int(forest_kw['min_samples_leaf']),
             random_state=int(forest_kw['seed']),
             n_jobs=int(os.environ.get('KIRBY_RF_JOBS', '1') or 1))
    X = np.asarray(X_std, dtype=np.float64)
    rf.fit(X[~held], np.asarray(y)[~held])
    L_rest = np.asarray(rf.apply(X[~held]), dtype=np.int32)
    L_held = np.asarray(rf.apply(X[held]), dtype=np.int32)
    n_trees = L_rest.shape[1]
    D = np.empty((len(L_held), len(L_rest)), dtype=np.float32)
    for a in range(0, len(L_held), 125):
        b = min(len(L_held), a + 125)
        same = (L_held[a:b, None, :] == L_rest[None, :, :])
        D[a:b] = 1.0 - (same.sum(axis=2) / float(n_trees))
    return D


def _screen_own_distance(dataset, kind, payload, space_metric, *, held,
                         y_noisy, is_clf, metric, ks, schemes, top,
                         forest_X=None, forest_kw=None, idx_all=None,
                         dist_all=None):
    """Rank one own distance's (k, scheme) combinations; keep the best `top`.

    DECIDED 2026-10-06 (Adelaide delegated). The held-back molecules (held,
    inner_holdout_mask) are predicted from their k nearest among the other
    training molecules in this distance, each neighbour counted by the scheme
    (n07._screen_prediction, the shared screen's prediction), and scored
    against the damaged training labels in the dataset's own metric. For
    forest proximity the forest is refitted on the other molecules
    (_inner_forest_distance). Importance-weighted and hybrid spaces are used
    as built on all the training molecules, as n07.screen_settings uses its
    spaces.

    held None (too few molecules to hold back): each training molecule is
    predicted from its own neighbours in the full lists (idx_all, dist_all,
    itself excluded) instead, and 'scored_on' says so.

    Returns (kept [(k, scheme)] best first, table [dict]).
    """
    metric_name, metric_fn, lower = metric
    y = np.asarray(y_noisy)
    if held is not None:
        scored_on = 'inner_holdout'
        y_src, y_true = y[~held], y[held]
        k_need = int(min(max(ks), int((~held).sum())))
        if kind == 'features':
            def _nb():
                X = np.asarray(payload)
                nn = make_nearest_neighbors(n_neighbors=k_need,
                                            metric=space_metric)
                nn.fit(X[~held])
                d, i = nn.kneighbors(X[held])
                return i, d
            idx, dist = _nc.cached(
                dataset, 'n21_inner_screen_neighbours_v1',
                {'X': _nc.array_hash(np.asarray(payload)),
                 'held': _nc.array_hash(held.astype(np.int8)),
                 'k': k_need, 'metric': str(space_metric)}, _nb)
            idx, dist = np.asarray(idx), np.asarray(dist)
        else:
            D = _inner_forest_distance(forest_X, y, held, is_clf, forest_kw)
            idx, dist = neighbours_in_rectangle(D, k_need)
    else:
        scored_on = 'training_rows'
        y_src = y_true = y
        idx, dist = np.asarray(idx_all), np.asarray(dist_all)
    table, cands = [], []
    for k in ks:
        kk = int(min(k, idx.shape[1]))
        for sch in schemes:
            try:
                pred = _screen_prediction(y_src, idx[:, :kk], dist[:, :kk],
                                          sch, is_clf, metric_name)
                sc = float(metric_fn(y_true, pred))
            except Exception as exc:
                table.append({'k': int(k), 'scheme': sch, 'score': None,
                              'error': f'{type(exc).__name__}: {exc}'})
                continue
            table.append({'k': int(k), 'scheme': sch, 'score': sc})
            if np.isfinite(sc):
                cands.append((sc, int(k), sch))
    cands.sort(key=lambda c: ((c[0] if lower else -c[0]), c[1], c[2]))
    kept = [(k, sch) for _, k, sch in (cands if top <= 0 else cands[:top])]
    for r in table:
        r['kept'] = (r['k'], r['scheme']) in kept
        r['scored_on'] = scored_on
    return kept, table


def neighbours_in_rectangle(D, k):
    """The k nearest columns of each row of a (rows, columns) distance, nearest
    first, and their distances. Rows and columns are different molecules, so
    nothing is dropped."""
    D = np.asarray(D)
    k = int(min(k, D.shape[1]))
    rows = np.arange(D.shape[0])[:, None]
    if k >= D.shape[1]:
        idx = np.argsort(D, axis=1, kind='stable')
    else:
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        order = np.argsort(D[rows, part], axis=1, kind='stable')
        idx = part[rows, order]
    return idx, np.take_along_axis(D, idx, axis=1)


def _detection(cmask, scores, condition, is_clf):
    """detection_scores, also on the classification flip kinds.

    detection_scores grades only conditions named in DETECTION_IS_DEFINED,
    which are regression names. A flip kind damages a share of the training
    labels (the level is that share), so a corrupted subset exists and the
    grading is defined. Before 2026-10-05 this file swept the regression names
    on a classification dataset too (noise_spec_for mapped them to flips), so
    the gate passed. Now the flip kinds are swept by name, and the gate is
    passed with outlier_p05 for them; 'detection_condition' is set back to the
    real kind. Shared change asked for: detection_is_defined should accept
    the flip kinds.
    """
    gate = condition
    if is_clf and not detection_is_defined(condition):
        gate = 'outlier_p05'
    det = detection_scores(cmask, scores, gate)
    if det is not None:
        det['detection_condition'] = str(condition)
    return det


def _takes_sample_weight(factory):
    """True when the model's fit() accepts sample_weight. gp (sklearn's
    GaussianProcessRegressor) does not; rf, xgboost, lightgbm, svm and dnn do."""
    try:
        params = inspect.signature(factory().fit).parameters
    except (TypeError, ValueError):
        return False
    return ('sample_weight' in params
            or any(p.kind == p.VAR_KEYWORD for p in params.values()))


# ---------------------------------------------------------------------------
# Building one pass's spaces
# ---------------------------------------------------------------------------

def _representation_spaces(rn, X_raw_tr, X_std_tr, y_noisy, is_clf,
                           importance_methods, metrics, exponents, forest_kw,
                           dataset, errors, err_prefix):
    """Every distance built on one base representation for one (fold, level,
    draw): plain under each metric, importance-weighted under each (method,
    exponent, metric), and forest proximity.

    Returns ({space key: ('features', X, metric) or ('distance', D, None)},
             {space key: spec}, {space key: True when the importance was all
             zero}). Space keys start with the representation name, so two
    models holding one representation share one set of numbers.
    """
    built, specs, zero = {}, {}, {}
    for m in metrics:
        key = f'{rn}/plain|{m}'
        built[key] = ('features', X_std_tr, m)
        specs[key] = {
            'family': 'plain', 'representation': rn, 'distance_metric': m,
            'importance_method': None, 'importance_exponent': None,
            'convention_matches_metric': None, 'plain_references': [],
            'what_it_is': f'{rn}, scaled, under a {m} distance'}

    for imp_method in importance_methods:
        try:
            def _imp(_m=imp_method):
                return np.asarray(compute_feature_importance(
                    {'base': X_raw_tr}, y_noisy, method=_m)['base'],
                    dtype=float)
            imp = np.asarray(_nc.cached(
                dataset, 'n21_importance',
                {'X': _nc.array_hash(X_raw_tr), 'y': _nc.array_hash(y_noisy),
                 'method': str(imp_method), 'random_state': 42}, _imp))
        except Exception as exc:
            errors[f'{err_prefix}|{rn}|importance|{imp_method}'] = \
                f'{type(exc).__name__}: {exc}'
            continue
        for e in exponents:
            Xw_cache = {}
            for m in metrics:
                key = f'{rn}/imp_pow{e:g}|{imp_method}|{m}'
                if 'X' not in Xw_cache:
                    Xw_cache['X'] = weight_features(X_std_tr, imp, e)
                built[key] = ('features', Xw_cache['X'], m)
                specs[key] = {
                    'family': 'importance_weighted', 'representation': rn,
                    'distance_metric': m, 'importance_method': imp_method,
                    'importance_exponent': float(e),
                    'convention_matches_metric': bool(
                        EXPONENT_MATCHED_TO_METRIC.get(m) == float(e)),
                    'plain_references': [f'{rn}/plain|{m}'],
                    'what_it_is': (
                        f'{rn}, scaled, each feature then multiplied by its '
                        f'{imp_method} importance raised to {e:g}, under a '
                        f'{m} distance')}
                if importance_is_all_zero(imp):
                    zero[key] = True

    key = f'{rn}/forest_proximity'
    try:
        leaves = forest_leaf_ids(X_std_tr, y_noisy, is_clf, **forest_kw)
        built[key] = ('distance', forest_proximity_distance(leaves), None)
        specs[key] = {
            'family': 'forest_proximity', 'representation': rn,
            'distance_metric': None, 'importance_method': None,
            'importance_exponent': None, 'convention_matches_metric': None,
            'plain_references': [f'{rn}/plain|{m}' for m in metrics],
            'what_it_is': (
                f'one minus the share of trees of a forest fitted on this '
                f'fold\'s training molecules in {rn}, against the damaged '
                f'labels, in which two molecules land in the same leaf')}
    except Exception as exc:
        errors[f'{err_prefix}|{rn}|forest_proximity'] = \
            f'{type(exc).__name__}: {exc}'
    return built, specs, zero


def _hybrid_spaces(model, hyb, base_names, metrics):
    """The hybrid neighbour spaces of one model: each of its built hybrid
    spaces under each metric. Their plain references are the plain distances
    of the model's two base representations under the same metric."""
    built, specs = {}, {}
    for sp, (X_tr, _X_va) in hyb.items():
        for m in metrics:
            key = f'{model}/{sp}|{m}'
            built[key] = ('features', X_tr, m)
            specs[key] = {
                'family': ('hybrid_similarity' if sp == 'hybrid_similarity'
                           else 'hybrid'),
                'hybrid_space': sp, 'recipe_model': model,
                'distance_metric': m, 'importance_method': None,
                'importance_exponent': None, 'convention_matches_metric': None,
                'plain_references': [f'{rn}/plain|{m}' for rn in base_names],
                'what_it_is': (
                    f"{model}'s {sp} feature space, rebuilt on this fold's "
                    f"training rows from the damaged labels, scaled, under a "
                    f"{m} distance")}
    return built, specs


# ---------------------------------------------------------------------------
# The shared similarity axis, one fitted configuration
# ---------------------------------------------------------------------------

def _sweep_pass(ctx, *, model_name, model_input, X_tr, X_va, y_tr_noisy,
                y_tr_clean, y_val, train_idx, val_idx, groups_tr, fold, draw,
                entry, corrupted_mask, detection_acc, grade_spaces,
                can_weight):
    """The record's neighbour settings for this dataset's group, both
    detectors at every (setting, k, counting scheme), for one fitted
    configuration on one (fold, level, draw).

    Detection (no model fit) is graded on every (setting, k, scheme) whose
    space is in grade_spaces (None = every space) into detection_acc. The
    per-dataset screen (n07.screen_settings) keeps the best SCREEN_TOP
    (setting, k, scheme), and only those are refitted, with the weights the
    detector's score gives, into the D-versus-D' rows, with this
    configuration's own decided model only (R1). Written out here, as in n12,
    because n07.similarity_sweep adds rf and svm to every row.
    """
    from experiments.noise import similarity_space as ss
    sink = ctx['sink']
    ds = ctx['dataset_name']
    is_clf = ctx['is_clf']
    settings = ctx['settings']
    spaces_tr, spaces_va, unavail = fold_similarity_spaces(
        ds, settings, model_input_tr=X_tr, model_input_va=X_va,
        reps=ctx['reps'], train_idx=train_idx, val_idx=val_idx,
        y_train=y_tr_noisy, groups_train=groups_tr,
        base_reps=entry.get('base') or [], recipes=entry.get('recipes') or {},
        augmentations=ctx['augmentations'], model_name=model_name,
        model_factory=ctx['factories'].get(model_name), metric=ctx['metric'],
        task_type=ctx['task_type'])
    for sp, why in unavail.items():
        ctx['unavailable'].setdefault(f'{model_input}|{sp}', why)
    smi = ctx.get('smiles')
    smi_tr = None if smi is None else [smi[i] for i in np.asarray(train_idx)]
    smi_va = None if smi is None else [smi[i] for i in np.asarray(val_idx)]
    # No scaffold groups are excluded from the lists (axis 6).
    nb = fold_neighbours(ds, settings, spaces_tr, spaces_va, y_tr_noisy,
                         smiles_train=smi_tr, smiles_val=smi_va, is_clf=is_clf)
    for s in settings:
        L = nb.get(s.name)
        if L is not None and not L.get('available'):
            ctx['unavailable'].setdefault(f'{model_input}|{s.name}',
                                          L.get('reason'))

    def _scores(L, k, scheme, detector):
        idx, dist = ss.neighbours_at(L, k)
        return suspicion_from_neighbours(idx, y_tr_noisy, detector, is_clf,
                                         dist=dist, scheme=scheme)

    if corrupted_mask is not None and detection_acc is not None:
        for s in settings:
            if grade_spaces is not None and s.space not in grade_spaces:
                continue
            L = nb.get(s.name)
            for k in usable_ks(L):
                for sch in sweep_schemes():
                    for detector in ctx['detectors']:
                        try:
                            sc = _scores(L, k, sch, detector)
                        except Exception as exc:
                            print(f"      {detector} on {s.name} k={k} "
                                  f"{sch} failed: {exc}")
                            continue
                        det = _detection(corrupted_mask, sc,
                                         ctx['condition'], is_clf)
                        if det is not None:
                            (detection_acc.setdefault(s.name, {})
                             .setdefault(f'k={k}|scheme={sch}', {})
                             )[detector] = det['detection_average_precision']

    # DECIDED 2026-10-06 (Adelaide delegated): the screen predicts a held-back
    # fifth of the training molecules, from lists built on the other four
    # fifths of the same spaces (n07.screen_settings inner=).
    kept, table = screen_settings(
        settings, nb, y_tr_noisy, metric=ctx['metric'], is_clf=is_clf,
        y_val=y_val,
        inner={'dataset_name': ds, 'spaces_train': spaces_tr,
               'y_train': y_tr_noisy, 'groups_train': groups_tr,
               'smiles_train': smi_tr, 'is_clf': is_clf})
    ctx.setdefault('screen', []).append({
        'model_input': model_input, 'fold': fold, 'draw': draw,
        'level': ctx['level'], 'condition': ctx['condition'],
        'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
        'table': table})
    if sink is None or not sink.enabled:
        return
    if not can_weight:
        # DECIDED 2026-10-06 (Adelaide delegated): a decided model that takes
        # no per-row weight (gp) records "not run" with the reason.
        ctx['unavailable'].setdefault(
            f'{model_input}|rows',
            "not run: this model's fit() takes no sample_weight")
        return
    base_id = dict(dataset_name=ds, experiment=EXP_FULL_NAME,
                   model_input=model_input, noise_condition=ctx['condition'],
                   level=ctx['level'], draw=draw, fold=fold)
    for s, k, scheme in kept:
        for detector in ctx['detectors']:
            try:
                sc = _scores(nb[s.name], k, scheme, detector)
                w = weights_from_scores(sc, strength=WEIGHT_STRENGTH,
                                        method=WEIGHT_METHOD)
                cleaned = cleaned_from_weights(X_tr, y_tr_noisy, w)
            except Exception as exc:
                print(f"      {detector} weights failed: {exc}")
                cleaned = None
            dprime_rows(sink, dataset_name=ds, cleaned=cleaned, X_tr=X_tr,
                        y_D=y_tr_noisy, X_va=X_va, y_va=y_val,
                        metric=ctx['metric'], task_type=ctx['task_type'],
                        y_clean=y_tr_clean, models=(model_name,),
                        is_clf=is_clf,
                        identity=row_identity(
                            method=f'{detector}_weight',
                            method_setting=(f'k={k}|scheme={scheme}|'
                                            f'strength={WEIGHT_STRENGTH}|'
                                            f'shape={WEIGHT_METHOD}'),
                            similarity_setting=s.name, **base_id))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _condition_levels(data, dataset_name, condition):
    """The decided levels of one damage kind on this dataset, level 0 first
    (noise_helpers.decided_levels; KNN_SEARCH_SPACE_RECORD "The noise grid",
    Adelaide 2026-10-05)."""
    from experiments.noise.noise_helpers import decided_levels
    return tuple(float(x) for x in decided_levels(data, dataset_name,
                                                  condition))


def run(data, config, output_dir, dataset_name, **kwargs):
    """N21 across every decided damage kind where a subset is damaged.

    The condition is an axis this experiment sweeps; the sweep, the
    per-condition files, the resume and the stride slice are
    noise_helpers.sweep_conditions. Average precision is n21's headline, so
    only the kinds that damage a SUBSET of the training labels are swept
    (conditions_for_task with detection as the headline): under gaussian and
    grouped_shifted every training label moves and average precision is 1.0
    by construction.

    2026-10-05 (chat 5): the condition files are fingerprinted with the
    levels, the draws, the outer fold count, the folds run and RESULT_LAYOUT,
    so a file written by the whole-dataset design is recomputed, not skipped.
    """
    from experiments.noise.noise_helpers import noise_n_folds
    conditions = list(conditions_for_task(data, detection_is_the_headline=True))
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: list(_condition_levels(data, dataset_name, c)) for c in conditions}
    fp_config['noise_seeds'] = list(NOISE_SEEDS)
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, EXP_FULL_NAME,
        data, fp_config, output_dir, dataset_name,
        default_conditions=conditions, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N21, ONE damage kind at every decided level. What each fold does:

      1. noise_helpers.fold_configurations: searches on the fold's training
         rows (clean labels, decided grid, pruning off) give each of the
         dataset's two decided models its 2 base representations, its 2
         general hybrids, and the recipes of its distance hybrid and
         similarity hybrid.
      2. Per (level, draw): the training labels are damaged
         (noise_helpers.inject_condition); each model's hybrids are rebuilt
         from the damaged labels (noise_helpers.fold_model_inputs); each of
         its 4 configurations is scored with clean and with damaged labels;
         then every distance -- per base representation and per hybrid
         space -- at every neighbour count, counting scheme and detector, is
         graded against the molecules that moved, and its weights refit the
         configurations it repairs at the 3 (k, scheme) its inner hold-out
         screen ranks best.
      3. The shared similarity axis (_sweep_pass) for each configuration.
    Each (fold, level, draw) is written to a checkpoint as it finishes.
    """
    from experiments._shared import input_fingerprint
    from experiments.decided_grid import decided_models
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_configurations, fold_model_inputs,
        configuration_record, inject_condition, dataset_knn_group, level_unit,
        fold_spread, draws_for, CONFIGURATION_SPACES)

    print("\n" + "=" * 70)
    print("EXPERIMENT N21: WHICH DISTANCE ARE THE NEIGHBOURS FOUND IN")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # auc is scored on the positive-class probability (noise_cache.
    # predict_mode_for), as n07._predict_for_metric did here before.
    predict_mode = _nc.predict_mode_for(metric_name)

    condition = kwargs.get('noise_condition') or NOISE_CONDITION
    allowed = list(conditions_for_task(data, detection_is_the_headline=True))
    if condition not in allowed:
        raise ValueError(
            f"n21: average precision is the headline here, and it needs a "
            f"damaged SUBSET of the training molecules. {condition!r} is not "
            f"one of the decided kinds that damage a subset on this dataset: "
            f"{allowed}.")
    for _k in _RETIRED_KWARGS:
        if kwargs.get(_k) is not None:
            print(f"  n21: '{_k}' is no longer read (2026-10-05, chat 5): "
                  f"the decided grid sets it")

    levels = _condition_levels(data, dataset_name, condition)
    importance_methods = list(kwargs.get('importance_methods')
                              or importance_methods_for(dataset_name))
    metrics = list(kwargs.get('metrics') or DISTANCE_METRICS_SWEPT)
    exponents = [float(e) for e in (kwargs.get('importance_exponents')
                                    or IMPORTANCE_EXPONENTS)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    schemes = list(sweep_schemes())
    k_max = max(K_VALUES)
    forest_kw = {
        'n_trees': int(kwargs.get('forest_trees', FOREST_TREES)),
        'min_samples_leaf': int(kwargs.get('forest_min_samples_leaf',
                                           FOREST_MIN_SAMPLES_LEAF)),
        'seed': int(kwargs.get('forest_seed', FOREST_SEED)),
    }
    knn_grp = dataset_knn_group(data, dataset_name)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    noise_groups = noise_groups_for(data)[0]

    # R1: the dataset's two decided models (DECISIONS.md Step 1), nothing
    # else. knn runs only inside the distance search. Replaces
    # config['models'] / DOWNSTREAM_MODELS_NOISE (five models) and rf's
    # ranking of the single representations.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    folds, n_folds = noise_outer_folds(data, config)

    print(f"  Dataset: {dataset_name} ({len(labels)} rows), group {knn_grp}")
    print(f"  Metric: {metric_name} "
          f"({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Models (decided): {models}")
    print(f"  Condition {condition}, levels {list(levels)} "
          f"({level_unit(condition, is_clf)}), draws "
          f"{list(draws_for(condition, NOISE_SEEDS))} per non-zero level")
    print(f"  Outer folds: {n_folds} (running {[f for f, _, _ in folds]})")
    print(f"  Own distances: metrics {metrics}, importance {importance_methods}"
          f" at exponents {exponents}, forest proximity, hybrid spaces "
          f"{list(NEIGHBOUR_HYBRID_SPACES)}")
    print(f"  Detectors {detectors}, k values {K_VALUES}, counting schemes "
          f"{schemes}")

    # The shared similarity axis: the neighbour settings this dataset's
    # group sweeps (KNN_SEARCH_SPACE_RECORD): no model_input, no importance
    # weighting, distances per group, scaffold-mates kept.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    # A new rows file: rows written by the whole-dataset design share model
    # input names with these and would be skipped as done.
    sink = RowSink(output_dir, EXP_FULL_NAME, dataset_name,
                   f'{condition}__{RESULT_LAYOUT}')
    sweep_ctx = sweep_context(
        experiment=EXP_FULL_NAME, dataset_name=dataset_name, sink=sink,
        settings=sim_settings, metric=metric, task_type=task_type,
        condition=condition, level=None, data=data, factories={})
    sweep_ctx['detectors'] = detectors
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x schemes {schemes}, top {screen_top()} "
          f"refitted; rows -> {sink.path}")

    ckpt = FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name, condition=condition,
        tag=f'units_{RESULT_LAYOUT}', fingerprint={
            'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
            'condition': condition, 'levels': list(levels),
            'n_folds': int(n_folds), 'models': models,
            'importance_methods': importance_methods, 'metrics': metrics,
            'exponents': exponents, 'detectors': detectors,
            'k_values': list(K_VALUES), 'schemes': schemes,
            'forest': forest_kw, 'seeds': list(NOISE_SEEDS),
            'repair': [WEIGHT_METHOD, WEIGHT_STRENGTH],
            'settings': [s.name for s in sim_settings],
            'screen_top': screen_top(),
            'own_refit_top': OWN_REFIT_TOP,
            'draws': list(draws_for(condition, NOISE_SEEDS))})

    fold_records = {}
    units_done = []      # [{'fold', 'level', 'draw', 'value'}]
    t0 = time.time()

    for fold_i, train_idx, val_idx in folds:
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        # n_samples = this fold's training rows, which is what the per-fold
        # search sized each model by.
        factories = get_model_factories(models, task_type=task_type,
                                        n_samples=len(train_idx))
        sweep_ctx['factories'] = factories
        y_tr_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]

        # Each model's two base configurations. A single representation
        # chooses no features, so they are the same matrices at every level.
        base_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            factory = factories.get(m)
            if entry is None or factory is None:
                why = ('the per-fold search failed for this model'
                       if entry is None else
                       'no model factory at this training size')
                frec['models'][m] = dict(configuration_record(entry),
                                         skipped=why)
                print(f"    {m}: skipped -- {why}")
                continue
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr_clean, factory=factory,
                metric=metric, spaces=BASE_SPACES)
            for i, sp in enumerate(HYBRID_FIT_SPACES):
                if len(entry.get('hybrids') or []) > i:
                    names[sp] = entry['hybrids'][i].get('label')
            base_inputs[m] = (entry, ins, unav, names)
            rec = configuration_record(entry, names)
            rec['base_configurations_unavailable'] = dict(unav)
            rec['takes_sample_weight'] = bool(_takes_sample_weight(factory))
            rec['hybrid_space_recipes'] = {
                sp: {'objective': getattr(bp, 'objective', None) or
                     'predictive',
                     'importance_method': getattr(bp, 'importance_method',
                                                  None),
                     'allocation_method': getattr(bp, 'allocation_method',
                                                  None),
                     'budget': getattr(bp, 'budget', None),
                     'internal_evaluator': getattr(bp, 'internal_evaluator',
                                                   None),
                     'n_reps': len(getattr(bp, 'reps', None) or []),
                     'aug_condition': aug}
                for sp, (bp, aug) in (entry.get('recipes') or {}).items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            # Training-row score per fitted configuration (handoff 2.4 item
            # 11, 2026-10-05): the per-fold search's own mean over its inner
            # folds on this fold's training rows. The hybrids carry it as
            # score_mean. The base representations' scores are ranked inside
            # fold_standard_selection but fold_neighbour_space_recipes does
            # not pass them on, so they are recorded as absent with why.
            trs = {}
            for i, sp in enumerate(BASE_SPACES):
                trs[sp] = {'score': None, 'absent_because': (
                    'fold_neighbour_space_recipes drops the base '
                    'representation scores of the per-fold search')}
            for i, sp in enumerate(HYBRID_FIT_SPACES):
                hs = entry.get('hybrids') or []
                if len(hs) > i:
                    trs[sp] = {'score': hs[i].get('score_mean'),
                               'label': hs[i].get('label')}
            rec['training_row_score'] = trs
            rec['training_row_score_basis'] = (
                "the per-fold search's mean over its inner folds, on this "
                "outer fold's training rows, clean labels")
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}, "
                  f"neighbour spaces {rec.get('neighbour_spaces')}")
        fold_records[str(fold_i)] = frec

        def _unit(level, draw, _tr=train_idx, _va=val_idx, _ytr=y_tr_clean,
                  _yva=y_val, _g=g_tr, _bi=base_inputs, _fold=fold_i,
                  _fac=factories):
            y_noisy, cmask, prov = inject_condition(
                data, dataset_name, condition, level, draw, train_idx=_tr,
                groups=noise_groups)
            cmask = np.asarray(cmask, dtype=bool)
            det_mask = cmask if cmask.any() else None
            pass_key = f'f{_fold}s{draw}'
            val = {'provenance': prov, 'n_corrupted': int(cmask.sum()),
                   'n_training': int(cmask.size),
                   'clean': {}, 'noisy': {}, 'detection': {}, 'repair': {},
                   'spaces': {}, 'zero_importance': {}, 'errors': {},
                   'unavailable': {}, 'weighted_undefined': {},
                   'hybrid_spaces_built': {}, 'sim_detection': {}}
            sweep_ctx['level'] = float(level)
            sweep_ctx['screen'] = []

            # ── The fitted configurations of every model (R5), and their
            #    clean and damaged references.
            unit_inputs, model_of_unit, can_weight = {}, {}, {}
            hyb_by_model = {}
            for m, (entry, base_ins, base_unav, _names) in _bi.items():
                factory = _fac[m]
                can_weight[m] = _takes_sample_weight(factory)
                if not can_weight[m]:
                    # DECIDED 2026-10-06 (Adelaide delegated): gp (a decided
                    # model on abagym_Ang2, abagym_VEGF, ginkgo_acsins,
                    # ginkgo_hic, ginkgo_pr_cho) takes no per-row weight, so
                    # its repaired fits are "not run". Detection is still
                    # graded.
                    val['weighted_undefined'][m] = (
                        "not run: this model's fit() takes no sample_weight")
                # The hybrid spaces, rebuilt from this draw's damaged labels.
                hyb, hyb_unav, _n = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=_tr, val_idx=_va,
                    y_train=y_noisy, factory=factory, metric=metric,
                    spaces=NEIGHBOUR_HYBRID_SPACES)
                hyb_by_model[m] = hyb
                val['hybrid_spaces_built'][m] = sorted(hyb)
                for sp, why in list(base_unav.items()) + list(hyb_unav.items()):
                    val['unavailable'].setdefault(m, {})[sp] = why
                for sp in CONFIGURATION_SPACES:
                    pair = base_ins.get(sp) if sp in BASE_SPACES else hyb.get(sp)
                    if pair is not None:
                        unit_inputs[f'{m}@{sp}'] = pair
                        model_of_unit[f'{m}@{sp}'] = m
            # A hybrid configuration's clean score is the clean labels in the
            # features chosen from this draw's damaged labels.
            for u, (xtr, xva) in unit_inputs.items():
                m = model_of_unit[u]
                for tag, yy in (('clean', _ytr), ('noisy', y_noisy)):
                    try:
                        pred = _nc.cached_fit_predict(
                            dataset_name, m, _fac[m], xtr, yy, xva,
                            predict_mode=predict_mode)
                        val[tag][u] = float(metric_fn(_yva, pred))
                    except Exception as exc:
                        val[tag][u] = None
                        val['errors'][f'{u}|{tag}'] = (
                            f'{type(exc).__name__}: {exc}')

            # ── Every distance of this pass, and which configurations its
            #    weights repair.
            built, repairs = {}, {}
            rep_units = {}     # representation name -> [units fitted on it]
            rep_mats = {}      # representation name -> (raw, scaled) training
            for m, (entry, base_ins, _bu, _names) in _bi.items():
                for i, sp in enumerate(BASE_SPACES):
                    rn = ((entry.get('base') or []) + [None, None])[i]
                    if rn is None or f'{m}@{sp}' not in unit_inputs:
                        continue
                    rep_units.setdefault(rn, []).append(f'{m}@{sp}')
                    rep_mats.setdefault(rn, (
                        np.asarray(data['reps'][rn])[_tr], base_ins[sp][0]))
            for rn, (X_raw, X_std) in rep_mats.items():
                b, s, z = _representation_spaces(
                    rn, X_raw, X_std, y_noisy, is_clf, importance_methods,
                    metrics, exponents, forest_kw, dataset_name,
                    val['errors'], pass_key)
                built.update(b)
                val['spaces'].update(s)
                val['zero_importance'].update(z)
                for key in b:
                    repairs[key] = list(rep_units[rn])
            for m, hyb in hyb_by_model.items():
                base_names = [rn for rn in (_bi[m][0].get('base') or [])
                              if rn in rep_mats]
                b, s = _hybrid_spaces(m, hyb, base_names, metrics)
                built.update(b)
                val['spaces'].update(s)
                for key in b:
                    repairs[key] = [u for u in unit_inputs
                                    if model_of_unit[u] == m]
            for key, units in repairs.items():
                val['spaces'][key]['repairs'] = list(units)

            # ── Grade every distance, then repair with it.
            # DECIDED 2026-10-06 (Adelaide delegated): detection is graded at
            # every (k, scheme, detector) of every distance; each distance
            # refits only at the 3 of its 6 (k, scheme) its inner hold-out
            # screen ranks best (_screen_own_distance), both detectors. With
            # 11 distances per base representation and 8 hybrid spaces per
            # model, that is about 4 x 11 x 6 + 2 x 8 x 4 x 6 = 648 fits per
            # (fold, level, draw) where it was 1,296.
            held = inner_holdout_mask(_g, len(y_noisy))
            val['own_screen'] = {}
            rows = []
            for key, (kind, payload, space_metric) in built.items():
                try:
                    if kind == 'features':
                        idx_all, dist_all = _cached_neighbours_in_space(
                            dataset_name, payload, k_max, space_metric)
                    else:
                        idx_all, dist_all = neighbours_in_distance_matrix(
                            payload, k_max, return_distance=True)
                except Exception as exc:
                    val['errors'][f'{pass_key}|{key}|neighbours'] = \
                        f'{type(exc).__name__}: {exc}'
                    continue
                refit_at = set()
                if any(can_weight[model_of_unit[u]]
                       for u in repairs.get(key, [])):
                    spec = val['spaces'].get(key) or {}
                    try:
                        kept_ks, s_table = _screen_own_distance(
                            dataset_name, kind, payload, space_metric,
                            held=held, y_noisy=y_noisy, is_clf=is_clf,
                            metric=metric, ks=K_VALUES, schemes=schemes,
                            top=OWN_REFIT_TOP,
                            forest_X=(rep_mats[spec['representation']][1]
                                      if kind == 'distance' else None),
                            forest_kw=forest_kw, idx_all=idx_all,
                            dist_all=dist_all)
                        refit_at = set(kept_ks)
                        val['own_screen'][key] = {
                            'kept': [f'k={k_}|{s_}' for k_, s_ in kept_ks],
                            'table': s_table}
                    except Exception as exc:
                        val['errors'][f'{pass_key}|{key}|own_screen'] = \
                            f'{type(exc).__name__}: {exc}'
                for k in K_VALUES:
                    kk = int(min(k, idx_all.shape[1]))
                    for sch in schemes:
                        for detector in detectors:
                            est_key = f'{detector}_k{k}|{sch}'
                            try:
                                scores = suspicion_from_neighbours(
                                    idx_all[:, :kk], y_noisy, detector, is_clf,
                                    dist=dist_all[:, :kk], scheme=sch)
                                weights = weights_from_scores(
                                    scores, strength=WEIGHT_STRENGTH,
                                    method=WEIGHT_METHOD)
                            except Exception as exc:
                                val['errors'][f'{pass_key}|{key}|{est_key}'] = \
                                    f'{type(exc).__name__}: {exc}'
                                continue
                            if det_mask is not None:
                                det = _detection(det_mask, scores, condition,
                                                 is_clf)
                                if det is not None:
                                    val['detection'].setdefault(key, {})[
                                        est_key] = {
                                        'ap': det['detection_average_precision'],
                                        'pk': det['detection_precision_at_k']}
                            if (k, sch) not in refit_at:
                                continue
                            for u in repairs.get(key, []):
                                m = model_of_unit[u]
                                if not can_weight[m]:
                                    continue
                                xtr, xva = unit_inputs[u]
                                try:
                                    pred = _nc.cached_fit_predict(
                                        dataset_name, m, _fac[m], xtr,
                                        y_noisy, xva, sample_weight=weights,
                                        predict_mode=predict_mode)
                                    sc = float(metric_fn(_yva, pred))
                                except Exception as exc:
                                    val['errors'][f'{key}|{u}|{est_key}'] = \
                                        f'{type(exc).__name__}: {exc}'
                                    continue
                                val['repair'].setdefault(key, {})[
                                    f'{u}|{est_key}'] = sc
                                # The repaired fit IS a D': one row beside the
                                # damaged fit and the clean one, from numbers
                                # already computed (no refit).
                                rows.append(((detector, k, sch, key, u, m),
                                             {'score_D': val['noisy'].get(u),
                                              'score_Dprime': sc,
                                              'score_clean':
                                                  val['clean'].get(u)}))
            if sink.enabled and rows:
                from experiments.noise.d_vs_dprime import rows_for
                out_rows = []
                for (detector, k, sch, key, u, m), s in rows:
                    ident = row_identity(
                        dataset_name=dataset_name, experiment=EXP_FULL_NAME,
                        method=f'{detector}_weight',
                        method_setting=(f'k={k}|scheme={sch}|'
                                        f'strength={WEIGHT_STRENGTH}|'
                                        f'shape={WEIGHT_METHOD}'),
                        # The prefix: this file's own distances are found by
                        # its own code under its own metric names and are not
                        # claimed to be any shared setting.
                        similarity_setting=f'n21:{key}',
                        model_input=f"{m}:{u.split('@', 1)[1]}",
                        noise_condition=condition, level=float(level),
                        draw=int(draw), fold=int(_fold))
                    if not sink.is_done(ident, (m,)):
                        out_rows += rows_for(
                            {m: {k_: (float('nan') if v is None else v)
                                 for k_, v in s.items()}},
                            metric=metric, **ident)
                sink.write(out_rows)

            # ── The shared similarity axis, once per fitted configuration.
            # The lists of every space but forest_proximity do not depend on
            # the configuration, so detection is graded on each model's first
            # configuration and only forest_proximity on the others.
            seen_model = set()
            for u, (xtr, xva) in unit_inputs.items():
                m = model_of_unit[u]
                first = m not in seen_model
                seen_model.add(m)
                _sweep_pass(
                    sweep_ctx, model_name=m,
                    model_input=f"{m}:{u.split('@', 1)[1]}",
                    X_tr=xtr, X_va=xva, y_tr_noisy=y_noisy, y_tr_clean=_ytr,
                    y_val=_yva, train_idx=_tr, val_idx=_va, groups_tr=_g,
                    fold=_fold, draw=draw, entry=_bi[m][0],
                    corrupted_mask=det_mask,
                    detection_acc=val['sim_detection'].setdefault(u, {}),
                    grade_spaces=None if first else ('forest_proximity',),
                    can_weight=can_weight[m])
            val['screen'] = sweep_ctx.pop('screen', [])
            return val

        for level in levels:
            draws = (NOISE_SEEDS[:1] if float(level) == 0.0
                     else draws_for(condition, NOISE_SEEDS))
            for draw in draws:
                key = ['unit', int(fold_i), float(level), int(draw)]
                v = ckpt.get(key)
                if v is None:
                    t_u = time.time()
                    print(f"    level {level:g}, draw {draw}")
                    v = _unit(float(level), int(draw))
                    ckpt.put(key, v)
                    print(f"    level {level:g}, draw {draw}: "
                          f"{time.time() - t_u:.0f}s, "
                          f"{len(v.get('errors') or {})} failures "
                          f"[{time.time() - t0:.0f}s]")
                else:
                    print(f"    level {level:g}, draw {draw}: read back from "
                          f"{ckpt.path}")
                units_done.append({'fold': int(fold_i), 'level': float(level),
                                   'draw': int(draw), 'value': v})

    # Held-out score per fold (handoff 2.4 item 11, 2026-10-05): each fitted
    # configuration's score on the fold's held-out molecules, with clean and
    # with damaged training labels, per level and draw. per_level's means
    # over folds are computed from these.
    for u in units_done:
        fr = fold_records.get(str(u['fold']))
        if fr is None or 'error' in fr:
            continue
        v = u['value']
        (fr.setdefault('held_out_score', {})
         .setdefault(f"{u['level']:g}", {}))[f"draw{u['draw']}"] = {
            'clean_training_labels': dict(v.get('clean') or {}),
            'damaged_training_labels': dict(v.get('noisy') or {})}
    for fr in fold_records.values():
        if 'held_out_score' in fr:
            fr['held_out_score_key'] = (
                "level -> draw -> clean_training_labels / "
                "damaged_training_labels -> '<model>@<configuration>'; "
                "in the dataset's own metric")

    results = {
        'experiment': EXP_FULL_NAME,
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'task_type': task_type,
        'noise_condition': condition,
        'levels': list(levels),
        'level_unit': level_unit(condition, is_clf),
        'noise_seeds': list(draws_for(condition, NOISE_SEEDS)),
        'draws_per_nonzero_level': len(draws_for(condition, NOISE_SEEDS)),
        'level_zero': 'the labels as delivered, run once (one draw)',
        'knn_group': knn_grp,
        'robustness_reported_as': (
            "damaged_minus_clean: the held-out score with damaged training "
            "labels minus the score with clean ones, per fitted configuration "
            "and level, in the dataset's own metric (see lower_is_better for "
            "its direction). This file never computed auc_norm."),
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _t, _v in folds],
        'selection_scope': 'fold_train_rows',
        'models': models,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'n_fitted_configurations': len(models) * len(CONFIGURATION_SPACES),
        'neighbour_hybrid_spaces': list(NEIGHBOUR_HYBRID_SPACES),
        'folds': fold_records,
        'k_values': list(K_VALUES),
        'schemes': schemes,
        'detectors': detectors,
        'importance_methods': importance_methods,
        'importance_methods_rule': (
            "the dataset's decided importance methods (decided_grid, "
            "'general' search)"),
        'own_refits': (
            f"each own distance refits at the {OWN_REFIT_TOP} of its "
            f"{len(K_VALUES) * len(schemes)} (k, scheme) combinations its "
            "inner hold-out screen ranks best (a fifth of the training "
            "molecules, as whole scaffold groups, predicted from their "
            "neighbours among the rest, against the damaged labels, in the "
            "dataset's own metric), both detectors. Detection is graded at "
            "every combination. Per unit: 'own_screen' in the checkpoint."),
        'own_screen': [
            {'fold': u['fold'], 'level': u['level'], 'draw': u['draw'],
             'kept': {key: e['kept'] for key, e in
                      (u['value'].get('own_screen') or {}).items()}}
            for u in units_done],
        'importance_exponents': exponents,
        'distance_metrics_swept': metrics,
        'exponent_matched_to_metric': dict(EXPONENT_MATCHED_TO_METRIC),
        'forest': dict(forest_kw, fitted_on=(
            'the fold\'s training molecules only, on the scaled single '
            'representation, against the damaged training labels'),
            distance='one minus the share of trees in which two molecules '
                     'land in the same leaf'),
        'weight_strength': WEIGHT_STRENGTH,
        'weight_method': WEIGHT_METHOD,
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
        'space_key': (
            "'<representation>/<distance>' for a distance built on a base "
            "representation (shared by every model holding it on that fold); "
            "'<model>/<hybrid space>|<metric>' for a hybrid space. Each "
            "space's 'repairs' names the fitted configurations its weights "
            "refit, as '<model>@<configuration>'."),
        'repair_key': "'<model>@<configuration>|<detector>_k<k>|<scheme>'",
        'exclude_groups': (
            'not used (KNN_SEARCH_SPACE_RECORD axis 6): scaffold-mates and '
            'repeat measurements of a molecule stay in its neighbour list; '
            'the molecule itself is dropped by index'),
        'recipe_note': (
            "Each fold's configurations and neighbour-space recipes come from "
            "searches on that fold's training rows with clean labels "
            "(noise_helpers.fold_configurations). Each hybrid's features are "
            "re-chosen from the damaged labels at every draw."),
        'similarity': similarity_record(
            sim_settings, sink, schemes_swept=schemes,
            screen_top=screen_top(), screen_scored_on=SCREEN_SCORED_ON,
            knn_group=knn_grp, spaces_unavailable=sweep_ctx['unavailable'],
            repair=f'{WEIGHT_METHOD} at strength {WEIGHT_STRENGTH}'),
        'screen': [dict(e, fold=u['fold'], level=u['level'])
                   for u in units_done
                   for e in (u['value'].get('screen') or [])],
        'checkpoint_file': ckpt.path,
        'n_units_resumed': ckpt.n_resumed,
        'rows': sink.summary(),
    }
    spaces = {}
    for u in units_done:
        for key, spec in (u['value'].get('spaces') or {}).items():
            spaces.setdefault(key, dict(spec))
    results['spaces'] = spaces
    results['per_level'] = {
        f'{lv:g}': convert_numpy(_level_block(
            [u for u in units_done if u['level'] == lv], lower_is_better,
            fold_spread))
        for lv in levels}
    results['summary'] = _summarise(results, spaces, metrics,
                                    importance_methods, exponents,
                                    lower_is_better)
    _print_summary(results)
    return convert_numpy(results)


# ---------------------------------------------------------------------------
# Reading the units back
# ---------------------------------------------------------------------------

def _mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float('nan')


def _by_fold(pairs):
    """[(fold, value)] -> {fold: mean over that fold's draws}."""
    acc = {}
    for f, v in pairs:
        if v is None or not np.isfinite(float(v)):
            continue
        acc.setdefault(f, []).append(float(v))
    return {f: float(np.mean(v)) for f, v in acc.items() if v}


def _recovery(c, nz, s, lower_is_better):
    """(recovery percent of the clean-to-corrupted gap, note)."""
    if any(x is None or not np.isfinite(x) for x in (c, nz)):
        return float('nan'), 'not computed: a score was not finite'
    gap = (nz - c) if lower_is_better else (c - nz)
    floor = max(1e-10, GAP_MIN_FRAC * abs(c))
    if not abs(gap) > floor:
        return float('nan'), (f'not computed: the clean-to-corrupted gap '
                              f'({gap:.6g}) is under {GAP_MIN_FRAC:.0%} of '
                              f'the clean score ({c:.6g})')
    if s is None or not np.isfinite(s):
        return float('nan'), 'not computed: no repaired score'
    return float((s - nz) / (c - nz) * 100.0), None


def _level_block(lv_units, lower_is_better, fold_spread):
    """One level's readings. Every number is averaged over draws within a
    fold, then summarised over folds with fold_spread (mean and spread, or
    'one fold, no spread'). No significance test.

    `clean_score_by_model` / `noisy_score_by_model` are keyed by fitted
    configuration ('<model>@<configuration>'), the shape
    noise_ceiling_analyse._read_n21 reads one curve per key from.
    """
    def _pass(u):
        return f"f{u['fold']}s{u['draw']}"

    clean_f, noisy_f = {}, {}
    det_f, det_cells = {}, {}
    rep_f, rep_cells = {}, {}
    sim_f = {}
    counts = {'n_corrupted': [], 'n_training': []}
    zero, errors, unavailable, provenance = {}, {}, {}, {}
    for u in lv_units:
        v = u['value']
        f, pk = u['fold'], _pass(u)
        counts['n_corrupted'].append(v.get('n_corrupted'))
        counts['n_training'].append(v.get('n_training'))
        for unit, s in (v.get('clean') or {}).items():
            clean_f.setdefault(unit, []).append((f, s))
        for unit, s in (v.get('noisy') or {}).items():
            noisy_f.setdefault(unit, []).append((f, s))
        for sp, by_est in (v.get('detection') or {}).items():
            for est, d in by_est.items():
                det_f.setdefault(sp, {}).setdefault(est, []).append(
                    (f, d.get('ap'), d.get('pk')))
                det_cells.setdefault(sp, {})[f'{est}|{pk}'] = d.get('ap')
        for sp, by_cell in (v.get('repair') or {}).items():
            for cell, s in by_cell.items():
                rep_f.setdefault(sp, {}).setdefault(cell, []).append((f, s))
                rep_cells.setdefault(sp, {})[f'{cell}|{pk}'] = s
        for unit, by_set in (v.get('sim_detection') or {}).items():
            for setting, by_ks in by_set.items():
                for ks, by_det in by_ks.items():
                    for det, ap in by_det.items():
                        (sim_f.setdefault(setting, {}).setdefault(ks, {})
                         .setdefault(det, []).append((f, ap)))
        for sp in (v.get('zero_importance') or {}):
            zero[sp] = zero.get(sp, 0) + 1
        for ek, why in (v.get('errors') or {}).items():
            errors[f'{pk}|{ek}'] = why
        for m, by_sp in (v.get('unavailable') or {}).items():
            unavailable.setdefault(m, {}).update(by_sp)
        if v.get('provenance'):
            provenance[pk] = v['provenance']

    clean_bf = {k: _by_fold(p) for k, p in clean_f.items()}
    noisy_bf = {k: _by_fold(p) for k, p in noisy_f.items()}
    units = sorted(set(clean_bf) | set(noisy_bf))
    clean_by_unit = {k: _mean((clean_bf.get(k) or {}).values()) for k in units}
    noisy_by_unit = {k: _mean((noisy_bf.get(k) or {}).values()) for k in units}
    damaged_minus_clean = {}
    for k in units:
        common = sorted(set(clean_bf.get(k) or {}) & set(noisy_bf.get(k) or {}))
        damaged_minus_clean[k] = fold_spread(
            [noisy_bf[k][f] - clean_bf[k][f] for f in common])

    per_space = {}
    for sp in sorted(set(det_f) | set(rep_f)):
        per_est = {}
        for est, trip in (det_f.get(sp) or {}).items():
            ap_bf = _by_fold([(f, a) for f, a, _p in trip])
            pk_bf = _by_fold([(f, p) for f, _a, p in trip])
            per_est[est] = {
                'average_precision': fold_spread(
                    [ap_bf[f] for f in sorted(ap_bf)]),
                'precision_at_k': fold_spread(
                    [pk_bf[f] for f in sorted(pk_bf)]),
                'n_passes': len(trip)}
        repair = {}
        for cell, pairs in (rep_f.get(sp) or {}).items():
            unit, est = cell.split('|', 1)
            s_bf = _by_fold(pairs)
            c_bf, n_bf = clean_bf.get(unit) or {}, noisy_bf.get(unit) or {}
            rec_by_fold = []
            for f in sorted(s_bf):
                r, _note = _recovery(c_bf.get(f), n_bf.get(f), s_bf[f],
                                     lower_is_better)
                rec_by_fold.append(r)
            denoised = _mean(s_bf.values())
            c, nz = clean_by_unit.get(unit), noisy_by_unit.get(unit)
            r_mean, r_note = _recovery(c, nz, denoised, lower_is_better)
            repair[cell] = {
                'configuration': unit, 'detector': est,
                'denoised_score': fold_spread([s_bf[f] for f in sorted(s_bf)]),
                'corrupted_score': nz, 'clean_score': c,
                'recovery_percent_of_gap_from_fold_means': r_mean,
                'recovery_note': r_note,
                'recovery_percent_of_gap_by_fold': fold_spread(rec_by_fold),
                'n_passes': len(pairs)}
        row = {
            'detection_per_detector': per_est,
            'detection_average_precision_mean': _mean(
                [e['average_precision']['mean'] for e in per_est.values()]),
            'repair': repair,
            'recovery_percent_of_gap_mean': _mean(
                [r['recovery_percent_of_gap_from_fold_means']
                 for r in repair.values()]),
            'detection_average_precision_per_pass': dict(
                det_cells.get(sp) or {}),
            'repair_score_per_pass': dict(rep_cells.get(sp) or {}),
        }
        if sp in zero:
            row['builds_with_all_zero_importance'] = zero[sp]
            row['builds_with_all_zero_importance_note'] = (
                'the importance was zero on every feature in this many builds, '
                'so the weighted space was the plain space and these numbers '
                'are a duplicate of the plain row')
        per_space[sp] = row

    similarity_axis = {}
    for setting, by_ks in sim_f.items():
        for ks, by_det in by_ks.items():
            for det, pairs in by_det.items():
                bf = _by_fold(pairs)
                (similarity_axis.setdefault(f'{setting}|{ks}', {}))[det] = {
                    'average_precision': fold_spread(
                        [bf[f] for f in sorted(bf)]),
                    'n_passes': len(pairs)}

    block = {
        'molecules_corrupted_mean': _mean(counts['n_corrupted']),
        'molecules_in_training_fold_mean': _mean(counts['n_training']),
        'clean_score_by_model': clean_by_unit,
        'clean_score_by_fold': clean_bf,
        'noisy_score_by_fold': noisy_bf,
        'noisy_score_by_model': noisy_by_unit,
        'damaged_minus_clean': damaged_minus_clean,
        'by_model_scores_note': (
            "keyed by fitted configuration, '<model>@<configuration>'; "
            "damaged_minus_clean is per fold (draw means), summarised over "
            "folds, in the dataset's own metric"),
        'per_space': per_space,
        'similarity_axis': similarity_axis,
    }
    if unavailable:
        block['unavailable'] = unavailable
    if errors:
        block['_errors'] = errors
    if provenance:
        block['noise_provenance'] = provenance
    return block


def _paired(a, b, higher_is_better):
    """Per-cell gains of `a` over `b`, over the cells both scored.

    A cell is one (detector, k, scheme, fold, draw) -- or one (configuration,
    detector, k, scheme, fold, draw) for the repair scores. So every gain
    compares two distances on the same molecules, the same corruption and the
    same detector settings.
    """
    shared = sorted(set(a) & set(b))
    g = np.array([(a[c] - b[c]) if higher_is_better else (b[c] - a[c])
                  for c in shared
                  if a[c] is not None and b[c] is not None], dtype=float)
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


def _summarise(results, specs, metrics, importance_methods, exponents,
               lower_is_better):
    """Two paired comparisons per level, plus the convention question.

    Everything is paired. Two distances are compared on the same fold, the same
    corruption draw, the same detector, the same k and the same counting
    scheme, so nothing but the distance differs between the two sides of a
    gain. A repair gain is also paired on the configuration refitted.
    """
    out = {
        'headline': ('detection average precision, each distance against the '
                     'plain one, paired'),
        'what_a_cell_is': (
            'detection: one (detector, k, scheme, fold, draw). repair: one '
            '(configuration, detector, k, scheme, fold, draw).'),
        'per_level': {},
    }
    ap_cell = 'one (detector, k, scheme, fold, draw)'
    rep_cell = 'one (configuration, detector, k, scheme, fold, draw)'

    for lvl, block in results['per_level'].items():
        ap = {s: r['detection_average_precision_per_pass']
              for s, r in block['per_space'].items()}
        rp = {s: r['repair_score_per_pass']
              for s, r in block['per_space'].items()}

        beats_plain = {}
        for space_key, spec in specs.items():
            if spec['family'] == 'plain':
                continue
            for ref in spec.get('plain_references') or []:
                if ref not in ap and ref not in rp:
                    continue
                beats_plain[f'{space_key} vs {ref}'] = {
                    'detection_average_precision': _compare(
                        ap.get(space_key, {}), ap.get(ref, {}), True, ap_cell),
                    'repair_score': _compare(
                        rp.get(space_key, {}), rp.get(ref, {}),
                        not lower_is_better, rep_cell),
                }

        # The convention question: at one representation, one metric and one
        # importance method, which exponent finds the corrupted molecules
        # better.
        convention = {}
        reps_seen = sorted({sp['representation'] for sp in specs.values()
                            if sp.get('representation')})
        for rn in reps_seen:
            for imp in importance_methods:
                for m in metrics:
                    keys = [f'{rn}/imp_pow{e:g}|{imp}|{m}' for e in exponents]
                    if not all(kk in block['per_space'] for kk in keys):
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
                        row[f'{exponents[0]:g} against {exponents[1]:g}'] = {
                            'detection_average_precision': _compare(
                                ap.get(keys[0], {}), ap.get(keys[1], {}), True,
                                ap_cell),
                            'repair_score': _compare(
                                rp.get(keys[0], {}), rp.get(keys[1], {}),
                                not lower_is_better, rep_cell),
                        }
                    if matched is not None and best is not None:
                        row['the_matched_exponent_won'] = bool(best == matched)
                    convention[f'{rn}/{imp}|{m}'] = row

        ranked = sorted(
            ((s, r['detection_average_precision_mean'])
             for s, r in block['per_space'].items()),
            key=lambda t: (-t[1] if np.isfinite(t[1]) else np.inf))

        out['per_level'][lvl] = {
            'average_precision_by_space': {s: v for s, v in ranked},
            'recovery_percent_of_gap_by_space': {
                s: r['recovery_percent_of_gap_mean']
                for s, r in block['per_space'].items()},
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
            print(f"    {space:56s} ap {ap:.4f}   recovery {rec:+7.1f}%")
        print("    against the plain distance, paired per (detector, k, "
              "scheme, fold, draw):")
        for name, cmp_row in row['beats_plain'].items():
            d = cmp_row['detection_average_precision']
            if not d.get('cells_paired'):
                print(f"      {name}: {d.get('absent_because')}")
                continue
            mark = '  >spread' if d['gain_exceeds_its_own_spread'] else ''
            print(f"      {name:70s} won {d['cells_won']} of "
                  f"{d['cells_paired']}, mean {d['mean_gain']:+.4f} "
                  f"(spread {d['spread_over_cell_gains']:.4f}){mark}")
        if row['weighting_convention']:
            print("    the scaling convention, at each metric:")
            for name, c in row['weighting_convention'].items():
                print(f"      {name:44s} matched exponent "
                      f"{c['exponent_matched_to_this_metric']}, higher average "
                      f"precision at "
                      f"{c['exponent_with_the_higher_average_precision']}"
                      + ("  MATCHED WON" if c.get('the_matched_exponent_won')
                         else ""))
    print("\n  A mean gain no larger than the spread of the per-cell gains "
          "behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N21: which distance are the nearest neighbours found in')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help='One damage kind. Default: every decided kind '
                             'that damages a subset of the training labels on '
                             'this dataset (outlier_p05, censoring, '
                             'shift_only_p10, sparse_dprime_up_p10 on '
                             'regression; the flip kinds on classification).')
    parser.add_argument('--metrics', type=str, default=None,
                        help='Comma-separated distance metrics for this '
                             "file's own distances. Default "
                             'euclidean,manhattan.')
    parser.add_argument('--importance-methods', type=str, default=None,
                        help="Comma-separated. Default: the dataset's "
                             'decided importance methods.')
    parser.add_argument('--forest-trees', type=int, default=FOREST_TREES)
    parser.add_argument('--forest-min-samples-leaf', type=int,
                        default=FOREST_MIN_SAMPLES_LEAF)
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {'forest_trees': args.forest_trees,
          'forest_min_samples_leaf': args.forest_min_samples_leaf}
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.metrics:
        kw['metrics'] = [x.strip() for x in args.metrics.split(',')]
    if args.importance_methods:
        kw['importance_methods'] = [x.strip() for x
                                    in args.importance_methods.split(',')]
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
