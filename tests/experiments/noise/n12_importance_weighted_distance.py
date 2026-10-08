#!/usr/bin/env python3
"""N12: Feature Importance-Weighted Distance for Noise Estimation.

Paper: Noise
SCOPE: SYNTHETIC noise only. This is a NARROW ablation, not a general claim:
does scaling features by importance before computing the neighbour distance
improve neighbour-based noise estimation, and is it redundant with
hybridization?

CAVEAT — READ BEFORE INTERPRETING RESULTS: the measured differences between
conditions are small and may fall entirely within run-to-run variance. Every
summary over folds reports the per-fold values and their spread
(noise_helpers.fold_spread). Do not read the boolean "helps"/"redundant" flags
as significance tests; there are none in this file.

2026-10-05, chat 5 of HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.0 (aligned
with section 2.4, Adelaide's rules R1-R6):
  - Folds come from noise_helpers.noise_outer_folds (config['n_splits'] or
    KIRBY_NOISE_N_FOLDS; the splits the per-fold searches are filed under).
    Any count, including 1.
  - Per outer fold, per each of the dataset's two decided models
    (DECISIONS.md Step 1, decided_grid.decided_models): the 2 best base
    representations and the 2 best general hybrids, all ranked by searches
    on that fold's training rows (noise_helpers.fold_configurations). Each
    model is fitted on those 4 configurations ('base_1', 'base_2', 'hybrid',
    'hybrid__general2'), so 2 models x 4 = 8 fitted configurations (R5).
    Nothing is read from a whole-dataset search any more:
    get_best_search_for_downstream, get_rule_hybrids, pool_recipes_by_family,
    noise_base_reps_by_model and the five-model roster are gone.
  - The distance-search hybrid ('hybrid__distance1') and the similarity-
    search hybrid ('hybrid_similarity') are neighbour spaces only: this
    file's conditions find neighbours in them, and no model is fitted on
    them (handoff 2.3 item 1).
  - The damage is the decided grid: noise_helpers.decided_conditions /
    decided_levels / inject_condition. 3 draws per non-zero level, level 0
    (the labels as delivered) once.
  - Every neighbour average in this file weights the neighbours by a
    counting scheme ('rank' or 'inverse_square', KNN_SEARCH_SPACE_RECORD
    axis 5) and both are run; the equal-weight mean of
    kirby.utils.noise_estimation's two estimators is not used any more
    (_neighbour_suspicion). A molecule's scaffold-mates stay in its
    neighbour list (axis 6), and the molecule itself is dropped by index
    (dirty-run bug 7).
  - Each (fold, level, draw) is written to a checkpoint file as it
    finishes (dirty-run bug 12).

2026-10-06, decisions Adelaide delegated (DECIDED, applied here):
  - This file's own conditions are refitted only at the per-dataset
    screen's top 3 (setting, k, counting scheme) of each fitted
    configuration on each (fold, level, draw). The screen predicts a held-back
    fifth of the training molecules (n07.screen_settings with inner=). The
    unweighted condition and its weighted twins are refitted at the same
    (k, scheme), so weighted against unweighted is compared within them.
    Detection (no model fit) is still graded at every k and scheme.
  - Molecular datasets only: a dataset without SMILES is skipped and the
    reason is recorded in the result file.
  - Importance weighting is kept, at the dataset's decided importance
    methods only (decided_grid, 'general' search).
  - gp takes no per-row weight: its repaired fits are recorded as "not run"
    with the reason. Censoring runs one draw per level
    (noise_helpers.draws_for).

WHAT THE CONDITIONS ARE. Per decided model and per fold, the neighbour
spaces are that model's two base representations and its four hybrid spaces
('hybrid', 'hybrid__general2', 'hybrid__distance1', 'hybrid_similarity').
  * base_k_unweighted / base_k_imp_weighted: neighbours in the model's rank-k
    base representation, unweighted or with each feature scaled by its
    importance. Scored for that model fitted on base_k only, as before.
  * <space>_unweighted / <space>_imp_base / <space>_imp_own, per hybrid
    space: unweighted; weighted by the importance method the recipe was
    searched with; weighted by the importance method this sweep is on.
    Scored for every one of that model's 4 fitted configurations.
The suspicion score of each condition, at every neighbour count in K_VALUES
and every counting scheme, is graded for detection. At the (k, counting
scheme) the screen kept for a fitted configuration (2026-10-06) it becomes a
per-row training weight (weights_from_scores), and that configuration is
refitted on the damaged labels with it.

Usage:
    python n12_importance_weighted_distance.py --dataset logd --mode intermediary
    python n12_importance_weighted_distance.py --dataset caco2 --mode full
"""
import copy
import inspect
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    detection_scores, sweep_conditions, conditions_for_task)
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, make_nearest_neighbors,
)
from kirby.hybrid import compute_feature_importance
from kirby.similarity.distance_config import (
    get_distance_metric as _get_distance_metric,
)
from kirby.utils.noise_estimation import weights_from_scores
# The similarity-sweep kit, the checkpoint and the disk caches (plan sections
# 1-3, 2026-09-25). See n07_remediation's SIMILARITY-SWEEP KIT block.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, sweep_schemes, screen_settings, screen_top,
    usable_ks, fold_similarity_spaces, fold_neighbours, RowSink,
    sweep_context, cleaned_from_weights, dprime_rows, row_identity,
    similarity_record, FoldCheckpoint, SCREEN_SCORED_ON,
)
from experiments.noise import noise_cache as _nc


EXPERIMENT = 'n12_importance_weighted_distance'

# What this file writes, since chat 5 (2026-10-05). In the condition files'
# fingerprint (run() below), the checkpoint's and the rows file's name, so a
# file written by the whole-dataset design is recomputed, never read back.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# The damage kind run when a caller names none. The sweep itself runs every
# decided kind (run() below).
NOISE_CONDITION = 'gaussian'

# Three corruptions per non-zero level, averaged within a fold (user decision
# 2026-09-12; the decided noise grid, 2026-10-05). Level 0 is run once.
# Censoring runs one draw (noise_helpers.draws_for, 2026-10-06): it reads no
# seed, so its draws are identical.
NOISE_SEEDS = [0, 1, 2]

# Neighbour counts of this file's own conditions. The same three the record
# keeps (KNN_SEARCH_SPACE_RECORD axis 4).
# DECIDED 2026-10-06 (Adelaide delegated): this file's own conditions are
# refitted only at the (k, counting scheme) of the screen's top 3 (setting, k,
# scheme) for each fitted configuration on each (fold, level, draw); the
# screen predicts a held-back fifth of the training molecules (inner=).
# Every condition of a configuration, unweighted and weighted, is refitted at
# the same (k, scheme), so weighted against unweighted is compared within
# them. Detection, which fits no model, is still graded at every k here.
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
# n21_neighbour_distance.GAP_MIN_FRAC.
GAP_MIN_FRAC = 0.02

# The importance methods the weighted conditions scale the distance by. This
# file's own axis, not a search setting.
# DECIDED 2026-10-06 (Adelaide delegated): importance weighting is this
# experiment's question, so its weighted conditions are kept, although
# KNN_SEARCH_SPACE_RECORD axis 3 cut it from the shared neighbour settings
# (the similarity sweep below follows that cut).
# DECIDED 2026-10-06 (Adelaide delegated): DECISIONS.md Step 3's cut applies
# here too. Each dataset runs its decided importance methods only
# (decided_grid, 'general' search; importance_methods_for below), which can
# include deeplift. IMPORTANCE_METHODS is the list used only when a dataset
# has no decided row.
IMPORTANCE_METHODS = ['lightgbm_gain', 'treeshap']


def importance_methods_for(dataset_name):
    """The dataset's decided importance methods (DECISIONS.md Step 3)."""
    from experiments.decided_grid import decided_search, UndecidedSearch
    try:
        return list(decided_search(dataset_name, 'general')
                    ['importance_methods'])
    except UndecidedSearch:
        return list(IMPORTANCE_METHODS)

# The two neighbour estimators. activity_cliff_aware CUT 2026-09-12 (user
# decision, measured): it ranks the molecules the same way as
# neighbor_residual at Spearman 0.989 on herg_ki in ecfp4.
# Since 2026-10-05 the formulas are computed in this file
# (_neighbour_suspicion) with the neighbours weighted by a counting scheme;
# the names are kirby.utils.noise_estimation's estimators whose formulas
# they follow.
ESTIMATOR_FACTORIES = {
    'neighbor_residual': 'NeighborResidualEstimator',
    'local_variance_ratio': 'LocalVarianceRatioEstimator',
}

# The configurations each decided model is fitted on (noise_helpers.
# CONFIGURATION_SPACES), and the hybrid spaces this file finds neighbours in.
BASE_SPACES = ('base_1', 'base_2')
HYBRID_FIT_SPACES = ('hybrid', 'hybrid__general2')
NEIGHBOUR_HYBRID_SPACES = ('hybrid', 'hybrid__general2', 'hybrid__distance1',
                           'hybrid_similarity')

# The three conditions each hybrid space carries, named after that space.
HYBRID_CONDITION_SUFFIXES = ('_unweighted', '_imp_base', '_imp_own')

# The two base conditions, once per rank: base_1_* in each model's rank-1
# representation, base_2_* in its rank-2 one.
BASE_CONDITION_NAMES = [f'base_{_k}{suf}'
                        for _k in (1, 2)
                        for suf in ('_unweighted', '_imp_weighted')]

CONDITION_NAMES = BASE_CONDITION_NAMES + [
    f'{sp}{suf}'
    for sp in NEIGHBOUR_HYBRID_SPACES
    for suf in HYBRID_CONDITION_SUFFIXES]


def zero_importance_twin(condition_name):
    """Which unweighted condition a weighted one collapses onto.

    When every importance score is zero _weight_features hands the unweighted
    matrix back, so the two spaces are the same geometry and their estimator
    scores are the same numbers. Recorded per (importance method, condition) in
    results['zero_importance_spaces'].
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


def _importance_is_all_zero(importance):
    """True when there is no importance to weight by.

    _weight_features hands back X_standardized untouched in that case, so the
    weighted space IS its unweighted twin: the same geometry, the same
    neighbours, the same estimator scores. The caller tests this and records
    it under 'zero_importance_spaces'.
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
    # The weights take the matrix's own precision, so a float32 representation
    # stays float32 (handoff 2.3 item 18, 2026-10-05). Before, the float64
    # weights promoted every weighted space to float64.
    X_standardized = np.asarray(X_standardized)
    Xw = X_standardized * w.astype(X_standardized.dtype, copy=False)

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


def _takes_sample_weight(factory):
    """True when the model's fit() accepts sample_weight. gp (sklearn's
    GaussianProcessRegressor) does not; rf, xgboost, lightgbm, svm and dnn do."""
    try:
        params = inspect.signature(factory().fit).parameters
    except (TypeError, ValueError):
        return False
    return ('sample_weight' in params
            or any(p.kind == p.VAR_KEYWORD for p in params.values()))


def _neighbour_lists(X, k_max):
    """Each training row's k_max nearest OTHER training rows: (idx, dist),
    both (n, k_eff), nearest first.

    The distance is the process-wide one (make_nearest_neighbors, manhattan
    unless --distance-metric says otherwise), as kirby.utils.noise_estimation
    used it for this file's estimators. Scaffold-mates are not excluded
    (KNN_SEARCH_SPACE_RECORD axis 6).

    Dirty-run bug 7 (handoff 2.7), 2026-10-05: the row itself is dropped BY
    INDEX. Dropping the first column, as noise_estimation._knn_indices does,
    assumes the first neighbour returned is the row itself; when another
    training row is identical in this space either can come first, and then
    the row's own label sits in the average it is compared with while one
    real neighbour is dropped. Same fix as n02 (736ec93).
    """
    X = np.asarray(X)
    n = len(X)
    k_eff = int(min(k_max, n - 1))
    nn = make_nearest_neighbors(n_neighbors=k_eff + 1)
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    own = idx == np.arange(n)[:, None]
    drop = own.copy()
    # A row whose own index did not come back (more than k_eff other rows
    # tied at distance 0) keeps its first k_eff neighbours.
    drop[~own.any(axis=1), -1] = True
    return (idx[~drop].reshape(n, k_eff), dist[~drop].reshape(n, k_eff))


def _neighbour_suspicion(est_name, y, idx, dist, scheme, is_clf):
    """One estimator's per-row suspicion score from a neighbour list cut to k,
    with each neighbour counted by `scheme` (similarity_space.neighbour_weights,
    KNN_SEARCH_SPACE_RECORD axis 5).

    The formulas are kirby.utils.noise_estimation's NeighborResidualEstimator
    and LocalVarianceRatioEstimator with every equal-weight neighbour mean,
    spread, share and vote replaced by its weighted one. Written here on
    2026-10-05 because those estimators average the neighbours with equal
    weight, which the record cut.

      neighbor_residual     regression: |y - weighted neighbour mean| / sd(y)
                            classification: weighted share of neighbours in
                            another class
      local_variance_ratio  regression: |y - weighted mean| / max(weighted
                            neighbour sd, max(0.01 sd(y), 1e-3))
                            classification: that weighted share / max(1 -
                            weighted share of the heaviest class, max(1/k,
                            1e-3))
    """
    from experiments.noise import similarity_space as ss
    y = np.asarray(y)
    idx = np.asarray(idx)
    w = ss.neighbour_weights(dist, scheme)
    W = w.sum(axis=1)
    nb = y[idx]
    if is_clf:
        disagree = (w * (nb != y[:, None])).sum(axis=1) / W
        if est_name == 'neighbor_residual':
            return disagree
        classes = np.unique(y)
        votes = np.stack([(w * (nb == c)).sum(axis=1) for c in classes], axis=1)
        purity = votes.max(axis=1) / W
        floor = max(1.0 / idx.shape[1], 1e-3)
        return disagree / np.maximum(1.0 - purity, floor)
    yf = y.astype(np.float64)
    nbf = nb.astype(np.float64)
    mean = (w * nbf).sum(axis=1) / W
    dev = np.abs(yf - mean)
    if est_name == 'neighbor_residual':
        return dev / (np.std(yf) + 1e-10)
    sd = np.sqrt((w * (nbf - mean[:, None]) ** 2).sum(axis=1) / W)
    floor = max(0.01 * np.std(yf), 1e-3)
    return dev / np.maximum(sd, floor)


def _build_conditions(base_spaces, hybrid_spaces, y_train, imp_method,
                      units_by_space, memo):
    """Every condition of one decided model for one importance method.

    Args:
        base_spaces: {'base_1': (X_scaled_train, X_raw_train, representation)}
            for the ranks this model has on this fold.
        hybrid_spaces: {space: (X_scaled_train, recipe importance method)}
            for the hybrid spaces built on this fold
            (NEIGHBOUR_HYBRID_SPACES).
        y_train: the training labels the experiment holds (damaged).
        imp_method: the importance method this sweep is on.
        units_by_space: {space: [unit names]}: base_k -> ['<model>@base_k'],
            every hybrid space -> all of the model's fitted configurations.
        memo: a dict kept across the importance methods of one (fold, level,
            draw, model), so an importance that does not depend on imp_method
            (the recipe's own, on a hybrid space) is computed once.

    Returns four dicts.
      {condition: X}                 the spaces to score.
      {condition: True}              every weighted condition whose importance
            scores were all zero, so it is its unweighted twin.
      {condition: {...}}             every condition that is a second copy of
            another in this same call, with the one it copies and why.
      {condition: [units]}           which fitted configurations it is scored
            for.
    """
    conditions, zero_importance, duplicate, units_for = {}, {}, {}, {}
    for sp, (X_std, X_raw, rn) in sorted(base_spaces.items()):
        _c1 = f'{sp}_unweighted'
        conditions[_c1] = X_std
        units_for[_c1] = list(units_by_space.get(sp, []))
        try:
            imp = compute_feature_importance(
                {rn: X_raw}, y_train, method=imp_method)[rn]
        except Exception as e:
            print(f"      {imp_method} importance failed on {rn}: {e}")
            continue
        _c2 = f'{sp}_imp_weighted'
        conditions[_c2] = _weight_features(X_std, imp)
        units_for[_c2] = list(units_by_space.get(sp, []))
        if _importance_is_all_zero(imp):
            zero_importance[_c2] = True

    for sp, (X_std, bp_imp) in hybrid_spaces.items():
        units = list(units_by_space.get(sp, []))
        conditions[f'{sp}_unweighted'] = X_std
        units_for[f'{sp}_unweighted'] = units
        # Condition 4: weighted by the importance method the recipe was
        # searched with. Independent of imp_method, so computed once.
        # Since 2026-10-05 the importance is computed on the scaled matrix
        # the distance is measured in (fold_model_inputs returns only that);
        # before, on the unscaled hybrid. Tree importances do not change
        # under a per-feature linear rescale; other methods may.
        key = ('imp_bp', sp)
        if key not in memo:
            try:
                memo[key] = (None if bp_imp is None else
                             compute_feature_importance(
                                 {'hybrid': X_std}, y_train,
                                 method=bp_imp)['hybrid'])
            except Exception as e:
                print(f"      {sp}: importance under the recipe's "
                      f"{bp_imp} failed: {e}")
                memo[key] = None
        imp_bp = memo[key]
        if imp_bp is not None:
            conditions[f'{sp}_imp_base'] = _weight_features(X_std, imp_bp)
            units_for[f'{sp}_imp_base'] = units
            if _importance_is_all_zero(imp_bp):
                zero_importance[f'{sp}_imp_base'] = True

        # Condition 5: weighted by the sweep's imp_method.
        #
        # ⛔ WHEN THE TWO NAMES AGREE, CONDITIONS 4 AND 5 ARE ONE CONDITION.
        # Both score the same hybrid matrix against the same labels, so when
        # the recipe's importance method is the method the sweep is on the two
        # spaces are the same matrix. Measured on flab_thermo_tresanco under
        # outlier_p10 at level 0.5: on the lightgbm_gain half of the sweep
        # hybrid_imp_base and hybrid_imp_own agreed to 1e-12 in 6 of 6
        # readings. The array is reused, and the copy recorded in
        # 'duplicate_conditions'.
        if bp_imp is not None and bp_imp == imp_method and imp_bp is not None:
            duplicate[f'{sp}_imp_own'] = {
                'duplicate_of': f'{sp}_imp_base',
                'reason': (f"the recipe for {sp} ranked its features by "
                           f"{bp_imp!r}, which is the importance method this "
                           f"sweep is on, so both conditions weight the same "
                           f"hybrid matrix by the same scores"),
                'importance_method': imp_method,
            }
            conditions[f'{sp}_imp_own'] = conditions[f'{sp}_imp_base']
            units_for[f'{sp}_imp_own'] = units
            if _importance_is_all_zero(imp_bp):
                zero_importance[f'{sp}_imp_own'] = True
            continue
        try:
            imp_own = compute_feature_importance(
                {'hybrid': X_std}, y_train, method=imp_method)['hybrid']
            conditions[f'{sp}_imp_own'] = _weight_features(X_std, imp_own)
            units_for[f'{sp}_imp_own'] = units
            if _importance_is_all_zero(imp_own):
                zero_importance[f'{sp}_imp_own'] = True
        except Exception as e:
            print(f"      {sp}_imp_own failed: {e}")
    return conditions, zero_importance, duplicate, units_for


def _run_condition(X_space, y_noisy, y_val, unit_inputs, units, model_of_unit,
                   factories, metric_fn, *, dataset_name, predict_mode,
                   is_clf=False, corrupted_mask=None, noise_condition=None,
                   can_weight=True, refit_at=None):
    """Both estimators x K_VALUES x counting schemes on one condition.

    refit_at: {unit: set of (k, scheme)}, the screen's kept (k, scheme) for
        each fitted configuration. A unit is refitted only at those; every k
        and scheme is still graded for detection. None refits nothing.

    X_space: the training rows of the space the neighbours are found in.
    unit_inputs: {unit: (X_train, X_val)}, the features each fitted
        configuration is fitted and scored on.
    units: the units this condition is scored for.

    Returns {f'{unit}|{estimator}_k{k}|{scheme}': repaired held-out score},
    plus '_detection' {f'{estimator}_k{k}|{scheme}': detection_scores reading}
    -- how well each found the molecules whose label actually moved, before
    any repair -- and '_errors'. Every fit goes through
    noise_cache.cached_fit_predict, so a repeat of the same (features,
    labels, weights, model) is fitted once.
    """
    results, errors, detection = {}, {}, {}
    refit_at = refit_at or {}
    ks_here = sorted(set(K_VALUES) | {int(k) for ps in refit_at.values()
                                      for k, _s in ps})
    k_max = max(ks_here)
    try:
        idx_all, dist_all = _neighbour_lists(X_space, k_max)
    except Exception as e:
        for est_name in ESTIMATOR_FACTORIES:
            for k in ks_here:
                for sch in sweep_schemes():
                    for u in units:
                        if (k, sch) not in refit_at.get(u, ()):
                            continue
                        key = f'{u}|{est_name}_k{k}|{sch}'
                        results[key] = None
                        errors[key] = f'{type(e).__name__}: {e}'
        results['_errors'] = errors
        return results
    for est_name in ESTIMATOR_FACTORIES:
        for k in ks_here:
            kk = int(min(k, idx_all.shape[1]))
            for sch in sweep_schemes():
                est_key = f'{est_name}_k{k}|{sch}'
                try:
                    scores = _neighbour_suspicion(
                        est_name, y_noisy, idx_all[:, :kk], dist_all[:, :kk],
                        sch, is_clf)
                    if corrupted_mask is not None:
                        _det = detection_scores(corrupted_mask, scores,
                                                noise_condition)
                        if _det is not None:
                            detection[est_key] = _det
                    weights = weights_from_scores(
                        scores, strength=REPAIR_STRENGTH,
                        method=REPAIR_WEIGHT_METHOD)
                except Exception as e:
                    for u in units:
                        if (k, sch) not in refit_at.get(u, ()):
                            continue
                        results[f'{u}|{est_key}'] = None
                        errors[f'{u}|{est_key}'] = f'{type(e).__name__}: {e}'
                    continue
                if not can_weight:
                    continue
                # DECIDED 2026-10-06 (Adelaide delegated): refitted only at
                # the screen's kept (k, scheme) of each configuration -- see
                # the note at K_VALUES.
                for u in units:
                    if (k, sch) not in refit_at.get(u, ()):
                        continue
                    key = f'{u}|{est_key}'
                    _xtr, _xva = unit_inputs[u]
                    try:
                        pred = _nc.cached_fit_predict(
                            dataset_name, model_of_unit[u],
                            factories[model_of_unit[u]], _xtr, y_noisy, _xva,
                            sample_weight=weights, predict_mode=predict_mode)
                        results[key] = float(metric_fn(y_val, pred))
                    except Exception as e:
                        results[key] = None
                        errors[key] = f'{type(e).__name__}: {e}'
    if errors:
        results['_errors'] = errors
    if detection:
        results['_detection'] = detection
    return results


def _sweep_pass(ctx, *, model_name, model_input, X_tr, X_va, y_tr_noisy,
                y_tr_clean, y_val, train_idx, val_idx, groups_tr, fold, draw,
                entry, corrupted_mask, detection_acc, grade_spaces,
                can_weight):
    """The similarity axis for one fitted configuration on one (fold, level,
    draw): the record's neighbour settings for this dataset's group, each
    estimator of this file at every (setting, k, counting scheme).

    Detection (no model fit) is graded on every (setting, k, scheme) whose
    space is in grade_spaces (None = every space) into detection_acc. The
    per-dataset screen (n07.screen_settings) keeps the best SCREEN_TOP
    (setting, k, scheme), and only those are refitted, with the weights the
    estimator's score gives, into the D-versus-D' rows.

    The same steps as n07.similarity_sweep, written out here so the rows are
    fitted with this configuration's own decided model only (R1). That
    function adds rf and svm to every row (n07.SWEEP_ROW_MODELS).

    Returns the screen's kept [(setting, k, scheme)], best first. The caller
    refits this file's own conditions at their (k, scheme) (2026-10-06).
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

    def _scores(L, k, scheme, est_name):
        idx, dist = ss.neighbours_at(L, k)
        return _neighbour_suspicion(est_name, y_tr_noisy, idx, dist, scheme,
                                    is_clf)

    if corrupted_mask is not None and detection_acc is not None:
        for s in settings:
            if grade_spaces is not None and s.space not in grade_spaces:
                continue
            L = nb.get(s.name)
            for k in usable_ks(L):
                for sch in sweep_schemes():
                    for est_name in ESTIMATOR_FACTORIES:
                        try:
                            sc = _scores(L, k, sch, est_name)
                        except Exception as exc:
                            print(f"      {est_name} on {s.name} k={k} "
                                  f"{sch} failed: {exc}")
                            continue
                        det = detection_scores(corrupted_mask, sc,
                                               ctx['condition'])
                        if det is not None:
                            (detection_acc.setdefault(est_name, {})
                             .setdefault(s.name, {})
                             .setdefault(f'k={k}|scheme={sch}', [])
                             .append(det))

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
    if not can_weight:
        # DECIDED 2026-10-06 (Adelaide delegated): the decided model takes no
        # per-row weight (gp), so its D' rows are "not run", with the reason.
        ctx['unavailable'].setdefault(
            f'{model_input}|rows',
            "not run: this model's fit() takes no sample_weight")
        return kept
    if sink is None or not sink.enabled:
        return kept
    base_id = dict(dataset_name=ds, experiment=ctx['experiment'],
                   model_input=model_input, noise_condition=ctx['condition'],
                   level=ctx['level'], draw=draw, fold=fold)
    for s, k, scheme in kept:
        for est_name in ESTIMATOR_FACTORIES:
            try:
                sc = _scores(nb[s.name], k, scheme, est_name)
                w = weights_from_scores(sc, strength=REPAIR_STRENGTH,
                                        method=REPAIR_WEIGHT_METHOD)
                cleaned = cleaned_from_weights(X_tr, y_tr_noisy, w)
            except Exception as exc:
                print(f"      {est_name} weights failed: {exc}")
                cleaned = None
            dprime_rows(sink, dataset_name=ds, cleaned=cleaned, X_tr=X_tr,
                        y_D=y_tr_noisy, X_va=X_va, y_va=y_val,
                        metric=ctx['metric'], task_type=ctx['task_type'],
                        y_clean=y_tr_clean, models=(model_name,),
                        is_clf=is_clf,
                        identity=row_identity(
                            method=est_name,
                            method_setting=(f'k={k}|scheme={scheme}|'
                                            f'{REPAIR_WEIGHT_METHOD}_'
                                            f'{REPAIR_STRENGTH}'),
                            similarity_setting=s.name, **base_id))
    return kept


def _condition_levels(data, dataset_name, condition):
    """The decided levels of one damage kind on this dataset, level 0 first
    (noise_helpers.decided_levels; KNN_SEARCH_SPACE_RECORD "The noise grid",
    Adelaide 2026-10-05)."""
    from experiments.noise.noise_helpers import decided_levels
    return tuple(float(x) for x in decided_levels(data, dataset_name,
                                                  condition))


# 2026-10-07 (Adelaide): the molecular-only rule added 2026-10-06 is removed.
# SMILES only adds one neighbour setting, Tanimoto on ECFP4
# (similarity_space refuses Tanimoto on any other features); without SMILES
# that setting is dropped and everything else runs.


def run(data, config, output_dir, dataset_name, **kwargs):
    """n12_importance_weighted_distance across every decided damage kind.

    The condition is an axis this experiment sweeps; the sweep, the
    per-condition files, the resume and the stride slice are
    noise_helpers.sweep_conditions. Its headline is a downstream score, which
    is defined under every condition, so it sweeps the whole list. Detection
    is recorded too, and detection_scores returns None where it is undefined.

    2026-10-05 (chat 5): the condition files are fingerprinted with the
    levels, the draws, the outer fold count, the folds run and RESULT_LAYOUT,
    so a file written by the whole-dataset design is recomputed, not skipped.
    """
    from experiments.noise.noise_helpers import noise_n_folds
    conditions = list(conditions_for_task(data))
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: list(_condition_levels(data, dataset_name, c)) for c in conditions}
    fp_config['noise_seeds'] = list(NOISE_SEEDS)
    fp_config['importance_methods'] = importance_methods_for(dataset_name)
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, EXPERIMENT,
        data, fp_config, output_dir, dataset_name,
        default_conditions=conditions, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _parse_estimator_key(ek):
    """'svm@hybrid|neighbor_residual_k10|rank' -> (unit, estimator, k,
    scheme)."""
    unit, est_k, scheme = ek.split('|', 2)
    est, k = est_k.rsplit('_k', 1)
    return unit, est, int(k), scheme


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N12, ONE damage kind at every decided level. What each fold does:

      1. noise_helpers.fold_configurations: searches on the fold's training
         rows (clean labels, decided grid, pruning off) give each of the
         dataset's two decided models its 2 base representations, its 2
         general hybrids, and the recipes of its distance hybrid and
         similarity hybrid.
      2. Per (level, draw): the training labels are damaged
         (noise_helpers.inject_condition); each model's hybrids are rebuilt
         from the damaged labels (noise_helpers.fold_model_inputs); each of
         its 4 configurations is scored with clean and with damaged labels;
         the similarity axis (_sweep_pass) runs for each configuration, and
         its screen keeps the top 3 (setting, k, scheme); then every
         condition of the model, at each decided importance method and each
         estimator, gives training weights, and each configuration it is
         scored for is refitted at that configuration's kept (k, scheme)
         only. Detection is graded at every k and scheme.
      A dataset without SMILES is skipped with the reason recorded.
    Each (fold, level, draw) is written to a checkpoint as it finishes.
    """
    from experiments._shared import input_fingerprint
    from experiments.decided_grid import decided_models
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_configurations, fold_model_inputs,
        configuration_record, inject_condition, dataset_knn_group, level_unit,
        fold_spread, CONFIGURATION_SPACES)

    from experiments.noise.noise_helpers import draws_for

    print("\n" + "=" * 70)
    print("EXPERIMENT N12: IMPORTANCE-WEIGHTED DISTANCE FOR NOISE ESTIMATION")
    print("=" * 70)

    importance_methods = importance_methods_for(dataset_name)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # auc is scored on the positive-class probability, never on hard class
    # labels (dirty-run bug 13, 2026-10-05: every fit here used predict()).
    predict_mode = _nc.predict_mode_for(metric_name)
    condition = kwargs.get('noise_condition') or NOISE_CONDITION
    levels = _condition_levels(data, dataset_name, condition)
    knn_grp = dataset_knn_group(data, dataset_name)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    noise_groups = noise_groups_for(data)[0]

    # R1: the dataset's two decided models (DECISIONS.md Step 1), nothing
    # else. knn runs only inside the distance search. Replaces
    # config['models'] / DOWNSTREAM_MODELS (five models) and the rf-only
    # recipe lookup.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    folds, n_folds = noise_outer_folds(data, config)

    print(f"  Dataset: {dataset_name} ({len(labels)} rows), group {knn_grp}")
    print(f"  Metric: {metric_name} "
          f"({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Models (decided): {models}")
    print(f"  Condition {condition}, levels {list(levels)} "
          f"({level_unit(condition, is_clf)})")
    print(f"  Outer folds: {n_folds} (running {[f for f, _, _ in folds]})")
    print(f"  K values: {K_VALUES}; counting schemes {list(sweep_schemes())}")
    print(f"  Importance methods (decided): {importance_methods}")

    # The neighbour settings this dataset's group sweeps (KNN_SEARCH_SPACE_
    # RECORD): group from the molecule count, no model_input, no importance
    # weighting, distances per group, scaffold-mates kept. ks and counting
    # schemes are swept and screened inside _sweep_pass.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    # A new rows file: rows written by the whole-dataset design share model
    # input names with these and would be skipped as done.
    sink = RowSink(output_dir, EXPERIMENT, dataset_name,
                   f'{condition}__{RESULT_LAYOUT}')
    sweep_ctx = sweep_context(
        experiment=EXPERIMENT, dataset_name=dataset_name, sink=sink,
        settings=sim_settings, metric=metric, task_type=task_type,
        condition=condition, level=None, data=data, factories={})
    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted; rows -> {sink.path}")

    ckpt = FoldCheckpoint(
        output_dir, EXPERIMENT, dataset_name, condition=condition,
        tag='units', fingerprint={
            'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
            'condition': condition, 'levels': list(levels),
            'n_folds': int(n_folds), 'models': models,
            'importance_methods': list(importance_methods),
            'k_values': list(K_VALUES), 'schemes': list(sweep_schemes()),
            'draws': list(draws_for(condition, NOISE_SEEDS)),
            'own_refits': 'screen_top_k_scheme_per_configuration',
            'importance_exponent': float(IMPORTANCE_EXPONENT),
            'distance_metric': _get_distance_metric(),
            'repair': [REPAIR_WEIGHT_METHOD, REPAIR_STRENGTH],
            'settings': [s.name for s in sim_settings],
            'screen_top': screen_top()})

    fold_records = {}
    units_done = []      # [{'fold', 'level', 'draw', 'value'}]

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
            val = {'provenance': prov, 'n_corrupted': int(cmask.sum()),
                   'clean': {}, 'noisy': {}, 'scores': {}, 'detection': {},
                   'errors': {}, 'zero_importance': {}, 'duplicate': {},
                   'duplicate_copied': {}, 'features_in_space': {},
                   'unavailable': {}, 'weighted_undefined': {},
                   'hybrid_spaces_built': {}, 'sim_detection': {},
                   'refit_at': {}}
            sweep_ctx['level'] = float(level)
            sweep_ctx['screen'] = []
            for m, (entry, base_ins, base_unav, _names) in _bi.items():
                factory = _fac[m]
                can_weight = _takes_sample_weight(factory)
                if not can_weight:
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
                val['hybrid_spaces_built'][m] = sorted(hyb)
                for sp, why in list(base_unav.items()) + list(hyb_unav.items()):
                    val['unavailable'].setdefault(m, {})[sp] = why
                # The fitted configurations (R5): 2 base + 2 hybrids.
                unit_inputs = {}
                for sp in CONFIGURATION_SPACES:
                    pair = base_ins.get(sp) if sp in BASE_SPACES else hyb.get(sp)
                    if pair is not None:
                        unit_inputs[f'{m}@{sp}'] = pair
                model_of_unit = {u: m for u in unit_inputs}
                # Clean and damaged references of every configuration, in the
                # features of this draw (a hybrid's are chosen from the
                # damaged labels, so its clean score is the clean labels in
                # those features).
                for u, (xtr, xva) in unit_inputs.items():
                    for tag, yy in (('clean', _ytr), ('noisy', y_noisy)):
                        try:
                            pred = _nc.cached_fit_predict(
                                dataset_name, m, factory, xtr, yy, xva,
                                predict_mode=predict_mode)
                            val[tag][u] = float(metric_fn(_yva, pred))
                        except Exception as exc:
                            val[tag][u] = None
                            val['errors'][f'{u}|{tag}'] = (
                                f'{type(exc).__name__}: {exc}')
                # The similarity axis, once per fitted configuration. The
                # lists of every space but forest_proximity do not depend on
                # the configuration, so detection is graded on the first one
                # and only forest_proximity on the others. Run before this
                # file's own conditions: its screen's kept (k, scheme) are
                # where they are refitted (2026-10-06).
                refit_at = {}
                for j, (u, (xtr, xva)) in enumerate(unit_inputs.items()):
                    kept = _sweep_pass(
                        sweep_ctx, model_name=m,
                        model_input=f"{m}:{u.split('@', 1)[1]}",
                        X_tr=xtr, X_va=xva, y_tr_noisy=y_noisy,
                        y_tr_clean=_ytr, y_val=_yva, train_idx=_tr,
                        val_idx=_va, groups_tr=_g, fold=_fold, draw=draw,
                        entry=entry, corrupted_mask=det_mask,
                        detection_acc=val['sim_detection'],
                        grade_spaces=(None if j == 0 else
                                      ('forest_proximity',)),
                        can_weight=can_weight) or []
                    refit_at[u] = {(int(k_), sch_) for _s, k_, sch_ in kept}
                    val['refit_at'].setdefault(m, {})[u] = {
                        'kept_settings': [f'{s_.name}|k={k_}|{sch_}'
                                          for s_, k_, sch_ in kept],
                        'k_scheme_refitted': sorted(
                            [int(k_), sch_] for k_, sch_ in refit_at[u])}
                # The neighbour spaces of this model.
                base_spaces = {}
                for i, sp in enumerate(BASE_SPACES):
                    rn = ((entry.get('base') or [None, None]) + [None, None])[i]
                    if sp in base_ins and rn is not None:
                        base_spaces[sp] = (
                            base_ins[sp][0],
                            np.asarray(data['reps'][rn])[_tr], rn)
                hybrid_spaces = {
                    sp: (hyb[sp][0],
                         getattr((entry.get('recipes') or {}).get(
                             sp, (None, None))[0], 'importance_method', None))
                    for sp in NEIGHBOUR_HYBRID_SPACES if sp in hyb}
                # OPEN (chat 5): which conditions each configuration is
                # scored under. Kept as before 2026-10-05: a base condition
                # only for the configuration fitted on that representation,
                # every hybrid-space condition for all of the model's
                # configurations (now 4, the hybrids included).
                units_by_space = {sp: [f'{m}@{sp}']
                                  for sp in BASE_SPACES
                                  if f'{m}@{sp}' in unit_inputs}
                for sp in hybrid_spaces:
                    units_by_space[sp] = list(unit_inputs)
                memo = {}
                for imp_method in importance_methods:
                    conds, zero, dup, units_for = _build_conditions(
                        base_spaces, hybrid_spaces, y_noisy, imp_method,
                        units_by_space, memo)
                    scored_here = {}
                    for cond, X_space in conds.items():
                        (val['features_in_space'].setdefault(imp_method, {})
                         .setdefault(cond, {}))[m] = int(np.shape(X_space)[1])
                        if cond in zero:
                            (val['zero_importance'].setdefault(imp_method, {})
                             .setdefault(cond, [])).append(m)
                        twin = (dup.get(cond) or {}).get('duplicate_of')
                        if cond in dup:
                            (val['duplicate'].setdefault(imp_method, {})
                             .setdefault(cond, dict(dup[cond], models=[]))
                             ['models'].append(m))
                        if twin is not None and twin in scored_here:
                            res = copy.deepcopy(scored_here[twin])
                            (val['duplicate_copied'].setdefault(imp_method, {})
                             .setdefault(cond, [])).append(m)
                        else:
                            res = _run_condition(
                                X_space, y_noisy, _yva, unit_inputs,
                                units_for.get(cond, []), model_of_unit, _fac,
                                metric_fn, dataset_name=dataset_name,
                                predict_mode=predict_mode, is_clf=is_clf,
                                corrupted_mask=det_mask,
                                noise_condition=condition,
                                can_weight=can_weight, refit_at=refit_at)
                        scored_here[cond] = res
                        out = (val['scores'].setdefault(imp_method, {})
                               .setdefault(cond, {}))
                        for ek, sc in res.items():
                            if ek == '_detection':
                                (val['detection'].setdefault(imp_method, {})
                                 .setdefault(cond, {}))[m] = sc
                            elif ek == '_errors':
                                (val['errors'].setdefault(imp_method, {})
                                 .setdefault(cond, {})).update(sc)
                            else:
                                out[ek] = sc
            val['screen'] = sweep_ctx.pop('screen', [])
            return val

        for level in levels:
            draws = (NOISE_SEEDS[:1] if float(level) == 0.0
                     else list(draws_for(condition, NOISE_SEEDS)))
            for draw in draws:
                key = ['unit', int(fold_i), float(level), int(draw)]
                v = ckpt.get(key)
                if v is None:
                    t0 = time.time()
                    print(f"    level {level:g}, draw {draw}")
                    v = _unit(float(level), int(draw))
                    ckpt.put(key, v)
                    print(f"    level {level:g}, draw {draw}: "
                          f"{time.time() - t0:.0f}s")
                else:
                    print(f"    level {level:g}, draw {draw}: read back from "
                          f"{ckpt.path}")
                units_done.append({'fold': int(fold_i), 'level': float(level),
                                   'draw': int(draw), 'value': v})

        # The held-out score of each fitted configuration in this fold
        # (handoff 2.4 item 11), per level: clean training labels and
        # damaged ones, the mean over this fold's draws, in the dataset's own
        # metric. Each draw's own score is kept beside it.
        held_out = {}
        fold_units = [u for u in units_done if u['fold'] == int(fold_i)]
        for level in levels:
            lv_units = [u for u in fold_units if u['level'] == float(level)]
            per_conf = {}
            for u in lv_units:
                for tag in ('clean', 'noisy'):
                    for conf, s in (u['value'].get(tag) or {}).items():
                        (per_conf.setdefault(conf, {}).setdefault(
                            tag + '_by_draw', {}))[str(u['draw'])] = s
            for conf, rec in per_conf.items():
                for tag in ('clean', 'noisy'):
                    vals = [x for x in (rec.get(tag + '_by_draw') or {})
                            .values() if x is not None and np.isfinite(x)]
                    rec[tag] = float(np.mean(vals)) if vals else None
                rec['damaged_minus_clean'] = (
                    None if rec.get('clean') is None or rec.get('noisy') is None
                    else float(rec['noisy'] - rec['clean']))
            held_out[str(float(level))] = per_conf
        fold_records[str(fold_i)]['held_out_score'] = held_out
        fold_records[str(fold_i)]['metric'] = metric_name
        fold_records[str(fold_i)]['lower_is_better'] = bool(lower_is_better)

    results = _aggregate(units_done, levels, lower_is_better, fold_spread)
    results.update({
        'experiment': EXPERIMENT,
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'task_type': task_type,
        'noise_condition': condition,
        'levels': list(levels),
        'sigmas': list(levels),
        'level_unit': level_unit(condition, is_clf),
        'draws_per_nonzero_level': len(draws_for(condition, NOISE_SEEDS)),
        'level_zero': 'the labels as delivered, run once (one draw)',
        'knn_group': knn_grp,
        'robustness_reported_as': (
            "damaged_minus_clean: the held-out score with damaged training "
            "labels minus the score with clean ones, per fitted "
            "configuration and level, in the dataset's own metric (see "
            "lower_is_better for its direction). This file never computed "
            "auc_norm."),
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _t, _v in folds],
        'selection_scope': 'fold_train_rows',
        'models': models,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'n_fitted_configurations': len(models) * len(CONFIGURATION_SPACES),
        'neighbour_hybrid_spaces': list(NEIGHBOUR_HYBRID_SPACES),
        'folds': fold_records,
        'k_values': list(K_VALUES),
        'schemes': list(sweep_schemes()),
        'estimators': list(ESTIMATOR_FACTORIES),
        'importance_methods': list(importance_methods),
        'importance_methods_rule': (
            "the dataset's decided importance methods "
            "(decided_grid, 'general' search)"),
        'own_conditions_refitted_at': (
            "per fitted configuration and (fold, level, draw), the (k, "
            "counting scheme) of the screen's top kept (setting, k, scheme); "
            "the screen predicts a held-back fifth of the training "
            "molecules. Unweighted and weighted conditions are refitted at "
            "the same (k, scheme). Detection is graded at every k in "
            "k_values and every scheme."),
        # The power the importance was raised to, the distance the neighbours
        # were found under, and the power that distance would match.
        'importance_exponent': float(IMPORTANCE_EXPONENT),
        'distance_metric': _get_distance_metric(),
        'exponent_matched_to_distance_metric': EXPONENT_MATCHED_TO_METRIC.get(
            _get_distance_metric()),
        'repair': f'{REPAIR_WEIGHT_METHOD} at strength {REPAIR_STRENGTH}',
        'conditions_run': sorted({c for u in units_done
                                  for by_c in (u['value'].get('scores')
                                               or {}).values()
                                  for c in by_c}),
        'estimator_key': (
            "'<model>@<configuration>|<estimator>_k<k>|<scheme>': that model "
            "fitted on that configuration (base_1, base_2, hybrid, "
            "hybrid__general2) with the training weights the estimator's "
            "score gives, neighbours counted by the scheme."),
        'recipe_note': (
            "Each fold's configurations and neighbour-space recipes come from "
            "searches on that fold's training rows with clean labels "
            "(noise_helpers.fold_configurations). The recipe is chosen "
            "without the damage; each hybrid's features are re-chosen from "
            "the damaged labels at every draw."),
        'similarity': similarity_record(
            sim_settings, sink, schemes_swept=list(sweep_schemes()),
            screen_top=screen_top(), screen_scored_on=SCREEN_SCORED_ON,
            knn_group=knn_grp, spaces_unavailable=sweep_ctx['unavailable'],
            repair=f'{REPAIR_WEIGHT_METHOD} at strength {REPAIR_STRENGTH}'),
        'screen': [dict(e, fold=u['fold'], level=u['level'])
                   for u in units_done
                   for e in (u['value'].get('screen') or [])],
        'checkpoint_file': ckpt.path,
        'n_units_resumed': ckpt.n_resumed,
    })
    results['summary'] = _compute_summary(results)

    print(f"\n  === SUMMARY ===")
    print("  Differences may fall within the fold-to-fold spread; read the")
    print("  boolean flags against recovery_by_fold, not as significance tests.")
    for sk, sv in results['summary'].items():
        if isinstance(sv, dict) and 'weighting_helps_base' in sv:
            print(f"  {sk}:")
            print(f"    Weighting helps base: {sv['weighting_helps_base']}")
            print(f"    Weighting helps hybrid: {sv['weighting_helps_hybrid']}")
            print(f"    Hybrid already weighted: {sv['hybrid_already_weighted']}")
            print(f"    Best k: {sv.get('best_k')}; best scheme: "
                  f"{sv.get('best_scheme')}")
            print(f"    Best importance method: {sv.get('best_importance_method')}")
    return convert_numpy(results)


def _aggregate(units_done, levels, lower_is_better, fold_spread):
    """Per level: every number averaged over draws within a fold, then
    summarised over folds with fold_spread. Recovery is computed from the
    fold means, and per fold beside it."""
    def _by_fold(rows):
        """[(fold, value)] -> {fold: mean over draws}."""
        acc = {}
        for f, v in rows:
            if v is None or not np.isfinite(v):
                continue
            acc.setdefault(f, []).append(float(v))
        return {f: float(np.mean(v)) for f, v in acc.items() if v}

    def _mean(d):
        return float(np.mean(list(d.values()))) if d else None

    def _recovery(c, nz, s):
        if any(x is None or not np.isfinite(x) for x in (c, nz)):
            return float('nan'), 'not computed: a score was not finite'
        gap = (nz - c) if lower_is_better else (c - nz)
        floor = max(1e-10, GAP_MIN_FRAC * abs(c))
        if not abs(gap) > floor:
            return float('nan'), (f'not computed: the clean-to-noisy gap '
                                  f'({gap:.6g}) is under {GAP_MIN_FRAC:.0%} '
                                  f'of the clean score ({c:.6g})')
        if s is None or not np.isfinite(s):
            return float('nan'), 'not computed: no repaired score'
        return float((s - nz) / (c - nz) * 100), None

    per_sigma = {}
    zero_acc, dup_acc, dup_copied, feat_acc = {}, {}, {}, {}
    for lv in levels:
        lv_units = [u for u in units_done if u['level'] == lv]
        clean_f, noisy_f = {}, {}
        # {unit: {(fold, draw): score}}: the no-cleaning score of every pass,
        # so a refit kept on fewer draws is compared on its own draws (A18,
        # 2026-10-08).
        noisy_pd = {}
        for u in lv_units:
            for unit, s in (u['value'].get('clean') or {}).items():
                clean_f.setdefault(unit, []).append((u['fold'], s))
            for unit, s in (u['value'].get('noisy') or {}).items():
                noisy_f.setdefault(unit, []).append((u['fold'], s))
                if s is not None and np.isfinite(s):
                    noisy_pd.setdefault(unit, {})[
                        (int(u['fold']), int(u['draw']))] = float(s)
        clean_bf = {k: _by_fold(v) for k, v in clean_f.items()}
        noisy_bf = {k: _by_fold(v) for k, v in noisy_f.items()}
        units_scored = sorted(k for k in clean_bf
                              if clean_bf[k] and noisy_bf.get(k))
        clean_by_unit = {k: _mean(clean_bf[k]) for k in units_scored}
        noisy_by_unit = {k: _mean(noisy_bf[k]) for k in units_scored}
        damaged_minus_clean = {}
        for k in units_scored:
            common = sorted(set(clean_bf[k]) & set(noisy_bf[k]))
            damaged_minus_clean[k] = fold_spread(
                [noisy_bf[k][f] - clean_bf[k][f] for f in common])

        grid = {}
        for u in lv_units:
            for imp, by_c in (u['value'].get('scores') or {}).items():
                for cond, by_ek in by_c.items():
                    for ek, s in by_ek.items():
                        (grid.setdefault(imp, {}).setdefault(cond, {})
                         .setdefault(ek, []).append(
                             (u['fold'], u['draw'], s)))
        per_imp = {}
        for imp, by_c in grid.items():
            imp_result = {}
            for cond, by_ek in by_c.items():
                cond_agg = {}
                for ek, rows_fd in by_ek.items():
                    rows = [(f, s) for f, _d, s in rows_fd]
                    bf = _by_fold(rows)
                    if not bf:
                        continue
                    unit, est, k, sch = _parse_estimator_key(ek)
                    # The passes this refit was kept on, and no cleaning on
                    # exactly those passes (draws averaged within a fold,
                    # then over folds, as everywhere else here).
                    _npd = noisy_pd.get(unit, {})
                    _pairs = [(int(f), int(d), float(s), _npd[(int(f), int(d))])
                              for f, d, s in rows_fd
                              if s is not None and np.isfinite(s)
                              and (int(f), int(d)) in _npd]
                    _den_sp = _mean(_by_fold([(f, s) for f, _d, s, _n
                                              in _pairs]))
                    _noi_sp = _mean(_by_fold([(f, n) for f, _d, _s, n
                                              in _pairs]))
                    _gain_sp = (None if _den_sp is None or _noi_sp is None
                                else float((_noi_sp - _den_sp)
                                           if lower_is_better
                                           else (_den_sp - _noi_sp)))
                    denoised = _mean(bf)
                    rec, note = _recovery(clean_by_unit.get(unit),
                                          noisy_by_unit.get(unit), denoised)
                    per_fold_rec = []
                    for f, s in bf.items():
                        c = clean_bf.get(unit, {}).get(f)
                        nz = noisy_bf.get(unit, {}).get(f)
                        if c is None or nz is None:
                            continue
                        r, _why = _recovery(c, nz, s)
                        if np.isfinite(r):
                            per_fold_rec.append(r)
                    cond_agg[ek] = {
                        'denoised': denoised,
                        # A18 (2026-10-08): the (fold, draw) passes behind
                        # 'denoised', each with its own no-cleaning score,
                        # and both averaged over those passes only. gain > 0
                        # means the refit beat no cleaning on them.
                        'passes': [[f, d] for f, d, _s, _n in _pairs],
                        'denoised_per_pass': [s_ for _f, _d, s_, _n
                                              in _pairs],
                        'noisy_per_pass': [n_ for _f, _d, _s, n_
                                           in _pairs],
                        'denoised_same_passes': _den_sp,
                        'noisy_same_passes': _noi_sp,
                        'gain_over_noisy_same_passes': _gain_sp,
                        'denoised_by_fold': fold_spread(
                            [bf[f] for f in sorted(bf)]),
                        'recovery': rec,
                        'recovery_note': note,
                        'recovery_by_fold': fold_spread(per_fold_rec),
                        'model': unit,
                        'downstream_model': unit.split('@', 1)[0],
                        'fitted_on': unit.split('@', 1)[1],
                        'estimator': est, 'k': k, 'scheme': sch,
                    }
                if cond_agg:
                    imp_result[cond] = cond_agg
            per_imp[imp] = imp_result

        det = {}
        for u in lv_units:
            for imp, by_c in (u['value'].get('detection') or {}).items():
                for cond, by_m in by_c.items():
                    for m, by_e in by_m.items():
                        for ek, row in by_e.items():
                            (det.setdefault(imp, {}).setdefault(cond, {})
                             .setdefault(m, {}).setdefault(ek, [])
                             .append(row))
        sim_det = {}
        for u in lv_units:
            for e, by_s in (u['value'].get('sim_detection') or {}).items():
                for sn, by_k in by_s.items():
                    for k, rows in by_k.items():
                        (sim_det.setdefault(e, {}).setdefault(sn, {})
                         .setdefault(k, [])).extend(rows)
        errors = {}
        for u in lv_units:
            for k, v in (u['value'].get('errors') or {}).items():
                if isinstance(v, dict):
                    for c, ev in v.items():
                        errors.setdefault(k, {}).setdefault(c, {}).update(
                            ev if isinstance(ev, dict) else {'_': ev})
                else:
                    errors[k] = v
        for u in lv_units:
            for imp, by_c in (u['value'].get('zero_importance') or {}).items():
                for cond, ms in by_c.items():
                    z = (zero_acc.setdefault(imp, {}).setdefault(
                        cond, {'n_builds_all_zero': 0,
                               'duplicate_of': zero_importance_twin(cond),
                               'sigmas': []}))
                    z['n_builds_all_zero'] += len(ms)
                    if lv not in z['sigmas']:
                        z['sigmas'].append(lv)
            for imp, by_c in (u['value'].get('duplicate') or {}).items():
                for cond, rec in by_c.items():
                    d = dup_acc.setdefault(imp, {}).setdefault(
                        cond, {k: v for k, v in rec.items() if k != 'models'})
                    d['n_builds'] = d.get('n_builds', 0) + len(
                        rec.get('models') or [])
                    d.setdefault('sigmas', [])
                    if lv not in d['sigmas']:
                        d['sigmas'].append(lv)
            for imp, by_c in (u['value'].get('duplicate_copied') or {}).items():
                for cond, ms in by_c.items():
                    dc = dup_copied.setdefault(imp, {}).setdefault(
                        cond, {'copied_from': cond.replace('_imp_own',
                                                           '_imp_base'),
                               'n_scorings_skipped': 0, 'sigmas': []})
                    dc['n_scorings_skipped'] += len(ms)
                    if lv not in dc['sigmas']:
                        dc['sigmas'].append(lv)
            for imp, by_c in (u['value'].get('features_in_space') or {}).items():
                for cond, by_m in by_c.items():
                    for m, n in by_m.items():
                        w = (feat_acc.setdefault(imp, {}).setdefault(cond, {})
                             .setdefault(m, []))
                        if n not in w:
                            w.append(n)

        per_sigma[str(lv)] = {
            'level': float(lv),
            'n_draws': len({u['draw'] for u in lv_units}),
            'units_scored': units_scored,
            'clean_score_by_model': clean_by_unit,
            'noisy_score_by_model': noisy_by_unit,
            # A18 (2026-10-08): no cleaning per (fold, draw) pass,
            # {unit: {'f<fold>d<draw>': score}}.
            'noisy_score_by_pass': {
                k: {f'f{f}d{d}': v for (f, d), v in sorted(
                    noisy_pd.get(k, {}).items())}
                for k in units_scored},
            # 'folds' names the fold of each entry of 'per_fold' (a fold
            # with no finite score is left out of both).
            'clean_score_by_fold': {k: dict(fold_spread(
                [clean_bf[k][f] for f in sorted(clean_bf[k])]),
                folds=sorted(clean_bf[k]))
                for k in units_scored},
            'noisy_score_by_fold': {k: dict(fold_spread(
                [noisy_bf[k][f] for f in sorted(noisy_bf[k])]),
                folds=sorted(noisy_bf[k]))
                for k in units_scored},
            'damaged_minus_clean': damaged_minus_clean,
            'per_importance_method': per_imp,
            'detection': {imp: {c: {m: {ek: _mean_detection(r)
                                        for ek, r in by_e.items()}
                                    for m, by_e in by_m.items()}
                                for c, by_m in by_c.items()}
                          for imp, by_c in det.items()},
            'detection_by_similarity': {
                e: {sn: {k: _mean_detection(r) for k, r in by_k.items()}
                    for sn, by_k in by_s.items()}
                for e, by_s in sim_det.items()},
            'errors': errors,
            'configurations_unavailable': {
                f"fold{u['fold']}_draw{u['draw']}": u['value'].get(
                    'unavailable') for u in lv_units
                if u['value'].get('unavailable')},
            'weighted_undefined': {
                m: why for u in lv_units
                for m, why in (u['value'].get('weighted_undefined')
                               or {}).items()},
            'noise_provenance': {
                f"fold{u['fold']}_draw{u['draw']}": u['value'].get(
                    'provenance') for u in lv_units},
        }
    return {'per_sigma': per_sigma,
            'zero_importance_spaces': zero_acc,
            'duplicate_conditions': dup_acc,
            'duplicate_conditions_copied': dup_copied,
            'features_in_space': feat_acc}


def _compute_summary(results):
    """The ablation answers per level. Every number is a mean of recoveries
    (percent of the damage recovered), each computed from fold means; the
    spread beside it is over the cells averaged, and each cell's own fold
    spread is in per_importance_method.<imp>.<condition>.<key>.
    recovery_by_fold. No significance test."""
    summary = {}
    _ORIGINAL_CONDITIONS = tuple(BASE_CONDITION_NAMES) + (
        'hybrid_unweighted', 'hybrid_imp_base', 'hybrid_imp_own')

    for sigma_key, sigma_data in results['per_sigma'].items():
        cond_recoveries = {}
        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            for cond_name, cond_data in imp_data.items():
                cond_recoveries.setdefault(cond_name, [])
                for ek, rv in cond_data.items():
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        cond_recoveries[cond_name].append(r)

        cond_avg = {c: float(np.mean(rs)) if rs else float('nan')
                    for c, rs in cond_recoveries.items()}
        cond_std = {c: float(np.std(rs)) if len(rs) > 1 else float('nan')
                    for c, rs in cond_recoveries.items()}
        cond_range = {c: [float(np.min(rs)), float(np.max(rs))] if rs
                      else [float('nan'), float('nan')]
                      for c, rs in cond_recoveries.items()}

        _base_ranks = [f'base_{_k}' for _k in (1, 2)
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

        hybrid_spaces_seen = sorted({
            c[:-len('_unweighted')] for c in cond_avg
            if c.endswith('_unweighted') and not c.startswith('base_')})
        answers_by_space = {hs: _answers_for(hs) for hs in hybrid_spaces_seen}
        _pred = answers_by_space.get('hybrid', {
            'weighting_helps_base': _weighting_helps_base(),
            'weighting_helps_hybrid': False,
            'hybrid_already_weighted': False})

        k_rec, k_rec_all, s_rec, m_rec, m_rec_all = {}, {}, {}, {}, {}
        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            for _cond_name, cond_data in imp_data.items():
                for ek, rv in cond_data.items():
                    r = rv.get('recovery', float('nan'))
                    if np.isnan(r):
                        continue
                    k_val, sch = rv.get('k'), rv.get('scheme')
                    k_rec_all.setdefault(k_val, []).append(r)
                    m_rec_all.setdefault(imp_method, []).append(r)
                    if _cond_name in _ORIGINAL_CONDITIONS:
                        k_rec.setdefault(k_val, []).append(r)
                        s_rec.setdefault(sch, []).append(r)
                        m_rec.setdefault(imp_method, []).append(r)

        def _best(d):
            avgs = {k: float(np.mean(v)) for k, v in d.items() if v}
            return (max(avgs, key=avgs.get) if avgs else None), avgs

        best_k, per_k = _best(k_rec)
        best_scheme, per_scheme = _best(s_rec)
        best_imp, per_method = _best(m_rec)
        summary[f'sigma_{sigma_key}'] = convert_numpy({
            'weighting_helps_base': _pred['weighting_helps_base'],
            'weighting_helps_hybrid': _pred['weighting_helps_hybrid'],
            'hybrid_already_weighted': _pred['hybrid_already_weighted'],
            'best_k': best_k,
            'best_scheme': best_scheme,
            'best_importance_method': best_imp,
            'per_condition_avg_recovery': cond_avg,
            'per_condition_recovery_std': cond_std,
            'per_condition_recovery_range': cond_range,
            'per_k_avg_recovery': {str(k): v for k, v in per_k.items()},
            'per_scheme_avg_recovery': per_scheme,
            'per_method_avg_recovery': per_method,
            'answers_by_hybrid_space': answers_by_space,
            'hybrid_spaces_scored': hybrid_spaces_seen,
            'per_k_avg_recovery_every_condition': {
                str(k): float(np.mean(rs)) for k, rs in k_rec_all.items() if rs},
            'per_method_avg_recovery_every_condition': {
                m: float(np.mean(rs)) for m, rs in m_rec_all.items() if rs},
            'caveat': ('Condition differences may fall within the '
                       'fold-to-fold spread; compare each mean against '
                       'recovery_by_fold before claiming an effect. Recipes '
                       'are chosen on each fold\'s clean training labels and '
                       'the hybrids rebuilt from the damaged ones.'),
        })
    return summary


if __name__ == '__main__':
    parser = create_experiment_parser('N12: Importance-Weighted Distance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, EXPERIMENT, args.dataset)
