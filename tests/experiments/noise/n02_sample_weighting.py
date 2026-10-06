#!/usr/bin/env python3
"""N02: Importance-weighted sample weighting for noise robustness.

Paper: Noise
Hypothesis: Feature importance scores define an importance-weighted distance
    space where per-sample reliability can be measured. Samples whose labels
    are consistent with their importance-weighted neighbors are "reliable";
    inconsistent samples are "unreliable" (noisy). Training with reliability-
    based sample weights should improve predictions.

Key question: Do importance-weighted sample weights help MORE with base reps
    or hybrids?
    - If base reps benefit more  -> hybrids already handle noise implicitly
    - If both benefit equally    -> effects are complementary

5 conditions:
    1. base_unweighted    — individual rep, no sample weights (baseline)
    2. base_weighted      — individual rep, importance-weighted sample weights
    3. hybrid_unweighted  — hybrid rep, no sample weights
    4. hybrid_weighted    — hybrid rep, importance-weighted sample weights
    5. hybrid_cliff_aware — hybrid rep, cliff-aware weights
                            (ActivityCliffAwareEstimator)

Conditions 3, 4 and 5 share one hybrid build per model per fold, and every
condition is dropped from a fold on which any of its fits failed, so the
means they are compared through cover the same folds.

2026-09-12, owner's decision: the scorer that defines the importance-weighted
distance is swept over lightgbm_gain and deeplift. It scales the axes of the
distance the neighbour search runs in; it selects no features and it does not
touch the hybrid's recipe. Superseded 2026-10-06: the scorers are the
dataset's decided importance methods (see below).

2026-10-05, chat 5 of HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.0 (aligned
with section 2.4, Adelaide's rules R1-R6):
  - Folds come from noise_helpers.noise_outer_folds (config['n_splits'],
    kirby.find_best_hybrid.nested_cv_outer_folds), any count including 1.
  - Per outer fold, per each of the dataset's two decided models: the 2 best
    base representations and the 2 best hybrids, all ranked by a search on
    that fold's training rows (noise_helpers.fold_configurations). 2 models x
    4 = 8 fitted configurations (R5). Nothing is read from a whole-dataset
    search any more. Conditions 1-2 run on the two base configurations,
    conditions 3-5 on the two hybrid configurations.
  - The distance-search hybrid and the similarity-search hybrid are
    neighbour spaces in the similarity sweep only; no model is fitted on them
    (handoff 2.3 item 1). From 2026-09-19 to 2026-10-05 conditions 3-5 were
    also fitted on the similarity hybrid; that arm is gone under R5.
  - Neighbour averaging uses a counting scheme (KNN_SEARCH_SPACE_RECORD axis
    5: 'rank' and 'inverse_square'); the unweighted mean is cut.
  - The result file is per fold: every model's configurations, their names,
    the training-row score where the fold search has one, and the held-out
    score of every condition. Means over folds are fold_spread (no
    significance tests).

2026-10-06, decisions Adelaide delegated:
  - The own conditions keep one neighbour count, 10 (--k-neighbors).
  - Clean labels only; n02 injects no label damage.
  - Importance weighting stays (it is this file's question), but the scorers
    are the dataset's decided importance methods
    (decided_grid.decided_search(dataset, 'general')['importance_methods']),
    not a fixed lightgbm_gain + deeplift pair.
  - gp's weighted conditions are recorded as not run, with the reason.
  - Condition 5 (cliff-aware) now averages its neighbours' labels under each
    swept counting scheme, as conditions 2 and 4 already did. Its result is
    per scheme: 'ca' is {scheme: score}.

Usage:
    python n02_sample_weighting.py --dataset logd --reps fast --mode intermediary
    python n02_sample_weighting.py --dataset caco2 --reps all --mode full
    python n02_sample_weighting.py --dataset logd --importance-methods lightgbm_gain
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import inspect

# Torch before anything that loads LightGBM. hybrid.py:3410 records the
# hazard: torch and LightGBM each load an OpenMP runtime and whichever loads
# second segfaults the process, and KIRBY_LGBM_THREADS=1 does not prevent it.
# n02 now scores with deeplift, which is a torch method, so this file reaches
# both. Measured here 2026-09-12 on a 160-row synthetic set: lightgbm_gain
# then deeplift exited 139 with no traceback; deeplift then lightgbm_gain ran
# to the end and gave the same per-scorer numbers. Importing torch up here
# lets the sweep keep lightgbm_gain first, which is what the flat
# weighted_mean and delta keys report. Guarded: a machine without torch runs
# every scorer that is not deeplift.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
from experiments._shared import make_nearest_neighbors  # honours --distance-metric

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric,
    compute_feature_importance, input_fingerprint,
)
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, CONFIGURATION_SPACES,
)

from kirby.utils.noise_estimation import ActivityCliffAwareEstimator
# The similarity-sweep kit, the checkpoint and the disk caches, added
# 2026-09-25 (plan sections 1-3). See n07_remediation's SIMILARITY-SWEEP KIT.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, sweep_schemes, RowSink, sweep_context,
    similarity_sweep, detector_scores_with, cleaned_from_weights,
    similarity_record, FoldCheckpoint,
)
from experiments.noise import noise_cache as _nc
from kirby.utils.noise_estimation import weights_from_scores


# 2026-10-05 (chat 5): WEIGHTED_MODELS (rf, xgboost, lightgbm, svm, dnn on
# every dataset) is gone. The models are the dataset's two decided ones
# (DECISIONS.md Step 1), handed out per fold by fold_configurations.

# The base and hybrid configurations of one model (CONFIGURATION_SPACES order).
BASE_SPACES = ('base_1', 'base_2')
HYBRID_FIT_SPACES = ('hybrid', 'hybrid__general2')
# 'hybrid' is the rank-1 hybrid of the fold search; results['hybrid'] holds it.
HYBRID_SPACE = 'hybrid'

# How hard the weights lean on the neighbour-disagreement score, through
# weights_from_scores: 1.0 is a 2-to-1 row weight between the most and least
# suspect molecule, 3.0 is 4 to 1, 9.0 is 10 to 1. 3.0 because that is the
# setting n18 measured as best on the same question over the same molecules --
# downweight at strength 3.0, 5 of 5 folds, +0.0069 rmse on herg_ki -- and
# because the normaliser this replaced left a ratio of 2.99 to 1 with half the
# molecules inside a nine-percent band (see compute_reliability_scores).
DEFAULT_STRENGTH = 3.0

# Scorers that define the importance-weighted distance.
#
# DECIDED 2026-10-06 (Adelaide delegated): n02's own conditions 2 and 4 keep
# importance weighting, because it is this file's question. The scorers are
# the dataset's decided importance methods (DECISIONS.md Step 3, read through
# decided_grid.decided_search(dataset, 'general')['importance_methods']),
# replacing the fixed lightgbm_gain + deeplift pair of 2026-09-12. The
# similarity sweep below still has no importance weighting
# (KNN_SEARCH_SPACE_RECORD axis 3).
def decided_importance_methods(dataset_name, requested=None):
    """The dataset's decided importance methods, lightgbm_gain first when
    it is one of them (the flat weighted_mean / delta keys hold the first).
    `requested` (from --importance-methods) narrows the list to the decided
    members it names; naming none of them raises."""
    from experiments.decided_grid import decided_search
    decided = list(decided_search(dataset_name, 'general')['importance_methods'])
    decided.sort(key=lambda m: m != 'lightgbm_gain')
    if not requested:
        return decided
    kept = [m for m in decided if m in requested]
    dropped = [m for m in requested if m not in decided]
    if not kept:
        raise ValueError(
            f'{dataset_name}: --importance-methods {list(requested)} names no '
            f'decided importance method (decided: {decided}).')
    if dropped:
        print(f'  [decided grid] {dataset_name}: importance methods {dropped} '
              f'not decided; running {kept}')
    return kept

# Neighbour count of n02's own conditions 2, 4 and 5 (--k-neighbors).
# DECIDED 2026-10-06 (Adelaide delegated): the own conditions keep one count,
# 10. The similarity sweep below covers 5, 10 and 20 through the screen.
DEFAULT_K = 10

# Bumped 2026-10-05 so a checkpoint written by the whole-dataset version is
# never read back into the per-fold one. Bumped 2026-10-06: the scorers are
# the decided importance methods and condition 5 is scored per counting
# scheme.
CHECKPOINT_TAG = 'folds_perfold_2026_10_06'


def model_takes_sample_weight(factory):
    """True when the model's fit() accepts sample_weight. gp (sklearn's
    GaussianProcessRegressor) does not; rf, xgboost, lightgbm, svm and dnn
    do (dnn since 2026-09-17, tests/experiments/noise/_smoke_dnn_sample_weight.py)."""
    try:
        params = inspect.signature(factory().fit).parameters
    except (TypeError, ValueError):
        return False
    return ('sample_weight' in params
            or any(p.kind == p.VAR_KEYWORD for p in params.values()))


def neighbours_without_self(X, k):
    """The k nearest other rows of every row of X (--distance-metric), as
    (indices, distances), each (n, k). The row itself is dropped BY INDEX.

    Dirty-run bug 7 (handoff 2.7), 2026-10-05: the molecule itself is
    dropped by index, as the dirty-local branch did (736ec93; same fix as
    c19 cbcd07e). Dropping column 0 assumed the first neighbour returned is
    the molecule itself, which fails when another training molecule has the
    identical row: at distance 0 either can come first, and then the
    molecule's own label sits in the mean it is compared with while one real
    neighbour is dropped. On herg_ki, 29 to 38 of about 941 training rows per
    fold share their ecfp4 row with another molecule before any importance
    weighting.
    """
    X = np.asarray(X)
    k_actual = min(k, X.shape[0] - 1)
    nn = make_nearest_neighbors(n_neighbors=k_actual + 1)  # +1: query includes self
    nn.fit(X)
    dist, indices = nn.kneighbors(X)
    own = indices == np.arange(len(indices))[:, None]
    has_own = own.any(axis=1)
    # A row whose own index did not come back at all (more than k_actual
    # other rows tied at distance 0) keeps its first k_actual neighbours.
    drop = own.copy()
    drop[~has_own, -1] = True
    return (indices[~drop].reshape(len(indices), k_actual),
            dist[~drop].reshape(len(indices), k_actual))


def compute_cliff_aware_weights(X, y, k=DEFAULT_K, strength=1.0, is_clf=False,
                                scheme=None, neighbours=None):
    """Use ActivityCliffAwareEstimator to compute sample weights.

    Unlike compute_reliability_scores, this distinguishes cliffs from noise:
    cliff compounds get higher weights (they're not noise), while genuinely
    noisy samples get lower weights.

    is_clf must be threaded through: the estimator's default (is_clf=False)
    runs the REGRESSION noise-scoring path (neighbor-mean deviation / local
    variance) on raw class labels, which is meaningless for classification.
    With is_clf=True it uses the class-disagreement x neighbor-purity path.

    DECIDED 2026-10-06 (Adelaide delegated): the estimator averages its
    neighbours' labels under the swept counting scheme (`scheme`, 'rank' or
    'inverse_square'), not with equal weight. The neighbours are found in X
    under --distance-metric with the row itself dropped by index
    (neighbours_without_self); pass `neighbours` = (indices, distances) to
    reuse one list across schemes.
    """
    if scheme is None:
        raise ValueError('compute_cliff_aware_weights needs a counting scheme '
                         '(KNN_SEARCH_SPACE_RECORD axis 5); equal weighting '
                         'is cut')
    idx, dist = neighbours if neighbours is not None else \
        neighbours_without_self(X, k)
    estimator = ActivityCliffAwareEstimator(
        k=idx.shape[1], metric=None, is_clf=is_clf,
        precomputed_neighbors={'idx': idx, 'dist': dist, 'scheme': scheme})
    estimator.fit(X, y)
    return estimator.get_weights(strength=strength)


def compute_reliability_scores(X, y, importance_scores, k=DEFAULT_K,
                               is_clf=False, strength=None, scheme=None):
    """Per-sample reliability as neighbour agreement in importance-weighted space.

    ⛔ THE REGRESSION NORMALISER WAS THE LABEL RANGE AND IT MADE THE WEIGHTS
    ALMOST UNIFORM. It read

        reliability = 1 - |y - mean(y_neighbours)| / (y.max() - y.min())

    and a deviation is on the scale of the label's SPREAD, not its range. On
    herg_ki the range is 5.34 log units against a label standard deviation of
    0.92, so the deviation was divided by about six times the scale it lives
    on. MEASURED there, 1,412 molecules, ecfp4, k=10:

        median weight 0.9449, interquartile spread 0.0862,
        highest-to-lowest ratio 2.99 to 1,
        and 0 of 1,412 molecules reached the 0.1 clip this docstring calls its
        floor.

    Half the molecules sat inside a nine-percent band of each other, so the
    weighted and unweighted fits were near enough the same fit -- which is what
    n02's deltas of about 0.001 in either direction have been saying.

    It now goes through `weights_from_scores`, which is the score-to-weight path
    n07, n11, n18 and n20 already use. That min-max normalises the score, so the
    spread of the weights is set by the spread of the evidence rather than by a
    range that has nothing to do with it, and `strength` says how hard to lean:
    1.0 gives 2 to 1 between the most and least suspect molecule, 3.0 gives 4 to
    1, 9.0 gives 10 to 1.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        y: Labels (n_samples,).
        importance_scores: Per-feature importance (n_features,).
        k: Number of neighbors.
        is_clf: If True, use class-agreement instead of label proximity.
        strength: how hard to lean on the score. None keeps DEFAULT_STRENGTH.
        scheme: how much each neighbour counts, one of
            similarity_space.SWEPT_NEIGHBOUR_SCHEMES (2026-10-05).

    Returns:
        reliability: one weight per row, mean-normalised to 1.
    """
    # Weight features by sqrt(importance). The weights take the matrix's own
    # precision, so a float32 representation stays float32 (handoff 2.3
    # item 18; before 2026-10-05 the float64 weights promoted it).
    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)
    X = np.asarray(X)
    X_weighted = X * weights.astype(X.dtype, copy=False)

    # k nearest neighbours in importance-weighted space, self dropped by index
    neighbor_idx, neighbor_dist = neighbours_without_self(X_weighted, k)
    return reliability_from_neighbours(y, neighbor_idx, is_clf=is_clf,
                                       strength=strength, dist=neighbor_dist,
                                       scheme=scheme)


def reliability_from_neighbours(y, neighbor_idx, is_clf=False, strength=None,
                                *, dist=None, scheme=None):
    """The reliability weight from a neighbour list, the second half of
    compute_reliability_scores, split out 2026-09-25 so the similarity sweep
    can hand it neighbours found in a similarity setting's features.

    neighbor_idx: (n, k) indices into the same rows, the row itself excluded.
    dist / scheme (2026-10-05, KNN_SEARCH_SPACE_RECORD axis 5): the (n, k)
        distances of the list and how much each neighbour counts ('rank' or
        'inverse_square'). The neighbours' mean label is
        similarity_space.weighted_neighbour_label under that scheme; on
        classification the share of neighbours in the same class is weighted
        the same way. The equal-weight mean used before is cut, so a missing
        scheme raises.
    """
    from experiments.noise import similarity_space as ss
    if scheme is None or dist is None:
        raise ValueError(
            'reliability_from_neighbours needs the neighbour distances and a '
            f'counting scheme from {ss.SWEPT_NEIGHBOUR_SCHEMES} '
            '(KNN_SEARCH_SPACE_RECORD axis 5); equal weighting is cut')
    y = np.asarray(y)
    idx = np.asarray(neighbor_idx)
    if is_clf:
        # Classification: weighted share of neighbours in the same class
        w_nb = ss.neighbour_weights(dist, scheme)
        same_class = (y[idx] == y[:, None])
        reliability = (w_nb * same_class).sum(axis=1) / w_nb.sum(axis=1)
        reliability = np.clip(reliability, 0.1, 1.0)
        # 2026-10-05, ported from the dirty-local branch (87a68ce): divided by
        # its mean, as the regression branch (weights_from_scores) and the
        # cliff-aware weight already are. Left on [0.1, 1] the weights
        # averaged below 1, and SVC multiplies C by each row's weight, so the
        # weighted arm was also a smaller-C fit. The ratio between rows is
        # unchanged.
        return reliability / reliability.mean()

    # Regression: how far this label sits from its neighbours' weighted mean,
    # in the label's own units. The number handed on is a SUSPICION score,
    # higher meaning more suspect, which is what weights_from_scores takes.
    neighbor_mean = ss.weighted_neighbour_label(
        y.astype(float), idx, dist, scheme, is_clf=False)
    deviation = np.abs(y - neighbor_mean)
    if not np.isfinite(deviation).any() or np.ptp(deviation) < 1e-12:
        # Every molecule agrees with its neighbours equally. There is nothing
        # to weight by, and min-max normalising a flat vector is undefined.
        return np.ones(len(y))
    return weights_from_scores(
        deviation,
        strength=DEFAULT_STRENGTH if strength is None else float(strength))


def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance-weighted sample weighting across 5 conditions
    (base_unweighted, base_weighted, hybrid_unweighted, hybrid_weighted,
    hybrid_cliff_aware), per outer fold, on each decided model's 2 base
    representations and 2 hybrids chosen on that fold's training rows."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N02: IMPORTANCE-WEIGHTED SAMPLE WEIGHTING")
    print("=" * 70)

    k_neighbors = int(kwargs.get('k_neighbors') or DEFAULT_K)
    # DECIDED 2026-10-06 (Adelaide delegated): the scorers that define the
    # weighted distance are the dataset's decided importance methods.
    # lightgbm_gain stays first where decided, so the flat weighted_mean /
    # delta keys hold it.
    importance_methods = decided_importance_methods(
        dataset_name, kwargs.get('importance_methods'))
    # 2026-10-05: each scorer is crossed with the two counting schemes
    # (KNN_SEARCH_SPACE_RECORD axis 5). A weighted variant is
    # '<scorer>|<scheme>'; the flat keys hold the first one.
    schemes = list(sweep_schemes())
    variants = [f'{m}|{s}' for m in importance_methods for s in schemes]
    primary_variant = variants[0]

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    n_samples = len(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    domain = data.get('domain', 'molecular')
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else n_samples

    # AUROC ('auc') expects continuous scores; passing hard 0/1 class labels
    # produces a degenerate score equivalent to balanced accuracy under
    # thresholding 0.5, not the true ranking-based AUROC. Use predict_proba
    # for binary AUROC; for everything else fall back to predict.
    def _predict_for_metric(model, X):
        if metric_name == 'auc' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return model.predict(X)

    folds, n_folds = noise_outer_folds(data, config)

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Outer folds: {n_folds} (running {[f for f, _, _ in folds]})")
    print(f"  k-neighbors (own conditions): {k_neighbors}")
    print(f"  Importance scorers swept: {', '.join(importance_methods)}; "
          f"counting schemes: {', '.join(schemes)} "
          f"(flat keys = {primary_variant})\n", flush=True)

    # =========================================================================
    # The similarity axis, the D-versus-D' rows and the checkpoint, added
    # 2026-09-25 (plan sections 1-3), aligned 2026-10-05.
    # =========================================================================
    # n02 injects no noise, so D is the clean training fold and D' is it
    # weighted. Its two weightings each read neighbours -- the reliability
    # weight (a label's distance from its neighbours' weighted mean) and the
    # cliff-aware weight -- and similarity_sweep screens every setting of this
    # dataset's group (n07.sweep_settings: KNN_SEARCH_SPACE_RECORD axes 1-6)
    # at every k and counting scheme with no model fit, then refits only the
    # screen's top 3. noise_condition 'none', level 0.0, draw 0.
    #
    # DECIDED 2026-10-06 (Adelaide delegated): n02 runs on clean labels only.
    # It injects no label damage, so the decided noise grid's (level, draw)
    # pairs that KNN_SEARCH_SPACE_RECORD's compute table counted for n02 are
    # not run.
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    sink = RowSink(output_dir, 'n02_sample_weighting', dataset_name)
    ckpt = FoldCheckpoint(output_dir, 'n02_sample_weighting', dataset_name,
                          tag=CHECKPOINT_TAG, fingerprint={
                              'input': input_fingerprint(data, config),
                              'alignment': 'per-fold 2 base + 2 hybrids per '
                                           'decided model, 2026-10-05; '
                                           'decided importance methods and '
                                           'cliff-aware per scheme, '
                                           '2026-10-06',
                              'n_folds': int(n_folds),
                              'importance_methods': list(importance_methods),
                              'schemes': list(schemes),
                              'k_neighbors': int(k_neighbors),
                              'strength': DEFAULT_STRENGTH})
    sweep_ctx = sweep_context(
        experiment='n02_sample_weighting', dataset_name=dataset_name,
        sink=sink, settings=sim_settings, metric=metric, task_type=task_type,
        condition='none', level=0.0, data=data, factories={})

    def _n02_sweep(*, model_name, model_input, X_tr, X_va, y_tr, y_va,
                   train_idx, val_idx, fold, entry):
        if not sink.enabled:
            return
        from experiments.noise.similarity_space import neighbours_at

        def _dprimes(setting, lists, k):
            idx, dist = neighbours_at(lists, k)
            w = reliability_from_neighbours(y_tr, idx, is_clf=is_clf,
                                            dist=dist,
                                            scheme=lists.get('scheme'))
            yield ('reliability_weight', f'strength={DEFAULT_STRENGTH}',
                   cleaned_from_weights(X_tr, y_tr, w))
            # The cliff-aware detector weights the neighbours by the list's
            # counting scheme (lists['scheme'], shared change 2026-10-06; the
            # detector cache is keyed by it).
            sc = detector_scores_with(dataset_name, 'activity_cliff_aware',
                                      X_tr, y_tr, lists, k, is_clf=is_clf)
            if sc is None:
                yield 'cliff_aware_weight', 'strength=1.0', None
                return
            # ActivityCliffAwareEstimator.get_weights(strength=1.0), the
            # weight compute_cliff_aware_weights hands the fit.
            w2 = 1.0 / (1.0 + 1.0 * np.asarray(sc, dtype=float))
            yield ('cliff_aware_weight', 'strength=1.0',
                   cleaned_from_weights(X_tr, y_tr, w2 / w2.mean()))
        # The neighbour spaces are this fold's: the model's 2 base
        # representations and the recipes of its fold search ('hybrid',
        # 'hybrid__general2', 'hybrid__distance1', 'hybrid_similarity').
        # No scaffold groups are excluded from the lists (axis 6).
        similarity_sweep(
            sweep_ctx, model_name=model_name, model_input=model_input,
            X_tr=X_tr, X_va=X_va, y_tr_noisy=y_tr, y_tr_clean=y_tr, y_val=y_va,
            train_idx=train_idx, val_idx=val_idx,
            groups_tr=(np.asarray(groups)[train_idx]
                       if groups is not None else None),
            fold=fold, draw=0, recipes=entry.get('recipes') or {},
            base_reps=entry.get('base') or [], make_dprimes=_dprimes)

    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())} x schemes {schemes}, screened to the top 3; "
          f"rows -> {sink.path}; checkpoint -> {ckpt.path}")

    # =========================================================================
    # One fold's conditions on one configuration
    # =========================================================================
    def _fit_conditions(X_train, X_val, y_train, y_val, model_factory, *,
                        with_cliff, can_weight):
        """Conditions 1-2 (base) or 3-5 (hybrid) on one fold. A fold enters
        every list or none of them, so unweighted, weighted and cliff-aware
        means cover one fold set."""
        try:
            model_uw = model_factory()
            model_uw.fit(X_train, y_train)
            fold_uw = float(metric_fn(
                y_val, _predict_for_metric(model_uw, X_val)))
        except Exception as exc:
            return {'dropped': f'the unweighted fit failed: '
                               f'{type(exc).__name__}: {exc}'}
        out = {'uw': fold_uw, 'n_features': int(np.asarray(X_train).shape[1])}
        if not can_weight:
            # DECIDED 2026-10-06 (Adelaide delegated): the decided model's
            # fit() takes no per-row weight (gp on abagym_Ang2, abagym_VEGF,
            # ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho), so the weighted
            # conditions are recorded as not run, with the reason. Only the
            # unweighted score is kept.
            out['weighted_undefined'] = ('not run: this model\'s fit() takes '
                                         'no sample_weight')
            return out

        fold_w, fold_stats = {}, {}
        for imp_method in importance_methods:
            try:
                # Importance on the SAME (scaled) matrix the reliability
                # scorer consumes.
                imp_scores = compute_feature_importance(
                    {'X': X_train}, y_train, method=imp_method)['X']
            except Exception as exc:
                return {'dropped': f'importance {imp_method} failed: '
                                   f'{type(exc).__name__}: {exc}'}
            for scheme in schemes:
                var = f'{imp_method}|{scheme}'
                try:
                    sample_weights = compute_reliability_scores(
                        X_train, y_train, imp_scores, k=k_neighbors,
                        is_clf=is_clf, scheme=scheme)
                    fold_stats[var] = {
                        'mean': float(np.mean(sample_weights)),
                        'std': float(np.std(sample_weights)),
                        'min': float(np.min(sample_weights)),
                        'pct_below_05': float(
                            np.mean(sample_weights < 0.5) * 100),
                    }
                    model_w = model_factory()
                    model_w.fit(X_train, y_train, sample_weight=sample_weights)
                    fold_w[var] = float(metric_fn(
                        y_val, _predict_for_metric(model_w, X_val)))
                except Exception as exc:
                    return {'dropped': f'the weighted fit {var} failed: '
                                       f'{type(exc).__name__}: {exc}'}
        out['w'] = fold_w
        out['stats'] = fold_stats
        if with_cliff:
            # Condition 5: same hybrid matrix, weights from
            # ActivityCliffAwareEstimator instead of neighbour agreement, one
            # fit per counting scheme (2026-10-06). One neighbour list serves
            # every scheme.
            ca = {}
            try:
                nbrs = neighbours_without_self(X_train, k_neighbors)
                for scheme in schemes:
                    ca_weights = compute_cliff_aware_weights(
                        X_train, y_train, k=k_neighbors, is_clf=is_clf,
                        scheme=scheme, neighbours=nbrs)
                    model_ca = model_factory()
                    model_ca.fit(X_train, y_train, sample_weight=ca_weights)
                    ca[scheme] = float(metric_fn(
                        y_val, _predict_for_metric(model_ca, X_val)))
            except Exception as exc:
                return {'dropped': f'the cliff-aware fit failed: '
                                   f'{type(exc).__name__}: {exc}'}
            out['ca'] = ca
        return out

    # =========================================================================
    # The outer folds
    # =========================================================================
    # {model: {space: [per-fold record]}}
    per_fold_scores = {}
    fold_records = []
    for fold_i, train_idx, val_idx in folds:
        print(f"\n  --- Fold {fold_i} of {n_folds} "
              f"({len(train_idx)} training, {len(val_idx)} held-out rows) ---")
        y_train, y_val = labels[train_idx], labels[val_idx]
        cfgs = fold_configurations(data, dataset_name, fold_i=fold_i,
                                   train_idx=train_idx, n_folds=n_folds)
        # n_samples = this fold's training rows, which is what the per-fold
        # search sized each model by (svm stops above 10,000 rows, gp above
        # 1,000).
        factories = get_model_factories(list(cfgs), task_type=task_type,
                                        n_samples=len(train_idx))
        sweep_ctx['factories'] = factories
        fold_rec = {'fold': int(fold_i), 'n_train': int(len(train_idx)),
                    'n_val': int(len(val_idx)), 'models': {}}
        for model_name, entry in cfgs.items():
            factory = factories.get(model_name)
            if entry is None or factory is None:
                why = ('the per-fold search failed for this model'
                       if entry is None else
                       'no model factory at this training size')
                fold_rec['models'][model_name] = dict(
                    configuration_record(entry), skipped=why)
                print(f"    {model_name}: skipped -- {why}")
                continue
            can_weight = model_takes_sample_weight(factory)
            inputs, unavail, names = fold_model_inputs(
                data, dataset_name, model_name, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_train, factory=factory,
                metric=metric)
            m_rec = configuration_record(entry, names)
            m_rec['configurations_unavailable'] = {
                sp: unavail[sp] for sp in CONFIGURATION_SPACES
                if sp not in inputs and sp in unavail}
            m_rec['takes_sample_weight'] = bool(can_weight)
            # The training-row score of each hybrid configuration: its mean
            # over the fold search's inner folds, on this fold's training rows
            # (configuration_record's 'score_mean'). The base representations'
            # training-row scores are not handed out by fold_configurations.
            train_scores = {}
            for i, sp in enumerate(HYBRID_FIT_SPACES):
                if len(entry.get('hybrids') or []) > i:
                    train_scores[sp] = entry['hybrids'][i].get('score_mean')
            m_rec['scores'] = {}
            for space in CONFIGURATION_SPACES:
                if space not in inputs:
                    print(f"    {model_name} [{space}]: unavailable -- "
                          f"{unavail.get(space)}")
                    continue
                X_tr, X_va = inputs[space]
                cfg_name = names.get(space)
                is_hyb = space in HYBRID_FIT_SPACES
                # The checkpoint key carries the configuration's name and the
                # content of its training matrix, so a changed recipe or
                # scaling recomputes instead of reading back an old score
                # (as dirty-local 401c873 did).
                key = ['fold', int(fold_i), model_name, space, cfg_name,
                       _nc.array_hash(X_tr)]
                rec = ckpt.cached(key, lambda X_tr=X_tr, X_va=X_va: (
                    _fit_conditions(X_tr, X_va, y_train, y_val, factory,
                                    with_cliff=is_hyb,
                                    can_weight=can_weight)))
                # D versus D' and the similarity axis with this configuration
                # as the model input.
                _n02_sweep(model_name=model_name,
                           model_input=f'{model_name}:{space}={cfg_name}',
                           X_tr=X_tr, X_va=X_va, y_tr=y_train, y_va=y_val,
                           train_idx=train_idx, val_idx=val_idx, fold=fold_i,
                           entry=entry)
                row = dict(rec, fold=int(fold_i), name=cfg_name,
                           training_row_score=train_scores.get(space))
                m_rec['scores'][space] = row
                per_fold_scores.setdefault(model_name, {}).setdefault(
                    space, []).append(row)
                if 'dropped' in rec:
                    print(f"    {model_name} [{space}={cfg_name}]: dropped "
                          f"-- {rec['dropped']}")
                    continue
                line = (f"    {model_name} [{space}={cfg_name}]: "
                        f"unweighted={rec['uw']:.4f}")
                if rec.get('w'):
                    _wp = rec['w'][primary_variant]
                    line += (f", weighted={_wp:.4f} "
                             f"(delta={_wp - rec['uw']:+.4f}) "
                             f"[{primary_variant}]")
                if rec.get('ca'):
                    line += (f", cliff_aware={rec['ca'][schemes[0]]:.4f} "
                             f"[{schemes[0]}]")
                print(line)
            fold_rec['models'][model_name] = m_rec
        fold_records.append(fold_rec)

    # =========================================================================
    # Over folds, per (model, configuration)
    # =========================================================================
    def _gain(delta):
        # A delta is weighted minus unweighted, so on rmse the improvement is
        # NEGATIVE and on spearman it is positive. `gain` flips that once, so a
        # positive gain always means the weights helped, whatever the metric.
        if delta is None or not np.isfinite(delta):
            return float('nan')
        return -float(delta) if lower_is_better else float(delta)

    def _summarise(rows):
        ok = [r for r in rows if 'dropped' not in r]
        out = {'per_fold': rows,
               'names_per_fold': {r['fold']: r.get('name') for r in rows},
               'unweighted': fold_spread([r['uw'] for r in ok]),
               'n_folds_scored': len(ok)}
        weighted_ok = [r for r in ok if r.get('w')]
        if weighted_ok:
            by_variant = {}
            for var in variants:
                by_variant[var] = {
                    'weighted': fold_spread([r['w'][var] for r in weighted_ok]),
                    'delta': fold_spread([r['w'][var] - r['uw']
                                          for r in weighted_ok]),
                    'gain': fold_spread([_gain(r['w'][var] - r['uw'])
                                         for r in weighted_ok]),
                    'reliability_stats': [r['stats'][var]
                                          for r in weighted_ok],
                }
            out['by_variant'] = by_variant
            prim = by_variant[primary_variant]
            # The flat keys carry the first variant, so a reader written
            # before the sweep still finds the same field with the same
            # meaning. 'delta' is the mean over folds of weighted minus
            # unweighted.
            out['unweighted_mean'] = out['unweighted']['mean']
            out['weighted_mean'] = prim['weighted']['mean']
            out['delta'] = prim['delta']['mean']
        else:
            out['weighted_undefined'] = (
                ok[0].get('weighted_undefined') if ok else 'no fold scored')
            out['unweighted_mean'] = out['unweighted']['mean']
        ca_ok = [r for r in ok if r.get('ca')]
        if ca_ok:
            # One entry per counting scheme (2026-10-06). The neighbour-
            # agreement weights it is compared with are the first scorer's
            # under the same scheme.
            by_scheme = {}
            for sch in schemes:
                agree_var = f'{importance_methods[0]}|{sch}'
                by_scheme[sch] = {
                    'score': fold_spread([r['ca'][sch] for r in ca_ok]),
                    'delta_vs_unweighted': fold_spread(
                        [r['ca'][sch] - r['uw'] for r in ca_ok]),
                    # Do cliff-aware weights beat plain neighbour-agreement
                    # weights on the hybrid? Positive = cliff-aware gained
                    # more.
                    'cliff_aware_minus_neighbour_agreement': fold_spread([
                        _gain(r['ca'][sch] - r['uw'])
                        - _gain(r['w'][agree_var] - r['uw'])
                        for r in ca_ok if r.get('w')]),
                    'neighbour_agreement_variant': agree_var,
                }
            out['cliff_aware_by_scheme'] = by_scheme
            # The flat key holds the first scheme, as the flat weighted keys
            # hold the first variant.
            out['cliff_aware'] = by_scheme[schemes[0]]
        return out

    configurations = {m: {sp: _summarise(rows) for sp, rows in by_sp.items()}
                      for m, by_sp in per_fold_scores.items()}

    # Question of this file: does weighting help a hybrid more than a single
    # representation? Per fold, the gain on each hybrid configuration minus
    # the gain on base_1 (the rank-1 representation of the same fold search),
    # with the first weighted variant.
    print(f"\n  --- Summary ({metric_name}; gain > 0 = the weights helped; "
          f"{primary_variant}) ---")
    per_model_summary = {}
    for model_name, by_sp in per_fold_scores.items():
        base1 = {r['fold']: r for r in by_sp.get('base_1', [])
                 if 'dropped' not in r and r.get('w')}
        s = {}
        for sp in HYBRID_FIT_SPACES:
            diffs = []
            for r in by_sp.get(sp, []):
                b = base1.get(r['fold'])
                if b is None or 'dropped' in r or not r.get('w'):
                    continue
                diffs.append(_gain(r['w'][primary_variant] - r['uw'])
                             - _gain(b['w'][primary_variant] - b['uw']))
            s[f'{sp}_gain_minus_base_1_gain'] = fold_spread(diffs)
        s['gain_by_configuration'] = {
            sp: (c.get('by_variant') or {}).get(primary_variant, {}).get('gain')
            for sp, c in configurations.get(model_name, {}).items()}
        s['note'] = ('A gain is how much the metric improved when the weights '
                     'were applied, sign-normalised so positive means they '
                     'helped. <hybrid>_gain_minus_base_1_gain is, per fold, '
                     'the hybrid\'s gain minus the gain on that fold\'s rank-1 '
                     'single representation, both chosen by the search on that '
                     'fold\'s training rows.')
        per_model_summary[model_name] = s
        for sp, g in s['gain_by_configuration'].items():
            if g and g.get('mean') is not None:
                sd = g.get('sd')
                print(f"    {model_name} [{sp}]: gain {g['mean']:+.4f}"
                      + (f" (sd {sd:.4f} over {g['n_folds']} folds)"
                         if sd is not None else f" ({g.get('note')})"))

    results = {
        'folds': fold_records,
        'configurations': configurations,
        'per_base': {sp: {m: c[sp] for m, c in configurations.items()
                          if sp in c} for sp in BASE_SPACES},
        'hybrid_by_space': {sp: {m: c[sp] for m, c in configurations.items()
                                 if sp in c} for sp in HYBRID_FIT_SPACES},
        'hybrid': {m: c[HYBRID_SPACE] for m, c in configurations.items()
                   if HYBRID_SPACE in c},
        'per_model_summary': per_model_summary,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'weighted_variants': variants,
        'primary_weighted_variant': primary_variant,
        # Per-dataset screen of the neighbour settings, one entry per
        # (model input, fold): every (setting, k, scheme) scored with no model
        # fit, and the 3 kept for refitting.
        'screen': sweep_ctx.get('screen', []),
        'similarity': similarity_record(
            sim_settings, sink, spaces_unavailable=sweep_ctx['unavailable'],
            checkpoint_file=ckpt.path, folds_read_back=ckpt.n_resumed),
    }

    models_used = sorted({m for fr in fold_records for m in fr['models']})
    return make_structured_output(
        experiment_id='n02',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': int(n_folds),
            'models': models_used,
            'models_rule': ('the dataset\'s two decided models (DECISIONS.md '
                            'Step 1); per fold 2 base representations + 2 '
                            'hybrids each, chosen on the fold\'s training rows'),
            'k_neighbors': k_neighbors,
            'importance_methods': list(importance_methods),
            'importance_methods_rule': ('the dataset\'s decided importance '
                                        'methods (DECISIONS.md Step 3), '
                                        '2026-10-06'),
            'neighbour_schemes': list(schemes),
            'primary_weighted_variant': primary_variant,
            'labels': 'clean only; no label damage injected (2026-10-06)',
            'conditions': [
                'base_unweighted', 'base_weighted',
                'hybrid_unweighted', 'hybrid_weighted',
                'hybrid_cliff_aware',
            ],
            'configuration_spaces': list(CONFIGURATION_SPACES),
            'neighbour_spaces_only': ['hybrid__distance1', 'hybrid_similarity'],
            'selection_scope': 'fold_train_rows',
        },
        results=results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser(
        'N02: Importance-Weighted Sample Weighting',
        has_delta=True, has_augment=True)
    parser.add_argument('--k-neighbors', type=int, default=DEFAULT_K,
                        help='Number of neighbors for n02\'s own reliability '
                             f'and cliff-aware weights (default: {DEFAULT_K})')
    parser.add_argument('--importance-methods', type=str, default='',
                        help='comma-separated scorers defining the weighted '
                             'distance, narrowed to the dataset\'s decided '
                             'importance methods (default: all decided ones, '
                             'lightgbm_gain first where decided)')
    args = parser.parse_args()
    config = get_config(args)

    importance_methods = [m.strip() for m in args.importance_methods.split(',')
                          if m.strip()]

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print("Models: the dataset's two decided models, per fold")
    print(f"k-neighbors: {args.k_neighbors}")
    print(f"Importance scorers: "
          f"{decided_importance_methods(args.dataset, importance_methods)}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_neighbors=args.k_neighbors,
                  importance_methods=importance_methods)
    save_experiment_results(results, args.output, 'n02_sample_weighting',
                            args.dataset)
