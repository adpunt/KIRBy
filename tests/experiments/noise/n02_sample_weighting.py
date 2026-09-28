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
touch the hybrid's recipe. Each scorer's own weighted score is under
results['per_rep'][rep][model]['by_importance'][method] and
results['hybrid'][model]['by_importance'][method]; the flat weighted_mean and
delta keys hold the first scorer in the sweep.

2026-09-12, owner's decision: the subset of the precomputed pool this file may
pick its hybrid from includes similarity hybrids — searches run under
objective='similarity', whose features were chosen to make the distance
geometry predict each molecule's activity from its five nearest neighbours —
not only prediction-built ones.

2026-09-19: both families are now BUILT, not only recorded. The pool names them
in preference order and returns the first that has an entry, so until today
conditions 3 to 5 ran in the prediction-built hybrid alone and the other family
appeared only as a line of metadata. Conditions 3 to 5 now run once per family.
results['hybrid'] and results['hybrid_cliff_aware'] are unchanged — the
prediction-built space, one entry per model — and every space is under
results['hybrid_by_space'] and results['hybrid_cliff_aware_by_space']. n02 does not pass distance=True and that is
unchanged; the pool subset is a separate thing. Every hybrid condition records
which objective its recipe came from, and results['pool_selection'] says which
numbers in the result file are in the dataset's own metric and which are read
straight out of the search.

Usage:
    python n02_sample_weighting.py --dataset logd --reps fast --mode intermediary
    python n02_sample_weighting.py --dataset caco2 --reps all --mode full
    python n02_sample_weighting.py --dataset flab_thermo_tresanco --reps fast --mode full
    python n02_sample_weighting.py --dataset logd --k-neighbors 15
    python n02_sample_weighting.py --dataset logd --importance-methods lightgbm_gain,deeplift
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

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
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    compute_feature_importance,
    get_precomputed_search, get_best_search_for_downstream, build_fold_hybrid,
    fold_hybrid_scoring_kwargs,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See SPACE_FOR_FAMILY below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
    extract_top3_base_reps,
    searched_models, resolve_comparison_models,
)

# 2026-09-18, owner's instruction: the top 2 rows go in everywhere as an extra,
# and nothing already reported changes. These three read rows the search already
# stored and fit no model, so they cost 0 new model fits.
#
# hybrid_deltas is imported under an alias because run() binds a local list
# named hybrid_deltas (the per-model weighted-minus-unweighted deltas it counts
# for the summary line). A plain import would make every mention of that name
# inside run() the local list, and the call below would raise UnboundLocalError.
from experiments._shared import (
    top_base_reps as shared_top_base_reps,
    top_hybrids as shared_top_hybrids,
    hybrid_deltas as shared_hybrid_deltas,
)

from kirby.utils.noise_estimation import (ActivityCliffAwareEstimator,
                                          weights_from_scores)
# The similarity-sweep kit, the checkpoint and the disk caches, added
# 2026-09-25 (plan sections 1-3). See n07_remediation's SIMILARITY-SWEEP KIT.
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, RowSink, sweep_context, similarity_sweep,
    detector_scores_with, cleaned_from_weights, similarity_record,
    FoldCheckpoint, cached_build_fold_hybrid,
)
from experiments.noise import noise_cache as _nc


# Models that support sample_weight in .fit().
#
# svm joined 2026-09-12, owner's decision. The comment above used to be the whole
# reason it was absent, and it does not cover svm: scikit-learn's SVR and SVC both
# accept sample_weight. Four of the roster's five now run here.
#
# dnn joined 2026-09-17. It was absent because DNNRegressor.fit() raised
# TypeError: unexpected keyword argument 'sample_weight', measured 2026-09-12.
# DNNRegressor.fit and DNNClassifier.fit now take sample_weight and multiply
# each row's squared error (or cross-entropy) by it, so all five of the roster
# run here. Checked by tests/experiments/noise/_smoke_dnn_sample_weight.py.
WEIGHTED_MODELS = ['rf', 'xgboost', 'lightgbm', 'svm', 'dnn']

# Which searches a hybrid recipe may come from, and which result-file space each
# family is built into. 'hybrid' is the predictive one and its name is
# unchanged. See the note above the per-family loop in run().
POOL_SUBSET = 'predictive_and_similarity'
HYBRID_SPACE = 'hybrid'
SPACE_FOR_FAMILY = {
    'predictive': HYBRID_SPACE,
    'similarity': 'hybrid_similarity',
}

# How hard the weights lean on the neighbour-disagreement score, through
# weights_from_scores: 1.0 is a 2-to-1 row weight between the most and least
# suspect molecule, 3.0 is 4 to 1, 9.0 is 10 to 1. 3.0 because that is the
# setting n18 measured as best on the same question over the same molecules --
# downweight at strength 3.0, 5 of 5 folds, +0.0069 rmse on herg_ki -- and
# because the normaliser this replaced left a ratio of 2.99 to 1 with half the
# molecules inside a nine-percent band (see compute_reliability_scores).
DEFAULT_STRENGTH = 3.0

# Scorers swept to define the importance-weighted distance. treeshap is
# deliberately absent: the owner excluded it, and it cannot run locally at all.
DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain', 'deeplift']


def compute_cliff_aware_weights(X, y, k=10, strength=1.0, is_clf=False):
    """Use ActivityCliffAwareEstimator to compute sample weights.

    Unlike compute_reliability_scores, this distinguishes cliffs from noise:
    cliff compounds get higher weights (they're not noise), while genuinely
    noisy samples get lower weights.

    is_clf must be threaded through: the estimator's default (is_clf=False)
    runs the REGRESSION noise-scoring path (neighbor-mean deviation / local
    variance) on raw class labels, which is meaningless for classification.
    With is_clf=True it uses the class-disagreement x neighbor-purity path.
    """
    # metric=None reads --distance-metric. It was 'cosine' here, 'cosine' in
    # n07 and 'euclidean' in n12, for the same estimator class.
    estimator = ActivityCliffAwareEstimator(k=k, metric=None, is_clf=is_clf)
    estimator.fit(X, y)
    return estimator.get_weights(strength=strength)


def compute_reliability_scores(X, y, importance_scores, k=10, is_clf=False,
                               strength=None):
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

    The classification branch is untouched: a fraction of disagreeing
    neighbours is already on [0, 1] and needed no normaliser.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        y: Labels (n_samples,).
        importance_scores: Per-feature importance (n_features,).
        k: Number of neighbors.
        is_clf: If True, use class-agreement instead of label proximity.
        strength: how hard to lean on the score. None keeps DEFAULT_STRENGTH.

    Returns:
        reliability: one weight per row, mean-normalised to 1 by
        weights_from_scores on the regression branch.
    """
    # Weight features by sqrt(importance)
    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)
    X_weighted = X * weights

    # Find k nearest neighbors in importance-weighted space
    k_actual = min(k, len(y) - 1)
    nn = make_nearest_neighbors(n_neighbors=k_actual + 1)  # +1 because query includes self
    nn.fit(X_weighted)
    _, indices = nn.kneighbors(X_weighted)
    # DIRTY-LOCAL 2026-09-28: exclude the molecule itself BY INDEX. Dropping
    # column 0 assumed the first neighbour returned is the molecule itself,
    # which fails when another training molecule has the identical row in
    # this weighted space: at distance 0 either can come first, and then the
    # molecule's own label sits in the mean it is compared with while one
    # real neighbour is dropped. On herg_ki, 29 to 38 of about 941 training
    # rows per fold share their ecfp4 row with another molecule before any
    # importance weighting; features with zero importance only add ties.
    # Same fix as c19 (cbcd07e).
    own = indices == np.arange(len(indices))[:, None]
    has_own = own.any(axis=1)
    # A row whose own index did not come back at all (more than k_actual
    # other rows tied at distance 0) keeps its first k_actual neighbours.
    drop = own.copy()
    drop[~has_own, -1] = True
    neighbor_idx = indices[~drop].reshape(len(indices), k_actual)
    return reliability_from_neighbours(y, neighbor_idx, is_clf=is_clf,
                                       strength=strength)


def reliability_from_neighbours(y, neighbor_idx, is_clf=False, strength=None):
    """The reliability weight from a neighbour list, the second half of
    compute_reliability_scores, split out 2026-09-25 so the similarity sweep
    can hand it neighbours found in a similarity setting's features. The
    arithmetic is the lines that followed the neighbour search there,
    unchanged, so compute_reliability_scores returns what it returned.

    neighbor_idx: (n, k) indices into the same rows, the row itself excluded.
    """
    if is_clf:
        # Classification: fraction of neighbors with same class
        same_class = (y[neighbor_idx] == y[:, None]).astype(float)
        reliability = np.mean(same_class, axis=1)
        # Already a fraction on [0, 1]; the clip below is the only guard it
        # needs.
        reliability = np.clip(reliability, 0.1, 1.0)
        # DIRTY-LOCAL 2026-09-28: divided by its mean, as the regression
        # branch (weights_from_scores) and the cliff-aware weight already are.
        # Left on [0.1, 1] the weights averaged below 1, and SVC multiplies C
        # by each row's weight, so the weighted arm was also a smaller-C fit.
        # The ratio between rows is unchanged.
        return reliability / reliability.mean()

    # Regression: how far this label sits from its neighbours' mean, in the
    # label's own units. The number handed on is a SUSPICION score, higher
    # meaning more suspect, which is what weights_from_scores takes.
    neighbor_mean = np.mean(y[neighbor_idx], axis=1)
    deviation = np.abs(y - neighbor_mean)
    if not np.isfinite(deviation).any() or np.ptp(deviation) < 1e-12:
        # Every molecule agrees with its neighbours equally. There is nothing
        # to weight by, and min-max normalising a flat vector is undefined.
        return np.ones(len(y))
    return weights_from_scores(
        deviation,
        strength=DEFAULT_STRENGTH if strength is None else float(strength))


def weight_vs_label_extremity(weights, y, is_clf=False):
    """Spearman correlation between each training molecule's weight and how
    far its label sits from the training-fold mean label. DIRTY-LOCAL
    2026-09-28, a diagnostic only: no fit reads it.

    On herg_ki svm lost 0.010 to 0.017 RMSE to these weights in every feature
    set, against at most 0.0011 for the same weights shuffled, and the loss
    grew with the number of neighbours (0.0148 at 5, 0.0224 at 20, mhg_gnn).
    With many neighbours the neighbour mean approaches the training mean, so
    the weight may be down-weighting the molecules with the most extreme
    labels. A strongly negative value here says it is. None on
    classification, or when the correlation is undefined.
    """
    if is_clf:
        return None
    try:
        from scipy.stats import spearmanr
        y = np.asarray(y, dtype=float)
        rho = spearmanr(np.asarray(weights, dtype=float),
                        np.abs(y - y.mean())).correlation
        return float(rho) if np.isfinite(rho) else None
    except Exception:
        return None


def search_fold_reps(bp, fold_idx, n_folds, pool):
    """The representations the stored search built this outer fold's hybrid on.

    DIRTY-LOCAL 2026-09-28. A recipe with rep_count N below the pool size is
    built on the N representations that fold's own training rows rank highest
    (find_best_hybrid.py:4939-4960). build_fold_hybrid, handed the whole pool
    and no per-fold scores, ranks by Blueprint.rep_scores instead
    (_shared.py:5270-5300): one cross-validated score per representation over
    EVERY molecule, the held-out fold's included. On herg_ki that put ecfp4 in
    every fold's hybrid, while the search used ecfp4 in none, and n02's rf
    hybrid scored 0.6152 RMSE against the search's 0.6058 on the same folds.

    Blueprint.allocation_per_fold records the representations each fold gave
    features to, chosen on that fold's training rows alone. Returns
    (names, source); names is None when the recipe does not cut the pool or
    carries no per-fold record, and build_fold_hybrid then does what it did.
    """
    rc = getattr(bp, 'rep_count', None)
    if rc is None or int(rc) >= len(pool):
        return None, 'whole_pool'
    apf = getattr(bp, 'allocation_per_fold', None)
    if not apf:
        return None, 'blueprint.rep_scores (no allocation_per_fold on the recipe)'
    bp_folds = getattr(bp, 'n_folds', None)
    if bp_folds is not None and int(bp_folds) != int(n_folds):
        return None, (f'blueprint.rep_scores (recipe searched on {bp_folds} '
                      f'folds, n02 runs {n_folds})')
    per = apf.get(fold_idx, apf.get(str(fold_idx)))
    if not per:
        return None, f'blueprint.rep_scores (no allocation for fold {fold_idx})'
    names = [r for r, n in per.items() if n and r in pool]
    if not names:
        return None, f'blueprint.rep_scores (fold {fold_idx} allocation empty)'
    return names, 'allocation_per_fold'


def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance-weighted sample weighting across 5 conditions
    (base_unweighted, base_weighted, hybrid_unweighted, hybrid_weighted,
    hybrid_cliff_aware)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N02: IMPORTANCE-WEIGHTED SAMPLE WEIGHTING")
    print("=" * 70)

    k_neighbors = kwargs.get('k_neighbors', 10)
    # 2026-09-12, owner's decision: sweep the scorer that defines the weighted
    # distance over lightgbm_gain and deeplift, not treeshap. The scorer only
    # scales the axes of the distance the k-nearest-neighbour search runs in —
    # it selects no features and it does not touch the hybrid's recipe.
    # lightgbm_gain stays first, so an unflagged run's flat weighted_mean /
    # delta keys hold the number they held before the sweep existed.
    importance_methods = kwargs.get('importance_methods') or DEFAULT_IMPORTANCE_METHODS
    importance_methods = list(importance_methods)
    primary_importance = importance_methods[0]

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    domain = data.get('domain', 'molecular')

    # AUROC ('auc') expects continuous scores; passing hard 0/1 class labels
    # produces a degenerate score equivalent to balanced accuracy under
    # thresholding 0.5, not the true ranking-based AUROC. Use predict_proba
    # for binary AUROC; for everything else fall back to predict (class
    # labels for F1/accuracy/MCC, raw floats for regression metrics).
    def _predict_for_metric(model, X):
        if metric_name == 'auc' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return model.predict(X)

    # Only use models that support sample_weight
    # DIRTY-LOCAL 2026-09-28: the configured model list (--models, or
    # config['models'] from run_all) was ignored and all five WEIGHTED_MODELS
    # were fitted. Now the intersection, in WEIGHTED_MODELS order; with no
    # configured list it is WEIGHTED_MODELS as before.
    _asked = kwargs.get('models') or config.get('models')
    _weighted = ([m for m in WEIGHTED_MODELS if m in list(_asked)]
                 if _asked else list(WEIGHTED_MODELS))
    model_factories = get_model_factories(
        _weighted, task_type=task_type, n_samples=n_samples)

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    # (Dead `augmentation_strategy = 'uncapped'` removed — was never read.
    # Per-model aug_condition is fetched from the pool below.)
    sample_X = next(iter(reps.values()))

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Models: {list(model_factories.keys())}")
    print(f"  Reps: {len(reps)} ({', '.join(sorted(reps.keys()))})")
    print(f"  k-neighbors: {k_neighbors}, folds: {n_folds}")
    print(f"  Importance scorers swept: {', '.join(importance_methods)} "
          f"(flat keys = {primary_importance})\n", flush=True)

    results = {
        'per_rep': {},      # conditions 1-2: per individual rep
        'hybrid': {},       # conditions 3-4: hybrid
        'summary': {},      # conditions 3-5 share one loop; 5 lands in
                            # results['hybrid_cliff_aware'] below

    }

    # =========================================================================
    # The similarity axis, the D-versus-D' rows and the checkpoint, added
    # 2026-09-25 (plan sections 1-3).
    # =========================================================================
    # n02 injects no noise, so D is the clean training fold and D' is it
    # weighted. Its two weightings each read neighbours -- the reliability
    # weight (a label's distance from its neighbours' mean) and the
    # cliff-aware weight (ActivityCliffAwareEstimator) -- and both are swept
    # over every similarity setting at every k in SWEPT_KS, on the same model
    # input each condition fits, one row per (weighting, setting, k, model):
    # RF, SVM and the condition's model. noise_condition 'none', level 0.0,
    # draw 0.
    #
    # ⛔ n02's own conditions weight the model input by sqrt(importance) before
    # finding neighbours (compute_reliability_scores), so its by_importance
    # numbers are not any swept setting: MODEL_INPUT is the unweighted model
    # input, and base_k|<weighting>|<distance> weights through fold_weights.
    # The conditions are kept exactly as they were.
    #
    # Every fold's conditions are written to a checkpoint as the fold finishes
    # (n02 wrote its result once, at the end) and read back on resume.
    from experiments._shared import input_fingerprint
    sim_settings = sweep_settings()
    sink = RowSink(output_dir, 'n02_sample_weighting', dataset_name)
    ckpt = FoldCheckpoint(output_dir, 'n02_sample_weighting', dataset_name,
                          tag='folds', fingerprint={
                              'input': input_fingerprint(data, config),
                              'models': list(model_factories),
                              'importance_methods': list(importance_methods),
                              'k_neighbors': int(k_neighbors),
                              'strength': DEFAULT_STRENGTH,
                              # DIRTY-LOCAL 2026-09-28: a fold stored before
                              # the classification weights were divided by
                              # their mean is not read back. '+self_by_index':
                              # nor one whose neighbours dropped column 0
                              # instead of the molecule's own index.
                              'weight_rule': 'clf_mean_1+self_by_index'})
    sweep_ctx = sweep_context(
        experiment='n02_sample_weighting', dataset_name=dataset_name,
        sink=sink, settings=sim_settings,
        metric=(metric_name, metric_fn, lower_is_better), task_type=task_type,
        condition='none', level=0.0, data=data, factories=model_factories)
    _sim_need = {s.space for s in sim_settings}
    _sim_info_memo = {}

    def _sim_info(mname):
        """(top-2 single representations, {space: (recipe, aug)}) for one
        model, read from the pool the way the hybrid block below reads it.
        Read only when a swept setting needs a base, augmented or hybrid
        space, and once per model."""
        if mname in _sim_info_memo:
            return _sim_info_memo[mname]
        base, rec = [], {}
        if _sim_need - {'model_input', 'tanimoto_ecfp4', 'forest_proximity'}:
            from experiments._shared import noise_base_reps_by_model
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups, downstream_model=mname, budget=None,
                    augmentations=augmentations, dataset_name=dataset_name,
                    task_type=task_type, data=data, pool_subset=POOL_SUBSET)
                _b, _ = noise_base_reps_by_model({mname: _s}, [mname],
                                                 reps=reps,
                                                 dataset_name=dataset_name,
                                                 verbose=False)
                base = _b.get(mname, [])
                _rs, _ = pool_recipes_by_family(
                    _m, SPACE_FOR_FAMILY,
                    eligible_families=POOL_SUBSETS[POOL_SUBSET])
                rec = {_r['space']: (_r['blueprint'],
                                     _r['aug_condition'] or 'uncapped')
                       for _r in _rs}
            except Exception as _exc:
                sweep_ctx['unavailable'].setdefault(
                    f'{mname}|pool', f'{type(_exc).__name__}: {_exc}')
        _sim_info_memo[mname] = (base, rec)
        return base, rec

    # ── Which single representations each model is fitted on ──
    # 2026-09-27, owner's decision: each model's own top 2, ranked by that
    # model's score in its stored search (_shared.noise_base_reps_by_model),
    # the same pair every other noise experiment uses. n02 used to fit every
    # loaded representation and then pick a "top 3" by the held-out score it
    # was reporting, so the comparison was chosen by its own result.
    _top2_memo = {}

    def _top2_for(mname):
        if mname in _top2_memo:
            return _top2_memo[mname]
        from experiments._shared import noise_base_reps_by_model
        try:
            _s, _ = get_best_search_for_downstream(
                reps, labels, groups, downstream_model=mname, budget=None,
                augmentations=augmentations, dataset_name=dataset_name,
                task_type=task_type, data=data, pool_subset=POOL_SUBSET)
            _b, _miss = noise_base_reps_by_model({mname: _s}, [mname],
                                                 reps=reps,
                                                 dataset_name=dataset_name,
                                                 verbose=False)
            _top2_memo[mname] = list(_b.get(mname, []))
            if not _top2_memo[mname]:
                print(f"    {mname}: no single representation ranked: "
                      f"{_miss.get(mname)}")
        except Exception as _exc:
            print(f"    {mname}: no stored search, so no single "
                  f"representation: {type(_exc).__name__}: {_exc}")
            _top2_memo[mname] = []
        return _top2_memo[mname]

    def _n02_sweep(*, model_name, model_input, X_tr, X_va, y_tr, y_va,
                   train_idx, val_idx, fold):
        if not sink.enabled:
            return
        from experiments.noise.similarity_space import neighbours_at
        base, rec = _sim_info(model_name)

        def _dprimes(setting, lists, k):
            idx = neighbours_at(lists, k)[0]
            w = reliability_from_neighbours(y_tr, idx, is_clf=is_clf)
            yield ('reliability_weight', f'strength={DEFAULT_STRENGTH}',
                   cleaned_from_weights(X_tr, y_tr, w))
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
        similarity_sweep(
            sweep_ctx, model_name=model_name, model_input=model_input,
            X_tr=X_tr, X_va=X_va, y_tr_noisy=y_tr, y_tr_clean=y_tr, y_val=y_va,
            train_idx=train_idx, val_idx=val_idx,
            groups_tr=(groups[train_idx] if groups is not None else None),
            fold=fold, draw=0, recipes=rec, base_reps=base,
            make_dprimes=_dprimes)

    print(f"  Similarity axis: {len(sim_settings)} settings x ks "
          f"{list(sweep_ks())}; rows -> {sink.path}; checkpoint -> "
          f"{ckpt.path}")

    # =========================================================================
    # CONDITIONS 1-2: Individual reps (base_unweighted, base_weighted)
    # =========================================================================
    print("  --- Conditions 1-2: Individual reps ---")

    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        print(f"\n  Rep: {rep_name} (dim={X_rep.shape[1]})")

        rep_results = {}
        for model_name, model_factory in model_factories.items():
            if rep_name not in _top2_for(model_name):
                continue
            # One unweighted list, and one weighted list per importance
            # scorer. A fold only enters any of them if every fit on it
            # succeeded, so all the means below cover the same folds and a
            # delta is a difference over one fold set.
            unweighted_scores = []
            weighted_scores = {m: [] for m in importance_methods}
            reliability_stats = {m: [] for m in importance_methods}

            for _fi, (train_idx, val_idx) in enumerate(cv_split(
                    groups, n_folds, sample_X, labels, mode=mode)):
                y_train, y_val = labels[train_idx], labels[val_idx]
                # noise_cache.cached_scaled, 2026-09-25: the same
                # StandardScaler arithmetic, stored once per (representation
                # content, fold).
                X_train, X_val = _nc.cached_scaled(dataset_name, X_rep,
                                                   train_idx, val_idx)

                # One fold's conditions 1 and 2, written to the checkpoint as
                # it finishes and read back on resume (2026-09-25). A fold a
                # failure dropped is stored as dropped, with the reason.
                def _rep_fold(X_train=X_train, X_val=X_val, y_train=y_train,
                              y_val=y_val, model_factory=model_factory):
                    # Condition 1: unweighted
                    try:
                        model_uw = model_factory()
                        model_uw.fit(X_train, y_train)
                        fold_uw = metric_fn(
                            y_val, _predict_for_metric(model_uw, X_val))
                    except Exception:
                        return {'dropped': 'the unweighted fit failed'}

                    # Condition 2: importance-weighted sample weights, once per
                    # scorer in the sweep.
                    fold_w = {}
                    fold_stats = {}
                    for imp_method in importance_methods:
                        try:
                            # Compute feature importance on the SAME (scaled)
                            # matrix the reliability scorer will consume.
                            # Previously importance was computed on raw
                            # X_rep[train_idx] while reliability ran on scaled
                            # X_train — fine for tree-based importance
                            # (scale-invariant) but breaks the moment `method`
                            # switches to a scale-sensitive one (permutation,
                            # deeplift on continuous features).
                            imp = compute_feature_importance(
                                {rep_name: X_train}, y_train,
                                method=imp_method)
                            imp_scores = imp[rep_name]

                            # Compute per-sample reliability in importance-weighted space
                            sample_weights = compute_reliability_scores(
                                X_train, y_train, imp_scores, k=k_neighbors,
                                is_clf=is_clf)

                            fold_stats[imp_method] = {
                                'mean': float(np.mean(sample_weights)),
                                'std': float(np.std(sample_weights)),
                                'min': float(np.min(sample_weights)),
                                'pct_below_05': float(
                                    np.mean(sample_weights < 0.5) * 100),
                                'rho_weight_vs_label_extremity':
                                    weight_vs_label_extremity(
                                        sample_weights, y_train, is_clf),
                            }

                            model_w = model_factory()
                            model_w.fit(X_train, y_train,
                                        sample_weight=sample_weights)
                            fold_w[imp_method] = metric_fn(
                                y_val, _predict_for_metric(model_w, X_val))
                        except Exception:
                            break

                    # Any failure drops the fold from every list, weighted and
                    # unweighted alike, so no mean is taken over a fold the
                    # others do not have.
                    if len(fold_w) != len(importance_methods):
                        return {'dropped': 'a weighted fit failed'}
                    return {'uw': fold_uw, 'w': fold_w, 'stats': fold_stats}


                # DIRTY-LOCAL 2026-09-28: the key carries the content of the
                # fold's matrices, so a change in how they are scaled (the
                # binary-raw rule of 808df51) recomputes the fold instead of
                # reading back a number made under the old rule.
                _rec = ckpt.cached(['per_rep', rep_name, model_name, _fi,
                                    _nc.array_hash(X_train, X_val)],
                                   _rep_fold)
                # D versus D' and the similarity axis in this representation,
                # 2026-09-25: the reliability weight and the cliff-aware
                # weight with neighbours from every setting at every k.
                _n02_sweep(model_name=model_name,
                           model_input=f'{model_name}:{rep_name}',
                           X_tr=X_train, X_va=X_val, y_tr=y_train, y_va=y_val,
                           train_idx=train_idx, val_idx=val_idx, fold=_fi)
                if 'dropped' in _rec:
                    continue
                fold_uw, fold_w, fold_stats = (_rec['uw'], _rec['w'],
                                               _rec['stats'])

                unweighted_scores.append(fold_uw)
                for imp_method in importance_methods:
                    weighted_scores[imp_method].append(fold_w[imp_method])
                    reliability_stats[imp_method].append(
                        fold_stats[imp_method])

            if not unweighted_scores:
                continue

            uw_mean = float(np.mean(unweighted_scores))
            uw_std = float(np.std(unweighted_scores))

            by_importance = {}
            for imp_method in importance_methods:
                m_scores = weighted_scores[imp_method]
                m_mean = float(np.mean(m_scores))
                by_importance[imp_method] = {
                    'weighted_mean': m_mean,
                    'weighted_std': float(np.std(m_scores)),
                    'delta': m_mean - uw_mean,
                    'reliability_stats': reliability_stats[imp_method],
                }

            # The flat keys carry the first scorer in the sweep, so a reader
            # written before the sweep still finds the same field with the
            # same meaning. Every scorer's own numbers sit in by_importance.
            primary = by_importance[primary_importance]
            w_mean = primary['weighted_mean']
            delta = primary['delta']

            rep_results[model_name] = {
                'unweighted_mean': uw_mean,
                'unweighted_std': uw_std,
                'weighted_mean': w_mean,
                'weighted_std': primary['weighted_std'],
                'delta': delta,
                'reliability_stats': primary['reliability_stats'],
                'n_folds_scored': len(unweighted_scores),
                'importance_methods': list(importance_methods),
                'primary_importance_method': primary_importance,
                'by_importance': by_importance,
            }

            sign = '+' if delta > 0 else ''
            print(f"    {model_name}: unweighted={uw_mean:.4f}, "
                  f"weighted={w_mean:.4f} (delta={sign}{delta:.4f}) "
                  f"[{primary_importance}]")
            _rhos = [s.get('rho_weight_vs_label_extremity')
                     for s in primary['reliability_stats']]
            _rhos = [r for r in _rhos if r is not None]
            if _rhos:
                print(f"      {'':<{len(model_name)}}  spearman(weight, "
                      f"|label - training mean|) per fold: "
                      f"{', '.join(f'{r:+.3f}' for r in _rhos)}")
            for imp_method in importance_methods[1:]:
                _b = by_importance[imp_method]
                print(f"      {'':<{len(model_name)}}  weighted="
                      f"{_b['weighted_mean']:.4f} "
                      f"(delta={_b['delta']:+.4f}) [{imp_method}]")

        results['per_rep'][rep_name] = rep_results

    # =========================================================================
    # CONDITIONS 3-5: Get hybrid blueprint ONCE (shared across all fold loops)
    # =========================================================================
    # 2026-09-12, owner's decision: n02's pool subset includes similarity
    # hybrids. The pool may now pick a hybrid whose features were chosen under
    # objective='similarity' — chosen to make the distance geometry predict
    # each molecule's activity from its five nearest neighbours — not only one
    # chosen to predict the label with a downstream model. This is NOT
    # distance=True; n02 does not pass distance=True and that is unchanged.
    # Every result below records which objective the winning hybrid came from.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset=POOL_SUBSET,
    )
    # 2026-09-12: which family this top-level lookup landed in, and in what
    # units its stored numbers are. Under 'similarity' the search's stored
    # score is a Spearman correlation between each held-out molecule's activity
    # and what its five nearest neighbours predict; under 'predictive' it is
    # the dataset's own metric. The two are not comparable and the pool does
    # not rank them against each other.
    pool_family = meta.get('objective') if meta else None
    pool_metric_name = meta.get('metric_name') if meta else None
    pool_family_choice = meta.get('objective_choice') if meta else None
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            # 2026-09-12: the label on this line used to say "by RF"
            # unconditionally. Under objective='similarity' find_best_hybrid
            # stores the SAME model-independent geometry number under every
            # model name, so "by RF" would name a model that never ran. Say
            # which family and which metric these numbers are in.
            _lbl = (f"by RF, {pool_metric_name}" if pool_family == 'predictive'
                    else f"family={pool_family}, {pool_metric_name}")
            print(f"  Top-3 base reps ({_lbl}): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    # DIRTY-LOCAL 2026-09-28: a missing random-forest recipe no longer ends the
    # run. It used to return {'error': 'no_blueprint'} here, which threw away
    # every condition 1 and 2 number already computed above and skipped the
    # hybrid for models that have a recipe of their own. This recipe is read
    # by nothing below; each model's hybrid comes from its own lookup, and a
    # model with none is recorded in hybrid_recipe_missing.
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: the pool holds no random-forest recipe; each model's "
              "hybrid is still looked up on its own below")
        results['rf_recipe_missing'] = True

    # Matched evaluator: fetch a per-model blueprint instead of reusing the
    # RF one for xgboost/lightgbm/ridge. Per project_model_matching the
    # downstream model used to evaluate a hybrid must match the model the
    # search optimised against. Each call is cached, so the extra fetches
    # cost ~nothing when the pool already has the entry. Also align each
    # per-model bp's aug_strategy with its own meta['aug_condition'] to
    # avoid the 2026-04-21 cache-collision pattern.
    bp_by_model = {}
    aug_by_model = {}
    # 2026-09-18: keep each model's winning SearchResult, not just the recipe
    # object taken out of it. Without it the top-2 block at the end of run()
    # had only the rf search to read, and it would have reported rf's stored
    # rows under all five downstream model names.
    search_by_model = {}
    # 2026-09-12, owner's decision: n02's pool subset includes similarity
    # hybrids, so each model's hybrid can come from either family. Record the
    # family per model — two models in the same run can land in different
    # families, and a reader cannot tell them apart from the score alone.
    family_by_model = {}
    # Each model's own pool metadata, kept so the per-family loop below can read
    # every family's best for that model and not only the one the pool returned.
    meta_by_model = {}
    # {downstream model: why it has no recipe of its own}. A model in here is
    # scored on its single representations only; conditions 3, 4 and 5 write no
    # row for it, because the only recipe on offer belonged to another model.
    hybrid_recipe_missing = {}
    for _mname in model_factories:
        try:
            _search, _meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset=POOL_SUBSET,
            )
            if _meta is not None:
                _bp = _meta['best_config']['blueprint']
                _aug_cond = _meta.get('aug_condition', 'uncapped')
                if getattr(_bp, 'augmentation_strategy', None) != _aug_cond:
                    try:
                        _bp.augmentation_strategy = _aug_cond
                    except Exception:
                        pass
                bp_by_model[_mname] = _bp
                aug_by_model[_mname] = _aug_cond
                meta_by_model[_mname] = _meta
                if _search is not None:
                    search_by_model[_mname] = _search
                # Every eligible family's own best sits beside the winner,
                # because the pool refuses to rank the two against each other
                # when their metric names differ.
                family_by_model[_mname] = {
                    'objective': _meta.get('objective'),
                    'objective_choice': _meta.get('objective_choice'),
                    'pool_subset': _meta.get('pool_subset'),
                    'search_metric_name': _meta.get('metric_name'),
                    'search_lower_is_better': _meta.get('lower_is_better'),
                    'search_score_mean': _meta.get('score_mean'),
                    'tier': _meta.get('tier'),
                    'aug_condition': _aug_cond,
                    'by_objective': {
                        _o: {'tier': _d.get('tier'),
                             'aug_condition': _d.get('aug_condition'),
                             'score_mean': _d.get('score_mean'),
                             'metric_name': _d.get('metric_name'),
                             'lower_is_better': _d.get('lower_is_better')}
                        for _o, _d in (_meta.get('by_objective') or {}).items()
                    },
                    'blueprint_source': 'own_model',
                }
                print(f"    {_mname}: hybrid from family "
                      f"{_meta.get('objective')!r} "
                      f"({_meta.get('objective_choice')}), search metric "
                      f"{_meta.get('metric_name')!r}")
            else:
                # ⛔ NO SUBSTITUTE RECIPE. Until 2026-09-28 the two branches
                # here copied `bp` -- the recipe the pool returned for the
                # random forest -- into this model, tagged
                # 'rf_fallback_no_pool_entry' or 'rf_fallback_fetch_failed'.
                # Conditions 3, 4 and 5 then trained this model on features a
                # random forest's search had chosen, and the row went into the
                # result file under this model's own name. The features were
                # chosen for one model and the score was produced by another,
                # so the number did not measure what its name said. It is now
                # recorded as unscored: no recipe, no hybrid space, no row.
                # The reason lands in results['hybrid_recipe_missing'] and in
                # results['hybrid_spaces_unavailable'].
                hybrid_recipe_missing[_mname] = (
                    'the precomputed pool holds no search for this downstream '
                    'model, so it has no recipe of its own; no other model\'s '
                    'recipe is put in its place')
                print(f"    {_mname}: no hybrid arm -- the pool holds no "
                      f"search for {_mname}, and the random forest's recipe "
                      f"is not substituted")
        except Exception as _exc:
            hybrid_recipe_missing[_mname] = (
                f'the pool lookup for this downstream model raised '
                f'{type(_exc).__name__}: {_exc}; it has no recipe of its own '
                f'and no other model\'s recipe is put in its place')
            print(f"  WARNING: blueprint fetch failed for {_mname} "
                  f"({_exc}); {_mname} is recorded as unscored on hybrid "
                  f"features, not given the random forest's recipe.")

    # ── One recipe per (model, family), not one per model ─────────────────
    # ⛔ Added 2026-09-19. POOL_SUBSETS names the two families in PREFERENCE
    # order and the pool returns the first that has an entry, because their
    # stored scores are in different units. A predictive recipe always exists,
    # so from 2026-09-12 to 2026-09-19 this file asked for both families, built
    # the predictive one and wrote the other into the result file as a score it
    # never measured in.
    #
    # It matters here because conditions 4 and 5 both weight a molecule by how
    # far it sits from its neighbours in the space the features define, and the
    # similarity search chooses features so that exactly that distance predicts
    # activity from the five nearest neighbours.
    #
    # 'hybrid' is still the predictive recipe, so every number reported before
    # 2026-09-19 still means what it meant.
    recipes_for_model = {}
    space_objective = {}
    hybrid_spaces_unavailable = {}
    for _mname in model_factories:
        _recipes, _unavail = pool_recipes_by_family(
            meta_by_model.get(_mname), SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS[POOL_SUBSET])
        _per_space = {}
        for _r in _recipes:
            _r_bp = _r['blueprint']
            _r_aug = _r['aug_condition'] or 'uncapped'
            # Same alignment the per-model fetch above does: a recipe whose
            # augmentation_strategy disagrees with the pool entry it came from
            # rebuilds under a different condition than the one that won, which
            # is the 2026-04-21 cache-collision pattern.
            if getattr(_r_bp, 'augmentation_strategy', None) != _r_aug:
                try:
                    _r_bp.augmentation_strategy = _r_aug
                except Exception:
                    pass
            _per_space[_r['space']] = (_r_bp, _r_aug)
            space_objective[_r['space']] = _r['objective']
        recipes_for_model[_mname] = _per_space
        if _unavail:
            hybrid_spaces_unavailable[_mname] = _unavail
    results['hybrid_spaces_unavailable'] = hybrid_spaces_unavailable
    results['hybrid_recipe_missing'] = dict(hybrid_recipe_missing)
    results['hybrid_space_objective'] = dict(space_objective)
    results['hybrid_spaces_note'] = (
        'One hybrid space per family of stored search, per model. A predictive '
        "search chose its features to predict the label; a similarity search "
        "chose them so that each molecule's activity is predicted by its five "
        'nearest neighbours in the distance those features define, and fits no '
        'downstream model. The two searches score their own candidates in '
        'different units and nothing here ranks one against the other. Every '
        "number under 'hybrid_by_space' is measured by n02 itself with the "
        "dataset's own metric, on the same folds, so those ARE comparable. "
        "'hybrid' and 'hybrid_cliff_aware' hold the predictive space alone and "
        'are unchanged.')
    for _sp in sorted(space_objective):
        print(f"  hybrid space {_sp}: family {space_objective[_sp]!r}")
    for _mname, _unavail in hybrid_spaces_unavailable.items():
        for _sp, _why in _unavail.items():
            print(f"  {_mname}: no {_sp} space — {_why}")

    # =========================================================================
    # CONDITIONS 3-5: Hybrid (hybrid_unweighted, hybrid_weighted,
    #                         hybrid_cliff_aware)
    # =========================================================================
    # 2026-09-12, owner's decision: condition 5 runs inside this loop instead
    # of its own. It used the same blueprint, the same augmentations and the
    # same fold indices, so its separate loop rebuilt the hybrid a second time
    # and threw the first copy away — 3 models x 5 folds = 15 rebuilds per
    # dataset. All three conditions now score the matrix this fold built once.
    print(f"\n  --- Conditions 3-5: Hybrid ---")

    # {space: {model: row}}. One space per family of stored search, keyed by
    # the result-file name in SPACE_FOR_FAMILY. Added 2026-09-19.
    hybrid_by_space = {}
    cliff_aware_by_space = {}
    # {space: {model: the first per-fold build failure}}. A space whose every
    # fold failed would otherwise leave no row and no reason.
    hybrid_build_errors = {}
    for model_name, model_factory in model_factories.items():
        # One pass per hybrid space: the recipe the pool returned for this
        # model, and every other family's own recipe for the same model.
        # Added 2026-09-19. Conditions 3, 4 and 5 are scored in each space,
        # on the same folds, with the same weights rule.
        for space_name, (model_bp, model_aug_cond) in \
                recipes_for_model.get(model_name, {}).items():
            model_augs = augmentations if model_aug_cond != 'none' else None
            # As in the base-rep block: a fold enters every list or none of them,
            # so unweighted, weighted and cliff-aware means cover one fold set.
            unweighted_scores = []
            weighted_scores = {m: [] for m in importance_methods}
            reliability_stats = {m: [] for m in importance_methods}
            ca_scores = []
            n_features_list = []
            fold_reps_used = {}

            for _fi, (train_idx, val_idx) in enumerate(cv_split(
                    groups, n_folds, sample_X, labels, mode=mode)):
                # DIRTY-LOCAL 2026-09-28: the fold's own representations, as
                # the search chose them on this fold's training rows
                # (search_fold_reps). Before this, the rebuild ranked by a
                # whole-dataset score that saw this fold's held-out labels.
                _keep, _keep_src = search_fold_reps(
                    model_bp, _fi, n_folds, list(reps))
                fold_reps_used[_fi] = {
                    'reps': (sorted(_keep) if _keep is not None else None),
                    'source': _keep_src}
                _fold_pool = (reps if _keep is None
                              else {k: v for k, v in reps.items() if k in _keep})
                train_reps = {k: v[train_idx] for k, v in _fold_pool.items()}
                val_reps = {k: v[val_idx] for k, v in _fold_pool.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Create hybrid per fold from THIS MODEL'S blueprint
                try:
                    train_augs = {k: v[train_idx] for k, v in model_augs.items()} if model_augs else None
                    val_augs = {k: v[val_idx] for k, v in model_augs.items()} if model_augs else None
                    # Stored through noise_cache since 2026-09-25
                    # (n07.cached_build_fold_hybrid): the same call, keyed on
                    # its inputs, so a resumed run does not rebuild it.
                    X_train_h, X_val_h, feature_info = cached_build_fold_hybrid(
                        dataset_name,
                        train_reps, val_reps, y_train, model_bp,
                        augmentations_train=train_augs, augmentations_val=val_augs,
                        groups=(groups[train_idx] if groups is not None else None),
                        # ⛔ Through the helper, not passed straight. A recipe
                        # from the similarity family fits no downstream model
                        # and build_fold_hybrid RAISES when a model and a
                        # metric arrive with one; the except below would have
                        # swallowed that and left the geometry space with no
                        # fold at all and nothing said. For a predictive recipe
                        # the three come back unchanged.
                        **fold_hybrid_scoring_kwargs(
                            model_bp, model=model_factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as e:
                    print(f"    {model_name} [{space_name}]: hybrid creation "
                          f"failed - {e}")
                    hybrid_build_errors.setdefault(space_name, {}).setdefault(
                        model_name, f'{type(e).__name__}: {e}')
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train_h)
                X_val_s = scaler.transform(X_val_h)

                # One fold's conditions 3 to 5, written to the checkpoint as
                # it finishes and read back on resume (2026-09-25).
                def _hyb_fold(X_train_s=X_train_s, X_val_s=X_val_s,
                              y_train=y_train, y_val=y_val,
                              model_factory=model_factory):
                    # Condition 3: hybrid unweighted
                    try:
                        model_uw = model_factory()
                        model_uw.fit(X_train_s, y_train)
                        fold_uw = metric_fn(
                            y_val, _predict_for_metric(model_uw, X_val_s))
                    except Exception:
                        return {'dropped': 'the unweighted fit failed'}

                    # Condition 4: hybrid + importance-weighted sample weights, once
                    # per scorer in the sweep.
                    fold_w = {}
                    fold_stats = {}
                    for imp_method in importance_methods:
                        try:
                            # Compute importance on the SAME (scaled) hybrid matrix
                            # the reliability scorer consumes — same alignment fix as
                            # the base-rep block above. The raw X_train_h was
                            # previously passed to compute_feature_importance while
                            # X_train_s (the scaled version) was passed to
                            # compute_reliability_scores.
                            imp = compute_feature_importance(
                                {'hybrid': X_train_s}, y_train, method=imp_method)
                            imp_scores = imp['hybrid']

                            sample_weights = compute_reliability_scores(
                                X_train_s, y_train, imp_scores, k=k_neighbors,
                                is_clf=is_clf)

                            fold_stats[imp_method] = {
                                'mean': float(np.mean(sample_weights)),
                                'std': float(np.std(sample_weights)),
                                'min': float(np.min(sample_weights)),
                                'pct_below_05': float(
                                    np.mean(sample_weights < 0.5) * 100),
                                'rho_weight_vs_label_extremity':
                                    weight_vs_label_extremity(
                                        sample_weights, y_train, is_clf),
                            }

                            model_w = model_factory()
                            model_w.fit(X_train_s, y_train,
                                        sample_weight=sample_weights)
                            fold_w[imp_method] = metric_fn(
                                y_val, _predict_for_metric(model_w, X_val_s))
                        except Exception:
                            break

                    if len(fold_w) != len(importance_methods):
                        return {'dropped': 'a weighted fit failed'}

                    # Condition 5: same hybrid matrix, weights from
                    # ActivityCliffAwareEstimator instead of neighbour agreement.
                    try:
                        ca_weights = compute_cliff_aware_weights(
                            X_train_s, y_train, k=k_neighbors, is_clf=is_clf)
                        model_ca = model_factory()
                        model_ca.fit(X_train_s, y_train, sample_weight=ca_weights)
                        fold_ca = metric_fn(
                            y_val, _predict_for_metric(model_ca, X_val_s))
                    except Exception:
                        return {'dropped': 'the cliff-aware fit failed'}
                    return {'uw': fold_uw, 'w': fold_w, 'stats': fold_stats,
                            'ca': fold_ca}

                # DIRTY-LOCAL 2026-09-28: the key carries the content of the
                # built hybrid. Without it a rerun whose pool returned a
                # different recipe read back the old recipe's scores under
                # the new recipe's name.
                _rec = ckpt.cached(['hybrid', model_name, space_name, _fi,
                                    _nc.array_hash(X_train_s, X_val_s)],
                                   _hyb_fold)
                # D versus D' and the similarity axis in this hybrid,
                # 2026-09-25.
                _n02_sweep(model_name=model_name,
                           model_input=f'{model_name}:{space_name}',
                           X_tr=X_train_s, X_va=X_val_s, y_tr=y_train,
                           y_va=y_val, train_idx=train_idx, val_idx=val_idx,
                           fold=_fi)
                if 'dropped' in _rec:
                    continue
                fold_uw, fold_w, fold_stats, fold_ca = (
                    _rec['uw'], _rec['w'], _rec['stats'], _rec['ca'])

                unweighted_scores.append(fold_uw)
                for imp_method in importance_methods:
                    weighted_scores[imp_method].append(fold_w[imp_method])
                    reliability_stats[imp_method].append(fold_stats[imp_method])
                ca_scores.append(fold_ca)
                n_features_list.append(X_train_h.shape[1])

            if not unweighted_scores:
                continue

            uw_mean = float(np.mean(unweighted_scores))
            uw_std = float(np.std(unweighted_scores))

            by_importance = {}
            for imp_method in importance_methods:
                m_scores = weighted_scores[imp_method]
                m_mean = float(np.mean(m_scores))
                by_importance[imp_method] = {
                    'weighted_mean': m_mean,
                    'weighted_std': float(np.std(m_scores)),
                    'delta': m_mean - uw_mean,
                    'reliability_stats': reliability_stats[imp_method],
                }

            primary = by_importance[primary_importance]
            w_mean = primary['weighted_mean']
            w_std = primary['weighted_std']
            delta = primary['delta']
            n_features = int(np.mean(n_features_list)) if n_features_list else 0

            hybrid_by_space.setdefault(space_name, {})[model_name] = {
                'unweighted_mean': uw_mean,
                'unweighted_std': uw_std,
                'weighted_mean': w_mean,
                'weighted_std': w_std,
                'delta': delta,
                'n_features': n_features,
                'fold_reps': fold_reps_used,
                'reliability_stats': primary['reliability_stats'],
                'n_folds_scored': len(unweighted_scores),
                'importance_methods': list(importance_methods),
                'primary_importance_method': primary_importance,
                'by_importance': by_importance,
                # 2026-09-12, owner's decision (pool subset includes similarity
                # hybrids): which family chose this hybrid's recipe. The two
                # scores above are measured by n02 itself with metric_fn, so they
                # are in the dataset's own metric whichever family chose the
                # recipe; this key says what chose it.
                'hybrid_objective': space_objective.get(space_name),
                'hybrid_objective_choice': family_by_model.get(
                    model_name, {}).get('objective_choice'),
                'pool_subset': family_by_model.get(model_name, {}).get(
                    'pool_subset'),
                # Which result-file space these numbers are in, and whether it
                # holds the recipe the pool returned for this model or another
                # family's own best. Added 2026-09-19.
                'space': space_name,
                'is_the_recipe_the_pool_returned': (
                    space_objective.get(space_name)
                    == family_by_model.get(model_name, {}).get('objective')),
                'features_chosen_by': describe_pool_family(
                    space_objective.get(space_name)),
            }

            # Condition 5, assembled from the lists the merged loop produced. Its
            # folds are the folds behind uw_mean, so this subtraction is over one
            # fold set.
            ca_mean = float(np.mean(ca_scores))
            cliff_aware_by_space.setdefault(space_name, {})[model_name] = {
                'score_mean': ca_mean,
                'score_std': float(np.std(ca_scores)),
                'delta_vs_unweighted': ca_mean - uw_mean,
                'n_folds_scored': len(ca_scores),
                # 2026-09-12: same hybrid recipe as conditions 3-4, so the
                # same family chose it. Recorded here too so this block reads
                # on its own.
                'hybrid_objective': space_objective.get(space_name),
                'space': space_name,
            }

            sign = '+' if delta > 0 else ''
            _fam = space_objective.get(space_name)
            print(f"    {model_name} [{space_name}]: unweighted={uw_mean:.4f}, "
                  f"weighted={w_mean:.4f} (delta={sign}{delta:.4f}, "
                  f"dim={n_features}, family={_fam}) [{primary_importance}]")
            for _f, _fr in sorted(fold_reps_used.items()):
                print(f"      {'':<{len(model_name)}}  fold {_f} "
                      f"representations: {_fr['reps'] or 'whole pool cut by the rebuild'}"
                      f" ({_fr['source']})")
            for imp_method in importance_methods[1:]:
                _b = by_importance[imp_method]
                print(f"      {'':<{len(model_name)}}  weighted="
                      f"{_b['weighted_mean']:.4f} "
                      f"(delta={_b['delta']:+.4f}) [{imp_method}]")
            print(f"      {'':<{len(model_name)}}  cliff_aware={ca_mean:.4f} "
                  f"(delta={ca_mean - uw_mean:+.4f})")

    # 'hybrid' and 'hybrid_cliff_aware' keep the shape and the numbers they had:
    # the space built from the predictive search, one entry per model. Every
    # space this run scored, that one included, is under the '*_by_space' keys.
    hybrid_results = hybrid_by_space.get(HYBRID_SPACE, {})
    cliff_aware_results = cliff_aware_by_space.get(HYBRID_SPACE, {})
    results['hybrid'] = hybrid_results
    results['hybrid_cliff_aware'] = cliff_aware_results
    results['hybrid_by_space'] = hybrid_by_space
    results['hybrid_cliff_aware_by_space'] = cliff_aware_by_space
    results['hybrid_spaces_scored'] = sorted(hybrid_by_space)
    results['hybrid_space_build_errors'] = hybrid_build_errors

    # =========================================================================
    # SUMMARY: Compare weighting benefit for base reps vs hybrid
    # =========================================================================
    print(f"\n  --- Summary ---")

    hybrid_deltas = []
    for model_name, model_res in results['hybrid'].items():
        d = model_res.get('delta', float('nan'))
        if not np.isnan(d):
            hybrid_deltas.append(d)
    n_hybrid_improved = int(sum(1 for d in hybrid_deltas if (d < 0 if lower_is_better else d > 0)))
    print(f"  Hybrid weighting: {n_hybrid_improved}/{len(hybrid_deltas)} models improved")

    # Per-model comparison: the model's own top-2 single representations vs
    # the hybrid. The two are taken in the order the model's STORED SEARCH
    # ranked them (_top2_for), never re-sorted by the held-out score reported
    # here, so the comparison is not chosen by its own result.
    print(f"\n  PER-MODEL COMPARISON (each model's top-2 single "
          f"representations vs hybrid):")
    results['per_model_top2'] = {}
    for model_name in model_factories:
        top2 = []
        for rep_name in _top2_for(model_name):
            _rr = results['per_rep'].get(rep_name, {}).get(model_name)
            if not _rr:
                continue
            uw = _rr.get('unweighted_mean', float('nan'))
            if not np.isnan(uw):
                top2.append((rep_name, uw,
                             _rr.get('weighted_mean', float('nan')),
                             _rr.get('delta', float('nan'))))

        model_hybrid_uw = float('nan')
        model_hybrid_w = float('nan')
        model_hybrid_delta = float('nan')
        if model_name in results['hybrid']:
            model_hybrid_uw = results['hybrid'][model_name].get('unweighted_mean', float('nan'))
            model_hybrid_w = results['hybrid'][model_name].get('weighted_mean', float('nan'))
            model_hybrid_delta = results['hybrid'][model_name].get('delta', float('nan'))

        # ── The two subtractions this file used to leave to the reader ──
        # EXPERIMENTS_FINALIZED.md records both as "never computed". Neither
        # costs a model fit: every number in them is already above.
        #
        # A delta is weighted minus unweighted, so on rmse the improvement is
        # NEGATIVE and on spearman it is positive. `gain` flips that once, so a
        # positive gain always means the weights helped, whatever the metric.
        def _gain(delta):
            if delta is None or np.isnan(delta):
                return float('nan')
            return -float(delta) if lower_is_better else float(delta)

        # "Best" = rank 1 in the model's stored search.
        _best_rep_name = top2[0][0] if top2 else None
        _best_rep_gain = _gain(top2[0][3]) if top2 else float('nan')
        _hybrid_gain = _gain(model_hybrid_delta)
        # Question 2 of this file: does the improvement differ between an
        # individual representation and the hybrid? Positive means the weights
        # bought the hybrid more than they bought the best single
        # representation for this model.
        _hyb_minus_base = (_hybrid_gain - _best_rep_gain
                           if not (np.isnan(_hybrid_gain)
                                   or np.isnan(_best_rep_gain))
                           else float('nan'))

        # Question 3: do cliff-aware weights beat plain neighbour-agreement
        # weights ON THE HYBRID? What was stored is the cliff-aware arm against
        # the UNWEIGHTED hybrid; the question asks for it against the
        # neighbour-agreement arm, which is that number minus 'delta'.
        _ca = (results.get('hybrid_cliff_aware', {})
               .get(model_name, {}).get('delta_vs_unweighted', float('nan')))
        _ca_gain = _gain(_ca)
        _ca_minus_neighbour = (_ca_gain - _hybrid_gain
                               if not (np.isnan(_ca_gain)
                                       or np.isnan(_hybrid_gain))
                               else float('nan'))

        results['per_model_top2'][model_name] = {
            'top2': [(n, uw, w, d) for n, uw, w, d in top2],
            'hybrid_unweighted': model_hybrid_uw,
            'hybrid_weighted': model_hybrid_w,
            'hybrid_delta': model_hybrid_delta,
            # Positive = the weights helped, whatever the metric's direction.
            'weighting_gain_hybrid': _hybrid_gain,
            'weighting_gain_best_base_rep': _best_rep_gain,
            'best_base_rep': _best_rep_name,
            'hybrid_gain_minus_best_base_gain': _hyb_minus_base,
            'cliff_aware_gain': _ca_gain,
            'cliff_aware_minus_neighbour_agreement': _ca_minus_neighbour,
            'gains_note': (
                'A gain is how much the metric improved when the weights were '
                'applied, sign-normalised so positive always means they '
                'helped. hybrid_gain_minus_best_base_gain is the hybrid\'s '
                'gain minus the gain on the rank-1 single representation '
                'in this model\'s stored search. '
                'cliff_aware_minus_neighbour_agreement is the cliff-aware '
                'arm\'s gain minus the neighbour-agreement arm\'s, both on '
                'the hybrid.'),
        }

        print(f"\n  {model_name}:")
        for rn, uw, w, d in top2:
            print(f"    {rn:<20} uw={uw:.4f}, w={w:.4f} (delta={d:+.4f})")
        print(f"    {'HYBRID':<20} uw={model_hybrid_uw:.4f}, w={model_hybrid_w:.4f} "
              f"(delta={model_hybrid_delta:+.4f})")
        print(f"    weighting helped the hybrid by {_hybrid_gain:+.4f} "
              f"{metric_name} and {_best_rep_name} by {_best_rep_gain:+.4f}; "
              f"gain on the hybrid minus gain on {_best_rep_name} "
              f"{_hyb_minus_base:+.4f} (gains: positive = the weights "
              f"helped)")
        print(f"    gain from cliff-aware weights minus gain from "
              f"neighbour-agreement weights, on the hybrid: "
              f"{_ca_minus_neighbour:+.4f} {metric_name}")

    # The RF-baseline ranking read straight from the search pool, as printed
    # at the top of the run. Each model's own pair is in
    # results['per_model_top2'].
    search_top3 = []
    if search is not None:
        try:
            search_top3 = [(n, float(s)) for n, s in extract_top3_base_reps(search)]
        except Exception:
            search_top3 = []
    results['top3_base_reps'] = search_top3

    # =========================================================================
    # 2026-09-12, owner's decision: n02's pool subset includes similarity
    # hybrids. Record which objective each hybrid was built under, and which
    # numbers in this file change units when that objective is 'similarity'.
    # =========================================================================
    # What is and is not in the dataset's own metric, read off this file:
    #
    #   Conditions 1-2 (per_rep) and conditions 3-5 (hybrid,
    #   hybrid_cliff_aware) are scored by n02 itself with metric_fn, on the
    #   held-out rows of its own folds. Every one of those numbers is in the
    #   dataset's own metric whichever family chose the hybrid's recipe, so
    #   'delta' (weighted minus unweighted) has the same feature space and the
    #   same metric top and bottom, and so does per_model_top2 (each model's
    #   top-2 single representations beside the hybrid).
    #
    #   'top3_base_reps' is the one field that does not: it is read straight
    #   out of the stored search. Under objective='predictive' those are RF
    #   scores in the dataset's own metric. Under objective='similarity'
    #   find_best_hybrid stores a model-independent number under every model
    #   name — the Spearman correlation between each held-out molecule's
    #   activity and what its five nearest neighbours predict — so the same
    #   field then holds a different quantity, ranked in a different space.
    #   'top3_base_reps_metric' says which, and when the winner is not the
    #   predictive family the predictive family's own top-3 is stored beside
    #   it so both numbers are on the page rather than one silently replacing
    #   the other.
    results['pool_selection'] = {
        'pool_subset': (meta.get('pool_subset') if meta else None),
        'top_level_objective': pool_family,
        'top_level_objective_choice': pool_family_choice,
        'top_level_search_metric': pool_metric_name,
        'by_model': family_by_model,
        'dataset_metric': metric_name,
        'scores_in_dataset_metric': [
            'per_rep', 'hybrid', 'hybrid_cliff_aware', 'per_model_top2'],
        'scores_read_from_the_search': ['top3_base_reps'],
    }
    results['top3_base_reps_metric'] = pool_metric_name
    results['top3_base_reps_objective'] = pool_family

    # The predictive family's top-3, in the dataset's own metric, whenever the
    # widened subset handed back a different family. A second pool lookup, not
    # a second search: 'predictive_only' is the subset every other experiment
    # already reads.
    results['top3_base_reps_predictive'] = None
    if pool_family is not None and pool_family != 'predictive':
        try:
            _p_search, _p_meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
                pool_subset='predictive_only',
            )
            if _p_search is not None:
                results['top3_base_reps_predictive'] = {
                    'metric_name': (_p_meta or {}).get('metric_name'),
                    'objective': (_p_meta or {}).get('objective'),
                    'top3': [(n, float(s))
                             for n, s in extract_top3_base_reps(_p_search)],
                }
        except Exception as _exc:
            results['top3_base_reps_predictive'] = {'error': str(_exc)}
        print(f"  NOTE: top3_base_reps came from family {pool_family!r}, "
              f"metric {pool_metric_name!r} — not the dataset's metric "
              f"({metric_name}). The predictive family's own top-3 is in "
              f"results['top3_base_reps_predictive'].")

    # =========================================================================
    # 2026-09-18, owner's instruction: the top 2 rows as an extra, and the two
    # hybrid deltas. Added beside the existing keys. Since 2026-09-27 the
    # per_rep loop fits each model on its own top 2 only (_top2_for).
    # =========================================================================
    # What was absent. results['per_model_top2'] holds only n02's own scores
    # on each model's top 2, and results['hybrid'] carries one
    # hybrid per downstream model whose only companion number is its own
    # weighted-minus-unweighted delta. Neither says how that hybrid stands
    # against the representations that fed it. A reader wanting that had to
    # re-open the stored search by hand.
    #
    # Each downstream model is read out of ITS OWN search — the same one that
    # gave that model its blueprint above. Two models in one run can land in
    # different search families, and their stored scores are then in different
    # units, so one search read under five model names would mix them.
    #
    # Every call below reads rows the search already stored. 0 new model fits.
    # require_full=False everywhere: a downstream model the search never scored
    # gives a short list with a shortfall_reason instead of raising and killing
    # the run.
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['hybrid_deltas_by_model'] = {}
    # WHICH DOWNSTREAM MODELS THIS TABLE COVERS. Two lists decide it and they
    # can disagree. model_factories is built from WEIGHTED_MODELS (:137), the
    # five models whose fit() takes a per-row weight. The other list is the
    # models a stored search scored, which searched_models reads off that
    # search. A model in the first and not in the second has no rows of its own
    # to rank. Until now such a model left this loop on a bare `continue` and
    # the result file said nothing about it, so a table of five models could be
    # a table of three and read as five. Both kinds of missing model are
    # written down now: the model whose pool lookup returned no search, and the
    # model whose own search scored other models and not it. Neither is read
    # out of another model's search -- resolve_comparison_models never
    # substitutes one model for another, and each model below keeps reading its
    # own search, which is what it did before.
    _requested = list(model_factories)
    _compare, _model_dropped = [], {}
    for _mname in _requested:
        _s_m = search_by_model.get(_mname)
        if _s_m is None:
            _model_dropped[_mname] = (
                'the pool returned no stored search for this downstream '
                'model, so there are no rows of its own to rank')
            continue
        _keep, _why = resolve_comparison_models(_s_m, [_mname])
        if not _keep:
            _model_dropped[_mname] = _why[0][1]
            continue
        _compare.append(_mname)
    results['top2_models_requested'] = _requested
    results['top2_models_compared'] = list(_compare)
    results['top2_models_dropped'] = _model_dropped
    results['top2_models_scored_by_each_search'] = {
        _m: searched_models(_s) for _m, _s in search_by_model.items()}
    for _mname, _why in _model_dropped.items():
        print(f"  top-2 rows absent for {_mname}: {_why}")
    for _mname in _compare:
        _msearch = search_by_model[_mname]
        try:
            results['top2_base_reps'][_mname] = shared_top_base_reps(
                _msearch, _mname, n=2, require_full=False,
                context=f'n02/{dataset_name}/{_mname}')
        except Exception as _exc:
            print(f"  WARNING: top-2 base representations failed for {_mname} "
                  f"({type(_exc).__name__}: {_exc}); "
                  f"results['top2_base_reps'] carries no entry for it.")
        # No budget= filter. n02 fetched its searches with budget=None, so the
        # stored rows span several budgets and filtering to one would report
        # the top 2 of a slice while calling it the top 2. Each selected entry
        # carries its own 'budget' field.
        _top2_hyb = None
        try:
            _top2_hyb = shared_top_hybrids(
                _msearch, _mname, n=2, require_full=False,
                context=f'n02/{dataset_name}/{_mname}')
            results['top2_hybrids'][_mname] = _top2_hyb
        except Exception as _exc:
            print(f"  WARNING: top-2 hybrids failed for {_mname} "
                  f"({type(_exc).__name__}: {_exc}); "
                  f"results['top2_hybrids'] carries no entry for it.")
        # The hybrid is passed in by hand. Called with hybrid=None,
        # hybrid_deltas reads _picked['selected'][0]['row'] (_shared.py:7432),
        # and the entry top_hybrids builds (_shared.py:7043) has no 'row' key,
        # so that path raises KeyError before it reaches either delta. The
        # entry itself carries 'score_mean' and 'blueprint', which is what
        # hybrid_deltas reads off the hybrid.
        try:
            if _top2_hyb and _top2_hyb['selected']:
                results['hybrid_deltas_by_model'][_mname] = shared_hybrid_deltas(
                    _msearch, _mname, hybrid=_top2_hyb['selected'][0],
                    context=f'n02/{dataset_name}/{_mname}')
        except Exception as _exc:
            print(f"  WARNING: hybrid deltas failed for {_mname} "
                  f"({type(_exc).__name__}: {_exc}); "
                  f"results['hybrid_deltas_by_model'] carries no entry for it.")
    # delta_vs_source_mean is the hybrid against the mean of the base
    # representations that actually fed it, read off Blueprint.
    # allocation_per_fold so a representation given 0 features is left out.
    # delta_vs_best_rep is the hybrid against the single best base
    # representation for that downstream model. Positive = the hybrid is
    # better, whichever way the metric runs. Both are in the units the search
    # stored, which for a model whose recipe came from the similarity family is
    # not the dataset's metric — results['pool_selection']['by_model'] says
    # which model is which.
    for _mname, _d in results['hybrid_deltas_by_model'].items():
        # delta_vs_best_rep is None when no base representation carries a score
        # for that downstream model; hybrid_deltas says so in its 'notes'.
        _vb = _d['delta_vs_best_rep']
        print(f"    {_mname}: hybrid vs mean of its "
              f"{_d['n_source_reps']} source representations "
              f"{_d['delta_vs_source_mean']:+.4f}, vs best single "
              f"representation ({_d['best_rep']}) "
              + (f"{_vb:+.4f}" if _vb is not None else "not defined"))

    # The similarity axis, 2026-09-25. Every key above is n02's own
    # conditions, unchanged; the swept numbers are the rows.
    results['similarity'] = similarity_record(
        sim_settings, sink, spaces_unavailable=sweep_ctx['unavailable'],
        checkpoint_file=ckpt.path, folds_read_back=ckpt.n_resumed)

    return make_structured_output(
        experiment_id='n02',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(model_factories.keys()),
            'k_neighbors': k_neighbors,
            # The scorers swept to define the weighted distance, in order.
            # The first one is what the flat weighted_mean / weighted_std /
            # delta / reliability_stats keys hold.
            'importance_methods': list(importance_methods),
            'primary_importance_method': primary_importance,
            'conditions': [
                'base_unweighted', 'base_weighted',
                'hybrid_unweighted', 'hybrid_weighted',
                'hybrid_cliff_aware',
            ],
            # 2026-09-12, owner's decision: n02's pool subset includes
            # similarity hybrids. Recorded in the config so a reader can tell
            # a run under the widened subset from an older one.
            'pool_subset': POOL_SUBSET,
            'hybrid_objective_by_model': {
                _m: _d.get('objective') for _m, _d in family_by_model.items()
            },
            # Which hybrid spaces this run actually scored the three hybrid
            # conditions in. One per family of stored search since 2026-09-19.
            'hybrid_spaces': sorted(results.get('hybrid_by_space') or {}),
        },
        results=results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser(
        'N02: Importance-Weighted Sample Weighting',
        has_delta=True, has_augment=True)
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='Number of neighbors for reliability scoring (default: 10)')
    parser.add_argument('--importance-methods', type=str,
                        default=','.join(DEFAULT_IMPORTANCE_METHODS),
                        help='comma-separated scorers defining the weighted '
                             'distance; the first is what the flat '
                             'weighted_mean/delta keys hold '
                             f'(default: {",".join(DEFAULT_IMPORTANCE_METHODS)})')
    args = parser.parse_args()
    config = get_config(args)

    importance_methods = [m.strip() for m in args.importance_methods.split(',')
                          if m.strip()]

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {[m for m in WEIGHTED_MODELS if m in config['models']]}")
    print(f"k-neighbors: {args.k_neighbors}")
    print(f"Importance scorers: {importance_methods}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_neighbors=args.k_neighbors,
                  importance_methods=importance_methods)
    save_experiment_results(results, args.output, 'n02_sample_weighting',
                            args.dataset)
