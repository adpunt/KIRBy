#!/usr/bin/env python3
"""N11: Noise Estimation × Remediation Grid.

Paper: Noise

SCOPE: MOLECULAR datasets only, and SYNTHETIC noise only (a dose-matched
condition from noiseInject on regression labels, defaulting to gaussian /
label-flipping on classification labels — see
`noise_helpers.inject_train_labels`). Results do NOT speak to real-world label
noise or to non-molecular domains.

Full grid from the confirmation report: estimator × smoothing/weighting × alpha ×
distance metric × representation. Extends with hybrid representations.

PRIMARY REPORTED CLAIM: the held-out RECOVERY metric (multi-seed) — how much of
the clean→noisy performance gap a given estimator × remediation config recovers.
This is scientifically sound.

CAVEAT (not the headline claim): the hybrid-vs-base space comparison is
ORACLE-BIASED. The hybrid blueprint (allocation / importance / internal
evaluator) is selected once from a precomputed pool fit on CLEAN labels, then
reused per fold under noisy labels. The *choice* of hybrid construction has thus
seen clean labels, so every hybrid space is a BEST-CASE hybrid — it is NOT a
fair hybrid-vs-base comparison and must not be reported as one.

2026-09-12, owner's decision: n11's pool subset includes similarity hybrids.
The pool lookup asks for subset 'predictive_and_similarity', so it may hand back
a recipe chosen under objective='similarity' — features chosen to make the
distance geometry predict each molecule's activity from its five nearest
neighbours — as well as one chosen to predict the label with a downstream model.
One estimation space is built per family that the pool holds, both through
create_hybrid on the same budget, the same folds and the same importance
machinery. Space names carry the family ('hybrid_predictive_<budget>',
'hybrid_similarity_<budget>') and results['pool_selection'] records what each
one was built from.

That REPLACES the four-block stack this file used to build by hand for its
'hybrid_sim' arm (count + physchem-minus-saboteurs + 3D shape + fine-2D, from
_similarity_hybrid_space). Two things change with it. The hand-built stack chose
its blocks against MCS structural overlap, which never reads activity; the
similarity objective scores a configuration by the correlation between each
held-out molecule's activity and what its five nearest neighbours predict, which
does. So the distance-based arm is now selected on clean activity labels, the
same way the prediction arm is, and carries the same oracle caveat. The stack
also only existed where a sim_<dataset>.npz cache covered every molecule; the
pool arm exists wherever a similarity-objective search has been precomputed
(tests/precompute_hybrids.py <dataset> --with-similarity).

~1,120 configs per sigma level. Each config: estimate noise scores, apply
remediation (smoothing or weighting), train RF on ECFP4 with remediated data,
compute recovery rate.

Usage:
    python n11_estimation_remediation_grid.py --dataset logd --mode intermediary
    python n11_estimation_remediation_grid.py --dataset caco2 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import inspect
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
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search, get_best_search_for_downstream, cap_dimensions,
    DOWNSTREAM_MODELS,
)
from kirby.hybrid import create_hybrid
from kirby.similarity.distance_config import DISTANCE_METRICS
from kirby.utils.noise_estimation import (
    get_noise_estimator, smooth_by_scores, weights_from_scores,
)

# 2026-09-12, owner's decision: n11's pool subset includes similarity hybrids,
# which replaces the four-block stack this file built by hand. The import of
# _similarity_hybrid_space (build_similarity_hybrid / similarity_hybrid_available)
# is gone with it. That module is still on disk and n13 still uses it.
#
# Which families of stored search the pool may draw a recipe from. Named here
# so the one decision is in one place; 'predictive_only' is what every other
# experiment reads and is what n11 read before today.
POOL_SUBSET = 'predictive_and_similarity'


# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
SIGMAS = None

# The noise condition. Gaussian is what this experiment has always injected; the
# other ten are reachable by passing noise_condition=..., one run each.
NOISE_CONDITION = 'gaussian'
ALPHAS = [0.1, 0.3, 0.5, 0.7]
WEIGHT_METHODS = ['inverse', 'exponential', 'soft']
SMOOTH_METHODS = ['proportional', 'threshold', 'soft_threshold', 'confidence']
# n11 is the one experiment that sweeps the distance INSIDE itself rather
# than across runs, so it gets the whole list. --distance-metric sets the
# metric for everything else n11 computes, not this sweep.
NEIGHBOR_METRICS = list(DISTANCE_METRICS)

# Estimators that use kNN internally — test with multiple distance metrics
NEIGHBOR_ESTIMATORS = {
    'neighbor_residual', 'local_variance_ratio',
    'distance_weighted', 'hybrid',
}

# is_clf-aware estimators: the kNN/neighbour-based ones whose constructor
# accepts is_clf (class-mismatch disagreement instead of a regression value
# residual). On a CLASSIFICATION dataset ONLY these produce a meaningful noise
# score (FIX 4f). Every other estimator in BASE_ESTIMATORS / TORCH_ESTIMATORS /
# OPTIONAL_ESTIMATORS is model-based and regression-only — it fits a regressor
# (or GP / NN / quantile / ngboost) on the integer class labels, which does not
# crash but yields a meaningless number. NOTE this is a STRICT SUBSET of
# NEIGHBOR_ESTIMATORS: 'hybrid' is in NEIGHBOR_ESTIMATORS (it uses kNN distance
# metrics) but is NOT is_clf-aware — HybridNoiseEstimator averages a
# CrossValResidualEstimator (a regressor on the labels) and exposes no is_clf
# branch — so it is correctly skipped on classification.
IS_CLF_AWARE_ESTIMATORS = {
    'neighbor_residual', 'local_variance_ratio',
    'distance_weighted', 'distance_weighted_residual',
}

# All estimators to try (torch-dependent ones are try-imported).
# leave_one_out removed (2026-05-26 audit decision): it required N model
# fits per use, dominating runtime on datasets above ~500 samples while
# the other estimators ran in seconds. The library function still exists
# in kirby.utils.noise_estimation if any other experiment wants it.
#
# activity_cliff_aware CUT 2026-09-12, measured, user decision. It scores a
# molecule by its distance from its ten nearest neighbours' activity values,
# multiplied by how much those neighbours agree with each other;
# neighbor_residual is the first half of that on its own. On herg_ki in ecfp4,
# 1,412 molecules, the two rank the molecules the same way at Spearman 0.989 —
# one method in two slots. The plain neighbour residual is the quantity the
# literature names, so the derived one goes. The class is still in
# ESTIMATOR_REGISTRY; n11, n13 and n06 stop running it.
BASE_ESTIMATORS = [
    'neighbor_residual', 'local_variance_ratio',
    'distance_weighted', 'hybrid',
    'cross_val', 'ensemble',
    'gaussian_process', 'conformal', 'bayesian_rf',
]
TORCH_ESTIMATORS = ['heteroscedastic_nn', 'deep_ensemble', 'quantile_regression']
OPTIONAL_ESTIMATORS = ['ngboost']

# GP max samples
GP_MAX = 1000

# Minimum clean->noisy gap, as a fraction of the clean score's magnitude,
# required before a recovery RATIO is reported. recovery = (denoised - noisy)
# / gap * 100 divides by the gap, so when the injected noise barely moved the
# model the gap is a tiny number and the ratio explodes into absurd values
# (e.g. 2827%). The old guard only caught a gap of almost exactly zero
# (abs(gap) > 1e-10), so any small-but-nonzero gap sailed through. Requiring
# the gap to be a meaningful fraction of the clean score turns those
# uninformative-noise cases into NaN (flagged) instead of garbage. Datasets
# with a real gap (e.g. buchwald_hartwig, gap ~20-60% of clean) are unaffected.
GAP_MIN_FRAC = 0.02


# The private `_inject_noise` this module used to carry has been deleted. It added
# Gaussian noise at `sigma_frac * (max(y) - min(y))` -- a fraction of the label RANGE,
# which is roughly 4-6x the label SD, so a level of 0.3 here delivered far more noise
# than a level of 0.3 anywhere else in the repo. Injection now goes through
# `noise_helpers.inject_train_labels`, which is the same definition the hybrid search,
# c10 and the Rust half of the study all use, and which records what was delivered.


def _get_correction_target(estimator):
    """Extract correction target from estimator for smoothing methods.

    Returns (target_array, source_name) or (None, None) if no target available.

    Look-up priority is fixed: neighbor-based correction targets win over
    model-based ones. This matters for ensemble estimators that expose both
    `neighbor_means_` AND e.g. `mu_` — without a stable priority the smoothing
    target would silently flip depending on which attribute was set first.
    The priority is intentional: neighbor-mean smoothing converges to local
    structure of the data, whereas a model's own prediction is the SAME
    quantity that defined the noise score, so smoothing toward it is
    self-fulfilling.
    """
    NEIGHBOR_ATTRS = ('neighbor_means_', 'weighted_means_')
    MODEL_ATTRS = ('mu_', 'cv_predictions_', 'prediction_means_',
                   'oob_predictions_', 'predictions_')

    for attr in NEIGHBOR_ATTRS:
        target = getattr(estimator, attr, None)
        if target is not None:
            return target, attr

    for attr in MODEL_ATTRS:
        target = getattr(estimator, attr, None)
        if target is not None:
            return target, attr

    return None, None


def _build_estimation_spaces(train_reps, y_train, augmentations,
                             hybrid_recipes,
                             dataset_name=None, groups=None,
                             build_errors=None):
    """Build estimation spaces: base rep + one hybrid per pool family.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        hybrid_recipes: list of the pool's winning recipes, one per search
            family, as built in run(). Each entry carries 'space' (the name the
            space gets), 'objective' ('predictive' or 'similarity') and
            'blueprint'.

    Returns (spaces_dict, base_rep_name).

    2026-09-12, owner's decision: n11's pool subset includes similarity
    hybrids. This argument was a single blueprint `bp` and built one hybrid
    space; it is now a list and builds one space per family. Every entry goes
    through the same create_hybrid call with the same budget, the same fold's
    training rows and the same importance method the search picked — only the
    blueprint and the objective differ between them.
    """
    spaces = {}

    # Control arm = one plain base representation. Shared helper: it will
    # NOT fall back onto a structure-tuned sim_* recipe, which is a study
    # treatment and must never become the thing treatments are measured
    # against. See _shared.pick_noise_base_rep.
    base_rep_name = pick_noise_base_rep(train_reps, dataset_name)

    X_base = train_reps[base_rep_name]
    scaler = StandardScaler()
    spaces[f'{base_rep_name}_raw'] = scaler.fit_transform(X_base)

    # One hybrid space per family the pool holds. The space name carries the
    # family and the blueprint's own budget (the name was hardcoded 'hybrid_100'
    # before 2026-06 even when bp.budget differed, and carried no family before
    # 2026-09-12).
    #
    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. `objective` is the family the recipe was searched under and is
    # passed to create_hybrid, so the allocators split the budget by the same
    # quantity the search ranked configurations by: the model's held-out score
    # under 'predictive', and under 'similarity' the correlation between each
    # held-out molecule's activity and what its five nearest neighbours predict.
    # Everything else in this call is identical between the families.
    for _recipe in (hybrid_recipes or []):
        bp = _recipe.get('blueprint')
        if bp is None:
            continue
        hybrid_key = _recipe['space']
        try:
            X_h, fi = create_hybrid(
                train_reps, y_train, groups=groups,
                allocation_method=bp.allocation_method,
                importance_method=bp.importance_method,
                internal_evaluator=bp.internal_evaluator,
                budget=bp.budget,
                augmentations=augmentations,
                augmentation_strategy=bp.augmentation_strategy,
                objective=_recipe.get('objective', 'predictive'),
            )
            scaler_h = StandardScaler()
            spaces[hybrid_key] = scaler_h.fit_transform(X_h)
        except Exception as e:
            print(f"      {hybrid_key} failed: {e}")
            # 2026-09-12: a family whose hybrid never builds would otherwise
            # leave no trace at all — the space is simply absent from
            # per_space. Count the failures per space and keep the first
            # message, so a reader can tell an arm that failed from an arm
            # that was never asked for.
            if build_errors is not None:
                _e = build_errors.setdefault(
                    hybrid_key, {'n_failures': 0, 'first_error': None,
                                 'objective': _recipe.get('objective')})
                _e['n_failures'] += 1
                if _e['first_error'] is None:
                    _e['first_error'] = f'{type(e).__name__}: {e}'

    # All other base rep spaces. Skip the canonical base rep itself — it's
    # already added above as f'{base_rep_name}_raw'. Adding it twice (under
    # both names, with different scaling/capping treatments) inflates the
    # "best space" win count and double-charges per-sigma config counts.
    for rep_name, X_rep in train_reps.items():
        if rep_name == base_rep_name:
            continue
        scaler = StandardScaler()
        spaces[rep_name] = cap_dimensions(scaler.fit_transform(X_rep))[0]

    return spaces, base_rep_name


def _resolve_estimators():
    """Determine which estimators are available (handle torch/ngboost imports)."""
    # Light local mode: neighbour-based estimators only. These are kNN-in-the-
    # space detectors — the on-point test for a SIMILARITY metric — and skip the
    # heavy model-based ones (GP is O(n^3), plus torch/ngboost), which the
    # out-of-fold refactor made 5x costlier. Set KIRBY_NOISE_LIGHT_ESTIMATORS=1.
    if os.environ.get('KIRBY_NOISE_LIGHT_ESTIMATORS') == '1':
        return [e for e in BASE_ESTIMATORS if e in NEIGHBOR_ESTIMATORS]

    available = list(BASE_ESTIMATORS)

    try:
        import torch  # noqa: F401
        available.extend(TORCH_ESTIMATORS)
    except ImportError:
        pass

    for opt in OPTIONAL_ESTIMATORS:
        try:
            get_noise_estimator(opt)
            available.append(opt)
        except Exception:
            pass

    return available


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes.

    Every reading is over the same condition and the same level, so they are the
    same quantity measured several times. The two counts are averaged too, and are
    written beside the fractions so no ratio appears without what it counts.
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


def _run_estimator_grid(est_name, metric_name, X_space, y_noisy,
                         X_eval_train, X_eval_val, y_val,
                         model_factories, metric_fn, n_samples,
                         is_clf=False, groups=None,
                         corrupted_mask=None, noise_condition=None):
    """Run full alpha × remediation grid for one estimator config.

    Returns: {remed_key: denoised_score} where remed_key =
    f'{model}|{method}_{alpha}'. The model is in the key because this experiment
    fits the whole roster as of 2026-09-12, and recovery is a fraction of that
    model's own clean-to-noisy gap. The bar is a separator that no model name and
    no remediation name contains, so the two halves split back apart exactly.

    Also returns `_detection`: how well this estimator's per-molecule suspicion
    score found the molecules whose label actually moved, before any repair.
    Owner's decision 2026-09-12. It is absent under the six conditions where every
    training molecule moves, because there it would be 1.0 by construction. It costs
    no extra model fit -- the suspicion scores and the list of corrupted molecules
    are both already in memory.

    For classification (is_clf=True) the smoothing branch is intentionally
    skipped: smooth_by_scores produces continuous floats which a
    RandomForestClassifier cannot consume — the old code's bare-except
    swallowed the resulting ValueError on every (estimator, alpha, method)
    triple, so half the grid silently produced no results for clf datasets.
    Smoothing results carry a `_smoothing_skipped_clf: True` marker so
    downstream readers don't confuse the empty cells with failed runs.
    """
    # Build estimator with appropriate kwargs
    kwargs = {}
    if est_name in NEIGHBOR_ESTIMATORS:
        kwargs['metric'] = metric_name
        kwargs['k'] = 10
    # Inspect the estimator constructor once so we only forward kwargs it
    # actually accepts (different estimator families expose different flags).
    try:
        est_params = inspect.signature(
            type(get_noise_estimator(est_name)).__init__).parameters
    except Exception:
        est_params = {}
    # On a classification task, pass is_clf to the kNN-based estimators that
    # accept it so the per-neighbour disagreement is a class-mismatch rate
    # rather than a regression value-residual on integer class labels. The
    # model-based estimators (cross_val, ensemble, gaussian_process, …) take
    # no is_clf flag, so we only set it when the constructor exposes it.
    # (Audit N08 "no clf branch", same defect applies to N11's kNN estimators.)
    if is_clf and 'is_clf' in est_params:
        kwargs['is_clf'] = True
    # Pass scaffold-group labels (this fold's training rows) to any estimator
    # that splits internally (e.g. conformal), so its calibration set stays
    # scaffold-disjoint from its training set instead of leaking near-copies.
    if groups is not None and 'groups' in est_params:
        kwargs['groups'] = groups
    # GP subsamples internally to GP_MAX. n_samples is passed in for future
    # heuristics; the dead `pass` block that used to guard it has been removed.

    try:
        estimator = get_noise_estimator(est_name, **kwargs)
        scores = estimator.estimate(X_space, y_noisy)
    except Exception as e:
        return {'_error': str(e)}

    target, target_source = _get_correction_target(estimator)
    results = {}
    # Grade the estimator on FINDING the corrupted molecules, not only on the score
    # after repair. Until 2026-09-12 every detector here was graded only after repair.
    if corrupted_mask is not None:
        _det = detection_scores(corrupted_mask, scores, noise_condition)
        if _det is not None:
            results['_detection'] = _det
    # Record per-(strategy, alpha) failure reasons so the grid's empty cells
    # can be distinguished from real failures during downstream analysis.
    failure_log = {}

    for alpha in ALPHAS:
        # Weighting methods (always available, sample_weight path works for
        # both regression and classification model factories)
        for wm in WEIGHT_METHODS:
            try:
                weights = weights_from_scores(scores, strength=alpha, method=wm)
            except Exception as exc:
                for mn in model_factories:
                    failure_log[f'{mn}|{wm}_{alpha}'] = f'{type(exc).__name__}: {exc}'
                continue
            for mn, mf in model_factories.items():
                key = f'{mn}|{wm}_{alpha}'
                try:
                    model = mf()
                    model.fit(X_eval_train, y_noisy, sample_weight=weights)
                    y_pred = model.predict(X_eval_val)
                    results[key] = float(metric_fn(y_val, y_pred))
                except Exception as exc:
                    failure_log[key] = f'{type(exc).__name__}: {exc}'

        # Smoothing methods (need a correction target AND a regression task)
        if target is not None and not is_clf:
            for sm in SMOOTH_METHODS:
                try:
                    y_smoothed = smooth_by_scores(
                        y_noisy, scores, target,
                        alpha=alpha, method=sm,
                    )
                except Exception as exc:
                    for mn in model_factories:
                        failure_log[f'{mn}|{sm}_{alpha}'] = f'{type(exc).__name__}: {exc}'
                    continue
                for mn, mf in model_factories.items():
                    key = f'{mn}|{sm}_{alpha}'
                    try:
                        model = mf()
                        model.fit(X_eval_train, y_smoothed)
                        y_pred = model.predict(X_eval_val)
                        results[key] = float(metric_fn(y_val, y_pred))
                    except Exception as exc:
                        failure_log[key] = f'{type(exc).__name__}: {exc}'

    if is_clf:
        results['_smoothing_skipped_clf'] = True
    if failure_log:
        results['_failures'] = failure_log

    return results


def run(data, config, output_dir, dataset_name, **kwargs):
    """N11: Noise estimation × remediation grid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N11: NOISE ESTIMATION × REMEDIATION GRID")
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
    n_samples = len(labels)

    # The whole roster, not random forest alone. Owner's decision 2026-09-12.
    # This experiment used to fit `rf` and nothing else, so every statement it made
    # about which detector and which repair works best was a statement about one
    # model. The estimators are unaffected: a suspicion score is computed once per
    # (space, estimator) and every model is then fitted on the same repaired labels,
    # so the added cost is the downstream fits only.
    downstream_models = list(kwargs.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_samples)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            "n11: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    noise_condition = kwargs.get('noise_condition', NOISE_CONDITION)
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

    available_estimators = _resolve_estimators()
    # FIX 4f: on a classification dataset, the model-based regression estimators
    # (cross_val, ensemble, gaussian_process, conformal, bayesian_rf, hybrid,
    # ngboost, and the torch heteroscedastic_nn / deep_ensemble /
    # quantile_regression) fit a regressor on integer class labels — no crash
    # but a meaningless noise score. Keep ONLY the kNN/is_clf-aware estimators
    # (IS_CLF_AWARE_ESTIMATORS), which _run_estimator_grid passes is_clf=True so
    # they score class disagreement. Previously only gaussian_process was
    # dropped, leaving the rest to emit junk. Record the dropped names + reason.
    skipped_estimators = []
    if is_clf:
        skipped_estimators = [e for e in available_estimators
                              if e not in IS_CLF_AWARE_ESTIMATORS]
        available_estimators = [e for e in available_estimators
                                if e in IS_CLF_AWARE_ESTIMATORS]

    # Control arm = one plain base representation. Shared helper: it will
    # NOT fall back onto a structure-tuned sim_* recipe, which is a study
    # treatment and must never become the thing treatments are measured
    # against. See _shared.pick_noise_base_rep.
    base_rep_name = pick_noise_base_rep(reps, dataset_name)

    print(f"  Dataset: {dataset_name} ({n_samples} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Estimators: {len(available_estimators)}")
    if skipped_estimators:
        print(f"  Skipped (regression-only on clf): {skipped_estimators}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Folds: {n_folds}")

    # Recipes selected from the CLEAN-label pool, per-fold construction under
    # noise (see _build_estimation_spaces). These are BEST-CASE hybrids, NOT a
    # fair baseline comparison: the structural recipe choice (allocation /
    # importance / internal evaluator) has already seen clean labels, so it is
    # ORACLE-BIASED and biases every hybrid space toward higher retention than
    # a fair baseline. Only the per-fold feature geometry is noisy. A
    # leak-free hybrid-vs-base design would re-select the recipe per fold
    # under noise; we do not. The RECOVERY metric (per estimator × remediation)
    # remains the sound primary claim — the hybrid-vs-base space comparison
    # must be read as "best-case hybrid", not a fair result. (Audit P1#5.)
    #
    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. One lookup asks the pool for subset 'predictive_and_similarity'
    # and every family it holds gets its own estimation space, built on the
    # same budget, the same folds and the same importance machinery as the
    # prediction arm it is compared against. This is NOT distance=True — n11
    # does not pass distance=True and that is unchanged.
    #
    # Local light-run mode, kept and re-pointed: it used to drop the predictive
    # hybrid and leave the hand-built four-block stack as the only hybrid arm.
    # That stack is gone, so the flag now asks the pool for 'similarity_only'
    # — the same intent (one hybrid arm, the distance-based one) through the
    # one mechanism. Both families are pool reads now, so neither runs a search
    # here; a family with no precomputed search simply has no arm.
    # Set KIRBY_SIMILARITY_HYBRID_ONLY=1.
    pool_subset = POOL_SUBSET
    if os.environ.get('KIRBY_SIMILARITY_HYBRID_ONLY') == '1':
        pool_subset = 'similarity_only'
        print("  similarity-hybrid-only mode (KIRBY_SIMILARITY_HYBRID_ONLY=1) "
              f"— pool subset {pool_subset!r}; the prediction-built hybrid is "
              "not asked for")

    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
        pool_subset=pool_subset,
    )
    if meta is None:
        print("  WARNING: No recipe found in the pool, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    # 2026-09-12: one hybrid space per family the pool holds. meta['objective']
    # names the family the pool returned as its winner; meta['by_objective']
    # carries EVERY eligible family's own best, which is what n11 builds from,
    # because the two families' stored scores are in different units and the
    # pool does not rank one against the other. Under 'predictive' the stored
    # score is the dataset's own metric; under 'similarity' it is the
    # correlation between each held-out molecule's activity and what its five
    # nearest neighbours predict.
    hybrid_recipes = []
    _by_obj = meta.get('by_objective') or {
        meta.get('objective', 'predictive'): {
            'tier': meta.get('tier'), 'aug_condition': meta.get('aug_condition'),
            'score_mean': meta.get('score_mean'),
            'metric_name': meta.get('metric_name'),
            'lower_is_better': meta.get('lower_is_better'),
            'best_config': meta.get('best_config'),
        }}
    for _obj, _entry in _by_obj.items():
        _bp = (_entry.get('best_config') or {}).get('blueprint')
        if _bp is None:
            print(f"  pool family {_obj!r}: no blueprint on its best entry — "
                  f"no space for it")
            continue
        _budget = int(getattr(_bp, 'budget', 0) or 0)
        hybrid_recipes.append({
            'space': f'hybrid_{_obj}_{_budget}' if _budget else f'hybrid_{_obj}',
            'objective': _obj,
            'blueprint': _bp,
            'tier': _entry.get('tier'),
            'aug_condition': _entry.get('aug_condition'),
            'budget': _budget,
            'allocation_method': getattr(_bp, 'allocation_method', None),
            'importance_method': getattr(_bp, 'importance_method', None),
            'internal_evaluator': getattr(_bp, 'internal_evaluator', None),
            'augmentation_strategy': getattr(_bp, 'augmentation_strategy', None),
            'search_score_mean': _entry.get('score_mean'),
            'search_metric_name': _entry.get('metric_name'),
            'search_lower_is_better': _entry.get('lower_is_better'),
        })

    if not hybrid_recipes:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    for _r in hybrid_recipes:
        print(f"  hybrid space {_r['space']}: family {_r['objective']!r}, "
              f"budget {_r['budget']} features, allocation "
              f"{_r['allocation_method']}, importance {_r['importance_method']}, "
              f"tier {_r['tier']}/{_r['aug_condition']}, search score "
              f"{_r['search_score_mean']} in {_r['search_metric_name']!r}")

    # 2026-09-12: per-space build failures, filled by _build_estimation_spaces.
    hybrid_build_errors = {}

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'estimators': available_estimators,
        # FIX 4f: regression-only estimators dropped on classification, with the
        # explicit reason, so empty cells aren't confused with failed runs.
        'skipped_estimators': {e: 'skipped_regression_estimator_on_clf'
                               for e in skipped_estimators},
        'alphas': ALPHAS,
        'sigmas': sigmas,
        'noise_condition': noise_condition,
        'noise_provenance': noise_provenance,
        'per_sigma': {},
    }

    # Multi-seed noise injection: average across NOISE_SEEDS to reduce
    # dependence on a single noise realization. The clean baseline is
    # noise-independent so it's only computed once per fold; the noisy
    # baseline + estimation spaces + grid are recomputed per seed.
    NOISE_SEEDS = kwargs.get('noise_seeds', [0, 1, 2])

    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. The per-fold cache the hand-built four-block stack needed
    # (sim_by_fold / sim_fold_info) is gone with it — both hybrid spaces are now
    # built inside the (fold, seed) loop by create_hybrid, on that seed's noisy
    # labels, exactly as the prediction arm already was.

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        clean_scores = {m: [] for m in downstream_models}
        # One list per model name, for both references. Recovery is a fraction of
        # THIS model's clean-to-noisy gap, so the two references cannot be shared.
        noisy_scores = {m: [] for m in downstream_models}
        # {space: {est_key: {remed_key: [scores]}}}
        grid_acc = {}
        # The same numbers again, kept fold by fold, so the winning config can be
        # re-picked from the OTHER folds and read on the held-out one. Without
        # this the reported recovery is the largest of ~4,800 numbers measured on
        # the folds that produced it.
        #   clean_by_fold[fold] = score           (noise-independent, one per fold)
        #   noisy_by_fold[fold] = [score, ...]    (one per noise seed)
        #   denoised_by_fold[space][est][remed][fold] = [score, ...]
        clean_by_fold = {m: {} for m in downstream_models}
        noisy_by_fold = {m: {} for m in downstream_models}
        denoised_by_fold = {m: {} for m in downstream_models}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}

            X_eval_train = reps[base_rep_name][train_idx]
            X_eval_val = reps[base_rep_name][val_idx]
            scaler_eval = StandardScaler()
            X_eval_train_s = scaler_eval.fit_transform(X_eval_train)
            X_eval_val_s = scaler_eval.transform(X_eval_val)

            # Clean baseline (noise-independent — compute once per fold per model)
            for _mn in downstream_models:
                _mdl = model_factories[_mn]()
                _mdl.fit(X_eval_train_s, y_train_clean)
                _clean_f = float(metric_fn(y_val, _mdl.predict(X_eval_val_s)))
                clean_scores[_mn].append(_clean_f)
                clean_by_fold[_mn][fold_idx] = _clean_f

            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

            for noise_seed in NOISE_SEEDS:
                # Per-(fold, seed) noise realization. Combining fold_idx and
                # noise_seed into the seed keeps every (fold, seed) draw
                # distinct without collisions.
                _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                       noise_seed=noise_seed,
                                       is_clf=is_clf)
                y_noisy, _cmask, _prov = inject_train_labels(
                    labels, _spec, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=dataset_name)
                noise_provenance.setdefault(_spec.spec_id, _prov)

                # 2026-09-12, owner's decision: n11's pool subset includes
                # similarity hybrids. `bp` (one blueprint) became
                # `hybrid_recipes` (one per family), and the hand-built
                # four-block stack that used to be pasted in below as
                # 'hybrid_sim' is gone — the distance-based arm is built here,
                # in this loop, on the same budget, the same fold and the same
                # importance machinery as the prediction arm.
                spaces, _brn = _build_estimation_spaces(
                    train_reps_fold, y_noisy, train_augs, hybrid_recipes,
                    dataset_name=dataset_name,
                    groups=(groups[train_idx] if groups is not None else None),
                    build_errors=hybrid_build_errors,
                )

                for _mn in downstream_models:
                    _mdl = model_factories[_mn]()
                    _mdl.fit(X_eval_train_s, y_noisy)
                    _noisy_fs = float(metric_fn(y_val, _mdl.predict(X_eval_val_s)))
                    noisy_scores[_mn].append(_noisy_fs)
                    noisy_by_fold[_mn].setdefault(fold_idx, []).append(_noisy_fs)

                for space_name, X_space in spaces.items():
                    if space_name not in grid_acc:
                        grid_acc[space_name] = {}

                    for est_name in available_estimators:
                        if est_name in NEIGHBOR_ESTIMATORS:
                            metrics_to_test = NEIGHBOR_METRICS
                        else:
                            metrics_to_test = [None]

                        for metric_var in metrics_to_test:
                            est_key = est_name
                            if metric_var:
                                est_key = f'{est_name}_{metric_var}'

                            grid_result = _run_estimator_grid(
                                est_name, metric_var,
                                X_space, y_noisy,
                                X_eval_train_s, X_eval_val_s, y_val,
                                model_factories, metric_fn, n_samples,
                                is_clf=is_clf,
                                groups=groups[train_idx],
                                corrupted_mask=_cmask,
                                noise_condition=noise_condition,
                            )

                            if est_key not in grid_acc[space_name]:
                                grid_acc[space_name][est_key] = {}

                            if '_error' in grid_result:
                                # Persist the per-(space, est) error so it's
                                # not silently dropped.
                                grid_acc[space_name][est_key].setdefault(
                                    '_errors', []).append(grid_result['_error'])
                                continue

                            for remed_key, score in grid_result.items():
                                if remed_key.startswith('_'):
                                    # Carry markers like `_failures` and
                                    # `_smoothing_skipped_clf` through to
                                    # the saved results so downstream
                                    # consumers can tell "skipped/failed"
                                    # apart from "no positive recoveries".
                                    if remed_key == '_failures':
                                        merged = grid_acc[space_name][est_key].setdefault(
                                            '_failures', {})
                                        merged.update(score)
                                    elif remed_key == '_detection':
                                        # One detection reading per (fold, draw).
                                        # Kept as a list and averaged below, like
                                        # every other per-fold number here, so a
                                        # later draw does not overwrite an earlier
                                        # one.
                                        (grid_acc[space_name][est_key]
                                         .setdefault('_detection_per_pass', [])
                                         .append(score))
                                    else:
                                        grid_acc[space_name][est_key][remed_key] = score
                                    continue
                                if remed_key not in grid_acc[space_name][est_key]:
                                    grid_acc[space_name][est_key][remed_key] = []
                                grid_acc[space_name][est_key][remed_key].append(score)
                                _mn_of_key = remed_key.split('|', 1)[0]
                                (denoised_by_fold[_mn_of_key]
                                 .setdefault(space_name, {})
                                 .setdefault(est_key, {})
                                 .setdefault(remed_key, {})
                                 .setdefault(fold_idx, []).append(score))

        # Aggregate across folds. One clean mean, one noisy mean and one gap PER
        # MODEL: recovery is a fraction of that model's own clean-to-noisy gap, and
        # sharing one gap across five models would divide every model's recovery by
        # random forest's damage.
        clean_by_model = {m: float(np.mean(v)) for m, v in clean_scores.items() if v}
        noisy_by_model = {m: float(np.mean(v)) for m, v in noisy_scores.items() if v}
        _models_scored = [m for m in downstream_models
                          if m in clean_by_model and m in noisy_by_model]
        if not _models_scored:
            raise RuntimeError(
                f"n11: no model produced both a clean and a noisy reference score at "
                f"level {sigma}. Asked for {downstream_models}.")
        gap_by_model = {}
        gap_floor_by_model = {}
        for _m in _models_scored:
            _c, _nz = clean_by_model[_m], noisy_by_model[_m]
            gap_by_model[_m] = (_nz - _c) if lower_is_better else (_c - _nz)
            gap_floor_by_model[_m] = max(1e-10, GAP_MIN_FRAC * abs(_c))
        # The first model in the roster still fills the three top-level fields, so a
        # reader written before the roster widened still finds a number where it
        # expects one -- and `reference_model` says which model that number is.
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
        # A recovery ratio is only meaningful when the noise actually moved the
        # model. Require the gap to exceed a small fraction of the clean score's
        # magnitude (not just a bare 1e-10) so a near-zero gap does not blow the
        # ratio up into absurd percentages.
        gap_floor = gap_floor_by_model[_ref_model]

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
            'reference_model': _ref_model,
            'models_scored': list(_models_scored),
            'clean_score_by_model': clean_by_model,
            'noisy_score_by_model': noisy_by_model,
            'gap_by_model': {k: float(v) for k, v in gap_by_model.items()},
            'per_space': {},
        }

        for space_name in sorted(grid_acc.keys()):
            space_agg = {}
            for est_key in sorted(grid_acc[space_name].keys()):
                est_agg = {}
                for remed_key, score_list in grid_acc[space_name][est_key].items():
                    # Skip marker fields (carried through for diagnostics);
                    # forward them as-is rather than treating them like
                    # score lists.
                    if remed_key == '_detection_per_pass':
                        est_agg['detection'] = _mean_detection(score_list)
                        continue
                    if remed_key.startswith('_'):
                        est_agg[remed_key] = score_list
                        continue
                    if score_list:
                        denoised = float(np.mean(score_list))
                        _mn = remed_key.split('|', 1)[0]
                        _c = clean_by_model.get(_mn)
                        _nz = noisy_by_model.get(_mn)
                        if (_c is not None and _nz is not None
                                and abs(gap_by_model[_mn]) > gap_floor_by_model[_mn]):
                            recovery = float((denoised - _nz) / (_c - _nz) * 100)
                        else:
                            recovery = float('nan')
                        est_agg[remed_key] = {
                            'denoised_score': denoised,
                            'lower_is_better': bool(lower_is_better),
                            'recovery': recovery,
                            # Named, not only encoded in the key, so no reader has
                            # to split a string to know which model this is.
                            'model': _mn,
                            'remediation': remed_key.split('|', 1)[-1],
                        }
                if est_agg:
                    space_agg[est_key] = est_agg
            sigma_result['per_space'][space_name] = space_agg

        # Recovery for a config picked without the fold it is read on. Pure
        # arithmetic over the per-fold scores just collected — no extra fitting.
        # One held-out recovery per model, for the same reason the gap is per model:
        # the config is re-picked from the other folds within that model's own arm.
        sigma_result['heldout_recovery_by_model'] = {
            _m: _heldout_recovery(clean_by_fold[_m], noisy_by_fold[_m],
                                  denoised_by_fold[_m], GAP_MIN_FRAC)
            for _m in _models_scored}
        sigma_result['heldout_recovery'] = (
            sigma_result['heldout_recovery_by_model'][_ref_model])

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_result)

        # Print compact summary
        n_total = 0
        n_positive = 0
        best_recovery = float('-inf')
        best_config = None
        for space_name, space_data in sigma_result['per_space'].items():
            for est_key, est_data in space_data.items():
                for rk, rv in est_data.items():
                    # Skip diagnostic markers (`_failures`, `_errors`,
                    # `_smoothing_skipped_clf`) so they don't inflate counts
                    # or trip the recovery-comparison logic.
                    if rk.startswith('_') or not isinstance(rv, dict):
                        continue
                    n_total += 1
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r) and r > 0:
                        n_positive += 1
                    if not np.isnan(r) and r > best_recovery:
                        best_recovery = r
                        best_config = f'{space_name}/{est_key}/{rk}'

        print(f"    Clean={clean_mean:.4f}, Noisy={noisy_mean:.4f}")
        print(f"    Configs: {n_total} total, {n_positive} positive "
              f"({100*n_positive/max(n_total,1):.1f}%)")
        if best_config:
            print(f"    Best over all configs on their own folds: "
                  f"{best_config} ({best_recovery:+.1f}%)")
            _ho = sigma_result.get('heldout_recovery')
            if _ho and not np.isnan(_ho['mean_recovery']):
                print(f"    Picked without the fold it is read on: "
                      f"{_ho['mean_recovery']:+.1f}% over "
                      f"{_ho['n_folds_scored']} fold(s), same config every "
                      f"fold={_ho['same_config_every_fold']}")
        elif n_total > 0:
            # All recoveries were NaN. Usually this means the (clean, noisy)
            # gap was near-zero for this sigma (no detectable noise effect)
            # so dividing by it produced NaN everywhere. Surface that
            # explicitly rather than silently skipping the "Best" line.
            print(f"    Best: None — all {n_total} configs returned NaN "
                  f"recovery (gap={gap:.4g}; sigma may be uninformative).")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # =========================================================================
    # 2026-09-12, owner's decision: n11's pool subset includes similarity
    # hybrids. Which objective each hybrid space was built under, and which
    # numbers in this file change units when that objective is 'similarity'.
    # This replaces results['hybrid_sim'], which recorded the coverage of the
    # hand-built four-block stack.
    # =========================================================================
    # Same denominator caution as n02, in n11's own terms:
    #
    #   clean_score, noisy_score, gap, denoised_score and recovery are all
    #   scored by n11 itself with metric_fn, on the held-out rows of its own
    #   folds, in the dataset's own metric. Every one of them is in that metric
    #   whichever family built the space.
    #
    #   Every space shares ONE denominator. `gap` is clean minus noisy on the
    #   base representation's evaluation model, computed once per sigma, and
    #   every space's recovery divides by that same number. So a recovery in a
    #   similarity-built space and a recovery in a prediction-built space are
    #   the same quantity over the same bottom, and the two can be read against
    #   each other. (The gap is refused below GAP_MIN_FRAC of the clean score's
    #   magnitude, which is unchanged.)
    #
    #   The search scores below are the one place the units move. They are read
    #   straight out of the stored search, not measured here. Under
    #   objective='predictive' that number is the dataset's own metric; under
    #   objective='similarity' find_best_hybrid stores a model-independent
    #   number under every model name — the correlation between each held-out
    #   molecule's activity and what its five nearest neighbours predict. Each
    #   family's own score sits beside its own metric name; nothing stored puts
    #   the two on one ruler and the pool does not rank them against each other.
    results['pool_selection'] = {
        'pool_subset': meta.get('pool_subset'),
        'pool_winner_objective': meta.get('objective'),
        'pool_winner_objective_choice': meta.get('objective_choice'),
        'dataset_metric': metric_name,
        'hybrid_spaces': {
            _r['space']: {
                'objective': _r['objective'],
                'tier': _r['tier'],
                'aug_condition': _r['aug_condition'],
                'budget_features': _r['budget'],
                'allocation_method': _r['allocation_method'],
                'importance_method': _r['importance_method'],
                'internal_evaluator': _r['internal_evaluator'],
                'augmentation_strategy': _r['augmentation_strategy'],
                'search_score_mean': _r['search_score_mean'],
                'search_metric_name': _r['search_metric_name'],
                'search_lower_is_better': _r['search_lower_is_better'],
            }
            for _r in hybrid_recipes
        },
        'hybrid_build_errors': convert_numpy(hybrid_build_errors),
        'scores_in_dataset_metric': [
            'per_sigma.*.clean_score', 'per_sigma.*.noisy_score',
            'per_sigma.*.gap', 'per_sigma.*.per_space.*.denoised_score',
            'per_sigma.*.per_space.*.recovery', 'heldout_recovery'],
        'scores_read_from_the_search': [
            'pool_selection.hybrid_spaces.*.search_score_mean'],
        'recipe_selection_rows': meta.get('selection_scope'),
        'recipe_selection_note': meta.get('selection_note'),
        'recipes_saw_clean_labels': (
            'Both families rank a configuration against activity labels, and '
            'the pool searched on the clean ones. The prediction family scores '
            "a model's held-out prediction; the similarity family scores the "
            "correlation between each held-out molecule's activity and what "
            'its five nearest neighbours predict. So every hybrid space here '
            'is a best-case hybrid, the caveat that already applied to the '
            'prediction arm.'),
    }

    # Summary
    results['summary'] = _compute_summary(results)

    return results


def _heldout_recovery(clean_by_fold, noisy_by_fold, denoised_by_fold,
                      gap_min_frac):
    """Recovery for a config chosen WITHOUT the fold it is read on.

    n11 reports the largest recovery among about 4,800 space x estimator x
    remediation x alpha configs, and every one of them was scored on the same
    folds that recovery is read from. The largest of 4,800 numbers measured on
    their own folds is partly luck, so it is not an estimate of how well that
    config recovers on rows it has not seen.

    This does the arithmetic that removes it, using per-fold scores that already
    exist — no extra fitting. For each fold in turn: rank every config by its
    recovery over the OTHER folds, take the top one, and read that config's
    recovery on this fold. Averaging those gives what picking a config this way
    is worth out of sample. The winner is allowed to differ from fold to fold,
    which is itself the answer to "is one config reliably best".

    Returns a dict, or None when fewer than two folds carry a usable gap.
    """
    folds = sorted(f for f in clean_by_fold if f in noisy_by_fold)
    if len(folds) < 2:
        return None

    clean_f = {f: float(clean_by_fold[f]) for f in folds}
    noisy_f = {f: float(np.mean(noisy_by_fold[f])) for f in folds}

    # {(space, est, remed): {fold: denoised}}
    flat = {}
    for space, ed in denoised_by_fold.items():
        for est, rd in ed.items():
            for remed, fd in rd.items():
                per_fold = {f: float(np.mean(v)) for f, v in fd.items()
                            if f in clean_f and len(v)}
                if len(per_fold) == len(folds):
                    flat[(space, est, remed)] = per_fold
    if not flat:
        return None

    def _recovery(cfg_folds, use):
        """Recovery over a set of folds, pooled the way the reported number is."""
        c = float(np.mean([clean_f[f] for f in use]))
        n = float(np.mean([noisy_f[f] for f in use]))
        d = float(np.mean([cfg_folds[f] for f in use]))
        gap = c - n
        if abs(gap) <= max(1e-10, gap_min_frac * abs(c)):
            return float('nan')
        return float((d - n) / gap * 100)

    per_fold_rows = []
    for f in folds:
        others = [g for g in folds if g != f]
        best_key, best_val = None, float('-inf')
        for key, cfg_folds in flat.items():
            r = _recovery(cfg_folds, others)
            if not np.isnan(r) and r > best_val:
                best_val, best_key = r, key
        if best_key is None:
            continue
        held = _recovery(flat[best_key], [f])
        per_fold_rows.append({
            'fold': int(f),
            'config': '/'.join(best_key),
            'recovery_on_other_folds': float(best_val),
            'recovery_on_this_fold': float(held),
        })

    scored = [r['recovery_on_this_fold'] for r in per_fold_rows
              if not np.isnan(r['recovery_on_this_fold'])]
    picks = {r['config'] for r in per_fold_rows}
    return {
        'mean_recovery': float(np.mean(scored)) if scored else float('nan'),
        'n_folds_scored': len(scored),
        'n_configs_ranked': len(flat),
        'per_fold': per_fold_rows,
        'same_config_every_fold': bool(len(picks) == 1),
        'configs_picked': sorted(picks),
        'what_this_is': (
            'For each fold the winning config is chosen from the other folds '
            'only, then read on this one. Compare it with best_config_search_max, '
            'which is the largest recovery over every config on the folds that '
            'produced it.'),
    }


def _compute_summary(results):
    """Compute summary: best configs, alpha effect, space comparison."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        gap = sigma_data.get('gap', 0)
        all_configs = []

        for space_name, space_data in sigma_data.get('per_space', {}).items():
            for est_key, est_data in space_data.items():
                for remed_key, rv in est_data.items():
                    # Skip diagnostic markers, same guard as the print loop.
                    # `_smoothing_skipped_clf` is a bool, so rv.get raised
                    # AttributeError here on every classification dataset —
                    # after the whole grid had run, which threw the run away.
                    if remed_key.startswith('_') or not isinstance(rv, dict):
                        continue
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        # Parse alpha from remed_key (e.g. 'inverse_0.1')
                        parts = remed_key.rsplit('_', 1)
                        alpha = float(parts[1]) if len(parts) == 2 else 0.0
                        method = parts[0] if len(parts) == 2 else remed_key
                        all_configs.append({
                            'space': space_name,
                            'estimator': est_key,
                            'method': method,
                            'alpha': alpha,
                            'recovery': r,
                        })

        if not all_configs:
            summary[f'sigma_{sigma_key}'] = {}
            continue

        # Sort by recovery (descending)
        all_configs.sort(key=lambda x: x['recovery'], reverse=True)

        # Alpha effect
        alpha_recoveries = {}
        for c in all_configs:
            a = str(c['alpha'])
            if a not in alpha_recoveries:
                alpha_recoveries[a] = []
            alpha_recoveries[a].append(c['recovery'])
        alpha_effect = {
            a: {
                'mean_recovery': float(np.mean(rs)),
                'pct_positive': float(sum(1 for r in rs if r > 0) / len(rs) * 100),
            }
            for a, rs in alpha_recoveries.items()
        }

        # Per-space recovery (base rep vs hybrid). This is the hybrid-vs-base
        # comparison. It is ORACLE-BIASED — the hybrid recipe was selected on
        # CLEAN labels (see run(): get_best_search_for_downstream on clean
        # labels, reused per fold under noise). Hybrid spaces are therefore
        # BEST-CASE and NOT a fair baseline comparison. The numbers are kept for
        # transparency but carry an explicit caveat so they are not read as a
        # fair result. Do NOT headline hybrid vs base from this field.
        #
        # 2026-09-12, owner's decision: n11's pool subset includes similarity
        # hybrids, so there can be two hybrid spaces here, one per family, named
        # hybrid_predictive_<budget> and hybrid_similarity_<budget>. Both
        # recipes were ranked against clean activity labels, so the caveat above
        # covers both. Every space's recovery divides by the same clean-minus-
        # noisy gap, so the rows are comparable with each other.
        space_recoveries = {}
        for c in all_configs:
            s = c['space']
            if s not in space_recoveries:
                space_recoveries[s] = []
            space_recoveries[s].append(c['recovery'])
        space_effect_oracle_biased = {
            s: {
                'mean_recovery': float(np.mean(rs)),
                'pct_positive': float(sum(1 for r in rs if r > 0) / len(rs) * 100),
            }
            for s, rs in space_recoveries.items()
        }

        n_positive = sum(1 for c in all_configs if c['recovery'] > 0)

        # best_config / top_10 are the top of ~4,800 configs, every one scored
        # on the same folds this recovery is read from. They say which config
        # came first in the search, NOT how much noise that config recovers on
        # rows it has not seen. heldout_recovery below is the second number:
        # for each fold the winner is picked from the other folds and read on
        # this one. Renamed so the two are not confused at the point of use;
        # the old names are kept alongside for readers written before this.
        s = {
            'n_total': len(all_configs),
            'n_positive': n_positive,
            'pct_positive': float(n_positive / len(all_configs) * 100),
            'best_config_search_max': all_configs[0],
            'top_10_search_max': all_configs[:10],
            'search_max_caveat': (
                'Largest recovery over every config, scored on the folds that '
                'chose it. A search result, not a measured recovery — read '
                'heldout_recovery for the out-of-fold number.'),
            'heldout_recovery': sigma_data.get('heldout_recovery'),
            'best_config': all_configs[0],
            'top_10': all_configs[:10],
            'alpha_effect': alpha_effect,
            # Renamed from 'space_effect' to make the bias explicit at the point
            # of consumption: hybrid blueprint selected on clean labels => this
            # is a best-case hybrid, NOT a fair hybrid-vs-base comparison.
            'space_effect_oracle_biased': space_effect_oracle_biased,
            'space_effect_caveat': (
                'ORACLE-BIASED: every hybrid recipe here was selected on clean '
                'labels — the prediction family against a model\'s held-out '
                'score, the similarity family against the correlation between '
                'each held-out molecule\'s activity and what its five nearest '
                'neighbours predict. NOT a fair hybrid-vs-base comparison — '
                'best-case hybrid. The recovery metric per estimator x '
                'remediation is the sound primary claim. Which family built '
                'each space is in results["pool_selection"]["hybrid_spaces"].'
            ),
        }

        # Robust configs (positive at BOTH sigma levels if we have multiple)
        # This is computed after all sigmas in the final summary merge
        summary[f'sigma_{sigma_key}'] = convert_numpy(s)

    # Cross-sigma robust configs
    sigma_keys = [k for k in results['per_sigma'] if k != '0.0']
    if len(sigma_keys) >= 2:
        # Collect all config keys that are positive at every sigma
        config_recovery = {}  # config_key -> {sigma: recovery}
        for sigma_key in sigma_keys:
            sigma_data = results['per_sigma'][sigma_key]
            for space_name, space_data in sigma_data.get('per_space', {}).items():
                for est_key, est_data in space_data.items():
                    for remed_key, rv in est_data.items():
                        # Same guard as the per-sigma loop above.
                        if remed_key.startswith('_') or not isinstance(rv, dict):
                            continue
                        r = rv.get('recovery', float('nan'))
                        if not np.isnan(r):
                            ck = f'{space_name}/{est_key}/{remed_key}'
                            if ck not in config_recovery:
                                config_recovery[ck] = {}
                            config_recovery[ck][sigma_key] = r

        robust = []
        for ck, sr in config_recovery.items():
            if len(sr) >= 2 and all(r > 0 for r in sr.values()):
                avg = float(np.mean(list(sr.values())))
                robust.append({'config': ck, 'avg_recovery': avg,
                               'per_sigma': sr})

        robust.sort(key=lambda x: x['avg_recovery'], reverse=True)
        # Same caveat as best_config: these 20 are the top of the same ~4,800
        # scored on the folds that ranked them. What they add is that each one
        # is positive at every sigma, which the single maximum does not check.
        summary['robust_configs_search_max'] = convert_numpy(robust[:20])
        summary['robust_configs'] = convert_numpy(robust[:20])
        summary['robust_configs_caveat'] = (
            'Ranked on the folds that chose them. Positive at every sigma is '
            'the claim; the size of the recovery is a search maximum.')
        summary['n_robust'] = len(robust)

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser('N11: Estimation × Remediation Grid')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n11_estimation_remediation_grid',
                            args.dataset)
