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
    inject_train_labels, noise_spec_for,
)
from experiments._shared import (
    pick_noise_base_rep,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search, get_best_search_for_downstream, cap_dimensions,
)
from kirby.hybrid import create_hybrid
from kirby.utils.noise_estimation import (
    get_noise_estimator, smooth_by_scores, weights_from_scores,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _similarity_hybrid_space import build_similarity_hybrid


# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
SIGMAS = None

# The noise condition. Gaussian is what this experiment has always injected; the
# other ten are reachable by passing noise_condition=..., one run each.
NOISE_CONDITION = 'gaussian'
ALPHAS = [0.1, 0.3, 0.5, 0.7]
WEIGHT_METHODS = ['inverse', 'exponential', 'soft']
SMOOTH_METHODS = ['proportional', 'threshold', 'soft_threshold', 'confidence']
NEIGHBOR_METRICS = ['euclidean', 'cosine']

# Estimators that use kNN internally — test with multiple distance metrics
NEIGHBOR_ESTIMATORS = {
    'neighbor_residual', 'local_variance_ratio',
    'activity_cliff_aware', 'distance_weighted', 'hybrid',
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
    'activity_cliff_aware', 'distance_weighted',
    'distance_weighted_residual',
}

# All estimators to try (torch-dependent ones are try-imported).
# leave_one_out removed (2026-05-26 audit decision): it required N model
# fits per use, dominating runtime on datasets above ~500 samples while
# the other estimators ran in seconds. The library function still exists
# in kirby.utils.noise_estimation if any other experiment wants it.
BASE_ESTIMATORS = [
    'neighbor_residual', 'local_variance_ratio', 'activity_cliff_aware',
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


def _build_estimation_spaces(train_reps, y_train, augmentations, bp,
                             dataset_name=None):
    """Build estimation spaces: base rep + hybrid.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.

    Returns (spaces_dict, base_rep_name).
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

    # Hybrid via blueprint. Label the space with the actual blueprint budget
    # (was previously hardcoded as 'hybrid_100' even when bp.budget differed).
    # bp is None when the predictive-hybrid arm is skipped (local light run);
    # then base + hybrid_sim + other base reps are the only spaces.
    if bp is not None:
        hybrid_key = f'hybrid_{int(bp.budget)}' if hasattr(bp, 'budget') else 'hybrid'
        try:
            X_h, fi = create_hybrid(
                train_reps, y_train,
                allocation_method=bp.allocation_method,
                importance_method=bp.importance_method,
                internal_evaluator=bp.internal_evaluator,
                budget=bp.budget,
                augmentations=augmentations,
                augmentation_strategy=bp.augmentation_strategy,
            )
            scaler_h = StandardScaler()
            spaces[hybrid_key] = scaler_h.fit_transform(X_h)
        except Exception as e:
            print(f"      {hybrid_key} failed: {e}")

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


def _run_estimator_grid(est_name, metric_name, X_space, y_noisy,
                         X_eval_train, X_eval_val, y_val,
                         model_factory, metric_fn, n_samples,
                         is_clf=False, groups=None):
    """Run full alpha × remediation grid for one estimator config.

    Returns: {remed_key: denoised_score} where remed_key = f'{method}_{alpha}'.

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
    # Record per-(strategy, alpha) failure reasons so the grid's empty cells
    # can be distinguished from real failures during downstream analysis.
    failure_log = {}

    for alpha in ALPHAS:
        # Weighting methods (always available, sample_weight path works for
        # both regression and classification model factories)
        for wm in WEIGHT_METHODS:
            key = f'{wm}_{alpha}'
            try:
                weights = weights_from_scores(scores, strength=alpha, method=wm)
                model = model_factory()
                model.fit(X_eval_train, y_noisy, sample_weight=weights)
                y_pred = model.predict(X_eval_val)
                results[key] = float(metric_fn(y_val, y_pred))
            except Exception as exc:
                failure_log[key] = f'{type(exc).__name__}: {exc}'

        # Smoothing methods (need a correction target AND a regression task)
        if target is not None and not is_clf:
            for sm in SMOOTH_METHODS:
                key = f'{sm}_{alpha}'
                try:
                    y_smoothed = smooth_by_scores(
                        y_noisy, scores, target,
                        alpha=alpha, method=sm,
                    )
                    model = model_factory()
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

    model_factory = get_model_factories(
        ['rf'], task_type=task_type,
        n_samples=n_samples)['rf']

    noise_condition = kwargs.get('noise_condition', NOISE_CONDITION)
    sigmas = kwargs.get('sigmas') or kwargs.get('levels') or SIGMAS
    if sigmas is None:
        sigmas = list(dataset_noise_levels(dataset_name, strategy=noise_condition))
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

    # Blueprint selected from the CLEAN-label pool, per-fold construction under
    # noise (see _build_estimation_spaces). This is a BEST-CASE hybrid, NOT a
    # fair baseline comparison: the structural blueprint choice (allocation /
    # importance / internal evaluator) has already seen clean labels, so it is
    # ORACLE-BIASED and biases every hybrid space toward higher retention than
    # a fair baseline. Only the per-fold feature geometry is noisy. A
    # leak-free hybrid-vs-base design would re-select the blueprint per fold
    # under noise; we do not. The RECOVERY metric (per estimator × remediation)
    # remains the sound primary claim — the hybrid-vs-base space comparison
    # must be read as "best-case hybrid", not a fair result. (Audit P1#5.)
    # Local light-run mode: make the similarity hybrid the ONLY hybrid arm and
    # drop the predictive one (which needs a full find_best_hybrid search —
    # uncached locally = hours). Base rep + other base reps still run, so this
    # gives base vs hybrid_sim. Set KIRBY_SIMILARITY_HYBRID_ONLY=1.
    if os.environ.get('KIRBY_SIMILARITY_HYBRID_ONLY') == '1':
        print("  similarity-hybrid-only mode (KIRBY_SIMILARITY_HYBRID_ONLY=1) "
              "— predictive hybrid dropped; base vs hybrid_sim")
        bp = None
    else:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            dataset_name=dataset_name, task_type=task_type,
            data=data,
        )
        bp = meta['best_config']['blueprint'] if meta else None
        if bp is None:
            print("  WARNING: No blueprint found, skipping")
            return {'error': 'no_blueprint', 'metric': metric_name}

    # Extra arm: the similarity-objective hybrid (label-blind winning recipe),
    # reused from the similarity study's precomputed MCS/3D caches and aligned by
    # canonical SMILES. Added as a space only when the cache covers EVERY row
    # (else skipped with a logged reason — never a partial arm). Unlike the
    # predictive `hybrid_*` space this selection never saw activity labels, so it
    # is NOT oracle-biased.
    X_sim_full, sim_info = build_similarity_hybrid(dataset_name, data.get('smiles'))
    if X_sim_full is not None:
        print(f"  hybrid_sim arm ON: dim={sim_info['dim']} "
              f"(cache n={sim_info['n_cache']}, full coverage)")
    else:
        print(f"  hybrid_sim arm OFF: {sim_info.get('skip')}")

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

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        clean_scores = []
        noisy_scores = []
        # {space: {est_key: {remed_key: [scores]}}}
        grid_acc = {}

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

            # Clean baseline (noise-independent — compute once per fold)
            rf_clean = model_factory()
            rf_clean.fit(X_eval_train_s, y_train_clean)
            clean_scores.append(float(metric_fn(y_val, rf_clean.predict(X_eval_val_s))))

            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

            for noise_seed in NOISE_SEEDS:
                # Per-(fold, seed) noise realization. Combining fold_idx and
                # noise_seed into the seed keeps every (fold, seed) draw
                # distinct without collisions.
                _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                       fold_idx=fold_idx, noise_seed=noise_seed,
                                       is_clf=is_clf)
                y_noisy, _cmask, _prov = inject_train_labels(
                    y_train_clean, _spec, is_clf=is_clf,
                    groups=noise_groups[train_idx] if (_spec.needs_groups and
                                                       noise_groups is not None) else None)
                noise_provenance.setdefault(_spec.spec_id, _prov)

                spaces, _brn = _build_estimation_spaces(
                    train_reps_fold, y_noisy, train_augs, bp,
                    dataset_name=dataset_name,
                )

                # similarity-objective hybrid arm (fold train slice)
                if X_sim_full is not None:
                    spaces['hybrid_sim'] = StandardScaler().fit_transform(
                        X_sim_full[train_idx])

                rf_noisy = model_factory()
                rf_noisy.fit(X_eval_train_s, y_noisy)
                noisy_scores.append(float(metric_fn(y_val, rf_noisy.predict(X_eval_val_s))))

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
                                est_name, metric_var or 'euclidean',
                                X_space, y_noisy,
                                X_eval_train_s, X_eval_val_s, y_val,
                                model_factory, metric_fn, n_samples,
                                is_clf=is_clf,
                                groups=groups[train_idx],
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
                                    else:
                                        grid_acc[space_name][est_key][remed_key] = score
                                    continue
                                if remed_key not in grid_acc[space_name][est_key]:
                                    grid_acc[space_name][est_key][remed_key] = []
                                grid_acc[space_name][est_key][remed_key].append(score)

        # Aggregate across folds
        clean_mean = float(np.mean(clean_scores))
        noisy_mean = float(np.mean(noisy_scores))
        gap = clean_mean - noisy_mean
        # A recovery ratio is only meaningful when the noise actually moved the
        # model. Require the gap to exceed a small fraction of the clean score's
        # magnitude (not just a bare 1e-10) so a near-zero gap does not blow the
        # ratio up into absurd percentages.
        gap_floor = max(1e-10, GAP_MIN_FRAC * abs(clean_mean))

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
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
                    if remed_key.startswith('_'):
                        est_agg[remed_key] = score_list
                        continue
                    if score_list:
                        denoised = float(np.mean(score_list))
                        if abs(gap) > gap_floor:
                            recovery = float((denoised - noisy_mean) / gap * 100)
                        else:
                            recovery = float('nan')
                        est_agg[remed_key] = {
                            'denoised_score': denoised,
                            'lower_is_better': bool(lower_is_better),
                            'recovery': recovery,
                        }
                if est_agg:
                    space_agg[est_key] = est_agg
            sigma_result['per_space'][space_name] = space_agg

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
            print(f"    Best: {best_config} ({best_recovery:+.1f}%)")
        elif n_total > 0:
            # All recoveries were NaN. Usually this means the (clean, noisy)
            # gap was near-zero for this sigma (no detectable noise effect)
            # so dividing by it produced NaN everywhere. Surface that
            # explicitly rather than silently skipping the "Best" line.
            print(f"    Best: None — all {n_total} configs returned NaN "
                  f"recovery (gap={gap:.4g}; sigma may be uninformative).")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # Summary
    results['summary'] = _compute_summary(results)

    return results


def _compute_summary(results):
    """Compute summary: best configs, alpha effect, space comparison."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        gap = sigma_data.get('gap', 0)
        all_configs = []

        for space_name, space_data in sigma_data.get('per_space', {}).items():
            for est_key, est_data in space_data.items():
                for remed_key, rv in est_data.items():
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
        # comparison. It is ORACLE-BIASED — the hybrid blueprint was selected on
        # CLEAN labels (see run(): get_best_search_for_downstream on clean
        # labels, reused per fold under noise). Hybrid spaces are therefore
        # BEST-CASE and NOT a fair baseline comparison. The numbers are kept for
        # transparency but carry an explicit caveat so they are not read as a
        # fair result. Do NOT headline hybrid vs base from this field.
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

        s = {
            'n_total': len(all_configs),
            'n_positive': n_positive,
            'pct_positive': float(n_positive / len(all_configs) * 100),
            'best_config': all_configs[0],
            'top_10': all_configs[:10],
            'alpha_effect': alpha_effect,
            # Renamed from 'space_effect' to make the bias explicit at the point
            # of consumption: hybrid blueprint selected on clean labels => this
            # is a best-case hybrid, NOT a fair hybrid-vs-base comparison.
            'space_effect_oracle_biased': space_effect_oracle_biased,
            'space_effect_caveat': (
                'ORACLE-BIASED: hybrid blueprint selected on clean labels; '
                'NOT a fair hybrid-vs-base comparison — best-case hybrid. '
                'The recovery metric per estimator x remediation is the sound '
                'primary claim.'
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
        summary['robust_configs'] = convert_numpy(robust[:20])
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
