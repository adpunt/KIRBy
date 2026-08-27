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

5 conditions × 3 estimators × 3 k values × 2 importance methods × 2 sigmas.

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
    inject_train_labels, noise_spec_for,
)
from experiments._shared import (
    pick_noise_base_rep,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search, get_best_search_for_downstream,
)
from kirby.hybrid import create_hybrid, compute_feature_importance
from kirby.utils.noise_estimation import (
    NeighborResidualEstimator,
    LocalVarianceRatioEstimator,
    ActivityCliffAwareEstimator,
    weights_from_scores,
)


# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy.
SIGMAS = None

# The noise condition. Gaussian is what this experiment has always injected; the
# other ten are reachable by passing noise_condition=..., one run each.
NOISE_CONDITION = 'gaussian'
K_VALUES = [5, 10, 20]
IMPORTANCE_METHODS = ['lightgbm_gain', 'treeshap']

ESTIMATOR_FACTORIES = {
    'neighbor_residual': NeighborResidualEstimator,
    'local_variance_ratio': LocalVarianceRatioEstimator,
    'activity_cliff_aware': ActivityCliffAwareEstimator,
}

CONDITION_NAMES = [
    'base_unweighted',
    'base_imp_weighted',
    'hybrid_unweighted',
    'hybrid_imp_base',
    'hybrid_imp_own',
]


# The private `_inject_noise` this module used to carry has been deleted. It added
# Gaussian noise at `sigma_frac * (max(y) - min(y))` -- a fraction of the label RANGE,
# which is roughly 4-6x the label SD, so a level of 0.3 here delivered far more noise
# than a level of 0.3 anywhere else in the repo. Injection now goes through
# `noise_helpers.inject_train_labels`, which is the same definition the hybrid search,
# c10 and the Rust half of the study all use, and which records what was delivered.


def _weight_features(X_standardized, importance):
    """Apply sqrt(importance) weighting to features that are ALREADY z-scored.

    CRITICAL: caller must pass a standardized X. The previous version of this
    helper applied the weight first and the caller standardized after, which
    is a mathematical no-op (per-column StandardScaler exactly cancels a
    per-column scalar multiply) — the entire importance-weighting ablation
    silently produced identical Euclidean geometry across weighted and
    unweighted conditions.

    Returns X * sqrt(w_normalized) where w_normalized has positive entries
    summing to len(importance) (mean ≈ 1) — chosen so the weighting expands
    or contracts the per-feature dimension relative to the unit-variance
    z-scored input rather than crushing every column toward zero.
    """
    w = np.sqrt(np.maximum(importance, 0))
    s = w.sum()
    if s < 1e-10:
        # All-zero importance → no signal to weight by; return X_standardized
        # unchanged so downstream Euclidean distances are well-defined.
        return X_standardized
    # Normalize to mean ≈ 1 (not sum ≈ 1) so unit-variance columns are
    # rescaled around their original magnitude rather than crushed to ~1/d.
    w = w * (len(w) / s)
    return X_standardized * w


def _build_fold_shared(train_reps, y_train, augmentations, bp, base_rep_name):
    """Build the pieces that don't depend on imp_method, ONCE per fold.

    Returns dict with:
      'X_base_std': standardized base rep
      'X_hyb_std' or None: standardized hybrid (or None if hybrid build failed)
      'imp_hybrid_via_blueprint' or None: importance of hybrid features under
            the blueprint's own importance method (used for hybrid_imp_base)
    """
    shared = {}
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

    X_hybrid = None
    try:
        X_hybrid, fi = create_hybrid(
            train_reps_for_bp, y_train,
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            budget=bp.budget,
            augmentations=augmentations,
            augmentation_strategy=bp.augmentation_strategy,
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
                       imp_method, base_rep_name, shared=None):
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

    Returns {condition_name: X_standardized}.
    """
    conditions = {}
    X_base = train_reps[base_rep_name]

    if shared is None:
        shared = _build_fold_shared(train_reps, y_train, augmentations, bp,
                                    base_rep_name)

    X_base_std = shared['X_base_std']
    X_hyb_std = shared['X_hyb_std']
    X_hybrid_raw = shared['X_hybrid_raw']
    imp_hybrid_via_blueprint = shared['imp_hybrid_via_blueprint']

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

    if X_hyb_std is not None:
        # Condition 3: hybrid unweighted
        conditions['hybrid_unweighted'] = X_hyb_std

        # Condition 4: hybrid weighted by blueprint's importance method.
        # imp_method-INDEPENDENT — pulled from `shared` so we don't recompute
        # it on every inner-loop iteration.
        if imp_base is not None and imp_hybrid_via_blueprint is not None:
            conditions['hybrid_imp_base'] = _weight_features(
                X_hyb_std, imp_hybrid_via_blueprint)

        # Condition 5: hybrid weighted by the sweep's imp_method (the only
        # imp_method-dependent hybrid weighting).
        try:
            imp_hybrid_own = compute_feature_importance(
                {'hybrid': X_hybrid_raw}, y_train, method=imp_method
            )['hybrid']
            conditions['hybrid_imp_own'] = _weight_features(
                X_hyb_std, imp_hybrid_own)
        except Exception as e:
            print(f"      hybrid_imp_own failed: {e}")

    return conditions


def _run_condition(X_space, y_noisy, X_eval_train, X_eval_val, y_val,
                    model_factory, metric_fn, is_clf=False):
    """Run 3 estimators × 3 k values on one condition.

    Fixed remediation: exponential weighting, alpha=0.1.
    Passes is_clf to each estimator so classification uses class-disagreement
    instead of the regression value-residual that's meaningless for class
    labels.

    Returns {f'{est_name}_k{k}': denoised_score}.
    """
    results = {}
    # Record per-(estimator, k) failures so a degenerate rerun isn't blind:
    # previously every error was swallowed into `results[key] = None`, the
    # aggregation dropped all-None cells, and per_importance_method came back
    # as empty {} with no trace of WHY. Now the error string is preserved.
    errors = {}

    for est_name, est_cls in ESTIMATOR_FACTORIES.items():
        for k in K_VALUES:
            key = f'{est_name}_k{k}'
            try:
                est = est_cls(k=k, metric='euclidean', is_clf=is_clf)
                scores = est.estimate(X_space, y_noisy)
                weights = weights_from_scores(scores, strength=0.1,
                                              method='exponential')
                model = model_factory()
                model.fit(X_eval_train, y_noisy, sample_weight=weights)
                y_pred = model.predict(X_eval_val)
                results[key] = float(metric_fn(y_val, y_pred))
            except Exception as e:
                results[key] = None
                errors[key] = f'{type(e).__name__}: {e}'

    if errors:
        results['_errors'] = errors

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

    model_factory = get_model_factories(
        ['rf'], task_type=task_type,
        n_samples=len(labels))['rf']

    noise_condition = kwargs.get('noise_condition', NOISE_CONDITION)
    sigmas = kwargs.get('sigmas') or kwargs.get('levels') or SIGMAS
    if sigmas is None:
        sigmas = list(dataset_noise_levels(dataset_name, strategy=noise_condition))
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

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'base_rep': base_rep_name,
        'k_values': K_VALUES,
        'importance_methods': IMPORTANCE_METHODS,
        'sigmas': sigmas,
        'noise_condition': noise_condition,
        'noise_provenance': noise_provenance,
        'per_sigma': {},
    }

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        clean_scores = []
        noisy_scores = []
        # {imp_method: {condition: {est_k: [scores]}}}
        grid_acc = {}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Fixed eval features
            X_eval_train = reps[base_rep_name][train_idx]
            X_eval_val = reps[base_rep_name][val_idx]
            scaler_eval = StandardScaler()
            X_eval_train_s = scaler_eval.fit_transform(X_eval_train)
            X_eval_val_s = scaler_eval.transform(X_eval_val)

            # Clean baseline
            rf_clean = model_factory()
            rf_clean.fit(X_eval_train_s, y_train_clean)
            clean_scores.append(float(metric_fn(y_val, rf_clean.predict(X_eval_val_s))))

            # Inject noise (training only)
            _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                   fold_idx=fold_idx, is_clf=is_clf)
            y_noisy, _cmask, _prov = inject_train_labels(
                y_train_clean, _spec, is_clf=is_clf,
                groups=noise_groups[train_idx] if noise_groups is not None else None)
            noise_provenance.setdefault(_spec.spec_id, _prov)

            # Noisy baseline
            rf_noisy = model_factory()
            rf_noisy.fit(X_eval_train_s, y_noisy)
            noisy_scores.append(float(metric_fn(y_val, rf_noisy.predict(X_eval_val_s))))

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

            # Build the imp-method-independent pieces (hybrid + via_blueprint
            # importance) ONCE per (sigma, fold); the inner imp_method loop
            # then only recomputes the truly imp-method-dependent pieces.
            fold_shared = _build_fold_shared(
                train_reps_fold, y_noisy, train_augs, bp, base_rep_name)

            for imp_method in IMPORTANCE_METHODS:
                if imp_method not in grid_acc:
                    grid_acc[imp_method] = {}

                conditions = _build_conditions(
                    train_reps_fold, y_noisy, train_augs, bp,
                    imp_method, base_rep_name, shared=fold_shared,
                )

                for cond_name, X_space in conditions.items():
                    if cond_name not in grid_acc[imp_method]:
                        grid_acc[imp_method][cond_name] = {}

                    cond_results = _run_condition(
                        X_space, y_noisy, X_eval_train_s, X_eval_val_s,
                        y_val, model_factory, metric_fn, is_clf=is_clf,
                    )

                    for ek, score in cond_results.items():
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
        clean_mean = float(np.mean(clean_scores))
        noisy_mean = float(np.mean(noisy_scores))
        gap = clean_mean - noisy_mean

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
            'per_importance_method': {},
        }

        for imp_method in IMPORTANCE_METHODS:
            imp_result = {}
            for cond_name in CONDITION_NAMES:
                cond_agg = {}
                cond_errors = None
                for ek, score_list in grid_acc.get(imp_method, {}).get(cond_name, {}).items():
                    if ek == '_errors':
                        # Diagnostic marker: pass the error dict straight
                        # through rather than averaging it as a score list.
                        cond_errors = score_list
                        continue
                    if score_list:
                        denoised = float(np.mean(score_list))
                        recovery = float((denoised - noisy_mean) / gap * 100) if abs(gap) > 1e-10 else float('nan')
                        cond_agg[ek] = {
                            'denoised': denoised,
                            'recovery': recovery,
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
        hyb_uw = cond_avg.get('hybrid_unweighted', float('nan'))
        hyb_w_base = cond_avg.get('hybrid_imp_base', float('nan'))
        hyb_w_own = cond_avg.get('hybrid_imp_own', float('nan'))

        weighting_helps_base = (not np.isnan(base_w) and not np.isnan(base_uw)
                                and base_w > base_uw)
        weighting_helps_hybrid = (not np.isnan(hyb_w_own) and not np.isnan(hyb_uw)
                                  and (hyb_w_own > hyb_uw or hyb_w_base > hyb_uw))
        # Redundant if hybrid_unweighted ≈ hybrid_imp_own (within 2%)
        hybrid_already_weighted = (not np.isnan(hyb_uw) and not np.isnan(hyb_w_own)
                                   and abs(hyb_uw - hyb_w_own) < 2.0)

        # Best k
        k_recoveries = {}
        for imp_data in sigma_data.get('per_importance_method', {}).values():
            for cond_data in imp_data.values():
                for ek, rv in cond_data.items():
                    # Parse k from key like 'neighbor_residual_k10'
                    if '_k' in ek:
                        k_val = int(ek.split('_k')[-1])
                        if k_val not in k_recoveries:
                            k_recoveries[k_val] = []
                        r = rv.get('recovery', float('nan'))
                        if not np.isnan(r):
                            k_recoveries[k_val].append(r)

        best_k = None
        best_k_avg = float('-inf')
        for k_val, rs in k_recoveries.items():
            avg = float(np.mean(rs))
            if avg > best_k_avg:
                best_k_avg = avg
                best_k = k_val

        # Best importance method
        method_recoveries = {}
        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            all_r = []
            for cond_data in imp_data.values():
                for rv in cond_data.values():
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        all_r.append(r)
            if all_r:
                method_recoveries[imp_method] = float(np.mean(all_r))

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
