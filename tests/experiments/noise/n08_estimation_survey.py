#!/usr/bin/env python3
"""N08: Noise Estimation Survey — benchmark estimators from noise_estimation.py.

Paper: Noise
Exercises the noise_estimation library. For each estimator: score
distributions, cross-estimator correlations, and cliff discrimination
(noise scores should be roughly independent of cliff status — both strong
positive and strong negative correlation are failure modes).

Note: the file's earlier "downstream impact via sample weighting" claim
was never implemented and has been removed from the docstring. Only
cliff discrimination + cross-estimator correlations + score statistics
are reported. The number of estimators varies based on which optional
dependencies (ngboost, torch) are present.

Usage:
    python n08_estimation_survey.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import inspect
import numpy as np
from scipy.stats import spearmanr, skew
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_fitted_hybrid, get_fitted_pool_hybrid,
    cap_dimensions,
)
from kirby.utils.noise_estimation import get_noise_estimator
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# All estimators to test (skip torch-dependent for screening)
ESTIMATORS = [
    'neighbor_residual',
    'local_variance_ratio',
    'distance_weighted_residual',
    'activity_cliff_aware',
    'cross_val_residual',
    'ensemble_disagreement',
    'leave_one_out',
    'hybrid_noise',
    'gaussian_process',
    'conformal',
    'bayesian_rf',
]

# Torch-dependent (try-import)
TORCH_ESTIMATORS = [
    'heteroscedastic_nn',
    'deep_ensemble',
    'quantile_regression',
]

# Optional
OPTIONAL_ESTIMATORS = [
    'ngboost',
]

# is_clf-aware estimators: the kNN/neighbour-based ones whose constructor
# accepts is_clf and switches the per-neighbour disagreement to a class-mismatch
# rate (see _make_estimator). On a CLASSIFICATION dataset ONLY these produce a
# meaningful noise score. Every OTHER estimator here is model-based and
# regression-only — it fits a RandomForestRegressor (or GP / NN / quantile /
# ngboost) on the integer class labels, which does NOT crash but yields a
# meaningless noise number (FIX 4f). `hybrid_noise` is included in the skip set
# because HybridNoiseEstimator internally averages a CrossValResidualEstimator
# (a regressor on the labels) and exposes no is_clf branch, so it has the same
# defect. On classification we SKIP the regression-only estimators and record an
# explicit reason instead of producing a meaningless score; the kNN ones run.
IS_CLF_AWARE_ESTIMATORS = {
    'neighbor_residual', 'local_variance_ratio',
    'distance_weighted', 'distance_weighted_residual',
    'activity_cliff_aware',
}

GP_MAX = 1000

# Activity-cliff percentile knob sweep. A pair counts as "structurally similar"
# if its Euclidean distance is below the `similar_percentile`-th percentile of
# off-diagonal distances (near-neighbour rule). The main cliff-discrimination
# result uses PRIMARY_AC_PERCENTILE; the additive `ac_percentile_sweep` output
# reports cliff prevalence + the best cliff-discriminating estimator at EACH
# knob. The per-estimator noise scores are computed ONCE and are independent of
# the cliff definition, so the sweep just recomputes the cheap
# Spearman(score, is_cliff) correlation against each knob's cliff mask.
#
# NOTE: the primary knob was switched 5.0 -> 2.0 (PRIMARY_AC_PERCENTILE) for
# consistency with n01/n04, whose "structurally similar" knob is 2%. The 5%
# rule still appears as one of the swept knobs.
AC_PERCENTILES = [1.0, 2.0, 5.0, 10.0]
PRIMARY_AC_PERCENTILE = 2.0


def _cliff_mask_from_distances(D, labels_for_cliffs, similar_percentile, is_clf):
    """Cliff mask via the near-neighbour rule at a given percentile knob.

    A sample is a cliff if it has at least one neighbour within the closest
    `similar_percentile`% of off-diagonal distances whose activity differs by
    more than 2 MAD (regression) or by class (classification). Returns
    (cliff_mask: bool[n], near_thresh: float).
    """
    n = D.shape[0]
    off_diag = D[~np.eye(n, dtype=bool)]
    near_thresh = float(np.percentile(off_diag, similar_percentile)) \
        if off_diag.size else 0.0
    near = D <= near_thresh
    np.fill_diagonal(near, False)
    if is_clf:
        delta_y = (labels_for_cliffs[:, None]
                   != labels_for_cliffs[None, :]).astype(float)
        cliff_mask = (near & (delta_y > 0.5)).any(axis=1)
    else:
        mad_y = float(np.median(np.abs(labels_for_cliffs
                                       - np.median(labels_for_cliffs))))
        # Guard against a degenerate (all-identical labels) MAD of 0, which
        # would mark every near-neighbour pair as a cliff.
        y_thresh = 2.0 * mad_y if mad_y > 1e-12 else float(np.std(labels_for_cliffs))
        delta_y = np.abs(labels_for_cliffs[:, None] - labels_for_cliffs[None, :])
        cliff_mask = (near & (delta_y > y_thresh)).any(axis=1)
    return cliff_mask, near_thresh


def _make_estimator(est_name, is_clf):
    """Construct a noise estimator, passing is_clf ONLY to those that accept it.

    Only the kNN-based estimators (neighbor_residual, local_variance_ratio,
    distance_weighted_residual, activity_cliff_aware) expose an `is_clf`
    constructor argument that switches the per-neighbour disagreement from a
    regression value-residual to a class-mismatch rate. Without this, a
    classification dataset (e.g. the TCGA PAM50 / LUAD-vs-LUSC surveys) would
    have its integer class labels fed into a regression residual estimator,
    producing a meaningless noise score. The model-based estimators
    (cross_val, ensemble, gaussian_process, conformal, bayesian_rf, …) are
    inherently regression residual methods and take no is_clf flag; they are
    constructed unchanged. (Audit N08 "estimators have no clf branch".)
    """
    estimator = get_noise_estimator(est_name)
    if is_clf:
        try:
            params = inspect.signature(type(estimator).__init__).parameters
        except (TypeError, ValueError):
            params = {}
        if 'is_clf' in params:
            estimator = get_noise_estimator(est_name, is_clf=True)
    return estimator


def run(data, config, output_dir, dataset_name, **kwargs):
    """N08: Noise estimation survey."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N08: NOISE ESTIMATION SURVEY")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    # metric_name surfaces in results; the fn / direction aren't used here —
    # this experiment correlates noise scores with cliff masks, not against
    # the domain metric. n_folds / mode were set but never read either; the
    # estimator survey runs a single in-sample fit on a fixed subsample by
    # design, so removing them removes a misleading config knob.
    metric_name, _metric_fn, _lower_is_better = get_domain_metric(data)
    budget = kwargs.get('budget', 100)
    n_samples = len(labels)

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    spaces = {'concat': X_concat_s}
    try:
        X_hybrid, fi, _search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            data=data,
            dataset_name=dataset_name,
            task_type=task_type)
        spaces['hybrid'] = StandardScaler().fit_transform(X_hybrid)
    except Exception as e:
        print(f"  Hybrid failed: {e}")

    # Per-rep spaces
    for rep_name, X_rep in reps.items():
        spaces[rep_name] = cap_dimensions(StandardScaler().fit_transform(X_rep))[0]

    # GP-only subsample. Most estimators run cheaply on the full dataset;
    # GP-based ones (`gaussian_process`, `bayesian_rf`, `conformal`) are
    # O(n²)-O(n³) in memory/time and need a smaller subset. Previously a
    # single `sub_idx` was applied to every estimator, throwing away 97% of
    # data even for cheap methods on a 30K-sample dataset. Now each
    # estimator pulls its own X/y subset based on whether it's GP-class.
    GP_ESTIMATORS = {'gaussian_process', 'bayesian_rf', 'conformal'}
    if n_samples > GP_MAX:
        rng = np.random.RandomState(42)
        gp_sub_idx = rng.choice(n_samples, GP_MAX, replace=False)
    else:
        gp_sub_idx = np.arange(n_samples)
    full_idx = np.arange(n_samples)
    # The downstream loop uses `sub_idx` for the cliff-mask / printing; that
    # subset corresponds to whatever the estimator actually ran on, so we
    # set it per-estimator below. For now define it as the GP subset for
    # compatibility with the rest of the function — cliff_mask_sub is
    # computed from sub_idx and is shared across estimators.
    sub_idx = gp_sub_idx

    # Identify cliff compounds for cliff discrimination test. The pairwise
    # distance matrix is O(n²) memory — on 30K samples that's ~7GB. Cap the
    # cliff computation to the GP subsample (≤GP_MAX) and use that mask for
    # all estimators. Estimators that ran on the full data will have their
    # cliff_correlation computed against the subset rather than the full N.
    X_for_cliffs = X_concat[gp_sub_idx]
    labels_for_cliffs = labels[gp_sub_idx]
    D = compute_distance_matrix(X_for_cliffs, standardize=True)
    # Near-neighbour criterion. The previous version converted distance to a
    # similarity via sim = 1/(1+D) and thresholded sim >= 0.55 — but on a
    # high-dimensional standardised space the pairwise Euclidean distances are
    # O(sqrt(d)) (≈20-30 for the few-hundred-dim concat here), so sim collapses
    # to ≈0.03-0.05 for EVERY pair and the 0.55 threshold matched NOTHING:
    # n_cliffs was 0 on every dataset and the entire cliff-discrimination test
    # (the experiment's whole point) returned null. Replace the scale-dependent
    # absolute similarity threshold with a scale-free near-neighbour rule: a
    # pair counts as "structurally similar" if its distance is below the
    # `similar_percentile`-th percentile of off-diagonal distances. This is
    # invariant to dimensionality and metric scale.
    #
    # The headline cliff mask uses PRIMARY_AC_PERCENTILE (=2%, switched from the
    # old hardcoded 5% for consistency with n01/n04). Per-knob masks for the
    # additive sweep are built up front and reused; the pairwise distance matrix
    # D is computed only once.
    cliff_masks_subset_by_pct = {}
    near_thresh_by_pct = {}
    for pct in AC_PERCENTILES:
        m, nt = _cliff_mask_from_distances(D, labels_for_cliffs, pct, is_clf)
        cliff_masks_subset_by_pct[pct] = m
        near_thresh_by_pct[pct] = nt
    if PRIMARY_AC_PERCENTILE not in cliff_masks_subset_by_pct:
        m, nt = _cliff_mask_from_distances(
            D, labels_for_cliffs, PRIMARY_AC_PERCENTILE, is_clf)
        cliff_masks_subset_by_pct[PRIMARY_AC_PERCENTILE] = m
        near_thresh_by_pct[PRIMARY_AC_PERCENTILE] = nt
    cliff_mask_subset = cliff_masks_subset_by_pct[PRIMARY_AC_PERCENTILE]
    near_thresh = near_thresh_by_pct[PRIMARY_AC_PERCENTILE]
    # Project the subset-cliff-mask back onto the full-index space: False
    # everywhere except at gp_sub_idx positions. Per-estimator code below
    # subsets this with cliff_mask[est_idx]; for full-index estimators we
    # only have cliff information for the gp_sub_idx ⊂ full_idx samples.
    cliff_mask = np.zeros(n_samples, dtype=bool)
    cliff_mask[gp_sub_idx] = cliff_mask_subset
    # Per-knob full-index cliff masks for the sweep.
    cliff_masks_by_pct = {}
    for pct, m in cliff_masks_subset_by_pct.items():
        fm = np.zeros(n_samples, dtype=bool)
        fm[gp_sub_idx] = m
        cliff_masks_by_pct[pct] = fm
    # Backwards-compat alias for code that still references cliff_mask_sub.
    cliff_mask_sub = cliff_mask[sub_idx]
    n_cliffs = int(cliff_mask_sub.sum())
    print(f"  Cliffs (primary {PRIMARY_AC_PERCENTILE}%): {n_cliffs}/{len(sub_idx)} "
          f"(computed on GP subsample of size {len(gp_sub_idx)})")
    for pct in AC_PERCENTILES:
        nc = int(cliff_masks_by_pct[pct][sub_idx].sum())
        print(f"    cliff knob {pct:>5.1f}%: {nc}/{len(sub_idx)} "
              f"({100*nc/max(len(sub_idx),1):.1f}%)")
    if n_cliffs == 0:
        # Self-diagnosing: if cliff detection still finds nothing, surface the
        # distance distribution that fed the near-neighbour rule so a rerun
        # isn't blind. (Historically this was silently 0 on every dataset.)
        print(f"  WARNING: 0 cliffs detected — near_thresh={near_thresh:.4g}, "
              f"D range=[{float(D.min()):.4g},{float(D.max()):.4g}]; "
              f"cliff-discrimination test will be empty for this dataset.")

    # Determine which estimators to test
    all_estimators = list(ESTIMATORS)
    try:
        import torch
        all_estimators.extend(TORCH_ESTIMATORS)
    except ImportError:
        print("  PyTorch not available — skipping torch-dependent estimators")
    for opt in OPTIONAL_ESTIMATORS:
        try:
            get_noise_estimator(opt)
            all_estimators.append(opt)
        except Exception:
            pass

    print(f"  Testing {len(all_estimators)} estimators × {len(spaces)} spaces")

    results = {
        'metric': metric_name,
        'n_samples': n_samples,
        'n_subsampled': len(sub_idx),
        'n_cliffs': n_cliffs,
        'cliff_near_thresh': near_thresh,
        'ac_percentiles': AC_PERCENTILES,
        'primary_ac_percentile': PRIMARY_AC_PERCENTILE,
        'estimators': all_estimators,
        'per_space': {},
    }

    all_scores_by_space = {}  # for cross-correlation

    for space_name, X_full in spaces.items():
        print(f"\n  === Space: {space_name} (dim={X_full.shape[1]}) ===")
        space_results = {}
        space_scores = {}

        for est_name in all_estimators:
            print(f"    {est_name}...", end='', flush=True)
            t0 = time.time()

            # FIX 4f: on a classification dataset, the model-based regression
            # estimators fit a regressor on integer class labels — they don't
            # crash but the noise score is meaningless. Skip them and record an
            # explicit reason rather than storing a junk number. The kNN/is_clf-
            # aware estimators (IS_CLF_AWARE_ESTIMATORS) handle classification
            # correctly via _make_estimator, so they still run.
            if is_clf and est_name not in IS_CLF_AWARE_ESTIMATORS:
                space_results[est_name] = {
                    'skipped': 'skipped_regression_estimator_on_clf'}
                print(" SKIP (regression-only estimator on clf)")
                continue

            # Per-estimator subsample: GP-class estimators use the small
            # subset; everything else sees the full data. Cross-estimator
            # correlations below will only compare estimators that ran on
            # the same indices (i.e. same length scores arrays).
            est_idx = gp_sub_idx if est_name in GP_ESTIMATORS else full_idx
            X_sub = X_full[est_idx]
            y_sub = labels[est_idx]

            try:
                estimator = _make_estimator(est_name, is_clf)
                estimator.fit(X_sub, y_sub)
                scores = estimator.noise_scores_
                elapsed = time.time() - t0

                entry = {
                    'time_s': float(elapsed),
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                    'score_skew': float(skew(scores)),
                    'score_min': float(np.min(scores)),
                    'score_max': float(np.max(scores)),
                }

                # Cliff discrimination: |Spearman(score, is_cliff)| should be
                # small. A good estimator's noise score is INDEPENDENT of
                # cliff status (cliffs are real chemistry signal, not noise).
                # Both strong positive correlation (rates cliffs as noisy) AND
                # strong negative correlation (rates cliffs as the cleanest
                # samples) are failure modes; we want |rho| ≈ 0.
                # Per-estimator subset of the cliff mask (estimators may now
                # have ran on different index subsets).
                cliff_mask_est = cliff_mask[est_idx]
                if int(cliff_mask_est.sum()) >= 5:
                    rho_cliff, p_cliff = spearmanr(scores, cliff_mask_est.astype(float))
                    entry['cliff_correlation'] = float(rho_cliff)
                    entry['cliff_abs_correlation'] = float(abs(rho_cliff))
                    entry['cliff_p'] = float(p_cliff)
                    entry['cliff_discriminates'] = abs(rho_cliff) < 0.1
                    entry['cliff_n_used'] = int(cliff_mask_est.sum())

                # Sample weights
                try:
                    weights = estimator.get_weights(strength=1.0)
                    entry['weight_mean'] = float(np.mean(weights))
                    entry['weight_std'] = float(np.std(weights))
                except Exception:
                    pass

                space_scores[est_name] = scores
                space_results[est_name] = entry
                _cliff_rho = entry.get('cliff_correlation')
                # N9: guard on numeric-finite, not just `is not None` — a
                # non-float sentinel (e.g. '?'/'n/a') would crash `:.3f`.
                cliff_str = (f", cliff_rho={_cliff_rho:.3f}"
                             if isinstance(_cliff_rho, (int, float))
                             and np.isfinite(_cliff_rho) else "")
                print(f" {elapsed:.1f}s, mean={entry['score_mean']:.4f}{cliff_str}")

            except Exception as e:
                print(f" FAILED: {e}")
                space_results[est_name] = {'error': str(e)}

        # Cross-estimator correlations. Keep the pair-level results in two
        # parallel structures: a string-keyed `_cross_correlations` (kept for
        # backwards compat with anything that reads it) and a tuple-keyed
        # internal map used for the aggregation. The aggregation uses
        # tuple-membership lookup rather than substring matching on the
        # string keys — substring matching would silently double-count if a
        # future estimator's name contains another's as a substring (e.g.
        # adding "ensemble" with existing "ensemble_disagreement" /
        # "deep_ensemble").
        est_names_ok = [e for e in all_estimators if e in space_scores]
        if len(est_names_ok) >= 2:
            corr_matrix = {}
            corr_pairs = {}
            for i, e1 in enumerate(est_names_ok):
                for e2 in est_names_ok[i+1:]:
                    # Only correlate estimators that ran on the same number
                    # of samples (i.e. either both GP-class on the small
                    # subset or both non-GP on the full data). Mismatched
                    # lengths would raise; previously this was impossible
                    # because every estimator used the same subset.
                    if len(space_scores[e1]) != len(space_scores[e2]):
                        continue
                    rho, _ = spearmanr(space_scores[e1], space_scores[e2])
                    corr_matrix[f'{e1}_vs_{e2}'] = float(rho)
                    corr_pairs[(e1, e2)] = float(rho)

            space_results['_cross_correlations'] = corr_matrix
            avg_corr = {}
            for e in est_names_ok:
                rhos = [v for (a, b), v in corr_pairs.items() if e in (a, b)]
                avg_corr[e] = float(np.mean(rhos)) if rhos else 0.0
            space_results['_avg_cross_correlation'] = avg_corr

        results['per_space'][space_name] = convert_numpy(space_results)
        all_scores_by_space[space_name] = space_scores

    # Summary — rank cliff discrimination by |rho|, smaller is better.
    # (Signed rho would happily pick an estimator with rho=-0.9 as the
    # 'best discriminator', which is just the inverted failure mode.)
    summary = {}
    for space_name in spaces:
        sr = results['per_space'].get(space_name, {})
        best_cliff_est = None
        best_cliff_abs = float('inf')
        best_cliff_signed = None
        for est_name in all_estimators:
            rho = sr.get(est_name, {}).get('cliff_correlation')
            if isinstance(rho, (int, float)) and abs(rho) < best_cliff_abs:
                best_cliff_abs = abs(rho)
                best_cliff_signed = float(rho)
                best_cliff_est = est_name
        summary[space_name] = {
            'best_cliff_discriminator': best_cliff_est,
            'cliff_abs_correlation': (float(best_cliff_abs)
                                      if best_cliff_est else None),
            'cliff_correlation_signed': best_cliff_signed,
            'n_estimators_ok': sum(
                1 for e in all_estimators
                if 'error' not in sr.get(e, {})
                and 'skipped' not in sr.get(e, {})),
        }

    results['summary'] = summary
    print(f"\n  Summary: {summary}")

    # =========================================================================
    # ADDITIVE: activity-cliff percentile knob SWEEP
    # =========================================================================
    # The per-estimator noise scores in all_scores_by_space are computed ONCE
    # and are independent of the cliff definition. For EACH percentile knob we
    # cheaply recompute Spearman(score, is_cliff) per estimator/space and pick
    # the best discriminator (smallest |rho|). No estimator re-fitting happens.
    print("\n  === ac_percentile_sweep ===")
    GP_ESTIMATORS = {'gaussian_process', 'bayesian_rf', 'conformal'}
    ac_sweep = {}
    for pct in AC_PERCENTILES:
        full_cmask = cliff_masks_by_pct[pct]
        n_cliffs_pct = int(full_cmask[sub_idx].sum())

        per_space_sweep = {}
        for space_name, space_scores in all_scores_by_space.items():
            per_est = {}
            for est_name, scores in space_scores.items():
                # Reconstruct the index subset this estimator ran on so the
                # cliff mask aligns with the stored scores.
                est_idx = gp_sub_idx if est_name in GP_ESTIMATORS else full_idx
                cmask_est = full_cmask[est_idx]
                if int(cmask_est.sum()) < 5 or len(scores) != len(cmask_est):
                    continue
                rho, p = spearmanr(scores, cmask_est.astype(float))
                per_est[est_name] = {
                    'cliff_correlation': float(rho),
                    'cliff_abs_correlation': float(abs(rho)),
                    'cliff_p': float(p),
                    'cliff_discriminates': bool(abs(rho) < 0.1),
                    'cliff_n_used': int(cmask_est.sum()),
                }
            # Best discriminator = smallest |rho| (independent of cliff status).
            best_est, best_abs, best_signed = None, float('inf'), None
            for est_name, e in per_est.items():
                if e['cliff_abs_correlation'] < best_abs:
                    best_abs = e['cliff_abs_correlation']
                    best_signed = e['cliff_correlation']
                    best_est = est_name
            per_space_sweep[space_name] = {
                'per_estimator': per_est,
                'best_cliff_discriminator': best_est,
                'cliff_abs_correlation': (float(best_abs)
                                          if best_est else None),
                'cliff_correlation_signed': best_signed,
            }

        ac_sweep[str(pct)] = convert_numpy({
            'similar_percentile': pct,
            'near_thresh': near_thresh_by_pct[pct],
            'n_cliffs': n_cliffs_pct,
            'pct_cliffs': float(100 * n_cliffs_pct / max(len(sub_idx), 1)),
            'cliffs_degenerate': bool(n_cliffs_pct < 5),
            'per_space': per_space_sweep,
        })
        # Print prevalence + the hybrid (or concat) best-discriminator |rho|.
        _ref = per_space_sweep.get('hybrid') or per_space_sweep.get('concat') or {}
        _abs = _ref.get('cliff_abs_correlation')
        print(f"    knob {pct:>5.1f}%: n_cliffs={n_cliffs_pct:>4} "
              f"({100*n_cliffs_pct/max(len(sub_idx),1):.1f}%)  "
              f"best_discr={_ref.get('best_cliff_discriminator')}  "
              # N9: numeric-finite guard so a non-float sentinel can't crash `:.3f`.
              f"|rho|={f'{_abs:.3f}' if isinstance(_abs, (int, float)) and np.isfinite(_abs) else 'n/a'}")

    results['ac_percentile_sweep'] = ac_sweep

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N08: Noise Estimation Survey')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n08_estimation_survey', args.dataset)
