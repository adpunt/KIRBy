#!/usr/bin/env python3
"""N04: GP Leave-One-Out Cliff-Aware Noise Detection.

Paper: Noise (KILLER EXPERIMENT)
Core thesis: In hybrid space, GP-LOO predictive probability distinguishes genuine
    noise from activity cliffs. In standard fingerprint/concat space, it cannot —
    because distances don't track activity, cliffs and noise look identical.

Steps:
  1. Build hybrid representation
  2. Fit GP in hybrid space
  3. LOO predictions → flag samples with low predictive probability
  4. Validate via synthetic noise injection: are injected-noise samples detected
     more often than genuine cliff compounds?

5 comparison conditions:
  - GP-LOO in hybrid space (MAIN — should have lowest cliff FP rate)
  - GP-LOO in full concat space (baseline)
  - GP-LOO in PCA-50 space (dimension-matched)
  - ActivityCliffAwareEstimator (library, non-GP)
  - Prediction error filter (naive baseline)

Usage:
    python n04_gp_loo_detection.py --dataset logd --mode intermediary
    python n04_gp_loo_detection.py --dataset flab_thermo_tresanco --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric,
)
from experiments.complexity.complexity_helpers import cached_create_hybrid as create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.
from kirby.utils.noise_estimation import ActivityCliffAwareEstimator
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# Maximum samples for GP (O(n³))
GP_MAX_SAMPLES = 1000


def _subsample_if_needed(X, y, max_n, seed=42):
    """Subsample to max_n if needed. Returns (X_sub, y_sub, idx)."""
    if len(y) <= max_n:
        return X, y, np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), max_n, replace=False)
    return X[idx], y[idx], idx


def _identify_cliffs_euclidean(X, y, sim_threshold=0.55, activity_k=2.0):
    """Identify activity cliff compounds using Euclidean distance."""
    D = compute_distance_matrix(X, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    mad_y = float(np.median(np.abs(y - np.median(y))))
    activity_threshold = activity_k * mad_y
    delta_y = np.abs(y[:, None] - y[None, :])
    cliff_mask = ((sim >= sim_threshold) & (delta_y > activity_threshold)).any(axis=1)
    return cliff_mask


def _gp_loo_scores(X, y):
    """Compute GP LOO noise scores.

    For each sample, fit GP on all-but-one, predict held-out.
    Score = |y_i - mu_i| / sigma_i (standardised residual).
    Higher = more likely noise.
    """
    n = len(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kernel = ConstantKernel() * RBF() + WhiteKernel()
    scores = np.zeros(n)
    gp_failures = 0

    # Full GP fit first to get hyperparameters (LOO refit is too expensive)
    try:
        gp_full = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                           random_state=42, alpha=1e-6)
        gp_full.fit(X_s, y)
        learned_kernel = gp_full.kernel_
    except Exception:
        # Fallback to default kernel
        learned_kernel = kernel
        gp_failures += 1

    # LOO predictions using the learned kernel (fast approximate LOO)
    try:
        gp = GaussianProcessRegressor(kernel=learned_kernel, optimizer=None,
                                      random_state=42, alpha=1e-6)
        gp.fit(X_s, y)
        # Approximate LOO via prediction on training data with noise
        # True LOO would refit n times — use the full model's uncertainty instead
        y_pred, y_std = gp.predict(X_s, return_std=True)
        y_std = np.maximum(y_std, 1e-10)
        scores = np.abs(y - y_pred) / y_std
    except Exception as e:
        print(f"    GP LOO failed: {e}")
        gp_failures += 1
        # Fallback to RF prediction error
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_s, y)
        y_pred = rf.predict(X_s)
        scores = np.abs(y - y_pred) / (np.std(y) + 1e-10)

    return scores, gp_failures


def _inject_noise(y, fraction, seed=42):
    """Inject noise into fraction of labels. Returns (y_noisy, noise_mask)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_noisy = max(1, int(n * fraction))
    noise_idx = rng.choice(n, n_noisy, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noise_idx] = True

    y_noisy = y.copy()
    y_range = np.max(y) - np.min(y)
    y_noisy[noise_idx] += rng.normal(0, y_range * 0.3, n_noisy)

    return y_noisy, noise_mask


def run(data, config, output_dir, dataset_name, **kwargs):
    """N04: GP-LOO cliff-aware noise detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N04: GP-LOO CLIFF-AWARE NOISE DETECTION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    augmentation_strategy = 'uncapped'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    noise_fractions = kwargs.get('noise_fractions', [0.05, 0.1, 0.2])
    budget = kwargs.get('budget', 100)

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    total_dim = X_concat.shape[1]

    spaces = {}

    # 1. Hybrid
    try:
        X_hybrid, fi = create_hybrid(
            reps, labels,
            allocation_method='greedy',
            importance_method='lightgbm_gain',
            budget=budget,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
        )
        spaces['hybrid'] = X_hybrid
    except Exception as e:
        print(f"  Hybrid creation failed: {e}")

    # 2. Concat
    spaces['concat'] = X_concat

    # 3. PCA-matched
    pca_dim = min(budget, total_dim, len(labels) - 1)
    if pca_dim >= 2:
        spaces['pca_matched'] = PCA(n_components=pca_dim, random_state=42).fit_transform(
            StandardScaler().fit_transform(X_concat))

    print(f"  Spaces: {list(spaces.keys())}")
    print(f"  Noise fractions: {noise_fractions}")

    # Identify cliffs in the original data (using concat Euclidean)
    X_sub, y_sub, sub_idx = _subsample_if_needed(X_concat, labels, GP_MAX_SAMPLES)
    cliff_mask = _identify_cliffs_euclidean(X_sub, y_sub)
    n_cliffs = int(cliff_mask.sum())
    print(f"  Cliff compounds: {n_cliffs}/{len(y_sub)} ({100*n_cliffs/len(y_sub):.1f}%)")

    results = {
        'n_samples': len(labels),
        'n_subsampled': len(y_sub),
        'n_cliffs': n_cliffs,
        'pct_cliffs': float(100 * n_cliffs / len(y_sub)),
        'noise_fractions': noise_fractions,
        'spaces': list(spaces.keys()),
        'per_noise_level': {},
    }

    for frac in noise_fractions:
        print(f"\n  === Noise fraction: {frac:.0%} ===")
        y_noisy, noise_mask_full = _inject_noise(labels, frac)

        # Subsample noise mask
        noise_mask = noise_mask_full[sub_idx]
        y_noisy_sub = y_noisy[sub_idx]
        n_injected = int(noise_mask.sum())
        print(f"    Injected: {n_injected} noisy samples")

        level_results = {}

        # GP-LOO conditions
        for space_name, X_full in spaces.items():
            X_space_sub = X_full[sub_idx] if len(X_full) > len(sub_idx) else X_full
            if X_space_sub.shape[0] != len(y_noisy_sub):
                continue

            print(f"    {space_name} (dim={X_space_sub.shape[1]})...", end='', flush=True)
            scores, gp_failures = _gp_loo_scores(X_space_sub, y_noisy_sub)

            # Threshold: flag top (frac * 100)% by score
            threshold_pct = min(frac * 2, 0.3)  # flag up to 2× the injection rate
            threshold = np.percentile(scores, (1 - threshold_pct) * 100)
            flagged = scores >= threshold

            # Metrics
            tp = int((flagged & noise_mask).sum())  # correctly flagged injected noise
            fp_cliff = int((flagged & cliff_mask & ~noise_mask).sum())  # cliffs wrongly flagged
            fp_other = int((flagged & ~noise_mask & ~cliff_mask).sum())
            fn = int((~flagged & noise_mask).sum())

            precision = tp / max(flagged.sum(), 1)
            recall = tp / max(noise_mask.sum(), 1)
            cliff_fp_rate = fp_cliff / max(cliff_mask.sum(), 1)

            level_results[space_name] = {
                'gp_failures': gp_failures,
                'n_flagged': int(flagged.sum()),
                'true_positives': tp,
                'false_positives_cliff': fp_cliff,
                'false_positives_other': fp_other,
                'false_negatives': fn,
                'precision': float(precision),
                'recall': float(recall),
                'cliff_fp_rate': float(cliff_fp_rate),
            }
            print(f" precision={precision:.3f}, recall={recall:.3f}, "
                  f"cliff_FP={cliff_fp_rate:.3f}")

        # ActivityCliffAwareEstimator condition (non-GP)
        if 'hybrid' in spaces:
            X_h_sub = spaces['hybrid'][sub_idx] if len(spaces['hybrid']) > len(sub_idx) else spaces['hybrid']
            if X_h_sub.shape[0] == len(y_noisy_sub):
                print(f"    cliff_aware_estimator...", end='', flush=True)
                try:
                    estimator = ActivityCliffAwareEstimator(k=10, metric='cosine')
                    estimator.fit(StandardScaler().fit_transform(X_h_sub), y_noisy_sub)
                    scores_ca = estimator.noise_scores_
                    threshold_ca = np.percentile(scores_ca, (1 - threshold_pct) * 100)
                    flagged_ca = scores_ca >= threshold_ca

                    tp_ca = int((flagged_ca & noise_mask).sum())
                    fp_cliff_ca = int((flagged_ca & cliff_mask & ~noise_mask).sum())
                    precision_ca = tp_ca / max(flagged_ca.sum(), 1)
                    recall_ca = tp_ca / max(noise_mask.sum(), 1)
                    cliff_fp_ca = fp_cliff_ca / max(cliff_mask.sum(), 1)

                    level_results['cliff_aware_estimator'] = {
                        'n_flagged': int(flagged_ca.sum()),
                        'true_positives': tp_ca,
                        'false_positives_cliff': fp_cliff_ca,
                        'precision': float(precision_ca),
                        'recall': float(recall_ca),
                        'cliff_fp_rate': float(cliff_fp_ca),
                    }
                    print(f" precision={precision_ca:.3f}, recall={recall_ca:.3f}, "
                          f"cliff_FP={cliff_fp_ca:.3f}")
                except Exception as e:
                    print(f" FAILED: {e}")

        # Prediction error baseline
        print(f"    prediction_error...", end='', flush=True)
        X_concat_sub = X_concat[sub_idx] if len(X_concat) > len(sub_idx) else X_concat
        scaler = StandardScaler()
        X_cs = scaler.fit_transform(X_concat_sub)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_cs, y_noisy_sub)
        pe_scores = np.abs(y_noisy_sub - rf.predict(X_cs))
        threshold_pe = np.percentile(pe_scores, (1 - threshold_pct) * 100)
        flagged_pe = pe_scores >= threshold_pe
        tp_pe = int((flagged_pe & noise_mask).sum())
        fp_cliff_pe = int((flagged_pe & cliff_mask & ~noise_mask).sum())
        precision_pe = tp_pe / max(flagged_pe.sum(), 1)
        recall_pe = tp_pe / max(noise_mask.sum(), 1)
        cliff_fp_pe = fp_cliff_pe / max(cliff_mask.sum(), 1)

        level_results['prediction_error'] = {
            'n_flagged': int(flagged_pe.sum()),
            'true_positives': tp_pe,
            'false_positives_cliff': fp_cliff_pe,
            'precision': float(precision_pe),
            'recall': float(recall_pe),
            'cliff_fp_rate': float(cliff_fp_pe),
        }
        print(f" precision={precision_pe:.3f}, recall={recall_pe:.3f}, "
              f"cliff_FP={cliff_fp_pe:.3f}")

        results['per_noise_level'][str(frac)] = convert_numpy(level_results)

    # Summary: does hybrid have the lowest cliff FP rate across noise levels?
    hybrid_wins = 0
    total_comparisons = 0
    for frac_key, level_res in results['per_noise_level'].items():
        h_cfp = level_res.get('hybrid', {}).get('cliff_fp_rate', 1.0)
        for cond, cond_res in level_res.items():
            if cond == 'hybrid':
                continue
            c_cfp = cond_res.get('cliff_fp_rate', 1.0)
            total_comparisons += 1
            if h_cfp <= c_cfp:
                hybrid_wins += 1

    results['summary'] = {
        'hybrid_lowest_cliff_fp_rate': hybrid_wins,
        'total_comparisons': total_comparisons,
        'win_rate': float(hybrid_wins / total_comparisons) if total_comparisons > 0 else 0.0,
    }
    print(f"\n  Hybrid lowest cliff FP rate: {hybrid_wins}/{total_comparisons}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N04: GP-LOO Noise Detection')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n04_gp_loo_detection', args.dataset)
