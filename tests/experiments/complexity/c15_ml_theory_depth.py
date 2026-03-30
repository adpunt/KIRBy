#!/usr/bin/env python3
"""C15: ML Theory Depth — concentration of measure, kernel degradation,
double descent, and approximation theory analysis.

Paper: Complexity
Conclusion: Compact hybrid reps avoid high-D pathologies: distance concentration,
    kernel degeneration, hubness, and interpolation-threshold instability.
Dependencies: c01 (rep characterization for comparison), c02 (grid results optional).

Four sub-analyses:
  15a: Distance concentration — Beyer et al. (1999) ratio + pairwise distance
       distribution comparison across budgets. The theoretical WHY behind everything.
  15b: Kernel matrix analysis — condition number, eigenspectrum, and kernel value
       distribution for RBF/Tanimoto at different dimensionalities. Explains why
       GP/SVM benefit disproportionately from compact reps.
  15c: Double descent probe — sweep budget through n ≈ p regime. Test whether
       hybrids skip the interpolation threshold entirely.
  15d: Approximation theory — Rademacher complexity estimation, generalization
       gap vs compression ratio. Empirical PAC-style bounds.

References:
  - Beyer et al. (ICDT 1999) "When Is Nearest Neighbor Meaningful?"
  - Aggarwal et al. (ICDT 2001) "On the Surprising Behavior of Distance Metrics"
  - Radovanovic et al. (JMLR 2010) "Hubs in Space"
  - Bengio et al. (2005) "The Curse of Dimensionality for Local Kernel Machines"

Usage:
    python c15_ml_theory_depth.py --dataset logd --reps all --mode full
    python c15_ml_theory_depth.py --dataset logd --reps all --mode full --sub 15a 15b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    compute_effective_rank,
    get_fitted_hybrid, get_precomputed_search, build_fold_hybrid,
)


# =========================================================================
# 15a: Distance Concentration (Beyer et al. 1999)
# =========================================================================

def _run_distance_concentration(data, budgets):
    """15a: Measure distance concentration at each dimensionality.

    The Beyer ratio: (max_dist - min_dist) / min_dist → 0 as d → ∞.
    Also measures:
    - Relative contrast (Aggarwal et al. 2001)
    - Pairwise distance CV (coefficient of variation)
    - Hubness: skewness of k-occurrence distribution (Radovanovic 2010)
    """
    print("\n  15a: Distance Concentration Analysis")
    print("  " + "-" * 60)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_samples = len(labels)

    results = {'per_rep': {}, 'per_budget': {}, 'concat': {}}

    # --- Per-rep analysis (raw representations) ---
    for rep_name, X in sorted(reps.items()):
        X_s = StandardScaler().fit_transform(X)
        metrics = _compute_concentration_metrics(X_s, k=5)
        results['per_rep'][rep_name] = {
            'dim': X.shape[1],
            **metrics,
        }
        print(f"    {rep_name:<20} dim={X.shape[1]:>5}  "
              f"beyer={metrics['beyer_ratio']:.4f}  "
              f"cv={metrics['distance_cv']:.4f}  "
              f"hubness={metrics['hubness_skewness']:.3f}")

    # --- Concatenated baseline ---
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)
    metrics = _compute_concentration_metrics(X_concat_s, k=5)
    results['concat'] = {
        'dim': X_concat.shape[1],
        **metrics,
    }
    print(f"    {'concat':<20} dim={X_concat.shape[1]:>5}  "
          f"beyer={metrics['beyer_ratio']:.4f}  "
          f"cv={metrics['distance_cv']:.4f}  "
          f"hubness={metrics['hubness_skewness']:.3f}")

    # --- Per-budget analysis (hybrid at each budget) ---
    print(f"\n    Budget sweep (hybrid):")
    for budget in sorted(budgets):
        try:
            X_hybrid, _, _search = get_fitted_hybrid(
                reps, labels, groups, budget=budget,
                aug_condition='none', speed='fast')
            X_hybrid_s = StandardScaler().fit_transform(X_hybrid)
            metrics = _compute_concentration_metrics(X_hybrid_s, k=5)
            results['per_budget'][budget] = {
                'dim': X_hybrid.shape[1],
                **metrics,
            }
            print(f"      budget={budget:>4}  dim={X_hybrid.shape[1]:>4}  "
                  f"beyer={metrics['beyer_ratio']:.4f}  "
                  f"cv={metrics['distance_cv']:.4f}  "
                  f"hubness={metrics['hubness_skewness']:.3f}")
        except Exception as e:
            print(f"      budget={budget}: FAILED - {e}")

    # --- PCA baselines at matching dimensions ---
    print(f"\n    PCA baselines (unsupervised):")
    results['pca'] = {}
    for budget in sorted(budgets):
        n_comp = min(budget, X_concat.shape[1], n_samples - 1)
        try:
            X_pca = PCA(n_components=n_comp).fit_transform(X_concat_s)
            metrics = _compute_concentration_metrics(X_pca, k=5)
            results['pca'][budget] = {
                'dim': n_comp,
                **metrics,
            }
            print(f"      pca_d={n_comp:>4}  "
                  f"beyer={metrics['beyer_ratio']:.4f}  "
                  f"cv={metrics['distance_cv']:.4f}  "
                  f"hubness={metrics['hubness_skewness']:.3f}")
        except Exception:
            pass

    # --- Summary ---
    print(f"\n    Summary:")
    print(f"    {'Space':<20} {'Dim':>5} {'Beyer':>8} {'CV':>8} {'Hubness':>8}")
    print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for rep_name, m in sorted(results['per_rep'].items(),
                               key=lambda x: x[1]['dim']):
        print(f"    {rep_name:<20} {m['dim']:>5} {m['beyer_ratio']:>8.4f} "
              f"{m['distance_cv']:>8.4f} {m['hubness_skewness']:>8.3f}")
    m = results['concat']
    print(f"    {'concat':<20} {m['dim']:>5} {m['beyer_ratio']:>8.4f} "
          f"{m['distance_cv']:>8.4f} {m['hubness_skewness']:>8.3f}")
    for budget in sorted(budgets):
        if budget in results['per_budget']:
            m = results['per_budget'][budget]
            print(f"    {'hybrid_b' + str(budget):<20} {m['dim']:>5} "
                  f"{m['beyer_ratio']:>8.4f} {m['distance_cv']:>8.4f} "
                  f"{m['hubness_skewness']:>8.3f}")

    return results


def _compute_concentration_metrics(X, k=5, max_subsample=2000):
    """Compute distance concentration metrics for a feature matrix.

    Returns dict with:
    - beyer_ratio: (max_dist - min_dist) / min_dist (→ 0 means concentrated)
    - relative_contrast: (max_dist - min_dist) / mean_dist
    - distance_cv: std(dists) / mean(dists)
    - distance_iqr_ratio: IQR / median
    - hubness_skewness: skewness of k-occurrence counts (Radovanovic 2010)
    - hubness_robin_hood: fraction of total occurrences held by top 1% of hubs
    """
    n = X.shape[0]

    # Pairwise distances (subsample if large)
    if n > max_subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
        idx = np.arange(n)

    dists = pdist(X_sub, metric='euclidean')

    if len(dists) == 0:
        return {
            'beyer_ratio': float('nan'), 'relative_contrast': float('nan'),
            'distance_cv': float('nan'), 'distance_iqr_ratio': float('nan'),
            'hubness_skewness': float('nan'), 'hubness_robin_hood': float('nan'),
        }

    min_d = np.min(dists[dists > 0]) if np.any(dists > 0) else 1e-10
    max_d = np.max(dists)
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    median_d = np.median(dists)
    q25, q75 = np.percentile(dists, [25, 75])

    beyer = (max_d - min_d) / min_d if min_d > 0 else float('inf')
    contrast = (max_d - min_d) / mean_d if mean_d > 0 else 0
    cv = std_d / mean_d if mean_d > 0 else 0
    iqr_ratio = (q75 - q25) / median_d if median_d > 0 else 0

    # Hubness: how many times each point appears as a k-NN of others
    k_use = min(k, len(X_sub) - 1)
    nn = NearestNeighbors(n_neighbors=k_use + 1, n_jobs=-1)
    nn.fit(X_sub)
    _, indices = nn.kneighbors(X_sub)
    indices = indices[:, 1:]  # exclude self

    # k-occurrence count: N_k(i) = how many times point i appears in others' k-NN
    n_occ = np.zeros(len(X_sub))
    for i in range(len(X_sub)):
        for j in range(k_use):
            n_occ[indices[i, j]] += 1

    hub_skewness = float(skew(n_occ))

    # Robin Hood index: fraction of total occurrences from top 1%
    n_top = max(1, int(0.01 * len(X_sub)))
    sorted_occ = np.sort(n_occ)[::-1]
    robin_hood = float(sorted_occ[:n_top].sum() / n_occ.sum()) if n_occ.sum() > 0 else 0

    return {
        'beyer_ratio': float(beyer),
        'relative_contrast': float(contrast),
        'distance_cv': float(cv),
        'distance_iqr_ratio': float(iqr_ratio),
        'hubness_skewness': hub_skewness,
        'hubness_robin_hood': robin_hood,
    }


# =========================================================================
# 15b: Kernel Matrix Analysis
# =========================================================================

def _run_kernel_analysis(data, budgets):
    """15b: Kernel matrix properties at different dimensionalities.

    For RBF kernel: k(x,x') = exp(-γ||x-x'||²).
    In high-D, all ||x-x'||² converge → all kernel values converge → matrix
    becomes near-singular and the GP/SVM loses discriminative power.

    Measures:
    - Kernel value distribution: mean, std, min, max of off-diagonal entries
    - Condition number of K (log10)
    - Eigenspectrum: effective rank, participation ratio
    - Kernel-target alignment: normalized Frobenius inner product <K, yy^T>
    """
    print("\n  15b: Kernel Matrix Analysis")
    print("  " + "-" * 60)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_samples = len(labels)

    # Subsample for computational feasibility (kernel matrix is n×n)
    max_n = 1000
    if n_samples > max_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, max_n, replace=False)
        sub_reps = {k: v[idx] for k, v in reps.items()}
        sub_labels = labels[idx]
    else:
        sub_reps = reps
        sub_labels = labels
        idx = np.arange(n_samples)

    n = len(sub_labels)
    results = {'per_rep': {}, 'per_budget': {}, 'concat': {}}

    # --- Per-rep kernel analysis ---
    for rep_name, X in sorted(sub_reps.items()):
        X_s = StandardScaler().fit_transform(X)
        metrics = _compute_kernel_metrics(X_s, sub_labels)
        results['per_rep'][rep_name] = {'dim': X.shape[1], **metrics}
        print(f"    {rep_name:<20} dim={X.shape[1]:>5}  "
              f"cond={metrics['log_condition']:>7.1f}  "
              f"k_cv={metrics['kernel_cv']:.4f}  "
              f"align={metrics['kernel_target_alignment']:.4f}")

    # --- Concat ---
    X_concat = np.hstack([sub_reps[k] for k in sorted(sub_reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)
    metrics = _compute_kernel_metrics(X_concat_s, sub_labels)
    results['concat'] = {'dim': X_concat.shape[1], **metrics}
    print(f"    {'concat':<20} dim={X_concat.shape[1]:>5}  "
          f"cond={metrics['log_condition']:>7.1f}  "
          f"k_cv={metrics['kernel_cv']:.4f}  "
          f"align={metrics['kernel_target_alignment']:.4f}")

    # --- Per-budget hybrid ---
    print(f"\n    Budget sweep (hybrid kernel):")
    for budget in sorted(budgets):
        try:
            X_hybrid, _, _search = get_fitted_hybrid(
                reps, labels, groups, budget=budget,
                aug_condition='none', speed='fast')
            # Subsample to match
            X_hybrid_sub = X_hybrid[idx] if n_samples > max_n else X_hybrid
            X_hybrid_s = StandardScaler().fit_transform(X_hybrid_sub)
            metrics = _compute_kernel_metrics(X_hybrid_s, sub_labels)
            results['per_budget'][budget] = {'dim': X_hybrid.shape[1], **metrics}
            print(f"      budget={budget:>4}  dim={X_hybrid.shape[1]:>4}  "
                  f"cond={metrics['log_condition']:>7.1f}  "
                  f"k_cv={metrics['kernel_cv']:.4f}  "
                  f"align={metrics['kernel_target_alignment']:.4f}")
        except Exception as e:
            print(f"      budget={budget}: FAILED - {e}")

    # --- RBF length-scale sensitivity (Moesser 2023 Ch.2, p.84) ---
    # Moesser found kernel length-scale critically affects GP/BO performance.
    # Sweep gamma across representative spaces to show that compact hybrid
    # representations are more robust to gamma choice.
    print(f"\n    Length-scale sensitivity (gamma sweep):")
    gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Pick representative spaces: highest-dim rep, concat, hybrid b50
    sweep_spaces = {}
    if results['per_rep']:
        highest_rep = max(results['per_rep'], key=lambda k: results['per_rep'][k]['dim'])
        sweep_spaces[highest_rep] = StandardScaler().fit_transform(
            sub_reps[highest_rep])
    sweep_spaces['concat'] = X_concat_s
    if 50 in results['per_budget']:
        try:
            X_h50, _, _search = get_fitted_hybrid(
                reps, labels, groups, budget=50,
                aug_condition='none', speed='fast')
            X_h50_sub = X_h50[idx] if n_samples > max_n else X_h50
            sweep_spaces['hybrid_b50'] = StandardScaler().fit_transform(X_h50_sub)
        except Exception:
            pass

    results['gamma_sweep'] = {}
    for space_name, X_s in sweep_spaces.items():
        dim = X_s.shape[1]
        sweep_row = {'dim': dim, 'gammas': {}}
        for g in gamma_values:
            m = _compute_kernel_metrics(X_s, sub_labels, gamma=g)
            sweep_row['gammas'][g] = {
                'log_condition': m['log_condition'],
                'kernel_cv': m['kernel_cv'],
                'kernel_target_alignment': m['kernel_target_alignment'],
            }
        results['gamma_sweep'][space_name] = sweep_row

    # Print sweep table
    print(f"    {'Space':<16} {'Dim':>5} | " +
          "  ".join(f"g={g:<7}" for g in gamma_values))
    print(f"    {'-'*16} {'-'*5}-+-" + "--".join("-" * 9 for _ in gamma_values))
    for space_name, sr in results['gamma_sweep'].items():
        conds = [sr['gammas'][g]['log_condition'] for g in gamma_values]
        cond_range = max(conds) - min(conds) if conds else 0
        line = f"    {space_name:<16} {sr['dim']:>5} | "
        line += "  ".join(f"{c:>9.1f}" for c in conds)
        line += f"  range={cond_range:.1f}"
        print(line)

    # Key insight: hybrid should have smallest condition range (most robust)
    if len(results['gamma_sweep']) >= 2:
        ranges = {}
        for sn, sr in results['gamma_sweep'].items():
            conds = [sr['gammas'][g]['log_condition'] for g in gamma_values]
            ranges[sn] = max(conds) - min(conds) if conds else float('inf')
        most_robust = min(ranges, key=ranges.get)
        print(f"    Most robust to gamma: {most_robust} "
              f"(logCond range={ranges[most_robust]:.1f})")

    # --- Summary table ---
    print(f"\n    Summary:")
    print(f"    {'Space':<20} {'Dim':>5} {'logCond':>8} {'K_cv':>8} "
          f"{'K_mean':>8} {'K_std':>8} {'Align':>8} {'EigRank':>8}")
    print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    all_items = list(results['per_rep'].items())
    all_items.append(('concat', results['concat']))
    for budget in sorted(budgets):
        if budget in results['per_budget']:
            all_items.append((f'hybrid_b{budget}', results['per_budget'][budget]))

    for name, m in all_items:
        print(f"    {name:<20} {m['dim']:>5} {m['log_condition']:>8.1f} "
              f"{m['kernel_cv']:>8.4f} {m['kernel_mean']:>8.4f} "
              f"{m['kernel_std']:>8.4f} {m['kernel_target_alignment']:>8.4f} "
              f"{m['kernel_effective_rank']:>8.1f}")

    return results


def _compute_kernel_metrics(X, y, gamma=None):
    """Compute RBF kernel matrix metrics.

    Uses the default sklearn gamma = 1 / (n_features * var(X)).
    """
    n, d = X.shape

    # Auto gamma (sklearn default for RBF)
    if gamma is None:
        var = np.var(X)
        gamma = 1.0 / (d * var) if var > 0 else 1.0

    # Compute RBF kernel matrix
    sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-gamma * sq_dists)

    # Off-diagonal entries only
    mask = ~np.eye(n, dtype=bool)
    k_off = K[mask]

    k_mean = float(np.mean(k_off))
    k_std = float(np.std(k_off))
    k_min = float(np.min(k_off))
    k_max = float(np.max(k_off))
    k_cv = k_std / k_mean if k_mean > 0 else 0

    # Condition number
    eigvals = np.linalg.eigvalsh(K)
    eigvals_pos = eigvals[eigvals > 1e-12]
    if len(eigvals_pos) > 0:
        cond = eigvals_pos[-1] / eigvals_pos[0]
        log_cond = float(np.log10(cond)) if cond > 0 else float('inf')
    else:
        log_cond = float('inf')

    # Effective rank of kernel matrix
    eigvals_norm = eigvals_pos / eigvals_pos.sum()
    k_erank = float(np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm + 1e-20))))

    # Participation ratio
    part_ratio = float((eigvals_pos.sum()) ** 2 / (eigvals_pos ** 2).sum())

    # Kernel-target alignment (Cristianini et al. 2001)
    # alignment = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    y_centered = y - y.mean()
    yy = np.outer(y_centered, y_centered)
    alignment = np.sum(K * yy) / (np.linalg.norm(K, 'fro') * np.linalg.norm(yy, 'fro') + 1e-10)

    return {
        'kernel_mean': k_mean,
        'kernel_std': k_std,
        'kernel_min': k_min,
        'kernel_max': k_max,
        'kernel_cv': float(k_cv),
        'log_condition': log_cond,
        'kernel_effective_rank': k_erank,
        'kernel_participation_ratio': part_ratio,
        'kernel_target_alignment': float(alignment),
        'gamma': float(gamma),
    }


# =========================================================================
# 15c: Double Descent Probe
# =========================================================================

def _run_double_descent(data, config):
    """15c: Probe for double descent near the interpolation threshold.

    Classical double descent: test error peaks when p ≈ n (number of features
    ≈ number of samples), then decreases again in the overparameterized regime.

    For KIRBy hybrids, p = budget. If budget << n always, the hybrid never
    enters the interpolation regime → no double descent → stable performance.

    Sweep budget through fine granularity around n_samples to detect any
    instability. Compare hybrid vs PCA vs random feature subsets.
    """
    print("\n  15c: Double Descent Probe")
    print("  " + "-" * 60)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    total_dim = X_concat.shape[1]

    # Fine-grained budget sweep: focus around n_samples
    # Include budgets well below and above n for context
    base_budgets = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    # Add budgets near n_samples for double descent detection
    n_region = [int(n_samples * f) for f in [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]]
    all_budgets = sorted(set(
        b for b in base_budgets + n_region
        if 5 <= b <= min(total_dim, n_samples * 3)
    ))

    print(f"    n_samples={n_samples}, total_dim={total_dim}")
    print(f"    Budgets: {all_budgets}")
    print(f"    Metric: {metric_name} ({'lower is better' if lower_is_better else 'higher is better'})")

    results = {
        'n_samples': n_samples,
        'total_dim': total_dim,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'budgets': [],
        'hybrid_scores': [],
        'pca_scores': [],
        'random_scores': [],
        'ridge_on_hybrid': [],
        'ridge_on_pca': [],
    }

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    for budget in all_budgets:
        n_feat = min(budget, total_dim, n_samples - 1)

        # --- Hybrid ---
        hybrid_score = _eval_at_budget(
            reps, labels, groups, n_folds, mode, metric_fn,
            budget=budget, method='hybrid'
        )

        # --- PCA ---
        pca_score = _eval_at_budget_pca(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_components=n_feat
        )

        # --- Random features ---
        random_score = _eval_at_budget_random(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_features=n_feat
        )

        # --- Ridge only (most sensitive to interpolation) ---
        ridge_hybrid = _eval_at_budget(
            reps, labels, groups, n_folds, mode, metric_fn,
            budget=budget, method='hybrid', model_name='ridge'
        )
        ridge_pca = _eval_at_budget_pca(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_components=n_feat, model_name='ridge'
        )

        results['budgets'].append(budget)
        results['hybrid_scores'].append(hybrid_score)
        results['pca_scores'].append(pca_score)
        results['random_scores'].append(random_score)
        results['ridge_on_hybrid'].append(ridge_hybrid)
        results['ridge_on_pca'].append(ridge_pca)

        ratio = budget / n_samples
        marker = " *** p ≈ n" if 0.8 <= ratio <= 1.2 else ""
        print(f"    budget={budget:>5} (p/n={ratio:.2f})  "
              f"hybrid={hybrid_score:>7.4f}  "
              f"pca={pca_score:>7.4f}  "
              f"random={random_score:>7.4f}  "
              f"ridge_hyb={ridge_hybrid:>7.4f}  "
              f"ridge_pca={ridge_pca:>7.4f}{marker}")

    # Detect instability near p ≈ n
    scores_arr = np.array(results['pca_scores'])
    budget_arr = np.array(results['budgets'])
    ratios = budget_arr / n_samples

    near_n_mask = (ratios >= 0.7) & (ratios <= 1.3)
    if np.sum(near_n_mask) >= 2:
        near_n_scores = scores_arr[near_n_mask]
        near_n_var = float(np.var(near_n_scores))
        far_scores = scores_arr[~near_n_mask]
        far_var = float(np.var(far_scores)) if len(far_scores) > 1 else 1e-10
        instability_ratio = near_n_var / (far_var + 1e-10)
        results['instability_ratio_pca'] = instability_ratio
        print(f"\n    PCA instability near p≈n: variance_ratio={instability_ratio:.2f} "
              f"({'UNSTABLE' if instability_ratio > 2 else 'stable'})")

    hybrid_arr = np.array(results['hybrid_scores'])
    if np.sum(near_n_mask) >= 2:
        near_n_hyb = hybrid_arr[near_n_mask]
        near_n_var_hyb = float(np.var(near_n_hyb))
        far_hyb = hybrid_arr[~near_n_mask]
        far_var_hyb = float(np.var(far_hyb)) if len(far_hyb) > 1 else 1e-10
        instability_hybrid = near_n_var_hyb / (far_var_hyb + 1e-10)
        results['instability_ratio_hybrid'] = instability_hybrid
        print(f"    Hybrid instability near p≈n: variance_ratio={instability_hybrid:.2f} "
              f"({'UNSTABLE' if instability_hybrid > 2 else 'stable'})")

    return results


def _eval_at_budget(reps, labels, groups, n_folds, mode, metric_fn,
                    budget, method='hybrid', model_name='rf'):
    """Evaluate a hybrid at a given budget with a given model."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    if model_name == 'ridge':
        make_model = lambda: Ridge(alpha=1.0)
    else:
        make_model = lambda: RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

    # Get blueprint from cached search
    search = get_precomputed_search(reps, labels, groups, budget=budget,
                                    aug_condition='none', speed='fast')
    bp = search.best_overall['blueprint']

    scores = []
    sample_X = next(iter(reps.values()))
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, X_val, feat_info = build_fold_hybrid(
                train_reps, val_reps_fold, y_train, bp)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = make_model()
            model.fit(X_train_s, y_train)
            scores.append(metric_fn(y_val, model.predict(X_val_s)))
        except Exception:
            continue

    return float(np.mean(scores)) if scores else float('nan')


def _eval_at_budget_pca(X_concat, labels, groups, n_folds, mode, metric_fn,
                        n_components, model_name='rf'):
    """Evaluate PCA-reduced features at a given dimensionality."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    if model_name == 'ridge':
        make_model = lambda: Ridge(alpha=1.0)
    else:
        make_model = lambda: RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels, mode=mode):
        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_concat[train_idx])
            X_val_s = scaler.transform(X_concat[val_idx])

            n_comp = min(n_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train_s)
            X_val_pca = pca.transform(X_val_s)

            model = make_model()
            model.fit(X_train_pca, labels[train_idx])
            scores.append(metric_fn(labels[val_idx], model.predict(X_val_pca)))
        except Exception:
            continue

    return float(np.mean(scores)) if scores else float('nan')


def _eval_at_budget_random(X_concat, labels, groups, n_folds, mode, metric_fn,
                           n_features, model_name='rf'):
    """Evaluate random feature subset at a given dimensionality."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    if model_name == 'ridge':
        make_model = lambda: Ridge(alpha=1.0)
    else:
        make_model = lambda: RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

    rng = np.random.RandomState(42)
    feat_idx = rng.choice(X_concat.shape[1], min(n_features, X_concat.shape[1]),
                          replace=False)
    X_sub = X_concat[:, feat_idx]

    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X_sub, labels, mode=mode):
        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_sub[train_idx])
            X_val_s = scaler.transform(X_sub[val_idx])

            model = make_model()
            model.fit(X_train_s, labels[train_idx])
            scores.append(metric_fn(labels[val_idx], model.predict(X_val_s)))
        except Exception:
            continue

    return float(np.mean(scores)) if scores else float('nan')


# =========================================================================
# 15d: Approximation Theory (Rademacher complexity)
# =========================================================================

def _run_approximation_theory(data, budgets, config):
    """15d: Empirical Rademacher complexity and generalization bounds.

    Rademacher complexity measures how well a hypothesis class can fit random
    labels. High Rademacher = model can memorize noise = poor generalization.

    For a linear model on d-dimensional data: R_n ∝ sqrt(d/n).
    Compressing from D to d should reduce Rademacher by sqrt(d/D).

    We estimate empirically: fit model on random labels, measure R².
    If hybrid compression reduces this R² → tighter generalization bound.
    """
    print("\n  15d: Approximation Theory (Rademacher Complexity)")
    print("  " + "-" * 60)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    results = {
        'n_samples': n_samples,
        'metric': metric_name,
        'budgets': [],
        'rademacher_hybrid': [],
        'rademacher_pca': [],
        'rademacher_full': None,
        'real_score_hybrid': [],
        'real_score_pca': [],
        'real_score_full': None,
    }

    n_random_trials = 5

    # --- Full-dimensional Rademacher ---
    # Use fresh RNG per condition for reproducibility across budgets
    full_rademacher = _estimate_rademacher(
        X_concat, labels, groups, n_folds, mode, n_random_trials,
        np.random.RandomState(42))
    results['rademacher_full'] = full_rademacher

    full_real = _eval_at_budget_pca(
        X_concat, labels, groups, n_folds, mode, metric_fn,
        n_components=min(X_concat.shape[1], n_samples - 1), model_name='ridge')
    results['real_score_full'] = full_real

    print(f"    Full dim ({X_concat.shape[1]}D): "
          f"Rademacher={full_rademacher:.4f}, real_{metric_name}={full_real:.4f}")

    # --- Per-budget ---
    for budget in sorted(budgets):
        n_feat = min(budget, X_concat.shape[1], n_samples - 1)

        # Hybrid — use per-fold creation to avoid label leakage
        # Rademacher: estimate on per-fold hybrid features (no leakage)
        rad_hybrid = _estimate_rademacher_hybrid(
            reps, labels, groups, n_folds, mode, n_random_trials,
            np.random.RandomState(42), budget)
        real_hybrid = _eval_at_budget(
            reps, labels, groups, n_folds, mode, metric_fn,
            budget=budget, method='hybrid', model_name='ridge')

        # PCA — fresh RNG per budget for fair comparison
        rad_pca = _estimate_rademacher_pca(
            X_concat, labels, groups, n_folds, mode, n_random_trials,
            np.random.RandomState(42), n_feat)
        real_pca = _eval_at_budget_pca(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_components=n_feat, model_name='ridge')

        results['budgets'].append(budget)
        results['rademacher_hybrid'].append(rad_hybrid)
        results['rademacher_pca'].append(rad_pca)
        results['real_score_hybrid'].append(real_hybrid)
        results['real_score_pca'].append(real_pca)

        ratio = budget / n_samples
        print(f"    budget={budget:>5} (p/n={ratio:.2f})  "
              f"Rad_hyb={rad_hybrid:.4f}  Rad_pca={rad_pca:.4f}  "
              f"Real_hyb={real_hybrid:.4f}  Real_pca={real_pca:.4f}")

    # Theoretical prediction: Rademacher ∝ sqrt(d/n)
    print(f"\n    Theoretical sqrt(d/n) scaling:")
    for budget in sorted(budgets):
        theoretical = np.sqrt(min(budget, X_concat.shape[1]) / n_samples)
        print(f"      budget={budget:>5}: sqrt(d/n) = {theoretical:.4f}")

    return results


def _estimate_rademacher(X, y, groups, n_folds, mode, n_trials, rng):
    """Estimate empirical Rademacher complexity: fit Ridge on random labels.

    Uses REAL labels for cv_split (consistent fold assignment) but fits on
    permuted labels to measure how well the model can memorize noise.
    """
    from sklearn.linear_model import Ridge

    # Pre-compute fold assignments using real labels (consistent splitting)
    folds = list(cv_split(groups, n_folds, X, y, mode=mode))

    random_scores = []
    for t in range(n_trials):
        y_random = rng.permutation(y)
        scores = []
        for train_idx, val_idx in folds:
            try:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X[train_idx])
                model = Ridge(alpha=1.0)
                model.fit(X_train_s, y_random[train_idx])
                # Training R²: how well can we fit random labels?
                r2_train = r2_score(y_random[train_idx], model.predict(X_train_s))
                scores.append(max(0, r2_train))
            except Exception:
                continue
        if scores:
            random_scores.append(np.mean(scores))

    return float(np.mean(random_scores)) if random_scores else float('nan')


def _estimate_rademacher_hybrid(reps, y, groups, n_folds, mode, n_trials, rng,
                                budget):
    """Estimate Rademacher complexity of hybrid features WITHOUT label leakage.

    Creates the hybrid per-fold on training data only, then measures how well
    Ridge can fit random labels on those features.
    """
    from sklearn.linear_model import Ridge

    # Get blueprint from cached search using REAL labels
    search = get_precomputed_search(reps, y, groups, budget=budget,
                                    aug_condition='none', speed='fast')
    bp = search.best_overall['blueprint']

    sample_X = next(iter(reps.values()))
    folds = list(cv_split(groups, n_folds, sample_X, y, mode=mode))

    random_scores = []
    for t in range(n_trials):
        y_random = rng.permutation(y)
        scores = []
        for train_idx, val_idx in folds:
            try:
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                # Build hybrid using REAL train labels (supervised selection)
                # but evaluate memorization on RANDOM labels
                X_train_hyb, _, _fi = build_fold_hybrid(
                    train_reps, train_reps, y[train_idx], bp)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train_hyb)
                model = Ridge(alpha=1.0)
                model.fit(X_train_s, y_random[train_idx])
                r2_train = r2_score(y_random[train_idx], model.predict(X_train_s))
                scores.append(max(0, r2_train))
            except Exception:
                continue
        if scores:
            random_scores.append(np.mean(scores))

    return float(np.mean(random_scores)) if random_scores else float('nan')


def _estimate_rademacher_pca(X_concat, y, groups, n_folds, mode, n_trials, rng,
                             n_components):
    """Estimate Rademacher complexity of PCA-reduced features."""
    from sklearn.linear_model import Ridge

    folds = list(cv_split(groups, n_folds, X_concat, y, mode=mode))

    random_scores = []
    for t in range(n_trials):
        y_random = rng.permutation(y)
        scores = []
        for train_idx, val_idx in folds:
            try:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_concat[train_idx])
                n_comp = min(n_components, X_train_s.shape[0] - 1,
                             X_train_s.shape[1])
                pca = PCA(n_components=n_comp)
                X_train_pca = pca.fit_transform(X_train_s)
                model = Ridge(alpha=1.0)
                model.fit(X_train_pca, y_random[train_idx])
                r2_train = r2_score(y_random[train_idx], model.predict(X_train_pca))
                scores.append(max(0, r2_train))
            except Exception:
                continue
        if scores:
            random_scores.append(np.mean(scores))

    return float(np.mean(random_scores)) if random_scores else float('nan')


# =========================================================================
# Main
# =========================================================================

def run(data, config, output_dir, dataset_name, sub_analyses=None, **kwargs):
    """C15: ML Theory Depth — 4 sub-analyses."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C15: ML THEORY DEPTH")
    print("=" * 70)

    budgets = config['budgets']
    all_subs = sub_analyses or ['15a', '15b', '15c', '15d']
    results = {}

    if '15a' in all_subs:
        results['distance_concentration'] = _run_distance_concentration(data, budgets)

    if '15b' in all_subs:
        results['kernel_analysis'] = _run_kernel_analysis(data, budgets)

    if '15c' in all_subs:
        results['double_descent'] = _run_double_descent(data, config)

    if '15d' in all_subs:
        results['approximation_theory'] = _run_approximation_theory(
            data, budgets, config)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C15: ML Theory Depth',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--sub', type=str, nargs='+', default=None,
                        choices=['15a', '15b', '15c', '15d'],
                        help='Sub-analyses to run (default: all)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  sub_analyses=args.sub)
    save_experiment_results(results, args.output, 'c15_ml_theory_depth',
                            args.dataset)
