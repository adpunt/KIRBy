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

from kirby.similarity.distance_config import scipy_metric as scipy_distance_metric
from scipy.stats import spearmanr, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from experiments._shared import make_nearest_neighbors  # honours --distance-metric

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    compute_effective_rank,
    get_fitted_hybrid, get_precomputed_search, build_fold_hybrid,
    get_best_for_downstream,
    get_best_search_for_downstream, get_fitted_pool_hybrid,
)


# =========================================================================
# 15a: Distance Concentration (Beyer et al. 1999)
# =========================================================================

def _run_distance_concentration(data, budgets, dataset_name=None):
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
    task_type = data['task_type']
    n_samples = len(labels)

    results = {'per_rep': {}, 'per_budget': {}}

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

    # X_concat kept as internal feature pool for PCA baselines / budget
    # sweeps below — NOT reported as a metric row (per design directive,
    # full-concat statistics aren't a meaningful baseline given how many
    # reps we have).
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    # --- Per-budget analysis (hybrid at each budget) ---
    print(f"\n    Budget sweep (hybrid):")
    for budget in sorted(budgets):
        try:
            X_hybrid, _, _search, _meta = get_fitted_pool_hybrid(
                reps, labels, groups, downstream_model='rf',
                budget=budget, augmentations=data.get('augmentations'),
                dataset_name=dataset_name, task_type=task_type)
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

    _m, _kw = scipy_distance_metric()
    dists = pdist(X_sub, metric=_m, **_kw)

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
    nn = make_nearest_neighbors(n_neighbors=k_use + 1, n_jobs=-1)
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

def _run_kernel_analysis(data, budgets, dataset_name=None):
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
    task_type = data['task_type']
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
    sub_groups = np.asarray(groups)[idx] if groups is not None else None
    results = {'per_rep': {}, 'per_budget': {}}

    # Kernel-target alignment is a score against the labels, and the hybrid's
    # features are chosen to fit those same labels while the representations
    # beside it never saw them. Measured over every row, the hybrid's number
    # counts the fit it was selected for. So a group-aware 80/20 split is cut
    # here: the hybrid is built on the training rows and every alignment,
    # hybrid and representation alike, is read on the held-out rows.
    # kernel_target_alignment stays in the table over every row, as
    # kernel_target_alignment_all_rows; the held-out one is what to read.
    _align_tr, _align_te = None, None
    try:
        if sub_groups is None:
            raise ValueError("no groups: a leak-free alignment split is "
                             "impossible, and there is no random fallback")
        _sample_X_15b = next(iter(sub_reps.values()))
        _align_tr, _align_te = next(iter(
            cv_split(sub_groups, 1, _sample_X_15b, sub_labels)))
        print(f"    Alignment held-out split: {len(_align_tr)} train rows, "
              f"{len(_align_te)} held-out rows (scaffold/identity groups)")
    except Exception as _e:
        print(f"    Held-out alignment unavailable ({_e}); the alignment "
              f"column is over every row and is not comparable between the "
              f"hybrid and the representations.")
    results['alignment_heldout_n'] = (int(len(_align_te))
                                      if _align_te is not None else 0)

    def _heldout_alignment(X_train, X_test):
        """Kernel-target alignment on the held-out rows.

        The scaler and the RBF length scale both come from the training rows,
        so nothing about the held-out rows shapes the kernel they are read in.
        """
        sc = StandardScaler().fit(X_train)
        Xtr_s, Xte_s = sc.transform(X_train), sc.transform(X_test)
        d = Xtr_s.shape[1]
        var = np.var(Xtr_s)
        gamma = 1.0 / (d * var) if var > 0 else 1.0
        m = _compute_kernel_metrics(Xte_s, sub_labels[_align_te], gamma=gamma)
        return float(m['kernel_target_alignment'])

    # --- Per-rep kernel analysis ---
    for rep_name, X in sorted(sub_reps.items()):
        X_s = StandardScaler().fit_transform(X)
        metrics = _compute_kernel_metrics(X_s, sub_labels)
        align_ho = float('nan')
        if _align_te is not None:
            try:
                align_ho = _heldout_alignment(X[_align_tr], X[_align_te])
            except Exception:
                pass
        row = {'dim': X.shape[1], **metrics}
        row['kernel_target_alignment_all_rows'] = row['kernel_target_alignment']
        row['kernel_target_alignment_heldout'] = align_ho
        results['per_rep'][rep_name] = row
        print(f"    {rep_name:<20} dim={X.shape[1]:>5}  "
              f"cond={metrics['log_condition']:>7.1f}  "
              f"k_cv={metrics['kernel_cv']:.4f}  "
              f"align_heldout={align_ho:.4f}  "
              f"align_all_rows={metrics['kernel_target_alignment']:.4f}")

    # X_concat kept only for the gamma sensitivity sweep below — not
    # reported as its own metric row (per design directive).
    X_concat = np.hstack([sub_reps[k] for k in sorted(sub_reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    # --- Per-budget hybrid ---
    print(f"\n    Budget sweep (hybrid kernel):")
    for budget in sorted(budgets):
        try:
            X_hybrid, _, _search, _meta = get_fitted_pool_hybrid(
                reps, labels, groups, downstream_model='rf',
                budget=budget, augmentations=data.get('augmentations'),
                dataset_name=dataset_name, task_type=task_type)
            # Subsample to match
            X_hybrid_sub = X_hybrid[idx] if n_samples > max_n else X_hybrid
            X_hybrid_s = StandardScaler().fit_transform(X_hybrid_sub)
            metrics = _compute_kernel_metrics(X_hybrid_s, sub_labels)

            # Held-out alignment: features are re-selected on the training
            # rows of the split above and read on the rows that did not choose
            # them. The recipe itself still came from a whole-dataset search,
            # which is a separate finding — recipe_selected_on records it.
            align_ho = float('nan')
            if _align_te is not None and _meta:
                try:
                    bp_15b = _meta['best_config']['blueprint']
                    _tr_reps = {k: v[_align_tr] for k, v in sub_reps.items()}
                    _te_reps = {k: v[_align_te] for k, v in sub_reps.items()}
                    _augs_15b = data.get('augmentations')
                    _tr_augs = _te_augs = None
                    if _augs_15b:
                        _sub_augs = {k: v[idx] for k, v in _augs_15b.items()}
                        _tr_augs = {k: v[_align_tr] for k, v in _sub_augs.items()}
                        _te_augs = {k: v[_align_te] for k, v in _sub_augs.items()}
                    X_h_tr, X_h_te, _ = build_fold_hybrid(
                        _tr_reps, _te_reps, sub_labels[_align_tr], bp_15b,
                        augmentations_train=_tr_augs,
                        augmentations_val=_te_augs,
                        groups=(sub_groups[_align_tr]
                                if sub_groups is not None else None),
                        budget_override=budget)
                    align_ho = _heldout_alignment(X_h_tr, X_h_te)
                except Exception as _be:
                    print(f"      budget={budget}: held-out alignment "
                          f"unavailable - {_be}")

            row = {'dim': X_hybrid.shape[1], **metrics}
            row['kernel_target_alignment_all_rows'] = row['kernel_target_alignment']
            row['kernel_target_alignment_heldout'] = align_ho
            row['recipe_selected_on'] = 'all rows'
            results['per_budget'][budget] = row
            print(f"      budget={budget:>4}  dim={X_hybrid.shape[1]:>4}  "
                  f"cond={metrics['log_condition']:>7.1f}  "
                  f"k_cv={metrics['kernel_cv']:.4f}  "
                  f"align_heldout={align_ho:.4f}  "
                  f"align_all_rows={metrics['kernel_target_alignment']:.4f}")
        except Exception as e:
            print(f"      budget={budget}: FAILED - {e}")

    # --- RBF length-scale sensitivity (Moesser 2023 Ch.2, p.84) ---
    # Moesser found kernel length-scale critically affects GP/BO performance.
    # Sweep gamma across representative spaces to show that compact hybrid
    # representations are more robust to gamma choice.
    print(f"\n    Length-scale sensitivity (gamma sweep):")
    gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Pick representative spaces: highest-dim rep, concat, hybrid b50.
    # sweep_splits carries each space's (training rows, held-out rows) pair so
    # the alignment at each gamma can also be read off rows that did not pick
    # the features. hybrid_b50's features ARE picked with the labels; the
    # representation beside it never sees them.
    sweep_spaces = {}
    sweep_splits = {}
    if results['per_rep']:
        highest_rep = max(results['per_rep'], key=lambda k: results['per_rep'][k]['dim'])
        sweep_spaces[highest_rep] = StandardScaler().fit_transform(
            sub_reps[highest_rep])
        if _align_te is not None:
            sweep_splits[highest_rep] = (sub_reps[highest_rep][_align_tr],
                                         sub_reps[highest_rep][_align_te])
    if 50 in results['per_budget']:
        try:
            X_h50, _, _search, _meta = get_fitted_pool_hybrid(
                reps, labels, groups, downstream_model='rf',
                budget=50, augmentations=data.get('augmentations'),
                dataset_name=dataset_name, task_type=task_type)
            X_h50_sub = X_h50[idx] if n_samples > max_n else X_h50
            sweep_spaces['hybrid_b50'] = StandardScaler().fit_transform(X_h50_sub)
            if _align_te is not None and _meta:
                bp50 = _meta['best_config']['blueprint']
                _tr_reps = {k: v[_align_tr] for k, v in sub_reps.items()}
                _te_reps = {k: v[_align_te] for k, v in sub_reps.items()}
                _augs50 = data.get('augmentations')
                _tr_a = _te_a = None
                if _augs50:
                    _sa = {k: v[idx] for k, v in _augs50.items()}
                    _tr_a = {k: v[_align_tr] for k, v in _sa.items()}
                    _te_a = {k: v[_align_te] for k, v in _sa.items()}
                Xh_tr, Xh_te, _ = build_fold_hybrid(
                    _tr_reps, _te_reps, sub_labels[_align_tr], bp50,
                    augmentations_train=_tr_a, augmentations_val=_te_a,
                    groups=(sub_groups[_align_tr]
                            if sub_groups is not None else None),
                    budget_override=50)
                sweep_splits['hybrid_b50'] = (Xh_tr, Xh_te)
        except Exception:
            pass

    results['gamma_sweep'] = {}
    for space_name, X_s in sweep_spaces.items():
        dim = X_s.shape[1]
        sweep_row = {'dim': dim, 'gammas': {}}
        _split = sweep_splits.get(space_name)
        if _split is not None:
            _sc_g = StandardScaler().fit(_split[0])
            _Xte_g = _sc_g.transform(_split[1])
        for g in gamma_values:
            m = _compute_kernel_metrics(X_s, sub_labels, gamma=g)
            align_ho_g = float('nan')
            if _split is not None:
                try:
                    m_ho = _compute_kernel_metrics(
                        _Xte_g, sub_labels[_align_te], gamma=g)
                    align_ho_g = float(m_ho['kernel_target_alignment'])
                except Exception:
                    pass
            sweep_row['gammas'][g] = {
                'log_condition': m['log_condition'],
                'kernel_cv': m['kernel_cv'],
                'kernel_target_alignment': m['kernel_target_alignment'],
                'kernel_target_alignment_all_rows': m['kernel_target_alignment'],
                'kernel_target_alignment_heldout': align_ho_g,
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
    # Squared Euclidean, and NOT --distance-metric: exp(-gamma * ||x-y||^2) is
    # what an RBF kernel is. Another distance here is a different kernel, not
    # the same kernel measured differently.
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

def _run_double_descent(data, config, dataset_name=None):
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
    task_type = data['task_type']
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    total_dim = X_concat.shape[1]

    # Each budget gets its own find_best_hybrid search (precompute only stored
    # canonical FULL_BUDGETS values — a pool lookup at 0.8n etc. would silently
    # return nothing, breaking the sweep).
    #
    # 2026-09-12, owner's decision: 15c's budget list becomes 0.25, 0.8, 0.9,
    # 1.0, 1.1, 1.2 and 4.0 times the row count. 0.25 and 4.0 are added, one at
    # each end; the five in the middle are what this file already swept.
    BUDGET_FACTORS = [0.25, 0.8, 0.9, 1.0, 1.1, 1.2, 4.0]
    all_budgets = sorted(set(
        int(n_samples * f) for f in BUDGET_FACTORS
    ))
    # Same decision, the ceiling it forces. The ceiling was
    # min(total_dim, 3 x row count, 2048); 4.0 times the row count cannot
    # survive either of the last two, so the budget the decision adds would
    # never have reached the sweep. The one ceiling kept is the number of
    # features there are to select from: a budget above total_dim asks for more
    # features than exist. The 3 x row count and the 2048 were compute caps,
    # and a budget is not cut for compute.
    all_budgets = [b for b in all_budgets if 5 <= b <= total_dim]

    print(f"    n_samples={n_samples}, total_dim={total_dim}")
    print(f"    Budgets: {all_budgets}")
    print(f"    Metric: {metric_name} ({'lower is better' if lower_is_better else 'higher is better'})")

    # Pre-fetch one search per budget. Both 'rf' and 'ridge' are
    # passed as downstream models so importance_cache is built once per
    # budget and reused across both models. Ridge stays here per
    # double-descent literature (Bartlett/Belkin) — see _eval_at_budget callers.
    #
    # 2026-09-12, owner's decision: every search c15 runs uses speed='fast'.
    # speed='fast' picks the short allocation, importance and internal-evaluator
    # lists inside find_best_hybrid. It does not touch the two downstream models
    # named below — an explicit models= dict is passed through whole — so 'ridge'
    # is still scored and _eval_at_budget still finds its blueprint.
    # Those three lists are in the search cache key, so a fast search files
    # separately from the full one; any c15c search already on disk is not
    # reused and is rebuilt fast.
    from kirby.find_best_hybrid import find_best_hybrid as _fbh
    from sklearn.ensemble import RandomForestRegressor as _RF15
    from sklearn.linear_model import Ridge as _Ridge15
    _augs = data.get('augmentations')
    _aug_strat = 'uncapped' if _augs else 'none'
    _searches_per_budget = {}
    for _b in all_budgets:
        try:
            _searches_per_budget[_b] = _fbh(
                base_reps=reps, labels=labels, groups=groups,
                models={
                    'rf': _RF15(n_estimators=100, max_depth=None,
                                random_state=42, n_jobs=-1),
                    'ridge': _Ridge15(alpha=1.0),
                },
                budgets=[_b],
                task_type='regression', metric=metric_name,
                cache=True,
                dataset_name=f"{dataset_name}__c15c_b{_b}",
                tier='c15c_fast',
                augmentations=_augs,
                augmentation_strategy=_aug_strat,
                speed='fast',
                verbose=False,
            )
        except Exception as _e:
            print(f"    c15c fast search at b={_b} failed: {_e}")

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
        hybrid_score, hybrid_w, _ = _eval_at_budget(
            reps, labels, groups, n_folds, mode, metric_fn,
            budget=budget, method='hybrid',
            data=data,
            dataset_name=dataset_name, task_type=task_type,
            prefetched_search=_searches_per_budget.get(budget),
        )

        # --- PCA ---
        pca_score, pca_w, _ = _eval_at_budget_pca(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_components=n_feat
        )

        # --- Random features ---
        random_score, random_w, _ = _eval_at_budget_random(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_features=n_feat
        )

        # --- Ridge only (most sensitive to interpolation) ---
        ridge_hybrid, ridge_hybrid_w, _ = _eval_at_budget(
            reps, labels, groups, n_folds, mode, metric_fn,
            budget=budget, method='hybrid', model_name='ridge',
            data=data,
            dataset_name=dataset_name, task_type=task_type,
            prefetched_search=_searches_per_budget.get(budget),
        )
        ridge_pca, ridge_pca_w, _ = _eval_at_budget_pca(
            X_concat, labels, groups, n_folds, mode, metric_fn,
            n_components=n_feat, model_name='ridge'
        )

        results['budgets'].append(budget)
        results['hybrid_scores'].append(hybrid_score)
        results['pca_scores'].append(pca_score)
        results['random_scores'].append(random_score)
        results['ridge_on_hybrid'].append(ridge_hybrid)
        results['ridge_on_pca'].append(ridge_pca)
        # The number of features each arm actually got, averaged over folds.
        # The budget is a maximum, not a target (hybrid.py:423), and the PCA
        # arm is capped at the training row count -- so the p/n printed below
        # comes from these, never from the budget.
        results.setdefault('realised_width_hybrid', []).append(hybrid_w)
        results.setdefault('realised_width_pca', []).append(pca_w)
        results.setdefault('realised_width_random', []).append(random_w)
        results.setdefault('realised_width_ridge_hybrid', []).append(ridge_hybrid_w)
        results.setdefault('realised_width_ridge_pca', []).append(ridge_pca_w)

        ratio = budget / n_samples
        marker = " *** p ≈ n" if 0.8 <= ratio <= 1.2 else ""
        print(f"    budget={budget:>5} (asked p/n={ratio:.2f})  "
              f"hybrid={hybrid_score:>7.4f} [{hybrid_w:.0f} feat, "
              f"built p/n={hybrid_w / n_samples:.2f}]  "
              f"pca={pca_score:>7.4f} [{pca_w:.0f} feat]  "
              f"random={random_score:>7.4f} [{random_w:.0f} feat]  "
              f"ridge_hyb={ridge_hybrid:>7.4f}  "
              f"ridge_pca={ridge_pca:>7.4f}{marker}")

    # Detect instability near p ≈ n
    scores_arr = np.array(results['pca_scores'])
    budget_arr = np.array(results['budgets'])
    ratios = budget_arr / n_samples

    near_n_mask = (ratios >= 0.7) & (ratios <= 1.3)
    far_mask = ~near_n_mask
    # The instability ratio is only meaningful when BOTH buckets have enough
    # points to have a real variance. If the "far" bucket has <2 points there
    # is no real denominator, and dividing by a fabricated 1e-10 floor
    # manufactures million-fold ratios that are pure artifact. In that case we
    # record nothing (leaving the metric blank) rather than emit garbage.
    far_ok = int(np.sum(far_mask)) >= 2
    if not far_ok:
        print("\n    Instability ratio not computable: <2 budgets far from p≈n "
              "(far bucket empty) — recording no instability_ratio for this run.")

    if np.sum(near_n_mask) >= 2 and far_ok:
        near_n_scores = scores_arr[near_n_mask]
        near_n_var = float(np.var(near_n_scores))
        far_scores = scores_arr[far_mask]
        far_var = float(np.var(far_scores))
        instability_ratio = near_n_var / (far_var + 1e-10)
        results['instability_ratio_pca'] = instability_ratio
        print(f"\n    PCA instability near p≈n: variance_ratio={instability_ratio:.2f} "
              f"({_stability_word(instability_ratio)})")

    hybrid_arr = np.array(results['hybrid_scores'])
    if np.sum(near_n_mask) >= 2 and far_ok:
        near_n_hyb = hybrid_arr[near_n_mask]
        near_n_var_hyb = float(np.var(near_n_hyb))
        far_hyb = hybrid_arr[far_mask]
        far_var_hyb = float(np.var(far_hyb))
        instability_hybrid = near_n_var_hyb / (far_var_hyb + 1e-10)
        results['instability_ratio_hybrid'] = instability_hybrid
        print(f"    Hybrid instability near p≈n: variance_ratio={instability_hybrid:.2f} "
              f"({_stability_word(instability_hybrid)})")

    return results


def _stability_word(ratio):
    """UNSTABLE / stable / not measured.

    A budget where every fold raised, or where no recipe existed, scores nan.
    np.var over a list holding a nan is nan, and `nan > 2` is False -- so the
    old inline conditional printed a budget that never ran as evidence of
    stability.
    """
    if ratio is None or not np.isfinite(ratio):
        return 'NOT MEASURED - a budget in this window scored nan'
    return 'UNSTABLE' if ratio > 2 else 'stable'


def _eval_at_budget(reps, labels, groups, n_folds, mode, metric_fn,
                    budget, method='hybrid', model_name='rf',
                    data=None, dataset_name=None, task_type=None,
                    prefetched_search=None):
    """Evaluate a hybrid at a given budget with a given model.

    If `prefetched_search` is provided, pull blueprint for `model_name`
    from that result (c15c fast-search path). Otherwise fall back to the
    pool helper (c15a/c15b/c15d use this path).

    REGRESSION-ONLY: the RF/Ridge regressors below are correct because
    run() gates the whole experiment to regression (see the regression
    guard + comment at the top of run()). Ridge is the double-descent
    interpolation probe (Bartlett/Belkin), not a sensitivity comparator.
    Never call this on classification data — clf metrics would be applied
    to continuous regressor outputs.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    if model_name == 'ridge':
        make_model = lambda: Ridge(alpha=1.0)
    else:
        make_model = lambda: RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

    if prefetched_search is not None:
        # c15c fast-search path: blueprint comes from the per-budget fast search.
        _best_by_model = getattr(prefetched_search, 'best_by_model', {}) or {}
        best = _best_by_model.get(model_name)
        if best is None:
            # Small-data case: the per-budget fast search produced no pool
            # candidate for this model (e.g. no result at budget=n). Degrade
            # gracefully (like c16/c02) — return nan so this budget drops out
            # of the double-descent curve instead of crashing run().
            print(f"    [c15c] no blueprint for model={model_name} at "
                  f"budget={budget}; recording nan (skipped)")
            return float('nan'), 0, 0
        bp = best['blueprint']
    else:
        # Pool helper: pick best (tier, aug_cond) cache file for this (model, budget).
        # Falls back to 'rf' if the requested model isn't in any pool candidate.
        augs = data.get('augmentations') if data is not None else None
        _search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=model_name, budget=budget,
            augmentations=augs, data=data,
            dataset_name=dataset_name, task_type=task_type,
        )
        if meta is None:
            _search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augs, data=data,
                dataset_name=dataset_name, task_type=task_type,
            )
        if meta is None:
            # Same small-data degeneracy as the prefetched path: no pool
            # candidate produced a result. Degrade gracefully (return nan)
            # so this budget drops out of the curve instead of crashing run().
            print(f"    [c15] no pool config for model={model_name} at "
                  f"budget={budget}; recording nan (skipped)")
            return float('nan'), 0, 0
        best = meta['best_config']
        bp = best['blueprint']

    scores, widths = [], []
    n_folds_tried = 0
    sample_X = next(iter(reps.values()))
    # Slice augmentations per-fold if present so build_fold_hybrid rebuilds
    # the same recipe the blueprint was scored under. Dropping augs here
    # would silently mismatch the score from the pool search.
    augs_data = data.get('augmentations') if data is not None else None
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        n_folds_tried += 1
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = ({k: v[train_idx] for k, v in augs_data.items()}
                      if augs_data else None)
        val_augs = ({k: v[val_idx] for k, v in augs_data.items()}
                    if augs_data else None)

        try:
            # budget_override: without it the rebuild uses bp.budget, which is
            # whatever width the looked-up recipe happened to carry -- and
            # get_best_for_downstream falls back to the best recipe across ALL
            # budgets when none exists at the one asked for. The whole point of
            # 15c/15d is a fine-grained sweep of p against n, so the loop's
            # budget has to govern the width, not the recipe's.
            X_train, X_val, feat_info = build_fold_hybrid(
                train_reps, val_reps_fold, y_train, bp,
                augmentations_train=train_augs,
                augmentations_val=val_augs,
                groups=(groups[train_idx] if groups is not None else None),
                budget_override=budget,
                model=make_model(),
                metric_fn=metric_fn)
            widths.append(int(X_train.shape[1]))

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = make_model()
            model.fit(X_train_s, y_train)
            scores.append(metric_fn(y_val, model.predict(X_val_s)))
        except Exception:
            continue

    if len(scores) < n_folds_tried:
        print(f"    [c15] budget={budget} model={model_name}: "
              f"{n_folds_tried - len(scores)} of {n_folds_tried} folds raised "
              f"and were dropped; the score below is the mean over "
              f"{len(scores)}.")
    return (float(np.mean(scores)) if scores else float('nan'),
            float(np.mean(widths)) if widths else 0.0,
            len(scores))


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

    scores, widths = [], []
    for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels, mode=mode):
        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_concat[train_idx])
            X_val_s = scaler.transform(X_concat[val_idx])

            # PCA cannot return more components than min(train rows, features).
            # At 5 folds that ceiling is 0.8 of the rows, so this arm CANNOT
            # reach p = n. The realised count is recorded so the curve is read
            # against what was built, not against what was asked for.
            n_comp = min(n_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
            widths.append(int(n_comp))
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train_s)
            X_val_pca = pca.transform(X_val_s)

            model = make_model()
            model.fit(X_train_pca, labels[train_idx])
            scores.append(metric_fn(labels[val_idx], model.predict(X_val_pca)))
        except Exception:
            continue

    return (float(np.mean(scores)) if scores else float('nan'),
            float(np.mean(widths)) if widths else 0.0,
            len(scores))


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
    width = int(X_sub.shape[1])
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

    return (float(np.mean(scores)) if scores else float('nan'),
            float(width), len(scores))


# =========================================================================
# 15d: Approximation Theory (Rademacher complexity)
# =========================================================================

def _run_approximation_theory(data, budgets, config, dataset_name=None):
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
    task_type = data['task_type']

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {
        'n_samples': n_samples,
        'metric': metric_name,
        'task_type': task_type,
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
        np.random.RandomState(42), task_type=task_type)
    results['rademacher_full'] = full_rademacher

    full_real, _full_w, _ = _eval_at_budget_pca(
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
            np.random.RandomState(42), budget, data=data,
            task_type=task_type, dataset_name=dataset_name)
        real_hybrid, _rh_w, _ = _eval_at_budget(
            reps, labels, groups, n_folds, mode, metric_fn,
            budget=budget, method='hybrid', model_name='ridge',
            data=data, dataset_name=dataset_name,
            task_type=task_type)

        # PCA — fresh RNG per budget for fair comparison
        rad_pca = _estimate_rademacher_pca(
            X_concat, labels, groups, n_folds, mode, n_random_trials,
            np.random.RandomState(42), n_feat, task_type=task_type)
        real_pca, _rp_w, _ = _eval_at_budget_pca(
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


def _fit_random_label_capacity(X_train_s, y_train_random, task_type):
    """Fit a linear model on random labels and return memorization score.

    Regression: Ridge + R² (fraction of variance in random continuous labels fit).
    Classification: RidgeClassifier + accuracy (fraction of random class labels fit).

    Both measure "how much noise can this feature geometry memorize", which is
    the empirical proxy for Rademacher complexity that scales with sqrt(d/n).
    """
    if task_type == 'classification':
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import accuracy_score
        model = RidgeClassifier(alpha=1.0)
        model.fit(X_train_s, y_train_random)
        return max(0.0, float(accuracy_score(y_train_random, model.predict(X_train_s))))
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train_random)
    return max(0.0, float(r2_score(y_train_random, model.predict(X_train_s))))


def _estimate_rademacher(X, y, groups, n_folds, mode, n_trials, rng,
                         task_type='regression'):
    """Estimate empirical Rademacher complexity: fit on random labels.

    Uses REAL labels for cv_split (consistent fold assignment) but fits on
    permuted labels to measure how well the model can memorize noise.
    """
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
                cap = _fit_random_label_capacity(
                    X_train_s, y_random[train_idx], task_type)
                scores.append(cap)
            except Exception:
                continue
        if scores:
            random_scores.append(np.mean(scores))

    return float(np.mean(random_scores)) if random_scores else float('nan')


def _estimate_rademacher_hybrid(reps, y, groups, n_folds, mode, n_trials, rng,
                                budget, data=None, task_type='regression',
                                dataset_name=None):
    """Estimate Rademacher complexity of hybrid features WITHOUT label leakage.

    Creates the hybrid per-fold on training data only, then measures how well
    a linear model can fit random labels on those features.
    """
    # Pool helper: best (tier, aug_cond) cache file for (ridge, budget),
    # falling back to rf if no ridge entry exists.
    augs = data.get('augmentations') if data is not None else None
    _search, meta = get_best_search_for_downstream(
        reps, y, groups,
        downstream_model='ridge', budget=budget,
        augmentations=augs, data=data,
        dataset_name=dataset_name, task_type=task_type,
    )
    if meta is None:
        _search, meta = get_best_search_for_downstream(
            reps, y, groups,
            downstream_model='rf', budget=budget,
            augmentations=augs, data=data,
            dataset_name=dataset_name, task_type=task_type,
        )
    if meta is None:
        return float('nan')
    bp = meta['best_config']['blueprint']

    sample_X = next(iter(reps.values()))
    folds = list(cv_split(groups, n_folds, sample_X, y, mode=mode))
    augs_rdm = data.get('augmentations') if data is not None else None

    # 2026-09-12, owner's decision — the dedupe already recorded. The hybrid
    # built here depends on the fold's training rows and on the blueprint. It
    # does not depend on the trial index: the trial index only draws a new label
    # permutation, and the permuted labels are never handed to
    # build_fold_hybrid — the real labels are, on the line below. So each fold's
    # features were rebuilt once per trial, 5 trials x 5 folds = 25 builds where
    # 5 differ, on each of 8 budgets: 160 identical rebuilds per dataset.
    # Build each fold once here, reuse it across the trials. The standardiser is
    # hoisted with it for the same reason. The numbers this returns do not
    # change: the permutation is still drawn once per trial in the same order,
    # and the Ridge fit below consumes no randomness.
    fold_features = []
    for train_idx, val_idx in folds:
        try:
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            train_augs = ({k: v[train_idx] for k, v in augs_rdm.items()}
                          if augs_rdm else None)
            # Build hybrid using REAL train labels (supervised selection)
            # but evaluate memorization on RANDOM labels. Augs sliced
            # per fold so the rebuild matches the scored blueprint.
            # budget_override for the same reason as _eval_at_budget: without
            # it the complexity below is measured on bp.budget features while
            # the score it is printed beside was measured on this budget's.
            X_train_hyb, _, _fi = build_fold_hybrid(
                train_reps, train_reps, y[train_idx], bp,
                augmentations_train=train_augs,
                augmentations_val=train_augs,
                budget_override=budget,
                groups=(groups[train_idx] if groups is not None else None))
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_hyb)
        except Exception:
            # A fold whose build raised was skipped in every trial before this
            # change too, so it is still skipped in every trial.
            continue
        fold_features.append((train_idx, X_train_s))

    random_scores = []
    for t in range(n_trials):
        y_random = rng.permutation(y)
        scores = []
        for train_idx, X_train_s in fold_features:
            try:
                cap = _fit_random_label_capacity(
                    X_train_s, y_random[train_idx], task_type)
                scores.append(cap)
            except Exception:
                continue
        if scores:
            random_scores.append(np.mean(scores))

    return float(np.mean(random_scores)) if random_scores else float('nan')


def _estimate_rademacher_pca(X_concat, y, groups, n_folds, mode, n_trials, rng,
                             n_components, task_type='regression'):
    """Estimate Rademacher complexity of PCA-reduced features."""
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
                cap = _fit_random_label_capacity(
                    X_train_pca, y_random[train_idx], task_type)
                scores.append(cap)
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

    task_type = data.get('task_type')
    if task_type != 'regression':
        # Sub-analyses 15c (_eval_at_budget*) and 15d hardcode RandomForestRegressor + Ridge.
        # Mirroring c04: gate the whole experiment to regression.
        print(f"  Skipping: c15 is regression-only (task_type={task_type}).")
        return {'skipped': True, 'reason': 'regression_only',
                'task_type': task_type}

    budgets = config['budgets']
    all_subs = sub_analyses or ['15a', '15b', '15c', '15d']
    results = {}

    if '15a' in all_subs:
        results['distance_concentration'] = _run_distance_concentration(
            data, budgets, dataset_name=dataset_name)

    if '15b' in all_subs:
        results['kernel_analysis'] = _run_kernel_analysis(
            data, budgets, dataset_name=dataset_name)

    if '15c' in all_subs:
        results['double_descent'] = _run_double_descent(
            data, config, dataset_name=dataset_name)

    if '15d' in all_subs:
        results['approximation_theory'] = _run_approximation_theory(
            data, budgets, config, dataset_name=dataset_name)

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
