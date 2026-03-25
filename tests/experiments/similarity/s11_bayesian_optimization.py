#!/usr/bin/env python3
"""S11: Bayesian Optimization in Compact Hybrid Space.

Paper: Similarity
Conclusion: GP-BO in compact KIRBy hybrid space is more sample-efficient than
    in high-dimensional fingerprint space, because distance concentration makes
    the GP surrogate uninformative when dimensionality is high.

Core hypothesis: In high-D, pairwise distances concentrate (Beyer et al. 1999),
    so the RBF kernel values all converge to the same number. The GP posterior
    becomes flat (near-zero variance everywhere), Expected Improvement collapses,
    and BO degenerates to random search. Compact KIRBy hybrids avoid this.

Design: Simulate a library-screening BO loop.
    1. Hold out 80% as "unlabeled pool"
    2. Initialize with random 5% seed set
    3. Build representation (hybrid/PCA/raw) on seed set
    4. BO loop: fit GP -> compute EI -> select top-1 -> reveal label -> repeat
    5. Track best-found-so-far at each iteration

Conditions:
    1. ecfp4_raw     — ECFP4 (2048D) + RBF GP: shows RBF failure in high-D
    2. best_single   — Best individual rep by seed-set GP + RBF GP
    3. pca_50d       — PCA to 50D from concat + RBF GP: unsupervised compression
    4. hybrid_50d    — KIRBy hybrid (budget=50) + RBF GP: MAIN CONDITION
    5. random_50d    — Random Gaussian projection to 50D + RBF GP: null compression
    6. random_acq    — Random acquisition baseline (no GP)
    7. multi_obj_hybrid — (optional, --multi-objective) scalarized multi-objective
       BO in hybrid space: geometric mean of normalized label + QED drug-likeness.
       Per Moesser (2023) Ch. 2. Molecular datasets with SMILES only.
    8. multi_obj_random — random baseline on combined objective (for comparison)

Metrics:
    - Primary: AUC of best-found-so-far curve (area under cumulative max)
    - Secondary: iterations to find top-1% compound, iterations to find top-5%

References:
    - Beyer et al. (ICDT 1999) "When Is Nearest Neighbor Meaningful?"
    - Snoek et al. (NIPS 2012) "Practical Bayesian Optimization"
    - Griffiths & Hernandez-Lobato (2020) "Constrained BO in Molecular Space"
    - Moesser (2023) PhD thesis, Ch. 2 (Oxford) — multi-objective BO

Usage:
    python s11_bayesian_optimization.py --dataset logd --reps all --mode intermediary
    python s11_bayesian_optimization.py --dataset logd --reps all --max-iterations 200 --n-seeds 5
    python s11_bayesian_optimization.py --dataset caco2 --reps all --rebuild-interval 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, get_domain_metric,
    make_structured_output,
)

from kirby.hybrid import apply_feature_selection
from experiments.complexity.complexity_helpers import cached_create_hybrid as create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.


# =============================================================================
# EXPECTED IMPROVEMENT
# =============================================================================

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement.

    EI(x) = (mu(x) - y_best - xi) * Phi(z) + sigma(x) * phi(z)
    where z = (mu(x) - y_best - xi) / sigma(x)

    Args:
        mu: predicted means, shape (n,)
        sigma: predicted stds, shape (n,)
        y_best: best observed value so far (scalar)
        xi: exploration-exploitation tradeoff (small positive value)

    Returns:
        ei: expected improvement values, shape (n,)
    """
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    # Set EI to 0 where sigma is negligible (GP is certain)
    ei[sigma < 1e-8] = 0.0
    return ei


# =============================================================================
# GP FITTING WITH ERROR HANDLING
# =============================================================================

def fit_gp(X_train, y_train, optimize=True):
    """Fit a GP with RBF + WhiteKernel. Returns (gp, success).

    In high-D, the kernel matrix may become singular. We catch LinAlgError
    and return (None, False) — this IS the expected result showing GP failure.
    """
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2 if optimize else 0,
        alpha=1e-6,
        normalize_y=True,
        random_state=42,
    )
    try:
        gp.fit(X_train, y_train)
        return gp, True
    except np.linalg.LinAlgError:
        return None, False
    except ValueError:
        return None, False


# =============================================================================
# MULTI-OBJECTIVE HELPERS
# =============================================================================

def compute_qed_scores(smiles_list):
    """Compute QED drug-likeness scores from SMILES.

    Returns np.array of QED values in [0, 1].
    """
    from rdkit import Chem
    from rdkit.Chem import QED as QEDModule

    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            scores.append(QEDModule.qed(mol))
        else:
            scores.append(0.5)  # neutral default
    return np.array(scores, dtype=np.float64)


def compute_combined_objective(y, qed_scores, lower_is_better=False):
    """Geometric mean of normalized label and QED for multi-objective BO.

    Mirrors Moesser (2023) Ch. 2: geometric mean of pIC50 across targets,
    adapted here as geometric mean of (normalized property, QED).

    Both components normalized to [0.01, 1] before combining.
    Always returns values to be MAXIMIZED.
    """
    y_work = -y if lower_is_better else y.copy()
    y_min, y_max = y_work.min(), y_work.max()
    y_range = y_max - y_min
    if y_range < 1e-10:
        y_norm = np.full_like(y_work, 0.5)
    else:
        y_norm = (y_work - y_min) / y_range

    y_norm = np.clip(y_norm, 0.01, 1.0)
    qed_clamped = np.clip(qed_scores, 0.01, 1.0)

    return np.sqrt(y_norm * qed_clamped)


# =============================================================================
# SINGLE BO RUN
# =============================================================================

def run_bo_loop(X_labeled, y_labeled, X_pool, y_pool,
                max_iterations, refit_interval=10, rng=None):
    """Run a single BO loop in a given representation space.

    Args:
        X_labeled: initial labeled features, shape (n_seed, d)
        y_labeled: initial labeled targets, shape (n_seed,)
        X_pool: pool features, shape (n_pool, d)
        y_pool: pool targets (hidden, revealed one at a time)
        max_iterations: BO budget
        refit_interval: refit GP hyperparameters every N iterations

    Returns:
        best_so_far: list of best-found-so-far at each iteration
        gp_failures: number of GP fitting failures
    """
    if rng is None:
        rng = np.random.RandomState(42)
    X_lab = X_labeled.copy()
    y_lab = y_labeled.copy()
    pool_mask = np.ones(len(y_pool), dtype=bool)  # True = still in pool

    best_so_far = [float(np.max(y_lab))]
    gp_failures = 0
    gp = None
    last_optimize = -refit_interval  # force optimization on first fit

    for it in range(max_iterations):
        if pool_mask.sum() == 0:
            break

        # Decide whether to optimize hyperparameters
        optimize = (it - last_optimize) >= refit_interval
        if optimize:
            last_optimize = it

        # Fit GP on labeled set
        gp, success = fit_gp(X_lab, y_lab, optimize=optimize)

        if not success:
            gp_failures += 1
            # Fall back to random selection when GP fails
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[rng.randint(len(available))]
        else:
            # Predict on pool
            X_pool_available = X_pool[pool_mask]
            mu, sigma = gp.predict(X_pool_available, return_std=True)

            # Compute EI
            y_best = float(np.max(y_lab))
            ei = expected_improvement(mu, sigma, y_best)

            # Select top-1 by EI
            best_ei_idx = np.argmax(ei)
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[best_ei_idx]

        # "Reveal" label: move from pool to labeled set
        x_new = X_pool[selected_pool_idx:selected_pool_idx + 1]
        y_new = y_pool[selected_pool_idx]

        X_lab = np.vstack([X_lab, x_new])
        y_lab = np.append(y_lab, y_new)
        pool_mask[selected_pool_idx] = False

        best_so_far.append(float(max(best_so_far[-1], y_new)))

    return best_so_far, gp_failures


def run_random_loop(y_labeled, y_pool, max_iterations, rng):
    """Run random acquisition baseline (no GP).

    Args:
        y_labeled: initial labeled targets
        y_pool: pool targets (hidden, revealed one at a time)
        max_iterations: budget
        rng: numpy RandomState

    Returns:
        best_so_far: list of best-found-so-far at each iteration
    """
    y_lab = y_labeled.copy()
    pool_mask = np.ones(len(y_pool), dtype=bool)

    best_so_far = [float(np.max(y_lab))]

    for it in range(max_iterations):
        if pool_mask.sum() == 0:
            break

        available = np.where(pool_mask)[0]
        selected = rng.choice(available)

        y_new = y_pool[selected]
        y_lab = np.append(y_lab, y_new)
        pool_mask[selected] = False

        best_so_far.append(float(max(best_so_far[-1], y_new)))

    return best_so_far


# =============================================================================
# REPRESENTATION BUILDERS
# =============================================================================

def build_ecfp4_features(reps, seed_idx, pool_idx):
    """Use ECFP4 raw (high-D) features.

    Falls back to the highest-dimensional rep if ecfp4 is not available.
    """
    if 'ecfp4' in reps:
        X_all = reps['ecfp4']
    else:
        # Use highest-dimensional individual rep as surrogate
        name = max(reps.keys(), key=lambda k: reps[k].shape[1])
        X_all = reps[name]
        print(f"    (ecfp4 not found, using {name} with dim={X_all.shape[1]})")

    scaler = StandardScaler()
    X_seed = scaler.fit_transform(X_all[seed_idx])
    X_pool = scaler.transform(X_all[pool_idx])
    return X_seed, X_pool, X_all.shape[1]


def build_best_single_features(reps, labels, seed_idx, pool_idx):
    """Use the best individual rep (highest GP R2 on seed set)."""
    y_seed = labels[seed_idx]
    best_name = None
    best_score = -np.inf

    for name, X_rep in reps.items():
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_rep[seed_idx])
        # Quick GP fit on seed to assess quality
        gp, ok = fit_gp(X_s, y_seed, optimize=True)
        if ok:
            mu = gp.predict(X_s)
            from sklearn.metrics import r2_score
            score = r2_score(y_seed, mu)
            if score > best_score:
                best_score = score
                best_name = name

    if best_name is None:
        # GP failed on all reps -- fall back to lowest-dim rep
        best_name = min(reps.keys(), key=lambda k: reps[k].shape[1])

    print(f"    Best single rep: {best_name} (dim={reps[best_name].shape[1]}, "
          f"seed GP R2={best_score:.4f})")

    X_rep = reps[best_name]
    scaler = StandardScaler()
    X_seed = scaler.fit_transform(X_rep[seed_idx])
    X_pool = scaler.transform(X_rep[pool_idx])
    return X_seed, X_pool, X_rep.shape[1]


def build_pca_features(reps, seed_idx, pool_idx, n_components=50):
    """PCA to n_components from concatenated reps."""
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    scaler = StandardScaler()
    X_seed_s = scaler.fit_transform(X_concat[seed_idx])
    X_pool_s = scaler.transform(X_concat[pool_idx])

    n_comp = min(n_components, X_seed_s.shape[0] - 1, X_seed_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_seed_pca = pca.fit_transform(X_seed_s)
    X_pool_pca = pca.transform(X_pool_s)

    return X_seed_pca, X_pool_pca, n_comp


def build_hybrid_features(reps, labels, seed_idx, pool_idx, budget=50):
    """Build KIRBy hybrid on seed set and transform pool.

    Returns (X_seed, X_pool, actual_dim, feature_info).
    """
    seed_reps = {k: v[seed_idx] for k, v in reps.items()}
    pool_reps = {k: v[pool_idx] for k, v in reps.items()}
    y_seed = labels[seed_idx]

    X_seed, feature_info = create_hybrid(
        seed_reps, y_seed,
        allocation_method='greedy',
        importance_method='lightgbm_gain',
        budget=budget,
    )
    X_pool = apply_feature_selection(pool_reps, feature_info)

    scaler = StandardScaler()
    X_seed_s = scaler.fit_transform(X_seed)
    X_pool_s = scaler.transform(X_pool)

    return X_seed_s, X_pool_s, X_seed.shape[1], feature_info, scaler


def build_random_features(reps, seed_idx, pool_idx, n_components=50, rng=None):
    """Random Gaussian projection to n_components."""
    if rng is None:
        rng = np.random.RandomState(42)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    d = X_concat.shape[1]

    scaler = StandardScaler()
    X_seed_s = scaler.fit_transform(X_concat[seed_idx])
    X_pool_s = scaler.transform(X_concat[pool_idx])

    # Random projection matrix (Gaussian)
    n_comp = min(n_components, d)
    projection = rng.randn(d, n_comp) / np.sqrt(n_comp)

    X_seed_rp = X_seed_s @ projection
    X_pool_rp = X_pool_s @ projection

    return X_seed_rp, X_pool_rp, n_comp


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_bo_metrics(best_so_far_curves, y_all, budgets):
    """Compute summary metrics from best-so-far curves across seeds.

    Args:
        best_so_far_curves: list of lists (one per seed)
        y_all: all target values (used for top-K thresholds)
        budgets: list of budget checkpoints to report

    Returns:
        dict with AUC, iterations-to-top-1%, iterations-to-top-5%
    """
    y_sorted = np.sort(y_all)[::-1]
    top_1pct_threshold = y_sorted[max(0, int(len(y_sorted) * 0.01) - 1)]
    top_5pct_threshold = y_sorted[max(0, int(len(y_sorted) * 0.05) - 1)]

    metrics = {}

    for budget in budgets:
        auc_values = []
        iters_to_top1 = []
        iters_to_top5 = []

        for curve in best_so_far_curves:
            # Truncate curve to budget (+1 for seed)
            c = curve[:budget + 1]

            # AUC of best-so-far curve (trapezoidal)
            if len(c) > 1:
                auc = float(np.trapz(c, dx=1.0))
            else:
                auc = float(c[0]) if c else 0.0
            auc_values.append(auc)

            # Iterations to find top-1%
            found_1 = None
            for i, val in enumerate(c):
                if val >= top_1pct_threshold:
                    found_1 = i
                    break
            iters_to_top1.append(found_1)

            # Iterations to find top-5%
            found_5 = None
            for i, val in enumerate(c):
                if val >= top_5pct_threshold:
                    found_5 = i
                    break
            iters_to_top5.append(found_5)

        # Aggregate across seeds
        metrics[f'budget_{budget}'] = {
            'auc_mean': float(np.mean(auc_values)),
            'auc_std': float(np.std(auc_values)),
            'iters_to_top1pct_mean': float(np.nanmean([
                x for x in iters_to_top1 if x is not None
            ])) if any(x is not None for x in iters_to_top1) else None,
            'iters_to_top1pct_found': sum(1 for x in iters_to_top1 if x is not None),
            'iters_to_top5pct_mean': float(np.nanmean([
                x for x in iters_to_top5 if x is not None
            ])) if any(x is not None for x in iters_to_top5) else None,
            'iters_to_top5pct_found': sum(1 for x in iters_to_top5 if x is not None),
        }

    return metrics


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S11: Bayesian Optimization in Compact Hybrid Space."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S11: BAYESIAN OPTIMIZATION IN HYBRID SPACE")
    print("=" * 70)

    max_iterations = kwargs.get('max_iterations', 100)
    n_seeds = kwargs.get('n_seeds', 3)
    rebuild_interval = kwargs.get('rebuild_interval', 0)
    hybrid_budget = kwargs.get('hybrid_budget', 50)
    target_dim = kwargs.get('target_dim', 50)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    n_samples = len(labels)

    # Domain-aware metric for determining "lower_is_better"
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # For BO we always MAXIMIZE. If the task's primary metric is lower-is-better
    # (e.g., RMSE), negate targets so BO maximizes -RMSE.
    if lower_is_better:
        y_bo = -labels.copy()
        print(f"  Metric: {metric_name} (lower is better) -> negating for BO maximization")
    else:
        y_bo = labels.copy()
        print(f"  Metric: {metric_name} (higher is better) -> direct maximization")

    # Budget checkpoints for metric computation
    budget_checkpoints = [b for b in [25, 50, 100, 200] if b <= max_iterations]
    if max_iterations not in budget_checkpoints:
        budget_checkpoints.append(max_iterations)

    print(f"  n_samples: {n_samples}")
    print(f"  n_reps: {len(reps)} ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")
    print(f"  max_iterations: {max_iterations}")
    print(f"  n_seeds: {n_seeds}")
    print(f"  rebuild_interval: {rebuild_interval}")
    print(f"  hybrid_budget: {hybrid_budget}")
    print(f"  budget_checkpoints: {budget_checkpoints}")

    # Top-K thresholds (on BO-oriented y)
    y_sorted = np.sort(y_bo)[::-1]
    top_1pct = y_sorted[max(0, int(len(y_sorted) * 0.01) - 1)]
    top_5pct = y_sorted[max(0, int(len(y_sorted) * 0.05) - 1)]
    print(f"  top-1% threshold: {top_1pct:.4f}")
    print(f"  top-5% threshold: {top_5pct:.4f}")

    # Multi-objective: combine primary label with QED (molecular only)
    multi_objective = kwargs.get('multi_objective', False)
    smiles = data.get('smiles', None)
    y_multi = None
    if multi_objective:
        if smiles is not None:
            try:
                qed_scores = compute_qed_scores(smiles)
                y_multi = compute_combined_objective(
                    labels, qed_scores, lower_is_better)
                print(f"  Multi-objective: label + QED geometric mean")
                print(f"    QED range: [{qed_scores.min():.3f}, "
                      f"{qed_scores.max():.3f}]")
                print(f"    Combined range: [{y_multi.min():.3f}, "
                      f"{y_multi.max():.3f}]")
            except Exception as e:
                print(f"  Multi-objective failed ({e}), disabled")
                multi_objective = False
        else:
            print(f"  Multi-objective skipped (no SMILES)")
            multi_objective = False

    conditions = [
        'ecfp4_raw',
        'best_single',
        'pca_50d',
        'hybrid_50d',
        'hybrid_50d_refit',
        'random_50d',
        'random_acq',
    ]
    if multi_objective and y_multi is not None:
        conditions.extend(['multi_obj_hybrid', 'multi_obj_random'])

    # Per-rep BO: run BO in each individual rep's space (Moesser Ch.2 comparison)
    per_rep_bo = kwargs.get('per_rep_bo', False)
    if per_rep_bo:
        for rname in sorted(reps.keys()):
            conditions.append(f'single_{rname}')
        print(f"  Per-rep BO: {len(reps)} individual rep conditions added")

    results = {
        'conditions': {},
        'config': {
            'max_iterations': max_iterations,
            'n_seeds': n_seeds,
            'rebuild_interval': rebuild_interval,
            'hybrid_budget': hybrid_budget,
            'target_dim': target_dim,
            'budget_checkpoints': budget_checkpoints,
            'lower_is_better': lower_is_better,
            'metric_name': metric_name,
            'n_samples': n_samples,
        },
    }

    for condition in conditions:
        print(f"\n  {'='*60}")
        print(f"  Condition: {condition}")
        print(f"  {'='*60}")

        all_curves = []
        total_gp_failures = 0
        cond_dim = target_dim  # default; overwritten by feature builders

        for seed_i in range(n_seeds):
            seed = 42 + seed_i
            rng = np.random.RandomState(seed)
            print(f"\n    Seed {seed_i + 1}/{n_seeds} (random_state={seed})")

            # --- Split: 5% seed, 80% pool, 15% unused (to keep pool at ~80%) ---
            # Respect groups if available
            seed_frac = 0.05
            pool_frac = 0.80
            # Total used = seed + pool = 85%, held out = 15%

            all_idx = np.arange(n_samples)
            if groups is not None and len(np.unique(groups)) >= 4:
                # Use GroupShuffleSplit: first split off seed set
                gss_seed = GroupShuffleSplit(
                    n_splits=1, test_size=1.0 - seed_frac, random_state=seed)
                seed_idx, rest_idx = next(gss_seed.split(all_idx, y_bo, groups))

                # From rest, take pool_frac / (1 - seed_frac) as pool
                pool_from_rest = pool_frac / (1.0 - seed_frac)
                pool_from_rest = min(pool_from_rest, 0.95)  # safety
                gss_pool = GroupShuffleSplit(
                    n_splits=1, test_size=1.0 - pool_from_rest, random_state=seed)
                rest_groups = groups[rest_idx]
                pool_local_idx, _ = next(gss_pool.split(
                    rest_idx, y_bo[rest_idx], rest_groups))
                pool_idx = rest_idx[pool_local_idx]
            else:
                # No groups: simple shuffle
                perm = rng.permutation(n_samples)
                n_seed = max(10, int(n_samples * seed_frac))
                n_pool = int(n_samples * pool_frac)
                seed_idx = perm[:n_seed]
                pool_idx = perm[n_seed:n_seed + n_pool]

            # Select target: combined objective for multi_obj, else standard
            y_cond = y_multi if condition.startswith('multi_obj_') else y_bo
            y_seed = y_cond[seed_idx]
            y_pool = y_cond[pool_idx]

            print(f"      seed: {len(seed_idx)}, pool: {len(pool_idx)}")

            # --- Build features for this condition ---
            try:
                if condition == 'ecfp4_raw':
                    X_seed, X_pool, dim = build_ecfp4_features(
                        reps, seed_idx, pool_idx)

                elif condition == 'best_single':
                    X_seed, X_pool, dim = build_best_single_features(
                        reps, y_bo, seed_idx, pool_idx)

                elif condition == 'pca_50d':
                    X_seed, X_pool, dim = build_pca_features(
                        reps, seed_idx, pool_idx, n_components=target_dim)

                elif condition == 'hybrid_50d':
                    if rebuild_interval > 0:
                        # BO with periodic hybrid rebuilding
                        curve, gp_fail = _run_bo_with_rebuild(
                            reps, y_bo, seed_idx, pool_idx,
                            max_iterations, hybrid_budget, rebuild_interval)
                        all_curves.append(curve)
                        total_gp_failures += gp_fail
                        best_final = curve[-1] if curve else float('nan')
                        print(f"      final best: {best_final:.4f}, "
                              f"GP failures: {gp_fail}")
                        continue
                    else:
                        X_seed, X_pool, dim, _, _ = build_hybrid_features(
                            reps, y_bo, seed_idx, pool_idx,
                            budget=hybrid_budget)

                elif condition == 'hybrid_50d_refit':
                    # Always rebuild hybrid every 25 iterations
                    curve, gp_fail = _run_bo_with_rebuild(
                        reps, y_bo, seed_idx, pool_idx,
                        max_iterations, hybrid_budget, 25)
                    all_curves.append(curve)
                    total_gp_failures += gp_fail
                    best_final = curve[-1] if curve else float('nan')
                    print(f"      final best: {best_final:.4f}, "
                          f"GP failures: {gp_fail}")
                    continue

                elif condition == 'random_50d':
                    X_seed, X_pool, dim = build_random_features(
                        reps, seed_idx, pool_idx,
                        n_components=target_dim, rng=rng)

                elif condition == 'random_acq':
                    curve = run_random_loop(y_seed, y_pool, max_iterations, rng)
                    all_curves.append(curve)
                    best_final = curve[-1] if curve else float('nan')
                    print(f"      final best: {best_final:.4f} (random)")
                    continue

                elif condition == 'multi_obj_hybrid':
                    X_seed, X_pool, dim, _, _ = build_hybrid_features(
                        reps, y_cond, seed_idx, pool_idx,
                        budget=hybrid_budget)

                elif condition == 'multi_obj_random':
                    curve = run_random_loop(y_seed, y_pool, max_iterations, rng)
                    all_curves.append(curve)
                    best_final = curve[-1] if curve else float('nan')
                    print(f"      final best: {best_final:.4f} (random, multi-obj)")
                    continue

                elif condition.startswith('single_'):
                    # Per-rep BO: use a single rep's features (scaled)
                    rep_name = condition[len('single_'):]
                    if rep_name not in reps:
                        print(f"      rep '{rep_name}' not found, skipping")
                        continue
                    X_rep = reps[rep_name]
                    scaler = StandardScaler()
                    X_seed = scaler.fit_transform(X_rep[seed_idx])
                    X_pool = scaler.transform(X_rep[pool_idx])
                    dim = X_rep.shape[1]

                else:
                    print(f"      Unknown condition: {condition}")
                    continue

            except Exception as e:
                print(f"      Feature build FAILED: {e}")
                all_curves.append([float('nan')])
                continue

            cond_dim = dim
            print(f"      dim: {dim}")

            # --- Run BO loop ---
            curve, gp_fail = run_bo_loop(
                X_seed, y_seed, X_pool, y_pool,
                max_iterations=max_iterations,
                refit_interval=10, rng=rng,
            )
            all_curves.append(curve)
            total_gp_failures += gp_fail

            best_final = curve[-1] if curve else float('nan')
            print(f"      final best: {best_final:.4f}, GP failures: {gp_fail}")

        # --- Aggregate metrics across seeds ---
        if not all_curves:
            results['conditions'][condition] = {'error': 'no successful runs'}
            continue

        y_for_metrics = y_multi if condition.startswith('multi_obj_') else y_bo
        metrics = compute_bo_metrics(all_curves, y_for_metrics, budget_checkpoints)

        # Mean curve for plotting
        max_len = max(len(c) for c in all_curves)
        padded = []
        for c in all_curves:
            if len(c) < max_len:
                c_ext = c + [c[-1]] * (max_len - len(c))
            else:
                c_ext = c
            padded.append(c_ext)
        mean_curve = list(np.nanmean(padded, axis=0))
        std_curve = list(np.nanstd(padded, axis=0))

        results['conditions'][condition] = {
            'metrics': metrics,
            'mean_curve': mean_curve,
            'std_curve': std_curve,
            'n_curves': len(all_curves),
            'total_gp_failures': total_gp_failures,
            'dim': cond_dim,
        }

        # Print summary for this condition
        for bk in budget_checkpoints:
            key = f'budget_{bk}'
            if key in metrics:
                m = metrics[key]
                print(f"    Budget {bk}: AUC={m['auc_mean']:.1f}+/-{m['auc_std']:.1f}, "
                      f"top1%: {m['iters_to_top1pct_mean']}"
                      f"/{m['iters_to_top1pct_found']} found, "
                      f"top5%: {m['iters_to_top5pct_mean']}"
                      f"/{m['iters_to_top5pct_found']} found")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY: Best-found AUC at max budget")
    print(f"{'='*70}")

    max_budget_key = f'budget_{max(budget_checkpoints)}'
    summary_rows = []
    for cond in conditions:
        if cond in results['conditions'] and 'metrics' in results['conditions'][cond]:
            m = results['conditions'][cond]['metrics'].get(max_budget_key, {})
            auc = m.get('auc_mean', float('nan'))
            auc_std = m.get('auc_std', float('nan'))
            gp_fail = results['conditions'][cond].get('total_gp_failures', 0)
            dim = results['conditions'][cond].get('dim', '?')
            summary_rows.append((cond, dim, auc, auc_std, gp_fail))

    print(f"  {'Condition':<16} {'Dim':>5} {'AUC':>10} {'Std':>8} {'GP Fail':>8}")
    print(f"  {'-'*16} {'-'*5} {'-'*10} {'-'*8} {'-'*8}")
    for cond, dim, auc, auc_std, gp_fail in summary_rows:
        print(f"  {cond:<16} {dim:>5} {auc:>10.1f} {auc_std:>8.1f} {gp_fail:>8}")

    # Highlight key comparison: hybrid vs ecfp4
    hybrid_res = results['conditions'].get('hybrid_50d', {}).get('metrics', {})
    ecfp4_res = results['conditions'].get('ecfp4_raw', {}).get('metrics', {})
    if hybrid_res and ecfp4_res:
        h_auc = hybrid_res.get(max_budget_key, {}).get('auc_mean', float('nan'))
        e_auc = ecfp4_res.get(max_budget_key, {}).get('auc_mean', float('nan'))
        if not (np.isnan(h_auc) or np.isnan(e_auc)):
            pct_gain = 100.0 * (h_auc - e_auc) / abs(e_auc) if e_auc != 0 else 0
            print(f"\n  Hybrid vs ECFP4: AUC gain = {h_auc - e_auc:+.1f} ({pct_gain:+.1f}%)")

    # Build structured output
    output = make_structured_output(
        's11_bayesian_optimization', dataset_name, data, config,
        results, primary_metric='bo_auc', lower_is_better=False,
    )
    return output


# =============================================================================
# HYBRID REBUILD BO
# =============================================================================

def _run_bo_with_rebuild(reps, y_bo, seed_idx, pool_idx,
                         max_iterations, hybrid_budget, rebuild_interval):
    """BO loop with periodic hybrid rebuilding.

    Every rebuild_interval iterations, re-create the hybrid using all
    labeled data accumulated so far, and re-transform the remaining pool.
    """
    y_seed = y_bo[seed_idx]
    y_pool = y_bo[pool_idx]

    # Initial hybrid build
    seed_reps = {k: v[seed_idx] for k, v in reps.items()}
    pool_reps_all = {k: v[pool_idx] for k, v in reps.items()}

    X_seed, feature_info = create_hybrid(
        seed_reps, y_seed,
        allocation_method='greedy',
        importance_method='lightgbm_gain',
        budget=hybrid_budget,
    )
    X_pool_all = apply_feature_selection(pool_reps_all, feature_info)

    scaler = StandardScaler()
    X_lab = scaler.fit_transform(X_seed)
    X_pool_transformed = scaler.transform(X_pool_all)

    y_lab = y_seed.copy()
    pool_mask = np.ones(len(y_pool), dtype=bool)
    labeled_original_idx = list(seed_idx)  # track original indices

    best_so_far = [float(np.max(y_lab))]
    gp_failures = 0
    last_optimize = -10

    for it in range(max_iterations):
        if pool_mask.sum() == 0:
            break

        # Rebuild hybrid if interval is set
        if rebuild_interval > 0 and it > 0 and it % rebuild_interval == 0:
            try:
                lab_idx_arr = np.array(labeled_original_idx)
                lab_reps = {k: v[lab_idx_arr] for k, v in reps.items()}
                remaining_pool_orig = pool_idx[pool_mask]
                remaining_reps = {k: v[remaining_pool_orig] for k, v in reps.items()}

                X_lab_new, feature_info = create_hybrid(
                    lab_reps, y_lab,
                    allocation_method='greedy',
                    importance_method='lightgbm_gain',
                    budget=hybrid_budget,
                )
                X_remaining = apply_feature_selection(remaining_reps, feature_info)

                scaler = StandardScaler()
                X_lab = scaler.fit_transform(X_lab_new)

                # Rebuild pool transform for remaining points
                X_pool_transformed = np.full(
                    (len(y_pool), X_remaining.shape[1]), np.nan)
                X_pool_transformed[pool_mask] = scaler.transform(X_remaining)

                print(f"        [rebuild at iter {it}, "
                      f"labeled={len(y_lab)}, pool={pool_mask.sum()}]")
            except Exception as e:
                print(f"        [rebuild failed at iter {it}: {e}]")

        # Fit GP
        optimize = (it - last_optimize) >= 10
        if optimize:
            last_optimize = it

        gp, success = fit_gp(X_lab, y_lab, optimize=optimize)

        if not success:
            gp_failures += 1
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[rng.randint(len(available))]
        else:
            X_pool_available = X_pool_transformed[pool_mask]
            mu, sigma = gp.predict(X_pool_available, return_std=True)
            y_best = float(np.max(y_lab))
            ei = expected_improvement(mu, sigma, y_best)
            best_ei_idx = np.argmax(ei)
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[best_ei_idx]

        # Reveal label
        x_new = X_pool_transformed[selected_pool_idx:selected_pool_idx + 1]
        y_new = y_pool[selected_pool_idx]

        X_lab = np.vstack([X_lab, x_new])
        y_lab = np.append(y_lab, y_new)
        pool_mask[selected_pool_idx] = False
        labeled_original_idx.append(pool_idx[selected_pool_idx])

        best_so_far.append(float(max(best_so_far[-1], y_new)))

    return best_so_far, gp_failures


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = create_experiment_parser(
        'S11: Bayesian Optimization in Compact Hybrid Space',
        has_delta=True, has_augment=True)
    parser.add_argument('--max-iterations', type=int, default=100,
                        help='Maximum BO iterations per run (default: 100)')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds to average (default: 3)')
    parser.add_argument('--rebuild-interval', type=int, default=0,
                        help='Rebuild hybrid every N iterations (0=never, default: 0)')
    parser.add_argument('--hybrid-budget', type=int, default=50,
                        help='KIRBy hybrid budget (default: 50)')
    parser.add_argument('--target-dim', type=int, default=50,
                        help='Target dimensionality for PCA/random (default: 50)')
    parser.add_argument('--multi-objective', action='store_true', default=False,
                        help='Enable multi-objective BO: optimize label + QED '
                             '(molecular datasets with SMILES only, per Moesser 2023)')
    parser.add_argument('--per-rep-bo', action='store_true', default=False,
                        help='Run BO in each individual rep space (shows which '
                             'reps give GP the best landscape, per Moesser Ch.2)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"N seeds: {args.n_seeds}")
    print(f"Rebuild interval: {args.rebuild_interval}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  max_iterations=args.max_iterations,
                  n_seeds=args.n_seeds,
                  rebuild_interval=args.rebuild_interval,
                  hybrid_budget=args.hybrid_budget,
                  target_dim=args.target_dim,
                  multi_objective=args.multi_objective,
                  per_rep_bo=args.per_rep_bo)
    save_experiment_results(results, args.output, 's11_bayesian_optimization',
                            args.dataset)
