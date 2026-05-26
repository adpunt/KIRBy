#!/usr/bin/env python3
"""C10: Noise Robustness & Feature Stability — NDS across noise strategies.

Paper: Complexity
Conclusion: Hybrids are more noise-robust than individual reps.
Dependencies: None (standalone).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, _cv_split,
    apply_feature_selection, get_domain_metric,
    get_best_search_for_downstream,
)
from kirby.hybrid import create_hybrid, apply_augmentation_selection

# Optional NoiseInject import
try:
    from noiseInject import NoiseInjectorRegression
    HAS_NOISEINJECT = True
except ImportError:
    HAS_NOISEINJECT = False


def _nogueira_stability(feature_sets, total_features):
    """Compute Nogueira et al. (2018) stability index from per-fold feature sets.

    Args:
        feature_sets: list of sets, each containing (rep_name, feature_idx) tuples.
        total_features: int, total number of features available.

    Returns:
        (phi, se): stability index in [-1, 1] and standard error.
        phi=1 means identical selection every fold.

    Reference:
        Nogueira, Sechidis & Brown (2018). "On the Stability of Feature
        Selection Algorithms." JMLR 18(174), 1-54.
    """
    M = len(feature_sets)
    if M < 2 or total_features == 0:
        return float('nan'), float('nan')

    # Build unified feature index from all selected features across folds
    all_features = set()
    for fs in feature_sets:
        all_features.update(fs)
    all_features = sorted(all_features)
    feat_to_idx = {f: i for i, f in enumerate(all_features)}
    d = len(all_features)

    if d == 0:
        return float('nan'), float('nan')

    # Binary matrix Z: M x d
    Z = np.zeros((M, d), dtype=float)
    for i, fs in enumerate(feature_sets):
        for f in fs:
            if f in feat_to_idx:
                Z[i, feat_to_idx[f]] = 1.0

    k_bar = Z.sum(axis=1).mean()
    if k_bar == 0 or k_bar == d:
        return float('nan'), float('nan')

    p_f = Z.mean(axis=0)
    numerator = (M / (M - 1)) * np.sum(p_f * (1 - p_f))
    denominator = d * k_bar * (1 - k_bar / d)

    if denominator == 0:
        return float('nan'), float('nan')

    phi = 1 - numerator / denominator
    var_terms = p_f * (1 - p_f)
    se = np.sqrt((4 / (M * d**2 * (k_bar * (1 - k_bar / d))**2)) *
                 np.sum(var_terms**2)) if M > 2 else float('nan')

    return float(phi), float(se)


def run(data, config, output_dir, dataset_name, sigmas=None, strategies=None):
    """Exp 10: Noise Degradation Slope (NDS) across noise strategies.

    Tests whether hybrids are more noise-robust than individual reps, and whether
    the optimal hybrid config changes under noise. Also tracks feature stability
    across noise levels (do the same features stay important when noise increases?).

    NDS = slope of R² vs sigma. More negative = more noise-sensitive.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: NOISE ROBUSTNESS & FEATURE STABILITY")
    print("=" * 70)

    if not HAS_NOISEINJECT:
        print("  NoiseInject not installed, skipping. Install: pip install noiseInject")
        return {'error': 'noiseInject not installed'}

    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  Skipping: c10 is regression-only "
              f"(task_type={task_type}). Noise injection mangles class labels.")
        return {'error': 'regression_only', 'task_type': task_type}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations') or {}
    budgets = config['budgets']
    n_folds = config['n_splits']

    # Precompute blueprints from cached search (clean labels). For each budget
    # the pool helper picks the best (tier, aug_condition) cache file for
    # downstream_model='rf', so different budgets can come from different files.
    blueprints = {}
    aug_conditions = {}
    try:
        for budget in budgets:
            search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augmentations or None,
                speed=config.get('search_speed', 'full'),
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
            if meta:
                blueprints[budget] = meta['best_config']['blueprint']
                # Track the actual aug_condition that produced this blueprint
                # (the blueprint's own augmentation_strategy field can be stale
                # due to the 2026-04-21 cache-collision bug).
                aug_conditions[budget] = meta.get('aug_condition', 'none')
                bp_model = meta.get('best_config', {}).get('model')
                if bp_model and bp_model != 'rf':
                    print(f"  WARNING: budget={budget} blueprint came from "
                          f"cross-model fallback ({bp_model}, not rf)")
                print(f"  Blueprint for budget={budget} "
                      f"(pool winner: {meta['tier']}/{meta['aug_condition']}): "
                      f"{blueprints[budget].allocation_method}, "
                      f"{blueprints[budget].importance_method}")
            else:
                print(f"  Blueprint for budget={budget}: no config found")
    except Exception as e:
        print(f"  Blueprint search failed: {e}")

    if sigmas is None:
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    if strategies is None:
        strategies = ['legacy', 'quantile', 'hetero']

    results = {
        'sigmas': sigmas,
        'strategies': strategies,
        'individual_reps': {},
        'hybrids': {},
        'feature_stability': {},
    }

    sample_X = next(iter(reps.values()))

    from sklearn.ensemble import RandomForestRegressor as _RF

    # --- Phase 1: Individual rep noise robustness ---
    print("\n  Phase 1: Individual rep NDS")
    for rep_name, X_rep in reps.items():
        print(f"    {rep_name} (dim={X_rep.shape[1]}):")
        rep_results = {}

        for strategy in strategies:
            sigma_scores = {}

            for sigma in sigmas:
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(
                        _cv_split(groups, n_folds, sample_X, labels)):
                    y_train = labels[train_idx].copy()
                    y_val = labels[val_idx]

                    # Inject noise into training labels only. Re-instantiate
                    # the injector per (sigma, fold) with a deterministic seed
                    # so the noise realization is i.i.d. across folds and
                    # repeatable across reps.
                    if sigma > 0:
                        _seed = 42 + fold_idx * 100 + int(sigma * 1000)
                        noise_injector = NoiseInjectorRegression(
                            strategy=strategy, random_state=_seed)
                        y_train = noise_injector.inject(y_train, sigma=sigma)

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_rep[train_idx])
                    X_va = scaler.transform(X_rep[val_idx])

                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, y_train)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va)))

                sigma_scores[sigma] = float(np.mean(fold_scores))

            # Compute NDS: slope of score vs sigma
            s_arr = np.array(list(sigma_scores.keys()))
            r2_arr = np.array(list(sigma_scores.values()))
            if len(s_arr) >= 2 and np.std(s_arr) > 0:
                slope, _ = np.polyfit(s_arr, r2_arr, 1)
                nds = float(slope)
            else:
                nds = 0.0

            baseline_r2 = sigma_scores.get(0.0, sigma_scores.get(min(sigmas)))
            max_noise_r2 = sigma_scores.get(max(sigmas), float('nan'))
            if baseline_r2 != 0 and not np.isnan(max_noise_r2):
                if lower_is_better:
                    # For lower-is-better (MAE/RMSE): retention = baseline / noisy * 100
                    # (lower noisy score = better retention)
                    retention = float(baseline_r2 / max_noise_r2 * 100)
                else:
                    # For higher-is-better (R²): retention = noisy / baseline * 100
                    retention = float(max_noise_r2 / baseline_r2 * 100)
            else:
                retention = 0.0

            rep_results[strategy] = {
                'scores_by_sigma': sigma_scores,
                'nds': nds,
                'baseline_r2': baseline_r2,
                'retention_pct': retention,
            }
            print(f"      {strategy}: NDS={nds:.4f}, base_R²={baseline_r2:.4f}, "
                  f"retention={retention:.1f}%")

        results['individual_reps'][rep_name] = rep_results

    # --- Phase 2: Hybrid noise robustness at each budget ---
    print("\n  Phase 2: Hybrid NDS by budget")

    for budget in budgets:
        print(f"\n    Budget={budget}:")
        budget_results = {}

        for strategy in strategies:
            sigma_scores = {}

            for sigma in sigmas:
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(
                        _cv_split(groups, n_folds, sample_X, labels)):
                    y_train = labels[train_idx].copy()
                    y_val = labels[val_idx]

                    if sigma > 0:
                        _seed = 42 + fold_idx * 100 + int(sigma * 1000)
                        noise_injector = NoiseInjectorRegression(
                            strategy=strategy, random_state=_seed)
                        y_train = noise_injector.inject(y_train, sigma=sigma)

                    train_reps = {k: v[train_idx] for k, v in reps.items()}
                    val_reps = {k: v[val_idx] for k, v in reps.items()}

                    train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                    val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                    try:
                        bp = blueprints[budget]
                        _aug_strat = aug_conditions.get(budget, bp.augmentation_strategy)
                        X_train, fi = create_hybrid(
                            train_reps, y_train,
                            allocation_method=bp.allocation_method,
                            budget=budget,  # pin to sweep axis (not bp.budget)
                            importance_method=bp.importance_method,
                            internal_evaluator=bp.internal_evaluator,
                            augmentations=train_augs,
                            augmentation_strategy=_aug_strat,
                        )
                        X_val = apply_feature_selection(val_reps, fi)
                        aug_info = fi.get('augmentation_info', {})
                        if aug_info and val_augs:
                            aug_val = apply_augmentation_selection(val_augs, aug_info)
                            if aug_val is not None:
                                X_val = np.hstack([X_val, aug_val])
                    except Exception:
                        continue

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_va = scaler.transform(X_val)

                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, y_train)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va)))

                if fold_scores:
                    sigma_scores[sigma] = float(np.mean(fold_scores))

            # NDS
            s_arr = np.array(list(sigma_scores.keys()))
            r2_arr = np.array(list(sigma_scores.values()))
            if len(s_arr) >= 2 and np.std(s_arr) > 0:
                slope, _ = np.polyfit(s_arr, r2_arr, 1)
                nds = float(slope)
            else:
                nds = 0.0

            baseline_r2 = sigma_scores.get(0.0, sigma_scores.get(min(sigmas), 0))
            max_noise_r2 = sigma_scores.get(max(sigmas), float('nan'))
            if baseline_r2 != 0 and not np.isnan(max_noise_r2):
                if lower_is_better:
                    retention = float(baseline_r2 / max_noise_r2 * 100)
                else:
                    retention = float(max_noise_r2 / baseline_r2 * 100)
            else:
                retention = 0.0

            budget_results[strategy] = {
                'scores_by_sigma': sigma_scores,
                'nds': nds,
                'baseline_r2': baseline_r2,
                'retention_pct': retention,
            }
            print(f"      {strategy}: NDS={nds:.4f}, base_R²={baseline_r2:.4f}, "
                  f"retention={retention:.1f}%")

        results['hybrids'][budget] = budget_results

    # --- Phase 2b: Noise × budget interaction ---
    # For each strategy, extract NDS per budget to show that lower budget = more noise-robust
    print("\n  Phase 2b: Noise × budget interaction")
    noise_budget = {}
    for strategy in strategies:
        budget_nds_pairs = []
        for budget in budgets:
            bdata = results['hybrids'].get(budget, {}).get(strategy)
            if bdata and not np.isnan(bdata.get('nds', float('nan'))):
                budget_nds_pairs.append((budget, bdata['nds']))
        if len(budget_nds_pairs) >= 2:
            b_arr = np.array([p[0] for p in budget_nds_pairs])
            nds_arr = np.array([p[1] for p in budget_nds_pairs])
            corr = float(np.corrcoef(b_arr, nds_arr)[0, 1]) if np.std(nds_arr) > 0 else 0.0
            noise_budget[strategy] = {
                'budget_nds': {int(b): float(n) for b, n in budget_nds_pairs},
                'correlation': corr,  # negative = higher budget → more negative NDS → less robust
            }
            print(f"    {strategy}: budget-NDS correlation={corr:.3f} "
                  f"({'higher budget = less robust' if corr < -0.3 else 'weak/no trend'})")
    results['noise_budget_interaction'] = noise_budget

    # --- Phase 3: Feature stability across noise levels ---
    print("\n  Phase 3: Feature stability across noise")
    budget_for_stability = budgets[len(budgets) // 2]  # middle budget
    print(f"    Using budget={budget_for_stability}")

    for strategy in strategies[:1]:  # just legacy for feature stability (expensive)
        features_by_sigma = {}

        for sigma in sigmas:
            fold_features = []
            for fold_idx, (train_idx, val_idx) in enumerate(
                    _cv_split(groups, n_folds, sample_X, labels)):
                y_train = labels[train_idx].copy()
                if sigma > 0:
                    _seed = 42 + fold_idx * 100 + int(sigma * 1000)
                    noise_injector = NoiseInjectorRegression(
                        strategy=strategy, random_state=_seed)
                    y_train = noise_injector.inject(y_train, sigma=sigma)

                train_reps = {k: v[train_idx] for k, v in reps.items()}
                train_augs_stab = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                try:
                    bp_stab = blueprints[budget_for_stability]
                    _aug_strat_stab = aug_conditions.get(
                        budget_for_stability, bp_stab.augmentation_strategy)
                    _, fi = create_hybrid(
                        train_reps, y_train,
                        allocation_method=bp_stab.allocation_method,
                        budget=budget_for_stability,
                        importance_method=bp_stab.importance_method,
                        internal_evaluator=bp_stab.internal_evaluator,
                        augmentations=train_augs_stab,
                        augmentation_strategy=_aug_strat_stab,
                    )
                    # Collect selected feature identifiers: (rep_name, feature_idx) tuples
                    selected = set()
                    for rep_name, rep_info in fi.items():
                        if isinstance(rep_info, dict) and 'selected_indices' in rep_info:
                            for idx in rep_info['selected_indices']:
                                selected.add((rep_name, int(idx)))
                    fold_features.append(selected)
                except Exception:
                    continue

            features_by_sigma[sigma] = fold_features

        # Feature stability across folds (at sigma=0)
        if features_by_sigma.get(0.0) and len(features_by_sigma[0.0]) >= 2:
            fold_sets = features_by_sigma[0.0]
            jaccard_across_folds = []
            for i in range(len(fold_sets)):
                for j in range(i + 1, len(fold_sets)):
                    inter = len(fold_sets[i] & fold_sets[j])
                    union = len(fold_sets[i] | fold_sets[j])
                    jaccard_across_folds.append(inter / union if union > 0 else 0)
            results['feature_stability']['fold_jaccard_sigma0'] = float(np.mean(jaccard_across_folds))
            print(f"    Fold stability (sigma=0): Jaccard={np.mean(jaccard_across_folds):.3f}")

            # Nogueira stability index (Nogueira, Sechidis & Brown, JMLR 2018)
            total_features = sum(v.shape[1] for v in reps.values())
            nogueira_phi, nogueira_se = _nogueira_stability(fold_sets, total_features)
            results['feature_stability']['nogueira_phi'] = nogueira_phi
            results['feature_stability']['nogueira_se'] = nogueira_se
            if not np.isnan(nogueira_phi):
                # SE can be NaN when M<=2 folds (Nogueira formula needs M>2);
                # writing NaN to JSON is not a valid value, so skip the CI.
                if not np.isnan(nogueira_se):
                    ci_lo = nogueira_phi - 1.96 * nogueira_se
                    ci_hi = nogueira_phi + 1.96 * nogueira_se
                    results['feature_stability']['nogueira_ci_95'] = [ci_lo, ci_hi]
                    print(f"    Nogueira stability (sigma=0): phi={nogueira_phi:.3f} "
                          f"(95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
                else:
                    print(f"    Nogueira stability (sigma=0): phi={nogueira_phi:.3f} "
                          f"(CI undefined: requires M>2 folds)")

        # Feature stability across noise levels (compare sigma=0 features to each sigma)
        if features_by_sigma.get(0.0):
            ref_features = set.union(*features_by_sigma[0.0])  # union across folds at sigma=0
            noise_stability = {}
            for sigma in sigmas:
                if sigma == 0.0 or sigma not in features_by_sigma:
                    continue
                if not features_by_sigma[sigma]:
                    continue
                noisy_features = set.union(*features_by_sigma[sigma])
                inter = len(ref_features & noisy_features)
                union = len(ref_features | noisy_features)
                jaccard = inter / union if union > 0 else 0.0
                noise_stability[sigma] = float(jaccard)
                print(f"    sigma={sigma:.1f} vs sigma=0: Jaccard={jaccard:.3f} "
                      f"(shared={inter}/{len(ref_features)} ref features)")

            results['feature_stability']['noise_jaccard'] = noise_stability

            # NDS of feature stability (does feature overlap degrade linearly with noise?)
            if len(noise_stability) >= 2:
                s_arr = np.array(list(noise_stability.keys()))
                j_arr = np.array(list(noise_stability.values()))
                slope, _ = np.polyfit(s_arr, j_arr, 1)
                results['feature_stability']['feature_nds'] = float(slope)
                print(f"    Feature stability NDS: {slope:.4f} (negative = features change with noise)")

        # Per-rep allocation stability across noise
        if features_by_sigma.get(0.0):
            print("\n    Per-rep feature count stability:")
            for sigma in [0.0] + [s for s in sigmas if s > 0]:
                if sigma not in features_by_sigma or not features_by_sigma[sigma]:
                    continue
                # Average feature count per rep across folds
                rep_counts = {}
                for fold_feats in features_by_sigma[sigma]:
                    fold_reps = set(r for r, _ in fold_feats)
                    for rep_name in fold_reps:
                        rep_counts.setdefault(rep_name, [])
                        count = sum(1 for r, _ in fold_feats if r == rep_name)
                        rep_counts[rep_name].append(count)
                counts_str = ", ".join(
                    f"{r}={np.mean(c):.0f}" for r, c in sorted(rep_counts.items()))
                print(f"      sigma={sigma:.1f}: {counts_str}")

    # --- Summary table ---
    print("\n  NOISE ROBUSTNESS SUMMARY (NDS, higher = more robust):")
    print(f"  {'Representation':<20} " +
          " ".join(f"{s:<12}" for s in strategies))
    print("  " + "-" * (20 + 13 * len(strategies)))

    # Individual reps
    for rep_name in sorted(results['individual_reps'].keys()):
        row = f"  {rep_name:<20} "
        for s in strategies:
            nds = results['individual_reps'][rep_name].get(s, {}).get('nds', float('nan'))
            row += f"{nds:>+8.4f}    "
        print(row)

    # Hybrids
    for budget in sorted(b for b in results['hybrids'] if isinstance(b, (int, float))):
        row = f"  hybrid_b{budget:<11} "
        for s in strategies:
            nds = results['hybrids'][budget].get(s, {}).get('nds', float('nan'))
            row += f"{nds:>+8.4f}    "
        print(row)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C10: Noise Robustness & Feature Stability',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--sigmas', type=float, nargs='+', default=None,
                        help='Noise sigma levels (default: 0.0 0.1 0.2 0.3 0.4 0.5 0.6)')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        choices=['legacy', 'quantile', 'hetero', 'threshold', 'outlier', 'valprop'],
                        help='Noise injection strategies (default: legacy quantile hetero)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  sigmas=args.sigmas, strategies=args.strategies)
    save_experiment_results(results, args.output, 'c10_noise_robustness', args.dataset)
