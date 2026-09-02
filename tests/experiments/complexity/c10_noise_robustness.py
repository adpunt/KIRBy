#!/usr/bin/env python3
"""C10: Noise Robustness & Feature Stability — auc_norm across noise strategies.

Paper: Complexity
Conclusion: Hybrids are more noise-robust than individual reps.
Dependencies: None (standalone).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import zlib

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, _cv_split,
    apply_feature_selection, get_domain_metric,
    get_best_search_for_downstream,
)
from experiments.dataset_policy import (
    noise_dose_units, noise_groups_for, noise_levels as dataset_noise_levels,
)
from kirby.hybrid import create_hybrid, apply_augmentation_selection
from kirby.noise_robustness import robustness_metrics
from kirby.noise_spec import (
    REGRESSION_STRATEGIES, NoiseSpec, make_noisy_labels_verbose,
    noise_provenance_row,
)

# Optional NoiseInject import
try:
    import noiseInject  # noqa: F401
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


def run(data, config, output_dir, dataset_name, levels=None, strategies=None,
        noise_seed=42, **retired):
    """Exp 10: Noise robustness (auc_norm / Weibull) across noise strategies.

    Tests whether hybrids are more noise-robust than individual reps, and whether
    the optimal hybrid config changes under noise. Also tracks feature stability
    across noise levels (do the same features stay important when noise increases?).

    Robustness is summarised by robustness_metrics(levels, scores): auc_norm
    (PRIMARY, higher = more robust) plus the supplementary Weibull tau/beta shape
    descriptors. This replaces the old NDS (linear slope of R² vs the noise level),
    which assumed a straight-line degradation and re-encoded baseline skill.

    NOISE LEVELS ARE NO LONGER SIGMAS. Under noiseInject 1.0.0 a level is the noise
    actually DELIVERED, as a root-mean-square, and every condition solves in closed
    form for the internal scale that delivers it. Before that each condition delivered
    whatever it happened to deliver -- between 0.49x and 2.00x the Gaussian amount --
    so the apparent severity ordering between conditions was entirely explained by
    amount rather than shape. `--sigmas` is REFUSED by name rather than ignored, so a
    job script written against the old scheme fails instead of running silently under
    the new one, where the number means something different.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: NOISE ROBUSTNESS & FEATURE STABILITY")
    print("=" * 70)

    # Order matters: gate on task_type BEFORE the noiseInject availability
    # check. A classification dataset is a clean skip regardless of whether
    # noiseInject is installed; and a missing noiseInject is a skip (not an
    # 'error') so it never surfaces as a wave failure.
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  Skipping: c10 is regression-only "
              f"(task_type={task_type}). Noise injection mangles class labels.")
        return {'skipped': True, 'reason': 'regression_only',
                'task_type': task_type}

    if not HAS_NOISEINJECT:
        print("  Skipping: noiseInject not installed "
              "(install: pip install noiseInject).")
        return {'skipped': True, 'reason': 'noiseInject_unavailable'}

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

    for _dead in ('sigmas', 'strategy'):
        if _dead in retired:
            raise TypeError(
                f"c10 no longer takes {_dead!r}. A noise level is now the amount of "
                f"noise DELIVERED, not a Gaussian sigma, so the old numbers mean "
                f"something different. Pass levels=... and strategies=... instead; "
                f"the per-dataset grids are in experiments.dataset_policy.")
    if retired:
        raise TypeError(f"unexpected arguments: {sorted(retired)}")

    if strategies is None:
        strategies = list(REGRESSION_STRATEGIES)
    dose_units = noise_dose_units(dataset_name)

    def _levels_for(strategy):
        lv = list(levels) if levels is not None else list(
            dataset_noise_levels(dataset_name, strategy=strategy))
        if len(lv) < 3:
            # auc_norm summarises the curve of score against noise level. With two
            # points there is no curve and robustness_metrics returns nan for every
            # cell, silently. Refuse rather than publish a table of nan.
            raise ValueError(
                f"c10 needs at least 3 noise levels to compute auc_norm; got "
                f"{len(lv)}: {lv}. Two points give nan in every cell.")
        return lv

    # Scaffold groups for grouped_wider / grouped_shifted. Acyclic molecules are
    # singletons (NOISE_DESIGN.md 2a rule 2) and this grouping is deliberately NOT the
    # one that sets the fold boundaries. None when the dataset has no SMILES, in which
    # case the two grouped conditions are dropped rather than run over invented groups.
    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    if noise_groups is None:
        dropped = [c for c in strategies if NoiseSpec(c, 0.1).needs_groups]
        if dropped:
            print(f"  {', '.join(dropped)}: DROPPED, this dataset has no SMILES to "
                  f"build scaffold groups from. Grouped noise over invented groups is "
                  f"uniform noise wearing a grouped name.")
            strategies = [c for c in strategies if c not in dropped]
    else:
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_group_share:.1%} of molecules")

    provenance = {}

    def _noisy(y_clean, strategy, level, fold_idx, train_idx):
        """Training labels for one (condition, level, fold).

        A fresh seed per (condition, level, fold): the injector advances its generator
        on every draw, so re-using one across the sweep would make a level's
        realisation depend on how many levels ran before it. The seed depends on the
        condition as well, or two conditions at one level would share a realisation
        and the comparison between them would measure nothing.

        WHICH molecules are hit is seeded from (condition, fold) alone, with the level
        left out, so one fold's levels hit the same molecules and auc_norm summarises
        a curve of rising dose rather than a curve of re-drawn victims.

        The dose is solved against THIS fold's clean training labels, matching
        rust/src/main.rs:512.
        """
        digest = zlib.crc32(strategy.encode('utf-8'))
        selection_seed = (int(noise_seed) * 1000003 + digest
                          + fold_idx * 104729) & 0x7FFFFFFF
        seed = (selection_seed
                + int(round(float(level) * 1000)) * 7919) & 0x7FFFFFFF
        spec = NoiseSpec(strategy=strategy, target_effective_noise=float(level),
                         noise_seed=seed, selection_seed=selection_seed,
                         dose_units=dose_units)
        # Passed for EVERY condition, not just the grouped two. The other nine
        # ignore it when drawing (bit-identical, verified) and it populates
        # n_groups / largest_group_share on their rows too.
        g = noise_groups[train_idx] if noise_groups is not None else None
        res = make_noisy_labels_verbose(y_clean, spec, groups=g)
        provenance.setdefault(spec.spec_id, noise_provenance_row(res, spec))
        return np.asarray(res.y_noisy)

    results = {
        'noise_levels': {s: _levels_for(s) for s in strategies},
        'dose_units': dose_units,
        'strategies': strategies,
        'noise_seed': int(noise_seed),
        'n_noise_groups': n_noise_groups,
        'largest_group_share': largest_group_share,
        'noise_provenance': provenance,
        'individual_reps': {},
        'hybrids': {},
        'feature_stability': {},
    }

    sample_X = next(iter(reps.values()))

    from sklearn.ensemble import RandomForestRegressor as _RF

    # --- Phase 1: Individual rep noise robustness ---
    print("\n  Phase 1: Individual rep robustness (auc_norm)")
    for rep_name, X_rep in reps.items():
        print(f"    {rep_name} (dim={X_rep.shape[1]}):")
        rep_results = {}

        for strategy in strategies:
            level_scores = {}
            _levels = _levels_for(strategy)

            for level in _levels:
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(
                        _cv_split(groups, n_folds, sample_X, labels)):
                    y_train = labels[train_idx].copy()
                    y_val = labels[val_idx]

                    # Training labels only. Level 0 goes through the same
                    # path and records exactly zero, which is the negative
                    # control the old reconstruction never had.
                    y_train = _noisy(y_train, strategy, level, fold_idx, train_idx)

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_rep[train_idx])
                    X_va = scaler.transform(X_rep[val_idx])

                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, y_train)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va)))

                level_scores[level] = float(np.mean(fold_scores))

            # Robustness metrics: auc_norm (primary) + Weibull shape, computed
            # on the score-vs-level degradation curve (no linear assumption).
            s_arr = np.array(list(level_scores.keys()))
            r2_arr = np.array(list(level_scores.values()))
            rmetrics = robustness_metrics(s_arr, r2_arr, lower_is_better=lower_is_better)

            baseline_r2 = level_scores.get(0.0, level_scores.get(min(_levels)))
            max_noise_r2 = level_scores.get(max(_levels), float('nan'))
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
                'scores_by_level': level_scores,
                'auc_norm': rmetrics['auc_norm'],
                'weibull_tau': rmetrics['weibull_tau'],
                'weibull_beta': rmetrics['weibull_beta'],
                'baseline_r2': baseline_r2,
                'retention_pct': retention,
            }
            print(f"      {strategy}: auc_norm={rmetrics['auc_norm']:.4f}, "
                  f"base_R²={baseline_r2:.4f}, retention={retention:.1f}%")

        results['individual_reps'][rep_name] = rep_results

    # --- Phase 2: Hybrid noise robustness at each budget ---
    print("\n  Phase 2: Hybrid robustness (auc_norm) by budget")

    for budget in budgets:
        print(f"\n    Budget={budget}:")
        if budget not in blueprints:
            # No cached search for this budget. Every create_hybrid call below would
            # raise KeyError into the `except` and leave fold_scores empty, and the
            # aggregation would then report baseline 0.0000 and retention 0.0% -- a
            # hybrid that scored zero, when in fact nothing ran. Say what happened.
            print(f"      NO BLUEPRINT for budget={budget}: the cached search has no "
                  f"entry for it. Skipped -- NOT reported as a zero score.")
            results['hybrids'][budget] = {'skipped': True,
                                          'reason': 'no_cached_search_for_budget'}
            continue
        budget_results = {}

        for strategy in strategies:
            level_scores = {}
            _levels = _levels_for(strategy)
            _hybrid_failures = {}

            for level in _levels:
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(
                        _cv_split(groups, n_folds, sample_X, labels)):
                    y_train = labels[train_idx].copy()
                    y_val = labels[val_idx]

                    y_train = _noisy(y_train, strategy, level, fold_idx, train_idx)

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
                    except Exception as _e:
                        _hybrid_failures.setdefault(f'{type(_e).__name__}: {_e}', 0)
                        _hybrid_failures[f'{type(_e).__name__}: {_e}'] += 1
                        continue

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_va = scaler.transform(X_val)

                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, y_train)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va)))

                if fold_scores:
                    level_scores[level] = float(np.mean(fold_scores))

            # Robustness metrics (auc_norm primary + Weibull shape)
            s_arr = np.array(list(level_scores.keys()))
            r2_arr = np.array(list(level_scores.values()))
            rmetrics = robustness_metrics(s_arr, r2_arr, lower_is_better=lower_is_better)

            if not level_scores:
                # Every fold raised. Reporting 0.0 here is how a broken hybrid arm
                # looks identical to one that genuinely scored nothing.
                print(f"      {strategy}: NO SCORES -- every fold raised. "
                      f"{_hybrid_failures}")
                budget_results[strategy] = {'skipped': True,
                                            'reason': 'all_folds_raised',
                                            'failures': _hybrid_failures}
                continue
            if _hybrid_failures:
                print(f"      {strategy}: WARNING, some folds raised: "
                      f"{_hybrid_failures}")

            baseline_r2 = level_scores.get(0.0, level_scores.get(min(_levels), 0))
            max_noise_r2 = level_scores.get(max(_levels), float('nan'))
            if baseline_r2 != 0 and not np.isnan(max_noise_r2):
                if lower_is_better:
                    retention = float(baseline_r2 / max_noise_r2 * 100)
                else:
                    retention = float(max_noise_r2 / baseline_r2 * 100)
            else:
                retention = 0.0

            budget_results[strategy] = {
                'scores_by_level': level_scores,
                'auc_norm': rmetrics['auc_norm'],
                'weibull_tau': rmetrics['weibull_tau'],
                'weibull_beta': rmetrics['weibull_beta'],
                'baseline_r2': baseline_r2,
                'retention_pct': retention,
            }
            print(f"      {strategy}: auc_norm={rmetrics['auc_norm']:.4f}, "
                  f"base_R²={baseline_r2:.4f}, retention={retention:.1f}%")

        results['hybrids'][budget] = budget_results

    # --- Phase 2b: Noise × budget interaction ---
    # For each strategy, extract auc_norm per budget to test whether budget
    # trades off against noise robustness.
    print("\n  Phase 2b: Noise × budget interaction")
    noise_budget = {}
    for strategy in strategies:
        budget_auc_pairs = []
        for budget in budgets:
            bdata = results['hybrids'].get(budget, {}).get(strategy)
            if bdata and not np.isnan(bdata.get('auc_norm', float('nan'))):
                budget_auc_pairs.append((budget, bdata['auc_norm']))
        if len(budget_auc_pairs) >= 2:
            b_arr = np.array([p[0] for p in budget_auc_pairs])
            # auc_norm is already direction-normalized: HIGHER = more robust for
            # any metric (it works on the retention curve). So a POSITIVE
            # budget-vs-auc_norm correlation = higher budget → more robust.
            auc_arr = np.array([p[1] for p in budget_auc_pairs])
            corr = float(np.corrcoef(b_arr, auc_arr)[0, 1]) if np.std(auc_arr) > 0 else 0.0
            noise_budget[strategy] = {
                'budget_auc_norm': {int(b): float(a) for b, a in budget_auc_pairs},
                'correlation': corr,  # positive = higher budget → more robust
            }
            print(f"    {strategy}: budget-auc_norm correlation={corr:.3f} "
                  f"({'higher budget = more robust' if corr > 0.3 else 'weak/no trend'})")
    results['noise_budget_interaction'] = noise_budget

    # --- Phase 3: Feature stability across noise levels ---
    print("\n  Phase 3: Feature stability across noise")
    budget_for_stability = budgets[len(budgets) // 2]  # middle budget
    print(f"    Using budget={budget_for_stability}")

    # One condition only -- feature stability re-runs the whole allocation at every
    # level, so the full set would be eleven times the cost for a question that is
    # about the level, not the shape.
    for strategy in strategies[:1]:
        features_by_level = {}
        _levels_stability = _levels_for(strategy)

        for level in _levels_stability:
            fold_features = []
            for fold_idx, (train_idx, val_idx) in enumerate(
                    _cv_split(groups, n_folds, sample_X, labels)):
                y_train = _noisy(labels[train_idx].copy(), strategy, level,
                                 fold_idx, train_idx)

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

            features_by_level[level] = fold_features

        # Feature stability across folds (at level=0)
        if features_by_level.get(0.0) and len(features_by_level[0.0]) >= 2:
            fold_sets = features_by_level[0.0]
            jaccard_across_folds = []
            for i in range(len(fold_sets)):
                for j in range(i + 1, len(fold_sets)):
                    inter = len(fold_sets[i] & fold_sets[j])
                    union = len(fold_sets[i] | fold_sets[j])
                    jaccard_across_folds.append(inter / union if union > 0 else 0)
            results['feature_stability']['fold_jaccard_level0'] = float(np.mean(jaccard_across_folds))
            print(f"    Fold stability (level=0): Jaccard={np.mean(jaccard_across_folds):.3f}")

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
                    print(f"    Nogueira stability (level=0): phi={nogueira_phi:.3f} "
                          f"(95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
                else:
                    print(f"    Nogueira stability (level=0): phi={nogueira_phi:.3f} "
                          f"(CI undefined: requires M>2 folds)")

        # Feature stability across noise levels (compare level=0 features to each level)
        if features_by_level.get(0.0):
            ref_features = set.union(*features_by_level[0.0])  # union across folds at level=0
            noise_stability = {}
            for level in _levels_stability:
                if level == 0.0 or level not in features_by_level:
                    continue
                if not features_by_level[level]:
                    continue
                noisy_features = set.union(*features_by_level[level])
                inter = len(ref_features & noisy_features)
                union = len(ref_features | noisy_features)
                jaccard = inter / union if union > 0 else 0.0
                noise_stability[level] = float(jaccard)
                print(f"    level={level:.1f} vs level=0: Jaccard={jaccard:.3f} "
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
        if features_by_level.get(0.0):
            print("\n    Per-rep feature count stability:")
            for level in [0.0] + [v for v in _levels_stability if v > 0]:
                if level not in features_by_level or not features_by_level[level]:
                    continue
                # Average feature count per rep across folds
                rep_counts = {}
                for fold_feats in features_by_level[level]:
                    fold_reps = set(r for r, _ in fold_feats)
                    for rep_name in fold_reps:
                        rep_counts.setdefault(rep_name, [])
                        count = sum(1 for r, _ in fold_feats if r == rep_name)
                        rep_counts[rep_name].append(count)
                counts_str = ", ".join(
                    f"{r}={np.mean(c):.0f}" for r, c in sorted(rep_counts.items()))
                print(f"      level={level:.1f}: {counts_str}")

    # --- Summary table ---
    # auc_norm = normalised area under the retention curve. It is already
    # direction-normalized: HIGHER = more robust for any metric direction.
    print("\n  NOISE ROBUSTNESS SUMMARY (auc_norm, higher = more robust):")
    print(f"  {'Representation':<20} " +
          " ".join(f"{s:<12}" for s in strategies))
    print("  " + "-" * (20 + 13 * len(strategies)))

    # Individual reps
    for rep_name in sorted(results['individual_reps'].keys()):
        row = f"  {rep_name:<20} "
        for s in strategies:
            auc = results['individual_reps'][rep_name].get(s, {}).get('auc_norm', float('nan'))
            row += f"{auc:>+8.4f}    "
        print(row)

    # Hybrids
    for budget in sorted(b for b in results['hybrids'] if isinstance(b, (int, float))):
        row = f"  hybrid_b{budget:<11} "
        for s in strategies:
            auc = results['hybrids'][budget].get(s, {}).get('auc_norm', float('nan'))
            row += f"{auc:>+8.4f}    "
        print(row)

    return results


if __name__ == '__main__':
    import argparse
    parser = create_experiment_parser('C10: Noise Robustness & Feature Stability',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--levels', type=float, nargs='+', default=None,
                        help='Noise levels. Default is the grid for this dataset in '
                             'experiments.dataset_policy: anchored to the published '
                             'assay error in the label\'s own units for herg_ki, '
                             'caco2 and logd, a fraction of the label SD otherwise, '
                             'and the censored fraction for censoring.')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        choices=list(REGRESSION_STRATEGIES),
                        help='Noise conditions (default: all eleven).')
    parser.add_argument('--noise-seed', type=int, default=42,
                        help='Base seed for the noise draw only.')
    # Refused by name, not ignored: a job script written against the old scheme would
    # otherwise run silently under the new one, where the number means something else.
    parser.add_argument('--sigmas', type=float, nargs='+', default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sigmas is not None:
        parser.error("--sigmas is gone. A noise level is now the amount of noise "
                     "DELIVERED (a root-mean-square), not a Gaussian sigma, so the "
                     "old numbers mean something different. Use --levels.")
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  levels=args.levels, strategies=args.strategies,
                  noise_seed=args.noise_seed)
    save_experiment_results(results, args.output, 'c10_noise_robustness', args.dataset)
