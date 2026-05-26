#!/usr/bin/env python3
"""C09: Formal Complexity Analysis — ANOVA + information retention + effective rank + bias-variance.

Paper: Complexity
Conclusion: ANOVA: model 50%+, budget 25%+, interaction 10%+ variance.
Dependencies: c02 (reads JSON for ANOVA), c06 (reads JSON for bias-variance).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import f as f_dist, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_model_factories, evaluate_cv_with_train,
    estimate_mutual_information_knn, compute_effective_rank,
    apply_feature_selection,
    find_best_hybrid, get_domain_metric,
    get_best_search_for_downstream,
    INTERNAL_EVALUATORS,
)
from kirby.hybrid import create_hybrid
from experiments._shared import create_hybrid_cached
# TODO: create_hybrid API not finalised — allocation_method / importance_method
#       semantics may change after framework experiments. Revisit call sites below.


# =========================================================================
# 9a: ANOVA helpers
# =========================================================================

def _compute_icc_check(configs):
    """Compute pairwise ICC(1,1) between ANOVA factor levels to detect redundancy.

    Following qsar_qm_models pattern: check whether model/budget pairs are
    redundant (ICC > 0.9) before running ANOVA.

    ICC(1,1) = (BMS - WMS) / (BMS + (k-1)*WMS)
    """
    models_set = sorted(set(c['model'] for c in configs))
    budgets_set = sorted(set(c['budget'] for c in configs))

    # Build mean R² matrix: rows = budgets, columns = models
    cell_means = {}
    for c in configs:
        key = (c['model'], c['budget'])
        fold_scores = c.get('fold_scores', [])
        vals = [s for s in fold_scores if not np.isnan(s)] if fold_scores else []
        if not vals and not np.isnan(c['score_mean']):
            vals = [c['score_mean']]
        if vals:
            cell_means[key] = float(np.mean(vals))

    result = {'model_pairs': {}, 'budget_pairs': {}, 'redundant_models': [], 'redundant_budgets': []}

    # --- Pairwise ICC between models (treating budgets as subjects) ---
    print("\n    ICC(1,1) between model pairs (budgets as subjects):")
    for i, m1 in enumerate(models_set):
        for m2 in models_set[i+1:]:
            # Get R² for each budget where both models have data
            rows = []
            for b in budgets_set:
                v1 = cell_means.get((m1, b))
                v2 = cell_means.get((m2, b))
                if v1 is not None and v2 is not None:
                    rows.append([v1, v2])

            if len(rows) < 2:
                continue

            data = np.array(rows)  # shape: (n_budgets, 2)
            n, k = data.shape
            grand_mean = data.mean()
            row_means = data.mean(axis=1)
            bms = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)
            wms = ((data - row_means.reshape(-1, 1)) ** 2).sum() / (n * (k - 1))

            if (bms + (k - 1) * wms) == 0:
                icc = 1.0
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            pair_key = f"{m1}_vs_{m2}"
            result['model_pairs'][pair_key] = float(icc)

            flag = " ** REDUNDANT" if icc > 0.9 else ""
            print(f"      {m1} vs {m2}: ICC={icc:.4f}{flag}")

            if icc > 0.9:
                result['redundant_models'].append(pair_key)

    # --- Pairwise ICC between budgets (treating models as subjects) ---
    print("    ICC(1,1) between budget pairs (models as subjects):")
    for i, b1 in enumerate(budgets_set):
        for b2 in budgets_set[i+1:]:
            rows = []
            for m in models_set:
                v1 = cell_means.get((m, b1))
                v2 = cell_means.get((m, b2))
                if v1 is not None and v2 is not None:
                    rows.append([v1, v2])

            if len(rows) < 2:
                continue

            data = np.array(rows)
            n, k = data.shape
            grand_mean = data.mean()
            row_means = data.mean(axis=1)
            bms = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)
            wms = ((data - row_means.reshape(-1, 1)) ** 2).sum() / (n * (k - 1))

            if (bms + (k - 1) * wms) == 0:
                icc = 1.0
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            pair_key = f"b{b1}_vs_b{b2}"
            result['budget_pairs'][pair_key] = float(icc)

            flag = " ** REDUNDANT" if icc > 0.9 else ""
            print(f"      b={b1} vs b={b2}: ICC={icc:.4f}{flag}")

            if icc > 0.9:
                result['redundant_budgets'].append(pair_key)

    if result['redundant_models']:
        print(f"    WARNING: {len(result['redundant_models'])} redundant model pair(s) detected")
    if result['redundant_budgets']:
        print(f"    WARNING: {len(result['redundant_budgets'])} redundant budget pair(s) detected")

    return result


def _run_anova_decomposition(exp2_results):
    """9a: ANOVA variance decomposition on Exp 2 grid results.

    Tests whether model x budget interaction explains significant variance in R².
    Uses per-fold CV scores as replicates for proper within-cell variance estimation.
    """
    print("\n  9a: ANOVA Variance Decomposition")

    configs = exp2_results.get('configs', [])
    if not configs:
        print("    No Exp 2 configs available")
        return {'error': 'no_data'}

    # ICC redundancy check (following qsar_qm_models pattern)
    icc_results = _compute_icc_check(configs)

    # Build data: each config is an observation with factors (model, budget) -> R²
    models_set = sorted(set(c['model'] for c in configs))
    budgets_set = sorted(set(c['budget'] for c in configs))

    if len(models_set) < 2 or len(budgets_set) < 2:
        print("    Need at least 2 models and 2 budgets for ANOVA")
        return {'error': 'insufficient_factors'}

    # Collect R² values into cells: cell[model][budget] = list of r2 values
    # Use per-fold scores as replicates (real variance) instead of mean R²
    # which can produce zero within-cell variance with identical configs
    cells = {}
    for c in configs:
        key = (c['model'], c['budget'])
        if key not in cells:
            cells[key] = []
        fold_scores = c.get('fold_scores', [])
        if fold_scores:
            cells[key].extend([s for s in fold_scores if not np.isnan(s)])
        elif not np.isnan(c['score_mean']):
            # Fallback: use mean if fold_scores not available (old cached results)
            cells[key].append(c['score_mean'])

    # Compute means
    grand_mean = np.mean([r2 for vals in cells.values() for r2 in vals])
    n_total = sum(len(v) for v in cells.values())

    model_means = {}
    for m in models_set:
        vals = [r2 for (mm, b), vv in cells.items() for r2 in vv if mm == m]
        model_means[m] = np.mean(vals) if vals else grand_mean

    budget_means = {}
    for b in budgets_set:
        vals = [r2 for (mm, bb), vv in cells.items() for r2 in vv if bb == b]
        budget_means[b] = np.mean(vals) if vals else grand_mean

    cell_means = {}
    for key, vals in cells.items():
        cell_means[key] = np.mean(vals) if vals else grand_mean

    # Sum of squares
    ss_total = sum((r2 - grand_mean) ** 2 for vals in cells.values() for r2 in vals)

    ss_model = sum(
        len([r2 for (mm, b), vv in cells.items() for r2 in vv if mm == m])
        * (model_means[m] - grand_mean) ** 2
        for m in models_set
    )

    ss_budget = sum(
        len([r2 for (mm, b), vv in cells.items() for r2 in vv if b == bb])
        * (budget_means[bb] - grand_mean) ** 2
        for bb in budgets_set
    )

    ss_interaction = 0
    for (m, b), vals in cells.items():
        expected = model_means[m] + budget_means[b] - grand_mean
        ss_interaction += len(vals) * (cell_means[(m, b)] - expected) ** 2

    ss_residual = max(0, ss_total - ss_model - ss_budget - ss_interaction)

    # Eta-squared (% variance)
    eta2 = {}
    if ss_total > 0:
        eta2['model'] = float(ss_model / ss_total * 100)
        eta2['budget'] = float(ss_budget / ss_total * 100)
        eta2['interaction'] = float(ss_interaction / ss_total * 100)
        eta2['residual'] = float(ss_residual / ss_total * 100)
    else:
        eta2 = {'model': 0, 'budget': 0, 'interaction': 0, 'residual': 0}

    # Degrees of freedom and F-statistics
    df_model = len(models_set) - 1
    df_budget = len(budgets_set) - 1
    df_interaction = df_model * df_budget
    df_residual = max(1, n_total - len(models_set) * len(budgets_set))

    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_budget = ss_budget / df_budget if df_budget > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 1e-10

    f_model = ms_model / ms_residual
    f_budget = ms_budget / ms_residual
    f_interaction = ms_interaction / ms_residual

    p_model = 1 - f_dist.cdf(f_model, df_model, df_residual)
    p_budget = 1 - f_dist.cdf(f_budget, df_budget, df_residual)
    p_interaction = 1 - f_dist.cdf(f_interaction, df_interaction, df_residual)

    result = {
        'n_observations': n_total,
        'n_models': len(models_set),
        'n_budgets': len(budgets_set),
        # ANOVA values are computed over whatever metric c02 used (R² for
        # molecular, CI for DTA, AUROC/MCC for classification, Spearman for
        # antibody) — naming the key "r2" was wrong for non-molecular runs.
        'grand_mean_score': float(grand_mean),
        'eta2': eta2,
        'f_model': float(f_model), 'p_model': float(p_model),
        'f_budget': float(f_budget), 'p_budget': float(p_budget),
        'f_interaction': float(f_interaction), 'p_interaction': float(p_interaction),
        'ss_total': float(ss_total),
        'icc': icc_results,
    }

    print(f"    Grand mean R²: {grand_mean:.4f}")
    print(f"    eta2 model:       {eta2['model']:.1f}%  (F={f_model:.2f}, p={p_model:.4f})")
    print(f"    eta2 budget:      {eta2['budget']:.1f}%  (F={f_budget:.2f}, p={p_budget:.4f})")
    print(f"    eta2 interaction: {eta2['interaction']:.1f}%  (F={f_interaction:.2f}, p={p_interaction:.4f})")
    print(f"    eta2 residual:    {eta2['residual']:.1f}%")

    return result


# =========================================================================
# 9b: Information retention
# =========================================================================

def _run_information_retention(data, budgets, search_speed='full', dataset_name=None):
    """9b: MI(X_selected, Y) and MI(X_selected, X_full) vs budget.

    At each budget, uses find_best_hybrid to discover the best hybrid config,
    then profiles its information content.
    """
    print("\n  9b: Information Retention Curves")

    metric_name_9b, _, lower_is_better_9b = get_domain_metric(data)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {'budgets': [], 'mi_xy': [], 'mi_xx_full': [], 'best_configs': []}

    augs = data.get('augmentations')
    sample_X = next(iter(reps.values()))
    from sklearn.linear_model import Ridge as _Ridge

    for budget in sorted(budgets):
        # Pool helper across (general, top) x (none, uncapped, capped) cache
        # files; pick winner across common downstream models for this budget.
        bp = None
        best_score = None
        for cand_model in ['rf', 'svm', 'lightgbm', 'mlp']:
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=cand_model, budget=budget,
                    augmentations=augs, speed=search_speed,
                    dataset_name=dataset_name,
                    task_type=data['task_type'], metric=metric_name_9b,
                )
            except Exception:
                continue
            if not _m:
                continue
            score = _m['best_config'].get('score_mean')
            if score is None:
                continue
            is_better = (best_score is None) or (
                score < best_score if lower_is_better_9b
                else score > best_score)
            if is_better:
                best_score = score
                bp = _m['best_config']['blueprint']

        alloc = bp.allocation_method if bp else 'greedy'
        imp = bp.importance_method if bp else 'treeshap'
        int_eval = bp.internal_evaluator if bp else 'ridge'
        aug_strat = bp.augmentation_strategy if bp else ('uncapped' if augs else 'none')
        results['best_configs'].append({'budget': budget, 'importance': imp, 'allocation': alloc})

        # Per-fold hybrid creation to avoid label leakage in MI measurement
        fold_mi_xy = []
        fold_mi_xx = []
        from experiments._shared import cv_split as _cv_split_9b
        for train_idx, val_idx in _cv_split_9b(groups, 3, sample_X, labels, mode='full'):
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            y_train = labels[train_idx]
            train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
            try:
                X_hybrid, _ = create_hybrid_cached(
                    train_reps, y_train, train_idx,
                    allocation_method=alloc,
                    importance_method=imp,
                    budget=budget,
                    internal_evaluator=int_eval,
                    augmentations=train_augs,
                    augmentation_strategy=aug_strat,
                )
            except Exception:
                continue

            # MI(X_selected, Y) on train fold
            n_comp = min(50, X_hybrid.shape[1], X_hybrid.shape[0] - 1)
            X_red = PCA(n_components=n_comp).fit_transform(
                StandardScaler().fit_transform(X_hybrid))
            try:
                mi_xy = estimate_mutual_information_knn(X_red, y_train, k=5)
                fold_mi_xy.append(float(mi_xy))
            except Exception:
                pass

            # MI(X_selected, X_full) -- approximate via explained variance on train
            X_train_concat = np.hstack([v[train_idx] for v in reps.values()])
            X_full_s = StandardScaler().fit_transform(X_train_concat)
            X_hyb_s = StandardScaler().fit_transform(X_hybrid)
            n_pcs = min(20, X_hybrid.shape[1], X_full_s.shape[0] - 1)
            try:
                pca_full = PCA(n_components=n_pcs)
                X_full_pcs = pca_full.fit_transform(X_full_s)
                explained_var_retained = 0
                for pc_i in range(n_pcs):
                    reg = _Ridge(alpha=1.0)
                    reg.fit(X_hyb_s, X_full_pcs[:, pc_i])
                    r2_pc = max(0, r2_score(X_full_pcs[:, pc_i], reg.predict(X_hyb_s)))
                    explained_var_retained += r2_pc * pca_full.explained_variance_ratio_[pc_i]
                fold_mi_xx.append(float(explained_var_retained))
            except Exception:
                pass

        mi_xy_avg = float(np.mean(fold_mi_xy)) if fold_mi_xy else float('nan')
        mi_xx_avg = float(np.mean(fold_mi_xx)) if fold_mi_xx else float('nan')

        results['budgets'].append(budget)
        results['mi_xy'].append(mi_xy_avg)
        results['mi_xx_full'].append(mi_xx_avg)

        print(f"    budget={budget}: MI(X,Y)={mi_xy_avg:.4f}, info_retained={mi_xx_avg:.4f}")

    # Find inflection: where MI(X,Y) gain drops below 10% of max gain
    if len(results['mi_xy']) >= 3:
        mi_arr = np.array(results['mi_xy'])
        gains = np.diff(mi_arr)
        max_gain = np.max(np.abs(gains)) if len(gains) > 0 else 0
        if max_gain > 0:
            plateau_mask = gains < 0.1 * max_gain
            if np.any(plateau_mask):
                inflection_idx = int(np.argmax(plateau_mask))
                # gains[i] is the MI increase from budgets[i]→budgets[i+1].
                # When plateau_mask[i] is first True the plateau STARTS at
                # budgets[i] (the last productive budget) — that is the
                # inflection point, not budgets[i+1].
                results['inflection_budget'] = results['budgets'][inflection_idx]
                print(f"    Inflection point: budget ~{results['inflection_budget']}")
            else:
                print("    No inflection detected (MI still increasing)")

    return results


# =========================================================================
# 9c: Effective rank analysis
# =========================================================================

def _run_effective_rank_analysis(data, budgets, search_speed='full', dataset_name=None):
    """9c: Effective rank of hybrid vs PCA vs random at each budget.

    At each budget, uses find_best_hybrid to discover the best config,
    then compares effective rank: best hybrid vs PCA vs random subset.
    """
    print("\n  9c: Effective Rank Profile")

    metric_name_9c, _, lower_is_better_9c = get_domain_metric(data)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {'budgets': [], 'hybrid_rank': [], 'pca_rank': [], 'random_rank': []}

    augs = data.get('augmentations')
    sample_X = next(iter(reps.values()))
    from experiments._shared import cv_split as _cv_split_9c

    for budget in sorted(budgets):
        n_feat = min(budget, X_concat.shape[1], X_concat.shape[0] - 1)

        # Pool helper picks best (tier, aug_cond) across common models for this budget.
        bp = None
        best_score = None
        for cand_model in ['rf', 'svm', 'lightgbm', 'mlp']:
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=cand_model, budget=budget,
                    augmentations=augs, speed=search_speed,
                    dataset_name=dataset_name,
                    task_type=data['task_type'], metric=metric_name_9c,
                )
            except Exception:
                continue
            if not _m:
                continue
            score = _m['best_config'].get('score_mean')
            if score is None:
                continue
            is_better = (best_score is None) or (
                score < best_score if lower_is_better_9c
                else score > best_score)
            if is_better:
                best_score = score
                bp = _m['best_config']['blueprint']

        alloc = bp.allocation_method if bp else 'greedy'
        imp = bp.importance_method if bp else 'treeshap'
        int_eval = bp.internal_evaluator if bp else 'ridge'
        aug_strat = bp.augmentation_strategy if bp else ('uncapped' if augs else 'none')

        # Per-fold hybrid creation to avoid label leakage in rank measurement
        fold_ranks = []
        for train_idx, _ in _cv_split_9c(groups, 3, sample_X, labels, mode='full'):
            try:
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
                X_hybrid, _ = create_hybrid_cached(
                    train_reps, labels[train_idx], train_idx,
                    allocation_method=alloc,
                    importance_method=imp,
                    budget=budget,
                    internal_evaluator=int_eval,
                    augmentations=train_augs,
                    augmentation_strategy=aug_strat,
                )
                fold_ranks.append(compute_effective_rank(X_hybrid))
            except Exception:
                pass
        hybrid_rank = float(np.mean(fold_ranks)) if fold_ranks else float('nan')

        # PCA — per-fold to match hybrid (avoids using val data in fit)
        pca_fold_ranks = []
        for train_idx, _ in _cv_split_9c(groups, 3, sample_X, labels, mode='full'):
            try:
                pca = PCA(n_components=n_feat)
                X_pca = pca.fit_transform(StandardScaler().fit_transform(X_concat[train_idx]))
                pca_fold_ranks.append(compute_effective_rank(X_pca))
            except Exception:
                pass
        pca_rank = float(np.mean(pca_fold_ranks)) if pca_fold_ranks else float('nan')

        # Random subset — per-fold to match hybrid
        random_fold_ranks = []
        rng = np.random.RandomState(42)
        random_idx = rng.choice(X_concat.shape[1], n_feat, replace=False)
        for train_idx, _ in _cv_split_9c(groups, 3, sample_X, labels, mode='full'):
            try:
                X_random = X_concat[train_idx][:, random_idx]
                random_fold_ranks.append(compute_effective_rank(X_random))
            except Exception:
                pass
        random_rank = float(np.mean(random_fold_ranks)) if random_fold_ranks else float('nan')

        results['budgets'].append(budget)
        results['hybrid_rank'].append(float(hybrid_rank))
        results['pca_rank'].append(float(pca_rank))
        results['random_rank'].append(float(random_rank))

        print(f"    budget={budget}: hybrid={hybrid_rank:.1f}, pca={pca_rank:.1f}, random={random_rank:.1f}")

    return results


# =========================================================================
# 9d: Bias-variance proxy
# =========================================================================

def _run_bias_variance_analysis(data, models, budgets, n_folds, exp6_results=None,
                                search_speed='full', dataset_name=None):
    """9d: Bias-variance proxy from train-val gap across budgets."""
    print("\n  9d: Bias-Variance Proxy")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    mode = 'intermediary' if n_folds == 1 else 'full'

    results = {'models': {}}

    # If Exp 6 results are available, extract train-val gaps from there
    if exp6_results and 'families' in exp6_results:
        print("    Using Exp 6 data for bias-variance analysis")
        for fam_name, fam_data in exp6_results['families'].items():
            fam_bv = {}
            for var_name, budget_data in fam_data.get('heatmap', {}).items():
                for budget, metrics in budget_data.items():
                    if isinstance(metrics, dict) and 'train_r2' in metrics:
                        key = f"{var_name}_b{budget}"
                        fam_bv[key] = {
                            'train_r2': metrics['train_r2'],
                            'val_r2': metrics['val_r2'],
                            'gap': metrics['train_r2'] - metrics['val_r2'],
                            'bias_proxy': 1 - metrics['train_r2'],
                            'budget': int(budget) if isinstance(budget, (int, float)) else budget,
                        }
            if fam_bv:
                results['models'][fam_name] = fam_bv
        if results['models']:
            # Print summary
            for fam_name, fam_data in results['models'].items():
                print(f"    {fam_name}:")
                for key, vals in sorted(fam_data.items()):
                    print(f"      {key}: gap={vals['gap']:.3f}, bias={vals['bias_proxy']:.3f}")
            return results

    # Fallback: discover best config per model at each budget via the pool
    # helper (best across (tier, aug_cond) for each model), then measure
    # train-val gap with that config.
    print("    Computing bias-variance from scratch (using pool helper per budget)")
    augs = data.get('augmentations')
    model_names = list(models.keys())

    for budget in budgets:
        for model_name, model_factory in models.items():
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=model_name, budget=budget,
                    augmentations=augs, speed=search_speed,
                    dataset_name=dataset_name,
                    task_type=data['task_type'], metric=metric_name,
                )
            except Exception as e:
                print(f"    {model_name}/b={budget}: pool helper failed - {e}")
                _s, _m = None, None
            if _m:
                bp = _m['best_config']['blueprint']
                alloc = bp.allocation_method
                imp = bp.importance_method
                int_eval = bp.internal_evaluator
                aug_strat = bp.augmentation_strategy
            else:
                alloc, imp, int_eval = 'greedy', 'treeshap', 'ridge'
                aug_strat = 'uncapped' if augs else 'none'

            try:
                val_r2, val_std, train_r2, train_std, n_feat = evaluate_cv_with_train(
                    reps, labels, groups, model_factory, n_folds,
                    allocation_method=alloc,
                    importance_method=imp,
                    budget=budget,
                    mode=mode,
                    metric_fn=metric_fn,
                    internal_evaluator=int_eval,
                    augmentations=augs,
                    augmentation_strategy=aug_strat,
                )
                if model_name not in results['models']:
                    results['models'][model_name] = {}
                results['models'][model_name][budget] = {
                    'train_r2': train_r2, 'val_r2': val_r2,
                    'gap': train_r2 - val_r2,
                    'bias_proxy': 1 - train_r2,
                    'budget': budget,
                    'importance': imp,
                    'allocation': alloc,
                }
                print(f"    {model_name}/b={budget}: gap={train_r2 - val_r2:.3f}, "
                      f"bias={1 - train_r2:.3f}, val_R²={val_r2:.4f} [{imp}/{alloc}]")
            except Exception as e:
                print(f"    {model_name}/b={budget}: FAILED - {e}")

    return results


# =========================================================================
# Main run function
# =========================================================================

def run(data, config, output_dir, dataset_name, sub_analyses=None,
        c02_results_dir=None, c06_results_dir=None):
    """Exp 9: Formal theoretical grounding with 4 sub-analyses."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: FORMAL COMPLEXITY ANALYSIS")
    print("=" * 70)

    budgets = config['budgets']
    search_speed = config['search_speed']
    n_folds = config['n_splits']
    model_names = config['models']
    models = get_model_factories(model_names,
                                    task_type=data['task_type'],
                                    n_samples=len(data['labels']))

    all_subs = sub_analyses or ['anova', 'information', 'effective_rank', 'bias_variance']
    results = {}

    # Load dependency results if directories provided
    exp2_results = None
    if c02_results_dir:
        exp2_results = load_experiment_results(c02_results_dir, 'c02_complexity_matching_grid',
                                               dataset_name)
        if exp2_results:
            print(f"  Loaded c02 results from {c02_results_dir}")
        else:
            print(f"  No c02 results found at {c02_results_dir}")

    exp6_results = None
    if c06_results_dir:
        exp6_results = load_experiment_results(c06_results_dir, 'c06_within_family_gradient',
                                               dataset_name)
        if exp6_results:
            print(f"  Loaded c06 results from {c06_results_dir}")
        else:
            print(f"  No c06 results found at {c06_results_dir}")

    if 'anova' in all_subs and exp2_results:
        results['anova'] = _run_anova_decomposition(exp2_results)

    if 'information' in all_subs:
        results['information'] = _run_information_retention(data, budgets, search_speed, dataset_name=dataset_name)

    if 'effective_rank' in all_subs:
        results['effective_rank'] = _run_effective_rank_analysis(data, budgets, search_speed, dataset_name=dataset_name)

    if 'bias_variance' in all_subs:
        results['bias_variance'] = _run_bias_variance_analysis(
            data, models, budgets, n_folds, exp6_results, search_speed, dataset_name=dataset_name)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C09: Formal Complexity Analysis',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--c02-results', type=str, default=None,
                        help='Path to c02 output directory (reads configs for ANOVA)')
    parser.add_argument('--c06-results', type=str, default=None,
                        help='Path to c06 output directory (reads train-val gaps for bias-variance)')
    parser.add_argument('--sub-analyses', type=str, nargs='+', default=None,
                        choices=['anova', 'information', 'effective_rank', 'bias_variance'],
                        help='Which sub-analyses to run (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  sub_analyses=args.sub_analyses,
                  c02_results_dir=args.c02_results,
                  c06_results_dir=args.c06_results)
    save_experiment_results(results, args.output, 'c09_formal_analysis', args.dataset)
