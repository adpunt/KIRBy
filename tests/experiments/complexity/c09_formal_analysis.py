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
from scipy.stats import spearmanr
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
    get_rule_hybrids, DOWNSTREAM_MODELS)
from kirby.hybrid import create_hybrid
from experiments._shared import create_hybrid_cached
# TODO: create_hybrid API not finalised — allocation_method / importance_method
#       semantics may change after framework experiments. Revisit call sites below.


# Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
# model, per family. Until then each sub-analysis below took ONE recipe per
# budget (9b, 9c: the best across four models; 9d: the best per model). The
# single-recipe curves keep their keys; every rule hybrid now gets its own.
def _rule_entries_for_budget(data, dataset_name, models, budget, metric,
                             missing_out):
    """[(model, 'model|family|slot', entry), ...] for one budget.

    A distance search stops at 500 features; asked for a larger budget its
    entries carry budget_relaxed=True (ranked over every budget it holds).
    Every slot the rule asked for and did not get is appended to missing_out.
    """
    out = []
    for m in models:
        try:
            r = get_rule_hybrids(
                data['reps'], data['labels'], data.get('groups'),
                downstream_model=m, budget=budget,
                dataset_name=dataset_name,
                augmentations=data.get('augmentations'),
                task_type=data['task_type'], metric=metric, data=data,
                fold_i=None)
        except Exception as exc:
            missing_out.append(f'{m} budget={budget}: the owner\'s set could '
                               f'not be read ({type(exc).__name__}: {exc})')
            continue
        missing_out.extend(f'{m} budget={budget}: {x}' for x in r['missing'])
        for h in r['hybrids']:
            out.append((m, f"{m}|{h.get('objective')}|{h.get('slot')}", h))
    return out


def _rule_meta(h):
    return {'slot': h.get('slot'), 'lineage': h.get('lineage'),
            'objective': h.get('objective'), 'tier': h.get('tier'),
            'aug_condition': h.get('aug_condition'),
            'recipe': h.get('recipe'),
            'budget_relaxed': bool(h.get('budget_relaxed'))}

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

            # ⛔ ZERO VARIANCE ON BOTH SIDES IS NO INFORMATION, NOT AGREEMENT.
            # bms + (k-1)*wms == 0 means every score in the block is identical:
            # nothing varies between the two members and nothing varies within
            # them. Returning 1.0 put that pair straight into the redundant
            # list, and "these two agree perfectly" is exactly the wrong
            # reading of "neither of these moved at all".
            if (bms + (k - 1) * wms) == 0:
                icc = float('nan')
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            pair_key = f"{m1}_vs_{m2}"
            result['model_pairs'][pair_key] = float(icc)

            flag = (" ** DEGENERATE (no variance either way)"
                    if not np.isfinite(icc)
                    else " ** REDUNDANT" if icc > 0.9 else "")
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

            # ⛔ ZERO VARIANCE ON BOTH SIDES IS NO INFORMATION, NOT AGREEMENT.
            # bms + (k-1)*wms == 0 means every score in the block is identical:
            # nothing varies between the two members and nothing varies within
            # them. Returning 1.0 put that pair straight into the redundant
            # list, and "these two agree perfectly" is exactly the wrong
            # reading of "neither of these moved at all".
            if (bms + (k - 1) * wms) == 0:
                icc = float('nan')
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            pair_key = f"b{b1}_vs_b{b2}"
            result['budget_pairs'][pair_key] = float(icc)

            flag = (" ** DEGENERATE (no variance either way)"
                    if not np.isfinite(icc)
                    else " ** REDUNDANT" if icc > 0.9 else "")
            print(f"      b={b1} vs b={b2}: ICC={icc:.4f}{flag}")

            if icc > 0.9:
                result['redundant_budgets'].append(pair_key)

    if result['redundant_models']:
        print(f"    WARNING: {len(result['redundant_models'])} redundant model pair(s) detected")
    if result['redundant_budgets']:
        print(f"    WARNING: {len(result['redundant_budgets'])} redundant budget pair(s) detected")

    return result


def _run_anova_decomposition(exp2_results, metric_name=None):
    """9a: ANOVA variance decomposition on Exp 2 grid results.

    Tests whether model x budget interaction explains significant variance in
    the dataset's own metric.

    One configuration is one observation, and its observation is its own mean
    over folds. The five fold scores of one configuration are five readings of
    the same configuration, not five independent configurations, so counting
    each of them inflated the observation count fivefold and the p-values with
    it. Within-cell variance now comes from the several allocation and
    importance settings c02 searched at the same (model, budget).

    The observations are c02's own selection scores, read off search.all_results
    (c02_complexity_grid.py:223-233). They are not held-out scores, and the
    payload says so in 'observations'.
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

    # ⛔ THE BUDGET FACTOR IS THE BUDGET ASKED FOR, AND c02 ALREADY STORES THE
    # WIDTH BUILT. Every configuration c02 writes carries 'n_features', the
    # mean of X_train.shape[1] over folds (c02:260, from
    # find_best_hybrid.py:5374). create_hybrid stops each representation at its
    # last feature scoring above zero, so two budget LEVELS can build the same
    # width and are counted below as two levels of the budget factor -- which
    # is the one thing an ANOVA on that factor cannot survive. The number
    # needed to see it is already in the file this was handed.
    _widths_by_budget = {}
    for _c in configs:
        _w = _c.get('n_features')
        if _w is not None and np.isfinite(_w):
            _widths_by_budget.setdefault(_c['budget'], []).append(float(_w))
    _collapsed = []
    _bl = sorted(_widths_by_budget)
    for _i in range(len(_bl) - 1):
        _b1, _b2 = _bl[_i], _bl[_i + 1]
        _m1, _m2 = np.mean(_widths_by_budget[_b1]), np.mean(_widths_by_budget[_b2])
        if abs(_m1 - _m2) < 1.0:
            _collapsed.append((_b1, _b2, round(float(_m1), 1)))
    if _collapsed:
        print(f"    budget levels that build the SAME width "
              f"(budget, next budget, features built): {_collapsed}. The "
              f"ANOVA below counts them as separate levels.")

    if len(models_set) < 2 or len(budgets_set) < 2:
        print("    Need at least 2 models and 2 budgets for ANOVA")
        return {'error': 'insufficient_factors'}

    # Collect scores into cells: cell[(model, budget)] = list of configuration
    # means, one entry per configuration. A configuration's mean is the mean of
    # its fold scores when they are stored, and score_mean otherwise.
    cells = {}
    for c in configs:
        key = (c['model'], c['budget'])
        if key not in cells:
            cells[key] = []
        fold_scores = [s for s in c.get('fold_scores', []) if not np.isnan(s)]
        if fold_scores:
            cells[key].append(float(np.mean(fold_scores)))
        elif not np.isnan(c['score_mean']):
            # Old cached results carry no fold scores, only the mean.
            cells[key].append(float(c['score_mean']))

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

    # Reported as computed. The design is unbalanced — c02 searches a different
    # number of allocation and importance settings at different (model, budget)
    # — so the three sums of squares are not orthogonal and their total can
    # exceed ss_total. Clamping that to zero used to hide it and then fabricated
    # a mean square out of the clamp. A negative residual is now stored as it
    # is, with decomposition_valid False.
    ss_residual = ss_total - ss_model - ss_budget - ss_interaction
    decomposition_valid = ss_residual >= 0

    # Eta-squared (% variance)
    eta2 = {}
    if ss_total > 0:
        eta2['model'] = float(ss_model / ss_total * 100)
        eta2['budget'] = float(ss_budget / ss_total * 100)
        eta2['interaction'] = float(ss_interaction / ss_total * 100)
        eta2['residual'] = float(ss_residual / ss_total * 100)
    else:
        eta2 = {'model': 0, 'budget': 0, 'interaction': 0, 'residual': 0}

    # Degrees of freedom. n_total counts configurations, so df_residual is the
    # number of configurations minus the number of filled (model, budget)
    # cells. F statistics and p-values were computed here and saved until
    # 2026-10-05; they are removed under the owner's rule of no significance
    # tests. The variance shares (eta2) above are what is kept.
    df_residual = n_total - len(models_set) * len(budgets_set)
    if not decomposition_valid:
        print(f"    ss_residual is negative ({ss_residual:.4g}): the "
              f"unbalanced design does not decompose orthogonally, so the "
              f"residual share below is negative and the other three "
              f"overlap.")

    result = {
        'n_observations': n_total,
        # One observation is one configuration, scored by c02's own search, not
        # on held-out rows.
        'observations': 'one per configuration, c02 selection score',
        'n_models': len(models_set),
        'n_budgets': len(budgets_set),
        'df_residual': int(df_residual),
        'ss_residual': float(ss_residual),
        'decomposition_valid': bool(decomposition_valid),
        # ANOVA values are computed over whatever metric c02 used (R² for
        # molecular, CI for DTA, AUROC/MCC for classification, Spearman for
        # antibody) — naming the key "r2" was wrong for non-molecular runs.
        'grand_mean_score': float(grand_mean),
        'eta2': eta2,
        'ss_total': float(ss_total),
        'icc': icc_results,
    }

    # The grand mean is in whatever metric c02 ran under — RMSE on the molecular
    # sets, not R². Name it. Fixed 2026-09-10 (metric-direction audit).
    print(f"    Grand mean {metric_name or 'score'}: {grand_mean:.4f}")
    print(f"    {n_total} configurations, df_residual={df_residual}")
    print(f"    eta2 model:       {eta2['model']:.1f}%")
    print(f"    eta2 budget:      {eta2['budget']:.1f}%")
    print(f"    eta2 interaction: {eta2['interaction']:.1f}%")
    print(f"    eta2 residual:    {eta2['residual']:.1f}%")

    return result


# =========================================================================
# 9b: Information retention
# =========================================================================

def _run_information_retention(data, budgets, dataset_name=None,
                               n_folds=5):
    """9b: MI(X_selected, Y) and variance of X_full retained, vs budget.

    The second curve is stored as 'mi_xx_full' but is a variance share read
    off a ridge regression, not a mutual information; see the loop.

    At each budget, uses find_best_hybrid to discover the best hybrid config,
    then profiles its information content.

    Features are selected on each fold's training rows and both quantities are
    measured on that fold's held-out rows. Measured on the training rows they
    would count the match the selection was chosen for: the features were
    picked because they track those labels.

    n_folds matches the suite-wide CV count (config['n_splits'], =5 in full
    mode). The per-fold train_idx sets must equal the rest of the suite's so
    the shared hybrid_fold_cache (keyed on train_idx) is reused instead of
    cold-recomputed — previously hardcoded to 3, which guaranteed a miss
    against the 5-fold precompute (CACHE-5 / PCB-19).
    """
    print("\n  9b: Information Retention Curves")

    metric_name_9b, _, lower_is_better_9b = get_domain_metric(data)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {'budgets': [], 'mi_xy': [], 'mi_xx_full': [], 'best_configs': [],
               # Budgets whose recipe named internal evaluator 'match',
               # which this path resolves to lightgbm; see the note in
               # the loop below.
               'match_resolved_to': {},
               # Features are selected on each fold's training rows; both
               # curves are read on that fold's held-out rows.
               'measured_on': 'held-out rows',
               # Owner's set (2026-10-02): one curve per rule hybrid,
               # keyed 'model|family|slot', and every slot not filled.
               'rule_curves': {}, 'rule_missing': []}

    augs = data.get('augmentations')
    sample_X = next(iter(reps.values()))
    from sklearn.linear_model import Ridge as _Ridge

    # Owner's set (2026-10-02): the per-fold computation is one function so
    # the single recipe and each rule hybrid are measured identically.
    def _mi_for(alloc, imp, int_eval, aug_strat, budget):
        # Features are selected on the training rows; both numbers below are
        # then measured on the held-out rows of the same fold.
        fold_mi_xy = []
        fold_mi_xx = []
        # Features the hybrid was built with, per fold (2026-10-05).
        fold_widths_9b = []
        from experiments._shared import cv_split as _cv_split_9b
        from kirby.hybrid import apply_feature_selection as _afs_9b
        from kirby.hybrid import apply_augmentation_selection as _aas_9b
        _mode_9b = 'intermediary' if n_folds == 1 else 'full'
        for train_idx, val_idx in _cv_split_9b(groups, n_folds, sample_X, labels, mode=_mode_9b):
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            y_train = labels[train_idx]
            y_val = labels[val_idx]
            train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
            val_augs = {k: v[val_idx] for k, v in augs.items()} if augs else None
            try:
                X_hybrid, _finfo = create_hybrid_cached(
                    train_reps, y_train, train_idx,
                    groups=groups[train_idx], dataset_name=dataset_name,
                    allocation_method=alloc,
                    importance_method=imp,
                    budget=budget,
                    internal_evaluator=int_eval,
                    augmentations=train_augs,
                    augmentation_strategy=aug_strat)
                fold_widths_9b.append(int(X_hybrid.shape[1]))
                # Same features, applied to the rows that did not choose them.
                X_hybrid_val = _afs_9b(val_reps, _finfo)
                _aug_info = _finfo.get('augmentation_info')
                if _aug_info and val_augs:
                    _X_aug_val = _aas_9b(val_augs, _aug_info)
                    if _X_aug_val is not None:
                        X_hybrid_val = np.hstack([X_hybrid_val, _X_aug_val])
            except Exception as exc:
                # N7: dropping a fold here silently shrinks the MI average;
                # log so "fold genuinely empty" != "create_hybrid threw".
                print(f"      9b budget={budget} fold hybrid build failed: "
                      f"{type(exc).__name__}: {exc}")
                continue

            # MI(X_selected, Y) on the held-out rows. Corrected 2026-10-05:
            # this said nothing about the held-out rows shaped the space the
            # estimate is made in. The scaler and the PCA to at most 50
            # components below are fitted on the training rows. But
            # estimate_mutual_information_knn (_shared.py) then standardises
            # its input again on the held-out rows themselves, and if there
            # are more components than held-out rows // 5 it fits a second PCA
            # on the held-out rows down to that many. It then scores each
            # component against the label on its own (sklearn's per-feature
            # Kraskov estimator), SUMS the per-component values, and subtracts
            # the mean of that sum over 3 label permutations. So the value is a
            # sum over components, not a joint MI of the selected features,
            # and the number of components summed depends on the held-out
            # fold's size.
            n_comp = min(50, X_hybrid.shape[1], X_hybrid.shape[0] - 1,
                         X_hybrid_val.shape[0] - 1)
            try:
                if n_comp < 1:
                    raise ValueError(
                        f"held-out fold too small for MI: n_comp={n_comp}")
                _sc_9b = StandardScaler().fit(X_hybrid)
                _pca_9b = PCA(n_components=n_comp).fit(_sc_9b.transform(X_hybrid))
                X_red_val = _pca_9b.transform(_sc_9b.transform(X_hybrid_val))
                mi_xy = estimate_mutual_information_knn(X_red_val, y_val, k=5)
                fold_mi_xy.append(float(mi_xy))
            except Exception as exc:
                # N7: a thrown MI(X;Y) estimate drops out of the average
                # silently; log it so the curve's NaN is explained.
                print(f"      9b budget={budget} MI(X;Y) estimate failed: "
                      f"{type(exc).__name__}: {exc}")

            # Stored as 'mi_xx_full' but NOT a mutual information (corrected
            # 2026-10-05): the share of the full concatenation's variance,
            # over its top 20 principal components, that a ridge from the
            # selected features predicts on the held-out rows. Each
            # component's held-out R2 is floored at 0 and weighted by its
            # explained-variance ratio. Scalers, PCA and the ridge are all
            # fitted on the training rows.
            X_train_concat = np.hstack([v[train_idx] for v in reps.values()])
            X_val_concat = np.hstack([v[val_idx] for v in reps.values()])
            _sc_full = StandardScaler().fit(X_train_concat)
            _sc_hyb = StandardScaler().fit(X_hybrid)
            X_full_s = _sc_full.transform(X_train_concat)
            X_full_s_val = _sc_full.transform(X_val_concat)
            X_hyb_s = _sc_hyb.transform(X_hybrid)
            X_hyb_s_val = _sc_hyb.transform(X_hybrid_val)
            n_pcs = min(20, X_hybrid.shape[1], X_full_s.shape[0] - 1)
            try:
                pca_full = PCA(n_components=n_pcs)
                X_full_pcs = pca_full.fit_transform(X_full_s)
                X_full_pcs_val = pca_full.transform(X_full_s_val)
                explained_var_retained = 0
                for pc_i in range(n_pcs):
                    reg = _Ridge(alpha=1.0)
                    reg.fit(X_hyb_s, X_full_pcs[:, pc_i])
                    r2_pc = max(0, r2_score(X_full_pcs_val[:, pc_i],
                                            reg.predict(X_hyb_s_val)))
                    explained_var_retained += r2_pc * pca_full.explained_variance_ratio_[pc_i]
                fold_mi_xx.append(float(explained_var_retained))
            except Exception as exc:
                # N7: a thrown MI(X;X_full) approximation drops out silently;
                # log it so the curve's NaN is explained.
                print(f"      9b budget={budget} MI(X;X_full) approx failed: "
                      f"{type(exc).__name__}: {exc}")

        mi_xy_avg = float(np.mean(fold_mi_xy)) if fold_mi_xy else float('nan')
        mi_xx_avg = float(np.mean(fold_mi_xx)) if fold_mi_xx else float('nan')
        # Per-fold values behind the two means, and the width built per fold
        # (2026-10-05). A fold whose estimate raised is absent from its list,
        # so the lists can be shorter than the fold count.
        return mi_xy_avg, mi_xx_avg, {'mi_xy_per_fold': list(fold_mi_xy),
                                      'mi_xx_full_per_fold': list(fold_mi_xx),
                                      'hybrid_width_per_fold': list(fold_widths_9b)}

    # {budget: 'lightgbm'} for budgets whose recipe said internal
    # evaluator 'match', which this path cannot honour. See the note
    # inside the loop.
    _match_resolved = {}
    for budget in sorted(budgets):
        # Pool helper across (general, top) x (none, uncapped, capped) cache
        # files; pick winner across common downstream models for this budget.
        bp = None
        best_score = None
        for cand_model in ['rf', 'svm', 'lightgbm', 'dnn']:
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=cand_model, budget=budget,
                    augmentations=augs, 
                    dataset_name=dataset_name,
                    task_type=data['task_type'], metric=metric_name_9b)
            except Exception as exc:
                # N7: a thrown search lookup is NOT the same as "no result";
                # log it so a genuine NaN curve is distinguishable from a crash.
                print(f"      9b budget={budget} model={cand_model} search "
                      f"lookup failed: {type(exc).__name__}: {exc}")
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

        # No-search fallback: the engine's own current defaults. Was
        # 'greedy'/'treeshap'/'ridge' -- greedy is cut, and treeshap/ridge are
        # neither the engine default nor what the search would have picked.
        # ⛔ internal_evaluator WAS 'lightgbm' HERE and the engine's has been
        # 'rf' since 2026-09-16 (hybrid.py:34 records the change, hybrid.py:
        # create_hybrid signature carries it). The comment claimed to track the
        # engine and did not, so a budget with no pool blueprint was built
        # under a different evaluator from one that had a blueprint naming
        # 'rf' -- inside the same curve.
        alloc = bp.allocation_method if bp else 'performance_weighted'
        imp = bp.importance_method if bp else 'lightgbm_gain'
        int_eval = bp.internal_evaluator if bp else 'rf'
        # ⛔ 'match' CANNOT BE HONOURED ON THIS PATH. create_hybrid_cached
        # refuses any keyword it cannot hash into its fold key, and model,
        # metric_fn and lower_is_better are not in _HYBRID_FOLD_KEY_KWARGS
        # (_shared.py:107). With no model, create_hybrid resolves
        # internal_evaluator='match' to 'lightgbm' (hybrid.py:5952) whatever
        # downstream model the recipe was chosen for -- and with no metric_fn
        # the budget split is ranked on R2 or accuracy rather than the
        # dataset's own metric. Recorded rather than silently different.
        if int_eval == 'match':
            _match_resolved.setdefault(budget, 'lightgbm')
        aug_strat = bp.augmentation_strategy if bp else ('uncapped' if augs else 'none')
        # from_fallback says the recipe is the engine's default, not a recipe
        # any search picked. Without it a fallback budget reads like a searched
        # one in the saved file.
        results['best_configs'].append({'budget': budget, 'importance': imp,
                                        'allocation': alloc,
                                        'from_fallback': bp is None})
        if bp is None:
            print(f"      9b budget={budget}: no search found, using engine "
                  f"defaults {alloc}/{imp}/{int_eval}")

        mi_xy_avg, mi_xx_avg, _per_fold_9b = _mi_for(
            alloc, imp, int_eval, aug_strat, budget)

        # Owner's set (2026-10-02): the same two curves for every rule
        # hybrid, built at this budget with that hybrid's recipe.
        for _rm, _key, _h in _rule_entries_for_budget(
                data, dataset_name, DOWNSTREAM_MODELS, budget,
                metric_name_9b, results['rule_missing']):
            _bp = _h['blueprint']
            _ie = _bp.internal_evaluator
            if _ie == 'match':
                _match_resolved.setdefault(budget, 'lightgbm')
            _x, _xx, _pf = _mi_for(_bp.allocation_method, _bp.importance_method,
                                   _ie, _h.get('aug_condition') or 'none', budget)
            _c = results['rule_curves'].setdefault(
                _key, {**_rule_meta(_h), 'model': _rm, 'budgets': [],
                       'mi_xy': [], 'mi_xx_full': [],
                       'mi_xy_per_fold': [], 'mi_xx_full_per_fold': [],
                       'hybrid_width_per_fold': [],
                       'budget_relaxed_at': []})
            _c['budgets'].append(budget)
            _c['mi_xy'].append(_x)
            _c['mi_xx_full'].append(_xx)
            for _k9, _v9 in _pf.items():
                _c[_k9].append(_v9)
            _c['budget_relaxed_at'].append(bool(_h.get('budget_relaxed')))
            print(f"    [{_key}] budget={budget}: MI(X,Y)={_x:.4f}, "
                  f"info_retained={_xx:.4f}")

        results['budgets'].append(budget)
        results['mi_xy'].append(mi_xy_avg)
        results['mi_xx_full'].append(mi_xx_avg)
        for _k9, _v9 in _per_fold_9b.items():
            results.setdefault(_k9, []).append(_v9)

        print(f"    budget={budget}: MI(X,Y)={mi_xy_avg:.4f}, info_retained={mi_xx_avg:.4f}")

    # Find inflection: where MI(X,Y) gain drops below 10% of max gain
    if len(results['mi_xy']) >= 3:
        mi_arr = np.array(results['mi_xy'], dtype=float)
        gains = np.diff(mi_arr)
        # ⛔ TWO WAYS THIS USED TO GO WRONG.
        #
        # First, a nan anywhere in the curve made max_gain nan, `max_gain > 0`
        # False, and NEITHER the inflection line nor the "no inflection" line
        # printed -- a curve that could not be read looked like a curve nobody
        # asked about. nanmax and an explicit count fix that.
        #
        # Second, the mask was `gains < 0.1 * max_gain` and argmax took the
        # FIRST True. A negative gain -- the mutual-information estimate simply
        # fell between two budgets, which it does, being an estimate -- is less
        # than any positive threshold, so one noisy step at the small end fixed
        # the inflection there whatever the rest of the curve did. A plateau by
        # definition CONTINUES, so the first index is now the first one from
        # which every later gain is also below the threshold.
        n_nan = int(np.sum(~np.isfinite(gains)))
        finite_gains = gains[np.isfinite(gains)]
        max_gain = float(np.max(np.abs(finite_gains))) if len(finite_gains) else 0.0
        if n_nan:
            print(f"    {n_nan} of {len(gains)} budget steps have no finite "
                  f"mutual-information gain; they cannot start or end a "
                  f"plateau and are excluded.")
            results['n_nonfinite_gains'] = n_nan
        if max_gain > 0:
            thresh = 0.1 * max_gain
            plateau_mask = np.array(
                [bool(np.isfinite(g) and g < thresh) for g in gains])
            # Sustained: True from here to the end of the curve.
            sustained = np.array(
                [bool(plateau_mask[i:].all()) for i in range(len(plateau_mask))])
            if np.any(sustained):
                inflection_idx = int(np.argmax(sustained))
                results['inflection_rule'] = (
                    'first budget from which every later MI gain stays under '
                    '10% of the largest absolute gain')
                # gains[i] is the MI increase from budgets[i]→budgets[i+1].
                # When plateau_mask[i] is first True the plateau STARTS at
                # budgets[i] (the last productive budget) — that is the
                # inflection point, not budgets[i+1].
                results['inflection_budget'] = results['budgets'][inflection_idx]
                print(f"    Inflection point: budget ~{results['inflection_budget']}")
            else:
                print("    No inflection detected: no budget from which the "
                      "MI gain stays under 10% of the largest for the rest of "
                      "the curve")

    results['match_resolved_to'] = dict(_match_resolved)
    if _match_resolved:
        print(f"    {len(_match_resolved)} budget(s) named internal "
              f"evaluator 'match'; this path built them under lightgbm "
              f"because create_hybrid_cached cannot take a model")
    return results


# =========================================================================
# 9c: Effective rank analysis
# =========================================================================

def _run_effective_rank_analysis(data, budgets, dataset_name=None,
                                 n_folds=5):
    """9c: Effective rank of hybrid vs PCA vs random at each budget.

    At each budget, uses find_best_hybrid to discover the best config,
    then compares effective rank: best hybrid vs PCA vs random subset.

    n_folds matches the suite-wide CV count so the per-fold train_idx sets
    align with the shared hybrid_fold_cache (CACHE-5 / PCB-19) — previously
    hardcoded to 3.
    """
    print("\n  9c: Effective Rank Profile")

    metric_name_9c, _, lower_is_better_9c = get_domain_metric(data)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {'budgets': [], 'hybrid_rank': [], 'pca_rank': [], 'random_rank': [],
               # Budgets whose recipe named internal evaluator 'match',
               # which this path resolves to lightgbm; see the note in
               # the loop below.
               'match_resolved_to': {},
               'n_feat_per_fold': [], 'pca_failures': [], 'from_fallback': [],
               # Owner's set (2026-10-02): one hybrid-rank curve per rule
               # hybrid, keyed 'model|family|slot', and every slot not filled.
               'rule_curves': {}, 'rule_missing': []}

    augs = data.get('augmentations')
    sample_X = next(iter(reps.values()))
    from experiments._shared import cv_split as _cv_split_9c
    _mode_9c = 'intermediary' if n_folds == 1 else 'full'

    # Owner's set (2026-10-02): the per-fold hybrid rank is one function so
    # the single recipe and each rule hybrid are measured identically.
    def _hyb_rank_for(alloc, imp, int_eval, aug_strat, budget):
        fold_ranks = []
        hybrid_fold_widths = []
        for train_idx, _ in _cv_split_9c(groups, n_folds, sample_X, labels, mode=_mode_9c):
            try:
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
                X_hybrid, _ = create_hybrid_cached(
                    train_reps, labels[train_idx], train_idx,
                    groups=groups[train_idx], dataset_name=dataset_name,
                    allocation_method=alloc,
                    importance_method=imp,
                    budget=budget,
                    internal_evaluator=int_eval,
                    augmentations=train_augs,
                    augmentation_strategy=aug_strat)
                fold_ranks.append(compute_effective_rank(X_hybrid))
                hybrid_fold_widths.append(int(X_hybrid.shape[1]))
            except Exception:
                pass
        # fold_ranks returned as well (2026-10-05) so the per-fold effective
        # ranks behind the mean are saved.
        return (float(np.mean(fold_ranks)) if fold_ranks else float('nan'),
                hybrid_fold_widths, list(fold_ranks))

    # {budget: 'lightgbm'} for budgets whose recipe said internal
    # evaluator 'match', which this path cannot honour. See the note
    # inside the loop.
    _match_resolved = {}
    for budget in sorted(budgets):
        # The widest the PCA and random arms can be before the fold is known.
        # The row-count cut is applied per fold below, on that fold's training
        # rows, because PCA is fitted on the training rows only: sizing it from
        # the whole dataset's row count asked for more components than a fold
        # has rows whenever the budget exceeded the training rows, and PCA then
        # raised.
        n_feat_cap = min(budget, X_concat.shape[1])

        # Pool helper picks best (tier, aug_cond) across common models for this budget.
        bp = None
        best_score = None
        for cand_model in ['rf', 'svm', 'lightgbm', 'dnn']:
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=cand_model, budget=budget,
                    augmentations=augs, 
                    dataset_name=dataset_name,
                    task_type=data['task_type'], metric=metric_name_9c)
            except Exception as exc:
                # N7: log the thrown search lookup so a genuine NaN is
                # distinguishable from a crash (mirrors the 9b path).
                print(f"      9c budget={budget} model={cand_model} search "
                      f"lookup failed: {type(exc).__name__}: {exc}")
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

        # No-search fallback: the engine's own current defaults. Was
        # 'greedy'/'treeshap'/'ridge' -- greedy is cut, and treeshap/ridge are
        # neither the engine default nor what the search would have picked.
        # ⛔ internal_evaluator WAS 'lightgbm' HERE and the engine's has been
        # 'rf' since 2026-09-16 (hybrid.py:34 records the change, hybrid.py:
        # create_hybrid signature carries it). The comment claimed to track the
        # engine and did not, so a budget with no pool blueprint was built
        # under a different evaluator from one that had a blueprint naming
        # 'rf' -- inside the same curve.
        alloc = bp.allocation_method if bp else 'performance_weighted'
        imp = bp.importance_method if bp else 'lightgbm_gain'
        int_eval = bp.internal_evaluator if bp else 'rf'
        # ⛔ 'match' CANNOT BE HONOURED ON THIS PATH. create_hybrid_cached
        # refuses any keyword it cannot hash into its fold key, and model,
        # metric_fn and lower_is_better are not in _HYBRID_FOLD_KEY_KWARGS
        # (_shared.py:107). With no model, create_hybrid resolves
        # internal_evaluator='match' to 'lightgbm' (hybrid.py:5952) whatever
        # downstream model the recipe was chosen for -- and with no metric_fn
        # the budget split is ranked on R2 or accuracy rather than the
        # dataset's own metric. Recorded rather than silently different.
        if int_eval == 'match':
            _match_resolved.setdefault(budget, 'lightgbm')
        aug_strat = bp.augmentation_strategy if bp else ('uncapped' if augs else 'none')
        # Same marker as 9b: the recipe came from the engine's defaults, not a
        # search.
        results['from_fallback'].append(bp is None)
        if bp is None:
            print(f"      9c budget={budget}: no search found, using engine "
                  f"defaults {alloc}/{imp}/{int_eval}")

        # Per-fold hybrid creation to avoid label leakage in rank measurement
        # ⛔ THE THREE ARMS ARE MEASURED AT DIFFERENT WIDTHS. The hybrid's width
        # is whatever create_hybrid built -- it stops each representation at its
        # last feature scoring above zero, so it is usually under the budget --
        # while the PCA and random arms are cut to
        # min(n_feat_cap, training rows - 1) below. Effective rank rises with
        # width, so the comparison printed at the end mixes a width difference
        # into a rank difference. Only the PCA/random width was recorded; the
        # hybrid's is now recorded beside it so the two can be read together.
        hybrid_rank, hybrid_fold_widths, hybrid_fold_ranks = _hyb_rank_for(
            alloc, imp, int_eval, aug_strat, budget)

        # Owner's set (2026-10-02): the hybrid's effective rank for every
        # rule hybrid too, built at this budget with that hybrid's recipe.
        for _rm, _key, _h in _rule_entries_for_budget(
                data, dataset_name, DOWNSTREAM_MODELS, budget,
                metric_name_9c, results['rule_missing']):
            _bp = _h['blueprint']
            _ie = _bp.internal_evaluator
            if _ie == 'match':
                _match_resolved.setdefault(budget, 'lightgbm')
            _rk, _wd, _rkf = _hyb_rank_for(
                _bp.allocation_method, _bp.importance_method, _ie,
                _h.get('aug_condition') or 'none', budget)
            _c = results['rule_curves'].setdefault(
                _key, {**_rule_meta(_h), 'model': _rm, 'budgets': [],
                       'hybrid_rank': [], 'hybrid_rank_per_fold': [],
                       'hybrid_width_per_fold': [],
                       'budget_relaxed_at': []})
            _c['budgets'].append(budget)
            _c['hybrid_rank'].append(float(_rk))
            _c['hybrid_rank_per_fold'].append(_rkf)
            _c['hybrid_width_per_fold'].append(list(_wd))
            _c['budget_relaxed_at'].append(bool(_h.get('budget_relaxed')))
            print(f"    [{_key}] budget={budget}: hybrid rank={_rk:.1f}")

        # PCA — per-fold to match hybrid (avoids using val data in fit)
        pca_fold_ranks = []
        fold_widths = []
        pca_failures = []
        for fold_i, (train_idx, _) in enumerate(
                _cv_split_9c(groups, n_folds, sample_X, labels, mode=_mode_9c)):
            n_feat_fold = min(n_feat_cap, len(train_idx) - 1)
            fold_widths.append(int(n_feat_fold))
            try:
                pca = PCA(n_components=n_feat_fold)
                X_pca = pca.fit_transform(StandardScaler().fit_transform(X_concat[train_idx]))
                pca_fold_ranks.append(compute_effective_rank(X_pca))
            except Exception as exc:
                # Recorded, not swallowed: a NaN pca_rank used to be
                # indistinguishable from a PCA that never ran.
                msg = f"fold {fold_i} at {n_feat_fold} features: {type(exc).__name__}: {exc}"
                pca_failures.append(msg)
                print(f"      9c budget={budget} PCA failed, {msg}")
        pca_rank = float(np.mean(pca_fold_ranks)) if pca_fold_ranks else float('nan')

        # Random subset — per-fold to match hybrid, and cut to the same per-fold
        # width as PCA. The draw is made once at the widest width and each fold
        # takes the leading n_feat_fold of it, so the folds are nested subsets
        # of one draw rather than independent ones.
        random_fold_ranks = []
        rng = np.random.RandomState(42)
        random_idx = rng.choice(X_concat.shape[1], n_feat_cap, replace=False)
        for fold_i, (train_idx, _) in enumerate(
                _cv_split_9c(groups, n_folds, sample_X, labels, mode=_mode_9c)):
            try:
                X_random = X_concat[train_idx][:, random_idx[:fold_widths[fold_i]]]
                random_fold_ranks.append(compute_effective_rank(X_random))
            except Exception:
                pass
        random_rank = float(np.mean(random_fold_ranks)) if random_fold_ranks else float('nan')

        _hw = (float(np.mean(hybrid_fold_widths)) if hybrid_fold_widths
               else float('nan'))
        _pw = float(np.mean(fold_widths)) if fold_widths else float('nan')
        if hybrid_fold_widths and fold_widths and abs(_hw - _pw) >= 1.0:
            print(f"      9c budget={budget}: hybrid built {_hw:.0f} features, "
                  f"PCA/random cut to {_pw:.0f}. Effective rank rises with "
                  f"width, so the ranks below are not width-matched.")
        results['budgets'].append(budget)
        results['hybrid_rank'].append(float(hybrid_rank))
        results.setdefault('hybrid_width_per_fold', []).append(
            list(hybrid_fold_widths))
        results.setdefault('hybrid_width_mean', []).append(_hw)
        results.setdefault('pca_random_width_mean', []).append(_pw)
        results['pca_rank'].append(float(pca_rank))
        results['random_rank'].append(float(random_rank))
        # Per-fold effective ranks behind the three means (2026-10-05).
        results.setdefault('hybrid_rank_per_fold', []).append(hybrid_fold_ranks)
        results.setdefault('pca_rank_per_fold', []).append(
            [float(v) for v in pca_fold_ranks])
        results.setdefault('random_rank_per_fold', []).append(
            [float(v) for v in random_fold_ranks])
        results['n_feat_per_fold'].append(fold_widths)
        results['pca_failures'].append(pca_failures)

        print(f"    budget={budget}: hybrid={hybrid_rank:.1f}, pca={pca_rank:.1f}, random={random_rank:.1f}")

    results['match_resolved_to'] = dict(_match_resolved)
    if _match_resolved:
        print(f"    {len(_match_resolved)} budget(s) named internal "
              f"evaluator 'match'; this path built them under lightgbm "
              f"because create_hybrid_cached cannot take a model")
    return results


# =========================================================================
# 9d: Bias-variance proxy
# =========================================================================

def _run_bias_variance_analysis(data, models, budgets, n_folds, exp6_results=None, dataset_name=None):
    """9d: Bias-variance proxy from train-val gap across budgets."""
    print("\n  9d: Bias-Variance Proxy")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    mode = 'intermediary' if n_folds == 1 else 'full'

    results = {'models': {}}

    # If Exp 6 results are available, extract train-val gaps from there
    # ⛔ THIS BRANCH HAS NEVER KEPT A SINGLE CELL. It requires 'train_r2' on
    # each heatmap cell; c06 writes val_score (val_r2 before 2026-10-05),
    # val_std, n_features, importance,
    # allocation, evaluator, evaluator_label, fold_scores, n_folds_scored and
    # n_folds_asked, and NOTHING ELSE -- `grep -c train_r2` over c06 returns 0,
    # and c06's own docstring says train scores are not collected there. So
    # results['models'] stayed empty and the expensive from-scratch path below
    # ran every time, after this printed that it was using c06's data. That
    # path is 5 models x 8 budgets x 5 folds = 200 create_hybrid calls, the
    # bulk of c09.
    #
    # Left as a branch rather than deleted: if c06 ever records a training
    # score this becomes live again. But it now says what it found.
    if exp6_results and 'families' in exp6_results:
        print("    Exp 6 results found — checking for a training score")
        for fam_name, fam_data in exp6_results['families'].items():
            fam_bv = {}
            for var_name, budget_data in fam_data.get('heatmap', {}).items():
                for budget, metrics in budget_data.items():
                    if isinstance(metrics, dict) and 'train_r2' in metrics:
                        key = f"{var_name}_b{budget}"
                        # NOTE: 'train_r2'/'val_r2' hold the domain metric (RMSE
                        # for molecular), so direction matters. gap>0 == overfit
                        # and bias_proxy high == underfit, for BOTH directions.
                        # c06 renamed 'val_r2' to 'val_score' 2026-10-05;
                        # both read so an older c06 file still works.
                        tr = metrics['train_r2']
                        vl = metrics.get('val_score', metrics.get('val_r2'))
                        fam_bv[key] = {
                            'train_r2': tr, 'val_r2': vl,
                            'train_score': tr, 'val_score': vl,
                            'gap': (vl - tr) if lower_is_better else (tr - vl),
                            'bias_proxy': tr if lower_is_better else (1 - tr),
                            'budget': int(budget) if isinstance(budget, (int, float)) else budget,
                        }
            if fam_bv:
                results['models'][fam_name] = fam_bv
        if not results['models']:
            print("    Exp 6 has no 'train_r2' on any heatmap cell, so it "
                  "cannot supply a train-validation gap. Falling through to "
                  "the from-scratch path below. THIS IS THE NORMAL CASE: c06 "
                  "does not record training scores.")
            results['exp6_unusable_reason'] = (
                "c06 heatmap cells carry no 'train_r2'; c06 does not record "
                "training scores")
        if results['models']:
            results['recipe_selected_on'] = 'c06 per-variant search'
            results['val_score_is_held_out'] = True
            # Print summary
            for fam_name, fam_data in results['models'].items():
                print(f"    {fam_name}:")
                for key, vals in sorted(fam_data.items()):
                    print(f"      {key}: gap={vals['gap']:.3f}, bias={vals['bias_proxy']:.3f}")
            return results

    # Fallback: discover best config per model at each budget via the pool
    # helper (best across (tier, aug_cond) for each model), then measure
    # train-val gap with that config.
    #
    # The allocation method, importance method, internal evaluator and
    # augmentation strategy below come from a search ranked on every row of the
    # dataset, including the rows scored as validation here. So val_score runs
    # high and gap runs small, by however much those four choices spread, and
    # they do so at every budget alike. The flag below travels with the numbers
    # so no reader takes val_score for a held-out score.
    print("    Computing bias-variance from scratch (using pool helper per budget)")
    print("    NOTE: recipes come from a whole-dataset search; val_score here "
          "is not held out from the rows that chose them.")
    results['recipe_selected_on'] = 'all rows'
    results['val_score_is_held_out'] = False
    # Owner's set (2026-10-02): one entry per rule hybrid, keyed
    # 'model|family|slot', and every slot not filled.
    results['rule_models'] = {}
    results['rule_missing'] = []
    augs = data.get('augmentations')
    model_names = list(models.keys())

    for budget in budgets:
        for model_name, model_factory in models.items():
            try:
                _s, _m = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=model_name, budget=budget,
                    augmentations=augs, 
                    dataset_name=dataset_name,
                    task_type=data['task_type'], metric=metric_name)
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
                # Fallback when no search is found: the engine's own current
                # defaults. Was 'greedy'/'treeshap'/'ridge' -- greedy is cut,
                # and treeshap/ridge are neither the engine default nor what
                # the search would have picked.
                alloc, imp, int_eval = ('performance_weighted',
                                        'lightgbm_gain', 'lightgbm')
                aug_strat = 'uncapped' if augs else 'none'
            _from_fallback = _m is None

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
                    augmentation_strategy=aug_strat)
                if model_name not in results['models']:
                    results['models'][model_name] = {}
                # train_r2/val_r2 hold the domain metric (RMSE for molecular);
                # gap>0 == overfit and bias_proxy high == underfit for BOTH
                # higher- and lower-is-better metrics.
                _gap = (val_r2 - train_r2) if lower_is_better else (train_r2 - val_r2)
                _bias = train_r2 if lower_is_better else (1 - train_r2)
                results['models'][model_name][budget] = {
                    'train_r2': train_r2, 'val_r2': val_r2,
                    'train_score': train_r2, 'val_score': val_r2,
                    # Spread over folds and the mean width built, 2026-10-05;
                    # evaluate_cv_with_train returned both and they were
                    # dropped. It returns no per-fold scores or per-fold
                    # widths, so those are not available here.
                    'val_score_std': val_std, 'train_score_std': train_std,
                    'n_features_mean': n_feat,
                    'gap': _gap,
                    'bias_proxy': _bias,
                    'budget': budget,
                    'importance': imp,
                    'allocation': alloc,
                    # These four came from a search over every row, so
                    # val_score is not held out from the rows that picked them.
                    'recipe_selected_on': 'all rows',
                    'val_score_is_held_out': False,
                    # True means no search was found and the recipe above is
                    # the engine's defaults.
                    'from_fallback': _from_fallback,
                }
                print(f"    {model_name}/b={budget}: gap={_gap:.3f}, "
                      f"bias={_bias:.3f}, val_{metric_name}={val_r2:.4f} [{imp}/{alloc}]"
                      f"{' (engine defaults, no search)' if _from_fallback else ''}")
            except Exception as e:
                print(f"    {model_name}/b={budget}: FAILED - {e}")

            # Owner's set (2026-10-02): the same train-validation gap for each
            # of this model's rule hybrids (top 2 general + top 1 distance).
            for _rm, _key, _h in _rule_entries_for_budget(
                    data, dataset_name, [model_name], budget, metric_name,
                    results['rule_missing']):
                _bp = _h['blueprint']
                try:
                    _v, _vs, _t, _ts, _nf = evaluate_cv_with_train(
                        reps, labels, groups, model_factory, n_folds,
                        allocation_method=_bp.allocation_method,
                        importance_method=_bp.importance_method,
                        budget=budget, mode=mode, metric_fn=metric_fn,
                        internal_evaluator=_bp.internal_evaluator,
                        augmentations=augs,
                        augmentation_strategy=(_h.get('aug_condition')
                                               or 'none'))
                except Exception as e:
                    results['rule_missing'].append(
                        f'{_key} budget={budget}: build or scoring failed '
                        f'({type(e).__name__}: {e})')
                    print(f"    [{_key}]/b={budget}: FAILED - {e}")
                    continue
                _gap = (_v - _t) if lower_is_better else (_t - _v)
                _bias = _t if lower_is_better else (1 - _t)
                results['rule_models'].setdefault(_key, {
                    **_rule_meta(_h), 'model': _rm, 'by_budget': {}}
                )['by_budget'][budget] = {
                    'train_score': _t, 'val_score': _v, 'gap': _gap,
                    'val_score_std': _vs, 'train_score_std': _ts,
                    'n_features_mean': _nf,
                    'bias_proxy': _bias, 'budget': budget,
                    'budget_relaxed': bool(_h.get('budget_relaxed')),
                    'recipe_selected_on': 'all rows',
                    'val_score_is_held_out': False,
                }
                print(f"    [{_key}]/b={budget}: gap={_gap:.3f}, "
                      f"bias={_bias:.3f}, val_{metric_name}={_v:.4f}")

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

    # ⛔ WHICH UPSTREAM FILES THIS RUN ACTUALLY READ. c09 decomposes c02's grid,
    # and run_ds.sh gives every experiment its own process, so run_all's
    # "move a dependant after its upstream" reorder no longer applies across
    # processes. Without this, c09 with c02 absent writes a file whose anova
    # block is simply missing -- and the resume check then matches that file's
    # fingerprint and skips c09 forever. The absence is now on the record.
    results['upstream'] = {
        'c02': {'dir': c02_results_dir, 'found': bool(exp2_results)},
        'c06': {'dir': c06_results_dir, 'found': bool(exp6_results)},
    }
    if not exp2_results:
        results['upstream']['anova_withheld_reason'] = (
            'c02 result file absent or unreadable, so the ANOVA over its grid '
            'was not computed. This is not "no variance explained".')

    if 'anova' in all_subs and exp2_results:
        results['anova'] = _run_anova_decomposition(
            exp2_results, metric_name=get_domain_metric(data)[0])

    if 'information' in all_subs:
        results['information'] = _run_information_retention(
            data, budgets, dataset_name=dataset_name, n_folds=n_folds)

    if 'effective_rank' in all_subs:
        results['effective_rank'] = _run_effective_rank_analysis(
            data, budgets, dataset_name=dataset_name, n_folds=n_folds)

    if 'bias_variance' in all_subs:
        results['bias_variance'] = _run_bias_variance_analysis(
            data, models, budgets, n_folds, exp6_results, dataset_name=dataset_name)

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
