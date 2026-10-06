#!/usr/bin/env python3
"""C09: Formal Complexity Analysis — ANOVA + information retention + effective rank + bias-variance.

Paper: Complexity
Conclusion: ANOVA: model 50%+, budget 25%+, interaction 10%+ variance.
Dependencies: c02 (reads JSON for ANOVA), c06 (reads JSON for bias-variance).

Per outer fold (aligned 2026-10-05, handoff 2.4 and 2.3 "Still open": standard
selection where it selects):
- Outer folds from kirby.find_best_hybrid.nested_cv_outer_folds at
  config['n_splits'] (1 = one held-out split, no spread over folds).
- One search per outer fold on that fold's training rows, on the dataset's
  decided grid (fold_standard_selection -> fold_searches; pruning off;
  augmentation only where decided_aug_conditions keeps it). Nothing is read
  from a whole-dataset search.
- Per decided model (DECISIONS.md Step 1; never knn), per budget in the
  decided budgets: the 2 best hybrids AT THAT BUDGET on the fold's training
  rows (a second hybrid whose selected features equal the first's is
  skipped for the next). Per decided model: the 2 best single
  representations, which have no budget and are measured once per fold.
- Each hybrid is rebuilt on the fold's training rows with build_fold_hybrid,
  with the per-representation scores of that same per-fold search (dirty-run
  bug 1) and with the downstream model and the dataset's metric, so an
  internal evaluator of 'match' is the model itself.
- 9b, 9c and 9d are measured on each built hybrid and each single
  representation; 9a reads c02's file and runs no search.
- No distance hybrid: c09 measures no neighbours or distances (handoff 2.3
  item 1). Before 2026-10-05 each model also had 1 distance hybrid.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_model_factories, parse_rep_args,
    estimate_mutual_information_knn, compute_effective_rank,
    get_domain_metric,
    fold_standard_selection, fold_searches,
    build_fold_hybrid, fold_hybrid_scoring_kwargs)
from kirby.scaling import FeatureScaler
from kirby.find_best_hybrid import nested_cv_outer_folds


EXPERIMENT_DIR = 'c09_formal_analysis'
SEED = 42
# Stamped into the result and into the per-fold sidecar's identity, so a
# partial file written by the whole-dataset version of this script is never
# read back as this one's (2026-10-05).
# 2026-10-06: 9b/9c top-level curves are per model (was best-of-models).
RESULT_FORMAT = 'c09_fold_standard_selection_per_model_2026_10_06'
# Fields of a top_hybrids entry written into each hybrid row.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')

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

    How much of the variance in the dataset's own metric the model, the
    budget and their interaction account for (shares only; no F or p).

    One configuration is one observation, and its observation is its own mean
    over folds. The five fold scores of one configuration are five readings of
    the same configuration, not five independent configurations, so counting
    each of them inflated the observation count fivefold and the p-values with
    it. Within-cell variance now comes from the several allocation and
    importance settings c02 searched at the same (model, budget).

    The observations are c02's own selection scores. Since c02 was aligned
    (2026-10-05) each configuration's 'fold_scores' are its training-row
    scores in each outer fold's own search (the mean over that search's inner
    folds), and 'score_mean' is their mean over outer folds
    (c02_complexity_grid.py, the 'configs' summary). They are not held-out
    scores, and the payload says so in 'observations'. c02 now holds only the
    dataset's decided models and budgets.
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
        # One observation is one configuration, scored by c02's own searches,
        # not on held-out rows.
        'observations': ('one per configuration: the mean over outer folds '
                         "of c02's per-fold search score on training rows"),
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
# 9b, 9c, 9d: the measurements on one feature matrix of one outer fold
# =========================================================================

def _info_retention(X_tr, X_va, y_va, X_full_tr, X_full_va):
    """9b on one outer fold: (mi_xy, mi_xx_full, errors).

    The features were chosen on the fold's training rows; both numbers are
    read on its held-out rows. Measured on the training rows they would count
    the match the selection was chosen for.

    mi_xy: MI(X_selected, Y) on the held-out rows. The scaler and the PCA to
    at most 50 components are fitted on the training rows. But
    estimate_mutual_information_knn (_shared.py) then standardises its input
    again on the held-out rows themselves, and if there are more components
    than held-out rows // 5 it fits a second PCA on the held-out rows down to
    that many. It then scores each component against the label on its own
    (sklearn's per-feature Kraskov estimator), SUMS the per-component values,
    and subtracts the mean of that sum over 3 label permutations. So the value
    is a sum over components, not a joint MI of the selected features, and the
    number of components summed depends on the held-out fold's size
    (corrected 2026-10-05).

    mi_xx_full: stored under that name but NOT a mutual information
    (corrected 2026-10-05): the share of the full concatenation's variance,
    over its top 20 principal components, that a ridge from the selected
    features predicts on the held-out rows. Each component's held-out R2 is
    floored at 0 and weighted by its explained-variance ratio. Scalers, PCA
    and the ridge are all fitted on the training rows. The ridge is a
    measuring tool here, not a downstream model.
    DECIDED 2026-10-06 (Adelaide delegated): this ridge-based curve is kept.

    Scaling: FeatureScaler since 2026-10-05 (0/1 features are not scaled,
    handoff 2.3 item 17); it was StandardScaler on every feature.
    """
    from sklearn.linear_model import Ridge as _Ridge
    errors = []
    mi_xy = None
    mi_xx = None

    n_comp = min(50, X_tr.shape[1], X_tr.shape[0] - 1, X_va.shape[0] - 1)
    try:
        if n_comp < 1:
            raise ValueError(f"held-out fold too small for MI: n_comp={n_comp}")
        _sc = FeatureScaler().fit(X_tr)
        _pca = PCA(n_components=n_comp).fit(_sc.transform(X_tr))
        X_red_va = _pca.transform(_sc.transform(X_va))
        mi_xy = float(estimate_mutual_information_knn(X_red_va, y_va, k=5))
    except Exception as exc:
        errors.append(f"MI(X;Y) estimate failed: {type(exc).__name__}: {exc}")

    try:
        _sc_full = FeatureScaler().fit(X_full_tr)
        _sc_h = FeatureScaler().fit(X_tr)
        X_full_s = _sc_full.transform(X_full_tr)
        X_full_s_va = _sc_full.transform(X_full_va)
        X_h_s = _sc_h.transform(X_tr)
        X_h_s_va = _sc_h.transform(X_va)
        n_pcs = min(20, X_tr.shape[1], X_full_s.shape[0] - 1)
        pca_full = PCA(n_components=n_pcs)
        X_full_pcs = pca_full.fit_transform(X_full_s)
        X_full_pcs_va = pca_full.transform(X_full_s_va)
        retained = 0.0
        for pc_i in range(n_pcs):
            reg = _Ridge(alpha=1.0)
            reg.fit(X_h_s, X_full_pcs[:, pc_i])
            r2_pc = max(0.0, r2_score(X_full_pcs_va[:, pc_i],
                                      reg.predict(X_h_s_va)))
            retained += r2_pc * pca_full.explained_variance_ratio_[pc_i]
        mi_xx = float(retained)
    except Exception as exc:
        errors.append(f"MI(X;X_full) approximation failed: "
                      f"{type(exc).__name__}: {exc}")
    return mi_xy, mi_xx, errors


def _effective_rank_centred_only(X):
    """Effective rank, exp(entropy of normalised singular values), with
    each feature centred but NOT re-scaled to unit variance.

    2026-10-05, owner delegated, 2026-10-05: used for the PCA arm only.
    compute_effective_rank (_shared.py) standardises every feature first. PCA
    components are uncorrelated, so after that standardisation every
    singular value is equal and the effective rank is the number of
    components whatever the variance spread (results audit C09a). Centring
    only keeps each component's variance, so this rank reflects how the
    variance is spread across the components. The SVD path matches
    compute_effective_rank otherwise (TruncatedSVD to 500 components above
    2000 features).

    The matrix keeps its own float precision (2026-10-05, handoff 2.3 item
    18: float32 representations); it was converted to float64 here.
    """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    n, d = Xc.shape
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(500, n - 1), random_state=42)
        svd.fit(Xc)
        sv = svd.singular_values_
    else:
        try:
            sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd as scipy_svd
            sv = scipy_svd(Xc, full_matrices=False, compute_uv=False,
                           lapack_driver='gesvd')
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _fit_scores(X_tr, X_va, y_tr, y_va, factory, metric_fn, lower_is_better):
    """9d on one outer fold: fit the downstream model on the training rows,
    score it on the training rows and on the held-out rows.

    Replaces evaluate_cv_with_train (2026-10-05), which built one recipe on
    every fold of its own split and returned only means and spreads. Here the
    recipe is this fold's own, so the fit is done per fold and both scores
    are kept per fold. Scores are the dataset's metric; gap > 0 is overfit and
    bias_proxy high is underfit, in both metric directions. FeatureScaler, as
    evaluate_cv_with_train used.
    """
    from kirby.find_best_hybrid import _predict_for_metric as _pfm
    sc = FeatureScaler()
    Xtr = sc.fit_transform(X_tr)
    Xva = sc.transform(X_va)
    model = factory()
    model.fit(Xtr, y_tr)
    tr = float(metric_fn(y_tr, _pfm(model, Xtr, metric_fn)))
    va = float(metric_fn(y_va, _pfm(model, Xva, metric_fn)))
    return {'train_score': tr, 'val_score': va,
            'gap': (va - tr) if lower_is_better else (tr - va),
            'bias_proxy': tr if lower_is_better else (1 - tr)}


def _measure_features(X_tr, X_va, y_va, X_full_tr, X_full_va, do):
    """The model-free measurements of one feature matrix: 9b and 9c."""
    out = {'n_features': int(np.asarray(X_tr).shape[1])}
    if 'information' in do:
        mi_xy, mi_xx, errs = _info_retention(X_tr, X_va, y_va,
                                             X_full_tr, X_full_va)
        out['mi_xy'] = mi_xy
        out['mi_xx_full'] = mi_xx
        if errs:
            out['information_errors'] = errs
    if 'effective_rank' in do:
        # On the training rows, as before 2026-10-05.
        try:
            out['effective_rank'] = float(compute_effective_rank(X_tr))
        except Exception as exc:
            out['effective_rank'] = None
            out['effective_rank_error'] = f'{type(exc).__name__}: {exc}'
    return out


def _search_failures(reps, labels, groups, tr, k, n_outer, dataset_name,
                     augs, task_type, metric_name):
    """The recorded failures of the search fold_standard_selection ran.

    Reads the in-process memo with the arguments fold_standard_selection
    passes (general tier, no models), so nothing is searched again.
    """
    try:
        srch = fold_searches(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            augmentations=augs, task_type=task_type, metric=metric_name,
            seed=SEED)
        return [{kk: (str(vv) if not isinstance(vv, (int, float, str,
                                                     type(None))) else vv)
                 for kk, vv in f.items()}
                for f in (getattr(srch, 'failures', None) or [])]
    except Exception as e:
        return [{'error': f'could not read the failure record: '
                          f'{type(e).__name__}: {e}'}]


def _run_fold(k, tr, va, *, reps, labels, groups, augs, dataset_name,
              n_outer, factories, budgets, task_type, metric, full_names, do):
    """Everything 9b, 9c and 9d measure on one outer fold.

    One search on the fold's training rows (memoised by fold_searches, so
    asking for the top hybrids at each budget does not search again). Per
    decided model: 2 single representations, and at each decided budget 2
    hybrids, each rebuilt on the training rows and measured.
    """
    metric_name, metric_fn, lower = metric
    tr = np.asarray(tr)
    va = np.asarray(va)
    tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
    va_reps = {r: np.asarray(X)[va] for r, X in reps.items()}
    y_tr, y_va = labels[tr], labels[va]
    g_tr = None if groups is None else np.asarray(groups)[tr]
    aug_tr = ({a: np.asarray(X)[tr] for a, X in augs.items()} if augs
              else None)
    aug_va = ({a: np.asarray(X)[va] for a, X in augs.items()} if augs
              else None)
    # The full concatenation 9b measures retention against and 9c draws its
    # PCA and random arms from: the dataset's general representations.
    X_full_tr = np.hstack([tr_reps[r] for r in full_names])
    X_full_va = np.hstack([va_reps[r] for r in full_names])

    rec = {'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower),
           'search_failures': [], 'shortfall': [],
           'base': {}, 'by_budget': []}

    rep_cache = {}      # rep -> model-free measurements (9b, 9c)
    failures_read = False
    for b in budgets:
        sel = fold_standard_selection(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            models=list(factories), augmentations=augs, task_type=task_type,
            metric=metric_name, seed=SEED, budget=b)

        failed = sorted(m for m, s in sel.items() if s is None)
        if failed and not failures_read:
            failures_read = True
            errs = _search_failures(reps, labels, groups, tr, k, n_outer,
                                    dataset_name, augs, task_type,
                                    metric_name)
            for m in failed:
                rec['search_failures'].append({
                    'dataset': dataset_name, 'fold': int(k), 'model': m,
                    'errors': errs})
                print(f"    fold {k} {m:8s} SEARCH FAILED, slots skipped: "
                      f"{[e.get('error') or e.get('error_type') for e in errs]}")
        if len(failed) == len(sel):
            rec['error_search'] = 'the per-fold search failed for every model'
            return rec

        # --- the 2 single representations per model (no budget) ----------
        for m, s in sel.items():
            if s is None or m in rec['base']:
                continue
            bd = s.get('base_detail') or {}
            rows = []
            for i, e in enumerate(bd.get('selected') or [], 1):
                rep = e['rep']
                row = {'slot': f'base{i}', 'rep': rep,
                       # The search's own mean over its inner folds, inside
                       # this outer fold's training rows.
                       'training_rows_score': e.get('score')}
                if rep not in rep_cache:
                    rep_cache[rep] = _measure_features(
                        tr_reps[rep], va_reps[rep], y_va,
                        X_full_tr, X_full_va, do)
                row.update(rep_cache[rep])
                if 'bias_variance' in do:
                    try:
                        row.update(_fit_scores(
                            tr_reps[rep], va_reps[rep], y_tr, y_va,
                            factories[m], metric_fn, lower))
                    except Exception as exc:
                        row['fit_error'] = f'{type(exc).__name__}: {exc}'
                rows.append(row)
                print(f"    fold {k} {m:8s} base{i}   {rep:24s} "
                      f"{_row_note(row, metric_name)}")
            if len(rows) < 2:
                rec['shortfall'].append({
                    'dataset': dataset_name, 'fold': int(k), 'model': m,
                    'budget': None, 'n_base': len(rows),
                    'reason': bd.get('shortfall_reason')})
            rec['base'][m] = rows

        # --- the 2 hybrids per model at this budget ------------------------
        brec = {'budget': int(b), 'models': {}}
        for m, s in sel.items():
            if s is None:
                continue
            factory = factories[m]
            if len(s['hybrids']) < 2:
                rec['shortfall'].append({
                    'dataset': dataset_name, 'fold': int(k), 'model': m,
                    'budget': int(b), 'n_hybrids': len(s['hybrids'])})
            hyb_rows = []
            for i, e in enumerate(s['hybrids'], 1):
                bp = e['blueprint']
                cond = e.get('aug_condition') or 'none'
                row = {'slot': f'hybrid{i}',
                       **{f: e.get(f) for f in HYBRID_FIELDS},
                       'aug_condition': cond,
                       'training_rows_score': e.get('score_mean')}
                try:
                    X_tr_h, X_va_h, fi = build_fold_hybrid(
                        tr_reps, va_reps, y_tr, bp,
                        augmentations_train=aug_tr if cond != 'none' else None,
                        augmentations_val=aug_va if cond != 'none' else None,
                        groups=g_tr,
                        # This fold's search measured these on this fold's
                        # training rows (dirty-run bug 1).
                        precomputed_alloc_scores=getattr(bp, 'rep_scores',
                                                         None),
                        **fold_hybrid_scoring_kwargs(
                            bp, model=factory(), metric_fn=metric_fn,
                            lower_is_better=lower))
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    print(f"    fold {k} {m:8s} hybrid{i} budget {b}: build "
                          f"failed: {row['error']}")
                    hyb_rows.append(row)
                    continue
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                row.update(_measure_features(X_tr_h, X_va_h, y_va,
                                             X_full_tr, X_full_va, do))
                if 'bias_variance' in do:
                    try:
                        row.update(_fit_scores(X_tr_h, X_va_h, y_tr, y_va,
                                               factory, metric_fn, lower))
                    except Exception as exc:
                        row['fit_error'] = f'{type(exc).__name__}: {exc}'
                hyb_rows.append(row)
                print(f"    fold {k} {m:8s} hybrid{i} budget {b:5d}: "
                      f"{row['n_features']} features "
                      f"({row['allocation_method']}, "
                      f"{row['importance_method']}, "
                      f"eval {row['internal_evaluator']}, "
                      f"rep_count {row['rep_count']}, {cond}) "
                      f"{_row_note(row, metric_name)}")
            brec['models'][m] = {
                'hybrids': hyb_rows,
                'hybrids_skipped_identical_features': [
                    x.get('label') for x in
                    (s.get('hybrids_skipped_identical_features') or [])]}

        # DECIDED 2026-10-06 (Adelaide delegated): the top-level 9b/9c
        # curves are reported per decided model, each from that model's own
        # first hybrid at this budget. The "best hybrid across models" pick
        # (by training-row score) that fed one curve is removed.

        # --- 9c: the PCA and random arms at this budget ---------------------
        if 'effective_rank' in do:
            brec.update(_pca_random_arms(X_full_tr, b, k))
            _hw = [row['n_features'] for mr in brec['models'].values()
                   for row in mr['hybrids'] if 'n_features' in row]
            if _hw and abs(float(np.mean(_hw)) - brec['pca_random_width']) >= 1.0:
                print(f"    fold {k} budget {b}: hybrids built "
                      f"{float(np.mean(_hw)):.0f} features on average, "
                      f"PCA/random cut to {brec['pca_random_width']}. "
                      f"Effective rank rises with width, so these ranks are "
                      f"not width-matched.")
        rec['by_budget'].append(brec)
    return rec


def _row_note(row, metric_name):
    """One short printed summary of a measured row."""
    parts = []
    if row.get('mi_xy') is not None:
        parts.append(f"MI(X,Y)={row['mi_xy']:.4f}")
    if row.get('effective_rank') is not None:
        parts.append(f"rank={row['effective_rank']:.1f}")
    if row.get('val_score') is not None:
        parts.append(f"held-out {metric_name}={row['val_score']:.4f} "
                     f"gap={row['gap']:.4f}")
    return ', '.join(parts)


def _pca_random_arms(X_full_tr, budget, fold_i):
    """9c's PCA and random arms on one fold's training rows at one budget.

    ⛔ THE ARMS ARE MEASURED AT DIFFERENT WIDTHS FROM THE HYBRIDS. A hybrid's
    width is whatever was built -- create_hybrid stops each representation at
    its last feature scoring above zero, so it is usually under the budget --
    while the PCA and random arms are cut to min(budget, features, training
    rows - 1). Effective rank rises with width, so a rank difference mixes in
    a width difference. Both widths are stored.

    The random draw is RandomState(42) at min(budget, features) and each
    fold takes its leading pca_random_width of it, so every fold draws the
    same features (as before 2026-10-05).
    """
    n_feat_cap = min(int(budget), X_full_tr.shape[1])
    width = int(min(n_feat_cap, X_full_tr.shape[0] - 1))
    out = {'pca_random_width': width, 'pca_rank_centred': None,
           'pca_error': None, 'random_rank': None, 'random_error': None}
    try:
        # FeatureScaler since 2026-10-05 (0/1 features not scaled).
        X_s = FeatureScaler().fit_transform(X_full_tr)
        X_pca = PCA(n_components=width).fit_transform(X_s)
        # Components are NOT re-scaled to unit variance before the rank
        # (owner delegated, 2026-10-05; see _effective_rank_centred_only).
        out['pca_rank_centred'] = _effective_rank_centred_only(X_pca)
    except Exception as exc:
        out['pca_error'] = (f"fold {fold_i} at {width} features: "
                            f"{type(exc).__name__}: {exc}")
        print(f"      9c budget={budget} PCA failed, {out['pca_error']}")
    try:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_full_tr.shape[1], n_feat_cap, replace=False)
        out['random_rank'] = float(compute_effective_rank(
            X_full_tr[:, idx[:width]]))
    except Exception as exc:
        # Recorded, not swallowed (it was a bare pass until 2026-10-05).
        out['random_error'] = f'{type(exc).__name__}: {exc}'
    return out


# =========================================================================
# Summaries over outer folds
# =========================================================================

def _finite(vals):
    return [float(v) for v in vals
            if v is not None and np.isfinite(v)]


def _mean_or_nan(vals):
    v = _finite(vals)
    return float(np.mean(v)) if v else float('nan')


def _spread(vals, n_outer):
    """Range over outer folds: per-fold values and their min and max only,
    no interval and no test."""
    v = _finite(vals)
    if n_outer == 1:
        return 'one fold, no spread'
    if len(v) < 2:
        return f'{len(v)} fold(s) measured, no spread'
    return {'min': float(min(v)), 'max': float(max(v))}


def _summarise_rows(per_fold, budgets, fields, n_outer):
    """{'curves': {'model|hybridN': per budget, per fold}, 'base':
    {'model|baseN': per fold}} for the given measurement fields."""
    curves = {}
    for b in budgets:
        for fr in per_fold:
            bb = next((x for x in fr.get('by_budget', [])
                       if x['budget'] == b), None)
            if bb is None:
                continue
            for m, mr in bb['models'].items():
                for row in mr['hybrids']:
                    c = curves.setdefault(f"{m}|{row['slot']}", {
                        'model': m, 'slot': row['slot'], 'by_budget': []})
                    d = next((x for x in c['by_budget'] if x['budget'] == b),
                             None)
                    if d is None:
                        d = {'budget': int(b), 'by_fold': []}
                        c['by_budget'].append(d)
                    d['by_fold'].append({
                        'fold': fr['fold'], 'label': row.get('label'),
                        'n_features': row.get('n_features'),
                        'error': row.get('error'),
                        **{f: row.get(f) for f in fields}})
    for c in curves.values():
        for d in c['by_budget']:
            for f in fields:
                vals = [x[f] for x in d['by_fold']]
                d[f'{f}_mean'] = _mean_or_nan(vals)
                d[f'{f}_spread'] = _spread(vals, n_outer)
    base = {}
    for fr in per_fold:
        for m, rows in (fr.get('base') or {}).items():
            for row in rows:
                d = base.setdefault(f"{m}|{row['slot']}", {
                    'model': m, 'slot': row['slot'], 'by_fold': []})
                d['by_fold'].append({'fold': fr['fold'], 'rep': row['rep'],
                                     'n_features': row.get('n_features'),
                                     **{f: row.get(f) for f in fields}})
    for d in base.values():
        for f in fields:
            vals = [x[f] for x in d['by_fold']]
            d[f'{f}_mean'] = _mean_or_nan(vals)
            d[f'{f}_spread'] = _spread(vals, n_outer)
    return curves, base


def _model_slot_values(per_fold, b, model, field, slot='hybrid1'):
    """Per fold at budget b: the field of this model's hybrid in this slot
    (None where that fold has none), and which hybrid it was.

    DECIDED 2026-10-06 (Adelaide delegated): the top-level curves are per
    model. 'hybrid1' is the model's best hybrid at that budget on the fold's
    training rows.
    """
    vals, picks = [], []
    for fr in per_fold:
        bb = next((x for x in fr.get('by_budget', []) if x['budget'] == b),
                  None)
        mr = (bb or {}).get('models', {}).get(model)
        row = next((r for r in (mr or {}).get('hybrids', [])
                    if r['slot'] == slot), None)
        if row is None or 'error' in row:
            vals.append(None)
            picks.append({'fold': fr['fold'], 'model': model, 'slot': slot,
                          'label': None if row is None else row.get('label'),
                          'error': ('no hybrid in this slot' if row is None
                                    else row.get('error'))})
            continue
        vals.append(row.get(field))
        picks.append({'fold': fr['fold'], 'model': model, 'slot': slot,
                      'label': row.get('label'),
                      'allocation': row.get('allocation_method'),
                      'importance': row.get('importance_method'),
                      'internal_evaluator': row.get('internal_evaluator'),
                      'rep_count': row.get('rep_count'),
                      'aug_condition': row.get('aug_condition'),
                      'n_features': row.get('n_features')})
    return vals, picks


def _peak_and_inflection(results):
    """Peak and inflection of the mean MI(X,Y) curve, in place."""
    # 2026-10-05, owner delegated, 2026-10-05: the budget (features) where
    # MI(X,Y) is highest, reported beside the inflection budget below. On
    # some datasets the inflection budget came after this peak.
    _mi_peak = np.array(results['mi_xy'], dtype=float)
    if _mi_peak.size and np.any(np.isfinite(_mi_peak)):
        _pk = int(np.nanargmax(_mi_peak))
        results['peak_budget'] = results['budgets'][_pk]
        results['peak_mi_xy'] = float(_mi_peak[_pk])
        print(f"    Peak MI(X,Y): budget {results['peak_budget']} "
              f"(MI={results['peak_mi_xy']:.4f})")
    else:
        results['peak_budget'] = None
        results['peak_mi_xy'] = None

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
                print(f"    Inflection point: budget ~{results['inflection_budget']} "
                      f"(peak MI at budget {results.get('peak_budget')})")
            else:
                print("    No inflection detected: no budget from which the "
                      "MI gain stays under 10% of the largest for the rest of "
                      "the curve")


def _summarise_information(per_fold, budgets, n_outer, models):
    """9b: MI(X_selected, Y) and variance of X_full retained, vs budget,
    one curve per decided model."""
    print("\n  9b: Information Retention Curves")
    fields = ('mi_xy', 'mi_xx_full')
    curves, base = _summarise_rows(per_fold, budgets, fields, n_outer)
    results = {'budgets': list(budgets),
               # Features are selected on each fold's training rows; both
               # curves are read on that fold's held-out rows.
               'measured_on': 'held-out rows',
               # DECIDED 2026-10-06 (Adelaide delegated): reported per model,
               # not best-of-models. mi_xx_full keeps its ridge measuring fit
               # (see _info_retention).
               'model_curve': ('per decided model: per fold and budget, that '
                               "model's first hybrid (its best on the fold's "
                               'training rows)'),
               'by_model': {},
               # Per model: 2 hybrids at each budget and 2 single
               # representations, all from the per-fold search.
               'curves': curves, 'base': base}
    for m in models:
        mres = {'budgets': list(budgets), 'mi_xy': [], 'mi_xx_full': [],
                'mi_xy_per_fold': [], 'mi_xx_full_per_fold': [],
                'hybrid_width_per_fold': [], 'configs': []}
        print(f"    model {m}:")
        for b in budgets:
            xy, picks = _model_slot_values(per_fold, b, m, 'mi_xy')
            xx, _ = _model_slot_values(per_fold, b, m, 'mi_xx_full')
            wd, _ = _model_slot_values(per_fold, b, m, 'n_features')
            mres['mi_xy'].append(_mean_or_nan(xy))
            mres['mi_xx_full'].append(_mean_or_nan(xx))
            mres['mi_xy_per_fold'].append(xy)
            mres['mi_xx_full_per_fold'].append(xx)
            mres['hybrid_width_per_fold'].append(wd)
            mres['configs'].append({'budget': int(b), 'by_fold': picks})
            print(f"      budget={b}: MI(X,Y)={mres['mi_xy'][-1]:.4f}, "
                  f"info_retained={mres['mi_xx_full'][-1]:.4f} "
                  f"({len(_finite(xy))} of {len(per_fold)} folds)")
        _peak_and_inflection(mres)
        results['by_model'][m] = mres
    return results


def _summarise_effective_rank(per_fold, budgets, n_outer, models):
    """9c: effective rank of hybrid vs PCA vs random at each budget. The
    hybrid curve is per decided model; PCA and random are model-free."""
    print("\n  9c: Effective Rank Profile")
    curves, base = _summarise_rows(per_fold, budgets, ('effective_rank',),
                                   n_outer)
    results = {'budgets': list(budgets),
               'pca_rank_centred': [], 'random_rank': [],
               'pca_rank_per_fold': [], 'random_rank_per_fold': [],
               'pca_random_width_mean': [],
               'n_feat_per_fold': [], 'pca_failures': [],
               'random_failures': [],
               # DECIDED 2026-10-06 (Adelaide delegated): the hybrid curve is
               # reported per model, not best-of-models.
               'model_curve': ('per decided model: per fold and budget, that '
                               "model's first hybrid (its best on the fold's "
                               'training rows)'),
               'by_model': {},
               'curves': curves, 'base': base,
               # 2026-10-05, owner delegated, 2026-10-05: how each arm's
               # features are scaled before its effective rank is taken.
               'effective_rank_scaling': {
                   'hybrid': ('each feature standardised to unit variance '
                              '(compute_effective_rank), as before'),
                   'random': ('each feature standardised to unit variance '
                              '(compute_effective_rank), as before'),
                   'pca': ('components centred, not re-scaled to unit '
                           'variance, so the rank reflects the variance '
                           'spread across components; before 2026-10-05 '
                           'they were standardised and the rank equalled the '
                           'number of components'),
               }}
    for b in budgets:
        bbs = [next((x for x in fr.get('by_budget', []) if x['budget'] == b),
                    {}) for fr in per_fold]
        pr = [bb.get('pca_rank_centred') for bb in bbs]
        rr = [bb.get('random_rank') for bb in bbs]
        pw = [bb.get('pca_random_width') for bb in bbs]
        results['pca_rank_centred'].append(_mean_or_nan(pr))
        results['random_rank'].append(_mean_or_nan(rr))
        results['pca_rank_per_fold'].append(pr)
        results['random_rank_per_fold'].append(rr)
        results['pca_random_width_mean'].append(_mean_or_nan(pw))
        results['n_feat_per_fold'].append(pw)
        results['pca_failures'].append(
            [bb.get('pca_error') for bb in bbs if bb.get('pca_error')])
        results['random_failures'].append(
            [bb.get('random_error') for bb in bbs if bb.get('random_error')])
        print(f"    budget={b}: pca={results['pca_rank_centred'][-1]:.1f}, "
              f"random={results['random_rank'][-1]:.1f}")
    for m in models:
        mres = {'hybrid_rank': [], 'hybrid_rank_per_fold': [],
                'hybrid_width_per_fold': [], 'hybrid_width_mean': [],
                'configs': []}
        for b in budgets:
            hr, picks = _model_slot_values(per_fold, b, m, 'effective_rank')
            hw, _ = _model_slot_values(per_fold, b, m, 'n_features')
            mres['hybrid_rank'].append(_mean_or_nan(hr))
            mres['hybrid_rank_per_fold'].append(hr)
            mres['hybrid_width_per_fold'].append(hw)
            mres['hybrid_width_mean'].append(_mean_or_nan(hw))
            mres['configs'].append({'budget': int(b), 'by_fold': picks})
            print(f"    {m} budget={b}: hybrid={mres['hybrid_rank'][-1]:.1f} "
                  f"({mres['hybrid_width_mean'][-1]:.0f} features)")
        results['by_model'][m] = mres
    return results


def _summarise_bias_variance(per_fold, budgets, n_outer, metric_name,
                             lower_is_better):
    """9d: train-to-held-out gap per model, per hybrid slot, per budget."""
    print("\n  9d: Bias-Variance Proxy")
    fields = ('train_score', 'val_score', 'gap', 'bias_proxy')
    curves, base = _summarise_rows(per_fold, budgets, fields, n_outer)
    # 'models' keeps its old shape, {model: {budget: ...}}, filled from each
    # model's first hybrid at that budget (the best of that model on the
    # fold's training rows). Every slot is under 'curves'.
    models = {}
    for key, c in curves.items():
        if c['slot'] != 'hybrid1':
            continue
        for d in c['by_budget']:
            models.setdefault(c['model'], {})[d['budget']] = {
                'train_score': d['train_score_mean'],
                'val_score': d['val_score_mean'],
                'gap': d['gap_mean'], 'bias_proxy': d['bias_proxy_mean'],
                'gap_spread': d['gap_spread'],
                'by_fold': d['by_fold'], 'budget': d['budget']}
            print(f"    {c['model']}/b={d['budget']}: "
                  f"gap={d['gap_mean']:.4f}, bias={d['bias_proxy_mean']:.4f}, "
                  f"held-out {metric_name}={d['val_score_mean']:.4f}")
    return {'models': models, 'curves': curves, 'base': base,
            'metric': metric_name, 'lower_is_better': bool(lower_is_better),
            # Every recipe was chosen on the fold's training rows; the
            # held-out rows took no part in choosing it.
            'recipe_selected_on': 'fold_train_rows',
            'val_score_is_held_out': True}


def _bias_variance_from_c06(exp6_results, lower_is_better):
    """9d from c06's file, if c06 recorded the training score 9d needs.

    ⛔ THIS HAS NEVER KEPT A SINGLE CELL. It requires 'train_r2' on each heatmap
    cell, and no version of c06 has written that key. Since 2026-10-05 c06
    writes 'train_score_by_fold' and 'held_out_score_by_fold' per cell, but
    its training score is the per-fold search's mean over inner folds, not a
    model fitted and scored on the same training rows, so it is not the
    quantity 9d computes and is not read here. Left as a branch, as before;
    it now returns ({}, reason) in every case seen.
    """
    out = {}
    if not exp6_results or 'families' not in exp6_results:
        return out, 'no c06 result was read'
    for fam_name, fam_data in exp6_results['families'].items():
        fam_bv = {}
        for var_name, budget_data in (fam_data.get('heatmap') or {}).items():
            for budget, metrics in budget_data.items():
                if isinstance(metrics, dict) and 'train_r2' in metrics:
                    tr = metrics['train_r2']
                    vl = metrics.get('val_score', metrics.get('val_r2'))
                    fam_bv[f"{var_name}_b{budget}"] = {
                        'train_score': tr, 'val_score': vl,
                        'gap': (vl - tr) if lower_is_better else (tr - vl),
                        'bias_proxy': tr if lower_is_better else (1 - tr),
                        'budget': budget}
        if fam_bv:
            out[fam_name] = fam_bv
    if not out:
        return out, ("c06 heatmap cells carry no 'train_r2'; c06's training "
                     "score is the search's inner-fold mean, not a fitted "
                     "model's training-row score")
    return out, None


# =========================================================================
# Main run function
# =========================================================================

def run(data, config, output_dir, dataset_name, sub_analyses=None,
        c02_results_dir=None, c06_results_dir=None):
    """Exp 9: Formal theoretical grounding with 4 sub-analyses."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: FORMAL COMPLEXITY ANALYSIS (per fold)")
    print("=" * 70)

    from experiments.decided_grid import decided_models
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)

    metric = get_domain_metric(data)
    metric_name, metric_fn, lower_is_better = metric
    task_type = data['task_type']
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    augs = data.get('augmentations') or None

    # Models: the dataset's two decided models (DECISIONS.md Step 1). knn is
    # never a general model here.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_dropped = [{'downstream_model': m,
                       'reason': 'no model factory at this dataset size'}
                      for m in models if m not in factories]
    # The decided budgets (features), from get_config / run_all's
    # decided_config. Was FULL_BUDGETS, 10 to 2000, on every dataset.
    budgets = sorted(int(b) for b in config['budgets'])

    # Representations: data['reps'] goes to the search unfiltered; the
    # full concatenation 9b and 9c measure against is the dataset's general
    # list only.
    general_names = list((DATASET_REPS.get(dataset_name) or {}).get(
        'general') or reps.keys())
    extra = sorted(set(reps) - set(general_names))
    if extra:
        print(f"  representations outside DATASET_REPS['{dataset_name}']"
              f"['general'] left out of the full concatenation: {extra}")
    full_names = sorted(r for r in reps if r in general_names)
    if not full_names:
        # None of the loaded representations is on the general list (a run
        # given its own --reps): measure against everything loaded, and say so.
        full_names = sorted(reps)
        print(f"  no loaded representation is on DATASET_REPS['{dataset_name}']"
              f"['general']; the full concatenation is every loaded one")

    all_subs = sub_analyses or ['anova', 'information', 'effective_rank', 'bias_variance']
    results = {'result_format': RESULT_FORMAT,
               'selection_scope': 'fold_train_rows',
               'metric_name': metric_name,
               'lower_is_better': bool(lower_is_better),
               'models': models, 'models_dropped': models_dropped,
               'budgets': budgets,
               'full_concatenation': full_names}

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
            exp2_results, metric_name=metric_name)

    fold_subs = [s for s in ('information', 'effective_rank', 'bias_variance')
                 if s in all_subs]
    if 'bias_variance' in fold_subs:
        _c06_bv, _c06_why = _bias_variance_from_c06(exp6_results,
                                                    lower_is_better)
        if _c06_bv:
            print("    9d: train-validation gaps read from c06")
            results['bias_variance'] = {
                'models': _c06_bv, 'recipe_selected_on': 'c06 per-variant search',
                'val_score_is_held_out': True}
            fold_subs.remove('bias_variance')
        elif exp6_results:
            print(f"    9d: c06 cannot supply a train-validation gap "
                  f"({_c06_why}); computed per fold below")
            results['bias_variance_c06_unusable_reason'] = _c06_why

    if not fold_subs:
        return results

    # Outer folds: one setting, config['n_splits'] (1 = one held-out split).
    n_requested = int(config['n_splits'])
    folds = nested_cv_outer_folds(reps, labels, groups, n_requested, SEED,
                                  task_type)
    n_outer = len(folds)
    print(f"  models {models}; budgets {budgets} features; {len(reps)} "
          f"representations; {n_outer} outer fold(s) (asked {n_requested}); "
          f"measuring {fold_subs}")

    # Each finished fold is stored at once and read back on the next attempt.
    # RESULT_FORMAT in the identity: a sidecar from an earlier version of
    # this script is moved aside, never read back.
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, config,
                     result_format=RESULT_FORMAT, sub_analyses=fold_subs))

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        if ck.done(fold=k):
            rec = ck.get(fold=k)
            print(f"  fold {k}: finished in an earlier attempt, read back")
        else:
            print(f"\n  --- outer fold {k + 1}/{n_outer}: {len(tr)} training "
                  f"rows, {len(va)} held out ---")
            rec = _run_fold(
                k, tr, va, reps=reps, labels=labels, groups=groups, augs=augs,
                dataset_name=dataset_name, n_outer=n_outer,
                factories=factories, budgets=budgets, task_type=task_type,
                metric=metric, full_names=full_names, do=fold_subs)
            if 'error_search' not in rec:
                ck.put(convert_numpy(rec), fold=k)
        per_fold.append(rec)

    ok = [r for r in per_fold if 'error_search' not in r]
    results.update({
        'n_outer_folds': n_outer,
        'n_outer_folds_asked': n_requested,
        'n_outer_folds_scored': len(ok),
        'per_fold': per_fold,
        'search_failures': [f for r in per_fold
                            for f in r.get('search_failures', [])],
        'shortfall': [s for r in per_fold for s in r.get('shortfall', [])],
    })

    if 'information' in fold_subs:
        results['information'] = _summarise_information(
            ok, budgets, n_outer, list(factories))
    if 'effective_rank' in fold_subs:
        results['effective_rank'] = _summarise_effective_rank(
            ok, budgets, n_outer, list(factories))
    if 'bias_variance' in fold_subs:
        results['bias_variance'] = _summarise_bias_variance(
            ok, budgets, n_outer, metric_name, lower_is_better)

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
    # A comma-separated --reps is a list of representation names, not a preset.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  sub_analyses=args.sub_analyses,
                  c02_results_dir=args.c02_results,
                  c06_results_dir=args.c06_results)
    _mname, _, _lower = get_domain_metric(data)
    save_experiment_results(results, args.output, EXPERIMENT_DIR, args.dataset,
                            metric_name=_mname, lower_is_better=_lower)
