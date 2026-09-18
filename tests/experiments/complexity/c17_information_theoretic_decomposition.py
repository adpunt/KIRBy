#!/usr/bin/env python3
"""C17: Information-Theoretic Feature Selection Comparison.

Paper: Complexity
Core thesis: The hybrid selects features that maximize MI with the target per unit of
    dimension — it finds the minimum description length representation. Compare against
    PCA (variance-based), mRMR (MI-greedy), RFE (model-based), and random.

5 strategies × multiple budgets → MI density curves.

Usage:
    python c17_information_theoretic_decomposition.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    estimate_mutual_information_knn, compute_effective_rank,
    cv_split, get_domain_metric,
)
from experiments._shared import (
    build_fold_hybrid, get_best_search_for_downstream,
)


# =========================================================================
# mRMR (greedy max-relevance minus redundancy)
# =========================================================================

def _subsample_for_mi(X, y, max_rows=2000, seed=42):
    """COST-1: row-subsample for KSG mutual-information estimation.

    KSG MI converges well before ~2000 samples; capping rows makes each MI call
    O(const) instead of O(n^~1.08) — the lever that keeps c17 finite at large n
    — while leaving feature selection essentially unchanged. Returns (X, y)
    unchanged when n <= max_rows. Deterministic (seeded).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] <= max_rows:
        return X, y
    idx = np.random.RandomState(seed).choice(X.shape[0], max_rows, replace=False)
    return X[idx], y[idx]


def _greedy_mrmr(X, y, budget, k_mi=5, max_features=None):
    """Greedy mRMR feature selection using KSG MI estimator.

    At each step, select the feature that maximizes:
        MI(x_j, y) - (1/|S|) * sum_{x_i in S} MI(x_j, x_i)

    Args:
        max_features: If set, pre-filter to top max_features by univariate MI
            before running greedy selection. Enables mRMR on large feature sets.

    Returns selected feature indices.
    """
    n_features = X.shape[1]
    budget = min(budget, n_features)

    # COST-1: cap rows for MI estimation (rows only — feature indices and the
    # full-data downstream slice are unaffected). Keeps c17 finite at large n.
    X, y = _subsample_for_mi(X, y)

    # Pre-filter for large feature sets: rank all features by MI(x_j, y)
    # and keep top max_features for greedy selection
    feature_indices = np.arange(n_features)
    if max_features is not None and n_features > max_features:
        pre_mi = np.zeros(n_features)
        for j in range(n_features):
            try:
                pre_mi[j] = estimate_mutual_information_knn(
                    X[:, j:j+1], y, k=k_mi)
            except Exception:
                pre_mi[j] = 0.0
        top_idx = np.argsort(pre_mi)[-max_features:]
        feature_indices = top_idx
        X = X[:, top_idx]
        n_features = max_features

    # Pre-compute MI(x_j, y) for all features
    relevance = np.zeros(n_features)
    for j in range(n_features):
        try:
            relevance[j] = estimate_mutual_information_knn(
                X[:, j:j+1], y, k=k_mi)
        except Exception:
            relevance[j] = 0.0

    selected = []
    remaining = list(range(n_features))

    # First feature: max relevance
    best_first = max(remaining, key=lambda j: relevance[j])
    selected.append(best_first)
    remaining.remove(best_first)

    # Running redundancy total per remaining feature.
    # When a feature s is added to `selected`, increment each remaining
    # candidate j's running total by MI(X[:,j], X[:,s]). Score for j is
    # then relevance[j] - (running_redundancy[j] / len(selected)).
    # Avoids the O(budget² × n_remaining × MI_cost) re-summation that made
    # large budgets effectively non-terminating.
    running_redundancy = {}
    for j in remaining:
        try:
            running_redundancy[j] = estimate_mutual_information_knn(
                X[:, j:j+1], X[:, best_first], k=k_mi)
        except Exception:
            running_redundancy[j] = 0.0

    # Greedy selection
    for _ in range(budget - 1):
        if not remaining:
            break
        n_sel = len(selected)
        best_score = -np.inf
        best_j = remaining[0]

        for j in remaining:
            score = relevance[j] - running_redundancy[j] / n_sel
            if score > best_score:
                best_score = score
                best_j = j

        selected.append(best_j)
        remaining.remove(best_j)
        running_redundancy.pop(best_j, None)

        # Update running redundancy for the remaining candidates against
        # the newly-selected feature.
        for j in remaining:
            try:
                running_redundancy[j] += estimate_mutual_information_knn(
                    X[:, j:j+1], X[:, best_j], k=k_mi)
            except Exception:
                pass

    # Map back to original feature indices if pre-filtered
    return [int(feature_indices[s]) for s in selected]


# =========================================================================
# RFE (recursive feature elimination with RF)
# =========================================================================

def _rfe_select(X, y, budget):
    """Select top features via sklearn RFE with RF.

    REGRESSION-ONLY: uses RandomForestRegressor. run() gates the whole
    experiment to regression (CXA-5), so this is never reached with
    classification labels.
    """
    from sklearn.feature_selection import RFE
    budget = min(budget, X.shape[1])
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rfe = RFE(rf, n_features_to_select=budget, step=max(1, X.shape[1] // 10))
    rfe.fit(X, y)
    return list(np.where(rfe.support_)[0])


# =========================================================================
# STRATEGY EVALUATION
# =========================================================================

def _evaluate_strategy(select_fn, y, splits, metric_fn):
    """Per-fold evaluation of one feature-selection strategy.

    `select_fn(train_idx, val_idx)` must choose its features using the training
    rows only and return (X_train, X_val) restricted to them. Mutual
    information, its per-feature density, effective rank and the downstream
    score are then all read on the validation rows, which never took part in
    the choice. Three of the five strategies here pick features by how well
    they track the labels, so measured on the rows that picked them the number
    counts that fit.

    REGRESSION-ONLY: the downstream model is RandomForestRegressor. run()
    gates the experiment to regression (CXA-5), so the classification
    metric_fn / regressor-output mismatch cannot occur.
    """
    mis, densities, eranks, scores, n_feats = [], [], [], [], []
    for train_idx, val_idx in splits:
        X_tr, X_va = select_fn(train_idx, val_idx)
        n_feat = X_va.shape[1]
        if n_feat == 0:
            continue
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # MI with labels on the held-out rows — estimate_mutual_information_knn
        # handles its own PCA internally (max_dim=50), the same for every
        # strategy.
        _mi_X, _mi_y = _subsample_for_mi(X_va_s, y[val_idx])  # COST-1 row cap
        try:
            mi = float(estimate_mutual_information_knn(_mi_X, _mi_y, k=5))
        except Exception:
            mi = float('nan')
        mis.append(mi)
        densities.append(mi / n_feat if n_feat > 0 else 0.0)
        try:
            eranks.append(float(compute_effective_rank(X_va)))
        except Exception:
            eranks.append(float('nan'))

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y[train_idx])
        scores.append(float(metric_fn(y[val_idx], rf.predict(X_va_s))))
        n_feats.append(int(n_feat))

    def _m(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    return {
        'mi': _m(mis),
        'mi_density': _m(densities),
        'effective_rank': _m(eranks),
        # Renamed from 'downstream_r2' on 2026-09-10 (metric-direction audit):
        # the value is the dataset's OWN metric — RMSE on logd, caco2 and
        # herg_ki — and nothing in the repo read the old name. The direction
        # is stored beside it as 'lower_is_better' in the results dict.
        'downstream_score': _m(scores),
        'n_features': int(np.mean(n_feats)) if n_feats else 0,
        'fold_scores': scores,
        'fold_mi_density': densities,
        'n_folds_used': len(scores),
        'measured_on': 'held-out rows',
    }


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C17: Information-theoretic feature selection comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C17: INFORMATION-THEORETIC DECOMPOSITION")
    print("=" * 70)

    task_type = data['task_type']
    if task_type != 'regression':
        # _rfe_select / _evaluate_strategy hardcode RandomForestRegressor.
        # Mirroring c04: gate the whole experiment to regression.
        # Clean skip sentinel (no 'error' key) so downstream collectors do not
        # treat a deliberate regression-only skip as a failure.
        print(f"  Skipping: c17 is regression-only (task_type={task_type}).")
        return {'skipped': True, 'reason': 'regression_only', 'task_type': task_type}

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # Concatenate all reps
    X_concat = np.hstack([v for v in reps.values()])
    total_dim = X_concat.shape[1]

    # Canonical budget list (matches FULL_BUDGETS used by precompute).
    # Hardcoded [25, 50, 100, 200] was divergent — 200 was never searched by
    # precompute, so blueprint lookups at b=200 returned None and the loop
    # silently skipped that column.
    # The `b <= n_samples` half of this filter is GONE since 2026-09-10:
    # find_best_hybrid's allow_budget_exceeds_n now defaults to True, so
    # precompute searches and stores every budget asked for, including ones
    # above the row count. Keeping the clause made this file ask for a shorter
    # list than the pool holds — on the small sets, three budgets out of eight —
    # which is the divergence that makes a blueprint lookup return None and
    # silently drop a column.
    from experiments._shared import FULL_BUDGETS
    budgets = [b for b in FULL_BUDGETS if b < total_dim]
    # mRMR is O(budget * n_features * n_samples). For large feature sets,
    # pre-filter to top 2000 features by univariate MI before greedy selection.
    mrmr_prefilter = 2000

    # ⛔ THE mRMR RANKING STOPS AT 200 FEATURES. Owner's decision 2026-09-18.
    #
    # The greedy loop picks features one at a time and, on each pick, measures
    # every remaining candidate against the one just chosen. Its cost is
    # therefore QUADRATIC in the budget it is asked for, and it is paid in
    # mutual-information estimates, not model fits -- so none of it shows up in
    # a fit count.
    #
    # Measured 2026-09-18 on one laptop core: one estimate on 2,000 molecules x
    # 1 feature is 74 ms. At a budget of 2,000 features the greedy loop alone is
    # about 2.0 million estimates per fold, 10.1 million over 5 folds, about 209
    # hours. At 200 features it is about 378,000 per fold.
    #
    # Nothing has ever run at 2,000. Every one of the 15 stored c17 results
    # (tests/results/**/c17_*/*.json) tops out at a budget of 200, so the
    # quadratic case was a cliff a wide dataset could fall off, not a cost
    # anything had paid. Naming the cap here is what stops that.
    #
    # WHY THE ARM STAYS. Over those 15 datasets and 60 (dataset, budget) cells,
    # scored on downstream R2, mrmr is best in 2 cells against 29 for rfe and 29
    # for hybrid, and its mean rank is 2.80 of 5 against 1.62 and 1.67. It is
    # third and it is not close -- but it beats pca and random in 58 of the 60,
    # so it is a real method losing to two better ones, which is a result worth
    # reporting. Cutting it would remove one of five arms; capping it removes a
    # cliff nothing was standing on.
    #
    # The OTHER four strategies are unaffected and still run at every budget.
    MRMR_MAX_BUDGET = 200

    print(f"  Total dim: {total_dim}, budgets: {budgets}")
    if total_dim > mrmr_prefilter:
        print(f"  mRMR: enabled with pre-filter to top {mrmr_prefilter} features")
    else:
        print(f"  mRMR: enabled (full feature set)")

    # One set of folds, shared by every strategy and every budget, so the five
    # strategies are compared on identical rows.
    splits = list(cv_split(groups, n_folds, X_concat, labels, mode=mode))
    labels = np.asarray(labels)

    # Each fold's scaler is fitted on that fold's training rows and applied to
    # its validation rows. Nothing is scaled over every row any more: the mean
    # and spread of the validation rows would otherwise reach the training
    # side through the scaler.
    fold_scalers = [StandardScaler().fit(X_concat[tr]) for tr, _va in splits]

    # COST-2: greedy mRMR is prefix-monotone — the budget-k selection is the
    # first k features of the full greedy ranking (the per-step argmax depends
    # only on already-selected features, not on the target budget; verified by
    # synthetic test in the fix that introduced this). So compute the full
    # ranking ONCE per fold at the largest budget and slice [:budget] per
    # budget, instead of re-running the O(budget·n_feat·n_samples) greedy from
    # scratch for every budget. Result-identical, ~Nbudgets× cheaper.
    #
    # The ranking is now built PER FOLD from that fold's training rows. Built
    # once over every row, it picked the features by how well they track labels
    # the validation rows then carry, which is the same defect the fold split
    # exists to prevent. That costs one greedy pass per fold instead of one.
    _mf = mrmr_prefilter if total_dim > mrmr_prefilter else None
    # COST-1 (signed off 2026-06-05): cap the rows used for the MI/mRMR
    # ranking at MRMR_MAX_ROWS. The KSG mutual-information estimator
    # converges well before ~2000 samples, so ranking on a fixed seeded
    # subsample is statistically equivalent and makes c17 cost independent
    # of n (the large-n wall-buster: ~thousands of h -> ~300h at n>=10k).
    # The returned indices are feature indices into the FULL feature axis.
    # This is a feature-selection row subsample, NOT an evaluation split.
    MRMR_MAX_ROWS = 2000
    fold_mrmr_ranking = []
    for _fi, (_tr, _va) in enumerate(splits):
        _rank = None
        if budgets:
            _Xtr_s = fold_scalers[_fi].transform(X_concat[_tr])
            _ytr = labels[_tr]
            if _Xtr_s.shape[0] > MRMR_MAX_ROWS:
                _ridx = np.random.RandomState(42).choice(
                    _Xtr_s.shape[0], MRMR_MAX_ROWS, replace=False)
                _X_rank, _y_rank = _Xtr_s[_ridx], _ytr[_ridx]
            else:
                _X_rank, _y_rank = _Xtr_s, _ytr
            try:
                # Capped at MRMR_MAX_BUDGET: the greedy loop is quadratic in
                # this number and it is paid in mutual-information estimates.
                # Above the cap the mrmr arm's ranking simply stops, so the
                # strategies it is compared against still cover every budget and
                # mrmr covers the ones it can afford.
                _mrmr_budget = min(max(budgets), MRMR_MAX_BUDGET)
                if max(budgets) > MRMR_MAX_BUDGET:
                    print(f"    mRMR ranking capped at {MRMR_MAX_BUDGET} "
                          f"features (budgets reach {max(budgets)}); the other "
                          f"four strategies are uncapped")
                _rank = _greedy_mrmr(_X_rank, _y_rank, budget=_mrmr_budget,
                                     k_mi=5, max_features=_mf)
            except Exception as _e:
                print(f"  mRMR ranking failed on fold {_fi}: {_e}")
        fold_mrmr_ranking.append(_rank)

    strategies = ['hybrid', 'pca', 'mrmr', 'rfe', 'random']

    # Every strategy now chooses its features on each fold's training rows and
    # is measured on that fold's validation rows. Before, hybrid, mRMR and
    # recursive elimination chose theirs using every row's label while PCA and
    # random used no label at all, so best_strategy favoured whichever had
    # already seen the labels it was scored against.

    results = {
        'metric': metric_name,
        # Which way that metric runs, so a reader of the file never has to
        # guess. Added 2026-09-10 (metric-direction audit).
        'lower_is_better': lower_is_better,
        'budgets': budgets,
        'total_dim': total_dim,
        'strategies': strategies,
        'n_folds': n_folds,
        'selection_rows': "each fold's training rows",
        'measured_on': 'held-out rows',
        'per_budget': {},
    }

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        budget_results = {}

        # The hybrid's blueprint comes from a search over every row — that is
        # a separate finding, recorded as recipe_selected_on below. Its
        # FEATURES are re-selected inside each fold from that fold's training
        # labels, which is what this loop fixes.
        augmentations = data.get('augmentations')
        hybrid_bp = None
        try:
            _hsearch, _hmeta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augmentations,
                data=data, dataset_name=dataset_name,
                task_type=task_type, metric=metric_name,
            )
            if _hmeta:
                hybrid_bp = _hmeta['best_config']['blueprint']
        except Exception as _he:
            print(f"    hybrid blueprint lookup failed: {_he}")

        def _fold_index(train_idx):
            for _i, (_tr, _va) in enumerate(splits):
                if len(_tr) == len(train_idx) and np.array_equal(_tr, train_idx):
                    return _i
            raise KeyError('fold not found')

        def _sel_hybrid(train_idx, val_idx):
            if hybrid_bp is None:
                raise RuntimeError('no hybrid blueprint for this budget')
            tr_reps = {k: v[train_idx] for k, v in reps.items()}
            va_reps = {k: v[val_idx] for k, v in reps.items()}
            tr_a = ({k: v[train_idx] for k, v in augmentations.items()}
                    if augmentations else None)
            va_a = ({k: v[val_idx] for k, v in augmentations.items()}
                    if augmentations else None)
            X_tr, X_va, _fi = build_fold_hybrid(
                tr_reps, va_reps, labels[train_idx], hybrid_bp,
                augmentations_train=tr_a, augmentations_val=va_a,
                budget_override=budget,
                groups=(groups[train_idx] if groups is not None else None),
                model=RandomForestRegressor(n_estimators=100, random_state=42,
                                            n_jobs=-1),
                metric_fn=metric_fn, lower_is_better=lower_is_better)
            return X_tr, X_va

        def _sel_pca(train_idx, val_idx):
            fi = _fold_index(train_idx)
            sc = fold_scalers[fi]
            Xtr_s = sc.transform(X_concat[train_idx])
            Xva_s = sc.transform(X_concat[val_idx])
            n_comp = min(budget, total_dim, len(train_idx) - 1)
            pca = PCA(n_components=n_comp, random_state=42).fit(Xtr_s)
            return pca.transform(Xtr_s), pca.transform(Xva_s)

        def _sel_mrmr(train_idx, val_idx):
            fi = _fold_index(train_idx)
            rank = fold_mrmr_ranking[fi]
            if rank is None:
                raise RuntimeError('mRMR ranking unavailable for this fold')
            # ⛔ REFUSE A BUDGET THE RANKING CANNOT FILL. Added 2026-09-18
            # with MRMR_MAX_BUDGET. `rank[:budget]` returns whatever it has, so
            # a ranking 200 long asked for 500 features would hand back 200 and
            # the row would be filed as a 500-feature mrmr arm — fewer features
            # than every other strategy in that cell, reported as the same.
            # Raising instead means the cell is absent, which the caller already
            # handles, and absent is readable where a short arm is not.
            if len(rank) < budget:
                raise RuntimeError(
                    f'mRMR ranking holds {len(rank)} features and this cell '
                    f'asks for {budget}. The ranking is capped at '
                    f'MRMR_MAX_BUDGET={MRMR_MAX_BUDGET} because the greedy loop '
                    f'is quadratic in it. The mrmr arm is absent above the cap; '
                    f'the other four strategies still run at this budget.')
            # COST-2: slice the prefix-monotone ranking instead of re-running
            # greedy mRMR at this budget (identical result).
            idxs = rank[:budget]
            sc = fold_scalers[fi]
            return (sc.transform(X_concat[train_idx])[:, idxs],
                    sc.transform(X_concat[val_idx])[:, idxs])

        def _sel_rfe(train_idx, val_idx):
            fi = _fold_index(train_idx)
            sc = fold_scalers[fi]
            Xtr_s = sc.transform(X_concat[train_idx])
            Xva_s = sc.transform(X_concat[val_idx])
            idxs = _rfe_select(Xtr_s, labels[train_idx], budget)
            return Xtr_s[:, idxs], Xva_s[:, idxs]

        def _sel_random(train_idx, val_idx):
            fi = _fold_index(train_idx)
            sc = fold_scalers[fi]
            rng = np.random.RandomState(42)
            idxs = rng.choice(total_dim, min(budget, total_dim), replace=False)
            return (sc.transform(X_concat[train_idx])[:, idxs],
                    sc.transform(X_concat[val_idx])[:, idxs])

        selectors = {'hybrid': _sel_hybrid, 'pca': _sel_pca, 'mrmr': _sel_mrmr,
                     'rfe': _sel_rfe, 'random': _sel_random}

        for strategy in strategies:
            print(f"    {strategy}...", end='', flush=True)

            try:
                metrics = _evaluate_strategy(selectors[strategy], labels,
                                             splits, metric_fn)
                if strategy == 'hybrid':
                    metrics['recipe_selected_on'] = 'all rows'
                    metrics['features_selected_on'] = "each fold's training rows"
                budget_results[strategy] = metrics
                print(f" MI={metrics['mi']:.3f}, density={metrics['mi_density']:.4f}, "
                      # downstream_score holds the dataset's OWN metric.
                      f"{metric_name}={metrics['downstream_score']:.4f}")

            except Exception as e:
                print(f" FAILED: {e}")
                budget_results[strategy] = {'error': str(e)}

        results['per_budget'][budget] = convert_numpy(budget_results)

    # Summary: which strategy has the highest information per feature, one
    # budget at a time. Every density here is measured on validation rows that
    # took no part in choosing the features, for all five strategies alike.
    #
    # There is deliberately no average across budgets. The MI estimate runs on
    # at most 50 principal components, so at a budget above 50 the numerator
    # stops growing with the budget while the denominator keeps growing. That
    # makes the number comparable between the five strategies at one budget and
    # meaningless between budgets, and averaging it mixes the two.
    per_budget_density = {}
    best_per_budget = {}
    for budget in budgets:
        cell = {}
        for strategy in strategies:
            d = results['per_budget'].get(budget, {}).get(strategy, {}).get('mi_density')
            cell[strategy] = (float(d) if d is not None and not np.isnan(d)
                              else float('nan'))
        per_budget_density[budget] = cell
        finite = {k: v for k, v in cell.items() if not np.isnan(v)}
        best_per_budget[budget] = (max(finite, key=finite.get) if finite else None)

    results['summary'] = {
        'mi_density_per_budget': per_budget_density,
        'best_strategy_per_budget': best_per_budget,
        'mi_density_measured_on': 'held-out rows',
        'mi_density_comparable_across_budgets': False,
    }
    for budget in budgets:
        print(f"\n  Budget={budget} MI density: {per_budget_density[budget]}")
        print(f"  Budget={budget} best strategy: {best_per_budget[budget]}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C17: Information-Theoretic Feature Selection Comparison')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'c17_information_theoretic_decomposition', args.dataset)
