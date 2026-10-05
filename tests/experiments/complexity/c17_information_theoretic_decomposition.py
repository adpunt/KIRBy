#!/usr/bin/env python3
"""C17: Information-Theoretic Feature Selection Comparison.

Paper: Complexity
Core thesis: The hybrid selects features that maximize MI with the target per unit of
    dimension — it finds the minimum description length representation. Compare against
    PCA (variance-based), mRMR (MI-greedy), RFE (model-based), and random.

5 strategies × multiple budgets → MI density curves. The hybrid strategy is
three arms since 2026-10-02 (owner's set: top 2 general + top 1 distance
hybrids for rf): hybrid, hybrid__general2, hybrid__distance1.

Usage:
    python c17_information_theoretic_decomposition.py --dataset logd --mode intermediary
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
# Ranking metrics need graded scores: an AUROC read off hard 0/1 labels
# collapses to the mean of sensitivity and specificity (hybrid.py:2661).
from kirby.find_best_hybrid import _predict_for_metric
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    estimate_mutual_information_knn, compute_effective_rank,
    cv_split, get_domain_metric,
)
from experiments._shared import (
    build_fold_hybrid, get_rule_hybrids, fold_hybrid_scoring_kwargs,
)

# Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per model,
# per family. c17 fits RandomForest only, so the model is 'rf' and there are
# three hybrid strategies instead of one. 'hybrid' keeps its old name and is
# general1, so a reader of per_budget[budget]['hybrid'] reads what it always
# read.
_C17_HYBRID_ARMS = {'general1': 'hybrid', 'general2': 'hybrid__general2',
                    'distance1': 'hybrid__distance1'}


# =========================================================================
# COST KNOBS (2026-09-25) — every default is the shipped value, so a server
# run with none of these set is bit-identical to before.
# =========================================================================

def _env_int(name, default):
    """Read an integer environment override, falling back to the shipped value."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        print(f"  [c17] ignoring {name}={raw!r} (not an integer); "
              f"using {default}")
        return default


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
        # COST-3 (2026-09-25): the pre-filter above already estimated
        # MI(x_j, y) for every feature it kept, on exactly these rows, and
        # MI(x_j, y) is what `relevance` holds. The loop below used to
        # re-estimate it, paying max_features estimates a second time for
        # numbers already in pre_mi. Carrying them over is exact -- same rows,
        # same estimator, same k -- and saves max_features estimates per fold.
        prefilter_relevance = pre_mi[top_idx]
    else:
        prefilter_relevance = None

    # MI(x_j, y) for every candidate feature.
    if prefilter_relevance is not None:
        relevance = prefilter_relevance
    else:
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
# HYBRID ARM: per-fold feature importance, computed once
# =========================================================================

_C17_FOLD_IMPORTANCE = {}


def _hybrid_fold_importance(fold_idx, train_reps, y_train, groups_train,
                            blueprint, augs_train, dataset_name):
    """Per-feature importance for one fold of the hybrid arm, computed once.

    WHAT THIS REPLACES. create_hybrid recomputes per-feature importance over
    the base representations on every call. That quantity depends on the fold
    and the importance method and on nothing else -- not the budget. c17 calls
    build_fold_hybrid once per (fold, budget), so with 3 folds and the budgets
    100 and 250 it paid for the same three importance vectors twice.

    Measured 2026-09-25 on flab_thermo_tresanco, 672 rows, 5 representations of
    1000 features each: the hybrid arm took 746 s for its 3 folds at budget 100,
    about 249 s per fold, and _shared.py:5258 records 261.9 s for one such build
    on this dataset with augmentation off. Handing the vector in removes the
    second pass entirely -- about 750 s of the run.

    SAME NUMBERS, NOT SIMILAR ONES. The recipe here is create_hybrid's own
    (hybrid.py:5988 and :6125): carve the fold's training rows with
    internal_holdout_split at its default 0.2, then run compute_feature_importance
    on the carve's training part with the blueprint's importance method.
    build_fold_hybrid passes neither validation_split nor apply_filters, so both
    defaults hold and no feature filter is in play.

    Before computing anything it looks for the vector find_best_hybrid's
    pre-pass already wrote to disk (find_best_hybrid.py:6456). That file is
    keyed by a hash of the exact matrices and labels the score is computed
    from, so a hit can only be the right vector and a miss only costs a fresh
    computation. On the run measured here that pre-pass reported "3/3
    successful, 3 read from disk" in 1.3 s.

    Returns None -- which puts create_hybrid back on its own computation -- in
    the three cases this has not been checked against:
      - augmentations are present (importance is then taken over a different
        set of matrices);
      - the recipe was searched over fewer representations than the pool holds,
        so build_fold_hybrid drops some before create_hybrid sees them;
      - the blueprint names no importance method.
    """
    if augs_train:
        return None
    _rc = getattr(blueprint, 'rep_count', None)
    if _rc is not None and int(_rc) < len(train_reps):
        return None
    method = getattr(blueprint, 'importance_method', None)
    if not method:
        return None

    key = (dataset_name, fold_idx, method)
    if key in _C17_FOLD_IMPORTANCE:
        return _C17_FOLD_IMPORTANCE[key]

    from kirby.hybrid import compute_feature_importance, internal_holdout_split
    y_train = np.asarray(y_train)
    inner_tr, _inner_va = internal_holdout_split(
        len(y_train), y_train, groups_train)
    reps_in = {k: np.asarray(v)[inner_tr] for k, v in train_reps.items()}
    y_in = y_train[inner_tr]

    imp = None
    if dataset_name:
        try:
            from kirby.find_best_hybrid import (
                _fold_input_hash, _fold_score_cache_path,
                _load_cache_with_meta)
            _h = _fold_input_hash(reps_in, y_in)
            imp = _load_cache_with_meta(
                _fold_score_cache_path(dataset_name, 'importance', fold_idx,
                                       method, _h),
                {'kind': 'fold_importance', 'method': method,
                 'fold_idx': fold_idx, 'input_hash': _h},
                kind='search', verbose=False)
            if imp is not None:
                print(f"      fold {fold_idx} importance ({method}): read the "
                      f"vector the search already wrote")
        except Exception as _ce:
            print(f"      fold {fold_idx} importance ({method}): the search's "
                  f"store could not be read ({_ce}); computing it here")
            imp = None

    if imp is None:
        _t0 = time.time()
        imp = compute_feature_importance(reps_in, y_in, method=method)
        print(f"      fold {fold_idx} importance ({method}): computed in "
              f"{time.time() - _t0:.1f}s")

    _C17_FOLD_IMPORTANCE[key] = imp
    return imp


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

        rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y[train_idx])
        scores.append(float(metric_fn(y[val_idx], _predict_for_metric(rf, X_va_s, metric_fn))))
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
        # 2026-10-05: features actually built, one per scored fold (same
        # order as fold_scores). 'n_features' is their mean, rounded down.
        'width_built_per_fold': [int(n) for n in n_feats],
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
    _t_run0 = time.time()

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
    #
    # ⛔ THE CANDIDATE COUNT IS WHAT c17 COSTS. Measured 2026-09-25 on
    # flab_thermo_tresanco (672 rows, whole dataset, nothing dropped), five
    # representations at 1000 features each, 3 folds, budgets 100 and 250.
    #
    # The greedy loop measures every surviving candidate against the feature it
    # just picked, so the number of mutual-information estimates it pays per
    # fold is about (features it builds) x (candidates the pre-filter kept):
    #   pre-filter over the whole width   ~5,000 estimates
    #   redundancy against the first pick ~2,000
    #   greedy loop, 99 further picks     ~193,000
    #   = about 200,000 estimates per fold, 600,000 over three folds.
    # One estimate on 448 training rows x 1 feature is ~12 ms on an idle
    # laptop core, which is the ~40 min/fold this experiment was measured at.
    #
    # ⛔ MEASURED AGAIN 2026-09-25 WHILE THE LAPTOP CARRIED 24 OTHER
    # EXPERIMENTS (load average 670): the SAME estimate took 547 ms, 45x the
    # idle figure. At the pre-filter set to 200 one fold's ranking had not
    # finished after 35 minutes. The count below is what this file controls;
    # how long a count takes is the machine's, and on a contended laptop no
    # setting of this knob brings c17 inside 40 minutes.
    #
    # KIRBY_C17_MRMR_PREFILTER sets how many candidates survive the
    # max-relevance pre-filter. It is c17's own condition, not a swept axis,
    # and the greedy cost is linear in it. Lowering it to 200 takes the greedy
    # loop from ~193,000 estimates per fold to ~14,900 and the fold total from
    # ~200,000 to ~20,000, about 10x.
    #
    # WHAT THAT COSTS, SAID PLAINLY: the mrmr arm then picks its 100 features
    # out of the 200 highest-relevance features instead of the 2,000 highest.
    # Its relevance ordering is unchanged; what shrinks is how much room the
    # redundancy term has to move a feature up. The other four strategies --
    # hybrid, pca, rfe, random -- still see the full width and are untouched.
    # NO ROW IS DROPPED by this or by anything else in c17: all 672 rows of
    # flab_thermo_tresanco are used, split into 3 group-aware folds.
    #
    # The default stays 2000, so a server run is unchanged.
    mrmr_prefilter = _env_int('KIRBY_C17_MRMR_PREFILTER', 2000)

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
    #
    # KIRBY_C17_MRMR_MAX_BUDGET overrides the cap. Default 200 = shipped value.
    MRMR_MAX_BUDGET = _env_int('KIRBY_C17_MRMR_MAX_BUDGET', 200)

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
    _t_mrmr0 = time.time()
    for _fi, (_tr, _va) in enumerate(splits):
        _t_fold0 = time.time()
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
                # BUILD TO THE LARGEST BUDGET A CELL CAN ACTUALLY SERVE,
                # which is the largest budget at or below the cap -- not the cap
                # itself. The greedy ranking is prefix-monotone, so a budget-k
                # cell reads the first k of it; a cell above MRMR_MAX_BUDGET
                # raises by design and reads nothing. Building to 200 when the
                # budgets are 10, 50, 250 and 1000 computed 150 of 200 picks per
                # fold and discarded every one. Measured on flab_thermo_tresanco:
                # the ranking is ~0.5 h per fold at 50 against ~7.8 h at 200, and
                # no reported cell changes.
                _servable = [b for b in budgets if b <= MRMR_MAX_BUDGET]
                _mrmr_budget = max(_servable) if _servable else 0
                if not _servable:
                    print(f"    mRMR: no budget at or below MRMR_MAX_BUDGET="
                          f"{MRMR_MAX_BUDGET} (budgets {sorted(budgets)}); the "
                          f"mrmr arm serves no cell and no ranking is built")
                    raise RuntimeError('no servable mrmr budget')
                print(f"    mRMR ranking built to {_mrmr_budget} features "
                      f"(the largest budget the mrmr arm can serve; cap is "
                      f"MRMR_MAX_BUDGET={MRMR_MAX_BUDGET}, budgets reach "
                      f"{max(budgets)})")
                _rank = _greedy_mrmr(_X_rank, _y_rank, budget=_mrmr_budget,
                                     k_mi=5, max_features=_mf)
            except Exception as _e:
                print(f"  mRMR ranking failed on fold {_fi}: {_e}")
        print(f"    [timing] mRMR ranking fold {_fi}: "
              f"{time.time() - _t_fold0:.1f}s")
        fold_mrmr_ranking.append(_rank)
    print(f"  [timing] mRMR ranking, all {len(splits)} folds: "
          f"{time.time() - _t_mrmr0:.1f}s")

    # Owner's set (2026-10-02): three hybrid arms, see _C17_HYBRID_ARMS.
    strategies = (list(_C17_HYBRID_ARMS.values())
                  + ['pca', 'mrmr', 'rfe', 'random'])

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
        # Owner's set (2026-10-02): the three hybrid recipes per budget, and
        # every slot the rule asked for and did not get, with the reason.
        'rule_hybrids_by_budget': {},
        'rule_base_reps_by_budget': {},
        'rule_missing': [],
    }

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        budget_results = {}

        # The hybrid's blueprint comes from a search over every row — that is
        # a separate finding, recorded as recipe_selected_on below. Its
        # FEATURES are re-selected inside each fold from that fold's training
        # labels, which is what this loop fixes.
        augmentations = data.get('augmentations')
        # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids for
        # rf, one strategy each. Above the largest distance budget (500) the
        # distance read ranks over every budget it holds and sets
        # budget_relaxed; the arm is still built at this cell's budget
        # (budget_override below), and the flag is recorded.
        hybrid_bp_by_arm = {}
        _t_bp0 = time.time()
        try:
            _rule = get_rule_hybrids(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augmentations,
                data=data, dataset_name=dataset_name,
                task_type=task_type, metric=metric_name,
            )
            _desc = []
            for _h in _rule['hybrids']:
                _arm = _C17_HYBRID_ARMS.get(_h.get('slot'))
                if _h.get('objective') not in (None, 'predictive'):
                    _arm = f"{_arm}__{_h.get('objective')}"
                    # 2026-10-05: a non-predictive hybrid (only read when
                    # KIRBY_POOL_SUBSET names a similarity family; c17 asks
                    # for the default, predictive only) gets a name that is
                    # not in `strategies`, so it is never scored. Say so
                    # instead of dropping it silently.
                    results['rule_missing'].append(
                        f"budget={budget}: {_h.get('slot')} "
                        f"objective={_h.get('objective')} hybrid was read "
                        f"but not run -- c17 scores predictive hybrids only "
                        f"(arm name {_arm} is not in its strategies)")
                if _arm and _h.get('blueprint') is not None:
                    hybrid_bp_by_arm[_arm] = _h['blueprint']
                _desc.append({k: _h.get(k) for k in (
                    'slot', 'lineage', 'tier', 'aug_condition', 'objective',
                    'recipe', 'budget', 'budget_relaxed', 'score_other_folds',
                    'fold_scores')} | {'arm': _arm})
            results['rule_hybrids_by_budget'][budget] = _desc
            results['rule_base_reps_by_budget'][budget] = convert_numpy(
                _rule['base_reps'])
            results['rule_missing'] += [f'budget={budget}: {m}'
                                        for m in _rule['missing']]
        except Exception as _he:
            print(f"    hybrid blueprint lookup failed: {_he}")
            results['rule_missing'].append(
                f'budget={budget}: the read of the owner\'s hybrid set raised '
                f'{type(_he).__name__}: {_he}')
        print(f"    [timing] hybrid blueprint lookup, budget={budget}: "
              f"{time.time() - _t_bp0:.1f}s")

        def _fold_index(train_idx):
            for _i, (_tr, _va) in enumerate(splits):
                if len(_tr) == len(train_idx) and np.array_equal(_tr, train_idx):
                    return _i
            raise KeyError('fold not found')

        def _sel_hybrid(train_idx, val_idx, arm='hybrid'):
            hybrid_bp = hybrid_bp_by_arm.get(arm)
            if hybrid_bp is None:
                raise RuntimeError(f'no hybrid blueprint for {arm} at this budget')
            tr_reps = {k: v[train_idx] for k, v in reps.items()}
            va_reps = {k: v[val_idx] for k, v in reps.items()}
            tr_a = ({k: v[train_idx] for k, v in augmentations.items()}
                    if augmentations else None)
            va_a = ({k: v[val_idx] for k, v in augmentations.items()}
                    if augmentations else None)
            _groups_tr = (groups[train_idx] if groups is not None else None)
            # The importance over this fold's representations does not depend
            # on the budget, and this function is called once per budget. See
            # _hybrid_fold_importance.
            _imp = _hybrid_fold_importance(
                _fold_index(train_idx), tr_reps, labels[train_idx],
                _groups_tr, hybrid_bp, tr_a, dataset_name)
            X_tr, X_va, _fi = build_fold_hybrid(
                tr_reps, va_reps, labels[train_idx], hybrid_bp,
                augmentations_train=tr_a, augmentations_val=va_a,
                budget_override=budget,
                groups=_groups_tr,
                precomputed_importance=_imp,
                **fold_hybrid_scoring_kwargs(
                    hybrid_bp,
                    model=RandomForestRegressor(n_estimators=100,
                                                random_state=42, n_jobs=-1),
                    metric_fn=metric_fn, lower_is_better=lower_is_better))
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

        selectors = {'pca': _sel_pca, 'mrmr': _sel_mrmr,
                     'rfe': _sel_rfe, 'random': _sel_random}
        # Owner's set (2026-10-02): one selector per hybrid arm.
        for _arm in _C17_HYBRID_ARMS.values():
            selectors[_arm] = (lambda tr, va, _a=_arm: _sel_hybrid(tr, va, _a))

        for strategy in strategies:
            print(f"    {strategy}...", end='', flush=True)
            _t_strat0 = time.time()

            try:
                metrics = _evaluate_strategy(selectors[strategy], labels,
                                             splits, metric_fn)
                if strategy in _C17_HYBRID_ARMS.values():
                    metrics['recipe_selected_on'] = 'all rows'
                    metrics['features_selected_on'] = "each fold's training rows"
                budget_results[strategy] = metrics
                print(f" MI={metrics['mi']:.3f}, density={metrics['mi_density']:.4f}, "
                      # downstream_score holds the dataset's OWN metric.
                      f"{metric_name}={metrics['downstream_score']:.4f}")

            except Exception as e:
                print(f" FAILED: {e}")
                budget_results[strategy] = {'error': str(e)}
            print(f"      [timing] budget={budget} {strategy}: "
                  f"{time.time() - _t_strat0:.1f}s")

        results['per_budget'][budget] = convert_numpy(budget_results)

    # Summary, one budget at a time. Every density here is measured on validation rows that
    # took no part in choosing the features, for all five strategies alike.
    #
    # There is deliberately no average across budgets. The MI estimate runs on
    # at most 50 principal components, so at a budget above 50 the numerator
    # stops growing with the budget while the denominator keeps growing. That
    # makes the number comparable between the five strategies at one budget and
    # meaningless between budgets, and averaging it mixes the two.
    # 2026-10-05, owner delegated, 2026-10-05: the 'best_strategy_per_budget'
    # label is removed. It picked the highest MI per feature, which favours
    # whichever method built the fewest features (results audit C17a). The
    # summary now reports, for every method at every budget, the MI with the
    # labels on the held-out rows (nats, mean over folds) and the width
    # actually built (features, mean over folds and per fold). MI per feature
    # is kept beside them for reference only.
    per_budget_density = {}
    per_budget_mi = {}
    per_budget_width = {}
    per_budget_width_per_fold = {}
    for budget in budgets:
        cell, cell_mi, cell_w, cell_wf = {}, {}, {}, {}
        for strategy in strategies:
            _r = results['per_budget'].get(budget, {}).get(strategy, {})
            d = _r.get('mi_density')
            cell[strategy] = (float(d) if d is not None and not np.isnan(d)
                              else float('nan'))
            _mi = _r.get('mi')
            cell_mi[strategy] = (float(_mi) if _mi is not None
                                 and not np.isnan(_mi) else float('nan'))
            _wf = _r.get('width_built_per_fold') or []
            cell_wf[strategy] = [int(w) for w in _wf]
            cell_w[strategy] = (float(np.mean(_wf)) if _wf else float('nan'))
        per_budget_density[budget] = cell
        per_budget_mi[budget] = cell_mi
        per_budget_width[budget] = cell_w
        per_budget_width_per_fold[budget] = cell_wf

    results['summary'] = {
        'mi_per_budget': per_budget_mi,
        'mi_unit': 'nats, held-out rows, mean over folds',
        'width_built_per_budget': per_budget_width,
        'width_built_per_budget_per_fold': per_budget_width_per_fold,
        'mi_density_per_budget': per_budget_density,
        'mi_density_measured_on': 'held-out rows',
        'mi_density_comparable_across_budgets': False,
    }
    for budget in budgets:
        print(f"\n  Budget={budget}:")
        for strategy in strategies:
            print(f"    {strategy:<24} MI={per_budget_mi[budget][strategy]:.4f} "
                  f"width built={per_budget_width[budget][strategy]:.1f} features")

    print(f"\n  [timing] c17 total: {time.time() - _t_run0:.1f}s")
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
