#!/usr/bin/env python3
"""C17: Information-Theoretic Feature Selection Comparison.

Paper: Complexity
Core thesis: The hybrid selects features that maximize MI with the target per unit of
    dimension — it finds the minimum description length representation. Compare against
    PCA (variance-based), mRMR (MI-greedy), RFE (model-based), and random.

Per outer fold, per decided budget, per decided model: the 2 best hybrids of
that fold's own search at that budget (hybrid, hybrid__general2) against
PCA, mRMR, recursive elimination and random, each scored with that model.

2026-10-05 (handoff 2.3 "Still open", selection rule B for c17; 2.4 steps
1-15): the models are the dataset's two decided models (DECISIONS.md Step 1),
not rf alone. Every hybrid comes from a search on the outer fold's training
rows (fold_standard_selection, pruning off, the decided grid), not from the
whole-dataset search. The distance hybrid is gone: c17 measures no neighbours
or distances (handoff 2.3 item 1). Budgets are the dataset's decided general
budgets. Feature scaling uses FeatureScaler (0/1 features never scaled).

Usage:
    python c17_information_theoretic_decomposition.py --dataset logd --mode full
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# Ranking metrics need graded scores: an AUROC read off hard 0/1 labels
# collapses to the mean of sensitivity and specificity (hybrid.py:2661).
from kirby.find_best_hybrid import _predict_for_metric, nested_cv_outer_folds
from kirby.find_best_hybrid import nested_cv_cache_namespace
from kirby.result_cache import StaleResultDB as _StaleResultDB
# 2026-10-05 (handoff 2.3 item 17): 0/1 features pass through unscaled, every
# other feature is standardised, feature by feature. Replaces StandardScaler.
from kirby.scaling import FeatureScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    estimate_mutual_information_knn, compute_effective_rank,
    get_domain_metric, get_model_factories,
)
from experiments._shared import (
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection, fold_searches,
)

SEED = 42

# 2026-10-05: names the result layout. A file without it, or with another
# value, was written by the whole-dataset version (one model, rf; three
# hybrids including a distance hybrid; recipes chosen on every row).
RESULT_IDENTITY = 'c17_per_fold_decided_models_2026_10_05'

# 2026-10-05: per model, per fold, the 2 hybrids of that fold's search at the
# cell's budget, rank 1 and rank 2. 'hybrid' keeps its old name for rank 1.
# hybrid__distance1 is removed (handoff 2.3 item 1: distance hybrids only
# where an experiment measures something with neighbours or distances).
_C17_HYBRID_ARMS = ('hybrid', 'hybrid__general2')
# The four methods that choose features without a search. Their features do
# not depend on the downstream model, so they are chosen once per fold and
# budget and then scored with each decided model.
_C17_METHODS = ('pca', 'mrmr', 'rfe', 'random')

# Fields of a fold_standard_selection hybrid written to the result file.
_HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                  'internal_evaluator', 'rep_count', 'step_size', 'tune')


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
    (2026-10-05: it is now called once per fold, budget, decided model and
    hybrid rank, so the saving is larger. The key below is the outer fold and
    the importance method, both of which a per-fold recipe still fixes.)

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

def _feature_measures(X_tr, X_va, y_va):
    """What a feature set holds on one fold's held-out rows, before any model.

    The features were chosen on the fold's training rows only. Mutual
    information with the labels, its per-feature density and the effective
    rank are all read on the held-out rows, which took no part in the choice.
    The scaler is fitted on the training rows (FeatureScaler, 2026-10-05: 0/1
    features pass through unscaled). Returns None for an empty feature set.
    """
    n_feat = int(np.asarray(X_va).shape[1])
    if n_feat == 0:
        return None
    scaler = FeatureScaler().fit(X_tr)
    X_va_s = scaler.transform(X_va)
    # MI with labels on the held-out rows -- estimate_mutual_information_knn
    # handles its own PCA internally (max_dim=50), the same for every method.
    _mi_X, _mi_y = _subsample_for_mi(X_va_s, y_va)  # COST-1 row cap
    try:
        mi = float(estimate_mutual_information_knn(_mi_X, _mi_y, k=5))
    except Exception:
        mi = float('nan')
    try:
        erank = float(compute_effective_rank(X_va))
    except Exception:
        erank = float('nan')
    return {'mi': mi,
            'mi_density': (mi / n_feat) if n_feat > 0 else 0.0,
            'effective_rank': erank,
            'n_features': n_feat}


def _model_score(X_tr, X_va, y_tr, y_va, factory, metric_fn):
    """Fit one decided model on a fold's training rows, score its held-out rows.

    The dataset's OWN metric (get_domain_metric). The scaler is fitted on the
    training rows only (FeatureScaler, 2026-10-05). The model comes from
    get_model_factories, so it is the search's own model: rf with
    max_features='sqrt' (4415a50), svm with its labels scaled (fb1df54).
    """
    scaler = FeatureScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    m = factory()
    m.fit(X_tr_s, y_tr)
    return float(metric_fn(y_va, _predict_for_metric(m, X_va_s, metric_fn)))


def _cell_summary(cells):
    """Mean over folds of one (budget, model, method) cell.

    cells: one dict per fold that produced this cell, each with 'fold', 'mi',
    'mi_density', 'effective_rank', 'n_features', 'heldout_score'. Folds where
    the cell raised are listed under 'errors' and left out of the means.
    """
    ok = [c for c in cells if 'error' not in c]

    def _m(key):
        vals = [c.get(key) for c in ok]
        vals = [float(v) for v in vals
                if v is not None and not np.isnan(float(v))]
        return float(np.mean(vals)) if vals else float('nan')

    widths = [int(c['n_features']) for c in ok]
    out = {
        'mi': _m('mi'),
        'mi_density': _m('mi_density'),
        'effective_rank': _m('effective_rank'),
        # The dataset's OWN metric; its direction is 'lower_is_better' at the
        # top of the result file.
        'downstream_score': _m('heldout_score'),
        'n_features': int(np.mean(widths)) if widths else 0,
        # Features actually built, one per scored fold (same order as
        # fold_scores). 'n_features' is their mean, rounded down.
        'width_built_per_fold': widths,
        'fold_scores': [c.get('heldout_score') for c in ok],
        'fold_mi_density': [c.get('mi_density') for c in ok],
        'folds': [c['fold'] for c in ok],
        'n_folds_used': len(ok),
        'errors': [{'fold': c['fold'], 'error': c['error']}
                   for c in cells if 'error' in c],
        'measured_on': 'held-out rows',
    }
    if len(ok) == 1:
        out['fold_spread'] = 'one fold, no spread'
    return out

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
        # _rfe_select hardcodes RandomForestRegressor as its elimination
        # model. Mirroring c04: gate the whole experiment to regression.
        # Clean skip sentinel (no 'error' key) so downstream collectors do not
        # treat a deliberate regression-only skip as a failure.
        print(f"  Skipping: c17 is regression-only (task_type={task_type}).")
        return {'skipped': True, 'reason': 'regression_only', 'task_type': task_type}

    from experiments.decided_grid import decided_models, decided_search
    from experiments.complexity.complexity_helpers import DATASET_REPS

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    augmentations = data.get('augmentations')
    # 2026-10-05 (handoff 2.4 step 14): the outer fold count is the config's,
    # never a literal. 1 = one held-out split (nested_cv_outer_folds gives a
    # group-aware 80/20 split), and every mean below is then over one fold.
    n_folds_requested = int(config['n_splits'])
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # 2026-10-05 (handoff 2.4 step 9): the representations are the dataset's
    # pruned list, DATASET_REPS[ds]['general'], which run_all loads. The
    # searches and the four other methods see the same set. One outside that
    # list is named here, not dropped, so the hybrid and the other methods
    # still draw from the same representations.
    _general = list((DATASET_REPS.get(dataset_name) or {}).get('general') or [])
    reps_outside_general = sorted(r for r in reps if r not in _general)
    if reps_outside_general:
        print(f"  WARNING: representations outside DATASET_REPS['{dataset_name}']"
              f"['general']: {reps_outside_general}")

    # Concatenate all reps. Kept in the dtype the loader gives (float32,
    # handoff 2.3 item 18); nothing here converts to float64.
    rep_order = list(reps.keys())
    X_concat = np.hstack([reps[r] for r in rep_order])
    total_dim = X_concat.shape[1]

    # 2026-10-05 (handoff 2.4 steps 2-3, rule R1 and R3): the downstream
    # models are the dataset's two decided models, and the budgets are its
    # decided general-search budgets. The per-fold search only runs those
    # budgets, so a budget outside them would have no hybrid to compare.
    # Before: rf alone, and every FULL_BUDGETS entry below the total width.
    model_names = list(decided_models(dataset_name, 'general'))
    factories = get_model_factories(model_names, 'regression',
                                    n_samples=len(labels))
    models_skipped = [m for m in model_names if m not in factories]
    _decided_budgets = list(decided_search(dataset_name, 'general')['budgets'])
    _asked = list(config.get('budgets') or _decided_budgets)
    _asked = [int(b) for b in _asked if int(b) in _decided_budgets]
    # A budget at or above the total width selects every feature for every
    # method, so the methods cannot differ there. Same rule as before
    # (b < total_dim); the budgets it leaves out are recorded.
    budgets = [b for b in _asked if b < total_dim]
    budgets_not_run = [{'budget': b, 'reason': (
        f'at or above the total width of {total_dim} features: every method '
        f'would keep every feature')} for b in _asked if b >= total_dim]
    print(f"  Models: {list(factories)}"
          + (f" (size guard skipped {models_skipped})" if models_skipped else ''))
    for _nr in budgets_not_run:
        print(f"  Budget {_nr['budget']} not run: {_nr['reason']}")

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
    _mrmr_cells = [b for b in budgets if b <= MRMR_MAX_BUDGET]
    if budgets and not _mrmr_cells:
        print(f"  mRMR: every decided budget {budgets} is above "
              f"MRMR_MAX_BUDGET={MRMR_MAX_BUDGET} features, so the mrmr method "
              f"has no cell on this dataset")

    # One set of folds, shared by every method, model and budget, so they are
    # compared on identical rows. 2026-10-05: the folds are
    # nested_cv_outer_folds, the split the per-fold searches are filed under
    # (ocv{n}f{i}s{seed}), so fold k's hybrids come from a search on exactly
    # fold k's training rows. Before: cv_split, with the recipes read from the
    # whole-dataset search.
    folds = nested_cv_outer_folds(reps, labels, groups, n_folds_requested,
                                  SEED, 'regression')
    n_outer = len(folds)
    if n_outer != n_folds_requested:
        print(f"  {n_outer} outer folds (asked {n_folds_requested}; the "
              f"splitter reduced it to the number of groups)")

    # Each fold's scaler is fitted on that fold's training rows and applied to
    # its validation rows. Nothing is scaled over every row any more: the mean
    # and spread of the validation rows would otherwise reach the training
    # side through the scaler. 2026-10-05: FeatureScaler, 0/1 features unscaled.
    fold_scalers = [FeatureScaler().fit(X_concat[tr]) for tr, _va in folds]

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
    for _fi, (_tr, _va) in enumerate(folds):
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
    print(f"  [timing] mRMR ranking, all {len(folds)} folds: "
          f"{time.time() - _t_mrmr0:.1f}s")

    strategies = list(_C17_HYBRID_ARMS) + list(_C17_METHODS)

    # Every method chooses its features on each fold's training rows and is
    # measured on that fold's held-out rows. Since 2026-10-05 that includes
    # the hybrids' recipes: each comes from a search on the fold's training
    # rows, so no method has seen the held-out rows' labels.

    def _sel_pca(fi, train_idx, val_idx, budget):
        sc = fold_scalers[fi]
        Xtr_s = sc.transform(X_concat[train_idx])
        Xva_s = sc.transform(X_concat[val_idx])
        n_comp = min(budget, total_dim, len(train_idx) - 1)
        pca = PCA(n_components=n_comp, random_state=42).fit(Xtr_s)
        return pca.transform(Xtr_s), pca.transform(Xva_s)

    def _sel_mrmr(fi, train_idx, val_idx, budget):
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
                f'the other strategies still run at this budget.')
        # COST-2: slice the prefix-monotone ranking instead of re-running
        # greedy mRMR at this budget (identical result).
        idxs = rank[:budget]
        sc = fold_scalers[fi]
        return (sc.transform(X_concat[train_idx])[:, idxs],
                sc.transform(X_concat[val_idx])[:, idxs])

    def _sel_rfe(fi, train_idx, val_idx, budget):
        sc = fold_scalers[fi]
        Xtr_s = sc.transform(X_concat[train_idx])
        Xva_s = sc.transform(X_concat[val_idx])
        idxs = _rfe_select(Xtr_s, labels[train_idx], budget)
        return Xtr_s[:, idxs], Xva_s[:, idxs]

    def _sel_random(fi, train_idx, val_idx, budget):
        sc = fold_scalers[fi]
        rng = np.random.RandomState(42)
        idxs = rng.choice(total_dim, min(budget, total_dim), replace=False)
        return (sc.transform(X_concat[train_idx])[:, idxs],
                sc.transform(X_concat[val_idx])[:, idxs])

    method_selectors = {'pca': _sel_pca, 'mrmr': _sel_mrmr,
                        'rfe': _sel_rfe, 'random': _sel_random}

    per_fold = []
    # cells[budget][model][strategy] = one dict per fold
    cells = {b: {m: {s: [] for s in strategies} for m in factories}
             for b in budgets}

    for k, (tr, va) in enumerate(folds):
        _t_fold0 = time.time()
        y_tr, y_va = labels[tr], labels[va]
        tr_reps = {r: X[tr] for r, X in reps.items()}
        va_reps = {r: X[va] for r, X in reps.items()}
        g_tr = (np.asarray(groups)[tr] if groups is not None else None)
        fold_rec = {
            'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
            'cache_namespace': nested_cv_cache_namespace(k, n_outer, SEED),
            'selection_scope': 'fold_train_rows',
            'metric': metric_name, 'lower_is_better': bool(lower_is_better),
            'models': {m: {'base': None, 'budgets': {}} for m in factories},
            'methods_unavailable': {},
            'search_failures': [],
        }
        print(f"\n  Fold {k} of {n_outer}: {len(tr)} training rows, "
              f"{len(va)} held-out rows")

        for budget in budgets:
            print(f"\n    Budget={budget}:")
            # The 2 best hybrids per decided model AT THIS BUDGET, from one
            # search on this fold's training rows (decided grid, pruning off;
            # memoised, so every budget reads the same search). A second
            # hybrid whose selected features equal the first's is skipped for
            # the next one (handoff 2.1).
            _t_sel0 = time.time()
            try:
                sel = fold_standard_selection(
                    reps, labels, groups, train_idx=tr, fold_i=k,
                    n_outer_folds=n_outer, dataset_name=dataset_name,
                    models=list(factories), augmentations=augmentations,
                    task_type='regression', metric=metric_name, seed=SEED,
                    budget=budget)
            except _StaleResultDB:
                # A result store written by older code stops the run, as it
                # does inside search_on_fold.
                raise
            except Exception as _se:
                print(f"      per-fold search raised: "
                      f"{type(_se).__name__}: {_se}")
                sel = {m: None for m in factories}
                fold_rec['search_failures'].append({
                    'budget': budget,
                    'error': f'{type(_se).__name__}: {_se}'})
            print(f"      [timing] per-fold selection, budget={budget}: "
                  f"{time.time() - _t_sel0:.1f}s")
            if any(s is None for s in sel.values()) and not any(
                    f.get('recorded_from') == 'search'
                    for f in fold_rec['search_failures']):
                try:
                    _srch = fold_searches(
                        reps, labels, groups, train_idx=tr, fold_i=k,
                        n_outer_folds=n_outer, dataset_name=dataset_name,
                        augmentations=augmentations, task_type='regression',
                        metric=metric_name, seed=SEED)
                    for _f in (getattr(_srch, 'failures', None) or []):
                        fold_rec['search_failures'].append(
                            dict(convert_numpy(_f), recorded_from='search'))
                except Exception:
                    pass

            # The four methods without a search: features chosen once per
            # fold and budget, then scored with each decided model.
            method_X, method_meas = {}, {}
            for s in _C17_METHODS:
                _t0 = time.time()
                try:
                    X_tr_s, X_va_s = method_selectors[s](k, tr, va, budget)
                    meas = _feature_measures(X_tr_s, X_va_s, y_va)
                    if meas is None:
                        raise RuntimeError('no features selected')
                    method_X[s] = (X_tr_s, X_va_s)
                    method_meas[s] = meas
                except Exception as _me:
                    fold_rec['methods_unavailable'].setdefault(
                        str(budget), {})[s] = f'{type(_me).__name__}: {_me}'
                    print(f"      {s}: unavailable ({_me})")
                print(f"      [timing] budget={budget} {s} features: "
                      f"{time.time() - _t0:.1f}s")

            for m, factory in factories.items():
                s_m = sel.get(m)
                brec = {'hybrids': [], 'hybrids_skipped_identical_features': []}
                fold_rec['models'][m]['budgets'][str(budget)] = brec
                if s_m is None:
                    brec['search_failed'] = True
                    for arm in _C17_HYBRID_ARMS:
                        cells[budget][m][arm].append({
                            'fold': int(k),
                            'error': 'the per-fold search failed'})
                else:
                    if fold_rec['models'][m]['base'] is None:
                        # The fold search's 2 best single representations for
                        # this model, recorded only. c17 does not score them
                        # (its comparison is between ways of choosing a fixed
                        # number of features).
                        _bd = s_m.get('base_detail') or {}
                        fold_rec['models'][m]['base'] = [
                            {'rep': e.get('rep'),
                             'training_rows_score': e.get('score')}
                            for e in (_bd.get('selected') or [])]
                        if _bd.get('shortfall_reason'):
                            fold_rec['models'][m]['base_shortfall'] = \
                                _bd.get('shortfall_reason')
                    brec['hybrids_skipped_identical_features'] = [
                        x.get('label') for x in
                        (s_m.get('hybrids_skipped_identical_features') or [])]
                    hybs = list(s_m.get('hybrids') or [])
                    for i, arm in enumerate(_C17_HYBRID_ARMS):
                        _t0 = time.time()
                        if i >= len(hybs):
                            why = (f'the per-fold search gave {m} '
                                   f'{len(hybs)} hybrid(s) with distinct '
                                   f'features at budget {budget}')
                            cells[budget][m][arm].append(
                                {'fold': int(k), 'error': why})
                            brec['hybrids'].append({'arm': arm,
                                                    'error': why})
                            print(f"      {m:8s} {arm}: {why}")
                            continue
                        e = hybs[i]
                        bp = e['blueprint']
                        aug_cond = e.get('aug_condition') or 'none'
                        hrow = {'arm': arm,
                                **{f: e.get(f) for f in _HYBRID_FIELDS},
                                'aug_condition': aug_cond,
                                # Mean over the search's inner folds, inside
                                # this outer fold's training rows.
                                'training_rows_score': e.get('score_mean')}
                        try:
                            tr_a = va_a = None
                            if aug_cond != 'none' and augmentations:
                                tr_a = {a: X[tr] for a, X in augmentations.items()}
                                va_a = {a: X[va] for a, X in augmentations.items()}
                            # The importance over this fold's representations
                            # does not depend on the budget or the model. See
                            # _hybrid_fold_importance.
                            _imp = _hybrid_fold_importance(
                                k, tr_reps, y_tr, g_tr, bp, tr_a,
                                dataset_name)
                            X_tr_h, X_va_h, _fi = build_fold_hybrid(
                                tr_reps, va_reps, y_tr, bp,
                                augmentations_train=tr_a,
                                augmentations_val=va_a,
                                groups=g_tr,
                                precomputed_importance=_imp,
                                # The per-representation scores THIS fold's
                                # search measured on this fold's training rows
                                # (dirty-run bug 1), never whole-dataset ones.
                                precomputed_alloc_scores=getattr(
                                    bp, 'rep_scores', None),
                                **fold_hybrid_scoring_kwargs(
                                    bp, model=factory(), metric_fn=metric_fn,
                                    lower_is_better=lower_is_better))
                            hrow['features_by_rep'] = {
                                r: int(len(info['selected_indices']))
                                for r, info in (_fi or {}).items()
                                if isinstance(info, dict)
                                and 'selected_indices' in info}
                            meas = _feature_measures(X_tr_h, X_va_h, y_va)
                            if meas is None:
                                raise RuntimeError('the hybrid built 0 features')
                            score = _model_score(X_tr_h, X_va_h, y_tr, y_va,
                                                 factory, metric_fn)
                            cell = dict(meas, fold=int(k), heldout_score=score)
                            hrow.update(meas)
                            hrow['heldout_score'] = score
                            cells[budget][m][arm].append(cell)
                            print(f"      {m:8s} {arm:18s} "
                                  f"{meas['n_features']:5d} features "
                                  f"({hrow['allocation_method']}, "
                                  f"{hrow['importance_method']}, "
                                  f"eval {hrow['internal_evaluator']}, "
                                  f"rep_count {hrow['rep_count']}) "
                                  f"MI={meas['mi']:.3f} "
                                  f"{metric_name}={score:.4f}")
                        except Exception as _he:
                            why = f'{type(_he).__name__}: {_he}'
                            hrow['error'] = why
                            cells[budget][m][arm].append(
                                {'fold': int(k), 'error': why})
                            print(f"      {m:8s} {arm}: FAILED: {why}")
                        hrow['seconds'] = round(time.time() - _t0, 1)
                        brec['hybrids'].append(convert_numpy(hrow))

                for s in _C17_METHODS:
                    if s not in method_X:
                        cells[budget][m][s].append({
                            'fold': int(k),
                            'error': fold_rec['methods_unavailable']
                            .get(str(budget), {}).get(s, 'unavailable')})
                        continue
                    _t0 = time.time()
                    X_tr_s, X_va_s = method_X[s]
                    try:
                        score = _model_score(X_tr_s, X_va_s, y_tr, y_va,
                                             factory, metric_fn)
                        cells[budget][m][s].append(
                            dict(method_meas[s], fold=int(k),
                                 heldout_score=score))
                        print(f"      {m:8s} {s:18s} "
                              f"{method_meas[s]['n_features']:5d} features "
                              f"MI={method_meas[s]['mi']:.3f} "
                              f"{metric_name}={score:.4f} "
                              f"({time.time() - _t0:.1f}s)")
                    except Exception as _fe:
                        why = f'{type(_fe).__name__}: {_fe}'
                        cells[budget][m][s].append({'fold': int(k),
                                                    'error': why})
                        print(f"      {m:8s} {s}: FAILED: {why}")

        fold_rec['seconds'] = round(time.time() - _t_fold0, 1)
        per_fold.append(convert_numpy(fold_rec))
        print(f"  [timing] fold {k}: {fold_rec['seconds']:.1f}s")

    # Per budget, per model, per method: the mean over folds.
    per_budget = {
        b: {m: {s: _cell_summary(cells[b][m][s]) for s in strategies}
            for m in factories}
        for b in budgets}

    # Summary, one budget and one model at a time. Every number here is
    # measured on held-out rows that took no part in choosing the features,
    # for every method alike.
    #
    # There is deliberately no average across budgets. The MI estimate runs on
    # at most 50 principal components, so at a budget above 50 the numerator
    # stops growing with the budget while the denominator keeps growing. That
    # makes the number comparable between the methods at one budget and
    # meaningless between budgets, and averaging it mixes the two.
    # 2026-10-05, owner delegated, 2026-10-05: the 'best_strategy_per_budget'
    # label is removed. It picked the highest MI per feature, which favours
    # whichever method built the fewest features (results audit C17a). The
    # summary reports, for every method at every budget, the MI with the
    # labels on the held-out rows (nats, mean over folds) and the width
    # actually built (features, mean over folds and per fold). MI per feature
    # is kept beside them for reference only.
    # 2026-10-05: one more level, the decided model. The four methods'
    # features, and so their MI and width, are the same under both models;
    # the hybrids differ because each model has its own 2.
    def _f(v):
        return (float(v) if v is not None and not np.isnan(v)
                else float('nan'))

    summary = {
        'mi_per_budget': {b: {m: {s: _f(per_budget[b][m][s]['mi'])
                                  for s in strategies} for m in factories}
                          for b in budgets},
        'mi_unit': 'nats, held-out rows, mean over folds',
        'heldout_score_per_budget': {
            b: {m: {s: _f(per_budget[b][m][s]['downstream_score'])
                    for s in strategies} for m in factories}
            for b in budgets},
        'heldout_score_metric': metric_name,
        'width_built_per_budget': {
            b: {m: {s: (float(np.mean(per_budget[b][m][s]
                                      ['width_built_per_fold']))
                        if per_budget[b][m][s]['width_built_per_fold']
                        else float('nan'))
                    for s in strategies} for m in factories}
            for b in budgets},
        'width_built_per_budget_per_fold': {
            b: {m: {s: list(per_budget[b][m][s]['width_built_per_fold'])
                    for s in strategies} for m in factories}
            for b in budgets},
        'mi_density_per_budget': {
            b: {m: {s: _f(per_budget[b][m][s]['mi_density'])
                    for s in strategies} for m in factories}
            for b in budgets},
        'mi_density_measured_on': 'held-out rows',
        'mi_density_comparable_across_budgets': False,
    }
    if n_outer == 1:
        summary['fold_spread'] = 'one fold, no spread'

    results = {
        'result_identity': RESULT_IDENTITY,
        'metric': metric_name,
        # Which way that metric runs, from get_domain_metric, so a reader of
        # the file never has to guess.
        'lower_is_better': bool(lower_is_better),
        'models': list(factories),
        'models_decided': model_names,
        'models_skipped_size_guard': models_skipped,
        'budgets': budgets,
        'budgets_decided': _decided_budgets,
        'budgets_not_run': budgets_not_run,
        'budget_unit': 'features',
        'total_dim': total_dim,
        'representations': rep_order,
        'representations_outside_general': reps_outside_general,
        'strategies': strategies,
        'hybrid_arms': list(_C17_HYBRID_ARMS),
        'n_folds': n_outer,
        'n_folds_requested': n_folds_requested,
        'outer_split': {'splitter': 'nested_cv_outer_folds', 'seed': SEED},
        'selection_scope': 'fold_train_rows',
        'selection': ("per outer fold, per decided model: the 2 best hybrids "
                      "at each budget from a search on that fold's training "
                      "rows (fold_standard_selection, budget=<cell budget>, "
                      "decided grid, pruning off); a second hybrid with the "
                      "same selected features as the first is skipped"),
        'selection_rows': "each fold's training rows",
        'measured_on': 'held-out rows',
        'mrmr_max_budget': MRMR_MAX_BUDGET,
        'mrmr_budgets_served': _mrmr_cells,
        'per_fold': per_fold,
        'per_budget': convert_numpy(per_budget),
        'summary': convert_numpy(summary),
    }

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        for m in factories:
            for strategy in strategies:
                print(f"    {m:8s} {strategy:<20} "
                      f"MI={summary['mi_per_budget'][budget][m][strategy]:.4f} "
                      f"{metric_name}="
                      f"{summary['heldout_score_per_budget'][budget][m][strategy]:.4f} "
                      f"width built="
                      f"{summary['width_built_per_budget'][budget][m][strategy]:.1f}"
                      f" features")

    print(f"\n  [timing] c17 total: {time.time() - _t_run0:.1f}s")
    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C17: Information-Theoretic Feature Selection Comparison')
    args = parser.parse_args()
    config = get_config(args)
    # 2026-10-05 (handoff 2.4 step 9): the dataset's pruned representations,
    # as run_all loads them. --reps is used only for a dataset with no list.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _gen = (DATASET_REPS.get(args.dataset) or {}).get('general')
    if _gen:
        data = load_data(args.dataset, rep_preset='all', rep_names=list(_gen))
    else:
        data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    _mn, _fn, _lib = get_domain_metric(data)
    save_experiment_results(results, args.output,
                            'c17_information_theoretic_decomposition', args.dataset,
                            metric_name=_mn, lower_is_better=_lib)
