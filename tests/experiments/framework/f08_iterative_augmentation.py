#!/usr/bin/env python3
"""F08: Iterative augmentation design sweep.

Paper: Framework
Question: What is the best way to handle small reps as augmentations?
    Tests auto-classification threshold, candidate ranking, worst-feature
    identification, patience, step sizes, and A/B eval model.

Setup:
    - 1-2 datasets per domain, single fold from group-aware splitter
    - XGBoost as the fast downstream model
    - All augmentation logic is self-contained here (not yet in hybrid.py)

Axes swept:
    A. auto_threshold:       [10, 20, 50] — reps with d < threshold become augmentations
    B. candidate_ranking:    ['rf_importance', 'residual_correlation']
    C. worst_feature_method: ['permutation', 'allocation_scores']
    D. patience:             [5, 10, 20, 50]
    E. aug_step_size:        [1, 5, 10]
    F. base_step_size:       [1, 2, 5, 10]
    G. ab_eval_model:        ['rf_proxy', 'downstream']

Usage:
    python f08_iterative_augmentation.py --dataset logd --reps all
    python f08_iterative_augmentation.py --dataset flab_thermo_tresanco --reps all
    python f08_iterative_augmentation.py --dataset tcga_luad_vs_lusc --reps all
    python f08_iterative_augmentation.py --dataset kiba --reps all
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from copy import deepcopy
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    parse_rep_args, FRAMEWORK_TASKS, FULL, INTERMEDIARY,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.hybrid import (
    create_hybrid, apply_feature_selection, apply_augmentation_selection,
    _is_classification_task,
)


# =========================================================================
# STANDALONE ITERATIVE AUGMENTATION (experiment-local, not in hybrid.py)
# =========================================================================

def _pool_augmentation_features(aug_dict_train, aug_dict_val):
    """Pool all augmentation features into single arrays with provenance tracking.

    Returns:
        X_pool_train, X_pool_val: pooled arrays
        provenance: list of (aug_name, col_idx) per pooled column
    """
    provenance = []
    parts_train = []
    parts_val = []

    for aug_name, X_tr in aug_dict_train.items():
        X_vl = aug_dict_val[aug_name]
        X_tr = np.clip(X_tr, -1e10, 1e10)
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e10, neginf=-1e10)
        X_vl = np.clip(X_vl, -1e10, 1e10)
        X_vl = np.nan_to_num(X_vl, nan=0.0, posinf=1e10, neginf=-1e10)
        for col in range(X_tr.shape[1]):
            provenance.append((aug_name, col))
        parts_train.append(X_tr)
        parts_val.append(X_vl)

    if not parts_train:
        return None, None, []

    return np.hstack(parts_train), np.hstack(parts_val), provenance


def _rank_by_rf_importance(X_pool_train, y_train, is_clf):
    """Rank pooled features by RF importance. Returns sorted indices (best first)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool_train)

    if is_clf:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                    random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                                   random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_train)
    imp = np.nan_to_num(rf.feature_importances_, nan=0.0)
    return np.argsort(imp)[::-1], imp


def _rank_by_residual_correlation(X_pool_train, X_base_train, y_train,
                                  eval_model, is_clf):
    """Rank pooled features by |correlation| with base hybrid residuals."""
    scaler = StandardScaler()
    X_base_s = scaler.fit_transform(X_base_train)
    m = deepcopy(eval_model)
    m.fit(X_base_s, y_train)
    residuals = y_train - m.predict(X_base_s)

    n_feats = X_pool_train.shape[1]
    correlations = np.zeros(n_feats)
    for j in range(n_feats):
        col = X_pool_train[:, j]
        if np.std(col) < 1e-12:
            continue
        correlations[j] = abs(np.corrcoef(col, residuals)[0, 1])
    correlations = np.nan_to_num(correlations, nan=0.0)
    return np.argsort(correlations)[::-1], correlations


def _eval_score(X_train, X_val, y_train, y_val, model, metric_fn,
                lower_is_better):
    """Fit model, predict, score. Returns score oriented so higher = better."""
    if X_train.shape[1] == 0:
        return -np.inf
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xvl = scaler.transform(X_val)
    m = deepcopy(model)
    m.fit(Xtr, y_train)
    score = metric_fn(y_val, m.predict(Xvl))
    return -score if lower_is_better else score


def _get_worst_feature_by_permutation(X_train, X_val, y_train, y_val,
                                      model, metric_fn, lower_is_better,
                                      n_base):
    """Return index of worst base feature (0..n_base-1) by permutation importance."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xvl = scaler.transform(X_val)
    m = deepcopy(model)
    m.fit(Xtr, y_train)

    scorer = make_scorer(metric_fn, greater_is_better=not lower_is_better)
    result = permutation_importance(m, Xvl, y_val, n_repeats=3,
                                    random_state=42, scoring=scorer)
    perm_imp = result.importances_mean[:n_base]
    # Lowest importance = least useful feature
    return int(np.argmin(perm_imp))


def _get_worst_feature_by_allocation(base_feature_info, n_base):
    """Return index of worst base feature using original allocation importance scores."""
    # Flatten importance scores from all base reps in column order
    all_imp = []
    for rep_name, info in base_feature_info.items():
        if 'importance_scores' not in info:
            continue
        all_imp.extend(info['importance_scores'].tolist())
    if not all_imp or len(all_imp) < n_base:
        return 0
    return int(np.argmin(all_imp[:n_base]))


def iterative_augmentation(
    X_base_train, X_base_val,
    base_feature_info,
    aug_dict_train, aug_dict_val,
    y_train, y_val,
    eval_model,
    metric_fn,
    lower_is_better,
    total_budget,
    candidate_ranking='rf_importance',
    worst_feature_method='permutation',
    patience=10,
    aug_step_size=1,
):
    """Self-contained iterative augmentation for the experiment.

    1. Pool all augmentation features, rank candidates.
    2. Fill phase: if under budget, greedily add aug features (step_size at a time).
    3. Replace phase: swap worst base feature with next candidate, keep if improves.

    Returns dict with results and diagnostics.
    """
    is_clf = _is_classification_task(y_train)
    n_base = X_base_train.shape[1]
    remaining_budget = max(0, total_budget - n_base)

    # Pool augmentation features
    X_pool_train, X_pool_val, provenance = _pool_augmentation_features(
        aug_dict_train, aug_dict_val)

    if X_pool_train is None or X_pool_train.shape[1] == 0:
        return {
            'baseline_score': float('nan'),
            'final_score': float('nan'),
            'fill_added': 0,
            'replace_swapped': 0,
            'total_aug_features': 0,
            'fill_evals': 0,
            'replace_evals': 0,
            'wall_time': 0.0,
        }

    t0 = time.time()

    # Rank candidates
    if candidate_ranking == 'rf_importance':
        ranked_idx, rank_scores = _rank_by_rf_importance(
            X_pool_train, y_train, is_clf)
    elif candidate_ranking == 'residual_correlation':
        ranked_idx, rank_scores = _rank_by_residual_correlation(
            X_pool_train, X_base_train, y_train, eval_model, is_clf)
    else:
        raise ValueError(f"Unknown candidate_ranking: {candidate_ranking}")

    # Baseline score
    baseline_score = _eval_score(X_base_train, X_base_val, y_train, y_val,
                                 eval_model, metric_fn, lower_is_better)

    X_cur_train = X_base_train.copy()
    X_cur_val = X_base_val.copy()
    current_score = baseline_score
    used_pool_indices = set()

    fill_added = 0
    fill_evals = 0
    consecutive_fail = 0

    # ----- Fill phase -----
    i = 0
    while remaining_budget > 0 and i < len(ranked_idx) and consecutive_fail < patience:
        # Gather next batch of aug_step_size candidates
        batch_indices = []
        while len(batch_indices) < aug_step_size and i < len(ranked_idx):
            idx = ranked_idx[i]
            i += 1
            if idx not in used_pool_indices:
                batch_indices.append(idx)
        if not batch_indices:
            break

        # Take only as many as remaining budget allows
        batch_indices = batch_indices[:remaining_budget]

        feat_train = X_pool_train[:, batch_indices]
        feat_val = X_pool_val[:, batch_indices]

        X_test_train = np.hstack([X_cur_train, feat_train])
        X_test_val = np.hstack([X_cur_val, feat_val])

        new_score = _eval_score(X_test_train, X_test_val, y_train, y_val,
                                eval_model, metric_fn, lower_is_better)
        fill_evals += 1

        if new_score > current_score:
            X_cur_train = X_test_train
            X_cur_val = X_test_val
            current_score = new_score
            remaining_budget -= len(batch_indices)
            fill_added += len(batch_indices)
            for idx in batch_indices:
                used_pool_indices.add(idx)
            consecutive_fail = 0
        else:
            consecutive_fail += 1

    # ----- Replace phase -----
    replace_swapped = 0
    replace_evals = 0
    consecutive_fail = 0

    # Continue from where fill phase left off in the ranking
    while i < len(ranked_idx) and consecutive_fail < patience:
        idx = ranked_idx[i]
        i += 1
        if idx in used_pool_indices:
            continue

        feat_train = X_pool_train[:, idx:idx + 1]
        feat_val = X_pool_val[:, idx:idx + 1]

        # Find worst base feature
        if worst_feature_method == 'permutation':
            worst_idx = _get_worst_feature_by_permutation(
                X_cur_train, X_cur_val, y_train, y_val,
                eval_model, metric_fn, lower_is_better, n_base)
        elif worst_feature_method == 'allocation_scores':
            worst_idx = _get_worst_feature_by_allocation(
                base_feature_info, n_base)
        else:
            raise ValueError(f"Unknown worst_feature_method: {worst_feature_method}")

        # Swap worst base feature with this aug feature
        X_test_train = X_cur_train.copy()
        X_test_val = X_cur_val.copy()
        X_test_train[:, worst_idx] = feat_train[:, 0]
        X_test_val[:, worst_idx] = feat_val[:, 0]

        new_score = _eval_score(X_test_train, X_test_val, y_train, y_val,
                                eval_model, metric_fn, lower_is_better)
        replace_evals += 1

        if new_score > current_score:
            X_cur_train = X_test_train
            X_cur_val = X_test_val
            current_score = new_score
            replace_swapped += 1
            used_pool_indices.add(idx)
            consecutive_fail = 0
        else:
            consecutive_fail += 1

    wall_time = time.time() - t0

    return {
        'baseline_score': float(baseline_score) if not lower_is_better
                          else float(-baseline_score),
        'final_score': float(current_score) if not lower_is_better
                       else float(-current_score),
        'fill_added': fill_added,
        'replace_swapped': replace_swapped,
        'total_aug_features': fill_added + replace_swapped,
        'fill_evals': fill_evals,
        'replace_evals': replace_evals,
        'total_evals': fill_evals + replace_evals,
        'wall_time': wall_time,
    }


# =========================================================================
# EXPERIMENT RUNNER
# =========================================================================

# Sweep axes
AUTO_THRESHOLDS = [10, 20, 50]
CANDIDATE_RANKINGS = ['rf_importance', 'residual_correlation']
WORST_FEATURE_METHODS = ['permutation', 'allocation_scores']
PATIENCE_VALUES = [5, 10, 20, 50]
AUG_STEP_SIZES = [1, 5, 10]
BASE_STEP_SIZES = [1, 2, 5, 10]
AB_EVAL_MODELS = ['rf_proxy', 'downstream']

HYBRID_BUDGET = 100


def _split_reps_by_threshold(reps, threshold):
    """Auto-classify reps into base and augmentation by dimension threshold."""
    base = {}
    aug = {}
    for name, X in reps.items():
        if X.shape[1] < threshold:
            aug[name] = X
        else:
            base[name] = X
    return base, aug


def _make_rf_proxy(is_clf):
    """Return an RF model for A/B testing."""
    if is_clf:
        return RandomForestClassifier(n_estimators=100, random_state=42,
                                      n_jobs=-1)
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run the iterative augmentation design sweep."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F08: ITERATIVE AUGMENTATION DESIGN SWEEP")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    domain = data.get('domain', 'molecular')
    task_type = data.get('task_type', 'regression')
    n_samples = len(labels)
    is_clf = task_type == 'classification'

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Reps: {len(reps)} total")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Rep dimensions: {', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())}")

    # Get downstream model (XGBoost)
    factories = get_model_factories(['xgboost'], task_type=task_type,
                                    n_samples=n_samples)
    if not factories:
        factories = get_model_factories(['rf'], task_type=task_type,
                                        n_samples=n_samples)
    downstream_name = next(iter(factories))
    downstream_factory = factories[downstream_name]
    print(f"  Downstream model: {downstream_name}")

    # Single fold from group-aware splitter
    sample_X = next(iter(reps.values()))
    fold_iter = cv_split(groups, 5, sample_X, labels, seed=42, mode='full')
    train_idx, val_idx = next(fold_iter)
    y_train, y_val = labels[train_idx], labels[val_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Show what each threshold would classify
    for thr in AUTO_THRESHOLDS:
        base, aug = _split_reps_by_threshold(reps, thr)
        aug_names = sorted(aug.keys())
        print(f"  Threshold {thr}: {len(base)} base, {len(aug)} aug "
              f"({', '.join(aug_names) if aug_names else 'none'})")

    # Count total configs
    n_configs = (len(AUTO_THRESHOLDS) * len(BASE_STEP_SIZES)
                 * len(CANDIDATE_RANKINGS) * len(WORST_FEATURE_METHODS)
                 * len(PATIENCE_VALUES) * len(AUG_STEP_SIZES)
                 * len(AB_EVAL_MODELS))
    print(f"\n  Total configs: {n_configs}")
    print(f"  (configs with no augmentation candidates will be fast skips)\n",
          flush=True)

    all_results = []
    idx = 0

    for auto_thr in AUTO_THRESHOLDS:
        base_reps_full, aug_reps_full = _split_reps_by_threshold(reps, auto_thr)

        # Need at least 2 base reps for hybrid creation
        if len(base_reps_full) < 2:
            print(f"  [threshold={auto_thr}] <2 base reps, skipping")
            idx += (len(BASE_STEP_SIZES) * len(CANDIDATE_RANKINGS)
                    * len(WORST_FEATURE_METHODS) * len(PATIENCE_VALUES)
                    * len(AUG_STEP_SIZES) * len(AB_EVAL_MODELS))
            continue

        # Split into train/val
        base_train = {k: v[train_idx] for k, v in base_reps_full.items()}
        base_val = {k: v[val_idx] for k, v in base_reps_full.items()}
        aug_train = {k: v[train_idx] for k, v in aug_reps_full.items()}
        aug_val = {k: v[val_idx] for k, v in aug_reps_full.items()}

        has_aug = len(aug_reps_full) > 0

        for base_ss in BASE_STEP_SIZES:
            # Build base hybrid with this step size via create_hybrid
            t_base = time.time()
            try:
                X_base_train, base_fi = create_hybrid(
                    base_train, labels[train_idx],
                    allocation_method='greedy',
                    budget=HYBRID_BUDGET,
                    step_size=base_ss,
                    patience=3,
                    internal_evaluator='rf',
                )
                X_base_val = apply_feature_selection(base_val, base_fi)
                n_base_features = X_base_train.shape[1]
                base_time = time.time() - t_base
            except Exception as e:
                print(f"  [thr={auto_thr}, base_ss={base_ss}] "
                      f"Base hybrid failed: {e}")
                idx += (len(CANDIDATE_RANKINGS) * len(WORST_FEATURE_METHODS)
                        * len(PATIENCE_VALUES) * len(AUG_STEP_SIZES)
                        * len(AB_EVAL_MODELS))
                continue

            # Baseline: base hybrid only, scored with downstream model
            base_only_score = _eval_score(
                X_base_train, X_base_val, y_train, y_val,
                downstream_factory(), metric_fn, lower_is_better)
            base_only_score_real = (float(-base_only_score) if lower_is_better
                                    else float(base_only_score))

            if not has_aug:
                # No augmentation candidates — record base-only result and skip
                idx += (len(CANDIDATE_RANKINGS) * len(WORST_FEATURE_METHODS)
                        * len(PATIENCE_VALUES) * len(AUG_STEP_SIZES)
                        * len(AB_EVAL_MODELS))
                for cr in CANDIDATE_RANKINGS:
                    for wf in WORST_FEATURE_METHODS:
                        for pat in PATIENCE_VALUES:
                            for ass in AUG_STEP_SIZES:
                                for abm in AB_EVAL_MODELS:
                                    all_results.append({
                                        'auto_threshold': auto_thr,
                                        'base_step_size': base_ss,
                                        'candidate_ranking': cr,
                                        'worst_feature_method': wf,
                                        'patience': pat,
                                        'aug_step_size': ass,
                                        'ab_eval_model': abm,
                                        'n_base_reps': len(base_reps_full),
                                        'n_aug_reps': 0,
                                        'n_base_features': n_base_features,
                                        'base_score': base_only_score_real,
                                        'final_score': base_only_score_real,
                                        'fill_added': 0,
                                        'replace_swapped': 0,
                                        'total_evals': 0,
                                        'base_time': base_time,
                                        'aug_time': 0.0,
                                        'no_aug_candidates': True,
                                    })
                print(f"  [thr={auto_thr}, base_ss={base_ss}] "
                      f"No aug candidates. base={base_only_score_real:.4f} "
                      f"({n_base_features} feats, {base_time:.1f}s)")
                continue

            # Sweep augmentation axes
            for cr, wf, pat, ass, abm in product(
                CANDIDATE_RANKINGS, WORST_FEATURE_METHODS,
                PATIENCE_VALUES, AUG_STEP_SIZES, AB_EVAL_MODELS,
            ):
                idx += 1

                # Select A/B eval model
                if abm == 'rf_proxy':
                    eval_model = _make_rf_proxy(is_clf)
                else:
                    eval_model = downstream_factory()

                try:
                    aug_result = iterative_augmentation(
                        X_base_train, X_base_val,
                        base_fi,
                        aug_train, aug_val,
                        y_train, y_val,
                        eval_model=eval_model,
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better,
                        total_budget=HYBRID_BUDGET,
                        candidate_ranking=cr,
                        worst_feature_method=wf,
                        patience=pat,
                        aug_step_size=ass,
                    )
                except Exception as e:
                    print(f"  [{idx}/{n_configs}] FAILED: {e}")
                    continue

                entry = {
                    'auto_threshold': auto_thr,
                    'base_step_size': base_ss,
                    'candidate_ranking': cr,
                    'worst_feature_method': wf,
                    'patience': pat,
                    'aug_step_size': ass,
                    'ab_eval_model': abm,
                    'n_base_reps': len(base_reps_full),
                    'n_aug_reps': len(aug_reps_full),
                    'n_base_features': n_base_features,
                    'base_score': base_only_score_real,
                    'final_score': aug_result['final_score'],
                    'fill_added': aug_result['fill_added'],
                    'replace_swapped': aug_result['replace_swapped'],
                    'total_evals': aug_result['total_evals'],
                    'base_time': base_time,
                    'aug_time': aug_result['wall_time'],
                    'no_aug_candidates': False,
                }
                all_results.append(entry)

                delta = entry['final_score'] - entry['base_score']
                sign = '+' if delta >= 0 else ''
                if idx % 50 == 0 or idx <= 5:
                    print(f"  [{idx}/{n_configs}] thr={auto_thr} "
                          f"bss={base_ss} cr={cr[:3]} wf={wf[:4]} "
                          f"pat={pat} ass={ass} abm={abm[:4]} | "
                          f"{metric_name}={entry['final_score']:.4f} "
                          f"({sign}{delta:.4f}) "
                          f"fill={aug_result['fill_added']} "
                          f"repl={aug_result['replace_swapped']} "
                          f"evals={aug_result['total_evals']} "
                          f"{aug_result['wall_time']:.1f}s",
                          flush=True)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if not all_results:
        print("  No results collected.")
        return make_structured_output(
            experiment_id='f08', dataset=dataset_name, data=data,
            config={}, results=[], primary_metric=metric_name,
            lower_is_better=lower_is_better)

    # Filter to configs that had augmentation candidates
    aug_results = [r for r in all_results if not r.get('no_aug_candidates')]

    if aug_results:
        # Best overall
        best = max(aug_results, key=lambda r: r['final_score'])
        if lower_is_better:
            best = min(aug_results, key=lambda r: r['final_score'])
        print(f"\n  Best config:")
        for k in ['auto_threshold', 'base_step_size', 'candidate_ranking',
                   'worst_feature_method', 'patience', 'aug_step_size',
                   'ab_eval_model']:
            print(f"    {k}: {best[k]}")
        print(f"    base_score: {best['base_score']:.4f}")
        print(f"    final_score: {best['final_score']:.4f}")
        print(f"    improvement: {best['final_score'] - best['base_score']:.4f}")
        print(f"    fill={best['fill_added']}, replace={best['replace_swapped']}")

        # Per-axis analysis: mean improvement by each axis value
        print(f"\n  Mean improvement by axis value:")
        axes = {
            'auto_threshold': AUTO_THRESHOLDS,
            'base_step_size': BASE_STEP_SIZES,
            'candidate_ranking': CANDIDATE_RANKINGS,
            'worst_feature_method': WORST_FEATURE_METHODS,
            'patience': PATIENCE_VALUES,
            'aug_step_size': AUG_STEP_SIZES,
            'ab_eval_model': AB_EVAL_MODELS,
        }
        for axis_name, axis_values in axes.items():
            print(f"    {axis_name}:")
            for val in axis_values:
                subset = [r for r in aug_results if r[axis_name] == val]
                if not subset:
                    continue
                improvements = [r['final_score'] - r['base_score']
                                for r in subset]
                times = [r['aug_time'] for r in subset]
                mean_imp = np.mean(improvements)
                mean_time = np.mean(times)
                sign = '+' if mean_imp >= 0 else ''
                print(f"      {str(val):<25} {sign}{mean_imp:.4f} "
                      f"({mean_time:.1f}s avg)")

        # Wall time distribution
        times = [r['aug_time'] for r in aug_results]
        print(f"\n  Aug wall time: min={min(times):.1f}s, "
              f"median={np.median(times):.1f}s, max={max(times):.1f}s")

    # Base-only results (no aug candidates)
    base_only = [r for r in all_results if r.get('no_aug_candidates')]
    if base_only:
        print(f"\n  Base-only configs (no aug candidates): {len(base_only)}")
        for bss in BASE_STEP_SIZES:
            subset = [r for r in base_only if r['base_step_size'] == bss]
            if subset:
                print(f"    base_step_size={bss}: "
                      f"{metric_name}={subset[0]['base_score']:.4f}")

    return make_structured_output(
        experiment_id='f08',
        dataset=dataset_name,
        data=data,
        config={
            'auto_thresholds': AUTO_THRESHOLDS,
            'candidate_rankings': CANDIDATE_RANKINGS,
            'worst_feature_methods': WORST_FEATURE_METHODS,
            'patience_values': PATIENCE_VALUES,
            'aug_step_sizes': AUG_STEP_SIZES,
            'base_step_sizes': BASE_STEP_SIZES,
            'ab_eval_models': AB_EVAL_MODELS,
            'hybrid_budget': HYBRID_BUDGET,
            'downstream_model': downstream_name,
            'n_configs': n_configs,
        },
        results=all_results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser(
        'F08: Iterative Augmentation Design Sweep',
        has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = INTERMEDIARY if args.mode == 'intermediary' else FULL

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'f08_iterative_augmentation',
                            args.dataset)
