#!/usr/bin/env python3
"""F11: Allocation × Evaluator interaction on small datasets.

Paper: Framework
Question: F01 showed smoother allocations (performance_weighted, softmax, etc.)
    beat greedy on all datasets. F10 showed internal evaluator fixes (ridge,
    3-fold CV) help greedy on small data. But nobody tested the interaction:
    do the smoother allocations also benefit from evaluator fixes? And can
    importance-weighted allocation bypass the noisy baseline eval entirely?

Conditions (internal evaluator variants):
    rf_single     — RF on single train/val split (current default)
    ridge_single  — RidgeCV on single train/val split
    rf_cv3        — RF with 3-fold CV (recombine train+val)
    ridge_cv3     — RidgeCV with 3-fold CV (recombine train+val)

Allocation strategies:
    performance_weighted, softmax_T2, power_2, min_floor_0.05, thresh_0.5

Plus importance-weighted allocation (new):
    imp_treeshap  — Aggregate treeshap importance per rep → allocation weights
    imp_lgbm      — Aggregate lightgbm_gain importance per rep → allocation weights

Budgets: [50, 100]
Downstream models: [ridge, rf, lightgbm]
Datasets: 4 small antibody (n~130) + flab_thermo_tresanco (n=672 control)

Usage:
    python f11_allocation_evaluator_interaction.py --dataset flab_pk_jain_fcrnrt --reps all --mode full
    python f11_allocation_evaluator_interaction.py --dataset flab_thermo_tresanco --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    parse_rep_args,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.hybrid import (
    create_hybrid, apply_feature_selection, compute_feature_importance,
    _compute_allocation_weights, _is_classification_task,
    _create_hybrid_with_allocation,
)
import kirby.hybrid as hybrid_module


# ── Constants ────────────────────────────────────────────────────────────────

BUDGETS = [50, 100]

# Use shared DOWNSTREAM_MODELS from _shared (imported above).
F11_MODELS = DOWNSTREAM_MODELS

ALLOCATION_STRATEGIES = [
    'performance_weighted', 'softmax_T2', 'power_2',
    'min_floor_0.05', 'thresh_0.5',
]

EVALUATOR_CONDITIONS = ['rf_single', 'ridge_single', 'rf_cv3', 'ridge_cv3']

IMPORTANCE_ALLOCATION_METHODS = ['imp_treeshap', 'imp_lgbm']

STEP_SIZE = 10
PATIENCE = 3


# ── Monkey-patch evaluator factories ─────────────────────────────────────────

def _make_rf_single():
    """Default: RF on single train/val split."""
    return hybrid_module._evaluate_allocation  # original


def _make_ridge_single():
    """RidgeCV on single train/val split."""
    def _evaluate_allocation(X_train, X_val, y_train, y_val,
                             model=None, metric_fn=None,
                             lower_is_better=False):
        if X_train.shape[1] == 0:
            return -np.inf
        if model is not None:
            m = deepcopy(model)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            m.fit(X_tr_s, y_train)
            y_pred = m.predict(X_va_s)
            score = metric_fn(y_val, y_pred)
            return -score if lower_is_better else score

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)

        is_clf = _is_classification_task(y_train)
        if is_clf:
            from sklearn.linear_model import RidgeClassifierCV
            ridge = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_tr_s, y_train)
            return ridge.score(X_va_s, y_val)
        else:
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_tr_s, y_train)
            y_pred = ridge.predict(X_va_s)
            return r2_score(y_val, y_pred)

    return _evaluate_allocation


def _make_rf_cv3():
    """RF with 3-fold CV on recombined train+val."""
    def _evaluate_allocation(X_train, X_val, y_train, y_val,
                             model=None, metric_fn=None,
                             lower_is_better=False):
        if X_train.shape[1] == 0:
            return -np.inf
        if model is not None:
            m = deepcopy(model)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            m.fit(X_tr_s, y_train)
            y_pred = m.predict(X_va_s)
            score = metric_fn(y_val, y_pred)
            return -score if lower_is_better else score

        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        n_cv = min(3, len(y_all))
        if n_cv < 2:
            return -np.inf

        is_clf = _is_classification_task(y_all)
        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_all):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_all[tr_idx])
            X_va_s = scaler.transform(X_all[va_idx])
            if is_clf:
                rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                            n_jobs=-1)
                rf.fit(X_tr_s, y_all[tr_idx])
                scores.append(rf.score(X_va_s, y_all[va_idx]))
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42,
                                           n_jobs=-1)
                rf.fit(X_tr_s, y_all[tr_idx])
                y_pred = rf.predict(X_va_s)
                scores.append(r2_score(y_all[va_idx], y_pred))

        return float(np.mean(scores))

    return _evaluate_allocation


def _make_ridge_cv3():
    """RidgeCV with 3-fold CV on recombined train+val."""
    def _evaluate_allocation(X_train, X_val, y_train, y_val,
                             model=None, metric_fn=None,
                             lower_is_better=False):
        if X_train.shape[1] == 0:
            return -np.inf
        if model is not None:
            m = deepcopy(model)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            m.fit(X_tr_s, y_train)
            y_pred = m.predict(X_va_s)
            score = metric_fn(y_val, y_pred)
            return -score if lower_is_better else score

        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        n_cv = min(3, len(y_all))
        if n_cv < 2:
            return -np.inf

        is_clf = _is_classification_task(y_all)
        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_all):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_all[tr_idx])
            X_va_s = scaler.transform(X_all[va_idx])
            if is_clf:
                from sklearn.linear_model import RidgeClassifierCV
                ridge = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_s, y_all[tr_idx])
                scores.append(ridge.score(X_va_s, y_all[va_idx]))
            else:
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_s, y_all[tr_idx])
                y_pred = ridge.predict(X_va_s)
                scores.append(r2_score(y_all[va_idx], y_pred))

        return float(np.mean(scores))

    return _evaluate_allocation


EVALUATOR_FACTORIES = {
    'rf_single': _make_rf_single,
    'ridge_single': _make_ridge_single,
    'rf_cv3': _make_rf_cv3,
    'ridge_cv3': _make_ridge_cv3,
}


# ── Importance-weighted allocation ───────────────────────────────────────────

def _importance_weighted_allocation(train_reps, y_train, budget, method):
    """Allocate budget proportional to per-rep aggregate importance.

    Instead of training a model per rep and using R²/accuracy (noisy on small
    data), compute feature importance on the concatenated reps and aggregate
    per rep. This bypasses the internal evaluator entirely.
    """
    # Compute importance on concatenated features
    importance_scores = compute_feature_importance(
        train_reps, y_train, method=method
    )

    # Aggregate: mean importance per rep
    rep_agg = {}
    for rep_name, scores in importance_scores.items():
        rep_agg[rep_name] = float(np.mean(np.abs(scores)))

    # Normalise to weights and allocate
    total_imp = sum(rep_agg.values())
    if total_imp == 0:
        # Fallback: equal allocation
        n_reps = len(train_reps)
        per_rep = budget // n_reps
        return {r: min(per_rep, train_reps[r].shape[1]) for r in train_reps}

    allocation = {}
    for rep_name, imp in rep_agg.items():
        n_features = int(round((imp / total_imp) * budget))
        max_features = train_reps[rep_name].shape[1]
        allocation[rep_name] = min(n_features, max_features)

    # Adjust to hit budget exactly
    current = sum(allocation.values())
    if current < budget:
        sorted_reps = sorted(rep_agg.items(), key=lambda x: x[1], reverse=True)
        remaining = budget - current
        for rep_name, _ in sorted_reps:
            if remaining <= 0:
                break
            can_add = train_reps[rep_name].shape[1] - allocation[rep_name]
            add = min(can_add, remaining)
            allocation[rep_name] += add
            remaining -= add
    elif current > budget:
        sorted_reps = sorted(rep_agg.items(), key=lambda x: x[1])
        excess = current - budget
        for rep_name, _ in sorted_reps:
            if excess <= 0:
                break
            reduce = min(allocation[rep_name], excess)
            allocation[rep_name] -= reduce
            excess -= reduce

    return allocation


def _extract_allocation(feature_info):
    """Extract {rep_name: n_features} from create_hybrid feature_info."""
    alloc = {}
    for rep_name, info in feature_info.items():
        if isinstance(info, dict) and 'selected_indices' in info:
            alloc[rep_name] = len(info['selected_indices'])
    return alloc


# ── Main run ─────────────────────────────────────────────────────────────────

def run(data, config, output_dir, dataset_name, **kwargs):
    """Run allocation × evaluator interaction experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F11: ALLOCATION × EVALUATOR INTERACTION")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    domain = data.get('domain', 'antibody')
    task_type = data.get('task_type', 'regression')
    n_samples = len(labels)
    n_folds = config['n_splits']

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Reps: {len(reps)} ({', '.join(sorted(reps.keys()))})")
    print(f"  Total features: {sum(v.shape[1] for v in reps.values())}")
    print(f"  Metric: {metric_name} "
          f"({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Allocations: {ALLOCATION_STRATEGIES}")
    print(f"  Evaluators: {EVALUATOR_CONDITIONS}")
    print(f"  Importance allocations: {IMPORTANCE_ALLOCATION_METHODS}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Downstream models: {DOWNSTREAM_MODELS}")
    print(f"  N folds: {n_folds}\n", flush=True)

    factories = get_model_factories(DOWNSTREAM_MODELS, task_type=task_type,
                                    n_samples=n_samples)
    print(f"  Available downstream models: {sorted(factories.keys())}\n",
          flush=True)

    original_evaluate = hybrid_module._evaluate_allocation

    # ── Phase 1: Single-rep baselines ────────────────────────────────────
    print("=" * 70)
    print("  PHASE 1: SINGLE-REP BASELINES")
    print("=" * 70, flush=True)

    baselines = {}
    best_single_rep = {}

    for rep_name in sorted(reps.keys()):
        baselines[rep_name] = {}
        X_rep = reps[rep_name]

        for model_name, model_factory in sorted(factories.items()):
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, X_rep, labels, mode='full')
            ):
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_rep[train_idx])
                X_va_s = scaler.transform(X_rep[val_idx])

                model = model_factory()
                model.fit(X_tr_s, labels[train_idx])
                score = metric_fn(labels[val_idx], model.predict(X_va_s))
                fold_scores.append(float(score))

            mean_s = float(np.mean(fold_scores))
            std_s = float(np.std(fold_scores))
            baselines[rep_name][model_name] = {
                'mean': mean_s, 'std': std_s, 'folds': fold_scores,
            }

            if model_name not in best_single_rep or (
                (not lower_is_better and mean_s > best_single_rep[model_name]['score']) or
                (lower_is_better and mean_s < best_single_rep[model_name]['score'])
            ):
                best_single_rep[model_name] = {
                    'rep': rep_name, 'score': mean_s,
                }

        print(f"  {rep_name}: "
              + ", ".join(f"{mn}={baselines[rep_name][mn]['mean']:.4f}"
                          for mn in sorted(factories.keys())
                          if mn in baselines[rep_name]),
              flush=True)

    print(f"\n  Best single rep per model:")
    for mn in sorted(best_single_rep.keys()):
        info = best_single_rep[mn]
        print(f"    {mn}: {info['rep']} ({metric_name}={info['score']:.4f})")

    # ── Phase 2: Allocation × Evaluator grid ─────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: ALLOCATION × EVALUATOR GRID")
    print("=" * 70, flush=True)

    all_results = []
    total_configs = (len(ALLOCATION_STRATEGIES) * len(EVALUATOR_CONDITIONS)
                     * len(BUDGETS))
    config_idx = 0

    for alloc_strategy in ALLOCATION_STRATEGIES:
        for eval_condition in EVALUATOR_CONDITIONS:
            for budget in BUDGETS:
                config_idx += 1
                label = (f"[{config_idx}/{total_configs}] "
                         f"{alloc_strategy} × {eval_condition} × b={budget}")
                print(f"\n  --- {label} ---", flush=True)

                # Set evaluator
                hybrid_module._evaluate_allocation = \
                    EVALUATOR_FACTORIES[eval_condition]()

                fold_results = []
                t0 = time.time()

                try:
                    for fold_idx, (train_idx, val_idx) in enumerate(
                        cv_split(groups, n_folds,
                                 next(iter(reps.values())), labels,
                                 mode='full')
                    ):
                        train_reps = {k: v[train_idx] for k, v in reps.items()}
                        val_reps = {k: v[val_idx] for k, v in reps.items()}
                        y_train = labels[train_idx]
                        y_val = labels[val_idx]

                        try:
                            t_fold = time.time()
                            X_train_h, feature_info = create_hybrid(
                                train_reps, y_train,
                                allocation_method=alloc_strategy,
                                budget=budget,
                                importance_method='treeshap',
                                step_size=STEP_SIZE,
                                patience=PATIENCE,
                            )
                            X_val_h = apply_feature_selection(
                                val_reps, feature_info)
                            hybrid_time = time.time() - t_fold

                            allocation = _extract_allocation(feature_info)

                            # Evaluate with downstream models
                            downstream_scores = {}
                            scaler = StandardScaler()
                            X_tr_s = scaler.fit_transform(X_train_h)
                            X_va_s = scaler.transform(X_val_h)

                            for model_name, model_factory in sorted(
                                factories.items()
                            ):
                                model = model_factory()
                                model.fit(X_tr_s, y_train)
                                score = metric_fn(
                                    y_val, model.predict(X_va_s))
                                downstream_scores[model_name] = float(score)

                            fold_results.append({
                                'fold': fold_idx,
                                'n_train': len(y_train),
                                'allocation': allocation,
                                'n_features': X_train_h.shape[1],
                                'hybrid_time': hybrid_time,
                                'downstream_scores': downstream_scores,
                            })
                        except Exception as e:
                            print(f"    Fold {fold_idx} FAILED: {e}")
                            fold_results.append({
                                'fold': fold_idx, 'error': str(e),
                            })
                finally:
                    hybrid_module._evaluate_allocation = original_evaluate

                total_time = time.time() - t0
                result_entry = _aggregate_fold_results(
                    fold_results, factories, metric_name, lower_is_better,
                    best_single_rep, total_time,
                    alloc_strategy=alloc_strategy,
                    eval_condition=eval_condition,
                    budget=budget,
                )
                all_results.append(result_entry)

    # ── Phase 3: Importance-weighted allocation ──────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: IMPORTANCE-WEIGHTED ALLOCATION")
    print("=" * 70, flush=True)

    imp_total = len(IMPORTANCE_ALLOCATION_METHODS) * len(BUDGETS)
    imp_idx = 0

    for imp_method in IMPORTANCE_ALLOCATION_METHODS:
        method = imp_method.replace('imp_', '')  # treeshap or lgbm
        if method == 'lgbm':
            method = 'lightgbm_gain'

        for budget in BUDGETS:
            imp_idx += 1
            label = f"[{imp_idx}/{imp_total}] {imp_method} × b={budget}"
            print(f"\n  --- {label} ---", flush=True)

            fold_results = []
            t0 = time.time()

            for fold_idx, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds,
                         next(iter(reps.values())), labels, mode='full')
            ):
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                val_reps = {k: v[val_idx] for k, v in reps.items()}
                y_train = labels[train_idx]
                y_val = labels[val_idx]

                try:
                    t_fold = time.time()

                    # Compute importance-based allocation
                    allocation = _importance_weighted_allocation(
                        train_reps, y_train, budget, method
                    )

                    # Build hybrid using the allocation
                    X_train_h, feature_info = _create_hybrid_with_allocation(
                        train_reps, y_train, allocation
                    )
                    X_val_h = apply_feature_selection(val_reps, feature_info)
                    hybrid_time = time.time() - t_fold

                    # Evaluate with downstream models
                    downstream_scores = {}
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train_h)
                    X_va_s = scaler.transform(X_val_h)

                    for model_name, model_factory in sorted(
                        factories.items()
                    ):
                        model = model_factory()
                        model.fit(X_tr_s, y_train)
                        score = metric_fn(y_val, model.predict(X_va_s))
                        downstream_scores[model_name] = float(score)

                    fold_results.append({
                        'fold': fold_idx,
                        'n_train': len(y_train),
                        'allocation': allocation,
                        'n_features': X_train_h.shape[1],
                        'hybrid_time': hybrid_time,
                        'downstream_scores': downstream_scores,
                    })
                except Exception as e:
                    print(f"    Fold {fold_idx} FAILED: {e}")
                    fold_results.append({
                        'fold': fold_idx, 'error': str(e),
                    })

            total_time = time.time() - t0
            result_entry = _aggregate_fold_results(
                fold_results, factories, metric_name, lower_is_better,
                best_single_rep, total_time,
                alloc_strategy=imp_method,
                eval_condition='importance',
                budget=budget,
            )
            all_results.append(result_entry)

    # ── Summary ──────────────────────────────────────────────────────────
    _print_summary(all_results, factories, metric_name, lower_is_better)

    # ── Structured output ────────────────────────────────────────────────
    return make_structured_output(
        experiment_id='f11',
        dataset=dataset_name,
        data=data,
        config={
            'allocation_strategies': ALLOCATION_STRATEGIES,
            'evaluator_conditions': EVALUATOR_CONDITIONS,
            'importance_methods': IMPORTANCE_ALLOCATION_METHODS,
            'budgets': BUDGETS,
            'downstream_models': DOWNSTREAM_MODELS,
            'step_size': STEP_SIZE,
            'patience': PATIENCE,
            'n_folds': n_folds,
            'n_samples': n_samples,
        },
        results={
            'baselines': baselines,
            'best_single_rep': best_single_rep,
            'grid_results': all_results,
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _aggregate_fold_results(fold_results, factories, metric_name,
                            lower_is_better, best_single_rep, total_time,
                            alloc_strategy, eval_condition, budget):
    """Aggregate per-fold results into a summary entry."""
    valid_folds = [f for f in fold_results if 'error' not in f]

    if valid_folds:
        # Mean allocation
        all_rep_names = sorted(
            set().union(*(f['allocation'].keys() for f in valid_folds)))
        mean_alloc = {
            r: float(np.mean([f['allocation'].get(r, 0)
                              for f in valid_folds]))
            for r in all_rep_names
        }

        # Mean downstream scores + delta vs best single rep
        mean_downstream = {}
        delta_vs_best = {}
        for model_name in sorted(factories.keys()):
            scores = [f['downstream_scores'].get(model_name)
                      for f in valid_folds
                      if model_name in f.get('downstream_scores', {})]
            if scores:
                m = float(np.mean(scores))
                s = float(np.std(scores))
                mean_downstream[model_name] = {'mean': m, 'std': s}

                best_info = best_single_rep.get(model_name)
                if best_info:
                    delta = m - best_info['score']
                    if lower_is_better:
                        delta = -delta
                    delta_vs_best[model_name] = {
                        'delta': float(delta),
                        'hybrid': m,
                        'best_rep': best_info['score'],
                        'best_rep_name': best_info['rep'],
                    }

        n_feats = float(np.mean([f['n_features'] for f in valid_folds]))

        # Best score across models
        best_score = None
        best_model = None
        for mn, ms in mean_downstream.items():
            if best_score is None or (
                (not lower_is_better and ms['mean'] > best_score) or
                (lower_is_better and ms['mean'] < best_score)
            ):
                best_score = ms['mean']
                best_model = mn

        print(f"    Features: {n_feats:.0f} | "
              f"Time: {total_time:.1f}s | "
              f"Folds OK: {len(valid_folds)}")
        for mn, ms in sorted(mean_downstream.items()):
            d = delta_vs_best.get(mn, {}).get('delta', 0)
            sign = '+' if d >= 0 else ''
            print(f"      {mn}: {metric_name}="
                  f"{ms['mean']:.4f}±{ms['std']:.4f} "
                  f"(Δ={sign}{d:.4f})")
    else:
        mean_alloc = {}
        mean_downstream = {}
        delta_vs_best = {}
        n_feats = 0
        best_score = None
        best_model = None
        print(f"    ALL FOLDS FAILED")

    return {
        'allocation_strategy': alloc_strategy,
        'evaluator_condition': eval_condition,
        'budget': budget,
        'wall_time': total_time,
        'n_folds_ok': len(valid_folds),
        'n_features_mean': n_feats,
        'mean_allocation': mean_alloc,
        'mean_downstream': mean_downstream,
        'delta_vs_best_rep': delta_vs_best,
        'best_score': best_score,
        'best_model': best_model,
        'fold_results': fold_results,
    }


def _print_summary(all_results, factories, metric_name, lower_is_better):
    """Print ranked summary table."""
    print("\n" + "=" * 70)
    print("  SUMMARY: RANKED BY BEST DOWNSTREAM SCORE")
    print("=" * 70)

    # Rank by best score
    scored = [(r, r['best_score']) for r in all_results
              if r['best_score'] is not None]
    scored.sort(key=lambda x: x[1], reverse=not lower_is_better)

    print(f"\n  {'Rank':>4} {'Allocation':<22} {'Evaluator':<14} "
          f"{'Budget':>6} {'Best':>8} {'Model':<10} ", end="")
    for mn in sorted(factories.keys()):
        print(f"{'Δ_' + mn:>10}", end="")
    print()
    print("  " + "-" * 100)

    for rank, (r, score) in enumerate(scored, 1):
        alloc = r['allocation_strategy']
        ev = r['evaluator_condition']
        b = r['budget']
        bm = r['best_model'] or '?'
        print(f"  {rank:>4} {alloc:<22} {ev:<14} {b:>6} "
              f"{score:>8.4f} {bm:<10} ", end="")
        for mn in sorted(factories.keys()):
            d = r['delta_vs_best_rep'].get(mn, {}).get('delta')
            if d is not None:
                sign = '+' if d >= 0 else ''
                print(f"{sign}{d:>9.4f}", end=" ")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    # Best per allocation strategy
    print(f"\n  BEST PER ALLOCATION STRATEGY:")
    by_alloc = defaultdict(list)
    for r in all_results:
        if r['best_score'] is not None:
            by_alloc[r['allocation_strategy']].append(r)

    for alloc in sorted(by_alloc.keys()):
        entries = by_alloc[alloc]
        entries.sort(key=lambda x: x['best_score'],
                     reverse=not lower_is_better)
        best = entries[0]
        print(f"    {alloc:<22} → {best['evaluator_condition']:<14} "
              f"b={best['budget']:>3} "
              f"{metric_name}={best['best_score']:.4f} "
              f"({best['best_model']})")

    # Best per evaluator condition
    print(f"\n  BEST PER EVALUATOR CONDITION:")
    by_eval = defaultdict(list)
    for r in all_results:
        if r['best_score'] is not None:
            by_eval[r['evaluator_condition']].append(r)

    for ev in sorted(by_eval.keys()):
        entries = by_eval[ev]
        entries.sort(key=lambda x: x['best_score'],
                     reverse=not lower_is_better)
        best = entries[0]
        print(f"    {ev:<14} → {best['allocation_strategy']:<22} "
              f"b={best['budget']:>3} "
              f"{metric_name}={best['best_score']:.4f} "
              f"({best['best_model']})")


if __name__ == '__main__':
    parser = create_experiment_parser(
        'F11: Allocation × Evaluator Interaction',
        has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'f11_allocation_evaluator_interaction',
                            args.dataset)
