#!/usr/bin/env python3
"""F10: Small dataset allocation fixes — hybrid underperformance on n<200.

Paper: Framework
Question: KIRBy's greedy allocation uses a single 80/20 split of the training
    fold for feature selection. On small datasets (n<200), this means ~22
    validation samples for greedy decisions — pure noise. Does fixing the
    internal evaluator, adding CV, or capping the budget recover hybrid
    performance vs the best single-rep baseline?

Conditions:
    A_baseline    — Current behavior (no changes)
    B_budget_cap  — min(budget, n_train // 2) to prevent p > n
    C_ridge_eval  — Replace RF with RidgeCV inside _evaluate_allocation
    D_cv3_eval    — Recombine train+val, do 3-fold CV with RF
    E_combined    — RidgeCV + 3-fold CV + budget cap

For each condition:
    1. Run single-rep baselines (12 reps × 4 models × 5-fold CV)
    2. Run hybrid creation with greedy allocation under each condition
    3. Compute delta = hybrid_score - best_single_rep_score

Usage:
    python f10_small_dataset_allocation.py --dataset flab_pk_jain_fcrnrt --reps blosum,onehot,aac --mode intermediary
    python f10_small_dataset_allocation.py --dataset flab_pk_jain_fcrnrt --reps all --mode full
    python f10_small_dataset_allocation.py --dataset flab_thermo_tresanco --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    parse_rep_args,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.hybrid import create_hybrid, apply_feature_selection
import kirby.hybrid as hybrid_module


# ── Constants ────────────────────────────────────────────────────────────────

BUDGETS = [25, 50, 100, 200]

# Use shared DOWNSTREAM_MODELS from _shared (imported above).
F10_MODELS = DOWNSTREAM_MODELS

CONDITIONS = ['A_baseline', 'B_budget_cap', 'C_ridge_eval', 'D_cv3_eval',
              'E_combined']

STEP_SIZE = 10
PATIENCE = 3


# ── Monkey-patch factories ──────────────────────────────────────────────────

def _make_ridge_evaluator():
    """Condition C: Replace RF with RidgeCV inside _evaluate_allocation."""
    def _evaluate_allocation(X_train, X_val, y_train, y_val,
                             model=None, metric_fn=None,
                             lower_is_better=False):
        if X_train.shape[1] == 0:
            return -np.inf

        # If a model is explicitly provided (augmentation callers), use it
        if model is not None:
            from copy import deepcopy
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

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_tr_s, y_train)
        y_pred = ridge.predict(X_va_s)
        return r2_score(y_val, y_pred)

    return _evaluate_allocation


def _make_cv3_evaluator():
    """Condition D: Recombine train+val, do 3-fold CV with RF."""
    def _evaluate_allocation(X_train, X_val, y_train, y_val,
                             model=None, metric_fn=None,
                             lower_is_better=False):
        if X_train.shape[1] == 0:
            return -np.inf

        if model is not None:
            from copy import deepcopy
            m = deepcopy(model)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            m.fit(X_tr_s, y_train)
            y_pred = m.predict(X_va_s)
            score = metric_fn(y_val, y_pred)
            return -score if lower_is_better else score

        # Recombine
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        n_cv = min(3, len(y_all))
        if n_cv < 2:
            return -np.inf

        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_all):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_all[tr_idx])
            X_va_s = scaler.transform(X_all[va_idx])
            rf = RandomForestRegressor(n_estimators=100, random_state=42,
                                       n_jobs=-1)
            rf.fit(X_tr_s, y_all[tr_idx])
            y_pred = rf.predict(X_va_s)
            scores.append(r2_score(y_all[va_idx], y_pred))

        return float(np.mean(scores))

    return _evaluate_allocation


def _make_combined_evaluator():
    """Condition E: Recombine + 3-fold CV + RidgeCV (all fixes)."""
    def _evaluate_allocation(X_train, X_val, y_train, y_val,
                             model=None, metric_fn=None,
                             lower_is_better=False):
        if X_train.shape[1] == 0:
            return -np.inf

        if model is not None:
            from copy import deepcopy
            m = deepcopy(model)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            m.fit(X_tr_s, y_train)
            y_pred = m.predict(X_va_s)
            score = metric_fn(y_val, y_pred)
            return -score if lower_is_better else score

        # Recombine
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        n_cv = min(3, len(y_all))
        if n_cv < 2:
            return -np.inf

        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_all):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_all[tr_idx])
            X_va_s = scaler.transform(X_all[va_idx])
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_tr_s, y_all[tr_idx])
            y_pred = ridge.predict(X_va_s)
            scores.append(r2_score(y_all[va_idx], y_pred))

        return float(np.mean(scores))

    return _evaluate_allocation


def _extract_allocation(feature_info):
    """Extract {rep_name: n_features} from create_hybrid feature_info."""
    alloc = {}
    for rep_name, info in feature_info.items():
        if isinstance(info, dict) and 'selected_indices' in info:
            alloc[rep_name] = len(info['selected_indices'])
    return alloc


# ── Main run ─────────────────────────────────────────────────────────────────

def run(data, config, output_dir, dataset_name, **kwargs):
    """Run small dataset allocation fix experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F10: SMALL DATASET ALLOCATION FIXES")
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
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Downstream models: {DOWNSTREAM_MODELS}")
    print(f"  N folds: {n_folds}\n", flush=True)

    # Get downstream model factories
    factories = get_model_factories(DOWNSTREAM_MODELS, task_type=task_type,
                                    n_samples=n_samples)
    print(f"  Available downstream models: {sorted(factories.keys())}\n",
          flush=True)

    # Save original for restoration
    original_evaluate = hybrid_module._evaluate_allocation

    # ── Phase 1: Single-rep baselines ────────────────────────────────────
    print("=" * 70)
    print("  PHASE 1: SINGLE-REP BASELINES")
    print("=" * 70, flush=True)

    baselines = {}  # {rep_name: {model_name: {mean, std, folds}}}
    best_single_rep = {}  # {model_name: {rep, score}}

    for rep_name in sorted(reps.keys()):
        baselines[rep_name] = {}
        X_rep = reps[rep_name]

        for model_name, model_factory in sorted(factories.items()):
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, X_rep, labels, mode='full')
            ):
                X_train = X_rep[train_idx]
                X_val = X_rep[val_idx]
                y_train = labels[train_idx]
                y_val = labels[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                X_va_s = scaler.transform(X_val)

                model = model_factory()
                model.fit(X_tr_s, y_train)
                score = metric_fn(y_val, model.predict(X_va_s))
                fold_scores.append(float(score))

            mean_s = float(np.mean(fold_scores))
            std_s = float(np.std(fold_scores))
            baselines[rep_name][model_name] = {
                'mean': mean_s,
                'std': std_s,
                'folds': fold_scores,
            }

            # Track best single rep per model
            if model_name not in best_single_rep or (
                (not lower_is_better and mean_s > best_single_rep[model_name]['score']) or
                (lower_is_better and mean_s < best_single_rep[model_name]['score'])
            ):
                best_single_rep[model_name] = {
                    'rep': rep_name,
                    'score': mean_s,
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

    # ── Phase 2: Condition sweeps ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: CONDITION SWEEPS")
    print("=" * 70, flush=True)

    all_conditions = []
    total_configs = len(CONDITIONS) * len(BUDGETS)
    config_idx = 0

    for condition in CONDITIONS:
        for budget in BUDGETS:
            config_idx += 1
            label = (f"[{config_idx}/{total_configs}] "
                     f"condition={condition} | budget={budget}")
            print(f"\n  --- {label} ---", flush=True)

            # Select monkey-patch
            if condition == 'C_ridge_eval':
                hybrid_module._evaluate_allocation = _make_ridge_evaluator()
            elif condition == 'D_cv3_eval':
                hybrid_module._evaluate_allocation = _make_cv3_evaluator()
            elif condition == 'E_combined':
                hybrid_module._evaluate_allocation = _make_combined_evaluator()
            else:
                # A_baseline and B_budget_cap use original evaluator
                hybrid_module._evaluate_allocation = original_evaluate

            fold_results = []
            effective_budgets = []
            t0 = time.time()

            try:
                for fold_idx, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, next(iter(reps.values())),
                             labels, mode='full')
                ):
                    train_reps = {k: v[train_idx] for k, v in reps.items()}
                    val_reps = {k: v[val_idx] for k, v in reps.items()}
                    y_train = labels[train_idx]
                    y_val = labels[val_idx]
                    n_train = len(y_train)

                    # Budget cap for B and E
                    if condition in ('B_budget_cap', 'E_combined'):
                        budget_eff = min(budget, n_train // 2)
                    else:
                        budget_eff = budget
                    effective_budgets.append(budget_eff)

                    try:
                        t_fold = time.time()
                        X_train_h, feature_info = create_hybrid(
                            train_reps, y_train,
                            allocation_method='greedy',
                            budget=budget_eff,
                            importance_method='treeshap',
                            step_size=STEP_SIZE,
                            patience=PATIENCE,
                            internal_evaluator='rf',
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
                            'n_train': n_train,
                            'budget_effective': budget_eff,
                            'allocation': allocation,
                            'n_features': X_train_h.shape[1],
                            'hybrid_time': hybrid_time,
                            'downstream_scores': downstream_scores,
                        })
                    except Exception as e:
                        print(f"    Fold {fold_idx} FAILED: {e}")
                        fold_results.append({
                            'fold': fold_idx,
                            'error': str(e),
                        })
            finally:
                # Always restore original evaluator
                hybrid_module._evaluate_allocation = original_evaluate

            total_time = time.time() - t0

            # Aggregate
            valid_folds = [f for f in fold_results if 'error' not in f]
            if valid_folds:
                # Mean allocation
                all_rep_names = sorted(
                    set().union(*(f['allocation'].keys()
                                  for f in valid_folds)))
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
                                delta = -delta  # positive = hybrid better
                            delta_vs_best[model_name] = {
                                'delta': float(delta),
                                'hybrid': m,
                                'best_rep': best_info['score'],
                                'best_rep_name': best_info['rep'],
                            }

                n_feats = np.mean([f['n_features'] for f in valid_folds])
                eff_b_mean = float(np.mean(effective_budgets)) if effective_budgets else budget
                print(f"    Features: {n_feats:.0f} | "
                      f"Budget eff: {eff_b_mean:.0f} | "
                      f"Time: {total_time:.1f}s | "
                      f"Folds OK: {len(valid_folds)}/{n_folds}")
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
                eff_b_mean = float(budget)
                print(f"    ALL FOLDS FAILED")

            all_conditions.append({
                'condition': condition,
                'budget': budget,
                'effective_budget_mean': eff_b_mean,
                'wall_time': total_time,
                'n_folds_ok': len(valid_folds),
                'mean_allocation': mean_alloc,
                'mean_downstream': mean_downstream,
                'delta_vs_best_rep': delta_vs_best,
                'fold_results': fold_results,
            })

    # Verify monkey-patch restoration
    assert hybrid_module._evaluate_allocation is original_evaluate, \
        "CRITICAL: _evaluate_allocation not restored!"

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: DELTA vs BEST SINGLE REP")
    print("=" * 70)

    # Group by condition
    by_condition = defaultdict(list)
    for r in all_conditions:
        by_condition[r['condition']].append(r)

    for condition in CONDITIONS:
        entries = by_condition[condition]
        print(f"\n  {condition}:")
        print(f"    {'Budget':>6} {'Eff':>4}  ", end="")
        for mn in sorted(factories.keys()):
            print(f"  {mn:>12}", end="")
        print(f"  {'mean_Δ':>8}")
        print("    " + "-" * (6 + 4 + 2 + 14 * len(factories) + 10))

        for entry in entries:
            b = entry['budget']
            eb = entry['effective_budget_mean']
            print(f"    {b:>6} {eb:>4.0f}  ", end="")
            deltas = []
            for mn in sorted(factories.keys()):
                d = entry['delta_vs_best_rep'].get(mn, {}).get('delta')
                if d is not None:
                    sign = '+' if d >= 0 else ''
                    print(f"  {sign}{d:>11.4f}", end="")
                    deltas.append(d)
                else:
                    print(f"  {'N/A':>12}", end="")
            if deltas:
                md = np.mean(deltas)
                sign = '+' if md >= 0 else ''
                print(f"  {sign}{md:>7.4f}")
            else:
                print()

    # Best condition per budget
    print(f"\n  BEST CONDITION PER BUDGET (mean Δ across models):")
    by_budget = defaultdict(list)
    for r in all_conditions:
        by_budget[r['budget']].append(r)
    for budget in BUDGETS:
        entries = by_budget[budget]
        best_cond = None
        best_delta = -np.inf
        for entry in entries:
            deltas = [v.get('delta', -np.inf)
                      for v in entry['delta_vs_best_rep'].values()]
            if deltas:
                md = np.mean(deltas)
                if md > best_delta:
                    best_delta = md
                    best_cond = entry['condition']
        if best_cond:
            sign = '+' if best_delta >= 0 else ''
            print(f"    budget={budget}: {best_cond} "
                  f"(mean Δ={sign}{best_delta:.4f})")

    # ── Structured output ────────────────────────────────────────────────
    return make_structured_output(
        experiment_id='f10',
        dataset=dataset_name,
        data=data,
        config={
            'conditions': CONDITIONS,
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
            'conditions': all_conditions,
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser(
        'F10: Small Dataset Allocation Fixes',
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
                            'f10_small_dataset_allocation', args.dataset)
