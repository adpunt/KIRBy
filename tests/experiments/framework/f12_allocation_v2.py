#!/usr/bin/env python3
"""F12: Allocation strategy comparison v2 — budget floor, swept evaluator & importance.

Paper: Framework
Question: With budget enforced as a hard floor+ceiling, which (allocation scheme ×
    importance method × internal evaluator) triple is the best default?

Changes from F01:
    - Budget is a hard constraint: strategies must produce exactly `budget` features
    - No tiering: all strategy × param combos tested on all datasets
    - Internal evaluator swept (rf, ridge, xgboost) — F07 showed Ridge wins antibody
    - Importance method swept (treeshap, deeplift, lightgbm_gain) — F05 families
    - Stability_selection dropped (F01/F11: catastrophic)
    - Two_stage dropped (F04: worse on 9/10 datasets)
    - Fixed dropped (ignores budget, not a real strategy)
    - Step_size {5, 10, 25} for greedy family (F03: useful range)

Usage:
    python f12_allocation_v2.py --dataset herg --reps all --mode full
    python f12_allocation_v2.py --dataset flab_thermo_tresanco --reps all --mode full
    python f12_allocation_v2.py --dataset herg --reps all --mode full --budgets 50,100,200,500
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from itertools import product

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


# =========================================================================
# STRATEGY CONFIGS: (display_name, allocation_method, extra_kwargs)
# =========================================================================

# Family 1: performance_weighted schemes
PERF_WEIGHTED_CONFIGS = [
    # linear (no transform)
    ('pw_linear', 'performance_weighted', {}),
    # softmax temperatures
    ('pw_softmax_T0.5', 'softmax_T0.5', {}),
    ('pw_softmax_T2', 'softmax_T2', {}),
    ('pw_softmax_T10', 'softmax_T10', {}),
    # power law
    ('pw_power_2', 'power_2', {}),
    ('pw_power_3', 'power_3', {}),
    # thresholded
    ('pw_thresh_0.3', 'performance_weighted',
     {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.3}}),
    ('pw_thresh_0.5', 'thresh_0.5', {}),
    ('pw_thresh_0.7', 'performance_weighted',
     {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.7}}),
    # top_k
    ('pw_top_2', 'performance_weighted',
     {'weighting_scheme': 'top_k', 'weighting_params': {'k': 2}}),
    ('pw_top_3', 'top_3', {}),
    ('pw_top_5', 'performance_weighted',
     {'weighting_scheme': 'top_k', 'weighting_params': {'k': 5}}),
    # rank_linear
    ('pw_rank_linear', 'rank_linear', {}),
    # rank_exp
    ('pw_rank_exp_d0.1', 'performance_weighted',
     {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.1}}),
    ('pw_rank_exp_d0.3', 'rank_exp', {}),
    ('pw_rank_exp_d0.5', 'performance_weighted',
     {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.5}}),
    # score_gap
    ('pw_score_gap_0.2', 'performance_weighted',
     {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.2}}),
    ('pw_score_gap_0.3', 'performance_weighted',
     {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.3}}),
    ('pw_score_gap_0.5', 'performance_weighted',
     {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.5}}),
    ('pw_score_gap_0.7', 'performance_weighted',
     {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.7}}),
    # entropy
    ('pw_entropy_0.5', 'performance_weighted',
     {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 0.5}}),
    ('pw_entropy_1.0', 'entropy', {}),
    ('pw_entropy_2.0', 'performance_weighted',
     {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 2.0}}),
    # min_floor
    ('pw_min_floor', 'min_floor_0.05', {}),
]

# Family 2: greedy
GREEDY_CONFIGS = [
    # greedy forced (no patience, must fill budget)
    ('greedy_forced_s5', 'greedy', {'step_size': 5, 'patience': None}),
    ('greedy_forced_s10', 'greedy', {'step_size': 10, 'patience': None}),
    ('greedy_forced_s25', 'greedy', {'step_size': 25, 'patience': None}),
    # greedy + fill (greedy stops naturally, fill from best single rep)
    ('greedy_fill_s5', 'greedy', {'step_size': 5, 'budget_fill': True}),
    ('greedy_fill_s10', 'greedy', {'step_size': 10, 'budget_fill': True}),
    ('greedy_fill_s25', 'greedy', {'step_size': 25, 'budget_fill': True}),
    # dominant_greedy with dominant_fraction sweep
    ('dom_greedy_f0.5', 'dominant_greedy', {'dominant_fraction': 0.5}),
    ('dom_greedy_f0.7', 'dominant_greedy', {'dominant_fraction': 0.7}),
    ('dom_greedy_f0.9', 'dominant_greedy', {'dominant_fraction': 0.9}),
    # greedy_feature
    ('greedy_feature', 'greedy_feature', {}),
]

ALL_STRATEGY_CONFIGS = PERF_WEIGHTED_CONFIGS + GREEDY_CONFIGS

IMPORTANCE_METHODS = ['deeplift', 'lightgbm_gain']
# Matched internal_evaluator = downstream_model pairs only (F13 finding)
MATCHED_PAIRS = [('rf', 'rf'), ('lightgbm', 'lightgbm'), ('mlp', 'mlp'), ('svm', 'svm')]
BUDGETS = [50, 150, 300]


def _incremental_save(all_configs, dataset_name, domain, metric_name,
                      lower_is_better, n_samples, reps, models, budgets,
                      n_folds, max_folds, output_dir, t_start):
    """Save partial results so crashes don't lose everything."""
    import json
    from pathlib import Path
    out_dir = Path(output_dir) / 'f12_allocation_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = {
        'experiment': 'f12',
        'dataset': dataset_name,
        'domain': domain,
        'primary_metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_samples': n_samples,
        'n_reps': len(reps),
        'partial': True,
        'n_configs_so_far': len(all_configs),
        'config': {
            'strategies': [s[0] for s in ALL_STRATEGY_CONFIGS],
            'importance_methods': IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': budgets,
            'models': list(models.keys()),
            'n_folds': n_folds,
            'max_folds': max_folds,
        },
        'results': {'configs': all_configs},
        'elapsed_s': round(time.time() - t_start, 1),
    }
    path = out_dir / f'{dataset_name}.json'
    path.write_text(json.dumps(partial, default=str))
    print(f"  [SAVED {len(all_configs)} configs to {path}]", flush=True)


def _eval_one(strat_display, alloc_method, extra_kwargs,
              budget, model_name, model_factory,
              reps, labels, groups, n_folds, max_folds,
              metric_fn, importance_method, internal_evaluator,
              precomputed_baselines=None, precomputed_importance=None):
    """Run a single config. Returns result dict."""
    mode = 'intermediary' if n_folds == 1 else 'full'
    sample_X = next(iter(reps.values()))
    fold_scores = []
    last_n_features = 0
    last_allocation = {}
    last_utilization = 0.0
    t0 = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(cv_split(
        groups, n_folds, sample_X, labels, mode=mode
    )):
        if fold_i >= max_folds:
            break
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        hybrid_kwargs = dict(
            allocation_method=alloc_method,
            budget=budget,
            importance_method=importance_method,
            internal_evaluator=internal_evaluator,
        )
        if extra_kwargs:
            hybrid_kwargs.update(extra_kwargs)
        if precomputed_baselines is not None:
            hybrid_kwargs['precomputed_baselines'] = precomputed_baselines
        if precomputed_importance is not None:
            hybrid_kwargs['precomputed_importance'] = precomputed_importance

        try:
            X_train_h, feature_info = create_hybrid(
                train_reps, y_train, **hybrid_kwargs,
            )
            X_val_h = apply_feature_selection(val_reps, feature_info)
        except Exception as e:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_h)
        X_val_s = scaler.transform(X_val_h)
        last_n_features = X_train_h.shape[1]
        last_allocation = feature_info.get('allocation', {})
        last_utilization = last_n_features / budget if budget > 0 else 0

        try:
            model = model_factory()
            model.fit(X_train_s, y_train)
            fold_scores.append(float(metric_fn(y_val, model.predict(X_val_s))))
        except Exception:
            continue

    wall_time = time.time() - t0

    if not fold_scores:
        score_mean, score_std = float('nan'), float('nan')
        n_features = 0
    else:
        score_mean = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        n_features = last_n_features

    return {
        'strategy': strat_display,
        'allocation_method': alloc_method,
        'model': model_name,
        'budget': budget,
        'importance': importance_method,
        'internal_evaluator': internal_evaluator,
        'score_mean': score_mean,
        'score_std': score_std,
        'fold_scores': fold_scores,
        'n_features': n_features,
        'utilization': round(last_utilization, 2),
        'allocation_dict': {k: int(v) for k, v in last_allocation.items()},
        'wall_time': round(wall_time, 2),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run flat allocation strategy comparison with budget floor."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F12: ALLOCATION STRATEGY v2 (budget floor)")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    # Build model factories for all downstream models in matched pairs
    all_model_names = sorted(set(m for _, m in MATCHED_PAIRS))
    models = get_model_factories(all_model_names, task_type=task_type,
                                 n_samples=n_samples)
    n_folds = config['n_splits']
    max_folds = config.get('max_folds', n_folds)
    budgets = BUDGETS  # Always use F12's own budgets

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    n_strats = len(ALL_STRATEGY_CONFIGS)
    n_imp = len(IMPORTANCE_METHODS)
    n_pairs = len(MATCHED_PAIRS)
    n_bud = len(budgets)
    total = n_strats * n_imp * n_pairs * n_bud

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples, {len(reps)} reps)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Strategies: {n_strats} ({len(PERF_WEIGHTED_CONFIGS)} perf_weighted + "
          f"{len(GREEDY_CONFIGS)} greedy)")
    print(f"  Importance methods: {IMPORTANCE_METHODS}")
    print(f"  Matched pairs: {MATCHED_PAIRS}")
    print(f"  Budgets: {budgets}")
    print(f"  Folds: {max_folds}/{n_folds}")
    print(f"  Total configs: {total}")
    print(flush=True)

    all_configs = []
    idx = 0
    t_start = time.time()

    # TEMPORARY: Precompute importance + baselines to avoid redundant
    # RF+SHAP fits across configs. Remove when experiment is done.
    from kirby.hybrid import compute_feature_importance, _evaluate_allocation
    _mode = 'intermediary' if n_folds == 1 else 'full'
    _sample_X = next(iter(reps.values()))
    _train_idx, _val_idx = next(iter(cv_split(groups, n_folds, _sample_X, labels, mode=_mode)))
    _train_reps_pre = {k: v[_train_idx] for k, v in reps.items()}
    _val_reps_pre = {k: v[_val_idx] for k, v in reps.items()}
    _train_labels_pre = labels[_train_idx]
    _val_labels_pre = labels[_val_idx]

    cached_importance = {}
    for imp_method in IMPORTANCE_METHODS:
        print(f"  Precomputing {imp_method} importance...", end='', flush=True)
        t_imp = time.time()
        cached_importance[imp_method] = compute_feature_importance(
            _train_reps_pre, _train_labels_pre, method=imp_method,
        )
        print(f" {time.time() - t_imp:.0f}s", flush=True)

    cached_baselines = {}
    for int_eval, _ in MATCHED_PAIRS:
        if int_eval in cached_baselines:
            continue
        print(f"  Precomputing {int_eval} baselines...", end='', flush=True)
        t_bl = time.time()
        bl = {}
        for rep_name, X_tr in _train_reps_pre.items():
            X_vl = _val_reps_pre[rep_name]
            X_tr_c = np.nan_to_num(np.clip(X_tr, -1e10, 1e10), nan=0.0)
            X_vl_c = np.nan_to_num(np.clip(X_vl, -1e10, 1e10), nan=0.0)
            bl[rep_name] = _evaluate_allocation(
                X_tr_c, X_vl_c, _train_labels_pre, _val_labels_pre,
                internal_evaluator=int_eval,
            )
        cached_baselines[int_eval] = bl
        print(f" {time.time() - t_bl:.0f}s", flush=True)

    print(f"  Precomputation done in {time.time() - t_start:.0f}s\n", flush=True)

    # Determine which strategies benefit from precomputed data
    _PW_ALLOCS = {'performance_weighted', 'softmax_T0.5', 'softmax_T2', 'softmax_T10',
                  'power_2', 'power_3', 'thresh_0.5', 'top_3', 'min_floor_0.05',
                  'rank_linear', 'rank_exp', 'score_gap', 'entropy'}

    for (strat_display, alloc_method, extra_kw) in ALL_STRATEGY_CONFIGS:
        is_pw = alloc_method in _PW_ALLOCS
        for imp_method in IMPORTANCE_METHODS:
            for int_eval, model_name in MATCHED_PAIRS:
                if model_name not in models:
                    continue
                model_factory = models[model_name]
                for budget in budgets:
                    idx += 1
                    entry = _eval_one(
                        strat_display, alloc_method, extra_kw,
                        budget, model_name, model_factory,
                        reps, labels, groups, n_folds, max_folds,
                        metric_fn, imp_method, int_eval,
                        precomputed_baselines=cached_baselines.get(int_eval) if is_pw else None,
                        precomputed_importance=cached_importance.get(imp_method) if is_pw else None,
                    )
                    all_configs.append(entry)

                    if idx <= 10 or idx % 50 == 0 or idx == total:
                        elapsed = time.time() - t_start
                        rate = idx / elapsed if elapsed > 0 else 0
                        eta = (total - idx) / rate if rate > 0 else 0
                        print(f"  [{idx}/{total}] {strat_display} / "
                              f"{imp_method} / {int_eval}={model_name} / "
                              f"b={budget}: "
                              f"{metric_name}={entry['score_mean']:.4f} "
                              f"dim={entry['n_features']} "
                              f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                              flush=True)

                    # Incremental save every 200 configs
                    if idx % 200 == 0:
                        _incremental_save(all_configs, dataset_name, domain,
                                          metric_name, lower_is_better,
                                          n_samples, reps, models, budgets,
                                          n_folds, max_folds, output_dir,
                                          t_start)

    total_time = time.time() - t_start

    # ── Summary ──
    valid = [c for c in all_configs if c['score_mean'] == c['score_mean']]
    if valid:
        if lower_is_better:
            best = min(valid, key=lambda c: c['score_mean'])
        else:
            best = max(valid, key=lambda c: c['score_mean'])

        print(f"\n{'=' * 70}")
        print(f"BEST: {best['strategy']} / {best['importance']} / "
              f"{best['internal_evaluator']} / b={best['budget']} / "
              f"{best['model']}: {metric_name}={best['score_mean']:.4f} "
              f"dim={best['n_features']}")
        print(f"Total: {len(all_configs)} configs in {total_time:.0f}s")
        print(f"NaN configs: {len(all_configs) - len(valid)}")

    return {
        'experiment': 'f12',
        'dataset': dataset_name,
        'domain': domain,
        'primary_metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_samples': n_samples,
        'n_reps': len(reps),
        'config': {
            'strategies': [s[0] for s in ALL_STRATEGY_CONFIGS],
            'importance_methods': IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': budgets,
            'models': list(models.keys()),
            'n_folds': n_folds,
            'max_folds': max_folds,
        },
        'results': {
            'configs': all_configs,
        },
        'total_time_s': round(total_time, 1),
    }


if __name__ == '__main__':
    parser = create_experiment_parser('F12: Allocation Strategy v2',
                                     has_delta=True, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    if hasattr(args, 'budgets') and args.budgets:
        config['budgets'] = [int(b) for b in args.budgets.split(',')]

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Budgets: {config.get('budgets', BUDGETS)}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'f12_allocation_v2',
                            args.dataset)
