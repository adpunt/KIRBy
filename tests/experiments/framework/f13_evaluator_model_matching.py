#!/usr/bin/env python3
"""F13: Internal evaluator × downstream model matching test.

Paper: Framework
Question: Should the internal evaluator used during allocation match the
    downstream model? Does RF as evaluator find better features for Ridge
    than Ridge itself would?

Tests two conditions per (allocation, budget, importance) combo:
    A. matched:   internal_evaluator = downstream_model
    B. cross:     internal_evaluator != downstream_model (all combos)

Uses the top allocation strategies from F12. If F12 hasn't run yet, uses
a sensible default set.

The internal_evaluator param can be set to 'match' to auto-match with the
downstream model — this is what would be used in production if matching wins.

Usage:
    python f13_evaluator_model_matching.py --dataset herg --reps all --mode full
    python f13_evaluator_model_matching.py --dataset flab_thermo_tresanco --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


# Representative allocation strategies (covers both families)
STRATEGY_CONFIGS = [
    ('pw_linear', 'performance_weighted', {}),
    ('pw_softmax_T2', 'softmax_T2', {}),
    ('pw_power_2', 'power_2', {}),
    ('greedy_s10', 'greedy', {'step_size': 10}),
    ('dom_greedy_f0.7', 'dominant_greedy', {'dominant_fraction': 0.7}),
]

IMPORTANCE_METHODS = ['treeshap', 'lightgbm_gain']
EVALUATORS = ['rf', 'xgboost', 'lightgbm', 'mlp', 'svm']
MODELS = ['rf', 'xgboost', 'lightgbm', 'mlp', 'svm']
BUDGETS = [150, 300]


def _eval_one(strat_display, alloc_method, extra_kwargs,
              budget, model_name, model_factory,
              reps, labels, groups, n_folds, max_folds,
              metric_fn, importance_method, internal_evaluator):
    """Run a single config. Returns result dict."""
    mode = 'intermediary' if n_folds == 1 else 'full'
    sample_X = next(iter(reps.values()))
    fold_scores = []
    last_n_features = 0
    last_allocation = {}
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

        try:
            X_train_h, feature_info = create_hybrid(
                train_reps, y_train, **hybrid_kwargs,
            )
            X_val_h = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_h)
        X_val_s = scaler.transform(X_val_h)
        last_n_features = X_train_h.shape[1]
        last_allocation = feature_info.get('allocation', {})

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
        'matched': internal_evaluator == model_name,
        'score_mean': score_mean,
        'score_std': score_std,
        'fold_scores': fold_scores,
        'n_features': n_features,
        'allocation_dict': {k: int(v) for k, v in last_allocation.items()},
        'wall_time': round(wall_time, 2),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run evaluator × model matching test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F13: EVALUATOR × MODEL MATCHING")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(MODELS, task_type=task_type, n_samples=n_samples)
    n_folds = config['n_splits']
    max_folds = config.get('max_folds', n_folds)
    budgets = BUDGETS

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    n_strats = len(STRATEGY_CONFIGS)
    n_imp = len(IMPORTANCE_METHODS)
    n_eval = len(EVALUATORS)
    n_bud = len(budgets)
    n_mod = len(models)
    total = n_strats * n_imp * n_eval * n_bud * n_mod

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples, {len(reps)} reps)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Strategies: {n_strats}")
    print(f"  Importance methods: {IMPORTANCE_METHODS}")
    print(f"  Internal evaluators: {EVALUATORS}")
    print(f"  Budgets: {budgets}")
    print(f"  Models: {list(models.keys())}")
    print(f"  Folds: {max_folds}/{n_folds}")
    print(f"  Total configs: {total}")
    print(flush=True)

    all_configs = []
    idx = 0
    t_start = time.time()

    for (strat_display, alloc_method, extra_kw) in STRATEGY_CONFIGS:
        for imp_method in IMPORTANCE_METHODS:
            for int_eval in EVALUATORS:
                for budget in budgets:
                    for model_name, model_factory in models.items():
                        idx += 1
                        entry = _eval_one(
                            strat_display, alloc_method, extra_kw,
                            budget, model_name, model_factory,
                            reps, labels, groups, n_folds, max_folds,
                            metric_fn, imp_method, int_eval,
                        )
                        all_configs.append(entry)

                        if idx <= 5 or idx % 20 == 0 or idx == total:
                            elapsed = time.time() - t_start
                            rate = idx / elapsed if elapsed > 0 else 0
                            eta = (total - idx) / rate if rate > 0 else 0
                            is_match = "MATCH" if entry['matched'] else "cross"
                            print(f"  [{idx}/{total}] {strat_display} / "
                                  f"{imp_method} / eval={int_eval} / "
                                  f"b={budget} / {model_name} [{is_match}]: "
                                  f"{metric_name}={entry['score_mean']:.4f} "
                                  f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                                  flush=True)

                        # Incremental save every 200 configs
                        if idx % 200 == 0:
                            import json as _json
                            from pathlib import Path as _Path
                            _out = _Path(output_dir) / 'f13_evaluator_model_matching'
                            _out.mkdir(parents=True, exist_ok=True)
                            _p = _out / f'{dataset_name}.json'
                            _p.write_text(_json.dumps({
                                'experiment': 'f13', 'dataset': dataset_name,
                                'partial': True, 'n_configs_so_far': len(all_configs),
                                'results': {'configs': all_configs},
                            }, default=str))
                            print(f"  [SAVED {len(all_configs)} configs]", flush=True)

    total_time = time.time() - t_start

    # ── Analysis: matched vs cross ──
    valid = [c for c in all_configs if c['score_mean'] == c['score_mean']]
    matched = [c for c in valid if c['matched']]
    crossed = [c for c in valid if not c['matched']]

    print(f"\n{'=' * 70}")
    print(f"MATCHED vs CROSS ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Total valid: {len(valid)}, matched: {len(matched)}, cross: {len(crossed)}")

    if matched and crossed:
        if lower_is_better:
            match_avg = np.mean([c['score_mean'] for c in matched])
            cross_avg = np.mean([c['score_mean'] for c in crossed])
            print(f"  Matched avg {metric_name}: {match_avg:.4f}")
            print(f"  Cross avg {metric_name}:   {cross_avg:.4f}")
            print(f"  Delta: {match_avg - cross_avg:+.4f} ({'matched better' if match_avg < cross_avg else 'cross better'})")
        else:
            match_avg = np.mean([c['score_mean'] for c in matched])
            cross_avg = np.mean([c['score_mean'] for c in crossed])
            print(f"  Matched avg {metric_name}: {match_avg:.4f}")
            print(f"  Cross avg {metric_name}:   {cross_avg:.4f}")
            print(f"  Delta: {match_avg - cross_avg:+.4f} ({'matched better' if match_avg > cross_avg else 'cross better'})")

        # Per-model breakdown
        print(f"\n  Per downstream model:")
        for model_name in models:
            m_match = [c for c in matched if c['model'] == model_name]
            m_cross = [c for c in crossed if c['model'] == model_name]
            if m_match and m_cross:
                mm = np.mean([c['score_mean'] for c in m_match])
                mc = np.mean([c['score_mean'] for c in m_cross])
                if lower_is_better:
                    winner = 'match' if mm < mc else 'cross'
                else:
                    winner = 'match' if mm > mc else 'cross'
                print(f"    {model_name:<10}: matched={mm:.4f}  cross={mc:.4f}  → {winner}")

    print(f"\n  Total time: {total_time:.0f}s")

    return {
        'experiment': 'f13',
        'dataset': dataset_name,
        'domain': domain,
        'primary_metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_samples': n_samples,
        'n_reps': len(reps),
        'config': {
            'strategies': [s[0] for s in STRATEGY_CONFIGS],
            'importance_methods': IMPORTANCE_METHODS,
            'internal_evaluators': EVALUATORS,
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
    parser = create_experiment_parser('F13: Evaluator × Model Matching',
                                     has_delta=True, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'f13_evaluator_model_matching',
                            args.dataset)
