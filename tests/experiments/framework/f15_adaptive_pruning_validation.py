#!/usr/bin/env python3
"""F15: Adaptive Pruning Validation

Runs the full search tier on 6 datasets (same as F12) with NO augmentation,
recording PER-FOLD scores for every config. Then simulates pruning rules:
- After fold 1, which strategies/evaluators/importance methods would be pruned?
- Would those pruned values have been the winner on folds 2-5?

This validates whether after-fold-1 pruning is safe before implementing it
in find_best_hybrid.

Datasets: abagym_EGFR, buchwald_hartwig, caco2, flab_pk_jain_fcrnrt, herg_ki, logd
Search: 13 strategies × 3 importance × 5 evaluators × 13 models × full budgets × 5 folds
Augmentation: none

Output: tests/results/framework/f15_adaptive_pruning/{dataset}.json
Each JSON has per-fold, per-config scores for full analysis.

Usage:
    python tests/experiments/framework/f15_adaptive_pruning_validation.py <dataset>
    python tests/experiments/framework/f15_adaptive_pruning_validation.py caco2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments._shared import load_data, FULL_BUDGETS
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import (
    SPEED_TIERS, MODEL_MAX_BUDGET, _resolve_models, _resolve_metric,
    _make_cv_splitter, _impute_nan,
)
from kirby.hybrid import compute_feature_importance, create_hybrid, apply_feature_selection

F15_DATASETS = [
    'abagym_EGFR', 'buchwald_hartwig', 'caco2',
    'flab_pk_jain_fcrnrt', 'herg_ki', 'logd',
]


def _get_domain(name):
    if name.startswith(('tcga_',)): return 'tcga'
    if name.startswith(('buchwald', 'bh_')): return 'reaction'
    if any(name.startswith(p) for p in ('flab_', 'abagym_', 'abcov_', 'ddg_',
           'tz_', 'humanness', 'ginkgo_')):
        return 'antibody'
    return 'molecular'


def _get_metric(domain, task_type, labels):
    if task_type == 'classification':
        return 'f1_weighted' if len(np.unique(labels)) > 2 else 'auc'
    if domain == 'dta': return 'ci'
    if domain == 'antibody': return 'spearman'
    return 'rmse'


def _budgets_for_model(model_name, all_budgets):
    """Filter budgets by per-model cap from MODEL_MAX_BUDGET."""
    cap = MODEL_MAX_BUDGET.get(model_name)
    if cap is None:
        return all_budgets
    return [b for b in all_budgets if b <= cap]


def _evaluators_for_model(model_name, evaluator_names):
    """Expand 'match' to the model's own type, deduplicate."""
    match_map = {
        'rf': 'rf', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost',
        'mlp': 'mlp', 'svm': 'svm', 'gp': 'gp', 'ridge': 'ridge',
        'knn': 'knn', 'catboost': 'catboost', 'dnn': 'dnn',
        'cnn': 'cnn', 'bnn': 'bnn', 'ngboost': 'ngboost',
    }
    explicit = [e for e in evaluator_names if e != 'match']
    evs = list(explicit)
    if 'match' in evaluator_names:
        matched = match_map.get(model_name)
        if matched and matched not in evs:
            evs.append(matched)
    return evs if evs else ['rf']


def _evaluate_single_fold(base_reps, labels, groups, fold_split,
                          allocation_method, budget, importance_method,
                          model, model_name, internal_evaluator,
                          metric_fn, metric_str, lower_is_better,
                          precomputed_importance):
    """Evaluate one config on one fold. Returns score or NaN."""
    from sklearn.preprocessing import StandardScaler

    train_idx, val_idx = fold_split
    train_reps = {k: v[train_idx] for k, v in base_reps.items()}
    val_reps = {k: v[val_idx] for k, v in base_reps.items()}
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Create hybrid
    X_train, feature_info = create_hybrid(
        train_reps, y_train,
        allocation_method=allocation_method,
        budget=budget,
        importance_method=importance_method,
        internal_evaluator=internal_evaluator,
        augmentations=None,
        augmentation_strategy='none',
        model=model,
        metric_fn=metric_fn,
        lower_is_better=lower_is_better,
        precomputed_importance=precomputed_importance,
    )
    X_val = apply_feature_selection(val_reps, feature_info)

    # Scale and impute
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_train_s, X_val_s, _ = _impute_nan(X_train_s, X_val_s)

    # Predict and score
    m = deepcopy(model)
    if hasattr(m, 'n_neighbors') and hasattr(m, 'set_params'):
        k = min(5, max(1, int(np.sqrt(len(train_idx)))))
        m.set_params(n_neighbors=k)
    m.fit(X_train_s, y_train)

    from kirby.find_best_hybrid import _predict_for_metric
    y_pred = _predict_for_metric(m, X_val_s, metric_fn)
    return float(metric_fn(y_val, y_pred))


def run_f15(dataset_name, output_dir):
    ds_config = DATASET_REPS.get(dataset_name)
    if ds_config is None:
        print(f"  {dataset_name}: not in DATASET_REPS")
        return

    domain = _get_domain(dataset_name)
    general_names = ds_config.get('general', [])
    needed = sorted(set(general_names))

    print(f"\n{'='*70}")
    print(f"F15 Adaptive Pruning Validation: {dataset_name}")
    print(f"{'='*70}")

    data = load_data(dataset_name, rep_names=needed)
    reps = {k: data['reps'][k] for k in general_names if k in data['reps']}
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    task_type = data.get('task_type', 'regression')
    metric_str = _get_metric(domain, task_type, labels)
    metric_fn, lower_is_better = _resolve_metric(metric_str)

    # Full tier config
    tier = SPEED_TIERS['full']
    strategies = tier['allocation_methods']
    importance_methods = tier['importance_methods']
    evaluator_names = tier['internal_evaluators']
    n_folds = tier['n_folds']

    # Budget filtering by n_train
    n_train_est = len(labels) - int(np.ceil(len(labels) / n_folds))
    budgets = [b for b in FULL_BUDGETS if b <= n_train_est]

    # Resolve models
    models = _resolve_models(None, task_type, 'full', n_samples=len(labels))

    print(f"  {len(reps)} reps, {len(labels)} samples, metric={metric_str} "
          f"(lower_is_better={lower_is_better})")
    print(f"  {len(strategies)} strategies × {len(importance_methods)} imp × "
          f"{len(evaluator_names)} eval × {len(models)} models × {len(budgets)} budgets")
    print(f"  {n_folds} folds, budgets={budgets}")

    # Pre-compute fold splits
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed=42)
    sample_X = next(iter(reps.values()))
    split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
    fold_splits = list(splitter.split(*split_args))

    # Pre-compute importance per (fold, method)
    print(f"\n  Pre-computing importance scores...")
    importance_cache = {}
    for fold_idx, (train_idx, _) in enumerate(fold_splits):
        fold_train_reps = {k: v[train_idx] for k, v in reps.items()}
        fold_train_labels = labels[train_idx]
        for imp_method in importance_methods:
            try:
                importance_cache[(fold_idx, imp_method)] = compute_feature_importance(
                    fold_train_reps, fold_train_labels, method=imp_method)
            except Exception as e:
                print(f"    WARNING: importance ({fold_idx}, {imp_method}) failed: {e}")

    # Count configs
    total = 0
    for model_name in models:
        model_budgets = _budgets_for_model(model_name, budgets)
        model_evals = _evaluators_for_model(model_name, evaluator_names)
        total += len(strategies) * len(importance_methods) * len(model_evals) * len(model_budgets)
    print(f"  {total} configs × {n_folds} folds = {total * n_folds} evaluations")

    # Build flat config list
    configs = []
    for strat in strategies:
        for imp in importance_methods:
            for model_name in models:
                model_budgets = _budgets_for_model(model_name, budgets)
                model_evals = _evaluators_for_model(model_name, evaluator_names)
                for ev in model_evals:
                    for bgt in model_budgets:
                        configs.append((strat, imp, model_name, ev, bgt))

    # Evaluate one config across all folds
    # Selection dedup: if two configs produce the same feature selection
    # (same strategy + importance + evaluator + budget → same allocation),
    # we can reuse scores across downstream models.
    _selection_cache = {}  # (fold_idx, feature_hash) -> {model_name: score}

    def _hash_features(feature_info):
        """Hash the feature selection for dedup."""
        parts = []
        for name in sorted(feature_info.keys()):
            info = feature_info[name]
            if isinstance(info, dict) and 'selected_indices' in info:
                parts.append(f"{name}:{tuple(info['selected_indices'])}")
        return hash(tuple(parts))

    def _eval_config(cfg):
        strat, imp, model_name, ev, bgt = cfg
        model = models[model_name]
        fold_scores = []

        for fold_idx in range(n_folds):
            fold_importance = importance_cache.get((fold_idx, imp))

            try:
                score = _evaluate_single_fold(
                    base_reps=reps,
                    labels=labels,
                    groups=groups,
                    fold_split=fold_splits[fold_idx],
                    allocation_method=strat,
                    budget=bgt,
                    importance_method=imp,
                    model=model,
                    model_name=model_name,
                    internal_evaluator=ev,
                    metric_fn=metric_fn,
                    metric_str=metric_str,
                    lower_is_better=lower_is_better,
                    precomputed_importance=fold_importance,
                )
                fold_scores.append(score)
            except Exception:
                fold_scores.append(float('nan'))

        valid = [s for s in fold_scores if s == s]
        return {
            'strategy': strat,
            'importance': imp,
            'evaluator': ev,
            'matched': (ev == model_name),
            'model': model_name,
            'budget': bgt,
            'fold_scores': fold_scores,
            'score_mean': float(np.mean(valid)) if valid else float('nan'),
            'score_std': float(np.std(valid)) if valid else float('nan'),
            'n_valid_folds': len(valid),
        }

    # Parallel evaluation
    print(f"\n  Evaluating {len(configs)} configs (parallel)...")
    t_start = time.time()
    n_jobs = min(len(configs), os.cpu_count() or 1)

    try:
        from joblib import Parallel, delayed
        all_results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_eval_config)(cfg) for cfg in configs
        )
        elapsed = time.time() - t_start
        print(f"  Parallel done: {len(all_results)} configs, {n_jobs} workers, "
              f"{elapsed:.0f}s")
    except ImportError:
        print(f"  joblib unavailable, running sequentially...")
        all_results = []
        for i, cfg in enumerate(configs, 1):
            if i % 200 == 0:
                elapsed = time.time() - t_start
                print(f"    {i}/{len(configs)} ({i/len(configs)*100:.0f}%) "
                      f"{elapsed:.0f}s")
            all_results.append(_eval_config(cfg))

    elapsed = time.time() - t_start
    print(f"\n  Done: {len(all_results)} configs in {elapsed:.0f}s")
    n_nan = sum(1 for r in all_results if r['n_valid_folds'] < n_folds)
    if n_nan:
        print(f"  WARNING: {n_nan} configs had NaN folds")

    # Save results
    output = {
        'experiment': 'f15_adaptive_pruning_validation',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dataset': dataset_name,
        'domain': domain,
        'metric': metric_str,
        'lower_is_better': lower_is_better,
        'n_samples': len(labels),
        'n_reps': len(reps),
        'n_folds': n_folds,
        'config': {
            'strategies': strategies,
            'importance_methods': importance_methods,
            'evaluators': evaluator_names,
            'budgets': budgets,
            'models': sorted(models.keys()),
        },
        'results': all_results,
    }

    out_dir = Path(output_dir) / 'f15_adaptive_pruning'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{dataset_name}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='F15: Adaptive Pruning Validation')
    parser.add_argument('dataset', type=str, help='Dataset name or "all"')
    parser.add_argument('-o', '--output', type=str,
                        default='tests/results/framework',
                        help='Output directory')
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in F15_DATASETS:
            run_f15(ds, args.output)
    else:
        run_f15(args.dataset, args.output)


if __name__ == '__main__':
    main()
