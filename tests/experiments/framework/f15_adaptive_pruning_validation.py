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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments._shared import load_data, FULL_BUDGETS
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import (
    SPEED_TIERS, _resolve_models, _make_cv_splitter,
    _evaluate_hybrid_config, DEFAULT_MODELS,
)
from kirby.hybrid import compute_feature_importance

F15_DATASETS = [
    'abagym_EGFR', 'buchwald_hartwig', 'caco2',
    'flab_pk_jain_fcrnrt', 'herg_ki', 'logd',
]

# Domain detection (matches precompute)
def _get_domain(name):
    if name.startswith(('tcga_',)): return 'tcga'
    if name.startswith(('buchwald', 'bh_')): return 'reaction'
    if any(name.startswith(p) for p in ('flab_', 'abagym_', 'abcov_', 'ddg_', 'tz_', 'humanness', 'ginkgo_')):
        return 'antibody'
    return 'molecular'

def _get_metric(domain, task_type, labels):
    if task_type == 'classification':
        return 'f1_weighted' if len(np.unique(labels)) > 2 else 'auc'
    if domain == 'dta': return 'ci'
    if domain == 'antibody': return 'spearman'
    return 'rmse'


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

    # Resolve metric function
    from kirby.find_best_hybrid import _resolve_metric
    metric_fn, lower_is_better = _resolve_metric(metric_str, task_type)

    # Full tier config
    tier = SPEED_TIERS['full']
    strategies = tier['allocation_methods']
    importance_methods = tier['importance_methods']
    evaluator_names = tier['internal_evaluators']
    n_folds = tier['n_folds']

    # Budget filtering
    n_train_est = len(labels) - int(np.ceil(len(labels) / n_folds))
    budgets = [b for b in FULL_BUDGETS if b <= n_train_est]

    # Resolve models
    models = _resolve_models(None, task_type, 'full', n_samples=len(labels))

    print(f"  {len(reps)} reps, {len(labels)} samples, metric={metric_str}")
    print(f"  {len(strategies)} strategies × {len(importance_methods)} imp × "
          f"{len(evaluator_names)} eval × {len(models)} models × {len(budgets)} budgets")
    print(f"  {n_folds} folds")

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

    # Pre-compute allocation scores per (fold, evaluator)
    print(f"  Pre-computing allocation scores...")
    from kirby.find_best_hybrid import _evaluate_allocation
    alloc_cache = {}
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold_train_reps = {k: v[train_idx] for k, v in reps.items()}
        fold_val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Resolve evaluator names (expand 'match')
        all_eval_names = set()
        for ev in evaluator_names:
            if ev == 'match':
                for m in models:
                    all_eval_names.add(m)
            else:
                all_eval_names.add(ev)

        for ev in all_eval_names:
            scores = {}
            for rep_name in fold_train_reps:
                try:
                    X_tr = fold_train_reps[rep_name]
                    X_va = fold_val_reps[rep_name]
                    s = _evaluate_allocation(X_tr, X_va, y_train, y_val,
                                            internal_evaluator=ev)
                    scores[rep_name] = max(s, 0.0)
                except:
                    scores[rep_name] = 0.0
            alloc_cache[(fold_idx, ev)] = scores

    # Run ALL configs, recording PER-FOLD scores
    print(f"\n  Evaluating configs...")
    t_start = time.time()

    all_results = []
    total_configs = len(strategies) * len(importance_methods) * len(models) * len(budgets)
    # For evaluators: each model gets its own set
    print(f"  ~{total_configs * 5} total config×evaluator combos (before eval expansion)")

    config_count = 0
    for strat in strategies:
        for imp in importance_methods:
            for model_name, model in models.items():
                # Resolve evaluators for this model
                from kirby.find_best_hybrid import _make_cv_splitter as _mcv
                explicit_evals = [e for e in evaluator_names if e != 'match']
                model_evals = list(explicit_evals)
                if 'match' in evaluator_names:
                    match_map = {
                        'rf': 'rf', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost',
                        'mlp': 'mlp', 'svm': 'svm', 'gp': 'gp', 'ridge': 'ridge',
                        'knn': 'knn', 'catboost': 'catboost', 'dnn': 'dnn',
                        'cnn': 'cnn', 'bnn': 'bnn', 'ngboost': 'ngboost',
                    }
                    matched = match_map.get(model_name)
                    if matched and matched not in model_evals:
                        model_evals.append(matched)

                for ev in model_evals:
                    for bgt in budgets:
                        config_count += 1
                        if config_count % 500 == 0:
                            elapsed = time.time() - t_start
                            print(f"    {config_count} configs evaluated ({elapsed:.0f}s)...")

                        fold_scores = []
                        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                            fold_importance = importance_cache.get((fold_idx, imp))
                            fold_alloc = alloc_cache.get((fold_idx, ev))

                            try:
                                result = _evaluate_hybrid_config(
                                    base_reps=reps,
                                    labels=labels,
                                    allocation_method=strat,
                                    budget=bgt,
                                    importance_method=imp,
                                    model=model,
                                    model_name=model_name,
                                    internal_evaluator=ev,
                                    groups=groups,
                                    n_folds=n_folds,
                                    metric_fn=metric_fn,
                                    metric_name=metric_str,
                                    seed=42,
                                    augmentations=None,
                                    lower_is_better=lower_is_better,
                                    precomputed_folds=[fold_splits[fold_idx]],
                                    importance_cache={(0, imp): fold_importance} if fold_importance else None,
                                    alloc_cache={(0, ev): fold_alloc} if fold_alloc else None,
                                )
                                fold_scores.append(result.get('score_mean', float('nan')))
                            except Exception:
                                fold_scores.append(float('nan'))

                        all_results.append({
                            'strategy': strat,
                            'importance': imp,
                            'evaluator': ev,
                            'model': model_name,
                            'budget': bgt,
                            'fold_scores': fold_scores,
                            'score_mean': float(np.nanmean(fold_scores)) if fold_scores else float('nan'),
                        })

    elapsed = time.time() - t_start
    print(f"\n  Done: {len(all_results)} configs in {elapsed:.0f}s")

    # Save results
    output = {
        'experiment': 'f15_adaptive_pruning_validation',
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
            'models': list(models.keys()),
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
    parser = argparse.ArgumentParser(description='F15: Adaptive Pruning Validation')
    parser.add_argument('dataset', type=str, help='Dataset name')
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
