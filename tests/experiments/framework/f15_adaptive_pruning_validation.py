#!/usr/bin/env python3
"""F15: Adaptive Pruning Validation

Runs the full search tier on 6 datasets (same as F12) with NO augmentation,
recording PER-FOLD scores for every config. Then simulates pruning rules:
- After fold 1, which strategies/evaluators/importance methods would be pruned?
- Would those pruned values have been the winner on folds 2-5?

This validates whether after-fold-1 pruning is safe before implementing it
in find_best_hybrid.

Datasets: abagym_EGFR, buchwald_hartwig, caco2, flab_pk_jain_fcrnrt, herg_ki, logd
Search: 13 strategies × 3 importance × 5 evaluators × 6 models × [250,500,1000,2000] budgets × 5 folds
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
# Force line-buffered stdout so SLURM output is visible in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import time
import argparse
print(f"[{time.strftime('%H:%M:%S')}] Starting imports...", flush=True)
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

print(f"[{time.strftime('%H:%M:%S')}] Importing kirby modules...", flush=True)
from experiments._shared import load_data
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import (
    SPEED_TIERS, MODEL_MAX_BUDGET, _resolve_models, _resolve_metric,
    _make_cv_splitter, _impute_nan,
)
from kirby.hybrid import compute_feature_importance, create_hybrid, apply_feature_selection
print(f"[{time.strftime('%H:%M:%S')}] Imports done.", flush=True)

F15_DATASETS = [
    'abagym_EGFR', 'buchwald_hartwig', 'caco2',
    'flab_pk_jain_fcrnrt', 'herg_ki', 'logd',
]

F15_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'mlp', 'lightgbm']
F15_BUDGETS = [50, 100, 250, 500, 1000, 2000]


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
                          precomputed_importance,
                          precomputed_alloc_scores=None):
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
        precomputed_alloc_scores=precomputed_alloc_scores,
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
        print(f"  {dataset_name}: not in DATASET_REPS", flush=True)
        return

    domain = _get_domain(dataset_name)
    general_names = ds_config.get('general', [])
    needed = sorted(set(general_names))

    print(f"\n{'='*70}", flush=True)
    print(f"F15 Adaptive Pruning Validation: {dataset_name}", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading data for {dataset_name}...", flush=True)
    data = load_data(dataset_name, rep_names=needed)
    reps = {k: data['reps'][k] for k in general_names if k in data['reps']}
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    task_type = data.get('task_type', 'regression')
    metric_str = _get_metric(domain, task_type, labels)
    metric_fn, lower_is_better = _resolve_metric(metric_str)
    print(f"[{time.strftime('%H:%M:%S')}] Data loaded: {len(reps)} reps, {len(labels)} samples", flush=True)

    # Full tier config
    tier = SPEED_TIERS['full']
    strategies = tier['allocation_methods']
    importance_methods = tier['importance_methods']
    evaluator_names = tier['internal_evaluators']
    n_folds = tier['n_folds']

    # Budget filtering by n_train
    n_train_est = len(labels) - int(np.ceil(len(labels) / n_folds))
    budgets = [b for b in F15_BUDGETS if b <= n_train_est]

    # Core 6 models (lit-validated: rf, xgboost, ridge, svm, mlp, lightgbm)
    models = _resolve_models(F15_MODELS, task_type, 'full', n_samples=len(labels))

    print(f"  {len(reps)} reps, {len(labels)} samples, metric={metric_str} "
          f"(lower_is_better={lower_is_better})", flush=True)
    print(f"  {len(strategies)} strategies × {len(importance_methods)} imp × "
          f"{len(evaluator_names)} eval × {len(models)} models × {len(budgets)} budgets", flush=True)
    print(f"  {n_folds} folds, budgets={budgets}", flush=True)

    # Pre-compute fold splits
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed=42)
    sample_X = next(iter(reps.values()))
    split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
    fold_splits = list(splitter.split(*split_args))

    # Pre-compute importance per (fold, method)
    print(f"\n[{time.strftime('%H:%M:%S')}] Pre-computing importance scores "
          f"({n_folds} folds × {len(importance_methods)} methods)...", flush=True)
    importance_cache = {}
    _imp_t0 = time.time()
    for fold_idx, (train_idx, _) in enumerate(fold_splits):
        fold_train_reps = {k: v[train_idx] for k, v in reps.items()}
        fold_train_labels = labels[train_idx]
        for imp_method in importance_methods:
            t0 = time.time()
            try:
                importance_cache[(fold_idx, imp_method)] = compute_feature_importance(
                    fold_train_reps, fold_train_labels, method=imp_method)
                print(f"    fold {fold_idx} {imp_method}: {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"    WARNING: fold {fold_idx} {imp_method} FAILED ({time.time()-t0:.1f}s): {e}", flush=True)
    print(f"  Importance done in {time.time()-_imp_t0:.0f}s, "
          f"{len(importance_cache)} entries cached", flush=True)

    # Pre-compute allocation scores per (fold, evaluator)
    # Same rep scored by the same evaluator produces the same alloc score
    # regardless of strategy/importance/budget/model — cache it.
    from kirby.hybrid import _evaluate_allocation
    alloc_cache = {}  # (fold_idx, evaluator_name) -> {rep_name: score}

    # Collect all unique evaluator names (expand 'match' for all models)
    all_eval_names = sorted(set(
        ev for model_name in models
        for ev in _evaluators_for_model(model_name, evaluator_names)
    ))

    print(f"\n[{time.strftime('%H:%M:%S')}] Pre-computing allocation scores "
          f"({n_folds} folds × {len(all_eval_names)} evaluators × {len(reps)} reps)...", flush=True)
    _alloc_t0 = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold_train_reps = {k: v[train_idx] for k, v in reps.items()}
        fold_val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train_fold = labels[train_idx]
        y_val_fold = labels[val_idx]

        for ev in all_eval_names:
            t0 = time.time()
            scores = {}
            for rep_name in fold_train_reps:
                X_tr = fold_train_reps[rep_name].copy()
                X_va = fold_val_reps[rep_name].copy()
                X_tr, X_va, _ = _impute_nan(X_tr, X_va)
                try:
                    s = _evaluate_allocation(
                        X_tr, X_va, y_train_fold, y_val_fold,
                        internal_evaluator=ev)
                    scores[rep_name] = max(float(s), 0.0)
                except Exception:
                    scores[rep_name] = 0.0
            alloc_cache[(fold_idx, ev)] = scores
            print(f"    fold {fold_idx} {ev}: {time.time()-t0:.1f}s", flush=True)

    print(f"  Allocation done in {time.time()-_alloc_t0:.0f}s, "
          f"{len(alloc_cache)} entries ({len(all_eval_names)} evaluators)", flush=True)

    # Count configs
    total = 0
    for model_name in models:
        model_budgets = _budgets_for_model(model_name, budgets)
        model_evals = _evaluators_for_model(model_name, evaluator_names)
        total += len(strategies) * len(importance_methods) * len(model_evals) * len(model_budgets)
    print(f"  {total} configs × {n_folds} folds = {total * n_folds} evaluations", flush=True)

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

    # Sequential evaluation with progress
    print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating {len(configs)} configs "
          f"(sequential, {n_folds} folds each)...", flush=True)
    t_start = time.time()
    all_results = []
    n_nan_configs = 0
    for i, cfg in enumerate(configs):
        strat, imp, model_name, ev, bgt = cfg
        model = deepcopy(models[model_name])
        fold_scores = []

        for fold_idx in range(n_folds):
            fold_importance = importance_cache.get((fold_idx, imp))
            fold_alloc = alloc_cache.get((fold_idx, ev))

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
                    precomputed_alloc_scores=fold_alloc,
                )
                fold_scores.append(score)
            except Exception as e:
                fold_scores.append(float('nan'))

        valid = [s for s in fold_scores if s == s]
        if len(valid) < n_folds:
            n_nan_configs += 1
        result = {
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
        all_results.append(result)

        # Progress every 100 configs
        if (i + 1) % 100 == 0 or (i + 1) == len(configs):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(configs) - i - 1) / rate if rate > 0 else 0
            print(f"    [{time.strftime('%H:%M:%S')}] {i+1}/{len(configs)} "
                  f"({(i+1)/len(configs)*100:.0f}%) {elapsed:.0f}s "
                  f"({rate:.1f}/s, ETA {eta:.0f}s) nan={n_nan_configs}",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\n[{time.strftime('%H:%M:%S')}] Done: {len(all_results)} configs "
          f"in {elapsed:.0f}s", flush=True)
    n_nan = sum(1 for r in all_results if r['n_valid_folds'] < n_folds)
    if n_nan:
        print(f"  WARNING: {n_nan} configs had NaN folds", flush=True)

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
    print(f"  Saved to {out_path}", flush=True)


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
