#!/usr/bin/env python3
"""
Covariate Shift Robustness Experiment (OPIG C2)

Tests whether hybrid reps maintain advantage under multiple distribution shifts,
following Q-SAVI evaluation protocol (Klarner et al., ICML 2023).

Split strategies:
  - scaffold: Murcko scaffold GroupKFold (standard)
  - mw: molecular weight percentile bins
  - spectral: Tanimoto-based spectral clustering

For each dataset x split strategy, evaluates all reps x models.
Scientific question: "Do hybrid reps maintain advantage under multiple
distribution shifts?"

Usage:
    python tests/experiments/covariate_shift.py --help
    python tests/experiments/covariate_shift.py --datasets logd --n-folds 2 --reps fast
    python tests/experiments/covariate_shift.py --datasets logd,caco2,herg
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    load_dataset,
    get_representations,
    assign_scaffold_groups,
    assign_mw_groups,
    assign_spectral_groups,
)
from run_baseline_survey import (
    evaluate_rep, get_default_models,
    _get_primary_metric, DATASET_REGISTRY,
)
from experiments._shared import convert_numpy

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATASETS = ['logd', 'caco2', 'herg_ki']
SPLIT_STRATEGIES = ['scaffold', 'mw', 'spectral']
SURVEY_MODELS = ['rf', 'ridge', 'xgboost']


# =============================================================================
# CORE EVALUATION
# =============================================================================

def evaluate_dataset_split(dataset_name, smiles, labels, reps, groups,
                           n_folds, task_type):
    """Evaluate all reps x models for one dataset + one split strategy.

    Returns dict of {rep_name: {model_name: {metric: value}}}.
    """
    default_models = get_default_models(task_type)
    # Use a subset of models for speed
    models = {k: v for k, v in default_models.items() if k in SURVEY_MODELS}
    metric_name, higher_is_better = _get_primary_metric(dataset_name, task_type)

    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            continue
        if X.ndim != 2 or X.shape[0] != len(labels):
            continue

        rep_results = {}
        for model_name, model in models.items():
            mean_m, std_m = evaluate_rep(
                X, labels, groups, model, n_folds, task_type)
            if mean_m:
                rep_results[model_name] = {
                    'mean': mean_m,
                    'std': std_m,
                }

        if rep_results:
            # Find best model for this rep
            best_model = None
            best_val = float('-inf') if higher_is_better else float('inf')
            for mn, mv in rep_results.items():
                val = mv['mean'].get(metric_name, float('nan'))
                if higher_is_better and val > best_val:
                    best_val = val
                    best_model = mn
                elif not higher_is_better and val < best_val:
                    best_val = val
                    best_model = mn

            results[rep_name] = {
                'models': rep_results,
                'best_model': best_model,
                'best_metric': best_val,
                'dim': X.shape[1],
            }

    return results


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(dataset_name, all_split_results, metric_name, higher_is_better):
    """Print comparison table across split strategies."""
    # Collect all reps that appear in at least one strategy
    all_reps = set()
    for split_res in all_split_results.values():
        all_reps.update(split_res.keys())
    all_reps = sorted(all_reps)

    print(f"\n{'=' * 90}")
    print(f"COVARIATE SHIFT COMPARISON: {dataset_name}  (metric: {metric_name})")
    print(f"{'=' * 90}")

    header = f"  {'Rep':30s}"
    for strategy in SPLIT_STRATEGIES:
        header += f"  {strategy:>14s}"
    header += f"  {'delta':>10s}"
    print(header)
    print(f"  {'-' * 30}  {'-' * 14}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    rows = []
    for rep_name in all_reps:
        vals = {}
        for strategy in SPLIT_STRATEGIES:
            res = all_split_results.get(strategy, {}).get(rep_name, {})
            vals[strategy] = res.get('best_metric', float('nan'))

        # Delta = worst - best performance across strategies
        valid = [v for v in vals.values() if not np.isnan(v)]
        if valid:
            delta = max(valid) - min(valid)
            rows.append((rep_name, vals, delta))
        else:
            rows.append((rep_name, vals, float('nan')))

    # Sort by scaffold performance
    rows.sort(key=lambda x: x[1].get('scaffold', float('inf') if not higher_is_better else float('-inf')),
              reverse=higher_is_better)

    for rep_name, vals, delta in rows:
        line = f"  {rep_name:30s}"
        for strategy in SPLIT_STRATEGIES:
            v = vals.get(strategy, float('nan'))
            line += f"  {v:>14.4f}" if not np.isnan(v) else f"  {'N/A':>14s}"
        line += f"  {delta:>10.4f}" if not np.isnan(delta) else f"  {'N/A':>10s}"
        print(line)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Covariate shift robustness experiment (Q-SAVI protocol). '
                    'Tests reps under scaffold, MW, and spectral splitting.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated molecular dataset names (default: logd,caco2,herg)')
    parser.add_argument(
        '--reps', type=str, default='all',
        help='Rep preset (fast, fast+hf, all)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path (default: tests/results/comparisons/covariate_shift.json)')

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)

    # Validate: only molecular datasets
    for ds in datasets:
        if ds not in DATASET_REGISTRY:
            print(f"ERROR: Unknown dataset '{ds}'")
            return 1
        domain, _ = DATASET_REGISTRY[ds]
        if domain != 'molecular':
            print(f"ERROR: {ds} is {domain}, not molecular. "
                  f"Covariate shift experiment is molecular-only.")
            return 1

    print("=" * 70)
    print("COVARIATE SHIFT ROBUSTNESS EXPERIMENT")
    print(f"  Datasets: {datasets}")
    print(f"  Reps: {args.reps}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Splits: {SPLIT_STRATEGIES}")
    print(f"  Models: {SURVEY_MODELS}")
    print("=" * 70)

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        try:
            smiles, labels, task_type = load_dataset(ds_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        reps = get_representations(smiles, args.reps)
        metric_name, higher_is_better = _get_primary_metric(ds_name, task_type)

        # Build hybrid from the viable reps and add as another "rep"
        try:
            from kirby.find_best_hybrid import find_best_hybrid
            from kirby.hybrid import apply_feature_selection
            from experiments._shared import FULL_BUDGETS
            scaffold_groups, _ = assign_scaffold_groups(smiles)
            search = find_best_hybrid(
                base_reps=reps, labels=labels, groups=scaffold_groups,
                budgets=FULL_BUDGETS,
                speed='full', return_fitted=True, verbose=False, cache=True,
                dataset_name=ds_name, tier='general',
                metric=metric_name, task_type=task_type,
                augmentation_strategy='none',
                adaptive_pruning=0.25)
            fitted = search.fitted_results.get('rf')
            if fitted is not None:
                fi = {r: {'selected_indices': idx}
                      for r, idx in fitted.selected_indices.items()}
                reps['kirby_hybrid'] = apply_feature_selection(reps, fi)
                print(f"  Hybrid built: {reps['kirby_hybrid'].shape[1]}D")
        except Exception as e:
            print(f"  Hybrid build failed: {e}")

        print(f"  {len(labels)} samples, {len(reps)} reps, task={task_type}")

        ds_results = {}
        for strategy in SPLIT_STRATEGIES:
            print(f"\n  --- Split: {strategy} ---")
            t0 = time.time()

            if strategy == 'scaffold':
                groups, n_groups = assign_scaffold_groups(smiles)
            elif strategy == 'mw':
                groups, n_groups = assign_mw_groups(smiles, n_groups=args.n_folds)
            elif strategy == 'spectral':
                groups, n_groups = assign_spectral_groups(smiles, n_groups=args.n_folds)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            print(f"  {n_groups} groups")

            split_res = evaluate_dataset_split(
                ds_name, smiles, labels, reps, groups,
                args.n_folds, task_type)

            elapsed = time.time() - t0
            print(f"  {len(split_res)} reps evaluated in {elapsed:.1f}s")

            ds_results[strategy] = split_res

        # Print comparison
        print_comparison(ds_name, ds_results, metric_name, higher_is_better)
        all_results[ds_name] = {
            'n_samples': len(labels),
            'task_type': task_type,
            'metric': metric_name,
            'higher_is_better': higher_is_better,
            'splits': ds_results,
        }

    # Save results
    output_path = args.output or 'tests/results/comparisons/covariate_shift.json'
    p = Path(output_path)
    if p.is_dir() or not p.suffix:
        p = p / 'covariate_shift.json'
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
