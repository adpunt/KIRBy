#!/usr/bin/env python3
"""
Molecular Baselines — fill the BOM table.

For each (dataset, rep) pair, loads from cache, runs 5-fold scaffold-split
GroupKFold with 5 models, prints R² scores. No hybrid search, no grid —
just individual rep baselines.

Usage:
    python scripts/run_molecular_baselines.py                # all datasets
    python scripts/run_molecular_baselines.py --dataset logd  # single dataset
    python scripts/run_molecular_baselines.py --dataset logd --reps ecfp4,maccs
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.modules.setdefault('bottleneck', None)
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

from find_best_hybrid_molecular import (
    load_dataset, assign_scaffold_groups, _rep_cache_path, REP_PRESETS
)

ALL_DATASETS = ['logd', 'caco2', 'herg', 'qm9',
                'adme_hlm', 'adme_rlm', 'adme_mdr1', 'adme_solubility']
ALL_REPS = REP_PRESETS['all'] + ['chemformer']


def evaluate_rep(X, y, groups, n_folds=5, seed=42):
    """Evaluate a single rep with multiple models. Returns {model: r2}."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    models = _get_models(seed)
    gkf = GroupKFold(n_splits=n_folds)

    results = {}
    for name, make_model in models.items():
        fold_scores = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = make_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_scores.append(r2_score(y_test, y_pred))

        results[name] = np.mean(fold_scores)

    return results


def _get_models(seed=42):
    """Return dict of {name: factory_fn} for baseline models."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    models = {
        'rf': lambda: RandomForestRegressor(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed
        ),
        'ridge': lambda: Ridge(alpha=1.0),
        'knn': lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }

    try:
        from xgboost import XGBRegressor
        models['xgb'] = lambda: XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=seed, verbosity=0
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        models['lgbm'] = lambda: LGBMRegressor(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            n_jobs=-1, random_state=seed, verbose=-1
        )
    except ImportError:
        pass

    return models


def main():
    parser = argparse.ArgumentParser(description="Run molecular baselines for BOM")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset (logd/caco2/herg/qm9) or all")
    parser.add_argument("--reps", type=str, default=None,
                        help="Comma-separated rep names (default: all cached)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to this file")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    rep_names = [r.strip() for r in args.reps.split(',')] if args.reps else ALL_REPS

    all_results = {}

    print("=" * 80)
    print("MOLECULAR BASELINES (individual rep evaluation)")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Reps:     {len(rep_names)}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    sys.stdout.flush()

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  DATASET: {dataset}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Load dataset
        smiles, labels, task_type = load_dataset(dataset)
        groups, n_groups = assign_scaffold_groups(smiles)
        print(f"  {len(smiles)} molecules, {n_groups} scaffolds\n")
        sys.stdout.flush()

        dataset_results = {}

        for rep_name in rep_names:
            cache_file = _rep_cache_path(smiles, rep_name)
            if not cache_file.exists():
                print(f"  {rep_name:20s}  NOT CACHED — skipping")
                sys.stdout.flush()
                continue

            X = np.load(cache_file)
            if X.shape[0] != len(smiles):
                print(f"  {rep_name:20s}  STALE CACHE ({X.shape[0]} != {len(smiles)}) — skipping")
                sys.stdout.flush()
                continue

            t0 = time.time()
            try:
                scores = evaluate_rep(X, labels, groups)
                elapsed = time.time() - t0

                best_model = max(scores, key=scores.get)
                best_r2 = scores[best_model]

                score_str = ', '.join(f'{m}={s:.3f}' for m, s in sorted(scores.items()))
                print(f"  {rep_name:20s}  {X.shape[1]:5d}D  best={best_model}={best_r2:.3f}  ({score_str})  [{elapsed:.0f}s]")

                dataset_results[rep_name] = {
                    'dim': X.shape[1],
                    'scores': scores,
                    'best_model': best_model,
                    'best_r2': best_r2,
                }
            except Exception as e:
                print(f"  {rep_name:20s}  ERROR: {e}")

            sys.stdout.flush()

        all_results[dataset] = dataset_results

    # Print BOM table
    print("\n\n" + "=" * 80)
    print("BOM TABLE (copy-paste ready)")
    print("=" * 80)

    # Header
    ds_list = list(all_results.keys())
    header = f"| {'Rep':20s} | {'Dim':>5s} |"
    for ds in ds_list:
        header += f" {ds + ' R²':>18s} |"
    print(header)
    print("|" + "-" * 22 + "|" + "-" * 7 + "|" + ("|".join(["-" * 20] * len(ds_list))) + "|")

    for rep_name in rep_names:
        has_any = any(rep_name in all_results.get(ds, {}) for ds in ds_list)
        if not has_any:
            continue

        dim = "?"
        row = f"| `{rep_name}`"
        row = row.ljust(23) + "| "

        for ds in ds_list:
            info = all_results.get(ds, {}).get(rep_name)
            if info:
                dim = str(info['dim'])
                row_ds = f"{info['best_model']}={info['best_r2']:.3f}"
            else:
                row_ds = "—"
            row += f"{row_ds:>18s} | "

        print(f"{row.rstrip()}")

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
