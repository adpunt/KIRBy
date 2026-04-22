"""Reaction Representation Comparison: Which rep works best for yield prediction?

Evaluates 9 reaction representations across 3 HTE yield datasets using
GroupKFold CV (groups provided by the dataset loader). All reps are static
(no fine-tuning) — this is a frozen-rep comparison.

Representations (9):
  drfp_2048         — Differential Reaction FP (Probst et al. 2022)
  product_ecfp4     — Product-only Morgan FP
  reactant_ecfp4    — Reactant-only Morgan FP (OR-combined)
  product_maccs     — Product-only MACCS keys (167-bit)
  reactant_maccs    — Reactant-only MACCS keys (OR-combined)
  product_pdv       — Product-only physicochemical descriptors
  reactant_pdv      — Reactant-only descriptors (mean-pooled)
  rxnfp             — RXNBERT learned FP (256D, Schwaller et al. 2021)
  onehot_components — One-hot categorical components (factorial baseline)

Models (3):
  Ridge, RandomForest, XGBoost

Datasets (3):
  buchwald_hartwig   — ~3955 C-N coupling reactions (additive groups)
  suzuki             — ~5760 Suzuki-Miyaura coupling (ligand groups)
  bh_doyle_plates    — ~4599 C-N coupling from Doyle lab (additive groups)

Usage:
    python tests/experiments/reaction_rep_comparison.py
    python tests/experiments/reaction_rep_comparison.py --datasets buchwald_hartwig,suzuki
    python tests/experiments/reaction_rep_comparison.py --n-folds 2
    python tests/experiments/reaction_rep_comparison.py --reps drfp_2048,rxnfp,concat_ecfp4
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kirby.datasets.reaction_yields import (
    load_buchwald_hartwig, load_suzuki, load_bh_doyle_plates,
)
from kirby.representations.reaction import (
    create_drfp,
    create_product_ecfp4,
    create_reactant_ecfp4,
    create_product_maccs,
    create_reactant_maccs,
    create_product_pdv,
    create_reactant_pdv,
    create_rxnfp,
    create_onehot_components,
)
from experiments._shared import (
    get_model_factories as _get_model_factories,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
)

warnings.filterwarnings('ignore')


# =============================================================================
# Constants
# =============================================================================

ALL_REPS = [
    'drfp_2048',
    'product_ecfp4', 'reactant_ecfp4',
    'product_maccs', 'reactant_maccs',
    'product_pdv', 'reactant_pdv',
    'rxnfp', 'onehot_components',
]

ALL_DATASETS = ['buchwald_hartwig', 'suzuki', 'bh_doyle_plates']

DATASET_LOADERS = {
    'buchwald_hartwig': load_buchwald_hartwig,
    'suzuki': load_suzuki,
    'bh_doyle_plates': load_bh_doyle_plates,
}


# =============================================================================
# Representation generation
# =============================================================================

def generate_rep(rep_name: str, rxn_smiles: List[str],
                 components: List[Dict]) -> np.ndarray:
    """Generate a single representation matrix.

    Some reps take rxn_smiles, others take components.

    Returns:
        np.ndarray of shape (n_reactions, D).
    """
    if rep_name == 'drfp_2048':
        return create_drfp(rxn_smiles, n_bits=2048)
    elif rep_name == 'product_ecfp4':
        return create_product_ecfp4(rxn_smiles)
    elif rep_name == 'reactant_ecfp4':
        return create_reactant_ecfp4(rxn_smiles)
    elif rep_name == 'product_maccs':
        return create_product_maccs(rxn_smiles)
    elif rep_name == 'reactant_maccs':
        return create_reactant_maccs(rxn_smiles)
    elif rep_name == 'product_pdv':
        return create_product_pdv(rxn_smiles)
    elif rep_name == 'reactant_pdv':
        return create_reactant_pdv(rxn_smiles)
    elif rep_name == 'rxnfp':
        return create_rxnfp(rxn_smiles)
    elif rep_name == 'onehot_components':
        return create_onehot_components(components)
    else:
        raise ValueError(f"Unknown representation: {rep_name}")


# =============================================================================
# Model factories
# =============================================================================

MODELS_LIST = DOWNSTREAM_MODELS


def get_model_factories(n_samples: int = 1000) -> Dict[str, callable]:
    """Return {name: factory_fn} using shared DOWNSTREAM_MODELS."""
    return _get_model_factories(MODELS_LIST, 'regression', n_samples=n_samples)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_rep_model(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory: callable,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Run GroupKFold CV for one (rep, model) pair.

    Returns dict with mean/std for R2, RMSE, Spearman.
    """
    n_groups = len(np.unique(groups))
    effective_folds = min(n_folds, n_groups)
    if effective_folds < 2:
        return {
            'r2_mean': float('nan'), 'r2_std': float('nan'),
            'rmse_mean': float('nan'), 'rmse_std': float('nan'),
            'spearman_mean': float('nan'), 'spearman_std': float('nan'),
            'n_folds': 0,
        }

    gkf = GroupKFold(n_splits=effective_folds)
    r2_scores = []
    rmse_scores = []
    spearman_scores = []

    for train_idx, test_idx in gkf.split(X, labels, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Handle NaN/Inf from scaling (can happen with constant columns)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        model = model_factory()
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        except Exception as e:
            print(f"    WARNING: model fit/predict failed: {e}")
            continue

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
        rho, _ = spearmanr(y_test, y_pred)
        spearman_scores.append(float(rho) if not np.isnan(rho) else 0.0)

    if not r2_scores:
        return {
            'r2_mean': float('nan'), 'r2_std': float('nan'),
            'rmse_mean': float('nan'), 'rmse_std': float('nan'),
            'spearman_mean': float('nan'), 'spearman_std': float('nan'),
            'n_folds': 0,
        }

    return {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'rmse_mean': float(np.mean(rmse_scores)),
        'rmse_std': float(np.std(rmse_scores)),
        'spearman_mean': float(np.mean(spearman_scores)),
        'spearman_std': float(np.std(spearman_scores)),
        'n_folds': effective_folds,
    }


# =============================================================================
# Summary printing
# =============================================================================

def print_dataset_summary(dataset_name: str, results: Dict,
                          model_names: List[str]) -> None:
    """Print a formatted table for one dataset."""
    print(f"\n{'=' * 90}")
    print(f"  {dataset_name.upper()}")
    print(f"{'=' * 90}")

    # Header
    header = f"{'Representation':<22}"
    for model in model_names:
        header += f" | {model + ' R2':>14} {model + ' RMSE':>14} {model + ' rho':>12}"
    print(header)
    print("-" * len(header))

    for rep_name in ALL_REPS:
        if rep_name not in results:
            continue
        row = f"{rep_name:<22}"
        for model in model_names:
            key = f"{rep_name}__{model}"
            if key in results:
                r = results[key]
                r2_str = f"{r['r2_mean']:>6.3f}+-{r['r2_std']:.3f}"
                rmse_str = f"{r['rmse_mean']:>6.2f}+-{r['rmse_std']:.2f}"
                rho_str = f"{r['spearman_mean']:>5.3f}+-{r['spearman_std']:.3f}"
                row += f" | {r2_str:>14} {rmse_str:>14} {rho_str:>12}"
            else:
                row += f" | {'FAILED':>14} {'':>14} {'':>12}"
        print(row)


def print_best_per_metric(all_results: Dict) -> None:
    """Print best rep per dataset per metric across all models."""
    print(f"\n{'=' * 90}")
    print("  BEST REPRESENTATION PER DATASET (across all models)")
    print(f"{'=' * 90}")

    for dataset_name, ds_results in all_results.items():
        print(f"\n  {dataset_name}:")
        # Find best R2
        best_r2_key = None
        best_r2_val = -float('inf')
        best_rmse_key = None
        best_rmse_val = float('inf')
        best_rho_key = None
        best_rho_val = -float('inf')

        for key, metrics in ds_results.items():
            if key.startswith('_'):
                continue
            if not isinstance(metrics, dict) or 'r2_mean' not in metrics:
                continue
            if np.isnan(metrics['r2_mean']):
                continue
            if metrics['r2_mean'] > best_r2_val:
                best_r2_val = metrics['r2_mean']
                best_r2_key = key
            if metrics['rmse_mean'] < best_rmse_val:
                best_rmse_val = metrics['rmse_mean']
                best_rmse_key = key
            if metrics['spearman_mean'] > best_rho_val:
                best_rho_val = metrics['spearman_mean']
                best_rho_key = key

        if best_r2_key:
            print(f"    Best R2:       {best_r2_key:<40} R2={best_r2_val:.4f}")
        if best_rmse_key:
            print(f"    Best RMSE:     {best_rmse_key:<40} RMSE={best_rmse_val:.2f}")
        if best_rho_key:
            print(f"    Best Spearman: {best_rho_key:<40} rho={best_rho_val:.4f}")


# =============================================================================
# JSON serialisation
# =============================================================================

def convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--datasets', type=str, default=','.join(ALL_DATASETS),
        help=f"Comma-separated dataset names (default: {','.join(ALL_DATASETS)})",
    )
    parser.add_argument(
        '--reps', type=str, default=','.join(ALL_REPS),
        help=f"Comma-separated rep names (default: all 8)",
    )
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help="Number of GroupKFold folds (default: 5)",
    )
    parser.add_argument(
        '--output', type=str, default='tests/results/comparisons',
        help="Output directory for results JSON",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    rep_names = [r.strip() for r in args.reps.split(',')]
    n_folds = args.n_folds

    # Validate inputs
    for d in datasets:
        if d not in DATASET_LOADERS:
            print(f"ERROR: unknown dataset '{d}'. Available: {list(DATASET_LOADERS.keys())}")
            sys.exit(1)
    for r in rep_names:
        if r not in ALL_REPS:
            print(f"ERROR: unknown rep '{r}'. Available: {ALL_REPS}")
            sys.exit(1)

    print(f"Reaction Representation Comparison")
    print(f"  Datasets:  {datasets}")
    print(f"  Reps:      {rep_names}")
    print(f"  Models:    {MODELS_LIST}")
    print(f"  Folds:     {n_folds}")
    print()

    all_results = {}
    total_t0 = time.time()

    for dataset_name in datasets:
        print(f"\n{'#' * 70}")
        print(f"  Loading {dataset_name}...")
        print(f"{'#' * 70}")

        data = DATASET_LOADERS[dataset_name]()
        rxn_smiles = data['rxn_smiles']
        labels = data['labels']
        groups = data['groups']
        components = data['components']
        n_reactions = len(labels)
        n_groups = len(np.unique(groups))

        model_factories = get_model_factories(n_samples=n_reactions)
        model_names = list(model_factories.keys())

        print(f"  {n_reactions} reactions, {n_groups} groups, "
              f"yield range [{labels.min():.1f}, {labels.max():.1f}]")

        ds_results = {
            '_dataset': dataset_name,
            '_n_reactions': n_reactions,
            '_n_groups': n_groups,
            '_n_folds': n_folds,
            '_yield_range': [float(labels.min()), float(labels.max())],
        }

        # Generate all representations up front
        rep_matrices = {}
        for rep_name in rep_names:
            t0 = time.time()
            try:
                X = generate_rep(rep_name, rxn_smiles, components)
                rep_matrices[rep_name] = X
                elapsed = time.time() - t0
                print(f"  {rep_name:<22} -> {X.shape[1]:>6}D  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  {rep_name:<22} -> FAILED: {e}")

        # Build hybrid from all successfully generated reps
        if len(rep_matrices) >= 2:
            try:
                from kirby.find_best_hybrid import find_best_hybrid
                from kirby.hybrid import apply_feature_selection
                search = find_best_hybrid(
                    base_reps=rep_matrices, labels=labels, groups=groups,
                    budgets=FULL_BUDGETS,
                    internal_evaluators=INTERNAL_EVALUATORS,
                    speed='full', return_fitted=True, verbose=False, cache=True,
                    dataset_name=dataset_name, tier='general',
                    metric='rmse', task_type='regression',
                    augmentation_strategy='none',
                    adaptive_pruning=0.25)
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
                fitted = search.fitted_results.get('rf')
                if fitted is not None:
                    fi = {r: {'selected_indices': idx}
                          for r, idx in fitted.selected_indices.items()}
                    rep_matrices['kirby_hybrid'] = apply_feature_selection(rep_matrices, fi)
                    if 'kirby_hybrid' not in rep_names:
                        rep_names = list(rep_names) + ['kirby_hybrid']
                    print(f"  {'kirby_hybrid':<22} -> {rep_matrices['kirby_hybrid'].shape[1]:>6}D")
            except Exception as e:
                print(f"  kirby_hybrid           -> FAILED: {e}")

        # Evaluate all (rep, model) combinations
        for rep_name in rep_names:
            if rep_name not in rep_matrices:
                ds_results[rep_name] = 'generation_failed'
                continue

            X = rep_matrices[rep_name]
            ds_results[rep_name] = {'dim': X.shape[1]}

            for model_name in model_names:
                key = f"{rep_name}__{model_name}"
                t0 = time.time()
                try:
                    metrics = evaluate_rep_model(
                        X, labels, groups, model_factories[model_name],
                        n_folds=n_folds,
                    )
                    elapsed = time.time() - t0
                    ds_results[key] = metrics
                    print(f"    {rep_name:<22} + {model_name:<8} -> "
                          f"R2={metrics['r2_mean']:.3f}+-{metrics['r2_std']:.3f}  "
                          f"RMSE={metrics['rmse_mean']:.2f}  "
                          f"rho={metrics['spearman_mean']:.3f}  "
                          f"({elapsed:.1f}s)")
                except Exception as e:
                    print(f"    {rep_name:<22} + {model_name:<8} -> FAILED: {e}")
                    ds_results[key] = {'error': str(e)}

        # Print per-dataset summary table
        print_dataset_summary(dataset_name, ds_results, model_names)
        all_results[dataset_name] = ds_results

    # Print cross-dataset best-rep summary
    print_best_per_metric(all_results)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Save JSON
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'reaction_rep_comparison.json'
    with open(out_file, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"[saved {out_file}]")


if __name__ == '__main__':
    main()
