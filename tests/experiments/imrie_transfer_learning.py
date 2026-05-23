#!/usr/bin/env python3
"""
Imrie Transfer Learning Experiment (OPIG C4)

Tests whether pre-training DNNs on broad ChEMBL bioactivity data improves
molecular property prediction, and whether this changes the rep ranking.

Scientific questions:
  1. Does TL help? (per-rep: dnn_tl_full RMSE vs dnn_scratch RMSE)
  2. Does TL change rep ranking? (Spearman rank correlation under DNN vs TL-DNN)
  3. Does model improvement beat rep improvement?
     (TL-DNN on ECFP4 vs Ridge on mordred)

Conditions per (dataset, rep):
  - dnn_scratch:  DNNRegressor (no pre-training, control)
  - dnn_tl_full:  PretrainedDNNRegressor(freeze=0, lr=1e-4) — full fine-tuning
  - dnn_tl_frozen: PretrainedDNNRegressor(freeze=-1, lr=1e-3) — linear probe
  - ridge:        RidgeCV (simple model control)
  - gp:           GaussianProcessRegressor (current best for many reps)

Usage:
    # Smoke test (local, ~5 min)
    python tests/experiments/imrie_transfer_learning.py \
        --datasets logd --reps ecfp4 --n-folds 2

    # Full evaluation
    python tests/experiments/imrie_transfer_learning.py \
        --datasets logd,caco2,herg \
        --reps ecfp4,maccs,pdv,mol2vec,mordred,avalon \
        --n-folds 5

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    load_dataset,
    get_representations,
    assign_scaffold_groups,
)
from run_baseline_survey import _make_cv, _score, _maybe_scale
from experiments._shared import convert_numpy

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATASETS = ['logd', 'caco2', 'herg_ki']
DEFAULT_REPS = ['ecfp4', 'maccs', 'pdv', 'mol2vec', 'mordred', 'avalon']
DEFAULT_PRETRAIN_DIR = '/Volumes/seagate/kirby/chembl_pretrain'
HIDDEN_SIZES = (256, 128, 64)


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_models(rep_name, pretrain_dir, rep_dim=None):
    """Build the 5 conditions for a given rep.

    Returns dict of {condition_name: model_instance}.
    GP is skipped for high-dim reps (>500D) where O(n^3) kernel is impractical.
    """
    from kirby.models.dnn import DNNRegressor
    from kirby.models.dnn_pretrained import PretrainedDNNRegressor
    from sklearn.linear_model import RidgeCV

    # Check for pre-trained weights
    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    pt_path = os.path.join(pretrain_dir, f'{rep_name}_{hs_str}.pt')
    has_pretrained = os.path.exists(pt_path)

    models = {
        'dnn_scratch': DNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, lr=0.001,
            epochs=200, batch_size=256, patience=20, random_state=42),
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 50)),
    }

    # GP is O(n^3) with full kernel matrix — impractical for high-dim reps
    # (e.g. ECFP4=2048D, mordred=1613D with n>3000 takes hours per fold)
    if rep_dim is not None and rep_dim <= 500:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        models['gp'] = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),
            n_restarts_optimizer=3, random_state=42)
    elif rep_dim is not None:
        print(f"    (skipping GP: dim={rep_dim} > 500)")

    if has_pretrained:
        models['dnn_tl_full'] = PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2,
            pretrained_path=pt_path, finetune_lr=0.0001,
            freeze_layers=0, epochs=200, batch_size=256,
            patience=20, random_state=42)
        models['dnn_tl_frozen'] = PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2,
            pretrained_path=pt_path, finetune_lr=0.001,
            freeze_layers=-1, epochs=200, batch_size=256,
            patience=20, random_state=42)
    else:
        print(f"    WARNING: No pre-trained weights at {pt_path}")
        print(f"    Skipping TL conditions for {rep_name}")

    return models


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_condition(X, labels, groups, model, n_folds):
    """Evaluate one condition via scaffold GroupKFold.

    Returns dict of {metric: value} with mean and std.
    """
    from sklearn.base import clone

    cv, cv_kwargs = _make_cv(groups, n_folds, labels)
    splits = list(cv.split(X, labels, **cv_kwargs))

    fold_metrics = []
    for train_idx, val_idx in splits:
        X_train, X_val = _maybe_scale(X[train_idx], X[val_idx])
        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            preds = m.predict(X_val)
            if not np.all(np.isfinite(preds)):
                print(" [NaN]", end='')
                continue
            fold_metrics.append(_score(labels[val_idx], preds, 'regression'))
        except Exception as e:
            print(f" [err: {e}]", end='')
            continue

    if not fold_metrics:
        return {'mean': {}, 'std': {}}

    all_keys = fold_metrics[0].keys()
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics if k in f]))
              for k in all_keys}
    std_m = {k: float(np.std([f[k] for f in fold_metrics if k in f]))
             for k in all_keys}
    return {'mean': mean_m, 'std': std_m}


def evaluate_dataset(dataset_name, rep_names, n_folds, pretrain_dir):
    """Evaluate all reps x conditions for one dataset.

    Returns dict of results.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    smiles, labels, task_type = load_dataset(dataset_name)
    # Registry cross-check — catches silent drift between molecular
    # loaders and DATASET_REGISTRY.
    from kirby.registry import crosscheck_task_type
    crosscheck_task_type(dataset_name, task_type)
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {len(labels)} samples, {n_scaffolds} scaffolds, task={task_type}")

    # Generate reps (only the requested fast ones)
    rep_str = ','.join(rep_names)
    reps = get_representations(smiles, rep_str)

    # Build hybrid and add as another "rep" (evaluated with non-TL models only)
    try:
        from kirby.hybrid import apply_feature_selection
        from experiments._shared import get_best_search_for_downstream
        search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=None,
            speed='full',
            metric='rmse', task_type=task_type,
            dataset_name=dataset_name,
            return_fitted=True,
            adaptive_pruning=0.25,
        )
        fitted = search.fitted_results.get('rf') if search is not None else None
        if fitted is not None:
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            reps['kirby_hybrid'] = apply_feature_selection(reps, fi)
            rep_names = list(rep_names) + ['kirby_hybrid']
            print(f"  Hybrid built: {reps['kirby_hybrid'].shape[1]}D")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    ds_results = {
        'n_samples': len(labels),
        'n_scaffolds': n_scaffolds,
        'task_type': task_type,
        'reps': {},
    }

    for rep_name in rep_names:
        if rep_name not in reps:
            print(f"  {rep_name}: SKIPPED (not generated)")
            continue

        X = reps[rep_name]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != len(labels):
            print(f"  {rep_name}: SKIPPED (shape mismatch)")
            continue

        print(f"\n  --- {rep_name} (dim={X.shape[1]}) ---")
        models = build_models(rep_name, pretrain_dir, rep_dim=X.shape[1])

        rep_results = {'dim': X.shape[1], 'conditions': {}}
        for cond_name, model in models.items():
            t0 = time.time()
            print(f"    {cond_name}...", end='', flush=True)
            result = evaluate_condition(X, labels, groups, model, n_folds)
            elapsed = time.time() - t0

            rmse = result['mean'].get('rmse', float('nan'))
            r2 = result['mean'].get('r2', float('nan'))
            rho = result['mean'].get('spearman', float('nan'))
            print(f" RMSE={rmse:.4f}, R²={r2:.4f}, ρ={rho:.4f} ({elapsed:.1f}s)")

            result['time'] = elapsed
            rep_results['conditions'][cond_name] = result

        ds_results['reps'][rep_name] = rep_results

    return ds_results


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print key analyses across all datasets."""
    print(f"\n{'='*80}")
    print("TRANSFER LEARNING ANALYSIS")
    print(f"{'='*80}")

    for ds_name, ds_res in all_results.items():
        print(f"\n--- {ds_name} (n={ds_res['n_samples']}) ---")

        # Collect per-rep RMSE for each condition
        conditions = set()
        rep_data = {}
        for rep_name, rep_res in ds_res['reps'].items():
            rep_data[rep_name] = {}
            for cond_name, cond_res in rep_res['conditions'].items():
                conditions.add(cond_name)
                rep_data[rep_name][cond_name] = cond_res['mean'].get('rmse', float('nan'))

        conditions = sorted(conditions)

        # Table header
        header = f"  {'Rep':15s}"
        for c in conditions:
            header += f"  {c:>14s}"
        if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions:
            header += f"  {'delta_TL':>10s}"
        print(header)
        print(f"  {'-'*15}" + f"  {'-'*14}" * len(conditions) +
              (f"  {'-'*10}" if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions else ""))

        for rep_name in sorted(rep_data.keys()):
            line = f"  {rep_name:15s}"
            for c in conditions:
                val = rep_data[rep_name].get(c, float('nan'))
                line += f"  {val:>14.4f}" if not np.isnan(val) else f"  {'N/A':>14s}"

            # Delta: negative = TL helped (lower RMSE is better)
            if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions:
                scratch = rep_data[rep_name].get('dnn_scratch', float('nan'))
                tl = rep_data[rep_name].get('dnn_tl_full', float('nan'))
                if not np.isnan(scratch) and not np.isnan(tl):
                    delta = tl - scratch
                    line += f"  {delta:>+10.4f}"
                else:
                    line += f"  {'N/A':>10s}"
            print(line)

        # Rank correlation analysis
        if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions:
            scratch_vals = []
            tl_vals = []
            rep_names_ordered = sorted(rep_data.keys())
            for rn in rep_names_ordered:
                s = rep_data[rn].get('dnn_scratch', float('nan'))
                t = rep_data[rn].get('dnn_tl_full', float('nan'))
                if not np.isnan(s) and not np.isnan(t):
                    scratch_vals.append(s)
                    tl_vals.append(t)

            if len(scratch_vals) >= 3:
                rho_rank, p_rank = spearmanr(scratch_vals, tl_vals)
                print(f"\n  Rep ranking preservation (DNN vs TL-DNN):")
                print(f"    Spearman ρ = {rho_rank:.4f} (p={p_rank:.4f})")
                if abs(rho_rank) > 0.8:
                    print(f"    → Rankings PRESERVED: TL doesn't change which reps are best")
                elif abs(rho_rank) > 0.5:
                    print(f"    → Rankings PARTIALLY preserved")
                else:
                    print(f"    → Rankings CHANGED: TL reshuffles rep ordering")

        # Model vs rep trade-off
        if 'ridge' in conditions and 'dnn_tl_full' in conditions:
            print(f"\n  Model vs Rep trade-off:")
            for rep_name in sorted(rep_data.keys()):
                tl_rmse = rep_data[rep_name].get('dnn_tl_full', float('nan'))
                ridge_rmse = rep_data[rep_name].get('ridge', float('nan'))
                if not np.isnan(tl_rmse) and not np.isnan(ridge_rmse):
                    if tl_rmse < ridge_rmse:
                        print(f"    {rep_name}: TL-DNN beats Ridge "
                              f"({tl_rmse:.4f} vs {ridge_rmse:.4f})")
                    else:
                        print(f"    {rep_name}: Ridge beats TL-DNN "
                              f"({ridge_rmse:.4f} vs {tl_rmse:.4f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Imrie transfer learning experiment (OPIG C4). '
                    'Tests pre-trained DNN vs scratch DNN across molecular reps.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset names (default: logd,caco2,herg)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: ecfp4,maccs,pdv,mol2vec,mordred,avalon)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of scaffold GroupKFold folds (default: 5)')
    parser.add_argument(
        '--pretrain-dir', type=str, default=DEFAULT_PRETRAIN_DIR,
        help=f'Directory with pre-trained weights (default: {DEFAULT_PRETRAIN_DIR})')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path (default: tests/results/comparisons/imrie_transfer_learning.json)')

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    rep_names = (args.reps.split(',') if args.reps
                 else DEFAULT_REPS)

    print("=" * 70)
    print("IMRIE TRANSFER LEARNING EXPERIMENT (OPIG C4)")
    print(f"  Datasets: {datasets}")
    print(f"  Reps: {rep_names}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Pre-trained weights: {args.pretrain_dir}")
    print("=" * 70)

    # Check pre-trained weights exist
    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    missing = []
    for rn in rep_names:
        pt_path = os.path.join(args.pretrain_dir, f'{rn}_{hs_str}.pt')
        if not os.path.exists(pt_path):
            missing.append(rn)
    if missing:
        print(f"\nWARNING: Missing pre-trained weights for: {missing}")
        print(f"  Run pre-training first:")
        print(f"  python -m kirby.models.pretrain_chembl_dnn "
              f"--rep {','.join(missing)} --output-dir {args.pretrain_dir}")
        print(f"  Continuing with available weights + scratch baselines...\n")

    all_results = {}
    for ds_name in datasets:
        try:
            ds_results = evaluate_dataset(ds_name, rep_names, args.n_folds,
                                          args.pretrain_dir)
            all_results[ds_name] = ds_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print analysis
    if all_results:
        print_analysis(all_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/imrie_transfer_learning.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
