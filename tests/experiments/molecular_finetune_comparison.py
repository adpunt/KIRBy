#!/usr/bin/env python3
"""
Molecular Fine-Tuning Comparison Experiment

Does fine-tuning molecular language models beat frozen extraction on QSAR datasets?

Compares frozen vs fine-tuned embeddings for LMs, GNNs, and GROVER:

  LM methods (per model: ChemBERTa, MolFormer, SELFormer, GP-MoLFormer, ChemBERT):
    1. Frozen + Ridge (baseline)
    2. LoRA + Ridge (molecular_lora.py factory)
    3. Freeze+Head + Ridge (molecular_freeze.py factory)
    4. Contrastive LoRA + Ridge (molecular_contrastive.py factory)

  GNN methods:
    5. Frozen MolCLR + Ridge
    6. Fine-tuned MolCLR + Ridge (molecular_gnn.py factory)
    7. Frozen GraphMVP + Ridge
    8. Fine-tuned GraphMVP + Ridge (molecular_gnn.py factory)

  GROVER:
    9. Frozen GROVER + Ridge
   10. Fine-tuned GROVER + Ridge (molecular_grover.py factory)

All evaluated with scaffold GroupKFold (no random splits). Primary metric: RMSE.
Secondary: R2, Spearman rho. Per-fold fine-tuning factories prevent leakage.

Usage:
    python tests/experiments/molecular_finetune_comparison.py --help
    python tests/experiments/molecular_finetune_comparison.py --datasets logd,caco2
    python tests/experiments/molecular_finetune_comparison.py --n-folds 2  # smoke test
    python tests/experiments/molecular_finetune_comparison.py --methods frozen,lora  # subset
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATASETS = ['logd', 'caco2', 'herg']

# LM models that support all three fine-tuning strategies (lora, freeze, contrastive)
LM_MODELS = ['chemberta', 'molformer', 'selformer', 'gp_molformer', 'chembert']

# GNN models (frozen vs fine-tuned)
GNN_MODELS = ['molclr', 'graphmvp']

# All valid method names for --methods filter
ALL_METHODS = ['frozen', 'lora', 'freeze_head', 'contrastive', 'gnn_ft', 'grover_ft']


# =============================================================================
# SCAFFOLD SPLITTING
# =============================================================================

def _scaffold_groups(smiles_list: List[str]) -> np.ndarray:
    """Assign Murcko scaffold group IDs for GroupKFold."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffolds = []
    for smi in smiles_list:
        try:
            scaf = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:
            scaf = smi
        scaffolds.append(scaf)

    unique = sorted(set(scaffolds))
    scaffold_to_idx = {s: i for i, s in enumerate(unique)}
    return np.array([scaffold_to_idx[s] for s in scaffolds])


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate_frozen(X: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray, n_folds: int = 5) -> List[dict]:
    """Evaluate a frozen rep with RidgeCV. Returns per-fold metric dicts."""
    n_unique = len(np.unique(groups))
    if n_unique < 2:
        print("  WARNING: <2 scaffold groups, skipping")
        return []

    effective_folds = min(n_folds, n_unique)
    cv = GroupKFold(n_splits=effective_folds)
    scores = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, labels, groups=groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        model = RidgeCV(alphas=np.logspace(-6, 6, 13))
        model.fit(X_tr, labels[train_idx])
        preds = model.predict(X_va)

        r2 = float(r2_score(labels[val_idx], preds))
        rmse = float(np.sqrt(mean_squared_error(labels[val_idx], preds)))
        rho = float(spearmanr(labels[val_idx], preds).correlation)
        scores.append({'fold': fold_i, 'r2': r2, 'rmse': rmse, 'spearman': rho})

    return scores


def evaluate_factory(factory: Callable, labels: np.ndarray,
                     groups: np.ndarray, n_folds: int = 5) -> List[dict]:
    """Evaluate a fine-tuning factory with RidgeCV. Returns per-fold metric dicts."""
    n_unique = len(np.unique(groups))
    if n_unique < 2:
        print("  WARNING: <2 scaffold groups, skipping")
        return []

    effective_folds = min(n_folds, n_unique)
    cv = GroupKFold(n_splits=effective_folds)
    dummy = np.zeros((len(labels), 1))
    scores = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(dummy, labels, groups=groups)):
        try:
            X_tr, X_va = factory(train_idx, val_idx, labels[train_idx])
        except Exception as e:
            print(f" [fold {fold_i} FT error: {e}]", end='')
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        model = RidgeCV(alphas=np.logspace(-6, 6, 13))
        model.fit(X_tr, labels[train_idx])
        preds = model.predict(X_va)

        r2 = float(r2_score(labels[val_idx], preds))
        rmse = float(np.sqrt(mean_squared_error(labels[val_idx], preds)))
        rho = float(spearmanr(labels[val_idx], preds).correlation)
        scores.append({'fold': fold_i, 'r2': r2, 'rmse': rmse, 'spearman': rho})

    return scores


def _summarize(scores: List[dict]) -> dict:
    """Aggregate per-fold scores to mean +/- std for each metric."""
    if not scores:
        return {
            'rmse_mean': float('nan'), 'rmse_std': float('nan'),
            'r2_mean': float('nan'), 'r2_std': float('nan'),
            'spearman_mean': float('nan'), 'spearman_std': float('nan'),
            'n_folds': 0,
        }
    rmse_vals = [s['rmse'] for s in scores]
    r2_vals = [s['r2'] for s in scores]
    rho_vals = [s['spearman'] for s in scores]
    return {
        'rmse_mean': round(float(np.mean(rmse_vals)), 4),
        'rmse_std': round(float(np.std(rmse_vals)), 4),
        'r2_mean': round(float(np.mean(r2_vals)), 4),
        'r2_std': round(float(np.std(r2_vals)), 4),
        'spearman_mean': round(float(np.mean(rho_vals)), 4),
        'spearman_std': round(float(np.std(rho_vals)), 4),
        'n_folds': len(scores),
    }


def _print_scores(method_name: str, summary: dict):
    """Print one-line summary for a method."""
    print(f"    {method_name:30s}  "
          f"RMSE={summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f}  "
          f"R2={summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}  "
          f"rho={summary['spearman_mean']:.4f} +/- {summary['spearman_std']:.4f}")


# =============================================================================
# FROZEN BASELINES
# =============================================================================

def compute_frozen_reps(smiles_list: List[str]) -> Dict[str, np.ndarray]:
    """Compute frozen embeddings for all LM + GNN + GROVER models.

    Returns {model_name: embeddings_array}.
    """
    from kirby.representations.molecular import (
        create_chemberta, create_molformer, create_selformer,
        create_gp_molformer, create_chembert,
        create_molclr, create_graphmvp, create_grover,
    )

    creators = {
        'chemberta': create_chemberta,
        'molformer': create_molformer,
        'selformer': create_selformer,
        'gp_molformer': create_gp_molformer,
        'chembert': create_chembert,
        'molclr': create_molclr,
        'graphmvp': create_graphmvp,
        'grover': create_grover,
    }

    frozen = {}
    for name, fn in creators.items():
        try:
            print(f"    Computing frozen {name}...", end='', flush=True)
            X = fn(smiles_list)
            frozen[name] = X
            print(f" ({X.shape[1]}D)")
        except Exception as e:
            print(f" FAILED: {e}")

    return frozen


# =============================================================================
# FINE-TUNING FACTORY CREATION
# =============================================================================

def _make_lm_factories(
    smiles_list: List[str],
    dataset_name: str,
    methods: List[str],
    epochs: int,
    cache: bool,
) -> Dict[str, Callable]:
    """Create LM fine-tuning factories for requested methods.

    Returns {method_display_name: factory}.
    """
    factories = {}

    for model_name in LM_MODELS:
        # LoRA
        if 'lora' in methods:
            try:
                from kirby.finetuning.molecular_lora import make_molecular_lora_factory
                key = f"{model_name}_lora"
                factories[key] = make_molecular_lora_factory(
                    smiles_list, dataset_name,
                    base_model=model_name, epochs=epochs, cache=cache,
                )
            except Exception as e:
                print(f"    {model_name}_lora: FAILED to create factory: {e}")

        # Freeze+Head
        if 'freeze_head' in methods:
            try:
                from kirby.finetuning.molecular_freeze import make_molecular_freeze_factory
                key = f"{model_name}_freeze_head"
                factories[key] = make_molecular_freeze_factory(
                    smiles_list, dataset_name,
                    base_model=model_name, epochs=epochs, cache=cache,
                )
            except Exception as e:
                print(f"    {model_name}_freeze_head: FAILED to create factory: {e}")

        # Contrastive LoRA
        if 'contrastive' in methods:
            try:
                from kirby.finetuning.molecular_contrastive import make_molecular_contrastive_factory
                key = f"{model_name}_contrastive"
                factories[key] = make_molecular_contrastive_factory(
                    smiles_list, dataset_name,
                    base_model=model_name, epochs=epochs, cache=cache,
                )
            except Exception as e:
                print(f"    {model_name}_contrastive: FAILED to create factory: {e}")

    return factories


def _make_gnn_factories(
    smiles_list: List[str],
    dataset_name: str,
    epochs: int,
    cache: bool,
) -> Dict[str, Callable]:
    """Create GNN fine-tuning factories (MolCLR, GraphMVP).

    Returns {method_display_name: factory}.
    """
    factories = {}

    for model_name in GNN_MODELS:
        try:
            from kirby.finetuning.molecular_gnn import make_molecular_gnn_factory
            key = f"{model_name}_ft"
            factories[key] = make_molecular_gnn_factory(
                smiles_list, dataset_name,
                base_model=model_name, epochs=epochs, cache=cache,
            )
        except Exception as e:
            print(f"    {model_name}_ft: FAILED to create factory: {e}")

    return factories


def _make_grover_factory(
    smiles_list: List[str],
    dataset_name: str,
    epochs: int,
    cache: bool,
) -> Dict[str, Callable]:
    """Create GROVER fine-tuning factory.

    Returns {method_display_name: factory}.
    """
    factories = {}
    try:
        from kirby.finetuning.molecular_grover import make_molecular_grover_factory
        factories['grover_ft'] = make_molecular_grover_factory(
            smiles_list, dataset_name, epochs=epochs, cache=cache,
        )
    except Exception as e:
        print(f"    grover_ft: FAILED to create factory: {e}")

    return factories


# =============================================================================
# PER-DATASET RUNNER
# =============================================================================

def run_dataset(
    dataset_name: str,
    smiles_list: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
    methods: List[str],
    n_folds: int,
    epochs: int,
    cache: bool,
) -> dict:
    """Run all requested methods on a single dataset.

    Returns a results dict with per-method summaries and per-fold scores.
    """
    n = len(labels)
    n_scaffolds = len(np.unique(groups))
    ds_result = {
        'n_samples': n,
        'n_scaffolds': n_scaffolds,
        'methods': {},
    }

    # ----- Frozen baselines -----
    frozen_reps = {}
    if 'frozen' in methods:
        print("\n  --- Frozen baselines ---")
        frozen_reps = compute_frozen_reps(smiles_list)

        for rep_name, X in frozen_reps.items():
            method_key = f"{rep_name}_frozen"
            t0 = time.time()
            try:
                fold_scores = evaluate_frozen(X, labels, groups, n_folds)
                summary = _summarize(fold_scores)
                summary['dim'] = X.shape[1]
                summary['time_s'] = round(time.time() - t0, 1)
                summary['per_fold'] = fold_scores
                ds_result['methods'][method_key] = summary
                _print_scores(method_key, summary)
            except Exception as e:
                print(f"    {method_key}: FAILED: {e}")
                ds_result['methods'][method_key] = {'error': str(e)}

    # ----- LM fine-tuning -----
    lm_methods_requested = [m for m in methods if m in ('lora', 'freeze_head', 'contrastive')]
    if lm_methods_requested:
        print(f"\n  --- LM fine-tuning ({', '.join(lm_methods_requested)}) ---")
        lm_factories = _make_lm_factories(
            smiles_list, dataset_name, lm_methods_requested, epochs, cache,
        )

        for method_key, factory in lm_factories.items():
            print(f"    {method_key}:", end='', flush=True)
            t0 = time.time()
            try:
                fold_scores = evaluate_factory(factory, labels, groups, n_folds)
                summary = _summarize(fold_scores)
                summary['time_s'] = round(time.time() - t0, 1)
                summary['per_fold'] = fold_scores
                ds_result['methods'][method_key] = summary
                print()
                _print_scores(method_key, summary)
            except Exception as e:
                print(f" FAILED: {e}")
                ds_result['methods'][method_key] = {'error': str(e)}

    # ----- GNN fine-tuning -----
    if 'gnn_ft' in methods:
        print("\n  --- GNN fine-tuning ---")
        gnn_factories = _make_gnn_factories(smiles_list, dataset_name, epochs, cache)

        for method_key, factory in gnn_factories.items():
            print(f"    {method_key}:", end='', flush=True)
            t0 = time.time()
            try:
                fold_scores = evaluate_factory(factory, labels, groups, n_folds)
                summary = _summarize(fold_scores)
                summary['time_s'] = round(time.time() - t0, 1)
                summary['per_fold'] = fold_scores
                ds_result['methods'][method_key] = summary
                print()
                _print_scores(method_key, summary)
            except Exception as e:
                print(f" FAILED: {e}")
                ds_result['methods'][method_key] = {'error': str(e)}

    # ----- GROVER fine-tuning -----
    if 'grover_ft' in methods:
        print("\n  --- GROVER fine-tuning ---")
        grover_factories = _make_grover_factory(smiles_list, dataset_name, epochs, cache)

        for method_key, factory in grover_factories.items():
            print(f"    {method_key}:", end='', flush=True)
            t0 = time.time()
            try:
                fold_scores = evaluate_factory(factory, labels, groups, n_folds)
                summary = _summarize(fold_scores)
                summary['time_s'] = round(time.time() - t0, 1)
                summary['per_fold'] = fold_scores
                ds_result['methods'][method_key] = summary
                print()
                _print_scores(method_key, summary)
            except Exception as e:
                print(f" FAILED: {e}")
                ds_result['methods'][method_key] = {'error': str(e)}

    return ds_result


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(all_results: Dict[str, dict]):
    """Print a final summary table across datasets, ranked by RMSE."""
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE  (ranked by RMSE, lower is better)")
    print(f"{'=' * 90}")

    for ds_name, ds_res in all_results.items():
        print(f"\n  {ds_name}  ({ds_res['n_samples']} samples, "
              f"{ds_res['n_scaffolds']} scaffolds)")
        print(f"  {'Method':30s}  {'RMSE':>12s}  {'R2':>12s}  {'rho':>12s}")
        print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")

        # Sort methods by RMSE (ascending = best first)
        method_items = []
        for method_name, summary in ds_res['methods'].items():
            if 'error' in summary:
                continue
            method_items.append((method_name, summary))

        method_items.sort(key=lambda x: x[1].get('rmse_mean', float('inf')))

        for method_name, summary in method_items:
            rmse_str = f"{summary['rmse_mean']:.4f}+/-{summary['rmse_std']:.4f}"
            r2_str = f"{summary['r2_mean']:.4f}+/-{summary['r2_std']:.4f}"
            rho_str = f"{summary['spearman_mean']:.4f}+/-{summary['spearman_std']:.4f}"
            print(f"  {method_name:30s}  {rmse_str:>12s}  {r2_str:>12s}  {rho_str:>12s}")

        # Highlight best frozen vs best fine-tuned
        frozen_items = [(n, s) for n, s in method_items if n.endswith('_frozen')]
        ft_items = [(n, s) for n, s in method_items if not n.endswith('_frozen')]

        if frozen_items and ft_items:
            best_frozen_rmse = frozen_items[0][1]['rmse_mean']
            best_ft_rmse = ft_items[0][1]['rmse_mean']
            delta = best_frozen_rmse - best_ft_rmse
            winner = "FT wins" if delta > 0 else "Frozen wins"
            print(f"\n  Best frozen: {frozen_items[0][0]} (RMSE={best_frozen_rmse:.4f})")
            print(f"  Best FT:     {ft_items[0][0]} (RMSE={best_ft_rmse:.4f})")
            print(f"  Delta: {delta:+.4f} ({winner})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Molecular fine-tuning comparison: frozen vs LoRA vs '
                    'freeze+head vs contrastive vs GNN FT vs GROVER FT')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset names (default: logd,caco2,herg)')
    parser.add_argument(
        '--methods', type=str, default=None,
        help='Comma-separated methods to run: '
             'frozen,lora,freeze_head,contrastive,gnn_ft,grover_ft '
             '(default: all)')
    parser.add_argument(
        '--epochs', type=int, default=5,
        help='Fine-tuning epochs (default: 5)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of scaffold GroupKFold splits (default: 5)')
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching of fine-tuned embeddings')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path (file or directory; default: '
             'tests/results/comparisons/molecular_finetune_comparison.json)')

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    methods = (args.methods.split(',') if args.methods
               else ALL_METHODS)
    cache = not args.no_cache

    # Validate method names
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method '{m}'. "
                         f"Choose from: {', '.join(ALL_METHODS)}")

    print("=" * 70)
    print("MOLECULAR FINE-TUNING COMPARISON")
    print(f"  Datasets: {datasets}")
    print(f"  Methods:  {methods}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Folds:    {args.n_folds}")
    print(f"  Cache:    {cache}")
    print("=" * 70)

    # Load dataset helper from molecular survey
    from find_best_hybrid_molecular import load_dataset

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        try:
            smiles_list, labels, task_type = load_dataset(ds_name)
        except Exception as e:
            print(f"  FAILED to load dataset: {e}")
            continue

        groups = _scaffold_groups(smiles_list)
        n_scaffolds = len(np.unique(groups))
        print(f"  {len(labels)} samples, {n_scaffolds} scaffolds, task={task_type}")

        t_ds = time.time()
        ds_result = run_dataset(
            dataset_name=ds_name,
            smiles_list=smiles_list,
            labels=labels,
            groups=groups,
            methods=methods,
            n_folds=args.n_folds,
            epochs=args.epochs,
            cache=cache,
        )
        ds_result['total_time_s'] = round(time.time() - t_ds, 1)
        all_results[ds_name] = ds_result

    # Summary table
    print_summary_table(all_results)

    # Save results
    output_path = args.output or 'tests/results/comparisons'
    p = Path(output_path)
    if p.suffix == '.json':
        out_file = p
    else:
        p.mkdir(parents=True, exist_ok=True)
        out_file = p / 'molecular_finetune_comparison.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    from experiments._shared import convert_numpy
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"\nResults saved: {out_file}")


if __name__ == '__main__':
    main()
