#!/usr/bin/env python3
"""
TCGA Genomic Foundation Model Comparison: Frozen vs Fine-Tuned

Does fine-tuning genomic foundation models beat frozen extraction
for cancer classification?

Methods:
  1. Frozen Geneformer + Ridge (baseline)
  2. Fine-tuned Geneformer + Ridge
  3. Frozen scFoundation + Ridge
  4. Fine-tuned scFoundation + Ridge
  5. Frozen BulkRNABert + Ridge
  6. Frozen scGPT + Ridge
  7. Frozen UCE + Ridge

Datasets:
  - tcga_luad_vs_lusc  (binary classification)
  - tcga_brca_pam50    (4-class classification)

Protocol: TSS GroupKFold (5-fold), AUROC (macro OVR) + F1-weighted.

Usage:
    python tests/experiments/tcga_foundation_comparison.py --dataset tcga_luad_vs_lusc
    python tests/experiments/tcga_foundation_comparison.py --dataset tcga_brca_pam50
    python tests/experiments/tcga_foundation_comparison.py --n-folds 2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# =========================================================================
# DATASETS
# =========================================================================

DEFAULT_DATASETS = [
    'tcga_luad_vs_lusc',
    'tcga_brca_pam50',
]


def load_tcga(dataset_name: str) -> dict:
    """Load a TCGA classification dataset.

    Returns dict with: expression, labels, gene_names, tss_groups,
                       sample_ids, task_type, class_names, n_classes.
    """
    from kirby.datasets.tcga import load_dataset

    expression, labels, task_type, metadata = load_dataset(dataset_name)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    tss_groups = metadata['tss_groups']
    gene_names = metadata['gene_names']
    sample_ids = metadata['sample_ids']
    class_names = metadata.get('class_names', [str(c) for c in le.classes_])

    print(f"  {dataset_name}: {len(y)} samples, {n_classes} classes "
          f"({', '.join(class_names)}), "
          f"{len(np.unique(tss_groups))} TSS groups")

    return {
        'expression': expression,
        'labels': y,
        'gene_names': gene_names,
        'tss_groups': tss_groups,
        'sample_ids': sample_ids,
        'task_type': task_type,
        'class_names': class_names,
        'n_classes': n_classes,
    }


# =========================================================================
# CLASSIFICATION EVALUATION
# =========================================================================

def evaluate_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """Evaluate a single fold: Ridge + LogReg for AUROC.

    Returns dict with auroc, f1_weighted, ridge_accuracy.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)

    # RidgeClassifier for F1
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(X_tr, y_train)
    ridge_pred = ridge.predict(X_va)
    f1w = float(f1_score(y_val, ridge_pred, average='weighted'))
    ridge_acc = float(np.mean(ridge_pred == y_val))

    # LogisticRegression for AUROC (needs predict_proba)
    if n_classes == 2:
        logreg = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        logreg.fit(X_tr, y_train)
        proba = logreg.predict_proba(X_va)[:, 1]
        auroc = float(roc_auc_score(y_val, proba))
    else:
        logreg = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        logreg.fit(X_tr, y_train)
        proba = logreg.predict_proba(X_va)
        y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
        auroc = float(roc_auc_score(
            y_val_bin, proba, average='macro', multi_class='ovr'))

    return {
        'auroc': auroc,
        'f1_weighted': f1w,
        'ridge_accuracy': ridge_acc,
    }


def evaluate_frozen_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Run GroupKFold CV on a frozen embedding matrix.

    Returns dict with (mean, std) for auroc, f1_weighted.
    """
    n_unique = len(np.unique(groups))
    effective_folds = min(n_folds, n_unique)
    if effective_folds < 2:
        return {
            'auroc': (float('nan'), float('nan')),
            'f1_weighted': (float('nan'), float('nan')),
        }

    cv = GroupKFold(n_splits=effective_folds)
    fold_results = []

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        result = evaluate_fold(
            X[train_idx], X[val_idx],
            y[train_idx], y[val_idx],
            n_classes,
        )
        fold_results.append(result)

    aurocs = [r['auroc'] for r in fold_results]
    f1ws = [r['f1_weighted'] for r in fold_results]

    return {
        'auroc': (float(np.mean(aurocs)), float(np.std(aurocs))),
        'f1_weighted': (float(np.mean(f1ws)), float(np.std(f1ws))),
        'per_fold': fold_results,
    }


def evaluate_factory_cv(
    factory: Callable,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Run GroupKFold CV with a fine-tuning factory.

    The factory signature: factory(train_idx, val_idx, y_train) -> (X_train, X_val)
    """
    n_unique = len(np.unique(groups))
    effective_folds = min(n_folds, n_unique)
    if effective_folds < 2:
        return {
            'auroc': (float('nan'), float('nan')),
            'f1_weighted': (float('nan'), float('nan')),
        }

    dummy = np.zeros((len(y), 1))
    cv = GroupKFold(n_splits=effective_folds)
    fold_results = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(dummy, y, groups=groups)):
        try:
            X_train, X_val = factory(train_idx, val_idx, y[train_idx])
        except Exception as e:
            print(f"      Fold {fold_i + 1} FT error: {e}")
            continue

        result = evaluate_fold(X_train, X_val, y[train_idx], y[val_idx], n_classes)
        fold_results.append(result)

    if not fold_results:
        return {
            'auroc': (float('nan'), float('nan')),
            'f1_weighted': (float('nan'), float('nan')),
        }

    aurocs = [r['auroc'] for r in fold_results]
    f1ws = [r['f1_weighted'] for r in fold_results]

    return {
        'auroc': (float(np.mean(aurocs)), float(np.std(aurocs))),
        'f1_weighted': (float(np.mean(f1ws)), float(np.std(f1ws))),
        'per_fold': fold_results,
    }


# =========================================================================
# FROZEN EMBEDDING GENERATORS
# =========================================================================

def get_frozen_embeddings(
    expression: np.ndarray,
    gene_names: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Generate all frozen foundation model embeddings.

    Wraps each in try/except so missing models are skipped gracefully.
    Returns {name: (n_samples, d) array}.
    """
    from kirby.representations.genomic import (
        create_geneformer_embeddings,
        create_scfoundation_embeddings,
        create_bulkrnabert_embeddings,
        create_scgpt_embeddings,
        create_uce_embeddings,
    )

    models = [
        ('geneformer', create_geneformer_embeddings),
        ('scfoundation', create_scfoundation_embeddings),
        ('bulkrnabert', create_bulkrnabert_embeddings),
        ('scgpt', create_scgpt_embeddings),
        ('uce', create_uce_embeddings),
    ]

    embeddings = {}
    for name, fn in models:
        print(f"    Generating frozen {name}...", end='', flush=True)
        t0 = time.time()
        try:
            X = fn(expression, gene_names)
            embeddings[name] = X
            elapsed = time.time() - t0
            print(f" {X.shape[1]}D ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f" FAILED ({elapsed:.1f}s): {e}")

    return embeddings


# =========================================================================
# FINE-TUNING FACTORIES
# =========================================================================

def get_finetuning_factories(
    expression: np.ndarray,
    gene_names: np.ndarray,
    sample_ids: List[str],
    dataset_name: str,
    epochs: int = 3,
    cache: bool = True,
) -> Dict[str, Callable]:
    """Create fine-tuning factories for Geneformer and scFoundation.

    Returns {name: factory_fn}.
    Each factory: factory(train_idx, val_idx, y_train) -> (X_train, X_val)
    """
    factories = {}

    # Geneformer fine-tuning
    print("    Creating Geneformer FT factory...", end='', flush=True)
    try:
        from kirby.finetuning.tcga_geneformer import make_geneformer_factory
        factories['geneformer_ft'] = make_geneformer_factory(
            expression_data=expression,
            gene_names=gene_names,
            sample_ids=sample_ids,
            dataset_name=dataset_name,
            task_type='classification',
            epochs=epochs,
            cache=cache,
        )
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    # scFoundation fine-tuning
    print("    Creating scFoundation FT factory...", end='', flush=True)
    try:
        from kirby.finetuning.tcga_scfoundation import make_scfoundation_factory
        factories['scfoundation_ft'] = make_scfoundation_factory(
            expression_data=expression,
            gene_names=gene_names,
            sample_ids=sample_ids,
            dataset_name=dataset_name,
            task_type='classification',
            epochs=epochs,
            cache=cache,
        )
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    return factories


# =========================================================================
# MAIN
# =========================================================================

def run_experiment(
    dataset_name: str,
    n_folds: int = 5,
    ft_epochs: int = 3,
    cache: bool = True,
) -> dict:
    """Run full frozen-vs-fine-tuned comparison on one dataset.

    Returns results dict ready for JSON serialization.
    """
    print(f"\n{'=' * 65}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 65}")

    # Load data
    data = load_tcga(dataset_name)
    expression = data['expression']
    y = data['labels']
    gene_names = data['gene_names']
    tss_groups = data['tss_groups']
    sample_ids = data['sample_ids']
    n_classes = data['n_classes']
    class_names = data['class_names']

    results = {
        'dataset': dataset_name,
        'n_samples': len(y),
        'n_classes': n_classes,
        'class_names': class_names,
        'n_folds': n_folds,
        'n_tss_groups': int(len(np.unique(tss_groups))),
        'frozen': {},
        'finetuned': {},
    }

    # ── Frozen baselines ────────────────────────────────────────────────
    print("\n  --- Frozen baselines ---")
    frozen_embs = get_frozen_embeddings(expression, gene_names)

    for name, X in frozen_embs.items():
        print(f"    Evaluating frozen {name}...", end='', flush=True)
        t0 = time.time()
        res = evaluate_frozen_cv(X, y, tss_groups, n_classes, n_folds)
        elapsed = time.time() - t0

        auroc_mean, auroc_std = res['auroc']
        f1w_mean, f1w_std = res['f1_weighted']

        results['frozen'][name] = {
            'auroc_mean': round(auroc_mean, 4),
            'auroc_std': round(auroc_std, 4),
            'f1w_mean': round(f1w_mean, 4),
            'f1w_std': round(f1w_std, 4),
            'dim': int(X.shape[1]),
            'time_s': round(elapsed, 1),
            'per_fold': res.get('per_fold', []),
        }
        print(f" AUROC={auroc_mean:.4f}+/-{auroc_std:.4f}"
              f"  F1w={f1w_mean:.4f}+/-{f1w_std:.4f} ({elapsed:.1f}s)")

    # ── Fine-tuned methods ──────────────────────────────────────────────
    print("\n  --- Fine-tuned methods ---")
    factories = get_finetuning_factories(
        expression, gene_names, sample_ids, dataset_name,
        epochs=ft_epochs, cache=cache,
    )

    for name, factory in factories.items():
        print(f"    Evaluating {name}...", end='', flush=True)
        t0 = time.time()
        res = evaluate_factory_cv(factory, y, tss_groups, n_classes, n_folds)
        elapsed = time.time() - t0

        auroc_mean, auroc_std = res['auroc']
        f1w_mean, f1w_std = res['f1_weighted']

        results['finetuned'][name] = {
            'auroc_mean': round(auroc_mean, 4),
            'auroc_std': round(auroc_std, 4),
            'f1w_mean': round(f1w_mean, 4),
            'f1w_std': round(f1w_std, 4),
            'time_s': round(elapsed, 1),
            'per_fold': res.get('per_fold', []),
        }
        print(f" AUROC={auroc_mean:.4f}+/-{auroc_std:.4f}"
              f"  F1w={f1w_mean:.4f}+/-{f1w_std:.4f} ({elapsed:.1f}s)")

        # Delta vs frozen baseline
        base_name = name.replace('_ft', '')
        if base_name in results['frozen']:
            base_auroc = results['frozen'][base_name]['auroc_mean']
            delta = auroc_mean - base_auroc
            sign = '+' if delta >= 0 else ''
            print(f"      vs frozen {base_name}: {sign}{delta:.4f} AUROC")

    return results


def print_summary_table(all_results: Dict[str, dict]):
    """Print a formatted summary table across all datasets."""
    print("\n")
    print("=" * 80)
    print("TCGA FOUNDATION MODEL COMPARISON — SUMMARY")
    print("=" * 80)

    for ds_name, res in all_results.items():
        n = res['n_samples']
        nc = res['n_classes']
        print(f"\n  {ds_name} ({n} samples, {nc} classes)")
        print(f"  {'Method':<30} {'AUROC':>16} {'F1-weighted':>16}")
        print(f"  {'-' * 62}")

        # Frozen
        for name, r in sorted(res['frozen'].items()):
            auroc_str = f"{r['auroc_mean']:.4f}+/-{r['auroc_std']:.4f}"
            f1w_str = f"{r['f1w_mean']:.4f}+/-{r['f1w_std']:.4f}"
            print(f"  {'frozen ' + name:<30} {auroc_str:>16} {f1w_str:>16}")

        # Fine-tuned
        for name, r in sorted(res['finetuned'].items()):
            auroc_str = f"{r['auroc_mean']:.4f}+/-{r['auroc_std']:.4f}"
            f1w_str = f"{r['f1w_mean']:.4f}+/-{r['f1w_std']:.4f}"
            label = name
            # Show delta vs frozen baseline
            base_name = name.replace('_ft', '')
            if base_name in res['frozen']:
                delta = r['auroc_mean'] - res['frozen'][base_name]['auroc_mean']
                sign = '+' if delta >= 0 else ''
                label = f"{name} ({sign}{delta:.4f})"
            print(f"  {label:<30} {auroc_str:>16} {f1w_str:>16}")

    # Best method per dataset
    print(f"\n  {'Best method per dataset':}")
    print(f"  {'-' * 50}")
    for ds_name, res in all_results.items():
        best_name = None
        best_auroc = -1.0
        for name, r in res['frozen'].items():
            if r['auroc_mean'] > best_auroc:
                best_auroc = r['auroc_mean']
                best_name = f"frozen {name}"
        for name, r in res['finetuned'].items():
            if r['auroc_mean'] > best_auroc:
                best_auroc = r['auroc_mean']
                best_name = name
        print(f"  {ds_name:<30} {best_name} (AUROC={best_auroc:.4f})")


def save_results(all_results: dict, output_dir: str):
    """Save results as JSON."""
    out = Path(output_dir) / 'tcga_foundation_comparison'
    out.mkdir(parents=True, exist_ok=True)

    # Per-dataset files
    for ds_name, res in all_results.items():
        filepath = out / f'{ds_name}.json'
        # Strip per_fold arrays for the combined file but keep in per-dataset
        with open(filepath, 'w') as f:
            json.dump(_convert_numpy(res), f, indent=2)
        print(f"  [saved {filepath}]")

    # Combined summary
    combined = out / 'summary.json'
    summary = {}
    for ds_name, res in all_results.items():
        ds_summary = {
            'n_samples': res['n_samples'],
            'n_classes': res['n_classes'],
            'frozen': {
                k: {kk: vv for kk, vv in v.items() if kk != 'per_fold'}
                for k, v in res['frozen'].items()
            },
            'finetuned': {
                k: {kk: vv for kk, vv in v.items() if kk != 'per_fold'}
                for k, v in res['finetuned'].items()
            },
        }
        summary[ds_name] = ds_summary

    with open(combined, 'w') as f:
        json.dump(_convert_numpy(summary), f, indent=2)
    print(f"  [saved {combined}]")


def _convert_numpy(obj):
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
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description='TCGA genomic foundation model comparison: '
                    'frozen vs fine-tuned for cancer classification')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name (default: run both '
                             'tcga_luad_vs_lusc and tcga_brca_pam50)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of GroupKFold folds (default: 5)')
    parser.add_argument('--ft-epochs', type=int, default=3,
                        help='Fine-tuning epochs (default: 3)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable fine-tuning cache')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS
    cache = not args.no_cache

    print("=" * 65)
    print("TCGA GENOMIC FOUNDATION MODEL COMPARISON")
    print(f"  Datasets: {datasets}")
    print(f"  Folds: {args.n_folds}, FT epochs: {args.ft_epochs}")
    print(f"  Cache: {cache}")
    print("=" * 65)

    all_results = {}
    for ds_name in datasets:
        try:
            res = run_experiment(
                ds_name,
                n_folds=args.n_folds,
                ft_epochs=args.ft_epochs,
                cache=cache,
            )
            all_results[ds_name] = res
        except Exception as e:
            print(f"\nFAILED on {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary_table(all_results)
        save_results(all_results, args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
