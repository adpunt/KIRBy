"""WSI Aggregation Comparison: Mean Pool vs Gated Attention vs ABMIL.

Compares WSI aggregation methods across all 7 WSI encoder models
with proper TSS GroupKFold cross-validation.

7 WSI encoders: ctranspath, phikon, hoptimus0, gigapath, moco, titan, uni2h
3 aggregation methods: Mean Pool, Gated Attention Pool, ABMIL (supervised)

Usage:
    python tests/experiments/wsi_abmil_comparison.py --dataset tcga_luad_vs_lusc
    python tests/experiments/wsi_abmil_comparison.py --dataset tcga_brca_pam50
    python tests/experiments/wsi_abmil_comparison.py --n-folds 2
    python tests/experiments/wsi_abmil_comparison.py --wsi-models ctranspath,phikon

Requires WSI patch-level embeddings on server.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

ALL_WSI_MODELS = [
    'ctranspath', 'phikon', 'hoptimus0', 'gigapath', 'moco', 'titan', 'uni2h',
]

AGGREGATION_METHODS = ['mean_pool', 'gated_attention', 'abmil']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='tcga_luad_vs_lusc',
                        help='TCGA dataset name')
    parser.add_argument('--wsi-models', default=None,
                        help=f'Comma-separated WSI models (default: all). '
                             f'Options: {",".join(ALL_WSI_MODELS)}')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--output', default='tests/results',
                        help='Output directory')
    args = parser.parse_args()

    wsi_models = (args.wsi_models.split(',') if args.wsi_models
                  else ALL_WSI_MODELS)

    from kirby.datasets.tcga import load_dataset
    from kirby.representations.genomic import (
        load_wsi_features,
        _load_wsi_patch_embeddings,
        gated_attention_pool,
        ABMIL,
    )

    # Load dataset
    print(f"Loading {args.dataset}...")
    expression, labels, task_type, metadata = load_dataset(args.dataset)
    sample_ids = metadata['sample_ids']
    wsi_dir = metadata.get('wsi_dir')
    tss_groups = metadata.get('tss_groups')

    if wsi_dir is None:
        print("ERROR: No WSI directory found in metadata. Run on server.")
        return

    # Encode labels for classification
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"  {len(sample_ids)} samples, {n_classes} classes: {list(le.classes_)}")

    # Groups for CV
    if tss_groups is not None:
        groups = np.array(tss_groups)
    else:
        groups = np.arange(len(y))

    n_unique_groups = len(np.unique(groups))
    effective_folds = min(args.n_folds, n_unique_groups)
    print(f"  {n_unique_groups} TSS groups, {effective_folds} folds")

    all_results = {}

    for wsi_model in wsi_models:
        print(f"\n{'='*60}")
        print(f"WSI Encoder: {wsi_model}")
        print(f"{'='*60}")

        # Load patch-level embeddings
        print(f"  Loading WSI patches ({wsi_model})...")
        try:
            all_patches, d = _load_wsi_patch_embeddings(
                sample_ids, wsi_dir, wsi_model)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        n_with_patches = sum(1 for p in all_patches if p is not None)
        print(f"  {n_with_patches}/{len(sample_ids)} slides have patches, dim={d}")

        if n_with_patches < 20:
            print("  SKIPPED: Too few slides with patches")
            continue

        model_results = {}
        cv = GroupKFold(n_splits=effective_folds)
        splits = list(cv.split(y, y, groups=groups))

        # --- Method 1: Mean Pool ---
        print(f"\n  --- Mean Pool ({wsi_model}) ---")
        t0 = time.time()
        try:
            X_mean = load_wsi_features(sample_ids, wsi_dir, wsi_model)
            scores = _evaluate_cv(X_mean, y, splits, n_classes)
            scores['time'] = time.time() - t0
            model_results['mean_pool'] = scores
            _print_scores('mean_pool', scores)
        except Exception as e:
            print(f"    FAILED: {e}")
            model_results['mean_pool'] = {'error': str(e)}

        # --- Method 2: Gated Attention Pool ---
        print(f"\n  --- Gated Attention Pool ({wsi_model}) ---")
        t0 = time.time()
        try:
            X_gated = gated_attention_pool(all_patches, d)
            scores = _evaluate_cv(X_gated, y, splits, n_classes)
            scores['time'] = time.time() - t0
            model_results['gated_attention'] = scores
            _print_scores('gated_attention', scores)
        except Exception as e:
            print(f"    FAILED: {e}")
            model_results['gated_attention'] = {'error': str(e)}

        # --- Method 3: ABMIL (supervised, per-fold) ---
        print(f"\n  --- ABMIL supervised ({wsi_model}) ---")
        t0 = time.time()
        try:
            scores = _evaluate_abmil_cv(
                all_patches, y, splits, d, n_classes)
            scores['time'] = time.time() - t0
            model_results['abmil'] = scores
            _print_scores('abmil', scores)
        except Exception as e:
            print(f"    FAILED: {e}")
            model_results['abmil'] = {'error': str(e)}

        all_results[wsi_model] = model_results

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"WSI Aggregation Comparison: {args.dataset}")
    print(f"{'='*70}")
    print(f"{'Encoder':<15} {'Method':<20} {'AUROC':>12} {'Accuracy':>12} {'F1w':>12}")
    print("-" * 72)
    for wsi_model in wsi_models:
        if wsi_model not in all_results:
            continue
        for method in AGGREGATION_METHODS:
            r = all_results[wsi_model].get(method, {})
            if 'error' in r:
                print(f"{wsi_model:<15} {method:<20} {'FAILED':>12}")
                continue
            auroc = r.get('auroc_mean', float('nan'))
            acc = r.get('accuracy_mean', float('nan'))
            f1w = r.get('f1w_mean', float('nan'))
            print(f"{wsi_model:<15} {method:<20} "
                  f"{auroc:>5.3f}+/-{r.get('auroc_std', 0):.3f} "
                  f"{acc:>5.3f}+/-{r.get('accuracy_std', 0):.3f} "
                  f"{f1w:>5.3f}+/-{r.get('f1w_std', 0):.3f}")

    # Save
    out_dir = Path(args.output) / 'wsi_abmil_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.dataset}.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=_convert)
    print(f"\n[saved {out_file}]")


def _convert(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _print_scores(method, scores):
    """Print AUROC/accuracy/F1w for a method."""
    auroc = scores.get('auroc_mean', float('nan'))
    acc = scores.get('accuracy_mean', float('nan'))
    f1w = scores.get('f1w_mean', float('nan'))
    print(f"    AUROC={auroc:.3f}  Accuracy={acc:.3f}  F1w={f1w:.3f}")


def _evaluate_cv(X, y, splits, n_classes):
    """Evaluate with RidgeClassifier across CV folds."""
    fold_scores = []

    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        # Decision function for AUROC
        dec = clf.decision_function(X_te)

        acc = float(accuracy_score(y_te, y_pred))
        f1w = float(f1_score(y_te, y_pred, average='weighted'))

        try:
            if n_classes == 2:
                auroc = float(roc_auc_score(y_te, dec))
            else:
                y_bin = label_binarize(y_te, classes=list(range(n_classes)))
                auroc = float(roc_auc_score(y_bin, dec, average='macro',
                                            multi_class='ovr'))
        except Exception:
            auroc = float('nan')

        fold_scores.append({'auroc': auroc, 'accuracy': acc, 'f1w': f1w})

    aurocs = [s['auroc'] for s in fold_scores if not np.isnan(s['auroc'])]
    accs = [s['accuracy'] for s in fold_scores]
    f1ws = [s['f1w'] for s in fold_scores]

    return {
        'auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
        'auroc_std': float(np.std(aurocs)) if aurocs else 0.0,
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'f1w_mean': float(np.mean(f1ws)),
        'f1w_std': float(np.std(f1ws)),
        'per_fold': fold_scores,
    }


def _evaluate_abmil_cv(all_patches, y, splits, d, n_classes):
    """Evaluate ABMIL with per-fold training."""
    from kirby.representations.genomic import ABMIL

    fold_scores = []

    for train_idx, test_idx in splits:
        train_patches = [all_patches[i] for i in train_idx]
        test_patches = [all_patches[i] for i in test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        abmil = ABMIL(input_dim=d, hidden_dim=128, lr=1e-3, epochs=20)
        abmil.fit(train_patches, y_tr,
                  task_type='classification')

        X_tr = abmil.transform(train_patches)
        X_te = abmil.transform(test_patches)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        dec = clf.decision_function(X_te)

        acc = float(accuracy_score(y_te, y_pred))
        f1w = float(f1_score(y_te, y_pred, average='weighted'))

        try:
            if n_classes == 2:
                auroc = float(roc_auc_score(y_te, dec))
            else:
                y_bin = label_binarize(y_te, classes=list(range(n_classes)))
                auroc = float(roc_auc_score(y_bin, dec, average='macro',
                                            multi_class='ovr'))
        except Exception:
            auroc = float('nan')

        fold_scores.append({'auroc': auroc, 'accuracy': acc, 'f1w': f1w})

    aurocs = [s['auroc'] for s in fold_scores if not np.isnan(s['auroc'])]
    accs = [s['accuracy'] for s in fold_scores]
    f1ws = [s['f1w'] for s in fold_scores]

    return {
        'auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
        'auroc_std': float(np.std(aurocs)) if aurocs else 0.0,
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'f1w_mean': float(np.mean(f1ws)),
        'f1w_std': float(np.std(f1ws)),
        'per_fold': fold_scores,
    }


if __name__ == '__main__':
    main()
