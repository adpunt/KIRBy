"""WSI Aggregation Comparison: Mean Pool vs Gated Attention vs ABMIL.

Single-run experiment (no CV) comparing three WSI aggregation methods
on a TCGA classification task.

Usage:
    python tests/experiments/wsi_abmil_comparison.py --dataset tcga_luad_vs_lusc
    python tests/experiments/wsi_abmil_comparison.py --dataset tcga_brca_pam50

Requires WSI patch-level embeddings on server.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='tcga_luad_vs_lusc',
                        help='TCGA dataset name')
    parser.add_argument('--wsi-model', default='ctranspath',
                        help='WSI encoder model (ctranspath, phikon, etc.)')
    parser.add_argument('--output', default='tests/results',
                        help='Output directory')
    args = parser.parse_args()

    from find_best_hybrid_tcga import load_dataset, assign_tss_groups
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

    if wsi_dir is None:
        print("ERROR: No WSI directory found in metadata. Run on server.")
        return

    print(f"  {len(sample_ids)} samples, task_type={task_type}")

    # Encode labels for classification
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"  {n_classes} classes: {list(le.classes_)}")

    # Single train/test split (80/20, stratified)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    y_train, y_test = y[train_idx], y[test_idx]
    train_ids = [sample_ids[i] for i in train_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    # Load patch-level embeddings
    print(f"\nLoading WSI patches ({args.wsi_model})...")
    all_patches, d = _load_wsi_patch_embeddings(sample_ids, wsi_dir, args.wsi_model)
    n_with_patches = sum(1 for p in all_patches if p is not None)
    print(f"  {n_with_patches}/{len(sample_ids)} slides have patches, dim={d}")

    if n_with_patches < 20:
        print("ERROR: Too few slides with patch embeddings. Check WSI data.")
        return

    train_patches = [all_patches[i] for i in train_idx]
    test_patches = [all_patches[i] for i in test_idx]

    results = {}

    # --- Method 1: Mean Pool (baseline) ---
    print("\n--- Method 1: Mean Pool ---")
    t0 = time.time()
    X_mean = load_wsi_features(sample_ids, wsi_dir, args.wsi_model)
    X_train_mean = X_mean[train_idx]
    X_test_mean = X_mean[test_idx]
    results['mean_pool'] = _evaluate(X_train_mean, X_test_mean, y_train, y_test,
                                      n_classes, time.time() - t0)

    # --- Method 2: Gated Attention Pool (frozen, unsupervised) ---
    print("\n--- Method 2: Gated Attention Pool ---")
    t0 = time.time()
    X_gated = gated_attention_pool(all_patches, d)
    X_train_gated = X_gated[train_idx]
    X_test_gated = X_gated[test_idx]
    results['gated_attention'] = _evaluate(X_train_gated, X_test_gated, y_train, y_test,
                                            n_classes, time.time() - t0)

    # --- Method 3: ABMIL (supervised, trained on train set) ---
    print("\n--- Method 3: ABMIL (supervised) ---")
    t0 = time.time()
    abmil = ABMIL(input_dim=d, hidden_dim=128, lr=1e-3, epochs=20)
    abmil.fit(train_patches, y_train,
              task_type='classification' if n_classes > 2 else 'regression')
    X_train_abmil = abmil.transform(train_patches)
    X_test_abmil = abmil.transform(test_patches)
    results['abmil'] = _evaluate(X_train_abmil, X_test_abmil, y_train, y_test,
                                  n_classes, time.time() - t0)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"WSI Aggregation Comparison: {args.dataset} / {args.wsi_model}")
    print("=" * 60)
    print(f"{'Method':<25} {'RF Acc':>8} {'Ridge Acc':>10} {'Time':>8}")
    print("-" * 60)
    for method, r in results.items():
        print(f"{method:<25} {r['rf_accuracy']:>8.3f} {r['ridge_accuracy']:>10.3f} "
              f"{r['time']:>7.1f}s")

    # Save
    out_dir = Path(args.output) / 'wsi_abmil_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.dataset}_{args.wsi_model}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved {out_file}]")


def _evaluate(X_train, X_test, y_train, y_test, n_classes, elapsed):
    """Evaluate with RF and Ridge."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_te))

    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(X_tr, y_train)
    ridge_acc = accuracy_score(y_test, ridge.predict(X_te))

    print(f"  RF Accuracy: {rf_acc:.3f}")
    print(f"  Ridge Accuracy: {ridge_acc:.3f}")

    return {
        'rf_accuracy': float(rf_acc),
        'ridge_accuracy': float(ridge_acc),
        'time': float(elapsed),
    }


if __name__ == '__main__':
    main()
