#!/usr/bin/env python3
"""
Quick test: ESM-C and Uni-Mol v2 on one dataset each.

Runs on the py311 venv to verify these Py3.10+-only reps work and
are worth including.

Usage (from KIRBy root, with py311 venv active):
    source /data/stat-cadd/scat9264/py311-kirby/bin/activate
    python tests/test_py311_reps.py
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def evaluate(X, y, groups, n_folds=5):
    """Quick CV evaluation with Ridge."""
    cv = GroupKFold(n_splits=n_folds)
    scores = []
    for train_idx, val_idx in cv.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        m = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
        m.fit(X_train, y[train_idx])
        scores.append(r2_score(y[val_idx], m.predict(X_val)))
    return np.mean(scores), np.std(scores)


def test_unimol_v2():
    """Test Uni-Mol v2 on LogD (molecular)."""
    print("=" * 60)
    print("TEST: Uni-Mol v2 on LogD")
    print("=" * 60)

    from kirby.datasets.esol import load_esol
    from kirby.representations.molecular import (
        create_unimol_v2, create_ecfp4, create_chemberta,
    )

    # Load LogD if available, fall back to ESOL
    try:
        from experiments._shared import load_data
        data = load_data('logd', rep_preset='fast')
        smiles = data.get('smiles')
        y = data['labels']
        groups = data['groups']
        dataset_name = 'LogD'
    except Exception:
        df = load_esol()
        smiles = df['smiles'].tolist()
        y = df['logS'].values
        groups = np.arange(len(y)) % 5
        dataset_name = 'ESOL'

    print(f"Dataset: {dataset_name}, n={len(y)}")

    # Baseline: ECFP4
    t0 = time.time()
    X_ecfp4 = create_ecfp4(smiles)
    t_ecfp4 = time.time() - t0
    mean, std = evaluate(X_ecfp4, y, groups)
    print(f"  ECFP4      ({X_ecfp4.shape[1]:4d}D): R²={mean:.3f} ± {std:.3f}  [{t_ecfp4:.1f}s]")

    # Uni-Mol v2
    t0 = time.time()
    X_unimol = create_unimol_v2(smiles)
    t_unimol = time.time() - t0
    mean, std = evaluate(X_unimol, y, groups)
    print(f"  Uni-Mol v2 ({X_unimol.shape[1]:4d}D): R²={mean:.3f} ± {std:.3f}  [{t_unimol:.1f}s]")

    print()


def test_esmc():
    """Test ESM-C on DTA Davis (protein reps)."""
    print("=" * 60)
    print("TEST: ESM-C on Davis (protein side)")
    print("=" * 60)

    from kirby.representations.protein import (
        create_esmc_protein_embeddings,
        create_esm2_protein_embeddings,
    )

    try:
        from kirby.datasets.dta import load_dta_dataset
        data = load_dta_dataset('davis')
        protein_seqs = data['protein_seqs']
        y = data['labels']
        # Use unique proteins for grouping
        protein_ids = data.get('protein_ids', np.arange(len(y)))
        unique_prots = list(set(protein_ids))
        prot_to_group = {p: i % 5 for i, p in enumerate(unique_prots)}
        groups = np.array([prot_to_group[p] for p in protein_ids])
        dataset_name = 'Davis'
    except Exception as e:
        print(f"  Could not load Davis: {e}")
        print("  Falling back to synthetic test...")
        protein_seqs = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
            "MKKYTCTVCGYIYNPEDGDPDNGVNPGTDFKDIPDDWVCPLCGVGKDQFEEVEE",
        ] * 50
        y = np.random.randn(100)
        groups = np.arange(100) % 5
        dataset_name = 'Synthetic'

    print(f"Dataset: {dataset_name}, n={len(y)}, unique seqs={len(set(protein_seqs))}")

    # Baseline: ESM-2
    t0 = time.time()
    X_esm2 = create_esm2_protein_embeddings(list(set(protein_seqs)))
    t_esm2 = time.time() - t0
    # Map back to full dataset
    unique_seqs = list(set(protein_seqs))
    seq_to_idx = {s: i for i, s in enumerate(unique_seqs)}
    X_esm2_full = X_esm2[[seq_to_idx[s] for s in protein_seqs]]
    mean, std = evaluate(X_esm2_full, y, groups)
    print(f"  ESM-2  ({X_esm2.shape[1]:4d}D): R²={mean:.3f} ± {std:.3f}  [{t_esm2:.1f}s]")

    # ESM-C
    t0 = time.time()
    X_esmc = create_esmc_protein_embeddings(list(set(protein_seqs)))
    t_esmc = time.time() - t0
    X_esmc_full = X_esmc[[seq_to_idx[s] for s in protein_seqs]]
    mean, std = evaluate(X_esmc_full, y, groups)
    print(f"  ESM-C  ({X_esmc.shape[1]:4d}D): R²={mean:.3f} ± {std:.3f}  [{t_esmc:.1f}s]")

    print()


if __name__ == '__main__':
    import torch
    print(f"Python {sys.version}")
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print()

    test_unimol_v2()
    test_esmc()

    print("Done.")
