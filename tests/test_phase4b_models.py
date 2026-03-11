#!/usr/bin/env python3
"""
Phase 4B Smoke Test — ESM3, SaProt, Boltz-2 on real data.

Runs on the server (py311 venv) to verify Phase 4B forward-compat models
produce valid embeddings and are not garbage.

Usage (from KIRBy root, with py311 venv active):
    source /data/stat-cadd/scat9264/py311-kirby/bin/activate
    python tests/test_phase4b_models.py              # all available
    python tests/test_phase4b_models.py --skip-boltz  # skip Boltz-2 (slow)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# Test sequences (fallback if Davis unavailable)
SYNTH_PROTEINS = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
    "MKKYTCTVCGYIYNPEDGDPDNGVNPGTDFKDIPDDWVCPLCGVGKDQFEEVEE",
    "MASMTGGQQMGRGSEFDLVPGSWEQVTARNTVSAGMRSGSAPDTPFQRAGGPFQVR",
    "MRSLLILVLCFLPLAALGKVFGRCELAQALKGEQGSNSN",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
] * 10  # 50 sequences

HEAVY_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
LIGHT_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"


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


def load_davis_subset(max_seqs=50):
    """Load Davis dataset, deduplicate proteins, take a small subset."""
    try:
        from kirby.datasets.dta import load_dta_dataset
        data = load_dta_dataset('davis')
        protein_seqs = data['protein_seqs']
        y = data['labels']
        protein_ids = data.get('protein_ids', np.arange(len(y)))
        unique_prots = list(set(protein_ids))
        prot_to_group = {p: i % 5 for i, p in enumerate(unique_prots)}
        groups = np.array([prot_to_group[p] for p in protein_ids])
        # Deduplicate for embedding (embed unique, map back)
        unique_seqs = list(dict.fromkeys(protein_seqs))[:max_seqs]
        return unique_seqs, y, groups, protein_seqs, 'Davis'
    except Exception:
        y = np.random.randn(len(SYNTH_PROTEINS))
        groups = np.arange(len(SYNTH_PROTEINS)) % 5
        return SYNTH_PROTEINS, y, groups, SYNTH_PROTEINS, 'Synthetic'


def test_esm3_protein():
    """Test ESM3 on Davis protein sequences."""
    print("=" * 60)
    print("ESM3 — Protein embeddings (1536D)")
    print("=" * 60)

    from kirby.representations.protein import create_esm3_protein_embeddings

    unique_seqs, y, groups, all_seqs, ds = load_davis_subset(50)
    print(f"  Dataset: {ds}, {len(unique_seqs)} unique seqs")

    t0 = time.time()
    X = create_esm3_protein_embeddings(unique_seqs)
    elapsed = time.time() - t0

    assert X.shape == (len(unique_seqs), 1536), f"Bad shape: {X.shape}"
    assert not np.isnan(X).any(), "NaN in embeddings"
    assert not np.all(X == 0), "All-zero embeddings"
    print(f"  Shape: {X.shape}, dtype: {X.dtype}, time: {elapsed:.1f}s")

    # Quick eval if real data
    if ds == 'Davis':
        seq_idx = {s: i for i, s in enumerate(unique_seqs)}
        X_full = X[[seq_idx.get(s, 0) for s in all_seqs[:len(y)]]]
        mean, std = evaluate(X_full, y, groups)
        print(f"  Ridge R²: {mean:.3f} ± {std:.3f}")

    print()


def test_esm3_antibody():
    """Test ESM3 antibody embeddings (full + CDR)."""
    print("=" * 60)
    print("ESM3 — Antibody embeddings (3072D)")
    print("=" * 60)

    from kirby.representations.antibody import embed_esm3, embed_esm3_cdr

    t0 = time.time()
    full = embed_esm3([HEAVY_SEQ], [LIGHT_SEQ])
    t1 = time.time()
    cdr = embed_esm3_cdr([HEAVY_SEQ], [LIGHT_SEQ])
    t2 = time.time()

    assert full.shape == (1, 3072), f"Full shape: {full.shape}"
    assert cdr.shape == (1, 3072), f"CDR shape: {cdr.shape}"
    assert not np.allclose(full, cdr), "Full and CDR should differ"
    print(f"  embed_esm3:     {full.shape} ({t1-t0:.1f}s)")
    print(f"  embed_esm3_cdr: {cdr.shape} ({t2-t1:.1f}s)")
    print()


def test_saprot_protein():
    """Test SaProt (ProstT5 → SaProt) on protein sequences."""
    print("=" * 60)
    print("SaProt — Protein embeddings (ProstT5 3Di + SaProt, 1280D)")
    print("=" * 60)

    from kirby.representations.protein import (
        create_saprot_protein_embeddings, _predict_3di_with_prostt5,
    )

    seqs = SYNTH_PROTEINS[:5]  # just 5 for speed
    print(f"  {len(seqs)} sequences")

    # Step 1: ProstT5 3Di prediction
    t0 = time.time()
    threedi = _predict_3di_with_prostt5(seqs, device='cpu')
    t1 = time.time()
    print(f"  ProstT5 3Di: {len(threedi)} seqs, lengths={[len(t) for t in threedi]} ({t1-t0:.1f}s)")
    for i, (s, d) in enumerate(zip(seqs[:2], threedi[:2])):
        print(f"    seq[{i}]: {s[:20]}... → 3Di: {d[:20]}...")

    # Step 2: Full SaProt pipeline
    t0 = time.time()
    X = create_saprot_protein_embeddings(seqs, device='cpu')
    elapsed = time.time() - t0

    assert X.shape == (5, 1280), f"Bad shape: {X.shape}"
    assert not np.isnan(X).any(), "NaN in embeddings"
    assert not np.all(X == 0), "All-zero embeddings"
    print(f"  Shape: {X.shape}, dtype: {X.dtype}, time: {elapsed:.1f}s")
    print()


def test_saprot_antibody():
    """Test SaProt antibody embeddings (full + CDR)."""
    print("=" * 60)
    print("SaProt — Antibody embeddings (2560D)")
    print("=" * 60)

    from kirby.representations.antibody import embed_saprot, embed_saprot_cdr

    t0 = time.time()
    full = embed_saprot([HEAVY_SEQ], [LIGHT_SEQ])
    t1 = time.time()
    cdr = embed_saprot_cdr([HEAVY_SEQ], [LIGHT_SEQ])
    t2 = time.time()

    assert full.shape == (1, 2560), f"Full shape: {full.shape}"
    assert cdr.shape == (1, 2560), f"CDR shape: {cdr.shape}"
    assert not np.allclose(full, cdr), "Full and CDR should differ"
    print(f"  embed_saprot:     {full.shape} ({t1-t0:.1f}s)")
    print(f"  embed_saprot_cdr: {cdr.shape} ({t2-t1:.1f}s)")
    print()


def test_boltz2_protein():
    """Test Boltz-2 on a short protein sequence."""
    print("=" * 60)
    print("Boltz-2 — Protein embeddings (384D, CLI wrapper)")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available, Boltz-2 may fail")

    from kirby.representations.protein import create_boltz2_protein_embeddings

    seqs = [SYNTH_PROTEINS[0]]  # single seq (Boltz is slow)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}, {len(seqs)} sequence")

    t0 = time.time()
    X = create_boltz2_protein_embeddings(seqs, device=device)
    elapsed = time.time() - t0

    assert X.shape == (1, 384), f"Bad shape: {X.shape}"
    assert not np.isnan(X).any(), "NaN in embeddings"
    print(f"  Shape: {X.shape}, dtype: {X.dtype}, time: {elapsed:.1f}s")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 4B smoke test")
    parser.add_argument('--skip-boltz', action='store_true',
                        help='Skip Boltz-2 (slow, needs CUDA)')
    parser.add_argument('--skip-esm3', action='store_true',
                        help='Skip ESM3 (needs esm package)')
    parser.add_argument('--skip-saprot', action='store_true',
                        help='Skip SaProt (needs transformers)')
    args = parser.parse_args()

    print(f"Python {sys.version}")
    try:
        import torch
        print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch: not installed")
    print()

    n_pass = 0
    n_fail = 0
    tests = []

    if not args.skip_esm3:
        tests.append(("ESM3 protein", test_esm3_protein))
        tests.append(("ESM3 antibody", test_esm3_antibody))
    if not args.skip_saprot:
        tests.append(("SaProt protein", test_saprot_protein))
        tests.append(("SaProt antibody", test_saprot_antibody))
    if not args.skip_boltz:
        tests.append(("Boltz-2 protein", test_boltz2_protein))

    for name, fn in tests:
        try:
            fn()
            n_pass += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            n_fail += 1

    print("=" * 60)
    print(f"Phase 4B: {n_pass} passed, {n_fail} failed, "
          f"{len(tests)} total")
    print("=" * 60)

    sys.exit(1 if n_fail > 0 else 0)
