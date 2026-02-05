#!/usr/bin/env python3
"""
Verify pretrained model setup for KIRBy.
Run: python scripts/verify_models.py
"""

import os
import sys

def check_env():
    """Check environment setup."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    kirby_dir = os.environ.get('KIRBY_MODELS_DIR')
    print(f"KIRBY_MODELS_DIR: {kirby_dir or '(not set)'}")

    # Check common locations
    locations = [
        '/Volumes/seagate/kirby_models',
        os.path.expanduser('~/kirby_models'),
        os.path.expanduser('~/repos'),
    ]

    print("\nSearching for model directories...")
    for loc in locations:
        if os.path.exists(loc):
            print(f"  [OK] {loc}")
        else:
            print(f"  [--] {loc}")

    return kirby_dir


def check_file_exists(paths, name="file"):
    """Check if any of the paths exist."""
    for path in paths:
        path = os.path.expanduser(path)
        if os.path.exists(path):
            return path
    return None


def main():
    print("\n" + "=" * 60)
    print("KIRBY PRETRAINED MODELS VERIFICATION")
    print("=" * 60 + "\n")

    check_env()

    print("\n" + "=" * 60)
    print("MANUAL SETUP MODELS (repos + weights)")
    print("=" * 60)

    # GROVER
    grover_repo = check_file_exists([
        '~/repos/grover',
        '~/kirby_models/grover',
        '/Volumes/seagate/kirby_models/grover',
    ])
    grover_weights = check_file_exists([
        '~/repos/grover/grover_large.pt',
        '~/repos/grover/grover_base.pt',
        '~/kirby_models/grover/grover_large.pt',
    ]) if grover_repo else None

    print(f"\nGROVER:")
    if grover_weights:
        print(f"  Status: READY")
        print(f"  Repo:   {grover_repo}")
        print(f"  Weight: {grover_weights}")
    elif grover_repo:
        print(f"  Status: WEIGHTS MISSING")
        print(f"  Repo:   {grover_repo}")
        print(f"  Action: Download grover_large.pt from OneDrive")
    else:
        print(f"  Status: NOT INSTALLED")
        print(f"  Action: git clone https://github.com/tencent-ailab/grover.git")

    # Chemformer
    chemformer_repo = check_file_exists([
        '~/repos/Chemformer',
        '~/kirby_models/Chemformer',
    ])
    chemformer_weights = check_file_exists([
        '~/repos/Chemformer/chemformer_pretrained.ckpt',
        '~/repos/Chemformer/step=1000000.ckpt',
        '~/kirby_models/Chemformer/chemformer_pretrained.ckpt',
    ]) if chemformer_repo else None

    print(f"\nChemformer:")
    if chemformer_weights:
        print(f"  Status: READY")
        print(f"  Repo:   {chemformer_repo}")
        print(f"  Weight: {chemformer_weights}")
    elif chemformer_repo:
        print(f"  Status: WEIGHTS MISSING")
        print(f"  Repo:   {chemformer_repo}")
        print(f"  Action: Download from az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq")
    else:
        print(f"  Status: NOT INSTALLED")
        print(f"  Action: git clone https://github.com/MolecularAI/Chemformer.git")

    # mol2vec
    mol2vec_weights = check_file_exists([
        '~/kirby_models/model_300dim.pkl',
        '/Volumes/seagate/kirby_models/model_300dim.pkl',
    ])

    print(f"\nmol2vec:")
    if mol2vec_weights:
        print(f"  Status: READY")
        print(f"  Weight: {mol2vec_weights}")
    else:
        print(f"  Status: WEIGHTS MISSING")
        print(f"  Action: wget https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl")

    # MolCLR
    molclr_repo = check_file_exists([
        '~/repos/MolCLR',
        '~/kirby_models/MolCLR',
    ])
    molclr_weights = check_file_exists([
        '~/repos/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth',
    ]) if molclr_repo else None

    print(f"\nMolCLR:")
    if molclr_weights:
        print(f"  Status: READY (but may crash due to torch-scatter)")
        print(f"  Repo:   {molclr_repo}")
        print(f"  Weight: {molclr_weights}")
    elif molclr_repo:
        print(f"  Status: WEIGHTS MISSING")
        print(f"  Repo:   {molclr_repo}")
    else:
        print(f"  Status: NOT INSTALLED (optional)")

    # GraphMVP
    graphmvp_repo = check_file_exists([
        '~/repos/GraphMVP',
        '~/kirby_models/GraphMVP',
    ])
    graphmvp_weights = check_file_exists([
        '~/repos/GraphMVP/MoleculeSTM_weights/pretrained_GraphMVP/GraphMVP_C/model.pth',
        '~/repos/GraphMVP/pretrained_models/GraphMVP.pth',
    ]) if graphmvp_repo else None

    print(f"\nGraphMVP:")
    if graphmvp_weights:
        print(f"  Status: READY (but may crash due to torch-scatter)")
        print(f"  Repo:   {graphmvp_repo}")
        print(f"  Weight: {graphmvp_weights}")
    elif graphmvp_repo:
        print(f"  Status: WEIGHTS MISSING")
        print(f"  Repo:   {graphmvp_repo}")
    else:
        print(f"  Status: NOT INSTALLED (optional)")

    print("\n" + "=" * 60)
    print("IMPORT TESTS")
    print("=" * 60)

    test_smiles = ['CCO', 'c1ccccc1']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Test fingerprints
    print("\n--- Fingerprints (no deps) ---")
    for name, func_name in [('ECFP4', 'create_ecfp4'), ('MACCS', 'create_maccs'), ('PDV', 'create_pdv')]:
        try:
            from kirby.representations import molecular
            func = getattr(molecular, func_name)
            emb = func(test_smiles)
            print(f"  {name}: OK ({emb.shape})")
        except Exception as e:
            print(f"  {name}: FAILED - {str(e)[:50]}...")

    # Test HuggingFace models
    print("\n--- HuggingFace Transformers (auto-download) ---")
    for name, func_name in [
        ('ChemBERTa', 'create_chemberta'),
        ('MolFormer', 'create_molformer'),
        ('SELFormer', 'create_selformer'),
        ('ChemBERT', 'create_chembert'),
    ]:
        try:
            from kirby.representations import molecular
            func = getattr(molecular, func_name)
            emb = func(test_smiles)
            print(f"  {name}: OK ({emb.shape})")
        except Exception as e:
            print(f"  {name}: FAILED - {str(e)[:50]}...")

    # Test other auto-download
    print("\n--- Other Auto-download ---")
    for name, func_name in [
        ('Uni-Mol', 'create_unimol'),
        ('MHG-GNN', 'create_mhg_gnn'),
        ('SMI-TED', 'create_smited'),
        ('SchNet', 'create_schnet'),
    ]:
        try:
            from kirby.representations import molecular
            func = getattr(molecular, func_name)
            emb = func(test_smiles)
            print(f"  {name}: OK ({emb.shape})")
        except Exception as e:
            print(f"  {name}: FAILED - {str(e)[:50]}...")

    # Test manual setup models
    print("\n--- Manual Setup Models ---")
    for name, func_name in [
        ('mol2vec', 'create_mol2vec'),
        ('GROVER', 'create_grover'),
        ('Chemformer', 'create_chemformer'),
        ('MolCLR', 'create_molclr'),
        ('GraphMVP', 'create_graphmvp'),
    ]:
        try:
            from kirby.representations import molecular
            func = getattr(molecular, func_name)
            emb = func(test_smiles)
            print(f"  {name}: OK ({emb.shape})")
        except Exception as e:
            err_str = str(e)[:60].replace('\n', ' ')
            print(f"  {name}: FAILED - {err_str}...")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
What works without any setup:
  - Fingerprints (ECFP4, MACCS, PDV)
  - HuggingFace models (ChemBERTa, MolFormer, SELFormer, ChemBERT)
  - Uni-Mol, MHG-GNN, SMI-TED, SchNet (auto-download)

What needs manual download:
  - GROVER: Download grover_large.pt from OneDrive
  - Chemformer: Download checkpoint from AZ Box
  - mol2vec: wget model_300dim.pkl

What has torch-scatter issues (macOS):
  - MolCLR, GraphMVP - may segfault due to version mismatch
    Fix: pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-X.X.X+cpu.html
""")


if __name__ == '__main__':
    main()
