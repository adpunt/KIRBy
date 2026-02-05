#!/usr/bin/env python3
"""
Quick verification of pretrained models on server.
Run: srun python scripts/verify_server_models.py
"""

import os
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def check_path(paths):
    """Return first existing path or None."""
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return p
    return None

def test_model(name, func_name, test_smiles=['CCO', 'c1ccccc1']):
    """Test a single model, return (success, shape_or_error)."""
    try:
        from kirby.representations import molecular
        func = getattr(molecular, func_name)
        emb = func(test_smiles)
        return True, emb.shape
    except Exception as e:
        return False, str(e)[:80]

def main():
    print("=" * 60)
    print("KIRBY SERVER MODEL VERIFICATION")
    print("=" * 60)

    # Environment
    print(f"\nKIRBY_MODELS_DIR: {os.environ.get('KIRBY_MODELS_DIR', '(not set)')}")
    print(f"Python: {sys.executable}")

    # Check paths
    print("\n--- Path Checks ---")
    checks = [
        ("GROVER repo", ['~/kirby_models/grover', '~/repos/grover']),
        ("GROVER weights", ['~/kirby_models/grover/grover_large.pt', '~/repos/grover/grover_large.pt']),
        ("Chemformer repo", ['~/kirby_models/Chemformer', '~/repos/Chemformer']),
        ("Chemformer weights", [
            '~/kirby_models/Chemformer/models/pre-trained/combined/step=1000000.ckpt',
            '~/kirby_models/Chemformer/step=1000000.ckpt',
            '~/repos/Chemformer/models/pre-trained/combined/step=1000000.ckpt',
        ]),
        ("mol2vec weights", ['~/kirby_models/model_300dim.pkl']),
        ("MolCLR repo", ['~/kirby_models/MolCLR', '~/repos/MolCLR']),
        ("MolCLR weights", ['~/kirby_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth']),
        ("GraphMVP repo", ['~/kirby_models/GraphMVP', '~/repos/GraphMVP']),
        ("GraphMVP weights", [
            '~/kirby_models/GraphMVP/MoleculeSTM_weights/pretrained_GraphMVP/GraphMVP_C/model.pth',
            '~/kirby_models/GraphMVP/pretrained_models/GraphMVP.pth',
        ]),
    ]

    for name, paths in checks:
        found = check_path(paths)
        status = f"OK: {found}" if found else "MISSING"
        print(f"  {name}: {status}")

    # Test models
    print("\n--- Model Tests ---")

    models = [
        # Fingerprints (always work)
        ("ECFP4", "create_ecfp4"),
        ("MACCS", "create_maccs"),

        # HuggingFace (auto-download)
        ("ChemBERTa", "create_chemberta"),
        ("MolFormer", "create_molformer"),

        # Other auto-download
        ("Uni-Mol", "create_unimol"),
        ("MHG-GNN", "create_mhg_gnn"),
        ("SMI-TED", "create_smited"),
        ("SchNet", "create_schnet"),

        # Manual setup
        ("mol2vec", "create_mol2vec"),
        ("GROVER", "create_grover"),
        ("Chemformer", "create_chemformer"),

        # Should work on Linux (torch-scatter issues are macOS-specific)
        ("MolCLR", "create_molclr"),
        ("GraphMVP", "create_graphmvp"),
    ]

    results = []
    for name, func_name in models:
        print(f"  Testing {name}...", end=" ", flush=True)
        ok, result = test_model(name, func_name)
        if ok:
            print(f"OK {result}")
        else:
            print(f"FAILED: {result}")
        results.append((name, ok, result))

    # Summary
    print("\n" + "=" * 60)
    working = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]
    print(f"Working: {len(working)}/{len(results)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
