#!/usr/bin/env python3
"""
Worker script: generate a single molecular representation and save to cache.

Called by precache_molecular_reps.py in an isolated subprocess so that
SIGBUS/SIGSEGV/hangs in one rep can't take down the orchestrator.

Usage:
    python scripts/_generate_single_rep.py <dataset> <rep_name> [--gpu]
"""

import os
import sys
import time
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Block stale system bottleneck (Anaconda 2022.05)
sys.modules.setdefault('bottleneck', None)

from pathlib import Path

# Add src/ and tests/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    parser = argparse.ArgumentParser(description="Generate one rep for one dataset")
    parser.add_argument("dataset", help="Dataset name (logd, caco2, herg, qm9)")
    parser.add_argument("rep_name", help="Representation name (e.g. chemberta)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # Force CPU if --gpu not passed
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    from find_best_hybrid_molecular import load_dataset, _rep_cache_path

    # --- Load dataset ---
    print(f"[worker] Loading dataset '{args.dataset}'...")
    smiles, labels, task_type = load_dataset(args.dataset)
    print(f"[worker] {len(smiles)} molecules loaded")

    # --- Check cache ---
    cache_file = _rep_cache_path(smiles, args.rep_name)
    if cache_file.exists():
        arr = np.load(cache_file)
        print(f"[worker] CACHED: {args.rep_name} ({arr.shape[1]}D) at {cache_file}")
        return

    # --- Import the rep function ---
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_pdv, create_mol2vec,
        create_chemberta, create_molformer, create_selformer,
        create_chembert, create_molclr, create_graphmvp, create_smited,
        create_mhg_gnn, create_grover, create_graph_kernel,
        create_schnet, create_chemformer
    )

    rep_funcs = {
        'ecfp4': create_ecfp4,
        'maccs': create_maccs,
        'pdv': create_pdv,
        'mol2vec': create_mol2vec,
        'chemberta': create_chemberta,
        'molformer': create_molformer,
        'selformer': create_selformer,
        'chembert': create_chembert,
        'molclr': create_molclr,
        'graphmvp': create_graphmvp,
        'smited': create_smited,
        'mhg_gnn': create_mhg_gnn,
        'grover': create_grover,
        'graph_kernel': create_graph_kernel,
        'schnet': create_schnet,
        'chemformer': create_chemformer,
    }

    if args.rep_name not in rep_funcs:
        print(f"[worker] ERROR: Unknown rep '{args.rep_name}'", file=sys.stderr)
        sys.exit(1)

    # --- Generate ---
    print(f"[worker] Generating {args.rep_name}...")
    t0 = time.time()
    rep = rep_funcs[args.rep_name](smiles)
    elapsed = time.time() - t0
    print(f"[worker] Generated {args.rep_name}: {rep.shape} in {elapsed:.1f}s")

    # --- Save ---
    np.save(cache_file, rep)
    print(f"[worker] Saved to {cache_file}")


if __name__ == '__main__':
    main()
