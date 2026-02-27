#!/usr/bin/env python3
"""
Quick evaluation of pretrained molecular embeddings on QM9 (HOMO-LUMO gap).
Purpose: Determine which pretrained embeddings are worth adding to the
QSAR noise robustness study's ANOVA.

Tests each embedding with RF (n=5000, scaffold split, 3 seeds).
Reports R² mean ± std and dimensionality.
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import time
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.datasets.qm9 import load_qm9, get_qm9_splits

# ── Embeddings to test ──────────────────────────────────────────────────
# Auto-download (HuggingFace): no manual setup needed
AUTO_EMBEDDINGS = {
    'molformer':  ('create_molformer',  {}),
    'selformer':  ('create_selformer',  {}),
    'chembert':   ('create_chembert',   {}),
    'chemberta':  ('create_chemberta',  {}),
    'mhg_gnn':    ('create_mhg_gnn',   {}),
    'smited':     ('create_smited',     {}),
}

# Needs specific setup (model file downloads)
MANUAL_EMBEDDINGS = {
    'mol2vec':    ('create_mol2vec',    {}),
    'unimol':     ('create_unimol',     {}),
    'schnet':     ('create_schnet',     {'hidden_channels': 128}),
}

# Baselines for context
BASELINES = {
    'ecfp4':      ('create_ecfp4',     {'n_bits': 2048}),
    'pdv':        ('create_pdv',       {}),
}


def import_creator(func_name):
    """Dynamically import a create_* function from kirby."""
    from kirby.representations import molecular
    return getattr(molecular, func_name)


def run_evaluation(name, creator_name, kwargs, train_smiles, train_labels,
                   test_smiles, test_labels, seeds=[42, 123, 456]):
    """Evaluate a single embedding with RF across multiple seeds."""
    t0 = time.time()

    try:
        creator = import_creator(creator_name)
    except AttributeError:
        return {'name': name, 'error': f'Function {creator_name} not found'}

    try:
        X_train = creator(train_smiles, **kwargs)
        X_test = creator(test_smiles, **kwargs)
    except Exception as e:
        return {'name': name, 'error': f'Embedding generation failed: {e}'}

    embed_time = time.time() - t0

    # Handle failed molecules (NaN rows)
    if hasattr(X_train, 'shape') and len(X_train.shape) == 2:
        dim = X_train.shape[1]
    else:
        return {'name': name, 'error': f'Unexpected shape: {type(X_train)}'}

    # Replace NaN with 0 for RF compatibility
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    r2_scores = []
    for seed in seeds:
        rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        rf.fit(X_train, train_labels)
        pred = rf.predict(X_test)
        r2_scores.append(r2_score(test_labels, pred))

    total_time = time.time() - t0

    return {
        'name': name,
        'dim': dim,
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'r2_scores': r2_scores,
        'embed_time': embed_time,
        'total_time': total_time,
        'error': None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-samples', type=int, default=10000)
    parser.add_argument('--include-manual', action='store_true',
                        help='Include embeddings that need manual model downloads')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Only test specific embeddings (e.g. --only molformer chembert)')
    args = parser.parse_args()

    print("=" * 80)
    print(f"PRETRAINED EMBEDDING EVALUATION - QM9 HOMO-LUMO gap")
    print(f"N = {args.n_samples}, random sample, scaffold split, RF (200 trees, 3 seeds)")
    print("=" * 80)

    # Load data
    print("\nLoading QM9...")
    raw_data = load_qm9(n_samples=args.n_samples, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')

    train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    test_smiles = splits['test']['smiles']
    test_labels = splits['test']['labels']

    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print(f"Target: {raw_data['property_name']}")

    # Build test list
    to_test = dict(BASELINES)
    to_test.update(AUTO_EMBEDDINGS)
    if args.include_manual:
        to_test.update(MANUAL_EMBEDDINGS)

    if args.only:
        to_test = {k: v for k, v in to_test.items() if k in args.only}

    # Run evaluations
    results = []
    for i, (name, (creator, kwargs)) in enumerate(to_test.items(), 1):
        print(f"\n[{i}/{len(to_test)}] {name}...")
        try:
            result = run_evaluation(name, creator, kwargs,
                                    train_smiles, train_labels,
                                    test_smiles, test_labels)
            results.append(result)

            if result['error']:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  dim={result['dim']}, R²={result['r2_mean']:.4f} ± {result['r2_std']:.4f} "
                      f"({result['embed_time']:.1f}s embed, {result['total_time']:.1f}s total)")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({'name': name, 'error': str(e)})

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Embedding':<20} {'Dim':>6} {'R² mean':>10} {'R² std':>10} {'Time (s)':>10} {'Note'}")
    print("-" * 70)

    # Sort by R² descending
    valid = [r for r in results if r.get('error') is None]
    failed = [r for r in results if r.get('error') is not None]
    valid.sort(key=lambda x: x['r2_mean'], reverse=True)

    for r in valid:
        note = ""
        if r['name'] in BASELINES:
            note = "(baseline)"
        print(f"{r['name']:<20} {r['dim']:>6} {r['r2_mean']:>10.4f} {r['r2_std']:>10.4f} "
              f"{r['total_time']:>10.1f} {note}")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  {r['name']}: {r['error']}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR ANOVA")
    print("=" * 80)

    baseline_r2 = {r['name']: r['r2_mean'] for r in valid if r['name'] in BASELINES}
    ecfp4_r2 = baseline_r2.get('ecfp4', 0)

    print(f"\nECFP4 baseline R² = {ecfp4_r2:.4f}")
    print("\nPretrained embeddings ranked by R²:")
    for r in valid:
        if r['name'] not in BASELINES:
            diff = r['r2_mean'] - ecfp4_r2
            verdict = "GOOD" if r['r2_mean'] > 0.5 else "WEAK"
            if r['r2_mean'] > ecfp4_r2:
                verdict = "BETTER THAN ECFP4"
            print(f"  {r['name']:<20} R²={r['r2_mean']:.4f} (Δ={diff:+.4f} vs ECFP4) → {verdict}")

    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    import csv
    csv_path = os.path.join(results_dir, 'pretrained_embedding_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['embedding', 'dim', 'r2_mean', 'r2_std', 'embed_time', 'total_time', 'error'])
        for r in valid:
            writer.writerow([r['name'], r['dim'], f"{r['r2_mean']:.6f}", f"{r['r2_std']:.6f}",
                             f"{r['embed_time']:.1f}", f"{r['total_time']:.1f}", ''])
        for r in failed:
            writer.writerow([r['name'], '', '', '', '', '', r.get('error', 'unknown')])
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
