#!/usr/bin/env python3
"""
Extract pairwise Spearman correlation matrix between importance methods.
Saves the full matrix to JSON for both LogD and QM9 datasets.

Each method is run in a subprocess to isolate crashes (segfaults etc).
"""

import json
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_importance_aggregation import (
    ALL_IMPORTANCE_METHODS,
    load_logd_dataset,
    load_qm9_homolumo,
    filter_valid_smiles,
    generate_representations,
)


WORKER_SCRIPT = Path(__file__).parent / '_importance_worker.py'


def run_single_method_subprocess(X_path, y_path, method, output_path, timeout=600):
    """Run a single importance method in a subprocess for crash isolation."""
    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        str(X_path), str(y_path), method, str(output_path)
    ]
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            scores = np.load(output_path)
            return scores, True
        else:
            stderr = result.stderr.strip()[-200:] if result.stderr else "no stderr"
            print(f"[rc={result.returncode}] {stderr}")
            return None, False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({timeout}s)")
        return None, False
    except Exception as e:
        print(f"ERROR: {e}")
        return None, False


def compute_correlation_matrix(dataset_name, X, y, methods, timeout_per_method=600):
    """Compute all importance vectors and the full pairwise Spearman matrix."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"{'='*60}")

    # Save X and y to temp files for subprocess
    tmp_dir = Path('/tmp/kirby_corr_matrix')
    tmp_dir.mkdir(exist_ok=True)
    X_path = tmp_dir / f'{dataset_name}_X.npy'
    y_path = tmp_dir / f'{dataset_name}_y.npy'
    np.save(X_path, X)
    np.save(y_path, y)

    importance_vectors = {}
    timings = {}

    for method in methods:
        print(f"  {method}...", end=' ', flush=True)
        output_path = tmp_dir / f'{dataset_name}_{method}_scores.npy'
        if output_path.exists():
            output_path.unlink()

        start = time.time()
        scores, success = run_single_method_subprocess(
            X_path, y_path, method, output_path, timeout=timeout_per_method
        )
        elapsed = time.time() - start

        if success and scores is not None and np.std(scores) > 0:
            importance_vectors[method] = scores
            timings[method] = elapsed
            print(f"{elapsed:.1f}s OK")
        elif success:
            print(f"FAILED (constant scores) {elapsed:.1f}s")
        # else: already printed error

    # Phase 2: Compute pairwise Spearman correlations
    successful_methods = list(importance_vectors.keys())
    n = len(successful_methods)
    print(f"\n  {n}/{len(methods)} methods succeeded")

    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = successful_methods[i], successful_methods[j]
            corr, _ = stats.spearmanr(importance_vectors[m1], importance_vectors[m2])
            matrix[i, j] = corr
            matrix[j, i] = corr

    return {
        'methods': successful_methods,
        'matrix': matrix.tolist(),
        'timings': timings,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
    }


def main():
    output_path = Path(__file__).parent / 'results' / 'importance_benchmark' / 'correlation_matrix.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the worker script
    worker_code = '''#!/usr/bin/env python3
"""Worker: compute single importance method. Called by extract_correlation_matrix.py."""
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def main():
    X_path, y_path, method, output_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    X = np.load(X_path)
    y = np.load(y_path)

    from kirby.hybrid import compute_feature_importance
    rep_dict = {'features': X}
    scores = compute_feature_importance(rep_dict, y, method=method)
    np.save(output_path, scores['features'])

if __name__ == '__main__':
    main()
'''
    WORKER_SCRIPT.write_text(worker_code)

    results = {}

    # Use same reps as full benchmark: ecfp4, maccs, pdv (sorted)
    reps = ['ecfp4', 'maccs', 'pdv']

    # --- LogD ---
    print("\nLoading LogD dataset...")
    smiles, labels = load_logd_dataset()
    smiles, labels = filter_valid_smiles(smiles, labels)
    base_reps = generate_representations(smiles, reps)
    X = np.hstack([base_reps[r] for r in sorted(base_reps.keys())])
    y = labels
    print(f"  Feature matrix: {X.shape}")

    methods_to_run = list(ALL_IMPORTANCE_METHODS)
    results['logd'] = compute_correlation_matrix(
        'logd', X, y, methods_to_run, timeout_per_method=600
    )

    # --- QM9 ---
    print("\nLoading QM9 dataset...")
    smiles_q, labels_q = load_qm9_homolumo(n_samples=5000)
    smiles_q, labels_q = filter_valid_smiles(smiles_q, labels_q)
    base_reps_q = generate_representations(smiles_q, reps)
    X_q = np.hstack([base_reps_q[r] for r in sorted(base_reps_q.keys())])
    y_q = labels_q
    print(f"  Feature matrix: {X_q.shape}")

    results['qm9'] = compute_correlation_matrix(
        'qm9', X_q, y_q, methods_to_run, timeout_per_method=600
    )

    # Save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Print tables
    for ds_name, ds_data in results.items():
        methods = ds_data['methods']
        matrix = np.array(ds_data['matrix'])
        n = len(methods)

        print(f"\n{'='*60}")
        print(f"  PAIRWISE SPEARMAN CORRELATIONS - {ds_name.upper()}")
        print(f"  ({n} methods)")
        print(f"{'='*60}")

        # Print all pairs sorted by correlation
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((methods[i], methods[j], matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  {'Method 1':<25} {'Method 2':<25} {'Spearman r':>10}")
        print(f"  {'-'*25} {'-'*25} {'-'*10}")
        for m1, m2, corr in pairs:
            print(f"  {m1:<25} {m2:<25} {corr:>10.4f}")

    # Cleanup worker
    if WORKER_SCRIPT.exists():
        WORKER_SCRIPT.unlink()


if __name__ == '__main__':
    main()
