#!/usr/bin/env python3
"""
Molecular Graph & Structural Augmentation Survey

Tests 8 new molecular graph/structural representations as both standalone
baselines and augmentations on top of existing base reps. Mirrors the
structural_geometry_survey.py pattern for antibodies.

New reps tested:
  atom_pair (2048D)         — Atom-pair fingerprints
  top_torsion (2048D)       — Topological torsion fingerprints
  coulomb_matrix (50D)      — Coulomb matrix eigenvalue spectrum
  usr (12D)                 — Ultrafast Shape Recognition
  usrcat (60D)              — USRCAT (USR + pharmacophore channels)
  pharmacophore3d (49D)     — 3D pharmacophore pair distances
  tda (28D)                 — Persistent homology / TDA
  e3fp (1024D)              — Extended 3D fingerprint
  all_graph (~5283D)        — All 8 concatenated

For each dataset, tests:
  1. Each graph rep standalone (vs Ridge/RF baseline)
  2. Base reps standalone (for delta computation)
  3. Each graph rep augmented onto each base rep
  4. Delta tracking: base RMSE vs augmented RMSE

Usage:
    python tests/experiments/molecular_graph_survey.py logd
    python tests/experiments/molecular_graph_survey.py herg --base-reps ecfp4,mhg_gnn,grover
    python tests/experiments/molecular_graph_survey.py all --models ridge,rf
    python tests/experiments/molecular_graph_survey.py all --output tests/results/molecular_graph_survey/
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# =============================================================================
# IMPORTS FROM EXISTING SURVEYS
# =============================================================================

from find_best_hybrid_molecular import (
    load_dataset, assign_scaffold_groups, _rep_cache_path,
)

from run_baseline_survey import evaluate_rep


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ['herg_ki', 'logd', 'qm9']

# New graph/structural reps to test
GRAPH_REP_NAMES = [
    'atom_pair', 'top_torsion', 'coulomb_matrix',
    'usr', 'usrcat', 'pharmacophore3d', 'tda', 'e3fp',
]

# Default base reps to test augmentation against
DEFAULT_BASE_REPS = [
    'ecfp4', 'mhg_gnn', 'grover', 'mordred', 'chemberta',
]


# =============================================================================
# REP GENERATION / LOADING
# =============================================================================

def _get_graph_rep_funcs():
    """Import and return graph rep generation functions."""
    from kirby.representations.molecular import (
        create_atom_pair, create_top_torsion,
        create_coulomb_matrix, create_usr, create_usrcat,
        create_pharmacophore3d, create_tda, create_e3fp,
    )
    return {
        'atom_pair': create_atom_pair,
        'top_torsion': create_top_torsion,
        'coulomb_matrix': create_coulomb_matrix,
        'usr': create_usr,
        'usrcat': create_usrcat,
        'pharmacophore3d': create_pharmacophore3d,
        'tda': create_tda,
        'e3fp': create_e3fp,
    }


def _load_or_compute_graph_reps(smiles: List[str]) -> Dict[str, np.ndarray]:
    """Load graph reps from cache or compute them. Also builds all_graph."""
    rep_funcs = _get_graph_rep_funcs()
    reps = {}

    for rep_name in GRAPH_REP_NAMES:
        # Try cache
        cache_file = _rep_cache_path(smiles, rep_name)
        if cache_file.exists():
            try:
                X = np.load(cache_file)
                if X.ndim == 2 and X.shape[0] == len(smiles):
                    print(f"  {rep_name}: loaded from cache ({X.shape[1]}D)")
                    reps[rep_name] = X
                    continue
            except Exception:
                pass

        # Compute
        try:
            print(f"  Generating {rep_name}...", end=' ', flush=True)
            t0 = time.time()
            X = rep_funcs[rep_name](smiles)
            elapsed = time.time() - t0
            print(f"done ({X.shape[1]}D, {elapsed:.1f}s)")
            reps[rep_name] = X
            np.save(cache_file, X)
        except Exception as e:
            print(f"FAILED ({e})")

    # Build all_graph (concatenation of all available graph reps)
    available = [reps[n] for n in GRAPH_REP_NAMES if n in reps]
    if available:
        reps['all_graph'] = np.hstack(available)
        print(f"  all_graph: {reps['all_graph'].shape[1]}D "
              f"(concat of {len(available)} reps)")

    return reps


def _load_base_reps(smiles: List[str], base_rep_names: List[str]) -> Dict[str, np.ndarray]:
    """Load base representations from cache. Skip any not found."""
    reps = {}
    for rep_name in base_rep_names:
        cache_file = _rep_cache_path(smiles, rep_name)
        if cache_file.exists():
            try:
                X = np.load(cache_file)
                if X.ndim == 2 and X.shape[0] == len(smiles):
                    reps[rep_name] = X
                    continue
            except Exception:
                pass
        print(f"    WARNING: base rep '{rep_name}' not cached, skipping")
    return reps


# =============================================================================
# MODELS
# =============================================================================

def get_models(model_names: List[str], n_samples: int) -> dict:
    """Get model instances for evaluation (regression only — all molecular)."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    models = {}
    for name in model_names:
        if name == 'ridge':
            models['ridge'] = Ridge(alpha=1.0)
        elif name == 'rf':
            n_est = 100 if n_samples < 5000 else 200
            models['rf'] = RandomForestRegressor(
                n_estimators=n_est, max_depth=None,
                n_jobs=-1, random_state=42)
        elif name == 'xgboost':
            try:
                from xgboost import XGBRegressor
                models['xgboost'] = XGBRegressor(
                    n_estimators=200, max_depth=6,
                    learning_rate=0.1, n_jobs=-1,
                    random_state=42, verbosity=0)
            except ImportError:
                print("  WARNING: xgboost not installed, skipping")
    return models


def _eval_and_format(X, labels, groups, model, n_folds, task_type):
    """Evaluate a rep and return a flat dict with _mean/_std keys."""
    mean_m, std_m = evaluate_rep(X, labels, groups, model, n_folds, task_type)
    result = {}
    result.update({f'{k}_mean': v for k, v in mean_m.items()})
    result.update({f'{k}_std': v for k, v in std_m.items()})
    return result


# =============================================================================
# SURVEY CORE
# =============================================================================

def run_graph_survey(
    dataset_name: str,
    base_rep_names: List[str],
    model_names: List[str],
    n_folds: int = 5,
) -> dict:
    """Run graph/structural rep augmentation survey for one dataset."""
    print(f"\n{'='*70}")
    print(f"MOLECULAR GRAPH SURVEY: {dataset_name}")
    print(f"{'='*70}")
    t0 = time.time()

    # 1. Load data
    smiles, labels, task_type = load_dataset(dataset_name)
    n_samples = len(smiles)
    groups, n_groups = assign_scaffold_groups(smiles)
    print(f"  {n_samples} molecules, {n_groups} scaffolds, task={task_type}")

    # 2. Compute/load graph reps
    graph_reps = _load_or_compute_graph_reps(smiles)

    # 3. Load base reps from cache
    base_reps = _load_base_reps(smiles, base_rep_names)
    print(f"  Graph reps: {sorted(graph_reps.keys())}")
    print(f"  Base reps loaded: {sorted(base_reps.keys())}")

    if not graph_reps:
        print("  ERROR: No graph reps available")
        return {'error': 'no graph reps'}

    # 4. Get models
    models = get_models(model_names, n_samples)
    print(f"  Models: {list(models.keys())}")

    # 5. Evaluate graph reps standalone
    print(f"\n--- Standalone graph rep baselines ---")
    graph_baselines = {}
    graph_names = GRAPH_REP_NAMES + ['all_graph']

    for rep_name in graph_names:
        X = graph_reps.get(rep_name)
        if X is None or np.all(X == 0):
            print(f"  {rep_name}: SKIPPED (missing or all zeros)")
            continue

        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        dim = X.shape[1]
        print(f"  {rep_name} ({dim}D):", end='', flush=True)

        entry = {'dimension': dim, 'scores': {}}
        for mname, model in models.items():
            try:
                result = _eval_and_format(X, labels, groups, model,
                                          n_folds, task_type)
                entry['scores'][mname] = result
                rmse = result.get('rmse_mean', float('nan'))
                rho = result.get('spearman_mean', float('nan'))
                print(f" {mname}=RMSE:{rmse:.3f}/ρ:{rho:.3f}",
                      end='', flush=True)
            except Exception:
                print(f" {mname}=FAIL", end='', flush=True)
        graph_baselines[rep_name] = entry
        print()

    # 6. Evaluate base reps standalone (for delta computation)
    print(f"\n--- Base rep standalone scores ---")
    base_scores = {}

    for base_name in sorted(base_reps.keys()):
        X_base = np.asarray(base_reps[base_name], dtype=np.float64)
        X_base = np.nan_to_num(X_base, nan=0.0, posinf=1e10, neginf=-1e10)
        dim = X_base.shape[1]
        print(f"  {base_name} ({dim}D):", end='', flush=True)

        entry = {'dimension': dim, 'scores': {}}
        for mname, model in models.items():
            try:
                result = _eval_and_format(X_base, labels, groups, model,
                                          n_folds, task_type)
                entry['scores'][mname] = result
                rmse = result.get('rmse_mean', float('nan'))
                rho = result.get('spearman_mean', float('nan'))
                print(f" {mname}=RMSE:{rmse:.3f}/ρ:{rho:.3f}",
                      end='', flush=True)
            except Exception:
                print(f" {mname}=FAIL", end='', flush=True)
        base_scores[base_name] = entry
        print()

    # 7. Augmentation tests: base + graph
    print(f"\n--- Augmentation tests ---")
    augmentations = {}

    for base_name in sorted(base_reps.keys()):
        X_base = np.asarray(base_reps[base_name], dtype=np.float64)
        X_base = np.nan_to_num(X_base, nan=0.0, posinf=1e10, neginf=-1e10)
        base_dim = X_base.shape[1]

        for rep_name in graph_names:
            X_graph = graph_reps.get(rep_name)
            if X_graph is None or np.all(X_graph == 0):
                continue

            X_graph = np.asarray(X_graph, dtype=np.float64)
            X_graph = np.nan_to_num(X_graph, nan=0.0, posinf=1e10, neginf=-1e10)

            X_aug = np.hstack([X_base, X_graph])
            aug_key = f"{base_name}+{rep_name}"
            aug_dim = X_aug.shape[1]

            print(f"  {aug_key} ({aug_dim}D):", end='', flush=True)

            entry = {
                'base': base_name,
                'graph_rep': rep_name,
                'base_dim': base_dim,
                'graph_dim': X_graph.shape[1],
                'concat_dim': aug_dim,
                'scores': {},
            }

            for mname, model in models.items():
                try:
                    result = _eval_and_format(X_aug, labels, groups, model,
                                              n_folds, task_type)
                    entry['scores'][mname] = result
                    rmse = result.get('rmse_mean', float('nan'))
                    rho = result.get('spearman_mean', float('nan'))
                    print(f" {mname}=RMSE:{rmse:.3f}/ρ:{rho:.3f}",
                          end='', flush=True)
                except Exception:
                    print(f" {mname}=FAIL", end='', flush=True)

            augmentations[aug_key] = entry
            print()

    elapsed = time.time() - t0

    return {
        'metadata': {
            'dataset': dataset_name,
            'n_samples': n_samples,
            'n_groups': n_groups,
            'n_folds': n_folds,
            'task_type': task_type,
            'base_reps': sorted(base_reps.keys()),
            'graph_reps': sorted(graph_reps.keys()),
            'models': list(models.keys()),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'graph_baselines': graph_baselines,
        'base_scores': base_scores,
        'augmentations': augmentations,
    }


# =============================================================================
# SUMMARY
# =============================================================================

def print_augmentation_summary(result: dict):
    """Print augmentation delta summary (RMSE ↓ is better)."""
    augmentations = result.get('augmentations', {})
    base_scores = result.get('base_scores', {})

    if not augmentations:
        return

    print(f"\n{'='*70}")
    print(f"AUGMENTATION DELTA SUMMARY (ΔRMSE, negative = improvement)")
    print(f"{'='*70}")

    # Group by base rep
    bases = {}
    for key, entry in augmentations.items():
        base = entry['base']
        if base not in bases:
            bases[base] = {}
        bases[base][entry['graph_rep']] = entry

    for base_name in sorted(bases.keys()):
        # Get best base RMSE per model
        base_entry = base_scores.get(base_name, {})
        base_model_rmse = {}
        for mname, s in base_entry.get('scores', {}).items():
            base_model_rmse[mname] = s.get('rmse_mean', float('inf'))

        # Find overall best base RMSE
        best_base_rmse = min(base_model_rmse.values()) if base_model_rmse else float('inf')
        best_base_model = min(base_model_rmse, key=base_model_rmse.get) if base_model_rmse else '?'
        print(f"\n  Base: {base_name} (RMSE={best_base_rmse:.4f}, {best_base_model})")

        for rep_name, entry in sorted(bases[base_name].items()):
            scores = entry.get('scores', {})
            # Find best augmented RMSE
            best_aug_rmse = float('inf')
            best_aug_model = None
            for mname, s in scores.items():
                rmse = s.get('rmse_mean', float('inf'))
                if rmse < best_aug_rmse:
                    best_aug_rmse = rmse
                    best_aug_model = mname

            if best_aug_model:
                # Delta: compare augmented vs base with same model
                base_rmse_same_model = base_model_rmse.get(best_aug_model, float('inf'))
                delta = best_aug_rmse - base_rmse_same_model
                marker = '✓' if delta < 0 else ' '
                print(f"    {marker} +{rep_name:<22} RMSE={best_aug_rmse:.4f} "
                      f"Δ={delta:+.4f} ({best_aug_model})")


def _convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Molecular graph/structural representation augmentation survey.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests 8 graph/structural molecular representations as standalone baselines
and augmentations on top of existing base reps. Reports RMSE deltas.

New reps:
  atom_pair (2048D)         top_torsion (2048D)
  coulomb_matrix (50D)      usr (12D)
  usrcat (60D)              pharmacophore3d (49D)
  tda (28D)                 e3fp (1024D)
  all_graph (~5283D)        all 8 concatenated

Examples:
  %(prog)s logd --models ridge,rf
  %(prog)s herg --base-reps ecfp4,mhg_gnn,grover
  %(prog)s all --output tests/results/molecular_graph_survey/
        """
    )

    parser.add_argument('dataset', type=str,
                        help='Dataset name (herg, logd, qm9) or "all"')
    parser.add_argument('--base-reps', type=str, default=None,
                        help=f'Comma-separated base reps '
                             f'(default: {",".join(DEFAULT_BASE_REPS)})')
    parser.add_argument('--models', type=str, default='ridge,rf',
                        help='Comma-separated model names (default: ridge,rf)')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for JSON results')

    args = parser.parse_args()

    # Resolve datasets
    if args.dataset == 'all':
        datasets = list(DATASETS)
    elif args.dataset in DATASETS:
        datasets = [args.dataset]
    else:
        datasets = [d.strip() for d in args.dataset.split(',')]

    # Resolve base reps
    if args.base_reps:
        base_rep_names = [r.strip() for r in args.base_reps.split(',')]
    else:
        base_rep_names = list(DEFAULT_BASE_REPS)

    model_names = [m.strip() for m in args.models.split(',')]

    print(f"Molecular Graph/Structural Augmentation Survey")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Base reps: {base_rep_names}")
    print(f"  Models: {model_names}")
    print(f"  Folds: {args.n_folds}")

    # Output
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run
    all_results = {}
    pass_count = 0
    fail_count = 0

    for ds in datasets:
        try:
            result = run_graph_survey(ds, base_rep_names, model_names,
                                       args.n_folds)
            result = _convert_numpy(result)
            print_augmentation_summary(result)
            all_results[ds] = result
            pass_count += 1

            if output_dir:
                out_path = output_dir / f"molecular_graph_{ds}.json"
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"\n  FAILED: {ds}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Cross-dataset summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DATASET GRAPH REP SUMMARY")
        print(f"{'='*70}")

        for ds, r in all_results.items():
            baselines = r.get('graph_baselines', {})
            print(f"\n  {ds} (standalone best per rep):")
            for rep_name, entry in sorted(baselines.items()):
                scores = entry.get('scores', {})
                best_rmse = min(
                    (s.get('rmse_mean', float('inf')) for s in scores.values()),
                    default=float('nan'))
                best_rho = max(
                    (s.get('spearman_mean', float('-inf')) for s in scores.values()),
                    default=float('nan'))
                dim = entry.get('dimension', '?')
                print(f"    {rep_name:<25} ({dim:>5}D)  RMSE={best_rmse:.4f}  ρ={best_rho:.4f}")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
