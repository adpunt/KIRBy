#!/usr/bin/env python3
"""
Structural Geometry Augmentation Survey

Tests 7 structure-based geometry features as augmentations on top of base
antibody representations across all antibody tasks.

Features tested:
  vhvl_angles (6D)          — ABangle-style VH-VL orientation
  cdr_shape (60D)           — USR-like CDR loop shape moments
  cdr_canonical (12D)       — CDR canonical class + geometry
  paratope_charge (12D)     — Paratope electrostatics
  paratope_hydrophobicity (12D) — Paratope hydrophobicity profile
  cdr_contacts (30D)        — Inter-CDR pairwise distances
  vhvl_interface (30D)      — VH-VL interface composition
  all_geometry (162D)       — All 7 concatenated

For each dataset, tests:
  1. Each geometry feature standalone (vs Ridge/RF baseline)
  2. Each geometry feature augmented onto each base rep
  3. All geometry combined augmented onto each base rep

Usage:
    python tests/experiments/structural_geometry_survey.py flab_thermo_jain
    python tests/experiments/structural_geometry_survey.py all_inter
    python tests/experiments/structural_geometry_survey.py all_inter --base-reps esm2,ablang2,prot_t5
    python tests/experiments/structural_geometry_survey.py all --models ridge,rf
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import hashlib
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# =============================================================================
# IMPORTS FROM EXISTING SURVEY
# =============================================================================

from antibody_baseline_survey import (
    DATASET_REGISTRY, DOMAIN_GROUPS, load_dataset,
    get_models, evaluate_rep, _rep_cache_path,
)


# Default base reps to test augmentation against
DEFAULT_BASE_REPS = [
    'esm2', 'ablang2', 'prot_t5', 'esm3', 'mint',
    'saprot', 'antiberty', 'blosum', 'aac',
]


# =============================================================================
# GEOMETRY FEATURE CACHING
# =============================================================================

def _geo_cache_dir(heavy_seqs: List[str], light_seqs: List[str]) -> Path:
    """Get geometry feature cache directory, keyed by sequence hash."""
    content = '\n'.join(heavy_seqs) + '\n\n' + '\n'.join(light_seqs)
    data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    base = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    cache_dir = base / 'reps' / f'ab_{data_hash}' / 'geometry'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_or_compute_geometry(
    heavy_seqs: List[str],
    light_seqs: List[str],
) -> Dict[str, np.ndarray]:
    """Load geometry features from cache or compute them."""
    from kirby.representations.antibody_geometry import (
        compute_all_geometry_features, GEOMETRY_DIMS)

    cache_dir = _geo_cache_dir(heavy_seqs, light_seqs)

    # Try loading all from cache
    cached = {}
    all_cached = True
    for name, dim in GEOMETRY_DIMS.items():
        cache_file = cache_dir / f'{name}.npy'
        if cache_file.exists():
            try:
                X = np.load(cache_file, allow_pickle=False)
                if X.shape == (len(heavy_seqs), dim):
                    cached[name] = X
                    continue
            except Exception:
                pass
        all_cached = False

    if all_cached and len(cached) == len(GEOMETRY_DIMS):
        print(f"  Loaded all geometry features from cache")
        # Build all_geometry
        order = ['vhvl_angles', 'cdr_shape', 'cdr_canonical', 'paratope_charge',
                 'paratope_hydrophobicity', 'cdr_contacts', 'vhvl_interface']
        cached['all_geometry'] = np.hstack([cached[n] for n in order])
        return cached

    # Compute all features
    geo = compute_all_geometry_features(heavy_seqs, light_seqs)

    # Save to cache
    for name in GEOMETRY_DIMS:
        cache_file = cache_dir / f'{name}.npy'
        np.save(cache_file, geo[name])

    # Build all_geometry
    order = ['vhvl_angles', 'cdr_shape', 'cdr_canonical', 'paratope_charge',
             'paratope_hydrophobicity', 'cdr_contacts', 'vhvl_interface']
    geo['all_geometry'] = np.hstack([geo[n] for n in order])

    return geo


# =============================================================================
# BASE REP LOADING
# =============================================================================

def _load_base_reps(
    heavy_seqs: List[str],
    light_seqs: List[str],
    base_rep_names: List[str],
) -> Dict[str, np.ndarray]:
    """Load base representations from cache. Skip any not found."""
    from kirby.representations.antibody import ANTIBODY_REPS

    reps = {}
    for rep_name in base_rep_names:
        cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
        if cache_file.exists():
            try:
                X = np.load(cache_file, allow_pickle=False)
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 2 and X.shape[0] == len(heavy_seqs):
                    reps[rep_name] = X
                    continue
            except Exception:
                pass

        # Not cached — try computing if it's a fast rep
        if rep_name in ANTIBODY_REPS and rep_name in {'onehot', 'blosum', 'aac', 'paratope'}:
            try:
                X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 2:
                    reps[rep_name] = X
                    np.save(cache_file, X)
                    continue
            except Exception:
                pass

        print(f"    WARNING: base rep '{rep_name}' not cached, skipping")

    return reps


# =============================================================================
# AUGMENTATION EVALUATION
# =============================================================================

def run_geometry_survey(
    dataset_name: str,
    base_rep_names: List[str],
    model_names: List[str],
    n_folds: int = 5,
) -> dict:
    """Run geometry augmentation survey for one dataset."""
    print(f"\n{'='*70}")
    print(f"GEOMETRY SURVEY: {dataset_name}")
    print(f"{'='*70}")
    t0 = time.time()

    # 1. Load data
    data = load_dataset(dataset_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data['groups']
    task_type = data.get('task_type', 'regression')
    is_clf = (task_type == 'classification')

    print(f"  {data['n_samples']} samples, {data['n_groups']} groups, "
          f"task={task_type}")

    # 2. Compute geometry features
    geo_reps = _load_or_compute_geometry(heavy_seqs, light_seqs)

    # 3. Load base reps
    base_reps = _load_base_reps(heavy_seqs, light_seqs, base_rep_names)
    print(f"  Base reps loaded: {sorted(base_reps.keys())}")

    if not base_reps:
        print("  ERROR: No base reps available — run antibody_baseline_survey first")
        return {'error': 'no base reps'}

    # 3b. Build hybrid from all available reps (base + geometry)
    all_reps_for_hybrid = {**base_reps, **geo_reps}
    all_reps_for_hybrid.pop('all_geometry', None)
    if len(all_reps_for_hybrid) >= 2:
        try:
            from kirby.find_best_hybrid import find_best_hybrid
            from kirby.hybrid import apply_feature_selection
            from experiments._shared import FULL_BUDGETS
            _metric = 'spearman' if not is_clf else 'auc'
            search = find_best_hybrid(
                base_reps=all_reps_for_hybrid, labels=labels, groups=groups,
                budgets=FULL_BUDGETS,
                speed='full', return_fitted=True, verbose=False, cache=True,
                dataset_name=dataset_name, tier='general',
                metric=_metric, adaptive_pruning=0.25)
            fitted = search.fitted_results.get('rf')
            if fitted is not None:
                fi = {r: {'selected_indices': idx}
                      for r, idx in fitted.selected_indices.items()}
                geo_reps['kirby_hybrid'] = apply_feature_selection(
                    all_reps_for_hybrid, fi)
                print(f"  Hybrid built: {geo_reps['kirby_hybrid'].shape[1]}D")
        except Exception as e:
            print(f"  Hybrid build failed: {e}")

    # 4. Get models
    models = get_models(model_names, n_samples=data['n_samples'],
                        task_type=task_type)
    print(f"  Models: {list(models.keys())}")

    # 5. Evaluate geometry features standalone
    print(f"\n--- Standalone geometry features ---")
    geo_baselines = {}
    geo_names = ['vhvl_angles', 'cdr_shape', 'cdr_canonical',
                 'paratope_charge', 'paratope_hydrophobicity',
                 'cdr_contacts', 'vhvl_interface', 'all_geometry']

    for geo_name in geo_names:
        X = geo_reps.get(geo_name)
        if X is None or np.all(X == 0):
            print(f"  {geo_name}: SKIPPED (all zeros)")
            continue

        X = np.asarray(X, dtype=np.float64)
        dim = X.shape[1]
        print(f"  {geo_name} ({dim}D):", end='', flush=True)

        entry = {'dimension': dim, 'scores': {}}
        for mname, model in models.items():
            try:
                result = evaluate_rep(X, labels, groups, model, n_folds,
                                      task_type=task_type)
                entry['scores'][mname] = result
                if is_clf:
                    print(f" {mname}=acc:{result['accuracy_mean']:.3f}/"
                          f"F1:{result['f1_weighted_mean']:.3f}",
                          end='', flush=True)
                else:
                    print(f" {mname}={result['r2_mean']:.3f}/"
                          f"\u03c1={result['spearman_mean']:.3f}",
                          end='', flush=True)
            except Exception as e:
                print(f" {mname}=FAIL", end='', flush=True)
        geo_baselines[geo_name] = entry
        print()

    # 6. Augmentation tests: base + geometry
    print(f"\n--- Augmentation tests ---")
    augmentations = {}

    for base_name in sorted(base_reps.keys()):
        X_base = base_reps[base_name]
        base_dim = X_base.shape[1]

        # Find best base score across models
        base_cache = _rep_cache_path(heavy_seqs, light_seqs, base_name)

        for geo_name in geo_names:
            X_geo = geo_reps.get(geo_name)
            if X_geo is None or np.all(X_geo == 0):
                continue

            X_aug = np.hstack([X_base, X_geo])
            aug_key = f"{base_name}+{geo_name}"
            aug_dim = X_aug.shape[1]

            print(f"  {aug_key} ({aug_dim}D):", end='', flush=True)

            entry = {
                'base': base_name,
                'geometry': geo_name,
                'base_dim': base_dim,
                'geo_dim': X_geo.shape[1],
                'concat_dim': aug_dim,
                'scores': {},
            }

            for mname, model in models.items():
                try:
                    result = evaluate_rep(X_aug, labels, groups, model,
                                          n_folds, task_type=task_type)
                    entry['scores'][mname] = result
                    if is_clf:
                        print(f" {mname}=acc:{result['accuracy_mean']:.3f}/"
                              f"F1:{result['f1_weighted_mean']:.3f}",
                              end='', flush=True)
                    else:
                        print(f" {mname}={result['r2_mean']:.3f}/"
                              f"\u03c1={result['spearman_mean']:.3f}",
                              end='', flush=True)
                except Exception as e:
                    print(f" {mname}=FAIL", end='', flush=True)

            augmentations[aug_key] = entry
            print()

    elapsed = time.time() - t0

    return {
        'metadata': {
            'dataset': dataset_name,
            'n_samples': data['n_samples'],
            'n_groups': data['n_groups'],
            'n_folds': n_folds,
            'task_type': task_type,
            'base_reps': sorted(base_reps.keys()),
            'geo_reps': geo_names,
            'models': list(models.keys()),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'geometry_baselines': geo_baselines,
        'augmentations': augmentations,
    }


# =============================================================================
# SUMMARY
# =============================================================================

def print_augmentation_summary(result: dict):
    """Print augmentation delta summary."""
    augmentations = result.get('augmentations', {})
    geo_baselines = result.get('geometry_baselines', {})
    is_clf = result.get('metadata', {}).get('task_type') == 'classification'
    primary = 'accuracy_mean' if is_clf else 'spearman_mean'
    metric_label = 'Acc' if is_clf else '\u03c1'

    if not augmentations:
        return

    print(f"\n{'='*70}")
    print(f"AUGMENTATION SUMMARY ({metric_label})")
    print(f"{'='*70}")

    # Group by base rep
    bases = {}
    for key, entry in augmentations.items():
        base = entry['base']
        if base not in bases:
            bases[base] = {}
        bases[base][entry['geometry']] = entry

    for base_name in sorted(bases.keys()):
        print(f"\n  Base: {base_name}")
        for geo_name, entry in sorted(bases[base_name].items()):
            scores = entry.get('scores', {})
            best_model = None
            best_val = -float('inf')
            for mname, s in scores.items():
                val = s.get(primary, float('nan'))
                if not np.isnan(val) and val > best_val:
                    best_val = val
                    best_model = mname
            if best_model:
                print(f"    +{geo_name:<30} best {metric_label}={best_val:.4f} "
                      f"({best_model})")


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
        description='Structural geometry augmentation survey for antibodies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests 7 structure-based geometry features as augmentations on top of
base antibody representations. Requires ImmuneBuilder for structure
prediction (structures are cached after first computation).

Geometry features:
  vhvl_angles (6D)          CDR loop shape (60D)
  cdr_canonical (12D)       paratope_charge (12D)
  paratope_hydrophobicity (12D)   cdr_contacts (30D)
  vhvl_interface (30D)      all_geometry (162D, all combined)

Examples:
  %(prog)s flab_thermo_jain --models ridge,rf
  %(prog)s all_inter --base-reps esm2,ablang2,prot_t5
  %(prog)s all --output tests/results/geometry_survey/
        """
    )

    parser.add_argument('dataset', type=str,
                        help='Dataset name, group shortcut, or "all"')
    parser.add_argument('--base-reps', type=str, default=None,
                        help=f'Comma-separated base reps (default: {",".join(DEFAULT_BASE_REPS)})')
    parser.add_argument('--models', type=str, default='ridge,rf',
                        help='Comma-separated model names (default: ridge,rf)')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for JSON results')

    args = parser.parse_args()

    # Resolve datasets
    ds_input = args.dataset
    if ds_input in DOMAIN_GROUPS:
        datasets = list(DOMAIN_GROUPS[ds_input])
    elif ds_input in DATASET_REGISTRY:
        datasets = [ds_input]
    else:
        print(f"ERROR: Unknown dataset '{ds_input}'")
        print(f"Available: {sorted(DATASET_REGISTRY.keys())}")
        print(f"Groups: {sorted(DOMAIN_GROUPS.keys())}")
        return 1

    # Resolve base reps
    if args.base_reps:
        base_rep_names = [r.strip() for r in args.base_reps.split(',')]
    else:
        base_rep_names = list(DEFAULT_BASE_REPS)

    model_names = [m.strip() for m in args.models.split(',')]

    print(f"Structural Geometry Augmentation Survey")
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
            result = run_geometry_survey(ds, base_rep_names, model_names,
                                         args.n_folds)
            result = _convert_numpy(result)
            print_augmentation_summary(result)
            all_results[ds] = result
            pass_count += 1

            if output_dir:
                out_path = output_dir / f"geometry_{ds}.json"
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
        print("CROSS-DATASET GEOMETRY SUMMARY")
        print(f"{'='*70}")

        for ds, r in all_results.items():
            geo = r.get('geometry_baselines', {})
            is_clf = r.get('metadata', {}).get('task_type') == 'classification'
            primary = 'accuracy_mean' if is_clf else 'spearman_mean'
            metric_label = 'Acc' if is_clf else '\u03c1'

            print(f"\n  {ds} (standalone):")
            for geo_name, entry in sorted(geo.items()):
                scores = entry.get('scores', {})
                best = max(
                    (s.get(primary, float('-inf')) for s in scores.values()),
                    default=float('nan'))
                print(f"    {geo_name:<30} best {metric_label}={best:.4f}")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
