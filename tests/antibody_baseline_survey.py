#!/usr/bin/env python3
"""
Antibody Baseline Survey

Simple LM embeddings → Ridge → Spearman rho, per the compass document.
No hybrids, no tuning, no complex presets.

Usage:
    python tests/antibody_baseline_survey.py flab_expression_koenig
    python tests/antibody_baseline_survey.py all_dms
    python tests/antibody_baseline_survey.py all
    python tests/antibody_baseline_survey.py flab_expression_koenig --skip-lm
    python tests/antibody_baseline_survey.py all --models ridge,rf,xgboost
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# {name: (loader_module, loader_func, group_strategy)}
# group_strategy: 'identity' (hash sequences), 'complex' (PDB ID), 'experiment'
DATASET_REGISTRY = {
    # DMS (intra-family)
    'flab_expression_koenig': ('flab', 'identity'),
    'flab_binding_koenig':    ('flab', 'identity'),
    'abagym_Ang2':            ('abagym', 'identity'),
    'abagym_EGFR':            ('abagym', 'identity'),
    'abagym_lysozyme':        ('abagym', 'identity'),
    'abagym_VEGF':            ('abagym', 'identity'),
    'abagym_HER2':            ('abagym', 'identity'),
    # Inter-family
    'ddg_combined':           ('ddg', 'complex'),
    'abagym_all':             ('abagym', 'experiment'),
    'abcov_kd':               ('abcov', 'identity'),
    'flab_thermo_jain':       ('flab', 'identity'),
    'flab_aggregation_jain':  ('flab', 'identity'),
    # FLAb2 expansion
    'flab_binding_phillips_h3':   ('flab', 'identity'),
    'flab_binding_adams':         ('flab', 'identity'),
    'flab_expression_adams':      ('flab', 'identity'),
    'flab_thermo_tresanco':       ('flab', 'identity'),
    'flab_thermo_hutchinson':     ('flab', 'identity'),
    'flab_immunogenicity_marks':  ('flab', 'identity'),
    'flab_aggregation_jain_hicrt':('flab', 'identity'),
    'flab_polyreact_jain_bvp':    ('flab', 'identity'),
    'flab_polyreact_jain_cic':    ('flab', 'identity'),
    'flab_pk_jain_fcrnrt':        ('flab', 'identity'),
    'flab_pk_jain_heprt':         ('flab', 'identity'),
    'flab_aggregation_jain_sas':  ('flab', 'identity'),
}

DOMAIN_GROUPS = {
    'all_dms': [
        'flab_expression_koenig', 'flab_binding_koenig',
        'flab_binding_phillips_h3', 'flab_binding_adams', 'flab_expression_adams',
        'abagym_Ang2', 'abagym_EGFR', 'abagym_lysozyme',
        'abagym_VEGF', 'abagym_HER2',
    ],
    'all_inter': [
        'ddg_combined', 'abagym_all', 'abcov_kd',
        'flab_thermo_jain', 'flab_aggregation_jain',
        'flab_thermo_tresanco', 'flab_thermo_hutchinson',
        'flab_immunogenicity_marks',
        'flab_aggregation_jain_hicrt', 'flab_aggregation_jain_sas',
        'flab_polyreact_jain_bvp', 'flab_polyreact_jain_cic',
        'flab_pk_jain_fcrnrt', 'flab_pk_jain_heprt',
    ],
    'all': list(DATASET_REGISTRY.keys()),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _abagym_full_name(short_name: str) -> str:
    """Convert short AbAgym name to full experiment name for the loader."""
    mapping = {
        'abagym_Ang2': 'abagym_Ang2_2017_G6',
        'abagym_EGFR': 'abagym_EGFR_2013_Cetuximab',
        'abagym_lysozyme': 'abagym_lysozyme_2019_D441',
        'abagym_VEGF': 'abagym_VEGF_2017b_G6',
        'abagym_HER2': 'abagym_HER2_2021_trastuzumab',
        'abagym_all': 'abagym_all',
    }
    return mapping.get(short_name, short_name)


def load_dataset(name: str) -> dict:
    """Load antibody dataset. Returns dict with heavy_seqs, light_seqs, labels, groups."""
    source, group_strategy = DATASET_REGISTRY[name]

    if source == 'flab':
        from kirby.datasets.flab import load_flab_dataset
        data = load_flab_dataset(name)
    elif source == 'abagym':
        from kirby.datasets.abagym import load_abagym_dataset
        data = load_abagym_dataset(_abagym_full_name(name))
    elif source == 'ddg':
        from kirby.datasets.ddg_combined import load_ddg_combined_dataset
        data = load_ddg_combined_dataset()
    elif source == 'abcov':
        from kirby.datasets.abcov import load_abcov_dataset
        data = load_abcov_dataset(name)
    else:
        raise ValueError(f"Unknown source: {source}")

    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']

    # Guard: if light_seqs is None (unpaired), use empty strings
    if light_seqs is None:
        light_seqs = [''] * len(heavy_seqs)

    # Assign groups
    groups, n_groups = _assign_groups(
        heavy_seqs, light_seqs, group_strategy, data)

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'groups': groups,
        'n_groups': n_groups,
        'n_samples': len(labels),
    }


def _assign_groups(heavy_seqs, light_seqs, strategy, data):
    """Assign CV groups based on strategy."""
    if strategy == 'complex':
        from kirby.datasets.ddg_combined import assign_complex_groups
        pdb_ids = data.get('pdb_ids', [])
        return assign_complex_groups(pdb_ids)
    elif strategy == 'experiment':
        # For abagym_all: group by experiment name from mutation metadata.
        # Each mutation dict has an 'experiment' key set by the loader.
        mutations = data.get('mutations', [])
        if mutations and mutations[0] and 'experiment' in mutations[0][0]:
            exp_labels = [m[0]['experiment'] for m in mutations]
            unique = sorted(set(exp_labels))
            label_to_id = {l: i for i, l in enumerate(unique)}
            groups = np.array([label_to_id[l] for l in exp_labels])
            return groups, len(unique)
        # Fallback: identity groups
        return _identity_groups(heavy_seqs)
    else:  # identity
        return _identity_groups(heavy_seqs)


def _identity_groups(seqs, min_groups=3):
    """Assign groups by unique sequence identity."""
    import hashlib
    unique = {}
    groups = np.zeros(len(seqs), dtype=int)
    for i, s in enumerate(seqs):
        if s not in unique:
            unique[s] = len(unique)
        groups[i] = unique[s]
    n_groups = len(unique)
    # If too few groups (e.g. DMS with identical-ish seqs), deterministic hash
    if n_groups < min_groups:
        n_buckets = max(min_groups, len(seqs) // 5)
        groups = np.array([
            int(hashlib.sha256(s.encode()).hexdigest(), 16) % n_buckets
            for s in seqs
        ])
        n_groups = len(np.unique(groups))
    return groups, n_groups


# =============================================================================
# MODELS
# =============================================================================

def get_models(model_names: List[str]) -> Dict[str, object]:
    """Return model instances by name."""
    catalog = {
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 50)),
    }

    if 'rf' in model_names:
        from sklearn.ensemble import RandomForestRegressor
        catalog['rf'] = RandomForestRegressor(
            n_estimators=100, max_features='sqrt', random_state=42, n_jobs=1)

    if 'xgboost' in model_names:
        try:
            from xgboost import XGBRegressor
            catalog['xgboost'] = XGBRegressor(
                n_estimators=100, colsample_bytree=0.5,
                random_state=42, n_jobs=1, verbosity=0)
        except ImportError:
            print("  WARNING: xgboost not available")

    if 'lightgbm' in model_names:
        try:
            from lightgbm import LGBMRegressor
            catalog['lightgbm'] = LGBMRegressor(
                n_estimators=100, random_state=42, n_jobs=1, verbose=-1)
        except ImportError:
            print("  WARNING: lightgbm not available")

    if 'knn' in model_names:
        from sklearn.neighbors import KNeighborsRegressor
        catalog['knn'] = KNeighborsRegressor(n_neighbors=5, n_jobs=1)

    return {k: v for k, v in catalog.items() if k in model_names}


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Evaluate one (rep, model) pair via GroupKFold.

    Returns dict with r2_mean, r2_std, spearman_mean, spearman_std, per_fold.
    """
    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)

    if actual_folds < 2:
        # Can't do CV with < 2 groups
        return {'r2_mean': float('nan'), 'r2_std': float('nan'),
                'spearman_mean': float('nan'), 'spearman_std': float('nan'),
                'per_fold': []}

    if n_unique >= actual_folds:
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(X, labels, groups=groups))
    else:
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, labels))

    r2_scores = []
    spearman_scores = []
    fold_details = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            y_pred = m.predict(X_val)

            r2 = r2_score(labels[val_idx], y_pred)
            rho, pval = spearmanr(labels[val_idx], y_pred)
            if np.isnan(rho):
                rho = 0.0  # constant predictions → no correlation

            r2_scores.append(r2)
            spearman_scores.append(rho)
            fold_details.append({
                'fold': fold_i,
                'r2': round(float(r2), 4),
                'spearman': round(float(rho), 4),
                'n_train': len(train_idx),
                'n_val': len(val_idx),
            })
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})

    if not r2_scores:
        return {'r2_mean': float('nan'), 'r2_std': float('nan'),
                'spearman_mean': float('nan'), 'spearman_std': float('nan'),
                'per_fold': fold_details}

    return {
        'r2_mean': round(float(np.mean(r2_scores)), 4),
        'r2_std': round(float(np.std(r2_scores)), 4),
        'spearman_mean': round(float(np.mean(spearman_scores)), 4),
        'spearman_std': round(float(np.std(spearman_scores)), 4),
        'per_fold': fold_details,
    }


# =============================================================================
# SURVEY
# =============================================================================

def run_survey(
    dataset_name: str,
    rep_names: List[str],
    model_names: List[str],
    n_folds: int = 5,
) -> dict:
    """Run baseline survey for one dataset. Returns result dict."""
    print(f"\n{'='*70}")
    print(f"SURVEY: {dataset_name}")
    print(f"{'='*70}")
    t0 = time.time()

    # 1. Load data
    data = load_dataset(dataset_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data['groups']

    print(f"  {data['n_samples']} samples, {data['n_groups']} groups")

    # 2. Generate representations
    from kirby.representations.antibody import ANTIBODY_REPS
    reps = {}
    for rep_name in rep_names:
        if rep_name not in ANTIBODY_REPS:
            print(f"  WARNING: unknown rep '{rep_name}', skipping")
            continue
        try:
            print(f"  Generating {rep_name}...", end='', flush=True)
            X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            if X.ndim != 2:
                print(f" FAILED: expected 2D array, got {X.ndim}D {X.shape}")
                continue
            reps[rep_name] = X
            print(f" ({X.shape[1]}D)")
        except Exception as e:
            print(f" FAILED: {e}")

    if not reps:
        raise RuntimeError("No representations generated successfully")

    # 3. Get models
    models = get_models(model_names)
    print(f"  Models: {list(models.keys())}")

    # 4. Evaluate each (rep, model) cell
    baselines = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        dim = X.shape[1]
        print(f"  {rep_name} ({dim}D):", end='', flush=True)

        entry = {'dimension': dim, 'scores': {}}
        for mname, model in models.items():
            result = evaluate_rep(X, labels, groups, model, n_folds)
            entry['scores'][mname] = result
            rho = result['spearman_mean']
            r2 = result['r2_mean']
            print(f" {mname}(rho={rho:.3f},r2={r2:.3f})", end='', flush=True)

        # Best model for this rep
        valid = [(m, s) for m, s in entry['scores'].items()
                 if not np.isnan(s['spearman_mean'])]
        if valid:
            best = max(valid, key=lambda x: x[1]['spearman_mean'])
            entry['best'] = {
                'model': best[0],
                'spearman': best[1]['spearman_mean'],
                'r2': best[1]['r2_mean'],
            }
        else:
            entry['best'] = None

        baselines[rep_name] = entry
        print()

    elapsed = time.time() - t0

    # 5. Summary
    all_bests = [(name, e['best']) for name, e in baselines.items()
                 if e['best'] is not None]
    if all_bests:
        winner = max(all_bests, key=lambda x: x[1]['spearman'])
        summary = {
            'best_rep': winner[0],
            'best_model': winner[1]['model'],
            'best_spearman': winner[1]['spearman'],
            'best_r2': winner[1]['r2'],
        }
    else:
        summary = {'best_rep': None, 'best_model': None,
                   'best_spearman': None, 'best_r2': None}

    return {
        'metadata': {
            'dataset': dataset_name,
            'n_samples': data['n_samples'],
            'n_groups': data['n_groups'],
            'n_folds': n_folds,
            'reps': sorted(reps.keys()),
            'models': list(models.keys()),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'baselines': baselines,
        'summary': summary,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def _convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj


def print_summary_table(result):
    """Print compact summary table."""
    meta = result['metadata']
    baselines = result['baselines']
    models = meta['models']

    print(f"\n{'='*70}")
    print(f"SUMMARY: {meta['dataset']} ({meta['n_samples']} samples, "
          f"{meta['n_groups']} groups, {meta['n_folds']}-fold)")
    print(f"{'='*70}")

    # Header
    model_cols = []
    for m in models:
        model_cols.append(f"{'rho_'+m:>12}")
        model_cols.append(f"{'r2_'+m:>10}")
    header = f"{'Rep':<15} {'Dim':>5}" + ''.join(model_cols) + f"  {'Best rho':>10}"
    print(header)
    print("-" * len(header))

    for rep_name in sorted(baselines.keys()):
        r = baselines[rep_name]
        row = f"{rep_name:<15} {r['dimension']:>5}"
        for m in models:
            s = r['scores'].get(m, {})
            rho = s.get('spearman_mean', float('nan'))
            r2 = s.get('r2_mean', float('nan'))
            row += f" {rho:>11.4f} {r2:>9.4f}"
        best = r.get('best')
        if best:
            row += f"  {best['spearman']:>10.4f}"
        print(row)

    summary = result['summary']
    print()
    if summary.get('best_rep'):
        print(f"WINNER: {summary['best_rep']} + {summary['best_model']} = "
              f"rho={summary['best_spearman']:.4f}, R2={summary['best_r2']:.4f}")
    print(f"  Time: {meta['elapsed_seconds']:.0f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Antibody baseline survey: simple LM embeddings + Ridge + Spearman rho.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  flab_expression_koenig, flab_binding_koenig       (FLAb DMS)
  abagym_Ang2/EGFR/lysozyme/VEGF/HER2              (AbAgym DMS)
  ddg_combined, abagym_all, abcov_kd,               (inter-family)
    flab_thermo_jain, flab_aggregation_jain

Shortcuts:
  all_dms       Run all 7 DMS datasets
  all_inter     Run all 5 inter-family datasets
  all           Run everything

Examples:
  %(prog)s flab_expression_koenig --skip-lm
  %(prog)s all_dms --models ridge,rf
  %(prog)s all --output tests/results/antibody_survey/
        """
    )

    parser.add_argument('dataset', type=str,
                        help='Dataset name, group shortcut, or "all"')
    parser.add_argument('--skip-lm', action='store_true',
                        help='Skip LM reps (only one-hot, blosum, aac)')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep names to use (overrides --skip-lm)')
    parser.add_argument('--models', type=str, default='ridge',
                        help='Comma-separated model names (default: ridge)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (JSON file or directory for multi-dataset)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Resolve datasets
    ds_input = args.dataset
    if ds_input in DOMAIN_GROUPS:
        datasets = DOMAIN_GROUPS[ds_input]
    elif ds_input in DATASET_REGISTRY:
        datasets = [ds_input]
    else:
        print(f"ERROR: Unknown dataset '{ds_input}'")
        print(f"Available: {sorted(DATASET_REGISTRY.keys())}")
        print(f"Groups: {sorted(DOMAIN_GROUPS.keys())}")
        return 1

    # Resolve reps
    from kirby.representations.antibody import (
        ANTIBODY_REPS, ANTIBODY_LM_REPS, ANTIBODY_FAST_REPS)

    if args.reps:
        rep_names = [r.strip() for r in args.reps.split(',')]
    elif args.skip_lm:
        rep_names = sorted(ANTIBODY_FAST_REPS)
    else:
        rep_names = sorted(ANTIBODY_REPS.keys())

    model_names = [m.strip() for m in args.models.split(',')]

    print(f"Antibody Baseline Survey")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {model_names}")
    print(f"  Folds: {args.n_folds}")

    # Output handling
    output_dir = None
    output_file = None
    if args.output:
        p = Path(args.output)
        if p.suffix == '.json':
            output_file = p
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = p
            output_dir.mkdir(parents=True, exist_ok=True)

    # Run surveys
    all_results = {}
    pass_count = 0
    fail_count = 0

    for ds in datasets:
        try:
            result = run_survey(ds, rep_names, model_names, args.n_folds)
            result = _convert_numpy(result)
            print_summary_table(result)
            all_results[ds] = result
            pass_count += 1

            if output_dir:
                out_path = output_dir / f"{ds}.json"
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"\n  FAILED: {ds}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Save single-file output
    if output_file and all_results:
        payload = (list(all_results.values())[0]
                   if len(all_results) == 1 else all_results)
        with open(output_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {output_file}")

    # Cross-dataset summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*70}")
        for ds, r in all_results.items():
            s = r['summary']
            if s.get('best_rep'):
                print(f"  {ds:<30} {s['best_rep']:<12} + {s['best_model']:<8} = "
                      f"rho={s['best_spearman']:.4f}")
            else:
                print(f"  {ds:<30} FAILED")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
