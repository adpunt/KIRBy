#!/usr/bin/env python3
"""
Antibody Baseline Survey

Simple LM embeddings → Ridge → R², per the compass document.
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


def _rep_cache_path(heavy_seqs: List[str], light_seqs: List[str],
                    rep_name: str) -> Path:
    """Get cache file path for a representation, keyed by sequence hash.

    Respects KIRBY_CACHE_DIR env var (set by setup.sh on the server).
    Falls back to ~/.cache/kirby.
    """
    import hashlib
    content = '\n'.join(heavy_seqs) + '\n\n' + '\n'.join(light_seqs)
    data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    base = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    cache_dir = base / 'reps' / f'ab_{data_hash}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{rep_name}.npy'


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

    Returns dict with r2_mean, r2_std, per_fold.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)

    if actual_folds < 2:
        return {'r2_mean': float('nan'), 'r2_std': float('nan'),
                'per_fold': []}

    if n_unique >= actual_folds:
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(X, labels, groups=groups))
    else:
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, labels))

    r2_scores = []
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
            r2_scores.append(r2)
            fold_details.append({
                'fold': fold_i,
                'r2': round(float(r2), 4),
                'n_train': len(train_idx),
                'n_val': len(val_idx),
            })
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})

    if not r2_scores:
        return {'r2_mean': float('nan'), 'r2_std': float('nan'),
                'per_fold': fold_details}

    return {
        'r2_mean': round(float(np.mean(r2_scores)), 4),
        'r2_std': round(float(np.std(r2_scores)), 4),
        'per_fold': fold_details,
    }


# =============================================================================
# SURVEY
# =============================================================================

def evaluate_ft_rep(factory, labels, groups, model, n_folds=5):
    """Evaluate a fine-tuning factory via CV. Returns dict with r2_mean, r2_std."""
    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)

    if actual_folds < 2:
        return {'r2_mean': float('nan'), 'r2_std': float('nan'), 'per_fold': []}

    if n_unique >= actual_folds:
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(np.zeros(len(labels)), labels, groups=groups))
    else:
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        splits = list(cv.split(np.zeros(len(labels)), labels))

    r2_scores = []
    fold_details = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        try:
            X_train, X_val = factory(train_idx, val_idx, labels[train_idx])
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            y_pred = m.predict(X_val)
            r2 = r2_score(labels[val_idx], y_pred)
            r2_scores.append(r2)
            fold_details.append({
                'fold': fold_i, 'r2': round(float(r2), 4),
                'n_train': len(train_idx), 'n_val': len(val_idx),
            })
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})

    if not r2_scores:
        return {'r2_mean': float('nan'), 'r2_std': float('nan'),
                'per_fold': fold_details}

    return {
        'r2_mean': round(float(np.mean(r2_scores)), 4),
        'r2_std': round(float(np.std(r2_scores)), 4),
        'per_fold': fold_details,
    }


def run_survey(
    dataset_name: str,
    rep_names: List[str],
    model_names: List[str],
    n_folds: int = 5,
    augment: bool = False,
    do_finetune: bool = False,
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

    # 2. Generate representations (with .npy caching)
    from kirby.representations.antibody import ANTIBODY_REPS
    reps = {}
    for rep_name in rep_names:
        if rep_name not in ANTIBODY_REPS:
            print(f"  WARNING: unknown rep '{rep_name}', skipping")
            continue

        # Try loading from cache
        cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
        if cache_file.exists():
            try:
                X = np.load(cache_file, allow_pickle=False)
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 2:
                    print(f"  {rep_name}: loaded from cache ({X.shape[1]}D)")
                    reps[rep_name] = X
                    continue
            except Exception:
                pass  # corrupted cache, recompute

        try:
            print(f"  Generating {rep_name}...", end='', flush=True)
            X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            if X.ndim != 2:
                print(f" FAILED: expected 2D array, got {X.ndim}D {X.shape}")
                continue
            reps[rep_name] = X
            print(f" ({X.shape[1]}D)")
            # Save to cache
            np.save(cache_file, X)
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
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            print(f"  {rep_name}: SKIPPED (array conversion failed)")
            continue
        dim = X.shape[1]
        print(f"  {rep_name} ({dim}D):", end='', flush=True)

        try:
            entry = {'dimension': dim, 'scores': {}}
            for mname, model in models.items():
                result = evaluate_rep(X, labels, groups, model, n_folds)
                entry['scores'][mname] = result
                r2 = result['r2_mean']
                print(f" {mname}={r2:.3f}", end='', flush=True)

            baselines[rep_name] = entry
            print()
        except Exception as e:
            print(f" EVAL FAILED: {e}")

    # 5. Fine-tuned reps (ESM-2 LoRA)
    ft_baselines = {}
    if do_finetune:
        try:
            from kirby.finetuning.antibody_lora import make_antibody_lora_factory
            factory = make_antibody_lora_factory(
                heavy_seqs=heavy_seqs, light_seqs=light_seqs,
                dataset_name=dataset_name, base_model='esm2',
            )
            ft_name = 'esm2_lora'
            print(f"\n--- Fine-tuned reps: [{ft_name}] ---")
            print(f"  {ft_name}:", end='', flush=True)

            entry = {'dimension': 1280, 'scores': {}}
            for mname, model in models.items():
                result = evaluate_ft_rep(factory, labels, groups, model, n_folds)
                entry['scores'][mname] = result
                r2 = result['r2_mean']
                print(f" {mname}={r2:.3f}", end='', flush=True)

            ft_baselines[ft_name] = entry
            print()
        except Exception as e:
            print(f"\n  esm2_lora factory failed: {e}")

    # 6. Augmentation test: concat small reps with top base
    augmentations = {}
    if augment and baselines:
        # Find top-scoring rep by best R² across models
        scored = []
        for name, e in baselines.items():
            for mname, s in e['scores'].items():
                if not np.isnan(s['r2_mean']):
                    scored.append((name, mname, s['r2_mean']))
        if scored:
            top_rep, top_model_name, top_score = max(scored, key=lambda x: x[2])
            X_top = reps[top_rep]
            top_model = models[top_model_name]

            # Small reps = ≤200D, excluding the top rep
            small_reps = {k: v for k, v in reps.items()
                          if v.shape[1] <= 200 and k != top_rep}

            if small_reps:
                print(f"\n--- Augmentation test: {top_rep} (R²={top_score:.4f}) "
                      f"+ {len(small_reps)} small reps ---")

                for aug_name in sorted(small_reps.keys()):
                    X_aug = np.hstack([X_top, small_reps[aug_name]])
                    result = evaluate_rep(X_aug, labels, groups, top_model, n_folds)
                    mean_s = result['r2_mean']
                    delta = mean_s - top_score
                    sign = '+' if delta >= 0 else ''
                    print(f"  {top_rep}+{aug_name} ({X_aug.shape[1]}d): "
                          f"{mean_s:.4f} ({sign}{delta:.4f})")
                    augmentations[f"{top_rep}+{aug_name}"] = {
                        'base': top_rep,
                        'augment': aug_name,
                        'base_dim': X_top.shape[1],
                        'aug_dim': small_reps[aug_name].shape[1],
                        'concat_dim': X_aug.shape[1],
                        'model': top_model_name,
                        'base_score': round(top_score, 4),
                        'concat_score': round(mean_s, 4),
                        'delta': round(delta, 4),
                    }

    elapsed = time.time() - t0

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
        'finetuned_baselines': ft_baselines or None,
        'augmentations': augmentations or None,
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
    header = f"{'Rep':<25} {'Dim':>5}"
    for m in models:
        header += f" {m:>8}"
    print(header)
    print("-" * len(header))

    for rep_name in sorted(baselines.keys()):
        r = baselines[rep_name]
        row = f"{rep_name:<25} {r['dimension']:>5}"
        for m in models:
            s = r['scores'].get(m, {})
            r2 = s.get('r2_mean', float('nan'))
            row += f" {r2:>8.4f}"
        print(row)

    # Fine-tuned scores
    ft = result.get('finetuned_baselines')
    if ft:
        print()
        print("FINE-TUNED:")
        print("-" * len(header))
        for ft_name in sorted(ft.keys()):
            r = ft[ft_name]
            dim = r.get('dimension', '?')
            row = f"{ft_name:<25} {dim:>5}"
            for m in models:
                s = r['scores'].get(m, {})
                r2 = s.get('r2_mean', float('nan'))
                row += f" {r2:>8.4f}"
            print(row)

    print(f"\n  Time: {meta['elapsed_seconds']:.0f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Antibody baseline survey: LM embeddings + Ridge/RF/XGBoost + R2.',
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
    parser.add_argument('--models', type=str, default='ridge,lightgbm',
                        help='Comma-separated model names (default: ridge,lightgbm)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--augment', action='store_true',
                        help='Test augmentation: concat each small rep (<=200D) with top base rep')
    parser.add_argument('--finetune', action='store_true',
                        help='Include fine-tuned reps (ESM-2 LoRA on VH, GPU required)')
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
            result = run_survey(ds, rep_names, model_names, args.n_folds,
                                augment=args.augment,
                                do_finetune=args.finetune)
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
            baselines = r.get('baselines', {})
            if baselines:
                print(f"\n  {ds}:")
                for rep_name in sorted(baselines.keys()):
                    scores = baselines[rep_name]['scores']
                    parts = []
                    for mname, s in scores.items():
                        parts.append(f"{mname}={s['r2_mean']:.4f}")
                    print(f"    {rep_name:<20} {', '.join(parts)}")
            else:
                print(f"  {ds:<30} FAILED")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
